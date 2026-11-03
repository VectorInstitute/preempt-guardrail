"""Generate PII-sanitizer-benchmark prompt templates using an LLM (Milestone 1, step 1).

A *template* is a prompt skeleton with typed placeholders (``{NAME}``,
``{AGE}``, ``{MONEY}``) that gets filled in later with concrete slot values
(Faker names, sampled ages, sampled money amounts) to build the actual
dataset items. The dataset-generation LLM (configured as ``claude-sonnet``
via ``create_dataset_generator_client``) only writes the template *wording*;
it never invents a concrete name/age/amount, and it never decides where PII
spans are -- those come for free from wherever the placeholders end up once
slots are filled in later.

Families already written by hand live in ``templates/``. Entries here describe
the same catalog, including repeated slots such as ``{MONEY_2}``.

Usage (from the ``pii-bench/`` directory)::

    uv run python -m dataset.generate_templates
    uv run python -m dataset.generate_templates --families retirement_eligibility --n-per-family 8

Requires the dataset-generator client's env vars to be set (e.g. in
``pii-bench/.env``): ``VP_API_KEY`` and optionally ``DATASET_GENERATOR_MODEL``.
"""

from __future__ import annotations

import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from llm.clients import OpenAICompatibleClient, create_dataset_generator_client
from models import PiiDependency, PromptTemplate


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")


@dataclass(frozen=True)
class TaskFamily:
    """A fixed catalog entry describing one task family to generate templates for."""

    name: str
    task_type: str
    pii_dependency: PiiDependency  # "incidental" (context only) | "semantic" (value-dependent) | "mixed"
    verifiable: bool
    required_slots: tuple[str, ...]
    guidance: str


# Hand-written YAML in templates/ is the dataset. This catalog matches it.
TASK_FAMILIES: dict[str, TaskFamily] = {
    "retirement_eligibility": TaskFamily(
        name="retirement_eligibility",
        task_type="eligibility",
        pii_dependency="semantic",
        verifiable=True,
        required_slots=("NAME", "AGE"),
        guidance=(
            "A user asks whether {NAME}, who is {AGE} years old, can withdraw "
            "from a retirement account without an early-withdrawal penalty, "
            "or asks a closely related age-gated eligibility question (e.g. "
            "penalty-free withdrawal age, catch-up contribution age). The "
            "prompt must end with this literal instruction, verbatim: "
            '"End your answer with a line: FINAL_ANSWER: <ELIGIBLE or '
            'NOT_ELIGIBLE>." Vary phrasing and scenario detail across '
            "templates, but keep {NAME} and {AGE} as the only placeholders, "
            "each used exactly once."
        ),
    ),
    "email_drafting": TaskFamily(
        name="email_drafting",
        task_type="drafting",
        pii_dependency="incidental",
        verifiable=False,
        required_slots=("NAME", "MONEY"),
        guidance=(
            "A user asks for help drafting a short email on {NAME}'s behalf "
            "that happens to mention a dollar amount {MONEY} (e.g. a salary "
            "negotiation ask, an invoice follow-up, a budget request). The "
            "name and amount are incidental to the task -- the quality of a "
            "good email does not depend on their exact values. Vary the "
            "scenario and tone across templates, but keep {NAME} and "
            "{MONEY} as the only placeholders, each used exactly once."
        ),
    ),
    "general_request": TaskFamily(
        name="general_request",
        task_type="explanation",
        pii_dependency="incidental",
        verifiable=False,
        required_slots=("NAME", "AGE", "MONEY"),
        guidance=(
            "A user asks for a general explanation or a general checklist. "
            "{NAME}, {AGE}, and {MONEY} appear as context, and the prompt "
            "explicitly says not to calculate anything from them and not to "
            "tailor the answer to those exact values. No FINAL_ANSWER line."
        ),
    ),
    "savings_rate": TaskFamily(
        name="savings_rate",
        task_type="calculation",
        pii_dependency="semantic",
        verifiable=True,
        required_slots=("NAME", "MONEY", "MONEY_2"),
        guidance=(
            "A user asks what percent of {NAME}'s salary ({MONEY}) they saved, "
            "given an amount saved ({MONEY_2}). {MONEY} is the salary and must "
            "appear before {MONEY_2}, which is the savings amount and must be "
            "a different placeholder. The prompt must end with this literal "
            'instruction, verbatim: "End your answer with a line: FINAL_ANSWER: '
            '<the percent as a number, not a fraction>."'
        ),
    ),
    "raise_calculation": TaskFamily(
        name="raise_calculation",
        task_type="calculation",
        pii_dependency="semantic",
        verifiable=True,
        required_slots=("NAME", "MONEY", "MONEY_2"),
        guidance=(
            "A user asks about a raise for {NAME}: current salary is {MONEY} "
            "and the raise is {MONEY_2} more per year. Ask either for the "
            "percent increase or for the new annual salary. {MONEY} must appear "
            "before {MONEY_2}. The prompt must end with this literal instruction, "
            'verbatim: "End your answer with a line: FINAL_ANSWER: <the number>."'
        ),
    ),
    "age_gate": TaskFamily(
        name="age_gate",
        task_type="eligibility",
        pii_dependency="semantic",
        verifiable=True,
        required_slots=("NAME", "AGE"),
        guidance=(
            "A user asks whether {NAME}, who is {AGE}, meets a cutoff that the "
            "prompt states explicitly: at least 18, at least 21, or at least 65. "
            "One cutoff per template. The prompt must end with this literal "
            'instruction, verbatim: "End your answer with a line: FINAL_ANSWER: '
            '<ELIGIBLE or NOT_ELIGIBLE>."'
        ),
    ),
}

SYSTEM_PROMPT = (
    "You write PROMPT TEMPLATES for a benchmark dataset. A template is a "
    "user message with typed placeholders like {NAME}, {AGE}, and {MONEY} "
    "that will be substituted with real-looking values later by code, not "
    "by you. Never invent a concrete name, age, or amount yourself -- use "
    "only the placeholders you are told to use, written exactly as "
    "{PLACEHOLDER}. Return strict JSON only, no markdown fences."
)


def _build_user_prompt(family: TaskFamily, n: int) -> str:
    slots = ", ".join(f"{{{s}}}" for s in family.required_slots)
    return (
        f"Task family: {family.name}\n"
        f"Task type: {family.task_type}\n"
        f"Required placeholders (use ALL of them, exactly once each, in "
        f"every template): {slots}\n\n"
        f"{family.guidance}\n\n"
        f"Write {n} distinct template variants for this task family. Each "
        "should be a single realistic user message (1-4 sentences), written "
        "as if a real person typed it, with the required placeholders "
        "inserted where a name/age/amount would naturally go.\n\n"
        "Respond with JSON of this exact shape:\n"
        '{"templates": [{"text": "...", "notes": "one short phrase on what '
        'varies in this variant"}, ...]}'
    )


def _validate_template(text: str, family: TaskFamily) -> list[str]:
    """Return a list of problems with a generated template; empty = valid."""
    problems = []
    found = set(PLACEHOLDER_RE.findall(text))
    required = set(family.required_slots)
    if found != required:
        problems.append(f"placeholders {sorted(found)} != required {sorted(required)}")
    if family.verifiable and "FINAL_ANSWER" not in text:
        problems.append("missing FINAL_ANSWER instruction for a verifiable family")
    if len(text.strip()) < 20:
        problems.append("template text looks too short")
    return problems


def generate_family_templates(
    client: OpenAICompatibleClient,
    family: TaskFamily,
    n: int,
    *,
    max_attempts: int = 2,
) -> list[PromptTemplate]:
    """Call the dataset-gen LLM (with one retry) to get up to `n` templates for a family."""
    templates: list[PromptTemplate] = []
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            payload = client.complete_json(
                _build_user_prompt(family, n),
                system=SYSTEM_PROMPT,
                temperature=0.9,
                max_tokens=2048,
            )
            raw_templates = payload.get("templates", [])
            if not isinstance(raw_templates, list) or not raw_templates:
                raise ValueError(f"no non-empty 'templates' list in response: {payload!r}")

            accepted: list[PromptTemplate] = []
            rejected: list[list[str]] = []
            for item in raw_templates:
                text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
                problems = _validate_template(text, family)
                if problems:
                    rejected.append(problems)
                    continue
                accepted.append(
                    PromptTemplate(
                        id=f"{family.name}-{len(templates) + len(accepted):03d}",
                        task_family=family.name,
                        task_type=family.task_type,
                        pii_dependency=family.pii_dependency,
                        verifiable=family.verifiable,
                        text=text,
                        slots=list(family.required_slots),
                        notes=str(item.get("notes", "")) if isinstance(item, dict) else "",
                    )
                )

            templates.extend(accepted)
            if rejected:
                logger.warning(
                    "[%s] rejected %d/%d templates on attempt %d: %s",
                    family.name,
                    len(rejected),
                    len(raw_templates),
                    attempt,
                    rejected,
                )
            if len(templates) >= n or attempt == max_attempts:
                break
        except Exception as exc:  # noqa: BLE001 - log and retry, return partial results
            last_error = exc
            logger.warning("[%s] attempt %d failed: %s", family.name, attempt, exc)

    if not templates and last_error is not None:
        raise last_error
    return templates[:n]


def write_templates(templates: list[PromptTemplate], out_dir: Path) -> Path:
    """Write one YAML file per task family (overwrites any existing file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    family = templates[0].task_family
    out_path = out_dir / f"{family}.yaml"
    out_path.write_text(
        yaml.safe_dump(
            {"templates": [t.model_dump() for t in templates]},
            sort_keys=False,
            allow_unicode=True,
        )
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(TASK_FAMILIES),
        default=sorted(TASK_FAMILIES),
        help="Task families to generate templates for.",
    )
    parser.add_argument("--n-per-family", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "templates",
        help="Directory to write one <family>.yaml file into per family.",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    client = create_dataset_generator_client()
    families = [TASK_FAMILIES[name] for name in args.families]

    results: dict[str, list[PromptTemplate]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(families))) as pool:
        futures = {
            pool.submit(generate_family_templates, client, fam, args.n_per_family): fam
            for fam in families
        }
        for future in as_completed(futures):
            fam = futures[future]
            try:
                results[fam.name] = future.result()
            except Exception:
                logger.exception("[%s] failed to generate templates", fam.name)

    for name, templates in results.items():
        if not templates:
            logger.warning("[%s] produced 0 valid templates", name)
            continue
        out_path = write_templates(templates, args.out_dir)
        logger.info("[%s] wrote %d templates -> %s", name, len(templates), out_path)


if __name__ == "__main__":
    main()
