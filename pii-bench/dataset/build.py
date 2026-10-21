"""Fill prompt templates with sampled PII values to build ``dataset.jsonl``.

Values are sampled deterministically from a seeded RNG, never from an LLM,
so every ``pii_spans`` entry is exact by construction. Each placeholder type
is filled once per item: a template that uses ``{NAME}`` twice gets the same
name (and the same ``entity_id``) in both places.

Usage (from the ``pii-bench/`` directory)::

    uv run --project .. python -m dataset.build
    uv run --project .. python -m dataset.build --fills-per-template 10
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from models import DatasetItem, PiiSpan, PromptTemplate


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BENCH_DIR = Path(__file__).resolve().parent.parent
PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")

NAMES: tuple[str, ...] = (
    "Sarah Lopez",
    "James Carter",
    "Priya Raman",
    "Wei Zhang",
    "Fatima Al-Sayed",
    "Olumide Adeyemi",
    "Mateo Fernández",
    "Hannah Schmidt",
    "Kenji Watanabe",
    "Aisha Mohammed",
    "Liam O'Connor",
    "Sofia Rossi",
    "Arjun Mehta",
    "Mei Lin Chen",
    "Diego Morales",
    "Emily Nguyen",
    "Tomasz Kowalski",
    "Amara Okafor",
    "Noah Thompson",
    "Yuki Tanaka",
    "Leila Hosseini",
    "Carlos Mendoza",
    "Grace Kim",
    "Ravi Shankar",
    "Isabelle Dubois",
    "Kwame Mensah",
    "Ana Petrovic",
    "Daniel Cohen",
    "Zainab Hussain",
    "Bill Rose",
)

# Per-family sampling ranges. Retirement ages straddle the 50 (catch-up),
# 55 (rule of 55) and 59.5 (IRA penalty-free) thresholds.
AGE_RANGES: dict[str, tuple[int, int]] = {
    "retirement_eligibility": (45, 70),
}
DEFAULT_AGE_RANGE = (18, 80)

MONEY_RANGES: dict[str, tuple[float, float]] = {
    "email_drafting": (500, 150_000),
}
DEFAULT_MONEY_RANGE = (100, 250_000)

MONEY_FORMATS = ("comma", "k_suffix", "words_suffix")

# Not generated yet: the preempt NER and sanitizer don't handle these, so they
# would only measure a known gap. Add them back as a separate stratum once supported.
# - AGE: spelled-out ("fifty-two"), birth date/year ("born in 1974"), decades
#   ("early 50s"), relative ("I'll be 60 next month").
# - MONEY: spelled-out ("eighty-five thousand"), colloquial ("85 grand"),
#   ranges ("between 80 and 90k"), relative ("twice my spouse's salary").


@dataclass(frozen=True)
class Fill:
    """A sampled value for one placeholder type within one item."""

    text: str
    value: float | None = None
    currency: str | None = None
    surface_format: str | None = None


def _round_money(value: float) -> int:
    step = 100 if value < 10_000 else 1_000
    return max(step, int(round(value / step) * step))


def _format_money(value: int, fmt: str) -> str:
    if fmt == "k_suffix":
        return f"${value / 1000:g}k"
    if fmt == "words_suffix":
        return f"{value:,} dollars"
    return f"${value:,}"


def sample_fill(slot: str, family: str, rng: random.Random) -> Fill:
    """Sample one value for a placeholder type."""
    if slot == "NAME":
        return Fill(text=rng.choice(NAMES), surface_format="full")
    if slot == "AGE":
        low, high = AGE_RANGES.get(family, DEFAULT_AGE_RANGE)
        age = rng.randint(low, high)
        return Fill(text=str(age), value=float(age), surface_format="numeric")
    if slot == "MONEY":
        low, high = MONEY_RANGES.get(family, DEFAULT_MONEY_RANGE)
        value = _round_money(math.exp(rng.uniform(math.log(low), math.log(high))))
        fmt = rng.choice(MONEY_FORMATS)
        return Fill(
            text=_format_money(value, fmt),
            value=float(value),
            currency="USD",
            surface_format=fmt,
        )
    raise ValueError(f"no sampler for placeholder {{{slot}}}")


def fill_template(template: PromptTemplate, rng: random.Random) -> tuple[str, list[PiiSpan]]:
    """Substitute every placeholder and return the text with exact spans."""
    fills: dict[str, Fill] = {}
    parts: list[str] = []
    spans: list[PiiSpan] = []
    cursor = 0
    length = 0

    for match in PLACEHOLDER_RE.finditer(template.text):
        slot = match.group(1)
        if slot not in fills:
            fills[slot] = sample_fill(slot, template.task_family, rng)
        fill = fills[slot]

        literal = template.text[cursor : match.start()]
        parts.append(literal)
        length += len(literal)

        spans.append(
            PiiSpan(
                label=slot,  # type: ignore[arg-type]
                start=length,
                end=length + len(fill.text),
                text=fill.text,
                entity_id=f"{slot.lower()}-1",
                value=fill.value,
                currency=fill.currency,
                surface_format=fill.surface_format,
            )
        )
        parts.append(fill.text)
        length += len(fill.text)
        cursor = match.end()

    parts.append(template.text[cursor:])
    return "".join(parts), spans


def validate_item(item: DatasetItem) -> list[str]:
    """Return problems with a built item; empty means valid."""
    problems = []
    for span in item.pii_spans:
        if item.text[span.start : span.end] != span.text:
            problems.append(f"span {span.start}:{span.end} does not match {span.text!r}")
    if PLACEHOLDER_RE.search(item.text):
        problems.append("unfilled placeholder left in text")
    return problems


def load_templates(templates_dir: Path) -> list[PromptTemplate]:
    templates: list[PromptTemplate] = []
    for path in sorted(templates_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        templates.extend(PromptTemplate(**raw) for raw in data.get("templates", []))
    return templates


def build_dataset(
    templates: list[PromptTemplate], fills_per_template: int, seed: int
) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    seen_texts: set[str] = set()

    for template in templates:
        for k in range(fills_per_template):
            item_seed = f"{seed}-{template.id}-{k}"
            text, spans = fill_template(template, random.Random(item_seed))
            if text in seen_texts:
                logger.warning("[%s] duplicate fill %d skipped", template.id, k)
                continue
            seen_texts.add(text)

            item = DatasetItem(
                id=f"{template.id}-{k:02d}",
                template_id=template.id,
                task_family=template.task_family,
                task_type=template.task_type,
                pii_dependency=template.pii_dependency,
                verifiable=template.verifiable,
                seed=item_seed,
                text=text,
                pii_spans=spans,
            )
            problems = validate_item(item)
            if problems:
                raise ValueError(f"{item.id} failed validation: {problems}")
            items.append(item)
    return items


def main() -> None:
    config = yaml.safe_load((BENCH_DIR / "config.yaml").read_text())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates-dir", type=Path, default=BENCH_DIR / "templates")
    parser.add_argument(
        "--out",
        type=Path,
        default=BENCH_DIR / config["paths"]["data_dir"] / "dataset.jsonl",
    )
    parser.add_argument("--fills-per-template", type=int, default=4)
    parser.add_argument("--seed", type=int, default=config["dataset"]["seed"])
    args = parser.parse_args()

    templates = load_templates(args.templates_dir)
    if not templates:
        raise SystemExit(f"no templates found in {args.templates_dir}")

    items = build_dataset(templates, args.fills_per_template, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")

    logger.info(
        "wrote %d items from %d templates -> %s", len(items), len(templates), args.out
    )


if __name__ == "__main__":
    main()
