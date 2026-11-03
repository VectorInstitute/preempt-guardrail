"""Fill prompt templates with sampled PII values to build ``dataset.jsonl``.

Values are sampled deterministically from a seeded RNG, never from an LLM,
so every ``pii_spans`` entry is exact by construction. ``{NAME}`` twice is
the same person (``name-1``). ``{NAME_2}`` is a different person (``name-2``);
the same rule applies to ``{AGE_2}`` and ``{MONEY_2}``.

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
from models import DatasetItem, EntityType, PiiSpan, PromptTemplate


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BENCH_DIR = Path(__file__).resolve().parent.parent
PLACEHOLDER_RE = re.compile(r"\{([A-Z]+(?:_[2-9])?)\}")
_SLOT_RE = re.compile(r"^(NAME|AGE|MONEY)(?:_([2-9]))?$")

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

# Inclusive integer windows, keyed by template id, sitting on either side of a
# decision threshold. Retirement: 50 catch-up, 55 rule of 55, 59.5 IRA penalty.
# Age gates: 18, 21, and 65, each stated in that template.
AGE_WINDOWS: dict[str, tuple[int, int]] = {
    "retirement_eligibility-000": (57, 62),
    "retirement_eligibility-001": (53, 57),
    "retirement_eligibility-002": (48, 52),
    "age_gate-000": (16, 20),
    "age_gate-001": (19, 23),
    "age_gate-002": (63, 67),
}
DEFAULT_AGE_RANGE = (18, 80)

# A tuple applies to every money slot. A dict is per placeholder (MONEY, MONEY_2).
MoneyBounds = tuple[float, float]
MONEY_RANGES: dict[str, MoneyBounds | dict[str, MoneyBounds]] = {
    "email_drafting": (500, 150_000),
    "savings_rate": {"MONEY": (40_000, 180_000), "MONEY_2": (2_000, 40_000)},
    "raise_calculation": {"MONEY": (45_000, 200_000), "MONEY_2": (1_000, 20_000)},
}
DEFAULT_MONEY_RANGE: MoneyBounds = (100, 250_000)

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


def parse_slot(token: str) -> tuple[EntityType, int]:
    """Split ``NAME_2`` into ``("NAME", 2)``. A bare ``NAME`` is entity 1."""
    match = _SLOT_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"no sampler for placeholder {{{token}}}")
    label: EntityType = match.group(1)  # type: ignore[assignment]
    index = int(match.group(2) or "1")
    return label, index


def _age_window(template: PromptTemplate) -> tuple[int, int]:
    return AGE_WINDOWS.get(template.id, DEFAULT_AGE_RANGE)


def _money_bounds(family: str, token: str) -> MoneyBounds:
    spec = MONEY_RANGES.get(family, DEFAULT_MONEY_RANGE)
    if isinstance(spec, dict):
        return spec.get(token, spec.get("MONEY", DEFAULT_MONEY_RANGE))
    return spec


def sample_fill(
    token: str,
    template: PromptTemplate,
    rng: random.Random,
    fills: dict[str, Fill],
) -> Fill:
    """Sample one concrete value for a single placeholder token.

    ``token`` is the placeholder without braces, such as ``NAME``, ``AGE``,
    or ``MONEY_2``. The task family and template id choose the age window
    and the money range. ``rng`` is seeded by the caller, so the same seed
    reproduces the same value. ``fills`` holds values already chosen for
    earlier placeholders in this item: a later name avoids those names, and
    a later amount (``MONEY_2``) is kept below the salary in ``MONEY``.

    The returned ``Fill`` carries the surface text. Ages and amounts also
    carry the numeric value that was formatted into that text.

    ``parse_slot`` splits the token into a label and an entity number.
    ``NAME_2`` becomes ``("NAME", 2)`` and a bare ``NAME`` becomes
    ``("NAME", 1)``. Only the label is used here, to pick the name, age, or
    money sampler. The entity number is unused, which is why it is bound to
    ``_index``. ``fill_template`` reads that number itself and stores it on
    the span as ``entity_id`` (``name-2``).
    """
    # Label picks the sampler. fill_template applies the entity number.
    label, _index = parse_slot(token)
    if label == "NAME":
        # Only names are stored with surface_format="full".
        used = {fill.text for fill in fills.values() if fill.surface_format == "full"}
        name = rng.choice(NAMES)
        for _ in range(len(NAMES)):
            if name not in used:
                break
            name = rng.choice(NAMES)
        return Fill(text=name, surface_format="full")
    if label == "AGE":
        low, high = _age_window(template)
        age = rng.randint(low, high)
        # ages are "numeric" because they are not formatted.
        return Fill(text=str(age), value=float(age), surface_format="numeric")
    low, high = _money_bounds(template.task_family, token)
    salary = fills.get("MONEY")
    if token != "MONEY" and salary is not None and salary.value is not None:
        high = min(high, salary.value - 1)
        low = min(low, high)
    value = _round_money(math.exp(rng.uniform(math.log(max(low, 1)), math.log(max(high, 1)))))
    if salary is not None and salary.value is not None and token != "MONEY" and value >= salary.value:
        value = max(100, _round_money(salary.value / 2))
    fmt = rng.choice(MONEY_FORMATS)
    return Fill(text=_format_money(value, fmt), value=float(value), currency="USD", surface_format=fmt)


def fill_template(
    template: PromptTemplate, 
    rng: random.Random
) -> tuple[str, list[PiiSpan]]:
    """
    Substitute every placeholder in a prompt template with sampled values and return the resulting text and the list of PiiSpan objects showing the location and metadata of each substituted PII value.

    Args:
        template (PromptTemplate): The prompt template containing placeholders to fill.
        rng (random.Random): A random number generator instance used for deterministic sampling of placeholder values.

    Returns:
        tuple[str, list[PiiSpan]]:
            - The filled-in template as a string.
            - A list of PiiSpan objects, each representing the label, location, and metadata of a filled placeholder within the text.
    """
    """Substitute every placeholder and return the text with exact spans."""
    fills: dict[str, Fill] = {}
    parts: list[str] = []
    spans: list[PiiSpan] = []
    cursor = 0
    length = 0

    for match in PLACEHOLDER_RE.finditer(template.text):
        # Extract the placeholder token (e.g., "NAME", "AGE", "MONEY_2") from the match.
        token = match.group(1)
        
        # Parse the token into a label and an index.
        label, index = parse_slot(token)
        
        # Sample a fill for the placeholder if it hasn't been sampled yet.
        if token not in fills:
            fills[token] = sample_fill(token, template, rng, fills)
        fill = fills[token]

        # Extract the literal text between the end of the previous match (or start of string) and the start of this placeholder.
        # This ensures that all non-PII portions of the template are preserved in order.
        literal = template.text[cursor : match.start()]
        parts.append(literal)
        length += len(literal)

        # Create a PiiSpan object for the filled placeholder.
        # The span includes the label, start/end indices, the formatted text, the entity ID, the numeric value, the currency, and the surface format.
        spans.append(
            PiiSpan(
                label=label,
                start=length,
                end=length + len(fill.text),
                text=fill.text,
                entity_id=f"{label.lower()}-{index}",
                value=fill.value,
                currency=fill.currency,
                surface_format=fill.surface_format,
            )
        )
        # Append the formatted text of the filled placeholder to the parts list.
        parts.append(fill.text)
        length += len(fill.text)
        # Update the cursor to the end of the current placeholder.
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


def build_dataset(templates: list[PromptTemplate], fills_per_template: int, seed: int) -> list[DatasetItem]:
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
    parser.add_argument("--fills-per-template", type=int, default=5)
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

    logger.info("wrote %d items from %d templates -> %s", len(items), len(templates), args.out)


if __name__ == "__main__":
    main()
