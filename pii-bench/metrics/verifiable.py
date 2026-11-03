"""Deterministic scoring for verifiable items that end with ``FINAL_ANSWER: <x>``.

No judge is involved: the answer line is parsed and compared exactly
(categorical labels) or within a relative tolerance (numbers).
"""

from __future__ import annotations

import re


# Allowed labels per task family. Families not listed are parsed as numbers.
ANSWER_LABELS: dict[str, frozenset[str]] = {
    "retirement_eligibility": frozenset({"ELIGIBLE", "NOT_ELIGIBLE"}),
}

_FINAL_RE = re.compile(r"FINAL[_ ]ANSWER\W*?:\s*([^\n]+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_label(raw: str) -> str:
    cleaned = re.sub(r"[*`\"'<>\[\]().]", " ", raw).strip()
    return re.sub(r"[\s\-]+", "_", cleaned).upper()


def parse_final_answer(response: str, labels: frozenset[str] | None = None) -> str | float | None:
    """Return the last valid ``FINAL_ANSWER`` value, or ``None`` if there isn't one.

    Matches are scanned from the end so an echoed instruction such as
    ``FINAL_ANSWER: <ELIGIBLE or NOT_ELIGIBLE>`` earlier in the text is ignored.
    """
    for match in reversed(_FINAL_RE.findall(response or "")):
        if labels is not None:
            label = _normalize_label(match)
            if label in labels:
                return label
        elif number := _NUMBER_RE.search(match):
            return float(number.group(0).replace(",", ""))
    return None


def answers_agree(a: str | float | None, b: str | float | None, *, rel_tol: float = 0.01) -> bool | None:
    """Exact match for labels, relative tolerance for numbers; ``None`` if either failed to parse."""
    if a is None or b is None:
        return None
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-9)
    return a == b
