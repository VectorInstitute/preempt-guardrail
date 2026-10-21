"""Text helpers: Unicode, spans, number and currency parsing, name rehydration."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from models import Substitution

from preempt.contracts import EntitySpan


def nfc(s: str) -> str:
    """NFC-normalize a string."""
    return unicodedata.normalize("NFC", s)


def resolve_overlaps(spans: Iterable[EntitySpan]) -> list[EntitySpan]:
    """Keep non-overlapping spans, preferring the earliest and then the longest."""
    kept: list[EntitySpan] = []
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if kept and span.start < kept[-1].end:
            continue
        kept.append(span)
    return kept


def dedupe(values: Iterable[str]) -> list[str]:
    """NFC-normalize, strip, and de-duplicate strings, preserving order."""
    seen: dict[str, None] = {}
    for v in values:
        if v and v.strip():
            seen.setdefault(nfc(v.strip()), None)
    return list(seen)


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_MULTIPLIER_RE = re.compile(r"^\s*(k|mm|m|bn|b|thousand|million|billion)\b", re.IGNORECASE)
_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
_CURRENCY_CODE_RE = re.compile(r"\b(USD|CAD|EUR|JPY|GBP|AUD|CHF|INR|CNY)\b")
_CURRENCY_PREFIXES = [("US$", "USD"), ("CA$", "CAD"), ("C$", "CAD"), ("A$", "AUD")]
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}


def numeric_core(s: str) -> re.Match[str] | None:
    """Return the first number-like token in ``s``."""
    return _NUMBER_RE.search(s)


def replace_numeric_core(original: str, new_core: str) -> str:
    """Swap the first number in ``original`` for ``new_core``, keeping "$", "k", etc."""
    m = numeric_core(original)
    if m is None:
        return new_core
    return original[: m.start()] + new_core + original[m.end() :]


def parse_age(s: str) -> float | None:
    """Parse the numeric value of an age mention."""
    m = numeric_core(s)
    return float(m.group(0).replace(",", "")) if m else None


def parse_money(s: str) -> float | None:
    """Parse a money mention into a float, honouring k/m/b-style multipliers."""
    m = numeric_core(s)
    if m is None:
        return None
    value = float(m.group(0).replace(",", ""))
    suffix = _MULTIPLIER_RE.match(s[m.end() :])
    if suffix:
        value *= _MULTIPLIERS[suffix.group(1).lower()]
    return value


def detect_currency(s: str) -> str | None:
    """Best-effort ISO currency code for a money mention (with surrounding context)."""
    code = _CURRENCY_CODE_RE.search(s)
    if code:
        return code.group(1)
    for prefix, iso in _CURRENCY_PREFIXES:
        if prefix in s:
            return iso
    for symbol, iso in _CURRENCY_SYMBOLS.items():
        if symbol in s:
            return iso
    return None


def rehydrate_names_from_map(text: str, substitutions: Sequence[Substitution]) -> str:
    """Replace fake names in ``text`` (e.g. an LLM reply) with the original names.

    Full fake names and their single words both match ("Maria", "Garcia's"),
    ignoring case. Everything is replaced in one pass, so a restored name is
    never replaced again.

    For examples, see ``test_rehydrate_full_names_tokens_and_possessives`` and
    ``test_rehydrate_does_not_chain`` in ``tests/test_sanitizers.py``.
    """
    pairs: dict[str, str] = {}
    full = [(nfc(s.surrogate), nfc(s.original)) for s in substitutions if s.label == "NAME"]
    for surrogate, original in full:
        pairs.setdefault(surrogate.lower(), original)

    token_pairs: dict[str, str] = {}
    ambiguous: set[str] = set()
    for surrogate, original in full:
        s_tokens, o_tokens = surrogate.split(), original.split()
        if len(s_tokens) < 2 or len(s_tokens) != len(o_tokens):
            continue
        for s_tok, o_tok in zip(s_tokens, o_tokens):
            key = s_tok.lower()
            if token_pairs.get(key, o_tok) != o_tok:
                ambiguous.add(key)
            token_pairs.setdefault(key, o_tok)
    for key, original in token_pairs.items():
        if key not in ambiguous:
            pairs.setdefault(key, original)

    if not pairs:
        return text
    alternation = "|".join(re.escape(k) for k in sorted(pairs, key=len, reverse=True))
    pattern = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)
    return pattern.sub(lambda m: pairs.get(nfc(m.group(0)).lower(), m.group(0)), nfc(text))
