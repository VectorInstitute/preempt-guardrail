"""Run preempt's ``Sanitizer`` on benchmark texts and report what changed and where.

Live mode finds PII with a ``Recognizer``. Oracle mode skips detection and uses
gold spans from the caller. Both then replace each value with the fake returned
by ``Sanitizer.encrypt_names`` / ``encrypt_age`` / ``encrypt_money``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Literal, cast

from models import ALL_ENTITY_TYPES, EntityType, Mechanism, Replacement, SanitizeResult, Substitution
from sanitizers.text_utils import (
    dedupe,
    detect_currency,
    nfc,
    numeric_core,
    parse_age,
    parse_money,
    rehydrate_names_from_map,
    replace_numeric_core,
    resolve_overlaps,
)

from preempt.contracts import EntitySpan, Recognizer, validate_span
from preempt.legacy import LegacyNERShim
from preempt.sanitizer import Sanitizer as PreemptSanitizer


# Benchmark label -> entity name used by preempt's NER.
_NER_ENTITIES: dict[EntityType, str] = {"NAME": "Name", "AGE": "Age", "MONEY": "Money"}

# {(label, original text): fake text} for one text.
# Example: {("NAME", "John Doe"): "Edward Smith"}
Fakes = dict[tuple[str, str], str]


class PreemptSanitizerAdapter:
    """Sanitize texts with preempt and return a ``SanitizeResult`` per text.

    Parameters
    ----------
    sanitizer
        A preempt ``Sanitizer``.
    recognizer
        Finds PII spans. Not used in oracle mode (when ``gold_spans`` is given).
    name_mechanism
        ``"fpe"`` (reversible) or ``"random"``.
    money_mechanism
        ``"mldp"`` (needs ``epsilon``) or ``"fpe"`` (reversible).
    """

    def __init__(
        self,
        sanitizer: PreemptSanitizer,
        recognizer: Recognizer,
        *,
        name_mechanism: Literal["fpe", "random"] = "fpe",
        money_mechanism: Literal["mldp", "fpe"] = "mldp",
    ) -> None:
        self.sanitizer = sanitizer
        self.ner = LegacyNERShim(recognizer)
        self.name_mechanism = name_mechanism
        self.money_mechanism = money_mechanism

    def sanitize(
        self,
        text: str,
        *,
        epsilon: float | None,
        seed: int,
        enabled_types: Iterable[str] = ALL_ENTITY_TYPES,
        gold_spans: Sequence[EntitySpan] | None = None,
    ) -> SanitizeResult:
        """Sanitize one text. Pass ``gold_spans`` to skip detection (oracle mode)."""
        return self.sanitize_batch(
            [text],
            epsilon=epsilon,
            seed=seed,
            enabled_types=enabled_types,
            gold_spans=None if gold_spans is None else [gold_spans],
        )[0]

    def sanitize_batch(
        self,
        texts: Sequence[str],
        *,
        epsilon: float | None,
        seed: int,
        enabled_types: Iterable[str] = ALL_ENTITY_TYPES,
        gold_spans: Sequence[Sequence[EntitySpan]] | None = None,
    ) -> list[SanitizeResult]:
        """Sanitize many texts.

        Each text is seeded with ``seed``, so results don't depend on batch order.
        """
        types = cast("set[EntityType]", set(enabled_types))
        if unknown := types - ALL_ENTITY_TYPES:
            raise ValueError(f"Unknown entity types: {sorted(unknown)}")
        if gold_spans is not None and len(gold_spans) != len(texts):
            raise ValueError("gold_spans must have one entry per text.")

        start = time.perf_counter()
        gold = None
        if gold_spans is not None:
            gold = []
            for spans in gold_spans:
                filtered_spans = [s for s in spans if s.label in types]
                gold.append(resolve_overlaps(filtered_spans))
       
            for text, spans in zip(texts, gold):
                for span in spans:
                    validate_span(text, span)
        detected = self._detect(texts, types) if gold is None else [_group_by_label(spans) for spans in gold]

        rows = []
        for i, text in enumerate(texts):
            fakes, warnings = self._encrypt(detected[i], seed=seed, epsilon=epsilon)
            # Live mode: the spans are wherever the detected values occur in the text.
            spans = gold[i] if gold is not None else resolve_overlaps(_locate_spans(text, fakes))
            replacements, substitutions = self._records(text, spans, fakes)
            rows.append((text, _apply_replacements(text, replacements), spans, replacements, substitutions, warnings))
        latency_ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)

        return [
            SanitizeResult(
                original_text=text,
                text=sanitized,
                spans=spans,
                replacements=replacements,
                substitutions=substitutions,
                epsilon=epsilon,
                seed=seed,
                enabled_types=types,
                oracle=gold is not None,
                latency_ms=latency_ms,
                warnings=warnings,
            )
            for text, sanitized, spans, replacements, substitutions, warnings in rows
        ]

    def rehydrate_names(self, text: str, substitutions: Sequence[Substitution]) -> str:
        """Restore original names in ``text``, e.g. an LLM reply."""
        return rehydrate_names_from_map(text, substitutions)

    def _detect(self, texts: Sequence[str], types: set[EntityType]) -> list[dict[str, list[str]]]:
        """Run NER once per enabled type; return ``{label: values}`` per text."""
        detected: list[dict[str, list[str]]] = [{} for _ in texts]
        for label, entity in _NER_ENTITIES.items():
            if label in types:
                values_per_text = self.ner.extract(list(texts), entity_type=entity)[entity]
                for row, values in zip(detected, values_per_text):
                    row[label] = dedupe(values or [])
        return detected

    def _encrypt(self, detected: dict[str, list[str]], *, seed: int, epsilon: float | None) -> tuple[Fakes, list[str]]:
        """Get a fake for each detected value in one text, plus warnings."""
        self.sanitizer.reseed(seed)
        fakes: Fakes = {}
        warnings: list[str] = []
        self._encrypt_names(detected.get("NAME", []), fakes, warnings)
        self._encrypt_numbers("AGE", detected.get("AGE", []), epsilon, fakes, warnings)
        self._encrypt_numbers("MONEY", detected.get("MONEY", []), epsilon, fakes, warnings)
        return fakes, warnings

    def _encrypt_names(self, names: list[str], fakes: Fakes, warnings: list[str]) -> None:
        if not names:
            return
        try:
            mapping, _ = self.sanitizer.encrypt_names([names], use_fpe=self.name_mechanism == "fpe")
        except Exception as err:  # noqa: BLE001 -- report and keep going, like other failed values
            warnings.append(f"NAME encryption failed: {err!r}")
            return
        fakes.update({("NAME", nfc(name)): fake for name, fake in mapping[0].items() if fake})

    def _encrypt_numbers(
        self, label: str, values: list[str], epsilon: float | None, fakes: Fakes, warnings: list[str]
    ) -> None:
        """Encrypt only the number inside each value, e.g. "85,000" in "$85,000"."""
        numbers = {}
        for value in values:
            if match := numeric_core(value):
                numbers[value] = match.group(0)
            else:
                warnings.append(f"{label} not parseable: {value!r}")
        if not numbers:
            return
        if epsilon is None and (label == "AGE" or self.money_mechanism == "mldp"):
            raise ValueError(f"epsilon is required to sanitize {label} with m-LDP.")

        if label == "AGE":
            mapping = self.sanitizer.encrypt_age([list(numbers.values())], epsilon=epsilon)
        else:
            use_fpe = self.money_mechanism == "fpe"
            mapping = self.sanitizer.encrypt_money([list(numbers.values())], use_fpe=use_fpe, epsilon=epsilon)
        for value, number in numbers.items():
            if fake := mapping[0].get(number):
                fakes[(label, value)] = replace_numeric_core(value, fake)
            else:
                warnings.append(f"{label} encryption failed: {value!r}")

    def _records(
        self, text: str, spans: list[EntitySpan], fakes: Fakes
    ) -> tuple[list[Replacement], list[Substitution]]:
        """One ``Replacement`` per span, and one ``Substitution`` per unique value."""
        replacements: list[Replacement] = []
        substitutions: dict[tuple[str, str], Substitution] = {}
        for span in spans:
            key = (span.label, nfc(span.text))
            fake = fakes.get(key)
            if fake is None:
                continue
            replacements.append(Replacement(span=span, replacement=fake))
            if key not in substitutions:
                substitutions[key] = self._substitution(text, span, fake)
        return replacements, list(substitutions.values())

    def _substitution(self, text: str, span: EntitySpan, fake: str) -> Substitution:
        """Describe one replaced value, with numbers and currency parsed for metrics."""
        mechanisms: dict[str, Mechanism] = {"NAME": self.name_mechanism, "AGE": "mldp", "MONEY": self.money_mechanism}
        mechanism = mechanisms[span.label]
        original_value = fake_value = currency = None
        if span.label == "AGE":
            original_value, fake_value = parse_age(span.text), parse_age(fake)
        elif span.label == "MONEY":
            # Nearby text holds suffixes ("85k") and currency ("USD").
            before, after = text[max(span.start - 4, 0) : span.start], text[span.end : span.end + 12]
            original_value, fake_value = parse_money(span.text + after), parse_money(fake + after)
            currency = detect_currency(before + span.text + after[:5])
        return Substitution(
            label=cast("EntityType", span.label),
            original=span.text,
            surrogate=fake,
            original_value=original_value,
            surrogate_value=fake_value,
            currency=currency,
            mechanism=mechanism,
            reversible=mechanism == "fpe",
        )


def _locate_spans(text: str, fakes: Fakes) -> list[EntitySpan]:
    """Find every occurrence of each original value in ``text``."""
    normalized = nfc(text)
    spans = []
    for label, original in fakes:
        start = 0
        while (idx := normalized.find(original, start)) != -1:
            end = idx + len(original)
            spans.append(
                EntitySpan(start=idx, end=end, text=text[idx:end], label=label, confidence=1.0, source="preempt-ner")
            )
            start = end
    return spans


def _group_by_label(spans: Iterable[EntitySpan]) -> dict[str, list[str]]:
    """Return ``{label: unique span texts}``."""
    grouped: dict[str, list[str]] = {}
    for span in spans:
        grouped.setdefault(span.label, []).append(span.text)
    return {label: dedupe(values) for label, values in grouped.items()}


def _apply_replacements(text: str, replacements: list[Replacement]) -> str:
    """Apply replacements right to left so earlier offsets stay valid."""
    for r in sorted(replacements, key=lambda r: r.span.start, reverse=True):
        text = text[: r.span.start] + r.replacement + text[r.span.end :]
    return text
