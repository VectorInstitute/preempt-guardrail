"""Tests for PreemptSanitizerAdapter, in both oracle mode and live detection."""

import pytest
from models import SanitizeResult, Substitution
from sanitizers.preempt_adapter import PreemptSanitizerAdapter
from sanitizers.text_utils import rehydrate_names_from_map

from preempt.contracts import EntitySpan
from preempt.sanitizer import Sanitizer as PreemptSanitizer


TEXT = "Sarah Lopez, 52, earns $85,000 a year. Sarah wants to retire at 60."


def gold(text: str, label: str, needle: str, occurrence: int = 0, source: str = "gold") -> EntitySpan:
    """Build a span for the ``occurrence``-th match of ``needle`` in ``text``."""
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(needle, start + 1)
    return EntitySpan(start=start, end=start + len(needle), text=needle, label=label, confidence=1.0, source=source)


GOLD = [
    gold(TEXT, "NAME", "Sarah Lopez"),
    gold(TEXT, "AGE", "52"),
    # Real recognizers (see spans.py's normalize_span/currency_spans) exclude
    # the currency symbol from the MONEY span, leaving it for the cipher to
    # pass through untouched.
    gold(TEXT, "MONEY", "85,000"),
    gold(TEXT, "NAME", "Sarah", occurrence=1),
]


class StubRecognizer:
    """Recognizer stand-in that returns a fixed list of spans for any text."""

    def __init__(self, spans: list[EntitySpan]):
        self.spans = spans

    def recognize(self, text: str) -> list[EntitySpan]:
        """Return the fixed spans this stub was built with, ignoring ``text``."""
        return list(self.spans)


class NeverCalledRecognizer:
    """Recognizer stand-in that fails if detection is invoked.

    Used to prove oracle mode (``gold_spans``) truly bypasses the recognizer.
    """

    def recognize(self, text: str) -> list[EntitySpan]:
        """Fail: detection should never run when gold_spans are supplied."""
        raise AssertionError("recognizer.recognize should not be called in oracle mode")


@pytest.fixture(scope="module")
def preempt_sanitizer() -> PreemptSanitizer:
    """Share one preempt Sanitizer across tests (NameDataset load is slow).

    ``ner_model`` is unused: the adapter detects via a separate
    ``Recognizer`` and calls ``encrypt_names``/``encrypt_age``/``encrypt_money``
    directly, so a plain ``Sanitizer(None)`` is enough.
    """
    return PreemptSanitizer(None)


@pytest.fixture
def oracle_adapter(preempt_sanitizer: PreemptSanitizer) -> PreemptSanitizerAdapter:
    """Adapter whose recognizer must never be called.

    Every test using this fixture passes ``gold_spans``.
    """
    return PreemptSanitizerAdapter(preempt_sanitizer, NeverCalledRecognizer())


def test_oracle_mode_skips_detection_and_replaces_in_place(oracle_adapter: PreemptSanitizerAdapter) -> None:
    """gold_spans bypasses the recognizer and replacements use original-text offsets."""
    result = oracle_adapter.sanitize(TEXT, epsilon=1.0, seed=0, gold_spans=GOLD)

    assert isinstance(result, SanitizeResult)
    assert result.oracle
    assert [r.span.label for r in result.replacements] == ["NAME", "AGE", "MONEY", "NAME"]
    for r in result.replacements:
        assert TEXT[r.span.start : r.span.end] == r.span.text
    assert "Sarah" not in result.text  # FPE names always change; m-LDP noise occasionally lands on the same number
    assert {s.label for s in result.substitutions} == {"NAME", "AGE", "MONEY"}


def test_oracle_parses_values_and_currency(oracle_adapter: PreemptSanitizerAdapter) -> None:
    """Numeric values and currency are parsed for AGE and MONEY substitutions."""
    result = oracle_adapter.sanitize(TEXT, epsilon=1.0, seed=0, gold_spans=GOLD)
    money = next(s for s in result.substitutions if s.label == "MONEY")
    age = next(s for s in result.substitutions if s.label == "AGE")

    assert money.original_value == 85000 and money.currency == "USD" and money.surrogate_value is not None
    assert age.original_value == 52 and age.surrogate_value is not None


def test_oracle_same_seed_same_draw(oracle_adapter: PreemptSanitizerAdapter) -> None:
    """The same seed reproduces the same draw."""
    a = oracle_adapter.sanitize(TEXT, epsilon=1.0, seed=7, gold_spans=GOLD)
    b = oracle_adapter.sanitize(TEXT, epsilon=1.0, seed=7, gold_spans=GOLD)
    assert a.text == b.text


def test_enabled_types_ablation(oracle_adapter: PreemptSanitizerAdapter) -> None:
    """Only enabled entity types are substituted."""
    result = oracle_adapter.sanitize(TEXT, epsilon=1.0, seed=0, enabled_types={"MONEY"}, gold_spans=GOLD)
    assert {r.span.label for r in result.replacements} == {"MONEY"}
    assert "Sarah Lopez" in result.text


def test_gold_span_mismatch_raises(oracle_adapter: PreemptSanitizerAdapter) -> None:
    """Gold spans that don't match the text are rejected via ``validate_span``."""
    bad = EntitySpan(start=0, end=5, text="Lopez", label="NAME", confidence=1.0, source="gold")
    with pytest.raises(ValueError):
        oracle_adapter.sanitize(TEXT, epsilon=1.0, seed=0, gold_spans=[bad])


def test_result_round_trips_through_json(oracle_adapter: PreemptSanitizerAdapter) -> None:
    """Results survive a JSON round trip unchanged."""
    result = oracle_adapter.sanitize(TEXT, epsilon=1.0, seed=0, gold_spans=GOLD)
    assert SanitizeResult.model_validate_json(result.model_dump_json()) == result


def test_rehydrate_full_names_tokens_and_possessives() -> None:
    """Full names, single tokens, casing, and possessives are all restored."""
    subs = [
        Substitution(label="NAME", original="Sarah Lopez", surrogate="Maria Garcia", mechanism="fpe", reversible=True)
    ]
    response = "MARIA GARCIA should talk to Maria. Garcia's plan is solid."
    assert rehydrate_names_from_map(response, subs) == "Sarah Lopez should talk to Sarah. Lopez's plan is solid."


def test_rehydrate_does_not_chain() -> None:
    """A restored original is never replaced again by another mapping."""
    subs = [
        Substitution(label="NAME", original="Bob", surrogate="Alice", mechanism="fpe", reversible=True),
        Substitution(label="NAME", original="Carol", surrogate="Bob", mechanism="fpe", reversible=True),
    ]
    assert rehydrate_names_from_map("Alice met Bob", subs) == "Bob met Carol"


def test_preempt_adapter_end_to_end(preempt_sanitizer: PreemptSanitizer) -> None:
    """Live detection: real FPE names and m-LDP numbers reach a SanitizeResult."""
    recognizer = StubRecognizer(
        [
            gold(TEXT, "NAME", "Sarah Lopez", source="stub"),
            gold(TEXT, "AGE", "52", source="stub"),
            gold(TEXT, "MONEY", "85,000", source="stub"),
            gold(TEXT, "NAME", "Sarah", occurrence=1, source="stub"),
        ]
    )
    adapter = PreemptSanitizerAdapter(preempt_sanitizer, recognizer)
    result = adapter.sanitize(TEXT, epsilon=1.0, seed=3)

    assert not result.oracle
    # Spans are rebuilt by locating substituted strings in the text (see
    # preempt_adapter._locate_spans), not passed through from the recognizer.
    assert all(s.source == "preempt-ner" for s in result.spans)
    labels = [r.span.label for r in result.replacements]
    assert labels.count("NAME") == 2  # full name span + standalone "Sarah" span
    assert "AGE" in labels and "MONEY" in labels
    assert "Sarah" not in result.text

    money = next(s for s in result.substitutions if s.label == "MONEY")
    assert money.mechanism == "mldp" and money.currency == "USD"

    name = next(s for s in result.substitutions if s.original == "Sarah Lopez")
    assert name.mechanism == "fpe" and name.reversible
    response = f"{name.surrogate} can retire comfortably."
    assert adapter.rehydrate_names(response, result.substitutions) == "Sarah Lopez can retire comfortably."

    again = adapter.sanitize(TEXT, epsilon=1.0, seed=3)
    assert again.text == result.text


def test_preempt_adapter_requires_epsilon_for_mldp(preempt_sanitizer: PreemptSanitizer) -> None:
    """m-LDP types need epsilon; disabling them makes epsilon optional."""
    recognizer = StubRecognizer([gold(TEXT, "MONEY", "85,000", source="stub")])
    adapter = PreemptSanitizerAdapter(preempt_sanitizer, recognizer)
    with pytest.raises(ValueError):
        adapter.sanitize(TEXT, epsilon=None, seed=0, enabled_types={"MONEY"})
    assert adapter.sanitize(TEXT, epsilon=None, seed=0, enabled_types=set()).text == TEXT
