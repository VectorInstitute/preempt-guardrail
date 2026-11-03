"""Tests for FINAL_ANSWER parsing and comparison."""

from metrics.verifiable import answers_agree, parse_final_answer


LABELS = frozenset({"ELIGIBLE", "NOT_ELIGIBLE"})


def test_parses_plain_and_markdown_labels() -> None:
    assert parse_final_answer("Reasoning...\nFINAL_ANSWER: ELIGIBLE", LABELS) == "ELIGIBLE"
    assert parse_final_answer("**FINAL_ANSWER:** `NOT_ELIGIBLE`.", LABELS) == "NOT_ELIGIBLE"
    assert parse_final_answer("FINAL ANSWER: not eligible", LABELS) == "NOT_ELIGIBLE"


def test_ignores_echoed_instruction_and_uses_last_valid() -> None:
    text = "You asked me to end with FINAL_ANSWER: <ELIGIBLE or NOT_ELIGIBLE>.\n...\nFINAL_ANSWER: ELIGIBLE"
    assert parse_final_answer(text, LABELS) == "ELIGIBLE"


def test_missing_or_invalid_answer_is_none() -> None:
    assert parse_final_answer("I think they qualify.", LABELS) is None
    assert parse_final_answer("FINAL_ANSWER: MAYBE", LABELS) is None


def test_numeric_answers() -> None:
    assert parse_final_answer("FINAL_ANSWER: $12,500.50") == 12500.5
    assert answers_agree(100.0, 100.5) is True
    assert answers_agree(100.0, 110.0) is False


def test_agreement_none_when_unparsed() -> None:
    assert answers_agree("ELIGIBLE", None) is None
    assert answers_agree("ELIGIBLE", "NOT_ELIGIBLE") is False
