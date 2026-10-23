"""Tests for filling templates into dataset items."""

import random

from dataset.build import build_dataset, fill_template
from models import PromptTemplate


def _template(text: str, slots: list[str], family: str = "retirement_eligibility") -> PromptTemplate:
    return PromptTemplate(
        id=f"{family}-000",
        task_family=family,
        task_type="eligibility",
        pii_dependency="semantic",
        verifiable=True,
        text=text,
        slots=slots,  # type: ignore[arg-type]
    )


def test_spans_index_into_filled_text() -> None:
    template = _template("{NAME} is {AGE}. Can {NAME} withdraw {MONEY}?", ["NAME", "AGE", "MONEY"])
    text, spans = fill_template(template, random.Random(0))

    assert "{" not in text
    assert len(spans) == 4
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_repeated_placeholder_reuses_value_and_entity() -> None:
    template = _template("{NAME} asked if {NAME} qualifies at {AGE}.", ["NAME", "AGE"])
    _, spans = fill_template(template, random.Random(1))

    names = [s for s in spans if s.label == "NAME"]
    assert names[0].text == names[1].text
    assert names[0].entity_id == names[1].entity_id


def test_age_respects_family_range() -> None:
    template = _template("{NAME} is {AGE}.", ["NAME", "AGE"])
    for seed in range(50):
        _, spans = fill_template(template, random.Random(seed))
        age = next(s for s in spans if s.label == "AGE")
        assert age.value is not None
        assert 45 <= age.value <= 70


def test_build_is_deterministic() -> None:
    templates = [_template("{NAME} is {AGE} and earns {MONEY}.", ["NAME", "AGE", "MONEY"])]
    first = build_dataset(templates, fills_per_template=5, seed=1234)
    second = build_dataset(templates, fills_per_template=5, seed=1234)

    assert [i.text for i in first] == [i.text for i in second]
