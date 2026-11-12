"""Tests for span contracts, BIO alignment, and the sanitizer shim."""

import pytest

from preempt.bio import IGNORE_TAG_ID, TAG_TO_ID, spans_from_tags, tags_from_spans
from preempt.contracts import (
    EntitySpan,
    validate_span,
)
from preempt.legacy import LegacyNERShim


def test_validate_span_accepts_exact_slice() -> None:
    """A span whose text is the source slice is valid."""
    validate_span("Ben paid.", EntitySpan(0, 3, "Ben", "NAME", 1.0, "test"))


def test_validate_span_rejects_mismatch() -> None:
    """A span that does not match the source slice is rejected."""
    with pytest.raises(ValueError, match="does not match"):
        validate_span("Ben paid.", EntitySpan(0, 3, "Be", "NAME", 1.0, "test"))


def test_validate_span_rejects_unknown_label() -> None:
    """Labels outside the canonical set are rejected."""
    with pytest.raises(ValueError, match="unknown label"):
        validate_span("Ben", EntitySpan(0, 3, "Ben", "PERSON", 1.0, "test"))


def test_bio_round_trip() -> None:
    """Tags decoded from character spans slice back to the original text."""
    text = "Ben paid $200."
    offsets = [(0, 0), (0, 3), (4, 8), (9, 13), (0, 0)]
    spans = [(0, 3, "NAME"), (9, 13, "MONEY")]
    tags = tags_from_spans(offsets, spans)
    assert tags[0] == IGNORE_TAG_ID
    assert tags[1] == TAG_TO_ID["B-NAME"]
    assert tags[2] == TAG_TO_ID["O"]
    assert tags[3] == TAG_TO_ID["B-MONEY"]
    decoded, dropped = spans_from_tags(text, offsets, tags, [1.0] * len(tags), "test")
    assert dropped == 0
    assert [(span.start, span.end, span.label, span.text) for span in decoded] == [
        (0, 3, "NAME", "Ben"),
        (9, 13, "MONEY", "$200"),
    ]


def test_multi_token_span_uses_i_tag() -> None:
    """The second token of one entity is an I- tag."""
    text = "Anne Marie"
    offsets = [(0, 4), (5, 10)]
    tags = tags_from_spans(offsets, [(0, 10, "NAME")])
    assert tags == [TAG_TO_ID["B-NAME"], TAG_TO_ID["I-NAME"]]
    decoded, dropped = spans_from_tags(text, offsets, tags, [0.8, 0.6], "test")
    assert dropped == 0
    assert decoded[0].text == "Anne Marie"
    assert decoded[0].confidence == 0.6


def test_overlapping_spans_are_rejected() -> None:
    """Two entities that cover the same token are an error."""
    with pytest.raises(ValueError, match="overlapping"):
        tags_from_spans([(0, 3)], [(0, 3, "NAME"), (0, 2, "ID")])


def test_stray_i_tag_is_dropped() -> None:
    """An I- tag with no open span is dropped rather than repaired."""
    spans, dropped = spans_from_tags("Ben", [(0, 3)], [TAG_TO_ID["I-NAME"]], [0.5], "test")
    assert spans == []
    assert dropped == 1


def test_shim_extract_shape() -> None:
    """extract() returns the string lists Sanitizer.encrypt reads."""

    class _ListRecognizer:
        """Returns a fixed name and a money span when the text contains them."""

        def recognize(self, text: str) -> list[EntitySpan]:
            """Return spans for a leading name and a four-character amount."""
            spans = []
            if text.startswith("Ben"):
                spans.append(EntitySpan(0, 3, "Ben", "NAME", 1.0, "fake"))
            if "$" in text:
                start = text.index("$")
                spans.append(EntitySpan(start, start + 4, text[start : start + 4], "MONEY", 1.0, "fake"))
            return spans

    shim = LegacyNERShim(_ListRecognizer())
    names = shim.extract(["Ben paid $200.", "Hello"], entity_type="Name", batch_size=4)
    money = shim.extract(["Ben paid $200.", "Hello"], entity_type="Money")
    assert names == {"Name": [["Ben"], []]}
    assert money["Money"] == [["$200"], []]


def test_shim_joins_names_separated_only_by_whitespace() -> None:
    """Ben and Parker become one name. A word between names keeps them apart."""

    class _SplitNameRecognizer:
        """Returns each name token as its own span."""

        def recognize(self, text: str) -> list[EntitySpan]:
            """Return the four name tokens in the demo sentence."""
            del text
            return [
                EntitySpan(0, 3, "Ben", "NAME", 1.0, "fake"),
                EntitySpan(4, 10, "Parker", "NAME", 1.0, "fake"),
                EntitySpan(15, 19, "John", "NAME", 1.0, "fake"),
                EntitySpan(20, 23, "Doe", "NAME", 1.0, "fake"),
            ]

    text = "Ben Parker and John Doe went to the bank."
    names = LegacyNERShim(_SplitNameRecognizer()).extract([text], entity_type="Name")
    assert names == {"Name": [["Ben Parker", "John Doe"]]}


def test_shim_rejects_unknown_entity_type() -> None:
    """Entity types other than Name, Money, and Age are rejected."""

    class _EmptyRecognizer:
        """Returns no spans."""

        def recognize(self, text: str) -> list[EntitySpan]:
            """Return no spans."""
            del text
            return []

    with pytest.raises(ValueError, match="unsupported entity type"):
        LegacyNERShim(_EmptyRecognizer()).extract(["Ben"], entity_type="Email")
