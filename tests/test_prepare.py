"""Tests for English span preparation."""

from collections import Counter

import pytest

from training.prepare import entities_from_mask, mask_dicts, split_name


def test_list_mask_maps_firstname_and_keeps_the_slice() -> None:
    """A FIRSTNAME span becomes NAME with the original offsets."""
    text = "Ben paid."
    unmapped: Counter[str] = Counter()
    entities, reason = entities_from_mask(
        text,
        [{"value": "Ben", "start": 0, "end": 3, "label": "FIRSTNAME"}],
        unmapped,
    )
    assert reason is None
    assert entities == [{"start": 0, "end": 3, "label": "NAME"}]
    assert unmapped == Counter()


def test_unmapped_label_is_not_an_entity() -> None:
    """Labels outside the map are counted and left out of the example."""
    unmapped: Counter[str] = Counter()
    entities, reason = entities_from_mask(
        "Use secret.",
        [{"value": "secret", "start": 4, "end": 10, "label": "PASSWORD"}],
        unmapped,
    )
    assert reason is None
    assert entities == []
    assert unmapped["PASSWORD"] == 1


def test_text_mismatch_rejects_the_row() -> None:
    """A span whose value is not the source slice drops the whole row."""
    entities, reason = entities_from_mask(
        "Ben paid.",
        [{"value": "Ann", "start": 0, "end": 3, "label": "FIRSTNAME"}],
        Counter(),
    )
    assert entities == []
    assert reason == "text_mismatch"


def test_overlap_rejects_the_row() -> None:
    """Overlapping canonical spans drop the row."""
    entities, reason = entities_from_mask(
        "Ben",
        [
            {"value": "Ben", "start": 0, "end": 3, "label": "FIRSTNAME"},
            {"value": "en", "start": 1, "end": 3, "label": "LASTNAME"},
        ],
        Counter(),
    )
    assert entities == []
    assert reason == "overlap"


def test_columnar_mask_matches_the_list_form() -> None:
    """A columnar privacy mask is read the same way as a list of objects."""
    text = "Ben"
    columnar = {"start": [0], "end": [3], "label": ["FIRSTNAME"], "value": ["Ben"]}
    assert mask_dicts(columnar) == mask_dicts([{"start": 0, "end": 3, "label": "FIRSTNAME", "value": "Ben"}])
    entities, reason = entities_from_mask(text, columnar, Counter())
    assert reason is None
    assert entities == [{"start": 0, "end": 3, "label": "NAME"}]


def test_bad_mask_rejects_the_row() -> None:
    """A mask that is not a list or columnar object drops the row."""
    entities, reason = entities_from_mask("Ben", 5, Counter())
    assert entities == []
    assert reason == "bad_mask"


def test_split_is_stable() -> None:
    """The same example id always lands in the same split."""
    assert split_name(165761, 0) == split_name(165761, 0)
    assert split_name(165761, 0) in {"train", "validation", "test"}


def test_columnar_length_mismatch_is_a_bad_mask() -> None:
    """Columns of different lengths are rejected."""
    with pytest.raises(ValueError, match="differ in length"):
        mask_dicts({"start": [0, 1], "end": [1], "label": ["AGE"], "value": ["1"]})
