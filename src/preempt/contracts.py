"""Span and recognizer types shared by the pipeline."""

from dataclasses import dataclass
from typing import Protocol


LABELS = ("NAME", "AGE", "MONEY", "EMAIL", "PHONE", "ADDRESS", "ID")

SANITIZER_ENTITY = {"Name": "NAME", "Money": "MONEY", "Age": "AGE"}


@dataclass(frozen=True)
class EntitySpan:
    """Exact character span in the source text."""

    start: int
    end: int
    text: str
    label: str
    confidence: float
    source: str


def validate_span(text: str, span: EntitySpan) -> None:
    """Raise ValueError when the span does not match the source text.

    Parameters
    ----------
    text : str
        Source text the span was taken from.
    span : EntitySpan
        Span to check.

    """
    if span.label not in LABELS:
        raise ValueError(f"unknown label: {span.label}")
    if span.start < 0 or span.end > len(text) or span.start >= span.end:
        raise ValueError(f"span offsets out of range: {span.start}:{span.end}")
    actual = text[span.start : span.end]
    if actual != span.text:
        raise ValueError(f"span text {span.text!r} does not match source slice {actual!r}")


class Recognizer(Protocol):
    """Finds PII spans and does not replace them."""

    def recognize(self, text: str) -> list[EntitySpan]:
        """Return exact character spans found in text."""
        ...
