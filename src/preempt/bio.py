"""Convert between character spans and BIO tags.

A token belongs to an entity when its start offset falls in ``[start, end)``.
Empty offsets, used for special tokens and padding, are ignored.
"""

from preempt.contracts import LABELS, EntitySpan, validate_span


IGNORE_TAG_ID = -100


def _build_tag_names() -> tuple[str, ...]:
    names = ["O"]
    for label in LABELS:
        names.append(f"B-{label}")
        names.append(f"I-{label}")
    return tuple(names)


TAG_NAMES = _build_tag_names()
TAG_TO_ID = {name: index for index, name in enumerate(TAG_NAMES)}
ID_TO_TAG = dict(enumerate(TAG_NAMES))


def tags_from_spans(
    offset_mapping: list[tuple[int, int]],
    spans: list[tuple[int, int, str]],
) -> list[int]:
    """Return a BIO tag id for each token.

    Parameters
    ----------
    offset_mapping : list[tuple[int, int]]
        ``(start, end)`` character offsets for each token.
    spans : list[tuple[int, int, str]]
        ``(start, end, label)`` character spans. Spans must not overlap.

    Returns
    -------
    list[int]
        Tag id per token. Empty offsets are ``-100``.

    """
    tags: list[int] = []
    active: tuple[int, int, str] | None = None
    for token_start, token_end in offset_mapping:
        if token_start == token_end:
            tags.append(IGNORE_TAG_ID)
            continue
        matches = [span for span in spans if span[0] <= token_start < span[1]]
        if len(matches) > 1:
            raise ValueError(f"overlapping spans at offset {token_start}")
        if not matches:
            tags.append(TAG_TO_ID["O"])
            active = None
            continue
        start, end, label = matches[0]
        tag_name = f"I-{label}" if active == (start, end, label) else f"B-{label}"
        if tag_name not in TAG_TO_ID:
            raise ValueError(f"unknown label: {label}")
        tags.append(TAG_TO_ID[tag_name])
        active = (start, end, label)
    return tags


def spans_from_tags(
    text: str,
    offset_mapping: list[tuple[int, int]],
    tag_ids: list[int],
    confidences: list[float],
    source: str,
) -> tuple[list[EntitySpan], int]:
    """Decode BIO tags into character spans.

    An ``I-`` tag with no open span of the same label is counted and dropped.
    Spans whose slice does not match the source text are counted and dropped.

    Parameters
    ----------
    text : str
        Original text.
    offset_mapping : list[tuple[int, int]]
        ``(start, end)`` character offsets for each token.
    tag_ids : list[int]
        BIO tag id per token.
    confidences : list[float]
        Score per token, parallel to ``tag_ids``.
    source : str
        Value stored on each emitted span.

    Returns
    -------
    tuple[list[EntitySpan], int]
        Accepted spans, and how many tag pieces were dropped.

    """
    if not (len(offset_mapping) == len(tag_ids) == len(confidences)):
        raise ValueError("offsets, tags, and confidences differ in length")

    spans: list[EntitySpan] = []
    dropped = 0
    open_label: str | None = None
    open_start = 0
    open_end = 0
    open_scores: list[float] = []

    def close() -> None:
        nonlocal dropped, open_label
        if open_label is None:
            return
        label = open_label
        start = open_start
        end = open_end
        scores = list(open_scores)
        open_label = None
        open_scores.clear()
        if start >= end or not scores:
            dropped += 1
            return
        span = EntitySpan(
            start=start,
            end=end,
            text=text[start:end],
            label=label,
            confidence=min(scores),
            source=source,
        )
        try:
            validate_span(text, span)
        except ValueError:
            dropped += 1
        else:
            spans.append(span)

    for (token_start, token_end), tag_id, score in zip(
        offset_mapping,
        tag_ids,
        confidences,
        strict=True,
    ):
        if token_start == token_end or tag_id == IGNORE_TAG_ID:
            continue
        tag = ID_TO_TAG[tag_id]
        if tag == "O":
            close()
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "I" and open_label == label:
            open_end = token_end
            open_scores.append(score)
            continue
        if prefix == "I":
            close()
            dropped += 1
            continue
        close()
        open_label = label
        open_start = token_start
        open_end = token_end
        open_scores = [score]

    close()
    return spans, dropped
