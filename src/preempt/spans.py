"""Character-span helpers for names, ages, and money."""

import re

from preempt.contracts import EntitySpan


# A currency symbol immediately before or after a number, including commas
# and a decimal part. Plain numbers are left for the tagger.
_CURRENCY_AMOUNT = re.compile(
    r"(?<![\w])[$£€¥₩฿₺₽₹₦₨₪៛]\s?(?P<before>\d+(?:,\d+)*(?:\.\d+)?)(?![\w])"
    r"|(?<![\w])(?P<after>\d+(?:,\d+)*(?:\.\d+)?)\s?[$£€¥₩฿₺₽₹₦₨₪៛](?![\w])"
)


def normalize_span(text: str, start: int, end: int, label: str) -> tuple[int, int]:
    """Snap AGE to its number and MONEY to the full amount.

    Parameters
    ----------
    text : str
        Source text.
    start : int
        Inclusive character start.
    end : int
        Exclusive character end.
    label : str
        Canonical label. Other labels are returned unchanged.

    Returns
    -------
    tuple[int, int]
        Normalized ``(start, end)``.

    """
    if label == "AGE":
        # Gold is "73", "73 years", or "73 years old". Keep the digits.
        while start < end and not text[start].isdigit():
            start += 1
        end = start
        while end < len(text) and text[end].isdigit():
            end += 1
        return start, end
    if label == "MONEY":
        # A span may start at "." or ",". Include the digits in front of it,
        # then the rest of the amount. Stop before currency words ("Rp", "kr").
        while start > 0 and text[start - 1].isdigit():
            start -= 1
        while end < len(text):
            if text[end].isdigit():
                end += 1
                continue
            # "." or "," only when a digit follows, so a sentence period stays out.
            if text[end] in ".," and end + 1 < len(text) and text[end + 1].isdigit():
                end += 1
                continue
            break
        if end < len(text) and text[end] in "kKmMbB":
            end += 1
    return start, end


def covering_windows(
    pieces: list[tuple[int, int]],
    *,
    size: int,
    overlap: int,
    text_length: int,
) -> list[tuple[int, int]]:
    """Return character slices that cover every piece.

    Parameters
    ----------
    pieces : list[tuple[int, int]]
        ``(start, end)`` offsets in order, one per token or word.
    size : int
        Maximum number of pieces in one slice.
    overlap : int
        Pieces repeated from the previous slice. Must be smaller than ``size``.
    text_length : int
        Length of the source text. The first slice starts at 0 and the last
        ends here, so a prefix or suffix with no piece is still included.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` slices. A text that fits in one window is
        ``[(0, text_length)]``.

    Raises
    ------
    ValueError
        ``size`` is below 1, or ``overlap`` is negative or not smaller than
        ``size``.

    """
    if size < 1:
        raise ValueError("window size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than the window size")
    if len(pieces) <= size:
        return [(0, text_length)]
    step = size - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    count = len(pieces)
    while start < count:
        end = min(start + size, count)
        char_start = 0 if start == 0 else pieces[start][0]
        char_end = text_length if end == count else pieces[end - 1][1]
        windows.append((char_start, max(char_end, char_start)))
        if end == count:
            break
        start += step
    return windows


def merge_overlapping_spans(spans: list[EntitySpan]) -> list[EntitySpan]:
    """Keep the longer span when two of the same label overlap.

    Parameters
    ----------
    spans : list[EntitySpan]
        Spans from one or more windows of the same text.

    Returns
    -------
    list[EntitySpan]
        Spans in start order. A same-label overlap keeps the longer span, or
        the higher confidence when the lengths match. Different labels are
        both kept.

    """
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    kept: list[EntitySpan] = []
    for span in ordered:
        overlap_at = None
        for index, current in enumerate(kept):
            if span.label == current.label and span.start < current.end and current.start < span.end:
                overlap_at = index
                break
        if overlap_at is None:
            kept.append(span)
            continue
        current = kept[overlap_at]
        current_rank = (current.end - current.start, current.confidence)
        span_rank = (span.end - span.start, span.confidence)
        if span_rank > current_rank:
            kept[overlap_at] = span
    kept.sort(key=lambda span: (span.start, span.end))
    return kept


def currency_spans(text: str) -> list[tuple[int, int]]:
    """Return character bounds of amounts written with a currency symbol.

    Parameters
    ----------
    text : str
        Source text.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` of the numeric amount, such as ``10,000`` in
        ``$10,000``. The symbol stays in the source text so encryption can
        leave it in front of the ciphertext.

    """
    spans: list[tuple[int, int]] = []
    for match in _CURRENCY_AMOUNT.finditer(text):
        group = "before" if match.group("before") else "after"
        spans.append(match.span(group))
    return spans


def join_name_spans(text: str, spans: list[EntitySpan]) -> list[str]:
    """Return one string per name, joining parts separated only by whitespace.

    Parameters
    ----------
    text : str
        Source text the spans were taken from.
    spans : list[EntitySpan]
        Name spans in that text.

    Returns
    -------
    list[str]
        Surface names. ``Ben`` and ``Parker`` with a space between them become
        ``Ben Parker``. A gap that contains other characters keeps them apart.

    """
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    names: list[str] = []
    block_start: int | None = None
    block_end = 0
    for span in ordered:
        if block_start is None:
            block_start = span.start
            block_end = span.end
            continue
        # Merge when the gap is only whitespace.
        if text[block_end : span.start].strip() == "":
            block_end = span.end
            continue
        names.append(text[block_start:block_end])
        block_start = span.start
        block_end = span.end
    if block_start is not None:
        names.append(text[block_start:block_end])
    return names
