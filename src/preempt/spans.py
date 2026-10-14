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
