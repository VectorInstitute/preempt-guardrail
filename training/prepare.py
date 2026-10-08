"""Filter English rows, check spans, and write train, validation, and test JSONL."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from preempt.contracts import LABELS
from training.labels import SOURCE_TO_LABEL, check_label_map


def _column(raw: dict[str, object], key: str) -> list[object]:
    """
    Retrieve a column from a dict assumed to be in columnar format.

    Args:
        raw: The raw privacy_mask object, a dict of column lists by key.
        key: The column name to extract.

    Returns:
        The column as a list of objects.

    Raises:
        TypeError: If the column is not present as a list.
    """
    column = raw.get(key)
    if not isinstance(column, list):
        raise TypeError("privacy_mask object is not columnar")
    return column


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError
    return int(value)


def split_name(example_id: int, seed: int) -> str:
    """Assign an example id to train, validation, or test.

    Parameters
    ----------
    example_id : int
        Dataset example id.
    seed : int
        Salt mixed into the hash so the split is repeatable.

    Returns
    -------
    str
        ``train``, ``validation``, or ``test``.

    """
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def mask_dicts(raw: object) -> list[dict[str, object]]:
    """Return privacy-mask entries as a list of dicts.

    Parameters
    ----------
    raw : object
        A list of dicts, a JSON string of that list, or a columnar dict. 

    Returns
    -------
    list[dict[str, object]]
        One dict per annotated span, with start, end, label, and value.

    """
    if isinstance(raw, str):
        parsed = json.loads(raw)
        return mask_dicts(parsed)
    if isinstance(raw, list):
        entries: list[dict[str, object]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise TypeError("privacy_mask entry is not an object")
            entries.append(item)
        return entries
    if isinstance(raw, dict):
        starts = _column(raw, "start")
        ends = _column(raw, "end")
        labels = _column(raw, "label")
        values = _column(raw, "value")
        if not (len(starts) == len(ends) == len(labels) == len(values)):
            raise ValueError("privacy_mask columns differ in length")
        return [
            {"start": starts[index], "end": ends[index], "label": labels[index], "value": values[index]}
            for index in range(len(starts))
        ]
    raise TypeError("privacy_mask has an unexpected type")


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


def entities_from_mask(
    text: str,
    raw_mask: object,
    unmapped: Counter[str],
) -> tuple[list[dict[str, object]], str | None]:
    """Convert one privacy mask into canonical entities.

    Parameters
    ----------
    text : str
        Source text. Offsets are checked against this string.
    raw_mask : object
        Raw ``privacy_mask`` value from the dataset.
    unmapped : Counter[str]
        Counts of source labels that are not in the canonical map.

    Returns
    -------
    tuple[list[dict[str, object]], str | None]
        Entities, and a reject reason when the row should be dropped.

    """
    try:
        masks = mask_dicts(raw_mask)
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], "bad_mask"

    entities: list[dict[str, object]] = []
    for item in masks:
        try:
            start = _as_int(item["start"])
            end = _as_int(item["end"])
            value = str(item["value"])
            source_label = str(item["label"])
        except (KeyError, TypeError, ValueError):
            return [], "bad_mask"
        label = SOURCE_TO_LABEL.get(source_label)
        if label is None:
            unmapped[source_label] += 1
            continue
        if start < 0 or end > len(text) or start >= end:
            return [], "bad_offsets"
        if text[start:end] != value:
            return [], "text_mismatch"
        norm_start, norm_end = normalize_span(text, start, end, label)
        if norm_start < norm_end:
            start, end = norm_start, norm_end
        entities.append({"start": start, "end": end, "label": label})

    entities.sort(key=lambda entity: (_as_int(entity["start"]), _as_int(entity["end"])))
    for previous, following in zip(entities, entities[1:], strict=False):
        if _as_int(following["start"]) < _as_int(previous["end"]):
            return [], "overlap"
    return entities, None


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one JSON object per line.

    Parameters
    ----------
    path : Path
        Destination file.
    rows : list[dict[str, object]]
        Records to write.

    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare English PII spans for DistilBERT.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="ai4privacy/pii-masking-200k")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Load the dataset, keep valid English rows, and write the splits."""
    from datasets import load_dataset  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    args = parse_args()
    check_label_map()
    dataset = load_dataset(args.dataset, split="train")

    reject_reasons: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    buckets: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}

    for row in tqdm(dataset, desc="prepare"):
        text = str(row["source_text"])
        entities, reason = entities_from_mask(text, row["privacy_mask"], unmapped)
        if reason is not None:
            reject_reasons[reason] += 1
            continue
        for entity in entities:
            label_counts[str(entity["label"])] += 1
        buckets[split_name(int(row["id"]), args.seed)].append({"text": text, "entities": entities})

    missing = [label for label in LABELS if label_counts[label] == 0]
    stats: dict[str, object] = {
        "dataset": args.dataset,
        "seed": args.seed,
        "rows_rejected": sum(reject_reasons.values()),
        "reject_reasons": dict(reject_reasons),
        "unmapped_labels": dict(unmapped),
        "label_counts": {label: label_counts[label] for label in LABELS},
        "splits": {name: len(rows) for name, rows in buckets.items()},
    }
    print(json.dumps(stats, indent=2))
    if missing:
        print("no spans for:", ", ".join(missing))
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
