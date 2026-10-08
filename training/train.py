"""Fine-tune DistilBERT and write exact-span metrics next to the checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from preempt.bio import ID_TO_TAG, TAG_NAMES, TAG_TO_ID, spans_from_tags, tags_from_spans
from preempt.contracts import LABELS
from training.labels import SOURCE_TO_LABEL
from training.prepare import normalize_span


Span = tuple[int, int, str]


class TokenDataset:
    """Token-classification examples built from character spans."""

    def __init__(self, rows: list[dict[str, object]], tokenizer: object, max_length: int) -> None:
        """Store rows and the tokenizer used to align spans.

        Parameters
        ----------
        rows : list[dict[str, object]]
            JSONL records with ``text`` and ``entities``.
        tokenizer : object
            Fast tokenizer. Called with ``return_offsets_mapping``.
        max_length : int
            Maximum tokens per example.

        """
        self._rows = rows
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """Tokenize one example and align its entity spans.

        Parameters
        ----------
        index : int
            Example index.

        Returns
        -------
        dict[str, list[int]]
            ``input_ids``, ``attention_mask``, and ``labels``.

        """
        row = self._rows[index]
        text = str(row["text"])
        raw_entities = row["entities"]
        if not isinstance(raw_entities, list):
            raise TypeError("entities must be a list")
        entities = [_entity_tuple(entity) for entity in raw_entities]
        encoded = self._tokenizer(  # type: ignore[operator]
            text,
            truncation=True,
            max_length=self._max_length,
            return_offsets_mapping=True,
        )
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        return {
            "input_ids": [int(token_id) for token_id in encoded["input_ids"]],
            "attention_mask": [int(flag) for flag in encoded["attention_mask"]],
            "labels": tags_from_spans(offsets, entities),
        }


def _entity_tuple(entity: object) -> Span:
    if not isinstance(entity, dict):
        raise TypeError("entity must be an object")
    return (int(entity["start"]), int(entity["end"]), str(entity["label"]))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a JSONL file of text and entity records.

    Parameters
    ----------
    path : Path
        File written by ``training.prepare``.

    Returns
    -------
    list[dict[str, object]]
        One record per line.

    """
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path} contains a non-object row")
            rows.append(record)
    return rows


def _counts_as_scores(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float | int]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    support = true_positive + false_negative
    false_negative_rate = false_negative / support if support else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": false_negative_rate,
        "support": support,
    }


def _boundary_errors(gold: list[Span], predicted: list[Span]) -> int:
    errors = 0
    for gold_start, gold_end, gold_label in gold:
        for pred_start, pred_end, pred_label in predicted:
            overlaps = gold_start < pred_end and pred_start < gold_end
            same_span = gold_start == pred_start and gold_end == pred_end
            if gold_label == pred_label and overlaps and not same_span:
                errors += 1
                break
    return errors


def exact_span_metrics(
    gold_rows: list[list[Span]],
    predicted_rows: list[list[Span]],
) -> dict[str, object]:
    """Score exact character spans, overall and per label.

    Parameters
    ----------
    gold_rows : list[list[Span]]
        Gold ``(start, end, label)`` spans per example.
    predicted_rows : list[list[Span]]
        Predicted spans per example, parallel to ``gold_rows``.

    Returns
    -------
    dict[str, object]
        Micro scores, per-label scores, and the boundary-error count.

    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    per_label = {label: [0, 0, 0] for label in LABELS}
    boundary_errors = 0
    for gold, predicted in zip(gold_rows, predicted_rows, strict=True):
        boundary_errors += _boundary_errors(gold, predicted)
        for label in LABELS:
            gold_set = {span for span in gold if span[2] == label}
            pred_set = {span for span in predicted if span[2] == label}
            matched = len(gold_set & pred_set)
            missed = len(gold_set - pred_set)
            extra = len(pred_set - gold_set)
            per_label[label][0] += matched
            per_label[label][1] += extra
            per_label[label][2] += missed
            true_positive += matched
            false_positive += extra
            false_negative += missed
    return {
        **_counts_as_scores(true_positive, false_positive, false_negative),
        "per_label": {
            label: _counts_as_scores(matched, extra, missed) for label, (matched, extra, missed) in per_label.items()
        },
        "boundary_errors": boundary_errors,
    }


def _last_kept_offset(offsets: list[tuple[int, int]]) -> int:
    last = 0
    for start, end in offsets:
        if start != end and end > last:
            last = end
    return last


def predict_split(
    model: torch.nn.Module,
    tokenizer: object,
    rows: list[dict[str, object]],
    max_length: int,
    batch_size: int,
) -> tuple[list[list[Span]], int, int]:
    """Decode one split into character spans.

    Parameters
    ----------
    model : torch.nn.Module
        Trained token classifier.
    tokenizer : object
        Fast tokenizer matching ``model``.
    rows : list[dict[str, object]]
        Examples with ``text`` and ``entities``.
    # Example row:
    # {
    #     "text": "My grandfather was 87 years old and earned $12,000.",
    #     "entities": [
    #         {"start": 21, "end": 23, "label": "AGE"},
    #         {"start": 44, "end": 50, "label": "MONEY"}
    #     ]
    # }
    max_length : int
        Maximum tokens per example.
    batch_size : int
        Examples per forward pass.

    Returns
    -------
    tuple[list[list[Span]], int, int]
        Predicted spans, truncated gold entities, and dropped tag pieces.

    """
    device = next(model.parameters()).device
    model.eval()
    predicted_rows: list[list[Span]] = []
    truncated_entities = 0
    dropped_spans = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [str(row["text"]) for row in batch]
        encoded = tokenizer(  # type: ignore[operator]
            texts,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        offset_batch = encoded.pop("offset_mapping").tolist()
        token_batch = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**token_batch).logits
            probs = torch.softmax(logits, dim=-1)
            tag_batch = probs.argmax(dim=-1)
            score_batch = probs.max(dim=-1).values
        for index, row in enumerate(batch):
            offsets = [(int(token_start), int(token_end)) for token_start, token_end in offset_batch[index]]
            tag_ids = [int(tag_id) for tag_id in tag_batch[index].tolist()]
            confidences = [float(score) for score in score_batch[index].tolist()]
            spans, dropped = spans_from_tags(texts[index], offsets, tag_ids, confidences, source="distilbert")
            dropped_spans += dropped
            # Same AGE/MONEY snap as the gold spans written by prepare.
            normalized: list[Span] = []
            for span in spans:
                norm_start, norm_end = normalize_span(texts[index], span.start, span.end, span.label)
                if norm_start >= norm_end:
                    norm_start, norm_end = span.start, span.end
                normalized.append((norm_start, norm_end, span.label))
            predicted_rows.append(normalized)
            raw_entities = row["entities"]
            if not isinstance(raw_entities, list):
                raise TypeError("entities must be a list")
            token_count = len(tokenizer.encode(texts[index], add_special_tokens=True))  # type: ignore[attr-defined]
            if token_count > max_length:
                last = _last_kept_offset(offsets)
                truncated_entities += sum(1 for entity in raw_entities if int(entity["end"]) > last)
    return predicted_rows, truncated_entities, dropped_spans


def gold_spans(rows: list[dict[str, object]]) -> list[list[Span]]:
    """Read gold spans from prepared rows.

    Parameters
    ----------
    rows : list[dict[str, object]]
        Examples with an ``entities`` list.

    Returns
    -------
    list[list[Span]]
        ``(start, end, label)`` per example.

    """
    columns: list[list[Span]] = []
    for row in rows:
        raw_entities = row["entities"]
        if not isinstance(raw_entities, list):
            raise TypeError("entities must be a list")
        columns.append([_entity_tuple(entity) for entity in raw_entities])
    return columns


def evaluate_split(
    model: torch.nn.Module,
    tokenizer: object,
    rows: list[dict[str, object]],
    max_length: int,
    batch_size: int,
) -> dict[str, object]:
    """Predict a split and attach truncation counts to the exact-span scores.

    Parameters
    ----------
    model : torch.nn.Module
        Trained token classifier.
    tokenizer : object
        Fast tokenizer matching ``model``.
    rows : list[dict[str, object]]
        Examples with ``text`` and ``entities``.
    max_length : int
        Maximum tokens per example.
    batch_size : int
        Examples per forward pass.

    Returns
    -------
    dict[str, object]
        Exact-span metrics plus truncation and dropped-span counts.

    """
    predicted, truncated_entities, dropped_spans = predict_split(model, tokenizer, rows, max_length, batch_size)
    metrics = exact_span_metrics(gold_spans(rows), predicted)
    metrics["truncated_entities"] = truncated_entities
    metrics["dropped_spans"] = dropped_spans
    return metrics


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for English PII spans.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Train DistilBERT and write ``metrics.json`` beside the checkpoint."""
    args = parse_args()
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    validation_rows = load_jsonl(args.data_dir / "validation.jsonl")
    test_rows = load_jsonl(args.data_dir / "test.jsonl")
    stats_path = args.data_dir / "stats.json"
    prepare_stats: object = None
    if stats_path.exists():
        prepare_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(TAG_NAMES),
        id2label=ID_TO_TAG,
        label2id=TAG_TO_ID,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"),
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            save_strategy="epoch",
            report_to="none",
            seed=args.seed,
            logging_steps=50,
        ),
        train_dataset=TokenDataset(train_rows, tokenizer, args.max_length),
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()

    checkpoint_dir = args.output_dir / "checkpoint"
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(checkpoint_dir)
    (args.output_dir / "label_map.json").write_text(
        json.dumps(SOURCE_TO_LABEL, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics = {
        "model": args.model,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "prepare": prepare_stats,
        "validation": evaluate_split(model, tokenizer, validation_rows, args.max_length, args.batch_size),
        "test": evaluate_split(model, tokenizer, test_rows, args.max_length, args.batch_size),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics_path)


if __name__ == "__main__":
    main()
