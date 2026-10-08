"""Write AGE and MONEY test misses from a saved DistilBERT checkpoint."""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from training.train import Span, gold_spans, load_jsonl, predict_split


FOCUS = {"AGE", "MONEY"}


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    run_dir = Path.home() / "preempt-distilbert" / "407111"
    parser = argparse.ArgumentParser(description="Write AGE and MONEY misses from a DistilBERT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=run_dir / "checkpoint")
    parser.add_argument("--test", type=Path, default=run_dir / "data" / "test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def _show(text: str, span: Span) -> str:
    snippet = text[span[0] : span[1]].replace("\n", "\\n")
    return f'{span[2]} [{span[0]}:{span[1]}] "{snippet}"'


def main() -> None:
    """Load the checkpoint and write missed AGE and MONEY spans."""
    args = parse_args()
    checkpoint = args.checkpoint.expanduser()
    test_path = args.test.expanduser()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_dir = args.output_dir.expanduser() if args.output_dir else Path.home() / "preempt-distilbert" / job_id

    rows = load_jsonl(test_path)
    gold_rows = gold_spans(rows)

    # Only keep rows with AGE or MONEY spans.
    keep = [index for index, spans in enumerate(gold_rows) if any(span[2] in FOCUS for span in spans)]
    rows = [rows[index] for index in keep]x
    gold_rows = [gold_rows[index] for index in keep]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    predicted_rows, _, _ = predict_split(model, tokenizer, rows, args.max_length, args.batch_size)

    blocks: list[str] = []
    for row, gold, predicted in zip(rows, gold_rows, predicted_rows, strict=True):
        text = str(row["text"])
        gold_focus = [span for span in gold if span[2] in FOCUS]
        predicted_focus = [span for span in predicted if span[2] in FOCUS]
        missed = [span for span in gold_focus if span not in predicted_focus]
        if not missed:
            continue
        lines = [text.replace("\n", "\\n"), "gold:"]
        lines.extend(f"  {_show(text, span)}" for span in gold_focus)
        lines.append("predicted:")
        lines.extend(f"  {_show(text, span)}" for span in predicted_focus)
        if not predicted_focus:
            lines.append("  none")
        lines.append("missed:")
        for span in missed:
            overlap = next(
                (
                    other
                    for other in predicted_focus
                    if other[2] == span[2] and span[0] < other[1] and other[0] < span[1]
                ),
                None,
            )
            if overlap is None:
                lines.append(f"  {_show(text, span)}  no prediction")
            else:
                lines.append(f"  {_show(text, span)}  overlaps {_show(text, overlap)}")
        blocks.append("\n".join(lines))

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "age-money-misses.txt"
    destination.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    print(f"{destination} ({len(blocks)} rows)")


if __name__ == "__main__":
    main()
