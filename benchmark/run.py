"""Compare fp32 ONNX with PyTorch, quantize to INT8, and score both.

Run from the repository root::

    uv run python -m benchmark.run

The report is exact-span F1 and false-negative rate (the same scores as
``training/train.py``) plus the average latency of one input on CPU.
"""

import argparse
import json
import platform
from pathlib import Path

from benchmark.latency import measure_latency
from benchmark.quantize import quantize_to_int8
from benchmark.session import (
    Predictions,
    load_onnx_session,
    load_pytorch,
    load_tokenizer,
    onnx_infer,
    predict,
    pytorch_infer,
)
from training.train import exact_span_metrics, gold_spans, load_jsonl


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Score fp32 and dynamic INT8 ONNX on CPU.")
    parser.add_argument("--onnx-dir", type=Path, default=Path("./onnx"))
    parser.add_argument("--checkpoint", type=Path, default=Path("./checkpoint"))
    parser.add_argument("--test-jsonl", type=Path, default=Path("./test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("./results.json"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N rows. 0 means the full file.")
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.999,
        help="Stop before INT8 when PyTorch and fp32 ONNX agree on fewer tags or spans than this.",
    )
    parser.add_argument(
        "--max-f1-drop",
        type=float,
        default=0.005,
        help="Largest micro F1 drop allowed versus fp32. 0.005 is half a point on the 0-1 F1 scale.",
    )
    parser.add_argument(
        "--max-fn-rise",
        type=float,
        default=0.01,
        help="Largest per-label false-negative rate rise allowed versus fp32.",
    )
    return parser.parse_args()


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def _number(metrics: dict[str, object], key: str) -> float:
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} is not a number")
    return float(value)


def _per_label(metrics: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = metrics["per_label"]
    if not isinstance(raw, dict):
        raise TypeError("per_label scores are missing")
    labels: dict[str, dict[str, object]] = {}
    for label, scores in raw.items():
        if not isinstance(scores, dict):
            raise TypeError(f"scores for {label} are not an object")
        labels[str(label)] = scores
    return labels


def agreement_stats(
    reference: Predictions,
    candidate: Predictions,
    minimum: float,
) -> dict[str, object]:
    """Compare PyTorch tags and spans with fp32 ONNX.

    Padding positions are ignored. Those tokens are not part of the text.
    A row agrees on spans when the two span sets are equal.

    Parameters
    ----------
    reference : Predictions
        PyTorch output.
    candidate : Predictions
        Fp32 ONNX output on the same rows.
    minimum : float
        Both agreement rates must reach this value to continue.

    Returns
    -------
    dict[str, object]
        Token and span agreement, and whether the check passed.

    """
    if len(reference.spans) != len(candidate.spans):
        raise ValueError("PyTorch and ONNX produced different numbers of rows")
    token_total = 0
    token_match = 0
    span_match = 0
    rows = zip(
        reference.tag_ids,
        candidate.tag_ids,
        reference.attention_mask,
        candidate.attention_mask,
        reference.spans,
        candidate.spans,
        strict=True,
    )
    for ref_tags, cand_tags, ref_mask, cand_mask, ref_spans, cand_spans in rows:
        if ref_mask != cand_mask:
            raise ValueError("attention masks differ; the models did not see the same tokens")
        for ref_tag, cand_tag, flag in zip(ref_tags, cand_tags, ref_mask, strict=True):
            if not flag:
                continue
            token_total += 1
            token_match += int(ref_tag == cand_tag)
        span_match += int(set(ref_spans) == set(cand_spans))
    examples = len(reference.spans)
    tag_agreement = token_match / token_total if token_total else 1.0
    span_agreement = span_match / examples if examples else 1.0
    return {
        "attended_tokens": token_total,
        "tag_matches": token_match,
        "tag_agreement": tag_agreement,
        "examples": examples,
        "span_matches": span_match,
        "span_agreement": span_agreement,
        "minimum": minimum,
        "passed": tag_agreement >= minimum and span_agreement >= minimum,
    }


def score_predictions(predictions: Predictions, rows: list[dict[str, object]]) -> dict[str, object]:
    """Exact-span precision, recall, F1, and false-negative rate.

    Parameters
    ----------
    predictions : Predictions
        Model spans for ``rows``.
    rows : list[dict[str, object]]
        JSONL records. Gold spans come from ``entities``.

    Returns
    -------
    dict[str, object]
        Metrics from ``training.train.exact_span_metrics``, plus truncation counts.

    """
    metrics = exact_span_metrics(gold_spans(rows), predictions.spans)
    metrics["truncated_entities"] = predictions.truncated_entities
    metrics["dropped_spans"] = predictions.dropped_spans
    return metrics


def promotion_decision(
    fp32: dict[str, object],
    int8: dict[str, object],
    max_f1_drop: float,
    max_fn_rise: float,
) -> tuple[bool, str]:
    """Decide whether INT8 is close enough to fp32 to keep.

    Parameters
    ----------
    fp32 : dict[str, object]
        Exact-span metrics for fp32 ONNX.
    int8 : dict[str, object]
        Exact-span metrics for INT8 ONNX.
    max_f1_drop : float
        Allowed drop in micro F1. Positive when INT8 is worse.
    max_fn_rise : float
        Allowed rise in each label's false-negative rate.

    Returns
    -------
    tuple[bool, str]
        The decision, and a sentence explaining it.

    """
    problems: list[str] = []
    f1_drop = _number(fp32, "f1") - _number(int8, "f1")
    if f1_drop > max_f1_drop:
        problems.append(f"micro F1 dropped by {f1_drop:.4f}, limit is {max_f1_drop:.4f}")
    for label, fp32_scores in _per_label(fp32).items():
        int8_scores = _per_label(int8)[label]
        rise = _number(int8_scores, "false_negative_rate") - _number(fp32_scores, "false_negative_rate")
        if rise > max_fn_rise:
            problems.append(f"{label} false-negative rate rose by {rise:.4f}, limit is {max_fn_rise:.4f}")
    if problems:
        return False, "; ".join(problems)
    return True, "INT8 is within the F1 and false-negative gates"


def _delta(
    fp32: dict[str, object],
    int8: dict[str, object],
    fp32_latency: dict[str, float | int],
    int8_latency: dict[str, float | int],
) -> dict[str, object]:
    per_label: dict[str, float] = {}
    for label, fp32_scores in _per_label(fp32).items():
        int8_scores = _per_label(int8)[label]
        rise = _number(int8_scores, "false_negative_rate") - _number(fp32_scores, "false_negative_rate")
        per_label[label] = rise
    fp32_mean = float(fp32_latency["mean"])
    int8_mean = float(int8_latency["mean"])
    speedup = fp32_mean / int8_mean if int8_mean else 0.0
    return {
        "f1_drop": _number(fp32, "f1") - _number(int8, "f1"),
        "false_negative_rate_rise": _number(int8, "false_negative_rate") - _number(fp32, "false_negative_rate"),
        "per_label_false_negative_rate_rise": per_label,
        "mean_latency_ms": {"fp32": fp32_mean, "int8": int8_mean, "speedup": round(speedup, 3)},
    }


def _write(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def _print_model(name: str, accuracy: dict[str, object], latency: dict[str, float | int]) -> None:
    print(
        f"{name}: F1={_number(accuracy, 'f1'):.4f} "
        f"FN={_number(accuracy, 'false_negative_rate'):.4f} "
        f"mean={float(latency['mean']):.1f} ms per input "
        f"p95={float(latency['p95']):.1f} ms"
    )


def main() -> None:
    """Run the fp32 check, quantize, and write the CPU report."""
    args = parse_args()
    _require(args.onnx_dir / "model.onnx", "fp32 ONNX model")
    _require(args.checkpoint / "config.json", "PyTorch checkpoint")
    _require(args.test_jsonl, "test jsonl")

    rows = load_jsonl(args.test_jsonl)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows in {args.test_jsonl}")

    tokenizer = load_tokenizer(args.onnx_dir)
    fp32_session = load_onnx_session(args.onnx_dir / "model.onnx")
    texts = [str(row["text"]) for row in rows]
    report: dict[str, object] = {
        "machine": platform.platform(),
        "examples": len(rows),
        "max_length": args.max_length,
        "paths": {
            "onnx_dir": str(args.onnx_dir),
            "checkpoint": str(args.checkpoint),
            "test_jsonl": str(args.test_jsonl),
        },
        "gates": {
            "min_agreement": args.min_agreement,
            "max_f1_drop": args.max_f1_drop,
            "max_false_negative_rate_rise": args.max_fn_rise,
        },
    }

    # Checkpoint is only for the export check. Scoring after this is ONNX on CPU.
    print(f"Comparing PyTorch and fp32 ONNX on {len(rows)} examples")
    fp32 = predict(
        onnx_infer(fp32_session),
        tokenizer,
        rows,
        max_length=args.max_length,
        batch_size=args.batch_size,
        source="fp32",
    )
    reference = predict(
        pytorch_infer(load_pytorch(args.checkpoint)),
        tokenizer,
        rows,
        max_length=args.max_length,
        batch_size=args.batch_size,
        source="pytorch",
    )
    agreement = agreement_stats(reference, fp32, args.min_agreement)
    report["agreement"] = agreement
    print(f"agreement: tags={float(agreement['tag_agreement']):.6f} spans={float(agreement['span_agreement']):.6f}")
    if not agreement["passed"]:
        report["promote_int8"] = False
        report["promote_reason"] = "ONNX and PyTorch disagree; INT8 was not created"
        _write(args.output, report)
        raise SystemExit("Fp32 ONNX does not match PyTorch. Left model.onnx unchanged and did not quantize.")

    print("Scoring and timing fp32 ONNX")
    fp32_accuracy = score_predictions(fp32, rows)
    fp32_latency = measure_latency(
        onnx_infer(fp32_session),
        tokenizer,
        texts,
        max_length=args.max_length,
        warmup=args.warmup,
        source="fp32",
    )

    int8_path = args.onnx_dir / "model.int8.onnx"
    print(f"Quantizing dynamic INT8 to {int8_path}")
    quantize_to_int8(args.onnx_dir / "model.onnx", int8_path)
    int8_session = load_onnx_session(int8_path)
    print("Scoring and timing INT8 ONNX")
    int8 = predict(
        onnx_infer(int8_session),
        tokenizer,
        rows,
        max_length=args.max_length,
        batch_size=args.batch_size,
        source="int8",
    )
    int8_accuracy = score_predictions(int8, rows)
    int8_latency = measure_latency(
        onnx_infer(int8_session),
        tokenizer,
        texts,
        max_length=args.max_length,
        warmup=args.warmup,
        source="int8",
    )

    promote, reason = promotion_decision(fp32_accuracy, int8_accuracy, args.max_f1_drop, args.max_fn_rise)
    report["fp32"] = {"accuracy": fp32_accuracy, "latency_ms": fp32_latency}
    report["int8"] = {"model": str(int8_path), "accuracy": int8_accuracy, "latency_ms": int8_latency}
    report["delta"] = _delta(fp32_accuracy, int8_accuracy, fp32_latency, int8_latency)
    report["promote_int8"] = promote
    report["promote_reason"] = reason
    _write(args.output, report)
    _print_model("fp32", fp32_accuracy, fp32_latency)
    _print_model("int8", int8_accuracy, int8_latency)
    print(reason)


if __name__ == "__main__":
    main()
