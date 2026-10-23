"""Validate the recognizer and sanitizer against the dataset's gold PII spans.

Two checks per item:

- **Detection:** the recognizer's raw spans vs. gold spans, per label, in
  ``strict`` (exact offsets) and ``partial`` (IoU >= 0.5) modes. Recall is
  also broken down by the gold span's ``surface_format`` (e.g. ``k_suffix``).
  Both sides are first normalized to the NER's labeling conventions: adjacent
  first/last NAME spans are merged, and gold MONEY spans are shrunk to the
  number ("$15,000" -> "15,000").
- **Leakage:** after a full sanitizer run, does any gold original value still
  appear verbatim in the sanitized text? Names also count as leaked if any
  single name token survives.

Usage (from the ``pii-bench/`` directory)::

    uv run --project .. python -m metrics.detection
    uv run --project .. python -m metrics.detection --model-path ../runs/407944/onnx --onnx-file model.int8.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from models import ALL_ENTITY_TYPES, DatasetItem, PiiSpan
from sanitizers.preempt_adapter import PreemptSanitizerAdapter
from sanitizers.text_utils import numeric_core

from preempt.contracts import EntitySpan, Recognizer
from preempt.distilbert_recognizer import DistilBertRecognizer
from preempt.sanitizer import Sanitizer as PreemptSanitizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BENCH_DIR.parent
PARTIAL_IOU = 0.5


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class DetectionReport:
    strict: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    partial: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    # (label, surface_format) -> [found, total] under partial matching.
    recall_by_format: dict[tuple[str, str], list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    leaked: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    items_with_leak: int = 0
    n_items: int = 0


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def comparable_gold(g: PiiSpan) -> PiiSpan:
    """Shrink MONEY gold spans to their number ("$15,000" -> "15,000"), as the NER labels them."""
    if g.label == "MONEY" and (m := numeric_core(g.text)):
        return g.model_copy(update={"start": g.start + m.start(), "end": g.start + m.end(), "text": m.group(0)})
    return g


def merge_adjacent_names(text: str, spans: Sequence[EntitySpan]) -> list[EntitySpan]:
    """Join NAME spans separated only by whitespace ("Wei" + "Zhang" -> "Wei Zhang")."""
    merged: list[EntitySpan] = []
    for span in sorted(spans, key=lambda s: s.start):
        prev = merged[-1] if merged else None
        if prev and prev.label == span.label == "NAME" and not text[prev.end : span.start].strip():
            merged[-1] = replace(prev, end=span.end, text=text[prev.start : span.end])
        else:
            merged.append(span)
    return merged


def match_spans(
    gold: Sequence[PiiSpan], predicted: Sequence[EntitySpan], *, strict: bool
) -> tuple[list[PiiSpan], list[PiiSpan], list[EntitySpan]]:
    """Greedy one-to-one matching on same label. Returns (matched, missed, extra)."""
    unused = list(predicted)
    matched: list[PiiSpan] = []
    missed: list[PiiSpan] = []
    for g in gold:
        best, best_iou = None, 0.0
        for p in unused:
            if p.label != g.label:
                continue
            if strict:
                if (p.start, p.end) == (g.start, g.end):
                    best, best_iou = p, 1.0
                    break
            elif (iou := _iou((g.start, g.end), (p.start, p.end))) >= PARTIAL_IOU and iou > best_iou:
                best, best_iou = p, iou
        if best is None:
            missed.append(g)
        else:
            unused.remove(best)
            matched.append(g)
    return matched, missed, unused


def leaked_spans(gold: Sequence[PiiSpan], sanitized_text: str) -> list[PiiSpan]:
    """Gold spans whose original value (or, for names, any name token) survives."""
    leaks = []
    for g in gold:
        if g.text in sanitized_text:
            leaks.append(g)
        elif g.label == "NAME" and any(len(tok) > 2 and tok in sanitized_text for tok in g.text.split()):
            leaks.append(g)
    return leaks


def evaluate(
    items: Sequence[DatasetItem],
    recognizer: Recognizer,
    adapter: PreemptSanitizerAdapter,
    *,
    epsilon: float,
    seed: int,
) -> tuple[DetectionReport, list[dict]]:
    report = DetectionReport()
    details = []
    sanitized = adapter.sanitize_batch([item.text for item in items], epsilon=epsilon, seed=seed)

    for item, result in zip(items, sanitized):
        report.n_items += 1
        raw = [s for s in recognizer.recognize(item.text) if s.label in ALL_ENTITY_TYPES]
        predicted = merge_adjacent_names(item.text, raw)
        gold = [comparable_gold(g) for g in item.pii_spans]

        for mode, counts in (("strict", report.strict), ("partial", report.partial)):
            matched, missed, extra = match_spans(gold, predicted, strict=mode == "strict")
            for g in matched:
                counts[g.label].tp += 1
            for g in missed:
                counts[g.label].fn += 1
            for p in extra:
                counts[p.label].fp += 1
            if mode == "partial":
                partial_missed = [orig for orig, g in zip(item.pii_spans, gold) if g in missed]
                partial_extra = extra
                for g in gold:
                    cell = report.recall_by_format[(g.label, g.surface_format or "unknown")]
                    cell[1] += 1
                    cell[0] += g not in missed

        leaks = leaked_spans(item.pii_spans, result.text)
        for g in item.pii_spans:
            report.leaked[g.label][1] += 1
            report.leaked[g.label][0] += g in leaks
        report.items_with_leak += bool(leaks)

        details.append(
            {
                "id": item.id,
                "text": item.text,
                "sanitized_text": result.text,
                "missed": [g.model_dump() for g in partial_missed],
                "extra": [{"label": p.label, "start": p.start, "end": p.end, "text": p.text} for p in partial_extra],
                "leaked": [g.model_dump() for g in leaks],
                "warnings": result.warnings,
            }
        )
    return report, details


def format_report(report: DetectionReport) -> str:
    lines = [f"Items: {report.n_items}", "", "Detection (precision / recall / F1)"]
    lines.append(f"{'label':<8}{'strict':>24}{'partial (IoU>=0.5)':>26}")
    for label in sorted(set(report.strict) | set(report.partial)):
        s, p = report.strict[label], report.partial[label]
        lines.append(
            f"{label:<8}{s.precision:>8.2f}{s.recall:>8.2f}{s.f1:>8.2f}"
            f"{p.precision:>10.2f}{p.recall:>8.2f}{p.f1:>8.2f}"
        )
    lines += ["", "Partial recall by surface format"]
    for (label, fmt), (found, total) in sorted(report.recall_by_format.items()):
        lines.append(f"  {label:<6} {fmt:<14} {found}/{total} = {found / total:.2f}")
    lines += ["", "Leakage (original value survives sanitization)"]
    for label, (leaked, total) in sorted(report.leaked.items()):
        lines.append(f"  {label:<6} {leaked}/{total} = {leaked / total:.2f}")
    lines.append(
        f"  items with any leak: {report.items_with_leak}/{report.n_items} = "
        f"{report.items_with_leak / max(report.n_items, 1):.2f}"
    )
    return "\n".join(lines)


def load_items(path: Path) -> list[DatasetItem]:
    with path.open() as f:
        return [DatasetItem.model_validate_json(line) for line in f if line.strip()]


def main() -> None:
    config = yaml.safe_load((BENCH_DIR / "config.yaml").read_text())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BENCH_DIR / config["paths"]["data_dir"] / "dataset.jsonl")
    parser.add_argument("--model-path", type=Path, default=REPO_DIR / "runs/407944/onnx")
    parser.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx")
    parser.add_argument("--onnx-file", default="model.onnx")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=config["dataset"]["seed"])
    parser.add_argument(
        "--out", type=Path, default=BENCH_DIR / config["paths"]["runs_dir"] / "detection_validation.jsonl"
    )
    args = parser.parse_args()

    items = load_items(args.dataset)
    recognizer = DistilBertRecognizer(
        model_path=str(args.model_path),
        backend=args.backend,
        onnx_file_name=args.onnx_file if args.backend == "onnx" else None,
    )
    adapter = PreemptSanitizerAdapter(PreemptSanitizer(recognizer), recognizer)

    report, details = evaluate(items, recognizer, adapter, epsilon=args.epsilon, seed=args.seed)
    print(format_report(report))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("per-item misses/extras/leaks -> %s", args.out)


if __name__ == "__main__":
    main()
