"""Per-input ONNX latency on CPU.

Each sample is one example: tokenize, run the session, decode spans.
Warmup calls run first and are not included in the average.
"""

import math
import time
from typing import Any

from benchmark.session import Infer, run_example


def _percentile(values: list[float], percent: float) -> float:
    """Nearest-rank percentile. ``percent`` is 50 for the median."""
    ordered = sorted(values)
    rank = math.ceil(percent / 100 * len(ordered))
    index = min(max(rank, 1), len(ordered)) - 1
    return ordered[index]


def measure_latency(
    infer: Infer,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    warmup: int,
    source: str,
) -> dict[str, float | int]:
    """Time one input at a time and summarize the samples.

    Parameters
    ----------
    infer : Infer
        Fp32 or INT8 ONNX forward pass.
    tokenizer : Any
        Fast tokenizer shared with the accuracy pass.
    texts : list[str]
        Examples to time. Batch size is always 1.
    max_length : int
        Truncation length.
    warmup : int
        Untimed runs of the first text before sampling starts.
    source : str
        Progress-line label, such as ``fp32``.

    Returns
    -------
    dict[str, float | int]
        Milliseconds: mean, p50, p95, and p99. The mean is the average
        latency per input.

    """
    if not texts:
        raise ValueError("no texts to time")
    for _ in range(warmup):
        run_example(infer, tokenizer, texts[0], max_length, source)
    samples_ms: list[float] = []
    total = len(texts)
    for index, text in enumerate(texts, start=1):
        started = time.perf_counter()
        run_example(infer, tokenizer, text, max_length, source)
        samples_ms.append((time.perf_counter() - started) * 1000)
        if index % 2000 == 0 or index == total:
            print(f"  {source} latency: {index}/{total}", flush=True)
    return {
        "warmup_excluded": warmup,
        "examples": len(samples_ms),
        "mean": round(sum(samples_ms) / len(samples_ms), 3),
        "p50": round(_percentile(samples_ms, 50), 3),
        "p95": round(_percentile(samples_ms, 95), 3),
        "p99": round(_percentile(samples_ms, 99), 3),
    }
