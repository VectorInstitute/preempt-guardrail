"""Generate A, A' and B responses for each dataset item and score verifiable ones.

- **A:** generator response to the original prompt.
- **A':** a second, independent sample on the original prompt (noise floor).
- **B:** response to the sanitized prompt, with surrogate names restored.

Verifiable items are scored by parsing ``FINAL_ANSWER`` from each response
and comparing A vs A' and A vs B. Every LLM call is cached on disk, and a
rerun with the same ``--run-id`` skips items already in ``results.jsonl``.

Usage (from the ``pii-bench/`` directory)::

    uv run --project .. python -m runner.pipeline --run-id pilot --limit 4
    uv run --project .. python -m runner.pipeline --run-id pilot --epsilon 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

from llm.clients import OpenAICompatibleClient, create_generator_client
from metrics.detection import load_items
from metrics.verifiable import ANSWER_LABELS, answers_agree, parse_final_answer
from models import DatasetItem, RunRecord, SanitizeResult
from runner.cache import CompletionCache
from sanitizers.preempt_adapter import PreemptSanitizerAdapter

from preempt.distilbert_recognizer import DistilBertRecognizer
from preempt.sanitizer import Sanitizer as PreemptSanitizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BENCH_DIR.parent
MAX_TOKENS = 4096


class Generator:
    """Cached wrapper around the generator client."""

    def __init__(self, client: OpenAICompatibleClient, cache: CompletionCache, temperature: float) -> None:
        self.client = client
        self.cache = cache
        self.temperature = temperature

    def __call__(self, prompt: str, *, sample: str) -> str:
        """``sample`` separates otherwise identical requests, e.g. A vs A'."""
        request = {
            "model": self.client.settings.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": MAX_TOKENS,
            "sample": sample,
        }
        key = self.cache.key(**request)
        if (cached := self.cache.get(key)) is not None:
            return cached
        response = self.client.complete(prompt, temperature=self.temperature, max_tokens=MAX_TOKENS)
        self.cache.put(key, request, response)
        return response


def item_seed(base_seed: int, item_id: str, draw: int) -> int:
    """Stable per-item, per-draw sanitizer seed; one shared seed gives every item the same noise."""
    digest = hashlib.sha256(f"{base_seed}-{item_id}-{draw}".encode()).hexdigest()
    return int(digest[:8], 16)


def run_item(
    item: DatasetItem,
    sanitized: SanitizeResult,
    generate: Generator,
    adapter: PreemptSanitizerAdapter,
    *,
    draw: int,
) -> RunRecord:
    response_a = generate(item.text, sample="A")
    response_a_prime = generate(item.text, sample="A_prime")
    response_b_raw = generate(sanitized.text, sample=f"B-draw{draw}")
    response_b = adapter.rehydrate_names(response_b_raw, sanitized.substitutions)

    # run_item() combines three things:
    # - The base seed from `config.yaml`
    # - The item's ID
    # - The draw number
    # Same item + same draw + same base seed → same noise, so the result is reproducible.
    record = RunRecord(
        item_id=item.id,
        task_family=item.task_family,
        verifiable=item.verifiable,
        generator_model=generate.client.settings.model,
        generator_temperature=generate.temperature,
        epsilon=sanitized.epsilon or 0.0,
        draw=draw,
        seed=sanitized.seed,
        original_prompt=item.text,
        sanitized_prompt=sanitized.text,
        substitutions=sanitized.substitutions,
        sanitizer_warnings=sanitized.warnings,
        response_a=response_a,
        response_a_prime=response_a_prime,
        response_b_raw=response_b_raw,
        response_b=response_b,
    )
    if item.verifiable:
        labels = ANSWER_LABELS.get(item.task_family)
        record.answer_a = parse_final_answer(response_a, labels)
        record.answer_a_prime = parse_final_answer(response_a_prime, labels)
        record.answer_b = parse_final_answer(response_b, labels)
        record.parity_a_a_prime = answers_agree(record.answer_a, record.answer_a_prime)
        record.parity_a_b = answers_agree(record.answer_a, record.answer_b)
    return record


def load_done(results_path: Path) -> set[tuple[str, float, int]]:
    """Load the set of items that have already been run. A rerun skips anything already done, 
    and a failed item is retried on the next run.
    Same item + same draw + same base seed → same noise, so the item is already done."""
    if not results_path.exists():
        return set()
    with results_path.open() as f:
        records = [RunRecord.model_validate_json(line) for line in f if line.strip()]
    return {(r.item_id, r.epsilon, r.draw) for r in records}


def summarize(results_path: Path) -> str:
    with results_path.open() as f:
        records = [RunRecord.model_validate_json(line) for line in f if line.strip()]
    verifiable = [r for r in records if r.verifiable]
    lines = [f"Records: {len(records)} ({len(verifiable)} verifiable)"]
    if not verifiable:
        return "\n".join(lines)

    def rate(values: list[bool | None]) -> str:
        scored = [v for v in values if v is not None]
        if not scored:
            return "n/a"
        return f"{sum(scored)}/{len(scored)} = {sum(scored) / len(scored):.2f}"

    parse_fail = sum(
        answer is None for r in verifiable for answer in (r.answer_a, r.answer_a_prime, r.answer_b)
    )
    lines += [
        f"Decision parity A vs A' (noise floor): {rate([r.parity_a_a_prime for r in verifiable])}",
        f"Decision parity A vs B  (sanitized):   {rate([r.parity_a_b for r in verifiable])}",
        f"FINAL_ANSWER parse failures: {parse_fail}/{3 * len(verifiable)}",
    ]
    return "\n".join(lines)


def main() -> None:
    load_dotenv(BENCH_DIR / ".env")
    config = yaml.safe_load((BENCH_DIR / "config.yaml").read_text())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="pilot")
    parser.add_argument("--dataset", type=Path, default=BENCH_DIR / config["paths"]["data_dir"] / "dataset.jsonl")
    parser.add_argument("--model-path", type=Path, default=REPO_DIR / "runs/407944/onnx")
    parser.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx")
    parser.add_argument("--onnx-file", default="model.onnx")
    # --draw picks which sanitizer run you're doing for each prompt. Same draw number, same noise. Different draw number, different noise.
    # Because seed is based on 3 values: base_seed (from config.yaml), item_id (from dataset.jsonl), draw
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--families", nargs="+", default=None, help="Only run these task families.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N items.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    runs_dir = BENCH_DIR / config["paths"]["runs_dir"]
    results_path = runs_dir / args.run_id / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    items = [i for i in load_items(args.dataset) if args.families is None or i.task_family in args.families]
    items = items[: args.limit]
    done = load_done(results_path)
    # Prepare a list of items that have not yet been completed.
    # Only include items whose (item.id, args.epsilon, args.draw) tuple is not found in the 'done' set.
    todo = []
    for item in items:
        key = (item.id, args.epsilon, args.draw)
        if key not in done:
            todo.append(item)
    logger.info("%d items, %d already done, %d to run (%d generator calls max)", len(items), len(items) - len(todo), len(todo), 3 * len(todo))

    if todo:
        recognizer = DistilBertRecognizer(
            model_path=str(args.model_path),
            backend=args.backend,
            onnx_file_name=args.onnx_file if args.backend == "onnx" else None,
        )
        adapter = PreemptSanitizerAdapter(PreemptSanitizer(recognizer), recognizer)
        # Sanitize up front on one thread; only the LLM calls run in parallel.
        base_seed = config["dataset"]["seed"]
        sanitized = [
            adapter.sanitize(item.text, epsilon=args.epsilon, seed=item_seed(base_seed, item.id, args.draw))
            for item in todo
        ]

        generate = Generator(
            create_generator_client(),
            CompletionCache(runs_dir / "cache"),
            temperature=config["llm"]["generator_temperature"],
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool, results_path.open("a") as out:
            futures = {
                pool.submit(run_item, item, result, generate, adapter, draw=args.draw): item
                for item, result in zip(todo, sanitized)
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception:
                    logger.exception("[%s] failed; rerun to retry", item.id)
                    continue
                out.write(record.model_dump_json() + "\n")
                out.flush()
                logger.info("[%s] done (A=%s A'=%s B=%s)", item.id, record.answer_a, record.answer_a_prime, record.answer_b)

    print(summarize(results_path))


if __name__ == "__main__":
    main()
