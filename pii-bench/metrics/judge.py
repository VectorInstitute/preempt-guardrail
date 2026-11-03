"""LLM judge for whether a sanitized answer serves the user as well as the original.

Sonnet (temperature 0) scores two pairs per pipeline record:

- **A vs B:** response to the original prompt vs response to the sanitized prompt,
  with surrogate names restored.
- **A vs A':** two independent samples of the original prompt (generator noise floor).

Each pair is judged twice, with the responses shown in opposite orders, and the
two 1–5 scores are averaged. Results append to ``runs/<run-id>/judge.jsonl``.
A rerun skips pairs already written. Completions go through ``runner.cache``.

Usage (from the ``pii-bench/`` directory, after ``runner.pipeline``)::

    uv run --project .. python -m metrics.judge --run-id pilot-e0.2
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from llm.clients import OpenAICompatibleClient, create_judge_client
from models import JudgePair, JudgeRecord, JudgeVerdict, RunRecord, Served, ServedConsensus, Substitution
from runner.cache import CompletionCache


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BENCH_DIR = Path(__file__).resolve().parent.parent
MAX_TOKENS = 512

SYSTEM = (
    "You compare two answers to one user request. "
    "Judge whether they reach the same decision and would serve the user equally well. "
    "Ignore wording that does not change the outcome."
)

_RUBRIC = """\
Score how compatible the two answers are as responses to the original prompt.
The order they appear in is arbitrary; do not prefer whichever answer is shown first.
A different age or dollar figure is expected when one answer saw a sanitized prompt.
Penalize that difference only when it changes the decision, the advice, or whether the user is equally well served.
A contradictory decision, a missed requirement, or advice that would lead the user to act differently is not compatible.

Return JSON with exactly these fields:
- overall_compatibility: integer from 1 to 5
  5 = same decision and equally useful
  4 = equally well served, with only minor differences
  3 = mostly useful, but a noticeable gap
  2 = a material omission or a shift in advice; not equally well served
  1 = contradictory decision, or one answer fails the request
- equally_well_served: "yes" or "no"
- reason: one or two sentences
"""


def format_substitution_table(substitutions: list[Substitution]) -> str:
    """One line per original value: ``LABEL: original -> surrogate``."""
    if not substitutions:
        return "none"
    lines = []
    for sub in substitutions:
        line = f"{sub.label}: {sub.original} -> {sub.surrogate}"
        if sub.original_value is not None or sub.surrogate_value is not None:
            line += f" ({sub.original_value} -> {sub.surrogate_value})"
        lines.append(line)
    return "\n".join(lines)


def comparison_context(record: RunRecord, pair: JudgePair) -> tuple[str, str, str]:
    """Return (note, prompt the second side saw, substitution table) for one pair."""
    if pair == "A_B":
        note = (
            "Answer A was generated from the original prompt. "
            "Answer B was generated from the sanitized prompt; names in Answer B have been restored."
        )
        return note, record.sanitized_prompt, format_substitution_table(record.substitutions)
    note = "Sample A and Sample A' are two independent answers to the original prompt. Nothing was sanitized."
    return note, record.original_prompt, "none"


def answer_blocks(record: RunRecord, pair: JudgePair) -> tuple[tuple[str, str], tuple[str, str]]:
    """Canonical (first, second) labeled answers: the forward ordering.

    A vs B is Answer A then Answer B. A vs A' is Sample A then Sample A'.
    The reverse ordering passes these two blocks swapped.
    """
    if pair == "A_B":
        return (
            ("Answer A (original prompt)", record.response_a),
            ("Answer B (sanitized prompt, names restored)", record.response_b),
        )
    return (
        ("Sample A (original prompt)", record.response_a),
        ("Sample A' (independent resample of the original prompt)", record.response_a_prime),
    )


def build_prompt(
    record: RunRecord,
    pair: JudgePair,
    blocks: tuple[tuple[str, str], tuple[str, str]],
) -> str:
    """User prompt for one ordering of one pair.

    ``blocks`` is whichever order this call should show: forward, or the swap
    used for reverse.
    """
    note, seen_prompt, table = comparison_context(record, pair)
    first, second = blocks
    same_prompt = seen_prompt == record.original_prompt
    seen = "same as the original prompt" if same_prompt else seen_prompt
    seen_heading = (
        "Prompt the sanitized answer was generated from" if pair == "A_B" else "Prompt both samples were generated from"
    )
    return (
        f"{note}\n\n"
        f"Original prompt:\n{record.original_prompt}\n\n"
        f"{seen_heading}:\n{seen}\n\n"
        f"Substitution table (original -> surrogate):\n{table}\n\n"
        f"{first[0]}:\n{first[1]}\n\n"
        f"{second[0]}:\n{second[1]}\n\n"
        f"{_RUBRIC}"
    )


def parse_verdict(payload: dict[str, Any]) -> JudgeVerdict:
    """Coerce a judge JSON object into a verdict, or raise ``ValueError``."""
    raw_score = payload.get("overall_compatibility")
    score_is_bool = isinstance(raw_score, bool)
    score_is_fraction = isinstance(raw_score, float) and not raw_score.is_integer()
    if score_is_bool or score_is_fraction:
        msg = f"overall_compatibility must be an integer 1-5, got {raw_score!r}"
        raise ValueError(msg)
    try:
        score = int(raw_score)
    except (TypeError, ValueError) as exc:
        msg = f"overall_compatibility must be an integer 1-5, got {raw_score!r}"
        raise ValueError(msg) from exc
    if score not in range(1, 6):
        msg = f"overall_compatibility must be an integer 1-5, got {score}"
        raise ValueError(msg)

    served = str(payload.get("equally_well_served", "")).strip().lower()
    if served not in ("yes", "no"):
        msg = f"equally_well_served must be 'yes' or 'no', got {payload.get('equally_well_served')!r}"
        raise ValueError(msg)

    reason = str(payload.get("reason", "")).strip()
    if not reason:
        msg = "reason must be a non-empty string"
        raise ValueError(msg)
    return JudgeVerdict(
        overall_compatibility=score,
        equally_well_served=cast(Served, served),
        reason=reason,
    )


def consensus(forward: Served, reverse: Served) -> ServedConsensus:
    """``yes`` or ``no`` when both orders agree, otherwise ``mixed``."""
    if forward == reverse:
        return forward
    return "mixed"


def _cached_verdict(
    client: OpenAICompatibleClient,
    cache: CompletionCache,
    prompt: str,
    *,
    temperature: float,
) -> JudgeVerdict:
    request = {
        "model": client.settings.model,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "prompt": prompt,
        "role": "judge",
    }
    key = cache.key(**request)
    if (cached := cache.get(key)) is not None:
        return parse_verdict(json.loads(cached))
    payload = client.complete_json(prompt, system=SYSTEM, temperature=temperature, max_tokens=MAX_TOKENS)
    verdict = parse_verdict(payload)
    cache.put(key, request, verdict.model_dump_json())
    return verdict


def judge_pair(
    record: RunRecord,
    pair: JudgePair,
    client: OpenAICompatibleClient,
    cache: CompletionCache,
    *,
    temperature: float,
) -> JudgeRecord:
    """Score one pair in both orderings.

    Forward shows the canonical block order. Reverse swaps the two answers so
    a preference for whichever answer appears first cannot decide the score.
    """
    blocks = answer_blocks(record, pair)
    forward = _cached_verdict(
        client,
        cache,
        build_prompt(record, pair, blocks),
        temperature=temperature,
    )
    reverse = _cached_verdict(
        client,
        cache,
        build_prompt(record, pair, (blocks[1], blocks[0])),
        temperature=temperature,
    )
    return JudgeRecord(
        item_id=record.item_id,
        task_family=record.task_family,
        epsilon=record.epsilon,
        draw=record.draw,
        pair=pair,
        judge_model=client.settings.model,
        forward=forward,
        reverse=reverse,
        score_mean=(forward.overall_compatibility + reverse.overall_compatibility) / 2,
        served_consensus=consensus(forward.equally_well_served, reverse.equally_well_served),
    )


def load_records(path: Path) -> list[RunRecord]:
    """Read pipeline ``results.jsonl``."""
    with path.open() as f:
        return [RunRecord.model_validate_json(line) for line in f if line.strip()]


def load_done(path: Path) -> set[tuple[str, float, int, str]]:
    """Pairs already written. A failed pair is absent, so a rerun retries it."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = JudgeRecord.model_validate_json(line)
            done.add((row.item_id, row.epsilon, row.draw, row.pair))
    return done


def summarize(path: Path) -> str:
    """Mean compatibility and equal-service rate for each pair."""
    with path.open() as f:
        rows = [JudgeRecord.model_validate_json(line) for line in f if line.strip()]
    lines = [f"Judgments: {len(rows)}"]
    pairs = sorted({row.pair for row in rows})
    for pair in pairs:
        group = [row for row in rows if row.pair == pair]
        mean = sum(row.score_mean for row in group) / len(group)
        yes = sum(row.served_consensus == "yes" for row in group)
        mixed = sum(row.served_consensus == "mixed" for row in group)
        lines.append(f"  {pair}: mean compatibility {mean:.2f}, equally well served {yes}/{len(group)}, mixed {mixed}")
    return "\n".join(lines)


def main() -> None:
    """Score each pair in a run and append ``judge.jsonl``."""
    load_dotenv(BENCH_DIR / ".env")
    config = yaml.safe_load((BENCH_DIR / "config.yaml").read_text())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="pilot-e0.2")
    parser.add_argument("--pairs", nargs="+", choices=["A_B", "A_A_prime"], default=["A_B", "A_A_prime"])
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only judge the first N pipeline records.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    runs_dir = BENCH_DIR / config["paths"]["runs_dir"]
    results_path = runs_dir / args.run_id / "results.jsonl"
    judge_path = runs_dir / args.run_id / "judge.jsonl"
    judge_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(results_path)
    if args.families is not None:
        records = [r for r in records if r.task_family in args.families]
    records = records[: args.limit]
    done = load_done(judge_path)
    todo = [
        (record, pair)
        for record in records
        for pair in args.pairs
        if (record.item_id, record.epsilon, record.draw, pair) not in done
    ]
    logger.info("%d records, %d pairs to judge", len(records), len(todo))

    if todo:
        client = create_judge_client()
        cache = CompletionCache(runs_dir / "cache")
        temperature = float(config["llm"]["judge_temperature"])
        with ThreadPoolExecutor(max_workers=args.workers) as pool, judge_path.open("a") as out:
            futures = {
                pool.submit(judge_pair, record, pair, client, cache, temperature=temperature): (record, pair)
                for record, pair in todo
            }
            for future in as_completed(futures):
                record, pair = futures[future]
                try:
                    judged = future.result()
                except Exception:
                    logger.exception("[%s %s] failed; rerun to retry", record.item_id, pair)
                    continue
                out.write(judged.model_dump_json() + "\n")
                out.flush()
                logger.info(
                    "[%s %s] %.1f (%s)",
                    record.item_id,
                    pair,
                    judged.score_mean,
                    judged.served_consensus,
                )

    print(summarize(judge_path))


if __name__ == "__main__":
    main()
