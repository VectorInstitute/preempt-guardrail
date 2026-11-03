"""Summarize a run by epsilon: agreement and judge score versus the noise floor.

Reads ``runs/<run-id>/results.jsonl`` and, when present, ``judge.jsonl``.
Verifiable rows only. A vs A' is the generator noise floor; A vs B is the
sanitized prompt with names restored.

Usage (from the ``pii-bench/`` directory)::

    uv run --project .. python -m report.summarize --run-id sweep
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml
from models import JudgeRecord, RunRecord


BENCH_DIR = Path(__file__).resolve().parent.parent


def _rate(values: list[bool | None]) -> str:
    scored = [value for value in values if value is not None]
    if not scored:
        return "n/a"
    agreed = sum(scored)
    return f"{agreed}/{len(scored)}"


def _judge_cell(rows: list[JudgeRecord], pair: str) -> str:
    group = [row for row in rows if row.pair == pair]
    if not group:
        return "n/a"
    mean = sum(row.score_mean for row in group) / len(group)
    yes = sum(row.served_consensus == "yes" for row in group)
    return f"{mean:.2f}, served {yes}/{len(group)}"


def _format_eps(epsilon: float) -> str:
    if epsilon == int(epsilon):
        return str(int(epsilon))
    return f"{epsilon:g}"


def render(results: list[RunRecord], judged: list[JudgeRecord]) -> str:
    """Markdown tables of agreement and judge scores, pooled and per family."""
    verifiable = [row for row in results if row.verifiable]
    incidental = [row for row in results if not row.verifiable]
    verifiable_families = {row.task_family for row in verifiable}
    by_eps: dict[float, list[RunRecord]] = defaultdict(list)
    judge_by_eps: dict[float, list[JudgeRecord]] = defaultdict(list)
    for row in verifiable:
        by_eps[row.epsilon].append(row)
    for row in judged:
        if row.task_family in verifiable_families:
            judge_by_eps[row.epsilon].append(row)

    lines = [
        "# Run summary",
        "",
        f"Verifiable rows: {len(verifiable)}. Incidental rows: {len(incidental)}. Judge rows: {len(judged)}.",
        "A vs A' is two samples of the original prompt. A vs B is the sanitized prompt with names restored.",
        "Judge cells are the mean 1-5 score across both presentation orders, then how often both orders said the user was equally well served.",
        "Verifiable prompts ask for a decision or a number, so a changed age or amount can change the answer.",
        "Incidental prompts ask for a draft or a general explanation. The name, age, and amount are in the prompt, and the request does not turn on their exact values.",
        "",
        "## Per epsilon (verifiable)",
        "",
        "| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |",
        "|---|---|---|---|---|",
    ]
    for epsilon in sorted(by_eps):
        rows = by_eps[epsilon]
        judges = judge_by_eps.get(epsilon, [])
        lines.append(
            f"| {_format_eps(epsilon)} | {_rate([r.parity_a_a_prime for r in rows])} "
            f"| {_rate([r.parity_a_b for r in rows])} "
            f"| {_judge_cell(judges, 'A_A_prime')} | {_judge_cell(judges, 'A_B')} |"
        )

    unparsed = sum(answer is None for row in verifiable for answer in (row.answer_a, row.answer_a_prime, row.answer_b))
    lines += ["", f"FINAL_ANSWER parse failures: {unparsed}/{3 * len(verifiable)}.", ""]

    families = sorted({row.task_family for row in verifiable})
    if len(families) > 1:
        lines += ["## Per family and epsilon", ""]
        for family in families:
            lines += [
                f"### {family}",
                "",
                "| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |",
                "|---|---|---|---|---|",
            ]
            family_eps = sorted({row.epsilon for row in verifiable if row.task_family == family})
            for epsilon in family_eps:
                rows = [row for row in by_eps[epsilon] if row.task_family == family]
                judges = [row for row in judge_by_eps.get(epsilon, []) if row.task_family == family]
                lines.append(
                    f"| {_format_eps(epsilon)} | {_rate([r.parity_a_a_prime for r in rows])} "
                    f"| {_rate([r.parity_a_b for r in rows])} "
                    f"| {_judge_cell(judges, 'A_A_prime')} | {_judge_cell(judges, 'A_B')} |"
                )
            lines.append("")

    incidental_families = sorted({row.task_family for row in incidental})
    if incidental_families:
        lines += [
            "## Incidental prompts",
            "",
            "No FINAL_ANSWER. The comparison is the judge against the A vs A' noise floor.",
            "",
        ]
        for family in incidental_families:
            lines += [
                f"### {family}",
                "",
                "| ε | Judge A vs A′ | Judge A vs B |",
                "|---|---|---|",
            ]
            family_rows = [row for row in incidental if row.task_family == family]
            for epsilon in sorted({row.epsilon for row in family_rows}):
                judges = [row for row in judged if row.task_family == family and row.epsilon == epsilon]
                lines.append(
                    f"| {_format_eps(epsilon)} | {_judge_cell(judges, 'A_A_prime')} | {_judge_cell(judges, 'A_B')} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_jsonl(path: Path, model: type[RunRecord] | type[JudgeRecord]) -> list:
    """Read a JSONL file of ``model`` rows. A missing file is an empty list."""
    if not path.exists():
        return []
    with path.open() as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def main() -> None:
    """Write ``reports/<run-id>.md`` for one pipeline run."""
    config = yaml.safe_load((BENCH_DIR / "config.yaml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=BENCH_DIR / config["paths"]["runs_dir"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.runs_dir / args.run_id
    results = load_jsonl(run_dir / "results.jsonl", RunRecord)
    judged = load_jsonl(run_dir / "judge.jsonl", JudgeRecord)
    if not results:
        raise SystemExit(f"no results in {run_dir / 'results.jsonl'}")

    text = render(results, judged)
    out = args.out or (BENCH_DIR / config["paths"]["reports_dir"] / f"{args.run_id}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, end="")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
