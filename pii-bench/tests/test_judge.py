"""Tests for the utility judge prompt, verdict parsing, and both presentation orders."""

import json

from metrics.judge import (
    answer_blocks,
    build_prompt,
    consensus,
    format_substitution_table,
    judge_pair,
    parse_verdict,
)
from models import JudgeRecord, RunRecord, Substitution
from runner.cache import CompletionCache


def _record() -> RunRecord:
    return RunRecord(
        item_id="item-1",
        task_family="retirement_eligibility",
        verifiable=True,
        generator_model="gpt-oss-120b",
        generator_temperature=0.7,
        epsilon=0.2,
        draw=0,
        seed=1,
        original_prompt="Is Sam, age 60, eligible?",
        sanitized_prompt="Is Lee, age 58, eligible?",
        substitutions=[
            Substitution(
                label="NAME",
                original="Sam",
                surrogate="Lee",
                mechanism="fpe",
                reversible=True,
            ),
            Substitution(
                label="AGE",
                original="60",
                surrogate="58",
                original_value=60,
                surrogate_value=58,
                mechanism="mldp",
                reversible=False,
            ),
        ],
        response_a="Sam qualifies.\nFINAL_ANSWER: ELIGIBLE",
        response_a_prime="Yes, Sam qualifies.\nFINAL_ANSWER: ELIGIBLE",
        response_b_raw="Lee qualifies.\nFINAL_ANSWER: ELIGIBLE",
        response_b="Sam qualifies.\nFINAL_ANSWER: ELIGIBLE",
    )


class _FakeJudge:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.settings = type("S", (), {"model": "claude-sonnet-5-5"})()

    def complete_json(self, prompt: str, **kwargs) -> dict:
        self.prompts.append(prompt)
        return {"overall_compatibility": 5, "equally_well_served": "Yes", "reason": "Same decision."}


def test_substitution_table_lists_values() -> None:
    """Names and numeric values both show up in the table the judge sees."""
    table = format_substitution_table(_record().substitutions)
    assert "NAME: Sam -> Lee" in table
    assert "AGE: 60 -> 58 (60.0 -> 58.0)" in table
    assert format_substitution_table([]) == "none"


def test_prompt_swaps_answer_order_and_hides_duplicate_prompt() -> None:
    """A vs A' does not repeat the prompt, and the reverse call flips answer order."""
    record = _record()
    blocks = answer_blocks(record, "A_A_prime")
    forward = build_prompt(record, "A_A_prime", blocks)
    reverse = build_prompt(record, "A_A_prime", (blocks[1], blocks[0]))
    assert "Nothing was sanitized." in forward
    assert "Prompt both samples were generated from:" in forward
    assert "same as the original prompt" in forward
    assert forward.index("Sample A (original prompt):") < forward.index("Sample A' (independent resample")
    assert reverse.index("Sample A' (independent resample") < reverse.index("Sample A (original prompt):")

    sanitized = build_prompt(record, "A_B", answer_blocks(record, "A_B"))
    assert "Is Lee, age 58, eligible?" in sanitized
    assert "NAME: Sam -> Lee" in sanitized


def test_parse_verdict_accepts_yes_no_and_rejects_bad_scores() -> None:
    """Scores must be integers 1-5 and equal-service must be yes or no."""
    verdict = parse_verdict({"overall_compatibility": "4", "equally_well_served": " NO ", "reason": "Gap."})
    assert verdict.overall_compatibility == 4
    assert verdict.equally_well_served == "no"
    for payload in (
        {"overall_compatibility": 6, "equally_well_served": "yes", "reason": "x"},
        {"overall_compatibility": 4.5, "equally_well_served": "yes", "reason": "x"},
        {"overall_compatibility": True, "equally_well_served": "yes", "reason": "x"},
        {"overall_compatibility": 3, "equally_well_served": "maybe", "reason": "x"},
        {"overall_compatibility": 3, "equally_well_served": "yes", "reason": "  "},
    ):
        try:
            parse_verdict(payload)
        except ValueError:
            continue
        raise AssertionError(payload)


def test_consensus() -> None:
    """The two orders agree, or the pair is marked mixed."""
    assert consensus("yes", "yes") == "yes"
    assert consensus("no", "no") == "no"
    assert consensus("yes", "no") == "mixed"


def test_judge_pair_calls_both_orders_and_uses_cache(tmp_path) -> None:
    """Each pair is judged twice, and a second call hits the disk cache."""
    client = _FakeJudge()
    cache = CompletionCache(tmp_path)
    judged = judge_pair(_record(), "A_B", client, cache, temperature=0)
    assert isinstance(judged, JudgeRecord)
    assert judged.score_mean == 5
    assert judged.served_consensus == "yes"
    assert len(client.prompts) == 2
    assert client.prompts[0] != client.prompts[1]

    judge_pair(_record(), "A_B", client, cache, temperature=0)
    assert len(client.prompts) == 2
    stored = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert stored["request"]["role"] == "judge"
