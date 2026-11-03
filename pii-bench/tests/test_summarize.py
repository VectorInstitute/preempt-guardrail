"""Tests for the per-epsilon run summary."""

from models import JudgeRecord, JudgeVerdict, RunRecord
from report.summarize import render


def _result(
    epsilon: float, *, agree_floor: bool, agree_sanitized: bool, family: str = "retirement_eligibility"
) -> RunRecord:
    return RunRecord(
        item_id=f"{family}-{epsilon}-{agree_sanitized}",
        task_family=family,
        verifiable=True,
        generator_model="gpt-oss-120b",
        generator_temperature=0.7,
        epsilon=epsilon,
        draw=0,
        seed=1,
        original_prompt="prompt",
        sanitized_prompt="sanitized",
        substitutions=[],
        response_a="a",
        response_a_prime="a'",
        response_b_raw="b",
        response_b="b",
        answer_a="ELIGIBLE",
        answer_a_prime="ELIGIBLE" if agree_floor else "NOT_ELIGIBLE",
        answer_b="ELIGIBLE" if agree_sanitized else "NOT_ELIGIBLE",
        parity_a_a_prime=agree_floor,
        parity_a_b=agree_sanitized,
    )


def _judge(epsilon: float, pair: str, score: float, served: str, family: str = "retirement_eligibility") -> JudgeRecord:
    half = int(score)
    other = int(round(score * 2 - half))
    verdict = JudgeVerdict(overall_compatibility=half, equally_well_served=served, reason="ok")
    swapped = JudgeVerdict(overall_compatibility=other, equally_well_served=served, reason="ok")
    return JudgeRecord(
        item_id="item",
        task_family=family,
        epsilon=epsilon,
        draw=0,
        pair=pair,
        judge_model="claude-sonnet-5-5",
        forward=verdict,
        reverse=swapped,
        score_mean=score,
        served_consensus=served,
    )


def test_render_pools_agreement_and_judge_by_epsilon() -> None:
    """One table row per epsilon, with the noise floor beside the sanitized pair."""
    results = [
        _result(0.1, agree_floor=True, agree_sanitized=False),
        _result(0.1, agree_floor=True, agree_sanitized=True),
        _result(1.0, agree_floor=False, agree_sanitized=True),
    ]
    judged = [
        _judge(0.1, "A_A_prime", 5.0, "yes"),
        _judge(0.1, "A_B", 2.0, "no"),
        _judge(1.0, "A_A_prime", 1.0, "no"),
        _judge(1.0, "A_B", 4.5, "yes"),
    ]
    text = render(results, judged)
    assert "| 0.1 | 2/2 | 1/2 | 5.00, served 1/1 | 2.00, served 0/1 |" in text
    assert "| 1 | 0/1 | 1/1 | 1.00, served 0/1 | 4.50, served 1/1 |" in text
    assert "FINAL_ANSWER parse failures: 0/9." in text


def test_incidental_scores_stay_out_of_the_verifiable_table() -> None:
    """A general prompt is judged on its own and does not dilute the verifiable row."""
    semantic = _result(0.2, agree_floor=True, agree_sanitized=True)
    general = _result(0.2, agree_floor=True, agree_sanitized=False, family="general_request")
    general.verifiable = False
    judged = [
        _judge(0.2, "A_B", 2.0, "no"),
        _judge(0.2, "A_B", 4.5, "yes", family="general_request"),
    ]
    text = render([semantic, general], judged)
    assert "| 0.2 | 1/1 | 1/1 | n/a | 2.00, served 0/1 |" in text
    assert "### general_request" in text
    assert "| 0.2 | n/a | 4.50, served 1/1 |" in text
