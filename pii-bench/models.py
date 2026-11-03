"""Pydantic schemas shared across the PII sanitizer benchmark."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from preempt.contracts import EntitySpan


EntityType = Literal["NAME", "AGE", "MONEY"]
Mechanism = Literal["fpe", "mldp", "random"]

ALL_ENTITY_TYPES: frozenset[str] = frozenset({"NAME", "AGE", "MONEY"})


class Replacement(BaseModel):
    """A detected span and the text written in its place."""

    span: EntitySpan
    replacement: str


class Substitution(BaseModel):
    """One unique original value and its surrogate, with parsed values for metrics.

    Per-occurrence offsets live in ``SanitizeResult.replacements``.
    """

    label: EntityType
    original: str
    surrogate: str
    original_value: float | None = None
    surrogate_value: float | None = None
    currency: str | None = None
    mechanism: Mechanism
    reversible: bool


# How a task's output depends on the concrete values of its PII slots:
# - "incidental": PII is background context/labels; task quality does not depend on exact values.
# - "semantic": Task reasoning or final answer directly depends on the numerical/semantic PII values.
# - "mixed": Combines both (some slots drive reasoning/calculation, others are incidental context).
PiiDependency = Literal["incidental", "semantic", "mixed"]


class PromptTemplate(BaseModel):
    """An LLM-authored prompt skeleton with typed PII placeholders.

    Concrete dataset items are built later by filling ``slots`` into
    ``text`` (Faker names, sampled ages/amounts). The LLM used to generate
    these templates only writes wording -- it never invents a concrete
    value, so ground-truth PII spans come for free once slots are filled.
    """

    id: str
    task_family: str
    task_type: str
    pii_dependency: PiiDependency
    verifiable: bool
    text: str
    # Placeholder tokens, including repeats such as MONEY_2. The label is the
    # part before the underscore; the suffix is a separate entity.
    slots: list[str]
    notes: str = ""


class PiiSpan(BaseModel):
    """A ground-truth PII span in a filled prompt, known from the slot fill."""

    label: EntityType
    start: int
    end: int
    text: str
    # Which real-world entity a span refers to. {NAME} twice is name-1 both
    # times. {NAME_2} is a different person, name-2.
    entity_id: str
    value: float | None = None
    currency: str | None = None
    surface_format: str | None = None


class DatasetItem(BaseModel):
    """One filled, single-turn benchmark prompt with exact PII labels."""

    id: str
    template_id: str
    task_family: str
    task_type: str
    pii_dependency: PiiDependency
    verifiable: bool
    seed: str
    text: str
    pii_spans: list[PiiSpan]


class SanitizeResult(BaseModel):
    """Output of a single sanitizer draw on a single text."""

    original_text: str
    text: str
    spans: list[EntitySpan]
    replacements: list[Replacement]
    substitutions: list[Substitution]
    epsilon: float | None
    seed: int
    enabled_types: set[EntityType]
    oracle: bool = False
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    """One pipeline result for one item, epsilon, and sanitizer draw."""

    item_id: str
    task_family: str
    verifiable: bool
    generator_model: str
    generator_temperature: float
    epsilon: float
    draw: int
    seed: int
    original_prompt: str
    sanitized_prompt: str
    substitutions: list[Substitution]
    sanitizer_warnings: list[str] = Field(default_factory=list)
    response_a: str
    response_a_prime: str
    response_b_raw: str
    response_b: str
    answer_a: str | float | None = None
    answer_a_prime: str | float | None = None
    answer_b: str | float | None = None
    parity_a_a_prime: bool | None = None
    parity_a_b: bool | None = None


JudgePair = Literal["A_B", "A_A_prime"]
Served = Literal["yes", "no"]
ServedConsensus = Literal["yes", "no", "mixed"]


class JudgeVerdict(BaseModel):
    """One ordering of a pair: a 1-5 score, equal service, and a short reason.

    An ordering is which of the two answers is shown first. The labels stay on
    their answers; only the position in the prompt changes.
    """

    overall_compatibility: int
    equally_well_served: Served
    reason: str


class JudgeRecord(BaseModel):
    """Both orderings of one comparison (A vs B, or A vs A').

    ``forward`` is the canonical order: A then B, or A then A'. ``reverse`` is
    that same pair with the two answers swapped. ``score_mean`` averages the
    two 1-5 scores. ``served_consensus`` is ``yes`` or ``no`` when both
    orderings agree, and ``mixed`` when one says equally well served and the
    other does not.
    """

    item_id: str
    task_family: str
    epsilon: float
    draw: int
    pair: JudgePair
    judge_model: str
    forward: JudgeVerdict
    reverse: JudgeVerdict
    score_mean: float
    served_consensus: ServedConsensus
