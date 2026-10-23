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
    slots: list[EntityType]
    notes: str = ""


class PiiSpan(BaseModel):
    """A ground-truth PII span in a filled prompt, known from the slot fill."""

    label: EntityType
    start: int
    end: int
    text: str
    #  it says which real-world entity a span refers to. If a template mentions {NAME}/
    #  twice, both spans get the same entity_id (name-1)
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
