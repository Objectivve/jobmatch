"""Pydantic contracts shared by the graph, the API and the tests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ApprovalDecision(BaseModel):
    """Human-in-the-loop payload accepted by POST /runs/{thread_id}/resume."""

    approved: bool = Field(
        description="True accepts the draft assessment; False asks for a revision."
    )
    note: str = Field(
        default="", description="Reviewer's reason or instructions for a revision."
    )
    override_summary: str | None = Field(
        default=None,
        description="Optional replacement text; when set it becomes the final answer verbatim.",
    )


class CVEvidence(BaseModel):
    """A grounded claim about a skill, tied to extracted CV text."""

    skill: str
    evidence: str
    source: str = "cv"


class JobRequirement(BaseModel):
    """A requirement extracted from a job summary."""

    requirement: str
    category: str = "required"
    source: str = "job"


class MatchEvaluation(BaseModel):
    """Typed assessment consumed by the API and the future eval harness."""

    summary: str
    overall_fit: Literal["strong", "moderate", "weak", "unclear"]
    matched_skills: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    cv_evidence: list[CVEvidence] = Field(default_factory=list)
    job_requirements: list[JobRequirement] = Field(default_factory=list)
    citations: list[str] = Field(
        default_factory=list, description="Rubric evidence IDs supporting the assessment."
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RunMetrics(BaseModel):
    model_calls: int = 0
    models_tried: list[str] = Field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    estimated_cost_usd: float = 0.0
    tool_calls: int = 0
    cache_hits: int = 0
    agent_seconds: float = 0.0
    wall_seconds: float | None = None


class RunCreateRequest(BaseModel):
    message: str = Field(min_length=1, description="What the agent should do.")
    cv_path: str | None = Field(default=None, description="Local CV path to extract.")
    job_links: list[str] = Field(
        default_factory=list, description="Job posting URLs to summarise."
    )
    thread_id: str | None = Field(
        default=None, description="Reuse an existing conversation thread."
    )


class ResumeRequest(BaseModel):
    decision: ApprovalDecision


RunStatus = Literal["awaiting_approval", "completed", "error"]


class RunResponse(BaseModel):
    thread_id: str
    status: RunStatus
    final_answer: str | None = None
    draft_assessment: str | None = None
    approval: ApprovalDecision | None = None
    evaluation: MatchEvaluation | None = None
    rubrics: list[dict] | None = None
    metrics: RunMetrics | None = None
    interrupt: dict | None = None
    error: str | None = None
