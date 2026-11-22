"""FastAPI integration layer over the LangGraph workflow."""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from langgraph.types import StateSnapshot

from ..schemas import (
    ApprovalDecision,
    MatchEvaluation,
    ResumeRequest,
    RunCreateRequest,
    RunResponse,
    RunMetrics,
)
from ..graph.workflow import (
    interrupt_payload,
    make_config,
    resume_run,
    start_run,
    status_from_result,
)

app = FastAPI(
    title="JobMatch Agent API",
    version="0.1.0",
    description="Stateful CV/job matching with a human approval gate.",
)


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """One compiled graph (and one SQLite connection) per process."""
    from ..graph.workflow import build_graph

    return build_graph()


def _snapshot_response(thread_id: str, snapshot: StateSnapshot) -> RunResponse:
    values = snapshot.values or {}
    approval = values.get("approval")
    pending = snapshot.interrupts
    payload = None
    if pending:
        value = pending[0].value
        payload = value if isinstance(value, dict) else {"value": value}

    return RunResponse(
        thread_id=thread_id,
        status="awaiting_approval" if pending else "completed",
        final_answer=values.get("final_answer"),
        draft_assessment=values.get("draft_assessment"),
        approval=ApprovalDecision.model_validate(approval) if approval else None,
        evaluation=(
            MatchEvaluation.model_validate(values["evaluation"])
            if values.get("evaluation")
            else None
        ),
        rubrics=values.get("rubrics"),
        metrics=(
            RunMetrics.model_validate(values["metrics"])
            if values.get("metrics")
            else None
        ),
        interrupt=payload,
    )


def _result_response(thread_id: str, result: dict[str, Any]) -> RunResponse:
    approval = result.get("approval")
    return RunResponse(
        thread_id=thread_id,
        status=status_from_result(result),  # type: ignore[arg-type]
        final_answer=result.get("final_answer"),
        draft_assessment=result.get("draft_assessment"),
        approval=ApprovalDecision.model_validate(approval) if approval else None,
        evaluation=(
            MatchEvaluation.model_validate(result["evaluation"])
            if result.get("evaluation")
            else None
        ),
        rubrics=result.get("rubrics"),
        metrics=(
            RunMetrics.model_validate(result["metrics"])
            if result.get("metrics")
            else None
        ),
        interrupt=interrupt_payload(result),
    )


@app.post("/runs", response_model=RunResponse, status_code=status.HTTP_200_OK)
def create_run(
    payload: RunCreateRequest, graph: Any = Depends(get_graph)
) -> RunResponse:
    """Start a run. It normally stops at the human approval gate."""
    thread_id = payload.thread_id or str(uuid.uuid4())
    result = start_run(
        graph,
        thread_id=thread_id,
        message=payload.message,
        cv_path=payload.cv_path,
        job_links=payload.job_links,
    )
    return _result_response(thread_id, result)


@app.get("/runs/{thread_id}", response_model=RunResponse)
def read_run(thread_id: str, graph: Any = Depends(get_graph)) -> RunResponse:
    """Read the latest checkpointed state for a thread."""
    snapshot = graph.get_state(make_config(thread_id))
    if snapshot is None or snapshot.created_at is None:
        raise HTTPException(status_code=404, detail=f"Unknown thread: {thread_id}")
    return _snapshot_response(thread_id, snapshot)


@app.post("/runs/{thread_id}/resume", response_model=RunResponse)
def resume(
    thread_id: str,
    payload: ResumeRequest,
    graph: Any = Depends(get_graph),
) -> RunResponse:
    """Resume a paused run with a validated approve/override decision."""
    snapshot = graph.get_state(make_config(thread_id))
    if snapshot is None or snapshot.created_at is None:
        raise HTTPException(status_code=404, detail=f"Unknown thread: {thread_id}")
    if not snapshot.interrupts:
        raise HTTPException(
            status_code=400, detail="This run is not awaiting human approval"
        )

    result = resume_run(graph, thread_id, payload.decision)
    return _result_response(thread_id, result)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
