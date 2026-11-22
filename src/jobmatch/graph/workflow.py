"""LangGraph workflow: tool loop -> human approval -> final answer."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from ..config import Settings, get_settings
from ..retrieval import retrieve_rubrics
from ..schemas import ApprovalDecision, CVEvidence, JobRequirement, MatchEvaluation
from ..telemetry import (
    add_duration,
    add_usage,
    empty_metrics,
    finish_metrics,
    record_attempt,
)
from ..tools import extract_cv_text, job_posting_tool
from .prompts import REVISION_INSTRUCTION, SYSTEM_MESSAGE
from .state import AgentState

logger = logging.getLogger(__name__)

TOOLS = [job_posting_tool, extract_cv_text]
ModelFactory = Callable[[str], Any]


def make_config(thread_id: str, settings: Settings | None = None) -> dict[str, Any]:
    """Every graph call needs the same thread id and recursion guard."""
    settings = settings or get_settings()
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.recursion_limit,
    }


def build_user_message(
    message: str, cv_path: str | None = None, job_links: list[str] | None = None
) -> HumanMessage:
    """Turn API fields into one user message the assistant can reason over."""
    parts = [message]
    if cv_path:
        parts.append(f"CV file: {cv_path}")
    for link in job_links or []:
        parts.append(f"Job posting: {link}")
    return HumanMessage(content="\n\n".join(parts))


def _text_from(content: Any) -> str:
    """Normalise str / list-of-blocks model content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif hasattr(block, "text") and block.text:
                parts.append(str(block.text))
        return "\n".join(parts)
    return str(content)


def _tool_evidence(messages: list[Any]) -> tuple[str, list[str]]:
    """Collect the CV text and job summaries produced by tool calls."""
    cv_text = ""
    job_summaries: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name == extract_cv_text.name:
            cv_text = _text_from(message.content)
        elif message.name == job_posting_tool.name:
            job_summaries.append(_text_from(message.content))
    return cv_text, job_summaries


def _route_after_assistant(state: AgentState) -> str:
    """Tools first; when the assistant is done, retrieve governed rubrics."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "retrieve"


def _default_model_factory(settings: Settings) -> ModelFactory:
    def factory(model_name: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.gemini_api_key,
            timeout=settings.model_timeout_seconds,
            max_retries=1,
        )

    return factory


def invoke_with_model_fallback(
    messages: list[Any],
    *,
    tools_list: list[Any] | None = None,
    settings: Settings | None = None,
    model_factory: ModelFactory | None = None,
    metrics: dict[str, Any] | None = None,
) -> Any:
    """Try each configured model in order; raise only if all of them fail."""
    settings = settings or get_settings()
    factory = model_factory or _default_model_factory(settings)
    last_error: Exception | None = None

    for model_name in settings.models:
        try:
            record_attempt(metrics, model_name)
            model = factory(model_name)
            runnable = model.bind_tools(tools_list) if tools_list else model
            response = runnable.invoke(messages)
            add_usage(metrics, response, model_name)
            return response
        except Exception as exc:
            last_error = exc
            logger.warning("Model %s failed, trying next: %s", model_name, exc)

    raise RuntimeError(f"All models failed. Last error: {last_error}")


def invoke_structured(
    messages: list[Any],
    schema: type,
    *,
    settings: Settings | None = None,
    model_factory: ModelFactory | None = None,
    metrics: dict[str, Any] | None = None,
) -> Any:
    """Return a Pydantic-validated model response, with the same fallback."""
    settings = settings or get_settings()
    factory = model_factory or _default_model_factory(settings)
    last_error: Exception | None = None

    for model_name in settings.models:
        try:
            record_attempt(metrics, model_name)
            model = factory(model_name)
            with_structured_output = getattr(model, "with_structured_output", None)
            if with_structured_output is None:
                raise TypeError("model does not support with_structured_output")
            result = with_structured_output(schema, include_raw=True).invoke(messages)
            if isinstance(result, dict) and "raw" in result:
                add_usage(metrics, result["raw"], model_name)
                parsed = result.get("parsed")
                if parsed is None:
                    raise ValueError("structured output could not be parsed")
                return parsed
            return result
        except Exception as exc:
            last_error = exc
            logger.warning("Structured output failed on %s: %s", model_name, exc)

    raise RuntimeError(f"Structured output failed on all models: {last_error}")


def assistant_node(
    state: AgentState,
    *,
    settings: Settings,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    metrics = dict(state.get("metrics") or empty_metrics())
    started = time.perf_counter()
    response = invoke_with_model_fallback(
        [SystemMessage(content=SYSTEM_MESSAGE), *state["messages"]],
        tools_list=TOOLS,
        settings=settings,
        model_factory=model_factory,
        metrics=metrics,
    )
    add_duration(metrics, started)
    update: dict[str, Any] = {"messages": [response], "metrics": metrics}
    if not getattr(response, "tool_calls", None):
        update["draft_assessment"] = _text_from(response.content)
    return update


def retrieve_node(state: AgentState, *, settings: Settings) -> dict[str, Any]:
    """Fetch verified rubrics for the request and keep their citations."""
    user_query = " ".join(
        _text_from(message.content)
        for message in state["messages"]
        if isinstance(message, HumanMessage)
    )
    matches = retrieve_rubrics(query=user_query, settings=settings)
    return {"rubrics": [match.to_dict() for match in matches]}


def structure_node(
    state: AgentState,
    *,
    settings: Settings,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    """Convert the draft plus tool evidence into a typed MatchEvaluation."""
    metrics = dict(state.get("metrics") or empty_metrics())
    cv_text, job_summaries = _tool_evidence(state["messages"])
    draft = state.get("draft_assessment", "")
    started = time.perf_counter()

    messages = [
        SystemMessage(
            content=(
                "Extract grounded evidence only. Never invent a skill, date or "
                "requirement. Governed rubrics are reference context: use them "
                "only for `citations`. A skill may appear in `matched_skills` "
                "only if it is present in the CV TEXT or JOB SUMMARIES. "
                "Return the required Pydantic schema."
            )
        ),
        HumanMessage(
            content=(
                f"CV TEXT:\n{cv_text[:8000]}\n\n"
                f"JOB SUMMARIES:\n{'\n---\n'.join(job_summaries)[:8000]}\n\n"
                "GOVERNED RUBRICS (reference the IDs in `citations`):\n"
                f"{json.dumps(state.get('rubrics', []))[:4000]}\n\n"
                f"DRAFT ASSESSMENT:\n{draft}"
            )
        ),
    ]

    try:
        evaluation = invoke_structured(
            messages,
            MatchEvaluation,
            settings=settings,
            model_factory=model_factory,
            metrics=metrics,
        )
    except Exception as exc:
        logger.warning("Using deterministic evaluation fallback: %s", exc)
        evaluation = MatchEvaluation(
            summary=draft or "No assessment available.",
            overall_fit="unclear",
            confidence=0.0,
            cv_evidence=[
                CVEvidence(skill="extracted_cv", evidence=cv_text[:300], source="cv")
            ]
            if cv_text
            else [],
            job_requirements=[
                JobRequirement(requirement=summary[:300], source="job")
                for summary in job_summaries[:5]
            ],
            citations=[
                str(rubric["citation"])
                for rubric in state.get("rubrics", [])
                if rubric.get("citation")
            ],
        )

    add_duration(metrics, started)
    tool_messages = [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ]
    metrics["tool_calls"] = len(tool_messages)
    metrics["cache_hits"] = sum(
        1
        for message in tool_messages
        if isinstance(message.artifact, dict) and message.artifact.get("cache_hit")
    )
    return {"evaluation": evaluation.model_dump(), "metrics": metrics}


def approval_node(state: AgentState) -> dict[str, Any]:
    """Pause with evidence; the resume value becomes a validated decision."""
    cv_text, job_summaries = _tool_evidence(state["messages"])
    payload = {
        "draft_assessment": state.get("draft_assessment", ""),
        "cv_excerpt": cv_text[:4000],
        "job_summaries": job_summaries,
        "evaluation": state.get("evaluation"),
        "rubrics": state.get("rubrics", []),
        "rubric_citations": [
            rubric.get("citation")
            for rubric in state.get("rubrics", [])
            if rubric.get("citation")
        ],
        "instruction": (
            "Approve this assessment, or supply a note / override_summary. "
            "The run is paused until you resume it."
        ),
    }
    raw_decision = interrupt(payload)
    decision = ApprovalDecision.model_validate(raw_decision)
    return {"approval": decision.model_dump()}


def finalize_node(
    state: AgentState,
    *,
    settings: Settings,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    """Apply the human decision deterministically, then record the answer."""
    decision = ApprovalDecision.model_validate(state.get("approval") or {})
    draft = state.get("draft_assessment", "")
    metrics = dict(state.get("metrics") or empty_metrics())
    started = time.perf_counter()

    if decision.override_summary:
        final = decision.override_summary
    elif decision.approved:
        final = draft
    else:
        revised = invoke_with_model_fallback(
            [
                SystemMessage(content=SYSTEM_MESSAGE),
                *state["messages"],
                HumanMessage(content=f"{REVISION_INSTRUCTION}\n\nReviewer note: {decision.note}"),
            ],
            settings=settings,
            model_factory=model_factory,
            metrics=metrics,
        )
        final = _text_from(revised.content)

    add_duration(metrics, started)
    finish_metrics(metrics)
    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
        "metrics": metrics,
    }


def create_sqlite_checkpointer(db_path: str | Path) -> SqliteSaver:
    """Persistent checkpointer; the caller keeps the returned saver alive."""
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(connection)


def create_checkpointer(settings: Settings, db_path: str | Path | None = None) -> Any:
    """Select SQLite locally or Postgres in production-style deployments."""
    backend = settings.checkpoint_backend.lower()
    if backend == "sqlite":
        return create_sqlite_checkpointer(db_path or settings.checkpoint_db_path)
    if backend == "postgres":
        try:
            import psycopg
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Postgres backend needs: uv add langgraph-checkpoint-postgres 'psycopg[binary]'"
            ) from exc
        if not settings.database_url:
            raise RuntimeError("JOBMATCH_DATABASE_URL is required for Postgres")
        return PostgresSaver(psycopg.connect(settings.database_url))
    raise ValueError(f"Unknown checkpoint backend: {settings.checkpoint_backend}")


def build_graph(
    *,
    checkpointer: Any | None = None,
    settings: Settings | None = None,
    model_factory: ModelFactory | None = None,
    db_path: str | Path | None = None,
    tools: list[Any] | None = None,
) -> Any:
    """Build the same compiled graph for the API, tests and the demo notebook."""
    settings = settings or get_settings()
    active_tools = list(tools) if tools is not None else list(TOOLS)

    builder = StateGraph(AgentState)
    builder.add_node(
        "assistant",
        partial(
            assistant_node, settings=settings, model_factory=model_factory
        ),
    )
    builder.add_node("tools", ToolNode(active_tools))
    builder.add_node("retrieve", partial(retrieve_node, settings=settings))
    builder.add_node(
        "structure",
        partial(structure_node, settings=settings, model_factory=model_factory),
    )
    builder.add_node("approval", approval_node)
    builder.add_node(
        "finalize",
        partial(
            finalize_node, settings=settings, model_factory=model_factory
        ),
    )

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        _route_after_assistant,
        {"tools": "tools", "retrieve": "retrieve"},
    )
    builder.add_edge("tools", "assistant")
    builder.add_edge("retrieve", "structure")
    builder.add_edge("structure", "approval")
    builder.add_edge("approval", "finalize")
    builder.add_edge("finalize", END)

    if checkpointer is None:
        checkpointer = create_checkpointer(settings, db_path)
    return builder.compile(checkpointer=checkpointer)


def start_run(
    graph: Any,
    *,
    thread_id: str,
    message: str,
    cv_path: str | None = None,
    job_links: list[str] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    return graph.invoke(
        {
            "messages": [build_user_message(message, cv_path, job_links)],
            "metrics": empty_metrics(),
        },
        config=make_config(thread_id, settings),
    )


def resume_run(
    graph: Any,
    thread_id: str,
    decision: ApprovalDecision,
    settings: Settings | None = None,
) -> dict[str, Any]:
    return graph.invoke(
        Command(resume=decision.model_dump()),
        config=make_config(thread_id, settings),
    )


def status_from_result(result: dict[str, Any]) -> str:
    return "awaiting_approval" if result.get("__interrupt__") else "completed"


def interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    items = result.get("__interrupt__") or ()
    if not items:
        return None
    value = items[0].value
    return value if isinstance(value, dict) else {"value": value}
