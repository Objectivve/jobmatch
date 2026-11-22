"""Graph transition, HITL and checkpoint durability tests."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from jobmatch.graph.workflow import make_config, resume_run, start_run
from jobmatch.schemas import ApprovalDecision
from jobmatch.tools.cv_parser import extract_text_with_meta
from jobmatch.tools.job_posting import fetch_job_posting

from conftest import RESUME_PDF, tool_call


def _tool_loop_script() -> tuple[AIMessage, AIMessage]:
    return (
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    "extract_cv_text", {"file_path": str(RESUME_PDF)}, "call_cv"
                ),
                tool_call(
                    "job_posting_tool",
                    {"job_link": "https://example.com/job"},
                    "call_job",
                ),
            ],
        ),
        AIMessage(content="Draft: strong LangGraph fit."),
    )


def test_tool_loop_then_pause_and_resume(
    settings, make_scripted_graph, fake_job_client
) -> None:
    graph, _model = make_scripted_graph(*_tool_loop_script())

    paused = start_run(
        graph,
        thread_id="thread-1",
        message="Compare my CV with this job",
        cv_path=str(RESUME_PDF),
        job_links=["https://example.com/job"],
        settings=settings,
    )

    assert paused["__interrupt__"]
    payload = paused["__interrupt__"][0].value
    assert payload["draft_assessment"] == "Draft: strong LangGraph fit."
    assert "Jane Doe" in payload["cv_excerpt"]
    assert any("Python" in summary for summary in payload["job_summaries"])
    assert paused["evaluation"]["overall_fit"] == "moderate"
    assert payload["evaluation"]["overall_fit"] == "moderate"
    assert paused["metrics"]["tool_calls"] == 2
    assert paused["metrics"]["model_calls"] >= 1
    assert any(
        isinstance(msg, ToolMessage) and msg.name == "extract_cv_text"
        for msg in paused["messages"]
    )

    completed = resume_run(
        graph, "thread-1", ApprovalDecision(approved=True), settings=settings
    )
    assert completed["final_answer"] == "Draft: strong LangGraph fit."
    assert completed["approval"]["approved"] is True
    assert completed["metrics"]["wall_seconds"] is not None
    assert "agent_seconds" in completed["metrics"]

    # A freshly compiled graph on the same SQLite file sees the answer.
    fresh_graph, _ = make_scripted_graph(AIMessage(content="unused"))
    snapshot = fresh_graph.get_state(make_config("thread-1", settings))
    assert snapshot.values["final_answer"] == "Draft: strong LangGraph fit."


def test_override_summary_wins_without_extra_model_call(
    settings, make_scripted_graph, fake_job_client
) -> None:
    graph, model = make_scripted_graph(AIMessage(content="Vague draft"))

    start_run(
        graph,
        thread_id="override-thread",
        message="Assess my CV",
        cv_path=str(RESUME_PDF),
        settings=settings,
    )
    completed = resume_run(
        graph,
        "override-thread",
        ApprovalDecision(approved=False, override_summary="Human summary"),
        settings=settings,
    )

    assert completed["final_answer"] == "Human summary"
    assert model.index == 1


def test_rejection_without_override_requests_revision(
    settings, make_scripted_graph, fake_job_client
) -> None:
    graph, model = make_scripted_graph(
        AIMessage(content="Vague draft"), AIMessage(content="Revised draft")
    )

    start_run(
        graph,
        thread_id="revision-thread",
        message="Assess my CV",
        cv_path=str(RESUME_PDF),
        settings=settings,
    )
    completed = resume_run(
        graph,
        "revision-thread",
        ApprovalDecision(approved=False, note="Be more specific"),
        settings=settings,
    )

    assert completed["final_answer"] == "Revised draft"
    assert model.index == 2


def test_cache_hits_are_counted_in_metrics(
    settings, make_scripted_graph, fake_job_client, isolated_cache
) -> None:
    url = "https://example.com/cached-job"
    fetch_job_posting(url)
    extract_text_with_meta(str(RESUME_PDF))

    graph, _ = make_scripted_graph(
        AIMessage(
            content="",
            tool_calls=[
                tool_call("extract_cv_text", {"file_path": str(RESUME_PDF)}, "hit_cv"),
                tool_call("job_posting_tool", {"job_link": url}, "hit_job"),
            ],
        ),
        AIMessage(content="Draft"),
    )
    result = start_run(
        graph,
        thread_id="cache-hit-thread",
        message="Assess this",
        cv_path=str(RESUME_PDF),
        job_links=[url],
        settings=settings,
    )

    assert result["metrics"]["cache_hits"] == 2
    assert len(fake_job_client.models.calls) == 1
