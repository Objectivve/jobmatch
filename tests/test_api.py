"""Integration tests for the FastAPI endpoints (graph and Gemini stubbed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from jobmatch.api.main import app, get_graph

from conftest import RESUME_PDF, tool_call


@pytest.fixture
def client(settings, make_scripted_graph, fake_job_client):
    graph, model = make_scripted_graph(
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    "extract_cv_text", {"file_path": str(RESUME_PDF)}, "api_cv"
                ),
                tool_call(
                    "job_posting_tool",
                    {"job_link": "https://example.com/job"},
                    "api_job",
                ),
            ],
        ),
        AIMessage(content="Draft from script."),
    )
    app.dependency_overrides[get_graph] = lambda: graph
    with TestClient(app) as test_client:
        yield test_client, model
    app.dependency_overrides.clear()


def test_full_run_lifecycle(client) -> None:
    api, _model = client

    created = api.post(
        "/runs",
        json={
            "message": "Compare my CV with this job",
            "cv_path": str(RESUME_PDF),
            "job_links": ["https://example.com/job"],
        },
    )
    assert created.status_code == 200
    body = created.json()
    thread_id = body["thread_id"]
    assert body["status"] == "awaiting_approval"
    assert body["interrupt"]["draft_assessment"] == "Draft from script."
    assert body["evaluation"]["overall_fit"] == "moderate"
    assert body["metrics"]["model_calls"] >= 1

    pending = api.get(f"/runs/{thread_id}")
    assert pending.status_code == 200
    assert pending.json()["status"] == "awaiting_approval"

    resumed = api.post(
        f"/runs/{thread_id}/resume", json={"decision": {"approved": True}}
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "completed"
    assert resumed.json()["final_answer"] == "Draft from script."
    assert resumed.json()["approval"]["approved"] is True

    assert api.get(f"/runs/{thread_id}").json()["status"] == "completed"


def test_unknown_thread_is_404(client) -> None:
    api, _model = client
    assert api.get("/runs/does-not-exist").status_code == 404


def test_resume_without_pending_approval_is_400(client) -> None:
    api, _model = client
    created = api.post("/runs", json={"message": "Assess my CV"}).json()
    thread_id = created["thread_id"]

    api.post(f"/runs/{thread_id}/resume", json={"decision": {"approved": True}})
    second = api.post(
        f"/runs/{thread_id}/resume", json={"decision": {"approved": True}}
    )
    assert second.status_code == 400


def test_invalid_decision_is_422(client) -> None:
    api, _model = client
    created = api.post("/runs", json={"message": "Assess my CV"}).json()
    response = api.post(
        f"/runs/{created['thread_id']}/resume",
        json={"decision": {"approved": "definitely"}},
    )
    assert response.status_code == 422


def test_health_endpoint(client) -> None:
    api, _model = client
    assert api.get("/health").json() == {"status": "ok"}
