"""Shared offline fixtures: no API key and no network calls in tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from jobmatch.cache import IdempotencyCache
from jobmatch.config import Settings
from jobmatch.graph.workflow import build_graph
from jobmatch.schemas import MatchEvaluation

RESUME_PDF = Path(__file__).resolve().parents[1] / "data" / "sample_cv.pdf"


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def default_evaluation() -> MatchEvaluation:
    return MatchEvaluation(
        summary="Structured draft",
        overall_fit="moderate",
        matched_skills=["Python"],
        confidence=0.8,
    )


class ScriptedModel:
    """Tiny stand-in for a chat model: scripted replies, recorded calls."""

    def __init__(self, *responses: AIMessage, structured=None) -> None:
        self.responses = list(responses)
        self.structured_responses = list(structured or [])
        self.index = 0
        self.structured_index = 0
        self.calls: list[list] = []

    def bind_tools(self, tools: list) -> "ScriptedModel":
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls.append(messages)
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return response

    def with_structured_output(self, schema: type, **kwargs):
        outer = self

        class Structured:
            def invoke(self, messages: list):
                if outer.structured_index < len(outer.structured_responses):
                    value = outer.structured_responses[outer.structured_index]
                elif outer.structured_responses:
                    value = outer.structured_responses[-1]
                else:
                    value = default_evaluation()
                outer.structured_index += 1
                return value

        return Structured()


class FakeGenAIModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class FakeGenAIClient:
    def __init__(self, text: str = "JOB TEXT: Python, LangGraph, RAG") -> None:
        self.models = FakeGenAIModels(text)


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> IdempotencyCache:
    cache = IdempotencyCache(str(tmp_path / "cache.sqlite"), ttl_seconds=60)
    monkeypatch.setattr("jobmatch.cache.get_cache", lambda: cache)
    monkeypatch.setattr("jobmatch.tools.job_posting.get_cache", lambda: cache)
    monkeypatch.setattr("jobmatch.tools.cv_parser.get_cache", lambda: cache)
    return cache


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        gemini_api_key="offline-test-key",
        models=("fake-model",),
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        recursion_limit=12,
        model_timeout_seconds=5,
    )


@pytest.fixture
def fake_job_client(monkeypatch: pytest.MonkeyPatch) -> FakeGenAIClient:
    client = FakeGenAIClient()
    monkeypatch.setattr(
        "jobmatch.tools.job_posting.make_genai_client", lambda: client
    )
    return client


@pytest.fixture
def make_scripted_graph(settings: Settings):
    def factory(*responses: AIMessage, structured=None):
        model = ScriptedModel(*responses, structured=structured)
        graph = build_graph(
            settings=settings,
            db_path=settings.checkpoint_db_path,
            model_factory=lambda _name: model,
        )
        return graph, model

    return factory
