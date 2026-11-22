"""Golden-set harness tests, fully offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobmatch.config import Settings
from jobmatch.evals.fake_model import DeterministicModel, OFFLINE_TOOLS
from jobmatch.evals.harness import (
    evaluate_case,
    evaluate_cases,
    load_golden_set,
    summarize,
    write_reports,
)
from jobmatch.graph.workflow import build_graph, create_sqlite_checkpointer

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "evals" / "golden_set.sample.jsonl"


@pytest.fixture
def offline_settings(tmp_path: Path) -> Settings:
    return Settings(
        gemini_api_key="offline",
        models=("offline",),
        checkpoint_db_path=str(tmp_path / "eval-checkpoints.sqlite"),
        rubric_path=str(ROOT / "data" / "rubrics" / "role_rubric.sample.json"),
    )


def _make_graph(settings: Settings):
    model = DeterministicModel()
    return build_graph(
        settings=settings,
        checkpointer=create_sqlite_checkpointer(":memory:"),
        model_factory=lambda _name: model,
        tools=OFFLINE_TOOLS,
    )


def test_golden_set_loads() -> None:
    cases = load_golden_set(GOLDEN)
    assert len(cases) == 3
    assert cases[0]["case_id"] == "case-001-clean-pdf"


def test_evaluate_case_succeeds_with_grounded_evidence(offline_settings) -> None:
    case = load_golden_set(GOLDEN)[0]
    result = evaluate_case(
        case, graph=_make_graph(offline_settings), settings=offline_settings, base_dir=ROOT
    )

    assert result.error is None
    assert result.status == "completed"
    assert result.task_success is True
    assert result.grounding_passed is True
    assert result.missing_expectations == []
    assert result.citations


def test_evaluate_case_flags_unsatisfied_expectation(offline_settings) -> None:
    case = {
        "case_id": "case-impossible",
        "cv_path": "data/sample_cv.pdf",
        "job_links": ["https://example.com/job"],
        "user_message": "Assess fit",
        "expected": {"must_contain": ["this phrase is never produced"]},
    }
    result = evaluate_case(
        case, graph=_make_graph(offline_settings), settings=offline_settings, base_dir=ROOT
    )

    assert result.task_success is False
    assert result.missing_expectations


def test_evaluate_cases_and_report_writing(offline_settings, tmp_path: Path) -> None:
    cases = load_golden_set(GOLDEN)
    results = evaluate_cases(
        cases,
        graph_factory=lambda _settings: _make_graph(offline_settings),
        settings=offline_settings,
        base_dir=ROOT,
    )
    summary = summarize(results)

    assert summary["cases"] == 3
    assert 0.0 <= summary["task_success_rate"] <= 1.0
    assert 0.0 <= summary["faithfulness_rate"] <= 1.0
    assert summary["cost_total_usd"] >= 0.0

    json_path = tmp_path / "summary.json"
    md_path = tmp_path / "summary.md"
    write_reports(summary, json_path, md_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["cases"] == 3
    assert "# Evaluation summary" in md_path.read_text(encoding="utf-8")
