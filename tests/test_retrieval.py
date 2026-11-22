"""Governed rubric retrieval: metadata filters, dates and citations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobmatch.config import Settings
from jobmatch.retrieval import load_rubrics, retrieve_rubrics


@pytest.fixture
def rubric_file(tmp_path: Path) -> Path:
    records = [
        {
            "rubric_id": "r-active",
            "role": "AI / LLM Engineer",
            "domain": "ai_engineering",
            "seniority_level": "junior_to_mid",
            "source_type": "curated_role_rubric",
            "verified_flag": True,
            "effective_date": "2026-01-01",
            "version": 2,
            "required_skills": ["Python", "LangGraph", "RAG"],
            "nice_to_have": ["Arabic NLP"],
        },
        {
            "rubric_id": "r-unverified",
            "role": "AI / LLM Engineer",
            "domain": "ai_engineering",
            "seniority_level": "junior_to_mid",
            "source_type": "community",
            "verified_flag": False,
            "effective_date": "2026-01-01",
            "version": 1,
            "required_skills": ["LangGraph"],
        },
        {
            "rubric_id": "r-future",
            "role": "AI / LLM Engineer",
            "domain": "ai_engineering",
            "seniority_level": "junior_to_mid",
            "source_type": "curated_role_rubric",
            "verified_flag": True,
            "effective_date": "2099-01-01",
            "version": 1,
            "required_skills": ["LangGraph"],
        },
    ]
    path = tmp_path / "rubrics.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_load_rubrics(rubric_file: Path) -> None:
    assert len(load_rubrics(rubric_file)) == 3


def test_only_verified_and_effective_records_are_eligible(rubric_file: Path) -> None:
    settings = Settings(rubric_path=str(rubric_file))
    matches = retrieve_rubrics(query="LangGraph RAG", settings=settings)

    assert [match.rubric_id for match in matches] == ["r-active"]
    assert matches[0].citation == "r-active@v2"


def test_unverified_can_be_included_explicitly(rubric_file: Path) -> None:
    settings = Settings(rubric_path=str(rubric_file))
    matches = retrieve_rubrics(
        query="LangGraph", verified_only=False, settings=settings
    )

    assert "r-unverified" in {match.rubric_id for match in matches}


def test_metadata_filters(rubric_file: Path) -> None:
    settings = Settings(rubric_path=str(rubric_file))
    assert retrieve_rubrics(domain="other", settings=settings) == []
    assert retrieve_rubrics(seniority_level="senior", settings=settings) == []
    assert retrieve_rubrics(role="LLM", settings=settings)[0].rubric_id == "r-active"


def test_ranking_uses_query_overlap(rubric_file: Path) -> None:
    settings = Settings(rubric_path=str(rubric_file))
    high = retrieve_rubrics(query="LangGraph RAG Python", settings=settings)
    low = retrieve_rubrics(query="unrelated topic", settings=settings)

    assert high[0].score > low[0].score
