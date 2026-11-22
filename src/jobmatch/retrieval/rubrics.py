"""Governed role-rubric retrieval with metadata filters and citations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings


@dataclass(frozen=True)
class RubricMatch:
    rubric_id: str
    role: str
    domain: str
    seniority_level: str
    source_type: str
    verified_flag: bool
    effective_date: str
    version: int
    required_skills: tuple[str, ...]
    nice_to_have: tuple[str, ...]
    score: float

    @property
    def citation(self) -> str:
        return f"{self.rubric_id}@v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "required_skills": list(self.required_skills),
            "nice_to_have": list(self.nice_to_have),
            "citation": self.citation,
        }


def load_rubrics(path: str | Path | None = None, settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    source = Path(path or settings.rubric_path)
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Rubric file must contain a JSON list")
    return records


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9+#.]+", text.lower()) if len(token) > 2}


def retrieve_rubrics(
    *,
    query: str = "",
    domain: str | None = None,
    seniority_level: str | None = None,
    role: str | None = None,
    verified_only: bool = True,
    limit: int = 3,
    path: str | Path | None = None,
    settings: Settings | None = None,
) -> list[RubricMatch]:
    """Filter governed rubrics by metadata, then rank by query overlap."""
    records = load_rubrics(path, settings)
    today = date.today().isoformat()

    eligible = [
        record
        for record in records
        if (not verified_only or record.get("verified_flag") is True)
        and (domain is None or record.get("domain") == domain)
        and (seniority_level is None or record.get("seniority_level") == seniority_level)
        and (role is None or role.lower() in str(record.get("role", "")).lower())
        and (str(record.get("effective_date", "9999-12-31")) <= today)
    ]

    query_tokens = _tokens(f"{query} {role or ''}")
    scored: list[RubricMatch] = []
    for record in eligible:
        haystack = " ".join(
            [
                str(record.get("role", "")),
                " ".join(record.get("required_skills", [])),
                " ".join(record.get("nice_to_have", [])),
            ]
        )
        overlap = len(query_tokens & _tokens(haystack))
        score = round(overlap / len(query_tokens), 3) if query_tokens else 0.0
        scored.append(
            RubricMatch(
                rubric_id=str(record.get("rubric_id", "unknown")),
                role=str(record.get("role", "")),
                domain=str(record.get("domain", "")),
                seniority_level=str(record.get("seniority_level", "")),
                source_type=str(record.get("source_type", "")),
                verified_flag=bool(record.get("verified_flag")),
                effective_date=str(record.get("effective_date", "")),
                version=int(record.get("version", 1)),
                required_skills=tuple(record.get("required_skills", [])),
                nice_to_have=tuple(record.get("nice_to_have", [])),
                score=score,
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]
