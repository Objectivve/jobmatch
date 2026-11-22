"""Golden-set evaluation: task success, faithfulness, latency and cost."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import ToolMessage

from ..config import Settings, get_settings
from ..graph.workflow import resume_run, start_run
from ..schemas import ApprovalDecision


@dataclass
class CaseResult:
    case_id: str
    status: str
    task_success: bool
    grounding_passed: bool
    missing_fields: list[str] = field(default_factory=list)
    missing_expectations: list[str] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    citations: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_golden_set(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    cases = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "\n".join(parts)
    return str(content)


def evaluate_case(
    case: dict[str, Any],
    *,
    graph: Any,
    settings: Settings | None = None,
    base_dir: Path | None = None,
    approve: bool = True,
) -> CaseResult:
    """Run one case to completion and score it against its expectations."""
    settings = settings or get_settings()
    expected = case.get("expected", {})
    cv_path = case.get("cv_path")
    if cv_path and base_dir and not Path(cv_path).is_absolute():
        cv_path = str(base_dir / cv_path)

    result = start_run(
        graph,
        thread_id=str(case.get("case_id", "case")),
        message=str(case.get("user_message", "Assess this")),
        cv_path=cv_path,
        job_links=list(case.get("job_links", [])),
        settings=settings,
    )
    paused = bool(result.get("__interrupt__"))
    if paused and approve:
        result = resume_run(
            graph,
            str(case.get("case_id", "case")),
            ApprovalDecision(approved=True),
            settings=settings,
        )

    tool_messages = [
        message for message in result.get("messages", []) if isinstance(message, ToolMessage)
    ]
    tool_names = [message.name for message in tool_messages]
    evidence = "\n".join(_message_text(message) for message in tool_messages)
    final_answer = str(result.get("final_answer") or "")
    evaluation = result.get("evaluation") or {}
    metrics = result.get("metrics") or {}
    blob = f"{final_answer}\n{evidence}\n{evaluation.get('summary', '')}".lower()

    required = set(expected.get("required_fields", [])) | {"final_answer"}
    missing_fields = [name for name in sorted(required) if not result.get(name)]
    missing_expectations = [
        phrase
        for phrase in expected.get("must_contain", [])
        if str(phrase).lower() not in blob
    ]
    forbidden = [
        phrase
        for phrase in expected.get("forbidden_claims", [])
        if str(phrase).lower() in blob
    ]
    tools_ok = all(tool in tool_names for tool in expected.get("tools", []))

    claims = [
        str(skill)
        for skill in evaluation.get("matched_skills", [])
        if str(skill).lower() not in evidence.lower()
    ]

    return CaseResult(
        case_id=str(case.get("case_id", "case")),
        status="completed" if not paused or approve else "awaiting_approval",
        task_success=tools_ok and not missing_fields and not missing_expectations,
        grounding_passed=not claims and not forbidden,
        missing_fields=missing_fields,
        missing_expectations=missing_expectations,
        ungrounded_claims=claims,
        forbidden_claims=forbidden,
        latency_seconds=float(metrics.get("agent_seconds") or 0.0),
        estimated_cost_usd=float(metrics.get("estimated_cost_usd") or 0.0),
        citations=list(evaluation.get("citations") or []),
    )


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    graph_factory: Callable[[Settings], Any],
    settings: Settings | None = None,
    base_dir: Path | None = None,
) -> list[CaseResult]:
    settings = settings or get_settings()
    results: list[CaseResult] = []
    for case in cases:
        try:
            results.append(
                evaluate_case(
                    case,
                    graph=graph_factory(settings),
                    settings=settings,
                    base_dir=base_dir,
                )
            )
        except Exception as exc:
            results.append(
                CaseResult(
                    case_id=str(case.get("case_id", "case")),
                    status="error",
                    task_success=False,
                    grounding_passed=False,
                    error=str(exc),
                )
            )
    return results


def _percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return round(ordered[index], 3)


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    count = len(results) or 1
    latencies = [result.latency_seconds for result in results]
    total_cost = sum(result.estimated_cost_usd for result in results)
    return {
        "cases": len(results),
        "task_success_rate": round(
            sum(result.task_success for result in results) / count, 3
        ),
        "faithfulness_rate": round(
            sum(result.grounding_passed for result in results) / count, 3
        ),
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "cost_total_usd": round(total_cost, 6),
        "cost_per_run_usd": round(total_cost / count, 6),
        "failures": [
            result.to_dict()
            for result in results
            if not (result.task_success and result.grounding_passed)
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {key} | {value} |"
        for key, value in summary.items()
        if key != "failures"
    )
    failures = summary.get("failures", [])
    failure_lines = "\n".join(
        f"- {item['case_id']}: {item.get('error') or item.get('missing_fields') or item.get('ungrounded_claims')}"
        for item in failures
    ) or "- None"
    return (
        "# Evaluation summary\n\n"
        "| Metric | Value |\n| --- | --- |\n"
        f"{rows}\n\n## Failures\n\n{failure_lines}\n"
    )


def write_reports(
    summary: dict[str, Any], json_path: str | Path, markdown_path: str | Path
) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_target.write_text(render_markdown(summary), encoding="utf-8")
