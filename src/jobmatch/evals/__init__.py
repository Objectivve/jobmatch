"""Offline evaluation toolkit for the golden set."""

from .fake_model import DeterministicModel, OFFLINE_TOOLS
from .harness import (
    CaseResult,
    evaluate_case,
    evaluate_cases,
    load_golden_set,
    render_markdown,
    summarize,
    write_reports,
)

__all__ = [
    "CaseResult",
    "DeterministicModel",
    "OFFLINE_TOOLS",
    "evaluate_case",
    "evaluate_cases",
    "load_golden_set",
    "render_markdown",
    "summarize",
    "write_reports",
]
