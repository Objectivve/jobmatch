"""Lightweight per-run telemetry: token counts, estimated cost and latency."""

from __future__ import annotations

import time
from typing import Any

# Rough USD per 1M tokens. Replace with your provider's current pricing.
COST_PER_MILLION: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3.8-flash": (0.30, 2.50),
}
DEFAULT_RATE = (0.30, 2.50)


def empty_metrics() -> dict[str, Any]:
    """Fresh metrics dict; safe to store in a LangGraph checkpoint."""
    return {
        "model_calls": 0,
        "models_tried": [],
        "tokens_input": 0,
        "tokens_output": 0,
        "estimated_cost_usd": 0.0,
        "tool_calls": 0,
        "cache_hits": 0,
        "agent_seconds": 0.0,
        "wall_seconds": None,
        "started_at": time.time(),
    }


def record_attempt(metrics: dict[str, Any] | None, model_name: str) -> None:
    if metrics is None:
        return
    metrics["model_calls"] += 1
    metrics.setdefault("models_tried", []).append(model_name)


def _token_pair(usage: Any) -> tuple[int, int]:
    if not usage:
        return 0, 0
    if isinstance(usage, dict):
        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_token_count")
            or usage.get("prompt_tokens")
            or 0
        )
        output_tokens = (
            usage.get("output_tokens")
            or usage.get("candidates_token_count")
            or usage.get("completion_tokens")
            or 0
        )
    else:
        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_token_count", None)
            or getattr(usage, "prompt_tokens", 0)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "candidates_token_count", None)
            or getattr(usage, "completion_tokens", 0)
            or 0
        )
    return int(input_tokens), int(output_tokens)


def add_usage(
    metrics: dict[str, Any] | None, response: Any, model_name: str = "unknown"
) -> None:
    """Add token counts and estimated cost from a model response."""
    if metrics is None:
        return
    usage = getattr(response, "usage_metadata", None) or getattr(
        response, "usage", None
    )
    input_tokens, output_tokens = _token_pair(usage)
    metrics["tokens_input"] += input_tokens
    metrics["tokens_output"] += output_tokens
    rate_in, rate_out = COST_PER_MILLION.get(model_name, DEFAULT_RATE)
    metrics["estimated_cost_usd"] = round(
        metrics["estimated_cost_usd"]
        + (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000,
        6,
    )


def add_duration(metrics: dict[str, Any] | None, started: float) -> None:
    """Accumulate time spent inside agent nodes (excludes human pause time)."""
    if metrics is not None:
        metrics["agent_seconds"] = round(
            metrics["agent_seconds"] + (time.perf_counter() - started), 3
        )


def finish_metrics(metrics: dict[str, Any] | None) -> None:
    """Record wall-clock time including any HITL pause."""
    if metrics is not None:
        metrics["wall_seconds"] = round(time.time() - metrics["started_at"], 3)
