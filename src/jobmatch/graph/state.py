"""State channels used by the LangGraph workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Conversation messages plus the fields the API needs after approval."""

    draft_assessment: str
    evaluation: dict[str, Any] | None
    rubrics: list[dict[str, Any]]
    approval: dict[str, Any] | None
    metrics: dict[str, Any]
    final_answer: str
