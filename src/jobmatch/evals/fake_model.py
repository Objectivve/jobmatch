"""Deterministic model and tools so evals can run offline, with grounded text."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from ..schemas import CVEvidence, JobRequirement, MatchEvaluation
from ..tools.cv_parser import extract_text_with_meta


@tool("extract_cv_text", response_format="content_and_artifact")
def fake_extract_cv_text(file_path: str):
    """Offline stand-in for CV extraction."""
    text, hit = extract_text_with_meta(file_path)
    return text, {"tool": "extract_cv_text", "cache_hit": hit}


@tool("job_posting_tool", response_format="content_and_artifact")
def fake_job_posting_tool(job_link: str):
    """Offline stand-in for job-posting extraction."""
    if "invalid" in job_link or "unreachable" in job_link:
        return (
            "Job posting unavailable or contains no job details.",
            {"tool": "job_posting_tool", "cache_hit": False},
        )
    return (
        "Job Title: Sample AI Engineer\n"
        "Required qualifications: Python, LangGraph, RAG\n"
        "Responsibilities: Build and evaluate agents.\n",
        {"tool": "job_posting_tool", "cache_hit": False},
    )


OFFLINE_TOOLS = [fake_job_posting_tool, fake_extract_cv_text]
KNOWN_SKILLS = ("Python", "LangGraph", "RAG", "TensorFlow", "SQL")


class DeterministicModel:
    """Calls tools once, then produces text and schemas grounded in their output."""

    def __init__(self) -> None:
        self.called = False

    def bind_tools(self, tools):
        return self

    @staticmethod
    def _evidence(messages: list) -> str:
        return "\n".join(
            str(getattr(message, "content", ""))
            for message in messages
            if isinstance(message, ToolMessage)
        )

    @staticmethod
    def _user_text(messages: list) -> str:
        return "\n".join(
            str(getattr(message, "content", ""))
            for message in messages
            if isinstance(message, HumanMessage)
        )

    def invoke(self, messages: list) -> AIMessage:
        if not self.called:
            self.called = True
            user = self._user_text(messages)
            calls: list[dict] = []
            cv_match = re.search(r"CV file:\s*(.+)", user)
            if cv_match:
                calls.append(
                    {
                        "name": "extract_cv_text",
                        "args": {"file_path": cv_match.group(1).strip()},
                        "id": "eval_cv",
                        "type": "tool_call",
                    }
                )
            for index, link in enumerate(re.findall(r"Job posting:\s*(.+)", user)):
                calls.append(
                    {
                        "name": "job_posting_tool",
                        "args": {"job_link": link.strip()},
                        "id": f"eval_job_{index}",
                        "type": "tool_call",
                    }
                )
            return AIMessage(content="", tool_calls=calls)

        evidence = self._evidence(messages)
        return AIMessage(content=f"Based on the supplied evidence: {evidence[:400]}")

    def with_structured_output(self, schema: type, **kwargs):
        outer = self

        class Structured:
            def invoke(self, messages: list):
                evidence = outer._evidence(messages)
                all_text = " ".join(
                    str(getattr(message, "content", ""))
                    for message in messages
                    if isinstance(message, HumanMessage)
                )
                citations = sorted(
                    set(re.findall(r"rubric-[a-z0-9-]+@v\d+", all_text))
                )[:3]
                skills = [
                    skill
                    for skill in KNOWN_SKILLS
                    if skill.lower() in evidence.lower()
                ]
                weak = any(
                    phrase in evidence.lower()
                    for phrase in ("not found", "unavailable", "no job details")
                )
                return schema(
                    summary=f"Evidence summary: {evidence[:220]}",
                    overall_fit="weak" if weak else "moderate",
                    matched_skills=skills,
                    gaps=[] if skills else ["No grounded skills identified"],
                    cv_evidence=[
                        CVEvidence(
                            skill=skill,
                            evidence=evidence[:220],
                            source=citations[0] if citations else "cv",
                        )
                        for skill in skills[:3]
                    ],
                    job_requirements=[
                        JobRequirement(requirement="Python", source="job")
                    ]
                    if "python" in evidence.lower()
                    else [],
                    citations=citations,
                    confidence=0.8,
                )

        return Structured()
