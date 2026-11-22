"""Job-posting extraction through Gemini's URL context tool, cached by URL."""

from __future__ import annotations

from google.genai import types
from langchain_core.tools import tool

from ..cache import get_cache, is_failure, normalize_url, request_key
from ..config import get_settings, make_genai_client
from ..graph.prompts import JOB_POSTING_PROMPT

UNAVAILABLE = "Job posting unavailable or contains no job details."


def fetch_job_posting(job_link: str, *, return_meta: bool = False):
    """Summarise a posting, returning a clear error string on failure.

    With return_meta=True, also report whether the result came from the cache.
    """
    model = get_settings().job_posting_model
    key = request_key(normalize_url(job_link), model, JOB_POSTING_PROMPT)

    def produce() -> str:
        try:
            client = make_genai_client()
            response = client.models.generate_content(
                model=model,
                contents=(
                    f"{JOB_POSTING_PROMPT}\n\nVisit this job posting and extract "
                    f"details:\n{job_link}"
                ),
                config=types.GenerateContentConfig(
                    tools=[types.Tool(url_context=types.UrlContext)],
                ),
            )
            return response.text or UNAVAILABLE
        except Exception as exc:
            return f"Error fetching job posting ({job_link}): {exc}"

    summary, hit = get_cache().get_or_set(key, produce)
    if not hit and is_failure(summary):
        get_cache().delete(key)
    return (summary, hit) if return_meta else summary


@tool(response_format="content_and_artifact")
def job_posting_tool(job_link: str):
    """Extract structured details from a job posting at the provided URL.

    Args:
        job_link: The URL of the job posting.
    """
    summary, hit = fetch_job_posting(job_link, return_meta=True)
    return summary, {"tool": "job_posting_tool", "cache_hit": hit}
