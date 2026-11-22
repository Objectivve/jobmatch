"""Configuration and lazy credential handling for the job-match agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

DEFAULT_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
)
DEFAULT_JOB_POSTING_MODEL = "gemini-3.5-flash-lite"
DEFAULT_CHECKPOINT_DB = "data/checkpoints.sqlite"
DEFAULT_CACHE_DB = "data/cache.sqlite"
DEFAULT_CACHE_TTL_SECONDS = 86400
DEFAULT_RUBRIC_PATH = "data/rubrics/role_rubric.sample.json"
DEFAULT_RECURSION_LIMIT = 20
DEFAULT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. Secrets are never stored in source control."""

    gemini_api_key: str | None = None
    models: tuple[str, ...] = DEFAULT_MODELS
    job_posting_model: str = DEFAULT_JOB_POSTING_MODEL
    checkpoint_db_path: str = DEFAULT_CHECKPOINT_DB
    cache_db_path: str = DEFAULT_CACHE_DB
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    rubric_path: str = DEFAULT_RUBRIC_PATH
    checkpoint_backend: str = "sqlite"
    database_url: str | None = None
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    model_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    @property
    def has_credentials(self) -> bool:
        return bool(self.gemini_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load .env once, then read typed settings from the environment."""
    load_dotenv(override=False)
    configured = os.getenv("JOBMATCH_MODELS", ",".join(DEFAULT_MODELS))
    models = tuple(model.strip() for model in configured.split(",") if model.strip())
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        models=models or DEFAULT_MODELS,
        job_posting_model=os.getenv(
            "JOBMATCH_JOB_MODEL", DEFAULT_JOB_POSTING_MODEL
        ),
        checkpoint_db_path=os.getenv("JOBMATCH_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB),
        cache_db_path=os.getenv("JOBMATCH_CACHE_DB", DEFAULT_CACHE_DB),
        cache_ttl_seconds=int(
            os.getenv("JOBMATCH_CACHE_TTL", str(DEFAULT_CACHE_TTL_SECONDS))
        ),
        rubric_path=os.getenv("JOBMATCH_RUBRIC_PATH", DEFAULT_RUBRIC_PATH),
        checkpoint_backend=os.getenv("JOBMATCH_CHECKPOINT_BACKEND", "sqlite"),
        database_url=os.getenv("JOBMATCH_DATABASE_URL"),
        recursion_limit=int(
            os.getenv("JOBMATCH_RECURSION_LIMIT", str(DEFAULT_RECURSION_LIMIT))
        ),
        model_timeout_seconds=int(
            os.getenv("JOBMATCH_MODEL_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
        ),
    )


def require_api_key(settings: Settings | None = None) -> str:
    """Fail fast with an actionable message when a live call is attempted."""
    settings = settings or get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "No Gemini API key found. Copy .env.example to .env and set "
            "GEMINI_API_KEY (or GOOGLE_API_KEY) before making live calls."
        )
    return settings.gemini_api_key


def make_genai_client(settings: Settings | None = None):
    """Create the google-genai client at call time, never at import time."""
    from google import genai

    return genai.Client(api_key=require_api_key(settings))
