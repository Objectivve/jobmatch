"""Persistent TTL cache that makes external/extraction calls idempotent."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Callable, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import get_settings

T = TypeVar("T")
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "promotion",
    "ref",
}
def normalize_url(url: str) -> str:
    """Lowercase host/scheme, drop tracking params, sort the rest."""
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(sorted(query)), "")
    )


def request_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def is_failure(value: str) -> bool:
    return (
        value.startswith("Error")
        or value.startswith("File not found")
        or value.startswith("Unsupported file format")
        or value == "Job posting unavailable or contains no job details."
    )


class IdempotencyCache:
    def __init__(self, db_path: str | Path, ttl_seconds: int | None = None) -> None:
        path = Path(db_path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        settings_ttl = (
            get_settings().cache_ttl_seconds if ttl_seconds is None else ttl_seconds
        )
        self.ttl_seconds = settings_ttl
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS cache_entries ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.connection.commit()

    def get(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value, created_at FROM cache_entries WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value, created_at = row
        if time.time() - created_at > self.ttl_seconds:
            self.delete(key)
            return None
        return str(value)

    def set(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO cache_entries (key, value, created_at) "
            "VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        self.connection.commit()

    def delete(self, key: str) -> None:
        self.connection.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        self.connection.commit()

    def get_or_set(self, key: str, producer: Callable[[], T]) -> tuple[T, bool]:
        cached = self.get(key)
        if cached is not None:
            return cached, True  # type: ignore[return-value]
        value = producer()
        self.set(key, str(value))
        return value, False


_cache: IdempotencyCache | None = None


def get_cache() -> IdempotencyCache:
    """One cache instance per process, configured from settings/.env."""
    global _cache
    if _cache is None:
        settings = get_settings()
        _cache = IdempotencyCache(settings.cache_db_path, settings.cache_ttl_seconds)
    return _cache
