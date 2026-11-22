"""Idempotency cache and URL-normalisation tests."""

from __future__ import annotations

from jobmatch.cache import IdempotencyCache, is_failure, normalize_url, request_key


def test_normalize_url_drops_tracking_and_sorts_query() -> None:
    url = "https://Example.com/jobs/123/?utm_source=x&b=2&a=1#section"
    assert normalize_url(url) == "https://example.com/jobs/123?a=1&b=2"


def test_request_key_is_stable() -> None:
    assert request_key("a", "b") == request_key("a", "b")
    assert request_key("a", "b") != request_key("b", "a")


def test_get_or_set_runs_producer_once(tmp_path) -> None:
    cache = IdempotencyCache(str(tmp_path / "cache.sqlite"), ttl_seconds=60)
    calls: list[int] = []

    def producer() -> str:
        calls.append(1)
        return "value"

    assert cache.get_or_set("key", producer) == ("value", False)
    assert cache.get_or_set("key", producer) == ("value", True)
    assert len(calls) == 1


def test_expired_entry_is_dropped(tmp_path) -> None:
    cache = IdempotencyCache(str(tmp_path / "cache.sqlite"), ttl_seconds=-1)
    cache.set("key", "value")
    assert cache.get("key") is None


def test_failure_detection() -> None:
    assert is_failure("Error fetching job posting (x): down")
    assert is_failure("File not found: x.pdf")
    assert not is_failure("Job Title: Engineer")
