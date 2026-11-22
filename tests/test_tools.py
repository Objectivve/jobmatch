"""Unit tests for the two agent tools."""

from __future__ import annotations

from types import SimpleNamespace

import docx

from jobmatch.tools.cv_parser import extract_text_from_cv
from jobmatch.tools.cv_parser import extract_cv_text
from jobmatch.tools.job_posting import fetch_job_posting, job_posting_tool

from conftest import RESUME_PDF, FakeGenAIClient


def test_extracts_docx_paragraphs(tmp_path) -> None:
    path = tmp_path / "cv.docx"
    document = docx.Document()
    document.add_paragraph("Mohamed Salaheldin")
    document.add_paragraph("")
    document.add_paragraph("AI Engineer")
    document.save(str(path))

    text = extract_text_from_cv(str(path))

    assert "Mohamed Salaheldin" in text
    assert "AI Engineer" in text
    assert "\n\n" not in text


def test_extracts_real_pdf_fixture() -> None:
    text = extract_text_from_cv(str(RESUME_PDF))
    assert not text.startswith("Error")
    assert "Jane Doe" in text


def test_missing_file_returns_clear_error(tmp_path) -> None:
    assert "File not found" in extract_text_from_cv(str(tmp_path / "missing.pdf"))


def test_unsupported_extension_returns_clear_error(tmp_path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("not a CV format", encoding="utf-8")
    assert "Unsupported file format" in extract_text_from_cv(str(path))


def test_tool_wrappers_keep_expected_names() -> None:
    assert extract_cv_text.name == "extract_cv_text"
    assert job_posting_tool.name == "job_posting_tool"


def test_job_posting_uses_url_context(fake_job_client: FakeGenAIClient) -> None:
    summary = fetch_job_posting("https://example.com/job")

    assert "Python" in summary
    assert fake_job_client.models.calls
    call = fake_job_client.models.calls[0]
    assert call["config"].tools
    assert "https://example.com/job" in str(call["contents"])


def test_job_posting_is_idempotent(fake_job_client, isolated_cache) -> None:
    first, first_hit = fetch_job_posting(
        "https://example.com/job?utm_source=newsletter", return_meta=True
    )
    second, second_hit = fetch_job_posting(
        "https://example.com/job", return_meta=True
    )

    assert first_hit is False
    assert second_hit is True
    assert first == second
    assert len(fake_job_client.models.calls) == 1


def test_job_posting_reports_fetch_errors(monkeypatch) -> None:
    class BrokenClient:
        models = SimpleNamespace(
            generate_content=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("network down")
            )
        )

    monkeypatch.setattr(
        "jobmatch.tools.job_posting.make_genai_client", lambda: BrokenClient()
    )

    assert "Error fetching job posting" in fetch_job_posting("https://example.com/job")
