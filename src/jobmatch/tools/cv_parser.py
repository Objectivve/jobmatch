"""Local CV text extraction (PDF, DOCX, DOC) with idempotent caching."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import docx
import pdfplumber
from langchain_core.tools import tool

from ..cache import get_cache, is_failure, request_key


def _extract_uncached(path: Path, ext: str) -> str:
    if ext == ".docx":
        try:
            document = docx.Document(str(path))
            return "\n".join(
                para.text for para in document.paragraphs if para.text.strip()
            )
        except Exception as exc:
            return f"Error reading the .docx file: {exc}"

    if ext == ".pdf":
        try:
            pages: list[str] = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
            return "\n".join(pages)
        except Exception as exc:
            return f"Error reading the .pdf file: {exc}"

    if ext == ".doc":
        converted = path.with_suffix(".docx")
        try:
            subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    str(path.resolve().parent),
                    str(path.resolve()),
                ],
                check=True,
            )
            document = docx.Document(str(converted))
            return "\n".join(
                para.text for para in document.paragraphs if para.text.strip()
            )
        except Exception as exc:
            return f"Error reading .doc file: {exc}"
        finally:
            if os.path.exists(converted):
                os.remove(converted)

    return "Unsupported file format. Please use .pdf, .docx, or .doc."


def extract_text_with_meta(file_path: str) -> tuple[str, bool]:
    """Return (text, cache_hit) so graph telemetry can record reuse."""
    path = Path(file_path).expanduser()
    if not path.is_file():
        return f"File not found: {file_path}", False

    stat = path.stat()
    key = request_key(
        "cv", str(path.resolve()), str(stat.st_mtime_ns), str(stat.st_size)
    )
    text, hit = get_cache().get_or_set(
        key, lambda: _extract_uncached(path, path.suffix.lower())
    )
    if not hit and is_failure(text):
        get_cache().delete(key)
    return text, hit


def extract_text_from_cv(file_path: str) -> str:
    """Return plain text from a CV, or a clear error string on failure."""
    return extract_text_with_meta(file_path)[0]


@tool(response_format="content_and_artifact")
def extract_cv_text(file_path: str):
    """Extract text from a local CV file in PDF, DOCX, or DOC format.

    Args:
        file_path: Local path (or filename) of the CV document.
    """
    text, hit = extract_text_with_meta(file_path)
    return text, {"tool": "extract_cv_text", "cache_hit": hit}
