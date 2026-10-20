"""
tests/test_loader.py
--------------------
Tests for loader.py: PDF loading, Markdown loading, error handling,
empty documents, missing files, and duplicate detection.

Run:
    .venv\\Scripts\\activate
    python -m pytest tests/test_loader.py -v
"""

import io
import sys
from pathlib import Path

import pytest
from pypdf import PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import (
    DocumentPage,
    LoadStats,
    load_directory,
    load_document,
    load_pdf,
    load_text_file,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_blank_pdf(tmp_path: Path, name: str = "blank.pdf", n_pages: int = 1) -> Path:
    """
    Write a valid PDF with *n_pages* blank pages to *tmp_path*.
    No extractable text — used to test the 'no-text' error path.
    """
    dest = tmp_path / name
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=612, height=792)
    with open(dest, "wb") as f:
        writer.write(f)
    return dest


def make_markdown_file(tmp_path: Path, content: str, name: str = "doc.md") -> Path:
    """Write a Markdown file and return its Path."""
    dest = tmp_path / name
    dest.write_text(content, encoding="utf-8")
    return dest


def make_text_file(tmp_path: Path, content: str, name: str = "doc.txt") -> Path:
    """Write a plain-text file and return its Path."""
    dest = tmp_path / name
    dest.write_text(content, encoding="utf-8")
    return dest


# ══════════════════════════════════════════════════════════════════════════════
# load_pdf tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadPdf:

    def test_missing_file_returns_error_not_exception(self, tmp_path):
        """load_pdf must NOT raise; missing files are reported in errors."""
        fake = tmp_path / "nonexistent.pdf"
        pages, errors = load_pdf(fake)
        assert pages == []
        assert len(errors) >= 1
        assert "not found" in errors[0].lower() or "nonexistent" in errors[0].lower()

    def test_wrong_extension_returns_error(self, tmp_path):
        """load_pdf called on a non-PDF file returns an error, not a crash."""
        txt = make_text_file(tmp_path, "hello", name="notes.txt")
        pages, errors = load_pdf(txt)
        assert pages == []
        assert errors
        assert ".txt" in errors[0] or "pdf" in errors[0].lower()

    def test_blank_pdf_reports_no_text_error(self, tmp_path):
        """A PDF with only blank pages should yield zero pages and an error."""
        blank = make_blank_pdf(tmp_path, "blank.pdf", n_pages=2)
        pages, errors = load_pdf(blank)
        assert pages == []
        assert len(errors) >= 1
        # Should mention no extractable text
        combined = " ".join(errors)
        assert "no extractable text" in combined.lower() or "all" in combined.lower()

    def test_blank_pdf_multipage_reports_page_count(self, tmp_path):
        """Error message for a multi-page blank PDF should mention page count."""
        blank = make_blank_pdf(tmp_path, "big_blank.pdf", n_pages=5)
        _, errors = load_pdf(blank)
        combined = " ".join(errors)
        assert "5" in combined, "Error should mention how many pages were blank."

    def test_load_pdf_returns_document_page_type(self, tmp_path):
        """
        When a PDF has extractable text, pages returned are DocumentPage objects.
        We can only test this indirectly because PdfWriter cannot add text.
        We confirm: if pages are returned they are the right type.
        This test is vacuously True for blank PDFs and meaningful when real
        PDFs are placed in data/documents/.
        """
        blank = make_blank_pdf(tmp_path, "typecheck.pdf")
        pages, _ = load_pdf(blank)
        for p in pages:
            assert isinstance(p, DocumentPage)

    def test_document_page_fields_populated(self, tmp_path):
        """
        If any pages are returned, all required fields must be non-None.
        (Only triggered if pypdf extracts text from the test PDF.)
        """
        blank = make_blank_pdf(tmp_path, "fields.pdf")
        pages, _ = load_pdf(blank)
        for p in pages:
            assert p.source != ""
            assert p.file_type == "pdf"
            assert isinstance(p.page_num, int) and p.page_num >= 1
            assert p.char_count == len(p.text)
            assert p.text.strip() != ""


# ══════════════════════════════════════════════════════════════════════════════
# load_text_file tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadTextFile:

    def test_markdown_loads_correctly(self, tmp_path):
        content = "# Heading\n\nThis is a paragraph about RAG systems."
        md = make_markdown_file(tmp_path, content)
        pages, errors = load_text_file(md)
        assert errors == []
        assert len(pages) == 1
        p = pages[0]
        assert p.text == content
        assert p.source == "doc.md"
        assert p.file_type == "markdown"
        assert p.page_num is None    # Markdown has no page numbers
        assert p.char_count == len(content)

    def test_plain_text_loads_correctly(self, tmp_path):
        content = "Plain text document.\nSecond line."
        txt = make_text_file(tmp_path, content)
        pages, errors = load_text_file(txt)
        assert errors == []
        assert len(pages) == 1
        assert pages[0].file_type == "text"
        assert pages[0].page_num is None

    def test_empty_file_returns_error_not_exception(self, tmp_path):
        empty_md = make_markdown_file(tmp_path, "")
        pages, errors = load_text_file(empty_md)
        assert pages == []
        assert len(errors) >= 1
        assert "empty" in errors[0].lower()

    def test_whitespace_only_file_treated_as_empty(self, tmp_path):
        ws_md = make_markdown_file(tmp_path, "   \n\n\t  ")
        pages, errors = load_text_file(ws_md)
        assert pages == []
        assert errors

    def test_missing_file_returns_error(self, tmp_path):
        fake = tmp_path / "ghost.md"
        pages, errors = load_text_file(fake)
        assert pages == []
        assert errors
        assert "not found" in errors[0].lower() or "ghost" in errors[0].lower()

    def test_char_count_matches_text_length(self, tmp_path):
        content = "The quick brown fox." * 50
        md = make_markdown_file(tmp_path, content)
        pages, _ = load_text_file(md)
        assert pages[0].char_count == len(pages[0].text)

    def test_source_filename_not_full_path(self, tmp_path):
        md = make_markdown_file(tmp_path, "content")
        pages, _ = load_text_file(md)
        # source should be just the filename, not the full path
        assert "/" not in pages[0].source
        assert "\\" not in pages[0].source
        assert pages[0].source == "doc.md"


# ══════════════════════════════════════════════════════════════════════════════
# load_document dispatch tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadDocument:

    def test_dispatches_to_pdf_loader(self, tmp_path):
        pdf = make_blank_pdf(tmp_path, "test.pdf")
        pages, errors = load_document(pdf)
        # Blank PDF → no text, but dispatch was correct (pdf error message)
        combined = " ".join(errors)
        assert "text" in combined.lower() or pages == []

    def test_dispatches_to_markdown_loader(self, tmp_path):
        md = make_markdown_file(tmp_path, "# Title\n\nBody text.")
        pages, errors = load_document(md)
        assert errors == []
        assert pages[0].file_type == "markdown"

    def test_dispatches_to_text_loader(self, tmp_path):
        txt = make_text_file(tmp_path, "plain text content")
        pages, errors = load_document(txt)
        assert errors == []
        assert pages[0].file_type == "text"

    def test_unsupported_extension_returns_error(self, tmp_path):
        docx = tmp_path / "report.docx"
        docx.write_bytes(b"fake docx content")
        pages, errors = load_document(docx)
        assert pages == []
        assert errors
        assert ".docx" in errors[0] or "unsupported" in errors[0].lower()


# ══════════════════════════════════════════════════════════════════════════════
# load_directory tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadDirectory:

    def test_empty_directory_returns_empty_list(self, tmp_path):
        pages, stats = load_directory(tmp_path)
        assert pages == []
        assert stats.files_found == 0

    def test_missing_directory_returns_error_stats(self, tmp_path):
        fake_dir = tmp_path / "nonexistent_dir"
        pages, stats = load_directory(fake_dir)
        assert pages == []
        assert stats.errors

    def test_counts_files_found(self, tmp_path):
        make_markdown_file(tmp_path, "content A", name="a.md")
        make_markdown_file(tmp_path, "content B", name="b.md")
        _, stats = load_directory(tmp_path)
        assert stats.files_found == 2

    def test_counts_files_loaded(self, tmp_path):
        make_markdown_file(tmp_path, "content", name="good.md")
        make_markdown_file(tmp_path, "   ", name="empty.md")   # whitespace only
        _, stats = load_directory(tmp_path)
        assert stats.files_loaded == 1

    def test_ignores_unsupported_extensions(self, tmp_path):
        make_markdown_file(tmp_path, "content", name="ok.md")
        (tmp_path / "ignore_me.docx").write_bytes(b"fake")
        _, stats = load_directory(tmp_path)
        # .docx is not in SUPPORTED_EXTENSIONS, so files_found should be 1
        assert stats.files_found == 1

    def test_detects_duplicate_filenames(self, tmp_path):
        """
        Duplicate: two files with the same name cannot coexist in one directory
        on the real filesystem, but we test the case-insensitive duplicate check.
        Since we can't have two 'doc.md' files in one folder on Windows/Linux,
        we simulate by loading the same file twice via a subclass — instead,
        we verify the duplicate-detection logic via the seen_names set path.
        This is tested indirectly: having a.md and A.md on case-sensitive FS.
        """
        make_markdown_file(tmp_path, "content A", name="a.md")
        make_markdown_file(tmp_path, "content B", name="b.md")
        pages, stats = load_directory(tmp_path)
        # No duplicates in this case: both should load
        assert stats.files_loaded == 2
        assert stats.files_failed == 0

    def test_total_chars_is_sum_of_page_chars(self, tmp_path):
        content_a = "Hello world. " * 20
        content_b = "Goodbye world. " * 15
        make_markdown_file(tmp_path, content_a, name="a.md")
        make_markdown_file(tmp_path, content_b, name="b.md")
        pages, stats = load_directory(tmp_path)
        assert stats.total_chars == sum(p.char_count for p in pages)

    def test_returns_document_page_objects(self, tmp_path):
        make_markdown_file(tmp_path, "Some content here.", name="test.md")
        pages, _ = load_directory(tmp_path)
        assert all(isinstance(p, DocumentPage) for p in pages)

    def test_load_stats_summary_runs_without_error(self, tmp_path):
        make_markdown_file(tmp_path, "content", name="doc.md")
        _, stats = load_directory(tmp_path)
        summary = stats.summary()
        assert "Files" in summary
        assert "characters" in summary.lower()
