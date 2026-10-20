"""
loader.py
---------
Loads PDF and Markdown documents from disk into a normalised list of
DocumentPage objects that chunking.py can consume.

Supported formats
-----------------
  .pdf   : text extracted page-by-page using pypdf
  .md    : entire file treated as a single "page"
  .txt   : entire file treated as a single "page"

Design notes
------------
- Each page is returned as an independent DocumentPage so that page-level
  metadata (source filename, page number) survives into the final chunks.
- Empty pages and pages with only whitespace are dropped and counted.
- Errors per file are collected and returned rather than raising, so a
  single bad file does not abort the whole ingestion batch.
- No embedding or ChromaDB interaction happens here.

Usage:
    from loader import load_directory, load_document
    pages, stats = load_directory(config.DOCUMENTS_DIR)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import config
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

# File extensions this loader handles
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class DocumentPage:
    """
    A single page (or section) of text from a source document.

    Attributes
    ----------
    text      : Extracted text content (stripped, non-empty guaranteed).
    source    : Original filename only (not full path).
    page_num  : 1-based page number for PDFs; None for Markdown / plain-text.
    char_count: len(text) — stored so callers don't have to recompute.
    file_type : "pdf", "markdown", or "text".
    """

    text: str
    source: str
    page_num: Optional[int]
    char_count: int
    file_type: str

    def __repr__(self) -> str:
        page = f"p{self.page_num}" if self.page_num else "whole"
        return (
            f"DocumentPage(source={self.source!r}, {page}, "
            f"chars={self.char_count}, type={self.file_type})"
        )


@dataclass
class LoadStats:
    """
    Aggregate statistics from loading a batch of documents.

    Attributes
    ----------
    files_found     : Total files discovered (matching supported extensions).
    files_loaded    : Files from which at least one page of text was extracted.
    files_empty     : Files that parsed correctly but had zero extractable text.
    files_failed    : Files that raised an error during loading.
    pages_total     : Total pages / sections seen (including blank ones).
    pages_with_text : Pages from which text was successfully extracted.
    pages_empty     : Pages skipped because they contained no text.
    total_chars     : Sum of char_count across all DocumentPage objects returned.
    errors          : Human-readable error messages for failed files.
    """

    files_found: int = 0
    files_loaded: int = 0
    files_empty: int = 0
    files_failed: int = 0
    pages_total: int = 0
    pages_with_text: int = 0
    pages_empty: int = 0
    total_chars: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Load Statistics",
            f"  Files found      : {self.files_found}",
            f"  Files loaded     : {self.files_loaded}",
            f"  Files empty      : {self.files_empty}",
            f"  Files failed     : {self.files_failed}",
            f"  Pages seen       : {self.pages_total}",
            f"  Pages with text  : {self.pages_with_text}",
            f"  Pages skipped    : {self.pages_empty}",
            f"  Total characters : {self.total_chars:,}",
        ]
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    - {e}")
        return "\n".join(lines)


# ── PDF loader ────────────────────────────────────────────────────────────────

def load_pdf(path: Path) -> Tuple[List[DocumentPage], List[str]]:
    """
    Extract text from a PDF file, one DocumentPage per PDF page.

    Parameters
    ----------
    path : Absolute Path to the .pdf file.

    Returns
    -------
    Tuple of:
      - List[DocumentPage]  — one entry per page that had extractable text.
      - List[str]           — error/warning messages (empty if all went fine).

    Does NOT raise; problems are returned as error strings.
    """
    pages: List[DocumentPage] = []
    errors: List[str] = []
    source = path.name

    if not path.exists():
        return [], [f"File not found: {path}"]
    if path.suffix.lower() != ".pdf":
        return [], [f"Expected .pdf extension, got: {path.suffix!r} ({source})"]

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        return [], [f"Cannot read PDF '{source}': {exc}"]
    except Exception as exc:
        return [], [f"Unexpected error opening '{source}': {exc}"]

    total_pages = len(reader.pages)
    empty_pages = 0

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
            text = raw.strip()
        except Exception as exc:
            errors.append(
                f"'{source}' page {page_num}: extraction error — {exc}"
            )
            text = ""

        if not text:
            empty_pages += 1
            logger.debug(
                "'%s' page %d: no extractable text (scanned image or blank).",
                source, page_num,
            )
            continue

        pages.append(
            DocumentPage(
                text=text,
                source=source,
                page_num=page_num,
                char_count=len(text),
                file_type="pdf",
            )
        )

    if empty_pages == total_pages:
        errors.append(
            f"'{source}': all {total_pages} page(s) had no extractable text. "
            "The PDF may be scanned or image-only."
        )
    elif empty_pages > 0:
        logger.info(
            "'%s': %d/%d pages skipped (no text).",
            source, empty_pages, total_pages,
        )

    logger.info(
        "PDF '%s': %d pages read, %d with text.",
        source, total_pages, len(pages),
    )
    return pages, errors


# ── Markdown / plain-text loader ──────────────────────────────────────────────

def load_text_file(path: Path) -> Tuple[List[DocumentPage], List[str]]:
    """
    Load a Markdown (.md) or plain-text (.txt) file as a single DocumentPage.

    Parameters
    ----------
    path : Absolute Path to the file.

    Returns
    -------
    Tuple of:
      - List[DocumentPage]  — zero or one element.
      - List[str]           — error messages if any.
    """
    source = path.name
    ext = path.suffix.lower()
    file_type = "markdown" if ext == ".md" else "text"

    if not path.exists():
        return [], [f"File not found: {path}"]

    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        return [], [f"Cannot read '{source}': {exc}"]

    if not text:
        return [], [f"'{source}': file is empty."]

    page = DocumentPage(
        text=text,
        source=source,
        page_num=None,
        char_count=len(text),
        file_type=file_type,
    )
    logger.info("Text file '%s': %d characters loaded.", source, len(text))
    return [page], []


# ── Dispatch ──────────────────────────────────────────────────────────────────

def load_document(path: Path) -> Tuple[List[DocumentPage], List[str]]:
    """
    Dispatch to the correct loader based on file extension.

    Parameters
    ----------
    path : Path to a PDF, Markdown, or plain-text file.

    Returns
    -------
    Tuple of (pages, errors) — same contract as load_pdf / load_text_file.

    Raises
    ------
    ValueError for extensions we don't support.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in {".md", ".txt"}:
        return load_text_file(path)
    else:
        return [], [
            f"Unsupported file type '{ext}' for '{path.name}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        ]


# ── Directory loader ──────────────────────────────────────────────────────────

def load_directory(
    directory: Path = config.DOCUMENTS_DIR,
) -> Tuple[List[DocumentPage], LoadStats]:
    """
    Load all supported documents found in *directory* (non-recursive).

    Duplicate filenames (same name, same extension) are detected and the
    second occurrence is skipped with a warning.

    Parameters
    ----------
    directory : Folder to scan. Defaults to config.DOCUMENTS_DIR.

    Returns
    -------
    Tuple of:
      - List[DocumentPage]  — all pages from all successfully loaded files.
      - LoadStats           — aggregate counts and any error messages.
    """
    if not directory.exists():
        return [], LoadStats(
            errors=[f"Directory not found: {directory}"]
        )

    # Collect supported files, detect duplicates by (stem, suffix)
    all_files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    stats = LoadStats(files_found=len(all_files))
    all_pages: List[DocumentPage] = []
    seen_names: set = set()

    if not all_files:
        logger.warning(
            "No supported files found in %s. "
            "Add .pdf, .md, or .txt files and re-run.",
            directory,
        )
        return [], stats

    for file_path in all_files:
        # Duplicate check (case-insensitive on Windows)
        name_key = file_path.name.lower()
        if name_key in seen_names:
            msg = (
                f"Duplicate filename skipped: '{file_path.name}'. "
                "Two files with the same name would produce identical chunk IDs."
            )
            logger.warning(msg)
            stats.errors.append(msg)
            stats.files_failed += 1
            continue
        seen_names.add(name_key)

        pages, errors = load_document(file_path)
        stats.errors.extend(errors)

        # Count PDF pages (we need the raw reader for total page count)
        if file_path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader as _R
                total_pdf_pages = len(_R(str(file_path)).pages)
            except Exception:
                total_pdf_pages = len(pages)
            stats.pages_total += total_pdf_pages
            stats.pages_with_text += len(pages)
            stats.pages_empty += total_pdf_pages - len(pages)
        else:
            # Markdown / text: counts as one "page"
            stats.pages_total += 1
            if pages:
                stats.pages_with_text += 1
            else:
                stats.pages_empty += 1

        if pages:
            stats.files_loaded += 1
            stats.total_chars += sum(p.char_count for p in pages)
            all_pages.extend(pages)
        elif not errors:
            # Parsed fine but yielded nothing (empty doc)
            stats.files_empty += 1
        else:
            stats.files_failed += 1

    logger.info(
        "Directory load complete. %d files -> %d pages, %d chars.",
        stats.files_found,
        stats.pages_with_text,
        stats.total_chars,
    )
    return all_pages, stats


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    pages, stats = load_directory()
    print()
    print(stats.summary())
    print(f"\nLoaded {len(pages)} page(s) total.")
    if pages:
        print("\nFirst page preview:")
        print(f"  {pages[0]}")
        print(f"  Text[:200]: {pages[0].text[:200]!r}")
