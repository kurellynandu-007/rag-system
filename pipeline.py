"""
pipeline.py
-----------
CLI script: load documents → chunk → print statistics + previews.

This script does NOT embed or store anything in ChromaDB.
It is the "dry run" step that lets you inspect the chunking output
before committing to the full ingestion pipeline.

Usage:
    python pipeline.py                         # scan data/documents/
    python pipeline.py path/to/file.pdf        # single file
    python pipeline.py path/to/dir/            # custom directory
    python pipeline.py --preview 5             # show 5 chunk previews (default 3)
"""

import argparse
import logging
import sys
from pathlib import Path

import config
from chunking import (
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    check_model_limits,
    chunk_document_pages,
    print_chunk_previews,
    print_stats,
)
from loader import load_directory, load_document

logger = logging.getLogger(__name__)


def run(target: Path, n_preview: int = 3) -> None:
    """Load documents from *target* (file or directory), chunk, and report."""

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if target.is_dir():
        print(f"\nScanning directory: {target}")
        pages, stats = load_directory(target)
        print(stats.summary())
    elif target.is_file():
        print(f"\nLoading single file: {target.name}")
        pages, errors = load_document(target)
        for e in errors:
            print(f"  [ERROR] {e}")
        if not pages:
            print("No text could be extracted. Exiting.")
            sys.exit(1)
        print(f"  Loaded {len(pages)} page(s), "
              f"{sum(p.char_count for p in pages):,} chars.")
        stats = None
    else:
        print(f"[ERROR] Path does not exist: {target}")
        sys.exit(1)

    if not pages:
        print("\nNo document pages to process. "
              "Add PDF or Markdown files to the target location.")
        sys.exit(0)

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    print(f"\nChunking with: {CHUNK_TOKENS} tokens/chunk, "
          f"{OVERLAP_TOKENS} tokens overlap …")
    chunks = chunk_document_pages(pages)

    if not chunks:
        print("No chunks produced (all pages were empty?).")
        sys.exit(0)

    # ── 3. Stats ──────────────────────────────────────────────────────────────
    print_stats(chunks, load_stats=stats)

    # ── 4. Preview ────────────────────────────────────────────────────────────
    print_chunk_previews(chunks, n=n_preview)

    print(f"\nReady to embed {len(chunks)} chunks.")
    print("Run  python ingest.py  to embed and store in ChromaDB.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load + chunk documents without embedding (dry run).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py\n"
            "  python pipeline.py data/documents/lecture.pdf\n"
            "  python pipeline.py --preview 5\n"
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(config.DOCUMENTS_DIR),
        help="File or directory to process (default: data/documents/)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        metavar="N",
        help="Number of chunk previews to display (default: 3)",
    )
    parser.add_argument(
        "--limits",
        action="store_true",
        help="Print model token-limit analysis and exit.",
    )
    args = parser.parse_args()

    if args.limits:
        check_model_limits()
        return

    check_model_limits()
    run(Path(args.target), n_preview=args.preview)


if __name__ == "__main__":
    main()
