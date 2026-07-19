"""Document ingestion and text chunking.

Turns raw uploaded files (PDF or plain text) into a list of ``Chunk`` objects
that carry their source filename and page number, so answers can cite exactly
where information came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single retrievable piece of text plus its provenance."""

    id: str
    text: str
    source: str  # original filename
    page: int  # 1-based page number (0 for plain-text files)
    metadata: dict = field(default_factory=dict)


# --- text extraction ---------------------------------------------------------


def extract_pages(data: bytes, filename: str) -> list[tuple[int, str]]:
    """Extract text as a list of ``(page_number, text)`` tuples.

    PDFs are read page by page; plain-text/markdown files are treated as a
    single page 0.
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(data)
    # treat everything else as UTF-8 text (with a lenient fallback)
    text = data.decode("utf-8", errors="replace")
    return [(0, text)]


def _extract_pdf(data: bytes) -> list[tuple[int, str]]:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, text))
    return pages


# --- normalisation & chunking ------------------------------------------------


def clean_text(text: str) -> str:
    """Collapse noisy whitespace produced by PDF extraction."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # cap blank-line runs
    return text.strip()


# Separators tried in order — from coarse (paragraph) to fine (character) so we
# break on the most natural boundary that fits the size limit.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively split ``text`` into pieces no larger than ``chunk_size``."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0]
    rest = separators[1:]

    if sep == "":  # last resort: hard character cut
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else current + sep + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # the single part is still too big -> split it further
            if len(part) > chunk_size:
                chunks.extend(_split_recursive(part, chunk_size, rest))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def _add_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prepend the tail of each chunk to the next for context continuity."""
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    result = [pieces[0]]
    for prev, curr in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        result.append((tail + " " + curr).strip())
    return result


def chunk_document(
    data: bytes,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Full pipeline: bytes -> cleaned, overlapping, page-tagged chunks."""
    chunks: list[Chunk] = []
    for page_no, raw in extract_pages(data, filename):
        text = clean_text(raw)
        if not text:
            continue
        pieces = _split_recursive(text, chunk_size, _SEPARATORS)
        pieces = _add_overlap(pieces, chunk_overlap)
        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            cid = f"{filename}::p{page_no}::c{idx}"
            chunks.append(
                Chunk(
                    id=cid,
                    text=piece,
                    source=filename,
                    page=page_no,
                    metadata={"source": filename, "page": page_no, "text": piece},
                )
            )
    return chunks
