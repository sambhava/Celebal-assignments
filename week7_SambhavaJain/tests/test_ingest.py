"""Tests for text cleaning and chunking (no external services needed)."""

from rag.ingest import Chunk, chunk_document, clean_text, extract_pages


def test_clean_text_collapses_whitespace():
    dirty = "Hello    world\r\n\r\n\r\n\r\nBye\t\tnow"
    assert clean_text(dirty) == "Hello world\n\nBye now"


def test_extract_pages_plain_text():
    pages = extract_pages(b"just some text", "notes.txt")
    assert pages == [(0, "just some text")]


def test_chunking_respects_size_and_overlap():
    text = ("Sentence number %d is here. " % 0) + " ".join(
        f"Sentence number {i} is here." for i in range(1, 200)
    )
    chunks = chunk_document(text.encode(), "doc.txt", chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    # every chunk should be within a reasonable bound of chunk_size (+overlap slack)
    for c in chunks:
        assert len(c.text) <= 200 + 40 + 20


def test_chunk_metadata_and_ids():
    chunks = chunk_document(b"Alpha beta gamma. Delta epsilon.", "a.txt", chunk_size=1000)
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.source == "a.txt"
    assert c.page == 0
    assert c.metadata["text"] == c.text
    assert c.id.startswith("a.txt::p0::c0")


def test_empty_document_yields_no_chunks():
    assert chunk_document(b"   \n\n  ", "empty.txt") == []


def test_overlap_carries_previous_context():
    # two clearly separated paragraphs, small size forces a split
    text = "AAAA BBBB CCCC DDDD.\n\nEEEE FFFF GGGG HHHH."
    chunks = chunk_document(text.encode(), "d.txt", chunk_size=25, chunk_overlap=10)
    assert len(chunks) >= 2
