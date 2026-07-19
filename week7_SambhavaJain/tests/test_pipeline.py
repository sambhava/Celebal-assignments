"""End-to-end pipeline test using fake Cohere + Pinecone clients.

These fakes mimic the shape of the real SDK responses so we can exercise the
full ingest -> embed -> store -> retrieve -> rerank -> generate flow offline.
"""

from types import SimpleNamespace

from rag.config import Config
from rag.pipeline import RAGPipeline


# --- fake Cohere client ------------------------------------------------------

class FakeCohere:
    def __init__(self):
        self.embed_calls = []
        self.rerank_calls = []
        self.chat_calls = []

    def embed(self, texts, model, input_type, embedding_types):
        self.embed_calls.append(input_type)
        # deterministic 3-dim vectors based on text length
        vecs = [[float(len(t)), float(input_type == "search_query"), 1.0] for t in texts]
        return SimpleNamespace(embeddings=SimpleNamespace(float_=vecs))

    def rerank(self, model, query, documents, top_n):
        # reverse order so we can prove rerank actually reorders results
        results = [
            SimpleNamespace(index=i, relevance_score=1.0 - i * 0.1)
            for i in reversed(range(len(documents)))
        ][:top_n]
        self.rerank_calls.append(len(documents))
        return SimpleNamespace(results=results)

    def chat(self, model, messages, documents):
        self.chat_calls.append(documents)
        content = [SimpleNamespace(text="Grounded answer.")]
        citations = [SimpleNamespace(text="answer", start=0, end=6, sources=[SimpleNamespace(id="doc_0")])]
        return SimpleNamespace(message=SimpleNamespace(content=content, citations=citations))


# --- fake Pinecone client ----------------------------------------------------

class FakeIndex:
    def __init__(self):
        self.vectors = {}

    def upsert(self, vectors):
        for v in vectors:
            self.vectors[v["id"]] = v

    def query(self, vector, top_k, include_metadata):
        matches = [
            SimpleNamespace(id=vid, score=0.9, metadata=v["metadata"])
            for vid, v in list(self.vectors.items())[:top_k]
        ]
        return SimpleNamespace(matches=matches)

    def delete(self, delete_all=False):
        if delete_all:
            self.vectors.clear()


class FakePinecone:
    def __init__(self):
        self._index = FakeIndex()

    def list_indexes(self):
        return SimpleNamespace(names=lambda: ["rag-document-qa"])

    def Index(self, name):
        return self._index


# --- tests -------------------------------------------------------------------

def _pipeline():
    cfg = Config(cohere_api_key="x", pinecone_api_key="y", top_k=5, rerank_top_n=2)
    return RAGPipeline(cfg, cohere_client=FakeCohere(), pinecone_client=FakePinecone())


def test_ingest_indexes_chunks():
    p = _pipeline()
    summary = p.ingest([(b"Hello world. This is a test document.", "doc.txt")])
    assert summary["files"] == 1
    assert summary["chunks"] >= 1


def test_ask_returns_grounded_answer_with_sources():
    p = _pipeline()
    p.ingest([(b"The sky is blue. Grass is green.", "facts.txt")])
    answer = p.ask("What colour is the sky?")

    assert answer.text == "Grounded answer."
    assert len(answer.sources) <= 2  # limited by rerank_top_n
    assert answer.sources  # at least one chunk was used
    assert answer.citations and answer.citations[0]["sources"] == ["doc_0"]


def test_embed_uses_correct_input_types():
    fake = FakeCohere()
    cfg = Config(cohere_api_key="x", pinecone_api_key="y")
    p = RAGPipeline(cfg, cohere_client=fake, pinecone_client=FakePinecone())
    p.ingest([(b"alpha beta gamma", "d.txt")])
    p.ask("question?")
    # documents indexed with search_document, query embedded with search_query
    assert "search_document" in fake.embed_calls
    assert "search_query" in fake.embed_calls


def test_ask_with_no_matches_is_graceful():
    p = _pipeline()  # nothing ingested
    answer = p.ask("anything?")
    assert "could not find" in answer.text.lower()
    assert answer.sources == []


def test_summarize_uses_ingested_chunks():
    p = _pipeline()
    summary_info = p.ingest([(b"The sky is blue. Grass is green.", "facts.txt")])
    assert "facts.txt" in summary_info["sources"]
    summary = p.summarize()
    assert summary == "Grounded answer."  # FakeCohere returns this for any chat


def test_summarize_without_documents_is_graceful():
    p = _pipeline()
    assert "No document content" in p.summarize()
