"""Cohere embedding wrapper.

Cohere's v3 embedding models require an ``input_type`` that differs between
indexing documents and embedding a search query — using the right one on each
side measurably improves retrieval quality.
"""

from __future__ import annotations


class Embedder:
    """Thin wrapper around Cohere's embed endpoint with batching."""

    # Cohere accepts up to 96 texts per embed call.
    BATCH_SIZE = 96

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start : start + self.BATCH_SIZE]
            resp = self._client.embed(
                texts=batch,
                model=self._model,
                input_type=input_type,
                embedding_types=["float"],
            )
            # The Cohere SDK names the field ``float_`` (``float`` is reserved);
            # accept either so we work across SDK versions and test fakes.
            emb = resp.embeddings
            data = getattr(emb, "float_", None)
            if data is None:
                data = getattr(emb, "float", None)
            if data is None:
                raise RuntimeError("Cohere embed response contained no float embeddings")
            vectors.extend(data)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for storage in the vector index."""
        if not texts:
            return []
        return self._embed(texts, "search_document")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question for retrieval."""
        return self._embed([text], "search_query")[0]
