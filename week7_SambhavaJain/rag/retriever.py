"""Retrieval = vector search followed by Cohere reranking.

Vector similarity is fast but approximate. Reranking the top candidates with a
cross-encoder (Cohere Rerank) reorders them by true relevance to the question,
so only the best few chunks reach the language model. This is the assignment's
"add re-ranking for better relevance" improvement.
"""

from __future__ import annotations


class Retriever:
    def __init__(self, embedder, store, client, rerank_model: str):
        self._embedder = embedder
        self._store = store
        self._client = client
        self._rerank_model = rerank_model

    def retrieve(self, question: str, top_k: int, rerank_top_n: int) -> list[dict]:
        """Return the ``rerank_top_n`` most relevant chunks for ``question``."""
        query_vec = self._embedder.embed_query(question)
        candidates = self._store.query(query_vec, top_k=top_k)
        if not candidates:
            return []
        return self._rerank(question, candidates, rerank_top_n)

    def _rerank(self, question: str, candidates: list[dict], top_n: int) -> list[dict]:
        top_n = min(top_n, len(candidates))
        try:
            res = self._client.rerank(
                model=self._rerank_model,
                query=question,
                documents=[c["text"] for c in candidates],
                top_n=top_n,
            )
        except Exception:
            # if rerank is unavailable, fall back to vector order
            return candidates[:top_n]

        reranked: list[dict] = []
        for r in res.results:
            item = dict(candidates[r.index])
            item["rerank_score"] = r.relevance_score
            reranked.append(item)
        return reranked
