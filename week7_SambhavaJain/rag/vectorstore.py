"""Pinecone serverless vector store.

Handles index creation (idempotent), upserting chunk embeddings with their
metadata, and similarity search. Text is stored in each vector's metadata so a
query returns the chunk content directly, without a second lookup.
"""

from __future__ import annotations

import time

from .ingest import Chunk


class VectorStore:
    def __init__(self, client, index_name: str, dimension: int, cloud: str, region: str):
        self._pc = client
        self._name = index_name
        self._dimension = dimension
        self._cloud = cloud
        self._region = region
        self._index = None

    # --- index lifecycle -----------------------------------------------------

    def ensure_index(self) -> None:
        """Create the index if it does not already exist, then connect."""
        if self._name not in self._existing_index_names():
            from pinecone import ServerlessSpec

            self._pc.create_index(
                name=self._name,
                dimension=self._dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
            # wait until the new index reports ready
            for _ in range(60):
                if self._index_ready(self._pc.describe_index(self._name)):
                    break
                time.sleep(1)
        self._index = self._pc.Index(self._name)

    def _existing_index_names(self) -> list[str]:
        """Return existing index names across SDK / fake response shapes."""
        listing = self._pc.list_indexes()
        if hasattr(listing, "names"):
            return list(listing.names())
        names = []
        for item in listing:
            names.append(item["name"] if isinstance(item, dict) else getattr(item, "name", None))
        return [n for n in names if n]

    @staticmethod
    def _index_ready(desc) -> bool:
        """Read the ``ready`` flag from a describe_index result (dict or Struct)."""
        status = desc.get("status") if isinstance(desc, dict) else getattr(desc, "status", None)
        if status is None:
            return False
        if isinstance(status, dict):
            return bool(status.get("ready"))
        return bool(getattr(status, "ready", False))

    @property
    def index(self):
        if self._index is None:
            self.ensure_index()
        return self._index

    # --- writes --------------------------------------------------------------

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 100) -> int:
        """Store chunk vectors + metadata. Returns the number upserted."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        vectors = [
            {"id": chunk.id, "values": emb, "metadata": chunk.metadata}
            for chunk, emb in zip(chunks, embeddings)
        ]
        for start in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[start : start + batch_size])
        return len(vectors)

    # --- reads ---------------------------------------------------------------

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        """Return the ``top_k`` most similar chunks as plain dicts."""
        res = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )
        matches = res.get("matches", []) if isinstance(res, dict) else res.matches
        out: list[dict] = []
        for m in matches:
            meta = m["metadata"] if isinstance(m, dict) else m.metadata
            score = m["score"] if isinstance(m, dict) else m.score
            out.append(
                {
                    "text": meta.get("text", ""),
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "score": score,
                }
            )
        return out

    def delete_all(self) -> None:
        """Clear every vector (used when re-indexing a fresh document set)."""
        try:
            self.index.delete(delete_all=True)
        except Exception:
            # a brand-new / empty index raises "namespace not found" — harmless
            pass
