"""High-level RAG orchestrator.

Wires the config, clients, and stages together behind two simple methods:
``ingest`` (index documents) and ``ask`` (answer a question). The Streamlit app
and the tests both talk to this class rather than the individual stages.
"""

from __future__ import annotations

from .config import Config
from .embeddings import Embedder
from .generator import Answer, Generator
from .ingest import chunk_document
from .retriever import Retriever
from .vectorstore import VectorStore


def _build_clients(cfg: Config):
    """Create the Cohere and Pinecone SDK clients."""
    import cohere
    from pinecone import Pinecone

    co = cohere.ClientV2(api_key=cfg.cohere_api_key)
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    return co, pc


class RAGPipeline:
    def __init__(self, cfg: Config, cohere_client=None, pinecone_client=None):
        cfg.validate()
        self.cfg = cfg

        if cohere_client is None or pinecone_client is None:
            cohere_client, pinecone_client = _build_clients(cfg)

        self._embedder = Embedder(cohere_client, cfg.embed_model)
        self._store = VectorStore(
            pinecone_client,
            index_name=cfg.index_name,
            dimension=cfg.embed_dim,
            cloud=cfg.cloud,
            region=cfg.region,
        )
        self._retriever = Retriever(
            self._embedder, self._store, cohere_client, cfg.rerank_model
        )
        self._generator = Generator(cohere_client, cfg.chat_model)

    # --- indexing ------------------------------------------------------------

    def ingest(self, files: list[tuple[bytes, str]], replace: bool = False) -> dict:
        """Chunk, embed, and store a list of ``(bytes, filename)`` documents.

        Returns a small summary dict for display in the UI.
        """
        self._store.ensure_index()
        if replace:
            self._store.delete_all()

        all_chunks = []
        for data, filename in files:
            all_chunks.extend(
                chunk_document(
                    data,
                    filename,
                    chunk_size=self.cfg.chunk_size,
                    chunk_overlap=self.cfg.chunk_overlap,
                )
            )

        if not all_chunks:
            return {"files": len(files), "chunks": 0}

        embeddings = self._embedder.embed_documents([c.text for c in all_chunks])
        self._store.upsert(all_chunks, embeddings)
        return {"files": len(files), "chunks": len(all_chunks)}

    # --- querying ------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        """Retrieve relevant context and generate a grounded answer."""
        chunks = self._retriever.retrieve(
            question,
            top_k=self.cfg.top_k,
            rerank_top_n=self.cfg.rerank_top_n,
        )
        return self._generator.generate(question, chunks)
