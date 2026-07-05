"""Central configuration.

Values are read from (in priority order):
  1. An explicit override passed in code / the Streamlit sidebar.
  2. Environment variables (loaded from a local ``.env`` if present).
  3. Streamlit secrets (``st.secrets``) when running on Streamlit Cloud.
  4. Built-in defaults.

Keeping every tunable knob here means the pipeline code stays clean and the
"experiment with chunking / models" tasks from the assignment become one-line
changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional: only needed for local development
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _secret(name: str) -> str | None:
    """Look up a value from Streamlit secrets without a hard dependency."""
    try:
        import streamlit as st

        if name in st.secrets:  # type: ignore[operator]
            return str(st.secrets[name])
    except Exception:
        pass
    return None


def _get(name: str, default: str | None = None) -> str | None:
    return os.getenv(name) or _secret(name) or default


@dataclass
class Config:
    """All settings for one RAG session."""

    # --- credentials ---
    cohere_api_key: str = ""
    pinecone_api_key: str = ""

    # --- Pinecone index ---
    index_name: str = "rag-document-qa"
    cloud: str = "aws"
    region: str = "us-east-1"

    # --- models ---
    embed_model: str = "embed-multilingual-v3.0"  # 1024-dim, handles many languages
    chat_model: str = "command-r-plus-08-2024"
    rerank_model: str = "rerank-v3.5"
    embed_dim: int = 1024

    # --- chunking ---
    chunk_size: int = 1000  # characters per chunk
    chunk_overlap: int = 150  # characters shared between neighbours

    # --- retrieval ---
    top_k: int = 20  # candidates pulled from the vector store
    rerank_top_n: int = 5  # chunks kept after reranking / fed to the model

    @classmethod
    def load(cls, **overrides: object) -> "Config":
        """Build a Config from env/secrets, then apply any explicit overrides.

        ``None`` and empty-string overrides are ignored so a blank sidebar
        field never clobbers a value already provided via the environment.
        """
        cfg = cls(
            cohere_api_key=_get("COHERE_API_KEY", "") or "",
            pinecone_api_key=_get("PINECONE_API_KEY", "") or "",
            index_name=_get("PINECONE_INDEX", cls.index_name),
            cloud=_get("PINECONE_CLOUD", cls.cloud),
            region=_get("PINECONE_REGION", cls.region),
            embed_model=_get("EMBED_MODEL", cls.embed_model),
            chat_model=_get("CHAT_MODEL", cls.chat_model),
            rerank_model=_get("RERANK_MODEL", cls.rerank_model),
            chunk_size=int(_get("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(_get("CHUNK_OVERLAP", str(cls.chunk_overlap))),
            top_k=int(_get("TOP_K", str(cls.top_k))),
            rerank_top_n=int(_get("RERANK_TOP_N", str(cls.rerank_top_n))),
        )
        for key, value in overrides.items():
            if value in (None, ""):
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def validate(self) -> None:
        """Raise a clear error if something is misconfigured."""
        missing = []
        if not self.cohere_api_key:
            missing.append("COHERE_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise ValueError(
                "Missing required API key(s): "
                + ", ".join(missing)
                + ". Set them in a .env file or the app sidebar."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.rerank_top_n > self.top_k:
            raise ValueError("rerank_top_n cannot exceed top_k.")
