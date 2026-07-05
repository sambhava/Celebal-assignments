"""Streamlit UI for the RAG Document Question Answering system.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from rag.config import Config
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")


# --- sidebar: credentials & settings ----------------------------------------

def sidebar_config() -> Config:
    st.sidebar.header("⚙️ Configuration")

    defaults = Config.load()  # picks up .env / secrets if present

    with st.sidebar:
        st.caption("Keys are read from `.env` if present — override here if needed.")
        cohere_key = st.text_input(
            "Cohere API key", value=defaults.cohere_api_key, type="password"
        )
        pinecone_key = st.text_input(
            "Pinecone API key", value=defaults.pinecone_api_key, type="password"
        )

        with st.expander("Advanced settings"):
            index_name = st.text_input("Pinecone index", value=defaults.index_name)
            chunk_size = st.slider("Chunk size", 300, 2000, defaults.chunk_size, 50)
            chunk_overlap = st.slider("Chunk overlap", 0, 400, defaults.chunk_overlap, 10)
            top_k = st.slider("Vector candidates (top_k)", 5, 50, defaults.top_k, 1)
            rerank_top_n = st.slider("Chunks after rerank", 1, 15, defaults.rerank_top_n, 1)

    return Config.load(
        cohere_api_key=cohere_key,
        pinecone_api_key=pinecone_key,
        index_name=index_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
    )


# --- main --------------------------------------------------------------------

def main() -> None:
    st.title("📄 Document Question Answering (RAG)")
    st.write(
        "Upload one or more documents, then ask questions. Answers are grounded "
        "in your documents and show their sources."
    )

    cfg = sidebar_config()

    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "history" not in st.session_state:
        st.session_state.history = []

    # --- step 1: upload & index ---
    st.header("1. Add documents")
    files = st.file_uploader(
        "PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    replace = st.checkbox("Replace existing index (clear old documents first)", value=True)

    if st.button("Process documents", type="primary", disabled=not files):
        try:
            cfg.validate()
        except ValueError as e:
            st.error(str(e))
            st.stop()

        payload = [(f.getvalue(), f.name) for f in files]
        with st.spinner("Chunking, embedding, and indexing…"):
            try:
                pipeline = RAGPipeline(cfg)
                summary = pipeline.ingest(payload, replace=replace)
            except Exception as e:  # surface any SDK / network error clearly
                st.error(f"Indexing failed: {e}")
                st.stop()
        st.session_state.pipeline = pipeline
        st.session_state.indexed = True
        st.success(
            f"Indexed {summary['chunks']} chunks from {summary['files']} file(s)."
        )

    # --- step 2: ask ---
    st.header("2. Ask a question")
    if not st.session_state.indexed:
        st.info("Process at least one document to start asking questions.")
        return

    question = st.text_input("Your question", placeholder="What is the main idea of the document?")
    if st.button("Ask", disabled=not question):
        with st.spinner("Retrieving context and generating an answer…"):
            try:
                answer = st.session_state.pipeline.ask(question)
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()
        st.session_state.history.insert(0, (question, answer))

    # --- answers ---
    for q, ans in st.session_state.history:
        st.markdown(f"**Q: {q}**")
        st.markdown(ans.text)

        if ans.sources:
            with st.expander(f"📚 Sources ({len(ans.sources)} chunks used)"):
                for i, src in enumerate(ans.sources, start=1):
                    score = src.get("rerank_score", src.get("score"))
                    score_str = f" · relevance {score:.3f}" if isinstance(score, (int, float)) else ""
                    st.markdown(
                        f"**{i}. {src['source']} — page {src['page']}**{score_str}"
                    )
                    st.caption(src["text"][:500] + ("…" if len(src["text"]) > 500 else ""))
        st.divider()


if __name__ == "__main__":
    main()
