"""NotebookLM-style Streamlit UI for the RAG Document Q&A system.

Layout:
  - Left rail (sidebar): your sources — upload documents and see what's loaded.
  - Main area: an auto-generated Overview of your documents, then a chat where
    you can ask follow-up questions grounded in those documents.

API keys are read silently from .env / Streamlit secrets. The app only asks for
a key if one is actually missing.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from rag.config import Config
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="DocMind — Document Q&A", page_icon="📓", layout="wide")


# --- styling -----------------------------------------------------------------

CSS = """
<style>
/* hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}

/* calm, paper-like canvas */
.stApp {background: #F7F8FA;}
section[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E8EAED;}

/* typography */
html, body, [class*="css"] {font-family: "Segoe UI", "Google Sans", system-ui, sans-serif;}
h1, h2, h3 {letter-spacing: -0.01em; color: #1F2733;}

/* brand mark */
.brand {display:flex; align-items:center; gap:.55rem; margin:.2rem 0 1.4rem;}
.brand .logo {font-size:1.6rem;}
.brand .name {font-size:1.35rem; font-weight:700; color:#1F2733;}
.brand .tag {font-size:.8rem; color:#6B7280; margin-top:-2px;}

/* overview card — the signature element (styled bordered container) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.eyebrow) {
  background:#FFFFFF; border:1px solid #E8EAED !important; border-left:4px solid #4B6FDB !important;
  border-radius:16px !important; box-shadow:0 1px 3px rgba(16,24,40,.05);
}
.eyebrow {
  display:inline-flex; align-items:center; gap:.4rem; font-size:.72rem;
  font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#4B6FDB;
}

/* source chips in the sidebar */
.source-item {
  display:flex; align-items:center; gap:.5rem; padding:.5rem .7rem; margin-bottom:.4rem;
  background:#F4F6FB; border:1px solid #E4E8F2; border-radius:10px; font-size:.88rem; color:#374151;
}

/* empty state */
.empty {
  text-align:center; color:#6B7280; padding:3.5rem 1rem;
  border:1px dashed #D6DAE1; border-radius:16px; background:#FFFFFFAA;
}
.empty .big {font-size:2.4rem; margin-bottom:.4rem;}

/* source citations under answers */
.cite {font-size:.82rem; color:#4B5563;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --- helpers -----------------------------------------------------------------

def get_config() -> Config:
    """Load config from env/secrets; prompt for keys only if missing."""
    cfg = Config.load()
    try:
        cfg.validate()
        st.session_state.keys_ok = True
    except ValueError:
        st.session_state.keys_ok = False
    return cfg


def key_setup(cfg: Config) -> Config:
    """Minimal, tucked-away key entry shown only when keys are absent."""
    with st.sidebar:
        st.warning("Add your API keys to get started.")
        c_key = st.text_input("Cohere API key", type="password", key="c_key")
        p_key = st.text_input("Pinecone API key", type="password", key="p_key")
        st.caption(
            "Tip: put these in a `.env` file and you'll never see this again. "
            "Free keys: cohere.com · pinecone.io"
        )
    return Config.load(cohere_api_key=c_key, pinecone_api_key=p_key)


def render_overview(summary: str) -> None:
    # a single contained card: eyebrow label + the grounded summary body
    with st.container(border=True):
        st.markdown('<span class="eyebrow">✦ Overview</span>', unsafe_allow_html=True)
        st.markdown(summary)


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"📚 Sources ({len(sources)} passages)"):
        for i, src in enumerate(sources, start=1):
            score = src.get("rerank_score", src.get("score"))
            score_str = f" · relevance {score:.2f}" if isinstance(score, (int, float)) else ""
            page = src.get("page", 0)
            loc = f"{src['source']}" + (f", p.{page}" if page else "")
            st.markdown(f"**{i}. {loc}**{score_str}")
            text = src["text"]
            st.caption(text[:400] + ("…" if len(text) > 400 else ""))


# --- app ---------------------------------------------------------------------

def main() -> None:
    st.session_state.setdefault("indexed", False)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("summary", "")
    st.session_state.setdefault("loaded_sources", [])

    cfg = get_config()
    if not st.session_state.keys_ok:
        cfg = key_setup(cfg)

    # ---- sidebar: sources rail ----
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="logo">📓</span>'
            '<div><div class="name">DocMind</div>'
            '<div class="tag">Chat with your documents</div></div></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Sources")
        files = st.file_uploader(
            "Add PDFs or text files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        add = st.button("＋ Add & summarize", type="primary", use_container_width=True,
                        disabled=not files)

        if st.session_state.loaded_sources:
            st.caption("Loaded")
            for name in st.session_state.loaded_sources:
                st.markdown(f'<div class="source-item">📄 {name}</div>', unsafe_allow_html=True)

    # ---- handle adding documents ----
    if add and files:
        try:
            cfg.validate()
        except ValueError as e:
            st.sidebar.error(str(e))
            st.stop()

        payload = [(f.getvalue(), f.name) for f in files]
        with st.spinner("Reading, indexing, and summarizing your document(s)…"):
            try:
                pipeline = RAGPipeline(cfg)
                summary_info = pipeline.ingest(payload, replace=True)
                summary = pipeline.summarize()
            except Exception as e:
                st.error(f"Couldn't process the document(s): {e}")
                st.stop()

        st.session_state.pipeline = pipeline
        st.session_state.indexed = True
        st.session_state.summary = summary
        st.session_state.loaded_sources = summary_info.get("sources", [f.name for f in files])
        st.session_state.messages = []  # fresh conversation per document set
        st.rerun()

    # ---- main area ----
    if not st.session_state.indexed:
        st.markdown(
            '<div class="empty"><div class="big">📄→💬</div>'
            "<b>Add a source to begin</b><br>"
            "Upload a document in the left panel. DocMind will read it, write a short "
            "overview, and then answer any follow-up questions you have.</div>",
            unsafe_allow_html=True,
        )
        return

    # overview (auto-summary)
    render_overview(st.session_state.summary)

    # conversation
    st.subheader("Ask a follow-up")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    prompt = st.chat_input("Ask anything about your document…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = st.session_state.pipeline.ask(prompt)
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    st.stop()
            st.markdown(answer.text)
            if answer.sources:
                render_sources(answer.sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer.text, "sources": answer.sources}
        )


if __name__ == "__main__":
    main()
