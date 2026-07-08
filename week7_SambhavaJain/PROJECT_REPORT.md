# DocMind — RAG Document Q&A · Full Project Report

A complete, self-contained explanation of the project: what it does, why every
piece exists, how the code works line-by-line, its strengths, its limits, and
how it differs from a typical beginner RAG. Read top-to-bottom to understand the
whole system; jump to a module section to learn one file.

---

## 1. What the project is (in one paragraph)

**DocMind** is a Retrieval-Augmented Generation (RAG) web app. You add a
document (PDF or text); it splits the text into chunks, turns each chunk into a
vector (embedding), and stores those vectors in a cloud vector database. It then
**automatically writes an overview** of the document, and lets you **chat** with
follow-up questions. For each question it finds the most relevant chunks,
re-ranks them for precision, and asks a language model to answer **using only
those chunks** — so answers are grounded in your document, not the model's
memory, and every answer shows the exact passages it used.

---

## 2. Why RAG at all (the core idea)

A plain LLM answers from what it memorised during training. That fails for:
- **private/custom data** it never saw (your notes, a company handbook), and
- **factual precision** — it may "hallucinate" a plausible-sounding wrong answer.

RAG fixes both by inserting a **retrieval** step before generation:

```
Retrieval  → find the passages most relevant to the question
Augmentation → paste those passages into the model's prompt as context
Generation → the model answers FROM that context, and cites it
```

The model becomes a *reasoning-and-writing* engine over *your* facts, instead of
a memory it might get wrong.

---

## 3. The technology stack (and why each was chosen)

| Layer | Choice | Why this one |
|---|---|---|
| Language | **Python 3.11** | Standard for ML/AI tooling. |
| Embeddings | **Cohere `embed-multilingual-v3.0`** (1024-dim) | One vendor for embed+rerank+chat = less glue code; multilingual variant tolerates non-English docs. |
| Reranking | **Cohere `rerank-v3.5`** | A cross-encoder that scores query↔passage pairs far more accurately than vector similarity alone. |
| Generation | **Cohere `command-r-plus-08-2024`** | Purpose-built for RAG: takes structured `documents` and returns **citations** natively. |
| Vector DB | **Pinecone** (serverless) | Managed, scalable similarity search; no infra to run. |
| UI | **Streamlit** | Pure-Python web apps; fast to build, easy to deploy free. |
| PDF parsing | **pypdf** | Lightweight, pure-Python page-by-page text extraction. |
| Config | **python-dotenv** | Loads API keys from a local `.env`. |
| Tests | **pytest** | The pipeline is unit-tested with the network mocked. |

> Note: Cohere has **no** separate "generation-only" limitation — it provides
> embeddings, reranking, and chat, which is exactly why using one provider keeps
> the codebase small.

---

## 4. The end-to-end data flow

```
        ┌───────────────────────── INGEST (once per document) ──────────────────────────┐
 upload → extract text per page → clean whitespace → recursive split (+overlap)
                                                                │  chunks (with source+page)
                                              Cohere embed(input_type="search_document")
                                                                │  1024-dim vectors
                                                        Pinecone upsert (vector + metadata)

        ┌───────────────────────── SUMMARIZE (auto, right after ingest) ────────────────┐
 first ~15 chunks → Cohere chat(SUMMARY_PROMPT, documents=chunks) → Overview text

        ┌───────────────────────── ASK (every question) ────────────────────────────────┐
 question → Cohere embed(input_type="search_query") → Pinecone query(top_k=20)
                                                                │  20 candidate chunks
                                              Cohere rerank(query, candidates, top_n=5)
                                                                │  5 best chunks
                                       Cohere chat(question, documents=5 chunks)
                                                                │
                                       grounded answer + citations + the 5 sources
```

Two ideas make this accurate:
1. **Asymmetric embeddings** — documents and queries are embedded with different
   `input_type`s (`search_document` vs `search_query`). Cohere's v3 models are
   trained for this and retrieval quality drops noticeably if you don't.
2. **Two-stage retrieval** — cheap vector search casts a wide net (20), then an
   expensive-but-accurate reranker keeps only the 5 truly relevant chunks.

---

## 5. Project structure

```
rag-document-qa/
├── app.py                     # Streamlit UI (NotebookLM-style, light/dark)
├── rag/                       # the reusable, UI-independent engine
│   ├── __init__.py
│   ├── config.py              # all settings + validation
│   ├── ingest.py              # PDF/text → cleaned, page-tagged, overlapping chunks
│   ├── embeddings.py          # Cohere embed wrapper (batched)
│   ├── vectorstore.py         # Pinecone create / upsert / query
│   ├── retriever.py           # vector search + Cohere rerank
│   ├── generator.py           # grounded answer + citations + summary
│   └── pipeline.py            # orchestrates everything: ingest / summarize / ask
├── tests/                     # 17 offline unit tests (Cohere + Pinecone mocked)
│   ├── test_config.py
│   ├── test_ingest.py
│   └── test_pipeline.py
├── sample_docs/rag_overview.txt   # a document to try immediately
├── scripts/capture_screenshots.py # Playwright script that drives the live app
├── docs/screenshots/          # README images (real runs)
├── .streamlit/config.toml     # base Streamlit theme
├── requirements.txt
├── .env.example               # template for your keys (real .env is gitignored)
└── README.md
```

**Design principle:** the `rag/` package knows nothing about Streamlit. The UI
is a thin client over `RAGPipeline`. That separation means you can reuse the
engine in a CLI, an API, a notebook, or *your own project* without touching the
UI code.

---

## 6. Module-by-module walkthrough

### 6.1 `rag/config.py` — one place for every knob
- A `@dataclass Config` holds credentials, index settings, model names, chunking
  params, and retrieval params.
- `Config.load(**overrides)` resolves each value in priority order:
  **explicit override → environment variable → Streamlit secret → default.**
  Blank/`None` overrides are ignored, so an empty sidebar box never wipes a key
  that came from `.env`.
- `validate()` fails fast with a clear message if a key is missing, if
  `chunk_overlap >= chunk_size` (would loop/duplicate), or if
  `rerank_top_n > top_k` (can't keep more than you fetched).
- **Learn from this:** centralising configuration + validating early is what
  keeps the rest of the code free of scattered `os.getenv` calls and defensive
  checks.

### 6.2 `rag/ingest.py` — turning files into good chunks
- `extract_pages(data, filename)` → list of `(page_number, text)`. PDFs go
  page-by-page (via pypdf); text/markdown is treated as one page `0`.
- `clean_text()` collapses the messy whitespace PDF extraction produces
  (`\r\n`, runs of spaces, 3+ blank lines).
- `_split_recursive()` is the heart: it tries separators **coarse→fine**
  (`"\n\n"` → `"\n"` → `". "` → `" "` → `""`). It packs as many pieces as fit
  under `chunk_size`, and only falls back to a hard character cut as a last
  resort. This keeps sentences and paragraphs intact instead of slicing
  mid-word.
- `_add_overlap()` prepends the tail (last `chunk_overlap` chars) of each chunk
  to the next, so context isn't lost at chunk boundaries.
- `chunk_document()` runs all of the above and produces `Chunk` objects with a
  stable id (`filename::pN::cM`) and metadata `{source, page, text}`.
- **Why chunking matters:** embeddings represent a *fixed-length* idea. Chunks
  too big dilute relevance; too small lose context. Recursive + overlap is the
  standard "good default."

### 6.3 `rag/embeddings.py` — text → vectors
- `Embedder` wraps Cohere's embed endpoint and **batches** (96 texts/call, the
  API max) so large documents don't exceed request limits.
- `embed_documents()` uses `input_type="search_document"`;
  `embed_query()` uses `"search_query"` — the asymmetry explained in §4.
- A defensive detail: the Cohere SDK returns embeddings under the attribute
  `float_` (trailing underscore, because `float` is a reserved word). The code
  reads `float_` and falls back to `float` so it works across SDK versions and
  with the test fakes. *(This was a real bug caught during verification.)*

### 6.4 `rag/vectorstore.py` — Pinecone wrapper
- `ensure_index()` creates the serverless index if missing (dimension 1024,
  cosine metric) and waits until it reports "ready," then connects.
- `_existing_index_names()` and `_index_ready()` normalise Pinecone's response
  shapes (the SDK returns objects with `.names()`, not dicts) — another real bug
  caught in verification.
- `upsert()` stores `{id, values, metadata}` in batches of 100. **The chunk text
  lives in the metadata**, so a query returns the content directly — no second
  round-trip to fetch text.
- `query()` returns the top-k matches as plain dicts `{text, source, page,
  score}`.
- `delete_all()` clears the index when you re-index a fresh document set
  (the "Replace existing index" behaviour).

### 6.5 `rag/retriever.py` — search + rerank
- `retrieve()` embeds the query, pulls `top_k` candidates from Pinecone, then
  reranks.
- `_rerank()` calls Cohere Rerank, which returns candidate indices sorted by a
  true relevance score; it keeps the top `n` and attaches `rerank_score`.
- **Graceful degradation:** if rerank ever fails, it falls back to plain vector
  order instead of crashing.

### 6.6 `rag/generator.py` — grounded answers + summaries
- `generate(question, chunks)` passes the chunks to Cohere Chat as structured
  `documents` and returns an `Answer(text, sources, citations)`.
  - `SYSTEM_PREAMBLE` instructs the model to answer **only** from the documents
    and to say so if the answer isn't there — this is what suppresses
    hallucination.
  - With no chunks, it returns a friendly "couldn't find it" instead of guessing.
- `summarize(chunks)` uses a dedicated `SUMMARY_PROMPT` (2–3 sentence summary +
  bulleted key points) — this powers the auto-Overview.
- `_extract_text()` / `_extract_citations()` are **defensive parsers**: they read
  the response shape but never crash if a field is missing — important because
  SDK response shapes vary by version.

### 6.7 `rag/pipeline.py` — the orchestrator
- `RAGPipeline` builds the four stages (embedder, store, retriever, generator)
  and exposes three methods:
  - `ingest(files, replace)` → chunk, embed, upsert; remembers the chunks and
    source names for the UI/summary.
  - `summarize()` → overview of the first ~15 ingested chunks.
  - `ask(question)` → retrieve+rerank, then generate a grounded `Answer`.
- **Dependency injection:** the constructor accepts optional `cohere_client` /
  `pinecone_client`. Production builds real ones; **tests inject fakes** — which
  is why the whole pipeline is testable offline with no keys and no network.

### 6.8 `app.py` — the NotebookLM-style UI
- **Palette-driven theming:** `LIGHT`/`DARK` dicts + `build_css(dark)` generate
  all CSS from one place, so the 🌙/☀️ sidebar toggle re-themes the entire app at
  runtime (Streamlit's static `config.toml` theme can't do runtime switching).
- **Keys hidden:** `get_config()` reads keys silently; `key_setup()` only shows
  password fields if a key is actually missing.
- **Layout:** left sidebar = *Sources* rail (upload + loaded files); main area =
  auto *Overview* card, then a chat (`st.chat_message` bubbles +
  `st.chat_input`) with a *Sources* expander under each answer.
- **State:** `st.session_state` holds the pipeline, the summary, the chat
  history, the loaded-source names, and the dark-mode flag.

---

## 7. Testing & verification

- **17 unit tests**, all with Cohere and Pinecone **mocked** (fake clients that
  mimic the real SDK response shapes), so they run in ~1 second with no keys:
  - `test_ingest.py` — cleaning, chunk sizing, overlap, metadata, empty input.
  - `test_config.py` — defaults, override rules, validation failures.
  - `test_pipeline.py` — full ingest→ask→summarize flow, correct `input_type`s,
    graceful "no matches" path.
- **Live verification** — the app was actually run against real Cohere+Pinecone
  and driven end-to-end with Playwright (upload → auto-summary → chat → dark
  mode) to capture the README screenshots. Two real SDK bugs (`float_`,
  Pinecone `.names()`) were found and fixed this way — the tests alone wouldn't
  have caught them because the fakes matched the *assumed* shapes.

---

## 8. What it's best at (strengths)

1. **Grounded, cited answers** — every answer shows the exact passages (file +
   page + relevance score) it used; the model is told not to invent.
2. **Higher retrieval precision** than a basic RAG, thanks to two-stage
   search + rerank and correct asymmetric embeddings.
3. **Great UX** — auto-summary on upload means you understand a document before
   asking anything; chat makes follow-ups natural.
4. **Clean, reusable engine** — `rag/` is UI-agnostic and fully unit-tested; you
   can lift it straight into another project.
5. **Multilingual-tolerant** and **multi-document** capable.
6. **Robust** — graceful fallbacks (rerank failure, empty results), defensive
   SDK parsing, fail-fast config validation.

---

## 9. Limitations (be honest about these)

1. **Needs internet + two API keys** (Cohere, Pinecone) — not fully offline.
2. **Cost/quota** — free tiers are fine for a demo; heavy use hits rate limits.
3. **Scanned PDFs / images** — pypdf extracts *text*, not OCR. Image-only PDFs
   yield little text. (Fix: add an OCR step like Tesseract.)
4. **Ephemeral chat memory** — follow-ups are independent retrievals; the model
   doesn't carry prior turns as conversational context (each question is
   answered fresh from the document). Good for factual QA, less so for
   multi-turn reasoning.
5. **First-run latency** — creating the Pinecone index takes ~30–60s the very
   first time.
6. **Single shared index** — re-indexing with "replace" clears previous docs;
   there's no per-user namespace separation (fine for a single-user demo).
7. **Chunk metadata size** — storing full text in Pinecone metadata is simple
   but has per-vector size limits for very large chunks.

---

## 10. How it differs from a typical beginner RAG / the reference repo

The reference project (Cohere + Pinecone + Streamlit) does the basic
embed→store→retrieve→answer loop. DocMind adds, specifically:

| Aspect | Typical beginner RAG | DocMind |
|---|---|---|
| Chunking | fixed-size cuts | recursive, boundary-aware, overlapping |
| Retrieval | vector search only | vector search **+ rerank** |
| Embedding types | same for query & doc | correct **asymmetric** `input_type`s |
| Answers | plain text | **grounded + citations + source passages** |
| First experience | must think of a question | **auto-summary** on upload |
| Interaction | one-shot Q&A box | **chat** with follow-ups |
| Keys | pasted in the UI | **hidden**, read from `.env`/secrets |
| Structure | one script | **modular package + 17 tests** |
| UI | default Streamlit | NotebookLM-style, **light/dark** |

---

## 11. How to reuse this in your own project

- The engine is the `rag/` package. Minimal usage:
  ```python
  from rag.config import Config
  from rag.pipeline import RAGPipeline

  cfg = Config.load()                 # reads .env
  rag = RAGPipeline(cfg)
  rag.ingest([(open("doc.pdf","rb").read(), "doc.pdf")], replace=True)
  print(rag.summarize())
  print(rag.ask("What is X?").text)
  ```
- Swap the vendor by editing three wrappers (`embeddings.py`, `retriever.py`,
  `generator.py`) — the pipeline and UI don't change.
- Reuse the **testing pattern** (inject fake clients) for any API-backed code.

---

## 12. Glossary

- **Embedding** — a list of numbers (here 1024) representing text meaning;
  similar texts have nearby vectors.
- **Vector database** — stores embeddings and finds nearest neighbours fast.
- **Cosine similarity** — the closeness metric used (angle between vectors).
- **Chunk** — a small passage of a document that gets embedded and retrieved.
- **Reranker** — a model that re-scores retrieved passages by true relevance.
- **Grounding** — forcing the answer to come from provided text, not model memory.
- **Citation** — a mapping from part of the answer back to its source passage.
