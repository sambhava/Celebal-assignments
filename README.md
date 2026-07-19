# Autonomous Data Science Co-Pilot

Turn a plain-English question and an uploaded dataset (CSV / Excel / JSON) into a
**finished analytical artefact** — a chart, a cleanliness report, a trend verdict —
rather than a wall of text describing how one *might* do the analysis.

The agent classifies the question's intent against the file's real schema, plans a
finite set of pandas steps, writes and **executes** the code in a hardened,
network-isolated sandbox, self-corrects on failure through a bounded,
error-taxonomy-gated repair loop, and always returns an honest report — including a
"what I could not do and why" section when a step can't be completed.

It is designed to run entirely on a **local open-source code model** with no external
API calls, so that even a fully successful prompt-injection attack from a malicious
upload produces only *inert* code: the sandbox has no network egress and holds no
secrets.

> **Status: full graph implemented end-to-end and tested.**
> All 11 nodes are implemented — ingest, profile, intent, planning, code-gen (live
> local LLM), sandboxed execution, error classification, RAG recovery, self-heal loop,
> grounded insight synthesis, and report assembly — covered by **160 passing tests**.
> `code_gen` runs against a real local model (qwen2.5-coder via Ollama); Docker and a
> FAISS docs index are optional enhancements, not required to run the suite. Nothing
> below claims to work that isn't backed by a test you can run.

---

## Why this exists (beyond the brief)

This started from an internship brief that asked for a Streamlit app that runs
LLM-generated pandas in a "subprocess sandbox" and "retries until it produces a clean
result." Both of those, taken literally, would fail a security review. This project
deliberately corrects them:

| Brief said | This project does | Why |
|---|---|---|
| "subprocess sandbox" | hardened container boundary (`--network none`, aggregate cgroup caps, dropped caps, read-only rootfs, no host env) | a same-user subprocess is not an isolation boundary; it can read host secrets and exfiltrate over the network |
| "retry until clean, never give up" | **bounded** self-heal loop → graceful degradation → honest partial report | "never give up" is a spec for an infinite loop and a cost incident; reliability means *degrading well* |
| RAG on every error | RAG **gated to one error class** (API-misuse) by a typed classifier | otherwise RAG masks genuine data problems (a missing column) as code problems and loops on unfixable input |

The full threat model, sandbox spec, and a red-team findings ledger (R1–R12, two of
them — R11/R12 — found and fixed during this build and verified against the real
libraries) are documented separately in the project docs (`SECURITY.md`).

---

## Architecture

An 11-node graph over a single `CoPilotState` object. The front half is fully
deterministic (no model needed); the model is only required for code generation and
grounded insight prose.

```
file ─▶ ingest_and_validate ─▶ schema_profile ─▶ intent_understanding ─▶ planning
                                                                            │
                        ┌───────────────────────────────────────────────────┘
                        ▼
                 ┌─▶ code_gen ─▶ sandboxed_execution ─▶ error_classification
                 │                                            │
                 │        ┌── retry (bounded) ◀── retry_router ┤
                 └────────┤                                    │
                          └── rag_recovery (API_MISUSE only)   │
                                                               ▼
                                        advance / degrade ─▶ insight_synthesis
                                                               │
                                                               ▼
                                                        report_assembly ─▶ Report
```

- **Two modes** (both required by the brief): *targeted* (question maps to a use-case
  template — deterministic skeleton, model only fills column names) and *open-ended*
  (vague question → a capped, ranked exploration battery, then stop).
- **Done-detection is structural.** The plan is finite and generated once; no node adds
  steps except the capped retry router. Infinite exploration is impossible by
  construction.
- **The error taxonomy gates everything.** Only `API_MISUSE` is RAG-eligible. A
  `KeyError` on a column that isn't in the schema is `DATA_PROBLEM` (surface it), not a
  code bug to retry. A clean-but-empty result is `SEMANTIC_EMPTY` (a finding, not an
  error). OOM/timeout is `RESOURCE_LIMIT` (degrade, don't re-run the same bomb).

See the technical design document (kept outside this repo) for per-node inputs,
outputs, and failure modes.

---

## Build status

| Node | Status | Needs |
|---|---|---|
| `ingest_and_validate` | ✅ implemented + tested | — |
| `schema_profile` | ✅ implemented + tested | — |
| `intent_understanding` | ✅ implemented + tested | — |
| `planning` | ✅ implemented + tested | — |
| `sandboxed_execution` | ✅ implemented + tested (Docker + dev backend) | Docker for the real boundary |
| `error_classification` | ✅ implemented + tested | — |
| `retry_router` + self-heal loop | ✅ implemented + tested | — |
| `report_assembly` | ✅ implemented + tested | — |
| `code_gen` | ✅ implemented + tested (live Ollama) | Ollama running `qwen2.5-coder:3b` |
| `rag_recovery` | ✅ implemented + tested (cheatsheet; FAISS optional) | FAISS index optional for long-tail |
| `insight_synthesis` | ✅ implemented + tested (grounded) | LLM optional (polish only) |

**All 11 nodes are implemented and the full graph runs end-to-end.** `code_gen` is
verified against a live local model (`qwen2.5-coder:3b` via Ollama); the sandbox
boundary and RAG's long-tail FAISS index are the two pieces that use optional external
infrastructure and degrade gracefully when absent.
The remaining three nodes are the ones that genuinely require it, and their input/output
contracts are already fixed in `copilot/graph/nodes.py`.

---

## Security highlights

The security boundary is code, not prose, and it is tested:

- **`copilot/sandbox/command.py`** builds the hardened `docker run` argv. Boundary flags
  (`--network none`, `--memory`==`--memory-swap`, `--cpus`, `--pids-limit`, `--cap-drop
  ALL`, `--read-only`, non-root user, no `-e`) are emitted as authoritative constants.
  Any boundary-defining flag a caller passes is **stripped**, so the boundary can only be
  tightened, never widened.
- **`copilot/sandbox/ci_check.py`** is a merge gate: it parses the emitted argv and fails
  the build if any aggregate cap is missing or if `--memory != --memory-swap`. A
  regression that drops a cap fails CI rather than shipping.
- **R11 (`copilot/ingest/field_count.py`)** — a ragged CSV whose rows carry one extra
  leading field is silently routed by pandas into `df.index`, bypassing a naive
  formula-defang and round-tripping a live `=cmd|...` formula into the exported CSV the
  reviewer opens in Excel. Guarded by a quote/delimiter-aware field-count invariant that
  hard-rejects before pandas can assign an implicit index. **Verified against pandas
  3.0.2.**
- **R12 (`copilot/ingest/xml_hardening.py`)** — the design originally specified
  `openpyxl.load_workbook(resolve_entities=False)`; that kwarg **does not exist** on
  openpyxl (it's an lxml parameter) — code written to the spec would crash. Replaced with
  a DTD/entity byte-screen over OOXML XML members + a pinned backend. **Verified against
  openpyxl 3.1.5.**

Both R11 and R12 were found by an adversarial red-team pass during this build and
confirmed by direct execution against the real libraries, not by assertion.

---

## Setup

Requires Python 3.11+. Docker is optional (needed only for the real execution boundary;
without it, tests use a guarded dev executor).

```bash
git clone <your-fork-url> autonomous-data-science-copilot
cd autonomous-data-science-copilot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # core + test deps
```

Core runtime deps are pinned to the versions everything is tested against
(`pandas==3.0.2`, `openpyxl==3.1.5`); see `requirements.txt` / `pyproject.toml`. The
charting and RAG dependencies are declared as optional extras (`.[viz]`, `.[rag]`) and
are not required to run the current test suite. Code generation talks to a local model
over HTTP via `httpx` (a core dep); the model itself runs in Ollama, installed
separately.

---

## Usage

Analyze a file with a plain-English question from the command line:

```bash
copilot analyze path/to/data.csv "which region has the highest total revenue?"
# or, without installing the console script:
python -m copilot analyze path/to/data.csv "which region has the highest total revenue?"
```

You get an honest report at the terminal: status, grounded insights, the path to
any chart produced, and a "what I could not do (and why)" section for any step
that couldn't be completed.

**Charts** require the `viz` extra (`pip install -e ".[viz]"`). **Code generation**
requires a local model — install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull qwen2.5-coder:3b
```

**Execution backend.** If Docker is installed, generated code runs inside the
hardened container automatically (the real security boundary). If Docker is *not*
present, you must pass `--allow-local` to opt into the local dev backend — which is
**not** a security boundary (same user, same machine, network access), so only use
it on files you trust:

```bash
copilot analyze data.csv "show revenue by region" --allow-local
```

---

## Running the tests

```bash
pytest                    # all 160 tests
pytest -q                 # quiet
python -m copilot.sandbox.ci_check   # the R1 launch-flag merge gate (exits non-zero on regression)
```

All 160 tests pass on Python 3.11 with the pinned dependencies, with no Docker and no
model required.

---

## Repository layout

```
copilot/
  __main__.py            enables `python -m copilot ...`
  cli.py                 the CLI front door (analyze command)
  errors.py              typed ingest errors (MalformedInput / SecurityViolation)
  ingest/
    preflight.py         host-side pre-parse gates (size, magic-byte, polyglot reject)
    field_count.py       R11 ragged-CSV guard
    xml_hardening.py     R12 OOXML XXE / entity-bomb screen
    load.py              guarded dataframe loaders (shape caps, R11/R12 wired in)
    profile.py           schema + frame profiling
  intent/understand.py   rule-based intent classification + schema validation
  planning/plan.py       finite plan builder (targeted templates + capped exploration)
  sandbox/
    command.py           hardened docker-run argv builder (the boundary)
    ci_check.py          R1 launch-flag merge gate
    execute.py           pluggable executor (Docker boundary + guarded dev backend)
  graph/
    state.py             CoPilotState + ErrorClass taxonomy
    classify.py          the RAG-gating error classifier
    router.py            bounded self-heal routing + escalation ladder
    loop.py              the always-terminating loop driver
    nodes.py             all 11 node bodies (implemented)
    run.py               linear walking-skeleton runner
  report/assemble.py     terminal report assembly (structural RunStatus, never LLM text)
tests/                   160 tests across 16 files
```

Design and reference documents (PRD, technical design, security ledger, build plan,
and the project report/explainer PDFs) are maintained separately, outside this
repository.

---

## Roadmap

Done during this build:

- ✅ `code_gen` wired against a local model (Qwen2.5-Coder via Ollama) with a pandas
  deprecation cheatsheet and per-step operation hints for weak models.
- ✅ `insight_synthesis` with strict grounding (references only values present in the
  execution result — no invented numbers).
- ✅ A CLI front door (`copilot analyze ...`) over the full graph.

Still ahead:

1. `rag_recovery` long-tail: index the *installed* pandas version's docs (FAISS +
   local embeddings); retrieval is already gated to `API_MISUSE` (cheatsheet path
   ships today, FAISS index optional).
2. Streamlit UI over the graph.
3. Run the full adversarial evaluation suite from the build-plan document.
