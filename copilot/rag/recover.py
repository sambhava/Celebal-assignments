"""Build ``rag_context`` for an API_MISUSE retry.

Pipeline (TECHNICAL_DESIGN.md §4), cheapest-first:

    1. Parse the offending symbol out of the traceback
       (e.g. ``AttributeError: 'DataFrame' object has no attribute 'append'``
        -> ``append``).
    2. Match it against the version-locked deprecation cheatsheet. A hit is the
       highest-signal, zero-cost fix for a weak model's most common error.
    3. Only if the cheatsheet misses, optionally query a FAISS index over pinned
       pandas docs. That index is optional infrastructure; when it is not built
       (no faiss / no embeddings / no artifact) this step is skipped gracefully
       and we return whatever the cheatsheet produced (possibly empty).

The result is a short text block the code-gen prompt injects under
"RELEVANT PANDAS DOCS / FIXES:" at the top of the escalation ladder. It is only
ever consulted for API_MISUSE — the classifier gates that upstream.
"""

from __future__ import annotations

import re
from typing import List, Optional

from copilot.codegen.cheatsheet import DEPRECATIONS

# Symbols an API_MISUSE traceback tends to name. We pull the attribute/method
# out of the common message shapes so we can match the cheatsheet precisely.
_ATTR_RE = re.compile(r"has no attribute ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]")
_NAME_RE = re.compile(r"name ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"] is not defined")
_MODULE_ATTR_RE = re.compile(r"module ['\"][\w.]+['\"] has no attribute ['\"]([A-Za-z_]\w*)['\"]")
_GENERIC_TOKEN_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")  # fallback: a called name


def extract_symbol(traceback: str) -> str:
    """Best-effort: the offending attribute/method/name from a traceback."""
    if not traceback:
        return ""
    for rx in (_ATTR_RE, _MODULE_ATTR_RE, _NAME_RE):
        m = rx.search(traceback)
        if m:
            return m.group(1)
    # last resort: the last called-name token on the final traceback line
    last = [ln for ln in traceback.strip().splitlines() if ln.strip()]
    if last:
        cands = _GENERIC_TOKEN_RE.findall(last[-1])
        if cands:
            return cands[-1]
    return ""


def _cheatsheet_hits(symbol: str) -> List[str]:
    """Deprecation entries whose wrong-form mentions the offending symbol."""
    if not symbol:
        return []
    hits = []
    for wrong, right in DEPRECATIONS:
        if re.search(rf"\b{re.escape(symbol)}\b", wrong):
            hits.append(f"`{wrong}` is removed/deprecated -> use `{right}`.")
    return hits


def _faiss_hits(query: str, k: int = 3) -> List[str]:
    """Optional long-tail retrieval over pinned pandas docs.

    Returns [] gracefully whenever the optional index or its dependencies are
    unavailable — RAG must never hard-fail a retry; a miss just means the model
    retries with cheatsheet + history only.
    """
    try:
        from copilot.rag.index import PandasDocIndex  # optional module
    except Exception:
        return []
    try:
        idx = PandasDocIndex.load_default()
        if idx is None:
            return []
        return idx.search(query, k=k)
    except Exception:
        return []


def build_rag_context(traceback: str, *, use_faiss: bool = True) -> str:
    """Return the retrieval block for the retry prompt (may be empty)."""
    symbol = extract_symbol(traceback)
    blocks: List[str] = []

    hits = _cheatsheet_hits(symbol)
    if hits:
        blocks.append("Known API changes for this environment (pandas 3.x):")
        blocks.extend(f"  - {h}" for h in hits)

    if use_faiss and not hits and symbol:
        doc_hits = _faiss_hits(f"pandas {symbol} correct usage")
        if doc_hits:
            blocks.append("From the pinned pandas docs:")
            blocks.extend(f"  - {d}" for d in doc_hits)

    return "\n".join(blocks)


def rag_recovery(state) -> object:
    """Node body: populate ``state.retrieved_docs`` / ``state.rag_context``.

    Only meaningful for API_MISUSE (the classifier gates this). We read the most
    recent attempt's error text, build a context block, and stash it for the next
    code-gen. If nothing is found, the retry simply proceeds without RAG.
    """
    from copilot.graph.state import ErrorClass

    if state.error_class is not ErrorClass.API_MISUSE:
        return state  # defensive: never retrieve for non-API errors

    traceback = ""
    if state.attempt_history:
        traceback = getattr(state.attempt_history[-1], "error", "") or ""
    if not traceback:
        traceback = state.execution_result.get("traceback", "") or state.execution_result.get("stderr", "")

    context = build_rag_context(traceback)
    if context:
        state.retrieved_docs = [context]
    return state
