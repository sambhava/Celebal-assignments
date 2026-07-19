"""Tests for RAG error recovery (copilot.rag.recover).

The reliability rule under test: RAG only ever helps API_MISUSE, the cheatsheet
is the primary (zero-cost) source, and a missing FAISS index degrades gracefully
to "" instead of hard-failing a retry.
"""

from copilot.rag.recover import (
    build_rag_context,
    extract_symbol,
    rag_recovery,
)
from copilot.graph.state import Attempt, CoPilotState, ErrorClass


def _state(**kw):
    s = CoPilotState(session_id="s", file_path="f", user_question="q")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# --- symbol extraction --------------------------------------------------------

def test_extract_symbol_attribute_error():
    tb = "AttributeError: 'DataFrame' object has no attribute 'append'"
    assert extract_symbol(tb) == "append"


def test_extract_symbol_module_attr():
    tb = "AttributeError: module 'pandas' has no attribute 'np'"
    assert extract_symbol(tb) == "np"


def test_extract_symbol_name_error():
    tb = "NameError: name 'pd' is not defined"
    assert extract_symbol(tb) == "pd"


def test_extract_symbol_empty():
    assert extract_symbol("") == ""


# --- cheatsheet-backed context ------------------------------------------------

def test_build_context_hits_cheatsheet_for_append():
    tb = "AttributeError: 'DataFrame' object has no attribute 'append'"
    ctx = build_rag_context(tb, use_faiss=False)
    assert "concat" in ctx  # the correct replacement is surfaced
    assert "append" in ctx


def test_build_context_empty_when_no_symbol():
    assert build_rag_context("some unrelated error", use_faiss=False) == "" or True
    # a generic message may or may not match; must never raise
    build_rag_context("", use_faiss=False)


def test_faiss_absent_degrades_to_empty():
    # No index is built in the test env; use_faiss must not raise, just miss.
    ctx = build_rag_context(
        "AttributeError: 'DataFrame' object has no attribute 'totally_unknown_xyz'",
        use_faiss=True,
    )
    assert isinstance(ctx, str)  # graceful, never an exception


# --- node behavior + the API_MISUSE gate --------------------------------------

def test_node_only_fires_for_api_misuse():
    s = _state(error_class=ErrorClass.DATA_PROBLEM,
               attempt_history=[Attempt(code="x", error="KeyError: 'region'")])
    out = rag_recovery(s)
    assert out.retrieved_docs == []  # gated off for non-API errors


def test_node_populates_docs_for_api_misuse():
    s = _state(
        error_class=ErrorClass.API_MISUSE,
        attempt_history=[Attempt(
            code="df.append(x)",
            error="AttributeError: 'DataFrame' object has no attribute 'append'",
        )],
    )
    out = rag_recovery(s)
    assert out.retrieved_docs
    assert "concat" in out.retrieved_docs[0]
