"""Tests for error_classification — the gate that keeps the self-heal loop honest.

The load-bearing property (TECHNICAL_DESIGN.md §1): only API_MISUSE is
RAG-eligible; a missing-column KeyError is a DATA_PROBLEM, not code to heal; a
clean-but-empty result is surfaced, never retried.
"""

import pytest

from copilot.graph.classify import classify
from copilot.graph.state import ErrorClass

SCHEMA = {"region": {}, "revenue": {}, "date": {}}


def _err(exc="", tb="", **kw):
    d = {"ok": False, "exception_type": exc, "traceback": tb}
    d.update(kw)
    return d


def test_missing_column_keyerror_is_data_problem_not_api_misuse():
    r = _err(exc="KeyError", tb="KeyError: 'profit'")
    cls = classify(r, SCHEMA)
    assert cls is ErrorClass.DATA_PROBLEM
    # the whole point: RAG must NOT fire on a missing column
    assert not cls.is_rag_eligible


def test_keyerror_on_existing_column_is_recoverable_not_data_problem():
    # key exists in schema -> it's a real code bug, not a missing-column data problem
    r = _err(exc="KeyError", tb="KeyError: 'region'")
    assert classify(r, SCHEMA) is ErrorClass.RUNTIME_RECOVERABLE


def test_attributeerror_is_api_misuse_and_rag_eligible():
    r = _err(exc="AttributeError",
             tb="AttributeError: 'DataFrame' object has no attribute 'append'")
    cls = classify(r, SCHEMA)
    assert cls is ErrorClass.API_MISUSE
    assert cls.is_rag_eligible  # the ONE class that gets RAG


def test_syntaxerror_retries_without_rag():
    r = _err(exc="SyntaxError", tb="SyntaxError: invalid syntax")
    cls = classify(r, SCHEMA)
    assert cls is ErrorClass.SYNTAX
    assert not cls.is_rag_eligible


def test_oom_is_resource_limit():
    r = _err(exc="", tb="", oom_killed=True)
    assert classify(r, SCHEMA) is ErrorClass.RESOURCE_LIMIT


def test_timeout_is_resource_limit():
    r = _err(oom_killed=False, timed_out=True)
    assert classify(r, SCHEMA) is ErrorClass.RESOURCE_LIMIT


def test_resource_limit_takes_precedence_over_stderr():
    # even with a traceback present, an OOM kill is the dominant signal
    r = _err(exc="AttributeError", tb="AttributeError: ...", oom_killed=True)
    assert classify(r, SCHEMA) is ErrorClass.RESOURCE_LIMIT


def test_security_violation_is_hard_fail():
    r = _err(exc="SecurityViolation", tb="SecurityViolation: DTD present")
    cls = classify(r, SCHEMA)
    assert cls is ErrorClass.SECURITY_VIOLATION
    assert not cls.is_rag_eligible


def test_clean_but_empty_is_semantic_empty():
    r = {"ok": True, "result_shape": [0, 3], "artifacts": []}
    cls = classify(r, SCHEMA)
    assert cls is ErrorClass.SEMANTIC_EMPTY
    assert not cls.is_rag_eligible


def test_clean_with_rows_is_success_no_error_class():
    r = {"ok": True, "result_shape": [42, 3], "artifacts": ["chart.png"]}
    assert classify(r, SCHEMA) is None


def test_clean_with_artifact_only_is_success():
    # a chart with 0 result rows but a real artifact is still a success
    r = {"ok": True, "result_shape": [0, 0], "artifacts": ["chart.png"]}
    assert classify(r, SCHEMA) is None


def test_generic_runtime_error_retries_without_rag():
    r = _err(exc="ValueError", tb="ValueError: something went wrong")
    cls = classify(r, SCHEMA)
    assert cls is ErrorClass.RUNTIME_RECOVERABLE
    assert not cls.is_rag_eligible
