"""Tests for intent_understanding — deterministic column-validation half.

These prove the design invariant: a question referencing a nonexistent column is
flagged NOW (before any code-gen), and a vague question routes to open_ended
rather than fabricating a targeted plan.
"""

import pytest

from copilot.intent.understand import understand_intent
from copilot.graph.state import Mode


SCHEMA = {
    "region": {"dtype_hint": "string", "cardinality": 4, "null_count": 0},
    "revenue": {"dtype_hint": "integer", "cardinality": 120, "null_count": 0},
    "order_date": {"dtype_hint": "datetime", "cardinality": 90, "null_count": 0},
}


def test_targeted_question_resolves_columns():
    intent = understand_intent("show revenue by region", SCHEMA)
    assert intent["mode"] == Mode.TARGETED
    assert "revenue" in intent["target_cols"]
    assert "region" in intent["target_cols"]
    assert intent["missing_cols"] == []


def test_trend_question_classified():
    intent = understand_intent("is revenue growing over time?", SCHEMA)
    assert intent["mode"] == Mode.TARGETED
    assert intent["analysis_type"] == "trend"
    # a datetime column is available for the time axis
    assert "order_date" in intent["target_cols"] or intent["time_col"] == "order_date"


def test_missing_column_flagged_before_codegen():
    # 'profit' does not exist -> must be flagged as a data/intent problem NOW
    intent = understand_intent("show profit by region", SCHEMA)
    assert "profit" in intent["missing_cols"]
    assert intent["needs_clarification"] is True
    # closest existing columns surfaced to help the user
    assert intent["suggestions"]  # non-empty


def test_vague_question_routes_open_ended():
    intent = understand_intent("what's interesting in this data?", SCHEMA)
    assert intent["mode"] == Mode.OPEN_ENDED
    assert intent["needs_clarification"] is False


def test_empty_question_routes_open_ended():
    intent = understand_intent("", SCHEMA)
    assert intent["mode"] == Mode.OPEN_ENDED


def test_quality_audit_classified():
    intent = understand_intent("audit this data for missing values and duplicates", SCHEMA)
    assert intent["analysis_type"] == "quality_audit"
    # a quality audit needs no specific column -> not a missing-col failure
    assert intent["needs_clarification"] is False


def test_segmentation_classified():
    intent = understand_intent("segment customers by region", SCHEMA)
    assert intent["analysis_type"] == "segmentation"
    assert "region" in intent["target_cols"]


def test_missing_column_does_not_crash_on_empty_schema():
    intent = understand_intent("show revenue by region", {})
    # no schema -> cannot resolve, but must not raise; routes to open_ended
    assert intent["mode"] == Mode.OPEN_ENDED
