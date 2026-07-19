"""Deterministic intent understanding.

Maps a plain-English question to an intent dict against the LIVE schema, so
column mistakes are caught HERE (before any code-gen) rather than discovered at
execution time. The whole thing is rule-based and testable without a model; a
later phase can swap the analysis-type classifier for an LLM while keeping this
validation gate in front of it.

Design points (TECHNICAL_DESIGN.md):
* Vague/unmapped question -> ``Mode.OPEN_ENDED`` (planner runs the capped
  exploration battery), never a guess at a specific analysis.
* A column named in the question that isn't in the schema is flagged as a
  DATA_PROBLEM (``missing_cols`` + ``needs_clarification``) with near-miss
  suggestions -- NOT silently substituted (that would answer a different
  question) and NOT raised (a hard raise would crash the graph on a recoverable
  problem; the report node surfaces it instead).
* No schema yet -> we can't ground columns, so route open-ended rather than
  invent missing-column errors.

Returns a plain dict so the node can drop it straight onto ``state.intent``.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Dict, List

from copilot.graph.state import Mode

# analysis_type -> trigger keywords. Order = priority (first match wins), so the
# more specific intents (audit, segmentation) come before generic ones.
_ANALYSIS_KEYWORDS = [
    ("quality_audit", ["audit", "missing value", "duplicat", "data quality", "clean the", "nulls"]),
    ("segmentation", ["segment", "cluster", "personas", "group into", "group the"]),
    ("correlation", ["correlat", "relationship", "related", "associat", "depend on"]),
    ("trend", ["trend", "over time", "growth", "growing", "trajectory", "time series", "seasonal"]),
    ("ranking", ["top ", "bottom ", "highest", "lowest", "rank", "most ", "least "]),
    ("distribution", ["distribut", "spread", "histogram", "outlier", "range of"]),
    ("aggregation", ["total", "sum of", "average", "mean ", "median", "count of", " per "]),
    ("comparison", ["compare", "versus", " vs ", "difference between", " by "]),
]

# Verbs that introduce a metric column: "<verb> <column>".
_METRIC_VERBS = r"(?:show|plot|chart|graph|display|visuali[sz]e|compute|calculate|find|get|of)"
# Prepositions that introduce a dimension column: "<prep> <column>".
_DIM_PREPS = r"(?:by|per|across)"

_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "data", "dataset",
    "values", "value", "each", "all", "any", "there", "me", "us", "it",
    "interesting", "anything", "something", "insights", "insight",
}


def _classify(q_lc: str) -> str:
    for analysis_type, keywords in _ANALYSIS_KEYWORDS:
        if any(kw in q_lc for kw in keywords):
            return analysis_type
    return ""


def _candidate_columns(question: str) -> List[str]:
    """Extract words that occupy a column slot (metric or dimension) in the text.

    Positional, not exhaustive: we only pull words the grammar clearly marks as
    column references, to avoid falsely flagging ordinary nouns as missing
    columns.
    """
    q = question.lower()
    cands: List[str] = []
    for pat in (
        _METRIC_VERBS + r"\s+(?:the\s+)?([a-z_]+)",
        _DIM_PREPS + r"\s+(?:the\s+)?([a-z_]+)",
    ):
        for m in re.findall(pat, q):
            if m not in _STOPWORDS and m not in cands:
                cands.append(m)
    return cands


def _quoted_refs(question: str) -> List[str]:
    return [q.strip() for q in re.findall(r"['\"]([^'\"]+)['\"]", question)]


def understand_intent(question: str, schema: Dict[str, dict]) -> Dict[str, object]:
    """Build the intent dict from ``question`` against ``schema``.

    Keys: mode, analysis_type, target_cols, missing_cols, time_col,
    suggestions, needs_clarification.
    """
    question = (question or "").strip()
    schema_cols = list(schema.keys())
    by_lc = {c.lower(): c for c in schema_cols}

    base = {
        "mode": Mode.OPEN_ENDED,
        "analysis_type": "",
        "target_cols": [],
        "missing_cols": [],
        "time_col": "",
        "suggestions": [],
        "needs_clarification": False,
    }

    # No question, or no schema to ground against -> explore.
    if not question or not schema_cols:
        base["analysis_type"] = _classify(question.lower()) if question else ""
        return base

    q_lc = question.lower()
    analysis_type = _classify(q_lc)

    target_cols: List[str] = []
    missing_cols: List[str] = []

    # 1. Whole schema column names appearing verbatim (handles multi-word headers).
    for lc, original in by_lc.items():
        if lc and lc in q_lc:
            target_cols.append(original)

    # 2. Explicit quoted references must resolve or they're flagged missing.
    # 3. Grammatical column-slot candidates likewise.
    for ref in _quoted_refs(question) + _candidate_columns(question):
        lc = ref.lower()
        if lc in by_lc:
            if by_lc[lc] not in target_cols:
                target_cols.append(by_lc[lc])
        elif ref not in missing_cols:
            missing_cols.append(ref)

    # Time axis: for a trend, surface the first datetime column as the axis.
    time_col = ""
    if analysis_type == "trend":
        for c in schema_cols:
            if schema[c].get("dtype_hint") == "datetime":
                time_col = c
                if c not in target_cols:
                    target_cols.append(c)
                break

    suggestions: List[str] = []
    if missing_cols:
        for name in missing_cols:
            suggestions += get_close_matches(name.lower(), list(by_lc), n=3, cutoff=0.4)
        if not suggestions:                       # nothing close -> list what's available
            suggestions = list(schema_cols)
        # de-dupe, map back to original casing
        seen = []
        for s in suggestions:
            orig = by_lc.get(s, s)
            if orig not in seen:
                seen.append(orig)
        suggestions = seen

    # Vague: nothing recognizable at all -> explore.
    if not analysis_type and not target_cols and not missing_cols:
        return base

    return {
        "mode": Mode.TARGETED,
        "analysis_type": analysis_type,
        "target_cols": target_cols,
        "missing_cols": missing_cols,
        "time_col": time_col,
        "suggestions": suggestions,
        "needs_clarification": bool(missing_cols),
    }
