"""Build a finite, ordered plan from an intent + schema/profile.

A plan is a list of ``dict`` steps. Each step has:
    - ``kind``: stable identifier for the analysis operation (drives code_gen)
    - ``desc``: human-readable one-liner (for the report + progress)
    - ``cols``: the columns the step operates on (already schema-validated
      upstream in intent_understanding, so these exist)

Design invariants (see TECHNICAL_DESIGN.md self-heal + done-detection):
    - The plan is FINITE and generated once. No step generates more steps.
    - Open-ended mode is capped at MAX_OPEN_ENDED_STEPS, ranked by a cheap
      interestingness heuristic, then STOPS.
    - Planning never calls an LLM. Targeted mode is template instantiation;
      the model (later) only fills column names inside a fixed skeleton.
"""

from __future__ import annotations

from typing import Any, Dict, List

MAX_OPEN_ENDED_STEPS = 5


def _num_cols(schema: Dict[str, dict]) -> List[str]:
    return [c for c, m in schema.items()
            if m.get("dtype_hint") in ("integer", "float")]


def _cat_cols(schema: Dict[str, dict]) -> List[str]:
    return [c for c, m in schema.items()
            if m.get("dtype_hint") in ("string", "boolean")]


def _time_cols(schema: Dict[str, dict]) -> List[str]:
    return [c for c, m in schema.items() if m.get("dtype_hint") == "datetime"]


# ---- targeted templates: analysis_type -> ordered step skeleton ------------

def _plan_quality_audit(schema, profile, intent) -> List[Dict[str, Any]]:
    cols = list(schema)
    return [
        {"kind": "missing_values", "desc": "Count and rank missing values per column", "cols": cols},
        {"kind": "duplicates", "desc": "Count duplicate rows", "cols": cols},
        {"kind": "outliers", "desc": "Flag numeric outliers (IQR)", "cols": _num_cols(schema)},
        {"kind": "cleanliness_report", "desc": "Assemble a cleanliness report", "cols": cols},
    ]


def _plan_trend(schema, profile, intent) -> List[Dict[str, Any]]:
    time_col = intent.get("time_col") or (_time_cols(schema)[0] if _time_cols(schema) else "")
    metrics = [c for c in intent.get("target_cols", []) if c in _num_cols(schema)] or _num_cols(schema)[:1]
    return [
        {"kind": "resample_aggregate", "desc": f"Aggregate {metrics} over {time_col}", "cols": [time_col, *metrics]},
        {"kind": "trend_slope", "desc": "Fit a trend and classify direction", "cols": [time_col, *metrics]},
        {"kind": "trend_plot", "desc": "Plot the time series with trend", "cols": [time_col, *metrics]},
        {"kind": "trend_verdict", "desc": "Plain-English growing/shrinking/flat verdict", "cols": metrics},
    ]


def _plan_comparison(schema, profile, intent) -> List[Dict[str, Any]]:
    tgt = intent.get("target_cols", [])
    dims = [c for c in tgt if c in _cat_cols(schema)] or _cat_cols(schema)[:1]
    metrics = [c for c in tgt if c in _num_cols(schema)] or _num_cols(schema)[:1]
    return [
        {"kind": "group_aggregate", "desc": f"Aggregate {metrics} by {dims}", "cols": [*dims, *metrics]},
        {"kind": "bar_chart", "desc": f"Bar chart of {metrics} by {dims}", "cols": [*dims, *metrics]},
    ]


def _plan_segmentation(schema, profile, intent) -> List[Dict[str, Any]]:
    dims = [c for c in intent.get("target_cols", []) if c in _cat_cols(schema)] or _cat_cols(schema)[:1]
    metrics = _num_cols(schema)[:2]
    return [
        {"kind": "cohort_group", "desc": f"Group into cohorts by {dims}", "cols": [*dims, *metrics]},
        {"kind": "cohort_aggregate", "desc": "Aggregate metrics per cohort", "cols": [*dims, *metrics]},
        {"kind": "cohort_plot", "desc": "Grouped visualization of cohorts", "cols": [*dims, *metrics]},
    ]


def _plan_aggregation(schema, profile, intent) -> List[Dict[str, Any]]:
    return _plan_comparison(schema, profile, intent)


def _plan_ranking(schema, profile, intent) -> List[Dict[str, Any]]:
    tgt = intent.get("target_cols", [])
    metrics = [c for c in tgt if c in _num_cols(schema)] or _num_cols(schema)[:1]
    dims = [c for c in tgt if c in _cat_cols(schema)] or _cat_cols(schema)[:1]
    return [
        {"kind": "rank_top_n", "desc": f"Rank {dims} by {metrics}", "cols": [*dims, *metrics]},
        {"kind": "bar_chart", "desc": "Bar chart of the ranking", "cols": [*dims, *metrics]},
    ]


def _plan_distribution(schema, profile, intent) -> List[Dict[str, Any]]:
    metrics = [c for c in intent.get("target_cols", []) if c in _num_cols(schema)] or _num_cols(schema)[:1]
    return [
        {"kind": "describe", "desc": f"Summary statistics for {metrics}", "cols": metrics},
        {"kind": "histogram", "desc": f"Distribution plot for {metrics}", "cols": metrics},
    ]


def _plan_correlation(schema, profile, intent) -> List[Dict[str, Any]]:
    nums = _num_cols(schema)
    return [
        {"kind": "correlation_matrix", "desc": "Compute pairwise correlations", "cols": nums},
        {"kind": "heatmap", "desc": "Correlation heatmap", "cols": nums},
    ]


_TARGETED_TEMPLATES = {
    "quality_audit": _plan_quality_audit,
    "trend": _plan_trend,
    "comparison": _plan_comparison,
    "aggregation": _plan_aggregation,
    "segmentation": _plan_segmentation,
    "ranking": _plan_ranking,
    "distribution": _plan_distribution,
    "correlation": _plan_correlation,
}


# ---- open-ended: capped, ranked exploration battery ------------------------

def _open_ended_battery(schema, profile) -> List[Dict[str, Any]]:
    """Rank candidate explorations by a cheap interestingness heuristic, cap at N.

    Heuristic (higher = more interesting), all read off the profile/schema:
      - a datetime column present  -> a trend is usually worth surfacing
      - >=2 numeric columns        -> correlation
      - a low-cardinality category + a numeric -> comparison
      - any numeric                -> distribution/outliers
      - always                     -> a quality audit is cheap and informative
    """
    nums = _num_cols(schema)
    cats = _cat_cols(schema)
    times = _time_cols(schema)
    candidates: List[tuple] = []  # (score, step)

    if times and nums:
        candidates.append((90, {"kind": "resample_aggregate",
                                "desc": f"Trend of {nums[0]} over {times[0]}",
                                "cols": [times[0], nums[0]]}))
    if len(nums) >= 2:
        candidates.append((80, {"kind": "correlation_matrix",
                                "desc": "Strongest correlations among numeric columns",
                                "cols": nums}))
    low_card_cats = [c for c in cats
                     if 1 < schema[c].get("cardinality", 0) <= 20]
    if low_card_cats and nums:
        candidates.append((70, {"kind": "group_aggregate",
                                "desc": f"{nums[0]} by {low_card_cats[0]}",
                                "cols": [low_card_cats[0], nums[0]]}))
    if nums:
        candidates.append((60, {"kind": "outliers",
                                "desc": f"Outliers / spread in {nums[0]}",
                                "cols": [nums[0]]}))
    candidates.append((50, {"kind": "missing_values",
                            "desc": "Data quality: missing values and duplicates",
                            "cols": list(schema)}))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [step for _, step in candidates[:MAX_OPEN_ENDED_STEPS]]


def build_plan(intent: Dict[str, Any], schema: Dict[str, dict],
               profile: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Return a finite, ordered plan for the given intent.

    Targeted with a known analysis_type -> that template. Open-ended or an
    unrecognized/blank analysis_type -> the capped exploration battery. The
    result is always finite and never empty for a non-empty schema.
    """
    profile = profile or {}
    mode = intent.get("mode")
    analysis_type = intent.get("analysis_type", "")

    if mode == "targeted" and analysis_type in _TARGETED_TEMPLATES:
        plan = _TARGETED_TEMPLATES[analysis_type](schema, profile, intent)
        # Drop steps that ended up with no usable columns (e.g. no numeric col
        # for a distribution) so we never emit an un-runnable step.
        return [s for s in plan if s.get("cols")]

    # open-ended, or targeted-but-unclassified -> explore
    if schema:
        return _open_ended_battery(schema, profile)
    return []
