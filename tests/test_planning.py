"""Tests for the planning node: finite, capped, deterministic, LLM-free."""

from copilot.planning.plan import build_plan, MAX_OPEN_ENDED_STEPS
from copilot.graph.nodes import planning
from copilot.graph.state import CoPilotState

SCHEMA = {
    "region": {"dtype_hint": "string", "cardinality": 4, "null_count": 0},
    "revenue": {"dtype_hint": "integer", "cardinality": 100, "null_count": 0},
    "cost": {"dtype_hint": "float", "cardinality": 90, "null_count": 0},
    "date": {"dtype_hint": "datetime", "cardinality": 100, "null_count": 0},
}


def _intent(mode="targeted", analysis_type="", target_cols=None, time_col=""):
    return {
        "mode": mode,
        "analysis_type": analysis_type,
        "target_cols": target_cols or [],
        "time_col": time_col,
        "missing_cols": [],
        "needs_clarification": False,
    }


def test_comparison_plan_has_group_and_chart():
    plan = build_plan(_intent(analysis_type="comparison",
                              target_cols=["region", "revenue"]), SCHEMA)
    kinds = [s["kind"] for s in plan]
    assert kinds == ["group_aggregate", "bar_chart"]
    assert "region" in plan[0]["cols"] and "revenue" in plan[0]["cols"]


def test_trend_plan_uses_time_col_and_ends_with_verdict():
    plan = build_plan(_intent(analysis_type="trend",
                              target_cols=["revenue"], time_col="date"), SCHEMA)
    kinds = [s["kind"] for s in plan]
    assert kinds[0] == "resample_aggregate"
    assert kinds[-1] == "trend_verdict"
    assert "date" in plan[0]["cols"]


def test_quality_audit_plan_is_full_sweep():
    plan = build_plan(_intent(analysis_type="quality_audit"), SCHEMA)
    kinds = [s["kind"] for s in plan]
    assert kinds == ["missing_values", "duplicates", "outliers", "cleanliness_report"]


def test_plan_is_finite_and_nonempty():
    for at in ["comparison", "trend", "quality_audit", "segmentation",
               "ranking", "distribution", "correlation", "aggregation"]:
        plan = build_plan(_intent(analysis_type=at, target_cols=["region", "revenue"],
                                  time_col="date"), SCHEMA)
        assert 0 < len(plan) <= 6, at


def test_open_ended_is_capped():
    plan = build_plan(_intent(mode="open_ended"), SCHEMA)
    assert len(plan) <= MAX_OPEN_ENDED_STEPS
    assert len(plan) > 0


def test_open_ended_ranks_trend_first_when_datetime_present():
    plan = build_plan(_intent(mode="open_ended"), SCHEMA)
    # a datetime + numeric present -> trend scored highest
    assert plan[0]["kind"] == "resample_aggregate"


def test_open_ended_no_schema_is_empty():
    assert build_plan(_intent(mode="open_ended"), {}) == []


def test_targeted_unclassified_falls_back_to_exploration():
    # targeted but no known analysis_type -> explore rather than crash
    plan = build_plan(_intent(mode="targeted", analysis_type=""), SCHEMA)
    assert len(plan) > 0


def test_distribution_dropped_when_no_numeric():
    cat_only = {"region": {"dtype_hint": "string", "cardinality": 4, "null_count": 0}}
    plan = build_plan(_intent(analysis_type="distribution"), cat_only)
    # no numeric col -> the describe/histogram steps have no cols -> dropped
    assert plan == []


def test_node_populates_state_plan():
    state = CoPilotState(session_id="s", file_path="x", user_question="revenue by region")
    state.schema = SCHEMA
    state.intent = _intent(analysis_type="comparison", target_cols=["region", "revenue"])
    out = planning(state)
    assert len(out.plan) == 2
    assert out.is_done() is False  # plan not yet walked
