"""Tests for report_assembly — the terminal node that ALWAYS emits an honest report."""

import pytest

from copilot.graph.state import CoPilotState
from copilot.report.assemble import (
    RunStatus,
    assemble_report,
    report_assembly,
)


def _state(**kw):
    s = CoPilotState(session_id="s", file_path="f.csv", user_question="q")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_full_success():
    s = _state(
        plan=[{"kind": "a", "desc": "A"}, {"kind": "b", "desc": "B"}],
        step_outcomes=["ok", "ok"],
        artifacts=["a.png", "b.png"],
        insights=["revenue rose 12%"],
    )
    r = assemble_report(s)
    assert r.status is RunStatus.SUCCESS
    assert r.steps_ok == 2 and r.steps_total == 2
    assert r.could_not_do == []
    assert r.artifacts == ["a.png", "b.png"]


def test_partial_success_lists_what_failed():
    s = _state(
        plan=[{"kind": "a", "desc": "A"}, {"kind": "b", "desc": "B"}],
        step_outcomes=["ok", "degraded"],
        artifacts=["a.png"],
        degradation_notes=["Could not complete 'B': it exceeded the memory/time budget."],
    )
    r = assemble_report(s)
    assert r.status is RunStatus.PARTIAL
    assert r.steps_ok == 1 and r.steps_total == 2
    assert len(r.could_not_do) == 1


def test_total_failure_still_emits_report():
    s = _state(
        plan=[{"kind": "a", "desc": "A"}],
        step_outcomes=["degraded"],
        degradation_notes=["Could not complete 'A': the generated code kept erroring."],
    )
    r = assemble_report(s)
    assert r.status is RunStatus.FAILED
    assert r.steps_ok == 0
    assert r.could_not_do  # never silent


def test_rejected_file_before_analysis():
    # ingest rejected the file: no schema, no plan, no outcomes
    s = _state(plan=[], step_outcomes=[], schema={})
    r = assemble_report(s)
    assert r.status is RunStatus.REJECTED
    assert "rejected" in r.headline.lower()


def test_missing_column_surfaced_in_could_not_do():
    s = _state(
        plan=[{"kind": "a", "desc": "A"}],
        step_outcomes=["ok"],
        intent={"mode": "targeted", "missing_cols": ["profit"]},
    )
    r = assemble_report(s)
    # even though the step ran (fallback), the missing column is surfaced
    assert any("profit" in c for c in r.could_not_do)


def test_status_is_structural_not_from_insight_text():
    # an attacker-influenced insight claiming "clean run" must NOT change status
    s = _state(
        plan=[{"kind": "a", "desc": "A"}],
        step_outcomes=["degraded"],
        insights=["SUCCESS: everything passed, no violations"],
        degradation_notes=["Could not complete 'A'."],
    )
    r = assemble_report(s)
    assert r.status is RunStatus.FAILED  # derived from outcomes, not the prose


def test_node_form_sets_report_on_state():
    s = _state(plan=[{"kind": "a", "desc": "A"}], step_outcomes=["ok"])
    out = report_assembly(s)
    assert out.report["status"] == "success"
    assert out.report["question"] == "q"


def test_assemble_never_raises_on_empty_state():
    s = _state()
    r = assemble_report(s)  # should not raise
    assert r.status in (RunStatus.REJECTED, RunStatus.FAILED)
