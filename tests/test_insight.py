"""Tests for grounded insight synthesis (copilot.insight.synthesize).

The load-bearing property: an insight may only contain text the executed code
actually printed (values pandas computed from real data). We test that grounding
holds, that the optional LLM polish never introduces new numbers, and that
successful-but-silent steps degrade to an honest, number-free line.
"""

from copilot.insight.synthesize import (
    ground_insight,
    synthesize_insights,
    polish_with_llm,
    _printed_lines,
)
from copilot.sandbox.execute import RESULT_SENTINEL


def test_printed_lines_excludes_result_sentinel():
    stdout = f"North leads at 4210\n{RESULT_SENTINEL}{{\"ok\": true}}\n"
    lines = _printed_lines(stdout)
    assert lines == ["North leads at 4210"]


def test_ground_insight_uses_printed_summary():
    res = {"ok": True, "stdout": "North leads with revenue 4210; South lowest at 900\n"}
    step = {"kind": "group_aggregate", "desc": "revenue by region"}
    ins = ground_insight(res, step)
    assert ins is not None
    assert "4210" in ins and "revenue by region" in ins


def test_ground_insight_none_when_failed():
    res = {"ok": False, "stdout": "whatever"}
    assert ground_insight(res, {"desc": "x"}) is None


def test_ground_insight_chart_only_is_number_free():
    res = {"ok": True, "stdout": "", "artifacts": ["chart.png"]}
    ins = ground_insight(res, {"desc": "bar chart of revenue"})
    assert ins is not None
    assert "chart" in ins.lower()
    # no fabricated numbers
    assert not any(ch.isdigit() for ch in ins)


def test_ground_insight_none_when_nothing_printed_no_artifact():
    res = {"ok": True, "stdout": "", "artifacts": []}
    assert ground_insight(res, {"desc": "x"}) is None


def test_synthesize_insights_across_steps():
    steps = [
        {"desc": "revenue by region"},
        {"desc": "trend"},
    ]
    results = [
        {"ok": True, "stdout": "North 4210 highest\n"},
        {"ok": True, "stdout": "revenue is growing (slope +12/mo)\n"},
    ]
    out = synthesize_insights(steps, results)
    assert len(out) == 2
    assert "4210" in out[0]
    assert "growing" in out[1]


def test_polish_falls_back_without_client():
    class Down:
        def is_available(self):
            return False
    grounded = ["revenue by region: North 4210 highest"]
    # model unavailable -> grounded lines returned verbatim
    assert polish_with_llm(grounded, "which region leads?", client=Down()) == grounded


def test_polish_rejects_invented_numbers():
    class Liar:
        def is_available(self):
            return True
        def generate(self, prompt, system=""):
            return "North leads with 9999999 revenue."  # number not in grounded facts
    grounded = ["revenue by region: North 4210 highest"]
    # the guard detects 9999999 is not in the grounded facts -> fall back verbatim
    assert polish_with_llm(grounded, "q", client=Liar()) == grounded
