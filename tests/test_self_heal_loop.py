"""Self-heal loop tests — drive the bounded loop with hand-injected failures.

No LLM, no Docker: `codegen` and `executor` are scripted stubs, so every branch
of the escalation ladder and every terminal path is exercised deterministically.
"""

import pytest

from copilot.graph.loop import run_plan, run_step
from copilot.graph.router import escalation_level, route, wants_rag, ADVANCE, RETRY, DEGRADE
from copilot.graph.state import CoPilotState, ErrorClass


SCHEMA = {"region": {"dtype_hint": "string"}, "revenue": {"dtype_hint": "integer"}}


def _state(plan, **kw):
    s = CoPilotState(session_id="t", file_path="x.csv", user_question="q", schema=SCHEMA)
    s.plan = plan
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class StubExecutor:
    """Returns a scripted sequence of ExecutionResult-shaped dicts."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def run(self, code, data_path):
        self.calls += 1
        r = self._results.pop(0) if self._results else {"ok": True, "artifacts": ["a.png"]}
        return r


def _ok():
    return {"ok": True, "artifacts": ["chart.png"], "result_shape": [5, 2]}


def _api_misuse():
    return {"ok": False, "exception_type": "AttributeError",
            "traceback": "AttributeError: 'DataFrame' object has no attribute 'append'"}


def _missing_col():
    return {"ok": False, "exception_type": "KeyError",
            "traceback": "KeyError: 'profit'"}


# --- router unit behaviour --------------------------------------------------

def test_success_advances():
    s = _state([{"kind": "x"}])
    s.error_class = None
    assert route(s) == ADVANCE


def test_data_problem_degrades_never_retries():
    s = _state([{"kind": "x"}])
    s.error_class = ErrorClass.DATA_PROBLEM
    assert route(s) == DEGRADE


def test_api_misuse_retries_within_budget():
    s = _state([{"kind": "x"}])
    s.error_class = ErrorClass.API_MISUSE
    s.attempt_count = 0
    assert route(s) == RETRY


def test_api_misuse_degrades_at_per_step_cap():
    s = _state([{"kind": "x"}])
    s.error_class = ErrorClass.API_MISUSE
    s.attempt_count = s.MAX_ATTEMPTS_PER_STEP
    assert route(s) == DEGRADE


def test_global_budget_forces_degrade():
    s = _state([{"kind": "x"}])
    s.error_class = ErrorClass.SYNTAX
    s.attempt_count = 0
    s.global_attempts = s.MAX_GLOBAL_ATTEMPTS
    assert route(s) == DEGRADE


def test_escalation_ladder():
    assert escalation_level(0) == "plain"
    assert escalation_level(1) == "with_history"
    assert escalation_level(2) == "with_rag_or_fallback"


def test_wants_rag_only_api_misuse_at_top_of_ladder():
    s = _state([{"kind": "x"}])
    s.error_class = ErrorClass.API_MISUSE
    s.attempt_count = 2
    assert wants_rag(s) is True
    # not at top of ladder
    s.attempt_count = 1
    assert wants_rag(s) is False
    # top of ladder but not API_MISUSE
    s.attempt_count = 2
    s.error_class = ErrorClass.SYNTAX
    assert wants_rag(s) is False


# --- loop driver behaviour --------------------------------------------------

def test_first_try_success():
    s = _state([{"kind": "comparison", "desc": "rev by region"}])
    ex = StubExecutor([_ok()])
    run_plan(s, codegen=lambda step, st, lvl: "code", executor=ex)
    assert s.step_outcomes == ["ok"]
    assert "chart.png" in s.artifacts
    assert s.degradation_notes == []
    assert ex.calls == 1


def test_heals_on_second_attempt():
    # fail once (api_misuse, retryable), then succeed
    s = _state([{"kind": "trend", "desc": "growth"}])
    ex = StubExecutor([_api_misuse(), _ok()])
    run_plan(s, codegen=lambda step, st, lvl: "code", executor=ex)
    assert s.step_outcomes == ["ok"]
    assert ex.calls == 2
    assert s.degradation_notes == []


def test_degrades_after_per_step_budget_exhausted():
    # always api_misuse -> retries to the per-step cap, then degrades
    s = _state([{"kind": "trend", "desc": "growth"}])
    ex = StubExecutor([_api_misuse()] * 10)
    run_plan(s, codegen=lambda step, st, lvl: "code", executor=ex)
    assert s.step_outcomes == ["degraded"]
    assert ex.calls == s.MAX_ATTEMPTS_PER_STEP        # capped, not infinite
    assert len(s.degradation_notes) == 1
    assert "growth" in s.degradation_notes[0]


def test_missing_column_degrades_immediately_no_retry():
    s = _state([{"kind": "comparison", "desc": "profit by region"}])
    ex = StubExecutor([_missing_col()])
    run_plan(s, codegen=lambda step, st, lvl: "code", executor=ex)
    assert s.step_outcomes == ["degraded"]
    assert ex.calls == 1                              # DATA_PROBLEM never retries
    assert "profit by region" in s.degradation_notes[0]


def test_partial_report_mixed_outcomes():
    # step1 ok, step2 unfixable data problem, step3 ok -> partial but complete
    plan = [
        {"kind": "a", "desc": "first"},
        {"kind": "b", "desc": "second"},
        {"kind": "c", "desc": "third"},
    ]
    s = _state(plan)
    ex = StubExecutor([_ok(), _missing_col(), _ok()])
    run_plan(s, codegen=lambda step, st, lvl: "code", executor=ex)
    assert s.step_outcomes == ["ok", "degraded", "ok"]
    assert len(s.degradation_notes) == 1
    assert s.is_done()


def test_global_budget_caps_total_attempts():
    # many steps each always failing -> global cap stops the whole run
    plan = [{"kind": f"s{i}", "desc": f"step {i}"} for i in range(10)]
    s = _state(plan)
    ex = StubExecutor([_api_misuse()] * 100)
    run_plan(s, codegen=lambda step, st, lvl: "code", executor=ex)
    # total executions never exceed the global budget
    assert ex.calls <= s.MAX_GLOBAL_ATTEMPTS
    # every step still accounted for (ok or degraded), none silently dropped
    assert len(s.degradation_notes) == len(plan)
    assert s.is_done()


def test_escalation_level_seen_by_codegen():
    # verify the codegen callable receives escalating levels across retries
    seen = []
    s = _state([{"kind": "trend", "desc": "g"}])
    ex = StubExecutor([_api_misuse(), _api_misuse(), _ok()])

    def codegen(step, st, level):
        seen.append(level)
        return "code"

    run_plan(s, codegen=codegen, executor=ex)
    assert seen == ["plain", "with_history", "with_rag_or_fallback"]
