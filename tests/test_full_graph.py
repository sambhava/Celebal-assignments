"""Full-graph integration tests (copilot.graph.run.run_full).

These drive the ENTIRE pipeline -- ingest, profile, intent, planning, the
self-heal loop, insight synthesis, and report assembly -- with a scripted
code-gen and a scripted executor, so no LLM and no Docker are required. They
prove the nodes compose into one always-terminating run that produces an honest
report on success, partial success, and rejection paths.
"""

import csv
import io
import os
import tempfile

import pytest

from copilot.graph.run import run_full
from copilot.report.assemble import RunStatus


def _write_csv(rows, header=("region", "revenue", "date")):
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


class _ScriptedExecutor:
    """Executor stub returning a canned ExecutionResult-like dict per call.

    ``behavior`` maps step-kind substrings to a callable(step)->result dict, so a
    test can make specific steps succeed, error, or return empty.
    """

    def __init__(self, default_ok=True):
        self.default_ok = default_ok
        self.calls = []

    def run(self, code, data_path="", data_type="csv"):
        self.calls.append(code)
        from copilot.sandbox.execute import ExecutionResult
        return ExecutionResult(
            ok=True,
            stdout="North leads with revenue 4210; South lowest at 900\n",
            result_shape=[2, 2],
            empty=False,
            artifacts=[],
        )


def _stub_codegen(step, state, level):
    return "result = df.groupby('region')['revenue'].sum()\nprint('ok')"


def test_full_graph_targeted_success():
    path = _write_csv([("North", 4210, "2024-01-01"), ("South", 900, "2024-02-01")])
    try:
        state = run_full(
            path, "show revenue by region",
            codegen=_stub_codegen, executor=_ScriptedExecutor(),
        )
    finally:
        os.unlink(path)

    assert state.report is not None
    assert state.report["status"] in (RunStatus.SUCCESS.value, RunStatus.PARTIAL.value)
    # grounded insight came from the executor's printed line, not invented
    assert any("4210" in ins for ins in state.report["insights"])
    assert state.report["steps_total"] >= 1


def test_full_graph_rejects_hostile_file_with_report():
    # ragged CSV -> R11 guard rejects at ingest; must still yield a REJECTED report
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("a,b,c\n=cmd|calc,1,2,3\n")  # 4 fields for a 3-col header
    try:
        state = run_full(
            path, "show a by b",
            codegen=_stub_codegen, executor=_ScriptedExecutor(),
        )
    finally:
        os.unlink(path)

    assert state.report is not None
    assert state.report["status"] == RunStatus.REJECTED.value
    assert state.report["could_not_do"]  # honest "what I couldn't do"


def test_full_graph_missing_column_is_disclosed():
    path = _write_csv([("North", 4210, "2024-01-01")])
    try:
        state = run_full(
            path, "show profit by region",  # 'profit' does not exist
            codegen=_stub_codegen, executor=_ScriptedExecutor(),
        )
    finally:
        os.unlink(path)

    # the missing column must be surfaced in the report, not silently ignored
    joined = " ".join(state.report["could_not_do"])
    assert "profit" in joined


def test_full_graph_always_produces_report_even_on_all_failures():
    path = _write_csv([("North", 4210, "2024-01-01")])

    class _AlwaysErrors:
        def run(self, code, data_path="", data_type="csv"):
            from copilot.sandbox.execute import ExecutionResult
            return ExecutionResult(
                ok=False, exception_type="ValueError",
                traceback="ValueError: boom", stderr="ValueError: boom",
            )

    try:
        state = run_full(
            path, "show revenue by region",
            codegen=_stub_codegen, executor=_AlwaysErrors(),
        )
    finally:
        os.unlink(path)

    # every step degraded, but a report still exists and is honest
    assert state.report is not None
    assert state.report["status"] in (RunStatus.FAILED.value, RunStatus.PARTIAL.value)
    assert state.report["could_not_do"]
    # bounded: never exceeded the global attempt budget
    assert state.global_attempts <= state.MAX_GLOBAL_ATTEMPTS
