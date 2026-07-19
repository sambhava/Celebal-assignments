"""Tests for the pluggable executor (copilot.sandbox.execute).

The DevExecutor is exercised here (Docker is not assumed present in CI). It is
NOT a security boundary — these tests only confirm the result-shape contract
and the env opt-in guard, so the classifier/loop can run backend-agnostically.
"""

import os

import pytest

from copilot.sandbox.execute import (
    DevExecutor,
    ExecutionResult,
    EXIT_OOM,
    EXIT_TIMEOUT,
    _parse_exception_type,
    get_executor,
)


def _dev():
    # 60s: the dev executor runs through the entrypoint, which cold-imports
    # pandas in a subprocess; on a small machine (shared with Ollama) that can
    # exceed a tight budget under full-suite load. This is a test-harness margin,
    # not a production timeout (production uses the container's own wall-clock).
    ex = DevExecutor(timeout_s=60)
    return ex


@pytest.fixture()
def data_csv(tmp_path):
    """A tiny CSV the entrypoint can load into the pre-provided ``df``."""
    p = tmp_path / "data.csv"
    p.write_text("region,revenue\nnorth,10\nsouth,20\n", encoding="utf-8")
    return str(p)


def test_dev_executor_refuses_without_optin(monkeypatch):
    monkeypatch.delenv("COPILOT_DEV_UNSAFE", raising=False)
    with pytest.raises(RuntimeError, match="not a security boundary"):
        _dev().run("result = df", data_csv)


def test_dev_executor_clean_run(monkeypatch, data_csv):
    # Code runs through the trusted entrypoint: df/pd are pre-provided, and the
    # RESULT line carries result_shape back for the classifier.
    monkeypatch.setenv("COPILOT_DEV_UNSAFE", "1")
    r = _dev().run("result = df.groupby('region')['revenue'].sum()\nprint('done')", data_csv)
    assert r.ok is True
    assert "done" in r.stdout
    assert r.exception_type == ""
    assert r.result_shape is not None


def test_dev_executor_captures_exception_type(monkeypatch, data_csv):
    monkeypatch.setenv("COPILOT_DEV_UNSAFE", "1")
    r = _dev().run("raise KeyError('region')", data_csv)
    assert r.ok is False
    assert r.exception_type == "KeyError"
    assert "KeyError" in r.traceback


def test_dev_executor_timeout(monkeypatch, data_csv):
    monkeypatch.setenv("COPILOT_DEV_UNSAFE", "1")
    r = DevExecutor(timeout_s=1).run("import time\ntime.sleep(5)", data_csv)
    assert r.ok is False
    assert r.timed_out is True
    assert r.exit_code == EXIT_TIMEOUT


def test_parse_exception_type_dotted():
    tb = "Traceback...\npandas.errors.EmptyDataError: No columns to parse"
    assert _parse_exception_type(tb) == "EmptyDataError"


def test_parse_exception_type_bare():
    assert _parse_exception_type("Traceback...\nSyntaxError: bad") == "SyntaxError"


def test_execution_result_to_dict_is_stable():
    r = ExecutionResult(ok=False, exception_type="TypeError", exit_code=1)
    d = r.to_dict()
    assert d["ok"] is False and d["exception_type"] == "TypeError"


def test_get_executor_returns_dev_without_docker(monkeypatch):
    # Force shutil.which to report no docker.
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert isinstance(get_executor(), DevExecutor)
