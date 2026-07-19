"""Pluggable code executor for the sandboxed_execution node.

Two backends behind one interface:

* ``DockerExecutor`` — the production path. Runs generated code in the hardened
  container (``build_sandbox_cmd``, see SECURITY.md §3 / R1): ``--network none``,
  aggregate cgroup caps, dropped caps, read-only rootfs, no host env. This is
  the actual security boundary.

* ``DevExecutor`` — a local subprocess runner for development on machines
  WITHOUT Docker. It is **NOT a security boundary** (same user, same host, same
  network) and refuses to run unless COPILOT_DEV_UNSAFE=1 is set. It exists only
  so the self-heal loop and classifier can be exercised end-to-end while the
  container image is unavailable.

Both return the same ``ExecutionResult`` shape so ``error_classification`` and
the retry loop are backend-agnostic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --- distinguished exit codes (match the SECURITY.md resource-limit contract) --
EXIT_TIMEOUT = 124        # wall-clock kill (GNU timeout convention)
EXIT_OOM = 137            # 128 + SIGKILL: cgroup OOM / docker OOMKilled


@dataclass
class ExecutionResult:
    """Uniform result of one execution attempt, for the classifier to inspect."""
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False
    oom_killed: bool = False
    artifacts: List[str] = field(default_factory=list)
    # error surface for the classifier (set when ok is False)
    exception_type: str = ""      # e.g. "KeyError", "AttributeError" (parsed from stderr)
    traceback: str = ""
    # structured result reported by the entrypoint RESULT protocol
    result_shape: Optional[list] = None
    empty: Optional[bool] = None
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "oom_killed": self.oom_killed,
            "artifacts": list(self.artifacts),
            "exception_type": self.exception_type,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result_shape": self.result_shape,
            "empty": self.empty,
            "summary": self.summary,
        }


def _parse_exception_type(stderr: str) -> str:
    """Pull the exception class name off the last traceback line.

    A Python traceback ends with ``<ExcType>: <msg>`` (or bare ``<ExcType>``).
    We return the class name for the classifier to route on.
    """
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        # last non-empty line is the exception line
        head = line.split(":", 1)[0].strip()
        # a bare identifier or dotted name like "pandas.errors.EmptyDataError"
        token = head.split()[-1] if head else ""
        if token and (token.isidentifier() or all(p.isidentifier() for p in token.split("."))):
            return token.split(".")[-1]
        return ""
    return ""


# Sentinel the trusted entrypoint prefixes to its single JSON result line.
# Must match copilot.sandbox.entrypoint.RESULT_SENTINEL.
RESULT_SENTINEL = "@@COPILOT_RESULT@@"


def _parse_result_line(stdout: str) -> Optional[Dict[str, Any]]:
    """Extract the entrypoint's structured RESULT payload from stdout, if any.

    The entrypoint emits exactly one ``<sentinel>{json}`` line. We scan from the
    end so a payload the model happened to print earlier can't shadow the real
    one. Returns None when no valid payload is found (e.g. the process was killed
    before it could emit, or ran code that never went through the entrypoint).
    """
    for line in reversed(stdout.splitlines()):
        idx = line.find(RESULT_SENTINEL)
        if idx == -1:
            continue
        blob = line[idx + len(RESULT_SENTINEL):].strip()
        try:
            payload = json.loads(blob)
        except (ValueError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _enrich_from_result(result: "ExecutionResult") -> "ExecutionResult":
    """Fold the entrypoint RESULT payload into an ExecutionResult.

    The RESULT line is authoritative over the raw exit code for *classification*
    fields: the entrypoint reports the structured exception type, artifacts, and
    result shape/emptiness the classifier needs. Resource kills (timeout/OOM) are
    detected out-of-band and are left untouched here.
    """
    payload = _parse_result_line(result.stdout)
    if payload is None:
        return result
    if payload.get("ok") is False:
        result.ok = False
        if payload.get("exception_type"):
            result.exception_type = payload["exception_type"]
        if payload.get("traceback"):
            result.traceback = payload["traceback"]
        elif payload.get("error"):
            result.traceback = result.traceback or str(payload["error"])
    elif payload.get("ok") is True:
        result.ok = True
        result.exception_type = ""
    result.artifacts = list(payload.get("artifacts") or result.artifacts)
    result.result_shape = payload.get("result_shape")
    result.empty = bool(payload.get("empty", False))
    return result


class DevExecutor:
    """Local subprocess runner — NOT a sandbox. Guarded by an env opt-in."""

    def __init__(self, timeout_s: int = 30):
        self.timeout_s = timeout_s

    def run(self, code: str, data_path: str = "", data_type: str = "csv") -> ExecutionResult:
        if os.environ.get("COPILOT_DEV_UNSAFE") != "1":
            raise RuntimeError(
                "DevExecutor is not a security boundary and is disabled. "
                "Set COPILOT_DEV_UNSAFE=1 to run generated code locally without "
                "Docker (development only — never in production)."
            )
        # Run the model code through the SAME trusted entrypoint the container
        # uses (code on stdin), so dev and prod share one execution contract:
        # df/pd/plt/save_artifact are pre-provided, and the RESULT line is emitted.
        artifact_dir = tempfile.mkdtemp(prefix="copilot_artifacts_")
        env = {
            "COPILOT_DATA_PATH": data_path,
            "COPILOT_DATA_TYPE": data_type,
            "COPILOT_ARTIFACT_DIR": artifact_dir,
            "PATH": os.environ.get("PATH", ""),
            # matplotlib reads MPLCONFIGDIR for its config/cache dir; without it,
            # import falls back to Path.home()/".matplotlib", which raises
            # RuntimeError when the host env (HOME/USERPROFILE) is stripped for
            # isolation. Point it at the disposable artifact dir so charting
            # works without leaking the host home into the sandbox.
            "MPLCONFIGDIR": artifact_dir,
        }
        # keep PYTHONPATH so the child can import copilot.* (dev machine only)
        if os.environ.get("PYTHONPATH"):
            env["PYTHONPATH"] = os.environ["PYTHONPATH"]
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "copilot.sandbox.entrypoint"],
                input=code,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            return ExecutionResult(
                ok=False,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + "\n[dev-executor] wall-clock timeout",
                exit_code=EXIT_TIMEOUT,
                timed_out=True,
            )

        ok = proc.returncode == 0
        return _enrich_from_result(ExecutionResult(
            ok=ok,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            oom_killed=proc.returncode == EXIT_OOM,
            exception_type="" if ok else _parse_exception_type(proc.stderr),
            traceback="" if ok else proc.stderr,
        ))


class DockerExecutor:
    """Production executor: runs code in the hardened container (the boundary)."""

    def __init__(self, timeout_s: int = 30, sandbox: str = "docker"):
        self.timeout_s = timeout_s
        self.sandbox = sandbox

    def run(self, code: str, data_path: str) -> ExecutionResult:
        from copilot.sandbox.command import build_sandbox_cmd

        # The generated code is passed to the in-container entrypoint on stdin;
        # the entrypoint (a pinned, trusted script) reads it and execs inside
        # the boundary. We never interpolate code into the argv.
        argv = build_sandbox_cmd(
            data_path=data_path,
            sandbox=self.sandbox,
            container_cmd=["python", "-m", "copilot.sandbox.entrypoint"],
        )
        try:
            proc = subprocess.run(
                argv,
                input=code,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            return ExecutionResult(
                ok=False,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + "\n[docker] wall-clock timeout; container killed",
                exit_code=EXIT_TIMEOUT,
                timed_out=True,
            )

        ok = proc.returncode == 0
        return _enrich_from_result(ExecutionResult(
            ok=ok,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            oom_killed=proc.returncode == EXIT_OOM,
            exception_type="" if ok else _parse_exception_type(proc.stderr),
            traceback="" if ok else proc.stderr,
        ))


def get_executor(timeout_s: int = 30) -> Any:
    """Return DockerExecutor if Docker is present, else the guarded DevExecutor."""
    import shutil

    if shutil.which("docker"):
        return DockerExecutor(timeout_s=timeout_s)
    return DevExecutor(timeout_s=timeout_s)
