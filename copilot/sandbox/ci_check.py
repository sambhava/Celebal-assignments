"""R1 [CI-REQUIRED] launch-flag gate.

Parses the argv that `build_sandbox_cmd()` would emit and fails if any aggregate
resource cap is missing or if `--memory != --memory-swap`. Wired into CI so a
regression that drops the boundary fails the build rather than shipping. Run as
`python -m copilot.sandbox.ci_check` (exit 1 on violation) or via the pytest
that calls `assert_launch_flags`.
"""

from __future__ import annotations

import sys
from typing import List, Sequence

from copilot.sandbox.command import DEFAULTS, build_sandbox_cmd

REQUIRED_FLAGS = ("--network", "--memory", "--memory-swap", "--cpus", "--pids-limit")


class LaunchFlagViolation(AssertionError):
    """Raised when the emitted sandbox argv is missing a required boundary flag."""


def _value(argv: Sequence[str], flag: str):
    for i, tok in enumerate(argv):
        if tok == flag:
            return argv[i + 1] if i + 1 < len(argv) else None
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def assert_launch_flags(argv: Sequence[str]) -> None:
    """Raise ``LaunchFlagViolation`` if the boundary argv is under-hardened."""
    missing = [f for f in REQUIRED_FLAGS if _value(argv, f) is None]
    if missing:
        raise LaunchFlagViolation("sandbox argv missing required flags: " + ", ".join(missing))

    mem, swap = _value(argv, "--memory"), _value(argv, "--memory-swap")
    if mem != swap:
        raise LaunchFlagViolation(
            "--memory ({}) must equal --memory-swap ({}) so swap is disabled".format(mem, swap)
        )

    if _value(argv, "--network") != "none":
        raise LaunchFlagViolation("--network must be 'none' (no egress)")


def main(argv: List[str] = None) -> int:
    # Check both the default (docker) and gVisor forms — both must be hardened.
    for sandbox in ("docker", "gvisor"):
        cmd = build_sandbox_cmd(data_path="/host/example.csv", sandbox=sandbox)
        try:
            assert_launch_flags(cmd)
        except LaunchFlagViolation as exc:
            print("R1 CI GATE FAILED ({}): {}".format(sandbox, exc), file=sys.stderr)
            return 1
    print("R1 CI gate OK: docker + gvisor argv both hardened "
          "(--memory==--memory-swap, network none, cpu/pids capped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
