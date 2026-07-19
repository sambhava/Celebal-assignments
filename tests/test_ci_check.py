"""Tests for the R1 CI launch-flag gate."""

import pytest

from copilot.sandbox.ci_check import LaunchFlagViolation, assert_launch_flags, main
from copilot.sandbox.command import build_sandbox_cmd


def test_gate_passes_on_real_builder_output():
    assert_launch_flags(build_sandbox_cmd(data_path="/host/d.csv"))
    assert_launch_flags(build_sandbox_cmd(data_path="/host/d.csv", sandbox="gvisor"))


def test_gate_fails_when_memory_swap_mismatched():
    # Simulate a regression where swap was left enabled.
    argv = ["docker", "run", "--network", "none", "--memory", "2g",
            "--memory-swap", "8g", "--cpus", "1.5", "--pids-limit", "128", "img"]
    with pytest.raises(LaunchFlagViolation):
        assert_launch_flags(argv)


def test_gate_fails_when_pids_limit_dropped():
    argv = ["docker", "run", "--network", "none", "--memory", "2g",
            "--memory-swap", "2g", "--cpus", "1.5", "img"]
    with pytest.raises(LaunchFlagViolation):
        assert_launch_flags(argv)


def test_gate_fails_when_network_not_none():
    argv = ["docker", "run", "--network", "bridge", "--memory", "2g",
            "--memory-swap", "2g", "--cpus", "1.5", "--pids-limit", "128", "img"]
    with pytest.raises(LaunchFlagViolation):
        assert_launch_flags(argv)


def test_main_returns_zero():
    assert main() == 0
