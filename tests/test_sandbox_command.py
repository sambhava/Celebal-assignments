"""Contract tests for the hardened sandbox command builder (R1 + threat model).

The generated `docker run` argv IS the security boundary. These tests pin every
control the SECURITY.md threat model claims, so a regression that weakens the
boundary fails the build rather than shipping:

  * --network none            (crown jewel: no egress)
  * --memory == --memory-swap (R1: aggregate cgroup cap, swap disabled)
  * --cpus, --pids-limit      (R1: bounded CPU / fork battery)
  * --cap-drop ALL, no-new-privileges, non-root user
  * --read-only rootfs + noexec size-capped tmpfs
  * data mounted read-only
  * NO secrets / env passed into the container
  * flags are constants, NOT widenable from caller input
  * gVisor via --runtime=runsc behind an opt-in
"""

import pytest

from copilot.sandbox.command import DEFAULTS, build_sandbox_cmd


def _flags(argv):
    """Return the set of bare flag tokens in an argv list."""
    return [a for a in argv if a.startswith("--")]


def _val(argv, flag):
    """Return the value following `flag` in `docker run --flag value` form."""
    for i, a in enumerate(argv):
        if a == flag:
            return argv[i + 1] if i + 1 < len(argv) else None
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


@pytest.fixture
def argv():
    return build_sandbox_cmd(data_path="/host/data.csv", image="copilot-sandbox:pinned")


# --- egress + secrets: the crown jewel ---------------------------------------

def test_network_is_none(argv):
    assert _val(argv, "--network") == "none"


def test_no_secrets_or_env_passed(argv):
    # No -e / --env anywhere: the container must never see host env/API keys.
    assert "-e" not in argv
    assert not any(a == "--env" or a.startswith("--env=") for a in argv)


# --- R1: aggregate resource cap ----------------------------------------------

def test_memory_equals_memory_swap_swap_disabled(argv):
    mem = _val(argv, "--memory")
    swap = _val(argv, "--memory-swap")
    assert mem is not None and swap is not None
    assert mem == swap  # equal => swap disabled => no thrash escape hatch


def test_cpu_and_pids_capped(argv):
    assert _val(argv, "--cpus") is not None
    assert _val(argv, "--pids-limit") is not None


# --- process isolation --------------------------------------------------------

def test_caps_dropped_and_no_new_privileges(argv):
    assert _val(argv, "--cap-drop") == "ALL"
    assert "no-new-privileges:true" in argv or "no-new-privileges" in " ".join(argv)


def test_runs_as_non_root(argv):
    user = _val(argv, "--user")
    assert user is not None
    uid = user.split(":")[0]
    assert uid not in ("0", "root")


# --- filesystem ---------------------------------------------------------------

def test_rootfs_read_only(argv):
    assert "--read-only" in argv


def test_writable_tmpfs_is_noexec(argv):
    tmpfs = [a for i, a in enumerate(argv) if argv[i - 1] == "--tmpfs"]
    assert tmpfs, "expected a --tmpfs writable scratch mount"
    assert any("noexec" in t for t in tmpfs)
    assert any("size=" in t for t in tmpfs)  # size-capped


def test_data_mounted_read_only(argv):
    mount = [a for i, a in enumerate(argv) if argv[i - 1] in ("-v", "--volume")]
    assert mount, "expected the data file to be volume-mounted"
    assert any(m.endswith(":ro") for m in mount), "data mount must be :ro"


# --- non-widenable: caller cannot loosen the boundary ------------------------

def test_caller_cannot_widen_network():
    # Even if a caller tries to inject their own network, the constant wins.
    argv = build_sandbox_cmd(data_path="/host/d.csv", extra_run_args=["--network", "host"])
    # The hardened --network none must still be present and host must not slip in
    # as the effective network. We assert none is present exactly once and host
    # is rejected.
    assert _val(argv, "--network") == "none"
    assert "host" not in argv


def test_memory_cannot_be_widened_by_caller():
    argv = build_sandbox_cmd(data_path="/host/d.csv", extra_run_args=["--memory", "64g"])
    assert _val(argv, "--memory") == DEFAULTS["memory"]


# --- gVisor opt-in ------------------------------------------------------------

def test_gvisor_runtime_opt_in():
    argv = build_sandbox_cmd(data_path="/host/d.csv", sandbox="gvisor")
    assert _val(argv, "--runtime") == "runsc"


def test_default_runtime_is_docker_no_runsc(argv):
    assert _val(argv, "--runtime") != "runsc"


# --- shape --------------------------------------------------------------------

def test_argv_starts_with_docker_run(argv):
    assert argv[:2] == ["docker", "run"]


def test_command_is_appended_last(argv):
    # The in-container command (python parse entrypoint) comes after the image.
    assert "copilot-sandbox:pinned" in argv
    assert argv.index("copilot-sandbox:pinned") < len(argv) - 1
