"""Hardened sandbox command builder — the security boundary as code (R1 + threat model).

The generated `docker run` argv *is* the boundary. This module builds it from a
single, non-overridable constant set, so no request path can widen it. Key
design rules (see SECURITY.md §3, R1):

* The hardened flags are appended **after** any caller-supplied `extra_run_args`.
  With `docker run`, the last occurrence of a repeated flag wins, so an injected
  `--network host` is overridden by the constant `--network none` that follows
  it. The boundary can only be *tightened* by a caller, never loosened.
* Resource caps are a **container cgroup** aggregate (`--memory == --memory-swap`,
  swap disabled), never a per-process rlimit (that was the R1 defect).
* No `-e`/`--env` is ever emitted: the container inherits no host secrets.
* The data file is bind-mounted **read-only**; the only writable space is a
  size-capped `noexec,nosuid` tmpfs.

The builder returns an argv **list** (never a shell string) so there is no shell
to inject into. It does not execute anything — running/lifecycle lives in
`runner.py`; keeping construction pure makes the boundary unit-testable and
CI-gateable without Docker present.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

# --- Non-overridable boundary constants --------------------------------------
# PARSE_RSS_BYTES default = 2 GiB. Sized against the documented minimum reviewer
# host RAM; a legitimately larger dataset is intentionally OOM-killed and
# surfaces as a typed RESOURCE_LIMIT (RAG-gated off), not a silent hang.
_2_GIB = 2 * 1024 * 1024 * 1024

DEFAULTS = {
    "memory": str(_2_GIB),        # --memory (bytes)
    "memory_swap": str(_2_GIB),   # --memory-swap == --memory  => swap disabled
    "cpus": "1.5",
    "pids_limit": "128",
    "user": "65534:65534",        # nobody:nogroup
    "tmpfs_size": "256m",
    "workdir": "/work",
    "data_mount": "/work/input",  # read-only mount point inside the container
}

SANDBOX_IMAGE = "copilot-sandbox:pinned"

# Flags a caller must never be able to set (they define the boundary). Any of
# these in `extra_run_args` is stripped, along with its value, before the
# authoritative constants are appended — the injection is removed, not merely
# overridden.
_BOUNDARY_FLAGS = frozenset({
    "--network", "--memory", "--memory-swap", "--memory-swappiness",
    "--cpus", "--pids-limit", "--oom-kill-disable", "--cap-drop", "--cap-add",
    "--security-opt", "--user", "-u", "--read-only", "--privileged",
    "--runtime", "--volume", "-v", "--mount", "--tmpfs", "--device",
    "--env", "-e", "--env-file",
})
# Of those, the ones that consume a following value token (space-separated form).
_VALUE_TAKING = frozenset(_BOUNDARY_FLAGS - {"--read-only", "--privileged"})


def _strip_boundary_flags(args: Sequence[str]) -> List[str]:
    """Remove any boundary-defining flag (and its value) a caller tried to pass."""
    out: List[str] = []
    skip_next = False
    for i, tok in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        base = tok.split("=", 1)[0]
        if base in _BOUNDARY_FLAGS:
            # `--flag=value` form drops in one token; `--flag value` form also
            # drops the following value token.
            if "=" not in tok and base in _VALUE_TAKING:
                skip_next = True
            continue
        out.append(tok)
    return out


def build_sandbox_cmd(
    *,
    data_path: str,
    image: str = SANDBOX_IMAGE,
    sandbox: str = "docker",
    extra_run_args: Optional[Sequence[str]] = None,
    container_cmd: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return the hardened ``docker run`` argv for one sandboxed execution.

    Parameters
    ----------
    data_path:
        Host path to the untrusted input file; bind-mounted read-only.
    image:
        Pinned sandbox image (digest-pinned in production).
    sandbox:
        ``"docker"`` (default) or ``"gvisor"`` (adds ``--runtime=runsc``).
    extra_run_args:
        Optional caller args for *non-boundary* concerns (labels, etc.). Any
        boundary-defining flag here is stripped (see ``_strip_boundary_flags``),
        so a caller can never widen the boundary.
    container_cmd:
        The in-container command; defaults to the parse/exec entrypoint.

    Notes
    -----
    Fresh container per call (``--rm``); no state bleed between runs.
    """
    argv: List[str] = ["docker", "run", "--rm"]

    # gVisor opt-in — same flag set, user-space kernel.
    if sandbox == "gvisor":
        argv += ["--runtime", "runsc"]

    # Sanitised caller extras first; authoritative constants below.
    if extra_run_args:
        argv += _strip_boundary_flags(extra_run_args)


    # --- The boundary constants (emitted last, authoritative) ---
    argv += [
        # egress: the crown jewel
        "--network", "none",
        # R1 aggregate cgroup caps (NOT per-process rlimit)
        "--memory", DEFAULTS["memory"],
        "--memory-swap", DEFAULTS["memory_swap"],
        "--memory-swappiness", "0",
        "--cpus", DEFAULTS["cpus"],
        "--pids-limit", DEFAULTS["pids_limit"],
        "--oom-kill-disable=false",
        # process isolation
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--user", DEFAULTS["user"],
        # filesystem
        "--read-only",
        "--tmpfs", "{workdir}:rw,noexec,nosuid,size={size}".format(
            workdir=DEFAULTS["workdir"], size=DEFAULTS["tmpfs_size"]
        ),
        "--workdir", DEFAULTS["workdir"],
        # untrusted data: read-only bind mount
        "--volume", "{host}:{dest}:ro".format(host=data_path, dest=DEFAULTS["data_mount"]),
    ]

    argv.append(image)
    argv += list(container_cmd) if container_cmd else ["python", "-m", "copilot.sandbox.entrypoint"]
    return argv
