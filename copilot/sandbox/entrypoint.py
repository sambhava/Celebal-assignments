"""Trusted in-container entrypoint — the only code in the sandbox we wrote.

The generated (semi-trusted) model code arrives on **stdin**. This script:

    1. Loads the untrusted data file into a pandas ``df`` (via the same hardened
       loaders used on the host: field-count guard, XML hardening, shape caps).
    2. Builds a restricted namespace exposing exactly what the data-contract
       promises the model: ``pd``, ``df``, ``plt``, and ``save_artifact(name)``.
       The model NEVER needs ``os``/``open``/``read_csv`` — so the code-gen
       pre-flight can keep rejecting those as a cheap signal, and legitimate
       code stays import-free.
    3. ``exec``s the model code in that namespace.
    4. Emits a single machine-readable RESULT line on stdout so the executor can
       populate ExecutionResult.result_shape / artifacts / empty for the
       classifier (SEMANTIC_EMPTY detection).

This is NOT the security boundary — the container (``--network none``, dropped
caps, read-only rootfs, no secrets) is. This script just makes the contract
concrete and the result parseable. It runs INSIDE that container.

Environment (set by the executor / container):
    COPILOT_DATA_PATH     absolute path to the (read-only mounted) data file
    COPILOT_DATA_TYPE     declared type: csv|tsv|json|xlsx  (from preflight)
    COPILOT_ARTIFACT_DIR  writable tmpfs dir for output charts
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# Sentinel prefixing the single JSON result line. Chosen to be unlikely to
# collide with anything the model prints; the executor scans stdout for it.
RESULT_SENTINEL = "@@COPILOT_RESULT@@"


def _emit(payload: dict) -> None:
    """Write the one structured result line the executor parses."""
    sys.stdout.write("\n" + RESULT_SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()


class _LazyPlt:
    """Deferred matplotlib.pyplot proxy.

    matplotlib is only needed for charting steps. Importing it eagerly would
    make a pure-aggregation step fail on a machine without matplotlib. This
    proxy imports (and forces the Agg backend) on first attribute access, so
    non-charting code never touches it.
    """

    _plt = None

    def _load(self):
        if self._plt is None:
            import matplotlib
            matplotlib.use("Agg")  # no display, no GUI, deterministic PNG
            import matplotlib.pyplot as plt
            self._plt = plt
        return self._plt

    def __getattr__(self, name):
        return getattr(self._load(), name)


def _build_namespace(df, artifact_dir):
    """The exact surface the data-contract promises the model."""
    import pandas as pd

    plt = _LazyPlt()
    saved: list = []

    def save_artifact(name: str = "chart.png"):
        """Save the current matplotlib figure into the artifact dir.

        Returns the absolute path. Only a basename is honored (no path
        traversal); the file lands in the writable tmpfs artifact dir.
        """
        base = os.path.basename(str(name)) or "chart.png"
        if not base.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
            base += ".png"
        path = os.path.join(artifact_dir, base)
        plt.savefig(path, bbox_inches="tight", dpi=100)
        plt.close("all")
        saved.append(path)
        return path

    ns = {
        "pd": pd,
        "plt": plt,
        "df": df,
        "save_artifact": save_artifact,
        "__builtins__": __builtins__,  # exec convenience; container is the boundary
    }
    return ns, saved


def _result_shape(ns) -> list | None:
    """Best-effort shape of a conventional ``result`` variable if present."""
    obj = ns.get("result")
    if obj is None:
        return None
    shape = getattr(obj, "shape", None)
    if shape is not None:
        try:
            return list(shape)
        except TypeError:
            return None
    try:
        return [len(obj)]
    except TypeError:
        return None


def main() -> int:
    data_path = os.environ.get("COPILOT_DATA_PATH", "")
    data_type = os.environ.get("COPILOT_DATA_TYPE", "csv")
    artifact_dir = os.environ.get("COPILOT_ARTIFACT_DIR", ".")

    code = sys.stdin.read()

    # Load the untrusted data through the hardened loaders (R11/R12/shape caps).
    try:
        from copilot.ingest.load import load_dataframe
        df = load_dataframe(data_path, data_type)
    except Exception as e:  # noqa: BLE001 - report any load failure structurally
        _emit({
            "ok": False,
            "phase": "load",
            "exception_type": type(e).__name__,
            "error": str(e)[:500],
        })
        return 3

    ns, saved = _build_namespace(df, artifact_dir)

    try:
        exec(compile(code, "<generated>", "exec"), ns)  # noqa: S102 - sandboxed
    except Exception as e:  # noqa: BLE001 - the whole point is to classify it
        _emit({
            "ok": False,
            "phase": "exec",
            "exception_type": type(e).__name__,
            "error": str(e)[:500],
            "traceback": traceback.format_exc()[-1500:],
            "artifacts": saved,
        })
        return 1

    shape = _result_shape(ns)
    empty = False
    if shape is not None:
        empty = (len(shape) >= 1 and shape[0] == 0) and not saved

    _emit({
        "ok": True,
        "phase": "done",
        "result_shape": shape,
        "artifacts": saved,
        "empty": empty,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
