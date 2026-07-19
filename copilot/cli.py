"""Command-line front door for the Autonomous Data Science Co-Pilot.

This is the missing "front door": it turns the tested ``run_full`` pipeline into
a command a person can type::

    python -m copilot analyze sales.csv "which region has the highest revenue?"

It wraps the existing graph unchanged — ingest -> profile -> intent -> planning
-> self-heal loop -> grounded insights -> report — and prints an honest report
(including a "what I could not do" section) plus the path to any chart produced.

Execution backend:
    * If Docker is installed, the hardened container is the boundary (see
      SECURITY.md) and is used automatically.
    * If Docker is NOT installed, the local dev backend is used. That backend is
      explicitly NOT a security boundary, so it is gated behind an env opt-in.
      This CLI sets that opt-in only when the user passes ``--allow-local`` (or
      the env var is already set), and it says so loudly. Never run an untrusted
      file this way.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List, Optional


def _has_docker() -> bool:
    return shutil.which("docker") is not None


def _print_report(state, *, verbose: bool = False) -> None:
    """Render the report dict to a readable terminal block."""
    report = getattr(state, "report", None) or {}
    status = report.get("status", "unknown")
    headline = report.get("headline", "")
    insights = report.get("insights", []) or []
    artifacts = report.get("artifacts", []) or []
    could_not = report.get("could_not_do", []) or []
    steps_ok = report.get("steps_ok", 0)
    steps_total = report.get("steps_total", 0)

    bar = "=" * 64
    print(bar)
    print(f"  STATUS : {status.upper()}   ({steps_ok}/{steps_total} steps succeeded)")
    if headline:
        print(f"  {headline}")
    print(bar)

    if insights:
        print("\nINSIGHTS")
        for ins in insights:
            print(f"  • {ins}")

    if artifacts:
        print("\nCHARTS / FILES")
        for art in artifacts:
            print(f"  • {art}")

    if could_not:
        print("\nWHAT I COULD NOT DO (and why)")
        for note in could_not:
            print(f"  • {note}")

    if verbose:
        meta = report.get("meta", {}) or {}
        if meta:
            print("\nRUN DETAILS")
            for k, v in meta.items():
                print(f"  {k}: {v}")
    print()


def _cmd_analyze(args: argparse.Namespace) -> int:
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"error: file not found: {file_path}", file=sys.stderr)
        return 2

    question = args.question or ""
    if not question.strip():
        print("error: please provide a question, e.g.\n"
              '  python -m copilot analyze data.csv "show revenue by region"',
              file=sys.stderr)
        return 2

    # Decide the execution backend and be explicit about safety.
    using_docker = _has_docker() and not args.force_local
    if using_docker:
        print("[sandbox] Docker found — running generated code in the hardened container.")
    else:
        if not (args.allow_local or os.environ.get("COPILOT_DEV_UNSAFE") == "1"):
            print(
                "error: Docker is not available, so generated code would run in the\n"
                "       LOCAL dev backend, which is NOT a security boundary (same user,\n"
                "       same machine, network access). Only do this with a file you TRUST.\n\n"
                "       Re-run with --allow-local to proceed on trusted data:\n"
                f'         python -m copilot analyze "{file_path}" "{question}" --allow-local',
                file=sys.stderr,
            )
            return 3
        os.environ["COPILOT_DEV_UNSAFE"] = "1"
        print("[sandbox] WARNING: Docker not found — using the LOCAL dev backend.")
        print("[sandbox] This is NOT a security boundary. Only run files you trust.")

    # Import lazily so `--help` and arg errors don't pay the import cost.
    from copilot.errors import IngestError
    from copilot.graph.run import run_full
    from copilot.sandbox.execute import DevExecutor, DockerExecutor

    # Build the executor explicitly so the chosen backend/runtime is honored.
    if using_docker:
        executor = DockerExecutor(sandbox=args.sandbox)
    else:
        executor = DevExecutor()

    print(f"[run] file     : {file_path}")
    print(f"[run] question : {question}")
    print("[run] working… (first model call can be slow while the model warms up)\n")

    try:
        state = run_full(file_path, question, executor=executor)
    except IngestError as e:
        # Should be turned into a REJECTED report by run_full, but guard anyway.
        print("=" * 64)
        print(f"  STATUS : REJECTED")
        print(f"  The file was rejected before analysis: {e}")
        print("=" * 64)
        return 1
    except Exception as e:  # noqa: BLE001 - top-level CLI guard
        print(f"error: the run failed unexpectedly: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    _print_report(state, verbose=args.verbose)

    report = getattr(state, "report", None) or {}
    status = report.get("status", "")
    # Exit code reflects outcome: 0 success, 1 partial/failed/rejected.
    return 0 if status == "success" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="copilot",
        description="Autonomous Data Science Co-Pilot — ask a plain-English "
                    "question about a data file and get a finished answer.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser(
        "analyze",
        help="analyze a data file with a plain-English question",
    )
    a.add_argument("file", help="path to the data file (.csv, .tsv, .txt, .xlsx, .json)")
    a.add_argument("question", help='your question, in quotes, e.g. "show revenue by region"')
    a.add_argument("--allow-local", action="store_true",
                   help="permit the local (non-sandboxed) backend when Docker is absent; "
                        "only use on files you trust")
    a.add_argument("--force-local", action="store_true",
                   help="use the local backend even if Docker is present (dev only)")
    a.add_argument("--sandbox", choices=["docker", "gvisor"], default="docker",
                   help="container runtime for the sandbox (default: docker)")
    a.add_argument("-v", "--verbose", action="store_true",
                   help="also print run details (file type, rows, attempts)")
    a.set_defaults(func=_cmd_analyze)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
