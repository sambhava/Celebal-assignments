"""Error classification — the gate that keeps the self-heal loop honest.

This is the highest-leverage node in the graph (TECHNICAL_DESIGN.md §1): a
misclassification is exactly what makes a RAG loop burn attempts on an unfixable
data problem. The rules, in priority order:

    1. Resource kill (OOM / timeout)      -> RESOURCE_LIMIT   (degrade, no re-run)
    2. Security violation                 -> SECURITY_VIOLATION (hard-fail)
    3. Clean run, but empty/degenerate    -> SEMANTIC_EMPTY    (surface, never retry)
    4. KeyError naming a column NOT in     -> DATA_PROBLEM      (surface; RAG can't
       the schema                                                fix a missing column)
    5. SyntaxError / IndentationError     -> SYNTAX            (retry, no RAG)
    6. AttributeError / ImportError /     -> API_MISUSE        (RAG-eligible retry)
       TypeError on a pandas symbol
    7. anything else at runtime           -> RUNTIME_RECOVERABLE (retry, no RAG)

Only API_MISUSE is RAG-eligible (ErrorClass.is_rag_eligible). Everything else
either retries WITHOUT RAG, surfaces to the user, or degrades — so RAG never
fires on a genuine data problem.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from copilot.graph.state import CoPilotState, ErrorClass

# Exceptions that mean "the code used the pandas/numpy API wrong" — RAG can help.
_API_MISUSE_EXC = {
    "AttributeError", "ImportError", "ModuleNotFoundError",
    "TypeError", "NameError",
}
_SYNTAX_EXC = {"SyntaxError", "IndentationError", "TabError"}
_SECURITY_EXC = {"SecurityViolation", "PermissionError"}

# pandas symbols in a traceback strengthen the API_MISUSE signal (vs a generic
# TypeError from unrelated code). Cheap heuristic, not load-bearing.
_PANDAS_HINT = re.compile(r"\b(pd|pandas|DataFrame|Series|numpy|np)\b")


def _extract_keyerror_key(traceback: str) -> str:
    """Return the key named by a KeyError, or '' if not a KeyError."""
    m = re.search(r"KeyError:\s*([\"'])(.*?)\1", traceback)
    return m.group(2) if m else ""


def _result_is_empty(execution_result: Dict[str, Any]) -> bool:
    """A clean run that produced nothing meaningful (0 rows / no artifact).

    The in-container entrypoint reports a ``result_shape`` and ``artifacts``;
    we treat 0-row output with no artifact as degenerate-but-clean.
    """
    shape = execution_result.get("result_shape")
    rows = shape[0] if isinstance(shape, (list, tuple)) and shape else None
    has_artifact = bool(execution_result.get("artifacts"))
    if execution_result.get("empty") is True:
        return True
    return rows == 0 and not has_artifact


def classify(execution_result: Dict[str, Any], schema: Dict[str, Any]) -> ErrorClass:
    """Pure function: map an execution result + schema to an ErrorClass."""
    # 1. resource kill takes precedence over any partial stderr
    if execution_result.get("oom_killed") or execution_result.get("timed_out"):
        return ErrorClass.RESOURCE_LIMIT

    exc = execution_result.get("exception_type", "")
    tb = execution_result.get("traceback", "") or execution_result.get("stderr", "")

    # 2. security violation
    if exc in _SECURITY_EXC or "SecurityViolation" in tb:
        return ErrorClass.SECURITY_VIOLATION

    # clean exit path
    if execution_result.get("ok"):
        # 3. clean but empty/degenerate -> surface, never retry
        if _result_is_empty(execution_result):
            return ErrorClass.SEMANTIC_EMPTY
        return None  # genuine success: no error class

    # 4. KeyError: is the missing key actually a missing column?
    key = _extract_keyerror_key(tb)
    if key:
        if key not in schema:
            return ErrorClass.DATA_PROBLEM   # RAG can't invent a column
        return ErrorClass.RUNTIME_RECOVERABLE  # key exists; a real code bug

    # 5. syntax
    if exc in _SYNTAX_EXC:
        return ErrorClass.SYNTAX

    # 6. API misuse (RAG-eligible)
    if exc in _API_MISUSE_EXC:
        return ErrorClass.API_MISUSE

    # 7. default: recoverable runtime error, retry without RAG
    return ErrorClass.RUNTIME_RECOVERABLE


#: Clearer public alias for the pure classification function.
classify_result = classify
