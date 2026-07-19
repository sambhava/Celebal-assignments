"""The self-heal retry router — bounded, escalating, always-terminating.

This is the control-flow heart of the reliability story (TECHNICAL_DESIGN.md
self-heal spec). It is a PURE decision function over the loop-control state; it
calls no model and touches no I/O, so the entire loop is testable with
hand-injected failures.

Routing contract, given the current ``error_class`` and attempt counters:

    success (error_class is None)          -> "advance"   (next plan step)
    terminal, never-retry classes          -> "degrade"   (surface + note)
        DATA_PROBLEM, SEMANTIC_EMPTY, SECURITY_VIOLATION, RESOURCE_LIMIT
    retryable classes, budget remaining    -> "retry"     (back to code_gen)
        SYNTAX, RUNTIME_RECOVERABLE, API_MISUSE
    retryable but per-step or global budget exhausted -> "degrade"

Escalation ladder (what the retry carries), by per-step attempt number:
    attempt 1 -> plain regen from step + schema
    attempt 2 -> + full traceback + attempt_history (model sees its own failure)
    attempt 3 -> + RAG docs (only if API_MISUSE) OR simplified-plan fallback

The two bounds are BOTH enforced:
    - MAX_ATTEMPTS_PER_STEP caps effort on a single step
    - MAX_GLOBAL_ATTEMPTS caps effort across the whole run

so the loop can never hang: every path eventually routes to "advance" or
"degrade", and "degrade" always records why.
"""

from __future__ import annotations

from typing import Optional

from copilot.graph.state import CoPilotState, ErrorClass

# Classes that a retry can plausibly fix.
_RETRYABLE = {
    ErrorClass.SYNTAX,
    ErrorClass.RUNTIME_RECOVERABLE,
    ErrorClass.API_MISUSE,
}
# Classes that must be surfaced/degraded immediately — retrying is pointless or
# unsafe (RAG can't invent a column; an empty result is a finding; a security
# violation must hard-fail; a resource kill must not re-run the same bomb).
_TERMINAL = {
    ErrorClass.DATA_PROBLEM,
    ErrorClass.SEMANTIC_EMPTY,
    ErrorClass.SECURITY_VIOLATION,
    ErrorClass.RESOURCE_LIMIT,
}

# Route names the graph builder wires edges over.
ADVANCE = "advance"
RETRY = "retry"
DEGRADE = "degrade"


def route(state: CoPilotState) -> str:
    """Return the next route: ``advance`` | ``retry`` | ``degrade``.

    Pure over ``state.error_class`` + the attempt counters; no side effects.
    """
    ec = state.error_class

    # genuine success -> move on to the next plan step
    if ec is None:
        return ADVANCE

    # terminal classes never retry
    if ec in _TERMINAL:
        return DEGRADE

    # retryable — but only if BOTH budgets allow another attempt
    if ec in _RETRYABLE:
        if state.attempt_count >= state.MAX_ATTEMPTS_PER_STEP:
            return DEGRADE
        if state.global_attempts >= state.MAX_GLOBAL_ATTEMPTS:
            return DEGRADE
        return RETRY

    # unknown class -> fail safe by degrading, never loop
    return DEGRADE


def escalation_level(attempt_count: int) -> str:
    """What the *next* retry should carry, per the ladder.

    attempt_count is the number of attempts ALREADY made on this step.
    """
    if attempt_count <= 0:
        return "plain"            # first try: step + schema only
    if attempt_count == 1:
        return "with_history"     # + traceback + attempt_history
    return "with_rag_or_fallback"  # + RAG (if API_MISUSE) or simplified plan


def wants_rag(state: CoPilotState) -> bool:
    """RAG is pulled only at the top of the ladder AND only for API_MISUSE."""
    return (
        state.error_class is ErrorClass.API_MISUSE
        and escalation_level(state.attempt_count) == "with_rag_or_fallback"
    )


def retry_router(state: CoPilotState) -> str:
    """Node form: the graph calls this to get the next edge to follow."""
    return route(state)
