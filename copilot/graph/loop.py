"""The self-heal loop driver — walk a finite plan, heal per step, always finish.

This ties together code_gen -> execute -> classify -> route into the bounded,
always-terminating loop specified in TECHNICAL_DESIGN.md. It is deliberately
parameterised on two callables so the whole loop can be tested with hand-injected
failures, no model and no Docker:

    codegen(step, state, level) -> code:str
        Produce code for a plan step at a given escalation level
        ("plain" | "with_history" | "with_rag_or_fallback"). In production this
        is the LLM node; in tests it is a scripted stub.
    executor.run(code, data_path) -> ExecutionResult
        Run the code (DockerExecutor in prod, DevExecutor/stub in tests).

Invariants this driver guarantees:
    * every plan step reaches a terminal outcome: "ok" or "degraded"
    * no step exceeds MAX_ATTEMPTS_PER_STEP; the run never exceeds
      MAX_GLOBAL_ATTEMPTS
    * on exhaustion the step is DEGRADED with a human-readable note, never
      retried forever, never silently dropped
    * the loop always returns; a report can always be assembled
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from copilot.codegen.generate import CodeGenError
from copilot.graph.classify import classify_result
from copilot.graph.router import ADVANCE, DEGRADE, RETRY, escalation_level, route, wants_rag
from copilot.graph.state import Attempt, CoPilotState, ErrorClass


def _degradation_note(step: Dict[str, Any], state: CoPilotState) -> str:
    """A concrete 'what I could not do and why' line for the final report."""
    ec = state.error_class
    kind = step.get("kind", "step")
    desc = step.get("desc", "")
    reason = {
        ErrorClass.DATA_PROBLEM: "the data doesn't support it (e.g. a referenced column is missing)",
        ErrorClass.SEMANTIC_EMPTY: "the analysis produced an empty/degenerate result",
        ErrorClass.SECURITY_VIOLATION: "the generated code tripped a security control",
        ErrorClass.RESOURCE_LIMIT: "it exceeded the memory/time budget",
        ErrorClass.SYNTAX: "the generated code could not be made to run within the attempt budget",
        ErrorClass.RUNTIME_RECOVERABLE: "the generated code kept erroring within the attempt budget",
        ErrorClass.API_MISUSE: "the generated code kept misusing the pandas API within the attempt budget",
    }.get(ec, "of an unrecoverable error")
    return f"Could not complete '{desc or kind}': {reason}."


def run_step(
    step: Dict[str, Any],
    state: CoPilotState,
    codegen: Callable[[Dict[str, Any], CoPilotState, str], str],
    executor: Any,
) -> str:
    """Drive one plan step through the bounded self-heal loop.

    Returns the terminal outcome for the step: "ok" or "degraded".
    Mutates ``state`` (attempt counters, history, artifacts, degradation_notes).
    """
    state.attempt_count = 0

    while True:
        # global budget guard first: never start an attempt we can't afford
        if state.global_attempts >= state.MAX_GLOBAL_ATTEMPTS:
            state.degradation_notes.append(_degradation_note(step, state))
            return "degraded"

        level = escalation_level(state.attempt_count)
        try:
            code = codegen(step, state, level)
        except CodeGenError as e:
            # Code-gen could not produce usable code (no code block, repeated
            # preflight failures, or a byte-identical re-emit). That is a
            # degraded step, NOT a crash: the "always produce a report"
            # invariant must hold even when the model is stuck.
            state.error_class = ErrorClass.RUNTIME_RECOVERABLE
            state.degradation_notes.append(
                f"Could not complete '{step.get('desc') or step.get('kind', 'step')}': "
                f"the model could not produce usable code ({e})."
            )
            return "degraded"
        state.generated_code = code

        result = executor.run(code=code, data_path=state.file_path)
        state.execution_result = result.to_dict() if hasattr(result, "to_dict") else result
        state.attempt_count += 1
        state.global_attempts += 1

        state.error_class = classify_result(state.execution_result, state.schema)

        # record the attempt so the NEXT retry can see what already failed
        state.attempt_history.append(
            Attempt(
                code=code,
                error=state.execution_result.get("traceback") or state.execution_result.get("stderr", ""),
                error_class=state.error_class,
            )
        )

        decision = route(state)
        if decision == ADVANCE:
            state.artifacts.extend(state.execution_result.get("artifacts", []))
            # record the successful step + its result so insight_synthesis can
            # ground insights in the code's own printed output (never invented).
            state.step_results.append({
                "step": dict(step),
                "result": dict(state.execution_result),
            })
            return "ok"
        if decision == DEGRADE:
            state.degradation_notes.append(_degradation_note(step, state))
            return "degraded"
        # RETRY: loop again; escalation_level(attempt_count) now steps up,
        # and wants_rag(state) tells the (future) rag_recovery node to fire.


def run_plan(
    state: CoPilotState,
    codegen: Callable[[Dict[str, Any], CoPilotState, str], str],
    executor: Any,
) -> CoPilotState:
    """Walk the finite plan once, healing each step within budget.

    Never adds steps; never revisits a completed step. Terminates when the plan
    is exhausted or the global attempt budget runs out — either way a report is
    assemblable from state.
    """
    outcomes: List[str] = []
    while state.current_step_idx < len(state.plan):
        if state.global_attempts >= state.MAX_GLOBAL_ATTEMPTS:
            # budget gone: degrade every remaining step honestly, then stop
            for remaining in state.plan[state.current_step_idx:]:
                state.degradation_notes.append(_degradation_note(remaining, state))
            break
        step = state.plan[state.current_step_idx]
        outcome = run_step(step, state, codegen, executor)
        outcomes.append(outcome)
        state.current_step_idx += 1
        # reset per-step history window (keep global attempt count)
        state.attempt_history = []

    state.execution_result = {}
    state.step_outcomes = outcomes
    return state
