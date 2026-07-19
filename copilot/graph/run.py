"""Walking-skeleton runner: the first executable end-to-end path.

Chains the implemented nodes linearly (ingest -> profile) with no LLM and no
planning yet, so a real file flows all the way to a schema + profile through the
same typed-error gates the full graph will use. This is Phase 1 of the build
plan (deterministic single-shot pipeline); later phases insert the LLM code-gen,
sandboxed execution, and self-heal loop between profile and report.

Every terminal path returns a ``CoPilotState`` -- a successful profile, or a
raised typed error (MalformedInput / SecurityViolation) that the caller renders
honestly. Nothing hangs, nothing silently succeeds on bad input.
"""

from __future__ import annotations

from copilot.graph.nodes import ingest_and_validate, schema_profile
from copilot.graph.state import CoPilotState

#: Ordered nodes for the walking skeleton. The full graph adds intent ->
#: planning -> code_gen -> execute -> classify/recover -> synthesis before
#: report_assembly.
SKELETON_NODES = (ingest_and_validate, schema_profile)


def run_walking_skeleton(
    file_path: str,
    user_question: str = "",
    session_id: str = "local",
) -> CoPilotState:
    """Run ingest + profile over ``file_path`` and return the populated state.

    Raises the ingest gate's typed errors (MalformedInput / SecurityViolation)
    on rejected input -- callers catch and surface them rather than crashing.
    """
    state = CoPilotState(
        session_id=session_id,
        file_path=file_path,
        user_question=user_question,
    )
    for node in SKELETON_NODES:
        state = node(state)
    return state


def run_full(
    file_path: str,
    user_question: str,
    *,
    session_id: str = "local",
    codegen: "Callable | None" = None,
    executor: "object | None" = None,
) -> CoPilotState:
    """Run the whole graph end-to-end and return the final state (with .report).

    Order: ingest -> profile -> intent -> planning -> [self-heal loop over the
    plan: code_gen -> execute -> classify -> route -> heal/degrade] -> insight
    synthesis -> report assembly. Every terminal path produces ``state.report``.

    ``codegen`` and ``executor`` are injectable so the loop can be driven by the
    real LLM + sandbox in production, or by scripted stubs in tests. Defaults wire
    the real Ollama code-gen and the auto-selected executor (Docker if present,
    else the guarded dev backend).

    Ingest/profile may raise typed MalformedInput / SecurityViolation on hostile
    input; those are caught and turned into a REJECTED report so the caller always
    gets an honest artefact.
    """
    from copilot.errors import IngestError
    from copilot.graph.loop import run_plan
    from copilot.graph.nodes import (
        code_gen as default_codegen_node,
        insight_synthesis,
        intent_understanding,
        planning,
        report_assembly,
    )

    state = CoPilotState(
        session_id=session_id, file_path=file_path, user_question=user_question,
    )

    # 1. ingest + profile (may reject hostile input with a typed error)
    try:
        state = ingest_and_validate(state)
        state = schema_profile(state)
    except IngestError as e:
        state.degradation_notes.append(f"File rejected before analysis: {e}")
        return report_assembly(state)

    # 2. understand + plan (deterministic, LLM-free)
    state = intent_understanding(state)
    state = planning(state)

    # 3. self-heal loop over the finite plan
    if executor is None:
        from copilot.sandbox.execute import get_executor
        executor = get_executor()
    if codegen is None:
        codegen = _default_codegen_adapter(default_codegen_node)
    state = run_plan(state, codegen, executor)

    # 4. grounded insights + report
    state = insight_synthesis(state)
    state = report_assembly(state)
    return state


def _default_codegen_adapter(_node):
    """Adapt the code_gen node to the loop's ``codegen(step, state, level)`` call.

    The node reads the *current* plan step off state; the loop passes the step
    explicitly. We align them by setting current_step_idx, then delegating to the
    real generator so production uses one code path.
    """
    from copilot.codegen.generate import generate_code

    def _codegen(step, state, level):
        sample = getattr(state, "sample", None)
        rag = state.retrieved_docs[0] if state.retrieved_docs else ""
        return generate_code(
            step, state.schema, level,
            sample=sample, attempt_history=state.attempt_history, rag_context=rag,
        )

    return _codegen
