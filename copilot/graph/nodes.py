"""LangGraph node stubs for the CoPilot graph.

Each node is a plain ``CoPilotState -> CoPilotState`` function so the graph is
testable headless and importable without langgraph. Bodies are stubs
(``NotImplementedError``) — this scaffold pins the *contract* (inputs, outputs,
failure modes) of each node per TECHNICAL_DESIGN.md; implementations land in
their own phases.

Nodes that parse or execute untrusted bytes (ingest, profile, execute) MUST run
inside the sandbox (see SECURITY.md); their stubs note this.
"""

from __future__ import annotations

from copilot.graph.state import CoPilotState


def ingest_and_validate(state: CoPilotState) -> CoPilotState:
    """G1-G7 untrusted-file gate. In: file_path. Out: validated file + file_meta.

    Failure modes: zip/decompression bomb, oversized file, magic-byte mismatch,
    ragged CSV (R11), formula/XXE payloads. Parsing runs INSIDE the sandbox.
    Rejections are typed MalformedInput/SecurityViolation (never RAG, never a
    burned attempt).
    """
    from copilot.ingest.preflight import preflight

    result = preflight(state.file_path)
    state.declared_type = result.declared_type
    state.file_meta = dict(result.file_meta)
    return state


def schema_profile(state: CoPilotState) -> CoPilotState:
    """In: validated file. Out: schema + profile (dtypes, nulls, cardinality).

    Failure modes: encoding errors, mixed-type columns, all-NaN columns. Runs
    INSIDE the sandbox (parsing untrusted files is attack surface).
    """
    from copilot.ingest.load import load_dataframe
    from copilot.ingest.profile import profile_dataframe

    df = load_dataframe(state.file_path, state.declared_type)
    state.dataframe = df
    state.schema, state.profile = profile_dataframe(df)
    return state


def intent_understanding(state: CoPilotState) -> CoPilotState:
    """In: user_question + schema. Out: intent{mode, analysis_type, target_cols}.

    Validates referenced columns exist against schema BEFORE any code-gen call.
    Failure modes: vague question -> route to open_ended; question references a
    nonexistent column -> flag now (data/intent problem), do not discover it at
    execution time.
    """
    from copilot.intent.understand import understand_intent

    state.intent = understand_intent(state.user_question, state.schema)
    return state


def planning(state: CoPilotState) -> CoPilotState:
    """In: intent + schema + profile. Out: finite ordered plan.

    Targeted mode: template instantiation (model only fills column names).
    Open-ended mode: capped top-N exploration battery, ranked by a cheap
    interestingness heuristic, then STOP. Failure mode: over-planning (guarded
    by the hard cap).
    """
    from copilot.planning.plan import build_plan

    state.plan = build_plan(state.intent, state.schema, state.profile)
    return state


def code_gen(state: CoPilotState) -> CoPilotState:
    """In: current step + schema + attempt_history + retrieved_docs. Out: code.

    Uses a strict-prompt + regex-extract + validate-and-repair path for the weak
    local model (Ollama; the vLLM grammar-constrained path is not used here).
    Always injects the deprecation cheatsheet and treats file-derived text as
    inert data. Failure modes: hallucinated API/columns (caught downstream, not
    here).
    """
    from copilot.graph.router import escalation_level, wants_rag

    step = state.plan[state.current_step_idx]
    level = escalation_level(state.attempt_count)
    rag = "\n".join(state.retrieved_docs) if wants_rag(state) else ""
    state.generated_code = make_code(step, state, level, rag_context=rag)
    return state


def make_code(step, state, level, *, rag_context: str = "") -> str:
    """Loop-compatible codegen callable: (step, state, level) -> code string.

    Adapts state -> generate_code(...) and turns an unreachable model into an
    honest, raised error so the loop degrades instead of emitting empty code.
    """
    from copilot.codegen.generate import generate_code
    from copilot.codegen.sanitize import sanitize_sample

    sample = None
    raw_sample = (state.profile or {}).get("sample_rows")
    if raw_sample:
        sample = sanitize_sample(raw_sample, list(state.schema.keys()))

    return generate_code(
        step,
        state.schema,
        level,
        sample=sample,
        attempt_history=state.attempt_history,
        rag_context=rag_context,
    )


def sandboxed_execution(state: CoPilotState) -> CoPilotState:
    """In: generated_code + data. Out: execution_result (artifacts/stdout/error).

    Runs in the hardened container (copilot.sandbox.command). Failure modes:
    timeout, OOM (-> RESOURCE_LIMIT), exception, empty/degenerate-but-clean
    result (-> SEMANTIC_EMPTY).
    """
    from copilot.sandbox.execute import get_executor

    executor = get_executor()
    result = executor.run(code=state.generated_code, data_path=state.file_path)
    state.execution_result = result.to_dict()
    return state


def error_classification(state: CoPilotState) -> CoPilotState:
    """In: execution_result. Out: error_class + routing decision.

    The highest-leverage node: misclassification is what makes RAG loop on data
    problems. Checks KeyError against schema (missing column -> DATA_PROBLEM, not
    API_MISUSE); empty result -> SEMANTIC_EMPTY; OOM/timeout -> RESOURCE_LIMIT.
    """
    from copilot.graph.classify import classify_result

    state.error_class = classify_result(state.execution_result, state.schema)
    return state


def rag_recovery(state: CoPilotState) -> CoPilotState:
    """In: error_class + traceback + schema. Out: retrieved_docs.

    Fires ONLY for ErrorClass.API_MISUSE. Retrieval query built from the
    exception type + offending attribute/method name; version-pinned pandas
    docs. Failure mode: retrieving irrelevant chunks (bounded to API errors).
    """
    from copilot.rag.recover import rag_recovery as _rag_recovery

    return _rag_recovery(state)


def retry_router(state: CoPilotState) -> str:
    """In: attempt_count + error_class. Out: next-node name.

    Escalation ladder: attempt1 plain; attempt2 + traceback + attempt_history;
    attempt3 + RAG (if API_MISUSE) or simplified-plan fallback. Bounded to
    MAX_ATTEMPTS_PER_STEP and MAX_GLOBAL_ATTEMPTS; on exhaustion routes to
    degrade (records a degradation_note), never hangs.
    """
    from copilot.graph.router import route

    return route(state)


def insight_synthesis(state: CoPilotState) -> CoPilotState:
    """In: successful execution_result(s). Out: plain-English insights.

    GROUNDED: may reference only values present in the execution result (no
    recomputation, no invented numbers). Failure mode: over-claiming
    (correlation stated as causation) — guarded by grounding.
    """
    from copilot.insight.synthesize import synthesize_insights

    # The loop records {"step", "result"} wrappers for each successful step;
    # unwrap them into parallel lists for grounded synthesis.
    recorded = list(getattr(state, "step_results", None) or [])
    if recorded:
        steps = [r.get("step", {}) for r in recorded]
        results = [r.get("result", {}) for r in recorded]
    elif state.execution_result:
        results = [state.execution_result]
        steps = [state.plan[state.current_step_idx - 1]] if state.plan else [{}]
    else:
        steps, results = [], []
    state.insights = synthesize_insights(steps, results)
    return state


def report_assembly(state: CoPilotState) -> CoPilotState:
    """In: insights + artifacts + degradation_notes. Out: final artefact.

    Always emits a report, possibly partial, with an explicit "what I could not
    do and why" section. This is the terminal node for every path.
    """
    from copilot.report.assemble import report_assembly as _assemble

    return _assemble(state)


#: Node registry — name -> callable. The graph builder wires edges over this.
NODES = {
    "ingest_and_validate": ingest_and_validate,
    "schema_profile": schema_profile,
    "intent_understanding": intent_understanding,
    "planning": planning,
    "code_gen": code_gen,
    "sandboxed_execution": sandboxed_execution,
    "error_classification": error_classification,
    "rag_recovery": rag_recovery,
    "retry_router": retry_router,
    "insight_synthesis": insight_synthesis,
    "report_assembly": report_assembly,
}
