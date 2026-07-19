"""Report assembly — the terminal node for EVERY path.

This node never raises and always produces a report object. A run may be a full
success, a partial success (some steps degraded), or a total failure (every step
degraded / the budget was exhausted / the file was rejected before analysis) —
in all cases the user gets an honest artefact, never a hang and never a silent
empty.

Design invariants (TECHNICAL_DESIGN.md report + self-heal specs):
    * always emits a report, possibly partial
    * carries an explicit "what I could not do and why" section built from
      state.degradation_notes
    * status is derived structurally from step outcomes, NOT from model prose
    * insights are passed through verbatim from insight_synthesis (which is the
      node responsible for grounding); this node does not invent numbers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from copilot.graph.state import CoPilotState


class RunStatus(str, Enum):
    """Structural run status, derived from step outcomes — never from LLM text.

    Referenced by SECURITY.md (R9): the report status is emitted from this enum
    so an attacker-controlled cell value can never forge a "clean run" line.
    """
    SUCCESS = "success"           # every planned step produced an artifact/insight
    PARTIAL = "partial"           # some steps ok, some degraded
    FAILED = "failed"             # no step succeeded (or nothing was planned/run)
    REJECTED = "rejected"         # file never entered analysis (ingest rejected it)


@dataclass
class Report:
    """The finished artefact handed back to the UI."""
    status: RunStatus
    question: str
    headline: str
    insights: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    could_not_do: List[str] = field(default_factory=list)
    steps_ok: int = 0
    steps_total: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "question": self.question,
            "headline": self.headline,
            "insights": list(self.insights),
            "artifacts": list(self.artifacts),
            "could_not_do": list(self.could_not_do),
            "steps_ok": self.steps_ok,
            "steps_total": self.steps_total,
            "meta": dict(self.meta),
        }


def _derive_status(state: CoPilotState) -> RunStatus:
    """Status from step outcomes — structural, not textual."""
    outcomes = state.step_outcomes
    total = len(state.plan)

    # nothing planned/run: either the file was rejected pre-analysis or the
    # question had nothing runnable. Distinguish by whether a schema exists.
    if total == 0 and not outcomes:
        if not state.schema:
            return RunStatus.REJECTED
        return RunStatus.FAILED

    n_ok = sum(1 for o in outcomes if o == "ok")
    if n_ok == 0:
        return RunStatus.FAILED
    if n_ok == len([o for o in outcomes if o in ("ok", "degraded")]) and n_ok == total:
        return RunStatus.SUCCESS
    return RunStatus.PARTIAL


def _headline(status: RunStatus, n_ok: int, total: int) -> str:
    if status is RunStatus.SUCCESS:
        return f"Completed all {total} planned step(s)."
    if status is RunStatus.PARTIAL:
        return f"Completed {n_ok} of {total} planned step(s); the rest are explained below."
    if status is RunStatus.REJECTED:
        return "The uploaded file was rejected before analysis could start."
    return "Could not complete the requested analysis."


def assemble_report(state: CoPilotState) -> Report:
    """Build the final Report from state. Pure, never raises."""
    status = _derive_status(state)
    n_ok = sum(1 for o in state.step_outcomes if o == "ok")
    total = len(state.plan)

    could_not_do = list(state.degradation_notes)

    # If intent flagged a missing column / clarification need, surface it too —
    # it is a "what I could not do" even when a fallback still produced output.
    intent = state.intent or {}
    for missing in intent.get("missing_cols", []) or []:
        note = f"Your question referenced '{missing}', which is not a column in this file."
        if note not in could_not_do:
            could_not_do.append(note)

    return Report(
        status=status,
        question=state.user_question,
        headline=_headline(status, n_ok, total),
        insights=list(state.insights),
        artifacts=list(state.artifacts),
        could_not_do=could_not_do,
        steps_ok=n_ok,
        steps_total=total,
        meta={
            "file_type": state.declared_type,
            "rows": state.profile.get("row_count"),
            "cols": state.profile.get("col_count"),
            "mode": intent.get("mode"),
            "analysis_type": intent.get("analysis_type"),
            "global_attempts": state.global_attempts,
        },
    )


def report_assembly(state: CoPilotState) -> CoPilotState:
    """Node form: assemble the report and stash it on state.execution_result.

    Stored under a dedicated key so downstream/UI can read it without colliding
    with the per-step execution_result (which the loop clears on completion).
    """
    report = assemble_report(state)
    state.report = report.to_dict()
    return state
