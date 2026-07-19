"""The CoPilot graph state — the object carried through every node.

Kept as a plain dataclass (not a LangGraph-specific type) so it is importable
and testable without langgraph installed; the graph wiring adapts it. See
TECHNICAL_DESIGN.md for the full node graph.

The one non-obvious, load-bearing field is ``attempt_history``: each retry must
see what was already tried and how it failed, or a weak local model re-emits the
same broken code. It is what makes the self-heal loop *converge* instead of
spinning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Mode(str, Enum):
    TARGETED = "targeted"        # question maps to a use-case template
    OPEN_ENDED = "open_ended"    # vague question -> capped exploration battery


class ErrorClass(str, Enum):
    """Error taxonomy that GATES the self-heal loop (see TECHNICAL_DESIGN.md).

    Only ``API_MISUSE`` is RAG-eligible. ``DATA_PROBLEM``, ``SEMANTIC_EMPTY``,
    ``RESOURCE_LIMIT`` and ``SECURITY_VIOLATION`` are surfaced/degraded, never
    RAG-looped.
    """
    SYNTAX = "syntax"                     # retry, no RAG
    API_MISUSE = "api_misuse"             # RAG-eligible retry
    RUNTIME_RECOVERABLE = "runtime_recoverable"  # retry, no RAG
    DATA_PROBLEM = "data_problem"         # surface (e.g. missing column)
    SEMANTIC_EMPTY = "semantic_empty"     # clean but empty -> surface, never retry
    RESOURCE_LIMIT = "resource_limit"     # OOM/timeout -> degrade, no re-run
    SECURITY_VIOLATION = "security_violation"    # hard-fail, never retry

    @property
    def is_retryable(self) -> bool:
        return self in {
            ErrorClass.SYNTAX,
            ErrorClass.API_MISUSE,
            ErrorClass.RUNTIME_RECOVERABLE,
        }

    @property
    def is_rag_eligible(self) -> bool:
        return self is ErrorClass.API_MISUSE


@dataclass
class Attempt:
    """One code-gen/execute attempt, recorded so the next retry can see it."""
    code: str
    error: Optional[str] = None
    error_class: Optional[ErrorClass] = None


@dataclass
class CoPilotState:
    # --- immutable inputs ---
    session_id: str
    file_path: str
    user_question: str
    file_meta: Dict[str, Any] = field(default_factory=dict)  # type, size, sha256

    # --- populated as the graph runs ---
    schema: Dict[str, Any] = field(default_factory=dict)
    profile: Dict[str, Any] = field(default_factory=dict)
    dataframe: Any = None                                    # loaded frame (in-sandbox in prod)
    declared_type: str = ""                                  # csv|tsv|json|xlsx (from preflight)
    intent: Dict[str, Any] = field(default_factory=dict)     # {mode, analysis_type, target_cols}
    plan: List[Dict[str, Any]] = field(default_factory=list)  # finite, ordered
    current_step_idx: int = 0

    # --- per-attempt (reset each step) ---
    generated_code: str = ""
    execution_result: Dict[str, Any] = field(default_factory=dict)
    error_class: Optional[ErrorClass] = None
    retrieved_docs: List[str] = field(default_factory=list)

    # --- loop control ---
    attempt_count: int = 0
    attempt_history: List[Attempt] = field(default_factory=list)
    global_attempts: int = 0

    # --- outputs ---
    insights: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    degradation_notes: List[str] = field(default_factory=list)
    step_outcomes: List[str] = field(default_factory=list)
    step_results: List[Dict[str, Any]] = field(default_factory=list)  # successful per-step results (for grounding)
    report: Dict[str, Any] = field(default_factory=dict)     # final artefact (report_assembly)

    # --- bounds (see self-heal loop spec) ---
    MAX_ATTEMPTS_PER_STEP: int = 3
    MAX_GLOBAL_ATTEMPTS: int = 15

    def is_done(self) -> bool:
        """Structural done-detection: the finite plan has been fully walked, or
        the global attempt budget is exhausted. No node adds steps mid-run
        except the capped retry router, so infinite exploration is impossible.
        """
        return (
            self.current_step_idx >= len(self.plan)
            or self.global_attempts >= self.MAX_GLOBAL_ATTEMPTS
        )
