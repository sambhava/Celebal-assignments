"""Scaffold tests for the graph state + node stubs.

These pin the invariants the design depends on, not the (unimplemented) node
bodies: the error taxonomy that gates RAG, and structural done-detection.
"""

from copilot.graph.state import CoPilotState, ErrorClass
from copilot.graph.nodes import NODES


def test_only_api_misuse_is_rag_eligible():
    rag = {e for e in ErrorClass if e.is_rag_eligible}
    assert rag == {ErrorClass.API_MISUSE}


def test_data_and_empty_and_resource_are_not_retryable():
    for e in (ErrorClass.DATA_PROBLEM, ErrorClass.SEMANTIC_EMPTY,
              ErrorClass.RESOURCE_LIMIT, ErrorClass.SECURITY_VIOLATION):
        assert not e.is_retryable
        assert not e.is_rag_eligible


def test_done_when_plan_fully_walked():
    s = CoPilotState(session_id="x", file_path="f", user_question="q",
                     plan=[{"step": 1}], current_step_idx=1)
    assert s.is_done()


def test_done_when_global_budget_exhausted():
    s = CoPilotState(session_id="x", file_path="f", user_question="q",
                     plan=[{"a": 1}] * 5, current_step_idx=0)
    s.global_attempts = s.MAX_GLOBAL_ATTEMPTS
    assert s.is_done()


def test_not_done_mid_plan():
    s = CoPilotState(session_id="x", file_path="f", user_question="q",
                     plan=[{"a": 1}, {"b": 2}], current_step_idx=1)
    assert not s.is_done()


def test_all_eleven_nodes_registered():
    assert len(NODES) == 11
    assert set(NODES) >= {"ingest_and_validate", "sandboxed_execution",
                          "error_classification", "rag_recovery", "report_assembly"}
