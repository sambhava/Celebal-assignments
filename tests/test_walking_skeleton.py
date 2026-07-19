"""End-to-end walking-skeleton tests: real files flow ingest -> schema+profile,
and adversarial inputs are rejected with typed errors (graceful failure = pass).
"""

import json

import pytest

from copilot.errors import MalformedInput, SecurityViolation
from copilot.graph.run import run_walking_skeleton
from copilot.ingest.preflight import MAX_UPLOAD_BYTES, preflight


# --- happy paths --------------------------------------------------------------

def test_csv_flows_to_schema_and_profile(tmp_path):
    pytest.importorskip("pandas")
    p = tmp_path / "sales.csv"
    p.write_text("region,revenue,date\nWest,100,2024-01-01\nEast,200,2024-02-01\n")
    state = run_walking_skeleton(str(p), "revenue by region")
    assert state.profile["row_count"] == 2
    assert state.profile["col_count"] == 3
    assert set(state.schema) == {"region", "revenue", "date"}
    assert state.schema["revenue"]["dtype_hint"] in ("integer", "float")
    assert state.schema["date"]["dtype_hint"] == "datetime"
    assert state.file_meta["sha256"]  # populated


def test_json_flows_to_profile(tmp_path):
    pytest.importorskip("pandas")
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]))
    state = run_walking_skeleton(str(p))
    assert state.profile["row_count"] == 2
    assert "a" in state.schema and "b" in state.schema


def test_xlsx_flows_to_profile(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    import openpyxl

    p = tmp_path / "d.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["name", "score"])
    ws.append(["alice", 10])
    ws.append(["bob", 20])
    wb.save(str(p))
    state = run_walking_skeleton(str(p))
    assert state.profile["row_count"] == 2
    assert set(state.schema) == {"name", "score"}


# --- adversarial: graceful, typed rejection = PASS ---------------------------

def test_ragged_csv_rejected_r11(tmp_path):
    p = tmp_path / "attack.csv"
    p.write_text('a,b,c\n=cmd|calc,10,20,30\n')  # N+1 fields -> would hit df.index
    with pytest.raises(MalformedInput):
        run_walking_skeleton(str(p))


def test_extension_content_mismatch_rejected(tmp_path):
    # A .csv whose bytes are actually a zip (polyglot) -> reject, don't guess.
    p = tmp_path / "evil.csv"
    p.write_bytes(b"PK\x03\x04rest-of-a-zip")
    with pytest.raises(MalformedInput):
        preflight(str(p))


def test_ole2_xls_rejected(tmp_path):
    p = tmp_path / "legacy.xlsx"  # allowed ext, but OLE2 content
    p.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1padding")
    with pytest.raises(SecurityViolation):
        preflight(str(p))


def test_disallowed_extension_rejected(tmp_path):
    p = tmp_path / "script.exe"
    p.write_bytes(b"MZ\x90\x00")
    with pytest.raises(MalformedInput):
        preflight(str(p))


def test_oversized_file_rejected(tmp_path, monkeypatch):
    # Avoid writing 100MB: shrink the cap for the test.
    import copilot.ingest.preflight as pf
    monkeypatch.setattr(pf, "MAX_UPLOAD_BYTES", 8)
    p = tmp_path / "big.csv"
    p.write_text("a,b,c\n1,2,3\n")  # > 8 bytes
    with pytest.raises(SecurityViolation):
        pf.preflight(str(p))


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_bytes(b"")
    with pytest.raises(MalformedInput):
        preflight(str(p))
