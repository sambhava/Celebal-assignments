"""Guarded dataframe loaders (G5-G6 + R11).

In production these run INSIDE the sandbox (parsing untrusted bytes is attack
surface). They apply the R11 field-count invariant for delimited files, a width
invariant for spreadsheets, a depth cap for JSON, and G6 post-parse shape caps.
Everything ingests as ``dtype=str`` — type inference is deferred (§3.3) so a
malicious cell can't drive parse-time coercion.

Dispatch is on the ``declared_type`` resolved by preflight, never re-sniffed
here (that would reopen the type-confusion window).
"""

from __future__ import annotations

import json
from typing import Any, List

from copilot.errors import MalformedInput
from copilot.ingest.field_count import (
    assert_uniform_row_width,
    read_csv_guarded,
)

MAX_ROWS = 5_000_000   # G6
MAX_COLS = 4_096       # G6
MAX_JSON_DEPTH = 64    # G4b (applied here for the skeleton)


def load_dataframe(path: str, declared_type: str):
    """Load ``path`` into a shape-capped, all-string DataFrame.

    Parameters
    ----------
    declared_type:
        One of ``csv | tsv | json | xlsx`` (from preflight). Not re-sniffed.
    """
    if declared_type in ("csv", "tsv"):
        delimiter = "\t" if declared_type == "tsv" else ","
        df = read_csv_guarded(path, delimiter=delimiter)  # R11 guard runs first
    elif declared_type == "json":
        df = _load_json(path)
    elif declared_type == "xlsx":
        df = _load_xlsx(path)
    else:
        raise MalformedInput("Unsupported declared_type {!r}.".format(declared_type))

    _enforce_shape(df)
    return df


def _enforce_shape(df) -> None:
    rows, cols = df.shape
    if rows > MAX_ROWS:
        raise MalformedInput("Row count {} exceeds cap {} (G6).".format(rows, MAX_ROWS))
    if cols > MAX_COLS:
        raise MalformedInput("Column count {} exceeds cap {} (G6).".format(cols, MAX_COLS))


def _json_depth(obj: Any, _depth: int = 0) -> int:
    if _depth > MAX_JSON_DEPTH:
        raise MalformedInput("JSON nesting exceeds depth cap {} (G4b).".format(MAX_JSON_DEPTH))
    if isinstance(obj, dict):
        return max((_json_depth(v, _depth + 1) for v in obj.values()), default=_depth)
    if isinstance(obj, list):
        return max((_json_depth(v, _depth + 1) for v in obj), default=_depth)
    return _depth


def _load_json(path: str):
    import pandas as pd

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    _json_depth(data)  # depth cap before we let pandas normalize/expand
    if isinstance(data, list):
        df = pd.json_normalize(data)
    elif isinstance(data, dict):
        # A single object, or a dict-of-columns; normalize a one-row frame.
        df = pd.json_normalize(data)
    else:
        raise MalformedInput("Top-level JSON must be an object or array.")
    return df.astype(str)


def _load_xlsx(path: str):
    import openpyxl
    import pandas as pd

    _screen_xlsx_xml(path)  # R12: reject DTD/entity XML before openpyxl parses
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_vba=False)
    try:
        ws = wb.active
        rows: List[List[Any]] = [list(r) for r in ws.iter_rows(values_only=True)]
    finally:
        wb.close()

    if not rows:
        raise MalformedInput("Workbook active sheet is empty.")
    header, *data = rows
    assert_uniform_row_width(header, data)  # width invariant (R11 analogue)
    df = pd.DataFrame(data, columns=[str(h) for h in header])
    return df.astype(str)


def _screen_xlsx_xml(path: str) -> None:
    """R12: screen each .xml member of the OOXML zip for DTD/entity payloads."""
    import zipfile

    from copilot.ingest.xml_hardening import screen_ooxml_bytes

    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".xml") or name.lower().endswith(".rels"):
                screen_ooxml_bytes(zf.read(name))
