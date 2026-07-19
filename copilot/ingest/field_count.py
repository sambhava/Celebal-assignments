"""Field-count invariant guard -- remediation for finding R11.

R11: pandas' C parser, given a header of N columns and data rows of N+1 fields,
silently routes the extra LEADING field into ``df.index`` (its implicit-index
heuristic) rather than raising. A formula-defang that scrubs only columns/bodies
misses the index, and ``to_csv()`` re-emits it live -> host-side CSV/formula
injection into the reviewer's spreadsheet.

Empirically (pandas 3.0.2): ``on_bad_lines='error'`` does not fire for this
shape, and ``index_col=False`` silently drops the extra field (data loss) with
only a ``ParserWarning``. So neither pandas knob is a security control.

The sound fix, implemented here: BEFORE pandas parses, walk the file with the
same delimiter dialect pandas will use and assert every record's field count
equals the header's. Any mismatch is a hard ``MalformedInput`` reject -- pandas
never gets the chance to assign an implicit index. The guard is quote- and
delimiter-aware (uses the stdlib ``csv`` reader) so quoted embedded delimiters
and newlines are counted as one field, exactly as pandas counts them.
"""

from __future__ import annotations

import csv
import io
from typing import Iterable, Optional, Sequence, TextIO, Union

from copilot.errors import MalformedInput

DEFAULT_DELIMITER = ","

# Bound the csv reader's field size so a single monster quoted field can't be
# used to exhaust memory inside the guard itself (defense-in-depth with G6).
_MAX_FIELD_BYTES = 1_000_000
csv.field_size_limit(_MAX_FIELD_BYTES)


def _text(source: Union[str, TextIO]) -> str:
    """Return the full text of ``source`` (a path or an already-open stream)."""
    if hasattr(source, "read"):
        return source.read()
    with open(source, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def assert_uniform_field_count(
    source: Union[str, TextIO],
    delimiter: str = DEFAULT_DELIMITER,
) -> int:
    """Assert every record in ``source`` has the header's field count.

    Parameters
    ----------
    source:
        A filesystem path or an open text stream (CSV/TSV/text).
    delimiter:
        The field delimiter. MUST match the ``sep`` handed to pandas, or the
        guard and the parser disagree. Defaults to comma.

    Returns
    -------
    int
        The header field count, on success.

    Raises
    ------
    MalformedInput
        If the input is empty, or any record's field count differs from the
        header's. ``row_index`` on the exception is the 0-based data-row index
        of the first offending row (the header is row -1 / not counted as data).
    """
    text = _text(source)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)

    try:
        header = next(reader)
    except StopIteration:
        raise MalformedInput("File is empty: no header row to validate against.")

    expected = len(header)
    if expected == 0:
        raise MalformedInput("Header row has zero fields.")

    for data_row_index, row in enumerate(reader):
        # A trailing blank line yields an empty record; treat it as benign EOF
        # padding, not a width violation.
        if row == []:
            continue
        if len(row) != expected:
            raise MalformedInput(
                (
                    "Ragged row: header declares {expected} fields but data row "
                    "{n} has {got}. Rejecting to prevent implicit-index type "
                    "confusion (R11); the file is not parsed."
                ).format(expected=expected, n=data_row_index, got=len(row)),
                row_index=data_row_index,
            )

    return expected


def assert_uniform_row_width(
    header: Sequence[object],
    rows: Iterable[Sequence[object]],
) -> int:
    """Excel-path analogue: assert every read row matches the header width.

    openpyxl yields Python lists per row; the same implicit-index confusion does
    not arise from pandas here, but a ragged matrix is still a structural red
    flag (and a spoofed-dimension worksheet can present extra trailing cells), so
    the same invariant is enforced before the frame is built.
    """
    expected = len(list(header))
    if expected == 0:
        raise MalformedInput("Header row has zero fields.")

    for i, row in enumerate(rows):
        if len(list(row)) != expected:
            raise MalformedInput(
                (
                    "Ragged row: header declares {expected} cells but row {n} "
                    "has {got}. Rejecting (R11)."
                ).format(expected=expected, n=i, got=len(list(row))),
                row_index=i,
            )
    return expected


def read_csv_guarded(
    source: Union[str, TextIO],
    delimiter: str = DEFAULT_DELIMITER,
    **read_csv_kwargs: object,
):
    """Guarded ``pd.read_csv``: run the field-count invariant, THEN parse.

    The guard runs first and raises ``MalformedInput`` before pandas can assign
    an implicit index. On success the same ``delimiter`` is passed to pandas as
    ``sep`` so the parse matches what was validated. ``dtype=str`` defers type
    inference (per §3.3) unless the caller overrides it.
    """
    import pandas as pd  # local import: keep the guard importable without pandas

    text = _text(source)
    assert_uniform_field_count(io.StringIO(text), delimiter=delimiter)

    read_csv_kwargs.setdefault("dtype", str)
    return pd.read_csv(io.StringIO(text), sep=delimiter, **read_csv_kwargs)
