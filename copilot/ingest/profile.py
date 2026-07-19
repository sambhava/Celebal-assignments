"""Schema + profile derivation over a loaded (all-string) DataFrame.

Produces the two structures later nodes and the model context rely on:

* ``schema``  -- per column: inferred-ish dtype hint, null count, cardinality,
  and a few sample values (samples are UNTRUSTED and must be wrapped as such by
  the prompt builder, never placed in the instruction channel).
* ``profile`` -- frame-level: row/col counts, duplicate rows, memory.

Cheap, dependency-light, and pure so it is unit-testable without the rest of the
graph.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

_SAMPLE_N = 3


def _dtype_hint(series) -> str:
    """A best-effort human dtype label without coercing the (str) column.

    Ingest keeps everything as str (deferred inference); this only *labels* what
    the column looks like for the profile/UI -- it does not change the data.
    """
    import pandas as pd

    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    coerced_num = pd.to_numeric(non_null, errors="coerce")
    if coerced_num.notna().all():
        return "integer" if (coerced_num % 1 == 0).all() else "float"
    coerced_dt = pd.to_datetime(non_null, errors="coerce", format="mixed")
    if coerced_dt.notna().all():
        return "datetime"
    return "string"


def profile_dataframe(df) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    schema: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        schema[str(col)] = {
            "dtype_hint": _dtype_hint(series),
            "null_count": int(series.isna().sum()),
            "cardinality": int(series.nunique(dropna=True)),
            "samples": [str(v) for v in series.dropna().unique()[:_SAMPLE_N]],
        }

    profile: Dict[str, Any] = {
        "row_count": int(df.shape[0]),
        "col_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }
    return schema, profile
