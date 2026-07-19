"""Version-locked pandas deprecation cheatsheet — cheap, high-value for weak models.

A 3B code model was trained on a corpus full of pandas 1.x idioms that are
*removed* in pandas 2.x/3.x. Injecting this short list into every code-gen prompt
(TECHNICAL_DESIGN.md §2.5) heads off the single most common class of API_MISUSE
errors *before* they cost a generate->execute->classify cycle. It is also the
first thing rag_recovery consults (§2.6) before falling back to the FAISS index.

Keep this list SHORT and high-frequency. It is a prompt-budget item on a small
context window, not an exhaustive migration guide.
"""

from __future__ import annotations

from typing import List, Tuple

# (removed/deprecated idiom, correct replacement). Ordered by how often a weak
# model emits the wrong form.
DEPRECATIONS: List[Tuple[str, str]] = [
    ("df.append(other)", "pd.concat([df, other], ignore_index=True)"),
    ("df.iteritems()", "df.items()"),
    ("series.iteritems()", "series.items()"),
    ("df.applymap(func)", "df.map(func)"),
    ("df.ix[...]", "df.loc[...] or df.iloc[...]"),
    ("df.mad()", "(df - df.mean()).abs().mean()  # .mad() was removed"),
    ("df.lookup(rows, cols)", "use df.melt / np.take_along_axis"),
    ("pd.np.*", "import numpy as np  # pd.np was removed"),
    ("df.get_value / set_value", "df.at[...] / df.iat[...]"),
    ("read_csv(..., squeeze=True)", "read_csv(...).squeeze('columns')"),
    ("inplace=True (chained)", "reassign: df = df.method(...)  # avoid inplace"),
    ("df.groupby(...).agg({'c': 'mean'}) with deprecated dict-of-lists",
     "df.groupby(...)['c'].mean() or named aggregation"),
]

# Column-name / value gotchas for robustness on messy uploads.
ROBUSTNESS_NOTES: List[str] = [
    "Column names may contain spaces/unicode; index with df['exact name'], not attribute access.",
    "Do not assume dtypes; coerce with pd.to_numeric(col, errors='coerce') / pd.to_datetime(col, errors='coerce').",
    "Never hardcode row counts or column values you have not been shown.",
]


def cheatsheet_block() -> str:
    """Render the cheatsheet as a compact prompt block."""
    lines = ["PANDAS API RULES (this environment is pandas 3.x — older idioms are REMOVED):"]
    for wrong, right in DEPRECATIONS:
        lines.append(f"  - Do NOT use `{wrong}`. Use `{right}`.")
    lines.append("ROBUSTNESS:")
    for note in ROBUSTNESS_NOTES:
        lines.append(f"  - {note}")
    return "\n".join(lines)
