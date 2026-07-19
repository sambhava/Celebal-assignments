"""Turn a model completion into runnable, pre-flighted pandas code.

Pipeline (TECHNICAL_DESIGN.md §2.5):
    prompt -> Ollama -> extract fenced python -> static pre-flight -> dedup guard

The static pre-flight is a CHEAP FILTER, not the security boundary — it rejects
obvious non-pandas imports and `os`/`subprocess`/`socket` usage so we don't waste
a sandbox execution on code that could never be legitimate for this task. The
real guarantee is the network-isolated, secret-free sandbox (SECURITY.md §7).

The dedup guard implements the anti-loop: a byte-identical re-emit of code that
already failed is rejected *before* it reaches the sandbox, so a stuck weak model
burns zero executions on repeats.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from copilot.codegen.prompt import SYSTEM_PROMPT, build_prompt
from copilot.llm.client import LLMUnavailable, OllamaClient

# Extract the first ```python ... ``` (or bare ``` ... ```) fenced block.
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Pre-flight: modules a legitimate pandas analysis step never needs. This is a
# convenience filter; the sandbox is the boundary.
_FORBIDDEN_IMPORT = re.compile(
    r"(?m)^\s*(?:import|from)\s+"
    r"(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|http|"
    r"ftplib|smtplib|ctypes|multiprocessing|threading|pickle|marshal)\b"
)
# Direct dangerous calls even without a top-level import (e.g. __import__).
_FORBIDDEN_CALL = re.compile(
    r"\b(__import__|eval|exec|compile|getattr\s*\(\s*__|"
    r"open\s*\(\s*['\"]/(?:etc|root|home)|os\.system|subprocess\.)"
)


class CodeGenError(RuntimeError):
    """Raised when a usable code block could not be produced from the model."""


@dataclass
class PreflightResult:
    ok: bool
    reason: str = ""


def extract_code(completion: str) -> str:
    """Pull the first fenced python block out of the model output.

    Falls back to the whole completion if the weak model forgot the fences but
    emitted only code (best-effort; pre-flight still runs on the result).
    """
    m = _FENCE_RE.search(completion)
    if m:
        return m.group(1).strip()
    # no fence: if it looks like code (has an import or assignment), take it raw
    stripped = completion.strip()
    if re.search(r"(?m)^\s*(import |from |df\s*=|import pandas)", stripped):
        return stripped
    raise CodeGenError("model output contained no usable code block")


def preflight(code: str) -> PreflightResult:
    """Cheap static gate. Returns ok/reason; does NOT raise."""
    if not code.strip():
        return PreflightResult(False, "empty code")
    m = _FORBIDDEN_IMPORT.search(code)
    if m:
        return PreflightResult(False, f"forbidden import: {m.group(1)}")
    if _FORBIDDEN_CALL.search(code):
        return PreflightResult(False, "forbidden call (dynamic import / eval / shell / sensitive path)")
    return PreflightResult(True)


def _hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()


def _prior_hashes(attempt_history: List[Any]) -> set:
    return {_hash(getattr(a, "code", "") or "") for a in attempt_history}


def generate_code(
    step: Dict[str, Any],
    schema: Dict[str, Any],
    level: str,
    *,
    sample: Optional[List[dict]] = None,
    attempt_history: Optional[List[Any]] = None,
    rag_context: str = "",
    client: Optional[OllamaClient] = None,
    max_tries: int = 2,
) -> str:
    """Produce runnable code for a step, or raise CodeGenError/LLMUnavailable.

    ``max_tries`` re-queries the model *within a single code_gen call* when the
    output is unusable (no code block, pre-flight fail, or byte-identical
    re-emit). This is distinct from the outer self-heal loop, which re-queries
    after *execution* fails. Cheap validate-and-repair for a weak model.
    """
    client = client or OllamaClient()
    attempt_history = attempt_history or []
    seen = _prior_hashes(attempt_history)

    prompt = build_prompt(
        step, schema, level,
        sample=sample, attempt_history=attempt_history, rag_context=rag_context,
    )

    last_reason = ""
    for _ in range(max_tries):
        completion = client.generate(prompt, system=SYSTEM_PROMPT, stop=None)
        try:
            code = extract_code(completion)
        except CodeGenError as e:
            last_reason = str(e)
            continue

        pf = preflight(code)
        if not pf.ok:
            last_reason = pf.reason
            # nudge the model with the specific reason on the retry
            prompt = build_prompt(
                step, schema, level,
                sample=sample, attempt_history=attempt_history, rag_context=rag_context,
            ) + f"\n\nYour previous output was rejected: {pf.reason}. Fix it."
            continue

        if _hash(code) in seen:
            last_reason = "byte-identical re-emit of a previously failed attempt"
            prompt += "\n\nThat code was already tried and failed. Produce a DIFFERENT solution."
            continue

        return code

    raise CodeGenError(f"could not produce usable code after {max_tries} tries: {last_reason}")
