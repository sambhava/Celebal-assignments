"""Ollama client — the only egress point, and it stays on localhost.

PRD G4: the system runs entirely on a local open-source model with no external
API calls. This client talks to a local Ollama daemon (default
``http://localhost:11434``) and nothing else. The generated code never reaches
this layer; only prompts do. Code *execution* happens in the sandbox, which has
no network at all.

Kept deliberately thin and dependency-light: a single ``generate()`` call over
Ollama's ``/api/generate`` endpoint, with a hard timeout and a clear, typed
error when the daemon is unreachable or the model isn't pulled — so callers can
degrade honestly instead of hanging.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_MODEL = os.environ.get("COPILOT_LLM_MODEL", "qwen2.5-coder:3b")
DEFAULT_HOST = os.environ.get("COPILOT_LLM_HOST", "http://localhost:11434")


class LLMUnavailable(RuntimeError):
    """The local model/daemon could not be reached or the model isn't pulled.

    Raised (not swallowed) so the code_gen node can turn it into an honest
    degradation rather than a silent empty result.
    """


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    # a 3B model on a 4GB card: keep outputs short, decoding tight/deterministic
    temperature: float = 0.1
    top_p: float = 0.9
    num_predict: int = 768          # generated code is small; cap to stay fast
    timeout_s: float = 120.0        # cold model load can take ~2min the first call
    seed: Optional[int] = 0         # reproducible-ish for tests/debugging


class OllamaClient:
    """Minimal Ollama HTTP client. One method: ``generate``."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    def generate(self, prompt: str, system: str = "", *, stop: Optional[list] = None) -> str:
        """Return the model's completion text for a single-turn prompt.

        Raises :class:`LLMUnavailable` on connection/HTTP/timeout errors and on
        a missing model, so the caller can degrade rather than hang.
        """
        # Imported lazily so the core package imports without httpx installed
        # (httpx is in the optional [llm] extra).
        try:
            import httpx
        except ImportError as e:  # pragma: no cover - env guard
            raise LLMUnavailable(
                "httpx is not installed. Install the LLM extra: pip install -e '.[llm]'"
            ) from e

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.num_predict,
                "seed": self.config.seed,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        url = f"{self.config.host.rstrip('/')}/api/generate"
        try:
            resp = httpx.post(url, json=payload, timeout=self.config.timeout_s)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            # 404 from Ollama == model not pulled
            hint = (
                f" — model '{self.config.model}' may not be pulled "
                f"(`ollama pull {self.config.model}`)"
                if e.response is not None and e.response.status_code == 404
                else ""
            )
            raise LLMUnavailable(f"Ollama returned {e.response.status_code}{hint}") from e
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.config.host}. Is `ollama serve` running?"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMUnavailable(
                f"Ollama timed out after {self.config.timeout_s}s "
                "(cold model load can be slow on the first call)."
            ) from e
        except json.JSONDecodeError as e:  # pragma: no cover - defensive
            raise LLMUnavailable("Ollama returned a non-JSON response.") from e

        return data.get("response", "")

    def is_available(self) -> bool:
        """Best-effort liveness probe, used to skip live tests when offline."""
        try:
            import httpx

            resp = httpx.get(f"{self.config.host.rstrip('/')}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False
