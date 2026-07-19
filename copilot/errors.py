"""Typed error taxonomy for the ingest / pre-parse gate.

The self-heal loop routes on error class (see TECHNICAL_DESIGN.md). Two classes
are defined here because they are terminal at the ingest boundary:

* ``MalformedInput``   -- the file is structurally invalid or violates a shape
  invariant. Surfaced to the user; NEVER routed to RAG (RAG over pandas docs
  cannot fix bad data) and NEVER consumes a code-gen attempt.
* ``SecurityViolation`` -- the file tripped a security gate (e.g. a rejected
  archive bomb or a disallowed type). Hard-fails immediately, logged loudly,
  never retried.

Both inherit from ``IngestError`` so callers can catch the boundary as one class
while still distinguishing the two when routing.
"""

from __future__ import annotations

from typing import Optional


class IngestError(Exception):
    """Base class for any terminal failure at the untrusted-file ingest gate."""


class MalformedInput(IngestError):
    """The uploaded file is structurally invalid or breaks a shape invariant.

    Attributes
    ----------
    row_index:
        0-based index of the offending row when the failure is row-local
        (``None`` for whole-file problems such as an empty file).
    """

    #: This error class must not be handed to the RAG-based recovery loop.
    rag_eligible = False
    #: This error class must not consume one of the bounded code-gen attempts.
    consumes_attempt = False

    def __init__(self, message: str, *, row_index: Optional[int] = None) -> None:
        super().__init__(message)
        self.row_index = row_index


class SecurityViolation(IngestError):
    """The uploaded file tripped a security gate. Hard-fail, log loudly."""

    rag_eligible = False
    consumes_attempt = False
