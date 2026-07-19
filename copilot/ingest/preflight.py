"""Host-side pre-parse gates G1-G4 (untrusted-file structural checks).

These run on the HOST, before any inflating/parsing, because they are what
protect the host. They allocate nothing beyond a bounded head read. Anything
that requires actually parsing bytes happens later, inside the sandbox
(see load.py). See SECURITY.md §5.

Gate map:
  G1  size cap            (streaming/stat, host)
  G2  magic-byte sniff    (first <=512 B: zip / ole2 / json / text)
  G3  extension allowlist + extension<->content cross-check (polyglot defense)
      -- OLE2 legacy .xls rejected by default (macro surface, weak parser)

Returns a ``PreflightResult`` carrying the resolved ``declared_type`` (which
parser the file may go to) plus ``file_meta`` (size, sha256). Never trusts the
extension alone; a mismatch between claimed extension and sniffed content is a
reject, not a guess.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict

from copilot.errors import MalformedInput, SecurityViolation

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # G1
_HEAD_BYTES = 512

# extension -> the content family we require it to sniff as (G3 cross-check)
_EXT_FAMILY = {
    ".csv": "text",
    ".tsv": "text",
    ".txt": "text",
    ".json": "json",
    ".xlsx": "zip",
    ".xlsm": "zip",
}
# extension -> the parser dispatch key used downstream
_EXT_DECLARED = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",
    ".json": "json",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
}


@dataclass
class PreflightResult:
    declared_type: str            # csv | tsv | json | xlsx
    file_meta: Dict[str, object]  # {size, sha256, ext, sniffed_family}


def sniff_family(head: bytes) -> str:
    """Classify a file by its leading bytes: zip | ole2 | json | text."""
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head[:4] == b"\xd0\xcf\x11\xe0":
        return "ole2"
    stripped = head.lstrip()
    if stripped[:1] in (b"{", b"["):
        return "json"
    return "text"


def preflight(path: str) -> PreflightResult:
    """Run G1-G4 on ``path``. Raise on any violation, else return a result.

    Raises
    ------
    SecurityViolation
        Oversized file, or a rejected type (OLE2 legacy .xls).
    MalformedInput
        Disallowed extension, or extension/content mismatch (polyglot).
    """
    # G1 — size cap (stat, no read)
    size = os.path.getsize(path)
    if size > MAX_UPLOAD_BYTES:
        raise SecurityViolation(
            "File is {} bytes; exceeds the {}-byte upload cap.".format(size, MAX_UPLOAD_BYTES)
        )
    if size == 0:
        raise MalformedInput("File is empty (0 bytes).")

    ext = os.path.splitext(path)[1].lower()
    if ext not in _EXT_FAMILY:
        raise MalformedInput(
            "Extension {!r} is not allowed; permitted: {}.".format(
                ext, ", ".join(sorted(_EXT_FAMILY))
            )
        )

    # G2 — magic-byte sniff on a bounded head read
    with open(path, "rb") as fh:
        head = fh.read(_HEAD_BYTES)
    family = sniff_family(head)

    # OLE2 legacy .xls: rejected by default regardless of extension.
    if family == "ole2":
        raise SecurityViolation(
            "OLE2 (legacy .xls) files are rejected by default (macro surface, weak parser)."
        )

    # G3 — extension <-> content cross-check (polyglot defense)
    expected = _EXT_FAMILY[ext]
    if family != expected:
        raise MalformedInput(
            "Content/extension mismatch: {} sniffed as {!r} but extension {!r} "
            "expects {!r}. Rejected (polyglot defense).".format(
                os.path.basename(path), family, ext, expected
            )
        )

    sha256 = _sha256(path)
    return PreflightResult(
        declared_type=_EXT_DECLARED[ext],
        file_meta={"size": size, "sha256": sha256, "ext": ext, "sniffed_family": family},
    )


def _sha256(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
