"""R12: harden OOXML XML parsing against entity attacks + pin the backend.

Empirically (openpyxl 3.1.5): load_workbook has NO resolve_entities kwarg, and
its stdlib ElementTree backend EXPANDS internal entities (billion-laughs),
bounded only by memory. External SYSTEM entities are not fetched on stdlib, but
would be if lxml were active -- so the backend must be pinned/asserted, and XML
must be screened for DTDs/entities before parsing.
"""

import pytest

from copilot.errors import MalformedInput, SecurityViolation
from copilot.ingest.xml_hardening import (
    assert_no_resolve_entities_kwarg,
    assert_xml_backend_pinned,
    screen_ooxml_bytes,
)

_BILLION_LAUGHS = b'''<?xml version="1.0"?>
<!DOCTYPE lolz [ <!ENTITY lol "lol"> <!ENTITY lol2 "&lol;&lol;&lol;"> ]>
<root>&lol2;</root>'''

_EXTERNAL_ENTITY = b'''<?xml version="1.0"?>
<!DOCTYPE r [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>
<root>&xxe;</root>'''

_CLEAN = b'<?xml version="1.0"?><sst><si><t>hello</t></si></sst>'


def test_screen_rejects_dtd_billion_laughs():
    with pytest.raises(SecurityViolation):
        screen_ooxml_bytes(_BILLION_LAUGHS)


def test_screen_rejects_external_entity():
    with pytest.raises(SecurityViolation):
        screen_ooxml_bytes(_EXTERNAL_ENTITY)


def test_screen_allows_clean_ooxml():
    screen_ooxml_bytes(_CLEAN)  # no raise


def test_resolve_entities_kwarg_is_fictional():
    # The spec-lie regression guard: passing it must NOT be a valid path.
    pytest.importorskip("openpyxl")
    assert_no_resolve_entities_kwarg()  # asserts openpyxl has no such kwarg


def test_backend_is_pinned_and_asserted():
    pytest.importorskip("openpyxl")
    # Must not raise on a correctly-pinned (stdlib) backend.
    assert_xml_backend_pinned()
