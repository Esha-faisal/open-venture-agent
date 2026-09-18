"""Normalize model/source text that arrives with broken accents or C1 junk."""

from __future__ import annotations

import re
import unicodedata

# C1 controls (U+0080–U+009F) often appear when an accented letter like é
# is mangled in transit — e.g. "Café" → "Caf" (U+0081).
_C1_CONTROLS = re.compile(r"[\u0080-\u009f]")

# Common UTF-8-read-as-Latin-1 mojibake for café / Café.
_MOJIBAKE_MAP = (
    ("CafÃ©", "Cafe"),
    ("cafÃ©", "cafe"),
    ("CAFÃ‰", "CAFE"),
    ("CafÃ¨", "Cafe"),
    ("cafÃ¨", "cafe"),
)

# "Caf" followed by a lone broken/control char or a real é → plain "Cafe".
_CAF_BROKEN = re.compile(r"([Cc][Aa][Ff])(?:[\u0080-\u009f\u00e8\u00e9\u00ea\u00eb]|Ã.|â.)")


def sanitize_display_text(text: str | None) -> str:
    """Fix mojibake and strip C1 controls so UI text stays readable."""
    if text is None:
        return ""
    cleaned = str(text)
    if not cleaned:
        return ""

    for bad, good in _MOJIBAKE_MAP:
        cleaned = cleaned.replace(bad, good)

    cleaned = _CAF_BROKEN.sub(r"\1e", cleaned)
    cleaned = _C1_CONTROLS.sub("", cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned)
    # Prefer ASCII "Cafe" over accented forms for consistent display.
    cleaned = cleaned.replace("Café", "Cafe").replace("café", "cafe")
    cleaned = cleaned.replace("CAFÉ", "CAFE")
    return cleaned
