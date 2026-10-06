"""Remove credentials from provider text before it reaches logs, audit or the UI."""

from __future__ import annotations

import re

# Generic credential shapes; the configured key is also removed by exact match.
_SECRET_PATTERNS = (
    re.compile(r"\b(sk|pk|rk|key)-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{8,}=*"),
)


def redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Return the text with known secrets and credential shapes replaced."""
    for secret in secrets:
        if len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text
