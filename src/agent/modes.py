"""Session modes: two permission profiles and the roles built on them.

A mode names the prompt (``prompts/<mode>.md``) and the model role for routing. The
permissions come only from its profile: "dev" (full) or "ask" (read only). A tool, a
tool source or a sandbox checks the profile, never the role name, so a new role needs
one line here and one prompt file.
"""

from __future__ import annotations

from typing import Literal, get_args

Profile = Literal["dev", "ask"]
SessionMode = Literal["dev", "ask", "arch", "pm", "qa", "devops", "cybersec", "analyst"]

ROLE_PROFILES: dict[str, Profile] = {
    "dev": "dev",
    "ask": "ask",
    "arch": "ask",
    "cybersec": "ask",
    "analyst": "ask",
    "pm": "dev",
    "qa": "dev",
    "devops": "dev",
}
MODES: tuple[str, ...] = get_args(SessionMode)


def tool_profile(mode: str) -> str:
    """Return the permission profile of a mode; an unknown mode stays itself and gets none."""
    return ROLE_PROFILES.get(mode, mode)
