"""Isolate the test run from the developer's .env and runtime database.

Runs before any test module imports ``agent``: settings and the engine are
module-level singletons, so the environment must be prepared first.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_DATA_DIR = Path(tempfile.mkdtemp(prefix="python-agent-tests-"))

for _name in [name for name in os.environ if name.upper().startswith("AGENT_")]:
    del os.environ[_name]

os.environ["AGENT_ENV_FILE"] = ""
os.environ["AGENT_HOST"] = "127.0.0.1"
os.environ["AGENT_DATA_DIR"] = str(_DATA_DIR)
os.environ["AGENT_DATABASE_URL"] = f"sqlite+aiosqlite:///{_DATA_DIR / 'agent.db'}"
os.environ["AGENT_EXECUTION_MODE"] = "embedded"
os.environ["AGENT_API_TOKEN"] = ""
# Empty value: explicit legacy AGENT_LLM_* mode, independent of a local config/models.yaml.
os.environ["AGENT_MODELS_FILE"] = ""
os.environ["AGENT_AUTH_OPEN_BROWSER"] = "false"
os.environ["AGENT_ALLOWED_HOSTS"] = '["testserver", "localhost", "127.0.0.1"]'
os.environ["AGENT_EXTRA_ORIGINS"] = '["http://testserver"]'
