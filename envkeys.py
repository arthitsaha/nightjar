"""
Where API keys come from, and where they must never be.

(Named `envkeys`, not `secrets`: a top-level `secrets.py` shadows the stdlib
module of that name for the whole process, and the MCP SDK imports it. That
mistake was made here and broke every connector with "the 'mcp' package is not
installed" while pip showed it present. Do not rename this back.)

One rule: a key is read from the environment or from a file the repo ignores,
and never from `config.json`. `config.json` is the file people paste into
issues and screenshot in bug reports, and this repo is public - a key that
lives there leaks by accident, not by attack.

Three sources, first hit wins:

1. the real process environment - right for CI, systemd, or `set OPENAI_API_KEY=`
2. `.env` beside nightjar.py - what most people expect, and gitignored
3. the OS config dir - same place the OAuth tokens live, so a machine-wide
   key survives re-cloning the repo

stdlib only. python-dotenv would do step 2 and nothing else, and the promise
is that `install.bat` stays short.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def config_dir() -> Path:
    """The OS config dir, matching where memory/mcp/registry.py puts tokens."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "nightjar"


def _parse_env_file(path: Path) -> dict[str, str]:
    """
    Minimal .env: KEY=value, one per line.

    Deliberately forgiving about the things people actually type - `export`
    prefixes pasted from a shell, quotes around the value, blank lines and
    `#` comments - and deliberately not a shell: no interpolation, no
    multiline values. A key is one line.
    """
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def get(name: str) -> str | None:
    """The named secret, or None. Never raises, never logs the value."""
    direct = os.environ.get(name)
    if direct:
        return direct.strip()
    for path in (ROOT / ".env", config_dir() / ".env"):
        value = _parse_env_file(path).get(name)
        if value:
            return value.strip()
    return None


def sources(name: str) -> list[str]:
    """Which of the three places currently hold this key - for `doctor`."""
    found = []
    if os.environ.get(name):
        found.append("environment")
    for label, path in (("./.env", ROOT / ".env"),
                        (str(config_dir() / ".env"), config_dir() / ".env")):
        if _parse_env_file(path).get(name):
            found.append(label)
    return found
