"""
Where things live, in source and in a frozen build.

Running from a clone, everything sits beside nightjar.py and that is the right
answer: it is a repo, you can see it, you can delete it. Running from an
installed app it is the wrong answer for everything writable - Program Files
and /Applications are read-only for the user, and a build that writes there
either fails or silently lands in a virtual store.

So one rule: **code and defaults ship with the app; data belongs to the user.**

    resource()   read-only, next to the binary       - default config, web UI
    data()       writable, per-user                  - memory.db, connections
    models()     writable, large, downloaded on demand

`models()` is deliberately user data rather than shipped: the weights are
~3.5 GB and change independently of the code, so bundling them would make
every release a 3.5 GB download to fix a typo. This is the same split Ollama
uses - small installer, models pulled on first use and cached.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "nightjar"


def frozen() -> bool:
    """True when running from a Nuitka or PyInstaller build."""
    return bool(getattr(sys, "frozen", False)
                or "__compiled__" in globals()
                or getattr(sys, "_MEIPASS", None))


def resource_root() -> Path:
    """Where the app's own read-only files are."""
    if getattr(sys, "_MEIPASS", None):          # PyInstaller onefile
        return Path(sys._MEIPASS)
    if frozen():                                 # Nuitka standalone
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def data_root() -> Path:
    """
    Writable per-user directory, created on demand.

    In source checkouts this stays the repo, so a clone behaves exactly as it
    always has and `memory.db` sits where the README says it does.
    """
    if not frozen():
        return Path(__file__).resolve().parent

    override = os.environ.get("NIGHTJAR_HOME")
    if override:
        base = Path(override)
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA")
                    or Path.home() / "AppData" / "Local") / APP
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP
    else:
        base = Path(os.environ.get("XDG_DATA_HOME")
                    or Path.home() / ".local" / "share") / APP
    base.mkdir(parents=True, exist_ok=True)
    return base


def resource(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def data(*parts: str) -> Path:
    path = data_root().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def models(*parts: str) -> Path:
    """Model weights - large, downloaded on first run, never in the installer."""
    base = data_root() / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base.joinpath(*parts)


def config_path() -> Path:
    """
    The user's config, seeded from the shipped default on first run.

    Copied rather than read in place so an app update cannot silently discard
    settings someone has edited, and so the file is somewhere they can find.
    """
    if not frozen():
        return resource("config.json")
    user = data_root() / "config.json"
    if not user.exists():
        shipped = resource("config.json")
        if shipped.exists():
            user.write_text(shipped.read_text(encoding="utf-8"), encoding="utf-8")
    return user
