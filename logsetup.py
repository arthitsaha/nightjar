"""
One log file for the whole app.

Nightjar prints a deliberately sparse, pretty transcript to the terminal - that
is the product surface, and burying it in DEBUG lines would ruin it. But when a
connector will not connect or a lookup silently returns nothing, there has to be
somewhere to look. So: the terminal stays clean, and everything goes to
`logs/nightjar.log` at DEBUG regardless. `--debug` additionally mirrors it to the
console.

MCP servers are subprocesses and talk on stderr, which the SDK would otherwise
throw away - `mcp_log_path` gives each one its own file so a server that dies at
startup still leaves its reason behind. That single detail is the difference
between "Connection closed" and "OAuth keys file not found".
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Logs are user data: writable, and beside the graph rather than inside the
# installed app, which on Windows and macOS the user cannot write to. Imported
# lazily-ish because logsetup is imported by nearly everything.
try:
    import paths

    LOG_DIR = paths.data_root() / "logs"
except Exception:
    LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "nightjar.log"

FORMAT = "%(asctime)s %(levelname)-5s %(name)-16s %(message)s"
DATEFMT = "%H:%M:%S"

_configured = False


def configure(debug: bool | None = None) -> Path:
    """
    Set up the 'nightjar' logger tree. Idempotent - call it from anywhere.

    `debug` mirrors the file log to the console; it defaults to the
    NIGHTJAR_DEBUG environment variable so a subprocess inherits the choice.
    """
    global _configured

    if debug is None:
        debug = os.environ.get("NIGHTJAR_DEBUG", "").lower() in ("1", "true", "yes")
    else:
        os.environ["NIGHTJAR_DEBUG"] = "1" if debug else "0"

    logger = logging.getLogger("nightjar")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not _configured:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
        handler.set_name("file")
        logger.addHandler(handler)
        _configured = True

    # Let the libraries talk into the same file. The MCP SDK logs the real
    # cause of an OAuth failure - "OAuth flow error", with the exception - and
    # then re-raises something the caller sees only as a bare CancelledError.
    # Without this the one message that explains the failure is discarded at
    # the exact moment it is needed.
    file_handler = next(h for h in logger.handlers if h.get_name() == "file")
    for name in ("mcp", "httpx", "httpcore"):
        lib = logging.getLogger(name)
        lib.setLevel(logging.DEBUG if name == "mcp" else logging.WARNING)
        if file_handler not in lib.handlers:
            lib.addHandler(file_handler)

    # The SSE back-channel is optional and several servers simply do not have
    # it: Supabase answers the SDK's GET with 405, which it retries twice and
    # logs with a full traceback each time. Four tracebacks per lookup for a
    # thing that is working as intended buried the lines that were not, so
    # this one transport is kept at WARNING while the rest of `mcp` stays at
    # DEBUG - the OAuth detail that only it reports is still wanted.
    logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)

    console = next((h for h in logger.handlers if h.get_name() == "console"), None)
    if debug and console is None:
        console = logging.StreamHandler()
        console.set_name("console")
        console.setFormatter(logging.Formatter("  \033[2m%(name)s: %(message)s\033[0m"))
        logger.addHandler(console)
    if console is not None:
        console.setLevel(logging.DEBUG if debug else logging.WARNING)
        if not debug and console in logger.handlers:
            logger.removeHandler(console)

    return LOG_PATH


def get(name: str) -> logging.Logger:
    """A child logger. Safe to call before `configure`."""
    configure()
    return logging.getLogger(f"nightjar.{name}")


def mcp_log_path(server: str) -> Path:
    """Where one MCP server's stderr is kept."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in server)
    return LOG_DIR / f"mcp-{safe}.log"


def tail(path: str | Path, lines: int = 40) -> str:
    """Last `lines` of a log file, or "" if it is not there yet."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:]).strip()
    except Exception:
        return ""
