#!/usr/bin/env python3
"""
Update Nightjar in place, with or without git.

    python update.py           fetch the latest and apply it
    python update.py --check   say what would change, touch nothing

If this folder is a git clone and git is installed, this is just `git pull`.
If it is not - a ZIP download, or a machine with no git - the same update is
fetched as a ZIP from GitHub and unpacked over the top.

Your `config.json` is never overwritten. If the shipped defaults change, the
new file is written alongside as `config.json.new` for you to compare.

Runs on any Python 3.9+ with nothing but the standard library, so it works
before (or instead of) the venv.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import io
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REPO = "arthitsaha/nightjar"
BRANCH = "main"
ZIP_URL = f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip"

# Never touched by an update: the venv, git's own metadata, local recordings.
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "wifi-monitor-logs"}
# Yours, not ours - updated only when it does not exist yet.
PRESERVE = {"config.json"}

GREEN, YELLOW, RED, BLUE, DIM, BOLD, OFF = (
    "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"
)


def enable_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(handle, ctypes.byref(mode)):
            k32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def ok(msg): print(f"    {GREEN}ok{OFF}   {msg}", flush=True)
def warn(msg): print(f"    {YELLOW}warn{OFF} {msg}", flush=True)
def fail(msg): print(f"    {RED}fail{OFF} {msg}", flush=True)
def dim(msg): print(f"    {DIM}{msg}{OFF}", flush=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# the git path
# --------------------------------------------------------------------------


def is_git_clone() -> bool:
    return (ROOT / ".git").exists() and shutil.which("git") is not None


def git_update(check_only: bool) -> tuple[bool, bool]:
    """Return (handled here, anything changed)."""
    def git(*args, **kw):
        return subprocess.run(["git", "-C", str(ROOT), *args],
                              capture_output=True, text=True, **kw)

    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        return False, False

    # Untracked files never block a fast-forward, so only modifications to
    # tracked files count as dirty here.
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        warn("you have local changes, so nothing was pulled:")
        for line in dirty.splitlines()[:10]:
            dim(f"  {line}")
        dim("commit or stash them, then run this again")
        return True, False

    if git("fetch", "--quiet").returncode != 0:
        warn("could not reach GitHub - check your network")
        return True, False

    behind = git("rev-list", "--count", "HEAD..@{u}").stdout.strip()
    if behind in ("", "0"):
        ok("already up to date")
        return True, False

    log = git("log", "--oneline", "HEAD..@{u}").stdout.strip()
    print(f"\n  {BOLD}{behind} new commit(s){OFF}")
    for line in log.splitlines():
        dim(f"  {line}")

    if check_only:
        dim("\n--check given, nothing pulled")
        return True, False

    pull = git("pull", "--ff-only")
    if pull.returncode != 0:
        fail("git pull failed")
        print(pull.stderr.strip())
        return True, False
    ok("updated")
    return True, True


# --------------------------------------------------------------------------
# the no-git path
# --------------------------------------------------------------------------


def download_zip() -> zipfile.ZipFile:
    dim(f"downloading {ZIP_URL}")
    request = urllib.request.Request(ZIP_URL, headers={"User-Agent": "nightjar-update"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    dim(f"{len(payload) / 1024:.0f} KB")
    return zipfile.ZipFile(io.BytesIO(payload))


def zip_update(check_only: bool) -> bool:
    try:
        archive = download_zip()
    except urllib.error.HTTPError as exc:
        fail(f"GitHub returned {exc.code} - is the repository still public?")
        sys.exit(1)
    except (urllib.error.URLError, OSError) as exc:
        fail(f"download failed ({exc}) - check your network")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        archive.extractall(tmp)
        roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if not roots:
            fail("the archive was empty")
            sys.exit(1)
        source = roots[0]

        changed: list[str] = []
        added: list[str] = []
        config_differs = False

        for path in sorted(source.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(source)
            if any(part in SKIP_DIRS for part in relative.parts):
                continue

            target = ROOT / relative

            if relative.name in PRESERVE and target.exists():
                # Keep the user's file; offer the new default beside it. Yours
                # will almost always differ from the shipped defaults, so the
                # question is not "does it differ" but "have the defaults
                # changed since the last time we offered them" - otherwise
                # every single update nags about the same file.
                offered = target.with_suffix(target.suffix + ".new")
                if digest(path) == digest(target):
                    if offered.exists() and not check_only:
                        offered.unlink()          # nothing left to compare
                elif not (offered.exists() and digest(offered) == digest(path)):
                    config_differs = True
                    if not check_only:
                        shutil.copy2(path, offered)
                continue

            if not target.exists():
                added.append(str(relative))
            elif not filecmp.cmp(path, target, shallow=False):
                changed.append(str(relative))
            else:
                continue

            if not check_only:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                # Zip archives carry no executable bit; put it back so
                # ./run.sh keeps working after an update.
                if target.suffix == ".sh":
                    target.chmod(target.stat().st_mode | 0o111)

        if not changed and not added and not config_differs:
            ok("already up to date")
            return False

        for name in added:
            print(f"    {GREEN}new{OFF}  {name}")
        for name in changed:
            print(f"    {BLUE}upd{OFF}  {name}")
        if config_differs:
            note = "config.json.new written beside yours" if not check_only else \
                   "config.json defaults changed"
            warn(f"your config.json was left alone - {note}")

        if check_only:
            dim("\n--check given, nothing written")
            return False

        touched = len(changed) + len(added)
        if touched:
            ok(f"updated {touched} file(s)")
        if any(n in ("requirements.txt", "install.py") for n in changed + added):
            warn("dependencies changed - re-run the installer:")
            dim("  install.bat        (Windows)")
            dim("  ./install.sh       (macOS / Linux)")
        return bool(touched)


def main() -> None:
    enable_ansi()
    parser = argparse.ArgumentParser(description="Update Nightjar")
    parser.add_argument("--check", action="store_true",
                        help="report what would change without writing anything")
    args = parser.parse_args()

    print(f"\n{BOLD}  Nightjar updater{OFF}")
    print(f"{DIM}  {ROOT}{OFF}\n")

    if is_git_clone():
        dim("git clone detected")
        handled, changed = git_update(args.check)
        if handled:
            if changed:
                print("\n  Restart Nightjar for the update to take effect.")
            print()
            return
        warn("git could not handle this folder - falling back to the ZIP")

    dim("no git clone here - updating from the ZIP on GitHub")
    if zip_update(args.check):
        print("\n  Restart Nightjar for the update to take effect.")
    print()


if __name__ == "__main__":
    main()
