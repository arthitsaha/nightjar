"""
Portable build, via Nuitka. `python build.py`

Produces `dist/nightjar-<os>-<arch>/` - a folder you can zip, copy to another
machine of the same OS, and run. No Python needed on the target.

**There is no cross-compilation.** Neither Nuitka nor PyInstaller can build a
macOS binary on Windows or vice versa: they bundle the running interpreter and
link against the host's C runtime. The Mac build has to run on a Mac. That is
why this is one script rather than two - same command, same flags, run once per
platform.

Nuitka rather than PyInstaller because the requirement is that the source not
ship: PyInstaller writes `.pyc` files that `pyinstxtractor` plus a decompiler
turns back into readable Python in about a minute, while Nuitka compiles to C
and then to a native binary, so there is no bytecode to recover. It is slower
to build and fussier about native dependencies, which is the trade.

Models are NOT bundled - see paths.py. The build is ~250 MB; the weights are
another ~3.5 GB fetched on first run into the user data dir.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "nightjar.py"
OUT = ROOT / "dist"

# Four components, because Windows file-version metadata demands that shape.
# Bump before cutting a build you hand to someone: it is what a tester quotes
# back when reporting a bug, and "the one from Tuesday" is not a version.
VERSION = "0.1.0.0"

IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"

# Pulled in dynamically, so Nuitka cannot see them by following imports.
# Each one here was a "ModuleNotFoundError" in a finished build, not a guess.
HIDDEN = [
    "onnx_asr",
    "sounddevice",
    "soundfile",
    "soxr",
    "numpy",
    "requests",
    "pyperclip",
    "sqlite_vec",
    "tokenizers",
    "huggingface_hub",
    "memory",
    "memory.mcp",
    # Reached only through `runpy.run_module`, which Nuitka cannot follow. Its
    # absence is not a missing-module error at startup - the connector window
    # is a child process - it shows up as pywebview being "missing", because
    # nothing in the visible import graph ever mentions webview and the plugin
    # that handles it therefore never runs.
    "memory.app",
    "memory.server",
    # Imported inside functions, so following imports does not reach them.
    "bootstrap",
    "paths",
    "envkeys",
]
HIDDEN += ["keyboard"] if IS_WIN else ["pynput"]

# Optional: absent in a minimal install, and the app degrades cleanly without
# them, so a missing one must not fail the build.
#
# `webview` is deliberately NOT here. Nuitka ships a pywebview plugin that
# excludes the platform backends for other OSes, and an explicit include of
# the package contradicts it - the build dies with "Conflict between user and
# plugin decision for module 'webview.platforms.android'". Let the plugin own
# it; it is followed from the import in memory/app.py anyway.
OPTIONAL = ["mcp", "fastapi", "uvicorn", "kokoro_onnx"]

# Plugins Nuitka needs told about explicitly. The overlay is Tk on Windows and
# Linux, and its TCL runtime is data rather than an import, so nothing in the
# import graph reveals that it is needed.
PLUGINS = ["tk-inter"]

# Read-only files that ship beside the binary.
DATA_DIRS = [("memory/static", "memory/static")]
DATA_FILES = [("config.json", "config.json")]

# Never worth compiling into a desktop build; each is minutes of build time.
EXCLUDE = ["tkinter.test", "test", "unittest", "pydoc_data", "setuptools",
           "pip", "nuitka", "IPython", "pytest"]

# Shipped as ordinary Python files rather than compiled in.
#
# Compiling these produces a build that *looks* fine and then fails at load
# with "DLL initialization routine failed" - the binaries are all present, but
# their loader shims call `os.add_dll_directory` at import time and that does
# not survive being turned into C. Left as plain modules they import exactly
# as they do from a venv.
#
# This is also where the source-protection line falls: these are third-party
# packages that are public on PyPI anyway, so shipping them readable gives
# nothing away. Nightjar's own modules stay compiled.
#
# `webview` is deliberately NOT here: Nuitka's pywebview plugin handles it
# correctly when compiled, and excluding it bypasses the plugin and breaks it.
# Tried both ways - compiled works, excluded does not.
AS_DATA = ["onnxruntime"]

# Nuitka copies the C runtime it built against - 14.36 here - into the dist
# root, where it shadows the system copy for every DLL loaded afterwards.
# onnxruntime's extension is built against a newer runtime and dies on the old
# one with "DLL initialization routine failed", which names neither the file
# nor the version and cost four build rounds to find. Replacing the bundled
# copies with the (newer) system ones fixes it and keeps the build
# self-contained on machines with no redistributable installed.
VC_RUNTIME = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll",
              "concrt140.dll"]

# Loadable SQLite extensions, which are data rather than imports: the Python
# side compiles in fine and then looks for a sibling DLL that Nuitka had no
# reason to copy. The failure is quiet - the store falls back to a brute-force
# scan and only gets slow - so it is worth asserting rather than eyeballing.
SIDECAR_FILES = [("sqlite_vec", "vec0.dll" if IS_WIN
                  else "vec0.dylib" if IS_MAC else "vec0.so")]

# Standalone executables the app shells out to, copied beside the binary
# because `resolve_command` looks there. The Gmail connector runs
# `uvx workspace-mcp`, so without these it fails with a bare "[WinError 2] The
# system cannot find the file specified" that names nothing. They are Rust
# binaries with no dependencies, so copying them is the whole job.
TOOLS = ["uv", "uvx"]


def site_packages() -> Path:
    """The venv's site-packages, whichever layout this platform uses."""
    win = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
    if win.is_dir():
        return win
    import sysconfig

    return Path(sysconfig.get_paths()["purelib"])


def copy_uncompiled(dest: Path) -> None:
    """
    Drop the AS_DATA packages into the finished build, verbatim.

    Whole directory including `.py`, so the package imports at runtime exactly
    as it does from a venv - which is the point of not compiling it.
    """
    for package in AS_DATA:
        source = site_packages() / package
        if not source.is_dir():
            print(f"  note: {package} not installed - skipping")
            continue
        target = dest / package
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        print(f"  copied {package} uncompiled "
              f"({sum(1 for _ in target.rglob('*'))} files)")


def copy_sidecars(dest: Path) -> None:
    """Copy loadable extension DLLs that Nuitka does not treat as imports."""
    for package, filename in SIDECAR_FILES:
        source = site_packages() / package / filename
        if not source.exists():
            print(f"  note: {package}/{filename} not found - skipping")
            continue
        target = dest / package / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"  copied {package}/{filename}")


def copy_package_data(dest: Path) -> None:
    """
    Mirror every non-Python file from site-packages, and the `.dist-info`
    directories with them.

    Nuitka compiles the code and leaves the data behind, and the resulting
    failures are individually cryptic and endless to chase one at a time:
    `fbanks.npz` missing killed speech-to-text at startup, `config.json`
    killed TTS, then `language_tags/data/json/index.json` killed it again,
    then `importlib.metadata` could not find kokoro-onnx because the
    `.dist-info` was gone. Copying the lot costs disk and ends the whole
    category.

    Only files that are not already present are copied, so nothing Nuitka
    placed deliberately gets overwritten.
    """
    site = site_packages()
    skip = {"pip", "setuptools", "nuitka", "wheel", "pkg_resources", "__pycache__"}
    copied = 0
    for package in sorted(p for p in site.iterdir() if p.is_dir()):
        if package.name in skip or package.name.endswith(".egg-info"):
            continue
        if package.name.endswith(".dist-info"):
            target = dest / package.name
            if not target.exists():
                shutil.copytree(package, target)
                copied += 1
            continue
        for item in package.rglob("*"):
            if not item.is_file() or item.suffix in (".py", ".pyc", ".pyi"):
                continue
            if "__pycache__" in item.parts:
                continue
            target = dest / item.relative_to(site)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied += 1
    print(f"  copied {copied} package data files and metadata")


def copy_tools(dest: Path) -> None:
    """Copy the standalone executables the connectors shell out to."""
    scripts = Path(sys.executable).resolve().parent
    if not (scripts / "python.exe").exists() and not (scripts / "python").exists():
        scripts = site_packages().parent.parent / ("Scripts" if IS_WIN else "bin")
    for name in TOOLS:
        exe = name + (".exe" if IS_WIN else "")
        source = scripts / exe
        if not source.exists():
            print(f"  note: {exe} not found in {scripts} - Gmail will not start")
            continue
        shutil.copy2(source, dest / exe)
        print(f"  copied {exe}")


def refresh_vc_runtime(dest: Path) -> None:
    """Swap Nuitka's bundled C runtime for the system one when it is newer."""
    if not IS_WIN:
        return
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for name in VC_RUNTIME:
        bundled, live = dest / name, system / name
        if not bundled.exists() or not live.exists():
            continue
        try:
            shutil.copy2(live, bundled)
            print(f"  refreshed {name} from the system runtime")
        except Exception as exc:
            print(f"  note: could not refresh {name} ({exc})")


def have(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def target_name() -> str:
    arch = {"AMD64": "x64", "x86_64": "x64", "arm64": "arm64",
            "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    osname = "win" if IS_WIN else "macos" if IS_MAC else "linux"
    return f"nightjar-{osname}-{arch}"


def build(jobs: int | None = None, quiet: bool = False) -> Path:
    if not ENTRY.exists():
        raise SystemExit(f"no entry point at {ENTRY}")

    work = ROOT / "build"
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        f"--output-dir={work}",
        "--output-filename=" + ("nightjar.exe" if IS_WIN else "nightjar"),
        "--company-name=Nightjar",
        "--product-name=Nightjar",
        # Nuitka refuses company/product metadata without a version to go with
        # it, and the version is what a tester reports back, so it is not
        # decoration.
        f"--product-version={VERSION}",
        f"--file-version={VERSION}",
        # anti-bloat is on by default in Nuitka 4 and passing it warns.
        "--nofollow-import-to=" + ",".join(EXCLUDE),
    ]
    for plugin in PLUGINS:
        cmd.append(f"--enable-plugin={plugin}")
    if jobs:
        cmd.append(f"--jobs={jobs}")
    if quiet:
        cmd.append("--quiet")

    for module in HIDDEN:
        cmd.append(f"--include-module={module}")
    for module in OPTIONAL:
        if have(module):
            cmd.append(f"--include-module={module}")
        else:
            print(f"  note: {module} not installed - building without it")

    # Excluded from compilation and copied in afterwards by hand.
    #
    # `--include-data-dir` is not enough: it copies data and skips `.py`,
    # so the package arrived with its licence files and no code at all.
    # And Nuitka's deployment mode refuses to import an excluded module at
    # runtime unless that flag is turned off, which is a sensible default and
    # wrong here - excluding it from *compilation* is the entire point.
    for package in AS_DATA:
        if (site_packages() / package).is_dir():
            cmd.append(f"--nofollow-import-to={package}")
    if AS_DATA:
        cmd.append("--no-deployment-flag=excluded-module-usage")

    for src, dst in DATA_DIRS:
        if (ROOT / src).exists():
            cmd.append(f"--include-data-dir={ROOT / src}={dst}")
    for src, dst in DATA_FILES:
        if (ROOT / src).exists():
            cmd.append(f"--include-data-files={ROOT / src}={dst}")

    if IS_MAC:
        # A plain binary cannot ask for microphone or accessibility rights;
        # macOS only prompts for a bundle with the usage strings present, and
        # a global hotkey listener without accessibility silently receives
        # nothing.
        cmd += [
            "--macos-create-app-bundle",
            "--macos-app-name=Nightjar",
            "--macos-app-mode=background",
            "--macos-app-protected-resource="
            "NSMicrophoneUsageDescription:Nightjar transcribes speech you dictate.",
        ]
    if IS_WIN:
        cmd.append("--windows-console-mode=attach")

    cmd.append(str(ENTRY))

    print(f"\n  building {target_name()}")
    print(f"  {len(cmd)} args, this takes a while - Nuitka compiles to C first\n")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"\n  build failed ({proc.returncode})")

    produced = work / ("nightjar.app" if IS_MAC else "nightjar.dist")
    if not produced.exists():
        raise SystemExit(f"  build reported success but {produced} is missing")

    OUT.mkdir(exist_ok=True)
    final = OUT / target_name()
    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(produced), str(final))

    # macOS puts the payload inside the bundle; everywhere else the folder is
    # the payload.
    payload = (final / "Contents" / "MacOS") if IS_MAC else final
    copy_uncompiled(payload)
    copy_sidecars(payload)
    copy_package_data(payload)
    copy_tools(payload)
    refresh_vc_runtime(payload)

    mins = (time.perf_counter() - t0) / 60
    size = sum(f.stat().st_size for f in final.rglob("*") if f.is_file())
    print(f"\n  done in {mins:.1f} min - {final}  ({size / 1e6:.0f} MB)")
    return final


def zip_it(folder: Path) -> Path:
    archive = folder.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    print(f"  zipping -> {archive.name}")
    shutil.make_archive(str(folder), "zip", root_dir=folder.parent,
                        base_dir=folder.name)
    return archive


def main() -> None:
    ap = argparse.ArgumentParser(description="portable Nightjar build")
    ap.add_argument("--jobs", type=int, help="parallel compile jobs")
    ap.add_argument("--zip", action="store_true", help="zip the result")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    folder = build(jobs=args.jobs, quiet=args.quiet)
    if args.zip:
        zip_it(folder)
    print("\n  test it:")
    print(f"    {folder / ('nightjar.exe' if IS_WIN else 'nightjar')} doctor\n")


if __name__ == "__main__":
    main()
