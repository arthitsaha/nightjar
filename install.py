#!/usr/bin/env python3
"""
One-shot setup for Nightjar, on Windows or macOS (Linux works too).

    python install.py              full setup
    python install.py --no-models  skip the model downloads
    python install.py --recreate   throw away .venv and start over

Creates an isolated .venv, installs the right dependencies for this OS, pulls
the cleanup model into Ollama, and pre-downloads the speech model so the first
real dictation is not competing with a 600 MB download.

Run it with any Python 3.9+; it does not need to be run from inside a venv.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
IS_LINUX = not IS_WIN and not IS_MAC

COMMON = [
    "onnx-asr[cpu,hub]",
    "sounddevice",
    "numpy",
    "pyperclip",
    "requests",
    "soundfile",
    "soxr",
]
# `keyboard` gives the best low-level hook on Windows but needs root on Unix
# and does not work properly on macOS; pynput covers those instead.
PLATFORM_DEPS = ["keyboard"] if IS_WIN else ["pynput"]

OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"
STT_MODEL = "nemo-parakeet-tdt-0.6b-v2"

GREEN, YELLOW, RED, BLUE, DIM, BOLD, OFF = (
    "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"
)


def enable_ansi() -> None:
    if not IS_WIN:
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


_step = 0


def step(msg: str) -> None:
    global _step
    _step += 1
    print(f"\n{BOLD}[{_step}] {msg}{OFF}", flush=True)


def ok(msg): print(f"    {GREEN}ok{OFF}   {msg}", flush=True)
def warn(msg): print(f"    {YELLOW}warn{OFF} {msg}", flush=True)
def fail(msg): print(f"    {RED}fail{OFF} {msg}", flush=True)
def dim(msg): print(f"    {DIM}{msg}{OFF}", flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def have(exe: str) -> bool:
    return shutil.which(exe) is not None


# --------------------------------------------------------------------------


def check_python() -> None:
    step("Checking Python")
    major, minor = sys.version_info[:2]
    dim(f"{sys.version.split()[0]} at {sys.executable}")
    if (major, minor) < (3, 9):
        fail(f"Python 3.9+ required, this is {major}.{minor}")
        sys.exit(1)
    ok(f"Python {major}.{minor}")


def check_tk() -> None:
    step("Checking Tk (needed for the blob overlay)")
    probe = run([sys.executable, "-c", "import tkinter; tkinter.Tk().destroy()"])
    if probe.returncode == 0:
        ok("tkinter works")
        return
    warn("tkinter is missing or broken - Nightjar will still run with ui.overlay=false")
    if IS_MAC:
        dim("Homebrew Python needs:  brew install python-tk")
    elif IS_LINUX:
        dim("Debian/Ubuntu needs:    sudo apt install python3-tk")


def check_audio_prereqs() -> None:
    if not IS_LINUX:
        return
    step("Checking PortAudio (Linux only)")
    if Path("/usr/include/portaudio.h").exists() or have("pactl"):
        ok("PortAudio present")
    else:
        warn("PortAudio headers not found")
        dim("Debian/Ubuntu needs:    sudo apt install portaudio19-dev python3-tk")


def make_venv(recreate: bool) -> None:
    step("Creating the virtual environment")
    if VENV.exists() and recreate:
        dim(f"removing {VENV}")
        shutil.rmtree(VENV, ignore_errors=True)
    if venv_python().exists():
        ok(f"reusing {VENV}")
        return
    venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    if not venv_python().exists():
        fail("venv creation did not produce an interpreter")
        sys.exit(1)
    ok(f"created {VENV}")


def pip_install() -> None:
    step("Installing Python dependencies")
    py = str(venv_python())
    run([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    packages = COMMON + PLATFORM_DEPS
    dim(f"{len(packages)} packages: {', '.join(packages)}")
    proc = run([py, "-m", "pip", "install", "--quiet", *packages])
    if proc.returncode != 0:
        fail("pip install failed")
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        sys.exit(1)
    ok("dependencies installed")


def setup_ollama(skip: bool) -> bool:
    step("Setting up the cleanup model (Ollama)")
    if not have("ollama"):
        warn("Ollama not found - Nightjar will paste raw transcripts without cleanup")
        dim("Install from https://ollama.com/download, then re-run this script")
        return False

    listing = run(["ollama", "list"])
    if OLLAMA_MODEL.split(":")[0] in listing.stdout and OLLAMA_MODEL in listing.stdout:
        ok(f"{OLLAMA_MODEL} already present")
        return True
    if skip:
        warn(f"skipping pull of {OLLAMA_MODEL} (--no-models)")
        return False

    dim(f"pulling {OLLAMA_MODEL} (~2 GB, one time)")
    proc = subprocess.run(["ollama", "pull", OLLAMA_MODEL])
    if proc.returncode != 0:
        warn("pull failed - cleanup will be skipped until it succeeds")
        return False
    ok(f"{OLLAMA_MODEL} ready")
    return True


def fetch_stt(skip: bool) -> None:
    step("Downloading the speech model")
    if skip:
        warn("skipped (--no-models); it will download on first run instead")
        return
    dim(f"{STT_MODEL} int8 (~600 MB, one time) - this is the slow step")
    code = (
        "import onnx_asr, numpy as np;"
        f"m = onnx_asr.load_model('{STT_MODEL}', quantization='int8');"
        "m.recognize(np.zeros(16000, dtype='float32'), sample_rate=16000);"
        "print('warm')"
    )
    proc = subprocess.run([str(venv_python()), "-c", code])
    if proc.returncode != 0:
        fail("model download failed - check your network and re-run")
        sys.exit(1)
    ok("speech model cached and warmed")


def verify() -> None:
    step("Verifying the install")
    mods = ["onnx_asr", "sounddevice", "numpy", "pyperclip", "requests", "soundfile", "soxr"]
    mods.append("keyboard" if IS_WIN else "pynput")
    code = (
        "import importlib, sys\n"
        f"bad = [m for m in {mods!r} if not importlib.util.find_spec(m)]\n"
        "print('MISSING:' + ','.join(bad) if bad else 'ALL_OK')\n"
    )
    proc = run([str(venv_python()), "-c", code])
    out = (proc.stdout or "").strip()
    if out.endswith("ALL_OK"):
        ok("every dependency imports")
    else:
        fail(out or proc.stderr.strip())
        sys.exit(1)

    devices = run([str(venv_python()), "-c",
                   "import sounddevice as sd; print(sd.query_devices(kind='input')['name'])"])
    if devices.returncode == 0 and devices.stdout.strip():
        ok(f"microphone: {devices.stdout.strip()}")
    else:
        warn("no default input device found - plug in a microphone")


def finish(llm_ready: bool) -> None:
    launcher = "run.bat" if IS_WIN else "./run.sh"
    key = "Right Option" if IS_MAC else "Right Ctrl"
    quit_combo = "Cmd+Alt+Q" if IS_MAC else "Ctrl+Alt+Q"

    print(f"\n{GREEN}{'=' * 62}{OFF}")
    print(f"{GREEN}{BOLD}  Nightjar is ready{OFF}")
    print(f"{GREEN}{'=' * 62}{OFF}")
    print(f"\n  Start it:   {BOLD}{launcher}{OFF}")
    print(f"  Dictate:    hold {BOLD}{key}{OFF}, speak, release")
    print(f"  Quit:       Ctrl+C, or {quit_combo} from anywhere")
    if not llm_ready:
        print(f"\n  {YELLOW}Cleanup is off{OFF} - install Ollama and re-run this script")
    if IS_MAC:
        print(f"\n  {YELLOW}macOS needs two permissions{OFF} before the hotkey works:")
        print("    System Settings > Privacy & Security > Accessibility  -> add Terminal")
        print("    System Settings > Privacy & Security > Input Monitoring -> add Terminal")
        print("    (add whichever app you launch Nightjar from, then restart it)")
    print()


def main() -> None:
    enable_ansi()
    ap = argparse.ArgumentParser(description="Install Nightjar")
    ap.add_argument("--no-models", action="store_true", help="skip model downloads")
    ap.add_argument("--recreate", action="store_true", help="rebuild .venv from scratch")
    args = ap.parse_args()

    system = "macOS" if IS_MAC else ("Windows" if IS_WIN else "Linux")
    print(f"\n{BOLD}  Nightjar installer{OFF} {DIM}- {system}{OFF}")
    print(f"{DIM}  {ROOT}{OFF}")

    check_python()
    check_tk()
    check_audio_prereqs()
    make_venv(args.recreate)
    pip_install()
    llm_ready = setup_ollama(args.no_models)
    fetch_stt(args.no_models)
    verify()
    finish(llm_ready)


if __name__ == "__main__":
    main()
