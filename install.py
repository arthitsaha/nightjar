#!/usr/bin/env python3
"""
One-shot setup for Nightjar, on Windows or macOS (Linux works too).

    python install.py              full setup
    python install.py --yes        ...without the Ollama install prompt
    python install.py --no-models  skip the model downloads
    python install.py --no-ollama  never install Ollama (no cleanup step)
    python install.py --recreate   throw away .venv and start over

Creates an isolated .venv, installs the right dependencies for this OS,
installs Ollama if it is missing (winget on Windows, Homebrew on macOS, the
official script on Linux) and pulls the cleanup model into it, then
pre-downloads the speech model so the first real dictation is not competing
with a 600 MB download.

Run it with any Python 3.9+; it does not need to be run from inside a venv.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
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
# The native AppKit overlay. pynput drags pyobjc in anyway, but the overlay
# depends on it directly, so ask for it directly.
if IS_MAC:
    PLATFORM_DEPS.append("pyobjc-framework-Cocoa")

OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"
STT_MODEL = "nemo-parakeet-tdt-0.6b-v2"
OLLAMA_HOST = "http://127.0.0.1:11434"

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


def find_ollama() -> str | None:
    """
    Locate the ollama binary.

    shutil.which alone is not enough right after an install: a fresh PATH entry
    does not reach an already-running process (most visibly on Windows), so the
    default install locations are checked too.
    """
    exe = shutil.which("ollama")
    if exe:
        return exe

    candidates: list[Path] = []
    if IS_WIN:
        local = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        if local:
            candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
        candidates.append(Path(program_files) / "Ollama" / "ollama.exe")
    elif IS_MAC:
        candidates += [
            Path("/usr/local/bin/ollama"),
            Path("/opt/homebrew/bin/ollama"),
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
        ]
    else:
        candidates += [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]

    for path in candidates:
        if path.exists():
            return str(path)
    return None


def confirm(question: str, assume_yes: bool) -> bool:
    """Ask before installing system-wide software; default to yes on Enter."""
    if assume_yes:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        # Non-interactive (CI, piped input): never install silently.
        return False
    try:
        return input(f"    {question} [Y/n] ").strip().lower() in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def install_ollama(assume_yes: bool) -> str | None:
    """Install Ollama with the platform's usual package manager."""
    if IS_WIN:
        if not have("winget"):
            warn("winget not available to install Ollama automatically")
            dim("Install from https://ollama.com/download, then re-run this script")
            return None
        label = "winget install Ollama.Ollama"
        cmd = ["winget", "install", "--id", "Ollama.Ollama", "-e", "--silent",
               "--accept-package-agreements", "--accept-source-agreements"]
    elif IS_MAC:
        if not have("brew"):
            warn("Homebrew not available to install Ollama automatically")
            dim("Install from https://ollama.com/download")
            return None
        # The desktop app, not the `ollama` formula: it keeps a server running
        # in the menu bar, so dictation works after a reboot without anyone
        # having to remember `ollama serve`. The cask was renamed from
        # `ollama` to `ollama-app` once the CLI formula took the short name.
        label = "brew install --cask ollama-app"
        cmd = ["brew", "install", "--cask", "ollama-app"]
    else:
        label = "curl -fsSL https://ollama.com/install.sh | sh"
        cmd = ["/bin/sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]

    dim(f"Ollama is not installed. It provides the cleanup model (~2 GB total).")
    if not confirm(f"Run `{label}`?", assume_yes):
        warn("skipped installing Ollama - transcripts will be pasted raw")
        dim("Install it yourself from https://ollama.com/download and re-run this script")
        return None

    dim(f"running: {label}")
    if subprocess.run(cmd).returncode != 0:
        warn("Ollama install failed - transcripts will be pasted raw")
        dim("Install from https://ollama.com/download, then re-run this script")
        return None

    exe = find_ollama()
    if not exe:
        warn("Ollama installed but not on PATH yet")
        dim("Open a new terminal and re-run this script to finish the model pull")
        return None
    ok("Ollama installed")
    return exe


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def start_server(exe: str) -> bool:
    """
    Bring the Ollama server up; `ollama pull` needs it listening.

    The desktop builds start it themselves once the app has been launched
    once, but a fresh headless install has nothing running yet.
    """
    if server_up():
        return True

    dim("starting the Ollama server")
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if IS_WIN:
        # Detach so the server outlives this installer and shows no console.
        kwargs["creationflags"] = 0x00000008 | 0x08000000  # DETACHED_PROCESS | CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    try:
        if IS_MAC and Path("/Applications/Ollama.app").exists():
            subprocess.Popen(["open", "-a", "Ollama"], **kwargs)
        else:
            subprocess.Popen([exe, "serve"], **kwargs)
    except OSError as exc:
        warn(f"could not start the Ollama server ({exc})")
        return False

    for _ in range(30):
        if server_up():
            ok("Ollama server responding")
            return True
        time.sleep(1)

    warn("Ollama server did not come up within 30s")
    dim("Start it yourself with `ollama serve`, then re-run this script")
    return False


def setup_ollama(skip: bool, assume_yes: bool, no_ollama: bool) -> bool:
    step("Setting up the cleanup model (Ollama)")

    exe = find_ollama()
    if not exe:
        if no_ollama:
            warn("Ollama not installed and --no-ollama given - transcripts will be raw")
            return False
        exe = install_ollama(assume_yes)
        if not exe:
            return False
    else:
        ok(f"Ollama found at {exe}")

    if not start_server(exe):
        return False

    listing = run([exe, "list"])
    if OLLAMA_MODEL in listing.stdout:
        ok(f"{OLLAMA_MODEL} already present")
        return True
    if skip:
        warn(f"skipping pull of {OLLAMA_MODEL} (--no-models)")
        return False

    dim(f"pulling {OLLAMA_MODEL} (~1.9 GB, one time)")
    if subprocess.run([exe, "pull", OLLAMA_MODEL]).returncode != 0:
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


KOKORO_FILES = {
    "kokoro-v1.0.onnx":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/voices-v1.0.bin",
}


def setup_tts() -> None:
    """
    Optional speech output: the kokoro-onnx package plus its model files.

    Kept behind a flag because it is only useful with the ask hotkey, and the
    model is another ~330 MB nobody should download by accident.
    """
    step("Setting up speech output (Kokoro)")
    py = str(venv_python())

    proc = run([py, "-m", "pip", "install", "--quiet", "kokoro-onnx"])
    if proc.returncode != 0:
        warn("kokoro-onnx install failed - answers will stay text only")
        print(proc.stderr[-1500:])
        return
    ok("kokoro-onnx installed")

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    for name, url in KOKORO_FILES.items():
        target = models / name
        if target.exists() and target.stat().st_size > 0:
            ok(f"{name} already present")
            continue
        dim(f"downloading {name}")
        # Straight to a temp name so an interrupted download cannot leave a
        # truncated file that looks complete on the next run.
        partial = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(target)
            ok(f"{name} ({target.stat().st_size / 1e6:.0f} MB)")
        except Exception as exc:
            partial.unlink(missing_ok=True)
            warn(f"could not download {name} ({exc})")
            return

    dim('set "enabled": true in the "tts" block of config.json to switch it on')


# `uv` is here for its `uvx` runner: the Gmail preset spawns workspace-mcp in
# an isolated environment because that package pins mcp<2 while Nightjar's
# client needs mcp>=2 - installing it here would downgrade the SDK under us.
MEMORY_PACKAGES = ["sqlite-vec", "mcp", "fastapi", "uvicorn",
                   "tokenizers", "huggingface_hub", "pywebview", "uv"]
EMBED_MODEL = "nomic-embed-text"
RERANKER_DIR_NAME = "ms-marco-MiniLM-L-6-v2"
# The reranker needs an ONNX export; BAAI publishes only PyTorch weights, so
# try known community exports and degrade to passthrough reranking otherwise.
# A 23M cross-encoder, not the 568M one the architecture doc assumed. Measured
# on the reference CPU against a real mailbox: 35 ms per candidate versus
# 513 ms for bge-reranker-v2-m3-int8, for equal or better recall on the same
# pools. Zen 2 has no VNNI, so quantising the big model does not rescue it -
# only a smaller model does. bge-reranker-v2-m3 remains installable by hand
# for multilingual corpora; set memory.rerank_model to its folder name.
RERANKER_SOURCES = [
    ("Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx"),
]
RERANKER_TOKENIZER_REPO = "Xenova/ms-marco-MiniLM-L-6-v2"


def setup_memory() -> None:
    """
    Optional context engine: pip packages, the embedding model, the reranker.

    Everything stays pip + Ollama - no Docker, no services. A failed reranker
    download is not fatal: retrieval runs without it, just less accurately.
    """
    step("Setting up memory (context engine)")
    py = str(venv_python())

    proc = run([py, "-m", "pip", "install", "--quiet", *MEMORY_PACKAGES])
    if proc.returncode != 0:
        warn("memory packages failed to install - memory stays off")
        print(proc.stderr[-1500:])
        return
    ok("memory packages installed (sqlite-vec, mcp, fastapi, uvicorn, tokenizers)")

    exe = find_ollama()
    if exe and server_up():
        listing = run([exe, "list"])
        if EMBED_MODEL in listing.stdout:
            ok(f"{EMBED_MODEL} already present")
        else:
            dim(f"pulling {EMBED_MODEL} (~274 MB, one time)")
            if subprocess.run([exe, "pull", EMBED_MODEL]).returncode != 0:
                warn(f"could not pull {EMBED_MODEL} - retrieval needs it")
    else:
        warn(f"Ollama not running - pull the embedding model later: ollama pull {EMBED_MODEL}")

    target = ROOT / "models" / RERANKER_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    model_file = target / "model.onnx"
    tok_file = target / "tokenizer.json"

    fetch = (
        "from huggingface_hub import hf_hub_download\n"
        "import shutil, sys\n"
        f"shutil.copy(hf_hub_download({RERANKER_TOKENIZER_REPO!r}, 'tokenizer.json'), {str(tok_file)!r})\n"
    )
    if run([py, "-c", fetch]).returncode == 0:
        ok("reranker tokenizer downloaded")
    else:
        warn("could not fetch the reranker tokenizer - reranking disabled")

    if model_file.exists() and model_file.stat().st_size > 0:
        ok("reranker model already present")
    else:
        got = False
        for repo, filename in RERANKER_SOURCES:
            dim(f"trying reranker ONNX from {repo}")
            # These exports keep their weights in a companion `.onnx.data`
            # file - the graph alone is ~100 KB and loads to an error about
            # missing external data. It must land beside the graph under the
            # exact name the graph references, so both are fetched together
            # and straight into the target directory: the HF cache would
            # otherwise hold a second 1.1 GB copy on whichever drive it lives
            # on, which is how this silently failed on a full system disk.
            fetch = (
                "from huggingface_hub import hf_hub_download, list_repo_files\n"
                "import shutil\n"
                f"files = list_repo_files({repo!r})\n"
                f"target = {str(target)!r}\n"
                f"shutil.copy(hf_hub_download({repo!r}, {filename!r}, "
                f"local_dir=target), {str(model_file)!r})\n"
                f"data = {filename!r} + '.data'\n"
                "if data in files:\n"
                f"    hf_hub_download({repo!r}, data, local_dir=target)\n"
            )
            if run([py, "-c", fetch]).returncode == 0:
                ok(f"reranker model downloaded ({repo})")
                got = True
                break
        if not got:
            warn("no reranker ONNX export reachable - retrieval runs without reranking")
            dim("export one yourself (needs torch, one time, any machine):")
            dim("  pip install optimum[exporters] && optimum-cli export onnx "
                f"-m {RERANKER_TOKENIZER_REPO} --task text-classification models/{RERANKER_DIR_NAME}")

    dim('set "enabled": true in the "memory" and "compose" blocks of config.json,')
    dim("then connect sources:  run.bat connectors")
    dim("to check the whole setup at any point:  run.bat doctor")


def verify() -> None:
    step("Verifying the install")
    mods = ["onnx_asr", "sounddevice", "numpy", "pyperclip", "requests", "soundfile", "soxr"]
    mods.append("keyboard" if IS_WIN else "pynput")
    code = (
        # importlib.util must be imported explicitly - `import importlib` alone
        # does not bind the submodule, and whether it happens to be there
        # already varies by platform and Python build.
        "import importlib.util\n"
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
    ap.add_argument("--no-ollama", action="store_true",
                    help="never install Ollama; run without cleanup if it is missing")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="answer yes to the Ollama install prompt")
    ap.add_argument("--tts", action="store_true",
                    help="also set up spoken answers (kokoro-onnx, ~330 MB)")
    ap.add_argument("--memory", action="store_true",
                    help="also set up the context engine (MCP connectors, "
                         "local retrieval)")
    args = ap.parse_args()

    system = "macOS" if IS_MAC else ("Windows" if IS_WIN else "Linux")
    print(f"\n{BOLD}  Nightjar installer{OFF} {DIM}- {system}{OFF}")
    print(f"{DIM}  {ROOT}{OFF}")

    check_python()
    check_tk()
    check_audio_prereqs()
    make_venv(args.recreate)
    pip_install()
    llm_ready = setup_ollama(args.no_models, args.yes, args.no_ollama)
    fetch_stt(args.no_models)
    if args.tts:
        setup_tts()
    if args.memory:
        setup_memory()
    verify()
    finish(llm_ready)


if __name__ == "__main__":
    main()
