# Building Nightjar

Produces a **portable folder** — copy it to another machine of the same OS,
run it, no Python required.

```bash
python build.py --jobs 6 --zip
```

Output: `dist/nightjar-<os>-<arch>/` and a matching `.zip`.

## There is no cross-compilation

Neither Nuitka nor PyInstaller can build a macOS binary on Windows, or the
reverse. They bundle the *running* interpreter and link against the host's C
runtime. **You must run `build.py` on each OS you want to ship.**

| Target | Build on | Toolchain needed |
|---|---|---|
| Windows x64 | Windows | MSVC build tools (Nuitka offers to fetch a compiler) |
| macOS arm64 | Apple Silicon Mac | Xcode command line tools (`xcode-select --install`) |
| macOS x64 | Intel Mac | same — an arm64 build will not run on Intel |

## Why Nuitka rather than PyInstaller

The requirement is that source not ship. PyInstaller bundles `.pyc` bytecode,
which `pyinstxtractor` plus a decompiler turns back into readable Python in
about a minute. Nuitka compiles to C and then to a native binary, so there is
no bytecode to recover.

It is not encryption, and nothing that runs on someone else's machine can be.
It moves extraction from "one command" to "serious effort", which is the real
bar for a shipped app.

Cost: builds take tens of minutes instead of seconds, and native dependencies
(`onnxruntime`, `numpy`, `tokenizers`) are fussier. That is the trade.

## What is and is not in the build

**In** — the compiled app, CPython, every pure-Python and native dependency,
`config.json` as a default, and the connector UI's static files. About 250 MB.

**Not in** — model weights, about 3.5 GB. They download on first run into the
user data directory, exactly as Ollama caches models. This is deliberate: a
typo fix should not be a 4 GB download, and the weights change independently
of the code.

## First run on a new machine

```bash
nightjar setup      # checks Ollama, pulls models, fetches the reranker
nightjar doctor     # verifies every moving part
nightjar            # run it
```

`setup` runs automatically on the first launch of an installed build if the
models are missing.

**Ollama is a prerequisite.** It ships its own per-platform GPU runtimes and
installs a background service; bundling it would mean shipping CUDA, ROCm and
Metal builds and owning every driver mismatch. `setup` detects it and prints
the download link if it is absent.

## Where the app keeps things

Source checkouts behave exactly as before — everything beside `nightjar.py`.
An installed build cannot write next to its own binary, so:

| | Windows | macOS |
|---|---|---|
| data, config, logs | `%LOCALAPPDATA%\nightjar` | `~/Library/Application Support/nightjar` |
| models | `…\nightjar\models` | `…/nightjar/models` |

Override with `NIGHTJAR_HOME`.

## macOS specifics

The build produces `Nightjar.app` with `NSMicrophoneUsageDescription` set — a
bare binary cannot prompt for microphone access, and a global hotkey listener
without **Accessibility** and **Input Monitoring** permission silently receives
nothing. Grant both in System Settings → Privacy & Security.

**Unsigned builds are for your own testing only.** macOS Sequoia removed the
Control-click override for unsigned apps, so anyone you send it to cannot open
it. Distribution needs an Apple Developer ID certificate and notarization —
that comes after the runtime is proven, not before.

## Reproducibility

The build is deterministic given a pinned toolchain: same Python version, same
`requirements.txt`, same Nuitka version. Bump `VERSION` in `build.py` before
cutting a build you hand to anyone — it is what a tester quotes back, and "the
one from Tuesday" is not a version.
