# Nightjar — a local voice keyboard for Windows and macOS

A Wispr Flow-style dictation app that runs entirely on your machine. Hold a key,
speak, release — cleaned-up text appears at your cursor in whatever app you're in.

No account, no cloud, no audio ever leaves the computer.

## Install

Get the code, then run the installer for your OS from inside that folder:

```bash
git clone https://github.com/arthitsaha/nightjar.git
cd nightjar
```

No git? Click the green **Code** button at the top of this page → **Download ZIP**,
unzip it, and open a terminal in the unzipped folder.

```bash
# Windows
install.bat

# macOS / Linux
chmod +x install.sh run.sh
./install.sh
```

All you need beforehand is **Python 3.9+**. The installer handles the rest: an
isolated `.venv`, the right dependencies for your OS, **Ollama itself if you
don't already have it**, the cleanup model pulled into it, and the speech model
pre-downloaded so your first dictation isn't racing a 600 MB download.

Ollama is installed with whatever your platform already uses, and it asks first:

| | |
|---|---|
| Windows | `winget install Ollama.Ollama` |
| macOS | `brew install --cask ollama-app` |
| Linux | the official `https://ollama.com/install.sh` |

If that package manager is missing, the installer says so and points you at
[ollama.com/download](https://ollama.com/download) rather than guessing. Either
way the install continues — Ollama is optional, and without it you get raw
transcripts instead of cleaned-up ones.

| Flag | |
|---|---|
| `--yes` / `-y` | don't prompt before installing Ollama |
| `--no-ollama` | never install Ollama; skip cleanup if it's absent |
| `--no-models` | skip the model downloads |
| `--recreate` | rebuild `.venv` from scratch |

### macOS needs two permissions

Global hotkeys and synthetic keystrokes are privileged on macOS. In
**System Settings → Privacy & Security**, add whichever app you launch Nightjar from
(Terminal, iTerm, VS Code) to **both**:

- **Accessibility**
- **Input Monitoring**

Then restart that app. Without these the hotkey silently never fires.

### Platform status

Nightjar is developed and used daily on **Windows**.

On **macOS**, the installer is confirmed working on a MacBook Pro: the venv,
every dependency, and the 661 MB speech model all install and warm correctly.
What is still unconfirmed is the *running* app — Right Option as the hotkey,
`Cmd+V` injection, and the overlay. Reports welcome.

While the model warms, macOS prints a wall of `Context leak detected,
CoreAnalytics returned false`. That is system-framework noise, not Nightjar
failing; the install continues normally underneath it.

If you run it on a Mac, `selftest.py` is the quickest check — it loads the model
and times one inference:

```bash
.venv/bin/python selftest.py
```

An issue reporting either success or failure, with your macOS version and chip,
is genuinely useful.

## Running it

| | Windows | macOS / Linux |
|---|---|---|
| Start | `run.bat` | `./run.sh` |
| Dictate | hold **Right Ctrl** | hold **Right Option** |
| Quit | `Ctrl+C`, or `Ctrl+Alt+Q` | `Ctrl+C`, or `Cmd+Alt+Q` |

## What's under the hood

| Stage | What runs | Where |
|---|---|---|
| Capture | `sounddevice` at the mic's native rate, soxr → 16 kHz | — |
| Transcribe | **NVIDIA Parakeet TDT 0.6B v2** (ONNX, int8) | CPU |
| Clean up | **qwen2.5:3b-instruct-q4_K_M** via Ollama | GPU |
| Insert | clipboard + `Ctrl+V` / `Cmd+V`, clipboard restored | — |

### The flow, key-down to paste

```
  [hold Right Ctrl]                                    nightjar.py
        │
        ▼
  Recorder ──── mic stream is already open, opened at startup ────┐
    arms a buffer, frames accumulate while the key is held        │
    a meter thread pushes the input level to the overlay ─────────┤
        │                                                         │
  [release]                                                       │
        │                                                         │
        ▼                                                    Overlay
  Recorder.stop() → float32 mono @ 16 kHz          (Tk blob, its own
    (native-rate capture, soxr downsample)          event loop on the
        │                                           main thread, fed
        │ shorter than 0.35 s → dropped             by a queue)
        ▼                                                         │
  Transcriber.transcribe()  Parakeet TDT 0.6B v2, ONNX int8, CPU  │
        │  raw: "um so ship it on friday comma if tests pass"     │
        ▼                                                         │
  Cleaner.clean()  POST /api/chat → Ollama, qwen2.5:3b on GPU     │
    system prompt + 7 few-shot pairs + the transcript alone       │
    unreachable, or reply suspiciously long → keep the raw text   │
        │  clean: "So ship it on Friday, if tests pass."          │
        ▼                                                         │
  Injector.send()  save clipboard → copy → Ctrl+V → restore ──────┘
        │
        ▼
  text is in whatever window had focus
```

Threading, since it matters: the Tk overlay owns the main thread, the keyboard
hook runs on its own listener thread, and each dictation is processed on a
throwaway worker thread guarded by a `busy` lock, so a second key press during
processing is ignored rather than queued. Everything the UI shows arrives
through one `queue.Queue`, so no worker ever touches Tk directly.

### Performance

| | |
|---|---|
| Startup | ~4.5 s (model cached) |
| STT, 2 s / 5 s audio | 237 ms / 496 ms (8–10× realtime) |
| LLM cleanup | ~580 ms median |
| **Full dictation** | **~0.8 s** for a short phrase |
| **Total VRAM** | **~2.2 GB** |

All 2.2 GB is the cleanup model — 1.9 GB of weights plus its KV cache, held in
VRAM between dictations by `keep_alive`. Speech recognition runs on the CPU and
uses no VRAM at all. Timings are from a Ryzen 5 4600H / GTX 1650 Ti.

### Why these choices

**Parakeet TDT 0.6B v2, not v3, and not Whisper.** For English-only, v2 *is* the
English model: 6.05% WER on the Open ASR Leaderboard versus 6.34% for the
multilingual v3 and ~7.4% for Whisper large-v3, at a quarter of Whisper's size.

**Parakeet on CPU, LLM on GPU.** On a 4 GB card the two won't sit together
comfortably. Parakeet int8 is fast on six cores; a 3B language model gains far
more from the GPU. Split, neither waits on the other.

**3B for cleanup, not 1.5B.** qwen2.5:1.5b was measured and rejected — it
inverted meanings ("cc sarah" → "No need for Sarah this time"), answered dictated
questions instead of punctuating them, and swapped words.

**No rule-based cleanup** — the LLM does all filler removal, punctuation,
capitalization, spoken punctuation, and self-corrections. If Ollama is
unreachable you get the raw transcript rather than an error.

## If you have less VRAM (Windows)

The 2.2 GB figure is entirely the cleanup model, so every lever below trades
cleanup quality or speed for VRAM. Transcription is unaffected — it is on the
CPU either way, and a machine with no discrete GPU at all can still run Nightjar.

Nothing here needs a code change; edit `config.json`.

| If you have | Do this | Cost |
|---|---|---|
| **~2 GB** | Nothing. Ollama fits what it can and runs the remaining layers on the CPU. | Cleanup slows to roughly 1–2 s. |
| **~1.5 GB** | `"model": "qwen2.5:1.5b-instruct-q4_K_M"` (986 MB) | Real quality loss — see the warning below. |
| **~1 GB** | `"keep_alive": "0"` alongside the 1.5b model — VRAM is released after each dictation instead of held. | Adds a 1–3 s model reload to *every* dictation. |
| **None** | `"enabled": false` in `"llm"`, or run with `--no-llm` | 0 VRAM, 0 cleanup: raw transcripts, no punctuation or filler removal. |

To keep the 3B model's quality with no GPU at all, hide the GPU from the Ollama
server (`CUDA_VISIBLE_DEVICES=` in its environment) and let it run on CPU and
system RAM. It needs about 2.5 GB of RAM and cleanup lands around 2–4 s per
dictation — slow for push-to-talk, but the text quality is identical.

> **The 1.5b downgrade is not free.** It was measured and rejected as the
> default for the reasons in *Why these choices* above: it inverted meanings
> ("cc sarah" → "No need for Sarah this time"), answered dictated questions
> instead of punctuating them, and swapped words. A swapped word is worse than
> no cleanup at all, so if you are choosing between 1.5b and `--no-llm`, try
> `--no-llm` first and see whether raw Parakeet output is good enough. It often is.

`qwen2.5:0.5b-instruct-q4_K_M` (398 MB) exists and will load, but it is not
worth trying — at that size the model rewrites more than it punctuates.

## If transcription quality is poor

Run the diagnostic. It records you once and pushes that *same* audio through four
pipelines so you can see exactly where quality is lost:

```
.venv\Scripts\python.exe diagnose.py        # Windows
.venv/bin/python diagnose.py                # macOS
```

It compares **int8 vs fp32** and **driver resampling vs soxr**, reports mic level
and clipping, and saves `diagnose_capture.wav` so you can listen to what the model
actually received. If that file sounds bad, the problem is the microphone.

Then adjust `config.json`:

- **fp32 wins** → set `"quantization": null` (more accurate, ~2× slower, 2.5 GB download)
- **soxr wins** → already the default (`"resample": "auto"`); `"direct"` reverts
- **audio is quiet or clipping** → fix the input level in your OS sound settings

## Other commands

| Command | What it does |
|---|---|
| `bench.bat` | Records 6 s and times each stage |
| `diagnose.py` | Compares model/resampling variants on one recording |
| `probe.bat` | Shows whether your Fn key reaches the OS (Windows) |
| `nightjar.py --devices` | Lists microphones |
| `nightjar.py --no-llm` | Raw transcripts, skips cleanup |

## Configuration

```jsonc
"hotkey": {
  "mode": "hold",              // "hold" = push-to-talk, "toggle" = press on/off
  "key": "right ctrl",         // Windows / Linux
  "key_mac": "right option",   // macOS (most Mac keyboards have no Right Ctrl)
  "scan_code": null            // Windows-only escape hatch, see probe.bat
},
"stt": {
  "quantization": "int8",      // null = fp32: more accurate, slower
  "resample": "auto",          // "auto" = native capture + soxr; "direct" = driver
  "device": "cpu"              // "cuda" needs onnxruntime-gpu + CUDA 12
},
"llm": { "enabled": true, "model": "qwen2.5:3b-instruct-q4_K_M" },
"ui":  { "overlay": true, "scale": 0.62 }
```

### The blob

A small blob sits in the bottom-right corner. No text — colour and motion carry
the status: dim grey idle, pink swelling with your voice, blue churning with an
orbiting bead while processing, green flash when pasted. `ui.scale` resizes the
whole thing from one number.

On Windows it's click-through and never takes focus. On macOS Tk offers no
click-through, so the blob can catch a click if you aim at it — it stays in the
corner to make that unlikely.

## If you tune the cleanup prompt

`SYSTEM_PROMPT` and `FEW_SHOT` in `nightjar.py` are more delicate than they look.
Four changes were measured and all four backfired:

1. **Stating self-correction as a *rule*** fixed self-corrections but made the
   model rewrite freely — `"before we ship"` became `"before we proceed"`.
   The same behaviour taught by a **single example** works without the side effect.
2. **That example must stay LAST in `FEW_SHOT`.** Moving it into the middle
   brought the word-swapping straight back. The ordering is load-bearing.
3. **Wrapping every turn in an instruction template** crowded out the examples
   and caused word swaps.
4. **Appending a reminder to the live turn** made the model treat the reminder as
   dictated content — `"Comma, the meeting is at noon"`.

The property that matters most is **word fidelity**: a surviving "uh" is
cosmetic, a swapped word is not.

Known imperfection: a leading "Um,"/"Uh," survives in roughly a third of cases.
Every fix cost word fidelity, so it was left alone deliberately.

## Notes and limits

- **Elevated windows (Windows).** Synthetic input is blocked into apps running as
  administrator unless Nightjar is elevated too.
- **Clipboard.** Dictation briefly replaces the clipboard and restores it ~0.5 s
  after pasting.
- **Long dictation.** Parakeet handles a couple of minutes in one pass; beyond
  that you'd want VAD chunking (`onnx_asr.load_vad("silero")`).
- **Quitting.** Tkinter reports exceptions raised in `after` callbacks and keeps
  looping, so `Ctrl+C` used to print a traceback and carry on. SIGINT now routes
  through a real handler, pending timers are daemons, and the keyboard hook is
  released on exit.

## Repo layout

| File | |
|---|---|
| `nightjar.py` | The whole app — capture, STT, cleanup, injection, overlay, hotkeys |
| `config.json` | Everything tunable |
| `install.py` | Venv, dependencies, model downloads (`install.bat` / `install.sh` wrap it) |
| `diagnose.py` | One recording through four pipelines, for quality problems |
| `selftest.py` | Model load and inference speed on this machine |
| `probe_fn.py` | Prints the scan code your Fn key emits, if any (Windows) |

## License

MIT — see [LICENSE](LICENSE).
