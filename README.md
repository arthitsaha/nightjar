# Nightjar — a local voice keyboard for Windows and macOS

A Wispr Flow-style dictation app that runs entirely on your machine. Hold a key,
speak, release — cleaned-up text appears at your cursor in whatever app you're in.

No account, no cloud, no audio ever leaves the computer.

With the optional **memory** engine switched on, that promise is stated
precisely: transcription, the index, embeddings, reranking and the language
models all run on this machine. Connecting a source (Gmail, Slack, a database)
obviously talks to that service — but what it returns is indexed **locally**,
and your data is never sent to a third-party model. Hosted compose is off by
default; if you ever turn it on, only the handful of retrieved snippets for one
query leaves the machine, never your corpus — and never anything the background
indexer reads.

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
| `--tts` | also set up spoken answers (kokoro-onnx, ~330 MB) |
| `--memory` | also set up the context engine (MCP connectors, local retrieval) |

### Updating

```bash
update.bat        # Windows
./update.sh       # macOS / Linux
```

You don't need git for this. In a clone it runs `git pull`; in a folder that
came from the ZIP — or on a machine with no git at all — it fetches the same
update as a ZIP and unpacks it over the top. Either way it prints what changed,
and `--check` shows that without writing anything.

**Your `config.json` is never overwritten.** If the shipped defaults change, the
new version lands beside yours as `config.json.new` to compare — once, not on
every update. `.venv` is left alone too, and you'll be told when dependencies
changed and the installer needs re-running.

### macOS needs two permissions

Global hotkeys and synthetic keystrokes are privileged on macOS. In
**System Settings → Privacy & Security**, add whichever app you launch Nightjar from
(Terminal, iTerm, VS Code) to **both**:

- **Accessibility**
- **Input Monitoring**

Then restart that app. Without these the hotkey silently never fires.

### Platform status

Nightjar is developed and used daily on **Windows**.

On **macOS** it installs and starts on a MacBook Pro. The speech model loads
through CoreML in ~5.6 s, the mic is captured at 48 kHz and resampled, and Right
Option binds correctly. Dictation end-to-end is still unconfirmed, and the
overlay has a known cosmetic problem — see below.

While the model warms, macOS prints a wall of `Context leak detected,
CoreAnalytics returned false`. That is system-framework noise, not Nightjar
failing; startup continues normally underneath it.

### Known macOS issues

**`KeyError: 'AXIsProcessTrusted'` on the listener thread.** pynput asks macOS
whether it is a trusted accessibility client, and on newer pyobjc builds that
lookup fails through the lazy loader — killing the hotkey while the app looks
like it started fine. Nightjar now binds the symbol itself, so this should not
appear; if it does, please open an issue with your Python and pyobjc versions.

**Cleanup silently off.** `Ollama unavailable ... Connection refused` means the
Ollama server isn't running — the menu-bar app isn't started, or it's still
installing. Dictation still works, it just pastes raw transcripts.

`selftest.py` is the quickest health check — it loads the model and times one
inference:

```bash
.venv/bin/python selftest.py
```

## Running it

| | Windows | macOS / Linux |
|---|---|---|
| Start | `run.bat` | `./run.sh` |
| Dictate | hold **Right Ctrl** | hold **Right Option** |
| Quit | `Ctrl+C`, or `Ctrl+Alt+Q` | `Ctrl+C`, or `Cmd+Alt+Q` |

## Memory — "hey jar, ..."

Optional, off by default. Keep dictating, and mid-sentence — without letting
go of the key — say **"hey jar, what was the Ramp invoice amount"**. What you
said before the trigger is typed as usual; the answer is retrieved from your
connected sources and lands at the cursor as a sentence fragment, `$4,820`,
not a paragraph. The same trigger on the ask key speaks the answer instead of
pasting it: the key chooses the destination, the trigger word chooses whether
memory is consulted. Spaces are ignored when matching, so "heyjar" works too,
and `memory.trigger` in `config.json` changes the phrase.

```bash
python install.py --memory          # packages + embedding model + reranker
run.bat                             # keyboard + connector window (./run.sh)
```

The connector window opens with the keyboard and closes with it. It is a
native window (pywebview drives the OS webview — WebView2 on
Windows, WKWebView on macOS; no Electron, no Node). It has two views: the
**connector grid** for adding and syncing sources, and the **memory graph** —
a force-directed map of every entity and fact the engine has extracted, where
clicking a node shows its facts with the exact source snippet each one came
from, superseded facts included. It also shows the log, so a connector that
will not connect says why on the same page as the button. `run.bat --no-ui`
starts the keyboard without the window.

### Connecting a source

Sources are the vendors' own **hosted MCP servers**, so there is nothing to
install — no Node, no `npx`, just an HTTPS URL and a browser sign-in. How much
setup that takes depends on one thing: whether the vendor lets an app register
itself (OAuth *dynamic client registration*). Verified against each server:

| Source | Setup | Why |
|---|---|---|
| **Supabase** | Click Connect, approve in the browser | Registration endpoint present — Nightjar registers itself |
| **Slack** | Create a Slack app once, paste id + secret, then sign in | Slack advertises no registration endpoint |
| **Gmail** | Create a Google OAuth client once, paste id + secret, then sign in | Google advertises no registration endpoint |

For the two that need a client, the card shows the exact steps with links to
each page, and the redirect URI to paste in. That part is once, ever; every
sign-in afterwards is a single browser round trip. The client id is stored with
the connection, the secret and the tokens go to your OS config folder — never
into this repo.

Products that make Gmail feel like one click do it by owning the OAuth client
and putting their servers in the middle of your mail. A local app has no middle
party, which is the whole point here and also the reason for the extra step.

### Setting it up for other people

You can do the console work **once** and let everyone else just sign in. Create
the OAuth client, add each person's Google address under **Audience → Test
users** (up to 100) — **including your own**, because the account that owns the
project is not a test user automatically, and Google will refuse it with
*"can only be accessed by developer-approved testers"* until you add it. Then
copy `oauth_client.example.json` to
`oauth_client.json`, fill in the id and secret, and put that file on each
machine — either beside `nightjar.py` or in the OS config folder. Their Gmail
card then shows a single **Sign in with Google** button and nothing to
configure. Each person still signs in as themselves and Nightjar only ever
reads their own mail.

`oauth_client.json` is a credential and is gitignored. Never commit it.

**The catch, and it is Google's rule, not ours:** while the app stays in
*Testing*, every sign-in expires after **7 days**. So people re-authorise about
weekly. Nightjar recognises an expired token, clears it, and the card asks for
one click rather than reporting an OAuth error. Publishing the app removes the
7-day limit, but Gmail read access is a *restricted* scope and cannot be
published without a paid annual security assessment — so for a household,
weekly re-sign-in is the cheaper trade. (A Google Workspace domain makes the
app "Internal", which is exempt from both, if you ever have one.)

Then set `"enabled": true` in the `"memory"` and `"compose"` blocks of
`config.json`. Sources connect over **MCP** — Gmail, Slack, Supabase and
GitNexus ship as presets, and any other MCP server works via a stdio command
or an HTTP URL. Corpus sources (mail, chat) are synced and indexed into a
local SQLite graph; live sources (databases) are queried at request time and
cached with a timestamp, so offline you get the last known value *labelled
with its age*, never a stale number presented as current.

If memory is unavailable or nothing matches, **the lookup is never pasted** —
"hey jar, what was..." is not dropped into your document. What you dictated
before the trigger still lands, because those are words you actually said.

### Remembering something

Memory reads from your sources, and it also takes dictation — but only when you
ask. Say **"hey jar, remember that the offsite moved to September 25"** and that
one sentence is stored and extracted into the graph; ask for it later and it
comes back. `remember`, `note that`, `make a note`, `store that` and `save that`
all work. Nothing else you dictate is ever kept: memory that fills itself is
memory nobody can predict.

`python -m memory.eval --seed` writes a small test corpus and question set;
`python -m memory.eval` prints recall@k so retrieval changes are measured,
not vibes.

### Using a hosted model instead (optional, off)

Everything above runs on a local 3B. On a 4 GB GPU that answers a factual
lookup in about a second, but a long open-ended answer generates at roughly
10 tokens a second, so it can take ten. If that trade is wrong for you, compose
and ask can go to a hosted model instead — **compose and ask only**. Dictation
cleanup and background fact extraction always stay local, and cannot be
configured otherwise: cleanup runs on everything you type, and extraction reads
your entire mailbox.

Put the key in `.env` beside `nightjar.py` (copy `.env.example`; it is
gitignored, and never goes in `config.json`), then switch it on:

```jsonc
"compose": {
  "hosted_fallback": {
    "enabled": true,
    "mode": "always",        // or "fallback": local first, hosted only if it comes back empty
    "model": "gpt-5-nano",
    "base_url": "https://api.openai.com/v1"
  }
}
```

`run.bat doctor` then reports where the key was found — never the key itself —
and every answer that left the machine is marked `cloud ·` on the overlay and
in the console. That marker is the condition for the feature existing at all.

Any OpenAI-shaped `/chat/completions` endpoint works, so a local vLLM or a
proxy is just a different `base_url`. An eight-chunk lookup is about 4,000
input tokens, which on `gpt-5-nano` is roughly **$0.00025 a request** — about
20,000 lookups for $5. Prompt caching does not apply: the stable prefix is 531
tokens, under the 1,024-token minimum, and the retrieved chunks differ every
time.

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

Everything runs through the one launcher — `run.bat` on Windows, `./run.sh`
elsewhere — so there is a single script to remember.

| Command | What it does |
|---|---|
| `run.bat` | The voice keyboard and the connector window |
| `run.bat --no-ui` | The keyboard on its own |
| `run.bat doctor` | Checks every moving part and names the broken one |
| `run.bat logs -f` | Follows the log while you test |
| `run.bat probe` | Shows whether a key reaches the OS (Windows) |
| `run.bat --debug` | The keyboard, with the log mirrored to the console |
| `run.bat --bench` | Records 6 s and times each stage |
| `run.bat --devices` | Lists microphones |
| `run.bat --no-llm` | Raw transcripts, skips cleanup |
| `diagnose.py` | Compares model/resampling variants on one recording |

Everything is logged to `logs/nightjar.log`, and each MCP server's own output
to `logs/mcp-<name>.log` — which is where a connector that refuses to start
explains itself.

## Configuration

```jsonc
"hotkey": {
  "mode": "hold",              // "hold" = push-to-talk, "toggle" = press on/off
  "key": "right ctrl",         // Windows / Linux
  "key_mac": "right option",   // macOS (most Mac keyboards have no Right Ctrl)
  "scan_code": null            // Windows-only escape hatch, see run.bat probe
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

It's click-through and never takes focus on both platforms, which matters more
than it sounds: a window that takes focus moves the text caret, and the paste
lands somewhere you didn't mean.

Getting there takes a different toolkit on each OS. Windows uses Tk with a
colour-key transparent window plus `WS_EX_TRANSPARENT | WS_EX_NOACTIVATE`. Tk on
macOS can do neither — hence the solid square earlier builds drew there — so
macOS uses a borderless `NSPanel` through pyobjc instead, with
`setIgnoresMouseEvents_` for click-through, an accessory activation policy so it
never becomes the active app, and `setHidesOnDeactivate_(False)` — panels hide
themselves whenever their app goes inactive, and an app that never activates
never gets them back. Both renderers share the same blob maths, so
the motion is identical; only the painting differs. If AppKit is unavailable,
it falls back to Tk and says so.

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
| `memory/` | The context engine — SQLite graph, MCP connectors, retrieval, connector UI |
| `ARCHITECTURE.md` | The full memory/compose design and its reasoning |
| `update.py` | Pulls the latest, with or without git (`update.bat` / `update.sh` wrap it) |
| `diagnose.py` | One recording through four pipelines, for quality problems |
| `selftest.py` | Model load and inference speed on this machine |
| `probe_fn.py` | Prints the scan code your Fn key emits, if any (Windows) |

## License

MIT — see [LICENSE](LICENSE).
