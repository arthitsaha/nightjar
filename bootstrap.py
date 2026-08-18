"""
First run: make sure the machine has what the app needs, then get out of the way.

The installer stays small and the weights arrive on demand - the same split
Ollama uses, and the reason a bug fix is a 250 MB download rather than a 4 GB
one. Everything here is idempotent and resumable: it checks before it fetches,
so running it twice costs one HTTP HEAD per model, and a killed download
restarts at the file that failed rather than at the beginning.

Ollama is a prerequisite rather than a bundled component. It ships its own
GPU runtimes per platform and installs a background service; vendoring that
inside another installer means shipping CUDA, ROCm and Metal builds and then
owning every driver mismatch. Detect it, and if it is missing say so in one
line with the download link.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import paths

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_DOWNLOAD = "https://ollama.com/download"

# Pulled through Ollama. Cleanup runs on every dictation and extraction reads
# the whole corpus, so both stay local whatever compose is configured to use.
OLLAMA_MODELS = [
    ("qwen2.5:3b-instruct-q4_K_M", "dictation cleanup", True),
    ("nomic-embed-text", "embeddings", True),
    ("qwen3:30b-a3b", "batch fact extraction", False),
]

# Fetched straight from HuggingFace into the user data dir. (repo, file, dest)
ONNX_MODELS = [
    ("Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx",
     "ms-marco-MiniLM-L-6-v2/model.onnx"),
    ("Xenova/ms-marco-MiniLM-L-6-v2", "tokenizer.json",
     "ms-marco-MiniLM-L-6-v2/tokenizer.json"),
]


def _get(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


def ollama_running() -> bool:
    return _get(f"{OLLAMA_HOST}/api/tags") is not None


def ollama_models() -> set[str]:
    data = _get(f"{OLLAMA_HOST}/api/tags") or {}
    return {m.get("name", "") for m in data.get("models", [])}


def pull(model: str) -> bool:
    """
    Ask Ollama to pull a model, streaming so progress is visible.

    A 2 GB download behind a silent prompt is how people conclude the app has
    hung and kill it.
    """
    body = json.dumps({"model": model}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/pull", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            last = ""
            for raw in resp:
                try:
                    line = json.loads(raw)
                except Exception:
                    continue
                status = line.get("status", "")
                total, done = line.get("total"), line.get("completed")
                if total and done:
                    status = f"{status} {done * 100 // total}%"
                if status != last:
                    print(f"\r    {status[:70]:<70}", end="", flush=True)
                    last = status
            print()
        return True
    except Exception as exc:
        print(f"\n    failed: {exc}")
        return False


def fetch_onnx() -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("    huggingface_hub missing - cannot fetch the reranker")
        return False

    ok = True
    for repo, remote, dest in ONNX_MODELS:
        target = paths.models(dest)
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"    {dest} ...", flush=True)
        try:
            got = hf_hub_download(repo, remote, local_dir=str(target.parent.parent))
            src = Path(got)
            if src.resolve() != target.resolve():
                target.write_bytes(src.read_bytes())
        except Exception as exc:
            print(f"      failed: {exc}")
            ok = False
    return ok


def check(cfg: dict | None = None) -> bool:
    """
    Returns True when the app can run. Prints what is missing when it cannot.
    """
    print("\n  Nightjar first-run check\n")
    print(f"    data dir   {paths.data_root()}")
    print(f"    models     {paths.models()}\n")

    if not ollama_running():
        print("  Ollama is not running.\n")
        print("  Nightjar uses it for dictation cleanup and embeddings, which")
        print("  stay on this machine. Install it once, then start Nightjar again:\n")
        print(f"      {OLLAMA_DOWNLOAD}\n")
        return False
    print("    ollama     running")

    have = ollama_models()
    for model, why, required in OLLAMA_MODELS:
        if any(name == model or name.startswith(model.split(":")[0] + ":")
               for name in have):
            print(f"    {model:32} present  ({why})")
            continue
        if not required:
            print(f"    {model:32} optional, skipped  ({why})")
            continue
        print(f"    {model:32} pulling  ({why})")
        if not pull(model):
            return False

    print("\n    reranker")
    fetch_onnx()
    print("\n  ready\n")
    return True


def main() -> int:
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
