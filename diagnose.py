"""
Find out why transcription quality is poor.

Records you once, then runs the SAME audio through four pipelines and prints
what each produced. Whichever line reads best tells us what to change.

    python diagnose.py            record 8 seconds
    python diagnose.py 12         record 12 seconds
    python diagnose.py --file x.wav

It also writes `diagnose_capture.wav` so you can listen to exactly what the
model was given. If that file sounds bad, the problem is the microphone, not
the model.

Read a script out loud rather than improvising - it makes the four outputs
comparable. Suggested:

    "The quick brown fox jumps over the lazy dog. Please schedule the
     deployment review for Thursday at half past four."
"""

from __future__ import annotations

import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

import onnx_asr

TARGET_SR = 16000
OUT_WAV = "diagnose_capture.wav"


def bar(value: float, width: int = 30) -> str:
    filled = int(max(0.0, min(1.0, value)) * width)
    return "#" * filled + "." * (width - filled)


def record(seconds: float) -> tuple[np.ndarray, int]:
    """Capture at the device's NATIVE rate, so no resampling happens yet."""
    info = sd.query_devices(kind="input")
    native = int(info["default_samplerate"])
    print(f"  device : {info['name']}")
    print(f"  native : {native} Hz")
    print()
    print(f"  Recording {seconds:.0f}s - speak normally, at your usual distance.")
    for n in (3, 2, 1):
        print(f"    {n}...", end="\r", flush=True)
        time.sleep(1)
    print("    GO         ")

    frames = int(seconds * native)
    audio = sd.rec(frames, samplerate=native, channels=1, dtype="float32")
    sd.wait()
    print("    done\n")
    return audio[:, 0], native


def audio_report(audio: np.ndarray, sr: int) -> None:
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    clipped = int(np.sum(np.abs(audio) > 0.99))
    # crude speech-presence measure: frames above a tenth of peak
    win = max(1, sr // 50)
    trimmed = audio[: len(audio) // win * win]
    env = np.abs(trimmed).reshape(-1, win).max(axis=1) if len(trimmed) else np.array([0.0])
    voiced = float(np.mean(env > peak * 0.1)) if peak else 0.0

    print("  AUDIO")
    print(f"    peak     {peak:.3f}  {bar(peak)}")
    print(f"    rms      {rms:.3f}  {bar(rms * 6)}")
    print(f"    voiced   {voiced * 100:.0f}% of the clip")
    print(f"    clipped  {clipped} samples")

    notes = []
    if peak < 0.05:
        notes.append("VERY QUIET - raise the mic level in Windows sound settings")
    elif peak < 0.15:
        notes.append("quiet - consider raising the mic level")
    if clipped > 50:
        notes.append("CLIPPING - lower the mic level")
    if voiced < 0.25:
        notes.append("mostly silence - speak sooner after 'GO'")
    for n in notes:
        print(f"    !! {n}")
    print()


def resample(audio: np.ndarray, src: int, dst: int, how: str) -> np.ndarray:
    if src == dst:
        return audio.astype(np.float32)
    if how == "soxr":
        import soxr
        return soxr.resample(audio, src, dst, quality="VHQ").astype(np.float32)
    # naive nearest-sample decimation, the cheap thing a driver might do
    idx = (np.arange(int(len(audio) * dst / src)) * src / dst).astype(np.int64)
    return audio[np.clip(idx, 0, len(audio) - 1)].astype(np.float32)


def main() -> None:
    args = [a for a in sys.argv[1:]]
    path = None
    seconds = 8.0
    if "--file" in args:
        path = args[args.index("--file") + 1]
    elif args:
        try:
            seconds = float(args[0])
        except ValueError:
            pass

    print()
    print("=" * 68)
    print("  TRANSCRIPTION DIAGNOSTIC")
    print("=" * 68)

    if path:
        audio, native = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        print(f"  file   : {path}  ({len(audio)/native:.1f}s @ {native} Hz)")
    else:
        audio, native = record(seconds)
        sf.write(OUT_WAV, audio, native)
        print(f"  saved  : {OUT_WAV}  <- listen to this")
        print()

    audio_report(audio, native)

    # what the live app currently feeds the model: the driver resamples to 16k
    driver_16k = resample(audio, native, TARGET_SR, "naive")
    # what a proper polyphase filter produces instead
    soxr_16k = resample(audio, native, TARGET_SR, "soxr")

    print("  Loading models (fp32 download is ~2.5GB the first time)...")
    variants = []
    for quant in ("int8", None):
        try:
            t0 = time.perf_counter()
            model = onnx_asr.load_model(
                "nemo-parakeet-tdt-0.6b-v2",
                **({"quantization": quant} if quant else {}),
            )
            print(f"    {quant or 'fp32':<5} loaded in {time.perf_counter()-t0:.1f}s")
            variants.append((quant or "fp32", model))
        except Exception as exc:
            print(f"    {quant or 'fp32':<5} FAILED: {exc}")
    print()

    print("=" * 68)
    print("  RESULTS")
    print("=" * 68)
    for name, model in variants:
        for label, wav in (("driver-resample", driver_16k), ("soxr-resample", soxr_16k)):
            t0 = time.perf_counter()
            try:
                text = model.recognize(wav, sample_rate=TARGET_SR)
            except Exception as exc:
                text = f"<error: {exc}>"
            dt = (time.perf_counter() - t0) * 1000
            print(f"\n  [{name} + {label}]  {dt:.0f}ms")
            print(f"  {text}")

    print()
    print("=" * 68)
    print("  Compare the four lines above and tell me which is closest to what")
    print("  you actually said. That identifies whether the loss is coming from")
    print("  int8 quantisation, from resampling, or from the microphone itself.")
    print("=" * 68)


if __name__ == "__main__":
    main()
