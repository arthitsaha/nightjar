"""Downloads the model, then measures load + inference speed on this machine."""

import time
import numpy as np
import onnx_asr

SR = 16000

print("downloading + loading nemo-parakeet-tdt-0.6b-v2 (int8)...", flush=True)
t0 = time.perf_counter()
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8")
print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

t0 = time.perf_counter()
model.recognize(np.zeros(SR, dtype=np.float32), sample_rate=SR)
print(f"warmup: {time.perf_counter() - t0:.2f}s", flush=True)

print("\nspeed on synthetic audio (accuracy needs a real voice):", flush=True)
rng = np.random.default_rng(0)
for secs in (2, 5, 10, 20):
    audio = (rng.standard_normal(SR * secs) * 0.02).astype(np.float32)
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        model.recognize(audio, sample_rate=SR)
        runs.append(time.perf_counter() - t0)
    best = min(runs)
    print(f"  {secs:>3}s audio -> {best * 1000:6.0f}ms   ({secs / best:5.0f}x realtime)", flush=True)

print("\nOK", flush=True)
