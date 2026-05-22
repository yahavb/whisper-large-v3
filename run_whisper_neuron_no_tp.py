"""OpenAI Whisper-large-v3 — PyTorch Native on Neuron (NO TP, single core).

Single NeuronCore inference using torch.compile(backend='neuron') on encoder only.
No tensor parallelism, no sharding, no all_reduce — pure compile test.
Validates output against ground-truth transcriptions using WER.
"""

import os
import sys
import time
import urllib.request

import torch
import torch._dynamo
import torch.nn as nn
import numpy as np

# Increase recompile limit for decoder KV cache guards
torch._dynamo.config.cache_size_limit = 64

TEST_SAMPLES = [
    {
        "url": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
        "ground_truth": "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE",
    },
    {
        "url": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/2.flac",
        "ground_truth": "STUFFILY FLOPPING ONTO THE MATS HAS THE GENTLEMAN BEEN MUCH INCONVENIENCED DURING THE NIGHT",
    },
]

# ═══════════════════════════════════════════════════════════════════════
# Setup single NeuronCore
# ═══════════════════════════════════════════════════════════════════════
torch.neuron.set_device(0)
NEURON_DEVICE = torch.device("neuron")

print("=" * 60)
print("  Whisper-large-v3 — PyTorch Native on Neuron (NO TP)")
print("  Single NeuronCore, no sharding, encoder compile only")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load model (full, unsharded)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n[STEP 1] Loading Whisper-large-v3 (full model, no sharding)...")

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/whisper-large-v3")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).eval().requires_grad_(False)

model.generation_config.max_new_tokens = 256

print(f"  Encoder layers: {len(model.model.encoder.layers)}")
print(f"  Decoder layers: {len(model.model.decoder.layers)}")
print(f"  Attention heads: {model.config.encoder_attention_heads}")
print(f"  d_model: {model.config.d_model}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Move to Neuron + Compile encoder only
# ═══════════════════════════════════════════════════════════════════════
print(f"\n[STEP 2] Moving to Neuron and compiling encoder layers...")

model = model.to(NEURON_DEVICE)

# Compile ONLY encoder layers (fixed mel spectrogram shape)
for i, layer in enumerate(model.model.encoder.layers):
    model.model.encoder.layers[i] = torch.compile(layer, backend='neuron', dynamic=False)

# Decoder layers: run eagerly (KV cache incompatible with static compile)
# proj_out: run eagerly too (to isolate any issues)

print(f"  Encoder: 32 layers compiled with backend='neuron'")
print(f"  Decoder: 32 layers running eagerly")
print(f"  proj_out: running eagerly")

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Download test audio
# ═══════════════════════════════════════════════════════════════════════
print(f"\n[STEP 3] Downloading test audio files...")

import soundfile as sf

test_samples = []
for i, sample_info in enumerate(TEST_SAMPLES):
    url = sample_info["url"]
    ground_truth = sample_info["ground_truth"]

    audio_path = f"/tmp/test_audio_{i}.flac"
    print(f"  Downloading sample {i+1}: {url}")
    urllib.request.urlretrieve(url, audio_path)

    audio_data, sample_rate = sf.read(audio_path, dtype='float32')
    print(f"    Sample rate: {sample_rate}Hz, duration: {len(audio_data)/sample_rate:.2f}s")

    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        print(f"    Resampled to 16kHz")

    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    test_samples.append({"inputs": inputs, "ground_truth": ground_truth})
    print(f"    Ground truth: {ground_truth[:80]}...")

print(f"  Prepared {len(test_samples)} test samples")

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Warmup run (triggers encoder compilation)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n[STEP 4] Warmup run (triggers compilation)...")

warmup_inputs = {k: v.to(dtype=torch.bfloat16, device=NEURON_DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in test_samples[0]["inputs"].items()}

warmup_start = time.time()
with torch.no_grad():
    generated_ids = model.generate(
        **warmup_inputs,
        max_new_tokens=256,
        language="en",
        task="transcribe",
    )
warmup_time = time.time() - warmup_start

transcription = processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
print(f"  Warmup time: {warmup_time:.2f}s (includes encoder compilation)")
print(f"  Transcription: {transcription[:150]}...")

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Timed runs with WER validation
# ═══════════════════════════════════════════════════════════════════════
print(f"\n[STEP 5] Timed inference runs with WER validation...")
print("=" * 60)
from jiwer import wer as compute_wer

NUM_RUNS = 5
run_times = []
wer_scores = []

for run_idx in range(NUM_RUNS):
    sample_idx = run_idx % len(test_samples)
    sample = test_samples[sample_idx]
    inputs = {k: v.to(dtype=torch.bfloat16, device=NEURON_DEVICE) if isinstance(v, torch.Tensor) else v
              for k, v in sample["inputs"].items()}
    ground_truth = sample["ground_truth"]

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            language="en",
            task="transcribe",
        )
    elapsed = time.time() - start
    run_times.append(elapsed)

    transcription = processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
    error_rate = compute_wer(ground_truth.lower(), transcription.lower())
    wer_scores.append(error_rate)

    cached = "✅ cached" if elapsed < warmup_time * 0.5 else "⚠️ possible recompile"
    wer_status = "✅ PASS" if error_rate < 0.10 else "⚠️ HIGH" if error_rate < 0.25 else "❌ FAIL"

    print(f"\n  [RUN {run_idx+1}/{NUM_RUNS}] Sample {sample_idx+1} — {elapsed:.2f}s {cached}")
    print(f"    Expected: {ground_truth[:120]}...")
    print(f"    Got:      {transcription[:120]}...")
    print(f"    WER:      {error_rate:.2%} {wer_status}")

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
avg_time = sum(run_times) / len(run_times)
avg_wer = sum(wer_scores) / len(wer_scores)
all_pass = all(w < 0.10 for w in wer_scores)

print("\n" + "=" * 60)
print("  FINAL SUMMARY — Whisper-large-v3 on Neuron (NO TP)")
print("=" * 60)
print(f"  Model:        openai/whisper-large-v3")
print(f"  Backend:      torch.compile(backend='neuron') encoder only")
print(f"  TP:           NONE (single NeuronCore)")
print(f"  Decoder:      eager (no compile)")
print(f"")
print(f"  ── Performance ──")
print(f"  Warmup:       {warmup_time:.2f}s (includes compilation)")
print(f"  Run times:    {[f'{t:.2f}s' for t in run_times]}")
print(f"  Average:      {avg_time:.2f}s")
print(f"")
print(f"  ── Accuracy (WER) ──")
print(f"  WER scores:   {[f'{w:.2%}' for w in wer_scores]}")
print(f"  Average WER:  {avg_wer:.2%}")
print(f"  All < 10%:    {'✅ PASS' if all_pass else '❌ FAIL'}")
print(f"")

if all_pass:
    print("  ✅ SUCCESS: torch.compile on encoder works correctly without TP!")
    print("  → Issue is in TP sharding logic, not in torch.compile")
else:
    print("  ❌ FAIL: Even without TP, output is incorrect")
    print("  → Issue is in torch.compile itself or model/Neuron interaction")
print("=" * 60)
