"""OpenAI Whisper-large-v3 — PyTorch Native on Neuron (TP-2).

Tensor-parallel inference using torch.compile(backend='neuron').
Demonstrates: compile once → reuse for multiple audio inputs.
Validates output against ground-truth transcriptions using WER.

Architecture: Encoder-Decoder (speech-to-text)
  - 32 encoder layers (self-attention + MLP)
  - 32 decoder layers (self-attention + cross-attention + MLP)
  - 20 attention heads → 10 per rank with TP-2
  - d_model=1280, FFN=5120 → 2560 per rank with TP-2
"""

import os
import sys
import time
import urllib.request

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Test audio samples with known ground-truth transcriptions
# From LibriSpeech test-clean (public domain)
# ═══════════════════════════════════════════════════════════════════════

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
# TP Wrapper Modules
# ═══════════════════════════════════════════════════════════════════════

class TPAttention(nn.Module):
    """Wraps attention with all_reduce after row-parallel out_proj."""

    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, *args, **kwargs):
        out = self.attn(*args, **kwargs)
        if isinstance(out, tuple):
            attn_out = out[0]
            dist.all_reduce(attn_out, op=dist.ReduceOp.SUM)
            return (attn_out,) + out[1:]
        else:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out


class TPMLP(nn.Module):
    """Wraps MLP (fc1 + fc2) with all_reduce after row-parallel fc2."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


class TPProjOut(nn.Module):
    """Wraps column-parallel proj_out with all_gather to reconstruct full vocab logits."""

    def __init__(self, proj_out, tp_size):
        super().__init__()
        self.proj_out = proj_out
        self.tp_size = tp_size

    def forward(self, *args, **kwargs):
        local_logits = self.proj_out(*args, **kwargs)
        gathered = [torch.zeros_like(local_logits) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local_logits)
        return torch.cat(gathered, dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# Sharding helpers
# ═══════════════════════════════════════════════════════════════════════

def shard_column(weight, rank, tp):
    """Shard along output dim (dim 0 for nn.Linear weight)."""
    chunk_size = weight.shape[0] // tp
    return weight[rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def shard_row(weight, rank, tp):
    """Shard along input dim (dim 1 for nn.Linear weight)."""
    chunk_size = weight.shape[1] // tp
    return weight[:, rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def shard_bias_column(bias, rank, tp):
    """Shard bias for column-parallel layer."""
    chunk_size = bias.shape[0] // tp
    return bias[rank * chunk_size : (rank + 1) * chunk_size].contiguous()


def shard_encoder_layer(layer, rank, tp):
    """Shard a Whisper encoder layer for TP."""
    attn = layer.self_attn

    # Column-parallel: q_proj, k_proj, v_proj
    attn.q_proj.weight = nn.Parameter(shard_column(attn.q_proj.weight.data, rank, tp), requires_grad=False)
    if attn.q_proj.bias is not None:
        attn.q_proj.bias = nn.Parameter(shard_bias_column(attn.q_proj.bias.data, rank, tp), requires_grad=False)

    attn.k_proj.weight = nn.Parameter(shard_column(attn.k_proj.weight.data, rank, tp), requires_grad=False)
    if attn.k_proj.bias is not None:
        attn.k_proj.bias = nn.Parameter(shard_bias_column(attn.k_proj.bias.data, rank, tp), requires_grad=False)

    attn.v_proj.weight = nn.Parameter(shard_column(attn.v_proj.weight.data, rank, tp), requires_grad=False)
    if attn.v_proj.bias is not None:
        attn.v_proj.bias = nn.Parameter(shard_bias_column(attn.v_proj.bias.data, rank, tp), requires_grad=False)

    # Row-parallel: out_proj
    attn.out_proj.weight = nn.Parameter(shard_row(attn.out_proj.weight.data, rank, tp), requires_grad=False)
    # out_proj bias is NOT sharded (it's after the all_reduce)

    # MLP: fc1 is column-parallel, fc2 is row-parallel
    layer.fc1.weight = nn.Parameter(shard_column(layer.fc1.weight.data, rank, tp), requires_grad=False)
    if layer.fc1.bias is not None:
        layer.fc1.bias = nn.Parameter(shard_bias_column(layer.fc1.bias.data, rank, tp), requires_grad=False)

    layer.fc2.weight = nn.Parameter(shard_row(layer.fc2.weight.data, rank, tp), requires_grad=False)
    # fc2 bias is NOT sharded (it's after the all_reduce)


def shard_decoder_layer(layer, rank, tp):
    """Shard a Whisper decoder layer for TP."""
    # Self-attention
    attn = layer.self_attn
    attn.q_proj.weight = nn.Parameter(shard_column(attn.q_proj.weight.data, rank, tp), requires_grad=False)
    if attn.q_proj.bias is not None:
        attn.q_proj.bias = nn.Parameter(shard_bias_column(attn.q_proj.bias.data, rank, tp), requires_grad=False)

    attn.k_proj.weight = nn.Parameter(shard_column(attn.k_proj.weight.data, rank, tp), requires_grad=False)
    if attn.k_proj.bias is not None:
        attn.k_proj.bias = nn.Parameter(shard_bias_column(attn.k_proj.bias.data, rank, tp), requires_grad=False)

    attn.v_proj.weight = nn.Parameter(shard_column(attn.v_proj.weight.data, rank, tp), requires_grad=False)
    if attn.v_proj.bias is not None:
        attn.v_proj.bias = nn.Parameter(shard_bias_column(attn.v_proj.bias.data, rank, tp), requires_grad=False)

    attn.out_proj.weight = nn.Parameter(shard_row(attn.out_proj.weight.data, rank, tp), requires_grad=False)

    # Cross-attention (encoder_attn)
    xattn = layer.encoder_attn
    xattn.q_proj.weight = nn.Parameter(shard_column(xattn.q_proj.weight.data, rank, tp), requires_grad=False)
    if xattn.q_proj.bias is not None:
        xattn.q_proj.bias = nn.Parameter(shard_bias_column(xattn.q_proj.bias.data, rank, tp), requires_grad=False)

    xattn.k_proj.weight = nn.Parameter(shard_column(xattn.k_proj.weight.data, rank, tp), requires_grad=False)
    if xattn.k_proj.bias is not None:
        xattn.k_proj.bias = nn.Parameter(shard_bias_column(xattn.k_proj.bias.data, rank, tp), requires_grad=False)

    xattn.v_proj.weight = nn.Parameter(shard_column(xattn.v_proj.weight.data, rank, tp), requires_grad=False)
    if xattn.v_proj.bias is not None:
        xattn.v_proj.bias = nn.Parameter(shard_bias_column(xattn.v_proj.bias.data, rank, tp), requires_grad=False)

    xattn.out_proj.weight = nn.Parameter(shard_row(xattn.out_proj.weight.data, rank, tp), requires_grad=False)

    # MLP: fc1 is column-parallel, fc2 is row-parallel
    layer.fc1.weight = nn.Parameter(shard_column(layer.fc1.weight.data, rank, tp), requires_grad=False)
    if layer.fc1.bias is not None:
        layer.fc1.bias = nn.Parameter(shard_bias_column(layer.fc1.bias.data, rank, tp), requires_grad=False)

    layer.fc2.weight = nn.Parameter(shard_row(layer.fc2.weight.data, rank, tp), requires_grad=False)


# ═══════════════════════════════════════════════════════════════════════
# Init distributed
# ═══════════════════════════════════════════════════════════════════════
dist.init_process_group(backend="neuron")
rank = dist.get_rank()
world_size = dist.get_world_size()
TP = world_size
assert TP == 2, f"Expected 2 ranks for TP-2, got {TP}"

torch.neuron.set_device(rank)
NEURON_DEVICE = torch.device("neuron")

if rank == 0:
    print("=" * 60)
    print(f"  Whisper-large-v3 — PyTorch Native on Neuron (TP-{TP})")
    print(f"  World size: {world_size}")
    print(f"  Compile once → reuse for multiple audio inputs")
    print(f"  Validation: WER against ground-truth transcriptions")
    print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load and shard model
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print(f"\n[STEP 1] Loading Whisper-large-v3 on CPU and sharding for TP-{TP}...")

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/whisper-large-v3")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).eval().requires_grad_(False)

# Enable static cache for torch.compile compatibility
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256

# Update config for sharded heads
model.config.encoder_attention_heads = model.config.encoder_attention_heads // TP
model.config.decoder_attention_heads = model.config.decoder_attention_heads // TP

if rank == 0:
    print(f"  Original: 20 heads → {model.config.encoder_attention_heads} heads/rank")
    print(f"  Encoder layers: {len(model.model.encoder.layers)}")
    print(f"  Decoder layers: {len(model.model.decoder.layers)}")
    print(f"  Static cache enabled, max_new_tokens=256")

# Shard encoder layers
for i, layer in enumerate(model.model.encoder.layers):
    shard_encoder_layer(layer, rank, TP)

# Shard decoder layers
for i, layer in enumerate(model.model.decoder.layers):
    shard_decoder_layer(layer, rank, TP)

# Shard proj_out (lm_head equivalent) — column parallel
proj_out = model.proj_out
vocab_size = proj_out.weight.shape[0]
chunk_size = vocab_size // TP
model.proj_out.weight = nn.Parameter(
    model.proj_out.weight.data[rank * chunk_size : (rank + 1) * chunk_size].contiguous(),
    requires_grad=False)
if model.proj_out.bias is not None:
    model.proj_out.bias = nn.Parameter(
        model.proj_out.bias.data[rank * chunk_size : (rank + 1) * chunk_size].contiguous(),
        requires_grad=False)

if rank == 0:
    print(f"  Sharded all encoder + decoder layers for TP-{TP}")
    print(f"  proj_out: {vocab_size} vocab → {chunk_size}/rank")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Move to Neuron + Compile + Wrap with TP
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print(f"\n[STEP 2] Moving to Neuron and compiling...")

model = model.to(NEURON_DEVICE)

# Compile encoder layers
for layer in model.model.encoder.layers:
    layer.self_attn = torch.compile(layer.self_attn, backend='neuron', dynamic=False)
    layer.fc1 = torch.compile(layer.fc1, backend='neuron', dynamic=False)
    layer.fc2 = torch.compile(layer.fc2, backend='neuron', dynamic=False)

# Compile decoder layers
for layer in model.model.decoder.layers:
    layer.self_attn = torch.compile(layer.self_attn, backend='neuron', dynamic=False)
    layer.encoder_attn = torch.compile(layer.encoder_attn, backend='neuron', dynamic=False)
    layer.fc1 = torch.compile(layer.fc1, backend='neuron', dynamic=False)
    layer.fc2 = torch.compile(layer.fc2, backend='neuron', dynamic=False)

# Compile proj_out
model.proj_out = torch.compile(model.proj_out, backend='neuron', dynamic=False)

# Wrap with TP all_reduce/all_gather
for layer in model.model.encoder.layers:
    layer.self_attn = TPAttention(layer.self_attn)
    layer.fc2 = TPMLP(layer.fc2)

for layer in model.model.decoder.layers:
    layer.self_attn = TPAttention(layer.self_attn)
    layer.encoder_attn = TPAttention(layer.encoder_attn)
    layer.fc2 = TPMLP(layer.fc2)

model.proj_out = TPProjOut(model.proj_out, TP)

dist.barrier()
if rank == 0:
    print(f"  TP-{TP} setup complete!")

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Download and prepare test audio (using soundfile, no torchcodec)
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print(f"\n[STEP 3] Downloading test audio files...")

AUDIO_INPUTS_PATH = "/tmp/whisper_test_inputs.pt"

if rank == 0:
    import soundfile as sf

    test_samples = []
    for i, sample_info in enumerate(TEST_SAMPLES):
        url = sample_info["url"]
        ground_truth = sample_info["ground_truth"]

        # Download audio file
        audio_path = f"/tmp/test_audio_{i}.flac"
        print(f"  Downloading sample {i+1}: {url}")
        urllib.request.urlretrieve(url, audio_path)

        # Load audio at native sample rate using soundfile
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        print(f"    Sample rate: {sample_rate}Hz, duration: {len(audio_data)/sample_rate:.2f}s")

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
            print(f"    Resampled to 16kHz")

        # Process through Whisper processor
        inputs = processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt",
        )
        test_samples.append({
            "inputs": inputs,
            "ground_truth": ground_truth,
        })
        print(f"    Ground truth: {ground_truth[:80]}...")

    torch.save(test_samples, AUDIO_INPUTS_PATH)
    print(f"  Prepared {len(test_samples)} test samples")

dist.barrier()
test_samples = torch.load(AUDIO_INPUTS_PATH, weights_only=False)

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Warmup run (triggers compilation)
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
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

if rank == 0:
    transcription = processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
    print(f"  Warmup time: {warmup_time:.2f}s (includes compilation)")
    print(f"  Transcription: {transcription[:150]}...")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Timed runs with WER validation
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print(f"\n[STEP 5] Timed inference runs with WER validation...")
    print("=" * 60)
    from jiwer import wer as compute_wer

NUM_RUNS = 5
run_times = []
wer_scores = []

for run_idx in range(NUM_RUNS):
    # Cycle through test samples
    sample_idx = run_idx % len(test_samples)
    sample = test_samples[sample_idx]
    inputs = {k: v.to(dtype=torch.bfloat16, device=NEURON_DEVICE) if isinstance(v, torch.Tensor) else v
              for k, v in sample["inputs"].items()}
    ground_truth = sample["ground_truth"]

    dist.barrier()
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

    if rank == 0:
        transcription = processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]

        # Compute WER
        error_rate = compute_wer(ground_truth.lower(), transcription.lower())
        wer_scores.append(error_rate)

        cached = "✅ cached" if elapsed < warmup_time * 0.5 else "⚠️ possible recompile"
        wer_status = "✅ PASS" if error_rate < 0.10 else "⚠️ HIGH WER" if error_rate < 0.25 else "❌ FAIL"

        print(f"\n  [RUN {run_idx+1}/{NUM_RUNS}] Sample {sample_idx+1} — {elapsed:.2f}s {cached}")
        print(f"    Expected: {ground_truth[:120]}...")
        print(f"    Got:      {transcription[:120]}...")
        print(f"    WER:      {error_rate:.2%} {wer_status}")

    dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    avg_time = sum(run_times) / len(run_times)
    avg_wer = sum(wer_scores) / len(wer_scores)
    all_cached = all(t < warmup_time * 0.5 for t in run_times)
    all_pass = all(w < 0.10 for w in wer_scores)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY — Whisper-large-v3 on Neuron")
    print("=" * 60)
    print(f"  Model:        openai/whisper-large-v3")
    print(f"  Backend:      torch.compile(backend='neuron', dynamic=False)")
    print(f"  TP:           {TP}")
    print(f"  Static cache: enabled")
    print(f"")
    print(f"  ── Performance ──")
    print(f"  Warmup:       {warmup_time:.2f}s (includes compilation)")
    print(f"  Run times:    {[f'{t:.2f}s' for t in run_times]}")
    print(f"  Average:      {avg_time:.2f}s")
    print(f"  Speedup:      {warmup_time/avg_time:.1f}x vs warmup")
    print(f"  All cached:   {'✅ YES' if all_cached else '❌ NO (recompilation detected)'}")
    print(f"")
    print(f"  ── Accuracy (WER) ──")
    print(f"  WER scores:   {[f'{w:.2%}' for w in wer_scores]}")
    print(f"  Average WER:  {avg_wer:.2%}")
    print(f"  All < 10%:    {'✅ PASS' if all_pass else '❌ FAIL'}")
    print(f"  Threshold:    10% WER (expected for Whisper large-v3 on LibriSpeech)")
    print(f"")

    if all_cached and all_pass:
        print("  ✅ SUCCESS: Accurate transcription + cached compilation confirmed!")
    elif all_pass:
        print("  ⚠️  Transcription accurate but some recompilations detected")
    elif all_cached:
        print("  ⚠️  Compilation cached but WER higher than expected")
    else:
        print("  ❌ Issues with both caching and accuracy")
    print("=" * 60)

dist.destroy_process_group()
