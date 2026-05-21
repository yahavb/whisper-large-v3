"""OpenAI Whisper-large-v3 — PyTorch Native on Neuron (TP-2).

Tensor-parallel inference using torch.compile(backend='neuron').
Demonstrates: compile once → reuse for multiple audio inputs.

Architecture: Encoder-Decoder (speech-to-text)
  - 32 encoder layers (self-attention + MLP)
  - 32 decoder layers (self-attention + cross-attention + MLP)
  - 20 attention heads → 10 per rank with TP-2
  - d_model=1280, FFN=5120 → 2560 per rank with TP-2
"""

import os
import sys
import time
import tempfile

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np


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
    print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load and shard model
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print(f"\n[STEP 1] Loading Whisper-large-v3 on CPU and sharding for TP-{TP}...")

from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/whisper-large-v3")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).eval().requires_grad_(False)

# Update config for sharded heads
model.config.encoder_attention_heads = model.config.encoder_attention_heads // TP
model.config.decoder_attention_heads = model.config.decoder_attention_heads // TP

if rank == 0:
    print(f"  Original: 20 heads → {model.config.encoder_attention_heads} heads/rank")
    print(f"  Encoder layers: {len(model.model.encoder.layers)}")
    print(f"  Decoder layers: {len(model.model.decoder.layers)}")

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
    # Compile fc1+fc2 as a unit via the layer forward
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
# STEP 3: Prepare audio inputs
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print(f"\n[STEP 3] Preparing audio inputs...")

import urllib.request
import io

# Sample audio URLs (LibriSpeech samples hosted publicly)
AUDIO_URLS = [
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/2.flac",
]

AUDIO_INPUTS_PATH = "/tmp/whisper_inputs.pt"

if rank == 0:
    import librosa

    all_inputs = []
    for i, url in enumerate(AUDIO_URLS):
        print(f"  Downloading audio {i+1}: {url}")
        audio_path = f"/tmp/audio_sample_{i}.flac"
        urllib.request.urlretrieve(url, audio_path)

        # Load audio at 16kHz (Whisper's expected sample rate)
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"    Duration: {len(audio)/sr:.2f}s, samples: {len(audio)}")

        # Process through Whisper processor
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        all_inputs.append(inputs)

    torch.save(all_inputs, AUDIO_INPUTS_PATH)
    print(f"  Saved {len(all_inputs)} processed inputs")

dist.barrier()
all_inputs = torch.load(AUDIO_INPUTS_PATH, weights_only=False)

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Warmup run (triggers compilation)
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print(f"\n[STEP 4] Warmup run (triggers compilation)...")

warmup_inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in all_inputs[0].items()}

warmup_start = time.time()
with torch.no_grad():
    generated_ids = model.generate(
        **warmup_inputs,
        max_new_tokens=128,
        language="en",
        task="transcribe",
    )
warmup_time = time.time() - warmup_start

if rank == 0:
    transcription = processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
    print(f"  Warmup time: {warmup_time:.2f}s (includes compilation)")
    print(f"  Transcription: {transcription[:200]}")

dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Timed runs (should reuse compiled graph)
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    print(f"\n[STEP 5] Timed inference runs (cached compilation)...")
    print("=" * 60)

NUM_RUNS = 5
run_times = []

for run_idx in range(NUM_RUNS):
    # Alternate between audio samples
    audio_idx = run_idx % len(all_inputs)
    inputs = {k: v.to(NEURON_DEVICE) if isinstance(v, torch.Tensor) else v
              for k, v in all_inputs[audio_idx].items()}

    dist.barrier()
    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            language="en",
            task="transcribe",
        )
    elapsed = time.time() - start
    run_times.append(elapsed)

    if rank == 0:
        transcription = processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
        cached = "✅ cached" if elapsed < warmup_time * 0.5 else "⚠️ possible recompile"
        print(f"\n  [RUN {run_idx+1}/{NUM_RUNS}] Audio {audio_idx+1} — {elapsed:.2f}s {cached}")
        print(f"    Transcription: {transcription[:150]}")

    dist.barrier()

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
if rank == 0:
    avg_time = sum(run_times) / len(run_times)
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY — Whisper-large-v3 on Neuron")
    print("=" * 60)
    print(f"  Model:       openai/whisper-large-v3")
    print(f"  Backend:     torch.compile(backend='neuron', dynamic=False)")
    print(f"  TP:          {TP}")
    print(f"  Warmup:      {warmup_time:.2f}s (includes compilation)")
    print(f"  Run times:   {[f'{t:.2f}s' for t in run_times]}")
    print(f"  Average:     {avg_time:.2f}s")
    print(f"  Speedup:     {warmup_time/avg_time:.1f}x vs warmup")
    all_cached = all(t < warmup_time * 0.5 for t in run_times)
    print(f"  All cached:  {'✅ YES' if all_cached else '❌ NO (recompilation detected)'}")
    print("=" * 60)

dist.destroy_process_group()
