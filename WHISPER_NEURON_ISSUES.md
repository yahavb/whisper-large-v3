# Whisper-large-v3 on Neuron — PyTorch Native Issues & Options

## Summary

Running `openai/whisper-large-v3` on AWS Trainium using `torch.compile(backend='neuron')` with TP-2.
The model compiles and runs, but produces incorrect output (repetitive tokens, 155-192% WER).

**Repository:** https://github.com/yahavb/whisper-large-v3  
**Script:** `run_whisper_neuron.py`  
**Hardware:** Trainium (trn1), 2 NeuronCores (TP-2)  
**Software:** torch-neuronx, transformers, torch.compile(backend='neuron')

---

## Architecture

Whisper-large-v3 is an encoder-decoder model:
- **Encoder:** 32 layers, processes mel spectrogram (fixed shape 1×80×3000)
- **Decoder:** 32 layers, autoregressive token generation with KV cache
- 20 attention heads (10/rank with TP-2), d_model=1280, FFN=5120

## What Works ✅

| Component | Status |
|-----------|--------|
| Model loading + TP sharding | ✅ |
| Move to neuron device | ✅ |
| TP wrappers (all_reduce/all_gather) | ✅ |
| torch.compile(backend='neuron') on layers | ✅ |
| Encoder compilation + execution | ✅ |
| Compilation caching (64x speedup on subsequent runs) | ✅ |
| Neuron profile artifact generation | ✅ |

## Current Issue: Decoder Recompilation → Broken Output ❌

### Symptom

```
Input:  "HE HOPED THERE WOULD BE STEW FOR DINNER..."
Output: "He hoped, he, he, he, he, he, he, he, he..."  (WER: 192%)
```

### Root Cause

TorchDynamo hits `config.recompile_limit (8)` during decoder autoregressive generation:

```
torch._dynamo hit config.recompile_limit (8)
function: '__call__' (transformers/modeling_layers.py:59)
last reason: kwargs['past_key_values'].is_updated[6] == False
```

The Whisper decoder layer's forward function accesses `past_key_values.is_updated[layer_idx]` — a Python dictionary that changes state as each layer writes to the KV cache. TorchDynamo places guards on this dictionary state, causing:

1. Each unique `is_updated` state triggers a new compilation
2. With 32 layers × pre/post-update states, there are many unique guard combinations
3. After 8 recompilations (default limit), Dynamo gives up
4. The function falls back to broken/partial execution
5. KV cache is not properly maintained → repetitive output

### Relevant Code Path

```python
# transformers/models/whisper/modeling_whisper.py:313
is_updated = past_key_values.is_updated.get(self.layer_idx)
```

This line creates a TorchDynamo guard on the dictionary lookup result. Different layers at different generation steps see different values.

---

## Options to Fix

### Option A: Increase `cache_size_limit` ❌ (Tried — doesn't fix)

```python
torch._dynamo.config.cache_size_limit = 64
```

**Result:** Still hits limit at 64. The REAL recompilation reason is:
```
tensor 'kwargs['past_key_values'].self_attention_cache.layers[30].keys' size mismatch at index 2. expected 4, actual 5
```

The KV cache tensor **grows by 1 token every decoding step**. This is NOT a finite set of states — it's unbounded. No `cache_size_limit` will fix this because each new sequence length requires a new compilation with `dynamic=False`.

### Option A2: Use `dynamic=True` for decoder layers ❌ (Tried — Neuron rejects)

```python
# Encoder: fixed input shape, dynamic=False is fine
model.model.encoder.layers[i] = torch.compile(layer, backend='neuron', dynamic=False)

# Decoder: KV cache grows every step, must use dynamic=True
model.model.decoder.layers[i] = torch.compile(layer, backend='neuron', dynamic=True)
```

**Rationale:** `dynamic=True` tells TorchDynamo to treat tensor dimensions as symbolic, so a single compiled graph handles all sequence lengths without recompilation.

**Tradeoffs:**
- ✅ Single compilation handles all sequence lengths
- ✅ No recompilation limit issues
- ❓ Depends on neuron backend supporting dynamic shapes
- ❓ May have performance overhead from symbolic shape handling

### Option B: Compile encoder only, run decoder eagerly

```python
# Compile encoder layers (fixed input shape, no KV cache)
for i, layer in enumerate(model.model.encoder.layers):
    model.model.encoder.layers[i] = torch.compile(layer, backend='neuron', dynamic=False)

# Do NOT compile decoder layers — run eagerly on neuron device
# (just wrap with TP, no torch.compile)
```

**Rationale:** The encoder processes fixed-shape mel spectrograms — ideal for compilation. The decoder's autoregressive KV cache makes it problematic for static compilation.

**Tradeoffs:**
- ✅ Encoder gets full compilation benefit
- ✅ Decoder runs correctly (no guard issues)
- ❌ Decoder runs unoptimized (eager on neuron)
- ❌ Misses decoder compilation speedup

### Option C: Disable Dynamo guards on `past_key_values`

```python
# Mark past_key_values as unguarded/dynamic
torch._dynamo.config.suppress_errors = True
# or use torch.compiler.allow_in_graph for KV cache access
```

**Rationale:** Tell Dynamo not to guard on KV cache state changes.

**Tradeoffs:**
- ✅ Would fix the recompilation issue
- ❌ May produce incorrect compiled code if KV cache state matters for graph structure
- ❌ `suppress_errors=True` is too broad (hides all dynamo errors)

### Option D: Use static KV cache with proper Neuron backend integration

```python
model.generation_config.cache_implementation = "static"
```

**Problem:** This triggers transformers to wrap `model.forward` with `torch.compile()` using the **default Inductor backend** (not neuron). Inductor can't handle neuron device tensors (`var_mean` lowering fails).

**What's needed from PyTorch Native team:** A way to tell transformers to use `backend='neuron'` when it does its implicit `torch.compile` for static cache. Or a Neuron-compatible static cache implementation.

---

## Issues Encountered During Development

### Issue 1: `torch_dtype` deprecation
```
`torch_dtype` is deprecated! Use `dtype` instead!
```
Cosmetic warning only.

### Issue 2: torchcodec/FFmpeg not available
**Fix:** Use `soundfile` library directly instead of torchcodec for audio loading.

### Issue 3: Input features dtype mismatch
**Fix:** Cast `input_features` to bfloat16 before passing to model.

### Issue 4: Attention head count mismatch after sharding
```
RuntimeError: shape mismatch in attention reshape
```
**Fix:** Patch `layer.self_attn.num_heads` directly on each attention module after sharding (config change alone doesn't propagate to already-constructed modules).

### Issue 5: Inductor `var_mean` lowering failure
```
torch._inductor.exc.InductorError: LoweringException: AssertionError
target: aten.var_mean.correction
```
**Root cause:** `cache_implementation="static"` triggers implicit `torch.compile()` with default Inductor backend on neuron device tensors.
**Fix:** Remove `cache_implementation="static"`, compile layers explicitly with `backend='neuron'`.

### Issue 6: Per-sub-module compilation misses layer norms
**Root cause:** Compiling only self_attn/fc1/fc2 leaves layer norms outside any compiled graph.
**Fix:** Compile the **entire layer** as one unit (includes layer norms).

### Issue 7: TP wrappers outside compiled graph
**Root cause:** Wrapping TP after compile means all_reduce calls are outside the neuron graph.
**Fix:** Wrap TP FIRST, then compile the full layer (TP wrapper is inside the compiled graph).

---

## Performance Results (Current State)

```
Warmup:     949.60s (includes compilation of all layer variants)
Run times:  ['20.64s', '11.86s', '20.38s', '0.63s', '20.41s']
Average:    14.79s
Speedup:    64.2x vs warmup
Caching:    ✅ All runs use cached compilations
Accuracy:   ❌ WER 155-192% (broken due to recompile limit)
```

---

## Ask for PyTorch Native Team

1. **How should encoder-decoder models with KV cache be handled under `torch.compile(backend='neuron')`?** The Whisper decoder's `past_key_values.is_updated` dictionary creates excessive Dynamo guards that exceed the recompile limit.

2. **Can `cache_implementation="static"` be made to use `backend='neuron'` instead of defaulting to Inductor?** Currently it forces the Inductor backend which doesn't support neuron device.

3. **Is there a recommended pattern for autoregressive generation with torch.compile on Neuron?** The Qwen3-VL model works because it's decoder-only and the generation logic is simpler. Whisper's encoder-decoder pattern with cross-attention KV cache introduces additional complexity.

4. **Would `torch._dynamo.config.cache_size_limit = 64` (or higher) be expected to fix the accuracy issue?** Or is there a deeper problem with how the compiled decoder handles the KV cache state transitions?
