# TODO: a faster Flint engine for this PC

The plan, why each step is in it, and what we expect. Overview: [README.md](README.md).

**Rule:** every number below marked *estimate* is a back-of-envelope figure from a bandwidth model. Only the
[Results log](#results-log) counts. Change one setting per measurement.

---

## 1. Background

### 1.1 Flint, from the GGUF header

| Item | Value |
|---|---|
| Layers | 49: 2 dense, then 47 MoE |
| Hidden size | 2048 |
| Attention | 32 query heads, 4 KV heads, head size 128, attention sinks |
| Attention layout | **14 full-attention layers** (0, 1, 5, 9, …, 45, 48); the other **35 use a 513-token sliding window** |
| Experts | 256 per MoE layer, **6 routed + 2 shared** per token; expert width 512, shared 1024 |
| Expert size (NVFP4) | **1.77 MB** each; 12,032 routed experts = **21.3 GB** of the 22.9 GB file |
| Everything else | ~0.95 GB read every token (attention, shared experts, routers, output head) + a 0.6 GB BF16 embedding table (one row per token) |
| Draft / MTP head | **None** |
| Context | 262,144 (Recursive recommends ≤ 256K) |

**Per generated token:** ~0.95 GB always read + 47 × 6 × 1.77 MB ≈ **0.50 GB of routed experts** + KV cache.

**KV cache:** ~29 KB per context token at f16 (full-attention layers only; the sliding-window layers add a fixed
~36 MB). 32K context ≈ 0.9 GB, 128K ≈ 3.7 GB, 262K ≈ 7.5 GB. Every full-attention layer reads all of it on every
token, so **long context competes one-for-one with cached experts for VRAM**.

### 1.2 The machine

| Tier | Size | Usable speed | Notes |
|---|---|---|---|
| VRAM (RTX 5060 Ti) | 16 GB | ~250 GB/s in practice (448 on paper) | Practical figure from Strata's measurements |
| PCIe 5.0 x8 | — | ~25 GB/s | GPU ↔ RAM |
| RAM (DDR5, dual channel) | 64 GB | ~60–70 GB/s | XMP status unknown |
| CPU (i9-14900KS) | 8 P + 16 E cores | ~25–42 GB/s of expert weights | AVX2 + AVX-VNNI, no AVX-512; E-cores slow down lock-step work |

### 1.3 Ideas we checked and rejected

| Idea | Verdict | Why |
|---|---|---|
| Split experts into smaller pieces so they transfer faster | ❌ | Transfer time depends on total bytes, not the number of pieces; each token still needs ~0.5 GB of experts. Flint's experts are already small (1.77 MB) |
| Copy missing experts to the GPU over PCIe during generation | ❌ | 0.5 GB at ~25 GB/s ≈ 20 ms/token (≤ 50 tok/s). Computing them on the CPU from RAM takes ~6 ms. PCIe streaming only pays for long prompts, where many tokens share each copy |
| Change the router so it picks "better" experts | ❌ (mostly) | Changes the model's outputs; real changes need retraining. A mild, training-free version (cache-aware routing) is in T7 |
| Skip parts of an expert (neuron-level hot/cold, PowerInfer-style) | Later | Exact slicing is possible, but Flint's SwiGLU activations are rarely exactly zero, so skipping costs quality |
| Run Recursive's command as-is (`-ngl all -c 262144`) | ❌ | Recursive recommends ≥ 48 GB GPU memory; this card has 16 |

### 1.4 What Strata taught us (measured on this PC)

Source: `../ai-content/strata/` (our `nikhil-opt` branch, `strata-iq3_xxs.log`, the Strata paper).

| Strata technique | Measured | Use for Flint |
|---|---|---|
| Adaptive expert cache in VRAM; CPU computes the misses **in parallel** with the GPU | 24% of experts in VRAM served **77–85%** of lookups | ✅ Core of the design |
| Adaptive vs. static cache | Static profile ~50% hits, adaptive ~72% (paper) | ✅ Adaptive |
| Cache-aware routing (`03f59246`) | CPU experts/layer 6.18 → 4.24, **+3.5% decode**, quality unchanged | ⚠️ Small, optional |
| CPU share of non-cached experts during prompt reading (`b1c88544`) | Agent prompt reading **−27%** | ✅ |
| One CPU worker per physical core, heaviest tasks first | E-core stragglers avoided | ✅ |
| Fused GPU kernels (`d0e3e690`, `698254df`) | +1–1.5% each, identical output | ✅ |
| MTP speculative decoding | **1.6–1.8×**, the biggest multiplier | ❌ Flint has no MTP head (see T9) |
| Prompt lookup / suffix drafts | Code edits +6–11% | ✅ More important without MTP |
| Quality checks: identical output, or teacher-forced nats/token | — | ✅ Same discipline here |

Why Flint suits this design better than Strata's model: **~54% of Flint's experts fit in VRAM** (Strata's: 24%), so
a 90%+ hit rate is plausible *if* Flint's coding traffic is skewed.

### 1.5 Expected gains (*estimates*, 4K context)

| Setup | How the work splits | Est. decode tok/s | vs. tuned llama.cpp |
|---|---|---|---|
| Ollama (if it supports Flint) | Untuned llama.cpp | ~40–65 | ~0.7× |
| **llama.cpp, tuned** | ~27 MoE layers on GPU, ~20 on CPU; CPU does ~0.21 GB/token **after** the GPU | **~55–80** | 1× |
| Engine, 80% cache hits | CPU does ~0.1 GB/token **in parallel** | ~110–130 | ~1.6–2× |
| Engine, 90%+ cache hits | CPU does ~0.05 GB/token, almost fully overlapped | ~130–150 | ~1.8–2.2× |
| + prompt-lookup drafts | Code-edit turns | +10–40% on edits | — |
| + trained draft head | Strata's MTP gave 1.6–1.8× | ~200–250 | ~2.5–3× |
| *Ceiling: all in VRAM, no speculation* | 1.45 GB/token at 250 GB/s | *~180* | — |

Prompt reading: *~1.2–1.5×* (fewer experts streamed + CPU share). For agents, prompt-prefix reuse between turns
matters more than either.

**What could move these numbers:** (1) the cache hit rate (near-uniform routing → only ~1.3×); (2) the CPU's
NVFP4 speed (a slow llama.cpp CPU path lowers the baseline); (3) long context taking VRAM from the cache.

---

## 2. Tasks

### Phase 0: setup ✅

- [x] **T0.1** Download `RecursiveSI/flint-nvfp4-gguf` → `flint-nvfp4-gguf/` (22,863,084,160 bytes, matches Hugging Face)
- [x] **T0.2** Build llama.cpp master (`19ab5a5`) with CUDA 13.4 for sm_120; includes Flint support (`a37cc461`,
  2026-10-21), NVFP4 CUDA kernels and `--n-cpu-moe`
- [ ] **T0.3** (optional) SHA-256 of the GGUF against the Hugging Face LFS hash

### Phase 1: free gains and the baseline (~½ day)

- [ ] **T1** Hardware checks (Strata H10)
  - RAM speed: `sudo dmidecode -t memory | grep -i speed`; enable XMP if below rated speed (*+10–25%* on the CPU half)
  - Plug the monitor into the motherboard (UHD 770) to free the VRAM the desktop uses
  - Confirm PCIe 5.0 x8 under load: `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv`
- [ ] **T2** Baseline: tuned llama.cpp
  - Sweep `--n-cpu-moe` (highest GPU share that fits), P-cores only (`taskset` + `-t 8/16`) vs. all threads,
    `-fa on`, KV f16 vs. q8_0, `-ub` for prompt speed
  - `llama-bench -p 512,4096 -n 128` at 4K / 32K context; record decode and prompt tok/s
  - Recommended sampling for real use: `--temp 1 --top-p 0.95 --top-k 20 --min-p 0 --jinja`
  - **Output:** the number every later change is compared with
- [ ] **T3** Ollama: does 0.40 load Flint at all? If yes, same benchmark, for the README comparison

### Phase 2: measure what decides the design (~1–2 days)

- [ ] **T4** Record expert routing (Strata H4)
  - Eval callback on the MoE top-k tensor in llama.cpp (start from `llama-eval-callback`); dump layer × token × 6 ids
  - Workload: 50–100 coding prompts + agent turns (SWE-bench style)
  - Offline: hit rate vs. cache size for static profile, LRU and adaptive policies, k-fold (profile on 9/10,
    score on 1/10), like Strata's `tools/routing_kfold.py`
  - Also: how often consecutive tokens reuse the same experts (temporal locality)
  - **Decides:** ≥ 80% hits at ~6,500 slots → build the engine; near-uniform → focus on T6/T9 instead
- [ ] **T5** Profile the CPU side
  - How fast is llama.cpp's NVFP4 CPU path on AVX2 (GB/s of expert weights streamed)?
  - Where does a token's time go (GPU, CPU, syncs)? `nsys` on a short decode

### Phase 3: decide the engine route

- [ ] **T6 (decision)** Route A or B
  - **A. Add Flint to Strata** (recommended): reuses the expert cache, CPU pool, prompt streaming, prompt lookup
    and conversation cache. New: Flint attention (sinks + sliding window), NVFP4 kernels (CPU + GPU), GGUF loader
  - **B. Hot/cold expert split in a llama.cpp fork** (Strata H5): permute experts by hotness (exact), split each
    layer's expert tensors into a GPU bank and a CPU bank, run both and add the results. Smaller, upstreamable,
    but none of Strata's other features

### Phase 4: engine work (weeks)

- [ ] **T7** Per-expert VRAM cache with CPU computing misses in parallel (the core; adaptive policy from T4)
  - Optional: cache-aware routing (Strata `03f59246`), checked with teacher-forced nats/token
- [ ] **T8** NVFP4 CPU kernel for AVX2 + AVX-VNNI: decode e2m1 to int8 with a 16-entry `pshufb` lookup, VNNI dot
  product with q8 activations, apply the FP8 block scale. Exact; about the cost of Q4_0. Useful for llama.cpp too
- [ ] **T9** Speculation without MTP
  - [ ] Prompt lookup / suffix drafts (cheap; code edits)
  - [ ] EAGLE-style draft head trained on Flint's hidden states (big; brings back the 1.6–1.8× multiplier)
- [ ] **T10** Next-layer expert prediction (pre-gating) to update the cache ahead of time
- [ ] **T11** Fewer GPU↔CPU round trips per token: CUDA graphs, fused per-layer kernels
- [ ] **T12** Prompt reading: CPU takes a share of non-cached experts (Strata `b1c88544`); skip streaming cached ones
- [ ] **T13** Agent prompt-prefix reuse across turns (Strata H7): check the chat template does not rewrite history

### Phase 5: later / research

- [ ] **T14** Precision by hotness: NVFP4 for hot experts, lower for cold ones (Strata H6b); quality check required
- [ ] **T15** Long context: measure how much hit rate drops at 64K / 128K as KV takes VRAM
- [ ] **T16** Batching: total throughput with 2–4 parallel requests (Strata H2)

---

## 3. Quality checks (every engine change)

- Greedy decoding gives the same tokens as llama.cpp, or
- Teacher-forced nats/token on held-out code text, reported as mean ± spread, like the Strata commits
- Never trust Recursive's benchmark numbers as a quality reference: they are self-reported (e.g. SWE-bench Pro v2
  89.7% > Verified 74.2%, which is unusual)

---

## Results log

One row per measurement. Never compare runs that changed more than one setting.

| Date | Task | Engine + version | Placement / flags | Context | Metric | Result | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-22 | T0.1 | — | — | — | GGUF size | 22,863,084,160 B | Matches Hugging Face |
| | | | | | | | |
