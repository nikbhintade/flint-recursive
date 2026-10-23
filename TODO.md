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
- [x] **T2** Baseline: tuned llama.cpp → **~79–92 tok/s decode, ~1,100 tok/s prompts at 8K** (results log). Best
  launch: `taskset -c 0-7,10-15 llama-server --n-cpu-moe 20 -c 32768 -fa on -ub 1024 -b 2048 -t 14`.
  Findings: pin to P-core threads (+12% decode); any E-core in the mix halves decode; larger `-ub` nearly triples
  prompt speed but costs expert layers on the GPU. CPUs 8–9 (one P-core) are offline on this PC.
  - Sweep `--n-cpu-moe` (highest GPU share that fits), P-cores only (`taskset` + `-t 8/16`) vs. all threads,
    `-fa on`, KV f16 vs. q8_0, `-ub` for prompt speed
  - `llama-bench -p 512,4096 -n 128` at 4K / 32K context; record decode and prompt tok/s
  - Recommended sampling for real use: `--temp 1 --top-p 0.95 --top-k 20 --min-p 0 --jinja`
  - **Output:** the number every later change is compared with
- [x] **T3** Ollama: Flint needs ≥ 0.40.9 on Linux (system install is 0.40.0, left untouched; 0.40.9 runs
  side-by-side from `ollama-0.40.9/` on port 11435). `flint:latest` is byte-identical to the Hugging Face NVFP4
  GGUF. Ollama uses llama.cpp's auto-fit (20 layers' experts on CPU) with 4 unpinned threads →
  **~68–73 tok/s decode, ~620–650 tok/s prompts**

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
  - [ ] EAGLE-3 draft head trained on Flint's hidden states (big; brings back the 1.6–1.8× multiplier).
    Plan and costs: [§4](#4-draft-head-plan-t9)
    - [ ] **T9.1** Local, $0: SpecForge model definition for Flint (attention sinks + sliding window), tested on
      the 5060 Ti with a cut-down Flint (a few layers)
    - [ ] **T9.2** Pick ~30K–70K coding/agent prompts; choose the 32K draft vocabulary from token frequencies
    - [ ] **T9.3** Cloud pilot, ~$2–3: ~5M generated tokens + short training run; check acceptance length
    - [ ] **T9.4** Cloud full run, ~$11–13: 30M tokens + SpecForge online EAGLE-3 training
    - [ ] **T9.5** Local: load the head next to NVFP4 Flint; measure acceptance; if it drops vs. BF16, fine-tune
      on a few million locally extracted NVFP4 hidden states
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

## 4. Draft-head plan (T9)

Flint has no MTP layer, so speculative decoding needs a trained draft head. No one has published one for Flint
yet (Hugging Face search, 2026-10-22).

### What we train

An **EAGLE-3-style head**: one decoder layer that reads hidden states from three Flint layers (low, middle, high)
and drafts the next few tokens; Flint checks them in one pass, so the output is unchanged.

| Part | Size (*estimate*) |
|---|---|
| One decoder layer (hidden 2048, Flint's attention shape) + layer combining the 3 hidden states | ~60–70M trainable |
| Output head over a **32K draft vocabulary** (most common tokens) | ~65M |
| Token embedding | Shared with Flint, frozen |

The reduced vocabulary matters at inference: Flint's full 150K output head is 0.6 GB in BF16, ~2.4 ms per draft
step on the 5060 Ti, which would eat most of the gain. (Strata's MTP head used a 73K draft vocabulary.)

### Where each step can run

| Step | 5060 Ti (local) | Rented GPU |
|---|---|---|
| Training memory: ~130M params with AdamW ≈ 2 GB + activations (output head computed in chunks) | ✅ Fits | ✅ |
| Training compute: ~3×10¹⁷ FLOP for 30M tokens (incl. EAGLE-3's 7-step unroll) | ⚠️ ~6 h per pass over the data | ✅ ≤ 1 h |
| Data: Flint's own answers to the prompts (drafts match better when trained on the target's outputs) | ❌ ~5 days at ~70 tok/s | ✅ < 1 h with SGLang batching |
| Hidden states: 3 layers × 2048 × BF16 ≈ 12 KB/token, ~360 GB for 30M tokens | ⚠️ Nearly fills the disk | ✅ SpecForge online mode computes them on the fly |

**Conclusion:** generate and train in the cloud with SGLang + SpecForge; test the code locally first.

### Which cloud GPU

BF16 (79.6 GB) is the only Flint checkpoint SGLang loads (the INT4 repo is MLX, the NVFP4 repo is GGUF). What
limits batching is the memory **left over for KV cache** (~30 KB per token), not the GPU's speed.

| Option | KV room | Requests at once | Est. throughput | 30M tokens | Est. whole job |
|---|---|---|---|---|---|
| **H200 SXM, BF16, Community $3.59/hr** | ~43 GB | ~450 | ~8–12K tok/s | ~45–60 min | **~3–3.5 h ≈ $11–13** |
| RTX PRO 6000, `--quantization fp8` at load, Secure $2.49/hr | ~45 GB | ~400 | ~5–8K tok/s | ~1–1.5 h | ~4–5 h ≈ $10–12, riskier |
| RTX PRO 6000, BF16, Secure $2.49/hr | ~5 GB | ~70 | ~1,000–1,400 tok/s | ~6–8 h | ~9–10 h ≈ $23–25 |

- **Pick: H200 SXM.** 2.7× the PRO 6000's bandwidth for 2.1× the price, plus room to batch. It is also on
  SGLang's verified list for Flint (H200, B200, B300, GB300, DGX Spark).
- The PRO 6000 is sm_120 (same as the 5060 Ti); it is not on SGLang's verified list for Flint, and FP8-at-load
  for this MoE on sm_120 is untested.
- The H100 80 GB cards cannot hold BF16 Flint.
- Prices and stock from Runpod `list-gpu-types`, 2026-10-22; they change, so re-read before creating anything.
  Stock was LOW, only on hosts with **CUDA 12.9 or 13.2** → set the pod's CUDA floor to 12.9.

### Keeping the spend low

1. **Setup and debugging time costs more than compute.** Do T9.1 locally so the paid GPU only runs code that
   already works.
2. Pilot (T9.3) before the full run (T9.4); stop if acceptance is poor.
3. Check the credit balance first: Runpod's REST API (and so the MCP) has no balance route; use the console's
   Billing page or `runpodctl user`.
4. Delete the pod as soon as the run finishes; copy the head and logs off first.

### Risks

- **SpecForge may not know Flint** (two days old). T9.1 covers it.
- **BF16 → NVFP4 mismatch:** the head is trained on BF16 hidden states but runs against NVFP4. Recursive says
  NVFP4 scores within 0.7 points of 16-bit, so expect slightly lower acceptance, not failure. T9.5 fixes it if needed.
- **Payoff** (*estimate*): Strata's MTP gave 1.6–1.8× on this PC. With Flint's higher expected cache hit rate,
  the extra tokens in each check mostly use experts already on the GPU, so the gain may be similar or larger.
- **Order:** do this after T4/T7; the head multiplies whatever engine speed we reach.

---

## Results log

One row per measurement. Never compare runs that changed more than one setting.

| Date | Task | Engine + version | Placement / flags | Context | Metric | Result | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-22 | T0.1 | — | — | — | GGUF size | 22,863,084,160 B | Matches Hugging Face |
| 2026-10-23 | T0.3 | — | — | — | SHA-256 | `97cc8009ff9b…` | Same digest as Hugging Face and Ollama's `flint:latest` (Ollama verified it on pull) |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 20`, 14 threads unpinned | 512+128 | pp512 / tg128 | 714 / 74.0 tok/s | Smoke test |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 15`, `16` | 512+128 | — | OOM | 17 fits at 15.75 GB peak but leaves no room for context |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 17` / `18`, 14 threads | 512+128 | tg128 | 78.5 / 79.6 tok/s | Unpinned |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 18`, 7 P-cores, 1 thread each | 512+128 | tg128 | 91.3 tok/s | |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 18`, 14 P-threads pinned | 512+128 | pp512 / tg128 | 778 / **92.0** tok/s | Best decode |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 18`, 14 threads unpinned | 512+128 | tg128 | 82.3 tok/s | |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 18`, 16 E-cores only | 512+128 | pp512 / tg128 | 487 / 76.5 tok/s | |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 18`, 7 P + 16 E / all 30 threads | 512+128 | tg128 | 39.7 / 54.5 tok/s | E-cores stall lock-step work |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 18`, P-pinned | depth 4K | pp4096 / tg128 | 767 / 86.2 tok/s | |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 19`, P-pinned | depth 16K | pp4096 / tg128 | 715 / 79.7 tok/s | |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 20`, P-pinned | depth 32K | pp4096 / tg128 | 659 / 71.8 tok/s | |
| 2026-10-23 | T2 | llama.cpp `19ab5a5` | `-ncmoe 20/21/23`, `-ub 1024/2048/4096` | 4K prompt | pp4096 | 1,211 / 1,734 / **2,102** tok/s | `-ub 2048` at `-ncmoe 18`: OOM |
| 2026-10-23 | T2 | llama-server `19ab5a5` | `-ncmoe 20 -c 32768 -ub 1024`, P-pinned | 19 / 2.2K / 8.2K prompts | prompt / gen tok/s | 190–212 / 85.8; 941–953 / 80.7; 1,108 / 78.5 | `bench/chat_bench.py`, 512 generated tokens; answer checked by hand, correct |
| 2026-10-23 | T3 | Ollama 0.40.9 | auto-fit (20 layers' experts on CPU), 4 threads, `num_ctx` 32768 | 20 / 2.2K / 8.2K prompts | prompt / gen tok/s | 348–401 / 72–73; 616 / 69–70; 655 / 47–68 | Same script; prompt figures from first runs only (repeats hit Ollama's prompt cache) |
