# Devlog

One entry per significant piece of work: **what we did, what we expected, what we got, and the takeaway.**
Newest at the bottom. Detailed numbers live in the [TODO.md results log](TODO.md#results-log); plans and
reasoning in [TODO.md](TODO.md).

---

## 2026-10-22 · Looked into Flint and Recursive

**Did:** Read up on Recursive's Flint, released the day before.

**Got:** A 39.8B-parameter MoE coding model, ~2.5B active per token, Apache 2.0, 262K context. All benchmarks are
self-reported (Terminal-Bench 2.1 54.9%, SWE-bench Verified 74.2%, SWE-bench Pro v2 89.7%; the last being above
Verified is unusual). Recursive is led by Richard Socher; Flint is its first model.

**Takeaway:** Only 2.5B active parameters per token makes it a good candidate for a 16 GB GPU + system RAM setup.

## 2026-10-22 · Can this PC run it?

**Did:** Checked the hardware (RTX 5060 Ti 16 GB, i9-14900KS, 64 GB DDR5, 376 GB free disk) against the model's
download options.

**Expected:** The 23 GB model can't fit in VRAM, but an MoE runs well with the experts in RAM.

**Got:** Yes, it can. Recursive's README recommends ≥ 48 GB of GPU memory, so its example command (`-ngl all
-c 262144`) can't be used as-is; `--n-cpu-moe` is the way to fit it.

## 2026-10-22 · Download and build

**Did:** Downloaded the official NVFP4 GGUF and built llama.cpp from master with CUDA 13.4 for the 5060 Ti (sm_120).

**Expected:** ~16 minutes for the download; a llama.cpp release with Flint support.

**Got:** 22,863,084,160 bytes, matching Hugging Face, at ~23–25 MB/s. No release supported Flint yet (support
landed on master on 2026-10-21, after the 2026-10-15 release), so we built master `19ab5a5`.

**Takeaway:** For a two-day-old model, build from master. (The `watch` progress command I suggested doesn't work
through `!`; it hung in the background and had to be killed.)

## 2026-10-22 · Can a custom engine beat llama.cpp? Design analysis

**Did:** Worked out the model's memory traffic from its GGUF header and compared ideas against measurements from
our earlier Strata work.

**Expected (the user's idea):** Split experts into smaller pieces, cache hot ones on the GPU, transfer the rest.

**Got:**
- Each token reads ~0.95 GB of always-used weights plus ~0.50 GB of routed experts (47 layers × 6 × 1.77 MB).
- Smaller experts don't transfer faster: the bytes per token stay the same. Copying missing experts over PCIe
  (~20 ms/token) is slower than computing them on the CPU from RAM (~6 ms/token).
- The cache half of the idea is right: Strata (24% of experts in VRAM) served 77–85% of lookups from the GPU, and
  ~54% of Flint's experts fit in VRAM.
- Estimate: tuned llama.cpp ~55–80 tok/s; a Strata-style engine ~110–150 tok/s (1.6–2.2×).

**Takeaway:** Build "GPU computes cached experts while the CPU computes the misses", not "stream experts to the
GPU". Everything depends on how skewed Flint's routing is (T4).

## 2026-10-22 · Repo set up

**Did:** Created `README.md` (overview) and `TODO.md` (plan, background numbers, tasks T0–T16) and pushed to
`nikbhintade/flint-recursive` (over SSH; HTTPS had no credentials). The model, llama.cpp and Ollama folders are
git-ignored.

## 2026-10-22 · Draft head and cloud plan (no money spent)

**Did:** Checked whether we can train a speculative-decoding draft head (Flint has no MTP layer), checked the Runpod
account, and compared GPUs.

**Expected:** Training needs a big GPU; an RTX PRO 6000 with batching might be cheapest.

**Got:**
- An EAGLE-3 head is ~130M parameters, so *training* fits on the 5060 Ti. *Generating the training data* doesn't
  (~5 days locally vs. < 1 hour on a rented H200).
- Runpod: no pods, endpoints or volumes; $0.90 spent in October; the balance can't be read through the API.
- The PRO 6000 can't batch much with BF16 Flint (80 GB of weights leave ~5 GB for KV cache), so the H200 is
  ~5× cheaper per token. Estimated whole job ≈ $11–13 on an H200 Community pod at $3.59/hr.

**Takeaway:** Test the SpecForge code locally first; then pilot (~$2–3), then a full run. Plan in TODO.md §4.

## 2026-10-23 · Baseline benchmarks: llama.cpp vs. Ollama

**Did:** Swept llama.cpp settings (expert layers on CPU, threads, P/E-core pinning, context, prompt batch), then
ran the same three coding prompts (19 / 2.2K / 8.2K tokens) through llama-server and Ollama.

**Expected:** Tuned llama.cpp 55–80 tok/s decode; Ollama ~0.7× of that (40–65 tok/s).

**Got:**

| | llama.cpp, tuned | Ollama 0.40.9 |
|---|---|---|
| Decode, short prompt | **85.8 tok/s** (92.0 in llama-bench) | 72–73 tok/s |
| Decode, 8.2K prompt | **78.5 tok/s** | 47–68 tok/s |
| Prompt reading, 8.2K | **1,108 tok/s** (2,102 with `-ub 4096`) | 655 tok/s |
| Decode at 32K context | 71.8 tok/s | — |

- Pinning to P-core threads: 82 → 92 tok/s. Adding any E-core more than halves decode (40–55 tok/s).
- Best split: 18–20 layers' experts on the CPU; 16 or fewer runs out of memory.
- Answers are correct, so the NVFP4 path works on sm_120.
- Ollama: Flint on Linux needs 0.40.9 (system has 0.40.0, left untouched; ran 0.40.9 side by side). It ignored our
  hard-linked GGUF and downloaded the identical file again (+23 GB disk). It uses 4 unpinned threads.
- CPUs 8–9 (one P-core) are offline on this PC.

**Takeaway:** Baseline is better than estimated (~92, not 55–80), so the custom-engine gain is revised to
**~1.4–1.6×** (estimate). Next: record expert routing (T4) to see whether a 90%+ cache hit rate is realistic.
