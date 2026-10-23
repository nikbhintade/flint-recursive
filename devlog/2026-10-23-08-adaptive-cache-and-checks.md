# Adaptive cache, workload checks, and an honest baseline (T7.3, T7.4)

*2026-10-23*

**Did:**
- T7.3: the cache now adapts. The host thread counts every expert use; every 16 tokens the hottest experts outside
  the cache replace the coldest inside (decayed counts, half-life 256 tokens). Swaps are made safe in three steps:
  the old expert leaves the slot map, a second CUDA stream uploads the new one after all queued work, and only then
  does the new one enter the map. The GPU also tells the CPU which experts it treated as misses, so they always
  agree.
- T7.4: long-context and agent-workload checks, against llama.cpp's own cache configured as well as ours
  (same `-ot ...exps\.weight=CPU` flag, pinned memory, 12 GB cache).

**Expected:** ~89% hits (T4), maybe a few % faster; no slowdown on prompts.

**Got:**
- Hit rate 79.9% -> 86.4%, CPU misses -30%, ~5.6 swaps per token in the background - but **no speed change**:
  since T7.2 the GPU is the limit, and the CPU work was already hidden. It will matter for batches and speculation.
- Quality unchanged (KL 0.00227 vs all-GPU).

| 32K context | llama.cpp cache, best flags | Ours | Gain |
|---|---|---|---|
| Generation, short prompt | 93.9 tok/s | 123.9 | +32% |
| Generation after 8K prompt | 88.4 | 112.0 | +27% |
| Generation after 30K prompt | 73.7 | 94.3 | +28% |
| Reading a 30K prompt | 1,025 tok/s | 1,033 | same |
| 25 agent turns | 49.4 s | 45.7 s | -7.5% |

- Correction: the "+36-45%" in the T7.2 entry compared with llama.cpp's *default* cache setup (`--n-cpu-moe`).
  With the same flag trick, llama.cpp's cache also gets faster; the fair gain is **+27-32%** in generation.
- Agent turns gain less: their time is mostly prompt reading, which neither engine changed.

**Takeaway:** T7 is done: ~124 tok/s short, ~112 at 8K, ~94 at 30K on this PC, quality unchanged. The GPU is now
the bottleneck (kernels at ~60% of bandwidth, ~1,850 kernels per token), and agent work is dominated by prompt
reading. Those are the next levers (T11, T12, T13), plus speculation (T9).
