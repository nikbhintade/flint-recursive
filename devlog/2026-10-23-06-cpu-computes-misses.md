# The CPU computes the misses: +17-28% (T7.1)

*2026-10-23*

**Did:** Changed llama.cpp's MoE cache (patch `patches/0001`, against `19ab5a5`). The cache is filled once from the
T4 routing profile and never changes; one slot stays zero and every uncached expert points to it, so on the GPU a
miss adds nothing. When llama.cpp stops at each layer to read the expert ids, the patch also reads the layer input
and router weights and hands the misses to 12 CPU threads (ggml's NVFP4 dot product). Their sum is added to the
layer output through a small host tensor, whose copy waits for the CPU while the GPU computes the cached experts.

**Expected:** ~100-120 tok/s if the CPU work really overlaps with the GPU; no change in output quality beyond
rounding.

**Got:**

| Same chat prompts, 32K context | Hit rate | Gen tok/s: short / 2K / 8K |
|---|---|---|
| llama.cpp cache, uploads misses (baseline) | 88.7% | 86.1 / - / 82.5-82.9 |
| **Static cache, CPU computes misses** | 79.5% | **110 / 100.5 / 96.5** |

- +17-28% faster with a lower hit rate and zero PCIe uploads. The hit rate matches T4's static prediction (79.8%).
- 8, 12 or 13 CPU threads give the same speed: the CPU is no longer the limit. What remains is the per-layer stop
  to read expert ids, and the hit rate.
- Quality, teacher-forced over 1,004 positions of real Flint answers: KL 0.00295 vs. all-GPU, top-1 agreement
  98.5%, nats/token 0.3511 vs. 0.3509. llama.cpp's own layer split differs by the same amount (KL 0.00285).
- Regression: batches of 2-32 tokens fall back to the upload path (a 19-token prompt read at 26 tok/s).

**Takeaway:** The core idea works. Next: remove the per-layer stop (T7.2), adapt the cache contents (T7.3, worth
~+10 points of hit rate), and fix small batches (T7.4).
