# Prompt batches overlap uploads with compute: agent work -35% (T12.1)

*2026-10-23*

**Did:** Made each prompt batch upload the next layer's uncached experts while the current layer computes, on a
second CUDA stream. Three things went wrong first, each found with nsys or the quality check:
1. Reading the experts in place from the cache (7,400 slots) made CUDA's MMQ visit every slot: 800 ms of matmuls
   for one 779-token prompt. Fix: *stage* each layer's 256 experts into one of two ordinary expert tensors
   (~0.9 GB taken from the cache) and run the matmuls on those.
2. Quality dropped (KL 0.0096). Not a race: Flint's GGUF says which expert matrices need 8-bit activations, and
   llama.cpp looks that up by the matmul's weight tensor - which is the cache copy, so every expert ran with 4-bit
   activations. Fixed by looking it up by the original weight. **This bug is in upstream llama.cpp's own
   `--moe-cache-mib`** for 9-32-token batches; a standalone patch is in `patches/`.
3. The copy stream stalled ~1.6 ms before each GPU-to-GPU copy of cached experts: such copies need the SMs, which
   the matmuls were using. Fix: only uploads go on the copy stream; cached experts are copied on the compute stream.

**Expected:** ~0.45 s per prompt batch instead of ~0.88 s; agent benchmark ~41 -> ~28 s.

**Got:**

| | llama.cpp cache (best flags) | T7 | T12.1 |
|---|---|---|---|
| Multi-turn agent benchmark (24 turns), wall | 46.4 s | 40.9 s | **30.2 s** |
| ... of which prompt reading | 28.3 s | 26.9 s | **18.0 s** |
| Prompt tok/s: 2K / 8K / 30K | - / 1,070 / 1,025 | ~920 / 1,075 / 1,033 | **1,436 / 1,798 / 1,848** |
| Generation tok/s: short / 8K / 30K | 93.9 / 88.4 / 73.7 | 123.9 / 112.0 / 94.3 | 121.0 / 111.6 / 92.6 |

- Quality unchanged: KL 0.00247 vs all-GPU.
- The cache lost ~0.9 GB to staging; the decode hit rate is still ~86.6% and generation speed did not move.

**Takeaway:** Agent work is now 35% faster than llama.cpp's best setup, close to the ~28 s estimate. Every prompt
batch still pays ~0.45 s of uploads (8.7 GB over PCIe), which is what remains for short agent turns; the CPU
computing the rarely used uncached experts (T12.2) is the next lever there.
