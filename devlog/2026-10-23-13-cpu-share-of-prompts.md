# The CPU computing cold prompt experts: slower, switched off (T8, T12.2)

*2026-10-23*

**Did:** Wrote a multi-token NVFP4 CPU kernel (T8: AVX2 + AVX-VNNI, one weight row decoded once for up to 8 tokens)
and a CPU pool grouped by expert, then let prompt batches give the coldest share of each layer's uncached experts
to the CPU instead of uploading them (T12.2): a per-layer map marks them, their staging slots are filled with zeros,
the GPU sends the layer input to the CPU through the mailbox (now up to 1,024 tokens) and adds the CPU's result.

**Expected:** The cold experts see one to three tokens per batch each, so the CPU handles them cheaply in parallel,
and fewer uploads cut the ~0.45 s per prompt batch: agent benchmark ~30 s -> ~20 s.

**Got:**

| CPU share | Agent benchmark | Prompt tok/s 2K / 8K / 30K |
|---|---|---|
| 0 (off) | 30.8 s | 1,428 / 1,802 / 1,864 |
| 0.5 | 39.7 s | 657 / 1,036 / 999 |
| 0.75 | 48.9 s | 493 / 636 / 694 |

- The assumption was wrong: the cache is tuned for *generation*, and prompt tokens (code, tool output) route
  differently. With a 0.5 share the CPU computed ~1.17 of every token's 6 experts, ~7.5 GFLOP per layer per batch,
  and the GPU waits for it at the end of every layer.
- The kernel itself is correct: with the share off, KL vs all-GPU is 0.00232 and nats/token 0.3470 (the best of any
  run). With the share on, KL rises to 0.0047 while nats/token stays better than the reference (0.3479 vs 0.3509):
  the CPU uses 8-bit activations where the GPU's batched kernel uses 4-bit ones.
- Generation speed did not change: its CPU work was already hidden.

**Takeaway:** Off by default (`LLAMA_MOE_CACHE_PF_CPU` stays as an option). To make the CPU useful for prompts it
would have to pick experts by their actual token counts, which needs the expert ids before the uploads start and so
gives up T12.1's prefetch. The remaining prompt cost is PCIe bandwidth: ~8.7 GB per batch.
