# llama.cpp already has a GPU expert cache (T7 start)

*2026-10-23*

**Did:** While reading llama.cpp's MoE code to plan T7, found `--moe-cache-mib`: an SLRU cache of the experts of
RAM-resident layers, added upstream recently. Benchmarked it with the same chat prompts as the baseline.

**Expected:** Nothing like this existed; our engine would be the first per-expert cache for this setup.

**Got:**

| Setup (32K context) | Decode hit rate | Gen tok/s, short / 8K prompt |
|---|---|---|
| Layer split, `--n-cpu-moe 20` (old baseline) | (62% on GPU) | 85.8 / 78.5 |
| All experts in RAM + 11 GB cache, mmap | — | 69–71 / 66 |
| Same, `--load-mode none` (pinned host memory) | 86.0% | 78–82 / 80 |
| **All in RAM + 12 GB cache, `--load-mode none`** | **88.7%** | **86.1 / 82.5–82.9** |
| Layer split 30 + 4 GB cache / 40 + 8 GB cache | 71% / 83% | — / 73–75, 78.5–81.5 |

- The hit rates match what our T4 simulator predicted for LRU at those sizes (~86–89%), so the simulator holds up.
- Every miss is **uploaded over PCIe** (~32 per token, ~56 MB, ~2.3 ms), and every layer has a host round trip
  to read the expert ids: ~11.9 ms per token vs. 6.98 ms if everything were on the GPU.

**Takeaway:** The new baseline is llama.cpp's own cache (~83–86 tok/s). Our work becomes smaller and sharper:
keep its cache, but compute misses on the CPU in parallel instead of uploading them, remove the per-layer host
round trip, and refresh the cache in the background. Target estimate unchanged: ~135–145 tok/s.
