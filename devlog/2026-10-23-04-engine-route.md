# Engine route: llama.cpp fork, not Strata (T6)

*2026-10-23*

**Did:** Compared the two routes on the actual code: Strata's source and llama.cpp's scheduler and MoE path.

**Expected:** Adding Flint to Strata (route A) would reuse most of Strata's engine; that was the earlier
recommendation.

**Got:**
- Strata is 66K lines tuned to one model: its layer code is Gated DeltaNet, QSA sparse attention and
  hyper-connections; it has its own tokenizer and chat handling; it has no NVFP4 kernels. For Flint we would write
  new attention, NVFP4 kernels, loader and template/tool handling. The *design* carries over, the code does not.
- llama.cpp already runs Flint correctly with all of that. What it lacks is exactly one thing: its scheduler runs
  GPU and CPU work one after another (that is the 52% GPU idle time from T5).

**Decision:** Route B, a llama.cpp fork. The work is one combined MoE operation (cached experts on the GPU and
misses on a CPU thread pool at the same time, then added) plus a cache manager using the decayed-LFU policy from
T4. Estimated 1–3 weeks.
