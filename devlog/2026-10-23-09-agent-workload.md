# Where agent time goes, and a realistic agent benchmark

*2026-10-23*

**Did:** Broke down the 25-turn agent run, profiled one ~780-token prompt with nsys, compared the chat endpoint with
raw completions, and wrote `bench/agent_bench.py`: 4 episodes x 6 turns where each turn appends a 300-1,500-token
tool result and prompt caching is on, like a real coding agent.

**Expected:** Prompt reading dominates agent time.

**Got:**
- Single-turn agent set: prompt reading was 37.8 of 45.7 s (83%), with a fixed ~1 s per prompt
  (`prompt ms = 995 + 0.71 x tokens`).
- One ~780-token prompt batch (~0.88 s): ~380 ms uploading the uncached experts (7.4 GB at 19.6 GB/s), ~320 ms of
  GPU compute, ~60 ms copying cached experts into a temporary tensor, ~120 ms idle. **Uploads and compute never
  overlap**: everything runs on one stream, one after another.
- The chat endpoint is ~0.55 s slower than a raw completion of the *same* text: for sliding-window models
  llama-server splits chat prompts (at user messages and 4 / 4+ubatch tokens before the end) to save context
  checkpoints, and each large piece pays the expert upload again.
- Multi-turn benchmark: prompt caching reuses 73% of prompt tokens, but each turn's new tokens still cost 0.7-1.7 s.

| 4 episodes x 6 turns | llama.cpp cache (best flags) | Ours (T7) |
|---|---|---|
| Wall time | 46.4 s | 40.9 s (-12%) |
| Prompt: processed / reused tokens | 19,368 / 52,550 | 19,365 / 52,378 |
| Prompt time | 28.3 s | 26.9 s |
| Generation time | 17.4 s | 13.1 s |

**Takeaway:** For agents the lever is the fixed cost of every prompt batch. Next: stream all uncached experts of the
next layer while the current layer computes (two scratch areas, second CUDA stream), estimated ~41 -> ~28 s.
