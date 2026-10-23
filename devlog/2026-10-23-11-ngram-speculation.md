# N-gram speculation: +34-69% on code edits

*2026-10-23*

**Did:** Tried llama.cpp's built-in n-gram speculative decoding (drafts copied from earlier text, checked in one
step) on two code-edit prompts whose answers copy ~4K characters of code, plus the usual prompts and agent runs.

**Expected:** A cheap win on copy-heavy answers, no cost elsewhere.

**Got:**
- At first it was **slower** (45-110 vs 117 tok/s) even at 91% acceptance. `tools/profile/batch_bench.cpp` showed
  why: one decode step of 8 tokens cost 16 ms, but 9 tokens cost **189 ms**. From 9 tokens CUDA switches to the MMQ
  kernel, which walks every one of the ~7,400 cache slots - the same problem T12.1 hit with prompts.
- Fix (patch `0005`): batches of 9-32 tokens read the cache in chunks of 8 tokens, which use the vector kernels:
  9 tokens 19.5 ms, 32 tokens 42 ms (5x faster). Quality checked against an independent layer-split run.
- Then, with drafts capped at 8 tokens (`--spec-type ngram-mod --spec-ngram-mod-n-max 8`):

| Greedy, 2,048 tokens | No speculation | Drafts <= 4 | **Drafts <= 8** | Drafts <= 16 |
|---|---|---|---|---|
| Rename a variable, output all the code | 117 tok/s | 176 | **198 (+69%)** | 126 |
| Add comments, output all the code | 117 | 143 | **157 (+34%)** | 123 |

- Answers that copy little (explanations) are unchanged; an answer quoting an 8K-token file went 110 -> 166 tok/s;
  the agent benchmark is neutral because Flint's agent answers are short tool calls.

**Takeaway:** Free speed for code edits, on by default from now on. Longer drafts lose accuracy, so 8 is the cap.
