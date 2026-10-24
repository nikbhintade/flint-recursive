# EAGLE-3 head, full local run: better head, no net speedup yet (T9)

*2026-10-25*

**Did:** Ran the full T9 pipeline unattended (`tools/t9/full_run.sh`), all on this PC with the NVFP4 model:

1. Flint answered 20,010 new prompts (SWE-bench *train* split for agent/issue prompts, no overlap with the pilot or
   the benchmarks), 8 parallel requests, capped at 20 h: **17,437 answers, 17.96M answer tokens**, a steady 249 tok/s.
2. Features for 23.2M tokens (answers + prompts): 381 GB on the internal NVMe, 4 h 25 min at ~1,460 tok/s.
3. Trained the head for 2 epochs (5,606 steps, 5 h 4 min) on sequences up to 4K tokens. The first attempt ran out of
   GPU memory: the 7-step unroll keeps ~2.4 GB of attention scores per step at 3K tokens. Fixed with per-step
   gradient checkpointing for long sequences (peak 7.1 GiB at 4K).
4. Compared speculation settings (`tools/t9/compare_spec.sh`): held-out prompts (never answered, so never trained
   on), the chat benchmark (short, 2K and 8K file questions) and the code-edit benchmark.

**Expected:** 55-65% first-token accuracy, ~1.2-1.8 accepted tokens per step, +10-25% on short prompts, weak on long
contexts.

**Got:**

| | Pilot (2.3M answer tokens) | Full run (18.0M) |
|---|---|---|
| Validation, first draft token correct | 42.2% | **62.8%** (steps 2-7: 56, 53, 51, 48, 46, 44%) |
| Held-out drafts accepted (3 per step) | 23% | **38%** (2 per step: 54%) |

Generation speed, tok/s (temperature 1, our engine):

| Workload | none | n-gram (default) | EAGLE, 2 drafts | EAGLE, 3 | EAGLE + n-gram, 3 | EAGLE, 3, p-min 0.75 |
|---|---|---|---|---|---|---|
| Held-out code answers | 120.7 | 130.8 | **133.1** | 117.0 | 119.3 | 110.3 |
| Short chat | 122.4 | **134.2** | 129.3 | 118.5 | 115.1 | 111.5 |
| 2K file question | 114.0 | 112.5 | 99.5 | 86.9 | 83.5 | 92.0 |
| 8K file question | 110.6 | **126.4** | 71.8 | 62.1 | 90.8 | 95.3 |
| Edit (rename) | 116.3 | **166.8** | 118.1 | 105.3 | 148.7 | 95.9 |
| Edit (comment) | 116.5 | 127.8 | 115.8 | 104.2 | **202.5** | 96.7 |

(4 drafts and p-min 0.5 were worse; all results in `data/t9/full/compare/`.)

- More data worked as hoped: accuracy kept rising through both epochs (25% at step 200, 43% at 1,000, 58% at
  3,000, 63% at the end), and acceptance tracks validation.
- Speed did not follow. Each speculative step costs ~15.5 ms (2 drafts) against 8.3 ms for one plain token: the
  verification of 3 tokens plus 2-3 small decodes of the draft model per step (feeding it the accepted tokens, a
  seed step, the drafts). Our engine makes one token so cheap that this fixed cost eats most of the gain.
- Long contexts stay the weak spot: on the 8K question the head drafts confidently wrong tokens. Confidence gating
  (p-min 0.75) cuts the loss (62 -> 95 tok/s) but also the gain elsewhere.
- EAGLE + n-gram helps edits the most (202 tok/s on the comment edit) but loses elsewhere.

**Takeaway:** The default stays n-gram. `SPEC=eagle DRAFT_MAX=2` is +10% on fresh code but much slower on long
contexts, so it stays an option, not a default. Next, in order of expected value: (1) profile and cut the per-step
draft overhead (draft-model decode launches, CUDA graphs for the draft context, the 3-token verify in our engine),
which is what decides whether *any* head pays off here; (2) long-context training data (8-32K agent sessions);
(3) more data only after (1), since accuracy alone is no longer the bottleneck.
