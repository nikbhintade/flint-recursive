# EAGLE-3 acceptance gate: long files fixed, still only a tie with n-gram (T9.8)

*2026-10-25*

**Did:** Made the EAGLE-3 drafter pause itself where it does not pay (`patches/0008-eagle3-acceptance-gate.patch`):

1. Per request, a running acceptance rate of EAGLE's drafts (starts at 0.6). Below 35% (the break-even with 2
   drafts, T9.6) it stops drafting: n-gram drafts if it can, otherwise a plain decode. After 16 steps it sends a
   probe draft. Each failed probe doubles the wait (up to 256 steps), and a passed one resets it.
   `LLAMA_SPEC_GATE` / `LLAMA_SPEC_GATE_PROBE` tune it.
2. Draft batches of 128 tokens (`--spec-draft-n-ubatch 128` in `run-flint.sh`). The draft model's compute buffer
   was 1.19 GB for 1,024-token batches, which is why the launcher held back 700 MiB of the expert cache for the head.
   Now the head needs ~350 MiB, which fits in the existing headroom. The full cache is back (also at the default 64K
   context).
3. Fixed 0007: its seed and draft-step timers were in the generic draft-model drafter, not EAGLE-3's. The server-side
   numbers in the T9.6 devlog were measured in the server and are unaffected.

**Expected:** `eagle+ngram` no worse than n-gram on long files, still ahead on new code, so it could become the
default.

**Got:** Long files are fixed, but overall it is a tie:

| Workload, tok/s | none | n-gram (default) | EAGLE + n-gram, no gate | EAGLE + n-gram, gate |
|---|---|---|---|---|
| Held-out code answers | 120.3 | 125.0 | 143.6 | 132.3-136.9 |
| Short chat | 122.5 | 122.8 | 136.5 | 127.4-134.3 |
| 2K file question | 113.4 | 112.9 | 104.1 | 108.0 |
| 8K file question | 110.8 | ~111 | 76.4 | 106.6 |
| Edit, rename (12 runs) | | **143.2** | | 126.4-139.2 |
| Edit, comment (12 runs) | | 148.5 | | 136.4-155.7 |
| Agent benchmark, 4 x 6 turns, wall s | 30.3 | 31.1 | | 31.5 |

(Ranges are separate sessions of the same setting; sampled runs vary by +-5-10%, edits even more: single runs of
115-285 tok/s.)

- The gate works where EAGLE is weak. At 8K it drafts ~38 times per 512 tokens instead of every step, and the long
  file goes from 76 to 107 tok/s. Most of that came from the probe backoff: a probe is a 3-token verify (~16 ms at
  8K) and slows the next plain steps too.
- Where EAGLE is strong (new code, ~50% accepted) the gate stays open and the gain stays: +6-9% over n-gram.
- Edits and agent turns: a tie. On edits n-gram already drafts the copied text, and EAGLE's 36-40% acceptance on the
  rest is right at break-even. Agent turns are short tool calls, where speculation hardly matters. Prompts take ~5%
  longer with the head (it reads every prompt token), down from ~10%.
- Tried and dropped: while gated, keep only the target's hidden states and run the draft model over them in one
  batch before the next probe. +2% at 8K, but the server saves the draft context with its prompt cache, without
  those rows. Restoring such a prompt left a gap in the draft context (HTTP 500 on the agent benchmark). Keeping the
  draft context complete is worth the 2%.

**Takeaway:** EAGLE is now safe to turn on (`SPEC=eagle+ngram`: never more than ~5% slower, +6-9% on new code), but
it is not a clear win, so n-gram stays the default. The overhead side is done: verify cost is fixed by the MoE, the
head's own cost is ~1.25 ms per step, and the gate removes the losses. What is left is acceptance. 5% at 8K and 20% at
2K come from training on sequences of at most 4K tokens (T9.7). A head that holds ~50% on long files would make
EAGLE + n-gram the default.
