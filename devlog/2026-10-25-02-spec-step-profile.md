# Where a speculative step's time goes: the verify, not the draft head (T9.6)

*2026-10-25*

**Did:** Profiled a speculative step with the EAGLE-3 head from the full run:

1. Timers behind `LLAMA_SPEC_PROF=1` (`patches/0007-spec-profiling.patch`): server step time, target decode time by
   batch size, the draft model's calls (encoder, decode of the verified batch, seed step, draft steps).
2. nsys timelines with per-kernel CUDA graph tracing, with and without the head.
3. Quantized the head (`llama-quantize head.gguf head-q4_0.gguf Q4_0`; also Q8_0) and re-ran the benchmarks.

**Expected:** Most of the extra ~7 ms per step (15.5 ms with 2 drafts vs 8.3 ms per plain token) in the draft
model's 4 small decodes; fix with CUDA graphs or fewer draft calls.

**Got:** The expectation was wrong. Most of the extra time is the target model verifying more tokens:

| Target decode, generation | 1 token | 3 | 5 | 9 |
|---|---|---|---|---|
| ms per batch | 8.2 | 12.5 | 17.2 | 24-27 |

- Each extra token costs ~2.1 ms, about linear. Flint picks 6 of 256 experts per layer, so 3 tokens read mostly
  different experts: more GPU expert reads, and the CPU misses grow too (GPU waiting for CPU experts: 2.1 ms per step
  vs 0.7 ms for one token). The extraction of the 3 hidden states EAGLE reads costs almost nothing.
- Draft model per step: 2.4 ms in bf16 (~590 MB of weights read per step over 4 calls), 1.6 ms in q8_0, 1.25 ms in
  q4_0. Acceptance did not drop (held-out: 48-51% in q4_0 vs 48% in bf16).
- So a 2-draft step is ~12.5 + 1.25 + ~0.6 ms. It beats a plain token only when more than ~35% of drafts are
  accepted.

Generation speed, tok/s, with the q4_0 head (2 runs each, temperature 1; bf16 numbers from the full-run devlog):

| Workload | n-gram (default) | EAGLE, 2 (bf16) | EAGLE, 2 (q4_0) | EAGLE + n-gram, 2 (q4_0) | EAGLE acceptance |
|---|---|---|---|---|---|
| Held-out code answers | 126.8 | 133.1 | 134.0-140.2 | **143.6** | 49-51% |
| Short chat | 133.5 | 129.3 | **140.1** | 136.5 | 48-55% |
| 2K file question | **113.6** | 99.5 | 103.3 | 104.1 | 19-22% |
| 8K file question | **110-187** | 71.8 | 77.6 | 76.4 | 5% |
| Edit (rename) | 155.4 | 118.1 | 125.7 | **181.9** | 36-37% |
| Edit (comment) | 124.0 | 115.8 | 122.5 | **166.5** | |

(3 drafts stay worse than 2 in q4_0 as well. The runs are noisy: n-gram's 8K run was 110 and 187.)

- q4_0 makes EAGLE 4-8% faster everywhere. It is now `run-flint.sh`'s default head for `SPEC=eagle`.
- EAGLE + n-gram with 2 drafts is the fastest setting on new code, short chat and edits, but still loses 8-33% on
  long file questions. There the head's drafts are nearly all wrong (5% accepted at 8K): the training sequences
  were at most 4K tokens.
- The head also slows prompts by ~10%, because the draft model runs over the whole prompt too.

**Takeaway:** The draft head's own cost is small now. The verify cost is how this MoE engine works and does not
shrink with draft-side tricks, so what matters is acceptance where it is low: (1) T9.8, a per-slot gate that stops
EAGLE drafting while its acceptance is below ~35%, which would make `eagle+ngram` no worse than n-gram on long
files; (2) T9.7, long-context training data. n-gram stays the default until the 8K case stops losing.
