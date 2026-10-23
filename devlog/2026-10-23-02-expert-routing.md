# Expert routing: how well would a GPU expert cache work? (T4)

*2026-10-23*

**Did:** Answered 100 coding prompts with Flint (25 each: HumanEval, SWE-bench Verified issues, agent turns with
tool definitions, explain/review/test tasks on real source files), then replayed all 181,329 tokens (68,839
generated) through a small llama.cpp tool that records each layer's 6 chosen experts. Prompts were replayed in
batches and answers one token at a time, like a real run. Then simulated GPU caches of different sizes, scored with
10-fold cross-validation over prompts.

**Expected:** Strata's model got 77–85% hits with 24% of experts cached. Flint can cache ~54%, so 90%+ *if* its
routing is skewed. Training usually pushes MoE routing toward even use, so skew was the open question.

**Got:**
- Skew is only moderate: a layer's top 25% of experts serve 49% of lookups (entropy 7.66 of 8 bits).
- But locality is strong: 84% of a token's experts were used in the previous 64 tokens.
- At 7,424 GPU slots (what llama.cpp's 29 GPU layers use at 32K context):

| Policy | Decode hit rate | CPU work per token |
|---|---|---|
| llama.cpp layer split | 61.7% | 108 experts (191 MB) |
| Static profile | 79.8% | 57 experts (101 MB) |
| **LRU / decayed LFU** | **89.5% / 89.3%** | **~30 experts (53 MB)** |
| Best fixed set, chosen with hindsight | 92.6% | — |

- Decayed LFU needs half of LRU's PCIe copies (~15 vs. ~30 experts per token). Warming the cache from the
  prompt's own routing added nothing.
- Agent turns hit less (83%): their answers are ~32 tokens, too short to adapt.
- Check: batched vs. token-by-token replay agree on 96.3% of expert choices and give the same statistics.

**Takeaway:** The cache works because of *locality*, not global skew, so it must be **adaptive**. At ~89% hits the
CPU's share drops from 191 MB to ~53 MB per token. That clears the T4 bar (≥ 80%): building the engine is worth it.
