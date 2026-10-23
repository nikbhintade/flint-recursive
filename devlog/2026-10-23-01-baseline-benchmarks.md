# Baseline benchmarks: llama.cpp vs. Ollama

*2026-10-23*

**Did:** Swept llama.cpp settings (expert layers on CPU, threads, P/E-core pinning, context, prompt batch), then
ran the same three coding prompts (19 / 2.2K / 8.2K tokens) through llama-server and Ollama.

**Expected:** Tuned llama.cpp 55–80 tok/s decode; Ollama ~0.7× of that (40–65 tok/s).

**Got:**

| | llama.cpp, tuned | Ollama 0.40.9 |
|---|---|---|
| Decode, short prompt | **85.8 tok/s** (92.0 in llama-bench) | 72–73 tok/s |
| Decode, 8.2K prompt | **78.5 tok/s** | 47–68 tok/s |
| Prompt reading, 8.2K | **1,108 tok/s** (2,102 with `-ub 4096`) | 655 tok/s |
| Decode at 32K context | 71.8 tok/s | — |

- Pinning to P-core threads: 82 → 92 tok/s. Adding any E-core more than halves decode (40–55 tok/s).
- Best split: 18–20 layers' experts on the CPU; 16 or fewer runs out of memory.
- Answers are correct, so the NVFP4 path works on sm_120.
- Ollama: Flint on Linux needs 0.40.9 (system has 0.40.0, left untouched; ran 0.40.9 side by side). It ignored our
  hard-linked GGUF and downloaded the identical file again (+23 GB disk). It uses 4 unpinned threads.
- CPUs 8–9 (one P-core) are offline on this PC.

**Takeaway:** Baseline is better than estimated (~92, not 55–80), so the custom-engine gain is revised to
**~1.4–1.6×** (estimate). Next: record expert routing (T4) to see whether a 90%+ cache hit rate is realistic.
