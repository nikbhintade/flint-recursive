# flint-recursive

Running **Recursive's Flint** coding model fast on a single consumer PC, and finding out whether a purpose-built
inference engine can beat llama.cpp / Ollama on it.

## The model

**Flint** (Recursive Superintelligence, released 2026-10-21) is an open-weight (Apache 2.0) mixture-of-experts coding
model: **39.8B parameters in total, about 2.5B used per token**, 256 experts per layer (6 picked per token + 2 shared),
and up to 262K tokens of context. We use the official **NVFP4 GGUF** (22.9 GB).

Recursive is an AI lab led by Richard Socher (formerly Salesforce's chief scientist) that builds automated AI-research
systems. Flint is its first model.

## The machine

| Part | Spec |
|---|---|
| GPU | RTX 5060 Ti, 16 GB (PCIe 5.0 x8) |
| CPU | i9-14900KS (8 P-cores + 16 E-cores, AVX2, no AVX-512) |
| RAM | 64 GB DDR5 |

The model does not fit in VRAM, so part of the work has to run on the CPU from system RAM.

## The idea

Flint reads only about 0.5 GB of expert weights per token, and about half of its 12,032 experts fit in VRAM. An
engine that **keeps the most-used experts on the GPU and computes the rest on the CPU at the same time**, as
[Strata](https://github.com/Niko1221/Strata) does for another MoE model, should be roughly **1.6–2.2× faster** at
generating text than a tuned llama.cpp. That is an estimate until we measure it.

## Status

- [x] NVFP4 weights downloaded and size-checked
- [x] llama.cpp built from master with CUDA (Flint support landed after the last release)
- [ ] Baseline speed and expert-routing measurements
- [ ] Decide: add Flint to Strata, or a hot/cold expert split in a llama.cpp fork

The plan, the reasoning behind it and every task: **[TODO.md](TODO.md)**.

## Layout

```
flint-nvfp4-gguf/   model weights + pi/OpenCode configs   (not in git)
llama.cpp/          llama.cpp source and CUDA build        (not in git)
README.md           this overview
TODO.md             plan, background numbers and tasks
```

## Links

- Model: [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) ·
  NVFP4 GGUF: [RecursiveSI/flint-nvfp4-gguf](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) · [Strata](https://github.com/Niko1221/Strata)
