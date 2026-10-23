# Can a custom engine beat llama.cpp? Design analysis

*2026-10-22*

**Did:** Worked out the model's memory traffic from its GGUF header and compared ideas against measurements from
our earlier Strata work.

**Expected (the user's idea):** Split experts into smaller pieces, cache hot ones on the GPU, transfer the rest.

**Got:**
- Each token reads ~0.95 GB of always-used weights plus ~0.50 GB of routed experts (47 layers × 6 × 1.77 MB).
- Smaller experts don't transfer faster: the bytes per token stay the same. Copying missing experts over PCIe
  (~20 ms/token) is slower than computing them on the CPU from RAM (~6 ms/token).
- The cache half of the idea is right: Strata (24% of experts in VRAM) served 77–85% of lookups from the GPU, and
  ~54% of Flint's experts fit in VRAM.
- Estimate: tuned llama.cpp ~55–80 tok/s; a Strata-style engine ~110–150 tok/s (1.6–2.2×).

**Takeaway:** Build "GPU computes cached experts while the CPU computes the misses", not "stream experts to the
GPU". Everything depends on how skewed Flint's routing is (T4).
