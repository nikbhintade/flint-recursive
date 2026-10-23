# Can this PC run it?

*2026-10-22*

**Did:** Checked the hardware (RTX 5060 Ti 16 GB, i9-14900KS, 64 GB DDR5, 376 GB free disk) against the model's
download options.

**Expected:** The 23 GB model can't fit in VRAM, but an MoE runs well with the experts in RAM.

**Got:** Yes, it can. Recursive's README recommends ≥ 48 GB of GPU memory, so its example command (`-ngl all
-c 262144`) can't be used as-is; `--n-cpu-moe` is the way to fit it.
