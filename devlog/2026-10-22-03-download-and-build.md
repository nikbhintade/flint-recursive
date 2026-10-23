# Download and build

*2026-10-22*

**Did:** Downloaded the official NVFP4 GGUF and built llama.cpp from master with CUDA 13.4 for the 5060 Ti (sm_120).

**Expected:** ~16 minutes for the download; a llama.cpp release with Flint support.

**Got:** 22,863,084,160 bytes, matching Hugging Face, at ~23–25 MB/s. No release supported Flint yet (support
landed on master on 2026-10-21, after the 2026-10-15 release), so we built master `19ab5a5`.

**Takeaway:** For a two-day-old model, build from master. (The `watch` progress command I suggested doesn't work
through `!`; it hung in the background and had to be killed.)
