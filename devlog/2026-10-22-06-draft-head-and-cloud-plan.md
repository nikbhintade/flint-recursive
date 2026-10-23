# Draft head and cloud plan (no money spent)

*2026-10-22*

**Did:** Checked whether we can train a speculative-decoding draft head (Flint has no MTP layer), checked the Runpod
account, and compared GPUs.

**Expected:** Training needs a big GPU; an RTX PRO 6000 with batching might be cheapest.

**Got:**
- An EAGLE-3 head is ~130M parameters, so *training* fits on the 5060 Ti. *Generating the training data* doesn't
  (~5 days locally vs. < 1 hour on a rented H200).
- Runpod: no pods, endpoints or volumes; $0.90 spent in October; the balance can't be read through the API.
- The PRO 6000 can't batch much with BF16 Flint (80 GB of weights leave ~5 GB for KV cache), so the H200 is
  ~5× cheaper per token. Estimated whole job ≈ $11–13 on an H200 Community pod at $3.59/hr.

**Takeaway:** Test the SpecForge code locally first; then pilot (~$2–3), then a full run. Plan in TODO.md §4.
