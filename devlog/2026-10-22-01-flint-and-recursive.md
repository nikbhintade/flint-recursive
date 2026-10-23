# Looked into Flint and Recursive

*2026-10-22*

**Did:** Read up on Recursive's Flint, released the day before.

**Got:** A 39.8B-parameter MoE coding model, ~2.5B active per token, Apache 2.0, 262K context. All benchmarks are
self-reported (Terminal-Bench 2.1 54.9%, SWE-bench Verified 74.2%, SWE-bench Pro v2 89.7%; the last being above
Verified is unusual). Recursive is led by Richard Socher; Flint is its first model.

**Takeaway:** Only 2.5B active parameters per token makes it a good candidate for a 16 GB GPU + system RAM setup.
