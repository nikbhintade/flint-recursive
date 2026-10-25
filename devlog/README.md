# Devlog

One file per significant piece of work: **what we did, what we expected, what we got, and the takeaway.** Files are named `YYYY-MM-DD-NN-topic.md`; the newest is at the bottom of this list. Detailed numbers live in the [TODO.md results log](../TODO.md#results-log); plans and reasoning in [TODO.md](../TODO.md).

| Date | Entry |
|---|---|
| 2026-10-22 | [Looked into Flint and Recursive](2026-10-22-01-flint-and-recursive.md) |
| 2026-10-22 | [Can this PC run it?](2026-10-22-02-can-this-pc-run-it.md) |
| 2026-10-22 | [Download and build](2026-10-22-03-download-and-build.md) |
| 2026-10-22 | [Can a custom engine beat llama.cpp? Design analysis](2026-10-22-04-engine-design-analysis.md) |
| 2026-10-22 | [Repo set up](2026-10-22-05-repo-setup.md) |
| 2026-10-22 | [Draft head and cloud plan (no money spent)](2026-10-22-06-draft-head-and-cloud-plan.md) |
| 2026-10-23 | [Baseline benchmarks: llama.cpp vs. Ollama](2026-10-23-01-baseline-benchmarks.md) |
| 2026-10-23 | [Expert routing: how well would a GPU expert cache work? (T4)](2026-10-23-02-expert-routing.md) |
| 2026-10-23 | [Where decode time goes: CPU and GPU profile (T5)](2026-10-23-03-cpu-profile.md) |
| 2026-10-23 | [Engine route: llama.cpp fork, not Strata (T6)](2026-10-23-04-engine-route.md) |
| 2026-10-23 | [llama.cpp already has a GPU expert cache (T7 start)](2026-10-23-05-llama-moe-cache.md) |
| 2026-10-23 | [The CPU computes the misses: +17-28% (T7.1)](2026-10-23-06-cpu-computes-misses.md) |
| 2026-10-23 | [No host round trips: +36-45% (T7.2)](2026-10-23-07-no-host-sync.md) |
| 2026-10-23 | [Adaptive cache, workload checks, and an honest baseline (T7.3, T7.4)](2026-10-23-08-adaptive-cache-and-checks.md) |
| 2026-10-23 | [Where agent time goes, and a realistic agent benchmark](2026-10-23-09-agent-workload.md) |
| 2026-10-23 | [Prompt batches overlap uploads with compute: agent work -35% (T12.1)](2026-10-23-10-faster-prompts.md) |
| 2026-10-23 | [N-gram speculation: +34-69% on code edits](2026-10-23-11-ngram-speculation.md) |
| 2026-10-23 | [Ready for daily use: launcher, agent configs, soak test](2026-10-23-12-daily-use.md) |
| 2026-10-23 | [The CPU computing cold prompt experts: slower, switched off (T8, T12.2)](2026-10-23-13-cpu-share-of-prompts.md) |
| 2026-10-23 | [EAGLE-3 draft head, local pilot: the pipeline works, the head is too weak yet (T9)](2026-10-23-14-eagle3-pilot.md) |
| 2026-10-25 | [EAGLE-3 head, full local run: better head, no net speedup yet (T9)](2026-10-25-01-eagle3-full-run.md) |
| 2026-10-25 | [Where a speculative step's time goes: the verify, not the draft head (T9.6)](2026-10-25-02-spec-step-profile.md) |
