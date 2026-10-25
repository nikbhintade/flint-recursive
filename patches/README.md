# llama.cpp patches

Our engine changes, kept as patches against a fixed llama.cpp commit. The work happens on the local branch
`flint-hybrid-moe` of `llama.cpp/` (not committed there); these files are exported with `git diff`. **Each patch is
the full diff against the base commit at that milestone** (apply one, not several); the newest is the current one.

| Patch | Base | What it does |
|---|---|---|
| `0001-static-moe-cache-cpu-misses.patch` | `19ab5a5` | T7.1: static GPU expert cache; the CPU computes the misses while the GPU computes the hits |
| `0002-moe-host-mailbox.patch` | `19ab5a5` | T7.2: no per-layer host sync; two ggml ops pass the misses to the CPU through pinned memory; 1-32 token batches |
| `0003-adaptive-moe-cache.patch` | `19ab5a5` | T7.3: the cache adapts (decayed use counts, swaps uploaded on a second stream); the CPU follows the device's slot choices |
| `0004-staged-prompt-experts.patch` | `19ab5a5` | T12.1: prompt batches stage each layer's experts while the previous layer computes; includes the precision fix below |
| `0005-chunked-small-batches.patch` | `19ab5a5` | Batches of 9-32 tokens read the cache in 8-token chunks (CUDA vector kernels): 5x faster, makes n-gram speculation pay |
| `0006-cpu-kernel-prompt-share.patch` | `19ab5a5` | Multi-token NVFP4 CPU kernel (AVX2/AVX-VNNI), grouped CPU pool, 1,024-token mailbox; optional CPU share of prompt experts (off: it was slower) |
| `0007-spec-profiling.patch` | `19ab5a5` | T9.6: `LLAMA_SPEC_PROF=1` logs per-step speculation timings (target decode by batch size, EAGLE-3 draft-model phases); off by default |
| `fix-moe-cache-prec-policy.patch` | `19ab5a5` | Standalone bug fix for upstream llama.cpp's own `--moe-cache-mib` (see below) |

## Apply and build

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 19ab5a5   # or the commit in the table
git apply ../patches/0007-spec-profiling.patch   # the newest patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

## Use

```bash
python3 tools/routing/make_profile.py    # data/routing/profile.txt from the T4 recording
LLAMA_MOE_CACHE_PROFILE=$PWD/data/routing/profile.txt \
taskset -c 0-7,10-15 llama.cpp/build/bin/llama-server -m flint-nvfp4-gguf/flint-nvfp4.gguf \
    -ngl 99 -ot 'blk\.[0-9]+\.ffn_(up|down|gate)_exps\.weight=CPU' --moe-cache-mib 12000 --load-mode none \
    -c 32768 -fa on -ub 1024 -t 14 --jinja --spec-type ngram-mod --spec-ngram-mod-n-max 8
```

`--spec-type ngram-mod --spec-ngram-mod-n-max 8`: n-gram speculation (drafts copied from earlier text, verified in one
step). +34-69% on answers that copy code, no change on others. Keep drafts at 8 or fewer tokens.

`-ot ... \.weight=CPU` keeps only the expert weights in RAM; `--n-cpu-moe` also moves the small expert scale
tensors, which then get copied to the GPU every layer (4-8% slower).

- `LLAMA_SPEC_PROF`: set to log speculation timings every 500 steps (`srv prof:` and `spec prof:` lines).
- `LLAMA_MOE_CACHE_PROFILE`: enables the static cache; a text file of `layer expert count` lines, most used first.
- `LLAMA_MOE_CACHE_THREADS`: CPU worker threads (default 12).
- `LLAMA_MOE_CACHE_ADAPT`: `0` keeps the profile's experts fixed; default: adapt.
- `LLAMA_MOE_CACHE_STAGE`: `0` turns off prompt-batch staging (frees ~0.9 GB for the cache); default: on.
- `LLAMA_MOE_CACHE_PF_CPU`: share (0-1) of each prompt batch's uncached experts that the CPU computes instead of
  uploading, coldest first; default 0, because it made prompts slower (see devlog 2026-10-23-13).
- Needs `--moe-cache-mib` and the experts in RAM. Batches of 1-32 tokens use the hybrid path; larger batches
  (prompts) use llama.cpp's path, which streams experts and reads cached ones from the cache.

## How 0004 speeds up prompts

- Batches of more than 32 tokens use two staging tensors per expert matrix, each holding all 256 experts of one
  layer, used by alternate layers. The matmuls read them with the original expert ids, so CUDA's MMQ sees the usual
  256 experts (reading the 7,400-slot cache in place made MMQ ~2.5x slower: it walks every expert).
- A `GGML_OP_MOE_HOST_CALL` node before each layer's experts runs host code as the graph is queued: it queues the
  upload of the uncached experts of this and the next layer on a second CUDA stream (after the event that says the
  staging area is free), copies the cached experts from the cache on the compute stream (device copies use the SMs,
  so on the copy stream they stalled behind the matmuls), and makes the compute stream wait for the upload. A second
  node after the experts records that the area is free. Graphs with these nodes are never captured as CUDA graphs.
- Per prompt batch: ~444 ms of uploads now overlap ~350 ms of compute and cache copies.

## Upstream bug fixed on the way (`fix-moe-cache-prec-policy.patch`)

Flint's NVFP4 GGUF stores per-tensor activation precision (`general.tensor_extra.prec_a4`): some expert matrices
must run with 8-bit activations. `llama_prec_policy::apply()` looks it up by the matmul's weight tensor, which is the
cache bank when experts come from `--moe-cache-mib`, so the lookup fails and CUDA's MMQ falls back to 4-bit
activations for every expert. In llama.cpp itself this hits batches of 9-32 tokens served from the cache (MMQ);
single tokens (MMVQ) are not affected. The fix looks the policy up by the original weight.

## How 0003 adds adaptation

- The send kernel also writes the slots it used to the mailbox; the CPU computes exactly the experts the device read
  from a zero slot, so the two always agree even while the cache changes.
- The host thread counts every expert use. Before each graph (`llama_context::graph_compute` calls `refresh()`),
  every 16 ubatches the counts are decayed (half-life 256 ubatches) and up to 256 hotter experts replace the coldest
  cached ones: (1) the victims leave the slot maps, (2) a second CUDA stream uploads the new experts after an event
  that follows all queued work, (3) at a later refresh, once the upload is done, they enter the slot maps.
  No kernel reads a slot while it is written.

## How 0002 works

- `GGML_OP_MOE_HOST_SEND` (CUDA): writes the layer input, expert ids and router weights to a mailbox in pinned host
  memory, then raises a request counter. It also returns the slots for the expert matmuls: the cached slot, or a
  zero slot for a miss. Each rank (1st..6th expert of a token) has its own zero slot, because `mul_mat_id` must not
  read one expert twice for a token (MMQ crashes on that).
- A host thread polls the counter, gives the misses to the CPU pool, writes their weighted sum to the mailbox and
  raises a done counter. No host sync.
- `GGML_OP_MOE_HOST_RECV` (CUDA): waits for the done counter (traps after ~10 s instead of hanging) and adds the sum.
- The slot maps live on the GPU, so a token's whole graph is one CUDA graph.

## How 0001 worked (replaced by 0002)

- One slot of the cache stays zero. The GPU slot map sends every expert that is not cached to it, so a miss adds 0.
- When the scheduler copies a layer's slot map (it already synchronizes there to read the expert ids), the patch
  also reads the layer input and router weights and gives the misses to a CPU thread pool
  (`ggml_vec_dot_nvfp4_q8_0`: gate, up, swiglu, down, scales, router weight).
- Each layer adds a host tensor `cpu_out` to its MoE output. Because it is a host weight, the scheduler starts a
  new split there and calls the copy callback, which waits for the CPU and uploads the sum. The GPU computes the
  cached experts in the meantime.
- Note for any upstream proposal: `src/` calling ggml-cpu directly is a new pattern there.
