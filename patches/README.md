# llama.cpp patches

Our engine changes, kept as patches against a fixed llama.cpp commit. The work happens on the local branch
`flint-hybrid-moe` of `llama.cpp/` (not committed there); these files are exported with `git diff`. **Each patch is
the full diff against the base commit at that milestone** (apply one, not several); the newest is the current one.

| Patch | Base | What it does |
|---|---|---|
| `0001-static-moe-cache-cpu-misses.patch` | `19ab5a5` | T7.1: static GPU expert cache; the CPU computes the misses while the GPU computes the hits |
| `0002-moe-host-mailbox.patch` | `19ab5a5` | T7.2: no per-layer host sync; two ggml ops pass the misses to the CPU through pinned memory; 1-32 token batches |
| `0003-adaptive-moe-cache.patch` | `19ab5a5` | T7.3: the cache adapts (decayed use counts, swaps uploaded on a second stream); the CPU follows the device's slot choices |

## Apply and build

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 19ab5a5   # or the commit in the table
git apply ../patches/0003-adaptive-moe-cache.patch   # the newest patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

## Use

```bash
python3 tools/routing/make_profile.py    # data/routing/profile.txt from the T4 recording
LLAMA_MOE_CACHE_PROFILE=$PWD/data/routing/profile.txt \
taskset -c 0-7,10-15 llama.cpp/build/bin/llama-server -m flint-nvfp4-gguf/flint-nvfp4.gguf \
    -ngl 99 -ot 'blk\.[0-9]+\.ffn_(up|down|gate)_exps\.weight=CPU' --moe-cache-mib 12000 --load-mode none \
    -c 32768 -fa on -ub 1024 -t 14 --jinja
```

`-ot ... \.weight=CPU` keeps only the expert weights in RAM; `--n-cpu-moe` also moves the small expert scale
tensors, which then get copied to the GPU every layer (4-8% slower).

- `LLAMA_MOE_CACHE_PROFILE`: enables the static cache; a text file of `layer expert count` lines, most used first.
- `LLAMA_MOE_CACHE_THREADS`: CPU worker threads (default 12).
- `LLAMA_MOE_CACHE_ADAPT`: `0` keeps the profile's experts fixed; default: adapt.
- Needs `--moe-cache-mib` and the experts in RAM. Batches of 1-32 tokens use the hybrid path; larger batches
  (prompts) use llama.cpp's path, which streams experts and reads cached ones from the cache.

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
