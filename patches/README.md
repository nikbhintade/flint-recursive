# llama.cpp patches

Our engine changes, kept as patches against a fixed llama.cpp commit. The work happens on the local branch
`flint-hybrid-moe` of `llama.cpp/` (not committed there); these files are exported with `git diff`.

| Patch | Base | What it does |
|---|---|---|
| `0001-static-moe-cache-cpu-misses.patch` | `19ab5a5` | T7.1: static GPU expert cache; the CPU computes the misses while the GPU computes the hits |

## Apply and build

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 19ab5a5   # or the commit in the table
git apply ../patches/0001-static-moe-cache-cpu-misses.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

## Use (0001)

```bash
python3 tools/routing/make_profile.py    # data/routing/profile.txt from the T4 recording
LLAMA_MOE_CACHE_PROFILE=$PWD/data/routing/profile.txt \
taskset -c 0-7,10-15 llama.cpp/build/bin/llama-server -m flint-nvfp4-gguf/flint-nvfp4.gguf \
    -ngl 99 --n-cpu-moe 49 --moe-cache-mib 12000 --load-mode none -c 32768 -fa on -ub 1024 -t 14 --jinja
```

- `LLAMA_MOE_CACHE_PROFILE`: enables the static cache; a text file of `layer expert count` lines, most used first.
- `LLAMA_MOE_CACHE_THREADS`: CPU worker threads (default 12).
- Needs `--moe-cache-mib` and the experts in RAM (`--n-cpu-moe`). Only single-token decoding uses the new path;
  batches of 2-32 tokens currently fall back to llama.cpp's upload path and are slow (fix planned in T7.4).

## How 0001 works

- One slot of the cache stays zero. The GPU slot map sends every expert that is not cached to it, so a miss adds 0.
- When the scheduler copies a layer's slot map (it already synchronizes there to read the expert ids), the patch
  also reads the layer input and router weights and gives the misses to a CPU thread pool
  (`ggml_vec_dot_nvfp4_q8_0`: gate, up, swiglu, down, scales, router weight).
- Each layer adds a host tensor `cpu_out` to its MoE output. Because it is a host weight, the scheduler starts a
  new split there and calls the copy callback, which waits for the CPU and uploads the sum. The GPU computes the
  cached experts in the meantime.
- Note for any upstream proposal: `src/` calling ggml-cpu directly is a new pattern there.
