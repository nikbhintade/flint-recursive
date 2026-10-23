# Expert-routing tools (T4)

Record which experts Flint picks for every token, then simulate GPU expert caches on that record.
Outputs go to `data/routing/` (git-ignored). Results: TODO.md, T4.

| Step | Command | Time on this PC |
|---|---|---|
| 1. Build the workload (100 prompts) | `python3 tools/routing/make_prompts.py` | seconds (needs internet) |
| 2. Answer them with Flint | start `llama-server … --jinja --port 8080`, then `python3 tools/routing/generate.py` | ~25 min |
| 3. Record routing | stop the server, then `python3 tools/routing/replay.py` | ~20 min |
| 4. Analyze | `python3 tools/routing/analyze.py` | ~1 min |
| 5. Static-cache profile (T7.1) | `python3 tools/routing/make_profile.py` | seconds |

## Building `route_dump`

```bash
L=llama.cpp
g++ -O2 -std=c++17 tools/routing/route_dump.cpp -I$L/include -I$L/ggml/include \
    -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/$L/build/bin -o tools/routing/build/route_dump
```

`route_dump` uses llama.cpp's eval callback to read the `ffn_moe_topk-<layer>` and `ffn_moe_weights-<layer>`
tensors. It replays each sequence's prompt in ubatches and the generated answer one token at a time, matching how
the tokens were produced. Every token is marked as an output, because llama.cpp drops non-output rows before the
last layer's MoE.

## Files in `data/routing/`

| File | Contents |
|---|---|
| `prompts.jsonl` | The 100 prompts (messages, tools) |
| `sequences.jsonl` | Prompt and generated token ids, text, speeds |
| `sequences.txt`, `meta.json` | `route_dump` input; per-sequence kind and lengths |
| `routes.ids` | uint8 `[token][47 MoE layers][6]` expert ids |
| `routes.w` | float32, same shape: the selected experts' raw router scores |

## CPU profiling (T5)

`tools/profile/cpu_moe_bench.cpp` times one MoE layer's CPU expert work (6 of 256 NVFP4 experts, one token).
Build like `route_dump` but link `-lggml -lggml-base -lggml-cpu`; run `taskset -c 0-7,10-15 cpu_moe_bench 14`.
