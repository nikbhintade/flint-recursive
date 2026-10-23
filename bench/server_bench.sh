#!/usr/bin/env bash
# Start llama-server with the given extra flags, run chat_bench.py against it, stop it, print MoE cache stats.
# Usage: [RUNS=2] [MAX_TOKENS=512] [BENCH_ARGS="--long"] bench/server_bench.sh <label> [llama-server flags...]
# Fixed: Flint NVFP4, 32K context, P-core pinning, flash attention, ubatch 1024. Logs: bench/server-<label>.log
set -euo pipefail
cd "$(dirname "$0")/.."
label=$1; shift
log=bench/server-$label.log
taskset -c 0-7,10-15 llama.cpp/build/bin/llama-server -m flint-nvfp4-gguf/flint-nvfp4.gguf --alias flint \
    -ngl 99 -c 32768 -np 1 -fa on -ub 1024 -b 2048 -t 14 --jinja --host 127.0.0.1 --port 8080 "$@" > "$log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
for _ in $(seq 180); do
    curl -s 127.0.0.1:8080/health | grep -q ok && break
    kill -0 $pid 2>/dev/null || { echo "server exited:"; grep -i -E 'error|fail|out of memory' "$log" | tail -5; exit 1; }
    sleep 1
done
echo "== $label: $*  (VRAM used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
python3 bench/chat_bench.py llamacpp --runs ${RUNS:-2} --max-tokens ${MAX_TOKENS:-512} ${BENCH_ARGS:-}
kill -INT $pid; wait $pid 2>/dev/null || true
grep -E 'MoE cache size|slots \(|hits = |large batches' "$log" | sed 's/^/   /'
