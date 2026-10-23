#!/usr/bin/env bash
# Start Flint (NVFP4) with our llama.cpp build: experts in RAM, adaptive GPU expert cache, CPU computes the misses,
# staged prompt batches, n-gram speculation. Serves an OpenAI-compatible API.
#
# Usage: ./run-flint.sh [extra llama-server flags...]
# Environment (all optional):
#   CTX=65536      context size; the expert cache shrinks by ~29 MB per extra 1K tokens of context
#   PORT=8080      port, on 127.0.0.1
#   CACHE_MIB=...  GPU expert cache size; default from CTX
#   SPEC=1         n-gram speculation (0 = off)
#   PARALLEL=1     parallel requests (slots); each slot gets CTX/PARALLEL tokens
set -euo pipefail
cd "$(dirname "$0")"

CTX=${CTX:-65536}
PORT=${PORT:-8080}
SPEC=${SPEC:-1}
PARALLEL=${PARALLEL:-1}
# 12000 MiB fits next to a 32K context on a 16 GB card (with ~0.5 GB used by the desktop); KV costs ~28 MiB per 1K tokens
CACHE_MIB=${CACHE_MIB:-$(( 12000 - (CTX - 32768) * 28 / 1024 ))}

MODEL=flint-nvfp4-gguf/flint-nvfp4.gguf
SERVER=llama.cpp/build/bin/llama-server
[ -f "$MODEL" ]  || { echo "missing $MODEL (see README: download)"; exit 1; }
[ -x "$SERVER" ] || { echo "missing $SERVER (see patches/README.md: apply and build)"; exit 1; }

# P-cores only: E-cores in the mix slow the CPU expert work down
CPUS=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || cat /sys/devices/system/cpu/online)
THREADS=$(taskset -c "$CPUS" nproc)

SPEC_FLAGS=()
if [ "$SPEC" = 1 ]; then
    SPEC_FLAGS=(--spec-type ngram-mod --spec-ngram-mod-n-max 8)
fi

echo "Flint on http://127.0.0.1:$PORT/v1 - context $CTX, expert cache $CACHE_MIB MiB, CPUs $CPUS ($THREADS threads)"
export LLAMA_MOE_CACHE_PROFILE=$PWD/profiles/flint-coding.txt
exec taskset -c "$CPUS" "$SERVER" -m "$MODEL" --alias flint \
    -ngl 99 -ot 'blk\.[0-9]+\.ffn_(up|down|gate)_exps\.weight=CPU' --moe-cache-mib "$CACHE_MIB" --load-mode none \
    -c "$CTX" -np "$PARALLEL" -fa on -ub 1024 -b 2048 -t "$THREADS" --jinja \
    --temp 1 --top-p 0.95 --top-k 20 --min-p 0 \
    "${SPEC_FLAGS[@]}" --host 127.0.0.1 --port "$PORT" "$@"
