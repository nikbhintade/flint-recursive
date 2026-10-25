#!/usr/bin/env bash
# Start Flint (NVFP4) with our llama.cpp build: experts in RAM, adaptive GPU expert cache, CPU computes the misses,
# staged prompt batches, n-gram speculation. Serves an OpenAI-compatible API.
#
# Usage: ./run-flint.sh [extra llama-server flags...]
# Environment (all optional):
#   CTX=65536      context size; the expert cache shrinks by ~29 MB per extra 1K tokens of context
#   PORT=8080      port, on 127.0.0.1
#   CACHE_MIB=...  GPU expert cache size; default from CTX
#   SPEC=1         n-gram speculation (0 = off; eagle = EAGLE-3 draft head, see tools/t9; eagle+ngram = both)
#   EAGLE=...      EAGLE-3 head GGUF (default data/t9/full/head-q4_0.gguf, made with llama-quantize head.gguf ... Q4_0);
#                  DRAFT_MAX=2 tokens drafted per step; EAGLE pauses itself while its acceptance is low
#                  (LLAMA_SPEC_GATE, see patches/README.md)
#   PARALLEL=1     parallel requests (slots); each slot gets CTX/PARALLEL tokens
set -euo pipefail
cd "$(dirname "$0")"

CTX=${CTX:-65536}
PORT=${PORT:-8080}
SPEC=${SPEC:-1}
PARALLEL=${PARALLEL:-1}
# 12000 MiB fits next to a 32K context on a 16 GB card (with ~0.5 GB used by the desktop); KV costs ~28 MiB per 1K tokens
EAGLE=${EAGLE:-data/t9/full/head-q4_0.gguf}
DRAFT_MAX=${DRAFT_MAX:-2}
# the EAGLE-3 head (q4_0) with 128-token draft batches needs ~350 MiB, which fits in the headroom left next to the cache
CACHE_MIB=${CACHE_MIB:-$(( 12000 - (CTX - 32768) * 28 / 1024 ))}

MODEL=flint-nvfp4-gguf/flint-nvfp4.gguf
SERVER=llama.cpp/build/bin/llama-server
[ -f "$MODEL" ]  || { echo "missing $MODEL (see README: download)"; exit 1; }
[ -x "$SERVER" ] || { echo "missing $SERVER (see patches/README.md: apply and build)"; exit 1; }

# P-cores only: E-cores in the mix slow the CPU expert work down
CPUS=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || cat /sys/devices/system/cpu/online)
THREADS=$(taskset -c "$CPUS" nproc)

SPEC_FLAGS=()
case "$SPEC" in
    1)            SPEC_FLAGS=(--spec-type ngram-mod --spec-ngram-mod-n-max 8) ;;
    eagle)        SPEC_FLAGS=(--spec-type draft-eagle3 -md "$EAGLE" -ngld 99 --spec-draft-n-max "$DRAFT_MAX"
                              --spec-draft-n-ubatch 128) ;;
    eagle+ngram)  SPEC_FLAGS=(--spec-type draft-eagle3,ngram-mod --spec-ngram-mod-n-max 8 -md "$EAGLE" -ngld 99
                              --spec-draft-n-max "$DRAFT_MAX" --spec-draft-n-ubatch 128) ;;
esac

echo "Flint on http://127.0.0.1:$PORT/v1 - context $CTX, expert cache $CACHE_MIB MiB, CPUs $CPUS ($THREADS threads)"
export LLAMA_MOE_CACHE_PROFILE=$PWD/profiles/flint-coding.txt
exec taskset -c "$CPUS" "$SERVER" -m "$MODEL" --alias flint \
    -ngl 99 -ot 'blk\.[0-9]+\.ffn_(up|down|gate)_exps\.weight=CPU' --moe-cache-mib "$CACHE_MIB" --load-mode none \
    -c "$CTX" -np "$PARALLEL" -fa on -ub 1024 -b 2048 -t "$THREADS" --jinja \
    --temp 1 --top-p 0.95 --top-k 20 --min-p 0 \
    "${SPEC_FLAGS[@]}" --host 127.0.0.1 --port "$PORT" "$@"
