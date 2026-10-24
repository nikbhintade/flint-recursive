#!/usr/bin/env bash
# Compare speculation settings for the T9 head: held-out acceptance/speed, then the chat benchmark (short, long, edit).
# Each config restarts the server. Output: data/t9/full/compare/<config>.txt
#
# EXTRA="flags" adds llama-server flags, TAG names the variant.
# Usage: tools/t9/compare_spec.sh "0" "1" "eagle:3" "eagle:2" "eagle+ngram:3"   (SPEC[:DRAFT_MAX])
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=data/t9/full/compare
mkdir -p "$OUT"
export EAGLE=${EAGLE:-data/t9/full/head.gguf}

stop_server() {
    if pgrep -x llama-server >/dev/null; then kill $(pgrep -x llama-server); fi
    for _ in $(seq 60); do pgrep -x llama-server >/dev/null || return 0; sleep 1; done
}

for cfg in "$@"; do
    spec=${cfg%%:*}; dmax=${cfg#*:}; [ "$dmax" = "$cfg" ] && dmax=3
    name="spec-${spec}-d${dmax}${TAG:-}"
    stop_server
    SPEC=$spec DRAFT_MAX=$dmax CTX=32768 ./run-flint.sh ${EXTRA:-} > "$OUT/$name.server.log" 2>&1 &
    for _ in $(seq 150); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 2; done
    {
        echo "== $name held-out (temp 1)"
        python3 tools/t9/accept_bench.py --n 12
        echo "== $name chat bench"
        python3 bench/chat_bench.py llamacpp --runs 2 --max-tokens 512
        echo "== $name code edits"
        python3 bench/chat_bench.py llamacpp --runs 2 --edit
    } > "$OUT/$name.txt" 2>&1
    grep -E '^total|^== ' "$OUT/$name.txt"
done
stop_server
