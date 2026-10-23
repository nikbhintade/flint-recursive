#!/usr/bin/env bash
# Profile llama-server decode with nsys: start it under nsys, send one greedy request, stop.
# Usage: tools/profile/profile_server.sh <name> [llama-server flags...]   -> data/profile/<name>.sqlite
cd "$(dirname "$0")/../.."
name=$1; shift
# nsys stops the server itself after 90 s (stopping it with pkill makes nsys-launcher abort)
taskset -c 0-7,10-15 nsys profile --trace=cuda,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none --duration=90 --kill=sigterm \
    -o data/profile/$name --force-overwrite true --export=sqlite \
    llama.cpp/build/bin/llama-server -m flint-nvfp4-gguf/flint-nvfp4.gguf -ngl 99 -c 8192 -np 1 -fa on -ub 1024 -t 14 \
    --jinja --host 127.0.0.1 --port 8080 --load-mode none "$@" > data/profile/$name.log 2>&1 &
for _ in $(seq 240); do curl -s 127.0.0.1:8080/health | grep -q ok && break; sleep 1; done
# REQ: a JSON file with the /completion request (default: a short greedy decode); REPEAT: times to send it
REQ=${REQ:-}
for _ in $(seq ${REPEAT:-1}); do
    if [ -n "$REQ" ]; then body=@$REQ; else body='{"prompt":"def merge_intervals(intervals):","n_predict":200,"temperature":0}'; fi
    curl -s 127.0.0.1:8080/completion -H 'Content-Type: application/json' -d "$body" | python3 -c 'import sys,json; t=json.load(sys.stdin)["timings"]; print("prompt %d tok %.1f tok/s, decode %.1f tok/s over %d tokens" % (t["prompt_n"], t["prompt_per_second"], t["predicted_per_second"], t["predicted_n"]))'
done
wait
