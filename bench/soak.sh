#!/usr/bin/env bash
# Soak test: run-flint.sh plus repeated agent episodes for MINUTES minutes; samples server memory and VRAM every 30 s.
# Usage: [MINUTES=30] [CLIENTS=1] [PARALLEL=1] bench/soak.sh <label>
# Output (git-ignored): bench/soak-<label>.txt (iterations, samples), bench/soak-<label>-server.log
cd "$(dirname "$0")/.."
label=$1
MINUTES=${MINUTES:-30}
CLIENTS=${CLIENTS:-1}
out=bench/soak-$label.txt
PARALLEL=${PARALLEL:-1} ./run-flint.sh -lv 4 > bench/soak-$label-server.log 2>&1 &
for _ in $(seq 180); do curl -s 127.0.0.1:8080/health | grep -q ok && break; sleep 1; done
pid=$(pgrep -f 'llama-server -m flint-nvfp4-gguf' | head -1)
echo "server pid $pid, $CLIENTS client(s), $MINUTES min" > "$out"

(while kill -0 "$pid" 2>/dev/null; do
    echo "sample $(date +%T) rss_mb $(( $(ps -o rss= -p "$pid") / 1024 )) vram_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$out"
    sleep 30
done) &
sampler=$!

end=$(( $(date +%s) + MINUTES*60 ))
client() {
    local n=0
    while [ "$(date +%s)" -lt "$end" ] && kill -0 "$pid" 2>/dev/null; do
        n=$((n + 1))
        r=$(python3 bench/agent_bench.py --episodes 4 --turns 12 2>&1 | tail -1)
        echo "client $1 iteration $n $(date +%T): $r" >> "$out"
    done
}
clients=()
for c in $(seq "$CLIENTS"); do client "$c" & clients+=($!); done
wait "${clients[@]}" 2>/dev/null
alive=$(kill -0 "$pid" 2>/dev/null && echo yes || echo NO)
kill -INT "$pid" 2>/dev/null; sleep 5; kill "$sampler" 2>/dev/null
echo "server alive at the end: $alive" >> "$out"
grep -c -i -E 'error|abort|assert|illegal|trap' bench/soak-$label-server.log | sed 's/^/error-like log lines: /' >> "$out"
grep -E 'hit rate|adaptive' bench/soak-$label-server.log | tail -3 >> "$out"
