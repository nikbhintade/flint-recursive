#!/usr/bin/env bash
# T9 full run, unattended: Flint answers the full prompt set until a deadline, then features, then training.
# Each stage is skipped when its output is complete, so the script can simply be started again after a failure.
# Progress: data/t9/full/status.log (stages), generate.log, features.log, train.log.
#
# Usage: GEN_HOURS=21 TRAIN_ARGS="--epochs 2" tools/t9/full_run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
D=data/t9/full
GEN_HOURS=${GEN_HOURS:-21}
TRAIN_ARGS=${TRAIN_ARGS:-"--epochs 2"}
MIN_FREE_GB=100
PY=.venv-t9/bin/python

log() { echo "$(date '+%F %T') $*" | tee -a "$D/status.log"; }
free_gb() { df --output=avail -BG . | tail -1 | tr -dc 0-9; }
server_up() { curl -sf 127.0.0.1:8080/health >/dev/null; }
stop_server() {
    if pgrep -x llama-server >/dev/null; then kill $(pgrep -x llama-server); fi
    for _ in $(seq 60); do pgrep -x llama-server >/dev/null || return 0; sleep 1; done
    kill -9 $(pgrep -x llama-server) 2>/dev/null
}
start_server() {
    stop_server
    CTX=40960 PARALLEL=8 ./run-flint.sh >> "$D/server.log" 2>&1 &
    for _ in $(seq 300); do server_up && return 0; sleep 2; done
    log "server did not come up"; return 1
}

# 1. prompts: batch a (+ b if built), one file
if [ ! -s "$D/prompts.jsonl" ]; then
    cat "$D"/prompts-a.jsonl $( [ -s "$D/prompts-b.jsonl" ] && echo "$D/prompts-b.jsonl" ) > "$D/prompts.jsonl"
fi
N_PROMPTS=$(wc -l < "$D/prompts.jsonl")

# 2. generation until the deadline (kept across restarts in $D/deadline)
[ -s "$D/deadline" ] || echo $(( $(date +%s) + GEN_HOURS * 3600 )) > "$D/deadline"
DEADLINE=$(cat "$D/deadline")
log "generation: $N_PROMPTS prompts, deadline $(date -d @"$DEADLINE" '+%F %T')"
while [ "$(date +%s)" -lt "$DEADLINE" ] && [ "$(wc -l < "$D/sequences.jsonl" 2>/dev/null || echo 0)" -lt "$N_PROMPTS" ]; do
    [ "$(free_gb)" -gt "$MIN_FREE_GB" ] || { log "disk below ${MIN_FREE_GB} GB, stopping"; exit 1; }
    server_up || start_server || { sleep 60; continue; }
    if python3 tools/t9/generate.py --dir "$D" --workers 8 --max-tokens 1536 --deadline "$DEADLINE" >> "$D/generate.log" 2>&1; then
        break  # all prompts answered (or skipped) or deadline reached
    fi
    log "generate.py failed (see generate.log), restarting the server"; stop_server; sleep 10
done
stop_server
log "generation done: $(wc -l < "$D/sequences.jsonl") answers"

# 3. features (all answers; sequences cut at 4096 tokens)
if [ ! -e "$D/features.done" ]; then
    log "features: start"
    python3 tools/t9/features.py --dir "$D" --max-len 4096 > "$D/features.log" 2>&1 || { log "features failed"; exit 1; }
    touch "$D/features.done"
    log "features: done, $(du -h "$D/features.bf16" | cut -f1)"
fi

# 4. training
if [ ! -s "$D/head/config.json" ] || [ ! -e "$D/train.done" ]; then
    log "training: $TRAIN_ARGS"
    $PY tools/t9/train.py --dir "$D" $TRAIN_ARGS > "$D/train.log" 2>&1 || { log "training failed"; exit 1; }
    touch "$D/train.done"
    log "training: done"
fi
log "all stages done"
