#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Collect a pid and all of its descendants (talker/predictor worker subprocesses
# are spawned children and don't show "examples/server.py" in their own cmdline).
collect_descendants() {
    local pid="$1"
    echo "$pid"
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        collect_descendants "$child"
    done
}

kill_existing_server() {
    local pids
    pids=$(pgrep -f "python.*examples/server\.py" || true)
    [ -z "$pids" ] && return 0

    local all_pids=""
    for pid in $pids; do
        all_pids="$all_pids $(collect_descendants "$pid")"
    done

    echo "Stopping existing server.py process(es):$all_pids"
    kill -TERM $all_pids 2>/dev/null || true
    for _ in $(seq 1 10); do
        sleep 1
        if ! kill -0 $all_pids 2>/dev/null; then
            break
        fi
    done
    kill -KILL $all_pids 2>/dev/null || true
}

kill_existing_server

export QWEN3_TTS_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base
export DEBUG_SAVE_AUDIO=0
export DECODER_MP_WORKER=1

python examples/server.py
