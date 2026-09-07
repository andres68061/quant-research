#!/usr/bin/env bash
# Unattended supervisor for the FMP ingestion waves.
#
# This script is the unit of work that launchd runs. It must be safe to start at
# any time, from any state, with no human watching:
#
#   - It never runs two ingesters at once. The rate limiter lives inside a single
#     process, so a second one would double the calls/minute and risk the vendor
#     throttling the key. If another ingester is already running (started by hand,
#     or a previous daemon invocation that outlived its supervisor), this one
#     waits for it rather than competing with it.
#   - Every wave is resumable, because completion is a file on disk. A wave that
#     is killed mid-flight loses only its in-flight requests.
#   - A wave that exits non-zero does not stop the ones after it. Partial coverage
#     of a later wave is more useful than none, and the journal records exactly
#     what was missed.
#   - It writes a heartbeat so progress can be checked without attaching to it.
#
# Exit codes matter to launchd: 0 means "all waves are complete, do not restart",
# non-zero means "something failed, restart me after ThrottleInterval".

set -uo pipefail

REPO="/Users/andres/Downloads/Cursor/quant"
PYTHON="/opt/anaconda3/envs/quant/bin/python"
WAVES="${INGEST_WAVES:-1 2 3 4}"
RATE="${INGEST_RATE:-600}"
WORKERS="${INGEST_WORKERS:-16}"
HEARTBEAT="$REPO/data/quality/ingest_heartbeat.txt"

cd "$REPO" || exit 1
mkdir -p logs data/quality

log() { echo "$(date -u +%FT%TZ) [daemon] $*"; }

beat() { printf '%s\n' "$(date -u +%FT%TZ) $*" > "$HEARTBEAT"; }

# Wait out any ingester already running, whoever started it.
waited=0
while pgrep -f "ingest_fmp.py" > /dev/null; do
    if [ "$waited" -eq 0 ]; then
        log "another ingest_fmp.py is running; waiting for it to finish"
        beat "waiting for an existing ingester"
    fi
    waited=$((waited + 1))
    # Give up after 24h so a wedged process cannot block ingestion forever.
    if [ "$waited" -gt 2880 ]; then
        log "existing ingester still running after 24h; exiting non-zero so launchd retries"
        exit 1
    fi
    sleep 30
done

log "starting waves: $WAVES (rate=$RATE workers=$WORKERS)"
failures=0

for wave in $WAVES; do
    log "=== wave $wave starting ==="
    beat "wave $wave running"
    "$PYTHON" scripts/ingest_fmp.py \
        --wave "$wave" \
        --rate "$RATE" \
        --workers "$WORKERS" \
        --log-file "logs/ingest_wave${wave}.log"
    status=$?
    log "=== wave $wave exited with status $status ==="
    [ "$status" -ne 0 ] && failures=$((failures + 1))
done

if [ "$failures" -gt 0 ]; then
    log "$failures wave(s) exited non-zero; exiting 1 so launchd retries the remainder"
    beat "finished with $failures failed wave(s)"
    exit 1
fi

log "all waves complete"
beat "all waves complete"
exit 0
