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
#   - It holds a power assertion while fetching. A laptop that sleeps stops
#     ingesting: one run lost 17 of 25.6 wall-clock hours to idle sleep while
#     averaging a healthy 568 calls/min whenever it was actually awake.
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

# Match only a real Python process running the ingester, never this script, a
# pgrep, or a shell command that merely mentions the name. A bare
# `pgrep -f ingest_fmp.py` matches any `grep ingest_fmp.py` a human or a status
# check happens to run, which once left the supervisor waiting on a process that
# did not exist.
running_ingester() {
    pgrep -f "bin/python.*scripts/ingest_fmp\.py" | grep -v "^$$\$"
}

beat() { printf '%s\n' "$(date -u +%FT%TZ) $*" > "$HEARTBEAT"; }

# Exclusive supervisor lock, held for this whole run.
#
# Checking "is an ingester running?" is not enough: between two waves there is a
# window with no ingester, and a second supervisor starting in that window would
# run its own waves alongside this one at twice the intended call rate. mkdir is
# atomic on every filesystem we care about, so it is the lock primitive here
# (macOS ships no flock binary).
#
# Exiting non-zero when the lock is held is deliberate: launchd's KeepAlive then
# retries after ThrottleInterval, so it takes over cleanly whenever the current
# supervisor finishes or dies, instead of giving up permanently.
LOCKDIR="$REPO/data/quality/ingest_supervisor.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    holder=$(cat "$LOCKDIR/pid" 2>/dev/null)
    if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
        log "supervisor pid $holder already holds the lock; exiting for a later retry"
        exit 1
    fi
    log "clearing a stale lock left by pid ${holder:-unknown}"
    rm -rf "$LOCKDIR"
    mkdir "$LOCKDIR" 2>/dev/null || { log "could not acquire lock; retrying later"; exit 1; }
fi
echo "$$" > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM

# An ingester may still be running from a supervisor that died without cleaning
# up. Wait it out rather than competing with it.
waited=0
while running_ingester > /dev/null; do
    if [ "$waited" -eq 0 ]; then
        log "an orphaned ingest_fmp.py is running; waiting for it to finish"
        beat "waiting for an orphaned ingester"
    fi
    waited=$((waited + 1))
    if [ "$waited" -gt 2880 ]; then
        log "orphaned ingester still running after 24h; exiting non-zero so launchd retries"
        exit 1
    fi
    sleep 30
done

log "starting waves: $WAVES (rate=$RATE workers=$WORKERS)"
failures=0

for wave in $WAVES; do
    log "=== wave $wave starting ==="
    beat "wave $wave running"
    # caffeinate holds the assertion only for as long as the ingester runs, so a
    # finished or crashed wave cannot leave the machine permanently awake.
    #   -i no idle sleep   -m no disk sleep   -s no sleep while on AC power
    # On battery a closed lid still sleeps: keep the machine plugged in for an
    # unattended overnight backfill.
    caffeinate -ims "$PYTHON" scripts/ingest_fmp.py \
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
