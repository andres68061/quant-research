#!/usr/bin/env bash
# Run the FMP ingestion waves in sequence, unattended.
#
# Waves run one at a time on purpose: the rate limiter lives inside a single
# process, so two concurrent runs would emit twice the intended calls/minute and
# risk the vendor throttling the key.
#
# Every wave is resumable — completion is a file on disk — so this script is safe
# to re-run, to interrupt, and to schedule. A wave that fails does not stop the
# ones after it: partial coverage of a later wave is more useful than none, and
# the journal records exactly what was missed.
#
# Usage:  nohup bash scripts/run_full_ingestion.sh "1 2 3 4" > logs/ingestion_all.out 2>&1 &

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/opt/anaconda3/envs/quant/bin/python"
WAVES="${1:-1 2 3 4}"
RATE="${RATE:-600}"
WORKERS="${WORKERS:-16}"

cd "$REPO" || exit 1
mkdir -p logs

# Refuse to start while another ingester is running, for the rate-limit reason above.
if pgrep -f "ingest_fmp.py" > /dev/null; then
    echo "$(date -u +%FT%TZ) another ingest_fmp.py is running; refusing to start a second"
    exit 1
fi

echo "$(date -u +%FT%TZ) starting ingestion for waves: $WAVES (rate=$RATE workers=$WORKERS)"

for wave in $WAVES; do
    echo "$(date -u +%FT%TZ) === wave $wave starting ==="
    "$PYTHON" scripts/ingest_fmp.py \
        --wave "$wave" \
        --rate "$RATE" \
        --workers "$WORKERS" \
        --log-file "logs/ingest_wave${wave}.log"
    status=$?
    echo "$(date -u +%FT%TZ) === wave $wave exited with status $status ==="
done

echo "$(date -u +%FT%TZ) all requested waves finished"
"$PYTHON" scripts/ingest_fmp.py --report
