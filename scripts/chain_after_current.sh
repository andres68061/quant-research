#!/usr/bin/env bash
# Wait for the running ingestion supervisor to finish, then start another set of
# waves. Used to queue follow-up work (a re-fetch after a spec fix) without
# running two ingesters at once, which would double the vendor call rate.
#
# Usage: INGEST_WAVES="1" bash scripts/chain_after_current.sh
set -uo pipefail
REPO="/Users/andres/Downloads/Cursor/quant"
cd "$REPO" || exit 1
WAVES="${INGEST_WAVES:-1}"

running() { pgrep -f "bin/python.*scripts/ingest_fmp\.py" | grep -v "^$$\$"; }

echo "$(date -u +%FT%TZ) [chain] waiting for the current ingestion to finish"
while running > /dev/null || pgrep -f "scripts/ingest_daemon\.sh" > /dev/null; do
    sleep 60
done
echo "$(date -u +%FT%TZ) [chain] current ingestion done; starting waves: $WAVES"
exec /opt/anaconda3/envs/quant/bin/python scripts/spawn_ingest_daemon.py --waves "$WAVES" --force
