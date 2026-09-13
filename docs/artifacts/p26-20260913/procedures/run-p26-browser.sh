#!/usr/bin/env bash
# P2.6: memo-ON full-runtime browser adjudication (accepted rolled protocol).
set -euo pipefail
WT=/home/keith/Desktop/github/lingbot-world-V2-halo-wt-p24
PRIMARY=/home/keith/Desktop/github/lingbot-world-V2-halo
OUT=/tmp/p26-20260913
VENV="$PRIMARY/.venv/bin/python"
UPSTREAM="$PRIMARY/.upstream/lingbot-world-v2"
MODEL="$PRIMARY/models/lingbot-world-v2-1.3b-causal-fast"
ACTIONS="w w j w l l s s j w d d w l a"

export LINGBOT_PYTHON="$VENV" LINGBOT_MODEL_DIR="$MODEL" LINGBOT_UPSTREAM_DIR="$UPSTREAM"
export PYTHONPATH="$WT/scripts" LINGBOT_TIMEPROJ_MEMO=${MEMO:-1}

cd "$WT" || exit 1
echo "HEAD: $(git rev-parse HEAD)"; git status --short --branch
rocm-smi --showuse | grep "GPU use"
mkdir -p "$OUT/memo-${MEMO:-1}"

export LINGBOT_BROWSER_OPEN_BROWSER=0 LINGBOT_BROWSER_MAX_ACTIONS=15
export LINGBOT_BROWSER_TIMEOUT_SECONDS=900 LINGBOT_BROWSER_PORT=${PORT:-8785}
export LINGBOT_BROWSER_OUTPUT_DIR="$OUT/memo-${MEMO:-1}"
env -u DEEPSEEK_API_KEY -u MODEL_API_KEY -u GOG_KEYRING_PASSWORD \
  bash scripts/run_browser.sh > "$OUT/memo-${MEMO:-1}-server.log" 2>&1 & SERVER=$!
READY=0
for i in $(seq 1 120); do
  if "$VENV" -c "import socket; s=socket.create_connection(('127.0.0.1',${PORT:-8785}), timeout=3); s.close()" 2>/dev/null; then READY=1; break; fi
  sleep 5
done
if [[ "$READY" != 1 ]]; then echo "no listen"; kill -INT $SERVER; wait $SERVER; exit 1; fi
"$VENV" /tmp/wait-ready.py "http://127.0.0.1:${PORT:-8785}/ws" 1500 || { echo "never READY"; kill -INT $SERVER; wait $SERVER; exit 1; }
set +e
"$VENV" scripts/browser_benchmark.py \
  --url http://127.0.0.1:${PORT:-8785}/ws --output "$OUT/memo-${MEMO:-1}/benchmark.json" \
  --ready-timeout 600 --action-timeout 180 --actions $ACTIONS
RC=$?
set -e
kill -INT $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true
if [[ $RC != 0 ]]; then echo "benchmark rc=$RC"; exit $RC; fi
grep -c "timeproj_memo" "$OUT/memo-${MEMO:-1}-server.log"
echo "P2.6 RUN COMPLETE"
