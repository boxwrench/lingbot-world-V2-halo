#!/usr/bin/env bash
# P2.0 baseline capture on tagged accepted main (9cc986c).
# Waterfall + 2 A/A rolled runs, 15-action string, warmed-Y protocol.
# No optimization changes: code under test == main == tag.
set -euo pipefail
WT=/home/keith/Desktop/github/lingbot-world-V2-halo-wt-p20
PRIMARY=/home/keith/Desktop/github/lingbot-world-V2-halo
OUT=/tmp/p20-20260913
VENV="$PRIMARY/.venv/bin/python"
UPSTREAM="$PRIMARY/.upstream/lingbot-world-v2"
MODEL="$PRIMARY/models/lingbot-world-v2-1.3b-causal-fast"
ACTIONS="w w j w l l s s j w d d w l a"

export LINGBOT_PYTHON="$VENV" LINGBOT_MODEL_DIR="$MODEL" LINGBOT_UPSTREAM_DIR="$UPSTREAM"
export PYTHONPATH="$WT/scripts"

cd "$WT" || exit 1
echo "HEAD: $(git rev-parse HEAD)"; git status --short --branch
rocm-smi --showuse | grep "GPU use"
mkdir -p "$OUT"

env -i PATH="$PATH" LANG="${LANG:-C.UTF-8}" \
  LINGBOT_PYTHON="$VENV" \
  PYTORCH_ROCM_ARCH=gfx1151 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/env_report.sh "$OUT/environment" || exit 1

run_one() { # name port
  local name="$1" port="$2"
  export LINGBOT_BROWSER_OPEN_BROWSER=0 LINGBOT_BROWSER_MAX_ACTIONS=15
  export LINGBOT_BROWSER_TIMEOUT_SECONDS=900 LINGBOT_BROWSER_PORT="$port"
  export LINGBOT_BROWSER_OUTPUT_DIR="$OUT/$name"
  env -u DEEPSEEK_API_KEY -u MODEL_API_KEY -u GOG_KEYRING_PASSWORD \
    bash scripts/run_browser.sh > "$OUT/$name-server.log" 2>&1 & SERVER=$!
  READY=0
  for i in $(seq 1 120); do
    if "$VENV" -c "import socket; s=socket.create_connection(('127.0.0.1',$port), timeout=3); s.close()" 2>/dev/null; then READY=1; break; fi
    sleep 5
  done
  if [[ "$READY" != 1 ]]; then echo "$name: no listen"; kill -INT $SERVER; wait $SERVER; exit 1; fi
  set +e
  "$VENV" scripts/browser_benchmark.py \
    --url "http://127.0.0.1:$port/ws" --output "$OUT/$name/benchmark.json" \
    --ready-timeout 600 --action-timeout 180 --actions $ACTIONS
  RC=$?
  set -e
  kill -INT $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true
  if [[ $RC != 0 ]]; then echo "$name: benchmark rc=$RC"; exit $RC; fi
}

run_one waterfall 8781
run_one aa-run-1 8782
run_one aa-run-2 8783
echo "P2.0 RUNS COMPLETE"
