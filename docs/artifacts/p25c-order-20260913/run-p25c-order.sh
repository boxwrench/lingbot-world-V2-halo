#!/usr/bin/env bash
# P2.5c: order-swap control. OFF, ON, OFF screen runs back-to-back.
# Separates memo effect from run-order confounds (clocks/thermals/drift).
set -euo pipefail
WT=/home/keith/Desktop/github/lingbot-world-V2-halo-wt-p24
PRIMARY=/home/keith/Desktop/github/lingbot-world-V2-halo
OUT=/tmp/p25c-20260913
VENV="$PRIMARY/.venv/bin/python"
UPSTREAM="$PRIMARY/.upstream/lingbot-world-v2"
MODEL="$PRIMARY/models/lingbot-world-v2-1.3b-causal-fast"

cd "$WT" || exit 1
mkdir -p "$OUT"
echo "HEAD: $(git rev-parse HEAD)"; git status --short --branch
rocm-smi --showuse | grep "GPU use"

run_screen() { # name memo_flag
  echo "=== $1 (memo=$2) ==="
  LINGBOT_TIMEPROJ_MEMO="$2" \
  PYTHONPATH="$WT/scripts:$UPSTREAM" PYTORCH_ROCM_ARCH=gfx1151 \
  HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  env -u DEEPSEEK_API_KEY -u MODEL_API_KEY -u GOG_KEYRING_PASSWORD \
  "$VENV" "$WT/scripts/c2_context_screen.py" \
    --upstream-dir "$UPSTREAM" --model-dir "$MODEL" --taehv-dir "$PRIMARY/.upstream/taehv" \
    --image "$UPSTREAM/examples/03/image.jpg" --action-path "$UPSTREAM/examples/03" \
    --screen-actions "w,w,j,w,l,l,s,s,j,w,d,d,w,l,a,a" --output-dir "$OUT/$1" 2> "$OUT/$1-stderr.log" || exit 1
  grep -h "timeproj_memo" "$OUT/$1-stderr.log" || echo "(no memo lines: memo off)"
}

run_screen run1-off 0
run_screen run2-on 1
run_screen run3-off 0

"$VENV" - <<EOF || exit 1
import json
trajs = {n: json.load(open(f"$OUT/{n}/screen_trajectory.json")) for n in ["run1-off", "run2-on", "run3-off"]}
base = [c["denoise_ms"] for c in trajs["run1-off"]["chunks"]]
for n in ["run2-on", "run3-off"]:
    ds = [cg - cb for cg, cb in zip([c["denoise_ms"] for c in trajs[n]["chunks"]], base)]
    print(f"{n} vs run1-off: mean {sum(ds)/len(ds):+.1f} range [{min(ds):+.1f},{max(ds):+.1f}]")
import torch
lats = {n: torch.load(f"$OUT/{n}/accepted_latents.pt", map_location="cpu", weights_only=False)["latents"].float() for n in trajs}
for n in ["run2-on", "run3-off"]:
    d = (lats[n] - lats["run1-off"]).abs().max().item()
    print(f"{n} vs run1-off latents max_abs: {d}")
EOF
echo P25C_DONE
