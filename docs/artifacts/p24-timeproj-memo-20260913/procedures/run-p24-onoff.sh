#!/usr/bin/env bash
# P2.5b: fair ON-vs-OFF pair, back-to-back on the same host state.
# Gate 1: ON latents bitwise 0.0 vs OFF latents (same noise/setup).
# Gate 2: per-chunk denoise_ms ON vs OFF (within-day comparison).
set -euo pipefail
WT=/home/keith/Desktop/github/lingbot-world-V2-halo-wt-p24
C2WT=/home/keith/Desktop/github/lingbot-world-V2-halo-wt-c2-screen
PRIMARY=/home/keith/Desktop/github/lingbot-world-V2-halo
OUT=/tmp/p24-eval-20260913
VENV="$PRIMARY/.venv/bin/python"
UPSTREAM="$PRIMARY/.upstream/lingbot-world-v2"
MODEL="$PRIMARY/models/lingbot-world-v2-1.3b-causal-fast"

cd "$WT" || exit 1
echo "HEAD: $(git rev-parse HEAD)"; git status --short --branch
rocm-smi --showuse | grep "GPU use"

run_screen() { # name memo_flag
  echo "=== $1 (memo=$2) ==="
  LINGBOT_TIMEPROJ_MEMO="$2" \
  PYTHONPATH="$WT/scripts:$UPSTREAM:$C2WT/scripts" PYTORCH_ROCM_ARCH=gfx1151 \
  HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  env -u DEEPSEEK_API_KEY -u MODEL_API_KEY -u GOG_KEYRING_PASSWORD \
  "$VENV" "$WT/scripts/c2_context_screen.py" \
    --upstream-dir "$UPSTREAM" --model-dir "$MODEL" --taehv-dir "$PRIMARY/.upstream/taehv" \
    --image "$UPSTREAM/examples/03/image.jpg" --action-path "$UPSTREAM/examples/03" \
    --screen-actions "w,w,j,w,l,l,s,s,j,w,d,d,w,l,a,a" --output-dir "$OUT/$1" 2>&1 | grep -E "timeproj_memo|status|chunks_scored" || exit 1
}

run_screen off-run 0
run_screen on-run 1

"$VENV" - <<EOF || exit 1
import json, torch
off = torch.load("$OUT/off-run/accepted_latents.pt", map_location="cpu", weights_only=False)["latents"].float()
on = torch.load("$OUT/on-run/accepted_latents.pt", map_location="cpu", weights_only=False)["latents"].float()
per = [(off[:, i:i+1] - on[:, i:i+1]).abs().max().item() for i in range(off.shape[1])]
print("ON-vs-OFF per-chunk max_abs:", [round(v, 6) for v in per])
print("worst:", max(per))
toff = json.load(open("$OUT/off-run/screen_trajectory.json"))
ton = json.load(open("$OUT/on-run/screen_trajectory.json"))
ds = [cg["denoise_ms"] - cr["denoise_ms"] for cr, cg in zip(toff["chunks"], ton["chunks"])]
print("per-chunk denoise delta ms (ON-OFF):", [round(v, 1) for v in ds])
print("mean delta:", round(sum(ds) / len(ds), 1))
EOF
echo P25B_DONE
