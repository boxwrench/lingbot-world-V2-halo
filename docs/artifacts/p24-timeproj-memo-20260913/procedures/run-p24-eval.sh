#!/usr/bin/env bash
# P2.5: timeproj-memo correctness + effect gate.
# Same 16-action C2 procedure on the p24 branch with LINGBOT_TIMEPROJ_MEMO=1.
# Gate 1 (correctness): accepted latents bitwise 0.0 vs committed C2 A1.
# Gate 2 (effect): per-chunk denoise_ms vs A1 trajectory (same noise/setup).
set -euo pipefail
WT=/home/keith/Desktop/github/lingbot-world-V2-halo-wt-p24
C2WT=/home/keith/Desktop/github/lingbot-world-V2-halo-wt-c2-screen
PRIMARY=/home/keith/Desktop/github/lingbot-world-V2-halo
OUT=/tmp/p24-eval-20260913
VENV="$PRIMARY/.venv/bin/python"
UPSTREAM="$PRIMARY/.upstream/lingbot-world-v2"
MODEL="$PRIMARY/models/lingbot-world-v2-1.3b-causal-fast"
C2A1="$C2WT/docs/artifacts/c2-screen-20260912/A1/accepted_latents.pt"

export LINGBOT_TIMEPROJ_MEMO=1
cd "$WT" || exit 1
echo "HEAD: $(git rev-parse HEAD)"; git status --short --branch
rocm-smi --showuse | grep "GPU use"
mkdir -p "$OUT"

PYTHONPATH="$WT/scripts:$UPSTREAM:$C2WT/scripts" PYTORCH_ROCM_ARCH=gfx1151 \
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
env -u DEEPSEEK_API_KEY -u MODEL_API_KEY -u GOG_KEYRING_PASSWORD \
"$VENV" "$WT/scripts/c2_context_screen.py" \
  --upstream-dir "$UPSTREAM" --model-dir "$MODEL" --taehv-dir "$PRIMARY/.upstream/taehv" \
  --image "$UPSTREAM/examples/03/image.jpg" --action-path "$UPSTREAM/examples/03" \
  --screen-actions "w,w,j,w,l,l,s,s,j,w,d,d,w,l,a,a" --output-dir "$OUT/memo-A1" || exit 1

"$VENV" - <<EOF || exit 1
import json, torch
ref = torch.load("$C2A1", map_location="cpu", weights_only=False)["latents"].float()
got = torch.load("$OUT/memo-A1/accepted_latents.pt", map_location="cpu", weights_only=False)["latents"].float()
assert ref.shape == got.shape, (tuple(ref.shape), tuple(got.shape))
per = [(ref[:, i:i+1] - got[:, i:i+1]).abs().max().item() for i in range(ref.shape[1])]
print("per-chunk max_abs:", [round(v, 6) for v in per])
print("worst:", max(per))
tref = json.load(open("$C2WT/docs/artifacts/c2-screen-20260912/A1/screen_trajectory.json"))
tgot = json.load(open("$OUT/memo-A1/screen_trajectory.json"))
ds = []
for cr, cg in zip(tref["chunks"], tgot["chunks"]):
    ds.append(cg["denoise_ms"] - cr["denoise_ms"])
print("per-chunk denoise delta ms (memo-A1):", [round(v, 1) for v in ds])
print("mean delta:", round(sum(ds) / len(ds), 1))
EOF
echo P24_EVAL_DONE
