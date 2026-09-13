#!/usr/bin/env bash
# P2.1b: headless DiT module-hook profile on P2.0 baseline state.
# CUDA-event module attribution for one steady-state rolled chunk (18)
# plus SDPA layout probe. Attribution only, never acceptance latency.
set -euo pipefail
WT=/home/keith/Desktop/github/lingbot-world-V2-halo-wt-p20
PRIMARY=/home/keith/Desktop/github/lingbot-world-V2-halo
OUT=/tmp/p21-20260913/module-profile

export LINGBOT_PYTHON="$PRIMARY/.venv/bin/python"
export LINGBOT_MODEL_DIR="$PRIMARY/models/lingbot-world-v2-1.3b-causal-fast"
export LINGBOT_UPSTREAM_DIR="$PRIMARY/.upstream/lingbot-world-v2"

cd "$WT" || exit 1
echo "HEAD: $(git rev-parse HEAD)"; git status --short --branch
rocm-smi --showuse | grep "GPU use"
mkdir -p "$OUT"

PYTHONPATH="$LINGBOT_UPSTREAM_DIR:$WT/scripts" PYTORCH_ROCM_ARCH=gfx1151 \
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
env -u DEEPSEEK_API_KEY -u MODEL_API_KEY -u GOG_KEYRING_PASSWORD \
"$LINGBOT_PYTHON" "$WT/scripts/run_interactive.py" \
  --upstream-dir "$LINGBOT_UPSTREAM_DIR" --model-dir "$LINGBOT_MODEL_DIR" \
  --output-dir "$OUT" --size '480*832' --max-area-pixels 264192 \
  --denoise-schedule 3-drop-957 --local-attn-size 12 --sink-size 6 --seed 42 \
  --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward." \
  --image "$LINGBOT_UPSTREAM_DIR/examples/03/image.jpg" \
  --action-path "$LINGBOT_UPSTREAM_DIR/examples/03" \
  --vae-attention-backend math --vae-dtype fp16 \
  --display-decoder taehv --taehv-dir "$PRIMARY/.upstream/taehv" \
  --defer-clean-kv --profile-dit-from-chunk 18 --profile-attention-from-chunk 18 || exit 1
echo "P2.1B MODULE PROFILE COMPLETE"
