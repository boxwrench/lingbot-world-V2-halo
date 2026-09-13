#!/usr/bin/env bash
# P2.1: quiet intrusive denoise profile on P2.0 baseline state (main 9cc986c).
# Same procedure as C4 action-13 profile; fresh waterfall already captured in P2.0.
set -euo pipefail
WT=/home/keith/Desktop/github/lingbot-world-V2-halo-wt-p20
PRIMARY=/home/keith/Desktop/github/lingbot-world-V2-halo
OUT=/tmp/p21-20260913

export LINGBOT_PYTHON="$PRIMARY/.venv/bin/python"
export LINGBOT_MODEL_DIR="$PRIMARY/models/lingbot-world-v2-1.3b-causal-fast"
export LINGBOT_UPSTREAM_DIR="$PRIMARY/.upstream/lingbot-world-v2"

cd "$WT" || exit 1
echo "HEAD: $(git rev-parse HEAD)"; git status --short --branch
rocm-smi --showuse | grep "GPU use"
mkdir -p "$OUT/detailed-profile"

PYTHONPATH="$LINGBOT_UPSTREAM_DIR:$WT/scripts" PYTORCH_ROCM_ARCH=gfx1151 \
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
env -u DEEPSEEK_API_KEY -u MODEL_API_KEY -u GOG_KEYRING_PASSWORD \
  -u PYTORCH_TUNABLEOP_ENABLED -u PYTORCH_TUNABLEOP_TUNING \
  -u PYTORCH_TUNABLEOP_RECORD_UNTUNED -u PYTORCH_TUNABLEOP_FILENAME \
"$LINGBOT_PYTHON" "$WT/scripts/run_pure_compile_live.py" \
  --pure-compile-block-count 30 \
  --tunableop-results "$WT/docs/artifacts/tunable-op-20260910/tunableop_results.csv" \
  --prewarm-pure-compile \
  --upstream-dir "$LINGBOT_UPSTREAM_DIR" --model-dir "$LINGBOT_MODEL_DIR" \
  --output-dir "$OUT/detailed-profile" --size '480*832' --max-area-pixels 264192 \
  --max-actions 15 --denoise-schedule 3-drop-957 \
  --local-attn-size 12 --sink-size 6 --seed 42 \
  --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward." \
  --image "$LINGBOT_UPSTREAM_DIR/examples/03/image.jpg" \
  --action-path "$LINGBOT_UPSTREAM_DIR/examples/03" \
  --vae-attention-backend math --vae-dtype fp16 \
  --display-decoder taehv --taehv-dir "$PRIMARY/.upstream/taehv" \
  --defer-clean-kv --max-seconds 1200 \
  --scripted-actions 'w,w,j,w,l,l,s,s,j,w,d,d,w,l,a' --profile-contexts 13 || exit 1
echo "P2.1 PROFILE COMPLETE"
