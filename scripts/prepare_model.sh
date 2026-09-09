#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${LINGBOT_MODEL_DIR:-"$ROOT_DIR/models/lingbot-world-v2-1.3b-causal-fast"}
MODEL_REPO=robbyant/lingbot-world-v2-1.3b-causal-fast
MODEL_REV=7e36a5f919f86cb4255cc9bfc30adb44963fbde1
AUX_REPO=robbyant/lingbot-world-v2-14b-causal-fast
AUX_REV=5c33dd40b213598c418fd25bff30fdbd23fd38a7

HF=${HF:-$(command -v hf || true)}
test -n "$HF" || { echo "error: Hugging Face 'hf' CLI not found" >&2; exit 1; }
mkdir -p "$MODEL_DIR"

"$HF" download "$MODEL_REPO" \
    model-00001-of-00006.safetensors \
    model-00002-of-00006.safetensors \
    model-00003-of-00006.safetensors \
    model-00004-of-00006.safetensors \
    model-00005-of-00006.safetensors \
    model-00006-of-00006.safetensors \
    model.safetensors.index.json \
    --revision "$MODEL_REV" --local-dir "$MODEL_DIR"

# The 1.3B repository currently publishes only the transformer shards. These
# assets are shared with the 14B release and are required by upstream's i2v
# pipeline. Do not include the 14B transformer shards.
"$HF" download "$AUX_REPO" \
    Wan2.1_VAE.pth \
    models_t5_umt5-xxl-enc-bf16.pth \
    google/umt5-xxl/special_tokens_map.json \
    google/umt5-xxl/spiece.model \
    google/umt5-xxl/tokenizer.json \
    google/umt5-xxl/tokenizer_config.json \
    --revision "$AUX_REV" --local-dir "$MODEL_DIR"

mkdir -p "$MODEL_DIR/transformers"
for shard in "$MODEL_DIR"/model-*.safetensors; do
    [[ -e "$shard" ]] || continue
    name=$(basename "$shard")
    [[ -e "$MODEL_DIR/transformers/$name" ]] || ln -s "../$name" "$MODEL_DIR/transformers/$name"
done
[[ -e "$MODEL_DIR/transformers/diffusion_pytorch_model.safetensors.index.json" ]] || \
    ln -s ../model.safetensors.index.json "$MODEL_DIR/transformers/diffusion_pytorch_model.safetensors.index.json"
cp "$ROOT_DIR/configs/wan-1.3b-causal-fast.json" "$MODEL_DIR/transformers/config.json"

for file in \
    "$MODEL_DIR/Wan2.1_VAE.pth" \
    "$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    "$MODEL_DIR/google/umt5-xxl/tokenizer.json" \
    "$MODEL_DIR/transformers/config.json" \
    "$MODEL_DIR/transformers/diffusion_pytorch_model.safetensors.index.json"; do
    test -e "$file" || { echo "error: missing prepared asset $file" >&2; exit 1; }
done

{
    echo "model_repo=$MODEL_REPO"
    echo "model_revision=$MODEL_REV"
    echo "aux_repo=$AUX_REPO"
    echo "aux_revision=$AUX_REV"
    find "$MODEL_DIR" -maxdepth 2 -type f -o -type l | sort | while read -r f; do stat -c '%s %n' "$f"; done
} > "$MODEL_DIR/manifest.txt"

echo "prepared model: $MODEL_DIR"
cat "$MODEL_DIR/manifest.txt"
