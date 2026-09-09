#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PYTHON=${LINGBOT_BASE_PYTHON:-/home/keith/ciru-ling-runtime/.venv/bin/python}
VENV_DIR=${LINGBOT_VENV_DIR:-"$ROOT_DIR/.venv"}

test -x "$BASE_PYTHON" || { echo "error: expected known-good ROCm Python at $BASE_PYTHON" >&2; exit 1; }

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
fi

BASE_SITE=$($BASE_PYTHON -c 'import site; print(site.getsitepackages()[0])')
"$VENV_DIR/bin/python" -c 'from pathlib import Path; import sys; Path(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "strix_halo_rocm_base.pth").write_text("'"$BASE_SITE"'" + chr(10))'

# Install only packages absent from the known-good ROCm environment. --no-deps
# is intentional: resolving torch here could replace ROCm Torch with a CUDA or
# CPU wheel. Dependencies are supplied by the pinned base environment.
"$VENV_DIR/bin/uv" --version >/dev/null 2>&1 || true
UV=${UV:-$(command -v uv || true)}
test -n "$UV" || { echo "error: uv is required to add packages to the isolated environment" >&2; exit 1; }

"$UV" pip install --python "$VENV_DIR/bin/python" --no-deps \
    'diffusers==0.31.0' \
    'imageio==2.37.0' \
    'easydict==1.13' \
    'ftfy==6.3.1' \
    'wcwidth==0.2.13'

"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata as md
import sys
import torch

print("python", sys.version)
print("torch", torch.__version__, "hip", torch.version.hip)
print("torchvision", md.version("torchvision"))
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
assert torch.version.hip, "isolated environment is not using ROCm Torch"
assert torch.cuda.is_available(), "ROCm device is not visible"
assert torch.cuda.is_bf16_supported(), "BF16 is not supported by the target runtime"
PY

echo "isolated environment ready: $VENV_DIR"
