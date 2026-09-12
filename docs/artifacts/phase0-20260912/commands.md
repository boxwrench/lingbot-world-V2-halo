# Phase 0 commands

Run from `/home/keith/Desktop/github/lingbot-world-V2-halo`.

The environment capture was deliberately sanitized so global API credentials
could not enter the artifact:

```bash
env -i PATH="$PATH" LANG="${LANG:-C.UTF-8}" \
  LINGBOT_PYTHON="$PWD/.venv/bin/python" \
  PYTORCH_ROCM_ARCH=gfx1151 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/env_report.sh results/raw/phase0-20260912/environment-gate
```

Each A/A server used a distinct port and output directory (`N=1,2,3`, ports
8771–8773):

```bash
env -u DEEPSEEK_API_KEY -u MODEL_API_KEY -u GOG_KEYRING_PASSWORD \
  LINGBOT_BROWSER_OPEN_BROWSER=0 \
  LINGBOT_BROWSER_MAX_ACTIONS=15 \
  LINGBOT_BROWSER_TIMEOUT_SECONDS=600 \
  LINGBOT_BROWSER_PORT=877N \
  LINGBOT_BROWSER_OUTPUT_DIR=results/raw/phase0-20260912/aa-run-N \
  bash scripts/run_browser.sh

.venv/bin/python scripts/browser_benchmark.py \
  --url http://127.0.0.1:877N/ws \
  --output results/raw/phase0-20260912/aa-run-N/benchmark.json \
  --ready-timeout 420 --action-timeout 120 \
  --actions w w j w l l s s j w d d w l a
```

The final telemetry-preserving waterfall run used port 8774 and output
directory `results/raw/phase0-20260912/waterfall-run` with the same arguments.

The separate intrusive action-13 mechanism profile used the existing
`run_pure_compile_live.py` wrapper, 30 pure-helper blocks, prewarm, the pinned
TunableOp CSV, the same 12+6/three-step/TAEHV configuration, 13 scripted
actions, and `--profile-contexts 13`. Its complete command-line arguments are
also retained in `detailed-profile/live_metrics.json`.

The current exact-shape SDPA dispatch probe allocated the Q/K/V shapes and
strides observed by that real action, warmed three calls, then captured one
call with `torch.profiler`. It reported:

```text
aten::scaled_dot_product_attention
aten::_scaled_dot_product_flash_attention
aten::_flash_attention_forward
attn_fwd.kd
```

Generate the compact evidence package from the ignored raw records:

```bash
.venv/bin/python scripts/build_phase0_evidence.py
```

Validate the telemetry-only browser change:

```bash
PYTHONPATH=.upstream/lingbot-world-v2:scripts \
  .venv/bin/python -m pytest -q \
  tests/test_browser_protocol.py tests/test_browser_server_smoke.py
```

The test import initializes the ROCm-aware upstream package, so it must run
where `/dev/kfd` and the validated gfx1151 device are visible.
