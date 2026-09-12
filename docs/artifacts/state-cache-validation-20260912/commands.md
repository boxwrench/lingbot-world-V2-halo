# Commands used

Repository isolation and provenance:

```bash
git status --short --branch
git worktree list
git log --oneline -10
git worktree add -b experiment/state-cache-validation \
  /home/keith/Desktop/github/lingbot-world-V2-halo-wt-state-cache-validation add8a69
git clone /home/keith/Desktop/github/lingbot-world-V2-halo/.upstream/lingbot-world-v2 \
  .upstream/lingbot-world-v2
bash scripts/prepare_upstream.sh
```

GPU ownership checks (run immediately before each correctness-sensitive run):

```bash
rocm-smi --showuse --showmemuse --showpids
ps -eo pid,etime,stat,pcpu,pmem,rss,cmd | \
  rg 'python|ffmpeg|ComfyUI|run_experiment|run_live|capture_states'
```

Deterministic minimal fixtures:

```bash
PYTHONPATH=.upstream/lingbot-world-v2:scripts \
  /home/keith/Desktop/github/lingbot-world-V2-halo/.venv/bin/python \
  scripts/state_cache_validation.py \
  --mode fixture \
  --output /tmp/state-cache-fixtures.json
```

Real LingBot gfx1151 validation:

```bash
PYTHONPATH=.upstream/lingbot-world-v2:scripts \
PYTORCH_ROCM_ARCH=gfx1151 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  /home/keith/Desktop/github/lingbot-world-V2-halo/.venv/bin/python \
  scripts/state_cache_validation.py \
  --mode gpu \
  --output /tmp/state-cache-gpu.json \
  --model-dir /home/keith/Desktop/github/lingbot-world-V2-halo/models/lingbot-world-v2-1.3b-causal-fast \
  --image .upstream/lingbot-world-v2/examples/03/image.jpg \
  --action-path .upstream/lingbot-world-v2/examples/03 \
  --chunks 9 \
  --local-attn-size 4 \
  --sink-size 1 \
  --max-area-pixels 264192 \
  --seed 42
```

Artifact validation:

```bash
python3 -m py_compile scripts/state_cache_validation.py
python3 -m json.tool docs/artifacts/state-cache-validation-20260912/summary.json
python3 -m json.tool docs/artifacts/state-cache-validation-20260912/environment.json
git diff --check
PYTHONPATH=.upstream/lingbot-world-v2:scripts \
  /home/keith/Desktop/github/lingbot-world-V2-halo/.venv/bin/python -m pytest -q
```

Final repository test result: `13 passed in 3.80s`.
