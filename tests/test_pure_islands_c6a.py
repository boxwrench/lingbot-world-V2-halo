from __future__ import annotations

import torch
import torch.nn.functional as F

from pure_compile_helpers import _norm_affine, _rmsnorm


def eager_layernorm_affine(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # WanLayerNorm (eps=1e-6, no affine params) then the block affine step.
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6).to(x.dtype).float() * (1 + scale) + bias


def eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # WanRMSNorm.forward on wan/modules/model.py (verified identical copy in
    # wan/modules/model_causal.py).
    raw = x.float()
    return (raw * torch.rsqrt(raw.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype) * weight


def test_norm_affine_matches_eager() -> None:
    torch.manual_seed(7)
    x = torch.randn(1, 32, 48, dtype=torch.bfloat16)
    scale = torch.randn(1, 32, 48)
    bias = torch.randn(1, 32, 48)
    delta = (_norm_affine(x, scale, bias).float() - eager_layernorm_affine(x, scale, bias)).abs()
    assert float(delta.max()) <= 2e-6, float(delta.max())


def test_rmsnorm_matches_eager() -> None:
    torch.manual_seed(11)
    x = torch.randn(1, 32, 48, dtype=torch.bfloat16)
    weight = torch.ones(48)
    for eps in (1e-6, 1e-5):
        delta = (_rmsnorm(x, weight, eps).float() - eager_rmsnorm(x, weight, eps)).abs()
        assert float(delta.max()) <= 2e-6, (eps, float(delta.max()))
