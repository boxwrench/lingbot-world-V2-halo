"""State-free tensor islands for the opt-in Inductor experiment.

Every callable in this module receives tensors produced by eager LingBot
modules.  In particular, Linear and attention calls are deliberately outside
these helpers so the accepted TunableOp and SDPA paths remain untouched.
"""

from __future__ import annotations

import types
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


def _modulation(modulation: torch.Tensor, e: torch.Tensor):
    return (modulation.unsqueeze(0) + e).chunk(6, dim=2)


def _affine(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + bias


def _scaled_residual(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x + y * scale


def _scaled_residual_bf16(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x + y * scale


def _scaled_residual_float32(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x + y * scale


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _camera_update(
    x: torch.Tensor,
    hidden: torch.Tensor,
    plucker: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = hidden + plucker
    return (1 + scale) * x + shift, hidden


class _ScaledResidualDispatch:
    """Keep denoise/clean dtype variants separate from Dynamo guards."""

    def __init__(self):
        common = {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": True,
        }
        self._bf16 = torch.compile(
            _scaled_residual_bf16,
            name="lingbot_pure_scaled_residual_bf16",
            **common,
        )
        self._float32 = torch.compile(
            _scaled_residual_float32,
            name="lingbot_pure_scaled_residual_float32",
            **common,
        )

    def __call__(self, x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32:
            return self._float32(x, y, scale)
        return self._bf16(x, y, scale)


@dataclass
class PureTensorIslands:
    """Compiled pure helpers plus names used in the experiment report."""

    modulation: Callable
    affine: Callable
    scaled_residual: Callable
    add: Callable
    silu: Callable
    gelu_tanh: Callable
    camera_update: Callable
    helper_names: tuple[str, ...]

    @classmethod
    def compile(cls) -> "PureTensorIslands":
        common = {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": True,
        }
        return cls(
            modulation=torch.compile(_modulation, name="lingbot_pure_modulation", **common),
            affine=torch.compile(_affine, name="lingbot_pure_affine", **common),
            scaled_residual=_ScaledResidualDispatch(),
            add=torch.compile(_add, name="lingbot_pure_add", **common),
            silu=torch.compile(_silu, name="lingbot_pure_silu", **common),
            gelu_tanh=torch.compile(_gelu_tanh, name="lingbot_pure_gelu_tanh", **common),
            camera_update=torch.compile(
                _camera_update, name="lingbot_pure_camera_update", **common
            ),
            helper_names=(
                "modulation",
                "affine",
                "scaled_residual",
                "add",
                "silu",
                "gelu_tanh",
                "camera_update",
            ),
        )


def prewarm_pure_tensor_islands(
    islands: PureTensorIslands,
    device: torch.device,
    *,
    tokens: int = 1008,
    hidden_dim: int = 1536,
    ffn_dim: int = 8960,
) -> dict[str, object]:
    """Instantiate the exact live helper graphs without touching model state.

    The live 384x672 path calls these helpers with fixed current-frame shapes.
    Inputs are zero-filled tensors with the observed dtypes/layouts, so this
    routine consumes neither the application RNG nor any KV/decoder state. It
    must receive the same ``PureTensorIslands`` instance later installed on the
    real blocks; warming a temporary helper bank would not warm the live path.
    """
    if device.type != "cuda":
        raise ValueError("pure-helper prewarm requires a CUDA/HIP device")

    torch.cuda.synchronize(device)
    cpu_rng_before = torch.random.get_rng_state()
    gpu_rng_before = torch.cuda.get_rng_state(device)
    t0 = time.perf_counter()

    # These are deliberately zeros rather than randn tensors.  They have the
    # same contiguous shape/dtype contract as the live block calls without
    # consuming either the CPU or GPU generation RNG.
    shape = (1, tokens, hidden_dim)
    modulation = torch.nn.Parameter(
        torch.zeros(1, 6, hidden_dim, device=device, dtype=torch.bfloat16),
        requires_grad=False,
    )
    e = torch.zeros(1, tokens, 6, hidden_dim, device=device, dtype=torch.float32)
    x_bf16 = torch.zeros(*shape, device=device, dtype=torch.bfloat16)
    y_bf16 = torch.zeros(*shape, device=device, dtype=torch.bfloat16)
    x_float = torch.zeros(*shape, device=device, dtype=torch.float32)
    hidden_bf16 = torch.zeros(*shape, device=device, dtype=torch.bfloat16)
    plucker_bf16 = torch.zeros(*shape, device=device, dtype=torch.bfloat16)
    camera_scale_bf16 = torch.zeros(*shape, device=device, dtype=torch.bfloat16)
    camera_shift_bf16 = torch.zeros(*shape, device=device, dtype=torch.bfloat16)
    ffn_hidden_bf16 = torch.zeros(
        1, tokens, ffn_dim, device=device, dtype=torch.bfloat16
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        modulation_parts = islands.modulation(modulation, e)
        # The live block passes these exact singleton-dimension views to the
        # affine/residual helpers. Keeping their strides is necessary to avoid
        # a second Dynamo graph merely for a contiguous synthetic tensor.
        scale0 = modulation_parts[0].squeeze(2)
        scale1 = modulation_parts[1].squeeze(2)
        scale2 = modulation_parts[2].squeeze(2)
        scale5 = modulation_parts[5].squeeze(2)
        islands.affine(x_float, scale1, scale0)
        islands.scaled_residual(x_bf16, y_bf16, scale2)
        islands.scaled_residual(x_float, y_bf16, scale5)
        islands.add(x_float, y_bf16)
        islands.silu(hidden_bf16)
        islands.gelu_tanh(ffn_hidden_bf16)
        islands.camera_update(
            x_float,
            hidden_bf16,
            plucker_bf16,
            camera_scale_bf16,
            camera_shift_bf16,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    cpu_rng_unchanged = torch.equal(cpu_rng_before, torch.random.get_rng_state())
    gpu_rng_unchanged = torch.equal(gpu_rng_before, torch.cuda.get_rng_state(device))
    return {
        "elapsed_seconds": elapsed,
        "rng": {
            "cpu_unchanged": cpu_rng_unchanged,
            "gpu_unchanged": gpu_rng_unchanged,
        },
        "shape_contract": {
            "tokens": tokens,
            "hidden_dim": hidden_dim,
            "ffn_dim": ffn_dim,
            "modulation": {"shape": [1, 6, hidden_dim], "dtype": "torch.bfloat16"},
            "conditioning": {"shape": [1, tokens, 6, hidden_dim], "dtype": "torch.float32"},
            "activation": {"shape": list(shape), "dtype": "torch.float32"},
            "activation_bf16": {"shape": list(shape), "dtype": "torch.bfloat16"},
            "ffn_hidden": {"shape": [1, tokens, ffn_dim], "dtype": "torch.bfloat16"},
        },
        "helpers_invoked": list(islands.helper_names),
        "graphs_expected": 8,
    }


def make_pure_block_forward(islands: PureTensorIslands):
    """Return a block forward preserving eager attention/Linear/state calls."""

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        cross_attn_first_call=None,
        seq_lens_int=None,
    ):
        assert e.dtype == torch.float32
        e = islands.modulation(self.modulation, e)
        assert e[0].dtype == torch.float32

        # LayerNorm, attention, cache indexing, and KV writes remain eager.
        self_input = islands.affine(
            self.norm1(x).float(),
            e[1].squeeze(2),
            e[0].squeeze(2),
        )
        y = self.self_attn(
            self_input,
            seq_lens,
            grid_sizes,
            freqs,
            kv_cache,
            current_start,
            max_attention_size,
            frame_seqlen=frame_seqlen,
            seq_lens_int=seq_lens_int,
        )
        x = islands.scaled_residual(x, y, e[2].squeeze(2))

        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            camera_hidden = self.cam_injector_layer1(c2ws_plucker_emb)
            camera_hidden = islands.silu(camera_hidden)
            camera_hidden = self.cam_injector_layer2(camera_hidden)
            x, camera_hidden = islands.camera_update(
                x,
                camera_hidden,
                c2ws_plucker_emb,
                self.cam_scale_layer(camera_hidden),
                self.cam_shift_layer(camera_hidden),
            )

        # Cross-attention stays eager; only its residual and the FFN's
        # state-free pieces are compiled.  Both FFN Linear calls remain eager
        # so TunableOp can dispatch them.
        x = islands.add(
            x,
            self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                crossattn_cache=crossattn_cache,
                cross_attn_first_call=cross_attn_first_call,
            ),
        )
        ffn_input = islands.affine(
            self.norm2(x).float(),
            e[4].squeeze(2),
            e[3].squeeze(2),
        )
        ffn_hidden = self.ffn[0](ffn_input)
        ffn_hidden = islands.gelu_tanh(ffn_hidden)
        ffn_output = self.ffn[2](ffn_hidden)
        return islands.scaled_residual(x, ffn_output, e[5].squeeze(2))

    return forward


def install_on_blocks(model: torch.nn.Module, block_count: int) -> dict[str, object]:
    """Install helpers on leading causal blocks and return restoration state."""
    blocks = list(model.blocks)
    count = min(block_count, len(blocks))
    islands = PureTensorIslands.compile()
    forward = make_pure_block_forward(islands)
    originals = []
    for index, block in enumerate(blocks):
        if index < count:
            originals.append((block, block.forward))
            block.forward = types.MethodType(forward, block)
    return {
        "islands": islands,
        "originals": originals,
        "compiled_block_indices": list(range(count)),
        "total_model_blocks": len(blocks),
    }


def restore_blocks(installation: dict[str, object]) -> None:
    for block, original in installation["originals"]:
        block.forward = original
