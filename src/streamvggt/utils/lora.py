"""
LoRA (Low-Rank Adaptation) utilities for VGGT test-time adaptation.

Injects lightweight low-rank adapters into frozen attention layers,
following the CAPA approach (Ke et al., 2026).
"""

import math
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple


class LoRALinear(nn.Module):
    """Drop-in wrapper that adds a LoRA branch to a frozen nn.Linear.

    output = frozen_linear(x) + (x @ A^T @ B^T) * (alpha / rank)

    Only A and B are trainable; the original weight is kept frozen.
    """

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = None):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(original).__name__}")
        if int(rank) <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        resolved_alpha = 2.0 * int(rank) if alpha is None else float(alpha)
        if not math.isfinite(resolved_alpha) or resolved_alpha <= 0.0:
            raise ValueError(f"LoRA alpha must be finite and positive, got {alpha}")

        self.original = original
        self._original_requires_grad = tuple(
            bool(parameter.requires_grad) for parameter in self.original.parameters()
        )
        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features
        device = original.weight.device
        dtype = torch.float32  # LoRA params always in fp32 for stability

        self.rank = int(rank)
        self.alpha = resolved_alpha
        self.scaling = self.alpha / self.rank

        # Low-rank factors: A projects down, B projects up
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))

        # Kaiming init for A, zero init for B → net zero at start
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        # LoRA path in fp32 for numerical stability, then cast back
        x_fp32 = x.float()
        lora_out = (x_fp32 @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base + lora_out.to(base.dtype)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = None,
    target_blocks: str = "all",
    target_layers: str = "qkv",
    target_block_indices: Optional[List[int]] = None,
) -> Tuple[List[nn.Parameter], int]:
    """Inject LoRA adapters into VGGT aggregator attention layers.

    Args:
        model: StreamVGGT model.
        rank: LoRA rank.
        alpha: LoRA scaling factor (default: 2 * rank).
        target_blocks: Which blocks to inject into.
            "all" — all 48 blocks (24 frame + 24 global)
            "frame" — only frame blocks
            "global" — only global blocks
        target_layers: Which linear layers within each attention block.
            "qkv" — only qkv projection
            "qkv_proj" — qkv + output projection
            "all" — qkv + proj + mlp fc1/fc2
        target_block_indices: Optional zero-based block indices within each
            selected frame/global block family. None selects every block.

    Returns:
        (lora_params, num_params): list of all trainable LoRA parameters
            and total parameter count.
    """
    rank = int(rank)
    alpha = 2.0 * rank if alpha is None else float(alpha)
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"LoRA alpha must be finite and positive, got {alpha}")
    if target_blocks not in {"all", "frame", "global"}:
        raise ValueError(
            "target_blocks must be one of {'all', 'frame', 'global'}, "
            f"got {target_blocks!r}"
        )
    if target_layers not in {"qkv", "qkv_proj", "all"}:
        raise ValueError(
            "target_layers must be one of {'qkv', 'qkv_proj', 'all'}, "
            f"got {target_layers!r}"
        )
    normalized_indices = (
        None
        if target_block_indices is None
        else sorted(set(int(index) for index in target_block_indices))
    )

    aggregator = model.aggregator
    if any(isinstance(module, LoRALinear) for module in aggregator.modules()):
        raise RuntimeError("LoRA adapters are already injected into this model")
    block_lists = []

    if target_blocks in ("all", "frame"):
        block_lists.append(("frame", aggregator.frame_blocks))
    if target_blocks in ("all", "global"):
        block_lists.append(("global", aggregator.global_blocks))

    lora_params = []
    n_injected = 0

    for name, blocks in block_lists:
        for idx, block in enumerate(blocks):
            if normalized_indices is not None and idx not in normalized_indices:
                continue
            attn = block.attn

            # QKV
            if "qkv" in target_layers or target_layers == "all":
                lora_qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)
                attn.qkv = lora_qkv
                lora_params.extend(lora_qkv.trainable_parameters())
                n_injected += 1

            # Output projection
            if target_layers in ("qkv_proj", "all"):
                lora_proj = LoRALinear(attn.proj, rank=rank, alpha=alpha)
                attn.proj = lora_proj
                lora_params.extend(lora_proj.trainable_parameters())
                n_injected += 1

            # MLP
            if target_layers == "all":
                mlp = block.mlp
                lora_fc1 = LoRALinear(mlp.fc1, rank=rank, alpha=alpha)
                mlp.fc1 = lora_fc1
                lora_params.extend(lora_fc1.trainable_parameters())
                n_injected += 1

                lora_fc2 = LoRALinear(mlp.fc2, rank=rank, alpha=alpha)
                mlp.fc2 = lora_fc2
                lora_params.extend(lora_fc2.trainable_parameters())
                n_injected += 1

    num_params = sum(p.numel() for p in lora_params)
    if n_injected == 0:
        raise ValueError(
            "LoRA target_block_indices selected no layers: "
            f"{normalized_indices!r}"
        )
    diagnostics = {
        "rank": rank,
        "alpha": alpha,
        "target_blocks": target_blocks,
        "target_layers": target_layers,
        "target_block_indices": normalized_indices,
        "injected_linear_count": int(n_injected),
        "trainable_parameter_count": int(num_params),
    }
    setattr(model, "_streamvggt_lora_diagnostics", diagnostics)
    print(f"[LoRA] Injected into {n_injected} layers, "
          f"{num_params:,} trainable params (rank={rank}, alpha={alpha:g})")

    return lora_params, num_params


def get_lora_diagnostics(model: nn.Module) -> Dict[str, Any]:
    """Return JSON-safe diagnostics for the currently injected adapters."""
    diagnostics = getattr(model, "_streamvggt_lora_diagnostics", None)
    return dict(diagnostics) if isinstance(diagnostics, dict) else {}


def remove_lora(model: nn.Module) -> None:
    """Remove all LoRA wrappers and restore the wrapped layers' grad flags.

    The operation is idempotent so it is safe to call from a ``finally`` block.
    """
    aggregator = model.aggregator
    removed = 0

    def _unwrap(module: nn.Module, name: str) -> None:
        nonlocal removed
        wrapped = getattr(module, name)
        if not isinstance(wrapped, LoRALinear):
            return
        original = wrapped.original
        for parameter, requires_grad in zip(
            original.parameters(), wrapped._original_requires_grad
        ):
            parameter.requires_grad_(requires_grad)
        setattr(module, name, original)
        removed += 1

    for blocks in [aggregator.frame_blocks, aggregator.global_blocks]:
        for block in blocks:
            attn = block.attn
            _unwrap(attn, "qkv")
            _unwrap(attn, "proj")

            mlp = block.mlp
            _unwrap(mlp, "fc1")
            _unwrap(mlp, "fc2")

    if hasattr(model, "_streamvggt_lora_diagnostics"):
        delattr(model, "_streamvggt_lora_diagnostics")
    if removed:
        print(f"[LoRA] Removed {removed} adapters; original layers restored.")
