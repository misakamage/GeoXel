from typing import Any, Dict, Tuple
import torch

from .config import GeoV3Config


def compute_model_space_losses(
    model_xyz_init: torch.Tensor,
    config: GeoV3Config,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Optimize a segment trajectory in model space with true gradients.

    This is a lightweight TTT objective that preserves the initial anchor,
    encourages temporal smoothness, and regularizes the correction magnitude.
    """
    device = model_xyz_init.device
    dtype = model_xyz_init.dtype
    num_frames = model_xyz_init.shape[0]

    if model_xyz_init.numel() == 0:
        empty = model_xyz_init.clone()
        diagnostics = {
            "skipped": True,
            "anchor_count": 0,
            "quality_score": 0.0,
            "residual": 0.0,
            "inlier_ratio": 0.0,
            "ttt_losses": {"total_loss": 0.0},
        }
        return empty, torch.zeros_like(empty), diagnostics

    delta = torch.zeros_like(model_xyz_init, device=device, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(config.kv_lr))

    anchor_target = model_xyz_init[:1].detach()
    last_losses: Dict[str, float] = {}

    for step in range(max(1, int(config.num_steps))):
        optimizer.zero_grad(set_to_none=True)
        model_pred = model_xyz_init + delta

        anchor_loss = torch.mean((model_pred[:1] - anchor_target) ** 2)
        if num_frames > 1:
            smooth_loss = torch.mean((model_pred[1:] - model_pred[:-1]) ** 2)
        else:
            smooth_loss = torch.zeros((), device=device, dtype=dtype)
        reg_loss = torch.mean(delta ** 2)

        total = anchor_loss + float(config.reg_weight) * smooth_loss + float(config.reg_weight) * reg_loss
        total.backward()
        grad_norm = float(delta.grad.norm().detach().cpu()) if delta.grad is not None else 0.0
        if config.max_grad_norm and config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_([delta], config.max_grad_norm)
        optimizer.step()

        last_losses = {
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "smooth_loss": float(smooth_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
            "grad_norm": grad_norm,
        }
        print(
            f"[GeoV3][TTT] step={step:02d} total={last_losses['total_loss']:.6f} "
            f"anchor={last_losses['anchor_loss']:.6f} smooth={last_losses['smooth_loss']:.6f} "
            f"reg={last_losses['reg_loss']:.6f} grad_norm={grad_norm:.6f}"
        )

    model_final = (model_xyz_init + delta).detach()
    residual = float(torch.norm(model_final - model_xyz_init, dim=-1).mean().detach().cpu())
    diagnostics = {
        "skipped": False,
        "anchor_count": 1,
        "quality_score": 1.0 / (1.0 + residual),
        "residual": residual,
        "inlier_ratio": 1.0,
        "ttt_losses": last_losses,
    }
    return model_final, delta.detach(), diagnostics
