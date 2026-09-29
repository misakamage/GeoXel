from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch


@dataclass
class LocalModelTransform:
    """Segment-local model-space transform.

    This object is intentionally light-weight: once a transform is selected
    for a segment, all subsequent computation stays in M_k.
    """

    anchor_world_xyz: torch.Tensor
    R: torch.Tensor
    s: torch.Tensor
    t: torch.Tensor

    def world_to_model(self, world_xyz: torch.Tensor) -> torch.Tensor:
        # enu = s * R @ model + t  →  model = R^T @ (enu - t) / s
        # Row-vector form: model = (enu - t) @ R / s
        return ((world_xyz - self.t[None, :]) @ self.R) / self.s

    def model_to_world(self, model_xyz: torch.Tensor) -> torch.Tensor:
        # enu = s * R @ model + t  (row-vector form: s * model @ R.T + t)
        return self.s * (model_xyz @ self.R.T) + self.t[None, :]


@dataclass
class GeoObservationCache:
    correspondences: Dict[int, Any]
    frame_pixels: Dict[int, np.ndarray]
    enu_positions: Dict[int, np.ndarray]
    frame_centers_enu: Dict[int, np.ndarray]
    pnp_results: Dict[int, Any]
    metadata: Dict[str, Any]


def build_local_model_transform(
    anchor_world_xyz: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> LocalModelTransform:
    anchor = torch.as_tensor(anchor_world_xyz, dtype=dtype or torch.float32, device=device)
    if anchor.ndim > 1:
        anchor = anchor[0]
    R = torch.eye(3, device=anchor.device, dtype=anchor.dtype)
    s = torch.tensor(1.0, device=anchor.device, dtype=anchor.dtype)
    t = anchor.clone()
    return LocalModelTransform(anchor_world_xyz=anchor, R=R, s=s, t=t)


def compose_transforms(T_outer: "LocalModelTransform", T_inner: "LocalModelTransform") -> "LocalModelTransform":
    """Compose T_outer ∘ T_inner: T_inner maps A→B, T_outer maps B→C, result maps A→C.

    Row-vector convention:  world = s * model @ R.T + t

    Derived composition:
        s_c = s_out * s_in
        R_c = R_out @ R_in
        t_c = s_out * t_in @ R_out.T + t_out
    anchor_world_xyz is set to t_c (ENU position of model-space origin under combined transform).
    """
    s_c = T_outer.s * T_inner.s
    R_c = T_outer.R @ T_inner.R
    t_c = T_outer.s * (T_inner.t @ T_outer.R.T) + T_outer.t
    return LocalModelTransform(
        anchor_world_xyz=t_c.detach().clone(),
        R=R_c,
        s=s_c,
        t=t_c,
    )


def canonicalize_world_xyz_to_model(
    world_xyz: torch.Tensor,
    transform: LocalModelTransform,
) -> torch.Tensor:
    return transform.world_to_model(world_xyz)


def canonicalize_segment_targets(
    world_xyz: torch.Tensor,
    transform: LocalModelTransform,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    model_xyz = canonicalize_world_xyz_to_model(world_xyz, transform)
    diagnostics = {
        "transform_R": transform.R.detach().cpu().numpy(),
        "transform_s": float(transform.s.detach().cpu()),
        "transform_t": transform.t.detach().cpu().numpy(),
    }
    return model_xyz, diagnostics
