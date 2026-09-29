from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from .io import SegmentInput
from .canonicalization import LocalModelTransform


def can_use_overlap_propagation(segment_input: SegmentInput) -> bool:
    return (
        segment_input.prev_summary is not None
        and segment_input.prev_trajectory_global is not None
        and segment_input.prev_trajectory_local is not None
        and segment_input.prev_trajectory_global.shape[0] > 0
        and segment_input.prev_trajectory_local.shape[0] > 0
    )


def build_overlap_transform_from_previous_segment(
    segment_input: SegmentInput,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[LocalModelTransform, Dict[str, Any]]:
    """Deprecated: old overlap propagation that ignored s/R.

    Kept for reference; the active path is _estimate_transform_from_overlap_frames
    in defaults.py which uses Umeyama on overlap-frame pairs.
    """
    prev_world = np.asarray(segment_input.prev_trajectory_global, dtype=np.float32)
    prev_local = np.asarray(segment_input.prev_trajectory_local, dtype=np.float32)
    if prev_world.ndim != 2 or prev_world.shape[0] == 0:
        raise ValueError("invalid previous trajectory for overlap propagation")

    anchor_world_xyz = torch.as_tensor(prev_world[-1], device=device, dtype=dtype or torch.float32)
    R = torch.eye(3, device=anchor_world_xyz.device, dtype=anchor_world_xyz.dtype)
    s = torch.tensor(1.0, device=anchor_world_xyz.device, dtype=anchor_world_xyz.dtype)
    t = anchor_world_xyz.clone()
    transform = LocalModelTransform(anchor_world_xyz=anchor_world_xyz, R=R, s=s, t=t)
    diagnostics = {
        "overlap_mode": "propagated",
        "overlap_anchor_world_xyz": anchor_world_xyz.detach().cpu().numpy(),
        "overlap_prev_last_local": prev_local[-1],
        "overlap_prev_last_world": prev_world[-1],
    }
    return transform, diagnostics
