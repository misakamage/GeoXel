from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import torch


@dataclass
class SegmentInput:
    """Canonical inputs for one GeoTTT-v3 segment.

    This object separates:
    - local segment frames
    - per-segment DOM / geo inputs
    - overlap propagation state
    - optional caller-provided context for losses
    """

    segment_id: int
    start_frame: int
    end_frame: int
    frames: List[dict]
    dom_image: Optional[torch.Tensor] = None
    anchor_world_xyz: Optional[Any] = None
    project_fn: Optional[Any] = None
    get_world_xyz_fn: Optional[Any] = None
    heading_fn: Optional[Any] = None
    query_points: Optional[torch.Tensor] = None
    image_paths: Optional[List[str]] = None
    inv_project_fn: Optional[Any] = None
    geo_elev: Optional[Any] = None
    dom_transform: Optional[Any] = None
    prev_summary: Optional[Dict[str, Any]] = None
    prev_submap_state: Optional[Any] = None
    prev_trajectory_global: Optional[np.ndarray] = None  # kept for backward compat; prefer prev_trajectory_local
    prev_trajectory_local: Optional[np.ndarray] = None
    prev_transform: Optional[Any] = None  # T_{k-1}: LocalModelTransform for previous segment
    # Segment-0 only: 3×3 GT first-frame camera R_c2w in ENU, derived from first-frame extrinsics
    gt_rotation: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class SegmentOutput:
    """Backward-compatible segment output bundle."""

    trajectory_global: np.ndarray
    trajectory_local: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    transform: Any = None
    submap_state: Any = None
    geo_cache: Any = None
