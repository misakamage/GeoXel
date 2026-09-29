from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SubmapState:
    """Explicit local submap state for one segment."""

    segment_id: int
    model_space: Any = None
    transform: Any = None
    anchor_world_xyz: Optional[np.ndarray] = None
    source_summary: Optional[Dict[str, Any]] = None


@dataclass
class GeoObservationCache:
    """Explicit cache for DOM/geo observations used within a segment."""

    correspondences: Dict[int, Any] = field(default_factory=dict)
    frame_pixels: Dict[int, np.ndarray] = field(default_factory=dict)
    enu_positions: Dict[int, np.ndarray] = field(default_factory=dict)
    frame_centers_enu: Dict[int, np.ndarray] = field(default_factory=dict)
    pnp_results: Dict[int, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentTTTResult:
    """Structured result for a single segment TTT pass."""

    trajectory_global: np.ndarray
    trajectory_local: np.ndarray
    transform: Any = None
    submap_state: Optional[SubmapState] = None
    geo_cache: Optional[GeoObservationCache] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentRuntimeState:
    """Mutable state that must not cross segment boundaries."""

    segment_id: int
    start_frame: int
    end_frame: int

    # Hard-reset state
    anchor_frame_idx: int = 0
    enu_offset: Optional[np.ndarray] = None
    past_kv: Any = None
    past_kv_cam: Any = None
    sim2: Any = None
    sim3: Any = None
    dom_crop_center: Optional[np.ndarray] = None

    # Segment-local optimization / caches
    optimizer_state: Dict[str, Any] = field(default_factory=dict)
    correspondence_cache: Dict[int, Any] = field(default_factory=dict)
    loss_history: List[float] = field(default_factory=list)
    ttt_outputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentSummary:
    """Read-only summary passed to the next segment."""

    segment_id: int
    start_frame: int
    end_frame: int
    last_frame_idx: int
    last_enu: Optional[np.ndarray] = None
    last_pose: Any = None
    final_sim2: Any = None
    final_sim3: Any = None
    scale: Optional[float] = None
    residual: Optional[float] = None
    inlier_ratio: Optional[float] = None
    anchor_count: int = 0
    quality_score: Optional[float] = None
    trajectory_global: Optional[np.ndarray] = None
    trajectory_local: Optional[np.ndarray] = None
    boundary_start_enu: Optional[np.ndarray] = None
    boundary_end_enu: Optional[np.ndarray] = None
    boundary_start_pose: Any = None
    boundary_end_pose: Any = None
    can_propagate: bool = True
    fallback_reason: Optional[str] = None
    overlap_len: int = 0
    transform: Optional[Any] = None  # T_k: LocalModelTransform (model space → ENU) for this segment
    post_ttt_overlap_m0: Optional[np.ndarray] = None  # post-TTT positions of last overlap_len frames in M_0 space (for next segment's T_k estimation)
    # Plan A (PGO): per-segment DOM anchor observations for overlap frames only.
    # Dict key: GLOBAL frame index (start_frame + local_idx) so PGO can match
    # the same overlap frame across neighboring segments.
    # Value: np.ndarray shape (3,) -- ENU XYZ observed via DOM + DEM.
    overlap_dom_targets_enu: Optional[Dict[int, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prev_summary(self) -> Dict[str, Any]:
        """Compact dict for the next segment's propagation decision."""
        return {
            "segment_id": self.segment_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "last_frame_idx": self.last_frame_idx,
            "last_enu": self.last_enu,
            "last_pose": self.last_pose,
            "final_sim2": self.final_sim2,
            "final_sim3": self.final_sim3,
            "scale": self.scale,
            "residual": self.residual,
            "inlier_ratio": self.inlier_ratio,
            "anchor_count": self.anchor_count,
            "quality_score": self.quality_score,
            "boundary_start_enu": self.boundary_start_enu,
            "boundary_end_enu": self.boundary_end_enu,
            "boundary_start_pose": self.boundary_start_pose,
            "boundary_end_pose": self.boundary_end_pose,
            "can_propagate": self.can_propagate,
            "fallback_reason": self.fallback_reason,
            "overlap_len": self.overlap_len,
            "trajectory_global": self.trajectory_global,
            "trajectory_local": self.trajectory_local,   # model-space trajectory for overlap chaining
            "transform": self.transform,                   # T_k for converting model → ENU on demand
            "post_ttt_overlap_m0": self.post_ttt_overlap_m0,  # post-TTT overlap positions in M_0 for next segment's T_k
            "overlap_dom_targets_enu": self.overlap_dom_targets_enu,
            "per_frame_R_c2w_model": self.metadata.get("per_frame_R_c2w_model") if isinstance(self.metadata, dict) else None,
            "transform_R": self.metadata.get("transform_R") if isinstance(self.metadata, dict) else None,
            "metadata": self.metadata,
        }


@dataclass
class GlobalChainState:
    """Global-only chaining state for output assembly."""

    summaries: List[SegmentSummary] = field(default_factory=list)
    # Stores model-space (local) trajectories; ENU is computed at the end via each segment's transform
    trajectories: List[np.ndarray] = field(default_factory=list)
    transforms: List[Any] = field(default_factory=list)  # T_k per segment; parallel to trajectories
    global_origin: Optional[np.ndarray] = None
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    # Counter: how many segments triggered the DOM-anchor fallback
    fallback_count: int = 0
