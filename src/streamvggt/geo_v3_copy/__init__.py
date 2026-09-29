"""GeoTTT-v3: segment-based local model-space submaps.

This package implements the GeoTTT-v3 architecture:
- hard reset at segment boundaries
- geo canonicalization into local model space M_k
- model-space TTT within each segment
- overlap-based propagation across segments
"""

from .config import GeoV3Config
from .state import SegmentRuntimeState, SegmentSummary, GlobalChainState
from .segment import SegmentRange, segment_sequence
from .io import SegmentInput, SegmentOutput
from .canonicalization import LocalModelTransform, build_local_model_transform, canonicalize_segment_targets
from .overlap import can_use_overlap_propagation, build_overlap_transform_from_previous_segment
from .defaults import default_segment_infer_fn
from .runner import GeoV3Runner
from .pgo import (
    PGOConfig,
    refine_transforms_with_pgo,
    assemble_trajectory,
    SegmentPGOConfig,
    refine_segment_pgo,
)

__all__ = [
    "GeoV3Config",
    "SegmentRuntimeState",
    "SegmentSummary",
    "GlobalChainState",
    "SegmentRange",
    "segment_sequence",
    "SegmentInput",
    "SegmentOutput",
    "LocalModelTransform",
    "build_local_model_transform",
    "canonicalize_segment_targets",
    "can_use_overlap_propagation",
    "build_overlap_transform_from_previous_segment",
    "default_segment_infer_fn",
    "GeoV3Runner",
    "PGOConfig",
    "refine_transforms_with_pgo",
    "assemble_trajectory",
    "SegmentPGOConfig",
    "refine_segment_pgo",
]
