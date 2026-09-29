from typing import Optional
import numpy as np

from .state import SegmentRuntimeState


def hard_reset_segment_state(
    state: SegmentRuntimeState,
    anchor_enu: np.ndarray,
    anchor_frame_idx: int = 0,
) -> SegmentRuntimeState:
    """Reset a segment runtime state to a fresh local model-space anchor."""
    state.anchor_frame_idx = anchor_frame_idx
    state.enu_offset = np.array(anchor_enu, copy=True)
    state.past_kv = None
    state.past_kv_cam = None
    state.sim2 = None
    state.sim3 = None
    state.dom_crop_center = None
    state.optimizer_state.clear()
    state.correspondence_cache.clear()
    state.loss_history.clear()
    state.ttt_outputs.clear()
    return state
