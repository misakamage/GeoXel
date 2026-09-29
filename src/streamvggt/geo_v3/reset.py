from typing import Optional
import numpy as np

from .state import SegmentRuntimeState


def hard_reset_segment_state(
    state: SegmentRuntimeState,
    anchor_enu: np.ndarray,
    anchor_frame_idx: int = 0,
) -> SegmentRuntimeState:
    """把分段运行时状态硬重置（hard reset）到一个全新的局部 model-space 锚点。

    GeoTTT-v3 要求分段之间不复用任何优化 / 缓存状态：每进入新的一段，都从
    干净的锚点重新开始。本函数就地修改并返回传入的 ``state``。

    参数：
        state:           待重置的分段运行时状态（会被就地修改）。
        anchor_enu:      该段锚点在 ENU 世界坐标下的位置，会被拷贝进 ``enu_offset``。
        anchor_frame_idx: 锚点对应的段内局部帧下标（默认 0，即段首帧）。

    返回：
        重置后的同一个 ``state`` 对象（便于链式调用）。
    """
    # 记录锚点帧下标，并以拷贝方式保存锚点 ENU 偏移（避免外部数组被改动牵连）。
    state.anchor_frame_idx = anchor_frame_idx
    state.enu_offset = np.array(anchor_enu, copy=True)
    # 清空所有跨帧 / 跨段的缓存张量：KV cache、Sim2/Sim3、DOM 裁剪中心。
    state.past_kv = None
    state.past_kv_cam = None
    state.sim2 = None
    state.sim3 = None
    state.dom_crop_center = None
    # 清空段内优化器状态、对应关系缓存、loss 历史与 TTT 输出，确保互不污染。
    state.optimizer_state.clear()
    state.correspondence_cache.clear()
    state.loss_history.clear()
    state.ttt_outputs.clear()
    return state
