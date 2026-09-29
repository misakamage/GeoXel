from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SubmapState:
    """单个分段的显式局部子图（submap）状态。

    保存该段在自己坐标系下的点云 / 变换等，供调试与跨段衔接参考。
    """

    segment_id: int                              # 段编号
    model_space: Any = None                      # 该段 model space 下的点云 / 结构
    transform: Any = None                        # T_k：model space -> ENU 的变换
    anchor_world_xyz: Optional[np.ndarray] = None    # 该段锚点的 ENU 坐标
    source_summary: Optional[Dict[str, Any]] = None  # 生成本子图所依据的上一段摘要


@dataclass
class GeoObservationCache:
    """段内 DOM / geo 观测的显式缓存。

    按段内帧下标聚合各类地理观测，供损失计算 / PGO 复用，避免重复求解。
    """

    correspondences: Dict[int, Any] = field(default_factory=dict)       # 帧 -> DOM 对应关系
    frame_pixels: Dict[int, np.ndarray] = field(default_factory=dict)   # 帧 -> 匹配像素坐标
    enu_positions: Dict[int, np.ndarray] = field(default_factory=dict)  # 帧 -> 对应 ENU 位置
    frame_centers_enu: Dict[int, np.ndarray] = field(default_factory=dict)  # 帧 -> 画面中心 ENU
    pnp_results: Dict[int, Any] = field(default_factory=dict)           # 帧 -> PnP 解算结果
    metadata: Dict[str, Any] = field(default_factory=dict)             # 其它元信息


@dataclass
class SegmentTTTResult:
    """单段一次 TTT（测试时训练）过程的结构化结果。"""

    trajectory_global: np.ndarray                # ENU 轨迹
    trajectory_local: np.ndarray                 # model-space 轨迹
    transform: Any = None                        # 该段 T_k
    submap_state: Optional[SubmapState] = None   # 该段子图状态
    geo_cache: Optional[GeoObservationCache] = None  # 该段地理观测缓存
    diagnostics: Dict[str, Any] = field(default_factory=dict)  # 诊断信息


@dataclass
class SegmentRuntimeState:
    """分段运行时的可变状态——**不允许跨分段边界传递**。

    每进入新的一段都会被 hard_reset_segment_state 清零（见 reset.py），
    确保各段的优化 / 缓存互不污染。
    """

    segment_id: int          # 段编号
    start_frame: int         # 段起始全局帧下标（含）
    end_frame: int           # 段结束全局帧下标（不含）

    # --- hard-reset 状态：每段开始时清零 ---
    anchor_frame_idx: int = 0                     # 锚点对应的段内局部帧下标
    enu_offset: Optional[np.ndarray] = None       # 锚点 ENU 偏移
    past_kv: Any = None                           # 点 / 聚合分支的 KV cache
    past_kv_cam: Any = None                       # 相机分支的 KV cache
    sim2: Any = None                              # 当前段的 Sim2（2D 带尺度变换）
    sim3: Any = None                              # 当前段的 Sim3（3D 带尺度变换）
    dom_crop_center: Optional[np.ndarray] = None  # DOM 裁剪中心

    # --- 段内优化 / 缓存：同样每段清零 ---
    optimizer_state: Dict[str, Any] = field(default_factory=dict)       # 优化器状态
    correspondence_cache: Dict[int, Any] = field(default_factory=dict)  # 对应关系缓存
    loss_history: List[float] = field(default_factory=list)             # loss 历史
    ttt_outputs: Dict[str, Any] = field(default_factory=dict)           # TTT 输出暂存


@dataclass
class SegmentSummary:
    """传给下一段的**只读**摘要：相邻分段衔接的关键载体。

    包含本段的最终位姿 / 变换 / 轨迹、重叠帧信息、质量指标，以及供 PGO 使用的
    重叠帧 DOM 锚点观测。下一段据此决定如何传播 / 对齐。
    """

    segment_id: int                              # 段编号
    start_frame: int                             # 段起始全局帧下标
    end_frame: int                               # 段结束全局帧下标
    last_frame_idx: int                          # 本段最后一帧的全局下标
    last_enu: Optional[np.ndarray] = None        # 最后一帧的 ENU 位置
    last_pose: Any = None                        # 最后一帧的位姿
    final_sim2: Any = None                       # 本段最终 Sim2
    final_sim3: Any = None                       # 本段最终 Sim3
    scale: Optional[float] = None                # 本段尺度估计
    residual: Optional[float] = None             # 拟合残差
    inlier_ratio: Optional[float] = None         # 内点比例
    anchor_count: int = 0                        # 锚点数量
    quality_score: Optional[float] = None        # 综合质量评分
    trajectory_global: Optional[np.ndarray] = None   # ENU 轨迹
    trajectory_local: Optional[np.ndarray] = None    # model-space 轨迹
    boundary_start_enu: Optional[np.ndarray] = None  # 段首边界 ENU
    boundary_end_enu: Optional[np.ndarray] = None    # 段尾边界 ENU
    boundary_start_pose: Any = None              # 段首边界位姿
    boundary_end_pose: Any = None                # 段尾边界位姿
    can_propagate: bool = True                   # 是否可向下一段传播
    fallback_reason: Optional[str] = None        # 若不可传播，记录回退原因
    overlap_len: int = 0                         # 与下一段的重叠帧数
    transform: Optional[Any] = None              # T_k：本段 model space -> ENU 的变换
    # 本段最后 overlap_len 帧在 M_0 空间中的 post-TTT 位置；
    # 供下一段估计自己的 T_k 时做参照。
    post_ttt_overlap_m0: Optional[np.ndarray] = None
    # Plan A（PGO）：仅针对重叠帧的逐帧 DOM 锚点观测。
    # key 为**全局帧下标**（start_frame + 段内下标），使 PGO 能在相邻段间
    # 匹配同一重叠帧；value 为形如 (3,) 的 ENU XYZ（由 DOM + DEM 观测得到）。
    overlap_dom_targets_enu: Optional[Dict[int, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)   # 其它附带信息

    def to_prev_summary(self) -> Dict[str, Any]:
        """打包成下一段做传播决策所需的紧凑字典（``prev_summary``）。"""
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
            "trajectory_local": self.trajectory_local,   # 供重叠链接用的 model-space 轨迹
            "transform": self.transform,                   # T_k：按需把 model -> ENU
            "post_ttt_overlap_m0": self.post_ttt_overlap_m0,  # M_0 下的 post-TTT 重叠位置，供下一段估 T_k
            "overlap_dom_targets_enu": self.overlap_dom_targets_enu,
            "per_frame_R_c2w_model": self.metadata.get("per_frame_R_c2w_model") if isinstance(self.metadata, dict) else None,
            "transform_R": self.metadata.get("transform_R") if isinstance(self.metadata, dict) else None,
            "metadata": self.metadata,
        }


@dataclass
class GlobalChainState:
    """全局链路状态：仅用于最终输出拼装的"跨段链接"信息。

    注意 ``trajectories`` 存的是**model-space（局部）轨迹**；最终 ENU 轨迹在
    末尾一次性通过各段的 transform 计算得到，过程中不在段间存 ENU。
    """

    summaries: List[SegmentSummary] = field(default_factory=list)   # 各段摘要
    # 各段的 model-space 轨迹；ENU 在最后用各段 transform 统一换算。
    trajectories: List[np.ndarray] = field(default_factory=list)
    transforms: List[Any] = field(default_factory=list)  # 各段 T_k，与 trajectories 一一对应
    global_origin: Optional[np.ndarray] = None           # 全局原点（通常取 GT 首帧 ENU）
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)  # 各段诊断信息
    # 计数器：有多少段触发了 DOM 锚点回退（fallback）。
    fallback_count: int = 0
