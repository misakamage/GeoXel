from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import torch


@dataclass
class SegmentInput:
    """单个 GeoTTT-v3 分段的标准输入。

    这个对象把一段推理所需的全部信息聚合在一处，按来源分为四类：
    - 局部分段帧本身（frames 及其起止下标）；
    - 该段的 DOM / geo 地理输入（正射影像、投影函数、高程等）；
    - 重叠传播状态（来自上一段的摘要 / 轨迹 / 变换）；
    - 可选的、由调用方提供的损失上下文（query_points、heading_fn 等）。
    """

    segment_id: int                     # 段编号（从 0 开始）
    start_frame: int                    # 该段在整段序列中的起始全局帧下标（含）
    end_frame: int                      # 该段的结束全局帧下标（不含）
    frames: List[dict]                  # 该段的逐帧数据（已按 [start, end) 切好）
    dom_image: Optional[torch.Tensor] = None        # 该段使用的 DOM 正射影像
    anchor_world_xyz: Optional[Any] = None          # 该段锚点的 ENU 世界坐标
    project_fn: Optional[Any] = None                # ENU -> 像素 的投影函数
    get_world_xyz_fn: Optional[Any] = None          # 由相机位姿取世界坐标的函数
    heading_fn: Optional[Any] = None                # 朝向 / heading 监督函数
    query_points: Optional[torch.Tensor] = None     # 用于损失的查询像素点
    image_paths: Optional[List[str]] = None         # 各帧图像路径（用于可视化 / 缓存）
    inv_project_fn: Optional[Any] = None            # 像素 -> ENU 的反投影函数
    geo_elev: Optional[Any] = None                  # 地理高程（DEM）查询入口
    dom_transform: Optional[Any] = None             # DOM 像素 <-> ENU 的仿射变换
    prev_summary: Optional[Dict[str, Any]] = None   # 上一段的摘要（用于传播决策）
    prev_submap_state: Optional[Any] = None         # 上一段的子图状态
    # 向后兼容字段：旧的全局轨迹；新代码优先用 prev_trajectory_local + prev_transform。
    prev_trajectory_global: Optional[np.ndarray] = None
    prev_trajectory_local: Optional[np.ndarray] = None   # 上一段的 model-space 轨迹
    prev_transform: Optional[Any] = None            # T_{k-1}：上一段的 LocalModelTransform
    # 仅 segment-0 使用：首帧相机 R_c2w（ENU 下的 3×3 旋转），由首帧外参推得，
    # 用来让 T_0 采用真实旋转而非单位阵。
    gt_rotation: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)   # 其它附带上下文（如 save_vis_dir）

    @property
    def length(self) -> int:
        """该段帧数 = end_frame - start_frame。"""
        return self.end_frame - self.start_frame


@dataclass
class SegmentOutput:
    """向后兼容的单段输出包。

    ``trajectory_local`` 是 model-space 轨迹（链路的权威数据），
    ``trajectory_global`` 是其 ENU 形式（仅用于边界报告 / 兼容旧接口）。
    """

    trajectory_global: np.ndarray       # ENU 世界坐标轨迹（边界报告用）
    trajectory_local: np.ndarray        # model-space 轨迹（跨段链接的权威数据）
    diagnostics: Dict[str, Any] = field(default_factory=dict)   # 诊断信息字典
    transform: Any = None               # T_k：该段 model space -> ENU 的变换
    submap_state: Any = None            # 该段的子图状态
    geo_cache: Any = None               # 该段的 DOM/geo 观测缓存
