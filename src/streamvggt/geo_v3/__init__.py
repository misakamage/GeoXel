"""GeoTTT-v3：基于分段（segment）的局部 model-space 子图（submap）框架。

本包实现 GeoTTT-v3 架构，核心思想是把长序列切成若干带重叠的分段，
每段在自己的局部坐标系（model space M_k）里独立做测试时训练（TTT），
再通过重叠帧把相邻分段串接起来：

- hard reset：在每个分段边界把运行时状态清零，分段之间不复用优化状态；
- geo canonicalization：把 ENU 世界坐标规范化到该段的局部 model space M_k；
- model-space TTT：在每个分段的局部坐标系内做测试时训练；
- overlap-based propagation：利用相邻分段的重叠帧做跨段传播 / 位姿对齐。

术语速查：
- ENU            —— East-North-Up 世界坐标系（地理参考系）。
- M_k            —— 第 k 段的局部 model space；T_k 把 M_k 映射回 ENU。
- Sim3 / Sim2    —— 带尺度的刚体变换（3D / 2D），(s, R, t)。
- TTT            —— Test-Time Training，测试时在线适配。
- PGO            —— Pose Graph Optimization，位姿图优化（见 pgo.py）。
- DOM / DEM      —— 正射影像 / 数字高程模型，用于提供地理监督。
"""

# 配置对象：所有分段适配与优化的超参集中在此。
from .config import GeoV3Config
# 状态对象：分段运行时状态、分段摘要、全局链路状态。
from .state import SegmentRuntimeState, SegmentSummary, GlobalChainState
# 分段切分：把整段序列切成带重叠的 SegmentRange 列表。
from .segment import SegmentRange, segment_sequence
# 段输入 / 输出的标准数据结构。
from .io import SegmentInput, SegmentOutput
# 局部 model space 变换 T_k 及其构造 / 规范化工具。
from .canonicalization import LocalModelTransform, build_local_model_transform, canonicalize_segment_targets
# 默认的单段推理入口：geo_v3 的核心实现都在 defaults.py。
from .defaults import default_segment_infer_fn
# 分段调度主循环：串起切分、hard reset、单段推理、输出拼装。
from .runner import GeoV3Runner
# 位姿图优化：段级 + 全局，后处理阶段精修各段 T_k。
from .pgo import (
    PGOConfig,
    refine_transforms_with_pgo,
    assemble_trajectory,
    SegmentPGOConfig,
    refine_segment_pgo,
)
from .vertical_scale import (
    apply_vertical_scale_propagation,
    build_vertical_scale_evidence,
    estimate_vertical_affines,
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
    "default_segment_infer_fn",
    "GeoV3Runner",
    "PGOConfig",
    "refine_transforms_with_pgo",
    "assemble_trajectory",
    "SegmentPGOConfig",
    "refine_segment_pgo",
    "apply_vertical_scale_propagation",
    "build_vertical_scale_evidence",
    "estimate_vertical_affines",
]
