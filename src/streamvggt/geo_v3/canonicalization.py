from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch


@dataclass
class LocalModelTransform:
    """分段局部 model-space 变换 T_k（model space M_k <-> ENU 世界系）。

    采用带尺度的 Sim3 表示，行向量约定下正变换为：``enu = s * model @ R.T + t``。
    刻意设计得很轻量：一旦为某段选定了变换，后续计算都留在 M_k 内进行。

    字段：
        anchor_world_xyz: 该段锚点在 ENU 下的坐标（即 model 原点映射到 ENU 的位置）。
        R: 3×3 旋转矩阵；s: 标量尺度；t: 3 维平移（ENU）。
    """

    anchor_world_xyz: torch.Tensor
    R: torch.Tensor
    s: torch.Tensor
    t: torch.Tensor

    def world_to_model(self, world_xyz: torch.Tensor) -> torch.Tensor:
        """ENU 世界坐标 -> 局部 model 坐标（T_k 的逆变换）。"""
        # 正变换 enu = s * R @ model + t  =>  逆变换 model = R^T @ (enu - t) / s
        # 行向量形式：model = (enu - t) @ R / s
        return ((world_xyz - self.t[None, :]) @ self.R) / self.s

    def model_to_world(self, model_xyz: torch.Tensor) -> torch.Tensor:
        """局部 model 坐标 -> ENU 世界坐标（T_k 的正变换）。"""
        # enu = s * R @ model + t（行向量形式：s * model @ R.T + t）
        return self.s * (model_xyz @ self.R.T) + self.t[None, :]


def build_local_model_transform(
    anchor_world_xyz: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> LocalModelTransform:
    """构造一个"仅平移"的初始 T_k：R=单位阵、s=1、t=锚点 ENU。

    用于段初始化——此时 model 原点正好落在锚点 ENU 上，朝向 / 尺度待后续估计。
    若传入的锚点是多帧数组，则取第一帧作为锚点。
    """
    anchor = torch.as_tensor(anchor_world_xyz, dtype=dtype or torch.float32, device=device)
    if anchor.ndim > 1:
        anchor = anchor[0]
    R = torch.eye(3, device=anchor.device, dtype=anchor.dtype)
    s = torch.tensor(1.0, device=anchor.device, dtype=anchor.dtype)
    t = anchor.clone()
    return LocalModelTransform(anchor_world_xyz=anchor, R=R, s=s, t=t)


def compose_transforms(T_outer: "LocalModelTransform", T_inner: "LocalModelTransform") -> "LocalModelTransform":
    """复合两个 Sim3 变换 T_outer ∘ T_inner（先内后外）。

    若 T_inner 把 A->B、T_outer 把 B->C，则结果把 A->C。
    行向量约定：``world = s * model @ R.T + t``。

    推导出的复合公式：
        s_c = s_out * s_in
        R_c = R_out @ R_in
        t_c = s_out * (t_in @ R_out.T) + t_out
    结果的 anchor_world_xyz 取 t_c（复合变换下 model 原点映射到 ENU 的位置）。
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
    """把 ENU 世界坐标规范化到 model space（``transform.world_to_model`` 的薄封装）。"""
    return transform.world_to_model(world_xyz)


def canonicalize_segment_targets(
    world_xyz: torch.Tensor,
    transform: LocalModelTransform,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """把一段的 ENU 目标点规范化到 model space，并返回所用变换的诊断信息。

    返回：
        (model_xyz, diagnostics)：model_xyz 为规范化后的 model 坐标；
        diagnostics 记录本次使用的 R / s / t（已转为 numpy / float，便于落盘）。
    """
    model_xyz = canonicalize_world_xyz_to_model(world_xyz, transform)
    diagnostics = {
        "transform_R": transform.R.detach().cpu().numpy(),
        "transform_s": float(transform.s.detach().cpu()),
        "transform_t": transform.t.detach().cpu().numpy(),
    }
    return model_xyz, diagnostics
