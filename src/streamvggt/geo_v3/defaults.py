"""GeoTTT-v3 单段推理的默认实现（geo_v3 的核心模块）。

本文件是整个 geo_v3 中体量最大、逻辑最集中的模块，对外的唯一主入口是
``default_segment_infer_fn``（被 runner.GeoV3Runner.run 逐段调用）。其余几十个
``_`` 开头的函数都是它的内部步骤，可大致归为以下几组（按职责，不按文件顺序）：

1. Sim2 / Sim3 估计与几何工具
   ``_estimate_sim2_2d* / _estimate_sim3_3d / _fit_ransac_srt_point_edges /
   _rotation_delta_deg`` 等：在 2D/3D 上拟合带尺度的相似变换、做 RANSAC、算角度差。

2. 重叠传播与 transform 选取
   ``_build_overlap_geo_prior_transform / _estimate_transform_from_geo_obs /
   _dom_anchor_fallback_transform / _build_transform_for_segment``：决定该段最终采用的
   model->ENU 变换 T_k，并在重叠先验、DOM 重拟合、回退之间做门控（gate）。

3. DOM / DEM 点级监督（dom_points 路径）
   ``_select_dom_point_correspondences* / _estimate_dom_point_camera_target* /
   _build_dom_point_camera_targets_from_geo_obs / _build_dom_point_ttt_teacher /
   _collect_dom_point_bootstrap_samples / _estimate_dom_point_bootstrap_transform``：
   从匹配像素采样 DOM/DEM 点，构造点对应、相机目标、TTT teacher 与 bootstrap 先验。

4. 相机目标 / 地面射线语义
   ``_geo_camera_target_enu / _frame_center_camera_target_enu /
   _pose_ground_camera_target_enu / _build_dom_point_ground_ray_cache /
   _ground_ray_intersection_xy_torch``：把 DOM 观测转成相机 XY/Z 目标或地面射线约束。

5. 模型前向与 token 适配（TTT）
   ``_frozen_forward_with_tokens_geo_v3 / _forward_pose_cached / _forward_pose_and_pts /
   _batched_forward_* / _prepare_token_leaves / _segment_model_space_ttt``：
   在冻结骨干上做带缓存的前向，并在 token 空间做测试时训练。

6. 观测收集与目标规范化
   ``_collect_geo_observations / _build_model_target_cache / _canonicalize_segment_targets /
   _build_overlap_dom_targets_enu``：把一段的所有地理观测整理成可供 TTT/PGO 使用的目标。

约定与术语见包级 __init__.py 的说明：ENU=地理参考系，M_k=第 k 段局部 model space，
T_k：M_k->ENU 的 Sim3，TTT=测试时训练，PGO=位姿图优化（见 pgo.py）。

注意：本模块内大量函数带有针对真实数据集（airzoo / Japan 等）观测到的现象写下的
调参注释与门控阈值；标注\"已弃用 / 忽略\"的分支为兼容保留。骨架注释只解释\"每个函数
做什么、输入输出是什么\"，不逐行展开数值细节。
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import copy
import os
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F

from .io import SegmentInput, SegmentOutput
from .state import SegmentRuntimeState, SubmapState, GeoObservationCache
from .config import GeoV3Config
from .canonicalization import (
    build_local_model_transform,
    compose_transforms,
    LocalModelTransform,
)
from streamvggt.utils.dom_state import DOMStateManager, DOMState, DOMStateConfig
from streamvggt.utils.lora import (
    LoRALinear,
    get_lora_diagnostics,
    inject_lora,
    remove_lora,
)
from streamvggt.utils.trajectory_chain import pose_enc_to_camera_centers


def _clone_geo_corr_for_overlap_reuse(corr: Any) -> Any:
    """为另一段的同一个全局帧克隆一个 DOM/PnP 地理观测。

    地理观测本身在重叠段之间是共享的，但克隆时会剔除与具体段相关的坐标
    （如 frame_center_m0）以及所有 GT 诊断字段（camera_z_gt、gt_agl、PnP
    diagnostics 里的 gt_*_err_m），避免把上一段的局部信息泄漏到当前段。
    返回深拷贝后的观测。
    """
    cloned = copy.deepcopy(corr)
    if isinstance(cloned, dict):
        for key in ("frame_center_m0", "camera_z_gt", "gt_agl"):
            cloned.pop(key, None)
        pnp = cloned.get("pnp_result")
        if isinstance(pnp, dict):
            diag = pnp.get("diagnostics")
            if isinstance(diag, dict):
                diag.pop("gt_xy_err_m", None)
                diag.pop("gt_z_err_m", None)
    return cloned


def _pose_to_world_xyz(
    pose: torch.Tensor,
    get_world_xyz_fn: Any,
) -> torch.Tensor:
    """从相机 pose 提取世界坐标系下的相机中心 xyz。

    若提供了 get_world_xyz_fn 则直接调用它；否则按启发式从 pose 中解析：
    pose 编码（最后一维 > 3）走 pose_enc_to_camera_centers，否则取前 3 维。
    返回 float tensor。当无法解析且缺少 get_world_xyz_fn 时抛 ValueError。
    """
    if get_world_xyz_fn is None:
        if pose.ndim >= 1 and pose.shape[-1] > 3:
            return pose_enc_to_camera_centers(pose).float()
        if pose.ndim == 1 and pose.numel() >= 3:
            return pose[:3].float()
        if pose.ndim >= 2 and pose.shape[-1] >= 3:
            return pose[..., :3].float()
        raise ValueError("camera pose does not provide world xyz and get_world_xyz_fn is missing")
    xyz = get_world_xyz_fn(pose)
    if not torch.is_tensor(xyz):
        xyz = torch.as_tensor(xyz, dtype=torch.float32, device=pose.device)
    return xyz.float()


def _get_segment_anchor_enu(segment_input: SegmentInput) -> Optional[np.ndarray]:
    """返回本段最可靠的 ENU 锚点（起始位置）。

    优先级：prev_summary.boundary_end_enu（上一段由 DOM 估出的段末位置）
            > anchor_world_xyz（首帧 GT，仅对 segment 0 有效）。
    坑：对 k > 0 绝不能用 anchor_world_xyz（首帧 GT）当作本段起点，否则会差出
    数公里。返回长度为 3 的 ENU 坐标（不足 3 维时补 0），都没有时返回 None。
    """
    prev_summary = segment_input.prev_summary
    if prev_summary is not None:
        bene = prev_summary.get("boundary_end_enu")
        if bene is not None:
            a = np.array(bene, dtype=np.float32).flatten()
            if len(a) >= 2:
                return a[:3] if len(a) >= 3 else np.append(a, 0.0)
    # Only safe for segment 0: anchor_world_xyz == first frame GT
    awxyz = segment_input.anchor_world_xyz
    if awxyz is not None:
        if torch.is_tensor(awxyz):
            awxyz = awxyz.detach().cpu().numpy()
        a = np.array(awxyz, dtype=np.float32).flatten()
        if len(a) >= 2:
            return a[:3] if len(a) >= 3 else np.append(a, 0.0)
    return None


def _estimate_sim2_2d(
    model_xy: np.ndarray,
    enu_xy: np.ndarray,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """2D Umeyama 算法：求 (s, R_2x2, t_2d) 使得 enu_xy ≈ s * R_2x2 @ model_xy + t_2d。

    输入为配对的 model 平面坐标与 ENU 平面坐标（各 n×2）。返回 (尺度 s,
    2×2 旋转 R2, 2D 平移 t2)。退化时返回 None（n < 2、方差为 0、或尺度过小）。
    """
    n = len(model_xy)
    if n < 2:
        return None
    mu_m = model_xy.mean(0)
    mu_e = enu_xy.mean(0)
    mc = model_xy - mu_m
    ec = enu_xy - mu_e
    var_m = float((mc ** 2).sum()) / n
    if var_m < 1e-12:
        return None
    H = mc.T @ ec / n           # (2, 2)
    U, sv, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if abs(d) < 0.5:
        d = 1.0
    D = np.diag([1.0, d])
    R2 = (Vt.T @ D @ U.T).astype(np.float32)
    s = float((sv * np.array([1.0, d])).sum() / var_m)
    if s < 1e-6:
        return None
    t2 = (mu_e - s * R2 @ mu_m).astype(np.float32)
    return s, R2, t2


def _estimate_sim2_2d_robust(
    model_xy: np.ndarray,
    enu_xy: np.ndarray,
    *,
    enable: bool = True,
    min_inliers: int = 2,
) -> Tuple[Optional[Tuple[float, np.ndarray, np.ndarray]], Dict[str, Any]]:
    """带 RANSAC 的鲁棒 2D Sim2 估计，对外点更稳健。

    在所有点对组合上拟合 _estimate_sim2_2d，用 MAD 阈值（med + 3*MAD）统计内点，
    取内点最多者，再在内点上重拟合。点对过少（n < 4）或 enable=False 时退回
    直接拟合。返回 ((s, R2, t2) 或 None, 诊断 dict)，诊断含模式、内点数、残差等。
    """
    X = np.asarray(model_xy, dtype=np.float64)
    Y = np.asarray(enu_xy, dtype=np.float64)
    diag: Dict[str, Any] = {"robust": bool(enable), "n": int(len(X))}
    finite = (
        X.ndim == 2 and Y.ndim == 2 and X.shape == Y.shape and X.shape[1] == 2
    )
    if not finite:
        diag["reason"] = "bad_shape"
        return None, diag
    mask_finite = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[mask_finite]
    Y = Y[mask_finite]
    n = int(len(X))
    diag["n_finite"] = n
    if n < 2:
        diag["reason"] = "too_few_pairs"
        return None, diag
    if not enable or n < 4:
        result = _estimate_sim2_2d(X.astype(np.float32), Y.astype(np.float32))
        diag.update({"mode": "direct", "n_inliers": n})
        return result, diag

    best: Optional[Tuple[int, float, np.ndarray, Tuple[float, np.ndarray, np.ndarray]]] = None
    for i in range(n - 1):
        for j in range(i + 1, n):
            result_ij = _estimate_sim2_2d(
                X[[i, j]].astype(np.float32), Y[[i, j]].astype(np.float32))
            if result_ij is None:
                continue
            s_ij, R2_ij, t2_ij = result_ij
            pred = float(s_ij) * (X @ R2_ij.T) + t2_ij[None, :]
            err = np.linalg.norm(pred - Y, axis=1)
            med = float(np.median(err))
            mad = float(1.4826 * np.median(np.abs(err - med)))
            cutoff = med + max(3.0 * mad, 1e-6)
            inlier_mask = err <= cutoff
            n_inl = int(inlier_mask.sum())
            score = (n_inl, -med)
            if best is None or score > (best[0], -best[1]):
                best = (n_inl, med, inlier_mask, result_ij)

    if best is None:
        result = _estimate_sim2_2d(X.astype(np.float32), Y.astype(np.float32))
        diag.update({"mode": "direct_fallback", "n_inliers": n, "reason": "ransac_failed"})
        return result, diag

    n_inl, med_best, inlier_mask, _ = best
    if n_inl < max(2, int(min_inliers)):
        keep = np.argsort(np.linalg.norm(X - X.mean(axis=0), axis=1))[:max(2, min(n, int(min_inliers)))]
        inlier_mask = np.zeros(n, dtype=bool)
        inlier_mask[keep] = True
        n_inl = int(inlier_mask.sum())
    result = _estimate_sim2_2d(
        X[inlier_mask].astype(np.float32), Y[inlier_mask].astype(np.float32))
    if result is None:
        result = _estimate_sim2_2d(X.astype(np.float32), Y.astype(np.float32))
        diag.update({"mode": "direct_fallback", "n_inliers": n, "reason": "refit_failed"})
        return result, diag

    s, R2, t2 = result
    err_final = np.linalg.norm(float(s) * (X @ R2.T) + t2[None, :] - Y, axis=1)
    diag.update({
        "mode": "pair_ransac_mad",
        "n_inliers": int(n_inl),
        "median_residual": float(np.median(err_final)),
        "max_residual": float(np.max(err_final)),
        "initial_best_median": float(med_best),
    })
    return result, diag


def _estimate_sim2_fixed_scale_2d(
    model_xy: np.ndarray,
    enu_xy: np.ndarray,
    scale: float,
    R2_hint: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """在尺度 scale 固定的前提下，估计 2D 旋转 R2 与平移 t2。

    旋转通过去中心后的 SVD 求得；点对不足或退化时退回 R2_hint，再不行用单位阵。
    平移取 enu - scale*R2@model 的均值。返回 (R2(2x2), t2(2d), 中位残差 m)，
    输入形状非法或 scale 非正时返回 None。
    """
    X = np.asarray(model_xy, dtype=np.float64)
    Y = np.asarray(enu_xy, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape or X.shape[1] != 2:
        return None
    finite = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[finite]
    Y = Y[finite]
    n = int(X.shape[0])
    if n < 1 or not np.isfinite(scale) or scale <= 1e-8:
        return None
    R2: Optional[np.ndarray] = None
    if n >= 2:
        Xc = X - X.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        if float((Xc ** 2).sum()) > 1e-12 and float((Yc ** 2).sum()) > 1e-12:
            try:
                H = Xc.T @ Yc / max(n, 1)
                U, _, Vt = np.linalg.svd(H)
                d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
                if abs(d) < 0.5:
                    d = 1.0
                D = np.diag([1.0, d])
                R2 = (Vt.T @ D @ U.T).astype(np.float32)
            except Exception:
                R2 = None
    if R2 is None and R2_hint is not None:
        R2_h = np.asarray(R2_hint, dtype=np.float64)
        if R2_h.shape == (2, 2) and np.isfinite(R2_h).all():
            R2 = R2_h.astype(np.float32)
    if R2 is None:
        R2 = np.eye(2, dtype=np.float32)
    pred = float(scale) * (X @ R2.T)
    t2 = (Y - pred).mean(axis=0).astype(np.float32)
    residual = float(np.median(np.linalg.norm(pred + t2[None, :] - Y, axis=1)))
    return R2.astype(np.float32), t2, residual


def _build_overlap_geo_prior_transform(
    segment_input: SegmentInput,
    per_frame_pose: Optional[List[torch.Tensor]],
    overlap_len: int,
    device: torch.device,
    dtype: torch.dtype,
    config: Optional[Any] = None,
) -> Tuple[Optional[LocalModelTransform], Dict[str, Any]]:
    """由重叠帧几何为当前段构造 model->ENU 的先验变换 T_k（Sim3）。

    取当前段重叠帧的相机中心与上一段轨迹（trajectory_global）末尾 n 帧 ENU 位置
    配对，用鲁棒 Sim2 联合估计尺度 s_k、平面旋转与平移；s_k 属于当前段 model
    space，因硬重置而与 prev_transform.s 独立。dom_points 模式下改用由上一段外参
    传递得到的完整 3D 旋转，并在该旋转下重估均匀尺度。Z 平移由 ENU 高程中位数推出。
    至少需要 3 对重叠点。返回 (LocalModelTransform 或 None, 诊断 dict)。
    """
    diag: Dict[str, Any] = {"enabled": False, "accepted": False, "reason": "disabled"}
    if int(segment_input.segment_id) <= 0 or overlap_len <= 0:
        return None, diag
    if not per_frame_pose:
        diag.update({"enabled": True, "reason": "no_pose"})
        return None, diag
    prev_summary = segment_input.prev_summary or {}
    prev_global = prev_summary.get("trajectory_global")
    if prev_global is None:
        diag.update({"enabled": True, "reason": "no_prev_enu"})
        return None, diag
    prev_enu = np.asarray(prev_global, dtype=np.float64)
    if prev_enu.ndim != 2 or prev_enu.shape[1] < 2:
        diag.update({"enabled": True, "reason": "bad_prev_enu"})
        return None, diag
    n = min(int(overlap_len), len(per_frame_pose), len(prev_enu))
    if n <= 0:
        diag.update({"enabled": True, "reason": "empty_overlap"})
        return None, diag
    target_enu = prev_enu[-n:, :3].copy()
    centers: List[np.ndarray] = []
    for oi in range(n):
        with torch.no_grad():
            center = pose_enc_to_camera_centers(per_frame_pose[oi])
            if center.ndim > 1:
                center = center[0]
            centers.append(center[:3].detach().cpu().float().numpy().astype(np.float64))
    model_xyz = np.stack(centers, axis=0)
    finite = np.isfinite(model_xyz[:, :2]).all(axis=1) & np.isfinite(target_enu[:, :2]).all(axis=1)
    finite_overlap_idx = np.arange(n, dtype=np.int32)[finite]
    if int(finite.sum()) < 1:
        diag.update({"enabled": True, "reason": "no_finite_pairs", "n_overlap": int(n)})
        return None, diag
    model_xyz = model_xyz[finite]
    target_enu = target_enu[finite]
    # Need at least 3 overlap pairs to estimate s + R + t (2-DoF rotation,
    # 1-DoF scale, 2-DoF translation = 5 params over 2*n measurements).
    _min_overlap = 3
    if len(model_xyz) < _min_overlap:
        diag.update({"enabled": True, "reason": f"too_few_overlap:{len(model_xyz)}<{_min_overlap}",
                     "n_overlap": int(len(model_xyz))})
        return None, diag
    # Estimate s_k, R, t jointly from overlap geometry (free Sim2).
    # s_k belongs to the *current* segment's model space, which is independent
    # of prev_transform.s due to hard resets.  prev_transform is only used for
    # the Z translation estimate below.
    free_result, _robust_diag = _estimate_sim2_2d_robust(
        model_xyz[:, :2], target_enu[:, :2], enable=True)
    if free_result is None:
        diag.update({"enabled": True, "reason": "fit_failed", "n_overlap": int(len(model_xyz))})
        return None, diag
    scale, R2, t2 = free_result
    residual = float(_robust_diag.get("median_residual", 0.0))
    R3 = np.eye(3, dtype=np.float32)
    R3[:2, :2] = R2
    rotation_source = "free_sim2"
    full_pose_diag: Dict[str, Any] = {}
    if _geo_target_mode_from_config(config) == "dom_points":
        R_full, full_pose_diag = _estimate_overlap_handoff_rotation_from_prev(
            segment_input=segment_input,
            per_frame_pose=per_frame_pose,
            overlap_indices=finite_overlap_idx.tolist(),
            overlap_len=int(n),
        )
        if R_full is not None:
            rotated = model_xyz @ R_full.T
            fixed_scale = _estimate_uniform_scale_after_rotation(rotated[:, :2], target_enu[:, :2])
            if fixed_scale is not None:
                scale = float(fixed_scale)
            pred_xy = float(scale) * rotated[:, :2]
            t2 = (target_enu[:, :2] - pred_xy).mean(axis=0).astype(np.float32)
            residual = float(np.median(np.linalg.norm(pred_xy + t2[None, :] - target_enu[:, :2], axis=1)))
            R3 = R_full.astype(np.float32)
            rotation_source = "overlap_full_extrinsic"
    if target_enu.shape[1] >= 3 and np.isfinite(target_enu[:, 2]).any():
        z_mask = np.isfinite(target_enu[:, 2]) & np.isfinite(model_xyz[:, 2])
        if z_mask.any():
            rotated_z = (model_xyz @ R3.T)[:, 2]
            t_z = float(np.median(target_enu[z_mask, 2] - scale * rotated_z[z_mask]))
        else:
            t_z = float(np.nanmean(target_enu[:, 2]))
    else:
        t_z = 0.0
    t3 = np.array([float(t2[0]), float(t2[1]), t_z], dtype=np.float32)
    anchor = torch.tensor(target_enu[0, :3].astype(np.float32), device=device, dtype=dtype)
    transform = LocalModelTransform(
        anchor_world_xyz=anchor,
        R=torch.tensor(R3, device=device, dtype=dtype),
        s=torch.tensor(float(scale), device=device, dtype=dtype),
        t=torch.tensor(t3, device=device, dtype=dtype),
    )
    angle_deg = float(np.degrees(np.arctan2(float(R3[1, 0]), float(R3[0, 0]))))
    diag.update({
        "enabled": True,
        "accepted": True,
        "reason": "ok",
        "n_overlap": int(len(model_xyz)),
        "scale": float(scale),
        "heading_deg": angle_deg,
        "median_xy_residual_m": float(residual),
        "rotation_source": str(rotation_source),
        "full_extrinsic_handoff": full_pose_diag,
    })
    print(
        f"[GeoV3][OverlapGeoPrior] seg {segment_input.segment_id}: "
        f"n={len(model_xyz)} s={scale:.4f} heading={angle_deg:.1f}deg "
        f"med_xy={residual:.2f}m ({rotation_source})"
    )
    return transform, diag


def _estimate_uniform_scale_after_rotation(
    rotated_xy: np.ndarray,
    target_xy: np.ndarray,
) -> Optional[float]:
    """在旋转已固定后，估计把 rotated_xy 对齐到 target_xy 的单一均匀尺度。

    用去中心后的最小二乘闭式解 s = <Xc, Yc> / <Xc, Xc>。点对少于 2、形状不符、
    或分母/结果非正时返回 None。
    """
    X = np.asarray(rotated_xy, dtype=np.float64)
    Y = np.asarray(target_xy, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape or X.shape[1] != 2:
        return None
    finite = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[finite]
    Y = Y[finite]
    if X.shape[0] < 2:
        return None
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    denom = float((Xc ** 2).sum())
    if denom <= 1e-12:
        return None
    scale = float((Xc * Yc).sum() / denom)
    if not np.isfinite(scale) or scale <= 1e-8:
        return None
    return scale


def _estimate_overlap_handoff_rotation_from_prev(
    segment_input: SegmentInput,
    per_frame_pose: Optional[List[torch.Tensor]],
    overlap_indices: List[int],
    overlap_len: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """由上一段的逐帧外参传递出当前段重叠处的完整 3D 交接旋转 R（model->ENU）。

    对每个重叠帧：取上一段该帧的 c2w model 旋转，左乘上一段 transform_R 得到
    c2w ENU 旋转，再右乘当前帧 c2w model 旋转的转置，得到 model_curr->ENU 的旋转
    估计；对所有重叠帧做旋转平均。返回 (平均旋转 R 或 None, 诊断 dict)，缺少上一段
    逐帧旋转或 transform_R 等信息时返回 None。
    """
    diag: Dict[str, Any] = {"accepted": False, "reason": "not_run", "n": 0}
    prev_summary = segment_input.prev_summary or {}
    prev_metadata = prev_summary.get("metadata") if isinstance(prev_summary, dict) else None
    if not isinstance(prev_metadata, dict):
        prev_metadata = {}
    prev_R_model_arr = prev_summary.get("per_frame_R_c2w_model")
    if prev_R_model_arr is None:
        prev_R_model_arr = prev_metadata.get("per_frame_R_c2w_model")
    if prev_R_model_arr is None:
        diag["reason"] = "no_prev_frame_rotations"
        return None, diag
    try:
        prev_R_model_arr = np.asarray(prev_R_model_arr, dtype=np.float64)
    except Exception:
        diag["reason"] = "bad_prev_frame_rotations"
        return None, diag
    if prev_R_model_arr.ndim != 3 or prev_R_model_arr.shape[1:] != (3, 3):
        diag["reason"] = "bad_prev_frame_rotation_shape"
        return None, diag
    prev_R_model_to_enu = prev_summary.get("transform_R")
    if prev_R_model_to_enu is None:
        prev_R_model_to_enu = prev_metadata.get("transform_R")
    if prev_R_model_to_enu is None and segment_input.prev_transform is not None:
        prev_R_model_to_enu = segment_input.prev_transform.R.detach().cpu().float().numpy()
    prev_R_model_to_enu = _nearest_rotation_np(prev_R_model_to_enu)
    if prev_R_model_to_enu is None:
        diag["reason"] = "no_prev_transform_rotation"
        return None, diag
    if not per_frame_pose:
        diag["reason"] = "no_current_pose"
        return None, diag
    prev_count = int(prev_R_model_arr.shape[0])
    rots: List[np.ndarray] = []
    for oi in overlap_indices:
        oi_i = int(oi)
        prev_idx = prev_count - int(overlap_len) + oi_i
        if prev_idx < 0 or prev_idx >= prev_count or oi_i < 0 or oi_i >= len(per_frame_pose):
            continue
        R_prev_model = _nearest_rotation_np(prev_R_model_arr[prev_idx])
        R_curr_model = _pose_to_R_c2w_model_np(per_frame_pose[oi_i])
        R_curr_model = _nearest_rotation_np(R_curr_model)
        if R_prev_model is None or R_curr_model is None:
            continue
        R_prev_c2w_enu = prev_R_model_to_enu @ R_prev_model
        rots.append(R_prev_c2w_enu @ R_curr_model.T)
    R_avg = _average_rotation_np(rots)
    if R_avg is None:
        diag.update({"reason": "no_valid_rotation_pairs", "n": int(len(rots))})
        return None, diag
    diag.update({"accepted": True, "reason": "ok", "n": int(len(rots))})
    return R_avg.astype(np.float64), diag


def _overlap_prior_reliability(
    segment_input: SegmentInput,
    overlap_prior_transform: Optional[LocalModelTransform],
    overlap_prior_diag: Optional[Dict[str, Any]],
    config: Optional[Any],
) -> Tuple[bool, Dict[str, Any]]:
    """判定重叠先验变换是否足够可靠，可作为 bootstrap/primary 使用。

    依次门控：开关是否开启、是否 segment 0、变换是否存在且 accepted、重叠帧数
    是否达到 min_frames、median_xy_residual_m 是否在 max_residual 内。任一不满足
    则返回 (False, 带原因的诊断)；全部通过返回 (True, 诊断)。
    """
    diag = dict(overlap_prior_diag or {})
    if config is not None and hasattr(config, "overlap_prior_bootstrap_enable"):
        enabled = (
            bool(getattr(config, "overlap_prior_bootstrap_enable"))
            and bool(getattr(config, "overlap_prior_primary_enable", True))
        )
    else:
        enabled = bool(getattr(config, "overlap_prior_primary_enable", True)) if config is not None else True
    max_residual = float(getattr(config, "overlap_prior_primary_max_residual_m", 2.0)) if config is not None else 2.0
    min_frames = int(getattr(config, "overlap_prior_primary_min_frames", 3)) if config is not None else 3
    if not enabled:
        diag["bootstrap_reason"] = "disabled"
        diag["primary_reason"] = "disabled"
        return False, diag
    if int(segment_input.segment_id) <= 0:
        diag["bootstrap_reason"] = "seg0"
        diag["primary_reason"] = "seg0"
        return False, diag
    if overlap_prior_transform is None:
        diag["bootstrap_reason"] = "no_transform"
        diag["primary_reason"] = "no_transform"
        return False, diag
    if not bool(diag.get("accepted", False)):
        _reason = str(diag.get("reason", "not_accepted"))
        diag["bootstrap_reason"] = _reason
        diag["primary_reason"] = _reason
        return False, diag
    n_overlap = int(diag.get("n_overlap", 0) or 0)
    if n_overlap < min_frames:
        _reason = f"too_few_overlap:{n_overlap}<{min_frames}"
        diag["bootstrap_reason"] = _reason
        diag["primary_reason"] = _reason
        return False, diag
    residual = float(diag.get("median_xy_residual_m", float("inf")))
    if not np.isfinite(residual) or residual > max_residual:
        _reason = f"residual:{residual:.3f}>{max_residual:.3f}"
        diag["bootstrap_reason"] = _reason
        diag["primary_reason"] = _reason
        return False, diag
    diag.update({
        "bootstrap_reason": "ok",
        "primary_reason": "ok",
        "primary_max_residual_m": float(max_residual),
        "primary_min_frames": int(min_frames),
    })
    return True, diag


def _filter_geo_obs_for_overlap_primary(
    geo_obs: Dict[int, Dict[str, Any]],
    per_frame_pose: Optional[List[torch.Tensor]],
    overlap_prior_transform: Optional[LocalModelTransform],
    overlap_len: int,
    config: Optional[Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """当重叠先验作为主约束时，过滤地理观测，剔除与先验不一致者。

    用 overlap_prior_transform 把各帧相机中心投到 ENU 得到预测 XY；逐帧比较观测的
    目标 XY（dom_points 模式取 DOM 点目标，否则取固定 R 的 PnP 中心或 DOM 单应帧
    中心）与预测 XY 的距离，超过 max_xy_m 则丢弃。重叠头部帧无条件保留（dom_points
    模式下仍需通过一致性检查）。返回 (过滤后的 geo_obs, 统计 dict)。
    """
    filtered: Dict[int, Dict[str, Any]] = {}
    head_overlap = 0
    new_kept = 0
    skipped = 0
    residuals: List[float] = []
    max_xy = float(getattr(config, "overlap_prior_consistency_max_xy_m", 25.0)) if config is not None else 25.0
    dom_points_mode = _geo_target_mode_from_config(config) == "dom_points"
    pred_xy_by_fi: Dict[int, np.ndarray] = {}
    if overlap_prior_transform is not None and per_frame_pose is not None:
        try:
            centers = []
            for pose in per_frame_pose:
                center = pose_enc_to_camera_centers(pose)
                if center.ndim > 1:
                    center = center[0]
                centers.append(center[:3].detach().cpu().float().numpy())
            if centers:
                center_t = torch.tensor(
                    np.asarray(centers, dtype=np.float32),
                    device=overlap_prior_transform.s.device,
                    dtype=overlap_prior_transform.s.dtype,
                )
                pred = overlap_prior_transform.model_to_world(center_t).detach().cpu().numpy()
                for fi in range(pred.shape[0]):
                    pred_xy_by_fi[int(fi)] = pred[fi, :2].astype(np.float64)
        except Exception:
            pred_xy_by_fi = {}

    def _target_xy(corr: Dict[str, Any]) -> Optional[np.ndarray]:
        if dom_points_mode:
            return _dom_point_target_xy_from_corr(corr)
        # Match the downstream target convention: fixed-R render PnP provides a
        # camera center; otherwise the DOM homography frame center is used.  A
        # free/nadir observation is allowed when this target is consistent with
        # the overlap-SRT prediction.
        pnp_center = _fixed_rotation_pnp_center(corr, min_dim=2)
        if pnp_center is not None:
            return np.asarray(pnp_center[:2], dtype=np.float64)
        fce = corr.get("frame_center_enu") if isinstance(corr, dict) else None
        if fce is None or len(fce) < 2:
            return None
        arr = np.asarray(fce, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 2 or not np.isfinite(arr[:2]).all():
            return None
        return arr[:2]

    for fi, corr in sorted((geo_obs or {}).items()):
        if corr is None:
            continue
        fi_i = int(fi)
        if dom_points_mode:
            ok_sem, reason_sem, sem_info = _dom_point_observation_semantics(
                corr, config=config, stage="overlap_primary")
            if not ok_sem:
                skipped += 1
                if isinstance(corr, dict):
                    corr.setdefault("dom_point_observation", {})["overlap_primary"] = {
                        "valid": False,
                        "reason": reason_sem,
                        **sem_info,
                    }
                continue
        if overlap_len > 0 and fi_i < int(overlap_len):
            if dom_points_mode and overlap_prior_transform is not None:
                target_xy = _target_xy(corr)
                pred_xy = pred_xy_by_fi.get(fi_i)
                if target_xy is not None and pred_xy is not None:
                    residual = float(np.linalg.norm(target_xy[:2] - pred_xy[:2]))
                    if np.isfinite(residual) and residual <= max_xy:
                        if isinstance(corr, dict):
                            corr["overlap_prior_xy_residual_m"] = residual
                            corr.setdefault("dom_point_observation", {})["overlap_primary"] = {
                                "valid": True,
                                "reason": "ok",
                                "overlap_prior_xy_residual_m": residual,
                                "max_xy_m": float(max_xy),
                            }
                        residuals.append(residual)
                    else:
                        skipped += 1
                        if isinstance(corr, dict):
                            corr.setdefault("dom_point_observation", {})["overlap_primary"] = {
                                "valid": False,
                                "reason": "head_overlap_conflicts_overlap_prior",
                                "overlap_prior_xy_residual_m": None if target_xy is None or pred_xy is None else residual,
                                "max_xy_m": float(max_xy),
                            }
                        continue
                else:
                    skipped += 1
                    if isinstance(corr, dict):
                        corr.setdefault("dom_point_observation", {})["overlap_primary"] = {
                            "valid": False,
                            "reason": "head_overlap_missing_overlap_prior_xy",
                            "max_xy_m": float(max_xy),
                        }
                    continue
            filtered[fi_i] = corr
            head_overlap += 1
            continue
        target_xy = _target_xy(corr)
        pred_xy = pred_xy_by_fi.get(fi_i)
        if target_xy is not None and pred_xy is not None:
            residual = float(np.linalg.norm(target_xy[:2] - pred_xy[:2]))
            if np.isfinite(residual) and residual <= max_xy:
                if isinstance(corr, dict):
                    corr["overlap_prior_xy_residual_m"] = residual
                residuals.append(residual)
                filtered[fi_i] = corr
                new_kept += 1
                continue
        elif overlap_prior_transform is None:
            filtered[fi_i] = corr
            new_kept += 1
            continue
        skipped += 1
    return filtered, {
        "kept": int(len(filtered)),
        "head_overlap": int(head_overlap),
        "new_kept": int(new_kept),
        "skipped": int(skipped),
        "max_xy_m": float(max_xy),
        "median_new_xy_m": float(np.median(residuals)) if residuals else None,
        "max_new_xy_m": float(np.max(residuals)) if residuals else None,
    }


def _estimate_sim3_3d(
    model_xyz: np.ndarray,
    enu_xyz: np.ndarray,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """3D Umeyama / Sim3 估计：求 (s, R_3x3, t_3d) 使 enu_xyz ≈ s * R @ model_xyz + t。

    输入为配对的 model 与 ENU 三维坐标（各 n×3，至少 3 对有限点）。经去中心 SVD
    求 R（含反射修正）并由奇异值/方差求尺度 s。返回 (s, R(3x3), t(3d))，形状非法、
    点数不足、方差/尺度退化或结果非有限时返回 None。
    """
    X = np.asarray(model_xyz, dtype=np.float64)
    Y = np.asarray(enu_xyz, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape or X.shape[1] != 3 or len(X) < 3:
        return None
    finite = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if finite.sum() < 3:
        return None
    X = X[finite]
    Y = Y[finite]
    n = int(X.shape[0])
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    var_x = float((Xc ** 2).sum()) / max(n, 1)
    if not np.isfinite(var_x) or var_x < 1e-12:
        return None
    H = Xc.T @ Yc / max(n, 1)
    try:
        U, sv, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    D = np.eye(3, dtype=np.float64)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1.0
    R = Vt.T @ D @ U.T
    s = float((sv * np.diag(D)).sum() / var_x)
    if not np.isfinite(s) or s <= 1e-8:
        return None
    t = (mu_y - float(s) * (mu_x @ R.T)).astype(np.float64)
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        return None
    return float(s), R.astype(np.float64), t.astype(np.float64)


def _rotation_delta_deg(R_new: np.ndarray, R_ref: np.ndarray) -> float:
    """计算两个 3D 旋转之间的测地角差（度）。

    由 R_delta = R_new @ R_ref.T 的迹经 arccos((trace-1)/2) 得到夹角，trace 项做
    clip 防止数值越界。返回旋转角的绝对大小。
    """
    R_delta = np.asarray(R_new, dtype=np.float64) @ np.asarray(R_ref, dtype=np.float64).T
    c = (float(np.trace(R_delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _fit_ransac_srt_point_edges(
    point_model: Optional[np.ndarray],
    point_targets_enu: Optional[np.ndarray],
    transform_init: LocalModelTransform,
    config: GeoV3Config,
    seg_id: int,
    min_inliers_override: Optional[int] = None,
    disable_pose_gate: bool = False,
    pose_gate_reference: Optional[LocalModelTransform] = None,
    pose_gate_reference_source: str = "transform_init",
) -> Tuple[Optional[LocalModelTransform], Optional[np.ndarray], Dict[str, Any]]:
    """用 RANSAC 从 DOM 点对 (model→ENU) 拟合 Sim3 变换 (s, R, t)。

    在 transform_init 基础上随机采样点对调用 _estimate_sim3_3d 估计候选 Sim3，
    用带 z_weight 的加权距离与 thresh_m 统计内点，并经 pose gate（尺度比、旋转角
    限制，可由 disable_pose_gate 关闭）筛选，最后对最优内点集重拟合。若提供
    pose_gate_reference，则尺度/旋转相对该变换比较，而不是默认的 transform_init。
    返回 (LocalModelTransform, 内点掩码, stats)；点数不足、RANSAC 失败或未过门控
    时返回 (None, None, stats)，stats 记录原因与各项误差指标。
    """
    stats: Dict[str, Any] = {"enabled": True, "accepted": False, "reason": "not_run"}
    if point_model is None or point_targets_enu is None:
        stats["reason"] = "no_points"
        return None, None, stats
    X = np.asarray(point_model, dtype=np.float64)
    Y = np.asarray(point_targets_enu, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape or X.shape[1] != 3:
        stats["reason"] = "bad_shape"
        return None, None, stats
    finite = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[finite]
    Y = Y[finite]
    n = int(X.shape[0])
    stats["num_points"] = n
    if min_inliers_override is not None:
        min_inliers = max(4, int(min_inliers_override))
    else:
        min_inliers = max(4, int(getattr(config, "dom_point_srt_min_inliers", 32)))
    if n < min_inliers:
        stats["reason"] = "too_few_points"
        return None, None, stats

    # ``transform_init`` is normally only an anchor translation (R=I, s=1).
    # For map-only, using that synthetic value as an absolute pose prior rejects
    # valid metric reconstructions whose model-to-ENU scale is tens/hundreds.
    # A reliable overlap handoff may be supplied explicitly as the reference;
    # otherwise the caller can disable this optional prior gate.
    gate_ref = pose_gate_reference if pose_gate_reference is not None else transform_init
    R_ref = gate_ref.R.detach().cpu().float().numpy().astype(np.float64)
    s_ref = float(gate_ref.s.detach().cpu().float())
    t_ref = gate_ref.t.detach().cpu().float().numpy().astype(np.float64)
    thresh_m = max(1e-6, float(getattr(config, "dom_point_srt_thresh_m", 15.0)))
    z_weight = max(0.0, float(getattr(config, "dom_point_srt_z_weight", 0.25)))
    min_ratio = max(0.0, float(getattr(config, "dom_point_srt_min_ratio", 0.05)))
    scale_ratio_min = max(1e-6, float(getattr(config, "dom_point_srt_scale_ratio_min", 0.5)))
    scale_ratio_max = max(scale_ratio_min, float(getattr(config, "dom_point_srt_scale_ratio_max", 2.0)))
    max_rot_deg = max(0.0, float(getattr(config, "dom_point_srt_max_rot_deg", 60.0)))
    iters = max(1, int(getattr(config, "dom_point_srt_ransac_iters", 256)))
    stats.update({
        "pose_gate_disabled": bool(disable_pose_gate),
        "pose_gate_reference": str(pose_gate_reference_source),
        "pose_gate_scale_ref": float(s_ref),
        "pose_gate_scale_ratio_min": float(scale_ratio_min),
        "pose_gate_scale_ratio_max": float(scale_ratio_max),
        "pose_gate_max_rot_deg": float(max_rot_deg),
    })

    def _passes_pose_gate(s_val: float, R_val: np.ndarray) -> Tuple[bool, float, float]:
        scale_ratio = float(s_val / s_ref) if abs(s_ref) > 1e-8 else 1.0
        rot_delta = _rotation_delta_deg(R_val, R_ref)
        if bool(disable_pose_gate):
            return True, scale_ratio, rot_delta
        ok = (scale_ratio_min <= scale_ratio <= scale_ratio_max and rot_delta <= max_rot_deg)
        return ok, scale_ratio, rot_delta

    def _eval_candidate(s_val: float, R_val: np.ndarray, t_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pred = float(s_val) * (X @ R_val.T) + t_val[None, :]
        diff = pred - Y
        err = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2 + (z_weight * diff[:, 2]) ** 2)
        return err <= thresh_m, err

    rng = np.random.RandomState(23 + int(seg_id))
    sample_n = 4 if n >= 4 else 3
    best_mask = None
    best_err = None
    best_score = (-1, np.inf)
    candidate_fits = 0
    pose_gate_rejected_candidates = 0
    pose_gate_passed_candidates = 0
    for _ in range(iters):
        idx = rng.choice(n, sample_n, replace=False)
        fit = _estimate_sim3_3d(X[idx], Y[idx])
        if fit is None:
            continue
        candidate_fits += 1
        s_cand, R_cand, t_cand = fit
        ok, _, _ = _passes_pose_gate(s_cand, R_cand)
        if not ok:
            pose_gate_rejected_candidates += 1
            continue
        pose_gate_passed_candidates += 1
        mask, err = _eval_candidate(s_cand, R_cand, t_cand)
        inl = int(mask.sum())
        med = float(np.median(err[mask])) if inl > 0 else float("inf")
        score = (inl, -med)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_err = err

    if best_mask is None or int(best_mask.sum()) < min_inliers:
        all_fitted_rejected = (
            not bool(disable_pose_gate)
            and candidate_fits > 0
            and pose_gate_rejected_candidates == candidate_fits
        )
        stats.update({
            "reason": "pose_gate_rejected_all_candidates" if all_fitted_rejected else "ransac_failed",
            "best_inliers": 0 if best_mask is None else int(best_mask.sum()),
            "candidate_fits": int(candidate_fits),
            "pose_gate_rejected_candidates": int(pose_gate_rejected_candidates),
            "pose_gate_passed_candidates": int(pose_gate_passed_candidates),
        })
        return None, None, stats
    if float(best_mask.mean()) < min_ratio:
        stats.update({"reason": "low_inlier_ratio", "best_inliers": int(best_mask.sum()), "inlier_ratio": float(best_mask.mean())})
        return None, None, stats

    final_fit = _estimate_sim3_3d(X[best_mask], Y[best_mask])
    if final_fit is None:
        stats.update({"reason": "refit_failed", "best_inliers": int(best_mask.sum())})
        return None, None, stats
    s_final, R_final, t_final = final_fit
    ok, scale_ratio, rot_delta = _passes_pose_gate(s_final, R_final)
    if not ok:
        stats.update({
            "reason": "pose_gate_failed",
            "best_inliers": int(best_mask.sum()),
            "scale_ratio": float(scale_ratio),
            "rot_delta_deg": float(rot_delta),
            "candidate_fits": int(candidate_fits),
            "pose_gate_rejected_candidates": int(pose_gate_rejected_candidates),
            "pose_gate_passed_candidates": int(pose_gate_passed_candidates),
        })
        return None, None, stats
    final_mask, final_err = _eval_candidate(s_final, R_final, t_final)
    if int(final_mask.sum()) < min_inliers:
        stats.update({"reason": "refit_too_few_inliers", "best_inliers": int(final_mask.sum())})
        return None, None, stats

    pred = float(s_final) * (X @ R_final.T) + t_final[None, :]
    diff = pred - Y
    xy_err = np.linalg.norm(diff[:, :2], axis=1)
    z_err = np.abs(diff[:, 2])
    stats.update({
        "accepted": True,
        "reason": "ok",
        "inliers": int(final_mask.sum()),
        "inlier_ratio": float(final_mask.mean()),
        "median_err_m": float(np.median(final_err[final_mask])),
        "median_xy_m": float(np.median(xy_err[final_mask])),
        "median_z_m": float(np.median(z_err[final_mask])),
        "scale": float(s_final),
        "scale_init": float(s_ref),
        "scale_ratio": float(scale_ratio),
        "rot_delta_deg": float(rot_delta),
        "t_delta_m": float(np.linalg.norm(t_final - t_ref)),
        "threshold_m": float(thresh_m),
        "z_weight": float(z_weight),
        "pose_gate_disabled": bool(disable_pose_gate),
        "pose_gate_reference": str(pose_gate_reference_source),
        "candidate_fits": int(candidate_fits),
        "pose_gate_rejected_candidates": int(pose_gate_rejected_candidates),
        "pose_gate_passed_candidates": int(pose_gate_passed_candidates),
    })
    device = transform_init.s.device
    dtype = transform_init.s.dtype
    transform = LocalModelTransform(
        anchor_world_xyz=transform_init.anchor_world_xyz,
        R=torch.tensor(R_final.astype(np.float32), device=device, dtype=dtype),
        s=torch.tensor(float(s_final), device=device, dtype=dtype),
        t=torch.tensor(t_final.astype(np.float32), device=device, dtype=dtype),
    )
    return transform, final_mask.astype(bool), stats


def _refit_point_transform_with_fixed_rotation(
    point_model: np.ndarray,
    point_targets_enu: np.ndarray,
    initial_inlier_mask: np.ndarray,
    fixed_rotation: np.ndarray,
    transform_init: LocalModelTransform,
    config: GeoV3Config,
) -> Tuple[Optional[LocalModelTransform], Optional[np.ndarray], Dict[str, Any]]:
    """Refit point-map scale/translation while keeping a trusted 3D rotation.

    DOM/DEM correspondences lie almost entirely on the ground plane. A free
    Sim(3) can fit that plane well while retaining a poorly constrained tilt;
    applying that tilt to an off-plane camera center then creates a large XY
    error. The free RANSAC fit is therefore used only to select correspondences.
    This refit estimates scale from centered XY point geometry and translation
    from robust coordinate-wise medians under a trusted pose-derived rotation.
    """
    stats: Dict[str, Any] = {
        "enabled": True,
        "accepted": False,
        "reason": "not_run",
    }
    X = np.asarray(point_model, dtype=np.float64)
    Y = np.asarray(point_targets_enu, dtype=np.float64)
    mask_in = np.asarray(initial_inlier_mask, dtype=bool).reshape(-1)
    R = _nearest_rotation_np(fixed_rotation)
    if (
        X.ndim != 2
        or Y.ndim != 2
        or X.shape != Y.shape
        or X.shape[1] != 3
        or mask_in.shape[0] != X.shape[0]
    ):
        stats["reason"] = "bad_shape"
        return None, None, stats
    if R is None:
        stats["reason"] = "bad_rotation"
        return None, None, stats

    finite = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    mask = mask_in & finite
    min_inliers = max(4, int(getattr(config, "dom_point_srt_min_inliers", 32)))
    if int(mask.sum()) < min_inliers:
        stats.update({"reason": "too_few_initial_inliers", "inliers": int(mask.sum())})
        return None, None, stats

    rotated = X @ R.T
    threshold_m = max(1e-6, float(getattr(config, "dom_point_srt_thresh_m", 15.0)))
    z_weight = max(0.0, float(getattr(config, "dom_point_srt_z_weight", 0.25)))
    scale: Optional[float] = None
    translation: Optional[np.ndarray] = None

    # Two robust refinement rounds are enough because R is fixed and only a
    # scalar scale plus translation remain.
    for _ in range(2):
        scale = _estimate_uniform_scale_after_rotation(rotated[mask, :2], Y[mask, :2])
        if scale is None:
            stats["reason"] = "scale_fit_failed"
            return None, None, stats
        offsets = Y[mask] - float(scale) * rotated[mask]
        translation = np.median(offsets, axis=0)
        pred = float(scale) * rotated + translation[None, :]
        diff = pred - Y
        weighted_err = np.sqrt(
            diff[:, 0] ** 2 + diff[:, 1] ** 2 + (z_weight * diff[:, 2]) ** 2
        )
        refined = finite & (weighted_err <= threshold_m)
        if int(refined.sum()) < min_inliers:
            stats.update({"reason": "too_few_refined_inliers", "inliers": int(refined.sum())})
            return None, None, stats
        mask = refined

    assert scale is not None and translation is not None
    pred = float(scale) * rotated + translation[None, :]
    diff = pred - Y
    xy_err = np.linalg.norm(diff[:, :2], axis=1)
    z_err = np.abs(diff[:, 2])
    weighted_err = np.sqrt(
        diff[:, 0] ** 2 + diff[:, 1] ** 2 + (z_weight * diff[:, 2]) ** 2
    )
    inlier_ratio = float(mask.mean()) if len(mask) else 0.0
    min_ratio = max(0.0, float(getattr(config, "dom_point_srt_min_ratio", 0.05)))
    if inlier_ratio < min_ratio:
        stats.update({"reason": "low_inlier_ratio", "inlier_ratio": inlier_ratio})
        return None, None, stats

    transform = LocalModelTransform(
        anchor_world_xyz=transform_init.anchor_world_xyz,
        R=torch.tensor(R.astype(np.float32), device=transform_init.s.device, dtype=transform_init.s.dtype),
        s=torch.tensor(float(scale), device=transform_init.s.device, dtype=transform_init.s.dtype),
        t=torch.tensor(translation.astype(np.float32), device=transform_init.s.device, dtype=transform_init.s.dtype),
    )
    stats.update({
        "accepted": True,
        "reason": "ok",
        "inliers": int(mask.sum()),
        "inlier_ratio": inlier_ratio,
        "scale": float(scale),
        "median_err_m": float(np.median(weighted_err[mask])),
        "median_xy_m": float(np.median(xy_err[mask])),
        "median_z_m": float(np.median(z_err[mask])),
        "threshold_m": float(threshold_m),
        "z_weight": float(z_weight),
    })
    return transform, mask.astype(bool), stats


def _smooth_l1_from_diff(diff: torch.Tensor, beta: float) -> torch.Tensor:
    """对差值张量 diff 计算 Smooth L1（Huber）损失并取均值。

    beta 为 L2 与 L1 分段的切换阈值（会被下限裁到 1e-6）：|diff|<beta 段用二次项，
    其余用线性项。返回标量损失。
    """
    beta = max(float(beta), 1e-6)
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta,
                       0.5 * abs_diff * abs_diff / beta,
                       abs_diff - 0.5 * beta)
    return loss.mean()


def _as_pts3d_tensor(pts3d_like: Any) -> torch.Tensor:
    """将各种容器形式的 pts3d 归一为形状 (H, W, 3) 的 float 张量。

    支持 dict（按 pts3d/points3d/points/xyz 取键）、tuple/list（取首元素）或张量；
    自动去除前导的 batch/序列维（5 维或 4 维）。返回仅含前 3 通道的坐标张量；
    类型不符或形状不是 (H,W,>=3) 时抛出 TypeError/ValueError。
    """
    if isinstance(pts3d_like, dict):
        for key in ("pts3d", "points3d", "points", "xyz"):
            if key in pts3d_like:
                pts3d_like = pts3d_like[key]
                break
    elif isinstance(pts3d_like, (tuple, list)):
        if not pts3d_like:
            raise ValueError("empty pts3d container")
        pts3d_like = pts3d_like[0]
    if not torch.is_tensor(pts3d_like):
        raise TypeError(f"expected pts3d tensor, got {type(pts3d_like).__name__}")
    pts = pts3d_like.float()
    if pts.ndim == 5:
        pts = pts[0, 0]
    elif pts.ndim == 4:
        pts = pts[0]
    if pts.ndim != 3 or pts.shape[-1] < 3:
        raise ValueError(f"expected pts3d shape (H,W,3), got {tuple(pts.shape)}")
    return pts[..., :3]


def _sample_pts3d_at_query_pixels(
    pts3d_fi: Any,
    px_arr: np.ndarray,
    query_w: float,
    query_h: float,
    detach: bool = False,
) -> torch.Tensor:
    """在 pts3d 图上按查询像素坐标双线性采样三维点。

    px_arr 为 (N,2) 的像素坐标，query_w/query_h 为其参考图宽高，用于把像素归一到
    [-1,1] 的 grid_sample 归一坐标（align_corners=True，边缘 padding）。detach 为 True
    时切断梯度。返回 (N,3) 的采样三维坐标张量。
    """
    pts = _as_pts3d_tensor(pts3d_fi)
    if detach:
        pts = pts.detach()
    pts_2hw = pts.permute(2, 0, 1).unsqueeze(0)
    px_np = np.asarray(px_arr, dtype=np.float32)
    u = torch.tensor(
        px_np[:, 0] / max(float(query_w) - 1.0, 1.0),
        device=pts_2hw.device,
        dtype=torch.float32,
    )
    v = torch.tensor(
        px_np[:, 1] / max(float(query_h) - 1.0, 1.0),
        device=pts_2hw.device,
        dtype=torch.float32,
    )
    grid = torch.stack([2.0 * u - 1.0, 2.0 * v - 1.0], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        pts_2hw, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return sampled[0, :, 0, :].T


def _select_dom_point_correspondences(
    corr: Dict[str, Any],
    road_only: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从 corr 中取出像素-ENU 配对点 (frame_pixels, enu_positions)。

    road_only 为 True 时优先取道路子集 (frame_pixels_road / enu_positions_road)，
    缺失则回退到全量配对。返回 ((N,2) 像素, (N,3) ENU)，校验维度且要求至少 4 对，
    不合法时返回 (None, None)。
    """
    px = None
    enu = None
    if road_only:
        px = corr.get("frame_pixels_road")
        enu = corr.get("enu_positions_road")
    if px is None or enu is None:
        px = corr.get("frame_pixels")
        enu = corr.get("enu_positions")
    if px is None or enu is None:
        return None, None
    px_arr = np.asarray(px, dtype=np.float32)
    enu_arr = np.asarray(enu, dtype=np.float64)
    if (px_arr.ndim != 2 or px_arr.shape[1] < 2
            or enu_arr.ndim != 2 or enu_arr.shape[1] < 3
            or px_arr.shape[0] != enu_arr.shape[0]
            or px_arr.shape[0] < 4):
        return None, None
    return px_arr[:, :2], enu_arr[:, :3]


def _select_dom_point_correspondences_road_then_nonbuilding(
    corr: Dict[str, Any],
    min_points: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """按"先道路、后非建筑"的优先级从 corr 选取像素-ENU 配对点。

    先尝试道路子集 (frame_pixels_road / enu_positions_road)；不足则在已做过建筑
    过滤 (building_filter_applied) 的前提下退回到全量非建筑配对。每组都要求至少
    min_points 对有效点。返回 ((N,2) 像素, (N,3) ENU, 来源标签)，标签为
    road/nonbuilding/no_building_filter/no_nonbuilding_points 之一，失败时前两项为 None。
    """
    min_points = max(1, int(min_points))

    def _valid_arrays(px_val: Any, enu_val: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if px_val is None or enu_val is None:
            return None, None
        px_arr = np.asarray(px_val, dtype=np.float32)
        enu_arr = np.asarray(enu_val, dtype=np.float64)
        if (px_arr.ndim != 2 or px_arr.shape[1] < 2
                or enu_arr.ndim != 2 or enu_arr.shape[1] < 3
                or px_arr.shape[0] != enu_arr.shape[0]
                or px_arr.shape[0] < min_points):
            return None, None
        return px_arr[:, :2], enu_arr[:, :3]

    px_road, enu_road = _valid_arrays(
        corr.get("frame_pixels_road"), corr.get("enu_positions_road"))
    if px_road is not None and enu_road is not None:
        return px_road, enu_road, "road"

    # R2 No-semantic-mask: use every geometrically valid correspondence.
    # This must be an explicit marker; otherwise a missing mask would look
    # identical to a broken/misconfigured semantic-filter input.
    if bool(corr.get("semantic_masks_disabled", False)):
        px_full, enu_full = _valid_arrays(
            corr.get("frame_pixels"), corr.get("enu_positions"))
        if px_full is not None and enu_full is not None:
            return px_full, enu_full, "unfiltered"
        return None, None, "no_unfiltered_points"

    if not bool(corr.get("building_filter_applied", False)):
        return None, None, "no_building_filter"
    px_nonbld, enu_nonbld = _valid_arrays(
        corr.get("frame_pixels"), corr.get("enu_positions"))
    if px_nonbld is not None and enu_nonbld is not None:
        return px_nonbld, enu_nonbld, "nonbuilding"
    return None, None, "no_nonbuilding_points"


def _dem_z_direct_sample_allowed(sample_source: str, config: GeoV3Config) -> bool:
    """Return whether a DEM-Z source can directly translate a segment."""
    source = str(sample_source).strip().lower()
    if source == "road":
        return True
    return (
        source == "nonbuilding_fallback"
        and bool(getattr(config, "dom_point_z_nonbuilding_fallback_enable", False))
    )


def _segment_pgo_should_skip_overlap_only(
    *,
    dom_points_mode: bool,
    fixed_rotation_bootstrap_accepted: bool,
    num_dom_targets: int,
    has_point_edges: bool,
    has_ground_ray_edges: bool,
) -> bool:
    """Do not replace a reliable map transform with an overlap-only refit."""
    return (
        bool(dom_points_mode)
        and bool(fixed_rotation_bootstrap_accepted)
        and int(num_dom_targets) <= 0
        and not bool(has_point_edges)
        and not bool(has_ground_ray_edges)
    )


def _dom_point_observation_semantics(
    corr: Dict[str, Any],
    *,
    config: Optional[GeoV3Config] = None,
    stage: str = "supervision",
) -> Tuple[bool, str, Dict[str, Any]]:
    """判断某帧的 DOM 点观测在语义上是否可信，用于决定能否用作监督。

    依据 corr 中的裁剪模式 (dom_crop_info.mode)、位姿裁剪失败原因、PnP 方法及内点数
    n_inliers 进行门控：位姿裁剪失败、nadir 模式下缓存阶段无 PnP 位姿、或内点不足 4
    均判为无效。可由 config.dom_point_observation_validity_enable 关闭（返回 disabled）。
    返回 (是否有效, 原因标签, 诊断信息 dict)。
    """
    if not isinstance(corr, dict):
        return False, "bad_corr", {}
    if config is not None and not bool(getattr(config, "dom_point_observation_validity_enable", True)):
        return True, "disabled", {"enabled": False}
    crop_info = corr.get("dom_crop_info") or {}
    if not isinstance(crop_info, dict):
        crop_info = {}
    mode = str(crop_info.get("mode") or "").lower()
    pose_fail = crop_info.get("pose_crop_failure_reason")
    pnp = corr.get("pnp_result") if isinstance(corr, dict) else None
    pnp_method = str(pnp.get("method") if isinstance(pnp, dict) else "none")
    n_inliers = int(corr.get("n_inliers", 0) or 0)
    info = {
        "enabled": True,
        "stage": str(stage),
        "crop_mode": mode,
        "pose_fail": pose_fail,
        "pnp_method": pnp_method,
        "n_inliers": n_inliers,
    }
    if pose_fail is not None:
        return False, f"pose_crop_failed:{pose_fail}", info
    if mode == "nadir" and pnp_method == "none" and stage in {"overlap_cache", "cache_reuse"}:
        return False, "nadir_without_pose_cache", info
    if n_inliers < 4:
        return False, "too_few_matches", info
    return True, "ok", info


def _dom_point_target_xy_from_corr(corr: Dict[str, Any]) -> Optional[np.ndarray]:
    """从 corr 中提取帧中心的 ENU 平面坐标 (frame_center_enu 的前 XY)。

    返回 (2,) 的 float64 数组；corr 非 dict、字段缺失或 XY 非有限时返回 None。
    """
    if not isinstance(corr, dict):
        return None
    try:
        arr = np.asarray(corr.get("frame_center_enu"), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return None
    return arr[:2].astype(np.float64)


def _dom_point_overlap_prior_xy_residual(
    corr: Dict[str, Any],
    fi: int,
    per_frame_pose: Optional[List[torch.Tensor]],
    overlap_prior_transform: Optional[LocalModelTransform],
) -> Optional[float]:
    """计算第 fi 帧相机中心经 overlap_prior_transform 投到 ENU 后与目标 XY 的平面残差。

    把 per_frame_pose[fi] 的相机中心用先验变换投到世界 ENU，与 corr 中帧中心目标 XY
    (_dom_point_target_xy_from_corr) 求平面距离（米）。缺先验/位姿、索引越界或数据无效
    时返回 None。
    """
    if overlap_prior_transform is None or per_frame_pose is None:
        return None
    if int(fi) < 0 or int(fi) >= len(per_frame_pose):
        return None
    target_xy = _dom_point_target_xy_from_corr(corr)
    if target_xy is None:
        return None
    try:
        center = pose_enc_to_camera_centers(per_frame_pose[int(fi)])
        if center.ndim > 1:
            center = center[0]
        center_t = center[:3].detach().to(
            device=overlap_prior_transform.s.device,
            dtype=overlap_prior_transform.s.dtype,
        ).view(1, 3)
        pred = overlap_prior_transform.model_to_world(center_t)[0].detach().cpu().numpy()
        if pred.shape[0] < 2 or not np.isfinite(pred[:2]).all():
            return None
        return float(np.linalg.norm(target_xy[:2] - pred[:2]))
    except Exception:
        return None


def _estimate_dom_point_camera_target_enu(
    point_model: np.ndarray,
    point_targets_enu: np.ndarray,
    center_model: np.ndarray,
    transform: LocalModelTransform,
    *,
    min_points: int = 8,
    max_points: int = 256,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """在固定 transform 的 (s, R) 下，估计使点对最一致的相机中心 ENU 目标。

    把每个 model 点相对相机中心 center_model 的偏移按 s*R 旋到 ENU 得到 rel_enu，
    则相机中心候选 = target_enu - rel_enu，对各点取中位数作为稳健估计。点数超过
    max_points 时按 seed 随机下采样。返回 ((3,) ENU 相机目标, stats)；点数不足或
    形状非法时返回 (None, stats)，stats 记录 XY/Z 残差统计。
    """
    stats: Dict[str, Any] = {"accepted": False, "reason": "not_run"}
    if transform is None:
        stats["reason"] = "no_transform"
        return None, stats
    pts = np.asarray(point_model, dtype=np.float64)
    tgt = np.asarray(point_targets_enu, dtype=np.float64)
    center = np.asarray(center_model, dtype=np.float64).reshape(-1)
    if (pts.ndim != 2 or tgt.ndim != 2 or pts.shape != tgt.shape
            or pts.shape[1] < 3 or center.size < 3):
        stats["reason"] = "bad_shape"
        return None, stats
    finite = np.isfinite(pts[:, :3]).all(axis=1) & np.isfinite(tgt[:, :3]).all(axis=1)
    finite = finite & np.isfinite(center[:3]).all()
    pts = pts[finite, :3]
    tgt = tgt[finite, :3]
    stats["valid_points"] = int(len(pts))
    min_points = max(4, int(min_points))
    if len(pts) < min_points:
        stats["reason"] = "too_few_points"
        return None, stats
    max_points = int(max_points)
    if max_points > 0 and len(pts) > max_points:
        rng = np.random.RandomState(int(seed))
        sel = rng.choice(len(pts), max_points, replace=False)
        sel.sort()
        pts = pts[sel]
        tgt = tgt[sel]

    R = transform.R.detach().cpu().float().numpy().astype(np.float64)
    s = float(transform.s.detach().cpu().float())
    rel_enu = float(s) * ((pts - center[:3][None, :]) @ R.T)
    camera_candidates = tgt - rel_enu
    camera_target = np.median(camera_candidates, axis=0).astype(np.float64)
    pred_pts = camera_target[None, :] + rel_enu
    diff = pred_pts - tgt
    xy = np.linalg.norm(diff[:, :2], axis=1)
    z = np.abs(diff[:, 2])
    stats.update({
        "accepted": True,
        "reason": "ok",
        "points": int(len(pts)),
        "xy_median_m": float(np.median(xy)),
        "xy_mean_m": float(np.mean(xy)),
        "xy_p90_m": float(np.percentile(xy, 90.0)),
        "z_median_m": float(np.median(z)),
        "z_mean_m": float(np.mean(z)),
    })
    return camera_target.astype(np.float32), stats


def _normalize_dom_point_camera_teacher_source(raw: Any) -> str:
    """把相机 teacher 信号来源的原始配置字符串归一为标准枚举值。

    接受各种别名/大小写/连字符写法，映射到 none / frame_center / ground_signal /
    fixedR_pnp / point_rays / dom_points 之一；无法识别时默认 dom_points。
    """
    raw = str(raw or "dom_points")
    source = raw.strip().lower().replace("-", "_")
    if source in {"off", "none", "disable", "disabled"}:
        return "none"
    if source in {"frame", "frame_center", "frame_center_xy", "dom_frame_center"}:
        return "frame_center"
    if source in {"ground", "ground_signal", "ground_signal_xy", "dom_ground_signal"}:
        return "ground_signal"
    if source in {"pnp", "fixedr_pnp", "fixed_r_pnp", "fixed_rotation_pnp", "fixedr", "fixed_r"}:
        return "fixedR_pnp"
    if source in {"point_ray", "point_rays", "ray_point", "ray_points", "fixedr_point_rays", "fixed_r_point_rays"}:
        return "point_rays"
    if source in {"points", "point", "dom_point", "dom_points", "point_camera"}:
        return "dom_points"
    return "dom_points"


def _dom_point_camera_teacher_source(config: Optional[GeoV3Config]) -> str:
    """读取并归一 config.dom_point_camera_teacher_source（监督阶段相机 teacher 来源）。

    config 为 None 时默认 dom_points，再经 _normalize_dom_point_camera_teacher_source 归一。
    """
    raw = getattr(config, "dom_point_camera_teacher_source", "dom_points") if config is not None else "dom_points"
    return _normalize_dom_point_camera_teacher_source(raw)


def _dom_point_ttt_camera_teacher_source(config: Optional[GeoV3Config]) -> str:
    """读取并归一 config.dom_point_ttt_camera_teacher_source（TTT 阶段相机 teacher 来源）。

    config 为 None 时默认 dom_points，再经 _normalize_dom_point_camera_teacher_source 归一。
    """
    raw = getattr(config, "dom_point_ttt_camera_teacher_source", "dom_points") if config is not None else "dom_points"
    return _normalize_dom_point_camera_teacher_source(raw)


def _model_camera_center_to_enu_np(
    center_model: np.ndarray,
    transform: Optional[LocalModelTransform],
) -> Optional[np.ndarray]:
    """把 model 坐标系下的相机中心经 transform 投到世界 ENU 坐标（numpy 版）。

    center_model 取前 3 维，调用 transform.model_to_world 得到 ENU 中心。transform 为
    None、坐标非有限或过程异常时返回 None。返回 (3,) float64。
    """
    if transform is None:
        return None
    try:
        center = np.asarray(center_model, dtype=np.float64).reshape(-1)
        if center.size < 3 or not np.isfinite(center[:3]).all():
            return None
        center_t = torch.tensor(
            center[:3].reshape(1, 3),
            device=transform.s.device,
            dtype=transform.s.dtype,
        )
        target = transform.model_to_world(center_t)[0].detach().cpu().float().numpy()
        return target.astype(np.float64)
    except Exception:
        return None


def _solve_camera_center_from_point_rays_np(
    points_enu: np.ndarray,
    directions_enu: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """由多条 ENU 射线（点 + 方向）用最小二乘求交点，估计相机中心。

    每条射线贡献投影矩阵 (I - d·dᵀ)，累加构成法方程 normal_matrix·c = rhs，再用
    lstsq 解出最接近所有射线的点 c。directions_enu 无需预先归一（内部会单位化），
    mask 可选地只用部分射线。至少需 2 条有效射线；形状非法、退化或非有限时返回 None。
    返回 (3,) ENU 相机中心。
    """
    points = np.asarray(points_enu, dtype=np.float64)
    directions = np.asarray(directions_enu, dtype=np.float64)
    if mask is not None:
        valid_mask = np.asarray(mask, dtype=bool).reshape(-1)
        points = points[valid_mask]
        directions = directions[valid_mask]
    if points.ndim != 2 or directions.ndim != 2 or points.shape != directions.shape or points.shape[1] < 3:
        return None
    if points.shape[0] < 2:
        return None
    normal_matrix = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for point_enu, direction_enu in zip(points[:, :3], directions[:, :3]):
        direction_norm = float(np.linalg.norm(direction_enu))
        if direction_norm <= 1e-8:
            continue
        direction_unit = direction_enu / direction_norm
        projection = eye - np.outer(direction_unit, direction_unit)
        normal_matrix += projection
        rhs += projection @ point_enu
    if not np.isfinite(normal_matrix).all() or not np.isfinite(rhs).all():
        return None
    try:
        center_enu, *_ = np.linalg.lstsq(normal_matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if center_enu.size < 3 or not np.isfinite(center_enu[:3]).all():
        return None
    return center_enu[:3].astype(np.float64)


def _point_ray_residuals_np(
    center_enu: np.ndarray,
    points_enu: np.ndarray,
    directions_enu: np.ndarray,
) -> np.ndarray:
    """计算相机中心 center_enu 到各条射线（点 points_enu + 方向 directions_enu）的垂直距离。

    取 center 相对每个点的偏移 diff，减去其在单位方向上的投影分量得到正交分量，
    返回各射线正交距离 (N,) ——即点到射线的最近距离，作为 point-ray 残差。
    """
    center = np.asarray(center_enu, dtype=np.float64).reshape(3)
    points = np.asarray(points_enu, dtype=np.float64)[:, :3]
    directions = np.asarray(directions_enu, dtype=np.float64)[:, :3]
    diff = center[None, :] - points
    projected = diff - np.sum(diff * directions, axis=1, keepdims=True) * directions
    return np.linalg.norm(projected, axis=1)


def _point_ray_depths_np(
    center_enu: np.ndarray,
    points_enu: np.ndarray,
    directions_enu: np.ndarray,
) -> np.ndarray:
    """计算各点沿其射线方向相对相机中心的有向深度（沿视线的投影距离）。

    即 (points - center) 在单位方向 directions 上的点积，(N,)。正值表示点在相机
    前方，用于剔除位于相机背面的点。
    """
    center = np.asarray(center_enu, dtype=np.float64).reshape(3)
    points = np.asarray(points_enu, dtype=np.float64)[:, :3]
    directions = np.asarray(directions_enu, dtype=np.float64)[:, :3]
    return np.sum((points - center[None, :]) * directions, axis=1)


def _estimate_dom_point_camera_target_point_rays_enu(
    point_model: np.ndarray,
    point_targets_enu: np.ndarray,
    center_model: np.ndarray,
    transform: LocalModelTransform,
    config: Optional[GeoV3Config],
    *,
    min_points: int = 8,
    max_points: int = 256,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """用 point-ray 三角化（RANSAC）估计相机中心的 ENU 目标位置。

    每个 model 点相对相机中心 center_model 的方向经 transform.R 旋到 ENU，与该点的
    ENU 目标位置构成一条射线；理想相机中心应落在所有射线上。用 RANSAC 采样子集调用
    _solve_camera_center_from_point_rays_np 求候选中心，以 _point_ray_residuals_np 正交
    距离和正深度筛内点，取最优后重解并按 config 中的内点比/中位/分位/先验偏移门控。
    默认仅采纳 XY（Z 沿用当前中心，除非 use_z 开启）。返回 ((3,) ENU 目标, stats)，
    失败时第一项为 None，stats 记录原因与残差/偏移指标。
    """
    stats: Dict[str, Any] = {"accepted": False, "reason": "not_run", "source": "point_rays"}
    if transform is None:
        stats["reason"] = "no_transform"
        return None, stats
    points_model = np.asarray(point_model, dtype=np.float64)
    targets_enu = np.asarray(point_targets_enu, dtype=np.float64)
    center = np.asarray(center_model, dtype=np.float64).reshape(-1)
    if (points_model.ndim != 2 or targets_enu.ndim != 2 or points_model.shape != targets_enu.shape
            or points_model.shape[1] < 3 or center.size < 3):
        stats["reason"] = "bad_shape"
        return None, stats
    current_center_enu = _model_camera_center_to_enu_np(center[:3], transform)
    if current_center_enu is None:
        stats["reason"] = "no_current_center"
        return None, stats

    directions_model = points_model[:, :3] - center[:3][None, :]
    direction_norm = np.linalg.norm(directions_model, axis=1)
    finite = (
        np.isfinite(points_model[:, :3]).all(axis=1)
        & np.isfinite(targets_enu[:, :3]).all(axis=1)
        & np.isfinite(directions_model).all(axis=1)
        & (direction_norm > 1e-6)
    )
    points_model = points_model[finite, :3]
    targets_enu = targets_enu[finite, :3]
    directions_model = directions_model[finite]
    direction_norm = direction_norm[finite]
    stats["valid_points"] = int(len(points_model))
    min_points = max(4, int(min_points))
    if len(points_model) < min_points:
        stats["reason"] = "too_few_points"
        return None, stats

    max_points = int(max_points)
    if max_points > 0 and len(points_model) > max_points:
        rng = np.random.RandomState(int(seed))
        selected = rng.choice(len(points_model), max_points, replace=False)
        selected.sort()
        points_model = points_model[selected]
        targets_enu = targets_enu[selected]
        directions_model = directions_model[selected]
        direction_norm = direction_norm[selected]

    directions_model = directions_model / np.maximum(direction_norm[:, None], 1e-8)
    transform_rotation = transform.R.detach().cpu().float().numpy().astype(np.float64)
    directions_enu = directions_model @ transform_rotation.T
    directions_enu_norm = np.linalg.norm(directions_enu, axis=1)
    directions_enu = directions_enu / np.maximum(directions_enu_norm[:, None], 1e-8)

    ransac_iters = max(0, int(getattr(config, "dom_point_camera_teacher_point_ray_ransac_iters", 128)) if config is not None else 128)
    inlier_thresh = float(getattr(config, "dom_point_camera_teacher_point_ray_inlier_thresh_m", 6.0)) if config is not None else 6.0
    min_ratio = float(getattr(config, "dom_point_camera_teacher_point_ray_min_ratio", 0.25)) if config is not None else 0.25
    max_median = float(getattr(config, "dom_point_camera_teacher_point_ray_max_median_m", 4.0)) if config is not None else 4.0
    max_p90 = float(getattr(config, "dom_point_camera_teacher_point_ray_max_p90_m", 10.0)) if config is not None else 10.0
    max_prior_xy = float(getattr(config, "dom_point_camera_teacher_point_ray_max_prior_xy_m", 50.0)) if config is not None else 50.0
    max_prior_z = float(getattr(config, "dom_point_camera_teacher_point_ray_max_prior_z_m", 120.0)) if config is not None else 120.0
    use_estimated_z = bool(getattr(config, "dom_point_camera_teacher_point_ray_use_z", False)) if config is not None else False

    rng = np.random.RandomState(int(seed))
    sample_size = min(3, len(targets_enu))
    candidate_masks: List[np.ndarray] = [np.ones(len(targets_enu), dtype=bool)]
    for _iter_idx in range(ransac_iters):
        if len(targets_enu) <= sample_size:
            sample_mask = np.ones(len(targets_enu), dtype=bool)
        else:
            sample_indices = rng.choice(len(targets_enu), sample_size, replace=False)
            sample_mask = np.zeros(len(targets_enu), dtype=bool)
            sample_mask[sample_indices] = True
        candidate_masks.append(sample_mask)

    best_center: Optional[np.ndarray] = None
    best_inlier_mask: Optional[np.ndarray] = None
    best_inlier_count = -1
    best_median_residual = float("inf")
    for candidate_mask in candidate_masks:
        candidate_center = _solve_camera_center_from_point_rays_np(targets_enu, directions_enu, candidate_mask)
        if candidate_center is None:
            continue
        residuals = _point_ray_residuals_np(candidate_center, targets_enu, directions_enu)
        ray_depths = _point_ray_depths_np(candidate_center, targets_enu, directions_enu)
        inlier_mask = (residuals <= inlier_thresh) & (ray_depths > 0.0)
        inlier_count = int(inlier_mask.sum())
        median_residual = float(np.median(residuals[inlier_mask])) if inlier_count > 0 else float("inf")
        if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and median_residual < best_median_residual):
            best_inlier_count = inlier_count
            best_median_residual = median_residual
            best_center = candidate_center
            best_inlier_mask = inlier_mask

    if best_center is None or best_inlier_mask is None or int(best_inlier_mask.sum()) < min_points:
        stats["reason"] = "ransac_failed"
        return None, stats

    refined_center = _solve_camera_center_from_point_rays_np(targets_enu, directions_enu, best_inlier_mask)
    if refined_center is None:
        stats["reason"] = "refine_failed"
        return None, stats
    residuals_refined = _point_ray_residuals_np(refined_center, targets_enu, directions_enu)
    ray_depths_refined = _point_ray_depths_np(refined_center, targets_enu, directions_enu)
    inlier_mask_refined = (residuals_refined <= inlier_thresh) & (ray_depths_refined > 0.0)
    inlier_count = int(inlier_mask_refined.sum())
    inlier_ratio = float(inlier_count / max(len(targets_enu), 1))
    residuals_in = residuals_refined[inlier_mask_refined] if inlier_count >= min_points else residuals_refined
    median_residual = float(np.median(residuals_in)) if len(residuals_in) else float("inf")
    p90_residual = float(np.percentile(residuals_in, 90.0)) if len(residuals_in) else float("inf")
    prior_shift = refined_center - current_center_enu[:3]
    prior_xy = float(np.linalg.norm(prior_shift[:2]))
    prior_z = float(abs(prior_shift[2]))

    stats.update({
        "points": int(len(targets_enu)),
        "inliers": int(inlier_count),
        "inlier_ratio": float(inlier_ratio),
        "median_residual_m": float(median_residual),
        "p90_residual_m": float(p90_residual),
        "prior_xy_m": float(prior_xy),
        "prior_z_m": float(prior_z),
        "estimated_z_shift_m": float(prior_z),
        "inlier_thresh_m": float(inlier_thresh),
        "front_ratio": float(np.mean(ray_depths_refined > 0.0)) if len(ray_depths_refined) else 0.0,
    })
    if inlier_count < min_points:
        stats["reason"] = "too_few_inliers"
        return None, stats
    if inlier_ratio < min_ratio:
        stats["reason"] = "low_inlier_ratio"
        return None, stats
    if np.isfinite(max_median) and median_residual > max_median:
        stats["reason"] = "median_residual_gate"
        return None, stats
    if np.isfinite(max_p90) and p90_residual > max_p90:
        stats["reason"] = "p90_residual_gate"
        return None, stats
    if np.isfinite(max_prior_xy) and prior_xy > max_prior_xy:
        stats["reason"] = "prior_xy_gate"
        return None, stats
    if np.isfinite(max_prior_z) and prior_z > max_prior_z:
        stats["reason"] = "prior_z_gate"
        return None, stats

    target = refined_center.astype(np.float64)
    if not use_estimated_z:
        target[2] = float(current_center_enu[2])
    stats.update({
        "accepted": True,
        "reason": "ok",
        "z_used": bool(use_estimated_z),
        "xy_shift_m": float(prior_xy),
        "z_shift_m": float(abs(target[2] - current_center_enu[2])),
    })
    return target.astype(np.float32), stats


def _estimate_geo_camera_teacher_target_enu(
    corr: Dict[str, Any],
    center_model: np.ndarray,
    transform: Optional[LocalModelTransform],
    config: Optional[GeoV3Config],
    *,
    source: str,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """按指定 source 估计单帧相机中心的 ENU 教师目标位置。

    source 可为 frame_center（用渲染帧中心 XY，可叠加 AGL 推 Z）、ground_signal
    （地面信号）、fixedR_pnp（固定旋转 PnP 中心，需先过 _fixed_rotation_pnp_gate），
    或 "config"（由 config 解析实际来源）。center_model 经 transform 转 ENU 作当前中心，
    各来源只改写 XY（Z 视情况）。返回 ((3,) float32 ENU 目标, stats)，被门控/数据缺失时
    第一项为 None，stats 记录 reason 与 xy/z 位移。
    """
    stats: Dict[str, Any] = {"accepted": False, "reason": "not_run", "source": source}
    if not isinstance(corr, dict):
        stats["reason"] = "bad_corr"
        return None, stats
    current_enu = _model_camera_center_to_enu_np(center_model, transform)
    if current_enu is None:
        stats["reason"] = "no_current_center"
        return None, stats

    source = _dom_point_camera_teacher_source(config) if source == "config" else str(source)
    if source == "frame_center":
        if _geo_target_mode_from_config(config) == "dom_points":
            crop_info = corr.get("dom_crop_info") or {}
            render_pitch = _finite_float_optional(crop_info.get("render_pitch_deg")) if isinstance(crop_info, dict) else None
            max_pitch_deg = max(0.0, float(getattr(config, "dom_point_bootstrap_camera_recenter_max_pitch_deg", 12.0))) if config is not None else 12.0
            if render_pitch is None:
                stats["reason"] = "no_render_pitch"
                return None, stats
            if abs(float(render_pitch)) > max_pitch_deg:
                stats["reason"] = "pitch_gate"
                stats["pitch_deg"] = float(abs(float(render_pitch)))
                stats["max_pitch_deg"] = float(max_pitch_deg)
                return None, stats
        fce = corr.get("frame_center_enu")
        try:
            fce_arr = np.asarray(fce, dtype=np.float64).reshape(-1)
        except Exception:
            fce_arr = np.empty((0,), dtype=np.float64)
        if fce_arr.size < 2 or not np.isfinite(fce_arr[:2]).all():
            stats["reason"] = "no_frame_center"
            return None, stats
        target = current_enu.copy()
        target[:2] = fce_arr[:2]
        label = "frame_center_xy"
        if (
            _geo_target_mode_from_config(config) == "dom_points"
            and bool(getattr(config, "dom_point_bootstrap_agl_z_recenter_enable", True))
        ):
            ground_z, _ground_source = _ground_z_from_corr_for_ray(corr)
            agl_value = _dom_point_agl_from_corr(corr)
            if ground_z is not None and agl_value is not None:
                target[2] = float(ground_z) + float(agl_value)
                label = "frame_center_xy_agl_z"
    elif source == "ground_signal":
        target_raw, label = _geo_camera_target_enu(
            corr,
            target_mode="ground_signal",
            prefer_ground_signal=True,
            prefer_pnp=False,
            allow_raw_pnp=False,
            ground_signal_max_delta_xy_m=float(getattr(config, "ground_signal_max_delta_xy_m", 120.0)) if config is not None else 120.0,
            ground_signal_min_coverage=float(getattr(config, "ground_signal_min_coverage", 0.01)) if config is not None else 0.01,
        )
        if target_raw is None:
            stats["reason"] = str(label)
            return None, stats
        if str(label) != "ground_signal":
            stats["reason"] = f"no_ground_signal:{label}"
            return None, stats
        target_arr = np.asarray(target_raw, dtype=np.float64).reshape(-1)
        if target_arr.size < 2 or not np.isfinite(target_arr[:2]).all():
            stats["reason"] = "bad_ground_signal"
            return None, stats
        target = current_enu.copy()
        target[:2] = target_arr[:2]
        label = f"{label}_xy"
    elif source == "fixedR_pnp":
        ok_pnp, pnp_gate_stats = _fixed_rotation_pnp_gate(corr, config)
        if not ok_pnp:
            stats.update(pnp_gate_stats)
            return None, stats
        pnp_center = _fixed_rotation_pnp_center(corr, min_dim=2)
        if pnp_center is None:
            stats["reason"] = "no_fixedR_pnp"
            return None, stats
        pnp_arr = np.asarray(pnp_center, dtype=np.float64).reshape(-1)
        if pnp_arr.size < 2 or not np.isfinite(pnp_arr[:2]).all():
            stats["reason"] = "bad_fixedR_pnp"
            return None, stats
        target = current_enu.copy()
        target[:2] = pnp_arr[:2]
        label = "fixedR_pnp_xy"
    else:
        stats["reason"] = "unsupported_source"
        return None, stats

    diff = target - current_enu
    stats.update({
        "accepted": True,
        "reason": "ok",
        "source": label,
        "xy_shift_m": float(np.linalg.norm(diff[:2])),
        "z_shift_m": float(abs(diff[2])),
    })
    return target.astype(np.float32), stats


def _build_dom_point_camera_targets_from_geo_obs(
    geo_obs: Dict[int, Dict[str, Any]],
    frames: List[dict],
    pts3d_by_frame: Any,
    camera_centers_model: Any,
    transform: LocalModelTransform,
    config: GeoV3Config,
    seg_id: int,
    *,
    road_only: bool = False,
    label: str = "post_ttt",
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """为各帧批量构建相机中心的 ENU 教师目标（用于相机 PGO）。

    遍历 geo_obs，按 config 选定的 teacher_source 逐帧估计目标：非 point 类来源走
    _estimate_geo_camera_teacher_target_enu；dom_points/point_rays 来源则在查询像素处采样
    pts3d 后调 _estimate_dom_point_camera_target_*_enu。dom_points 模式下还会做语义校验与
    segment0 门控。pts3d_by_frame/camera_centers_model 支持 dict、list 或 tensor。返回
    ({帧号: (3,) float32 目标}, stats)；stats 汇总帧数、位移中位数与各拒绝原因。
    """
    enabled = bool(getattr(config, "dom_point_camera_teacher_enable", False)) if config is not None else False
    stats: Dict[str, Any] = {
        "enabled": bool(enabled),
        "label": str(label),
        "accepted": False,
        "frames": 0,
        "raw_points": 0,
        "valid_frames": 0,
    }
    if not enabled:
        stats["reason"] = "disabled"
        return {}, stats
    if not geo_obs or transform is None:
        stats["reason"] = "no_geo_obs"
        return {}, stats
    teacher_source = _dom_point_camera_teacher_source(config)
    dom_points_mode = _geo_target_mode_from_config(config) == "dom_points"
    stats["source"] = teacher_source
    if teacher_source == "none":
        stats["reason"] = "source_disabled"
        return {}, stats
    if (
        dom_points_mode
        and teacher_source == "frame_center"
        and int(seg_id) == 0
        and not bool(getattr(config, "dom_point_bootstrap_camera_recenter_segment0_enable", False))
    ):
        stats["reason"] = "segment0_disabled"
        return {}, stats
    min_points = int(getattr(config, "dom_point_camera_teacher_min_points", 8))
    max_points = int(getattr(config, "dom_point_camera_teacher_max_points_per_frame", 256))
    targets: Dict[int, np.ndarray] = {}
    xy_medians: List[float] = []
    z_medians: List[float] = []
    xy_shifts: List[float] = []
    z_shifts: List[float] = []
    reject_reasons: Dict[str, int] = {}
    for fi, corr in sorted((geo_obs or {}).items()):
        fi = int(fi)
        if corr is None:
            continue
        if isinstance(pts3d_by_frame, dict):
            pts3d_fi = pts3d_by_frame.get(fi)
        elif 0 <= fi < len(pts3d_by_frame):
            pts3d_fi = pts3d_by_frame[fi]
        else:
            pts3d_fi = None
        if pts3d_fi is None or fi < 0 or fi >= len(frames):
            continue
        if dom_points_mode and teacher_source == "frame_center":
            ok_sem, reason_sem, sem_info = _dom_point_observation_semantics(
                corr, config=config, stage="camera_pgo")
            if not ok_sem:
                reject_reasons[reason_sem] = int(reject_reasons.get(reason_sem, 0)) + 1
                if isinstance(corr, dict):
                    corr.setdefault("dom_point_observation", {})["camera_pgo"] = {
                        "valid": False,
                        "reason": reason_sem,
                        **sem_info,
                    }
                continue
            if isinstance(corr, dict):
                corr.setdefault("dom_point_observation", {})["camera_pgo"] = {
                    "valid": True,
                    "reason": reason_sem,
                    **sem_info,
                }
        if torch.is_tensor(camera_centers_model):
            if fi >= int(camera_centers_model.shape[0]):
                continue
            center_np = camera_centers_model[fi].detach().cpu().float().numpy()
        elif 0 <= fi < len(camera_centers_model):
            center_i = camera_centers_model[fi]
            center_np = (center_i.detach().cpu().float().numpy()
                         if torch.is_tensor(center_i) else np.asarray(center_i, dtype=np.float32))
        else:
            continue
        if teacher_source not in {"dom_points", "point_rays"}:
            target, target_stats = _estimate_geo_camera_teacher_target_enu(
                corr,
                center_np,
                transform,
                config,
                source=teacher_source,
            )
            stats["frames"] += 1
            if target is None:
                reason = str(target_stats.get("reason", "rejected"))
                reject_reasons[reason] = int(reject_reasons.get(reason, 0)) + 1
                continue
            targets[fi] = target.astype(np.float32)
            stats["valid_frames"] += 1
            xy_shifts.append(float(target_stats.get("xy_shift_m", 0.0)))
            z_shifts.append(float(target_stats.get("z_shift_m", 0.0)))
            continue
        if dom_points_mode:
            px, enu, _ = _select_dom_point_correspondences_road_then_nonbuilding(corr, min_points=min_points)
        else:
            px, enu = _select_dom_point_correspondences(corr, road_only=road_only)
        if px is None or enu is None:
            continue
        stats["frames"] += 1
        stats["raw_points"] += int(len(px))
        query_w = float(corr.get("query_W", frames[fi]["img"].shape[-1]))
        query_h = float(corr.get("query_H", frames[fi]["img"].shape[-2]))
        sampled = _sample_pts3d_at_query_pixels(
            pts3d_fi, px, query_w, query_h, detach=True)
        sampled_np = sampled.detach().cpu().float().numpy()
        if teacher_source == "point_rays":
            target, target_stats = _estimate_dom_point_camera_target_point_rays_enu(
                sampled_np,
                enu,
                center_np,
                transform,
                config,
                min_points=min_points,
                max_points=max_points,
                seed=101 + 17 * int(seg_id) + fi,
            )
        else:
            target, target_stats = _estimate_dom_point_camera_target_enu(
                sampled_np,
                enu,
                center_np,
                transform,
                min_points=min_points,
                max_points=max_points,
                seed=101 + 17 * int(seg_id) + fi,
            )
        if target is None:
            reason = str(target_stats.get("reason", "rejected"))
            reject_reasons[reason] = int(reject_reasons.get(reason, 0)) + 1
            continue
        targets[fi] = target.astype(np.float32)
        stats["valid_frames"] += 1
        xy_medians.append(float(target_stats.get("xy_median_m", target_stats.get("median_residual_m", 0.0))))
        z_medians.append(float(target_stats.get("z_median_m", target_stats.get("p90_residual_m", 0.0))))
        xy_shifts.append(float(target_stats.get("xy_shift_m", target_stats.get("prior_xy_m", 0.0))))
        z_shifts.append(float(target_stats.get("z_shift_m", target_stats.get("prior_z_m", 0.0))))
    if targets:
        stats.update({
            "accepted": True,
            "reason": "ok",
            "target_frames": int(len(targets)),
            "xy_median_m": float(np.median(np.asarray(xy_medians, dtype=np.float64))) if xy_medians else 0.0,
            "z_median_m": float(np.median(np.asarray(z_medians, dtype=np.float64))) if z_medians else 0.0,
            "xy_shift_median_m": float(np.median(np.asarray(xy_shifts, dtype=np.float64))) if xy_shifts else 0.0,
            "z_shift_median_m": float(np.median(np.asarray(z_shifts, dtype=np.float64))) if z_shifts else 0.0,
            "rejected": reject_reasons,
        })
        if teacher_source in {"dom_points", "point_rays"}:
            print(
                f"[GeoV3][PointCamera] seg {seg_id}: targets={len(targets)} "
                f"source={label}/{teacher_source} med_res={stats['xy_median_m']:.2f}m "
                f"p90_res={stats['z_median_m']:.2f}m med_shift={stats['xy_shift_median_m']:.2f}m "
                f"road_only={road_only}"
            )
        else:
            print(
                f"[GeoV3][PointCamera] seg {seg_id}: targets={len(targets)} "
                f"source={label}/{teacher_source} med_shift={stats['xy_shift_median_m']:.2f}m "
                f"med_z_shift={stats['z_shift_median_m']:.2f}m road_only={road_only}"
            )
    else:
        stats["reason"] = "no_valid_targets"
        stats["rejected"] = reject_reasons
        print(f"[GeoV3][PointCamera] seg {seg_id}: no camera targets source={label} rejected={reject_reasons}")
    return targets, stats


def _collect_dom_point_bootstrap_samples(
    geo_obs: Dict[int, Dict[str, Any]],
    frames: List[dict],
    per_frame_pts3d: List[Any],
    config: GeoV3Config,
    seg_id: int,
    frame_max_exclusive: Optional[int] = None,
    per_frame_pose: Optional[List[torch.Tensor]] = None,
    overlap_prior_transform: Optional[LocalModelTransform] = None,
    overlap_len: int = 0,
    filter_head_overlap: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """收集 bootstrap 所需的 model<->ENU 点对应样本（供 SRT 拟合 transform）。

    逐帧做语义校验，并可选地用 overlap_prior 一致性剔除头部重叠帧（filter_head_overlap，
    残差超 overlap_prior_consistency_max_xy_m 即丢弃）；road-then-nonbuilding 选点后在查询
    像素采样 pts3d，过滤非有限值，最多保留 dom_point_bootstrap_max_points 个（随 seg_id 设种子
    随机下采样）。frame_max_exclusive 可限定只用前若干帧。返回 (model 点 (N,3), ENU 点 (N,3),
    stats)，无有效点时前两项为 None。
    """
    stats: Dict[str, Any] = {
        "frames": 0,
        "raw_points": 0,
        "valid_points": 0,
        "stored_points": 0,
        "frame_max_exclusive": None if frame_max_exclusive is None else int(frame_max_exclusive),
        "semantic_rejected": {},
        "head_overlap_rejected": 0,
    }
    if not geo_obs or not per_frame_pts3d:
        stats["reason"] = "no_geo_obs"
        return None, None, stats
    model_chunks: List[np.ndarray] = []
    enu_chunks: List[np.ndarray] = []
    max_head_xy = float(getattr(config, "overlap_prior_consistency_max_xy_m", 25.0))
    for fi, corr in sorted((geo_obs or {}).items()):
        if corr is None or int(fi) < 0 or int(fi) >= len(per_frame_pts3d):
            continue
        if frame_max_exclusive is not None and int(fi) >= int(frame_max_exclusive):
            continue
        ok_sem, reason_sem, sem_info = _dom_point_observation_semantics(
            corr, config=config, stage="bootstrap")
        if not ok_sem:
            stats["semantic_rejected"][reason_sem] = stats["semantic_rejected"].get(reason_sem, 0) + 1
            if isinstance(corr, dict):
                corr.setdefault("dom_point_observation", {})["bootstrap"] = {"valid": False, "reason": reason_sem, **sem_info}
            continue
        if (
            filter_head_overlap
            and int(overlap_len) > 0
            and int(fi) < int(overlap_len)
            and overlap_prior_transform is not None
        ):
            residual = _dom_point_overlap_prior_xy_residual(
                corr, int(fi), per_frame_pose, overlap_prior_transform)
            if residual is None or not np.isfinite(residual) or residual > max_head_xy:
                stats["head_overlap_rejected"] += 1
                if isinstance(corr, dict):
                    corr.setdefault("dom_point_observation", {})["bootstrap"] = {
                        "valid": False,
                        "reason": "head_overlap_conflicts_overlap_prior",
                        "overlap_prior_xy_residual_m": None if residual is None else float(residual),
                        "max_xy_m": float(max_head_xy),
                    }
                continue
            if isinstance(corr, dict):
                corr.setdefault("dom_point_observation", {})["bootstrap_head_overlap"] = {
                    "valid": True,
                    "reason": "ok",
                    "overlap_prior_xy_residual_m": float(residual),
                    "max_xy_m": float(max_head_xy),
                }
        px, enu, point_source = _select_dom_point_correspondences_road_then_nonbuilding(corr, min_points=4)
        if px is None or enu is None:
            continue
        query_w = float(corr.get("query_W", frames[int(fi)]["img"].shape[-1]))
        query_h = float(corr.get("query_H", frames[int(fi)]["img"].shape[-2]))
        sampled = _sample_pts3d_at_query_pixels(
            per_frame_pts3d[int(fi)], px, query_w, query_h, detach=True)
        sampled_np = sampled.detach().cpu().float().numpy().astype(np.float64)
        finite = (
            np.isfinite(sampled_np).all(axis=1)
            & np.isfinite(enu).all(axis=1)
            & np.isfinite(px).all(axis=1)
        )
        if int(finite.sum()) < 4:
            continue
        model_chunks.append(sampled_np[finite])
        enu_chunks.append(enu[finite].astype(np.float64))
        stats["frames"] += 1
        stats["raw_points"] += int(len(px))
        stats["valid_points"] += int(finite.sum())
        stats[f"frames_{point_source}"] = int(stats.get(f"frames_{point_source}", 0)) + 1
        stats[f"points_{point_source}"] = int(stats.get(f"points_{point_source}", 0)) + int(finite.sum())
    if not model_chunks:
        stats["reason"] = "no_valid_points"
        return None, None, stats
    model_np = np.concatenate(model_chunks, axis=0)
    enu_np = np.concatenate(enu_chunks, axis=0)
    max_points = int(getattr(config, "dom_point_bootstrap_max_points", 4096))
    if max_points <= 0:
        stats["reason"] = "max_points_zero"
        return None, None, stats
    if len(model_np) > max_points:
        rng = np.random.RandomState(43 + int(seg_id))
        sel = rng.choice(len(model_np), max_points, replace=False)
        sel.sort()
        model_np = model_np[sel]
        enu_np = enu_np[sel]
    stats["stored_points"] = int(len(model_np))
    return model_np, enu_np, stats


def _finite_float_optional(value: Any) -> Optional[float]:
    """将 value 转为 float，转换失败或结果非有限（NaN/Inf）时返回 None。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _dom_point_agl_from_corr(corr: Dict[str, Any]) -> Optional[float]:
    """从 corr 中提取相机离地高度 AGL（米）。

    依次查 dom_crop_info 内的 pose_agl_m/estimated_agl_m/agl_m，再查 corr 顶层的
    estimated_agl/gt_agl，取首个落在 [20, 1000] 米合理区间内的有限值；都没有则返回 None。
    """
    if not isinstance(corr, dict):
        return None
    crop_info = corr.get("dom_crop_info") or {}
    if isinstance(crop_info, dict):
        for key in ("pose_agl_m", "estimated_agl_m", "agl_m"):
            agl_value = _finite_float_optional(crop_info.get(key))
            if agl_value is not None and 20.0 <= agl_value <= 1000.0:
                return agl_value
    for key in ("estimated_agl", "gt_agl"):
        agl_value = _finite_float_optional(corr.get(key))
        if agl_value is not None and 20.0 <= agl_value <= 1000.0:
            return agl_value
    return None


def _apply_dom_point_camera_bootstrap_recenter(
    transform: LocalModelTransform,
    geo_obs: Dict[int, Dict[str, Any]],
    per_frame_pose: List[torch.Tensor],
    config: GeoV3Config,
    seg_id: int,
) -> Tuple[LocalModelTransform, Dict[str, Any]]:
    """用 frame-center 教师对 bootstrap transform 做相机平移再中心化。

    遍历各帧（过语义校验、render_pitch 不超过 max_pitch_deg），计算渲染帧中心 XY 与预测
    相机中心 XY 的差 delta_xy；启用 AGL-Z 时再用 ground_z + AGL 算 delta_z。对差值取中位数、
    按 inlier 阈值剔除离群帧，乘不确定度权重并限幅（max_apply_xy/max_apply_z），把该平移加到
    transform.t 上返回新 transform。帧数不足或位移无效则原样返回。segment0 默认关闭。返回
    (transform, stats)。
    """
    stats: Dict[str, Any] = {
        "enabled": bool(getattr(config, "dom_point_bootstrap_camera_recenter_enable", True)),
        "accepted": False,
        "reason": "disabled",
    }
    if not stats["enabled"]:
        return transform, stats
    if int(seg_id) == 0 and not bool(getattr(config, "dom_point_bootstrap_camera_recenter_segment0_enable", False)):
        stats["reason"] = "segment0_disabled"
        return transform, stats
    if not geo_obs or not per_frame_pose:
        stats["reason"] = "no_geo_obs"
        return transform, stats

    max_pitch_deg = max(0.0, float(getattr(config, "dom_point_bootstrap_camera_recenter_max_pitch_deg", 12.0)))
    min_frames = max(1, int(getattr(config, "dom_point_bootstrap_camera_recenter_min_frames", 3)))
    xy_inlier_m = max(1e-6, float(getattr(config, "dom_point_bootstrap_camera_recenter_inlier_m", 15.0)))
    xy_weight_base = float(np.clip(float(getattr(config, "dom_point_bootstrap_camera_recenter_xy_weight", 1.0)), 0.0, 1.0))
    max_apply_xy = max(0.0, float(getattr(config, "dom_point_bootstrap_camera_recenter_max_apply_xy_m", 80.0)))
    z_enabled = bool(getattr(config, "dom_point_bootstrap_agl_z_recenter_enable", True))
    z_weight_base = float(np.clip(float(getattr(config, "dom_point_bootstrap_agl_z_weight", 1.0)), 0.0, 1.0))
    max_apply_z = max(0.0, float(getattr(config, "dom_point_bootstrap_agl_z_max_apply_m", 80.0)))
    z_inlier_m = max(1e-6, float(getattr(config, "dom_point_bootstrap_agl_z_inlier_m", 15.0)))

    entries: List[Dict[str, Any]] = []
    reject_reasons: Dict[str, int] = {}

    def _reject(reason: str) -> None:
        reject_reasons[reason] = int(reject_reasons.get(reason, 0)) + 1

    with torch.no_grad():
        for frame_index_raw, corr in sorted((geo_obs or {}).items()):
            try:
                frame_index = int(frame_index_raw)
            except (TypeError, ValueError):
                _reject("bad_frame_index")
                continue
            if corr is None or frame_index < 0 or frame_index >= len(per_frame_pose):
                _reject("bad_frame")
                continue
            ok_sem, reason_sem, sem_info = _dom_point_observation_semantics(
                corr, config=config, stage="camera_recenter")
            if not ok_sem:
                _reject(reason_sem)
                if isinstance(corr, dict):
                    corr.setdefault("dom_point_observation", {})["camera_recenter"] = {
                        "valid": False,
                        "reason": reason_sem,
                        **sem_info,
                    }
                continue
            crop_info = corr.get("dom_crop_info") or {}
            render_pitch = None
            if isinstance(crop_info, dict):
                render_pitch = _finite_float_optional(crop_info.get("render_pitch_deg"))
            if render_pitch is None:
                _reject("no_render_pitch")
                continue
            if abs(render_pitch) > max_pitch_deg:
                _reject("pitch_gate")
                continue
            target_xy = _dom_point_target_xy_from_corr(corr)
            if target_xy is None:
                _reject("no_frame_center_xy")
                continue
            center_model = pose_enc_to_camera_centers(per_frame_pose[frame_index])
            if center_model.ndim > 1:
                center_model = center_model[0]
            center_model_np = center_model[:3].detach().cpu().float().numpy()
            pred_center = _model_camera_center_to_enu_np(center_model_np, transform)
            if pred_center is None:
                _reject("bad_pred_center")
                continue
            delta_xy = target_xy[:2] - pred_center[:2]
            if not np.isfinite(delta_xy).all():
                _reject("bad_delta_xy")
                continue
            entry: Dict[str, Any] = {
                "frame_index": frame_index,
                "delta_xy": delta_xy.astype(np.float64),
                "pitch_deg": float(abs(render_pitch)),
            }
            if z_enabled:
                ground_z, _ground_source = _ground_z_from_corr_for_ray(corr)
                agl_value = _dom_point_agl_from_corr(corr)
                if ground_z is not None and agl_value is not None:
                    target_z = float(ground_z) + float(agl_value)
                    delta_z = target_z - float(pred_center[2])
                    if np.isfinite(delta_z):
                        entry["delta_z"] = float(delta_z)
                        entry["target_z"] = float(target_z)
                        entry["ground_z"] = float(ground_z)
                        entry["agl_m"] = float(agl_value)
            entries.append(entry)
            if isinstance(corr, dict):
                corr.setdefault("dom_point_observation", {})["camera_recenter"] = {
                    "valid": True,
                    "reason": "ok",
                    "pitch_deg": float(abs(render_pitch)),
                    **sem_info,
                }

    stats["frames"] = int(len(entries))
    stats["rejected"] = reject_reasons
    stats["max_pitch_deg"] = float(max_pitch_deg)
    if len(entries) < min_frames:
        stats["reason"] = "too_few_frames"
        return transform, stats

    delta_xy_array = np.stack([entry["delta_xy"] for entry in entries], axis=0).astype(np.float64)
    median_delta_xy = np.median(delta_xy_array, axis=0)
    xy_residuals = np.linalg.norm(delta_xy_array - median_delta_xy[None, :], axis=1)
    xy_inlier_mask = xy_residuals <= xy_inlier_m
    if int(xy_inlier_mask.sum()) < min_frames:
        stats.update({
            "reason": "too_few_xy_inliers",
            "xy_inliers": int(xy_inlier_mask.sum()),
            "xy_scatter_m": float(np.median(xy_residuals)) if len(xy_residuals) else 0.0,
        })
        return transform, stats

    inlier_delta_xy = delta_xy_array[xy_inlier_mask]
    median_delta_xy = np.median(inlier_delta_xy, axis=0)
    xy_scatter = float(np.median(np.linalg.norm(inlier_delta_xy - median_delta_xy[None, :], axis=1))) if len(inlier_delta_xy) else 0.0
    xy_uncertainty_weight = min(1.0, xy_inlier_m / max(xy_scatter, 1e-6))
    xy_weight = float(xy_weight_base * xy_uncertainty_weight)
    apply_delta_xy = median_delta_xy * xy_weight
    apply_norm = float(np.linalg.norm(apply_delta_xy))
    if max_apply_xy > 0.0 and apply_norm > max_apply_xy:
        apply_delta_xy = apply_delta_xy * (max_apply_xy / max(apply_norm, 1e-6))
        apply_norm = float(np.linalg.norm(apply_delta_xy))

    apply_delta_z = 0.0
    z_inliers = 0
    z_scatter = None
    if z_enabled and z_weight_base > 0.0:
        z_candidates = [
            float(entry["delta_z"])
            for entry, is_inlier in zip(entries, xy_inlier_mask.tolist())
            if is_inlier and "delta_z" in entry and np.isfinite(float(entry["delta_z"]))
        ]
        if len(z_candidates) >= min_frames:
            delta_z_array = np.asarray(z_candidates, dtype=np.float64)
            median_delta_z = float(np.median(delta_z_array))
            z_residuals = np.abs(delta_z_array - median_delta_z)
            z_inlier_mask = z_residuals <= z_inlier_m
            z_inliers = int(z_inlier_mask.sum())
            if z_inliers >= min_frames:
                inlier_delta_z = delta_z_array[z_inlier_mask]
                median_delta_z = float(np.median(inlier_delta_z))
                z_scatter = float(np.median(np.abs(inlier_delta_z - median_delta_z))) if len(inlier_delta_z) else 0.0
                z_uncertainty_weight = min(1.0, z_inlier_m / max(z_scatter, 1e-6))
                apply_delta_z = float(median_delta_z * z_weight_base * z_uncertainty_weight)
                if max_apply_z > 0.0 and abs(apply_delta_z) > max_apply_z:
                    apply_delta_z = float(np.sign(apply_delta_z) * max_apply_z)

    delta_np = np.array([float(apply_delta_xy[0]), float(apply_delta_xy[1]), float(apply_delta_z)], dtype=np.float32)
    if not np.isfinite(delta_np).all():
        stats["reason"] = "bad_apply_delta"
        return transform, stats
    if float(np.linalg.norm(delta_np)) <= 1e-6:
        stats["reason"] = "zero_delta"
        return transform, stats

    delta_tensor = torch.tensor(delta_np, device=transform.t.device, dtype=transform.t.dtype)
    recentered = LocalModelTransform(
        anchor_world_xyz=transform.anchor_world_xyz,
        R=transform.R,
        s=transform.s,
        t=transform.t + delta_tensor,
    )
    pitch_values = np.asarray([entry["pitch_deg"] for entry in entries], dtype=np.float64)
    stats.update({
        "accepted": True,
        "reason": "ok",
        "xy_inliers": int(xy_inlier_mask.sum()),
        "xy_weight": float(xy_weight),
        "xy_scatter_m": float(xy_scatter),
        "xy_delta_m": [float(apply_delta_xy[0]), float(apply_delta_xy[1])],
        "xy_delta_norm_m": float(apply_norm),
        "z_inliers": int(z_inliers),
        "z_delta_m": float(apply_delta_z),
        "z_scatter_m": None if z_scatter is None else float(z_scatter),
        "pitch_median_deg": float(np.median(pitch_values)) if len(pitch_values) else 0.0,
    })
    print(
        f"[GeoV3][CameraBootstrap] seg {seg_id}: frame_center_recenter "
        f"frames={len(entries)} xy_inliers={int(xy_inlier_mask.sum())}/{len(entries)} "
        f"pitch_med={stats['pitch_median_deg']:.1f}deg "
        f"dxy={apply_norm:.2f}m weight={xy_weight:.2f} "
        f"dz={apply_delta_z:+.2f}m z_inliers={z_inliers}"
    )
    return recentered, stats


def _estimate_dom_point_bootstrap_transform(
    geo_obs: Dict[int, Dict[str, Any]],
    frames: List[dict],
    per_frame_pose: List[torch.Tensor],
    per_frame_pts3d: List[Any],
    segment_input: SegmentInput,
    config: GeoV3Config,
    device: torch.device,
    dtype: torch.dtype,
    overlap_len: int = 0,
    overlap_prior_transform: Optional[LocalModelTransform] = None,
    overlap_prior_diag: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[LocalModelTransform], Dict[str, Any]]:
    """用 dom-point 对应估计 segment 的 bootstrap LocalModelTransform（Sim3）。

    先收集 model<->ENU 样本（segment>0 且有重叠时优先用头部重叠帧，否则全帧），样本不足则放弃；
    再用 _fit_ransac_srt_point_edges 的自由 RANSAC SRT 选出内点。由于地图对应主要位于
    地面平面，自由 3D 旋转存在倾角歧义；默认使用首帧 GT 或可靠 overlap full-extrinsic
    的朝向固定 R，再从地图内点重拟合尺度和平移。segment0 可选做 anchor 对齐
    （_anchor_aligned_t）使首帧落在 anchor。最后调 _apply_dom_point_camera_bootstrap_recenter
    做相机再中心化。map-only 的直接地图注册只由对应点的 RANSAC/残差质量决定；短
    overlap 轨迹的 Sim(3) 可用于裁图、连续传播和地图失败后的 fallback，但不能作为硬
    尺度/旋转先验否决直接地图证据。返回 (transform, stats)，被拒时第一项为 None，
    stats 含内点数/比例/尺度 s/样本来源。
    """
    stats: Dict[str, Any] = {
        "enabled": bool(getattr(config, "dom_point_bootstrap_enable", False)),
        "accepted": False,
        "reason": "disabled",
    }
    if not stats["enabled"]:
        return None, stats
    min_inliers = max(4, int(getattr(config, "dom_point_srt_min_inliers", 32)))
    point_model = None
    point_enu = None
    collect_stats: Dict[str, Any] = {}
    sample_source = "all_frames"
    # Prefer the explicitly computed current-segment overlap handoff.  The
    # metadata lookup is retained for compatibility with older callers.
    overlap_prior_for_bootstrap = overlap_prior_transform
    if overlap_prior_for_bootstrap is None and isinstance(segment_input.metadata, dict):
        overlap_prior_for_bootstrap = segment_input.metadata.get("overlap_prior_transform")
    prefer_overlap_samples = (
        int(segment_input.segment_id) > 0
        and int(overlap_len) > 0
        and bool(getattr(config, "dom_point_bootstrap_prefer_overlap_frames", True))
    )
    if prefer_overlap_samples:
        overlap_model, overlap_enu, overlap_stats = _collect_dom_point_bootstrap_samples(
            geo_obs,
            frames,
            per_frame_pts3d,
            config,
            int(segment_input.segment_id),
            frame_max_exclusive=int(overlap_len),
            per_frame_pose=per_frame_pose,
            overlap_prior_transform=overlap_prior_for_bootstrap,
            overlap_len=int(overlap_len),
            filter_head_overlap=bool(getattr(config, "dom_point_bootstrap_filter_head_overlap", True)),
        )
        stats["overlap_samples"] = overlap_stats
        if overlap_model is not None and overlap_enu is not None and int(overlap_model.shape[0]) >= min_inliers:
            point_model = overlap_model
            point_enu = overlap_enu
            collect_stats = overlap_stats
            sample_source = "head_overlap_dom_points"
        else:
            print(
                f"[GeoV3][PointBootstrap] seg {segment_input.segment_id}: "
                f"head-overlap samples unavailable; fallback to all frames "
                f"points={0 if overlap_model is None else int(overlap_model.shape[0])}"
            )
    if point_model is None or point_enu is None:
        point_model, point_enu, collect_stats = _collect_dom_point_bootstrap_samples(
            geo_obs,
            frames,
            per_frame_pts3d,
            config,
            int(segment_input.segment_id),
            per_frame_pose=per_frame_pose,
            overlap_prior_transform=overlap_prior_for_bootstrap,
            overlap_len=int(overlap_len),
            filter_head_overlap=bool(getattr(config, "dom_point_bootstrap_filter_head_overlap", True)),
        )
        sample_source = "all_frames"
    stats["samples"] = collect_stats
    stats["sample_source"] = sample_source
    if point_model is None or point_enu is None:
        stats["reason"] = collect_stats.get("reason", "no_points")
        return None, stats
    if int(point_model.shape[0]) < min_inliers:
        stats["reason"] = "too_few_points"
        return None, stats

    anchor_enu_np = _get_segment_anchor_enu(segment_input)
    if anchor_enu_np is None:
        anchor_enu_np = np.zeros(3, dtype=np.float32)
    anchor_t = torch.tensor(anchor_enu_np[:3].astype(np.float32), device=device, dtype=dtype)
    init_transform = build_local_model_transform(anchor_world_xyz=anchor_t, device=device, dtype=dtype)
    map_registration_only = bool(getattr(config, "map_anchor_only", False))
    fixed_rotation_refit = bool(getattr(
        config, "dom_point_bootstrap_fixed_rotation_refit_enable", False))
    pose_gate_reference = None
    pose_gate_reference_source = "full_path_permissive"
    disable_pose_gate = True
    if map_registration_only:
        overlap_gate_ok, overlap_gate_diag = _overlap_prior_reliability(
            segment_input=segment_input,
            overlap_prior_transform=overlap_prior_for_bootstrap,
            overlap_prior_diag=overlap_prior_diag,
            config=config,
        )
        stats["pose_gate_prior"] = overlap_gate_diag
        if overlap_gate_ok and overlap_prior_for_bootstrap is not None:
            # A short overlap can have a tiny self-residual while its scale is
            # poorly identifiable.  Keep it for crop/fallback diagnostics, but
            # let independent map correspondences estimate the absolute Sim(3).
            overlap_gate_diag["direct_registration_role"] = "diagnostic_only"
            pose_gate_reference_source = "map_registration_direct_geometry"
        elif int(segment_input.segment_id) == 0:
            # Segment 0 has a known position/orientation but no independent
            # absolute model-scale prior.  Geometry must estimate free scale.
            pose_gate_reference_source = "seg0_no_absolute_scale_prior"
        else:
            pose_gate_reference_source = (
                "no_reliable_overlap_prior:"
                f"{overlap_gate_diag.get('primary_reason', overlap_gate_diag.get('reason', 'unknown'))}"
            )
    transform, mask, srt_stats = _fit_ransac_srt_point_edges(
        point_model,
        point_enu,
        init_transform,
        config,
        int(segment_input.segment_id),
        min_inliers_override=min_inliers,
        # Direct map registration is geometry-only.  Overlap handoff remains a
        # crop/fallback source, not a hard prior over an independently fitted T_k.
        disable_pose_gate=bool(disable_pose_gate),
        pose_gate_reference=pose_gate_reference,
        pose_gate_reference_source=pose_gate_reference_source,
    )
    if transform is None or mask is None:
        stats["srt"] = srt_stats
        stats["reason"] = f"srt_{srt_stats.get('reason', 'rejected')}"
        print(
            f"[GeoV3][PointBootstrap] seg {segment_input.segment_id}: reject "
            f"reason={stats['reason']} points={int(point_model.shape[0])} "
            f"pose_ref={srt_stats.get('pose_gate_reference')} "
            f"pose_reject={srt_stats.get('pose_gate_rejected_candidates', 0)}/"
            f"{srt_stats.get('candidate_fits', 0)}"
        )
        return None, stats

    if fixed_rotation_refit:
        free_srt_stats = dict(srt_stats)
        safe_rotation, safe_rotation_diag = _safe_pose_orientation_rotation_for_segment(
            segment_input=segment_input,
            per_frame_pose=per_frame_pose,
            overlap_prior_transform=overlap_prior_for_bootstrap,
            overlap_prior_diag=overlap_prior_diag,
        )
        stats["orientation_source"] = safe_rotation_diag
        if safe_rotation is None:
            stats["srt"] = free_srt_stats
            stats["reason"] = (
                "fixed_rotation_unavailable:"
                f"{safe_rotation_diag.get('reason', 'unknown')}"
            )
            print(
                f"[GeoV3][PointBootstrap] seg {segment_input.segment_id}: reject "
                f"reason={stats['reason']}"
            )
            return None, stats
        fixed_transform, fixed_mask, fixed_stats = _refit_point_transform_with_fixed_rotation(
            point_model=point_model,
            point_targets_enu=point_enu,
            initial_inlier_mask=mask,
            fixed_rotation=safe_rotation,
            transform_init=init_transform,
            config=config,
        )
        fixed_stats["rotation_source"] = safe_rotation_diag.get("source", "unknown")
        fixed_stats["free_scale"] = free_srt_stats.get("scale")
        fixed_stats["free_median_xy_m"] = free_srt_stats.get("median_xy_m")
        stats["fixed_rotation_refit"] = fixed_stats
        if fixed_transform is None or fixed_mask is None:
            stats["srt"] = free_srt_stats
            stats["reason"] = f"fixed_rotation_refit:{fixed_stats.get('reason', 'failed')}"
            print(
                f"[GeoV3][PointBootstrap] seg {segment_input.segment_id}: reject "
                f"reason={stats['reason']}"
            )
            return None, stats
        transform = fixed_transform
        mask = fixed_mask
        srt_stats = {
            **fixed_stats,
            "pose_gate_reference": safe_rotation_diag.get("source", "unknown"),
            "free_fit": free_srt_stats,
        }
        print(
            f"[GeoV3][PointBootstrap-FixedR] seg {segment_input.segment_id}: "
            f"source={safe_rotation_diag.get('source', 'unknown')} "
            f"s={float(transform.s.detach().cpu()):.3f} "
            f"free_s={float(free_srt_stats.get('scale', float('nan'))):.3f} "
            f"med_xy={float(fixed_stats.get('median_xy_m', float('nan'))):.2f}m"
        )
    else:
        stats["fixed_rotation_refit"] = {
            "enabled": False,
            "accepted": False,
            "reason": "disabled",
        }
    stats["srt"] = srt_stats

    stats["head_overlap_bootstrap"] = sample_source == "head_overlap_dom_points"
    stats["overlap_scale_controlled"] = stats["head_overlap_bootstrap"]

    if int(segment_input.segment_id) == 0 and bool(getattr(config, "dom_point_bootstrap_anchor_align", True)):
        try:
            with torch.no_grad():
                mc0 = pose_enc_to_camera_centers(per_frame_pose[0])
                if mc0.ndim > 1:
                    mc0 = mc0[0]
                model0 = mc0[:3].detach().cpu().float().numpy().astype(np.float32)
            R_np = transform.R.detach().cpu().float().numpy().astype(np.float32)
            t_np = _anchor_aligned_t(
                anchor_enu_np,
                model0,
                float(transform.s.detach().cpu().float()),
                R_np,
            )
            transform = LocalModelTransform(
                anchor_world_xyz=transform.anchor_world_xyz,
                R=transform.R,
                s=transform.s,
                t=torch.tensor(t_np, device=device, dtype=dtype),
            )
            stats["anchor_aligned"] = True
        except Exception as exc:
            stats["anchor_align_warning"] = str(exc)

    recenter_after_fixed_refit = bool(getattr(
        config,
        "dom_point_bootstrap_recenter_after_fixed_rotation_refit_enable",
        False,
    ))
    if fixed_rotation_refit and not recenter_after_fixed_refit:
        camera_bootstrap_stats = {
            "enabled": bool(getattr(config, "dom_point_bootstrap_camera_recenter_enable", True)),
            "accepted": False,
            "reason": "fixed_rotation_refit_preserves_map_translation",
        }
    else:
        transform, camera_bootstrap_stats = _apply_dom_point_camera_bootstrap_recenter(
            transform,
            geo_obs,
            per_frame_pose,
            config,
            int(segment_input.segment_id),
        )
    stats["camera_bootstrap"] = camera_bootstrap_stats

    stats.update({
        "accepted": True,
        "reason": "ok",
        "inliers": int(mask.sum()),
        "inlier_ratio": float(mask.mean()) if len(mask) else 0.0,
        "s": float(transform.s.detach().cpu()),
    })
    if map_registration_only and bool(getattr(config, "dom_point_quality_gate_enable", True)):
        registration_quality = _dom_point_bootstrap_quality_stats(
            stats,
            config,
            require_road_support=False,
        )
        stats["registration_quality_gate"] = registration_quality
        if not bool(registration_quality.get("point_good", False)):
            stats["accepted"] = False
            stats["reason"] = f"quality_gate:{registration_quality.get('reason', 'failed')}"
            print(
                f"[GeoV3][PointBootstrap] seg {segment_input.segment_id}: reject "
                f"reason={stats['reason']} "
                f"ratio={registration_quality.get('inlier_ratio')} "
                f"med_xy={registration_quality.get('median_xy_m')}"
            )
            return None, stats
    print(
        f"[GeoV3][PointBootstrap] seg {segment_input.segment_id}: accept "
        f"points={int(point_model.shape[0])} "
        f"inliers={int(mask.sum())}/{int(len(mask))} "
        f"ratio={float(mask.mean()) if len(mask) else 0.0:.2f} "
        f"s={float(transform.s.detach().cpu()):.3f} "
        f"med_xy={float(srt_stats.get('median_xy_m', 0.0)):.2f}m "
        f"pose_ref={srt_stats.get('pose_gate_reference')} "
        f"source={sample_source}"
    )
    return transform, stats


def _build_dom_point_ttt_teacher(
    geo_obs: Dict[int, Dict[str, Any]],
    frames: List[dict],
    per_frame_pose: List[torch.Tensor],
    per_frame_pts3d: List[torch.Tensor],
    transform: LocalModelTransform,
    config: GeoV3Config,
    segment_input: SegmentInput,
    device: torch.device,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """为 TTT（test-time training）构建 dom-point 点教师与位姿教师缓存。

    逐帧采样 model<->ENU<->像素 对应，按 max_points 下采样。点教师可经 SRT 拟合（use_srt_teacher）
    或原始稀疏深度门控（raw_gate），得到内点对应；位姿教师按 pose_teacher_source 取 srt/transform，
    或经相机教师（geo 目标 / dom_point / point_rays，内含嵌套 _build_pose_cache_from_* 构建）给出各帧
    相机中心 ENU 目标。点教师与位姿教师可独立成立。返回 (point_cache, pose_cache, stats)：point_cache
    按帧存像素+ENU 目标，pose_cache 按帧存相机中心 ENU 目标。
    """
    stats: Dict[str, Any] = {
        "enabled": bool(getattr(config, "dom_point_ttt_enable", False)),
        "accepted": False,
        "reason": "disabled",
    }
    if not stats["enabled"]:
        return {}, {}, stats
    if not geo_obs or not per_frame_pts3d:
        stats["reason"] = "no_geo_obs"
        return {}, {}, stats

    dom_points_mode = _geo_target_mode_from_config(config) == "dom_points"
    road_only = bool(getattr(config, "dom_point_ttt_road_only", True))
    max_points = int(getattr(config, "dom_point_ttt_max_points", 2048))
    min_inliers = max(4, int(getattr(config, "dom_point_ttt_min_inliers", 64)))
    if max_points <= 0:
        stats["reason"] = "max_points_zero"
        return {}, {}, stats

    model_chunks: List[np.ndarray] = []
    enu_chunks: List[np.ndarray] = []
    pixel_chunks: List[np.ndarray] = []
    frame_chunks: List[np.ndarray] = []
    query_wh_by_frame: Dict[int, Tuple[float, float]] = {}
    semantic_rejected: Dict[str, int] = {}
    for fi, corr in sorted((geo_obs or {}).items()):
        fi = int(fi)
        if corr is None or fi < 0 or fi >= len(per_frame_pts3d):
            continue
        ok_sem, reason_sem, sem_info = _dom_point_observation_semantics(
            corr, config=config, stage="ttt")
        if not ok_sem:
            semantic_rejected[reason_sem] = semantic_rejected.get(reason_sem, 0) + 1
            if isinstance(corr, dict):
                corr.setdefault("dom_point_observation", {})["ttt"] = {"valid": False, "reason": reason_sem, **sem_info}
            continue
        if dom_points_mode:
            px, enu, _ = _select_dom_point_correspondences_road_then_nonbuilding(corr, min_points=4)
        else:
            px, enu = _select_dom_point_correspondences(corr, road_only=road_only)
        if px is None or enu is None:
            continue
        query_w = float(corr.get("query_W", frames[fi]["img"].shape[-1]))
        query_h = float(corr.get("query_H", frames[fi]["img"].shape[-2]))
        query_wh_by_frame[int(fi)] = (query_w, query_h)
        sampled = _sample_pts3d_at_query_pixels(
            per_frame_pts3d[fi], px, query_w, query_h, detach=True)
        sampled_np = sampled.detach().cpu().float().numpy().astype(np.float64)
        finite = (
            np.isfinite(sampled_np).all(axis=1)
            & np.isfinite(enu).all(axis=1)
            & np.isfinite(px).all(axis=1)
        )
        if int(finite.sum()) < 4:
            continue
        model_chunks.append(sampled_np[finite])
        enu_chunks.append(enu[finite])
        pixel_chunks.append(px[finite])
        frame_chunks.append(np.full(int(finite.sum()), int(fi), dtype=np.int32))
        if isinstance(corr, dict):
            corr.setdefault("dom_point_observation", {})["ttt"] = {"valid": True, "reason": "ok", **sem_info}

    point_teacher_rejected = False
    point_reject_reason = None
    if not model_chunks:
        stats["reason"] = "no_valid_points"
        stats["semantic_rejected"] = semantic_rejected
        model_np = np.empty((0, 3), dtype=np.float64)
        enu_np = np.empty((0, 3), dtype=np.float64)
        pixels_np = np.empty((0, 2), dtype=np.float64)
        frame_ids = np.empty((0,), dtype=np.int32)
        stats["raw_points"] = 0
        stats["raw_frames"] = 0
        stats["sampled_points"] = 0
        point_teacher_rejected = True
        point_reject_reason = stats["reason"]
    else:
        model_np = np.concatenate(model_chunks, axis=0)
        enu_np = np.concatenate(enu_chunks, axis=0)
        pixels_np = np.concatenate(pixel_chunks, axis=0)
        frame_ids = np.concatenate(frame_chunks, axis=0)
        stats["raw_points"] = int(len(model_np))
        stats["raw_frames"] = int(len(set(frame_ids.tolist())))
        stats["semantic_rejected"] = semantic_rejected
        if len(model_np) > max_points:
            rng = np.random.RandomState(31 + int(segment_input.segment_id))
            sel = rng.choice(len(model_np), max_points, replace=False)
            sel.sort()
            model_np = model_np[sel]
            enu_np = enu_np[sel]
            pixels_np = pixels_np[sel]
            frame_ids = frame_ids[sel]
        stats["sampled_points"] = int(len(model_np))
        if len(model_np) < min_inliers:
            stats["reason"] = "too_few_points"
            point_teacher_rejected = True
            point_reject_reason = stats["reason"]
    raw_model_np = model_np
    raw_enu_np = enu_np
    raw_pixels_np = pixels_np
    raw_frame_ids = frame_ids

    use_srt_teacher = bool(getattr(config, "dom_point_ttt_use_srt_teacher", True))
    pose_teacher_source = str(getattr(config, "dom_point_ttt_pose_teacher_source", "srt") or "srt").strip().lower()
    if pose_teacher_source in {"bootstrap", "bootstrap_transform", "current", "current_transform"}:
        pose_teacher_source = "transform"
    elif pose_teacher_source in {"off", "none", "disable", "disabled"}:
        pose_teacher_source = "none"
    elif pose_teacher_source not in {"srt", "transform"}:
        pose_teacher_source = "srt"
    stats["pose_teacher_source"] = pose_teacher_source

    if (
        not use_srt_teacher
        and bool(getattr(config, "dom_point_ttt_raw_gate_enable", True))
        and transform is not None
    ):
        raw_gate_stats: Dict[str, Any] = {"enabled": True, "accepted": False}
        try:
            with torch.no_grad():
                pts_t = torch.tensor(
                    raw_model_np.astype(np.float32),
                    device=transform.s.device,
                    dtype=transform.s.dtype,
                )
                pred_enu = transform.model_to_world(pts_t).detach().cpu().float().numpy().astype(np.float64)
            diff = pred_enu[:, :3] - raw_enu_np[:, :3]
            finite_raw = np.isfinite(diff).all(axis=1)
            if int(finite_raw.sum()) < min_inliers:
                raw_gate_stats.update({"reason": "too_few_finite", "finite": int(finite_raw.sum())})
            else:
                xy_all = np.linalg.norm(diff[finite_raw, :2], axis=1)
                z_all = np.abs(diff[finite_raw, 2])
                med_xy = float(np.median(xy_all))
                p90_xy = float(np.percentile(xy_all, 90.0))
                med_z = float(np.median(z_all))
                max_med_xy = float(getattr(config, "dom_point_ttt_raw_max_median_xy_m", 30.0))
                max_p90_xy = float(getattr(config, "dom_point_ttt_raw_max_p90_xy_m", 80.0))
                min_ratio_raw = float(getattr(config, "dom_point_ttt_raw_min_inlier_ratio", 0.25))
                finite_indices = np.flatnonzero(finite_raw)
                inlier_local = xy_all <= max_med_xy
                inlier_indices = finite_indices[inlier_local]
                inlier_ratio = float(len(inlier_indices) / max(int(finite_raw.sum()), 1))
                accepted = len(inlier_indices) >= min_inliers and inlier_ratio >= min_ratio_raw
                raw_gate_stats.update({
                    "accepted": bool(accepted),
                    "reason": "ok" if accepted else "too_few_initial_inliers",
                    "points": int(finite_raw.sum()),
                    "inliers": int(len(inlier_indices)),
                    "inlier_ratio": float(inlier_ratio),
                    "median_xy_m": med_xy,
                    "p90_xy_m": p90_xy,
                    "median_z_m": med_z,
                    "max_median_xy_m": float(max_med_xy),
                    "max_p90_xy_m": float(max_p90_xy),
                    "min_inlier_ratio": float(min_ratio_raw),
                })
                if accepted:
                    raw_model_np = raw_model_np[inlier_indices]
                    raw_enu_np = raw_enu_np[inlier_indices]
                    raw_pixels_np = raw_pixels_np[inlier_indices]
                    raw_frame_ids = raw_frame_ids[inlier_indices]
                    model_np = raw_model_np
                    enu_np = raw_enu_np
                    pixels_np = raw_pixels_np
                    frame_ids = raw_frame_ids
                    stats["sampled_points_after_raw_gate"] = int(len(model_np))
        except Exception as exc:
            raw_gate_stats.update({"reason": "error", "error": str(exc)})
        stats["raw_gate"] = raw_gate_stats
        if not bool(raw_gate_stats.get("accepted", False)):
            stats["reason"] = f"raw_gate_{raw_gate_stats.get('reason', 'rejected')}"
            print(
                f"[GeoV3][PointTTT] seg {segment_input.segment_id}: reject raw sparse-depth teacher "
                f"reason={raw_gate_stats.get('reason')} "
                f"med_xy={float(raw_gate_stats.get('median_xy_m', 0.0)):.2f}m "
                f"p90_xy={float(raw_gate_stats.get('p90_xy_m', 0.0)):.2f}m "
                f"inliers={int(raw_gate_stats.get('inliers', 0))}/{int(raw_gate_stats.get('points', 0))}"
            )
            point_teacher_rejected = True
            point_reject_reason = stats["reason"]

    camera_teacher_enabled = bool(getattr(config, "dom_point_camera_teacher_enable", False))
    camera_teacher_source = _dom_point_ttt_camera_teacher_source(config)
    stats["camera_teacher_source"] = camera_teacher_source
    pose_cache: Dict[int, Dict[str, Any]] = {}
    pose_source_label = "none"

    def _build_pose_cache_from_geo_targets() -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
        cache: Dict[int, Dict[str, Any]] = {}
        xy_shifts: List[float] = []
        z_shifts: List[float] = []
        source_labels: List[str] = []
        with torch.no_grad():
            for fi, corr in sorted((geo_obs or {}).items()):
                fi = int(fi)
                if corr is None or fi < 0 or fi >= len(per_frame_pose):
                    continue
                center = pose_enc_to_camera_centers(per_frame_pose[fi])
                if center.ndim > 1:
                    center = center[0]
                target_enu_np, cam_stats = _estimate_geo_camera_teacher_target_enu(
                    corr,
                    center[:3].detach().cpu().float().numpy(),
                    transform,
                    config,
                    source=camera_teacher_source,
                )
                if target_enu_np is None:
                    continue
                source_label = str(cam_stats.get("source", camera_teacher_source))
                cache[fi] = {
                    "target_enu": torch.tensor(target_enu_np, device=device, dtype=torch.float32),
                    "source": source_label,
                }
                source_labels.append(source_label)
                xy_shifts.append(float(cam_stats.get("xy_shift_m", 0.0)))
                z_shifts.append(float(cam_stats.get("z_shift_m", 0.0)))
        if not cache:
            return {}, {"accepted": False, "reason": "no_geo_camera_targets", "source": camera_teacher_source}
        return cache, {
            "accepted": True,
            "reason": "ok",
            "source": source_labels[0] if source_labels else camera_teacher_source,
            "frames": int(len(cache)),
            "xy_shift_median_m": float(np.median(np.asarray(xy_shifts, dtype=np.float64))) if xy_shifts else 0.0,
            "z_shift_median_m": float(np.median(np.asarray(z_shifts, dtype=np.float64))) if z_shifts else 0.0,
        }

    def _build_pose_cache_from_point_arrays(
        point_model_np: np.ndarray,
        point_enu_np: np.ndarray,
        point_frame_ids: np.ndarray,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
        cache: Dict[int, Dict[str, Any]] = {}
        cam_xy_meds: List[float] = []
        cam_z_meds: List[float] = []
        reject_reasons: Dict[str, int] = {}
        min_cam_points = int(getattr(config, "dom_point_camera_teacher_min_points", 8))
        max_cam_points = int(getattr(config, "dom_point_camera_teacher_max_points_per_frame", 256))
        if point_model_np is None or point_enu_np is None or point_frame_ids is None:
            return {}, {"accepted": False, "reason": "no_point_arrays", "source": camera_teacher_source}
        if len(point_model_np) == 0:
            return {}, {"accepted": False, "reason": "empty_point_arrays", "source": camera_teacher_source}
        with torch.no_grad():
            for fi in sorted(set(point_frame_ids.tolist())):
                fi = int(fi)
                if fi < 0 or fi >= len(per_frame_pose):
                    continue
                idx = np.where(point_frame_ids == fi)[0]
                center = pose_enc_to_camera_centers(per_frame_pose[fi])
                if center.ndim > 1:
                    center = center[0]
                center_np = center[:3].detach().cpu().float().numpy()
                if camera_teacher_source == "point_rays":
                    target_enu_np, cam_stats = _estimate_dom_point_camera_target_point_rays_enu(
                        point_model_np[idx],
                        point_enu_np[idx],
                        center_np,
                        transform,
                        config,
                        min_points=min_cam_points,
                        max_points=max_cam_points,
                        seed=71 + 19 * int(segment_input.segment_id) + fi,
                    )
                else:
                    target_enu_np, cam_stats = _estimate_dom_point_camera_target_enu(
                        point_model_np[idx],
                        point_enu_np[idx],
                        center_np,
                        transform,
                        min_points=min_cam_points,
                        max_points=max_cam_points,
                        seed=71 + 19 * int(segment_input.segment_id) + fi,
                    )
                if target_enu_np is None:
                    reason = str(cam_stats.get("reason", "rejected"))
                    reject_reasons[reason] = int(reject_reasons.get(reason, 0)) + 1
                    continue
                cache[fi] = {
                    "target_enu": torch.tensor(target_enu_np, device=device, dtype=torch.float32),
                    "source": "dom_point_point_rays" if camera_teacher_source == "point_rays" else "dom_point_camera",
                }
                cam_xy_meds.append(float(cam_stats.get("xy_median_m", cam_stats.get("median_residual_m", 0.0))))
                cam_z_meds.append(float(cam_stats.get("z_median_m", cam_stats.get("p90_residual_m", 0.0))))
        if not cache:
            return {}, {
                "accepted": False,
                "reason": "no_valid_point_camera_targets",
                "source": camera_teacher_source,
                "rejected": reject_reasons,
            }
        return cache, {
            "accepted": True,
            "reason": "ok",
            "source": "dom_point_point_rays" if camera_teacher_source == "point_rays" else "dom_point_camera",
            "frames": int(len(cache)),
            "xy_median_m": float(np.median(np.asarray(cam_xy_meds, dtype=np.float64))) if cam_xy_meds else 0.0,
            "z_median_m": float(np.median(np.asarray(cam_z_meds, dtype=np.float64))) if cam_z_meds else 0.0,
            "rejected": reject_reasons,
        }

    if camera_teacher_enabled and camera_teacher_source not in {"none", "dom_points", "point_rays"}:
        pose_cache, camera_stats = _build_pose_cache_from_geo_targets()
        stats["camera_teacher"] = camera_stats
        if pose_cache:
            pose_source_label = str(camera_stats.get("source", camera_teacher_source))

    teacher_transform = None
    teacher_mask = np.ones(len(model_np), dtype=bool)
    if use_srt_teacher and not point_teacher_rejected:
        teacher_transform, srt_mask, srt_stats = _fit_ransac_srt_point_edges(
            model_np, enu_np, transform, config, int(segment_input.segment_id),
            min_inliers_override=min_inliers)
        stats["srt"] = srt_stats
        if teacher_transform is None or srt_mask is None:
            stats["reason"] = f"srt_{srt_stats.get('reason', 'rejected')}"
            print(
                f"[GeoV3][PointTTT] seg {segment_input.segment_id}: reject teacher "
                f"reason={stats['reason']} points={len(model_np)}")
            point_teacher_rejected = True
            point_reject_reason = stats["reason"]
            teacher_mask = np.zeros(len(model_np), dtype=bool)
        else:
            teacher_mask = np.asarray(srt_mask, dtype=bool)
            inliers = int(teacher_mask.sum())
            ratio = float(teacher_mask.mean()) if len(teacher_mask) else 0.0
            median_xy = float(srt_stats.get("median_xy_m", np.inf))
            min_ratio = float(getattr(config, "dom_point_ttt_min_ratio", 0.25))
            max_median_xy = float(getattr(config, "dom_point_ttt_max_median_xy_m", 3.0))
            if (inliers < min_inliers or ratio < min_ratio
                    or (np.isfinite(max_median_xy) and median_xy > max_median_xy)):
                stats.update({
                    "reason": "teacher_gate_failed",
                    "srt_inliers": int(inliers),
                    "srt_inlier_ratio": float(ratio),
                    "srt_median_xy_m": float(median_xy),
                })
                print(
                    f"[GeoV3][PointTTT] seg {segment_input.segment_id}: reject teacher gate "
                    f"inliers={inliers}/{len(model_np)} ratio={ratio:.2f} med_xy={median_xy:.2f}m")
                point_teacher_rejected = True
                point_reject_reason = "teacher_gate_failed"
                teacher_mask = np.zeros(len(model_np), dtype=bool)

    if camera_teacher_enabled and camera_teacher_source in {"dom_points", "point_rays"} and not pose_cache:
        pose_points_model = raw_model_np if point_teacher_rejected else raw_model_np[teacher_mask]
        pose_points_enu = raw_enu_np if point_teacher_rejected else raw_enu_np[teacher_mask]
        pose_points_frame_ids = raw_frame_ids if point_teacher_rejected else raw_frame_ids[teacher_mask]
        pose_cache, camera_stats = _build_pose_cache_from_point_arrays(
            pose_points_model,
            pose_points_enu,
            pose_points_frame_ids,
        )
        stats["camera_teacher"] = camera_stats
        if pose_cache:
            pose_source_label = str(camera_stats.get("source", "dom_point_camera"))

    if point_teacher_rejected and not pose_cache:
        return {}, {}, stats

    if point_teacher_rejected:
        model_np = raw_model_np[:0]
        enu_np = raw_enu_np[:0]
        pixels_np = raw_pixels_np[:0]
        frame_ids = raw_frame_ids[:0]
    else:
        model_np = raw_model_np[teacher_mask]
        enu_np = raw_enu_np[teacher_mask]
        pixels_np = raw_pixels_np[teacher_mask]
        frame_ids = raw_frame_ids[teacher_mask]

    point_cache: Dict[int, Dict[str, Any]] = {}
    for fi in sorted(set(frame_ids.tolist())):
        idx = np.where(frame_ids == int(fi))[0]
        if len(idx) < 4:
            continue
        point_cache[int(fi)] = {
            "pixels": pixels_np[idx].astype(np.float32),
            "target_enu": torch.tensor(enu_np[idx].astype(np.float32), device=device),
            "query_W": float(query_wh_by_frame.get(int(fi), (frames[int(fi)]["img"].shape[-1], frames[int(fi)]["img"].shape[-2]))[0]),
            "query_H": float(query_wh_by_frame.get(int(fi), (frames[int(fi)]["img"].shape[-1], frames[int(fi)]["img"].shape[-2]))[1]),
            "source": "srt_teacher" if use_srt_teacher else "raw_dom",
        }

    if not pose_cache:
        pose_teacher_transform = None
        if pose_teacher_source == "transform":
            pose_teacher_transform = transform
            pose_source_label = "transform"
        elif pose_teacher_source == "srt":
            pose_teacher_transform = teacher_transform
            pose_source_label = "srt_teacher"
        if pose_teacher_transform is not None and bool(getattr(config, "dom_point_ttt_pose_all_frames", True)):
            pose_fis = range(len(per_frame_pose))
        else:
            pose_fis = sorted(point_cache.keys())
        if pose_teacher_transform is not None:
            with torch.no_grad():
                for fi in pose_fis:
                    if fi >= len(per_frame_pose):
                        continue
                    center = pose_enc_to_camera_centers(per_frame_pose[fi])
                    if center.ndim > 1:
                        center = center[0]
                    center_t = center[:3].detach().to(
                        device=pose_teacher_transform.s.device,
                        dtype=pose_teacher_transform.s.dtype,
                    ).view(1, 3)
                    target_enu = pose_teacher_transform.model_to_world(center_t)[0].detach().to(device=device, dtype=torch.float32)
                    pose_cache[int(fi)] = {"target_enu": target_enu, "source": pose_source_label}

    stats.update({
        "accepted": bool(point_cache or pose_cache),
        "reason": (
            "ok" if point_cache else
            (f"pose_only_{point_reject_reason}" if pose_cache and point_teacher_rejected else
             ("pose_only" if pose_cache else "no_grouped_points"))
        ),
        "inliers": int(len(enu_np)),
        "inlier_ratio": float(len(enu_np) / max(stats.get("sampled_points", len(enu_np)), 1)),
        "frames": int(len(point_cache)),
        "pose_frames": int(len(pose_cache)),
        "pose_source": pose_source_label if pose_cache else "none",
        "point_source": "rejected" if point_teacher_rejected else ("srt" if use_srt_teacher else "raw"),
    })
    print(
        f"[GeoV3][PointTTT] seg {segment_input.segment_id}: "
        f"teacher points={stats['inliers']}/{stats.get('sampled_points', stats['inliers'])} "
        f"frames={stats['frames']} pose_frames={stats['pose_frames']} "
        f"source={stats['point_source']} pose_source={stats['pose_source']}")
    return point_cache, pose_cache, stats


def _pose_to_pitch_deg(pose_enc: torch.Tensor) -> Optional[float]:
    """从单帧 pose 编码提取俯仰角 pitch（度）。

    将相机前向轴（-Z_cam 在世界系中）投影到世界水平面，取 tilt = atan2(|水平分量|, |竖直分量|)：
    nadir 正下视约为 0，斜视越大越大，与 world->model 朝向无关。任何异常返回 None，便于调用方
    优雅回退（门控 PnP 路径为可选启用）。
    """
    try:
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        _extri, _ = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(1, 1), build_intrinsics=False)
        _R = _extri[..., :3, :3]
        if _R.dim() > 2:
            _R = _R.reshape(-1, 3, 3)[0]
        # In a nadir-down body frame, R_w2c maps +Z_world -> -Z_cam.
        # Pitch from horizontal: angle between camera optical axis (-Z_cam in world)
        # and -Z_world. Equivalent to: pitch = arcsin(R_w2c[2,2]) deviating from -1.
        # Use arcsin(R_w2c[2, 0]) for x-axis tilt; we just need a stable scalar
        # that goes from ~0 in nadir to large positive in oblique views.
        # Choose the projection of the camera forward (-Z_cam in world frame) onto
        # the world horizontal plane; tilt = atan2(|h|, |v|).
        _R_np = _R.detach().cpu().float().numpy()
        # camera forward in world = R_c2w @ [0,0,1] = R_w2c.T[:, 2] = R_w2c[2, :]
        _fwd_w = _R_np[2, :]
        _h = float(np.hypot(_fwd_w[0], _fwd_w[1]))
        _v = float(abs(_fwd_w[2]))
        return float(np.degrees(np.arctan2(_h, _v)))
    except Exception:
        return None


def _is_fixed_rotation_render_pnp(corr: Dict[str, Any]) -> bool:
    """判断该 corr 是否为 render 裁切下的 fixed_rotation_center PnP 结果。

    要求 corr["pnp_result"].method == "fixed_rotation_center" 且
    dom_crop_info.mode == "render"。是则返回 True。
    """
    if not isinstance(corr, dict):
        return False
    pnp = corr.get("pnp_result")
    crop_info = corr.get("dom_crop_info") or {}
    return (
        isinstance(pnp, dict)
        and pnp.get("method") == "fixed_rotation_center"
        and crop_info.get("mode") == "render"
    )


def _fixed_rotation_pnp_center(corr: Dict[str, Any], min_dim: int = 2) -> Optional[np.ndarray]:
    """取 fixed_rotation PnP 的世界系相机中心 t_world（(N,) float32）。

    仅当 corr 通过 _is_fixed_rotation_render_pnp 时返回；要求 t_world 前 min_dim 维存在且有限，
    否则返回 None。
    """
    if not _is_fixed_rotation_render_pnp(corr):
        return None
    pnp = corr.get("pnp_result") or {}
    t_world = pnp.get("t_world")
    if t_world is None:
        return None
    arr = np.asarray(t_world, dtype=np.float32).reshape(-1)
    if arr.size < min_dim or not np.all(np.isfinite(arr[:min_dim])):
        return None
    return arr


def _fixed_rotation_pnp_gate(
    corr: Dict[str, Any],
    config: Optional[GeoV3Config],
) -> Tuple[bool, Dict[str, Any]]:
    """对单帧 fixed_rotation PnP 结果做质量门控。

    依据 pnp_result.diagnostics 检查内点数、正深度比、重投影中位误差(px)、射线残差中位(m)，
    可选检查相对先验的 XY 位移，任一不达 config 阈值即拒绝。返回 (是否通过, stats)，stats
    记录 reason 与各项指标。
    """
    stats: Dict[str, Any] = {"accepted": False, "reason": "not_run"}
    if _fixed_rotation_pnp_center(corr, min_dim=2) is None:
        stats["reason"] = "no_fixedR_pnp"
        return False, stats
    pnp = corr.get("pnp_result") if isinstance(corr, dict) else None
    diag = pnp.get("diagnostics") if isinstance(pnp, dict) else None
    if not isinstance(diag, dict):
        stats["reason"] = "no_diagnostics"
        return False, stats
    min_inliers = int(getattr(config, "dom_point_pnp_fallback_min_inliers", 128)) if config is not None else 128
    min_positive = float(getattr(config, "dom_point_pnp_fallback_min_positive_ratio", 0.95)) if config is not None else 0.95
    max_reproj = float(getattr(config, "dom_point_pnp_fallback_max_reproj_px", 30.0)) if config is not None else 30.0
    max_ray = float(getattr(config, "dom_point_pnp_fallback_max_ray_residual_m", 2.0)) if config is not None else 2.0
    max_prior_xy = float(getattr(config, "dom_point_pnp_fallback_max_prior_xy_m", 40.0)) if config is not None else 40.0
    gate_prior_xy = bool(getattr(config, "dom_point_pnp_fallback_gate_prior_xy", True)) if config is not None else True
    n_inliers = int(pnp.get("n_inliers", diag.get("n_good", 0)) or 0)
    positive_ratio = _finite_float_optional(diag.get("positive_depth_ratio"))
    reproj = _finite_float_optional(diag.get("median_reproj_px"))
    ray = _finite_float_optional(diag.get("median_ray_residual_m"))
    prior_xy = _finite_float_optional(diag.get("delta_xy_m"))
    reason = None
    if n_inliers < min_inliers:
        reason = "few_inliers"
    elif positive_ratio is None or positive_ratio < min_positive:
        reason = "low_positive_depth"
    elif reproj is None or reproj > max_reproj:
        reason = "high_reproj"
    elif ray is None or ray > max_ray:
        reason = "high_ray_residual"
    elif gate_prior_xy and (prior_xy is None or prior_xy > max_prior_xy):
        reason = "large_prior_shift"
    stats.update({
        "accepted": reason is None,
        "reason": "ok" if reason is None else reason,
        "n_inliers": int(n_inliers),
        "positive_ratio": None if positive_ratio is None else float(positive_ratio),
        "median_reproj_px": None if reproj is None else float(reproj),
        "median_ray_residual_m": None if ray is None else float(ray),
        "prior_xy_m": None if prior_xy is None else float(prior_xy),
        "gate_prior_xy": bool(gate_prior_xy),
        "max_prior_xy_m": float(max_prior_xy),
    })
    return reason is None, stats


def _dom_point_bootstrap_quality_stats(
    point_bootstrap_diag: Optional[Dict[str, Any]],
    config: Optional[GeoV3Config],
    *,
    require_road_support: bool = True,
) -> Dict[str, Any]:
    """评估 dom-point bootstrap 诊断的整体质量，判定 point_good。

    从 point_bootstrap_diag 的 srt/samples 子项读取内点比、median_xy、road 帧占比，分别与
    dom_point_quality_* 阈值比较。Full 路径要求三项（含 accepted）全过；Map Registration
    Only 传入 require_road_support=False，只用几何拟合质量决定是否接受 T_k。道路覆盖率仍
    记录在诊断中，但只用于决定 Full 路径能否把该证据交给 TTT/PGO。
    """
    min_ratio = float(getattr(config, "dom_point_quality_min_inlier_ratio", 0.65)) if config is not None else 0.65
    max_median_xy = float(getattr(config, "dom_point_quality_max_median_xy_m", 5.0)) if config is not None else 5.0
    min_road_ratio = float(getattr(config, "dom_point_quality_min_road_frame_ratio", 0.30)) if config is not None else 0.30
    stats: Dict[str, Any] = {
        "accepted": False,
        "point_good": False,
        "reason": "no_bootstrap_diag",
        "min_inlier_ratio": float(min_ratio),
        "max_median_xy_m": float(max_median_xy),
        "min_road_frame_ratio": float(min_road_ratio),
        "road_support_required": bool(require_road_support),
    }
    if not isinstance(point_bootstrap_diag, dict):
        return stats
    srt_stats = point_bootstrap_diag.get("srt") if isinstance(point_bootstrap_diag.get("srt"), dict) else {}
    sample_stats = point_bootstrap_diag.get("samples") if isinstance(point_bootstrap_diag.get("samples"), dict) else {}
    accepted = bool(point_bootstrap_diag.get("accepted", False)) and bool(srt_stats.get("accepted", True))
    inlier_ratio = _finite_float_optional(point_bootstrap_diag.get("inlier_ratio"))
    if inlier_ratio is None:
        inlier_ratio = _finite_float_optional(srt_stats.get("inlier_ratio"))
    median_xy = _finite_float_optional(srt_stats.get("median_xy_m"))
    frame_count = int(sample_stats.get("frames", 0) or 0)
    road_frames = int(sample_stats.get("frames_road", 0) or 0)
    nonbuilding_frames = int(sample_stats.get("frames_nonbuilding", 0) or 0)
    road_ratio = float(road_frames / max(frame_count, 1)) if frame_count > 0 else 1.0
    ratio_ok = inlier_ratio is not None and inlier_ratio >= min_ratio
    median_ok = median_xy is not None and (not np.isfinite(max_median_xy) or median_xy <= max_median_xy)
    road_ok = (
        not require_road_support
        or min_road_ratio <= 0.0
        or frame_count <= 0
        or road_ratio >= min_road_ratio
    )
    point_good = bool(accepted and ratio_ok and median_ok and road_ok)
    failed = []
    if not accepted:
        failed.append(str(point_bootstrap_diag.get("reason", srt_stats.get("reason", "not_accepted"))))
    if not ratio_ok:
        failed.append("low_inlier_ratio")
    if not median_ok:
        failed.append("high_median_xy")
    if not road_ok:
        failed.append("low_road_frame_ratio")
    stats.update({
        "accepted": bool(accepted),
        "point_good": bool(point_good),
        "reason": "ok" if point_good else "+".join(failed or ["quality_failed"]),
        "inlier_ratio": None if inlier_ratio is None else float(inlier_ratio),
        "median_xy_m": None if median_xy is None else float(median_xy),
        "frames": int(frame_count),
        "road_frames": int(road_frames),
        "nonbuilding_frames": int(nonbuilding_frames),
        "road_frame_ratio": float(road_ratio),
    })
    return stats


def _dom_point_fixedr_pnp_quality_stats(
    geo_obs: Dict[int, Dict[str, Any]],
    config: Optional[GeoV3Config],
) -> Dict[str, Any]:
    """统计各帧 fixed_rotation PnP 的质量，判断该 fallback 是否可用。

    遍历 geo_obs，对每个候选帧调 _fixed_rotation_pnp_gate，统计通过帧数 good_frames 及内点/重投影/
    射线残差/先验 XY/正深度比的中位数。good_frames 达到 min_frames 即 accepted。返回汇总 stats，
    含各中位指标与拒绝原因计数。
    """
    min_frames = int(getattr(config, "dom_point_pnp_fallback_min_frames", 3)) if config is not None else 3
    min_inliers = int(getattr(config, "dom_point_pnp_fallback_min_inliers", 128)) if config is not None else 128
    min_positive = float(getattr(config, "dom_point_pnp_fallback_min_positive_ratio", 0.95)) if config is not None else 0.95
    max_reproj = float(getattr(config, "dom_point_pnp_fallback_max_reproj_px", 30.0)) if config is not None else 30.0
    max_ray = float(getattr(config, "dom_point_pnp_fallback_max_ray_residual_m", 2.0)) if config is not None else 2.0
    max_prior_xy = float(getattr(config, "dom_point_pnp_fallback_max_prior_xy_m", 40.0)) if config is not None else 40.0
    gate_prior_xy = bool(getattr(config, "dom_point_pnp_fallback_gate_prior_xy", True)) if config is not None else True
    good_frames = 0
    candidate_frames = 0
    inlier_values: List[int] = []
    reproj_values: List[float] = []
    ray_values: List[float] = []
    prior_xy_values: List[float] = []
    positive_values: List[float] = []
    reject_reasons: Dict[str, int] = {}
    for _frame_index, corr in sorted((geo_obs or {}).items()):
        if _fixed_rotation_pnp_center(corr, min_dim=2) is None:
            reject_reasons["no_fixedR_pnp"] = int(reject_reasons.get("no_fixedR_pnp", 0)) + 1
            continue
        candidate_frames += 1
        ok_pnp, gate_stats = _fixed_rotation_pnp_gate(corr, config)
        n_inliers = int(gate_stats.get("n_inliers", 0) or 0)
        positive_ratio = gate_stats.get("positive_ratio")
        reproj = gate_stats.get("median_reproj_px")
        ray = gate_stats.get("median_ray_residual_m")
        prior_xy = gate_stats.get("prior_xy_m")
        inlier_values.append(int(n_inliers))
        if positive_ratio is not None:
            positive_values.append(float(positive_ratio))
        if reproj is not None:
            reproj_values.append(float(reproj))
        if ray is not None:
            ray_values.append(float(ray))
        if prior_xy is not None:
            prior_xy_values.append(float(prior_xy))
        if ok_pnp:
            good_frames += 1
        else:
            reason = str(gate_stats.get("reason", "rejected"))
            reject_reasons[reason] = int(reject_reasons.get(reason, 0)) + 1
    accepted = good_frames >= max(1, min_frames)
    return {
        "accepted": bool(accepted),
        "reason": "ok" if accepted else "too_few_good_fixedR_pnp_frames",
        "good_frames": int(good_frames),
        "candidate_frames": int(candidate_frames),
        "min_frames": int(min_frames),
        "min_inliers": int(min_inliers),
        "min_positive_ratio": float(min_positive),
        "max_reproj_px": float(max_reproj),
        "max_ray_residual_m": float(max_ray),
        "max_prior_xy_m": float(max_prior_xy),
        "gate_prior_xy": bool(gate_prior_xy),
        "median_inliers": float(np.median(np.asarray(inlier_values, dtype=np.float64))) if inlier_values else 0.0,
        "median_reproj_px": float(np.median(np.asarray(reproj_values, dtype=np.float64))) if reproj_values else float("inf"),
        "median_ray_residual_m": float(np.median(np.asarray(ray_values, dtype=np.float64))) if ray_values else float("inf"),
        "median_prior_xy_m": float(np.median(np.asarray(prior_xy_values, dtype=np.float64))) if prior_xy_values else float("inf"),
        "median_positive_ratio": float(np.median(np.asarray(positive_values, dtype=np.float64))) if positive_values else 0.0,
        "rejected": reject_reasons,
    }


def _dom_point_point_ttt_rejected(point_ttt_stats: Optional[Dict[str, Any]]) -> bool:
    """判断 dom-point TTT 的点教师是否被拒（仅在 TTT enabled 时有意义）。

    当 reason 为 teacher_gate_failed 或以 srt_/raw_gate_ 开头，或 srt 子项 accepted 为 False 时返回
    True；未启用或正常则返回 False。
    """
    if not isinstance(point_ttt_stats, dict) or not bool(point_ttt_stats.get("enabled", False)):
        return False
    srt_stats = point_ttt_stats.get("srt") if isinstance(point_ttt_stats.get("srt"), dict) else {}
    reason = str(point_ttt_stats.get("reason", ""))
    if reason == "teacher_gate_failed" or reason.startswith("srt_") or reason.startswith("raw_gate_"):
        return True
    if srt_stats and not bool(srt_stats.get("accepted", False)):
        return True
    return False


def _geo_ground_z_mean(corr: Dict[str, Any], prefer_road: bool = False) -> float:
    """估计该 corr 的地面 ENU 高程 Z 均值（米）。

    优先用 enu_positions_road（prefer_road 时），否则用 enu_positions，取其中有限 Z 值的均值。
    无有效数据或异常时返回 0.0。
    """
    if not isinstance(corr, dict):
        return 0.0
    enu_pos = None
    if prefer_road:
        enu_pos = corr.get("enu_positions_road")
    if enu_pos is None:
        enu_pos = corr.get("enu_positions")
    try:
        arr = np.asarray(enu_pos, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            z = arr[:, 2]
            z = z[np.isfinite(z)]
            if z.size > 0:
                return float(np.mean(z))
    except Exception:
        pass
    return 0.0


def _dom_point_ground_ray_stage_enabled(config: Optional[GeoV3Config], stage: str) -> bool:
    """判断指定 stage（ttt/seg_pgo/pgo）的 dom-point ground ray 约束是否启用。

    总开关 dom_point_ground_ray_enable 或该 stage 的 dom_point_ground_ray_{stage}_enable
    任一为 True 即启用。config 为 None 时返回 False。
    """
    if config is None:
        return False
    master = bool(getattr(config, "dom_point_ground_ray_enable", False))
    stage_flag = bool(getattr(config, f"dom_point_ground_ray_{stage}_enable", False))
    return bool(master or stage_flag)


def _ground_z_from_corr_for_ray(corr: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """从 corr 推断 ground ray 所需的地面 ENU 高程 Z（米）。

    依次尝试 enu_positions_road / enu_positions 的有限 Z 中位数；再退到 dom_crop_info 中
    pose_image_center_ground_enu / pose_center_ground_enu 的 Z。返回 (z, 来源标签)，全部缺失时
    返回 (None, "no_ground_z")。
    """
    if not isinstance(corr, dict):
        return None, "bad_corr"
    for key in ("enu_positions_road", "enu_positions"):
        vals = corr.get(key)
        try:
            arr = np.asarray(vals, dtype=np.float64)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[1] >= 3:
            z = arr[:, 2]
            z = z[np.isfinite(z)]
            if z.size > 0:
                return float(np.median(z)), f"{key}_median_z"
    crop_info = corr.get("dom_crop_info") or {}
    if isinstance(crop_info, dict):
        for key in ("pose_image_center_ground_enu", "pose_center_ground_enu"):
            val = crop_info.get(key)
            try:
                arr = np.asarray(val, dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if arr.size >= 3 and np.isfinite(float(arr[2])):
                return float(arr[2]), key
    return None, "no_ground_z"


def _ground_ray_target_from_corr(corr: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
    """构造 ground ray 的目标点 target_enu（(3,) float32 ENU 坐标）。

    XY 取 frame_center_enu，Z 优先用 frame_center_enu 的有限 Z，否则回退到
    _ground_z_from_corr_for_ray。返回 (target, z_source)；XY 缺失或非有限时返回 (None, 原因)。
    """
    if not isinstance(corr, dict):
        return None, "bad_corr"
    fce = corr.get("frame_center_enu")
    try:
        fce_arr = np.asarray(fce, dtype=np.float64).reshape(-1)
    except Exception:
        return None, "no_frame_center"
    if fce_arr.size < 2 or not np.isfinite(fce_arr[:2]).all():
        return None, "bad_frame_center_xy"
    if fce_arr.size >= 3 and np.isfinite(float(fce_arr[2])):
        z_val = float(fce_arr[2])
        z_source = "frame_center_enu_z"
    else:
        z_opt, z_source = _ground_z_from_corr_for_ray(corr)
        if z_opt is None or not np.isfinite(float(z_opt)):
            return None, z_source
        z_val = float(z_opt)
    target = np.array([float(fce_arr[0]), float(fce_arr[1]), z_val], dtype=np.float32)
    return target, z_source


def _build_dom_point_ground_ray_cache(
    geo_obs: Dict[int, Dict[str, Any]],
    config: Optional[GeoV3Config],
    device: torch.device,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """为各帧预计算 dom-point ground ray 的目标点缓存。

    任一 stage（ttt/seg_pgo/pgo）启用才执行。遍历 geo_obs，先过语义门控
    _dom_point_observation_semantics，再用 _ground_ray_target_from_corr 取目标点，写入每帧的
    dom_point_observation["ground_ray"] 并打印汇总。返回 (cache: {frame_index -> {target_enu(tensor),
    z_source}}, stats)。未启用/无数据时返回空 cache。
    """
    enabled = any(
        _dom_point_ground_ray_stage_enabled(config, stage)
        for stage in ("ttt", "seg_pgo", "pgo")
    )
    stats: Dict[str, Any] = {
        "enabled": bool(enabled),
        "accepted": False,
        "frames": 0,
        "rejected": 0,
        "z_sources": {},
    }
    if not enabled:
        stats["reason"] = "disabled"
        return {}, stats
    if not geo_obs:
        stats["reason"] = "no_geo_obs"
        return {}, stats
    cache: Dict[int, Dict[str, Any]] = {}
    reject_reasons: Dict[str, int] = {}
    z_sources: Dict[str, int] = {}
    for fi_raw, corr in sorted((geo_obs or {}).items()):
        try:
            fi = int(fi_raw)
        except (TypeError, ValueError):
            continue
        ok_sem, reason_sem, sem_info = _dom_point_observation_semantics(
            corr, config=config, stage="ground_ray")
        if not ok_sem:
            reject_reasons[reason_sem] = reject_reasons.get(reason_sem, 0) + 1
            if isinstance(corr, dict):
                corr.setdefault("dom_point_observation", {})["ground_ray"] = {"valid": False, "reason": reason_sem, **sem_info}
            continue
        target, reason = _ground_ray_target_from_corr(corr)
        if target is None:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            continue
        if isinstance(corr, dict):
            corr.setdefault("dom_point_observation", {})["ground_ray"] = {"valid": True, "reason": "ok", **sem_info}
        cache[fi] = {
            "target_enu": torch.tensor(target, dtype=torch.float32, device=device),
            "z_source": reason,
        }
        z_sources[reason] = z_sources.get(reason, 0) + 1
    stats.update({
        "accepted": bool(cache),
        "reason": "ok" if cache else "no_valid_targets",
        "frames": int(len(cache)),
        "rejected": int(sum(reject_reasons.values())),
        "reject_reasons": reject_reasons,
        "z_sources": z_sources,
    })
    if cache:
        print(
            f"[GeoV3][GroundRay] targets={len(cache)} "
            f"z_sources={z_sources} rejected={reject_reasons}"
        )
    else:
        print(f"[GeoV3][GroundRay] no valid targets rejected={reject_reasons}")
    return cache, stats


def _pose_center_ray_model(pose_enc: torch.Tensor) -> torch.Tensor:
    """从 pose_enc 解出相机在 model 坐标系下的中心射线方向（单位向量）。

    解码 extrinsic 得 R_w2c，取其第 2 行（相机前向在世界系的投影）并归一化。返回 (3,) 张量。
    """
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    extri, _ = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(1, 1), build_intrinsics=False)
    R_w2c = extri[..., :3, :3]
    if R_w2c.dim() > 2:
        R_w2c = R_w2c.reshape(-1, 3, 3)[0]
    ray = R_w2c[2, :].float()
    return F.normalize(ray, dim=0, eps=1e-8)


def _pose_center_ray_model_variants(pose_enc: torch.Tensor) -> Dict[str, torch.Tensor]:
    """返回 model 坐标系下相机射线方向的 4 种候选（用于消除朝向/轴歧义）。

    分别取 R_w2c 的第 2 行(row2)与第 2 列(col2)及其取负，各自归一化，键为
    row2_plus_z / row2_minus_z / col2_plus_z / col2_minus_z，对应不同的轴选择与 Z 符号约定。
    """
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    extri, _ = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(1, 1), build_intrinsics=False)
    R_w2c = extri[..., :3, :3]
    if R_w2c.dim() > 2:
        R_w2c = R_w2c.reshape(-1, 3, 3)[0]
    row2 = R_w2c[2, :].float()
    col2 = R_w2c[:, 2].float()
    return {
        "row2_plus_z": F.normalize(row2, dim=0, eps=1e-8),
        "row2_minus_z": F.normalize(-row2, dim=0, eps=1e-8),
        "col2_plus_z": F.normalize(col2, dim=0, eps=1e-8),
        "col2_minus_z": F.normalize(-col2, dim=0, eps=1e-8),
    }


def _signed_safe_denominator_torch(values: torch.Tensor, min_abs: float) -> torch.Tensor:
    """对张量做保号的分母下限裁剪，避免除零。

    绝对值小于 max(min_abs, 1e-8) 的元素被替换为 sign * min_abs（保留原符号），其余原样返回。
    """
    min_abs_val = max(float(min_abs), 1e-8)
    sign = torch.where(values >= 0.0, torch.ones_like(values), -torch.ones_like(values))
    return torch.where(values.abs() < min_abs_val, sign * min_abs_val, values)


def _ground_ray_intersection_xy_torch(
    center_enu: torch.Tensor,
    dir_enu: torch.Tensor,
    target_z: torch.Tensor,
    min_abs_dir_z: float,
) -> torch.Tensor:
    """求相机中心射线与水平面 z=target_z 交点的 ENU XY 坐标。

    射线起点 center_enu、方向 dir_enu（归一化后），用 lam=(target_z-center_z)/dir_z 求参数，
    分母经 _signed_safe_denominator_torch(min_abs_dir_z) 保号裁剪以防近水平射线发散。返回 (2,) XY。
    """
    direction = F.normalize(dir_enu.view(-1)[:3], dim=0, eps=1e-8)
    denom = _signed_safe_denominator_torch(direction[2], min_abs_dir_z)
    lam = (target_z.reshape(()) - center_enu.view(-1)[2]) / denom
    return center_enu.view(-1)[:2] + lam * direction[:2]


def _normalize_geo_target_mode(
    target_mode: str,
    *,
    prefer_ground_signal: bool = True,
    prefer_pnp: bool = True,
    allow_raw_pnp: bool = False,
) -> str:
    """将外部传入的 geo target_mode 字符串规范化为内部标准取值。

    做小写/去横线/别名映射，标准值为 ground_signal / frame_center / fixedR_pnp / pose_ground /
    dom_points。"auto" 按 prefer_ground_signal、prefer_pnp+allow_raw_pnp 决策；未知值回退到
    ground_signal 或 frame_center。
    """
    mode = str(target_mode or "auto").strip().lower().replace("-", "_")
    aliases = {
        "ground": "ground_signal",
        "ground_signal_preferred": "ground_signal",
        "frame": "frame_center",
        "framecenter": "frame_center",
        "raw_pnp": "fixedR_pnp",
        "fixedr_pnp": "fixedR_pnp",
        "pnp": "fixedR_pnp",
        "dompoints": "dom_points",
        "point_edges": "dom_points",
        "points": "dom_points",
        # Explicit camera-center branch.  The CLI keeps the branch name for
        # diagnostics, while the proven DOM-point implementation consumes the
        # normalized target mode below.
        "pointmap_sim3": "dom_points",
        "pointmap_camera": "dom_points",
        "pointmap_transform": "dom_points",
        "poseground": "pose_ground",
    }
    mode = aliases.get(mode, mode)
    if mode == "auto":
        if prefer_ground_signal:
            return "ground_signal"
        if prefer_pnp and allow_raw_pnp:
            return "fixedR_pnp"
        return "frame_center"
    if mode in ("ground_signal", "frame_center", "fixedR_pnp", "pose_ground", "dom_points"):
        return mode
    return "ground_signal" if prefer_ground_signal else "frame_center"


def _geo_target_mode_from_config(config: Optional[GeoV3Config]) -> str:
    """从 config 读取并规范化 geo target_mode。

    取 config.geo_target_mode 及 prefer_ground_signal_targets / prefer_pnp_camera_targets /
    allow_raw_pnp_camera_targets，调 _normalize_geo_target_mode。config 为 None 时返回 ground_signal。
    """
    if config is None:
        return "ground_signal"
    return _normalize_geo_target_mode(
        getattr(config, "geo_target_mode", "auto"),
        prefer_ground_signal=bool(getattr(config, "prefer_ground_signal_targets", True)),
        prefer_pnp=bool(getattr(config, "prefer_pnp_camera_targets", True)),
        allow_raw_pnp=bool(getattr(config, "allow_raw_pnp_camera_targets", False)),
    )


def _frame_center_camera_target_enu(
    corr: Dict[str, Any],
    *,
    prefer_road_z: bool = False,
) -> Tuple[Optional[np.ndarray], str]:
    """由 frame_center_enu 构造相机中心 ENU 目标点（(3,) float32）。

    XY 取 frame_center_enu。相机高度 Z 优先用 camera_z_gt，其次 frame_center 自带 Z，否则按
    地面高程 + estimated_agl（AGL）估计；当 render 裁切且 AGL 源为 legacy/affine 时视为不可靠，
    回退 cam_z=0。返回 (target, z_source)；XY 缺失时返回 (None, "none")。
    """
    fce = corr.get("frame_center_enu")
    if fce is None:
        return None, "none"
    fce_arr = np.asarray(fce, dtype=np.float32).reshape(-1)
    if fce_arr.size < 2 or not np.all(np.isfinite(fce_arr[:2])):
        return None, "none"

    camera_z_gt = corr.get("camera_z_gt")
    if camera_z_gt is not None and np.isfinite(float(camera_z_gt)):
        cam_z = float(camera_z_gt)
        z_source = "camera_z_gt"
    elif fce_arr.size >= 3 and np.isfinite(float(fce_arr[2])):
        cam_z = float(fce_arr[2])
        z_source = "frame_center_enu_z"
    else:
        ground_z = _geo_ground_z_mean(corr, prefer_road=prefer_road_z)
        est_agl = corr.get("estimated_agl")
        est_source = str(corr.get("estimated_agl_source") or "legacy_affine")
        crop_info = corr.get("dom_crop_info") or {}
        crop_mode = crop_info.get("mode") if isinstance(crop_info, dict) else None
        bad_render_affine = (
            crop_mode == "render"
            and est_source in ("legacy_affine", "affine_metric_crop")
        )
        if est_agl is not None and not bad_render_affine and float(est_agl) > 0.0:
            cam_z = ground_z + float(est_agl)
            z_source = str(est_source)
        else:
            cam_z = 0.0
            z_source = "fallback_zero_z"

    return np.array([float(fce_arr[0]), float(fce_arr[1]), cam_z], dtype=np.float32), z_source


def _pose_ground_camera_target_enu(
    corr: Dict[str, Any],
    *,
    prefer_road_z: bool = False,
) -> Tuple[Optional[np.ndarray], str]:
    """由 pose 提供的图像中心地面点构造相机中心 ENU 目标点（(3,) float32）。

    XY 取 dom_crop_info 的 pose_image_center_ground_enu（缺则 pose_center_ground_enu）；相机高度 Z
    优先复用 _frame_center_camera_target_enu 的结果，否则用地面点自身 Z 或 0。返回
    (target, "pose_ground")；缺失时返回 (None, "none")。
    """
    crop_info = corr.get("dom_crop_info") or {}
    if not isinstance(crop_info, dict):
        return None, "none"
    pose_ground = crop_info.get("pose_image_center_ground_enu")
    if pose_ground is None:
        pose_ground = crop_info.get("pose_center_ground_enu")
    try:
        ground_arr = np.asarray(pose_ground, dtype=np.float32).reshape(-1)
    except Exception:
        return None, "none"
    if ground_arr.size < 2 or not np.all(np.isfinite(ground_arr[:2])):
        return None, "none"
    frame_target, _ = _frame_center_camera_target_enu(corr, prefer_road_z=prefer_road_z)
    if frame_target is not None and len(frame_target) >= 3:
        cam_z = float(frame_target[2])
    else:
        cam_z = float(ground_arr[2]) if ground_arr.size >= 3 and np.isfinite(float(ground_arr[2])) else 0.0
    return np.array([float(ground_arr[0]), float(ground_arr[1]), cam_z], dtype=np.float32), "pose_ground"


def _dem_ground_z_at_enu_xy(
    xy_enu: np.ndarray,
    project_fn: Any,
    geo_elev: Any,
    dom_transform: Any,
) -> Optional[float]:
    """查询给定 ENU XY 处的 DEM 地面高程 Z（米）。

    用 project_fn 把 (x,y,0) 投到 DOM 像素 uv，再经 geo_elev.enu_3d_from_dom 配合 dom_transform
    取地面 Z。任一依赖为 None、输入非有限或异常时返回 None。
    """
    if project_fn is None or geo_elev is None or dom_transform is None:
        return None
    try:
        xy_arr = np.asarray(xy_enu, dtype=np.float64).reshape(-1)
        if xy_arr.size < 2 or not np.all(np.isfinite(xy_arr[:2])):
            return None
        xyz = torch.tensor([[float(xy_arr[0]), float(xy_arr[1]), 0.0]], dtype=torch.float32)
        uv = project_fn(xyz)
        if torch.is_tensor(uv):
            uv = uv.detach().cpu().numpy()
        uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1)
        if uv_arr.size < 2 or not np.all(np.isfinite(uv_arr[:2])):
            return None
        _, _, ground_z = geo_elev.enu_3d_from_dom(float(uv_arr[0]), float(uv_arr[1]), dom_transform)
        ground_z = float(ground_z)
        return ground_z if np.isfinite(ground_z) else None
    except Exception:
        return None


def _fmt_signed_summary(vals: List[float]) -> str:
    """把一组带符号误差(米)格式化为含中位/均值(带正负号)/RMSE/样本数的摘要串。

    仅统计有限值；空集返回 "None"。
    """
    arr = np.asarray([float(v) for v in vals if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return "None"
    rmse = float(np.sqrt(np.mean(arr ** 2)))
    return (
        f"med={float(np.median(arr)):+.2f}m "
        f"mean={float(np.mean(arr)):+.2f}m "
        f"rmse={rmse:.2f}m n={int(arr.size)}"
    )


def _fmt_unsigned_summary(vals: List[float]) -> str:
    """把一组无符号量(米)格式化为含中位/均值/样本数的摘要串。

    仅统计有限值；空集返回 "None"。
    """
    arr = np.asarray([float(v) for v in vals if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return "None"
    return f"med={float(np.median(arr)):.2f}m mean={float(np.mean(arr)):.2f}m n={int(arr.size)}"


def _geo_camera_target_enu(
    corr: Dict[str, Any],
    *,
    target_mode: str = "auto",
    prefer_ground_signal: bool = True,
    prefer_pnp: bool = True,
    allow_raw_pnp: bool = False,
    prefer_road_z: bool = False,
    ground_signal_max_delta_xy_m: float = 120.0,
    ground_signal_min_coverage: float = 0.01,
) -> Tuple[Optional[np.ndarray], str]:
    """为 DOM 监督返回统一的相机中心 ENU 目标点 (target, source)。

    先用 _normalize_geo_target_mode 归一化 target_mode，按模式取目标：
    - dom_points：返回 None（仅走 point-edge 通道）。
    - ground_signal：用 RoMa+homography 得到的地面足迹位移（render/oblique 裁切下较可靠），
      在保持 render pose 固定的前提下，把该 XY 位移叠加到先验相机中心；位移过大
      (>ground_signal_max_delta_xy_m) 或像素覆盖率过低 (<ground_signal_min_coverage) 则放弃。
    - pose_ground：用 _pose_ground_camera_target_enu。
    - fixedR_pnp（或 ground_signal 且开启 prefer_pnp+allow_raw_pnp）：用 fixed-R PnP 中心，
      因其 Z/中心可观测性弱，属 opt-in 回退。
    其余情况回退到 _frame_center_camera_target_enu。
    """
    if not isinstance(corr, dict):
        return None, "none"
    mode = _normalize_geo_target_mode(
        target_mode,
        prefer_ground_signal=bool(prefer_ground_signal),
        prefer_pnp=bool(prefer_pnp),
        allow_raw_pnp=bool(allow_raw_pnp),
    )
    if mode == "dom_points":
        return None, "dom_points_point_edges_only"

    if mode == "ground_signal":
        crop_info = corr.get("dom_crop_info") or {}
        if isinstance(crop_info, dict):
            cam_prior = crop_info.get("pose_cam_center_enu")
            ground_prior = crop_info.get("pose_image_center_ground_enu")
            if ground_prior is None:
                ground_prior = crop_info.get("pose_center_ground_enu")
            frame_center = corr.get("frame_center_enu")
            try:
                cam_arr = np.asarray(cam_prior, dtype=np.float64).reshape(-1)
                g0_arr = np.asarray(ground_prior, dtype=np.float64).reshape(-1)
                fc_arr = np.asarray(frame_center, dtype=np.float64).reshape(-1)
            except Exception:
                cam_arr = g0_arr = fc_arr = np.empty((0,), dtype=np.float64)
            if cam_arr.size >= 3 and g0_arr.size >= 2 and fc_arr.size >= 2:
                if (np.all(np.isfinite(cam_arr[:3])) and np.all(np.isfinite(g0_arr[:2]))
                        and np.all(np.isfinite(fc_arr[:2]))):
                    pixels = corr.get("frame_pixels")
                    coverage = 1.0
                    try:
                        pix_arr = np.asarray(pixels, dtype=np.float64)
                        q_w = float(corr.get("query_W", 0) or 0)
                        q_h = float(corr.get("query_H", 0) or 0)
                        if pix_arr.ndim == 2 and pix_arr.shape[0] >= 8 and q_w > 1.0 and q_h > 1.0:
                            span = np.maximum(pix_arr[:, :2].max(axis=0) - pix_arr[:, :2].min(axis=0), 0.0)
                            coverage = float((span[0] * span[1]) / max(q_w * q_h, 1.0))
                    except Exception:
                        coverage = 0.0
                    min_cov = float(ground_signal_min_coverage)
                    if np.isfinite(min_cov) and min_cov > 0.0 and coverage < min_cov:
                        return None, "ground_signal_low_coverage"
                    delta_xy = fc_arr[:2] - g0_arr[:2]
                    delta_norm = float(np.linalg.norm(delta_xy))
                    max_delta = float(ground_signal_max_delta_xy_m)
                    if not np.isfinite(max_delta) or max_delta <= 0.0:
                        max_delta = 120.0
                    if delta_norm <= max_delta:
                        target = cam_arr[:3].copy()
                        target[:2] = target[:2] + delta_xy
                        return target.astype(np.float32), "ground_signal"

    if mode == "pose_ground":
        pose_target, pose_source = _pose_ground_camera_target_enu(corr, prefer_road_z=prefer_road_z)
        if pose_target is not None:
            return pose_target, pose_source

    if mode == "fixedR_pnp" or (mode == "ground_signal" and prefer_pnp and allow_raw_pnp):
        pnp_center = _fixed_rotation_pnp_center(corr, min_dim=3)
        if pnp_center is not None:
            return np.asarray(pnp_center[:3], dtype=np.float32), "fixedR_pnp"

    return _frame_center_camera_target_enu(corr, prefer_road_z=prefer_road_z)


def _ground_signal_flip_target_enu(corr: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """诊断用的反号备选 ground_signal 目标点。

    把地面足迹位移取反后叠加到先验相机中心：target = pose_cam - (frame_center - pose_ground)。
    用于排查 ground_signal 位移符号是否搞反。返回 (target(3,) float32, 位移模长)；缺数据时返回 (None, None)。
    """
    if not isinstance(corr, dict):
        return None, None
    crop_info = corr.get("dom_crop_info") or {}
    if not isinstance(crop_info, dict):
        return None, None
    try:
        cam_arr = np.asarray(crop_info.get("pose_cam_center_enu"), dtype=np.float64).reshape(-1)
        ground_prior = crop_info.get("pose_image_center_ground_enu")
        if ground_prior is None:
            ground_prior = crop_info.get("pose_center_ground_enu")
        g0_arr = np.asarray(ground_prior, dtype=np.float64).reshape(-1)
        fc_arr = np.asarray(corr.get("frame_center_enu"), dtype=np.float64).reshape(-1)
    except Exception:
        return None, None
    if cam_arr.size < 3 or g0_arr.size < 2 or fc_arr.size < 2:
        return None, None
    if not (np.all(np.isfinite(cam_arr[:3])) and np.all(np.isfinite(g0_arr[:2])) and np.all(np.isfinite(fc_arr[:2]))):
        return None, None
    delta_xy = fc_arr[:2] - g0_arr[:2]
    target = cam_arr[:3].copy()
    target[:2] = target[:2] - delta_xy
    return target.astype(np.float32), float(np.linalg.norm(delta_xy))


def _pose_to_R_c2w_model_np(pose_enc: torch.Tensor) -> Optional[np.ndarray]:
    """从 pose_enc 解出相机在 model 坐标系下的 R_c2w（(3,3) float32 numpy）。

    解码 extrinsic 得 R_w2c，取首个并转置得到 R_c2w。解码失败或异常时返回 None。
    """
    try:
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        with torch.no_grad():
            extri, _ = pose_encoding_to_extri_intri(
                pose_enc, image_size_hw=(1, 1), build_intrinsics=False)
            R_w2c = extri[..., :3, :3]
            if R_w2c.dim() > 2:
                R_w2c = R_w2c.reshape(-1, 3, 3)[0]
            return R_w2c.detach().cpu().float().numpy().T.astype(np.float32)
    except Exception:
        return None


def _nearest_rotation_np(R: np.ndarray) -> Optional[np.ndarray]:
    """把任意 3x3 矩阵投影到最接近的旋转矩阵 SO(3)（(3,3) float64）。

    经 SVD 取 U@Vt；若 det<0 则翻转 U 末列以保证 det=+1（避免反射）。输入非有限或异常时返回 None。
    """
    try:
        arr = np.asarray(R, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(arr).all():
            return None
        U, _, Vt = np.linalg.svd(arr)
        rot = U @ Vt
        if np.linalg.det(rot) < 0.0:
            U[:, -1] *= -1.0
            rot = U @ Vt
        if not np.isfinite(rot).all():
            return None
        return rot.astype(np.float64)
    except Exception:
        return None


def _average_rotation_np(rotations: List[np.ndarray]) -> Optional[np.ndarray]:
    """对一组旋转矩阵做平均，返回最接近的旋转矩阵。

    先用 _nearest_rotation_np 清洗每个输入（丢弃无效项），再对求和矩阵做一次
    _nearest_rotation_np 投影回 SO(3)。无有效输入时返回 None。
    """
    valid: List[np.ndarray] = []
    for R in rotations:
        rot = _nearest_rotation_np(R)
        if rot is not None:
            valid.append(rot)
    if not valid:
        return None
    return _nearest_rotation_np(np.sum(valid, axis=0))


def _seg0_gt_model_to_enu_rotation(
    segment_input: SegmentInput,
    per_frame_pose: Optional[List[torch.Tensor]],
) -> Optional[np.ndarray]:
    """计算 segment 0 的 model→ENU 旋转 R_model_to_enu（(3,3) float32）。

    gt_rotation 是首帧相机 R_c2w（在 ENU 下）。无 per_frame_pose 时直接返回该 GT 旋转；
    否则用首帧 model 系下的 R_c2w_model0，组合得 R_gt_c2w @ R_c2w_model0.T。
    gt_rotation 缺失或形状非 (3,3) 时返回 None。
    """
    gt_rot = getattr(segment_input, "gt_rotation", None)
    if gt_rot is None:
        return None
    R_gt_c2w = np.asarray(gt_rot, dtype=np.float32)
    if R_gt_c2w.shape != (3, 3):
        return None
    if not per_frame_pose:
        return R_gt_c2w
    R_c2w_model0 = _pose_to_R_c2w_model_np(per_frame_pose[0])
    if R_c2w_model0 is None:
        return R_gt_c2w
    return (R_gt_c2w @ R_c2w_model0.T).astype(np.float32)


def _anchor_aligned_t(
    anchor_enu_np: np.ndarray,
    model_anchor_xyz: np.ndarray,
    scale: float,
    R_model_to_enu: np.ndarray,
) -> np.ndarray:
    """由 anchor、首帧 model 中心和已知 scale/R 反解 Sim3 的平移 t（(3,) float32）。

    满足 anchor_enu = scale * (model_anchor @ R_model_to_enu.T) + t，即
    t = anchor - scale * (model0 @ R.T)。anchor 不足 3 维时补零。
    """
    anchor3 = np.asarray(anchor_enu_np, dtype=np.float32).reshape(-1)
    if anchor3.size < 3:
        anchor3 = np.pad(anchor3, (0, 3 - anchor3.size), constant_values=0.0)
    model0 = np.asarray(model_anchor_xyz, dtype=np.float32).reshape(-1)[:3]
    return (anchor3[:3] - float(scale) * (model0 @ R_model_to_enu.T)).astype(np.float32)


def _estimate_fixedR_anchor_scale(
    model_xyzs: List[np.ndarray],
    enu_xys: List[np.ndarray],
    anchor_enu_np: np.ndarray,
    R_model_to_enu: np.ndarray,
) -> Optional[Tuple[float, np.ndarray, Dict[str, Any]]]:
    """在 R_model_to_enu 固定的前提下，估计 fixed-R Sim3 的 scale 和平移 t。

    给定 model 中心点集与对应 ENU XY 目标，估计两种 scale：
    - anchor_scale：以首帧为锚的相对位移最小二乘解。
    - pairwise_scale：用所有帧两两位移做 MAD-gate 鲁棒最小二乘（优先，可消除相对首帧
      GT anchor 的相干 XY 平移偏置）。
    优先用 pairwise，否则用 anchor；scale 无效（NaN 或 <1e-6）时返回 None。
    返回 (scale, t3(3,), diag)，diag 含 scale 来源与残差等诊断。
    """
    if len(model_xyzs) < 2 or len(enu_xys) < 2:
        return None
    model_arr = np.asarray(model_xyzs, dtype=np.float64)
    enu_arr = np.asarray(enu_xys, dtype=np.float64)
    if model_arr.ndim != 2 or model_arr.shape[1] < 3:
        return None
    if enu_arr.ndim != 2 or enu_arr.shape[1] < 2:
        return None
    R = np.asarray(R_model_to_enu, dtype=np.float64)
    anchor_xy = np.asarray(anchor_enu_np, dtype=np.float64).reshape(-1)[:2]
    d_model = model_arr[:, :3] - model_arr[0:1, :3]
    pred_xy_unit = (d_model @ R.T)[:, :2]
    obs_xy = enu_arr[:, :2] - anchor_xy[None, :]
    a = pred_xy_unit[1:].reshape(-1)
    b = obs_xy[1:].reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    anchor_scale: Optional[float] = None
    if valid.sum() >= 2:
        a_anchor = a[valid]
        b_anchor = b[valid]
        denom = float(np.dot(a_anchor, a_anchor))
        if denom >= 1e-12:
            _anchor_scale = float(np.dot(a_anchor, b_anchor) / denom)
            if np.isfinite(_anchor_scale) and _anchor_scale >= 1e-6:
                anchor_scale = _anchor_scale

    # Prefer scale from DOM-target relative displacements.  This removes the
    # coherent XY translation bias that can affect all ground-signal targets
    # with respect to the first-frame GT anchor.
    pairwise_scale: Optional[float] = None
    pairwise_median_residual: Optional[float] = None
    pairwise_pairs = 0
    pairwise_inliers = 0
    pair_pred: List[np.ndarray] = []
    pair_obs: List[np.ndarray] = []
    pair_scales: List[float] = []
    pred_xy_abs = (model_arr[:, :3] @ R.T)[:, :2]
    for ia in range(1, len(model_arr)):
        for ib in range(ia + 1, len(model_arr)):
            pred = pred_xy_abs[ib] - pred_xy_abs[ia]
            obs = enu_arr[ib, :2] - enu_arr[ia, :2]
            if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(obs))):
                continue
            pred_norm2 = float(np.dot(pred, pred))
            obs_norm = float(np.linalg.norm(obs))
            if pred_norm2 < 1e-10 or obs_norm < 0.25:
                continue
            scale_ij = float(np.dot(pred, obs) / pred_norm2)
            if not np.isfinite(scale_ij) or scale_ij <= 1e-6 or scale_ij > 10000.0:
                continue
            pair_pred.append(pred.astype(np.float64))
            pair_obs.append(obs.astype(np.float64))
            pair_scales.append(scale_ij)
    pairwise_pairs = len(pair_scales)
    if pairwise_pairs >= 3:
        scales = np.asarray(pair_scales, dtype=np.float64)
        scale_med = float(np.median(scales))
        scale_mad = float(np.median(np.abs(scales - scale_med)))
        scale_gate = max(5.0, 3.0 * 1.4826 * scale_mad, 0.15 * abs(scale_med))
        inlier_mask = np.abs(scales - scale_med) <= scale_gate
        if int(inlier_mask.sum()) < 3:
            inlier_mask = np.abs(scales - scale_med) <= max(10.0, 0.25 * abs(scale_med))
        pairwise_inliers = int(inlier_mask.sum())
        if pairwise_inliers >= 3:
            pred_in = np.stack(pair_pred, axis=0)[inlier_mask]
            obs_in = np.stack(pair_obs, axis=0)[inlier_mask]
            a_pair = pred_in.reshape(-1)
            b_pair = obs_in.reshape(-1)
            denom_pair = float(np.dot(a_pair, a_pair))
            if denom_pair >= 1e-12:
                _pairwise_scale = float(np.dot(a_pair, b_pair) / denom_pair)
                if np.isfinite(_pairwise_scale) and 1e-6 <= _pairwise_scale <= 10000.0:
                    pairwise_scale = _pairwise_scale
                    residual = np.linalg.norm(pairwise_scale * pred_in - obs_in, axis=1)
                    pairwise_median_residual = float(np.median(residual)) if residual.size else 0.0

    scale_source = "fixedR_pairwise_anchor" if pairwise_scale is not None else "fixedR_anchor"
    scale = pairwise_scale if pairwise_scale is not None else anchor_scale
    if scale is None or not np.isfinite(scale) or scale < 1e-6:
        return None
    t3 = _anchor_aligned_t(anchor_enu_np, model_arr[0, :3], scale, R_model_to_enu)
    diag: Dict[str, Any] = {
        "mode": scale_source,
        "anchor_scale": None if anchor_scale is None else float(anchor_scale),
        "pairwise_scale": None if pairwise_scale is None else float(pairwise_scale),
        "pairwise_pairs": int(pairwise_pairs),
        "pairwise_inliers": int(pairwise_inliers),
        "pairwise_median_residual_m": (
            None if pairwise_median_residual is None else float(pairwise_median_residual)
        ),
    }
    if anchor_scale is not None and pairwise_scale is not None and abs(anchor_scale) > 1e-8:
        diag["pairwise_anchor_scale_ratio"] = float(pairwise_scale / anchor_scale)
    return scale, t3, diag


def _estimate_transform_from_geo_obs(
    geo_obs: Dict[int, Dict[str, Any]],
    per_frame_pose: List[torch.Tensor],
    segment_input: SegmentInput,
    device: torch.device,
    dtype: torch.dtype,
    t0_transform: Optional[LocalModelTransform] = None,
    target_mode: str = "auto",
    use_pnp_target: bool = False,
    use_ground_signal_target: bool = True,
    allow_raw_pnp_target: bool = False,
    ground_signal_max_delta_xy_m: float = 120.0,
    ground_signal_min_coverage: float = 0.01,
    robust_segment_fit: bool = True,
    forced_world_scale: Optional[float] = None,
) -> Tuple[LocalModelTransform, Dict[str, Any]]:
    """从 DOM 观测 + segment anchor 估计 T_k（model 空间 → ENU）。

    始终把首帧 anchor 作为一个约束加入，使得即便只有单个 DOM 观测也能凑成 2 点
    Umeyama（解出 scale + heading）。

    首帧 anchor 来源：
      - segment 0：anchor_world_xyz（首帧 GT）
      - segment k>0：prev_summary.boundary_end_enu（上一段由 DOM 估计的 ENU）

    回退顺序：
      n >= 2 → 2D Umeyama (s, R, t)
      n == 1 → 仅 anchor 平移（无 DOM 匹配，s=1, R=I）

    use_ground_signal_target=True 时，render/pose 的 DOM 匹配通过地面足迹位移贡献
    相机中心 XY 修正。allow_raw_pnp_target 保留旧的 fixed-R PnP 中心作为显式 opt-in
    回退；默认关闭，因为高空近平面场景下 PnP-Z 可观测性弱。

    forced_world_scale 预留给显式的外部 scale 源；overlap bootstrap 先验必须在此传
    None，因为它不是绝对 scale 观测。
    """
    anchor_enu_np = _get_segment_anchor_enu(segment_input)
    if anchor_enu_np is None:
        anchor_enu_np = np.zeros(3, dtype=np.float32)
    ref_z = float(anchor_enu_np[2]) if len(anchor_enu_np) >= 3 else 0.0

    # --- Always include first-frame model center ↔ anchor ENU as pair 0 ---
    with torch.no_grad():
        mc0 = pose_enc_to_camera_centers(per_frame_pose[0])
        if mc0.ndim > 1:
            mc0 = mc0[0]
        model_anchor_xyz = mc0[:3].detach().cpu().float().numpy().astype(np.float32)
        model_anchor = model_anchor_xyz[:2]

    model_xys: List[np.ndarray] = [model_anchor]
    model_xyzs: List[np.ndarray] = [model_anchor_xyz]
    enu_xys: List[np.ndarray] = [anchor_enu_np[:2].copy()]

    n_pnp_used = 0
    n_pnp_skip = 0
    n_ground_signal_used = 0
    n_pose_ground_used = 0

    # --- Add DOM observations for other frames ---
    for fi, corr in sorted(geo_obs.items()):
        if corr is None:
            continue
        target_enu, target_source = _geo_camera_target_enu(
            corr,
            target_mode=target_mode,
            prefer_ground_signal=bool(use_ground_signal_target),
            prefer_pnp=bool(use_pnp_target),
            allow_raw_pnp=bool(allow_raw_pnp_target),
            ground_signal_max_delta_xy_m=float(ground_signal_max_delta_xy_m),
            ground_signal_min_coverage=float(ground_signal_min_coverage),
        )
        if target_enu is None or len(target_enu) < 2:
            continue
        if fi >= len(per_frame_pose):
            continue
        with torch.no_grad():
            mc = pose_enc_to_camera_centers(per_frame_pose[fi])
            if mc.ndim > 1:
                mc = mc[0]
            model_xyz = mc[:3].detach().cpu().float().numpy().astype(np.float32)
            model_xy = model_xyz[:2]
        if target_source == "ground_signal":
            n_ground_signal_used += 1
        elif target_source == "pose_ground":
            n_pose_ground_used += 1
        elif use_pnp_target and target_source == "fixedR_pnp":
            n_pnp_used += 1
        elif use_pnp_target:
            n_pnp_skip += 1
        enu_xy = np.array([float(target_enu[0]), float(target_enu[1])], dtype=np.float32)
        model_xys.append(model_xy)
        model_xyzs.append(model_xyz)
        enu_xys.append(enu_xy)

    n = len(model_xys)
    n_dom = n - 1  # exclude the anchor pair

    def _make_transform(s: float, R2: np.ndarray, t2: np.ndarray) -> LocalModelTransform:
        R3 = np.eye(3, dtype=np.float32)
        R3[:2, :2] = R2
        # Z: model_z → enu_z = s*model_z + t_z; model_z=0 at anchor → t_z = ref_z
        t3 = np.array([t2[0], t2[1], ref_z], dtype=np.float32)
        anc_t = torch.tensor(anchor_enu_np[:3].astype(np.float32), device=device, dtype=dtype)
        return LocalModelTransform(
            anchor_world_xyz=anc_t,
            R=torch.tensor(R3, device=device, dtype=dtype),
            s=torch.tensor(float(s), device=device, dtype=dtype),
            t=torch.tensor(t3, device=device, dtype=dtype),
        )

    # ── Case 1: Umeyama ──
    if t0_transform is not None:
        # seg k>0: pure M_0 Umeyama using the unified camera target mapped by T_0.
        m0_model_pts: List[np.ndarray] = []
        m0_target_pts: List[np.ndarray] = []
        m0_z_vals: List[float] = []
        for fi, corr in sorted(geo_obs.items()):
            if corr is None:
                continue
            target_enu, target_source = _geo_camera_target_enu(
                corr,
                target_mode=target_mode,
                prefer_ground_signal=bool(use_ground_signal_target),
                prefer_pnp=bool(use_pnp_target),
                allow_raw_pnp=bool(allow_raw_pnp_target),
                ground_signal_max_delta_xy_m=float(ground_signal_max_delta_xy_m),
                ground_signal_min_coverage=float(ground_signal_min_coverage),
            )
            if target_enu is None or len(target_enu) < 3:
                continue
            if fi >= len(per_frame_pose):
                continue
            with torch.no_grad():
                mc = pose_enc_to_camera_centers(per_frame_pose[fi])
                if mc.ndim > 1:
                    mc = mc[0]
                model_xy_fi = mc[:2].detach().cpu().float().numpy().astype(np.float32)
            with torch.no_grad():
                target_t = torch.tensor(
                    target_enu[:3], device=device, dtype=dtype
                ).unsqueeze(0)
                target_m0 = (
                    t0_transform.world_to_model(target_t)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .astype(np.float32)
                )
            if target_source == "ground_signal":
                n_ground_signal_used += 1
            elif target_source == "pose_ground":
                n_pose_ground_used += 1
            elif use_pnp_target and target_source == "fixedR_pnp":
                n_pnp_used += 1
            elif use_pnp_target:
                n_pnp_skip += 1
            m0_model_pts.append(model_xy_fi)
            m0_target_pts.append(np.array(
                [float(target_m0[0]), float(target_m0[1])], dtype=np.float32))
            m0_z_vals.append(float(target_m0[2]))
        n_dom_m0 = len(m0_model_pts)
        if n_dom_m0 >= 2:
            ref_z_target = float(np.mean(m0_z_vals))
            if forced_world_scale is not None:
                fixed_world_scale = float(forced_world_scale)
                t0_s = float(t0_transform.s.detach().cpu().float())
                if np.isfinite(fixed_world_scale) and fixed_world_scale > 1e-8 and np.isfinite(t0_s) and abs(t0_s) > 1e-8:
                    s_rel_fixed = float(fixed_world_scale / t0_s)
                    fixed_result = _estimate_sim2_fixed_scale_2d(
                        np.stack(m0_model_pts), np.stack(m0_target_pts), scale=s_rel_fixed)
                    if fixed_result is not None:
                        R2, t2, fixed_residual = fixed_result
                        R3 = np.eye(3, dtype=np.float32)
                        R3[:2, :2] = R2
                        t3_rel = np.array([t2[0], t2[1], ref_z_target], dtype=np.float32)
                        T_k_rel = LocalModelTransform(
                            anchor_world_xyz=torch.zeros(3, device=device, dtype=dtype),
                            R=torch.tensor(R3, device=device, dtype=dtype),
                            s=torch.tensor(float(s_rel_fixed), device=device, dtype=dtype),
                            t=torch.tensor(t3_rel, device=device, dtype=dtype),
                        )
                        transform = compose_transforms(t0_transform, T_k_rel)
                        transform = _apply_gt_rotation_seg0(
                            transform, segment_input, device, dtype,
                            per_frame_pose=per_frame_pose,
                            align_model_frame=use_pnp_target,
                        )
                        angle_deg = float(np.degrees(np.arctan2(float(R2[1, 0]), float(R2[0, 0]))))
                        _pnp_tag = (
                            f" target_mode={target_mode} ground_used={n_ground_signal_used}"
                            f" pose_ground_used={n_pose_ground_used} raw_pnp_used={n_pnp_used} fallback={n_pnp_skip}"
                            if (use_pnp_target or use_ground_signal_target) else ""
                        )
                        print(
                            f"[GeoV3][Sim2] seg {segment_input.segment_id}: fixed_scale_m0"
                            f" n_dom={n_dom_m0} s_rel={s_rel_fixed:.4f}"
                            f" (s_combined={float(transform.s.detach().cpu()):.2f})"
                            f" heading={angle_deg:.1f}deg residual={fixed_residual:.2f}m" + _pnp_tag
                        )
                        return transform, {
                            "sim2_mode": "fixed_scale_m0", "n_pairs": n_dom_m0, "n_dom": n_dom_m0,
                            "s": float(transform.s.detach().cpu()), "s_rel": float(s_rel_fixed),
                            "forced_world_scale": float(fixed_world_scale),
                            "fixed_scale_residual_m": float(fixed_residual),
                            "overlap_mode": "fixed_scale_m0", "can_propagate": True, "fallback_reason": None,
                            "pnp_used": int(n_pnp_used), "pnp_skip": int(n_pnp_skip),
                            "ground_signal_used": int(n_ground_signal_used),
                            "pose_ground_used": int(n_pose_ground_used),
                            "geo_target_mode": str(target_mode),
                            "robust_fit": {
                                "mode": "fixed_scale",
                                "n_inliers": int(n_dom_m0),
                                "median_residual": float(fixed_residual),
                            },
                        }
                print(
                    f"[GeoV3][Sim2] seg {segment_input.segment_id}: fixed_scale_m0 unavailable; "
                    "falling back to free Sim2"
                )
            result, robust_diag = _estimate_sim2_2d_robust(
                np.stack(m0_model_pts), np.stack(m0_target_pts),
                enable=bool(robust_segment_fit))
            if result is not None:
                s_rel, R2, t2 = result
                R3 = np.eye(3, dtype=np.float32)
                R3[:2, :2] = R2
                t3_rel = np.array([t2[0], t2[1], ref_z_target], dtype=np.float32)
                T_k_rel = LocalModelTransform(
                    anchor_world_xyz=torch.zeros(3, device=device, dtype=dtype),
                    R=torch.tensor(R3, device=device, dtype=dtype),
                    s=torch.tensor(float(s_rel), device=device, dtype=dtype),
                    t=torch.tensor(t3_rel, device=device, dtype=dtype),
                )
                transform = compose_transforms(t0_transform, T_k_rel)
                transform = _apply_gt_rotation_seg0(
                    transform, segment_input, device, dtype,
                    per_frame_pose=per_frame_pose,
                    align_model_frame=use_pnp_target,
                )
                angle_deg = float(np.degrees(np.arctan2(float(R2[1, 0]), float(R2[0, 0]))))
                _pnp_tag = (
                    f" target_mode={target_mode} ground_used={n_ground_signal_used}"
                    f" pose_ground_used={n_pose_ground_used} raw_pnp_used={n_pnp_used} fallback={n_pnp_skip}"
                    if (use_pnp_target or use_ground_signal_target) else ""
                )
                print(
                    f"[GeoV3][Sim2] seg {segment_input.segment_id}: umeyama_2d(M0)"
                    f" n_dom={n_dom_m0} s_rel={s_rel:.4f}"
                    f" (s_combined={float(transform.s.detach().cpu()):.2f})"
                    f" heading={angle_deg:.1f}deg"
                    f" robust={robust_diag.get('mode', 'none')}"
                    f" inliers={robust_diag.get('n_inliers', n_dom_m0)}/{n_dom_m0}" + _pnp_tag
                )
                return transform, {
                    "sim2_mode": "umeyama_2d", "n_pairs": n_dom_m0, "n_dom": n_dom_m0,
                    "s": float(transform.s.detach().cpu()), "s_rel": float(s_rel),
                    "overlap_mode": "umeyama_2d", "can_propagate": True, "fallback_reason": None,
                    "pnp_used": int(n_pnp_used), "pnp_skip": int(n_pnp_skip),
                    "ground_signal_used": int(n_ground_signal_used),
                    "pose_ground_used": int(n_pose_ground_used),
                    "geo_target_mode": str(target_mode),
                    "robust_fit": robust_diag,
                }
    elif n >= 2:
        # seg 0: Umeyama in ENU space.
        if use_pnp_target:
            R_fixed = _seg0_gt_model_to_enu_rotation(segment_input, per_frame_pose)
            if R_fixed is not None:
                fixed = _estimate_fixedR_anchor_scale(
                    model_xyzs, enu_xys, anchor_enu_np, R_fixed)
                if fixed is not None:
                    s_fixed, t3_fixed, fixed_diag = fixed
                    anc_t = torch.tensor(anchor_enu_np[:3].astype(np.float32), device=device, dtype=dtype)
                    transform = LocalModelTransform(
                        anchor_world_xyz=anc_t,
                        R=torch.tensor(R_fixed, device=device, dtype=dtype),
                        s=torch.tensor(float(s_fixed), device=device, dtype=dtype),
                        t=torch.tensor(t3_fixed, device=device, dtype=dtype),
                    )
                    angle_deg = float(np.degrees(np.arctan2(float(R_fixed[1, 0]), float(R_fixed[0, 0]))))
                    _scale_mode = str(fixed_diag.get("mode", "fixedR_anchor"))
                    _anchor_s = fixed_diag.get("anchor_scale")
                    _pair_s = fixed_diag.get("pairwise_scale")
                    _pair_res = fixed_diag.get("pairwise_median_residual_m")
                    _scale_tag = ""
                    if _pair_s is not None:
                        _scale_tag += (
                            f" pair_s={float(_pair_s):.4f}"
                            f" pair_inliers={int(fixed_diag.get('pairwise_inliers', 0))}"
                        )
                        if _pair_res is not None:
                            _scale_tag += f" pair_res={float(_pair_res):.2f}m"
                    if _anchor_s is not None:
                        _scale_tag += f" anchor_s={float(_anchor_s):.4f}"
                    print(
                        f"[GeoV3][Sim2] seg {segment_input.segment_id}: {_scale_mode}"
                        f" n_dom={n_dom} s={s_fixed:.4f} heading={angle_deg:.1f}deg"
                        f" t=[{t3_fixed[0]:.1f},{t3_fixed[1]:.1f}]"
                        f"{_scale_tag}"
                        f" target_mode={target_mode} ground_used={n_ground_signal_used}"
                        f" pose_ground_used={n_pose_ground_used} raw_pnp_used={n_pnp_used} fallback={n_pnp_skip}"
                    )
                    return transform, {
                        "sim2_mode": _scale_mode, "n_pairs": n, "n_dom": n_dom,
                        "s": float(s_fixed), "overlap_mode": _scale_mode,
                        "can_propagate": True, "fallback_reason": None,
                        "pnp_used": int(n_pnp_used), "pnp_skip": int(n_pnp_skip),
                        "ground_signal_used": int(n_ground_signal_used),
                        "pose_ground_used": int(n_pose_ground_used),
                        "geo_target_mode": str(target_mode),
                        "fixedR_scale_diag": fixed_diag,
                    }
        result, robust_diag = _estimate_sim2_2d_robust(
            np.stack(model_xys), np.stack(enu_xys),
            enable=bool(robust_segment_fit))
        if result is not None:
            s, R2, t2 = result
            transform = _make_transform(s, R2, t2)
            transform = _apply_gt_rotation_seg0(
                transform, segment_input, device, dtype,
                per_frame_pose=per_frame_pose,
                align_model_frame=use_pnp_target,
            )
            angle_deg = float(np.degrees(np.arctan2(float(R2[1, 0]), float(R2[0, 0]))))
            _pnp_tag = (
                f" target_mode={target_mode} ground_used={n_ground_signal_used}"
                f" pose_ground_used={n_pose_ground_used} raw_pnp_used={n_pnp_used} fallback={n_pnp_skip}"
                if (use_pnp_target or use_ground_signal_target) else ""
            )
            print(
                f"[GeoV3][Sim2] seg {segment_input.segment_id}: umeyama_2d"
                f" n_dom={n_dom} s={s:.4f} heading={angle_deg:.1f}deg"
                f" t=[{t2[0]:.1f},{t2[1]:.1f}]"
                f" robust={robust_diag.get('mode', 'none')}"
                f" inliers={robust_diag.get('n_inliers', n)}/{n}" + _pnp_tag
            )
            return transform, {
                "sim2_mode": "umeyama_2d", "n_pairs": n, "n_dom": n_dom, "s": s,
                "overlap_mode": "umeyama_2d", "can_propagate": True, "fallback_reason": None,
                "pnp_used": int(n_pnp_used), "pnp_skip": int(n_pnp_skip),
                "ground_signal_used": int(n_ground_signal_used),
                "pose_ground_used": int(n_pose_ground_used),
                "geo_target_mode": str(target_mode),
                "robust_fit": robust_diag,
            }
    anc_t = torch.tensor(anchor_enu_np[:3].astype(np.float32), device=device, dtype=dtype)
    transform = build_local_model_transform(anchor_world_xyz=anc_t, device=device, dtype=dtype)
    transform = _apply_gt_rotation_seg0(
        transform, segment_input, device, dtype,
        per_frame_pose=per_frame_pose,
        align_model_frame=use_pnp_target,
    )
    print(
        f"[GeoV3][Sim2] seg {segment_input.segment_id}: anchor_only"
        f" enu={anchor_enu_np[:2].tolist()}"
    )
    return transform, {
        "sim2_mode": "anchor_only", "n_pairs": 1, "n_dom": 0, "s": 1.0,
        "overlap_mode": "anchor_only", "can_propagate": False, "fallback_reason": "no_dom_obs",
    }


def _apply_gt_rotation_seg0(
    transform: LocalModelTransform,
    segment_input: SegmentInput,
    device: torch.device,
    dtype: torch.dtype,
    per_frame_pose: Optional[List[torch.Tensor]] = None,
    align_model_frame: bool = False,
) -> LocalModelTransform:
    """segment 0 专用：用首帧 GT 旋转覆盖 transform 里的 R。

    历史上的 direct-R 路径保留给非 PnP/正射(nadir)运行。在 oblique fixed-R PnP 路径下，
    segment_input.gt_rotation 是首帧相机在 ENU 下的 R_c2w，需与预测的 model 系 R 组合
    （align_model_frame=True），并据此重算平移 t 以保持 anchor 对齐。
    非 seg0、gt_rotation 缺失或形状非 (3,3) 时原样返回 transform。
    """
    if segment_input.segment_id != 0:
        return transform
    gt_rot = getattr(segment_input, "gt_rotation", None)
    if gt_rot is None:
        return transform
    R_gt = np.asarray(gt_rot, dtype=np.float32)
    if R_gt.shape != (3, 3):
        print(f"[GeoV3][GT-R] seg0: unexpected gt_rotation shape {R_gt.shape}, skipping")
        return transform
    R_apply = R_gt
    t_apply = transform.t
    mode = "direct"
    if align_model_frame:
        R_model_to_enu = _seg0_gt_model_to_enu_rotation(segment_input, per_frame_pose)
        if R_model_to_enu is not None:
            R_apply = R_model_to_enu.astype(np.float32)
            mode = "model_aligned"
            anchor_enu_np = _get_segment_anchor_enu(segment_input)
            if anchor_enu_np is not None and per_frame_pose:
                with torch.no_grad():
                    mc0 = pose_enc_to_camera_centers(per_frame_pose[0])
                    if mc0.ndim > 1:
                        mc0 = mc0[0]
                    model0 = mc0[:3].detach().cpu().float().numpy().astype(np.float32)
                t_np = _anchor_aligned_t(
                    anchor_enu_np, model0,
                    float(transform.s.detach().cpu().float()), R_apply)
                t_apply = torch.tensor(t_np, device=device, dtype=dtype)
    print(f"[GeoV3][GT-R] seg0: overriding R with GT first-frame rotation ({mode})")
    return LocalModelTransform(
        anchor_world_xyz=transform.anchor_world_xyz,
        R=torch.tensor(R_apply, device=device, dtype=dtype),
        s=transform.s,
        t=t_apply,
    )


def _safe_pose_orientation_rotation_for_segment(
    segment_input: SegmentInput,
    per_frame_pose: Optional[List[torch.Tensor]],
    overlap_prior_transform: Optional[LocalModelTransform],
    overlap_prior_diag: Optional[Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """为某 segment 取出一个可安全使用的 pose 朝向旋转（model→ENU）及诊断。

    segment 0：用首帧 GT 旋转 _seg0_gt_model_to_enu_rotation（经 _nearest_rotation_np 清洗）。
    segment k>0：仅当 overlap 先验的 rotation_source 为 overlap_full_extrinsic 且 full
    extrinsic handoff 被接受时，才采用其 R。否则返回 (None, diag)，diag 记录拒绝原因。
    """
    diag: Dict[str, Any] = {"accepted": False, "source": "none", "reason": "unavailable"}
    if int(segment_input.segment_id) == 0:
        rotation = _seg0_gt_model_to_enu_rotation(segment_input, per_frame_pose)
        if rotation is not None:
            diag.update({"accepted": True, "source": "gt_first", "reason": "ok"})
            return _nearest_rotation_np(rotation), diag
        diag["reason"] = "no_gt_first_rotation"
        return None, diag

    prior_diag = dict(overlap_prior_diag or {})
    rotation_source = str(prior_diag.get("rotation_source", ""))
    full_diag = prior_diag.get("full_extrinsic_handoff")
    full_ok = not isinstance(full_diag, dict) or bool(full_diag.get("accepted", False))
    if (
        overlap_prior_transform is not None
        and rotation_source == "overlap_full_extrinsic"
        and full_ok
    ):
        rotation = overlap_prior_transform.R.detach().cpu().float().numpy()
        rotation = _nearest_rotation_np(rotation)
        if rotation is not None:
            diag.update({"accepted": True, "source": "overlap_full_extrinsic", "reason": "ok"})
            return rotation, diag

    diag.update({
        "reason": "no_pose_overlap_orientation",
        "overlap_rotation_source": rotation_source or "none",
    })
    if isinstance(full_diag, dict):
        diag["full_extrinsic_handoff"] = full_diag
    return None, diag


def _keep_pose_orientation_for_transform(
    transform: LocalModelTransform,
    safe_rotation: Optional[np.ndarray],
    per_frame_pose: Optional[List[torch.Tensor]],
    source_diag: Dict[str, Any],
) -> Tuple[LocalModelTransform, Dict[str, Any]]:
    """用 safe_rotation 替换 transform 的 R，同时保持首帧世界中心不动。

    把 R 换成 safe_rotation（经 _nearest_rotation_np 清洗），并重算平移 t 使首帧 model 中心
    仍映射到原来的世界中心（frame0_world_center 不变），scale 不变。
    safe_rotation 无效、无 per_frame_pose 或计算异常时，原样返回 transform 并在 diag 记录原因。
    """
    diag = dict(source_diag or {})
    if safe_rotation is None:
        diag.setdefault("accepted", False)
        diag.setdefault("reason", "no_safe_rotation")
        return transform, diag
    safe_rotation = _nearest_rotation_np(safe_rotation)
    if safe_rotation is None:
        diag.update({"accepted": False, "reason": "bad_safe_rotation"})
        return transform, diag
    if not per_frame_pose:
        diag.update({"accepted": False, "reason": "no_pose_anchor"})
        return transform, diag
    try:
        with torch.no_grad():
            model_anchor = pose_enc_to_camera_centers(per_frame_pose[0])
            if model_anchor.ndim > 1:
                model_anchor = model_anchor[0]
            model_anchor_t = model_anchor[:3].to(
                device=transform.s.device,
                dtype=transform.s.dtype,
            ).view(1, 3)
            world_anchor_t = transform.model_to_world(model_anchor_t)[0]
        model_anchor_np = model_anchor_t[0].detach().cpu().float().numpy().astype(np.float64)
        world_anchor_np = world_anchor_t.detach().cpu().float().numpy().astype(np.float64)
        scale_value = float(transform.s.detach().cpu().float())
        translation_np = world_anchor_np - scale_value * (model_anchor_np @ safe_rotation.T)
        guarded = LocalModelTransform(
            anchor_world_xyz=world_anchor_t.detach().clone(),
            R=torch.tensor(safe_rotation.astype(np.float32), device=transform.s.device, dtype=transform.s.dtype),
            s=transform.s,
            t=torch.tensor(translation_np.astype(np.float32), device=transform.s.device, dtype=transform.s.dtype),
        )
        diag.update({
            "accepted": True,
            "reason": "ok",
            "preserve": "frame0_world_center",
        })
        return guarded, diag
    except Exception as exc:
        diag.update({"accepted": False, "reason": f"exception:{exc}"})
        return transform, diag


def _dom_anchor_fallback_transform(
    geo_obs: Dict[int, Dict[str, Any]],
    per_frame_pose: List[torch.Tensor],
    segment_input: SegmentInput,
    device: torch.device,
    dtype: torch.dtype,
    t0_transform: Optional[LocalModelTransform] = None,
) -> Tuple[LocalModelTransform, Dict[str, Any]]:
    """回退方案：用首个可用 DOM 匹配帧重新估计完整的 s,R,t。

    当 overlap 传播和常规 Umeyama 都失败时触发。把首个 DOM 匹配帧的 ENU 中心与 segment
    anchor 凑成 2 点 Umeyama；有 t0_transform 时在 M_0 空间拟合，否则在 ENU 空间。
    完全无 DOM 数据时退化为纯 anchor-only。诊断里标记 is_fallback=True。
    """
    anchor_enu_np = _get_segment_anchor_enu(segment_input)
    if anchor_enu_np is None:
        anchor_enu_np = np.zeros(3, dtype=np.float32)
    ref_z = float(anchor_enu_np[2]) if len(anchor_enu_np) >= 3 else 0.0
    # Anchor pair from segment's first frame model center ↔ anchor ENU
    with torch.no_grad():
        mc0 = pose_enc_to_camera_centers(per_frame_pose[0]) if per_frame_pose else None
    if mc0 is None:
        anc_enu = torch.tensor(anchor_enu_np[:3].astype(np.float32), device=device, dtype=dtype)
        if t0_transform is not None:
            anc_m0 = t0_transform.world_to_model(anc_enu.unsqueeze(0)).squeeze(0)
            T_k_rel = LocalModelTransform(
                anchor_world_xyz=torch.zeros(3, device=device, dtype=dtype),
                R=torch.eye(3, device=device, dtype=dtype),
                s=torch.tensor(1.0, device=device, dtype=dtype),
                t=anc_m0,
            )
            transform = compose_transforms(t0_transform, T_k_rel)
        else:
            transform = build_local_model_transform(anchor_world_xyz=anc_enu, device=device, dtype=dtype)
        return transform, {
            "overlap_mode": "fallback_anchor_only", "can_propagate": False,
            "fallback_reason": "no_pose_data", "is_fallback": True, "overlap_len": 0,
        }
    if mc0.ndim > 1:
        mc0 = mc0[0]
    mc0_np = mc0[:2].detach().cpu().float().numpy().astype(np.float32)

    model_xys: List[np.ndarray] = [mc0_np]
    enu_xys: List[np.ndarray] = [anchor_enu_np[:2].copy()]

    # Find first DOM-matched frame and add its center pair
    first_fi = next((fi for fi in sorted(geo_obs.keys()) if geo_obs[fi] is not None), None)
    if first_fi is not None and first_fi < len(per_frame_pose):
        fce = geo_obs[first_fi].get("frame_center_enu")
        if fce is not None and len(fce) >= 2:
            with torch.no_grad():
                mc_fi = pose_enc_to_camera_centers(per_frame_pose[first_fi])
                if mc_fi.ndim > 1:
                    mc_fi = mc_fi[0]
                mc_fi_np = mc_fi[:2].detach().cpu().float().numpy().astype(np.float32)
            model_xys.append(mc_fi_np)
            enu_xys.append(np.array([float(fce[0]), float(fce[1])], dtype=np.float32))
            # Note: enu_positions[:, 2] is ground-point altitude, not camera altitude — ref_z stays as anchor Z

    n = len(model_xys)
    if t0_transform is not None:
        # seg k>0: pure M_0 Umeyama using frame_center_m0 (pre-computed at ENU→M_0 boundary).
        m0_model_pts: List[np.ndarray] = []
        m0_target_pts: List[np.ndarray] = []
        m0_z_vals_fb: List[float] = []
        anc_enu = torch.tensor(anchor_enu_np[:3].astype(np.float32), device=device, dtype=dtype)
        anc_m0 = t0_transform.world_to_model(anc_enu.unsqueeze(0)).squeeze(0)
        m0_model_pts.append(mc0_np)
        m0_target_pts.append(anc_m0[:2].detach().cpu().float().numpy().astype(np.float32))
        m0_z_vals_fb.append(float(anc_m0[2].detach().cpu().float()))
        for fi_fb in sorted(geo_obs.keys()):
            corr_fb = geo_obs[fi_fb]
            if corr_fb is None:
                continue
            fcm_fb = corr_fb.get("frame_center_m0")
            if fcm_fb is None or len(fcm_fb) < 2 or fi_fb >= len(per_frame_pose):
                continue
            with torch.no_grad():
                mc_fb = pose_enc_to_camera_centers(per_frame_pose[fi_fb])
                if mc_fb.ndim > 1:
                    mc_fb = mc_fb[0]
                mc_fb_np = mc_fb[:2].detach().cpu().float().numpy().astype(np.float32)
            m0_model_pts.append(mc_fb_np)
            m0_target_pts.append(np.array([float(fcm_fb[0]), float(fcm_fb[1])], dtype=np.float32))
            m0_z_vals_fb.append(float(fcm_fb[2]) if len(fcm_fb) > 2 else 0.0)
        if len(m0_model_pts) >= 2:
            ref_z_target = float(np.mean(m0_z_vals_fb))
            result = _estimate_sim2_2d(np.stack(m0_model_pts), np.stack(m0_target_pts))
        else:
            result = None
            ref_z_target = ref_z
    elif n >= 2:
        result = _estimate_sim2_2d(np.stack(model_xys), np.stack(enu_xys))
        ref_z_target = ref_z
    else:
        result = None
        ref_z_target = ref_z

    if result is not None:
        s_est, R2, t2 = result
        R3 = np.eye(3, dtype=np.float32)
        R3[:2, :2] = R2
        t3 = np.array([t2[0], t2[1], ref_z_target], dtype=np.float32)
        if t0_transform is not None:
            T_k_rel = LocalModelTransform(
                anchor_world_xyz=torch.zeros(3, device=device, dtype=dtype),
                R=torch.tensor(R3, device=device, dtype=dtype),
                s=torch.tensor(float(s_est), device=device, dtype=dtype),
                t=torch.tensor(t3, device=device, dtype=dtype),
            )
            transform = compose_transforms(t0_transform, T_k_rel)
        else:
            anc_t = torch.tensor(anchor_enu_np[:3].astype(np.float32), device=device, dtype=dtype)
            transform = LocalModelTransform(
                anchor_world_xyz=anc_t,
                R=torch.tensor(R3, device=device, dtype=dtype),
                s=torch.tensor(float(s_est), device=device, dtype=dtype),
                t=torch.tensor(t3, device=device, dtype=dtype),
            )
        angle_deg = float(np.degrees(np.arctan2(float(R2[1, 0]), float(R2[0, 0]))))
        print(
            f"[GeoV3][Fallback] seg {segment_input.segment_id}: dom_anchor_fallback"
            f" first_fi={first_fi} s_est={s_est:.4f} heading={angle_deg:.1f}deg"
        )
        return transform, {
            "overlap_mode": "dom_anchor_fallback", "can_propagate": True,
            "fallback_reason": "dom_anchor_refitted", "is_fallback": True,
            "overlap_len": 0, "s": float(transform.s.detach().cpu()),
            "s_rel": float(s_est) if t0_transform is not None else None,
        }

    # No DOM data available at all: pure anchor-only
    anc_enu = torch.tensor(anchor_enu_np[:3].astype(np.float32), device=device, dtype=dtype)
    if t0_transform is not None:
        anc_m0 = t0_transform.world_to_model(anc_enu.unsqueeze(0)).squeeze(0)
        T_k_rel = LocalModelTransform(
            anchor_world_xyz=torch.zeros(3, device=device, dtype=dtype),
            R=torch.eye(3, device=device, dtype=dtype),
            s=torch.tensor(1.0, device=device, dtype=dtype),
            t=anc_m0,
        )
        transform = compose_transforms(t0_transform, T_k_rel)
    else:
        transform = build_local_model_transform(anchor_world_xyz=anc_enu, device=device, dtype=dtype)
    print(f"[GeoV3][Fallback] seg {segment_input.segment_id}: anchor_only (no DOM data)")
    return transform, {
        "overlap_mode": "fallback_anchor_only", "can_propagate": False,
        "fallback_reason": "no_dom_no_overlap", "is_fallback": True, "overlap_len": 0,
    }


def _build_transform_for_segment(
    segment_input: SegmentInput,
    world_xyz_init: torch.Tensor,
    geo_obs: Optional[Dict[int, Dict[str, Any]]] = None,
    per_frame_pose: Optional[List[torch.Tensor]] = None,
    point_bootstrap_transform: Optional[LocalModelTransform] = None,
    point_bootstrap_diag: Optional[Dict[str, Any]] = None,
    overlap_len: int = 0,
    s_drift_threshold: float = 0.30,
    prev_post_ttt_overlap_m0: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
    overlap_prior_transform: Optional[LocalModelTransform] = None,
    overlap_prior_diag: Optional[Dict[str, Any]] = None,
    overlap_dom_consistency_diag: Optional[Dict[str, Any]] = None,
) -> Tuple[LocalModelTransform, Dict[str, Any]]:
    """为某 segment 选择并构建 model→ENU 的 LocalModelTransform，按优先级单出口返回。

    从 prev_summary 取出固定参考 T_0（M_0→ENU；seg0 自身为 None）并读取 config 决定
    target_mode（含 dom_points 模式）、ground_signal 阈值、是否用 PnP/robust 拟合等。
    按优先级依次尝试：1) dom_points 模式下的 DOM/DEM point-bootstrap；2) 可靠的
    overlap full-extrinsic handoff；3) 非 dom_points 模式下的 DOM 拟合；4) 当前段地图
    注册失败时复用上一 submap 的 transform；5) DOM-anchor fallback。旧 submap 只作失败
    回退，不能短路当前段地图注册。最后始终向下游传播 T_0 并合并诊断。
    """
    device = world_xyz_init.device
    dtype = world_xyz_init.dtype

    # T_0 (M_0 → ENU): fixed reference transform from segment 0.
    # Propagated through segment metadata; None for seg 0 itself (not estimated yet).
    _pm = segment_input.prev_summary
    t0_transform: Optional[LocalModelTransform] = (
        _pm.get("metadata", {}).get("t0_transform") if _pm is not None else None
    )

    use_pnp_target = True if config is None else bool(
        getattr(config, "prefer_pnp_camera_targets", True))
    use_ground_signal_target = True if config is None else bool(
        getattr(config, "prefer_ground_signal_targets", True))
    allow_raw_pnp_target = False if config is None else bool(
        getattr(config, "allow_raw_pnp_camera_targets", False))
    geo_target_mode = _geo_target_mode_from_config(config)
    dom_points_mode = geo_target_mode == "dom_points"
    ground_signal_max_delta_xy_m = 120.0 if config is None else float(
        getattr(config, "ground_signal_max_delta_xy_m", 120.0))
    ground_signal_min_coverage = 0.01 if config is None else float(
        getattr(config, "ground_signal_min_coverage", 0.01))
    robust_segment_fit = True if config is None else bool(
        getattr(config, "segment_target_robust_fit", True))
    pitch_diag: Dict[str, Any] = {
        "pnp_target_mode": geo_target_mode,
        "geo_target_mode": geo_target_mode,
        "pointmap_camera_transform": bool(
            getattr(config, "pointmap_camera_transform_enable", False)
        ),
        "use_pnp_target": bool(use_pnp_target),
        "use_ground_signal_target": bool(use_ground_signal_target),
        "allow_raw_pnp_target": bool(allow_raw_pnp_target),
    }
    if config is not None and bool(getattr(config, "use_pnp_gate", False)):
        pitch_diag["deprecated_use_pnp_gate_ignored"] = True
    print(
        f"[GeoV3][Geo-target] seg {segment_input.segment_id}: "
        f"mode={pitch_diag['pnp_target_mode']}"
    )

    prev_traj_m = segment_input.prev_trajectory_local
    prev_transform = getattr(segment_input, "prev_transform", None)

    # Accumulate through priorities; single exit so t0_transform is always propagated.
    _result: Optional[Tuple[LocalModelTransform, Dict[str, Any]]] = None

    # Preserve the previous submap transform only as a final continuity
    # fallback.  Selecting it here used to bypass every current-segment map fit.
    prev_submap_state = getattr(segment_input, "prev_submap_state", None)
    prev_submap_transform = (
        getattr(prev_submap_state, "transform", None)
        if prev_submap_state is not None else None
    )

    overlap_prior_bootstrap, overlap_prior_bootstrap_diag = _overlap_prior_reliability(
        segment_input=segment_input,
        overlap_prior_transform=overlap_prior_transform,
        overlap_prior_diag=overlap_prior_diag,
        config=config,
    )

    _overlap_consistency_diag = dict(overlap_dom_consistency_diag or {})

    # For camera/DOM-target modes the overlap prior remains crop/bootstrap-only.
    # In dom_points mode, let the absolute DOM/DEM point-edge bootstrap own the
    # segment transform when it succeeds; overlap handoff remains a crop seed and
    # fallback continuity source instead of defining the map position.
    _forced_world_scale: Optional[float] = None
    _dom_point_bootstrap_primary = (
        dom_points_mode
        and bool(getattr(config, "dom_point_prefer_point_bootstrap_transform", True))
    )
    _point_bootstrap_allowed = point_bootstrap_transform is not None
    _point_bootstrap_skip_reason = None
    if (
        _point_bootstrap_allowed
        and dom_points_mode
        and int(segment_input.segment_id) > 0
        and bool(getattr(config, "dom_point_bootstrap_require_overlap_consistency", True))
        and _overlap_consistency_diag
    ):
        _min_kept = max(1, int(getattr(config, "dom_point_bootstrap_min_overlap_consistency", 1)))
        _kept = int(_overlap_consistency_diag.get("kept", 0) or 0)
        if _kept < _min_kept:
            _point_bootstrap_allowed = False
            _point_bootstrap_skip_reason = f"overlap_consistency_kept_{_kept}_lt_{_min_kept}"
            print(
                f"[GeoV3][OverlapBootstrap] seg {segment_input.segment_id}: "
                f"skip DOM point bootstrap transform reason={_point_bootstrap_skip_reason}"
            )

    point_bootstrap_transform_for_use = point_bootstrap_transform
    point_bootstrap_orientation_diag: Dict[str, Any] = {"enabled": False}
    if (
        _point_bootstrap_allowed
        and dom_points_mode
        and bool(getattr(config, "dom_point_bootstrap_keep_pose_orientation", False))
    ):
        safe_rotation, safe_diag = _safe_pose_orientation_rotation_for_segment(
            segment_input=segment_input,
            per_frame_pose=per_frame_pose,
            overlap_prior_transform=overlap_prior_transform,
            overlap_prior_diag=overlap_prior_diag,
        )
        guarded_transform, point_bootstrap_orientation_diag = _keep_pose_orientation_for_transform(
            point_bootstrap_transform,
            safe_rotation,
            per_frame_pose,
            safe_diag,
        )
        point_bootstrap_orientation_diag["enabled"] = True
        if bool(point_bootstrap_orientation_diag.get("accepted", False)):
            point_bootstrap_transform_for_use = guarded_transform
            print(
                f"[GeoV3][PointBootstrap] seg {segment_input.segment_id}: "
                f"preserve pose orientation source={point_bootstrap_orientation_diag.get('source')} "
                f"mode={point_bootstrap_orientation_diag.get('preserve')}"
            )
        else:
            _point_bootstrap_allowed = False
            _point_bootstrap_skip_reason = (
                "orientation_guard_"
                f"{point_bootstrap_orientation_diag.get('reason', 'failed')}"
            )
            print(
                f"[GeoV3][PointBootstrap] seg {segment_input.segment_id}: "
                f"skip DOM point bootstrap transform reason={_point_bootstrap_skip_reason}"
            )

    if _result is None and dom_points_mode and _dom_point_bootstrap_primary and _point_bootstrap_allowed:
        _pb_diag = dict(point_bootstrap_diag or {})
        _head_overlap_bootstrap = bool(_pb_diag.get("head_overlap_bootstrap", False))
        _pb_diag.update({
            "sim2_mode": "dom_point_bootstrap_primary",
            "overlap_mode": "dom_point_bootstrap_primary",
            "can_propagate": True,
            "fallback_reason": None,
            "geo_target_mode": str(geo_target_mode),
            "overlap_primary": False,
            "overlap_bootstrap": bool(overlap_prior_bootstrap),
            "head_overlap_bootstrap": _head_overlap_bootstrap,
            "point_bootstrap_source": _pb_diag.get("sample_source", "unknown"),
            "point_bootstrap_primary": True,
            "overlap_scale_controlled": _head_overlap_bootstrap,
            "point_bootstrap_gate": "ok",
            "point_bootstrap_orientation": point_bootstrap_orientation_diag,
        })
        if overlap_prior_transform is not None:
            print(
                f"[GeoV3][OverlapBootstrap] seg {segment_input.segment_id}: "
                "DOM point bootstrap selected as primary transform; overlap prior seeds crop/bootstrap only"
            )
        _result = (point_bootstrap_transform_for_use, _pb_diag)

    if _result is None and dom_points_mode and overlap_prior_bootstrap and overlap_prior_transform is not None:
        _ov_diag = dict(overlap_prior_bootstrap_diag or overlap_prior_diag or {})
        _ov_diag.update({
            "sim2_mode": "overlap_full_extrinsic_handoff",
            "overlap_mode": "overlap_full_extrinsic_handoff",
            "can_propagate": True,
            "fallback_reason": None,
            "geo_target_mode": str(geo_target_mode),
            "overlap_primary": True,
            "overlap_bootstrap": True,
            "point_bootstrap_available": bool(point_bootstrap_transform is not None),
            "point_bootstrap_primary": False,
        })
        _result = (overlap_prior_transform, _ov_diag)

    # Priority 3: DOM fitting in M_0 space (or ENU for seg 0).
    # In dom_points mode this is used when primary point-bootstrap is disabled or unavailable.
    if _result is None and dom_points_mode and _point_bootstrap_allowed:
        _pb_diag = dict(point_bootstrap_diag or {})
        _head_overlap_bootstrap = bool(_pb_diag.get("head_overlap_bootstrap", False))
        _pb_diag.update({
            "sim2_mode": "dom_point_bootstrap",
            "overlap_mode": "dom_point_bootstrap",
            "can_propagate": True,
            "fallback_reason": None,
            "geo_target_mode": str(geo_target_mode),
            "overlap_bootstrap": bool(overlap_prior_bootstrap),
            "head_overlap_bootstrap": _head_overlap_bootstrap,
            "point_bootstrap_source": _pb_diag.get("sample_source", "unknown"),
            "point_bootstrap_primary": True,
            "overlap_scale_controlled": _head_overlap_bootstrap,
            "point_bootstrap_gate": "ok",
            "point_bootstrap_orientation": point_bootstrap_orientation_diag,
        })
        if overlap_prior_transform is not None:
            if _head_overlap_bootstrap:
                print(
                    f"[GeoV3][OverlapBootstrap] seg {segment_input.segment_id}: "
                    "point bootstrap uses current head-overlap DOM points; overlap prior still seeds crop/bootstrap"
                )
            else:
                print(
                    f"[GeoV3][OverlapBootstrap] seg {segment_input.segment_id}: "
                    "point bootstrap keeps free scale; overlap prior used only for crop/bootstrap"
                )
        _result = (point_bootstrap_transform_for_use, _pb_diag)

    if _result is None and not dom_points_mode and geo_obs is not None and per_frame_pose is not None:
        transform, sim2_diag = _estimate_transform_from_geo_obs(
            geo_obs=geo_obs,
            per_frame_pose=per_frame_pose,
            segment_input=segment_input,
            device=device,
            dtype=dtype,
            t0_transform=t0_transform,
            target_mode=geo_target_mode,
            use_pnp_target=use_pnp_target,
            use_ground_signal_target=use_ground_signal_target,
            allow_raw_pnp_target=allow_raw_pnp_target,
            ground_signal_max_delta_xy_m=ground_signal_max_delta_xy_m,
            ground_signal_min_coverage=ground_signal_min_coverage,
            robust_segment_fit=robust_segment_fit,
            forced_world_scale=_forced_world_scale,
        )
        if sim2_diag.get("n_dom", 0) >= 1:
            sim2_diag["overlap_primary"] = False
            sim2_diag["overlap_bootstrap"] = bool(overlap_prior_bootstrap)
            sim2_diag["overlap_dom_consistency"] = _overlap_consistency_diag
            sim2_diag["overlap_scale_controlled"] = False
            if overlap_prior_transform is not None:
                print(
                    f"[GeoV3][OverlapBootstrap] seg {segment_input.segment_id}: "
                    "DOM fit keeps free scale; overlap prior used only for crop/bootstrap"
                )
            _result = (transform, sim2_diag)

    # A free Sim2 fitted only between
    # current overlap poses and the previous predicted trajectory is internally
    # self-consistent, but not an external geo observation.  Let the anchor/DOM
    # fallback below handle missing DOM without propagating that prior as truth.
    if _result is None and overlap_prior_transform is not None:
        _prior_diag = dict(overlap_prior_diag or {})
        print(
            f"[GeoV3][OverlapBootstrap] seg {segment_input.segment_id}: "
            f"no DOM transform; overlap prior is not used as fallback "
            f"n={int(_prior_diag.get('n_overlap', 0))} "
            f"med_xy={float(_prior_diag.get('median_xy_residual_m', 0.0)):.2f}m "
            f"s={float(overlap_prior_transform.s.detach().cpu()):.4f}"
        )

    # Priority 4: if point-SRT and reliable overlap handoff fail, use current
    # segment DOM camera-center observations before reusing the previous map.
    if (
        _result is None
        and bool(getattr(config, "map_anchor_only", False))
        and geo_obs
        and per_frame_pose
    ):
        _dom_fallback_transform, _dom_fallback_diag = _dom_anchor_fallback_transform(
            geo_obs=geo_obs,
            per_frame_pose=per_frame_pose,
            segment_input=segment_input,
            device=device,
            dtype=dtype,
            t0_transform=t0_transform,
        )
        if str(_dom_fallback_diag.get("fallback_reason", "")) != "no_dom_no_overlap":
            _dom_fallback_diag["transform_source"] = "dom_anchor_fallback"
            _result = (_dom_fallback_transform, _dom_fallback_diag)

    # Priority 5: when all current-segment map registration paths fail, retain
    # the previous submap transform as a continuity fallback.
    if _result is None and prev_submap_transform is not None:
        print(
            f"[GeoV3][MapRegistration] seg {segment_input.segment_id}: "
            "current map registration unavailable; reuse previous submap transform"
        )
        _result = (prev_submap_transform, {
            "sim2_mode": "previous_submap_fallback",
            "overlap_mode": "previous_submap_fallback",
            "transform_source": "previous_submap_fallback",
            "can_propagate": True,
            "fallback_reason": "current_map_registration_unavailable",
            "is_fallback": True,
            "overlap_len": int(
                getattr(segment_input.prev_summary, "overlap_len", 0)
                if segment_input.prev_summary is not None else 0
            ),
            "reused_submap": True,
        })

    # Priority 6: anchor-only fallback when no current or previous map evidence
    # is available.
    if _result is None:
        _result = _dom_anchor_fallback_transform(
            # Registration-only may still use current-segment DOM camera
            # observations if point-SRT was rejected.  This is a deterministic
            # map fallback, not token or trajectory optimization.
            geo_obs=(geo_obs or {}) if bool(getattr(config, "map_anchor_only", False))
            else ({} if dom_points_mode else (geo_obs or {})),
            per_frame_pose=per_frame_pose or [],
            segment_input=segment_input,
            device=device,
            dtype=dtype,
            t0_transform=t0_transform,
        )

    result_transform, result_diag = _result
    # All methods share the declared first-frame position/orientation
    # initialization. Apply it once to the final segment-0 transform so the
    # point-bootstrap and fallback paths have identical protocol semantics.
    if int(segment_input.segment_id) == 0 and bool(getattr(config, "map_anchor_only", False)):
        result_transform = _apply_gt_rotation_seg0(
            result_transform,
            segment_input,
            device,
            dtype,
            per_frame_pose=per_frame_pose,
            align_model_frame=True,
        )
        result_diag["gt_first_rotation_available"] = (
            getattr(segment_input, "gt_rotation", None) is not None
        )
    result_diag.setdefault("transform_source", result_diag.get("overlap_mode", "unknown"))
    if bool(getattr(config, "pointmap_camera_transform_enable", False)):
        result_diag["transform_source"] = "pointmap_sim3"
        result_diag["camera_transform_source"] = "window_model_points_to_enu"
    result_diag["map_registration_only"] = bool(
        getattr(config, "map_anchor_only", False)
    )
    result_diag["pointmap_camera_transform"] = bool(
        getattr(config, "pointmap_camera_transform_enable", False)
    )
    if dom_points_mode:
        result_diag["point_bootstrap_attempt"] = dict(point_bootstrap_diag or {})
        result_diag["map_registration_accepted"] = bool(
            result_diag.get("point_bootstrap_primary", False)
            or result_diag.get("overlap_primary", False)
            or str(result_diag.get("sim2_mode", "")) in {
                "dom_point_bootstrap_primary", "dom_point_bootstrap",
                "overlap_full_extrinsic_handoff", "fixed_scale_m0",
                "umeyama_2d",
            }
            or result_diag.get("fallback_reason") == "dom_anchor_refitted"
        )
    # Propagate T_0 for all downstream segments.
    # seg 0: T_0 = transform just estimated.  seg k>0: T_0 unchanged, passed forward.
    result_diag["t0_transform"] = t0_transform if t0_transform is not None else result_transform
    if overlap_prior_bootstrap_diag:
        result_diag.setdefault("overlap_geo_prior", overlap_prior_bootstrap_diag)
    elif overlap_prior_diag:
        result_diag.setdefault("overlap_geo_prior", overlap_prior_diag)
    if pitch_diag:
        result_diag.update(pitch_diag)
    return result_transform, result_diag


def _frozen_forward_with_tokens_geo_v3(model: Any, frames: List[dict], query_points: Any = None):
    """对帧序列做流式 frozen 前向，逐帧缓存 KV 并收集 tokens / pose / pts3d。

    维护 aggregator 与 camera_head 的 past_key_values 实现增量推理；逐帧返回：
    DPT 层 tokens、全部层 tokens、pose_enc、(pts3d, conf)，以及 patch_start_idx。
    全程 no_grad，张量均 detach。返回 5 元组
    (per_frame_tokens, per_frame_all_tokens, per_frame_pose, per_frame_pts3d, psi)。
    """
    aggregator = model.aggregator
    camera_head = model.camera_head
    dpt_layers = list(model.point_head.intermediate_layer_idx)

    past_kv = [None] * aggregator.depth
    past_kv_cam = [None] * camera_head.trunk_depth
    per_frame_tokens = []
    per_frame_all_tokens = []
    per_frame_pose = []
    per_frame_pts3d = []
    psi = aggregator.patch_start_idx

    with torch.no_grad():
        for i, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0)
            agg_out = aggregator(images, past_key_values=past_kv, use_cache=True, past_frame_idx=i)
            if isinstance(agg_out, tuple) and len(agg_out) == 3:
                agg_tokens, psi_out, past_kv = agg_out
            else:
                agg_tokens, psi_out = agg_out
            psi = psi_out

            pose_out = camera_head(
                agg_tokens, past_key_values_camera=past_kv_cam, use_cache=True
            )
            if isinstance(pose_out, tuple):
                pose_enc = pose_out[0]
                past_kv_cam = pose_out[1] if len(pose_out) > 1 else past_kv_cam
            else:
                pose_enc = pose_out
            pose_enc = pose_enc[-1]
            per_frame_pose.append(pose_enc[:, 0, :].detach().float())
            per_frame_all_tokens.append([t.detach() for t in agg_tokens])
            per_frame_tokens.append({li: agg_tokens[li].detach().float() for li in dpt_layers})

            pts3d, pts3d_conf = model.point_head(
                agg_tokens, images=images, patch_start_idx=psi
            )
            per_frame_pts3d.append((pts3d[:, 0].detach().float(), pts3d_conf[:, 0].detach().float()))

    return per_frame_tokens, per_frame_all_tokens, per_frame_pose, per_frame_pts3d, psi


def _streaming_aggregator_forward_geo_v3(
    model: Any,
    frames: List[dict],
    *,
    detach: bool,
    gradient_checkpointing: bool = False,
) -> Tuple[List[List[torch.Tensor]], int]:
    """Replay one segment through the streaming aggregator.

    Unlike :func:`_frozen_forward_with_tokens_geo_v3`, this helper does not run
    the prediction heads and can retain the autograd graph.  LoRA TTT calls it
    once per optimization step so the common GeoV3 loss can reach adapters in
    the aggregator.  KV state is rebuilt from scratch on every call, enforcing
    segment-local adaptation.
    """
    aggregator = model.aggregator
    blocks = list(aggregator.frame_blocks) + list(aggregator.global_blocks)
    previous_checkpoint_flags = [
        bool(getattr(block, "gradient_checkpointing", False)) for block in blocks
    ]
    for block in blocks:
        block.gradient_checkpointing = bool(gradient_checkpointing)
    past_kv = [None] * aggregator.depth
    per_frame_all_tokens: List[List[torch.Tensor]] = []
    psi = int(aggregator.patch_start_idx)
    try:
        for frame_idx, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0)
            agg_out = aggregator(
                images,
                past_key_values=past_kv,
                use_cache=True,
                past_frame_idx=frame_idx,
            )
            if isinstance(agg_out, tuple) and len(agg_out) == 3:
                agg_tokens, psi, past_kv = agg_out
            else:
                agg_tokens, psi = agg_out
            if detach:
                agg_tokens = [token.detach() for token in agg_tokens]
            per_frame_all_tokens.append(list(agg_tokens))
    finally:
        for block, previous in zip(blocks, previous_checkpoint_flags):
            block.gradient_checkpointing = previous
    return per_frame_all_tokens, int(psi)


def _build_lora_prefix_cache_geo_v3(
    model: Any,
    per_frame_all_tokens: List[List[torch.Tensor]],
    frames: List[dict],
    start_block: int,
    retained_layers: List[int],
) -> Optional[Dict[str, Any]]:
    """Cache the frozen aggregator prefix preceding the first LoRA block.

    The regular frozen segment forward has already produced every alternating
    attention output.  Output ``i`` concatenates the frame-block state and the
    global-block state, so the latter half of output ``start_block - 1`` is
    exactly the token state consumed by frame block ``start_block``.  Reusing
    that detached state avoids replaying blocks that cannot depend on LoRA.
    """
    aggregator = model.aggregator
    start_block = int(start_block)
    if start_block <= 0:
        return None
    if start_block >= int(aggregator.depth):
        raise ValueError(
            f"LoRA prefix boundary {start_block} outside aggregator depth "
            f"{aggregator.depth}"
        )
    if not per_frame_all_tokens or len(per_frame_all_tokens) != len(frames):
        raise ValueError("LoRA prefix cache requires the frozen tokens for every frame")

    prefix_layer = start_block - 1
    prefix_tokens: List[torch.Tensor] = []
    prefix_outputs: List[Dict[int, torch.Tensor]] = []
    token_dim = None
    token_count = None
    for frame_tokens in per_frame_all_tokens:
        prefix_concat = frame_tokens[prefix_layer]
        if prefix_concat.ndim != 4 or prefix_concat.shape[-1] % 2 != 0:
            raise ValueError(
                "Unexpected aggregator intermediate shape for LoRA prefix: "
                f"{tuple(prefix_concat.shape)}"
            )
        current_dim = int(prefix_concat.shape[-1] // 2)
        global_state = prefix_concat[..., current_dim:].detach()
        prefix_tokens.append(global_state.reshape(-1, global_state.shape[-2], current_dim))
        prefix_outputs.append({
            int(layer_idx): frame_tokens[int(layer_idx)].detach()
            for layer_idx in retained_layers
            if int(layer_idx) < start_block
        })
        token_dim = current_dim
        token_count = int(global_state.shape[-2])

    first_image = frames[0]["img"]
    height, width = int(first_image.shape[-2]), int(first_image.shape[-1])
    pos = None
    if aggregator.rope is not None:
        pos = aggregator.position_getter(
            1,
            height // int(aggregator.patch_size),
            width // int(aggregator.patch_size),
            device=first_image.device,
        )
        pos = pos + 1
        pos_special = torch.zeros(
            1,
            int(aggregator.patch_start_idx),
            2,
            device=first_image.device,
            dtype=pos.dtype,
        )
        pos = torch.cat([pos_special, pos], dim=1)
    if token_count is not None and pos is not None and int(pos.shape[1]) != token_count:
        raise ValueError(
            f"LoRA prefix position/token mismatch: pos={pos.shape[1]} tokens={token_count}"
        )

    return {
        "start_block": start_block,
        "retained_layers": sorted(set(int(x) for x in retained_layers)),
        "prefix_tokens": prefix_tokens,
        "prefix_outputs": prefix_outputs,
        "pos": pos,
        "token_dim": int(token_dim),
        "token_count": int(token_count),
        "psi": int(aggregator.patch_start_idx),
    }


def _streaming_aggregator_suffix_forward_geo_v3(
    model: Any,
    prefix_cache: Dict[str, Any],
    *,
    detach: bool,
    gradient_checkpointing: bool = False,
) -> Tuple[List[List[Optional[torch.Tensor]]], int]:
    """Replay only the trainable aggregator suffix from a detached prefix."""
    aggregator = model.aggregator
    start_block = int(prefix_cache["start_block"])
    retained_layers = set(int(x) for x in prefix_cache["retained_layers"])
    prefix_tokens = prefix_cache["prefix_tokens"]
    prefix_outputs = prefix_cache["prefix_outputs"]
    pos = prefix_cache.get("pos")
    token_dim = int(prefix_cache["token_dim"])
    token_count = int(prefix_cache["token_count"])

    suffix_blocks = (
        list(aggregator.frame_blocks[start_block:])
        + list(aggregator.global_blocks[start_block:])
    )
    previous_checkpoint_flags = [
        bool(getattr(block, "gradient_checkpointing", False))
        for block in suffix_blocks
    ]
    for block in suffix_blocks:
        block.gradient_checkpointing = bool(gradient_checkpointing)

    suffix_past_kv: List[Any] = [None] * int(aggregator.depth)
    per_frame_outputs: List[List[Optional[torch.Tensor]]] = []
    try:
        for frame_idx, cached_tokens in enumerate(prefix_tokens):
            tokens = cached_tokens
            outputs: List[Optional[torch.Tensor]] = [None] * int(aggregator.depth)
            for layer_idx, value in prefix_outputs[frame_idx].items():
                outputs[int(layer_idx)] = value

            for block_idx in range(start_block, int(aggregator.depth)):
                frame_tokens = aggregator.frame_blocks[block_idx](tokens, pos=pos)
                frame_intermediate = frame_tokens.reshape(1, 1, token_count, token_dim)
                global_input = frame_tokens.reshape(1, token_count, token_dim)
                global_tokens, new_kv = aggregator.global_blocks[block_idx](
                    global_input,
                    pos=pos,
                    past_key_values=suffix_past_kv[block_idx],
                    use_cache=True,
                )
                suffix_past_kv[block_idx] = new_kv
                if block_idx in retained_layers:
                    global_intermediate = global_tokens.reshape(
                        1, 1, token_count, token_dim
                    )
                    output = torch.cat(
                        [frame_intermediate, global_intermediate], dim=-1
                    )
                    outputs[block_idx] = output.detach() if detach else output
                tokens = global_tokens
            per_frame_outputs.append(outputs)
    finally:
        for block, previous in zip(suffix_blocks, previous_checkpoint_flags):
            block.gradient_checkpointing = previous
    return per_frame_outputs, int(prefix_cache["psi"])


def _slice_camera_kv_value(value: Any, prefix_len: int) -> Any:
    """对 camera KV cache 沿 token 维裁剪到前 prefix_len 个（递归处理嵌套结构）。

    张量（ndim>=3）切 [:, :, :prefix_len] 并 detach；tuple/list 递归处理；None 与其他类型原样返回。
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim >= 3:
            return value[:, :, :prefix_len].detach()
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_slice_camera_kv_value(v, prefix_len) for v in value)
    if isinstance(value, list):
        return [_slice_camera_kv_value(v, prefix_len) for v in value]
    return value


def _camera_kv_cache_len(value: Any) -> Optional[int]:
    """从 camera_head 的 KV cache 中推断已缓存的 token 序列长度。

    支持张量（取 dim=2 的长度，需 ndim>=3）或嵌套 tuple/list（递归取首个有效值）。
    无法判定时返回 None。
    """
    if value is None:
        return None
    if torch.is_tensor(value) and value.ndim >= 3:
        return int(value.shape[2])
    if isinstance(value, (tuple, list)):
        for item in value:
            cache_len = _camera_kv_cache_len(item)
            if cache_len is not None:
                return cache_len
    return None


def _infer_camera_kv_tokens_per_frame(
    frozen_camera_kv_full: Optional[List[Any]],
    num_frames: Optional[int],
) -> int:
    """推断 frozen camera KV cache 中每帧占用的 token 数。

    用各层 KV cache 长度对 num_frames 取模，整除则返回 cache_len//num_frames。
    无法推断时回退为 1。
    """
    if frozen_camera_kv_full is None or num_frames is None or int(num_frames) <= 0:
        return 1
    n = int(num_frames)
    for kv in frozen_camera_kv_full:
        cache_len = _camera_kv_cache_len(kv)
        if cache_len is None or cache_len <= 0:
            continue
        if cache_len % n == 0:
            return max(int(cache_len // n), 1)
    return 1


def _slice_camera_kv_prefix(
    frozen_camera_kv_full: Optional[List[Any]],
    frame_idx: int,
    num_frames: Optional[int] = None,
) -> Optional[List[Any]]:
    """将 frozen camera KV cache 截断为 frame_idx 之前的前缀。

    按每帧 token 数 × frame_idx 计算前缀长度，对每个 KV 逐一切片。
    用于流式推理中只暴露当前帧之前的因果上下文。
    """
    if frozen_camera_kv_full is None:
        return None
    tokens_per_frame = _infer_camera_kv_tokens_per_frame(frozen_camera_kv_full, num_frames)
    prefix_len = max(int(frame_idx), 0) * max(int(tokens_per_frame), 1)
    return [_slice_camera_kv_value(kv, prefix_len) for kv in frozen_camera_kv_full]


def _build_frozen_camera_kv_full(
    model: Any,
    per_frame_all_tokens: List[List[torch.Tensor]],
) -> Optional[List[Any]]:
    """逐帧前向 camera_head 以构建覆盖全部帧的 frozen camera KV cache。

    依次喂入每帧的 agg tokens（use_cache=True）累积 KV，全程 no_grad 且关闭 autocast。
    无 camera_head 或 trunk_depth<=0 时返回 None。
    """
    camera_head = getattr(model, "camera_head", None)
    trunk_depth = int(getattr(camera_head, "trunk_depth", 0) or 0)
    if camera_head is None or trunk_depth <= 0:
        return None
    camera_kv = [None] * trunk_depth
    with torch.no_grad():
        for agg_tokens in per_frame_all_tokens:
            with torch.amp.autocast("cuda", enabled=False):
                agg_f = [
                    t.float() if torch.is_tensor(t) else t
                    for t in agg_tokens
                ]
                pose_out = camera_head(
                    agg_f,
                    past_key_values_camera=camera_kv,
                    use_cache=True,
                )
            if isinstance(pose_out, tuple) and len(pose_out) > 1:
                camera_kv = pose_out[1]
    return camera_kv


def _forward_pose_cached(
    model: Any,
    agg_tokens: List[torch.Tensor],
    camera_kv_prefix: Optional[List[Any]],
) -> torch.Tensor:
    """用（可选的）frozen camera KV 前缀前向 camera_head，返回该帧的 pose_enc。

    有前缀时走 use_cache=True 的缓存路径（流式），否则一次性 use_cache=False。
    取最后一层、第 0 个 token，返回形状 (B, C) 的 pose_enc。
    """
    with torch.amp.autocast("cuda", enabled=False):
        agg_f = [t.float() if torch.is_tensor(t) else t for t in agg_tokens]
        if camera_kv_prefix is not None:
            pose_out = model.camera_head(
                agg_f,
                past_key_values_camera=camera_kv_prefix,
                use_cache=True,
            )
        else:
            pose_out = model.camera_head(agg_f, use_cache=False)
        pose_enc = pose_out[0] if isinstance(pose_out, tuple) else pose_out
        pose_enc = pose_enc[-1]
    return pose_enc[:, 0, :]


def _forward_pose_and_pts(
    model: Any,
    frame: dict,
    agg_tokens: List[torch.Tensor],
    psi: int,
    camera_kv_prefix: Optional[List[Any]] = None,
):
    """单帧同时前向 camera_head 与 point_head，返回 (pose_enc, pts3d, pts3d_conf)。

    camera_head 走可选的 frozen KV 前缀（流式）；point_head 由 agg tokens + 原图重建点云。
    psi 为 patch_start_idx。三个返回值均已去掉最外层 batch/序列维。
    """
    images = frame["img"].unsqueeze(0)
    with torch.amp.autocast("cuda", enabled=False):
        agg_f = [t.float() if torch.is_tensor(t) else t for t in agg_tokens]
        if camera_kv_prefix is not None:
            pose_out = model.camera_head(
                agg_f,
                past_key_values_camera=camera_kv_prefix,
                use_cache=True,
            )
        else:
            pose_out = model.camera_head(agg_f, use_cache=False)
        if isinstance(pose_out, tuple):
            pose_enc = pose_out[0]
        else:
            pose_enc = pose_out
        pose_enc = pose_enc[-1]
        pts3d, pts3d_conf = model.point_head(agg_f, images=images, patch_start_idx=psi)
    return pose_enc[:, 0, :], pts3d[:, 0], pts3d_conf[:, 0]


def _build_batched_agg_tokens(
    per_frame_all_tokens: List[List[torch.Tensor]],
    per_frame_leaf_tokens: List[Dict[int, torch.Tensor]],
    fis: List[int],
) -> List[torch.Tensor]:
    """将 B 帧的逐层 tokens 沿 batch 维堆叠成批。

    对在 per_frame_leaf_tokens[fi] 中的层使用 leaf 张量（保留在计算图中，使梯度能
    回传到 optimizer 的叶子参数），其余层用原始 agg tokens。每层返回形状 (B, S, N, C)。
    """
    if len(fis) == 0:
        return []
    n_layers = len(per_frame_all_tokens[fis[0]])
    out: List[torch.Tensor] = []
    for li in range(n_layers):
        if per_frame_all_tokens[fis[0]][li] is None:
            out.append(None)
            continue
        toks = []
        for fi in fis:
            leaves_fi = per_frame_leaf_tokens[fi]
            if li in leaves_fi:
                toks.append(leaves_fi[li])
            else:
                toks.append(per_frame_all_tokens[fi][li])
        out.append(torch.cat(toks, dim=0))
    return out


def _batched_forward_pts3d(
    model: Any,
    frames: List[dict],
    per_frame_all_tokens: List[List[torch.Tensor]],
    per_frame_leaf_tokens: List[Dict[int, torch.Tensor]],
    fis: List[int],
    psi: int,
):
    """对 fis 指定的多帧批量前向 point_head，返回 (pts3d, pts3d_conf)。

    先用 _build_batched_agg_tokens 把多帧 agg tokens 堆成批（保留 leaf 梯度），
    再拼接对应原图一次性前向。psi 为 patch_start_idx。
    """
    agg_list = _build_batched_agg_tokens(per_frame_all_tokens, per_frame_leaf_tokens, fis)
    with torch.amp.autocast("cuda", enabled=False):
        agg_f = [t.float() if torch.is_tensor(t) else t for t in agg_list]
        images_B = torch.cat([frames[fi]["img"].unsqueeze(0) for fi in fis], dim=0)
        pts3d, pts3d_conf = model.point_head(agg_f, images=images_B, patch_start_idx=psi)
    return pts3d[:, 0], pts3d_conf[:, 0]


def _prepare_token_leaves(per_frame_all_tokens, dpt_layers, device):
    """为 TTT 准备可优化的 token 叶子张量。

    对每帧、每个 dpt_layers 指定的层，克隆出一个 requires_grad 的 leaf（送至 device），
    同时保留一份冻结的原始副本用于正则项。返回
    (per_frame_leaf_tokens, all_leaves, all_originals)。
    """
    per_frame_leaf_tokens = []
    all_leaves = []
    all_originals = []
    for fi in range(len(per_frame_all_tokens)):
        tok_leaves = {}
        for li in dpt_layers:
            leaf = per_frame_all_tokens[fi][li].to(device).requires_grad_(True)
            tok_leaves[li] = leaf
            all_leaves.append(leaf)
            all_originals.append(per_frame_all_tokens[fi][li].to(device).clone())
        per_frame_leaf_tokens.append(tok_leaves)
    return per_frame_leaf_tokens, all_leaves, all_originals


def _build_agg_tokens(per_frame_all_tokens, per_frame_leaf_tokens, fi: int):
    """组装单帧 fi 的逐层 agg tokens 列表。

    在 per_frame_leaf_tokens[fi] 中的层取可优化的 leaf 张量，其余层取原始 token，
    从而让该帧前向时仅 dpt_layers 参与梯度回传。
    """
    n_layers = len(per_frame_all_tokens[fi])
    agg_tokens = []
    for li in range(n_layers):
        if li in per_frame_leaf_tokens[fi]:
            agg_tokens.append(per_frame_leaf_tokens[fi][li])
        else:
            agg_tokens.append(per_frame_all_tokens[fi][li])
    return agg_tokens


def _collect_geo_observations(
    segment_input: SegmentInput,
    frames: List[dict],
    dom_image: Optional[torch.Tensor],
    per_frame_pose: List[torch.Tensor],
    roma_model: Any,
    save_vis_dir: Optional[str],
    geo_consist_stride: int,
    geo_consist_max_corr: int,
    project_fn: Any,
    inv_project_fn: Any,
    geo_elev: Any,
    dom_transform: Any,
    device: torch.device,
    overlap_len: int = 0,
    crop_transform: Optional[LocalModelTransform] = None,
    crop_transform_source: str = "propagated",
    overlap_prior_cache_reliable_only: bool = False,
    overlap_prior_cache_max_xy_m: float = 25.0,
    overlap_prior_pose_from_cache: bool = True,
    overlap_prior_pose_min_frames: int = 1,
    target_mode: str = "auto",
    dom_point_pose_crop_scale: float = 1.4,
    semantic_masks_enable: bool = True,
    disable_pnp: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """收集一段内逐帧的 DOM/geo 观测，是 geo_v3 地理监督的核心采集步骤。

    总体流程：对该段每一帧，按当前位姿估计在 DOM 正射影像上裁剪出对应窗口，用
    RoMa（roma_model）做帧图 <-> DOM 像素稠密匹配；再借 dom_transform / geo_elev /
    inv_project_fn 把匹配到的 DOM 像素抬升为 ENU 3D 点，进而解出该帧的相机 ENU 目标、
    地面点、（可选）固定旋转 PnP、heading 等观测。pointmap_sim3 分支将 disable_pnp
    置为 True：PnP 完全不执行，相机位置只由窗口级 model-point→ENU Sim(3) 传递。

    关键行为：
    - crop 中心由 crop_transform（重叠传播或先验）给出；不同数据集（airzoo/uavscene 等）
      的 GT 先验在 metadata 中按严格策略受控使用——method 输入仅限首帧 GT 位姿/锚点。
    - overlap_len>0 时，头尾重叠帧的观测可写入跨段共享缓存 geo_observation_cache_global，
      供相邻段复用 / 一致性检查（受 overlap_prior_cache_* 系列参数门控）。
    - target_mode 决定相机目标语义（见 _geo_camera_target_enu）。

    返回：``{local_frame_idx: corr_dict}``。无 roma_model / 无 image_paths / 无 dom_image
    时返回空字典。corr_dict 内含该帧的像素、ENU 点、相机/地面目标、PnP、诊断等字段。
    """
    if roma_model is None or segment_input.image_paths is None:
        return {}
    from streamvggt.utils.dom_matching import get_frame_dom_correspondences
    from PIL import Image as PILImage
    if dom_image is None:
        return {}
    dom_cpu = dom_image[0].detach().cpu() if dom_image.dim() == 4 else dom_image.detach().cpu()

    _meta = segment_input.metadata or {}
    _dataset = str(_meta.get("dataset", "")).lower()
    _airzoo_gt_enu = _meta.get("airzoo_gt_enu")
    _airzoo_use_gnss_crop_center = bool(
        _meta.get("airzoo_use_gnss_crop_center", False))
    _airzoo_intr = _meta.get("airzoo_intrinsics")
    _airzoo_default_intr = _meta.get("airzoo_default_intrinsic")
    _strict_dom_crop = bool(_meta.get("strict_dom_crop", False))
    _uavscene_gt_c2w = _meta.get("uavscene_gt_c2w_enu")
    _uavscene_gt_c2w_diag = _uavscene_gt_c2w
    _debug_allow_gt_pose_crop = bool(_meta.get("debug_allow_gt_pose_crop", False))
    if _dataset == "uavscene" and not _debug_allow_gt_pose_crop:
        if _strict_dom_crop or _uavscene_gt_c2w is not None or _airzoo_gt_enu is not None:
            print(
                "[GeoV3][DOM] ignoring UAVScene per-frame GT pose/z priors; "
                "method input is restricted to the first-frame GT pose/anchor"
            )
        _strict_dom_crop = False
        _uavscene_gt_c2w = None
        _airzoo_gt_enu = None
        _airzoo_use_gnss_crop_center = False
    _geo_observation_cache_global = _meta.get("geo_observation_cache_global")
    if not isinstance(_geo_observation_cache_global, dict):
        _geo_observation_cache_global = None
    _target_diag_gt_rows: List[Dict[str, float]] = []

    def _gt_camera_enu_for_global_frame(global_frame_idx: int) -> Optional[np.ndarray]:
        try:
            if _uavscene_gt_c2w_diag is not None:
                _gt_pose_arr = np.asarray(_uavscene_gt_c2w_diag, dtype=np.float64)
                if 0 <= int(global_frame_idx) < len(_gt_pose_arr):
                    _pose = _gt_pose_arr[int(global_frame_idx)]
                    if _pose.shape == (4, 4) and np.isfinite(_pose[:3, 3]).all():
                        return _pose[:3, 3].copy()
            if _airzoo_gt_enu is not None:
                _gt_arr = np.asarray(_airzoo_gt_enu, dtype=np.float64)
                if 0 <= int(global_frame_idx) < len(_gt_arr):
                    _gt_vec = np.asarray(_gt_arr[int(global_frame_idx)], dtype=np.float64).reshape(-1)
                    if _gt_vec.size >= 3 and np.isfinite(_gt_vec[:3]).all():
                        return _gt_vec[:3].copy()
        except Exception:
            return None
        return None

    def _xy_err_to_gt(value: Any, gt_vec: np.ndarray) -> Optional[float]:
        try:
            _arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if _arr.size >= 2 and np.isfinite(_arr[:2]).all():
                return float(np.linalg.norm(_arr[:2] - gt_vec[:2]))
        except Exception:
            pass
        return None
    _cache_overlap_geo_observations = bool(_meta.get("cache_overlap_geo_observations", True))
    _camera_fov_by_path = _meta.get("camera_fov_by_path") or {}
    _gt_first_R = getattr(segment_input, "gt_rotation", None)
    _img_hw_cache: Dict[str, Tuple[int, int]] = {}

    def _resolve_camera_fov(query_path: str, agl_m: float) -> Dict[str, float]:
        """Resolve camera intrinsics from query image size for AirZoo; fallback to legacy constants."""
        if isinstance(_camera_fov_by_path, dict):
            _meta_fov = _camera_fov_by_path.get(query_path)
            if _meta_fov is None:
                _meta_fov = _camera_fov_by_path.get(os.path.basename(str(query_path)))
            if _meta_fov is not None:
                _fov = dict(_meta_fov)
                _fov["altitude"] = float(agl_m)
                if not _fov.get("orig_w") or not _fov.get("orig_h"):
                    try:
                        with PILImage.open(query_path) as _im:
                            _fov["orig_w"], _fov["orig_h"] = _im.size
                    except Exception:
                        pass
                return _fov
        if isinstance(_airzoo_intr, dict):
            try:
                if query_path in _img_hw_cache:
                    h_img, w_img = _img_hw_cache[query_path]
                else:
                    with PILImage.open(query_path) as _im:
                        w_img, h_img = _im.size
                    _img_hw_cache[query_path] = (h_img, w_img)
                _K = _airzoo_intr.get((h_img, w_img), _airzoo_default_intr)
                if _K is not None:
                    return {
                        "fx": float(_K[0, 0]),
                        "fy": float(_K[1, 1]),
                        "cx": float(_K[0, 2]),
                        "cy": float(_K[1, 2]),
                        "orig_w": int(w_img),
                        "orig_h": int(h_img),
                        "altitude": float(agl_m),
                    }
            except Exception:
                pass
        return {
            "fx": 1931.7,
            "fy": 1931.7,
            "cx": 800.0,
            "cy": 600.0,
            "orig_w": 1600,
            "orig_h": 1200,
            "altitude": float(agl_m),
        }

    # Determine a single approx ENU for the DOM crop center (same for all frames).
    # Segment 0: use anchor_world_xyz (first frame GT).
    # Segment k>0: prefer prev_summary.boundary_end_enu (DOM-estimated from previous segment).
    # A 250m crop is more than wide enough to cover a 32-frame UAV segment.
    _seg_anchor_enu: Optional[np.ndarray] = None
    _prev_sum = segment_input.prev_summary
    if _prev_sum is not None and _prev_sum.get("boundary_end_enu") is not None:
        _a = np.array(_prev_sum["boundary_end_enu"], dtype=np.float64).flatten()
        if len(_a) >= 2:
            _seg_anchor_enu = _a[:3] if len(_a) >= 3 else np.append(_a, 0.0)
    if _seg_anchor_enu is None and segment_input.anchor_world_xyz is not None:
        _awxyz = segment_input.anchor_world_xyz
        if torch.is_tensor(_awxyz):
            _awxyz = _awxyz.detach().cpu().numpy()
        _a = np.array(_awxyz, dtype=np.float64).flatten()
        if len(_a) >= 2:
            _seg_anchor_enu = _a[:3] if len(_a) >= 3 else np.append(_a, 0.0)
    if _seg_anchor_enu is None and crop_transform is not None and per_frame_pose:
        try:
            _mc0 = pose_enc_to_camera_centers(per_frame_pose[0])
            if _mc0.ndim > 1:
                _mc0 = _mc0[0]
            _mc0_t = _mc0[:3].detach().to(device=crop_transform.s.device, dtype=crop_transform.s.dtype).view(1, 3)
            _seg_anchor_enu = crop_transform.model_to_world(_mc0_t)[0].detach().cpu().numpy().astype(np.float64)
        except Exception:
            _seg_anchor_enu = None
    if _seg_anchor_enu is None:
        print(f"[GeoV3][DOM] seg {segment_input.segment_id}: no anchor_enu, skipping DOM search")
        return {}
    approx_enu_base = _seg_anchor_enu.copy()

    # Compute anchor AGL dynamically from DEM + GT flight altitude
    _anchor_agl = 435.0  # fallback
    if geo_elev is not None:
        try:
            _ground_alt = geo_elev.elevation_at_lonlat(geo_elev.lon0, geo_elev.lat0)
            _anchor_agl = max(50.0, geo_elev.alt0 - _ground_alt)  # clamp to sensible min
        except Exception:
            pass

    _R_c2w_model_list: List[Optional[np.ndarray]] = [None] * len(per_frame_pose)
    _R_model_to_enu: Optional[np.ndarray] = None
    _R_model_to_enu_source = "none"
    _dom_points_mode_for_crop = (
        _normalize_geo_target_mode(
            target_mode,
            prefer_ground_signal=False,
            prefer_pnp=False,
            allow_raw_pnp=False,
        ) == "dom_points"
    )
    try:
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        for _pose_idx, _pose_fi in enumerate(per_frame_pose):
            _extri, _ = pose_encoding_to_extri_intri(
                _pose_fi, image_size_hw=(1, 1), build_intrinsics=False)
            _R_w2c = _extri[..., :3, :3]
            if _R_w2c.dim() > 2:
                _R_w2c = _R_w2c.reshape(-1, 3, 3)[0]
            _R_c2w_model_list[_pose_idx] = _R_w2c.detach().cpu().float().numpy().T

        if segment_input.prev_transform is not None:
            _prev_R = segment_input.prev_transform.R.detach().cpu().float().numpy()
            # After a hard reset the model's internal frame 0 may have a different
            # orientation than the previous segment's frame 0.  We apply the same
            # correction as the seg-0 GT formula:
            #   R_model_to_enu = prev_transform.R @ R_c2w_model_0.T
            # This maps the current segment's model vectors to ENU via the previous
            # segment's transform, after aligning the current model's frame-0 orientation.
            if _R_c2w_model_list and _R_c2w_model_list[0] is not None:
                _R_model_to_enu = _prev_R @ _R_c2w_model_list[0].T.astype(_prev_R.dtype)
                _R_model_to_enu_source = "prev_transform_frame0"
            else:
                _R_model_to_enu = _prev_R
                _R_model_to_enu_source = "prev_transform"
        elif _gt_first_R is not None and _R_c2w_model_list and _R_c2w_model_list[0] is not None:
            _R_gt = np.asarray(_gt_first_R, dtype=np.float64)
            _R_model_to_enu = _R_gt @ _R_c2w_model_list[0].T
            _R_model_to_enu_source = "gt_first"
        if (
            crop_transform is not None
            and str(crop_transform_source) != "overlap_geo_prior"
        ):
            _R_model_to_enu = crop_transform.R.detach().cpu().float().numpy().astype(np.float64)
            _R_model_to_enu_source = str(crop_transform_source or "crop_transform")
        if (
            crop_transform is not None
            and str(crop_transform_source) == "overlap_geo_prior"
            and _dom_points_mode_for_crop
        ):
            _R_model_to_enu = crop_transform.R.detach().cpu().float().numpy().astype(np.float64)
            _R_model_to_enu_source = "overlap_geo_prior"
        if (
            crop_transform is not None
            and str(crop_transform_source) == "overlap_geo_prior"
            and bool(overlap_prior_pose_from_cache)
            and _geo_observation_cache_global is not None
            and _R_c2w_model_list
        ):
            _overlap_pose_len = max(0, min(int(overlap_len), len(_R_c2w_model_list)))
            _min_pose_frames = max(1, int(overlap_prior_pose_min_frames))
            _preferred_rots: List[np.ndarray] = []
            _fallback_rots: List[np.ndarray] = []
            for _fi_pose in range(_overlap_pose_len):
                _R_m = _R_c2w_model_list[_fi_pose]
                if _R_m is None:
                    continue
                _corr_cached = _geo_observation_cache_global.get(int(segment_input.start_frame) + int(_fi_pose))
                if not isinstance(_corr_cached, dict):
                    continue
                # The standalone point-map branch must not inherit a cached
                # PnP rotation, even when the cache was produced by a legacy
                # run.  Its crop orientation is estimated from the model/map
                # transform only.
                if bool(disable_pnp):
                    continue
                _pnp_cached = _corr_cached.get("pnp_result")
                if not isinstance(_pnp_cached, dict):
                    continue
                _R_cached = _nearest_rotation_np(_pnp_cached.get("R_c2w"))
                _R_model = _nearest_rotation_np(_R_m)
                if _R_cached is None or _R_model is None:
                    continue
                _R_candidate = _R_cached @ _R_model.T
                _crop_info_cached = _corr_cached.get("dom_crop_info") or {}
                _is_preferred = (
                    _pnp_cached.get("method") == "fixed_rotation_center"
                    and _crop_info_cached.get("mode") == "render"
                )
                if _is_preferred:
                    _preferred_rots.append(_R_candidate)
                else:
                    _fallback_rots.append(_R_candidate)
            _pose_rots = _preferred_rots
            if len(_pose_rots) >= _min_pose_frames:
                _R_overlap_pose = _average_rotation_np(_pose_rots)
                if _R_overlap_pose is not None:
                    _R_model_to_enu = _R_overlap_pose
                    _R_model_to_enu_source = f"overlap_pose_cache:{len(_pose_rots)}"
                    print(
                        f"[GeoV3][DOM] seg {segment_input.segment_id}: "
                        f"using overlap cached pose orientation n={len(_pose_rots)} "
                        f"preferred={len(_preferred_rots)}"
                    )
    except Exception as _e:
        print(f"[GeoV3][DOM] per-frame pose rotation estimation failed ({_e}), no pose crop")
        _R_c2w_model_list = [None] * len(per_frame_pose)
        _R_model_to_enu = None
        _R_model_to_enu_source = "none"

    # Build per-frame approx ENU using frozen model-space centers.
    # For each frame fi: approx_enu[fi] = anchor_enu + s * delta_mk[fi] @ R.T
    # where delta_mk[fi] = center_mk[fi] - center_mk[0] is the relative model displacement.
    # R and s are taken from prev_transform, but after a hard reset the current
    # model frame-0 orientation may differ from the previous segment.  Use the
    # corrected current M_k -> ENU rotation when available.  For seg 0 (no
    # prev_transform), fall back to an AGL-scale estimate.
    _per_frame_approx_enu: List[np.ndarray] = []
    _mc_arr_for_crop: Optional[np.ndarray] = None
    try:
        _mc_list = []
        for _pose_fi in per_frame_pose:
            _mc = pose_enc_to_camera_centers(_pose_fi)
            if _mc.ndim > 1:
                _mc = _mc[0]
            _mc_list.append(_mc.detach().cpu().float().numpy())
        _mc_arr = np.array(_mc_list, dtype=np.float64)  # (N, 3)
        _mc_arr_for_crop = _mc_arr
        _delta_mk = _mc_arr - _mc_arr[0]  # (N, 3) relative to frame 0

        _prev_transform = segment_input.prev_transform
        if crop_transform is not None:
            _mc_t = torch.tensor(
                _mc_arr[:, :3], device=crop_transform.s.device,
                dtype=crop_transform.s.dtype)
            _enu_arr = crop_transform.model_to_world(_mc_t).detach().cpu().numpy().astype(np.float64)
            for _fi in range(len(per_frame_pose)):
                _per_frame_approx_enu.append(_enu_arr[_fi, :3].copy())
        elif _prev_transform is not None:
            _s = float(_prev_transform.s.detach().cpu().float())
            if _R_model_to_enu is not None:
                _R = np.asarray(_R_model_to_enu, dtype=np.float64)  # current M_k -> ENU
            else:
                _R = _prev_transform.R.detach().cpu().float().numpy()  # fallback
            # delta_enu ≈ s * delta_mk @ R.T  (row-vector convention matches model_to_world)
            _delta_enu = _s * (_delta_mk @ _R.T)  # (N, 3)
        else:
            # Seg 0: no prev transform.  Use AGL-derived rough scale.
            _scale_est = _anchor_agl / 500.0  # empirical: model coords ≈ 1 unit per ~500m AGL
            if _R_model_to_enu is not None:
                _delta_enu = _scale_est * (_delta_mk @ _R_model_to_enu.T)
            else:
                _delta_enu = _scale_est * _delta_mk

        if crop_transform is None:
            for _fi in range(len(per_frame_pose)):
                _enu_fi = approx_enu_base.copy()
                _enu_fi[:3] = approx_enu_base[:3] + _delta_enu[_fi, :3]
                _per_frame_approx_enu.append(_enu_fi)
    except Exception as _e:
        print(f"[GeoV3][DOM] per-frame ENU estimation failed ({_e}), using anchor for all frames")
        _per_frame_approx_enu = [approx_enu_base.copy() for _ in range(len(per_frame_pose))]

    # --- Per-frame yaw (deg, 0=north, CW+) in ENU, used to pre-rotate DOM crop
    # so that its "up" aligns with query image "up".  Sources:
    #   seg 0 (no prev_transform): use gt_first_frame_rotation as R_model->enu anchor
    #                              (via R_model->enu = R_gt_first @ R_c2w_model[0].T)
    #   seg k>=1: reuse prev_transform.R as R_model->enu
    # yaw per frame: up_enu = R_model->enu @ R_c2w_model[fi] @ [0,-1,0]
    #               yaw_deg = atan2(up_enu[0], up_enu[1])
    _per_frame_yaw_deg: List[Optional[float]] = [None] * len(per_frame_pose)
    _per_frame_R_c2w_enu: List[Optional[np.ndarray]] = [None] * len(per_frame_pose)
    try:
        if _R_model_to_enu is not None:
            _up_cam = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            for _fi, _R_c2w_m in enumerate(_R_c2w_model_list):
                if _R_c2w_m is None:
                    continue
                _R_c2w_enu = _R_model_to_enu @ _R_c2w_m
                _per_frame_R_c2w_enu[_fi] = _R_c2w_enu.astype(np.float64)
                _up_enu = _R_c2w_enu @ _up_cam
                _per_frame_yaw_deg[_fi] = float(
                    np.degrees(np.arctan2(_up_enu[0], _up_enu[1])))
    except Exception as _e:
        print(f"[GeoV3][DOM] per-frame yaw estimation failed ({_e}), no crop rotation")

    observations: Dict[int, Dict[str, Any]] = {}
    print(
        f"[GeoV3][DOM] save_vis_dir={save_vis_dir}"
        f" image_paths={len(segment_input.image_paths) if segment_input.image_paths is not None else 0}"
        f" start_frame={segment_input.start_frame} anchor_enu={approx_enu_base[:2].tolist()}"
        f" anchor_agl={_anchor_agl:.1f}m"
        f" yaw_src={_R_model_to_enu_source}"
        f" crop_prior={crop_transform_source if crop_transform is not None else 'propagated'}"
        f" gnss_crop_center={_airzoo_use_gnss_crop_center}"
    )
    _running_agl = _anchor_agl  # updated from GT/PnP/metric-crop AGL estimates
    _overlap_len_eff = max(0, min(int(overlap_len), len(frames)))
    _online_crop_scale: Optional[float] = None
    _online_crop_t: Optional[np.ndarray] = None
    _online_crop_R: Optional[np.ndarray] = None
    _overlap_handoff_frame: Optional[int] = None
    _online_crop_pnp_frames: List[int] = []
    _online_crop_pnp_centers: List[np.ndarray] = []
    _dom_points_mode = _dom_points_mode_for_crop
    _prev_overlap_R_c2w_enu_by_fi: Dict[int, np.ndarray] = {}

    def _build_prev_overlap_exact_crop_rotations() -> Dict[int, np.ndarray]:
        if not (
            _dom_points_mode
            and crop_transform is not None
            and str(crop_transform_source) == "overlap_geo_prior"
            and _overlap_len_eff > 0
        ):
            return {}
        prev_summary = segment_input.prev_summary or {}
        if not isinstance(prev_summary, dict):
            return {}
        prev_metadata = prev_summary.get("metadata")
        if not isinstance(prev_metadata, dict):
            prev_metadata = {}
        prev_R_model_arr = prev_summary.get("per_frame_R_c2w_model")
        if prev_R_model_arr is None:
            prev_R_model_arr = prev_metadata.get("per_frame_R_c2w_model")
        if prev_R_model_arr is None:
            return {}
        try:
            prev_R_model_arr = np.asarray(prev_R_model_arr, dtype=np.float64)
        except Exception:
            return {}
        if prev_R_model_arr.ndim != 3 or prev_R_model_arr.shape[1:] != (3, 3):
            return {}
        prev_R_model_to_enu = prev_summary.get("transform_R")
        if prev_R_model_to_enu is None:
            prev_R_model_to_enu = prev_metadata.get("transform_R")
        if prev_R_model_to_enu is None and segment_input.prev_transform is not None:
            prev_R_model_to_enu = segment_input.prev_transform.R.detach().cpu().float().numpy()
        prev_R_model_to_enu = _nearest_rotation_np(prev_R_model_to_enu)
        if prev_R_model_to_enu is None:
            return {}
        try:
            prev_start = int(prev_summary.get("start_frame"))
        except Exception:
            prev_start = None
        prev_count = int(prev_R_model_arr.shape[0])
        result: Dict[int, np.ndarray] = {}
        for _fi in range(_overlap_len_eff):
            if prev_start is not None:
                prev_idx = int(segment_input.start_frame) + int(_fi) - prev_start
            else:
                prev_idx = prev_count - int(_overlap_len_eff) + int(_fi)
            if prev_idx < 0 or prev_idx >= prev_count:
                continue
            R_prev_model = _nearest_rotation_np(prev_R_model_arr[prev_idx])
            if R_prev_model is None:
                continue
            result[int(_fi)] = (prev_R_model_to_enu @ R_prev_model).astype(np.float64)
        if result:
            print(
                f"[GeoV3][DOM] seg {segment_input.segment_id}: "
                f"using prev exact overlap crop rotations n={len(result)}"
            )
        return result

    _prev_overlap_R_c2w_enu_by_fi = _build_prev_overlap_exact_crop_rotations()

    def _sanitize_prev_overlap_crop_corr(
        corr: Optional[Dict[str, Any]],
        frame_idx: int,
        used_prev_overlap_exact: bool,
    ) -> Optional[Dict[str, Any]]:
        if corr is None or not bool(used_prev_overlap_exact):
            return corr
        if not isinstance(corr, dict):
            return corr
        corr["overlap_exact_crop_rotation"] = True
        corr["overlap_exact_crop_rotation_source"] = "prev_segment"
        crop_info = corr.setdefault("dom_crop_info", {})
        if isinstance(crop_info, dict):
            crop_info["rotation_source"] = "prev_overlap_exact"
            crop_info["pnp_disabled_reason"] = "prev_overlap_exact_crop_rotation"
        if "pnp_result" in corr or "pnp_rotation_check" in corr:
            corr.pop("pnp_result", None)
            corr.pop("pnp_rotation_check", None)
            corr["pnp_result_disabled_reason"] = "prev_overlap_exact_crop_rotation"
        return corr

    _enable_overlap_crop_handoff = (
        crop_transform is not None
        and str(crop_transform_source) == "overlap_geo_prior"
        and _mc_arr_for_crop is not None
        and not _airzoo_use_gnss_crop_center
    )
    _enable_online_crop_bootstrap = (
        crop_transform is None
        and segment_input.prev_transform is None
        and _mc_arr_for_crop is not None
        and _R_model_to_enu is not None
        and not _airzoo_use_gnss_crop_center
    )

    def _online_crop_approx_enu(frame_idx: int) -> Optional[np.ndarray]:
        if (_online_crop_scale is None or _online_crop_t is None
                or _mc_arr_for_crop is None):
            return None
        if frame_idx < 0 or frame_idx >= len(_mc_arr_for_crop):
            return None
        _R_online = _online_crop_R if _online_crop_R is not None else _R_model_to_enu
        if _R_online is None:
            return None
        _model_xyz = np.asarray(_mc_arr_for_crop[frame_idx, :3], dtype=np.float64)
        _enu = float(_online_crop_scale) * (_model_xyz @ np.asarray(_R_online, dtype=np.float64).T) + _online_crop_t
        if not np.all(np.isfinite(_enu[:3])):
            return None
        return _enu.astype(np.float64)

    def _update_online_crop_bootstrap(frame_idx: int, corr: Optional[Dict[str, Any]]) -> None:
        nonlocal _online_crop_scale, _online_crop_t, _online_crop_R, _overlap_handoff_frame
        if corr is None:
            return
        if _dom_points_mode:
            return
        _center = _fixed_rotation_pnp_center(corr, min_dim=3)
        if (
            _enable_overlap_crop_handoff
            and _overlap_handoff_frame is None
            and frame_idx >= _overlap_len_eff
            and crop_transform is not None
            and _mc_arr_for_crop is not None
            and _center is not None
        ):
            _R_crop = crop_transform.R.detach().cpu().float().numpy().astype(np.float64)
            _s_crop = float(crop_transform.s.detach().cpu().float())
            _t_crop = crop_transform.t.detach().cpu().float().numpy().astype(np.float64)
            _model_xyz = np.asarray(_mc_arr_for_crop[frame_idx, :3], dtype=np.float64)
            _center_arr = np.asarray(_center[:3], dtype=np.float64)
            if np.all(np.isfinite(_model_xyz[:3])) and np.all(np.isfinite(_center_arr[:3])) and _s_crop > 0.0:
                _prior_enu = _s_crop * (_model_xyz @ _R_crop.T) + _t_crop
                _online_crop_scale = _s_crop
                _online_crop_R = _R_crop
                _online_crop_t = _center_arr - _s_crop * (_model_xyz @ _R_crop.T)
                _overlap_handoff_frame = int(frame_idx)
                _dxy = float(np.linalg.norm(_center_arr[:2] - _prior_enu[:2]))
                print(
                    f"[GeoV3][OverlapBootstrap] seg={segment_input.segment_id} "
                    f"first_new_dom_frame={frame_idx} crop_handoff "
                    f"dxy_prior={_dxy:.2f}m s={_s_crop:.2f}"
                )
                return
        if not _enable_online_crop_bootstrap:
            return
        if _mc_arr_for_crop is None or _R_model_to_enu is None:
            return
        if frame_idx < 0 or frame_idx >= len(_mc_arr_for_crop):
            return
        if _center is None:
            return
        if frame_idx in _online_crop_pnp_frames:
            return
        _online_crop_pnp_frames.append(int(frame_idx))
        _online_crop_pnp_centers.append(np.asarray(_center[:3], dtype=np.float64))
        if len(_online_crop_pnp_frames) < 2:
            return

        _R = np.asarray(_R_model_to_enu, dtype=np.float64)
        _pred_diffs: List[np.ndarray] = []
        _obs_diffs: List[np.ndarray] = []
        for _ia in range(len(_online_crop_pnp_frames)):
            _fa = _online_crop_pnp_frames[_ia]
            _ma = np.asarray(_mc_arr_for_crop[_fa, :3], dtype=np.float64)
            _ca = _online_crop_pnp_centers[_ia]
            for _ib in range(_ia + 1, len(_online_crop_pnp_frames)):
                _fb = _online_crop_pnp_frames[_ib]
                _mb = np.asarray(_mc_arr_for_crop[_fb, :3], dtype=np.float64)
                _cb = _online_crop_pnp_centers[_ib]
                _pred = ((_mb - _ma) @ _R.T)[:2]
                _obs = (_cb - _ca)[:2]
                if not (np.all(np.isfinite(_pred)) and np.all(np.isfinite(_obs))):
                    continue
                if float(np.linalg.norm(_pred)) < 1e-5 or float(np.linalg.norm(_obs)) < 0.25:
                    continue
                _pred_diffs.append(_pred.astype(np.float64))
                _obs_diffs.append(_obs.astype(np.float64))
        if not _pred_diffs:
            return
        _pred_arr = np.concatenate(_pred_diffs, axis=0)
        _obs_arr = np.concatenate(_obs_diffs, axis=0)
        _den = float(np.dot(_pred_arr, _pred_arr))
        if _den <= 1e-12:
            return
        _scale = float(np.dot(_pred_arr, _obs_arr) / _den)
        if not np.isfinite(_scale) or _scale <= 0.0 or _scale > 10000.0:
            return
        _t = _anchor_aligned_t(
            approx_enu_base,
            np.asarray(_mc_arr_for_crop[0, :3], dtype=np.float32),
            _scale,
            np.asarray(_R_model_to_enu, dtype=np.float32),
        ).astype(np.float64)
        _old = _online_crop_scale
        _online_crop_scale = _scale
        _online_crop_t = _t
        _online_crop_R = np.asarray(_R_model_to_enu, dtype=np.float64)
        _old_s = "None" if _old is None else f"{float(_old):.2f}"
        print(
            f"[GeoV3][CropBootstrap] seg={segment_input.segment_id} frame={frame_idx} "
            f"scale={_scale:.2f} old={_old_s} "
            f"pnp_frames={len(_online_crop_pnp_frames)}"
        )

    def _overlap_pair_vis_paths(frame_idx: int) -> List[str]:
        if save_vis_dir is None or _overlap_len_eff <= 0 or segment_input.image_paths is None:
            return []
        _paths: List[str] = []
        _seg_id = int(segment_input.segment_id)
        _gfi = int(segment_input.start_frame) + int(frame_idx)
        _query_path = segment_input.image_paths[segment_input.start_frame + frame_idx]
        _stem = os.path.splitext(os.path.basename(_query_path))[0]
        _prev_sum_local = segment_input.prev_summary or {}
        if frame_idx < _overlap_len_eff and _prev_sum_local:
            _prev_id = int(_prev_sum_local.get("segment_id", _seg_id - 1))
            _pair_dir = os.path.join(save_vis_dir, "overlap_pairs", f"seg{_prev_id:02d}_{_seg_id:02d}")
            _paths.append(os.path.join(
                _pair_dir,
                f"g{_gfi:06d}_{_stem}_next_seg{_seg_id:02d}_f{frame_idx:04d}.jpg",
            ))
        if (frame_idx >= len(frames) - _overlap_len_eff
                and segment_input.end_frame < len(segment_input.image_paths)):
            _next_id = _seg_id + 1
            _pair_dir = os.path.join(save_vis_dir, "overlap_pairs", f"seg{_seg_id:02d}_{_next_id:02d}")
            _paths.append(os.path.join(
                _pair_dir,
                f"g{_gfi:06d}_{_stem}_prev_seg{_seg_id:02d}_f{frame_idx:04d}.jpg",
            ))
        return _paths

    def _overlap_pair_cached_alias_pairs(frame_idx: int) -> List[Tuple[str, str]]:
        if save_vis_dir is None or _overlap_len_eff <= 0 or segment_input.image_paths is None:
            return []
        _seg_id = int(segment_input.segment_id)
        _prev_sum_local = segment_input.prev_summary or {}
        if frame_idx >= _overlap_len_eff or not _prev_sum_local:
            return []
        _gfi = int(segment_input.start_frame) + int(frame_idx)
        _query_path = segment_input.image_paths[segment_input.start_frame + frame_idx]
        _stem = os.path.splitext(os.path.basename(_query_path))[0]
        _prev_id = int(_prev_sum_local.get("segment_id", _seg_id - 1))
        _prev_start = int(_prev_sum_local.get("start_frame", segment_input.start_frame - _overlap_len_eff))
        _prev_local_idx = _gfi - _prev_start
        if _prev_local_idx < 0:
            return []
        _pair_dir = os.path.join(save_vis_dir, "overlap_pairs", f"seg{_prev_id:02d}_{_seg_id:02d}")
        _src = os.path.join(
            _pair_dir,
            f"g{_gfi:06d}_{_stem}_prev_seg{_prev_id:02d}_f{_prev_local_idx:04d}.jpg",
        )
        _dst = os.path.join(
            _pair_dir,
            f"g{_gfi:06d}_{_stem}_next_seg{_seg_id:02d}_f{frame_idx:04d}.jpg",
        )
        return [(_src, _dst)]

    def _materialize_overlap_pair_cached_aliases(frame_idx: int) -> None:
        for _src, _dst in _overlap_pair_cached_alias_pairs(frame_idx):
            if os.path.exists(_dst):
                continue
            if not os.path.exists(_src):
                print(f"[GeoV3][DOM] overlap_vis_alias missing source src={_src} dst={_dst}")
                continue
            try:
                os.makedirs(os.path.dirname(_dst), exist_ok=True)
                shutil.copyfile(_src, _dst)
                print(f"[GeoV3][DOM] overlap_vis_alias copy src={_src} dst={_dst}")
            except OSError as _e:
                print(
                    f"[GeoV3][DOM] overlap_vis_alias copy failed ({_e}), continuing "
                    f"src={_src} dst={_dst}"
                )

    def _ground_z_for_crop(cam_center_enu: np.ndarray, running_agl: float) -> float:
        _fallback_z = float(cam_center_enu[2] - running_agl)
        if geo_elev is not None and dom_transform is not None:
            try:
                _xyz = torch.tensor([[float(cam_center_enu[0]), float(cam_center_enu[1]), 0.0]])
                _uv = project_fn(_xyz)
                if torch.is_tensor(_uv):
                    _uv = _uv.detach().cpu().numpy()
                _uv = np.asarray(_uv, dtype=np.float64)[0]
                _, _, _ground_z = geo_elev.enu_3d_from_dom(float(_uv[0]), float(_uv[1]), dom_transform)
                _dem_agl = float(cam_center_enu[2] - _ground_z)
                _min_agl = max(10.0, 0.25 * float(running_agl))
                _max_agl = max(300.0, 4.0 * float(running_agl))
                if np.isfinite(_ground_z) and _min_agl <= _dem_agl <= _max_agl:
                    return float(_ground_z)
            except Exception:
                pass
        return _fallback_z

    def _camera_tilt_deg(R_c2w: np.ndarray) -> float:
        try:
            _fwd = np.asarray(R_c2w, dtype=np.float64) @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            _v = abs(float(_fwd[2]))
            _h = float(np.hypot(_fwd[0], _fwd[1]))
            return float(np.degrees(np.arctan2(_h, max(_v, 1e-8))))
        except Exception:
            return 0.0

    def _sanitize_crop_camera_center(
        cam_center_enu: np.ndarray,
        ground_z: float,
        running_agl: float,
        R_c2w: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        _cam = np.asarray(cam_center_enu, dtype=np.float64).copy()
        if _cam.shape[0] < 3 or not np.all(np.isfinite(_cam[:3])) or not np.isfinite(ground_z):
            return _cam
        _tilt = _camera_tilt_deg(R_c2w)
        # Only touch the oblique render/PnP path.  In this branch a corrupted
        # trajectory Z makes the rendered DOM FOV and PnP prior collapse; XY is
        # still taken from the propagated segment estimate.
        if _tilt <= 5.0:
            return _cam
        _target_agl = float(running_agl) if np.isfinite(running_agl) and running_agl > 20.0 else float(_anchor_agl)
        if not np.isfinite(_target_agl) or _target_agl <= 20.0:
            return _cam
        _agl = float(_cam[2] - ground_z)
        _min_agl = max(20.0, 0.50 * _target_agl)
        _max_agl = max(80.0, 2.00 * _target_agl)
        if (not np.isfinite(_agl)) or _agl < _min_agl or _agl > _max_agl:
            _old_agl = _agl
            _cam[2] = float(ground_z + _target_agl)
            print(
                f"[GeoV3][DOM] frame={frame_idx} crop_z_clamp "
                f"tilt={_tilt:.1f}deg agl={_old_agl:.1f}m -> {_target_agl:.1f}m"
            )
        return _cam

    for fi, frame in enumerate(frames):
        _gfi = int(segment_input.start_frame) + int(fi)
        _is_overlap_frame = (
            _overlap_len_eff > 0
            and (fi < _overlap_len_eff or fi >= len(frames) - _overlap_len_eff)
        )
        if fi % geo_consist_stride != 0 and fi != len(frames) - 1 and not _is_overlap_frame:
            continue
        approx_enu = _per_frame_approx_enu[fi] if fi < len(_per_frame_approx_enu) else approx_enu_base
        _crop_prior_frame_source = "precomputed"
        _online_enu = _online_crop_approx_enu(fi)
        if _online_enu is not None:
            approx_enu = _online_enu
            _crop_prior_frame_source = "online_pnp"
        _strict_R_c2w_enu = None
        _strict_pose_crop = False
        if _strict_dom_crop and _uavscene_gt_c2w is not None:
            try:
                _gt_pose_arr = np.asarray(_uavscene_gt_c2w, dtype=np.float64)
                if (0 <= _gfi < len(_gt_pose_arr)
                        and _gt_pose_arr[_gfi].shape == (4, 4)
                        and np.isfinite(_gt_pose_arr[_gfi]).all()):
                    approx_enu = _gt_pose_arr[_gfi, :3, 3].copy()
                    _strict_R_c2w_enu = _gt_pose_arr[_gfi, :3, :3].copy()
                    _strict_pose_crop = True
            except Exception as _e:
                print(f"[GeoV3][DOM] WARN: strict UAVScene crop prior failed for frame={fi}: {_e}")
        if _airzoo_use_gnss_crop_center and _airzoo_gt_enu is not None:
            try:
                _gt_arr = np.asarray(_airzoo_gt_enu, dtype=np.float64)
                if 0 <= _gfi < len(_gt_arr):
                    _gt_crop_center = np.asarray(_gt_arr[_gfi], dtype=np.float64).reshape(-1)
                    if _gt_crop_center.shape[0] >= 3 and np.all(np.isfinite(_gt_crop_center[:3])):
                        approx_enu = _gt_crop_center[:3].copy()
            except Exception as _e:
                print(f"[GeoV3][DOM] WARN: failed to use GNSS crop center for frame={fi}: {_e}")
        vis_path = None if save_vis_dir is None else f"{save_vis_dir}/seg{segment_input.segment_id:02d}_f{fi:04d}.jpg"
        _yaw_fi = _per_frame_yaw_deg[fi] if fi < len(_per_frame_yaw_deg) else None
        _yaw_src_fi = "model_pose" if _yaw_fi is not None else "none"
        _camera_fov_fi = _resolve_camera_fov(segment_input.image_paths[segment_input.start_frame + fi], _running_agl)
        _pred_pose_crop = None
        _R_c2w_enu_fi = _per_frame_R_c2w_enu[fi] if fi < len(_per_frame_R_c2w_enu) else None
        _used_prev_overlap_exact_crop_rotation = False
        _R_prev_overlap_exact = _prev_overlap_R_c2w_enu_by_fi.get(int(fi))
        if _R_prev_overlap_exact is not None:
            _R_c2w_enu_fi = _R_prev_overlap_exact
            _used_prev_overlap_exact_crop_rotation = True
            try:
                _up_enu_overlap = _R_prev_overlap_exact @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
                if np.all(np.isfinite(_up_enu_overlap[:2])) and float(np.linalg.norm(_up_enu_overlap[:2])) > 1e-8:
                    _yaw_fi = float(np.degrees(np.arctan2(_up_enu_overlap[0], _up_enu_overlap[1])))
                    _yaw_src_fi = "prev_overlap_exact"
            except Exception as _e:
                print(f"[GeoV3][DOM] WARN: prev overlap exact yaw failed for frame={fi}: {_e}")
        if _strict_R_c2w_enu is not None:
            _R_c2w_enu_fi = _strict_R_c2w_enu
            _used_prev_overlap_exact_crop_rotation = False
            try:
                _up_enu_strict = _strict_R_c2w_enu @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
                if np.all(np.isfinite(_up_enu_strict[:2])) and float(np.linalg.norm(_up_enu_strict[:2])) > 1e-8:
                    _yaw_fi = float(np.degrees(np.arctan2(_up_enu_strict[0], _up_enu_strict[1])))
                    _yaw_src_fi = "strict_pose"
            except Exception as _e:
                print(f"[GeoV3][DOM] WARN: strict yaw prior failed for frame={fi}: {_e}")
        if _R_c2w_enu_fi is not None:
            _cam_center_crop = np.asarray(approx_enu, dtype=np.float64).flatten()
            if _cam_center_crop.shape[0] >= 3 and np.all(np.isfinite(_cam_center_crop[:3])):
                _ground_z = _ground_z_for_crop(_cam_center_crop, _running_agl)
                if not _strict_pose_crop:
                    _cam_center_crop = _sanitize_crop_camera_center(
                        _cam_center_crop, _ground_z, _running_agl, _R_c2w_enu_fi, fi)
                if _strict_pose_crop:
                    _crop_scale = 1.30
                elif _dom_points_mode:
                    _crop_scale = float(dom_point_pose_crop_scale)
                    if not np.isfinite(_crop_scale) or _crop_scale <= 0.0:
                        _crop_scale = 1.4
                else:
                    _crop_scale = 1.15
                _min_half_size = 2.0 if _strict_pose_crop else 45.0
                _max_half_size = 600.0 if _strict_pose_crop else 300.0
                _max_center_shift = 2000.0 if _strict_pose_crop else 400.0
                _pred_pose_crop = {
                    "cam_center_enu": _cam_center_crop[:3],
                    "R_c2w": _R_c2w_enu_fi,
                    "ground_z": _ground_z,
                    "crop_scale": _crop_scale,
                    "min_half_size_m": _min_half_size,
                    "max_half_size_m": _max_half_size,
                    "max_center_shift_m": _max_center_shift,
                    "strict_footprint": bool(_strict_pose_crop),
                    "max_hit_dist_m": 2500.0 if _strict_pose_crop else None,
                    "disable_render": bool(_strict_pose_crop),
                    "render_min_valid_ratio": 0.30,
                }
        _crop_mode = "strict_pose" if _strict_pose_crop else ("pose" if _pred_pose_crop is not None else "nadir")
        _vis_extra = _overlap_pair_vis_paths(fi)
        print(f"[GeoV3][DOM] frame={fi} approx_enu={approx_enu[:2].tolist()} prior={_crop_prior_frame_source} yaw={None if _yaw_fi is None else f'{_yaw_fi:.1f}deg'} yaw_src={_yaw_src_fi} crop={_crop_mode} vis_path={vis_path}")
        _cached_corr = None
        if (_cache_overlap_geo_observations and _is_overlap_frame
                and _geo_observation_cache_global is not None):
            _cached_corr = _geo_observation_cache_global.get(_gfi)
        _dom_points_cache_mode = _normalize_geo_target_mode(target_mode) == "dom_points"
        if _cached_corr is not None and _dom_points_cache_mode:
            _cached_candidate = _clone_geo_corr_for_overlap_reuse(_cached_corr)
            _ok_cache, _cache_reason, _cache_info = _dom_point_observation_semantics(
                _cached_candidate, config=None, stage="cache_reuse")
            _cache_reliable_flag = (
                _cached_candidate.get("overlap_geo_cache_reliable")
                if isinstance(_cached_candidate, dict) else None
            )
            if (
                not _ok_cache
                or (
                    True
                    and _cache_reliable_flag is False
                )
            ):
                if isinstance(_cached_candidate, dict):
                    _cached_candidate.setdefault("dom_point_observation", {})["cache_reuse"] = {
                        "valid": False,
                        "reason": _cache_reason,
                        **_cache_info,
                    }
                print(
                    f"[GeoV3][DOM] frame={fi} global={_gfi} skip cached overlap geo observation "
                    f"reason={_cache_reason} reliable={_cache_reliable_flag}"
                )
                _cached_corr = None
        if _cached_corr is not None:
            corr = _clone_geo_corr_for_overlap_reuse(_cached_corr)
            corr = _sanitize_prev_overlap_crop_corr(corr, fi, _used_prev_overlap_exact_crop_rotation)
            if bool(disable_pnp) and isinstance(corr, dict):
                corr.pop("pnp_result", None)
                corr.pop("pnp_rotation_check", None)
                corr["pnp_disabled"] = True
                corr["pnp_result_disabled_reason"] = "pointmap_camera_transform"
            if isinstance(corr, dict):
                corr["overlap_geo_cache_reused"] = True
                if _dom_points_cache_mode:
                    _ok_cache, _cache_reason, _cache_info = _dom_point_observation_semantics(
                        corr, config=None, stage="cache_reuse")
                    corr.setdefault("dom_point_observation", {})["cache_reuse"] = {
                        "valid": bool(_ok_cache),
                        "reason": _cache_reason,
                        **_cache_info,
                    }
            _materialize_overlap_pair_cached_aliases(fi)
            print(f"[GeoV3][DOM] frame={fi} global={_gfi} reusing cached overlap geo observation")
        else:
            corr = get_frame_dom_correspondences(
                segment_input.image_paths[segment_input.start_frame + fi],
                dom_cpu,
                approx_enu,
                project_fn,
                inv_project_fn,
                roma_model,
                crop_size_m=250,
                device=device,
                max_correspondences=geo_consist_max_corr,
                camera_fov=_camera_fov_fi,
                geo_elev=geo_elev,
                dom_transform=dom_transform,
                save_vis=vis_path,
                save_vis_extra=_vis_extra,
                pred_pose_crop=_pred_pose_crop,
                query_yaw_deg=_yaw_fi,
                compute_pnp=not bool(disable_pnp),
            )
            corr = _sanitize_prev_overlap_crop_corr(corr, fi, _used_prev_overlap_exact_crop_rotation)
            if bool(disable_pnp) and isinstance(corr, dict):
                # Defensive cleanup also covers matcher implementations or
                # future cache paths that return a PnP field unexpectedly.
                corr.pop("pnp_result", None)
                corr.pop("pnp_rotation_check", None)
                corr["pnp_disabled"] = True
                corr["pnp_result_disabled_reason"] = "pointmap_camera_transform"
            if corr is not None and not semantic_masks_enable:
                # Propagate the ablation semantics into downstream point-edge,
                # bootstrap, TTT and PGO selection code.
                corr["semantic_masks_disabled"] = True
            if (corr is not None and _cache_overlap_geo_observations and _is_overlap_frame
                    and _geo_observation_cache_global is not None):
                _cache_reliable_only = bool(overlap_prior_cache_reliable_only) or (
                    _dom_points_cache_mode and True
                )
                _cache_xy_residual = None
                _cache_center = None
                _cache_center_source = "none"
                _cache_sem_ok = True
                _cache_sem_reason = "ok"
                _cache_sem_info: Dict[str, Any] = {}
                if _dom_points_cache_mode:
                    _cache_sem_ok, _cache_sem_reason, _cache_sem_info = _dom_point_observation_semantics(
                        corr, config=None, stage="overlap_cache")
                    _cache_center = _dom_point_target_xy_from_corr(corr)
                    _cache_center_source = "dom_points_frame_center"
                else:
                    _cache_center, _cache_center_source = _geo_camera_target_enu(
                        corr,
                        target_mode=target_mode,
                    )
                if _cache_center is not None:
                    _cache_xy_residual = float(np.linalg.norm(
                        np.asarray(_cache_center[:2], dtype=np.float64)
                        - np.asarray(approx_enu[:2], dtype=np.float64)))
                _cache_is_reliable = (
                    bool(_cache_sem_ok)
                    and _cache_xy_residual is not None
                    and np.isfinite(_cache_xy_residual)
                    and _cache_xy_residual <= float(overlap_prior_cache_max_xy_m)
                )
                if not _cache_sem_ok:
                    if isinstance(corr, dict):
                        corr.setdefault("dom_point_observation", {})["overlap_cache"] = {
                            "valid": False,
                            "reason": _cache_sem_reason,
                            **_cache_sem_info,
                        }
                    print(
                        f"[GeoV3][DOM] frame={fi} global={_gfi} skip overlap geo cache "
                        f"reason={_cache_sem_reason}"
                    )
                elif _cache_reliable_only and not _cache_is_reliable:
                    print(
                        f"[GeoV3][DOM] frame={fi} global={_gfi} skip overlap geo cache "
                        f"(source={_cache_center_source} xy_residual={_cache_xy_residual}m > "
                        f"{float(overlap_prior_cache_max_xy_m):.1f}m)"
                    )
                else:
                    if isinstance(corr, dict):
                        corr["overlap_geo_cache_reliable"] = bool(_cache_is_reliable)
                        corr["overlap_geo_cache_target_source"] = str(_cache_center_source)
                        if _cache_xy_residual is not None:
                            corr["overlap_geo_cache_xy_residual_m"] = float(_cache_xy_residual)
                        if _dom_points_cache_mode:
                            corr.setdefault("dom_point_observation", {})["overlap_cache"] = {
                                "valid": True,
                                "reason": "ok",
                                **_cache_sem_info,
                            }
                    _geo_observation_cache_global[_gfi] = _clone_geo_corr_for_overlap_reuse(corr)
                    print(f"[GeoV3][DOM] frame={fi} global={_gfi} cached overlap geo observation")
        if corr is not None:
            _crop_info = corr.get("dom_crop_info", {}) if isinstance(corr, dict) else {}
            if _crop_info:
                _half_w = _crop_info.get("pose_half_w_m")
                _half_h = _crop_info.get("pose_half_h_m")
                _shift = _crop_info.get("pose_center_shift_m")
                _agl = _crop_info.get("pose_agl_m")
                _half_w_s = "None" if _half_w is None else f"{_half_w:.1f}"
                _half_h_s = "None" if _half_h is None else f"{_half_h:.1f}"
                _shift_s = "None" if _shift is None else f"{_shift:.1f}m"
                _agl_s = "None" if _agl is None else f"{_agl:.1f}m"
                _pose_fail = _crop_info.get("pose_crop_failure_reason")
                _render_fail = _crop_info.get("render_failure_reason")
                _render_pitch = _crop_info.get("render_pitch_deg")
                _diag_tail = ""
                if _pose_fail is not None:
                    _diag_tail += f" pose_fail={_pose_fail}"
                if _render_fail is not None:
                    _diag_tail += f" render_fail={_render_fail}"
                if _render_pitch is not None:
                    _diag_tail += f" render_pitch={float(_render_pitch):.1f}deg"
                _render_valid = _crop_info.get("render_valid_ratio")
                if _render_valid is not None:
                    _diag_tail += f" render_valid={float(_render_valid):.2f}"
                print(
                    f"[GeoV3][DOM] frame={fi} matched crop={_crop_info.get('mode')} "
                    f"half_m=({_half_w_s},{_half_h_s}) "
                    f"center_shift={_shift_s} pose_agl={_agl_s} "
                    f"yaw_applied={float(_crop_info.get('yaw_applied_deg', 0.0)):.1f}deg"
                    f"{_diag_tail}"
                )
            _pnp = corr.get("pnp_result") if isinstance(corr, dict) else None
            if isinstance(_pnp, dict):
                _diag = _pnp.get("diagnostics")
                _method = _pnp.get("method", "free")
                if isinstance(_diag, dict):
                    print(
                        f"[GeoV3][PnPDiag] seg={segment_input.segment_id} frame={fi} "
                        f"method={_method} n={int(_pnp.get('n_inliers', 0))}/"
                        f"{int(_diag.get('n_valid', 0))} "
                        f"reproj={float(_diag.get('median_reproj_px', float('nan'))):.1f}px "
                        f"ray={float(_diag.get('median_ray_residual_m', float('nan'))):.1f}m "
                        f"pos={float(_diag.get('positive_depth_ratio', 0.0)):.2f} "
                        f"dxy_prior={float(_diag.get('delta_xy_m', float('nan'))):.1f}m "
                        f"dz_prior={float(_diag.get('delta_z_m', float('nan'))):.1f}m"
                    )
                else:
                    print(
                        f"[GeoV3][PnPDiag] seg={segment_input.segment_id} frame={fi} "
                        f"method={_method} n={int(_pnp.get('n_inliers', 0))}"
                    )
            _gt_vec_diag = _gt_camera_enu_for_global_frame(_gfi)
            if _gt_vec_diag is not None:
                _target_enu_diag, _target_src_diag = _geo_camera_target_enu(
                    corr,
                    target_mode=target_mode,
                    prefer_pnp=True,
                    prefer_road_z=True,
                )
                _target_xy = _xy_err_to_gt(_target_enu_diag, _gt_vec_diag)
                _prior_xy = _xy_err_to_gt(approx_enu, _gt_vec_diag)
                _frame_xy = _xy_err_to_gt(corr.get("frame_center_enu"), _gt_vec_diag)
                _pnp_xy = None
                if isinstance(_pnp, dict) and _pnp.get("t_world") is not None:
                    _pnp_xy = _xy_err_to_gt(_pnp.get("t_world"), _gt_vec_diag)
                _rot_check = corr.get("pnp_rotation_check") if isinstance(corr, dict) else None
                if isinstance(_rot_check, dict):
                    _rt_tw = _rot_check.get("t_world")
                    _rt_diag = _rot_check.get("diagnostics")
                    _rt_xy = _xy_err_to_gt(_rt_tw, _gt_vec_diag)
                    _rt_z = None
                    try:
                        _rt_arr = np.asarray(_rt_tw, dtype=np.float64).reshape(-1)
                        if _rt_arr.size >= 3 and np.isfinite(_rt_arr[:3]).all():
                            _rt_z = float(_rt_arr[2] - float(_gt_vec_diag[2]))
                    except Exception:
                        _rt_z = None
                    _cur_z = None
                    if isinstance(_pnp, dict) and _pnp.get("t_world") is not None:
                        try:
                            _cur_arr = np.asarray(_pnp.get("t_world"), dtype=np.float64).reshape(-1)
                            if _cur_arr.size >= 3 and np.isfinite(_cur_arr[:3]).all():
                                _cur_z = float(_cur_arr[2] - float(_gt_vec_diag[2]))
                        except Exception:
                            _cur_z = None
                    def _fmt_rot(_val: Optional[float], _unit: str = "m") -> str:
                        return "None" if _val is None or not np.isfinite(float(_val)) else f"{float(_val):.2f}{_unit}"
                    _rt_reproj = None
                    _rt_ray = None
                    _rt_dxy_prior = None
                    _rt_dz_prior = None
                    if isinstance(_rt_diag, dict):
                        _rt_reproj = _rt_diag.get("median_reproj_px")
                        _rt_ray = _rt_diag.get("median_ray_residual_m")
                        _rt_dxy_prior = _rt_diag.get("delta_xy_m")
                        _rt_dz_prior = _rt_diag.get("delta_z_m")
                    print(
                        f"[GeoV3][PnPRotCheckGT] seg={segment_input.segment_id} frame={fi} "
                        f"global={_gfi} cur_xy={_fmt_rot(_pnp_xy)} cur_z={_fmt_rot(_cur_z)} "
                        f"rt_xy={_fmt_rot(_rt_xy)} rt_z={_fmt_rot(_rt_z)} "
                        f"rt_reproj={_fmt_rot(_rt_reproj, 'px')} "
                        f"rt_ray={_fmt_rot(_rt_ray)} "
                        f"rt_dxy_prior={_fmt_rot(_rt_dxy_prior)} "
                        f"rt_dz_prior={_fmt_rot(_rt_dz_prior)}"
                    )
                _flip_enu, _ground_delta_m = _ground_signal_flip_target_enu(corr)
                _flip_xy = _xy_err_to_gt(_flip_enu, _gt_vec_diag)
                _crop_info_diag = corr.get("dom_crop_info", {}) if isinstance(corr, dict) else {}
                _pose_cam_xy = None
                _pose_ground_xy = None
                _pose_crop_ground_xy = None
                if isinstance(_crop_info_diag, dict):
                    _pose_cam_xy = _xy_err_to_gt(_crop_info_diag.get("pose_cam_center_enu"), _gt_vec_diag)
                    _pose_ground_prior = _crop_info_diag.get("pose_image_center_ground_enu")
                    if _pose_ground_prior is None:
                        _pose_ground_prior = _crop_info_diag.get("pose_center_ground_enu")
                    _pose_ground_xy = _xy_err_to_gt(_pose_ground_prior, _gt_vec_diag)
                    _pose_crop_ground_xy = _xy_err_to_gt(_crop_info_diag.get("pose_crop_center_ground_enu"), _gt_vec_diag)
                _diag_row: Dict[str, float] = {}
                for _key, _val in (
                    ("target_xy", _target_xy),
                    ("prior_xy", _prior_xy),
                    ("frame_xy", _frame_xy),
                    ("pnp_xy", _pnp_xy),
                    ("flip_xy", _flip_xy),
                    ("pose_cam_xy", _pose_cam_xy),
                    ("pose_ground_xy", _pose_ground_xy),
                    ("pose_crop_ground_xy", _pose_crop_ground_xy),
                    ("ground_delta_m", _ground_delta_m),
                ):
                    if _val is not None and np.isfinite(float(_val)):
                        _diag_row[_key] = float(_val)
                if _diag_row:
                    _target_diag_gt_rows.append(_diag_row)
                    corr["gt_target_diag"] = {
                        "global_frame": int(_gfi),
                        "target_source": str(_target_src_diag),
                        **_diag_row,
                    }
                    def _fmt_diag(_val: Optional[float]) -> str:
                        return "None" if _val is None or not np.isfinite(float(_val)) else f"{float(_val):.2f}m"
                    print(
                        f"[GeoV3][TargetDiagGT] seg={segment_input.segment_id} frame={fi} "
                        f"global={_gfi} src={_target_src_diag} "
                        f"target_xy={_fmt_diag(_target_xy)} "
                        f"prior_xy={_fmt_diag(_prior_xy)} "
                        f"frame_xy={_fmt_diag(_frame_xy)} "
                        f"pnp_xy={_fmt_diag(_pnp_xy)} "
                        f"flip_xy={_fmt_diag(_flip_xy)} "
                        f"ground_delta={_fmt_diag(_ground_delta_m)} "
                        f"pose_cam_xy={_fmt_diag(_pose_cam_xy)} "
                        f"pose_ground_xy={_fmt_diag(_pose_ground_xy)} "
                        f"pose_crop_ground_xy={_fmt_diag(_pose_crop_ground_xy)}"
                    )
            _update_online_crop_bootstrap(fi, corr)
        if corr is not None:
            try:
                _gfi = int(segment_input.start_frame + fi)
                _gt_vec = _gt_camera_enu_for_global_frame(_gfi)
                if _gt_vec is None:
                    raise ValueError("no_gt_camera_diag")
                _gt_vec = np.asarray(_gt_vec, dtype=np.float64).reshape(-1)
                if _gt_vec.size < 3 or not np.isfinite(_gt_vec[:3]).all():
                    raise ValueError("bad_gt_camera_diag")
                _gt_z = float(_gt_vec[2])
                corr["camera_z_gt_diag"] = _gt_z
                if _airzoo_gt_enu is not None:
                    corr["camera_z_gt"] = _gt_z
                _pnp = corr.get("pnp_result") if isinstance(corr, dict) else None
                if isinstance(_pnp, dict) and _pnp.get("t_world") is not None:
                    _tw = np.asarray(_pnp.get("t_world"), dtype=np.float64).reshape(-1)
                    if _tw.shape[0] >= 3 and _gt_vec.shape[0] >= 3:
                        _err_xy = float(np.linalg.norm(_tw[:2] - _gt_vec[:2]))
                        _err_z = float(_tw[2] - _gt_vec[2])
                        _diag = _pnp.get("diagnostics")
                        if isinstance(_diag, dict):
                            _diag["gt_xy_err_m"] = _err_xy
                            _diag["gt_z_err_m"] = _err_z
                        print(
                            f"[GeoV3][PnPDiagGT] seg={segment_input.segment_id} frame={fi} "
                            f"global={_gfi} gt_xy={_err_xy:.2f}m gt_z={_err_z:.2f}m"
                        )
                _ground_z_diag = _dem_ground_z_at_enu_xy(
                    _gt_vec[:2], project_fn, geo_elev, dom_transform)
                if _ground_z_diag is None:
                    _ep = corr.get("enu_positions")
                    if _ep is not None:
                        _ep_arr = np.asarray(_ep)
                        if _ep_arr.ndim == 2 and _ep_arr.shape[1] >= 3:
                            _gz_arr = _ep_arr[:, 2]
                            _gz_arr = _gz_arr[np.isfinite(_gz_arr)]
                            if _gz_arr.size > 0:
                                _ground_z_diag = float(np.mean(_gz_arr))
                if _ground_z_diag is not None and np.isfinite(float(_ground_z_diag)):
                    _agl_gt = _gt_z - float(_ground_z_diag)
                    if 20.0 < _agl_gt < 2000.0:
                        corr["gt_agl_diag"] = _agl_gt
                        if _airzoo_gt_enu is not None:
                            corr["gt_agl"] = _agl_gt
            except Exception:
                pass
        # Update running AGL estimate from matched frames (improves crop size for later frames).
        # For rendered DOM crops, corr["estimated_agl"] must not be the query->render
        # affine scale interpreted with DOM px/m; that quantity is not metric.
        if corr is not None:
            _agl_est = None
            _agl_source = None
            _crop_info = corr.get("dom_crop_info", {}) if isinstance(corr, dict) else {}
            _crop_mode = _crop_info.get("mode") if isinstance(_crop_info, dict) else None
            if _agl_est is None and corr.get("gt_agl") is not None:
                _agl_est = float(corr["gt_agl"])
                _agl_source = "gt_agl"
            if _agl_est is None and corr.get("estimated_agl") is not None:
                _est_source = str(corr.get("estimated_agl_source") or "legacy_affine")
                _is_bad_render_affine = (_crop_mode == "render" and _est_source in (
                    "legacy_affine", "affine_metric_crop", "render_pose_agl"))
                if not _is_bad_render_affine:
                    _agl_est = float(corr["estimated_agl"])
                    _agl_source = _est_source
            if _agl_est is not None and 50.0 < float(_agl_est) < 1000.0:
                _old_agl = float(_running_agl)
                _running_agl = 0.7 * _running_agl + 0.3 * float(_agl_est)  # EMA
                print(
                    f"[GeoV3][AGL] frame={fi} source={_agl_source} "
                    f"raw={float(_agl_est):.1f}m running={_old_agl:.1f}->{_running_agl:.1f}m"
                )
        print(f"[GeoV3][DOM] frame={fi} corr={'None' if corr is None else 'ok'}")
        if corr is not None:
            observations[fi] = corr
    if _target_diag_gt_rows:
        def _summ_diag(_key: str) -> str:
            _vals = np.asarray([
                float(_row[_key]) for _row in _target_diag_gt_rows
                if _key in _row and np.isfinite(float(_row[_key]))
            ], dtype=np.float64)
            if _vals.size == 0:
                return "None"
            return f"med={float(np.median(_vals)):.2f}m mean={float(np.mean(_vals)):.2f}m n={int(_vals.size)}"
        print(
            f"[GeoV3][TargetDiagGT-summary] seg={segment_input.segment_id} "
            f"target({_summ_diag('target_xy')}) "
            f"prior({_summ_diag('prior_xy')}) "
            f"frame({_summ_diag('frame_xy')}) "
            f"pnp({_summ_diag('pnp_xy')}) "
            f"flip({_summ_diag('flip_xy')}) "
            f"ground_delta({_summ_diag('ground_delta_m')}) "
            f"pose_cam({_summ_diag('pose_cam_xy')}) "
            f"pose_ground({_summ_diag('pose_ground_xy')}) "
            f"pose_crop_ground({_summ_diag('pose_crop_ground_xy')})"
        )
    return observations


def _build_overlap_dom_targets_enu(
    geo_obs: Dict[int, Dict[str, Any]],
    start_frame: int,
    num_frames: int,
    overlap_len: int,
    target_mode: str = "auto",
) -> Dict[int, np.ndarray]:
    """为重叠帧（每段头部 + 尾部各 overlap_len 帧）构建 DOM 锚点 ENU 目标字典。

    仅处理 head/tail 重叠帧；对每帧调 _geo_camera_target_enu 取 ENU 相机目标，
    以**全局帧下标**（start_frame + fi）为 key 存入。供 PGO 在相邻段间匹配同一
    重叠帧。overlap_len<=0 或无观测时返回空字典。
    """
    targets: Dict[int, np.ndarray] = {}
    if overlap_len <= 0 or not geo_obs:
        return targets
    for fi, corr in geo_obs.items():
        if corr is None:
            continue
        is_head = int(fi) < int(overlap_len)
        is_tail = int(fi) >= int(num_frames) - int(overlap_len)
        if not (is_head or is_tail):
            continue
        target_enu, _target_source = _geo_camera_target_enu(
            corr, target_mode=target_mode, prefer_pnp=True)
        if target_enu is None or len(target_enu) < 3:
            continue
        targets[int(start_frame) + int(fi)] = np.asarray(target_enu[:3], dtype=np.float32)
    return targets


def _build_model_target_cache(
    geo_obs: Dict[int, Dict[str, Any]],
    query_W: int,
    query_H: int,
    target_mode: str = "auto",
) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """Extract (camera_center_3d, per-point ground ENU, frame_pixels, raw_corr) from geo_obs."""
    cache: Dict[int, Dict[str, Any]] = {}
    matched_fis: List[int] = []
    for fi, corr in geo_obs.items():
        if corr is None:
            continue
        center_3d, center_source = _geo_camera_target_enu(
            corr, target_mode=target_mode, prefer_pnp=True, prefer_road_z=True)
        if center_3d is None:
            continue
        use_road_dense = _is_fixed_rotation_render_pnp(corr) and corr.get("enu_positions_road") is not None
        enu_pos = corr.get("enu_positions_road", None) if use_road_dense else corr.get("enu_positions", None)
        pixels = corr.get("frame_pixels_road", None) if use_road_dense else corr.get("frame_pixels", None)
        if pixels is None or len(pixels) == 0:
            continue
        pixels = np.asarray(pixels, dtype=np.float32)
        # Per-point ground ENU for dense loss (K, 3) or (K, 2)
        if enu_pos is not None:
            ground_enu = np.asarray(enu_pos, dtype=np.float32)
        else:
            ground_enu = None
        raw_corr = {
            "query_W": corr.get("query_W", query_W),
            "query_H": corr.get("query_H", query_H),
            "dense_source": "road" if use_road_dense else "all",
            "center_source": center_source,
        }
        cache[fi] = {"world": np.asarray(center_3d[:3], dtype=np.float32), "pixels": pixels, "corr": raw_corr, "ground_enu": ground_enu}
        matched_fis.append(fi)
    return cache, sorted(matched_fis)


def _canonicalize_segment_targets(
    geo_obs: Dict[int, Dict[str, Any]],
    transform: LocalModelTransform,
    query_W: int,
    query_H: int,
    device: torch.device,
    target_mode: str = "auto",
) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """把各帧的 ENU 目标规范化到 model space M_k，产出供段内 TTT 直接使用的缓存。

    先用 _build_model_target_cache 抽取每帧的相机中心 / 地面点 / 像素等 ENU 目标，
    再用 transform.world_to_model 逐一换算到 M_k：包括相机目标 target_model、
    可选的 heading 方向（ENU XY 单位向量，供 heading_loss）、以及稠密的逐点
    ground_model（K×3）。返回 (model_target_cache_model, matched_fis)。
    """
    model_target_cache, matched_fis = _build_model_target_cache(
        geo_obs, query_W=query_W, query_H=query_H, target_mode=target_mode)
    model_target_cache_model: Dict[int, Dict[str, Any]] = {}
    for fi, mt in model_target_cache.items():
        world = torch.tensor(mt["world"], device=device, dtype=torch.float32).view(1, 3)
        entry = {
            "target_model": transform.world_to_model(world).view(-1),
            "pixels": mt["pixels"],
            "corr": mt["corr"],
        }
        # Heading target: 2D unit vector in ENU (XY) representing query-image
        # camera-right direction. Used by heading_loss. Convention matches v2.
        _corr_fi = mt.get("corr", {}) or {}
        _hdeg = _corr_fi.get("heading_deg") if isinstance(_corr_fi, dict) else None
        if _hdeg is not None:
            _hrad = float(np.radians(float(_hdeg)))
            entry["target_heading_dir_enu"] = torch.tensor(
                [float(np.cos(_hrad)), float(np.sin(_hrad))],
                dtype=torch.float32, device=device,
            )
        # Per-point ground targets in model space for dense loss
        ground_enu = mt.get("ground_enu")
        if ground_enu is not None and len(ground_enu) > 0:
            ge = np.asarray(ground_enu, dtype=np.float32)
            if ge.ndim == 2 and ge.shape[1] >= 3:
                ge_t = torch.tensor(ge[:, :3], device=device, dtype=torch.float32)
                entry["ground_model"] = transform.world_to_model(ge_t)  # (K, 3) in M_k
            elif ge.ndim == 2 and ge.shape[1] == 2:
                # 2D ENU only — pad Z=0
                ge_3d = np.zeros((len(ge), 3), dtype=np.float32)
                ge_3d[:, :2] = ge[:, :2]
                ge_t = torch.tensor(ge_3d, device=device, dtype=torch.float32)
                entry["ground_model"] = transform.world_to_model(ge_t)
        model_target_cache_model[fi] = entry
    return model_target_cache_model, matched_fis


def _segment_model_space_ttt(
    model: Any,
    frames: List[dict],
    per_frame_all_tokens: List[List[torch.Tensor]],
    per_frame_leaf_tokens: List[Dict[int, torch.Tensor]],
    matched_fis: List[int],
    model_target_cache_model: Dict[int, Dict[str, Any]],
    transform: LocalModelTransform,
    optimizer: Any,
    num_steps: int,
    psi: int,
    reg_weight: float,
    mse_weight: float,
    dense_reproj_weight: float,
    heading_weight: float,
    cam_token_weight: float,
    segment_input: SegmentInput,
    geo_consist_stride: int,
    geo_consist_max_corr: int,
    overlap_target_cache: Optional[Dict[int, torch.Tensor]] = None,
    overlap_weight: float = 1.0,
    all_leaves: Optional[List[torch.Tensor]] = None,
    all_originals: Optional[List[torch.Tensor]] = None,
    dom_supervised_fis: Optional[set] = None,
    ttt_batch: int = 0,
    point_teacher_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    pose_teacher_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    ground_ray_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    config: Optional[GeoV3Config] = None,
    step_token_provider: Optional[
        Callable[[], Tuple[List[List[torch.Tensor]], int]]
    ] = None,
    optimized_parameters: Optional[List[torch.nn.Parameter]] = None,
) -> Dict[str, Any]:
    """Segment-internal TTT.

    dom_supervised_fis: if not None, only these local frame indices contribute
    DOM-derived pose/dense losses (Plan A: overlap frames only). Other losses
    (smoothness, overlap continuity) are unaffected.

    ttt_batch: point-head batch size for main blocks. 0 = one batch over all
    frames. Camera pose supervision is evaluated per frame with the frozen
    camera-head KV prefix so it matches the streaming pose used for DOM crops
    and transform fitting.
    """
    if getattr(config, "pointmap_consistency", "off") == "joint":
        from .pointmap_consistency import run_joint_ttt
        return run_joint_ttt(
            model=model, frames=frames, per_frame_all_tokens=per_frame_all_tokens,
            per_frame_leaf_tokens=per_frame_leaf_tokens, transform=transform,
            optimizer=optimizer, num_steps=num_steps, psi=psi, segment_input=segment_input,
            point_teacher_cache=point_teacher_cache, all_leaves=all_leaves,
            all_originals=all_originals, config=config, overlap_target_cache=overlap_target_cache,
        )
    loss_diag = {
        "ttt_losses": {"total_loss": 0.0},
        "pose_loss": 0.0,
        "dense_loss": 0.0,
        "point_ttt_loss": 0.0,
        "pose_ttt_loss": 0.0,
        "ground_ray_loss": 0.0,
        "ground_ray_xy_med_m": 0.0,
        "reg_loss": 0.0,
        "smooth_loss": 0.0,
        "overlap_loss": 0.0,
        "grad_norm": 0.0,
        "grad_norm_pre_clip": 0.0,
        "max_grad_norm": float(getattr(config, "max_grad_norm", 0.0) or 0.0),
        "grad_clip_applied": False,
    }
    gradient_probe_enabled = bool(getattr(config, "ttt_gradient_probe", False))
    device = frames[0]["img"].device
    point_teacher_cache = point_teacher_cache or {}
    pose_teacher_cache = pose_teacher_cache or {}
    ground_ray_cache = ground_ray_cache or {}

    # Pre-build chunk lists (batched forward). 0 => single full-batch chunk.
    def _chunk(lst: List[int], B: int) -> List[List[int]]:
        if B <= 0 or B >= len(lst):
            return [list(lst)] if lst else []
        return [list(lst[i:i + B]) for i in range(0, len(lst), B)]

    main_fis = sorted(set(int(x) for x in matched_fis)
                      | set(int(x) for x in point_teacher_cache.keys())
                      | set(int(x) for x in pose_teacher_cache.keys())
                      | set(int(x) for x in ground_ray_cache.keys()))
    main_chunks = _chunk(main_fis, ttt_batch)
    overlap_fis_all = (
        [oi for oi in overlap_target_cache.keys() if oi < len(frames)]
        if overlap_target_cache else []
    )
    overlap_chunks = _chunk(overlap_fis_all, ttt_batch)

    # Precompute 2D rotation from model→ENU for heading loss (detached)
    _R_m2e_2x2 = transform.R.detach()[:2, :2].float() if transform is not None else None
    _point_ttt_w_xy = float(getattr(config, "dom_point_ttt_w_point_xy", 0.0)) if config is not None else 0.0
    _point_ttt_w_z = float(getattr(config, "dom_point_ttt_w_point_z", 0.0)) if config is not None else 0.0
    _pose_ttt_w_xy = float(getattr(config, "dom_point_ttt_w_pose_xy", 0.0)) if config is not None else 0.0
    _pose_ttt_w_z = float(getattr(config, "dom_point_ttt_w_pose_z", 0.0)) if config is not None else 0.0
    _point_ttt_huber = float(getattr(config, "dom_point_ttt_point_huber_m", 5.0)) if config is not None else 5.0
    _pose_ttt_huber = float(getattr(config, "dom_point_ttt_pose_huber_m", 10.0)) if config is not None else 10.0
    _ground_ray_enabled = _dom_point_ground_ray_stage_enabled(config, "ttt")
    _ground_ray_w_xy = (
        float(getattr(config, "dom_point_ground_ray_ttt_w_xy", 0.0))
        if _ground_ray_enabled and config is not None else 0.0
    )
    _ground_ray_huber = float(getattr(config, "dom_point_ground_ray_ttt_huber_m", 10.0)) if config is not None else 10.0
    _ground_ray_min_abs_dir_z = float(getattr(config, "dom_point_ground_ray_min_abs_dir_z", 0.05)) if config is not None else 0.05
    _replace_dense_with_teacher = bool(getattr(config, "dom_point_ttt_replace_dense", True)) if config is not None else True
    _ray_diag_enabled = str(os.environ.get("GEOV3_RAY_DIAG", "")).strip().lower() in {"1", "true", "yes", "on"}
    if _ray_diag_enabled:
        try:
            _ray_diag_max_seg = int(os.environ.get("GEOV3_RAY_DIAG_MAX_SEG", "1"))
        except Exception:
            _ray_diag_max_seg = 1
        _ray_diag_enabled = int(segment_input.segment_id) <= _ray_diag_max_seg
    _ray_diag_stats: Dict[str, List[Tuple[float, float, float, float]]] = {}
    _ray_diag_error_printed = False
    frozen_camera_kv_full = (
        None
        if step_token_provider is not None
        else _build_frozen_camera_kv_full(model, per_frame_all_tokens)
    )
    grad_parameters = (
        list(optimized_parameters)
        if optimized_parameters is not None
        else list(all_leaves or [])
    )

    for step in range(max(1, num_steps)):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        # LoRA replays the complete streaming aggregator at every optimization
        # step.  Drop references to the previous replay *before* constructing
        # the next one.  In particular, an assignment of the form
        # ``per_frame_all_tokens = step_token_provider()`` keeps the old value
        # alive until the right-hand side has finished, temporarily placing two
        # segment-sized token/graph sets on the GPU.  That transient overlap is
        # enough to OOM even though the adapters themselves are tiny.
        pose_losses, dense_losses, heading_losses = [], [], []
        point_teacher_losses, pose_teacher_losses = [], []
        ground_ray_losses, pose_centers = [], []
        pose_teacher_xy_residuals, pose_teacher_z_residuals = [], []
        ground_ray_xy_residuals = []
        if step_token_provider is not None:
            per_frame_all_tokens = []
            per_frame_leaf_tokens = []
            frozen_camera_kv_full = None
            new_per_frame_all_tokens, new_psi = step_token_provider()
            per_frame_all_tokens = new_per_frame_all_tokens
            psi = int(new_psi)
            del new_per_frame_all_tokens, new_psi
            per_frame_leaf_tokens = [{} for _ in per_frame_all_tokens]
            frozen_camera_kv_full = _build_frozen_camera_kv_full(
                model, per_frame_all_tokens
            )

        # --- Main block: batch point_head, but keep camera_head causal per frame. ---
        for chunk_fis in main_chunks:
            if not chunk_fis:
                continue
            pts3d_B, _ = _batched_forward_pts3d(
                model, frames, per_frame_all_tokens, per_frame_leaf_tokens,
                chunk_fis, psi,
            )
            for bi, fi in enumerate(chunk_fis):
                frame = frames[fi]
                agg_tokens_fi = _build_agg_tokens(per_frame_all_tokens, per_frame_leaf_tokens, fi)
                pose_pred = _forward_pose_cached(
                    model,
                    agg_tokens_fi,
                    _slice_camera_kv_prefix(frozen_camera_kv_full, fi, len(per_frame_all_tokens)),
                )
                pts3d_pred = pts3d_B[bi:bi + 1]        # (1, H, W, 3)
                # Get model-space camera center directly (bypasses get_world_xyz_fn ambiguity)
                pose_model_center = pose_enc_to_camera_centers(pose_pred)
                if pose_model_center.ndim > 1:
                    pose_model_center = pose_model_center[0]
                pose_model = pose_model_center.view(-1)
                pose_centers.append(pose_model)
                mtm = model_target_cache_model.get(fi)
                if mtm is not None and (dom_supervised_fis is None or fi in dom_supervised_fis):
                    # target_model already in model space (world_to_model applied to ENU during canonicalization)
                    pose_losses.append(mse_weight * cam_token_weight * F.smooth_l1_loss(pose_model[:2], mtm["target_model"][:2]))

                    # Heading (yaw) loss: compare predicted camera-right in ENU to DOM-estimated direction
                    if heading_weight > 0 and _R_m2e_2x2 is not None:
                        _tgt_h = mtm.get("target_heading_dir_enu")
                        if _tgt_h is not None:
                            from streamvggt.utils.rotation import quat_to_mat
                            _quat = pose_pred[:, 3:7]
                            _R_raw = quat_to_mat(_quat)  # (1, 3, 3)
                            # v2 convention: transpose then take first col XY -> cam-right in model XY
                            _cam_right_model_xy = _R_raw.transpose(1, 2)[:, :2, 0]  # (1, 2)
                            _cam_right_enu_xy = _cam_right_model_xy @ _R_m2e_2x2.T  # (1, 2)
                            _cam_right_enu_n = F.normalize(_cam_right_enu_xy, dim=-1)
                            heading_losses.append(
                                heading_weight * F.mse_loss(_cam_right_enu_n.squeeze(0), _tgt_h)
                            )
                    skip_dense = _replace_dense_with_teacher and fi in point_teacher_cache
                    if not skip_dense:
                        pixels = mtm["pixels"]
                        corr = mtm["corr"]
                        query_W = float(corr.get("query_W", frame["img"].shape[-1]))
                        query_H = float(corr.get("query_H", frame["img"].shape[-2]))
                        sampled_pts = _sample_pts3d_at_query_pixels(
                            pts3d_pred, pixels[:, :2], query_W, query_H, detach=False)
                        sampled_pts_xy = sampled_pts[:, :2]
                        ground_model = mtm.get("ground_model")
                        if ground_model is not None and len(ground_model) == len(pixels):
                            gt_xy = torch.as_tensor(ground_model[:len(pixels), :2], dtype=torch.float32, device=pts3d_pred.device)
                            dense_losses.append(dense_reproj_weight * F.smooth_l1_loss(sampled_pts_xy, gt_xy))
                        else:
                            dense_losses.append(dense_reproj_weight * F.smooth_l1_loss(sampled_pts_xy, mtm["target_model"][:2]))

                teacher = point_teacher_cache.get(fi)
                if teacher is not None and (_point_ttt_w_xy > 0.0 or _point_ttt_w_z > 0.0):
                    pixels_t = teacher.get("pixels")
                    target_enu = teacher.get("target_enu")
                    if pixels_t is not None and target_enu is not None and len(pixels_t) >= 4:
                        query_W_t = float(teacher.get("query_W", frame["img"].shape[-1]))
                        query_H_t = float(teacher.get("query_H", frame["img"].shape[-2]))
                        sampled_model = _sample_pts3d_at_query_pixels(
                            pts3d_pred, pixels_t[:, :2], query_W_t, query_H_t, detach=False)
                        pred_enu = transform.model_to_world(sampled_model)
                        tgt_enu = target_enu.to(device=pred_enu.device, dtype=pred_enu.dtype)
                        losses = []
                        if _point_ttt_w_xy > 0.0:
                            losses.append(_point_ttt_w_xy * _smooth_l1_from_diff(
                                pred_enu[:, :2] - tgt_enu[:, :2], _point_ttt_huber))
                        if _point_ttt_w_z > 0.0:
                            losses.append(_point_ttt_w_z * _smooth_l1_from_diff(
                                pred_enu[:, 2:3] - tgt_enu[:, 2:3], _point_ttt_huber))
                        if losses:
                            point_teacher_losses.append(torch.stack(losses).sum())

                pose_teacher = pose_teacher_cache.get(fi)
                if pose_teacher is not None and (_pose_ttt_w_xy > 0.0 or _pose_ttt_w_z > 0.0):
                    target_enu = pose_teacher.get("target_enu")
                    if target_enu is not None:
                        pred_center_enu = transform.model_to_world(pose_model[:3].view(1, 3))[0]
                        tgt_center_enu = target_enu.to(device=pred_center_enu.device, dtype=pred_center_enu.dtype).view(-1)
                        losses = []
                        pose_diff_xy = pred_center_enu[:2] - tgt_center_enu[:2]
                        pose_diff_z = pred_center_enu[2:3] - tgt_center_enu[2:3]
                        pose_teacher_xy_residuals.append(torch.linalg.norm(pose_diff_xy.detach()))
                        pose_teacher_z_residuals.append(pose_diff_z.detach().abs().reshape(-1).mean())
                        if _pose_ttt_w_xy > 0.0:
                            losses.append(_pose_ttt_w_xy * _smooth_l1_from_diff(
                                pose_diff_xy, _pose_ttt_huber))
                        if _pose_ttt_w_z > 0.0:
                            losses.append(_pose_ttt_w_z * _smooth_l1_from_diff(
                                pose_diff_z, _pose_ttt_huber))
                        if losses:
                            pose_teacher_losses.append(torch.stack(losses).sum())

                ground_ray = ground_ray_cache.get(fi)
                if ground_ray is not None and _ground_ray_w_xy > 0.0:
                    target_enu = ground_ray.get("target_enu")
                    if target_enu is not None:
                        pred_center_enu = transform.model_to_world(pose_model[:3].view(1, 3))[0]
                        ray_model = _pose_center_ray_model(pose_pred).to(
                            device=pred_center_enu.device,
                            dtype=pred_center_enu.dtype,
                        )
                        ray_enu = ray_model @ transform.R.to(
                            device=pred_center_enu.device,
                            dtype=pred_center_enu.dtype,
                        ).T
                        tgt_ground = target_enu.to(
                            device=pred_center_enu.device,
                            dtype=pred_center_enu.dtype,
                        ).view(-1)
                        pred_ground_xy = _ground_ray_intersection_xy_torch(
                            pred_center_enu, ray_enu, tgt_ground[2], _ground_ray_min_abs_dir_z)
                        diff_xy = pred_ground_xy - tgt_ground[:2]
                        ground_ray_xy_residuals.append(torch.linalg.norm(diff_xy.detach()))
                        if _ray_diag_enabled and step == 0:
                            with torch.no_grad():
                                try:
                                    _variant_rays = _pose_center_ray_model_variants(pose_pred)
                                    _rot_enu = transform.R.to(
                                        device=pred_center_enu.device,
                                        dtype=pred_center_enu.dtype,
                                    )
                                    _center_d = pred_center_enu.detach()
                                    _target_d = tgt_ground.detach()
                                    for _ray_name, _ray_m in _variant_rays.items():
                                        _ray_m = _ray_m.to(
                                            device=pred_center_enu.device,
                                            dtype=pred_center_enu.dtype,
                                        ).detach()
                                        for _rot_name, _rot_mat in (("Rt", _rot_enu.T), ("R", _rot_enu)):
                                            _dir_enu = F.normalize(_ray_m @ _rot_mat, dim=0, eps=1e-8)
                                            _denom = _signed_safe_denominator_torch(
                                                _dir_enu[2], _ground_ray_min_abs_dir_z)
                                            _lam = (_target_d[2].reshape(()) - _center_d.view(-1)[2]) / _denom
                                            _pred_xy = _center_d.view(-1)[:2] + _lam * _dir_enu[:2]
                                            _diff_xy = _pred_xy - _target_d[:2]
                                            _norm = float(torch.linalg.norm(_diff_xy).detach().cpu())
                                            _loss_like = float((_ground_ray_w_xy * _smooth_l1_from_diff(
                                                _diff_xy, _ground_ray_huber)).detach().cpu())
                                            _ray_diag_stats.setdefault(
                                                f"{_ray_name}@{_rot_name}", []).append((
                                                    _norm,
                                                    _loss_like,
                                                    float(_lam.detach().cpu()),
                                                    float(_dir_enu[2].detach().cpu()),
                                                ))
                                except Exception as _e:
                                    if not _ray_diag_error_printed:
                                        print(f"[GeoV3][RayDiag] seg={segment_input.segment_id} error={_e}")
                                        _ray_diag_error_printed = True
                        ground_ray_losses.append(
                            _ground_ray_w_xy * _smooth_l1_from_diff(diff_xy, _ground_ray_huber)
                        )
        pose_loss = torch.stack(pose_losses).mean() if pose_losses else torch.zeros((), device=device)
        dense_loss = torch.stack(dense_losses).mean() if dense_losses else torch.zeros((), device=device)
        heading_loss = torch.stack(heading_losses).mean() if heading_losses else torch.zeros((), device=device)
        point_teacher_loss = torch.stack(point_teacher_losses).mean() if point_teacher_losses else torch.zeros((), device=device)
        pose_teacher_loss = torch.stack(pose_teacher_losses).mean() if pose_teacher_losses else torch.zeros((), device=device)
        ground_ray_loss = torch.stack(ground_ray_losses).mean() if ground_ray_losses else torch.zeros((), device=device)
        pose_teacher_xy_med_m = (
            float(torch.stack(pose_teacher_xy_residuals).median().detach().cpu())
            if pose_teacher_xy_residuals else 0.0
        )
        pose_teacher_z_med_m = (
            float(torch.stack(pose_teacher_z_residuals).median().detach().cpu())
            if pose_teacher_z_residuals else 0.0
        )
        ground_ray_xy_med_m = (
            float(torch.stack(ground_ray_xy_residuals).median().detach().cpu())
            if ground_ray_xy_residuals else 0.0
        )
        # Temporal smoothness over matched-frame camera centers (encourages consistent motion)
        if len(pose_centers) >= 2:
            centers_seq = torch.stack(pose_centers)   # (M, 3)
            smooth_loss = reg_weight * F.smooth_l1_loss(centers_seq[1:], centers_seq[:-1].detach())
        else:
            smooth_loss = torch.zeros((), device=device)
        # Overlap frames: enforce consistency with previous segment's ENU output.
        # overlap_target_cache maps local frame index → model-space target tensor.
        overlap_losses = []
        if overlap_target_cache:
            for chunk_fis in overlap_chunks:
                if not chunk_fis:
                    continue
                for oi in chunk_fis:
                    target_model = overlap_target_cache[oi]
                    agg_tokens_oi = _build_agg_tokens(per_frame_all_tokens, per_frame_leaf_tokens, oi)
                    ov_pose_pred = _forward_pose_cached(
                        model,
                        agg_tokens_oi,
                        _slice_camera_kv_prefix(frozen_camera_kv_full, oi, len(per_frame_all_tokens)),
                    )
                    ov_center = pose_enc_to_camera_centers(ov_pose_pred)
                    if ov_center.ndim > 1:
                        ov_center = ov_center[0]
                    overlap_losses.append(
                        overlap_weight * F.smooth_l1_loss(ov_center.view(-1)[:3], target_model[:3])
                    )
        overlap_loss = torch.stack(overlap_losses).mean() if overlap_losses else torch.zeros((), device=device)
        reg_loss = torch.zeros((), device=device)
        total_loss = (pose_loss + dense_loss + point_teacher_loss + pose_teacher_loss
              + ground_ray_loss
                  + smooth_loss + overlap_loss + heading_loss + reg_weight * reg_loss)
        if not total_loss.requires_grad:
            # No DOM/overlap targets for this segment — nothing to optimize, skip backward.
            continue
        total_loss.backward()
        if gradient_probe_enabled:
            # Compare the four DPT taps in a common frozen backward pass.  Raw
            # Frobenius norms depend on tensor size and activation scale, so the
            # selection score is grad_rms / token_rms, aggregated over all
            # frames in this segment.  No optimizer update is performed.
            probe_acc = {}
            for leaves_fi in per_frame_leaf_tokens:
                for layer_idx, leaf in leaves_fi.items():
                    if leaf.grad is None:
                        continue
                    grad = leaf.grad.detach().float()
                    token = leaf.detach().float()
                    entry = probe_acc.setdefault(int(layer_idx), {
                        "grad_sq_sum": 0.0,
                        "token_sq_sum": 0.0,
                        "numel": 0,
                        "tensors_with_grad": 0,
                    })
                    entry["grad_sq_sum"] += float((grad * grad).sum().cpu())
                    entry["token_sq_sum"] += float((token * token).sum().cpu())
                    entry["numel"] += int(grad.numel())
                    entry["tensors_with_grad"] += 1
            layer_scores = {}
            for layer_idx, entry in probe_acc.items():
                denom = max(int(entry["numel"]), 1)
                grad_rms = float(np.sqrt(entry["grad_sq_sum"] / denom))
                token_rms = float(np.sqrt(entry["token_sq_sum"] / denom))
                layer_scores[str(layer_idx)] = {
                    "grad_rms": grad_rms,
                    "token_rms": token_rms,
                    "relative_grad_rms": grad_rms / max(token_rms, 1e-12),
                    "grad_frobenius": float(np.sqrt(entry["grad_sq_sum"])),
                    "numel": int(entry["numel"]),
                    "tensors_with_grad": int(entry["tensors_with_grad"]),
                }
            ranking = sorted(
                layer_scores,
                key=lambda layer_idx: layer_scores[layer_idx]["relative_grad_rms"],
                reverse=True,
            )
            probe_diag = {
                "score": "relative_grad_rms = grad_rms / token_rms",
                "step": int(step),
                "layers": layer_scores,
                "ranking": ranking,
            }
            loss_diag.update({
                "ttt_losses": {"total_loss": float(total_loss.detach().cpu())},
                "pose_loss": float(pose_loss.detach().cpu()),
                "dense_loss": float(dense_loss.detach().cpu()),
                "point_ttt_loss": float(point_teacher_loss.detach().cpu()),
                "pose_ttt_loss": float(pose_teacher_loss.detach().cpu()),
                "ground_ray_loss": float(ground_ray_loss.detach().cpu()),
                "smooth_loss": float(smooth_loss.detach().cpu()),
                "overlap_loss": float(overlap_loss.detach().cpu()),
                "heading_loss": float(heading_loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "gradient_probe": probe_diag,
            })
            _scores = ", ".join(
                f"L{layer_idx}={layer_scores[layer_idx]['relative_grad_rms']:.3e}"
                for layer_idx in ranking
            )
            print(f"[GeoV3][TTTGradientProbe] seg={segment_input.segment_id} "
                  f"ranking={' > '.join('L' + layer_idx for layer_idx in ranking)} "
                  f"({_scores})")
            break
        grad_parameters_with_grad = [
            parameter for parameter in grad_parameters
            if parameter.grad is not None
        ]
        if grad_parameters_with_grad:
            grad_norm_pre_clip = float(torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().float().norm()
                for parameter in grad_parameters_with_grad
            ])).cpu())
        else:
            grad_norm_pre_clip = 0.0
        max_grad_norm = float(getattr(config, "max_grad_norm", 0.0) or 0.0)
        grad_clip_applied = False
        if optimizer is not None and max_grad_norm > 0.0 and grad_parameters_with_grad:
            torch.nn.utils.clip_grad_norm_(grad_parameters_with_grad, max_grad_norm)
            grad_clip_applied = grad_norm_pre_clip > max_grad_norm
        if grad_parameters_with_grad:
            grad_norm = float(torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().float().norm()
                for parameter in grad_parameters_with_grad
            ])).cpu())
        else:
            grad_norm = 0.0
        if optimizer is not None:
            optimizer.step()
        token_delta_rms = 0.0
        token_delta_max = 0.0
        token_delta_rel_rms = 0.0
        if all_leaves and all_originals and len(all_leaves) == len(all_originals):
            with torch.no_grad():
                delta_sq_sum = 0.0
                orig_sq_sum = 0.0
                elem_count = 0
                max_abs_delta = 0.0
                for leaf, original in zip(all_leaves, all_originals):
                    delta = leaf.detach() - original.to(device=leaf.device, dtype=leaf.dtype)
                    delta_sq_sum += float((delta.float() ** 2).sum().detach().cpu())
                    orig_sq_sum += float((original.to(device=leaf.device, dtype=leaf.dtype).float() ** 2).sum().detach().cpu())
                    elem_count += int(delta.numel())
                    if delta.numel() > 0:
                        max_abs_delta = max(max_abs_delta, float(delta.float().abs().max().detach().cpu()))
                if elem_count > 0:
                    token_delta_rms = float(np.sqrt(delta_sq_sum / max(elem_count, 1)))
                    token_delta_max = float(max_abs_delta)
                    token_delta_rel_rms = float(np.sqrt(delta_sq_sum / max(orig_sq_sum, 1e-12)))
        loss_diag = {
            "ttt_losses": {"total_loss": float(total_loss.detach().cpu())},
            "pose_loss": float(pose_loss.detach().cpu()),
            "dense_loss": float(dense_loss.detach().cpu()),
            "point_ttt_loss": float(point_teacher_loss.detach().cpu()),
            "pose_ttt_loss": float(pose_teacher_loss.detach().cpu()),
            "pose_ttt_xy_med_m": pose_teacher_xy_med_m,
            "pose_ttt_z_med_m": pose_teacher_z_med_m,
            "ground_ray_loss": float(ground_ray_loss.detach().cpu()),
            "ground_ray_xy_med_m": ground_ray_xy_med_m,
            "smooth_loss": float(smooth_loss.detach().cpu()),
            "overlap_loss": float(overlap_loss.detach().cpu()),
            "heading_loss": float(heading_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "grad_norm": grad_norm,
            "grad_norm_pre_clip": grad_norm_pre_clip,
            "max_grad_norm": max_grad_norm,
            "grad_clip_applied": bool(grad_clip_applied),
            "token_delta_rms": token_delta_rms,
            "token_delta_max": token_delta_max,
            "token_delta_rel_rms": token_delta_rel_rms,
        }
        _parts = [
            f"[GeoV3][TTT] seg={segment_input.segment_id} step={step:02d}",
            f"total={loss_diag['ttt_losses']['total_loss']:.4f}",
        ]
        for _name, _key in (
            ("pose", "pose_loss"),
            ("dense", "dense_loss"),
            ("point_ttt", "point_ttt_loss"),
            ("pose_ttt", "pose_ttt_loss"),
            ("ground_ray", "ground_ray_loss"),
            ("smooth", "smooth_loss"),
            ("overlap", "overlap_loss"),
            ("heading", "heading_loss"),
            ("reg", "reg_loss"),
        ):
            _val = float(loss_diag.get(_key, 0.0))
            if abs(_val) > 1e-8:
                _parts.append(f"{_name}={_val:.4f}")
        if abs(float(grad_norm)) > 1e-8:
            _parts.append(f"grad={grad_norm:.4f}")
        if grad_clip_applied:
            _parts.append(f"grad_preclip={grad_norm_pre_clip:.4f}")
        if token_delta_max > 0.0:
            _parts.append(f"tok_rms={token_delta_rms:.2e}")
            _parts.append(f"tok_max={token_delta_max:.2e}")
            _parts.append(f"tok_rel={token_delta_rel_rms:.2e}")
        if pose_teacher_xy_residuals:
            _parts.append(f"pose_xy_med={pose_teacher_xy_med_m:.2f}m")
            _parts.append(f"pose_z_med={pose_teacher_z_med_m:.2f}m")
        if ground_ray_xy_residuals:
            _parts.append(f"ray_xy_med={ground_ray_xy_med_m:.2f}m")
        print(" ".join(_parts))
        if _ray_diag_enabled and step == 0 and _ray_diag_stats:
            for _name, _rows in sorted(_ray_diag_stats.items()):
                if not _rows:
                    continue
                _arr = np.asarray(_rows, dtype=np.float64)
                _xy = _arr[:, 0]
                _loss_like = _arr[:, 1]
                _lam = _arr[:, 2]
                _dir_z = _arr[:, 3]
                _current = " current" if _name == "row2_plus_z@Rt" else ""
                print(
                    f"[GeoV3][RayDiag] seg={segment_input.segment_id} variant={_name}{_current} "
                    f"n={len(_rows)} xy_med={float(np.median(_xy)):.2f}m "
                    f"xy_q25={float(np.percentile(_xy, 25)):.2f}m "
                    f"xy_q75={float(np.percentile(_xy, 75)):.2f}m "
                    f"xy_max={float(np.max(_xy)):.2f}m "
                    f"xy_mean={float(np.mean(_xy)):.2f}m "
                    f"loss_like={float(np.mean(_loss_like)):.4f} "
                    f"lambda_pos={float(np.mean(_lam > 0.0)):.2f} "
                    f"dir_z_med={float(np.median(_dir_z)):.4f}"
                )

        if step_token_provider is not None:
            # Only the JSON-safe scalar diagnostics survive a LoRA step.  The
            # caller performs a detached replay after the final optimizer
            # update, so none of these graph-bearing tensors are needed here.
            # Clearing them now also prevents Python loop locals from keeping a
            # completed replay resident while the next replay is evaluated.
            per_frame_all_tokens = []
            per_frame_leaf_tokens = []
            frozen_camera_kv_full = None
            pose_losses.clear()
            dense_losses.clear()
            heading_losses.clear()
            point_teacher_losses.clear()
            pose_teacher_losses.clear()
            ground_ray_losses.clear()
            pose_centers.clear()
            overlap_losses.clear()
            total_loss = None
            pose_loss = None
            dense_loss = None
            heading_loss = None
            point_teacher_loss = None
            pose_teacher_loss = None
            ground_ray_loss = None
            smooth_loss = None
            overlap_loss = None
            reg_loss = None
            pts3d_B = None
            agg_tokens_fi = None
            pose_pred = None
            pts3d_pred = None
            pose_model_center = None
            pose_model = None
            agg_tokens_oi = None
            ov_pose_pred = None
            ov_center = None
    return loss_diag


def default_segment_infer_fn(
    model: Any,
    segment_input: SegmentInput,
    runtime_state: SegmentRuntimeState,
    config: GeoV3Config,
) -> SegmentOutput:
    """geo_v3 的单段推理主入口：被 GeoV3Runner.run 对每个分段逐一调用。

    它把本文件其余几十个 `_` 辅助函数串成一条完整的单段流水线，大致顺序为：
    1. 冻结骨干前向，拿到逐帧 token / pose / pts3d（_frozen_forward_with_tokens_geo_v3
       与 _forward_* 系列）；
    2. 采集 DOM/geo 观测（_collect_geo_observations），并按需做 dom_points 路径的
       点对应 / bootstrap / 相机目标 / ground-ray 构造；
    3. 选定并构建本段 model→ENU 的 T_k（_build_transform_for_segment，含重叠先验、
       DOM 重拟合、回退之间的门控）；
    4. 把 ENU 目标规范化到 M_k（_canonicalize_segment_targets），在 token 空间做
       model-space TTT（_segment_model_space_ttt）；可选段内 PGO（pgo.refine_segment_pgo）；
    5. 解码最终逐帧位姿 / 点，产出 model-space 轨迹与诊断，并组装重叠帧的 DOM 目标
       （_build_overlap_dom_targets_enu）供全局 PGO 使用。

    参数：
        model:          StreamVGGT 模型（骨干冻结，仅 token 叶子参与 TTT）。
        segment_input:  本段输入（帧、DOM、锚点、上一段传播状态等，见 io.SegmentInput）。
        runtime_state:  本段运行时状态（已 hard reset，见 state.SegmentRuntimeState）。
        config:         GeoV3Config 超参与各类开关。

    返回：
        SegmentOutput——含 model-space 轨迹 trajectory_local（跨段链接的权威数据）、
        其 ENU 形式 trajectory_global（边界报告用）、本段 transform=T_k、submap_state、
        geo_cache 以及详尽的 diagnostics。空段（length<=0）返回标记 skipped 的空输出。
    """
    num_frames = segment_input.length
    if num_frames <= 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return SegmentOutput(trajectory_global=empty, trajectory_local=empty, diagnostics={"skipped": True})

    # Keep stage timings local to this segment so the runner can aggregate them
    # without parsing stdout.  ``perf_counter`` is monotonic and has enough
    # resolution for the short map/TTT stages.
    _segment_t0 = time.perf_counter()
    _phase_timing_s: Dict[str, float] = {}

    def _record_phase(name: str, started: float) -> None:
        _phase_timing_s[str(name)] = float(max(0.0, time.perf_counter() - started))

    frames = segment_input.frames
    dom_mgr = DOMStateManager(DOMStateConfig())
    dom_state = DOMState(DOMStateConfig())

    dom_image = segment_input.dom_image
    if dom_image is not None and dom_image.dim() == 3:
        dom_image = dom_image.unsqueeze(0)

    anchor_world_xyz = segment_input.anchor_world_xyz
    if anchor_world_xyz is not None and torch.is_tensor(anchor_world_xyz) and anchor_world_xyz.dim() == 1:
        anchor_world_xyz = anchor_world_xyz.unsqueeze(0)

    get_world_xyz_fn = segment_input.get_world_xyz_fn
    if get_world_xyz_fn is None:
        def get_world_xyz_fn(pose_params):
            centers = pose_enc_to_camera_centers(pose_params)
            if anchor_world_xyz is not None:
                anchor = anchor_world_xyz.to(device=centers.device, dtype=centers.dtype)
                if anchor.dim() == 1:
                    anchor = anchor.unsqueeze(0)
                return centers + (anchor - anchor[:1])
            return centers

    print(f"[GeoV3] seg {segment_input.segment_id}: entering default_segment_infer_fn with {len(frames)} frames")
    _frozen_t0 = time.perf_counter()
    per_frame_tokens, per_frame_all_tokens, per_frame_pose, per_frame_pts3d, psi = _frozen_forward_with_tokens_geo_v3(
        model, frames, segment_input.query_points
    )
    adaptation_type = str(
        getattr(config, "ttt_adaptation_type", "token")
    ).strip().lower()
    if adaptation_type not in {"token", "lora"}:
        raise ValueError(f"Unsupported TTT adaptation type: {adaptation_type!r}")
    if adaptation_type == "lora" and bool(
        getattr(config, "ttt_gradient_probe", False)
    ):
        raise ValueError("ttt_gradient_probe is only defined for token adaptation")

    # ttt_layers selects which DPT layers become leaves in token mode.  LoRA
    # mode keeps these empty because gradients must flow through the replayed
    # aggregator instead of stopping at cached token leaves.
    _cfg_ttt_layers = getattr(config, "ttt_layers", None)
    if _cfg_ttt_layers:
        dpt_layers = [int(li) for li in _cfg_ttt_layers]
    else:
        dpt_layers = list(model.point_head.intermediate_layer_idx)
    if adaptation_type == "token":
        prepare_leaves = _prepare_token_leaves
        if getattr(config, "pointmap_consistency", "off") != "off":
            from .pointmap_consistency import prepare_token_leaves
            prepare_leaves = prepare_token_leaves
            dpt_layers = sorted(set(dpt_layers) | {len(per_frame_all_tokens[0]) - 1})
        per_frame_leaf_tokens, all_leaves, all_originals = prepare_leaves(
            per_frame_all_tokens, dpt_layers, device=frames[0]["img"].device
        )
    else:
        per_frame_leaf_tokens = [{} for _ in per_frame_all_tokens]
        all_leaves, all_originals = [], []
    _record_phase("frozen_forward_s", _frozen_t0)

    kv_lr = float(getattr(config, "geo_ttt_v2_kv_lr", config.kv_lr))
    num_steps = int(getattr(config, "geo_ttt_v2_steps", config.num_steps))
    reg_weight = float(getattr(config, "geo_ttt_v2_reg_weight", config.reg_weight))
    mse_weight = float(getattr(config, "mse_weight", 1.0))
    dense_reproj_weight = float(getattr(config, "dense_reproj_weight", 0.5))
    heading_weight = float(getattr(config, "heading_weight", 0.0))
    cam_token_weight = float(getattr(config, "cam_token_weight", 10.0))
    geo_consist_stride = max(1, int(getattr(config, "geo_consist_stride", 1)))
    geo_consist_max_corr = int(getattr(config, "geo_consist_max_corr", 500))

    ttt_enable = (
        bool(getattr(config, "ttt_enable", True))
        and not bool(getattr(config, "map_anchor_only", False))
        and num_steps > 0
    )
    optimizer = (
        torch.optim.Adam(all_leaves, lr=kv_lr)
        if (ttt_enable and adaptation_type == "token" and all_leaves)
        else None
    )
    pose_target = torch.cat(per_frame_pose, dim=0).detach() if per_frame_pose else None
    pose_target = pose_target.to(frames[0]["img"].device) if pose_target is not None else None
    prev_submap_state = segment_input.prev_submap_state
    correspondences = segment_input.metadata.get("correspondences") if segment_input.metadata else None
    roma_model = segment_input.metadata.get("roma_model") if segment_input.metadata else None
    save_vis_dir = segment_input.metadata.get("save_vis_dir") if segment_input.metadata else None
    geo_elev = segment_input.geo_elev
    dom_transform = segment_input.dom_transform
    collect_geo_observations = _collect_geo_observations
    canonicalize_segment_targets = _canonicalize_segment_targets
    segment_model_space_ttt = _segment_model_space_ttt
    geo_target_mode = _geo_target_mode_from_config(config)
    dom_points_mode = geo_target_mode == "dom_points"

    # Resolve overlap config BEFORE calling _build_transform_for_segment (which needs overlap_len)
    overlap_weight = float(getattr(config, "overlap_weight", 1.0))
    overlap_len = int(getattr(config, "overlap", 0))

    _map_matching_t0 = time.perf_counter()
    overlap_prior_transform, overlap_prior_diag = _build_overlap_geo_prior_transform(
        segment_input=segment_input,
        per_frame_pose=per_frame_pose,
        overlap_len=overlap_len,
        device=frames[0]["img"].device,
        dtype=frames[0]["img"].dtype,
        config=config,
    )

    geo_obs = collect_geo_observations(
        segment_input=segment_input,
        frames=frames,
        dom_image=dom_image,
        per_frame_pose=per_frame_pose,
        roma_model=roma_model,
        save_vis_dir=save_vis_dir,
        geo_consist_stride=geo_consist_stride,
        geo_consist_max_corr=geo_consist_max_corr,
        project_fn=segment_input.project_fn,
        inv_project_fn=segment_input.inv_project_fn,
        geo_elev=geo_elev,
        dom_transform=dom_transform,
        device=frames[0]["img"].device,
        overlap_len=overlap_len,
        crop_transform=overlap_prior_transform,
        crop_transform_source="overlap_geo_prior" if overlap_prior_transform is not None else "propagated",
        overlap_prior_cache_reliable_only=bool(getattr(config, "overlap_prior_cache_reliable_only", False)),
        overlap_prior_cache_max_xy_m=float(getattr(config, "overlap_prior_consistency_max_xy_m", 25.0)),
        overlap_prior_pose_from_cache=bool(getattr(config, "overlap_prior_pose_from_cache", True)),
        overlap_prior_pose_min_frames=int(getattr(config, "overlap_prior_pose_min_frames", 1)),
        target_mode=geo_target_mode,
        dom_point_pose_crop_scale=float(getattr(config, "dom_point_pose_crop_scale", 1.4)),
        semantic_masks_enable=bool(getattr(config, "semantic_masks_enable", True)),
        disable_pnp=bool(getattr(config, "pointmap_camera_transform_enable", False)),
    )
    _candidate_frame_indices = [
        int(fi)
        for fi in range(int(num_frames))
        if (
            fi % int(geo_consist_stride) == 0
            or fi == int(num_frames) - 1
            or (
                overlap_len > 0
                and (fi < overlap_len or fi >= int(num_frames) - overlap_len)
            )
        )
    ]
    _matched_frame_indices = sorted(
        int(fi) for fi, corr in geo_obs.items() if corr is not None)
    _failed_frame_indices = sorted(
        set(_candidate_frame_indices) - set(_matched_frame_indices))
    _inlier_counts = [
        max(0, int(geo_obs[fi].get("n_inliers", 0) or 0))
        for fi in _matched_frame_indices
        if isinstance(geo_obs.get(fi), dict)
    ]
    _cache_reused_frame_indices = [
        int(fi) for fi in _matched_frame_indices
        if bool(geo_obs[fi].get("overlap_geo_cache_reused", False))
    ]
    map_observation_stats: Dict[str, Any] = {
        "candidate_frames": int(len(_candidate_frame_indices)),
        "matched_frames": int(len(_matched_frame_indices)),
        "failed_frames": int(len(_failed_frame_indices)),
        "match_rate": float(
            len(_matched_frame_indices) / max(len(_candidate_frame_indices), 1)),
        "total_inliers": int(sum(_inlier_counts)),
        "mean_inliers_per_matched_frame": (
            float(np.mean(np.asarray(_inlier_counts, dtype=np.float64)))
            if _inlier_counts else 0.0),
        "median_inliers_per_matched_frame": (
            float(np.median(np.asarray(_inlier_counts, dtype=np.float64)))
            if _inlier_counts else 0.0),
        "zero_match_segment": bool(not _matched_frame_indices),
        "candidate_frame_indices": _candidate_frame_indices,
        "matched_frame_indices": _matched_frame_indices,
        "failed_frame_indices": _failed_frame_indices,
        "cache_reused_frames": int(len(_cache_reused_frame_indices)),
        "cache_reused_frame_indices": _cache_reused_frame_indices,
    }
    # ENU→M_0 boundary: convert the unified camera target to M_0 immediately after DOM collection.
    # From this point on, all optimization uses M_0 coordinates only.
    # seg 0: no T_0 yet, conversion skipped (seg 0 Umeyama works in ENU — that IS the boundary).
    _pm_for_geo = segment_input.prev_summary
    _t0_for_geo: Optional[LocalModelTransform] = (
        _pm_for_geo.get("metadata", {}).get("t0_transform") if _pm_for_geo is not None else None
    )
    if _t0_for_geo is not None and geo_obs:
        _geo_dev = frames[0]["img"].device
        for _fi, _corr in geo_obs.items():
            if _corr is None:
                continue
            _target_enu, _target_source = _geo_camera_target_enu(
                _corr,
                target_mode=geo_target_mode,
                prefer_ground_signal=bool(getattr(config, "prefer_ground_signal_targets", True)),
                prefer_pnp=bool(getattr(config, "prefer_pnp_camera_targets", True)),
                allow_raw_pnp=bool(getattr(config, "allow_raw_pnp_camera_targets", False)),
                ground_signal_max_delta_xy_m=float(getattr(config, "ground_signal_max_delta_xy_m", 120.0)),
                ground_signal_min_coverage=float(getattr(config, "ground_signal_min_coverage", 0.01)),
            )
            if _target_enu is None or len(_target_enu) < 3:
                continue
            _enu_t = torch.tensor(
                [_target_enu[:3]], dtype=torch.float32, device=_geo_dev
            )
            _m0_t = _t0_for_geo.world_to_model(_enu_t)
            _corr["frame_center_m0"] = _m0_t[0].cpu().numpy().tolist()
            _corr["frame_center_m0_source"] = _target_source

    _record_phase("map_matching_s", _map_matching_t0)

    # --- Plan A: collect overlap-frame DOM targets in ENU (for post-processing PGO) ---
    # Key = GLOBAL frame index so neighboring segments can match the same overlap frame.
    # Value = np.ndarray (3,) ENU XYZ from the unified DOM camera target.
    overlap_dom_targets_enu = _build_overlap_dom_targets_enu(
        geo_obs, int(segment_input.start_frame), int(num_frames), int(overlap_len),
        target_mode=geo_target_mode)
    world_xyz_init = torch.stack(
        [_pose_to_world_xyz(p, segment_input.get_world_xyz_fn) for p in per_frame_pose], dim=0
    ).detach()
    _prev_post_ttt_m0 = segment_input.prev_summary.get("post_ttt_overlap_m0") if segment_input.prev_summary is not None else None
    point_bootstrap_transform: Optional[LocalModelTransform] = None
    point_bootstrap_diag: Optional[Dict[str, Any]] = None
    _bootstrap_t0 = time.perf_counter()
    if dom_points_mode:
        point_bootstrap_transform, point_bootstrap_diag = _estimate_dom_point_bootstrap_transform(
            geo_obs=geo_obs,
            frames=frames,
            per_frame_pose=per_frame_pose,
            per_frame_pts3d=per_frame_pts3d,
            segment_input=segment_input,
            config=config,
            device=world_xyz_init.device,
            dtype=world_xyz_init.dtype,
            overlap_len=overlap_len,
            overlap_prior_transform=overlap_prior_transform,
            overlap_prior_diag=overlap_prior_diag,
        )
    _overlap_prior_pre_reliable, _overlap_prior_pre_diag = _overlap_prior_reliability(
        segment_input=segment_input,
        overlap_prior_transform=overlap_prior_transform,
        overlap_prior_diag=overlap_prior_diag,
        config=config,
    )
    geo_obs_for_overlap_consistency = geo_obs
    overlap_dom_consistency_diag: Dict[str, Any] = {}
    _overlap_filter_strong = bool(getattr(config, "overlap_prior_restrict_strong_supervision", False))
    if _overlap_prior_pre_reliable:
        _geo_obs_overlap_filtered, overlap_dom_consistency_diag = _filter_geo_obs_for_overlap_primary(
            geo_obs=geo_obs,
            per_frame_pose=per_frame_pose,
            overlap_prior_transform=overlap_prior_transform,
            overlap_len=overlap_len,
            config=config,
        )
        if _overlap_filter_strong:
            geo_obs_for_overlap_consistency = _geo_obs_overlap_filtered
        print(
            f"[GeoV3][OverlapBootstrap] seg {segment_input.segment_id}: "
            f"DOM consistency diagnostic kept {overlap_dom_consistency_diag['kept']}/{len(geo_obs)} "
            f"(head={overlap_dom_consistency_diag['head_overlap']}, "
            f"new={overlap_dom_consistency_diag['new_kept']}, "
            f"skipped={overlap_dom_consistency_diag['skipped']}, "
            f"max_xy={overlap_dom_consistency_diag['max_xy_m']:.1f}m "
            f"filter_strong={_overlap_filter_strong})"
        )
    bootstrap_quality_diag = _dom_point_bootstrap_quality_stats(point_bootstrap_diag, config) if dom_points_mode else {}
    transform, transform_diag = _build_transform_for_segment(
        segment_input, world_xyz_init, geo_obs=geo_obs_for_overlap_consistency, per_frame_pose=per_frame_pose,
        point_bootstrap_transform=point_bootstrap_transform,
        point_bootstrap_diag=point_bootstrap_diag,
        overlap_len=overlap_len, prev_post_ttt_overlap_m0=_prev_post_ttt_m0,
        config=config,
        overlap_prior_transform=overlap_prior_transform,
        overlap_prior_diag=overlap_prior_diag,
        overlap_dom_consistency_diag=overlap_dom_consistency_diag,
    )

    _overlap_primary_active = bool(transform_diag.get("overlap_primary", False))
    if getattr(config, "pointmap_consistency", "off") != "off":
        if not transform_diag.get("point_bootstrap_primary", False):
            raise RuntimeError("PointMap A/B requires an accepted map Sim3; refusing an implicit fallback")
        transform_diag["pointmap_consistency"] = config.pointmap_consistency
    _overlap_consistency_active = bool(
        _overlap_prior_pre_reliable and overlap_dom_consistency_diag and _overlap_filter_strong)
    geo_obs_for_strong_supervision = geo_obs_for_overlap_consistency if _overlap_consistency_active else geo_obs
    if _overlap_consistency_active:
        transform_diag["overlap_primary_strong_supervision"] = overlap_dom_consistency_diag
    _record_phase("map_bootstrap_s", _bootstrap_t0)

    segment_config = config
    dom_point_route_stats: Dict[str, Any] = {}
    if dom_points_mode and bool(getattr(config, "dom_point_adaptive_pnp_enable", True)):
        pnp_quality_diag = _dom_point_fixedr_pnp_quality_stats(geo_obs_for_strong_supervision, config)
        route_source = "point_rays"
        route_reason = "point_quality_ok"
        if not bool(bootstrap_quality_diag.get("point_good", False)):
            if bool(pnp_quality_diag.get("accepted", False)):
                route_source = "fixedR_pnp"
                route_reason = "point_quality_bad_pnp_good"
            else:
                route_source = "none"
                route_reason = "point_quality_bad_pnp_bad"
        dom_point_route_stats = {
            "enabled": True,
            "teacher_source": route_source,
            "reason": route_reason,
            "bootstrap_quality": bootstrap_quality_diag,
            "fixedR_pnp_quality": pnp_quality_diag,
        }
        if route_source != "point_rays":
            segment_config = copy.copy(config)
            setattr(segment_config, "dom_point_ttt_camera_teacher_source", route_source)
            pgo_camera_source = (
                "fixedR_pnp"
                if route_source == "fixedR_pnp"
                and bool(getattr(config, "dom_point_camera_teacher_pgo_enable", False))
                else "none"
            )
            setattr(segment_config, "dom_point_camera_teacher_source", pgo_camera_source)
            dom_point_route_stats["pgo_camera_source"] = pgo_camera_source
        else:
            dom_point_route_stats["pgo_camera_source"] = str(
                getattr(config, "dom_point_camera_teacher_source", "dom_points"))
        transform_diag["dom_point_quality_route"] = dom_point_route_stats
        print(
            f"[GeoV3][PointRoute] seg {segment_input.segment_id}: "
            f"source={route_source} reason={route_reason} "
            f"pgo_source={dom_point_route_stats.get('pgo_camera_source')} "
            f"point_good={bool(bootstrap_quality_diag.get('point_good', False))} "
            f"pb_ratio={bootstrap_quality_diag.get('inlier_ratio')} "
            f"pb_med_xy={bootstrap_quality_diag.get('median_xy_m')} "
            f"road_ratio={bootstrap_quality_diag.get('road_frame_ratio')} "
            f"pnp_good={bool(pnp_quality_diag.get('accepted', False))} "
            f"pnp_frames={int(pnp_quality_diag.get('good_frames', 0))}/{int(pnp_quality_diag.get('candidate_frames', 0))}"
        )

    _ttt_t0 = time.perf_counter()
    model_target_cache_model, matched_fis = canonicalize_segment_targets(
        geo_obs=geo_obs_for_strong_supervision,
        transform=transform,
        query_W=frames[0]["img"].shape[-1],
        query_H=frames[0]["img"].shape[-2],
        device=frames[0]["img"].device,
        target_mode=geo_target_mode,
    )
    point_ttt_cache, pose_ttt_cache, point_ttt_stats = _build_dom_point_ttt_teacher(
        geo_obs=geo_obs_for_strong_supervision,
        frames=frames,
        per_frame_pose=per_frame_pose,
        per_frame_pts3d=per_frame_pts3d,
        transform=transform,
        config=segment_config,
        segment_input=segment_input,
        device=frames[0]["img"].device,
    )
    ground_ray_cache, ground_ray_stats = _build_dom_point_ground_ray_cache(
        geo_obs=geo_obs_for_strong_supervision,
        config=config if dom_points_mode else None,
        device=frames[0]["img"].device,
    )
    if bool(getattr(config, "dom_point_ttt_enable", False)) and (num_steps <= 0 or kv_lr <= 0.0):
        print(
            f"[GeoV3][PointTTT] seg {segment_input.segment_id}: WARN token teacher is enabled "
            f"but steps={num_steps} lr={kv_lr}; token outputs will not meaningfully update")

    # Build overlap frame targets in model space.
    # Overlap frames are local indices [0, overlap_len) of the current segment,
    # corresponding to the last overlap_len frames of the previous segment.
    # Preferred: use stored post-TTT positions in M_0 → convert M_0 → M_k.
    # Fallback: prev_trajectory_local → ENU via T_{k-1} → M_k via T_k.
    overlap_target_cache: Dict[int, torch.Tensor] = {}
    _prev_sum = segment_input.prev_summary
    _t0 = transform_diag.get("t0_transform")
    _prev_post_ttt = _prev_sum.get("post_ttt_overlap_m0") if _prev_sum is not None else None
    if _prev_sum is not None and overlap_len > 0 and _prev_post_ttt is not None and _t0 is not None:
        # Path A: M_0 → ENU (via T_0) → M_k (via T_k inverse). No inter-segment ENU storage.
        m0_arr = np.asarray(_prev_post_ttt, dtype=np.float32)
        if len(m0_arr) >= overlap_len:
            _dev = frames[0]["img"].device
            for oi in range(overlap_len):
                m0_pt = m0_arr[-(overlap_len - oi)]  # (3,)
                m0_t = torch.tensor(m0_pt, device=_dev, dtype=torch.float32).view(1, 3)
                # M_0 → ENU → M_k (mathematically equivalent to T_k_to_m0⁻¹)
                enu_t = _t0.model_to_world(m0_t)
                overlap_target_cache[oi] = transform.world_to_model(enu_t).view(-1).detach()
            print(f"[GeoV3][Overlap] seg {segment_input.segment_id}: "
                  f"{len(overlap_target_cache)} overlap targets built (M_0→M_k)")
    elif _prev_sum is not None and overlap_len > 0:
        # Path B: Legacy fallback via prev trajectory + prev transform
        _prev_transform = _prev_sum.get("transform")
        if _prev_transform is not None:
            prev_traj_local_stored = _prev_sum.get("trajectory_local")
            if prev_traj_local_stored is not None:
                prev_traj_local_np = np.asarray(prev_traj_local_stored, dtype=np.float32)
                if len(prev_traj_local_np) >= overlap_len:
                    _dev = frames[0]["img"].device
                    _R_prev = _prev_transform.R.detach().cpu().float().numpy()
                    _s_prev = float(_prev_transform.s.detach().cpu())
                    _t_prev = _prev_transform.t.detach().cpu().float().numpy()
                    for oi in range(overlap_len):
                        prev_local_pt = prev_traj_local_np[-(overlap_len - oi)]
                        prev_enu_np = _s_prev * (_R_prev @ prev_local_pt) + _t_prev
                        prev_enu_t = torch.tensor(prev_enu_np, device=_dev, dtype=torch.float32).view(1, 3)
                        overlap_target_cache[oi] = transform.world_to_model(prev_enu_t).view(-1).detach()
                    print(f"[GeoV3][Overlap] seg {segment_input.segment_id}: "
                          f"{len(overlap_target_cache)} overlap targets built (legacy M_{{k-1}}→ENU→M_k)")
    if not overlap_target_cache and overlap_len > 0 and _prev_sum is not None:
        _reason = "no post_ttt_m0 and no prev_transform" if _prev_post_ttt is None else "insufficient data"
        print(f"[GeoV3][Overlap] seg {segment_input.segment_id}: skip overlap targets ({_reason})")

    # Plan A: build set of frames allowed to contribute DOM loss.
    # If dom_supervision_overlap_only=True, restrict to head+tail overlap frames.
    dom_supervised_fis: Optional[set] = None
    if bool(getattr(config, "dom_supervision_overlap_only", False)) and overlap_len > 0:
        _head = set(range(min(overlap_len, num_frames)))
        _tail = set(range(max(0, num_frames - overlap_len), num_frames))
        dom_supervised_fis = _head | _tail
        print(f"[GeoV3][PlanA] seg {segment_input.segment_id}: DOM supervision "
              f"restricted to {len(dom_supervised_fis)} overlap frames "
              f"(head+tail; segment has {num_frames} frames)")

    ttt_memory_diag: Dict[str, Any] = {
        "measured": False,
        "device": str(frames[0]["img"].device),
    }
    _ttt_mem_device = frames[0]["img"].device
    if ttt_enable and torch.cuda.is_available() and _ttt_mem_device.type == "cuda":
        # Isolate allocations made while TTT is active.  The leaf tokens,
        # geographic targets, and other persistent segment state already exist
        # at this boundary and therefore belong to the pre-TTT baseline.
        torch.cuda.synchronize(_ttt_mem_device)
        _mb = float(1024.0 ** 2)
        ttt_memory_diag = {
            "measured": True,
            "device": str(_ttt_mem_device),
            "pre_allocated_mb": float(torch.cuda.memory_allocated(_ttt_mem_device) / _mb),
            "pre_reserved_mb": float(torch.cuda.memory_reserved(_ttt_mem_device) / _mb),
            "peak_before_reset_allocated_mb": float(
                torch.cuda.max_memory_allocated(_ttt_mem_device) / _mb
            ),
            "peak_before_reset_reserved_mb": float(
                torch.cuda.max_memory_reserved(_ttt_mem_device) / _mb
            ),
        }
        torch.cuda.reset_peak_memory_stats(_ttt_mem_device)

    if ttt_enable:
        _ttt_common_kwargs = dict(
            model=model,
            frames=frames,
            per_frame_all_tokens=per_frame_all_tokens,
            per_frame_leaf_tokens=per_frame_leaf_tokens,
            matched_fis=matched_fis,
            model_target_cache_model=model_target_cache_model,
            transform=transform,
            num_steps=num_steps,
            psi=psi,
            reg_weight=reg_weight,
            mse_weight=mse_weight,
            dense_reproj_weight=dense_reproj_weight,
            heading_weight=heading_weight,
            cam_token_weight=cam_token_weight,
            segment_input=segment_input,
            geo_consist_stride=geo_consist_stride,
            geo_consist_max_corr=geo_consist_max_corr,
            overlap_target_cache=overlap_target_cache,
            overlap_weight=overlap_weight,
            all_leaves=all_leaves,
            all_originals=all_originals,
            dom_supervised_fis=dom_supervised_fis,
            ttt_batch=int(getattr(config, "ttt_batch", 0)),
            point_teacher_cache=point_ttt_cache,
            pose_teacher_cache=pose_ttt_cache,
            ground_ray_cache=ground_ray_cache,
            config=config,
        )
        if adaptation_type == "lora":
            _lora_started = time.perf_counter()
            _pre_lora_grad_flags = [
                (parameter, bool(parameter.requires_grad))
                for parameter in model.parameters()
            ]
            for parameter, _ in _pre_lora_grad_flags:
                parameter.requires_grad_(False)

            lora_params: List[torch.nn.Parameter] = []
            lora_diag: Dict[str, Any] = {}
            try:
                lora_params, _ = inject_lora(
                    model,
                    rank=int(getattr(config, "lora_rank", 4)),
                    alpha=float(getattr(config, "lora_alpha", 8.0)),
                    target_blocks=str(getattr(config, "lora_target_blocks", "all")),
                    target_layers=str(getattr(config, "lora_target_layers", "qkv")),
                    target_block_indices=getattr(config, "lora_block_indices", None),
                )
                lora_diag = get_lora_diagnostics(model)
                _lora_initial = [parameter.detach().clone() for parameter in lora_params]
                lora_optimizer = torch.optim.Adam(
                    lora_params,
                    lr=float(getattr(config, "lora_lr", 1e-4)),
                )

                _prefix_cache_enabled = str(
                    os.environ.get("GEOV3_LORA_PREFIX_CACHE", "1")
                ).strip().lower() not in {"0", "false", "no", "off"}
                _target_block_indices = list(
                    lora_diag.get("target_block_indices") or []
                )
                _lora_start_block = (
                    min(int(index) for index in _target_block_indices)
                    if _target_block_indices else 0
                )
                _lora_retained_layers = sorted(set(
                    int(index)
                    for index in (
                        list(model.point_head.intermediate_layer_idx)
                        + [int(model.aggregator.depth) - 1]
                    )
                ))
                _lora_prefix_cache = (
                    _build_lora_prefix_cache_geo_v3(
                        model,
                        per_frame_all_tokens,
                        frames,
                        _lora_start_block,
                        _lora_retained_layers,
                    )
                    if _prefix_cache_enabled and _lora_start_block > 0
                    else None
                )
                if _lora_prefix_cache is not None:
                    print(
                        f"[LoRA] Frozen prefix cached through block "
                        f"{_lora_start_block - 1}; replaying blocks "
                        f"{_lora_start_block}-{int(model.aggregator.depth) - 1}"
                    )

                def _lora_step_tokens():
                    if _lora_prefix_cache is not None:
                        return _streaming_aggregator_suffix_forward_geo_v3(
                            model,
                            _lora_prefix_cache,
                            detach=False,
                            gradient_checkpointing=bool(
                                getattr(config, "lora_gradient_checkpointing", True)
                            ),
                        )
                    return _streaming_aggregator_forward_geo_v3(
                        model, frames, detach=False,
                        gradient_checkpointing=bool(
                            getattr(config, "lora_gradient_checkpointing", True)
                        ),
                    )

                # The frozen pass is needed to construct geographic targets,
                # but LoRA immediately replays the aggregator with gradients.
                # Release its all-layer token copies before the first replay;
                # otherwise both segment-sized representations coexist.  Pose
                # and point predictions remain available for downstream map
                # registration and diagnostics.
                per_frame_tokens = []
                per_frame_all_tokens = []
                per_frame_leaf_tokens = []
                _ttt_common_kwargs["per_frame_all_tokens"] = []
                _ttt_common_kwargs["per_frame_leaf_tokens"] = []
                _ttt_common_kwargs["all_leaves"] = []
                _ttt_common_kwargs["all_originals"] = []

                loss_diag = segment_model_space_ttt(
                    **_ttt_common_kwargs,
                    optimizer=lora_optimizer,
                    step_token_provider=_lora_step_tokens,
                    optimized_parameters=lora_params,
                )

                # The loss tokens precede the final optimizer.step(). Replay
                # once more to materialize outputs from the final adapters.
                with torch.no_grad():
                    if _lora_prefix_cache is not None:
                        per_frame_all_tokens, psi = (
                            _streaming_aggregator_suffix_forward_geo_v3(
                                model,
                                _lora_prefix_cache,
                                detach=True,
                                gradient_checkpointing=False,
                            )
                        )
                    else:
                        per_frame_all_tokens, psi = _streaming_aggregator_forward_geo_v3(
                            model, frames, detach=True
                        )
                per_frame_leaf_tokens = [{} for _ in per_frame_all_tokens]
                all_leaves, all_originals = [], []

                with torch.no_grad():
                    delta_sq_sum = 0.0
                    initial_sq_sum = 0.0
                    delta_max = 0.0
                    elem_count = 0
                    for parameter, initial in zip(lora_params, _lora_initial):
                        delta = parameter.detach().float() - initial.float()
                        delta_sq_sum += float((delta * delta).sum().cpu())
                        initial_sq_sum += float(
                            (initial.float() * initial.float()).sum().cpu()
                        )
                        elem_count += int(delta.numel())
                        if delta.numel() > 0:
                            delta_max = max(
                                delta_max, float(delta.abs().max().cpu())
                            )
                lora_diag.update({
                    "lr": float(getattr(config, "lora_lr", 1e-4)),
                    "steps": int(num_steps),
                    "segment_share": False,
                    "prefix_cache_enabled": bool(_lora_prefix_cache is not None),
                    "prefix_end_block": (
                        int(_lora_start_block - 1)
                        if _lora_prefix_cache is not None else None
                    ),
                    "replayed_block_count": int(
                        int(model.aggregator.depth) - _lora_start_block
                        if _lora_prefix_cache is not None
                        else int(model.aggregator.depth)
                    ),
                    "parameter_delta_rms": float(
                        np.sqrt(delta_sq_sum / max(elem_count, 1))
                    ),
                    "parameter_delta_max": float(delta_max),
                    "parameter_delta_rel_rms": float(
                        np.sqrt(delta_sq_sum / max(initial_sq_sum, 1e-12))
                    ),
                })
            finally:
                remove_lora(model)
                for parameter, requires_grad in _pre_lora_grad_flags:
                    parameter.requires_grad_(requires_grad)
            lora_diag["model_restored"] = not any(
                isinstance(module, LoRALinear)
                for module in model.aggregator.modules()
            )
            lora_diag["ttt_time_s"] = float(
                max(0.0, time.perf_counter() - _lora_started)
            )
            loss_diag["adaptation_type"] = "lora"
            loss_diag["lora"] = lora_diag
        else:
            loss_diag = segment_model_space_ttt(
                **_ttt_common_kwargs,
                optimizer=optimizer,
            )
            loss_diag["adaptation_type"] = "token"
            loss_diag["ttt_layers"] = [int(layer) for layer in dpt_layers]
    else:
        print(f"[GeoV3][TTT] disabled seg={segment_input.segment_id} "
              f"(ttt_enable={bool(getattr(config, 'ttt_enable', True))}, steps={num_steps})")
        loss_diag = {
            "ttt_losses": {"total_loss": 0.0},
            "pose_loss": 0.0,
            "dense_loss": 0.0,
            "point_ttt_loss": 0.0,
            "pose_ttt_loss": 0.0,
            "ground_ray_loss": 0.0,
            "ground_ray_xy_med_m": 0.0,
            "smooth_loss": 0.0,
            "overlap_loss": 0.0,
            "heading_loss": 0.0,
            "reg_loss": 0.0,
            "grad_norm": 0.0,
            "grad_norm_pre_clip": 0.0,
            "max_grad_norm": float(getattr(config, "max_grad_norm", 0.0) or 0.0),
            "grad_clip_applied": False,
            "token_delta_rms": 0.0,
            "token_delta_max": 0.0,
            "token_delta_rel_rms": 0.0,
            "adaptation_type": adaptation_type,
        }
    if bool(ttt_memory_diag.get("measured", False)):
        torch.cuda.synchronize(_ttt_mem_device)
        _mb = float(1024.0 ** 2)
        _peak_allocated_mb = float(
            torch.cuda.max_memory_allocated(_ttt_mem_device) / _mb
        )
        _peak_reserved_mb = float(
            torch.cuda.max_memory_reserved(_ttt_mem_device) / _mb
        )
        _post_allocated_mb = float(torch.cuda.memory_allocated(_ttt_mem_device) / _mb)
        _post_reserved_mb = float(torch.cuda.memory_reserved(_ttt_mem_device) / _mb)
        ttt_memory_diag.update({
            "peak_allocated_mb": _peak_allocated_mb,
            "peak_reserved_mb": _peak_reserved_mb,
            "post_allocated_mb": _post_allocated_mb,
            "post_reserved_mb": _post_reserved_mb,
            "incremental_peak_allocated_mb": float(max(
                0.0, _peak_allocated_mb - float(ttt_memory_diag["pre_allocated_mb"])
            )),
            "incremental_peak_reserved_mb": float(max(
                0.0, _peak_reserved_mb - float(ttt_memory_diag["pre_reserved_mb"])
            )),
        })
        print(
            f"[GeoV3][TTT-memory] seg={segment_input.segment_id} "
            f"layers={dpt_layers} "
            f"allocated={ttt_memory_diag['pre_allocated_mb']:.1f}->"
            f"{_peak_allocated_mb:.1f} MiB "
            f"(increment={ttt_memory_diag['incremental_peak_allocated_mb']:.1f} MiB)"
        )
    _record_phase("ttt_s", _ttt_t0)
    loss_diag["ground_ray_stats"] = ground_ray_stats
    loss_diag["ttt_memory"] = ttt_memory_diag

    with torch.no_grad():
        # Re-run inference with updated token leaves to get post-TTT camera centers.
        # model_final: (N, 3) in model space M_k.
        # world_enu_final: (N, 3) in absolute ENU via T_k (model_to_world).
        _PTS3D_FRAME_STRIDE = 4    # collect pts3d every N frames (keep memory low)
        _PTS3D_MAX_PER_FRAME = 256  # max points sampled per frame
        _export_full_pts3d = bool(getattr(config, "export_full_pts3d", False))
        post_ttt_centers = []
        post_ttt_ray_dirs_model: List[torch.Tensor] = []
        _pts3d_model_list = []  # model-space pts3d chunks for footprint vis
        # Full per-frame pts3d / conf in model space (only populated when
        # config.export_full_pts3d=True). Each entry: (H, W, 3) / (H, W).
        _pts3d_model_full_list: List[torch.Tensor] = []
        _pts3d_conf_full_list: List[torch.Tensor] = []
        _road_pts_model_list: List[torch.Tensor] = []   # per-frame (K_i,3) road pts in M_k
        _road_dem_z_list: List[np.ndarray] = []          # per-frame (K_i,) DEM ground Z (ENU)
        _road_frame_count = 0
        _nonbld_pts_model_list: List[torch.Tensor] = []  # non-building DEM-Z diagnostics only
        _nonbld_dem_z_list: List[np.ndarray] = []
        _nonbld_frame_count = 0
        _dom_pts_model_list: List[torch.Tensor] = []     # matched DOM/DEM point edges in M_k
        _dom_pts_enu_list: List[np.ndarray] = []          # matched DOM/DEM point targets in ENU
        _dom_point_frame_count = 0
        # Retain per-frame visual ground samples for the optional post-PGO
        # camera-only vertical-scale solve. Model-space points are stored so
        # the final PGO transform, rather than a stale pre-PGO transform,
        # determines the visual AGL.
        _visual_ground_points_model_by_frame: Dict[int, np.ndarray] = {}
        _visual_ground_dem_z_by_frame: Dict[int, np.ndarray] = {}
        _visual_ground_source_by_frame: Dict[int, str] = {}
        _geo_target_mode_norm = _normalize_geo_target_mode(
            str(getattr(config, "geo_target_mode", "")),
            prefer_ground_signal=bool(getattr(config, "prefer_ground_signal_targets", True)),
            prefer_pnp=bool(getattr(config, "prefer_pnp_camera_targets", True)),
            allow_raw_pnp=bool(getattr(config, "allow_raw_pnp_camera_targets", False)),
        )
        _dom_points_mode = _geo_target_mode_norm == "dom_points"
        _collect_dom_point_edges = bool(getattr(config, "dom_point_edges_enable", False))
        _dom_point_road_only = (
            True if _dom_points_mode
            else bool(getattr(config, "dom_point_edges_road_only", True))
        )
        _dom_point_z_nonbuilding_fallback = (
            _dom_points_mode
            and bool(getattr(config, "dom_point_z_nonbuilding_fallback_enable", False))
        )
        dom_point_edge_stats: Dict[str, Any] = {
            "enabled": bool(_collect_dom_point_edges),
            "road_only_requested": bool(_dom_point_road_only),
            "dom_points_mode": bool(_dom_points_mode),
            "z_nonbuilding_fallback": bool(_dom_point_z_nonbuilding_fallback),
            "frames_road": 0,
            "frames_fallback_full": 0,
            "frames_nonbuilding": 0,
            "frames_full": 0,
            "frames_invalid": 0,
            "points_road_raw": 0,
            "points_fallback_full_raw": 0,
            "points_nonbuilding_raw": 0,
            "points_full_raw": 0,
            "points_valid_before_subsample": 0,
            "points_stored": 0,
            "semantic_rejected": {},
        }
        dom_point_camera_targets_enu: Dict[int, np.ndarray] = {}
        dom_point_camera_stats: Dict[str, Any] = {
            "enabled": bool(getattr(config, "dom_point_camera_teacher_enable", False)),
            "accepted": False,
            "reason": "not_run",
        }
        ground_ray_centers_model_pgo = None
        ground_ray_dirs_model_pgo = None
        ground_ray_targets_enu_pgo = None
        ground_ray_edge_stats: Dict[str, Any] = dict(ground_ray_stats)
        _point_ttt_disables_direct_srt = (
            bool(getattr(config, "dom_point_ttt_enable", False))
            and bool(getattr(config, "dom_point_ttt_disable_direct_srt", True))
        )
        if _point_ttt_disables_direct_srt and bool(getattr(config, "dom_point_srt_enable", False)):
            print(
                f"[GeoV3][PointSRT] seg {segment_input.segment_id}: direct SRT skipped "
                "because PointTTT teacher is enabled")
        _enable_dom_point_srt = (
            bool(getattr(config, "dom_point_srt_enable", False))
            and _collect_dom_point_edges
            and not _point_ttt_disables_direct_srt
        )
        _enable_dom_point_srt_rematch = bool(getattr(config, "dom_point_srt_rematch", False)) and _enable_dom_point_srt
        _post_pts3d_cache: Dict[int, torch.Tensor] = {}
        point_srt_stats: Dict[str, Any] = {
            "enabled": bool(_enable_dom_point_srt),
            "accepted": False,
            "disabled_by_point_ttt": bool(_point_ttt_disables_direct_srt),
        }
        _frozen_camera_kv_full = _build_frozen_camera_kv_full(model, per_frame_all_tokens)
        for _fi in range(num_frames):
            _agg = _build_agg_tokens(per_frame_all_tokens, per_frame_leaf_tokens, _fi)
            _pose_post, _pts3d_fi, _pts3d_conf_fi = _forward_pose_and_pts(
                model,
                frames[_fi],
                _agg,
                psi,
                camera_kv_prefix=_slice_camera_kv_prefix(_frozen_camera_kv_full, _fi, len(per_frame_all_tokens)),
            )
            _post_pts3d_cache[_fi] = _pts3d_fi.detach().float()
            if getattr(config, "pointmap_consistency", "off") == "joint":
                per_frame_pose[_fi] = _pose_post.detach().float()
            _mc = pose_enc_to_camera_centers(_pose_post)
            if _mc.ndim > 1:
                _mc = _mc[0]
            post_ttt_centers.append(_mc.detach().float())
            try:
                _ray_model = _pose_center_ray_model(_pose_post).detach().float()
            except Exception:
                _ray_model = torch.full((3,), float("nan"), device=_mc.device, dtype=torch.float32)
            post_ttt_ray_dirs_model.append(_ray_model)
            # Collect downsampled pts3d for footprint visualisation (every _PTS3D_FRAME_STRIDE frames)
            if _fi % _PTS3D_FRAME_STRIDE == 0:
                _p = _pts3d_fi.detach().float()
                if _p.ndim == 4:
                    _p = _p[0]  # (H, W, 3)
                _p_flat = _p.reshape(-1, 3)  # (H*W, 3)
                if len(_p_flat) > _PTS3D_MAX_PER_FRAME:
                    _sample_gen = torch.Generator(device=_p_flat.device)
                    _sample_gen.manual_seed(
                        170003 + 1009 * int(segment_input.segment_id) + int(_fi))
                    _idx = torch.randperm(
                        len(_p_flat), device=_p_flat.device, generator=_sample_gen
                    )[:_PTS3D_MAX_PER_FRAME]
                    _p_flat = _p_flat[_idx]
                _pts3d_model_list.append(_p_flat)

            if _export_full_pts3d:
                _pf = _pts3d_fi.detach().float()
                if _pf.ndim == 4:
                    _pf = _pf[0]  # (H, W, 3)
                _cf = _pts3d_conf_fi.detach().float()
                if _cf.ndim == 3:
                    _cf = _cf[0]  # (H, W)
                _pts3d_model_full_list.append(_pf.cpu())
                _pts3d_conf_full_list.append(_cf.cpu())

        model_final = torch.stack(post_ttt_centers, dim=0)          # (N, 3) model space
        world_enu_final = transform.model_to_world(model_final)      # (N, 3) ENU

        if pose_ttt_cache:
            _pose_xy_resids: List[torch.Tensor] = []
            _pose_z_resids: List[torch.Tensor] = []
            for _fi_pose, _teacher_pose in sorted(pose_ttt_cache.items()):
                _fi_pose = int(_fi_pose)
                if _fi_pose < 0 or _fi_pose >= int(world_enu_final.shape[0]):
                    continue
                _target_pose = _teacher_pose.get("target_enu") if isinstance(_teacher_pose, dict) else None
                if _target_pose is None:
                    continue
                _target_pose_t = _target_pose.to(device=world_enu_final.device, dtype=world_enu_final.dtype).view(-1)
                if _target_pose_t.numel() < 3:
                    continue
                _diff_pose = world_enu_final[_fi_pose, :3] - _target_pose_t[:3]
                _pose_xy_resids.append(torch.linalg.norm(_diff_pose[:2]).detach())
                _pose_z_resids.append(_diff_pose[2].abs().detach())
            if _pose_xy_resids:
                _pose_xy_stack = torch.stack(_pose_xy_resids).float().cpu()
                _pose_z_stack = torch.stack(_pose_z_resids).float().cpu()
                _post_pose_xy_med = float(torch.median(_pose_xy_stack).item())
                _post_pose_z_med = float(torch.median(_pose_z_stack).item())
                loss_diag["post_pose_ttt_xy_med_m"] = _post_pose_xy_med
                loss_diag["post_pose_ttt_z_med_m"] = _post_pose_z_med
                loss_diag["post_pose_ttt_frames"] = int(len(_pose_xy_resids))
                print(
                    f"[GeoV3][TTT-post] seg={segment_input.segment_id} "
                    f"pose_xy_med={_post_pose_xy_med:.2f}m "
                    f"pose_z_med={_post_pose_z_med:.2f}m "
                    f"frames={len(_pose_xy_resids)}"
                )

        def _collect_point_samples_from_obs(_geo_obs_src: Dict[int, Dict[str, Any]], collect_dom_points: bool = True):
            _dom_model_list: List[torch.Tensor] = []
            _dom_enu_list: List[np.ndarray] = []
            _dom_frame_count = 0
            _road_model_list: List[torch.Tensor] = []
            _road_z_list: List[np.ndarray] = []
            _road_count = 0
            _nonbld_model_list: List[torch.Tensor] = []
            _nonbld_z_list: List[np.ndarray] = []
            _nonbld_count = 0
            for _fi_s, _corr_s in sorted((_geo_obs_src or {}).items()):
                if _corr_s is None or _fi_s not in _post_pts3d_cache:
                    continue
                _pts3d_s = _post_pts3d_cache[_fi_s]
                if collect_dom_points and _collect_dom_point_edges:
                    if _dom_points_mode:
                        _ok_sem, _reason_sem, _sem_info = _dom_point_observation_semantics(
                            _corr_s, config=config, stage="point_edges")
                        if not _ok_sem:
                            _rej = dom_point_edge_stats.setdefault("semantic_rejected", {})
                            _rej[_reason_sem] = int(_rej.get(_reason_sem, 0)) + 1
                            if isinstance(_corr_s, dict):
                                _corr_s.setdefault("dom_point_observation", {})["point_edges"] = {
                                    "valid": False,
                                    "reason": _reason_sem,
                                    **_sem_info,
                                }
                            continue
                    if _dom_points_mode:
                        _px_edge, _enu_edge, _edge_source = _select_dom_point_correspondences_road_then_nonbuilding(
                            _corr_s, min_points=4)
                    else:
                        _px_edge = None
                        _enu_edge = None
                        _edge_source = "full"
                        if _dom_point_road_only:
                            _px_edge = _corr_s.get('frame_pixels_road')
                            _enu_edge = _corr_s.get('enu_positions_road')
                            _edge_source = "road"
                        if _px_edge is None or _enu_edge is None:
                            _px_edge = _corr_s.get('frame_pixels')
                            _enu_edge = _corr_s.get('enu_positions')
                            _edge_source = "fallback_full" if _dom_point_road_only else "full"
                    if _px_edge is not None and _enu_edge is not None:
                        _px_edge = np.asarray(_px_edge, dtype=np.float32)
                        _enu_edge = np.asarray(_enu_edge, dtype=np.float64)
                        if (_px_edge.ndim == 2 and _px_edge.shape[1] >= 2
                                and _enu_edge.ndim == 2 and _enu_edge.shape[1] >= 3
                                and _px_edge.shape[0] == _enu_edge.shape[0]
                                and _px_edge.shape[0] >= 4):
                            _qW_edge = float(_corr_s.get('query_W', frames[_fi_s]["img"].shape[-1]))
                            _qH_edge = float(_corr_s.get('query_H', frames[_fi_s]["img"].shape[-2]))
                            _samp_edge = _sample_pts3d_at_query_pixels(
                                _pts3d_s, _px_edge[:, :2], _qW_edge, _qH_edge, detach=True)
                            _dom_model_list.append(_samp_edge)
                            _dom_enu_list.append(_enu_edge[:, :3].astype(np.float64))
                            _visual_ground_points_model_by_frame[int(_fi_s)] = (
                                _samp_edge.detach().cpu().float().numpy().astype(np.float32)
                            )
                            _visual_ground_dem_z_by_frame[int(_fi_s)] = (
                                _enu_edge[:, 2].astype(np.float64)
                            )
                            _visual_ground_source_by_frame[int(_fi_s)] = str(_edge_source)
                            _dom_frame_count += 1
                            if _dom_points_mode and isinstance(_corr_s, dict):
                                _corr_s.setdefault("dom_point_observation", {})["point_edges"] = {
                                    "valid": True,
                                    "reason": "ok",
                                    **_sem_info,
                                }
                            if _edge_source == "road":
                                dom_point_edge_stats["frames_road"] += 1
                                dom_point_edge_stats["points_road_raw"] += int(_px_edge.shape[0])
                            elif _edge_source == "nonbuilding":
                                dom_point_edge_stats["frames_fallback_full"] += 1
                                dom_point_edge_stats["frames_nonbuilding"] += 1
                                dom_point_edge_stats["points_fallback_full_raw"] += int(_px_edge.shape[0])
                                dom_point_edge_stats["points_nonbuilding_raw"] += int(_px_edge.shape[0])
                            elif _edge_source == "fallback_full":
                                dom_point_edge_stats["frames_fallback_full"] += 1
                                dom_point_edge_stats["points_fallback_full_raw"] += int(_px_edge.shape[0])
                            else:
                                dom_point_edge_stats["frames_full"] += 1
                                dom_point_edge_stats["points_full_raw"] += int(_px_edge.shape[0])
                        else:
                            dom_point_edge_stats["frames_invalid"] += 1

                _is_rd = _corr_s.get('is_road')
                _enu_rd = _corr_s.get('enu_positions')
                _px_rd = _corr_s.get('frame_pixels')
                _has_nonbuilding_filter = bool(_corr_s.get('building_filter_applied', False))
                if (isinstance(_enu_rd, np.ndarray)
                        and _enu_rd.ndim == 2 and _enu_rd.shape[1] >= 3
                        and isinstance(_px_rd, np.ndarray)
                        and _px_rd.ndim == 2 and _px_rd.shape[1] >= 2
                        and _px_rd.shape[0] == _enu_rd.shape[0]):
                    _valid_z = (
                        np.isfinite(_enu_rd[:, 2])
                        & np.isfinite(_px_rd[:, 0])
                        & np.isfinite(_px_rd[:, 1])
                    )
                    if _has_nonbuilding_filter:
                        _px_nonbld = _px_rd[_valid_z]
                        _dem_z_nonbld = _enu_rd[_valid_z, 2]
                        if _px_nonbld.shape[0] >= 3:
                            _qW = float(_corr_s.get('query_W', frames[_fi_s]["img"].shape[-1]))
                            _qH = float(_corr_s.get('query_H', frames[_fi_s]["img"].shape[-2]))
                            _samp_nonbld = _sample_pts3d_at_query_pixels(
                                _pts3d_s, _px_nonbld, _qW, _qH, detach=True)
                            _nonbld_model_list.append(_samp_nonbld)
                            _nonbld_z_list.append(_dem_z_nonbld.astype(np.float64))
                            _nonbld_count += 1
                    if not (isinstance(_is_rd, np.ndarray) and _is_rd.shape[0] == _enu_rd.shape[0]):
                        continue
                    _mask_rd = _is_rd.astype(bool) & _valid_z
                    _px_road = _px_rd[_mask_rd]
                    _dem_z_road = _enu_rd[_mask_rd, 2]
                    if _px_road.shape[0] >= 3:
                        _qW = float(_corr_s.get('query_W', frames[_fi_s]["img"].shape[-1]))
                        _qH = float(_corr_s.get('query_H', frames[_fi_s]["img"].shape[-2]))
                        _samp = _sample_pts3d_at_query_pixels(
                            _pts3d_s, _px_road, _qW, _qH, detach=True)
                        _road_model_list.append(_samp)
                        _road_z_list.append(_dem_z_road.astype(np.float64))
                        _road_count += 1
            return (_dom_model_list, _dom_enu_list, _dom_frame_count,
                    _road_model_list, _road_z_list, _road_count,
                    _nonbld_model_list, _nonbld_z_list, _nonbld_count)

        def _finalize_dom_point_edges(
            _model_list: List[torch.Tensor],
            _enu_list: List[np.ndarray],
            _frame_count: int,
            _label: str,
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            if not (_model_list and _enu_list):
                return None, None
            _dom_pts_model_cat = torch.cat(_model_list, dim=0)
            _dom_pts_enu_cat = np.concatenate(_enu_list, axis=0).astype(np.float64)
            _finite = np.isfinite(_dom_pts_enu_cat).all(axis=1)
            _model_np = _dom_pts_model_cat.detach().cpu().float().numpy().astype(np.float64)
            _finite = _finite & np.isfinite(_model_np).all(axis=1)
            if not _finite.any():
                return None, None
            _dom_pts_model_cat = _dom_pts_model_cat[torch.from_numpy(_finite).to(device=_dom_pts_model_cat.device)]
            _dom_pts_enu_cat = _dom_pts_enu_cat[_finite]
            _P_full = int(_dom_pts_model_cat.shape[0])
            dom_point_edge_stats["points_valid_before_subsample"] = int(_P_full)
            _P_max = max(0, int(getattr(config, "dom_point_edges_max", 512)))
            if _P_max > 0 and _P_full > _P_max:
                _idx_pt = np.random.RandomState(17).choice(_P_full, _P_max, replace=False)
                _idx_pt.sort()
                _dom_pts_model_cat = _dom_pts_model_cat[_idx_pt]
                _dom_pts_enu_cat = _dom_pts_enu_cat[_idx_pt]
            _model_pgo = _dom_pts_model_cat.detach().cpu().float().numpy().astype(np.float32)
            _enu_pgo = _dom_pts_enu_cat.astype(np.float32)
            dom_point_edge_stats["points_stored"] = int(len(_enu_pgo))
            print(f"[GeoV3][PointEdges] seg {segment_input.segment_id}: "
                  f"stored {len(_enu_pgo)} DOM/DEM point edges "
                f"from {_frame_count} frames road_only={_dom_point_road_only} source={_label} "
                f"road_frames={dom_point_edge_stats['frames_road']} "
                                f"nonbuilding_frames={dom_point_edge_stats.get('frames_nonbuilding', 0)}")
            return _model_pgo, _enu_pgo

        _geo_obs_for_xy_edges = geo_obs_for_strong_supervision if _overlap_consistency_active else geo_obs
        (_dom_pts_model_list, _dom_pts_enu_list, _dom_point_frame_count,
         _road_pts_model_list, _road_dem_z_list, _road_frame_count,
         _nonbld_pts_model_list, _nonbld_dem_z_list, _nonbld_frame_count) = _collect_point_samples_from_obs(_geo_obs_for_xy_edges)
        if _geo_obs_for_xy_edges is not geo_obs and not _overlap_primary_active:
            (_, _, _,
             _road_pts_model_list, _road_dem_z_list, _road_frame_count,
             _nonbld_pts_model_list, _nonbld_dem_z_list, _nonbld_frame_count) = _collect_point_samples_from_obs(geo_obs, collect_dom_points=False)
        dom_pts_model_pgo, dom_pts_enu_pgo = _finalize_dom_point_edges(
            _dom_pts_model_list, _dom_pts_enu_list, _dom_point_frame_count, "initial")
        if _dom_points_mode and bool(getattr(config, "dom_point_quality_gate_enable", True)):
            ttt_rejected = _dom_point_point_ttt_rejected(point_ttt_stats)
            point_good_for_edges = bool(bootstrap_quality_diag.get("point_good", False)) and not ttt_rejected
            route_source_for_edges = str(dom_point_route_stats.get("teacher_source", "point_rays")) if dom_point_route_stats else "point_rays"
            point_edge_gate = {
                "enabled": True,
                "accepted": bool(point_good_for_edges and route_source_for_edges == "point_rays"),
                "reason": "ok",
                "bootstrap_quality": bootstrap_quality_diag,
                "point_ttt_rejected": bool(ttt_rejected),
                "teacher_source": route_source_for_edges,
            }
            if not point_good_for_edges:
                point_edge_gate["reason"] = "point_teacher_or_bootstrap_bad"
            elif route_source_for_edges != "point_rays":
                point_edge_gate["reason"] = f"teacher_source_{route_source_for_edges}"
            dom_point_edge_stats["quality_gate"] = point_edge_gate
            if not bool(point_edge_gate["accepted"]):
                dom_point_edge_stats["points_stored_before_quality_gate"] = int(dom_point_edge_stats.get("points_stored", 0))
                dom_point_edge_stats["points_stored"] = 0
                dom_pts_model_pgo = None
                dom_pts_enu_pgo = None
                print(
                    f"[GeoV3][PointEdges-gate] seg {segment_input.segment_id}: skip "
                    f"reason={point_edge_gate['reason']} "
                    f"point_good={bool(bootstrap_quality_diag.get('point_good', False))} "
                    f"ttt_rejected={ttt_rejected} source={route_source_for_edges}"
                )

        if _enable_dom_point_srt and dom_pts_model_pgo is not None and dom_pts_enu_pgo is not None:
            _srt_T, _srt_mask, _srt_stats = _fit_ransac_srt_point_edges(
                dom_pts_model_pgo, dom_pts_enu_pgo, transform, config, int(segment_input.segment_id))
            point_srt_stats = _srt_stats
            if _srt_T is not None and _srt_mask is not None:
                transform = _srt_T
                _prev_t0 = (segment_input.prev_summary or {}).get("metadata", {}).get("t0_transform") if segment_input.prev_summary is not None else None
                if _prev_t0 is None:
                    transform_diag["t0_transform"] = transform
                dom_pts_model_pgo = dom_pts_model_pgo[_srt_mask]
                dom_pts_enu_pgo = dom_pts_enu_pgo[_srt_mask]
                world_enu_final = transform.model_to_world(model_final)
                print(
                    f"[GeoV3][PointSRT] seg {segment_input.segment_id}: accept "
                    f"inliers={_srt_stats.get('inliers', 0)}/{_srt_stats.get('num_points', 0)} "
                    f"ratio={_srt_stats.get('inlier_ratio', 0.0):.2f} "
                    f"s={_srt_stats.get('scale', 0.0):.3f} "
                    f"rot_delta={_srt_stats.get('rot_delta_deg', 0.0):.1f}deg "
                    f"med_xy={_srt_stats.get('median_xy_m', 0.0):.2f}m"
                )
                if _enable_dom_point_srt_rematch and roma_model is not None:
                    print(f"[GeoV3][PointSRT] seg {segment_input.segment_id}: rematching DOM crops with SRT prior")
                    _geo_obs_rematch = collect_geo_observations(
                        segment_input=segment_input,
                        frames=frames,
                        dom_image=dom_image,
                        per_frame_pose=per_frame_pose,
                        roma_model=roma_model,
                        save_vis_dir=save_vis_dir,
                        geo_consist_stride=geo_consist_stride,
                        geo_consist_max_corr=geo_consist_max_corr,
                        project_fn=segment_input.project_fn,
                        inv_project_fn=segment_input.inv_project_fn,
                        geo_elev=geo_elev,
                        dom_transform=dom_transform,
                        device=frames[0]["img"].device,
                        overlap_len=overlap_len,
                        crop_transform=transform,
                        crop_transform_source="point_srt",
                        overlap_prior_cache_reliable_only=bool(getattr(config, "overlap_prior_cache_reliable_only", False)),
                        overlap_prior_cache_max_xy_m=float(getattr(config, "overlap_prior_consistency_max_xy_m", 25.0)),
                        target_mode=geo_target_mode,
                        semantic_masks_enable=bool(getattr(config, "semantic_masks_enable", True)),
                    )
                    if _geo_obs_rematch:
                        geo_obs = _geo_obs_rematch
                        overlap_dom_targets_enu = _build_overlap_dom_targets_enu(
                            geo_obs, int(segment_input.start_frame), int(num_frames), int(overlap_len),
                            target_mode=geo_target_mode)
                        (_dom_pts_model_list, _dom_pts_enu_list, _dom_point_frame_count,
                         _road_pts_model_list, _road_dem_z_list, _road_frame_count,
                         _nonbld_pts_model_list, _nonbld_dem_z_list, _nonbld_frame_count) = _collect_point_samples_from_obs(geo_obs)
                        dom_pts_model_pgo, dom_pts_enu_pgo = _finalize_dom_point_edges(
                            _dom_pts_model_list, _dom_pts_enu_list, _dom_point_frame_count, "srt_rematch")
                        if dom_pts_model_pgo is not None and dom_pts_enu_pgo is not None:
                            _srt_T2, _srt_mask2, _srt_stats2 = _fit_ransac_srt_point_edges(
                                dom_pts_model_pgo, dom_pts_enu_pgo, transform, config, int(segment_input.segment_id))
                            point_srt_stats["rematch"] = _srt_stats2
                            if _srt_T2 is not None and _srt_mask2 is not None:
                                transform = _srt_T2
                                if _prev_t0 is None:
                                    transform_diag["t0_transform"] = transform
                                dom_pts_model_pgo = dom_pts_model_pgo[_srt_mask2]
                                dom_pts_enu_pgo = dom_pts_enu_pgo[_srt_mask2]
                                world_enu_final = transform.model_to_world(model_final)
                                print(
                                    f"[GeoV3][PointSRT] seg {segment_input.segment_id}: rematch accept "
                                    f"inliers={_srt_stats2.get('inliers', 0)}/{_srt_stats2.get('num_points', 0)} "
                                    f"ratio={_srt_stats2.get('inlier_ratio', 0.0):.2f} "
                                    f"med_xy={_srt_stats2.get('median_xy_m', 0.0):.2f}m"
                                )
                    else:
                        point_srt_stats["rematch"] = {"accepted": False, "reason": "no_rematch_obs"}
            else:
                print(
                    f"[GeoV3][PointSRT] seg {segment_input.segment_id}: reject "
                    f"reason={_srt_stats.get('reason')} points={_srt_stats.get('num_points', 0)}"
                )

        if _dom_points_mode:
            dom_point_camera_targets_enu, dom_point_camera_stats = _build_dom_point_camera_targets_from_geo_obs(
                geo_obs=geo_obs,
                frames=frames,
                pts3d_by_frame=_post_pts3d_cache,
                camera_centers_model=model_final,
                transform=transform,
                config=segment_config,
                seg_id=int(segment_input.segment_id),
                road_only=bool(_dom_point_road_only),
                label="post_ttt",
            )

        _ground_ray_seg_enabled = (
            _dom_points_mode
            and _dom_point_ground_ray_stage_enabled(config, "seg_pgo")
            and bool(ground_ray_cache)
        )
        _ground_ray_dirs_for_seg_pgo: Dict[int, np.ndarray] = {}
        _ground_ray_targets_for_seg_pgo: Dict[int, np.ndarray] = {}
        if _ground_ray_seg_enabled:
            for _fi_gr, _entry_gr in sorted(ground_ray_cache.items()):
                _fi_i = int(_fi_gr)
                if _fi_i < 0 or _fi_i >= len(post_ttt_ray_dirs_model):
                    continue
                _target_t = _entry_gr.get("target_enu") if isinstance(_entry_gr, dict) else None
                if _target_t is None:
                    continue
                _dir_np = post_ttt_ray_dirs_model[_fi_i].detach().cpu().float().numpy().astype(np.float64)
                _target_np = _target_t.detach().cpu().float().numpy().astype(np.float64)
                if not (np.isfinite(_dir_np).all() and np.isfinite(_target_np).all()):
                    continue
                _ground_ray_dirs_for_seg_pgo[_fi_i] = _dir_np[:3].copy()
                _ground_ray_targets_for_seg_pgo[_fi_i] = _target_np[:3].copy()

        # --- In-segment PGO (Method-A) ---
        # After KV-only TTT, run a single scipy LS to refine (omega, t, delta);
        # log_s frozen. Replaces fused-Adam path. Writes back to transform and
        # adds delta to model_final so downstream Z-correct, post_ttt_overlap_m0,
        # and dedupe all see the corrected geometry.
        _seg_pgo_t0 = time.perf_counter()
        if (bool(getattr(config, "seg_pgo_enable", False))
                and not bool(getattr(config, "map_anchor_only", False))):
            from .pgo import SegmentPGOConfig as _SegPGOCfg, refine_segment_pgo as _ref_seg_pgo
            # Build dom_targets_enu directly from geo_obs (camera ENU center per fi).
            _dom_tgts: Dict[int, np.ndarray] = {}
            _mt_enu_cache, _ = _build_model_target_cache(
                geo_obs_for_strong_supervision if _overlap_consistency_active else geo_obs,
                query_W=int(frames[0]["img"].shape[-1]),
                query_H=int(frames[0]["img"].shape[-2]),
                target_mode=geo_target_mode,
            )
            for _fi_d, _mt_d in _mt_enu_cache.items():
                _wd = _mt_d.get("world")
                if _wd is None:
                    continue
                _arr = np.asarray(_wd, dtype=np.float64).reshape(-1)
                if _arr.size < 3 or not np.isfinite(_arr[:3]).all():
                    continue
                _dom_tgts[int(_fi_d)] = _arr[:3].copy()
            for _fi_c, _target_c in dom_point_camera_targets_enu.items():
                _arr_c = np.asarray(_target_c, dtype=np.float64).reshape(-1)
                if _arr_c.size >= 3 and np.isfinite(_arr_c[:3]).all():
                    _dom_tgts[int(_fi_c)] = _arr_c[:3].copy()

            _seg_pgo_cfg = _SegPGOCfg(
                enable=True,
                w_dom_xy=float(getattr(config, "seg_pgo_w_dom_xy", 5.0)),
                w_dom_z=float(getattr(config, "seg_pgo_w_dom_z", 2.0)),
                w_consist=float(getattr(config, "seg_pgo_w_consist", 8.0)),
                w_delta_reg=float(getattr(config, "seg_pgo_w_delta_reg", 2.0)),
                w_pose_rot=float(getattr(config, "seg_pgo_w_pose_rot", 50.0)),
                w_pose_trans=float(getattr(config, "seg_pgo_w_pose_trans", 2.0)),
                yaw_only_rotation=bool(
                    getattr(config, "seg_pgo_yaw_only_rotation", False)
                    or (
                        dom_points_mode
                        and bool(getattr(
                            config,
                            "dom_point_bootstrap_fixed_rotation_refit_enable",
                            False,
                        ))
                    )
                ),
                use_point_edges=bool(getattr(config, "seg_pgo_use_point_edges", False)),
                w_point_xy=float(getattr(config, "seg_pgo_w_point_xy", 0.5)),
                w_point_z=float(getattr(config, "seg_pgo_w_point_z", 0.1)),
                use_ground_ray=bool(_ground_ray_dirs_for_seg_pgo),
                w_ground_ray_xy=float(getattr(config, "dom_point_ground_ray_seg_pgo_w_xy", 2.0)),
                ground_ray_min_abs_dir_z=float(getattr(config, "dom_point_ground_ray_min_abs_dir_z", 0.05)),
                freeze_delta=bool(getattr(config, "seg_pgo_freeze_delta", True)),
                max_iter=int(getattr(config, "seg_pgo_max_iter", 20)),
                delta_clip=float(getattr(config, "seg_pgo_delta_clip", 5.0)),
                verbose=bool(getattr(config, "seg_pgo_verbose", False)),
            )
            _seg_pgo_use_overlap_consist = bool(getattr(config, "seg_pgo_use_overlap_consistency", False))
            _fixed_rotation_bootstrap_accepted = bool(
                isinstance(point_bootstrap_diag, dict)
                and point_bootstrap_diag.get("accepted", False)
                and isinstance(point_bootstrap_diag.get("fixed_rotation_refit"), dict)
                and point_bootstrap_diag["fixed_rotation_refit"].get("accepted", False)
            )
            _seg_pgo_skip_overlap_only = _segment_pgo_should_skip_overlap_only(
                dom_points_mode=bool(_dom_points_mode),
                fixed_rotation_bootstrap_accepted=_fixed_rotation_bootstrap_accepted,
                num_dom_targets=len(_dom_tgts),
                has_point_edges=(
                    dom_pts_model_pgo is not None and dom_pts_enu_pgo is not None
                ),
                has_ground_ray_edges=bool(_ground_ray_dirs_for_seg_pgo),
            )
            if _seg_pgo_skip_overlap_only:
                _seg_pgo_use_overlap_consist = False
                transform_diag["seg_pgo_overlap_consistency"] = {
                    "enabled": False,
                    "reason": "fixed_rotation_map_transform_no_absolute_edges",
                    "available_overlap": int(
                        min(
                            int(overlap_len),
                            0 if _prev_post_ttt is None else len(_prev_post_ttt),
                        )
                    ),
                }
                print(
                    f"[GeoV3][SegPGO] seg {segment_input.segment_id}: skip overlap-only "
                    "refit; preserving accepted fixed-R map transform"
                )
            if overlap_len > 0 and _prev_post_ttt is not None and not _seg_pgo_use_overlap_consist:
                if not _seg_pgo_skip_overlap_only:
                    transform_diag["seg_pgo_overlap_consistency"] = {
                        "enabled": False,
                        "reason": "bootstrap_only",
                        "available_overlap": int(min(int(overlap_len), len(_prev_post_ttt))),
                    }
                    print(
                        f"[GeoV3][SegPGO] seg {segment_input.segment_id}: "
                        f"overlap consistency edge disabled (bootstrap-only prior)"
                    )
            _model_pts_np = model_final.detach().cpu().float().numpy().astype(np.float64)
            _t0_for_pgo = transform_diag.get("t0_transform")
            _refined_T, _delta_np, _seg_pgo_stats = _ref_seg_pgo(
                model_pts=_model_pts_np,
                transform_init=transform,
                dom_targets_enu=_dom_tgts,
                prev_M0_overlap=(np.asarray(_prev_post_ttt, dtype=np.float64)
                                 if (_seg_pgo_use_overlap_consist and _prev_post_ttt is not None) else None),
                t0_transform=(_t0_for_pgo if _seg_pgo_use_overlap_consist else None),
                overlap_len=(int(overlap_len) if _seg_pgo_use_overlap_consist else 0),
                seg_id=int(segment_input.segment_id),
                cfg=_seg_pgo_cfg,
                point_model=dom_pts_model_pgo,
                point_targets_enu=dom_pts_enu_pgo,
                ground_ray_dirs_model=_ground_ray_dirs_for_seg_pgo,
                ground_ray_targets_enu=_ground_ray_targets_for_seg_pgo,
            )
            _pgo_accept = not _seg_pgo_skip_overlap_only
            # Gate SegPGO unconditionally whenever overlap frames and a previous
            # segment reference are available.  Scale is already locked by the
            # overlap-prior path, so the gate only guards against large rotational
            # or translational outliers from the PGO solver.
            _pgo_gate_mode: Optional[str] = None
            if (
                not _seg_pgo_skip_overlap_only
                and overlap_len > 0
                and _prev_post_ttt is not None
            ):
                _pgo_gate_mode = "overlap_continuity"
                _max_omega = float(getattr(config, "seg_pgo_bootstrap_max_omega_rad", 0.25))
                _max_motion = float(getattr(config, "seg_pgo_bootstrap_max_overlap_motion_m", 5.0))
                _max_prev_overlap = float(getattr(config, "seg_pgo_bootstrap_max_prev_overlap_xy_m", 5.0))
                _max_t_delta = float(getattr(config, "seg_pgo_bootstrap_max_t_delta_m", 15.0))
            if _pgo_gate_mode is not None:
                try:
                    _ov_len_pgo = max(0, min(int(overlap_len), int(model_final.shape[0])))
                    if _ov_len_pgo > 0:
                        _delta_gate_t = torch.tensor(_delta_np[:_ov_len_pgo], device=model_final.device, dtype=model_final.dtype)
                        _before = transform.model_to_world(model_final[:_ov_len_pgo]).detach().cpu().numpy()
                        _after = _refined_T.model_to_world(model_final[:_ov_len_pgo] + _delta_gate_t).detach().cpu().numpy()
                        _motion = np.linalg.norm(_after[:, :2] - _before[:, :2], axis=1)
                        _overlap_motion = float(np.median(_motion)) if _motion.size else 0.0
                    else:
                        _before = None
                        _after = None
                        _overlap_motion = 0.0
                except Exception:
                    _before = None
                    _after = None
                    _overlap_motion = float("inf")
                _has_prev_overlap_ref = False
                _pre_prev_overlap = 0.0
                _post_prev_overlap = 0.0
                try:
                    if (
                        _before is not None and _after is not None
                        and _prev_post_ttt is not None and _t0_for_pgo is not None
                    ):
                        _prev_m0 = np.asarray(_prev_post_ttt, dtype=np.float64)
                        _n_prev = max(0, min(int(_before.shape[0]), int(_prev_m0.shape[0])))
                        if _n_prev > 0:
                            _prev_t = torch.tensor(
                                _prev_m0[:_n_prev, :3],
                                device=model_final.device,
                                dtype=model_final.dtype,
                            )
                            _prev_enu = _t0_for_pgo.model_to_world(_prev_t).detach().cpu().numpy()
                            _pre_prev_diff = np.linalg.norm(_before[:_n_prev, :2] - _prev_enu[:, :2], axis=1)
                            _post_prev_diff = np.linalg.norm(_after[:_n_prev, :2] - _prev_enu[:, :2], axis=1)
                            _pre_prev_overlap = float(np.median(_pre_prev_diff)) if _pre_prev_diff.size else 0.0
                            _post_prev_overlap = float(np.median(_post_prev_diff)) if _post_prev_diff.size else 0.0
                            _has_prev_overlap_ref = True
                except Exception:
                    _pre_prev_overlap = float("inf")
                    _post_prev_overlap = float("inf")
                    _has_prev_overlap_ref = True
                _omega = float(_seg_pgo_stats.get("omega_delta", float("inf")))
                _t_delta = float(_seg_pgo_stats.get("t_delta", float("inf")))
                _pgo_accept = (
                    np.isfinite(_omega) and _omega <= _max_omega
                    and np.isfinite(_overlap_motion) and _overlap_motion <= _max_motion
                    and np.isfinite(_t_delta) and _t_delta <= _max_t_delta
                    and (
                        (not _has_prev_overlap_ref)
                        or (np.isfinite(_post_prev_overlap) and _post_prev_overlap <= _max_prev_overlap)
                    )
                )
                _gate_base_mode = str(_pgo_gate_mode)
                _reanchor_enabled = bool(getattr(config, "dom_point_seg_pgo_absolute_reanchor_enable", True))
                _point_n = int(_seg_pgo_stats.get("point_n", 0) or 0)
                _ground_ray_n = int(_seg_pgo_stats.get("ground_ray_n", 0) or 0)
                _min_reanchor_points = int(getattr(config, "dom_point_seg_pgo_reanchor_min_points", 32))
                _min_reanchor_rays = int(getattr(config, "dom_point_seg_pgo_reanchor_min_ground_rays", 4))
                _init_cost = float(_seg_pgo_stats.get("init_cost", float("inf")))
                _final_cost = float(_seg_pgo_stats.get("final_cost", float("inf")))
                _cost_ratio = float("inf")
                if np.isfinite(_init_cost) and _init_cost > 1e-9 and np.isfinite(_final_cost):
                    _cost_ratio = float(_final_cost / max(_init_cost, 1e-9))
                _max_reanchor_ratio = float(getattr(config, "dom_point_seg_pgo_reanchor_max_cost_ratio", 0.50))
                _reanchor_abs_edges_ok = (_point_n >= _min_reanchor_points) or (_ground_ray_n >= _min_reanchor_rays)
                _reanchor_cost_ok = (
                    np.isfinite(_cost_ratio)
                    and _cost_ratio <= _max_reanchor_ratio
                    and np.isfinite(_final_cost)
                    and np.isfinite(_init_cost)
                    and _final_cost < _init_cost
                )
                _reanchor_overlap_blocked = (
                    (np.isfinite(_overlap_motion) and _overlap_motion > _max_motion)
                    or (np.isfinite(_t_delta) and _t_delta > _max_t_delta)
                    or (
                        _has_prev_overlap_ref
                        and np.isfinite(_post_prev_overlap)
                        and _post_prev_overlap > _max_prev_overlap
                    )
                )
                _reanchor_omega_ok = np.isfinite(_omega) and _omega <= _max_omega
                _dom_points_reanchor = bool(
                    (not _pgo_accept)
                    and _dom_points_mode
                    and _reanchor_enabled
                    and _reanchor_abs_edges_ok
                    and _reanchor_cost_ok
                    and not _reanchor_overlap_blocked
                    and _reanchor_omega_ok
                )
                _reanchor_reason = "not_needed" if _pgo_accept else "not_eligible"
                if _dom_points_reanchor:
                    _pgo_accept = True
                    _pgo_gate_mode = "dom_points_absolute_reanchor"
                    _reanchor_reason = "absolute_dom_evidence_cost_drop"
                transform_diag["seg_pgo_gate"] = {
                    "accepted": bool(_pgo_accept),
                    "mode": str(_pgo_gate_mode),
                    "base_mode": _gate_base_mode,
                    "omega_delta_rad": float(_omega),
                    "max_omega_rad": float(_max_omega),
                    "overlap_motion_m": float(_overlap_motion),
                    "max_overlap_motion_m": float(_max_motion),
                    "pre_prev_overlap_xy_m": float(_pre_prev_overlap),
                    "post_prev_overlap_xy_m": float(_post_prev_overlap),
                    "max_prev_overlap_xy_m": float(_max_prev_overlap),
                    "has_prev_overlap_ref": bool(_has_prev_overlap_ref),
                    "t_delta_m": float(_t_delta),
                    "max_t_delta_m": float(_max_t_delta),
                    "dom_points_absolute_reanchor": bool(_dom_points_reanchor),
                    "reanchor_reason": str(_reanchor_reason),
                    "reanchor_cost_ratio": float(_cost_ratio),
                    "reanchor_max_cost_ratio": float(_max_reanchor_ratio),
                    "reanchor_point_n": int(_point_n),
                    "reanchor_ground_ray_n": int(_ground_ray_n),
                }
                if _pgo_accept:
                    print(
                        f"[GeoV3][SegPGO-gate] seg {segment_input.segment_id}: accept "
                        f"mode={_pgo_gate_mode} omega={_omega:.4f}rad "
                        f"overlap_motion={_overlap_motion:.2f}m "
                        f"post_prev={_post_prev_overlap:.2f}m "
                        f"t_delta={_t_delta:.2f}m"
                    )
                else:
                    print(
                        f"[GeoV3][SegPGO-gate] seg {segment_input.segment_id}: reject "
                        f"mode={_pgo_gate_mode} omega={_omega:.4f}/{_max_omega:.4f}rad "
                        f"overlap_motion={_overlap_motion:.2f}/{_max_motion:.2f}m "
                        f"post_prev={_post_prev_overlap:.2f}/{_max_prev_overlap:.2f}m "
                        f"t_delta={_t_delta:.2f}/{_max_t_delta:.2f}m"
                    )
            if _pgo_accept:
                # Write back refined T_k (R, t; s frozen)
                transform.R = _refined_T.R.to(device=transform.R.device, dtype=transform.R.dtype)
                transform.t = _refined_T.t.to(device=transform.t.device, dtype=transform.t.dtype)
                # Apply delta to model_final
                _delta_t = torch.tensor(_delta_np, device=model_final.device, dtype=model_final.dtype)
                model_final = model_final + _delta_t
                world_enu_final = transform.model_to_world(model_final)
            print(
                f"[GeoV3][SegPGO] seg {segment_input.segment_id}: "
                f"dom={_seg_pgo_stats['dom_n']} consist={_seg_pgo_stats['consist_n']} "
                f"points={_seg_pgo_stats.get('point_n', 0)} "
                f"ground_rays={_seg_pgo_stats.get('ground_ray_n', 0)} "
                f"init={_seg_pgo_stats['init_cost']:.1f} final={_seg_pgo_stats['final_cost']:.1f} "
                f"nfev={_seg_pgo_stats['nfev']} "
                f"|t|={_seg_pgo_stats['t_delta']:.3f}m "
                f"|omega|={_seg_pgo_stats['omega_delta']:.4f}rad "
                f"|delta|max={_seg_pgo_stats['delta_max']:.3f}m "
                f"mean={_seg_pgo_stats['delta_mean']:.3f}m "
                f"applied={_pgo_accept}"
            )

        _record_phase("segment_pgo_s", _seg_pgo_t0)

        # --- DEM-anchored Z offset correction (road-only, no AGL) ---
        # For each road DOM/DEM pixel in the segment, we have:
        #   pred_ground_model = pts3d_fi sampled at (col, row) of the query pixel
        #   true_ground_z     = DEM elevation at the same pixel (enu_positions[.., 2])
        # Map pred to ENU via T_k and take median(true - pred_z) as dz.
        # This is pure DEM alignment (no AGL estimate), so no dependence on
        # camera scale/AGL. Only transform.t[2] is touched.
        # Also subsample (model-space road pts ↔ DEM Z) for the global PGO
        # DEM-Z edge (see pgo.py::_DemZEdge).
        _dem_z_t0 = time.perf_counter()
        _dem_z_stats: Dict[str, Any] = {
            "enabled": bool(getattr(config, "dem_z_correct_enable", True))
            and not bool(getattr(config, "map_anchor_only", False)),
            "direct_enabled": bool(getattr(config, "dem_z_road_direct_correct_enable", False))
            and bool(getattr(config, "dem_z_direct_correct_enable", True))
            and not bool(getattr(config, "map_anchor_only", False)),
            "direct_applied": False,
            "quality_ok": False,
            "reason": "not_run",
            "dz_m": None,
            "dz_std_m": None,
            "n_frames": 0,
            "n_points": 0,
            "sample_source": "none",
            "pgo_edges_enabled": False,
        }
        _PGO_ROAD_MAX = 200          # cap per-segment road-pt count for PGO speed
        road_pts_model_pgo: Optional[np.ndarray] = None
        road_dem_z_pgo: Optional[np.ndarray] = None
        dem_z_sample_source = "none"
        camera_z_pts_model_pgo: Optional[np.ndarray] = None
        camera_z_target_pgo: Optional[np.ndarray] = None
        camera_z_sample_source = "none"
        _z_min_frames = int(getattr(config, "dem_z_correct_min_frames", 5)) if config is not None else 5
        _z_min_points = int(getattr(config, "dem_z_correct_min_points", 50)) if config is not None else 50
        _road_z_points = int(sum(int(_pts.shape[0]) for _pts in _road_pts_model_list)) if _road_pts_model_list else 0
        _use_nonbld_z_fallback = (
            _dom_point_z_nonbuilding_fallback
            and bool(_nonbld_pts_model_list)
            and (_road_frame_count < _z_min_frames or _road_z_points < _z_min_points)
        )
        if _use_nonbld_z_fallback:
            _z_pts_model_list = _nonbld_pts_model_list
            _z_dem_z_list = _nonbld_dem_z_list
            _z_frame_count = _nonbld_frame_count
            _z_sample_label = "nonbuilding_fallback"
        else:
            _z_pts_model_list = _road_pts_model_list
            _z_dem_z_list = _road_dem_z_list
            _z_frame_count = _road_frame_count
            _z_sample_label = "road"
        _dem_z_stats.update({
            "sample_source": str(_z_sample_label),
            "n_frames": int(_z_frame_count),
            "n_points": int(sum(int(_pts.shape[0]) for _pts in _z_pts_model_list))
            if _z_pts_model_list else 0,
        })
        if _z_pts_model_list:
            dem_z_sample_source = _z_sample_label
            _road_pts_model_cat = torch.cat(_z_pts_model_list, dim=0)  # (K_tot, 3) M_k
            _road_dem_z_cat = np.concatenate(_z_dem_z_list, axis=0)     # (K_tot,)
            # ---- subsample for global PGO ----
            _K_full = int(_road_pts_model_cat.shape[0])
            if _K_full > 0:
                if _K_full > _PGO_ROAD_MAX:
                    _idx_sub = np.random.RandomState(0).choice(
                        _K_full, _PGO_ROAD_MAX, replace=False
                    )
                    _idx_sub.sort()
                    road_pts_model_pgo = (
                        _road_pts_model_cat[_idx_sub]
                        .detach().cpu().float().numpy().astype(np.float32)
                    )
                    road_dem_z_pgo = _road_dem_z_cat[_idx_sub].astype(np.float64)
                else:
                    road_pts_model_pgo = (
                        _road_pts_model_cat.detach().cpu().float().numpy().astype(np.float32)
                    )
                    road_dem_z_pgo = _road_dem_z_cat.astype(np.float64)
            _road_world = transform.model_to_world(_road_pts_model_cat).detach().cpu().numpy()
            _pred_z_arr = _road_world[:, 2].astype(np.float64)
            _dz_arr = _road_dem_z_cat - _pred_z_arr                        # (K_tot,)
            _K_tot = int(_dz_arr.size)
            if _K_tot >= 3:
                _dz = float(np.median(_dz_arr))
                _dz_std = float(np.std(_dz_arr))
                _agl_samples = []
                for _corr in geo_obs.values():
                    if _corr is None:
                        continue
                    _ga = _corr.get("gt_agl")
                    if _ga is None:
                        continue
                    _ga = float(_ga)
                    if 20.0 < _ga < 2000.0:
                        _agl_samples.append(_ga)
                _agl_ref = float(np.median(_agl_samples)) if _agl_samples else None
                _dz_limit = max(200.0, 1.5 * _agl_ref) if _agl_ref is not None else 200.0
                _dem_z_enabled = (
                    bool(getattr(config, "dem_z_correct_enable", True))
                    and not bool(getattr(config, "map_anchor_only", False))
                ) if config is not None else True
                _z_diag_only = bool(getattr(config, "dem_z_diagnostic_only", False)) if config is not None else False
                _direct_enabled = (
                    _dem_z_enabled
                    and not _z_diag_only
                    and bool(getattr(config, "dem_z_direct_correct_enable", True))
                    and _dem_z_direct_sample_allowed(_z_sample_label, config)
                    and bool(getattr(config, "dem_z_road_direct_correct_enable", False))
                )
                _pgo_edges_enabled = (
                    _dem_z_enabled
                    and not _z_diag_only
                    and bool(getattr(config, "dem_z_pgo_edges_enable", True))
                )
                _dem_z_stats.update({
                    "enabled": bool(_dem_z_enabled),
                    "direct_enabled": bool(_direct_enabled),
                    "pgo_edges_enabled": bool(_pgo_edges_enabled),
                    "dz_m": float(_dz),
                    "dz_std_m": float(_dz_std),
                    "n_frames": int(_z_frame_count),
                    "n_points": int(_K_tot),
                    "sample_source": str(_z_sample_label),
                })
                _z_std_limit = float(getattr(config, "dem_z_correct_max_std_m", 8.0)) if config is not None else 8.0
                _dz_limit = float(getattr(config, "dem_z_correct_max_abs_m", 20.0)) if config is not None else 20.0
                _legacy_z_gate = _overlap_primary_active and bool(getattr(config, "overlap_prior_gate_z_correct", False))
                if _legacy_z_gate:
                    _dz_limit = min(_dz_limit, float(getattr(config, "overlap_prior_z_correct_max_abs_m", 50.0)))
                    _z_std_limit = min(_z_std_limit, float(getattr(config, "overlap_prior_z_correct_max_std_m", 80.0)))
                # Diagnostic: show pred_z and dem_z medians, plus transform scalars
                _pred_z_med = float(np.median(_pred_z_arr))
                _dem_z_med = float(np.median(_road_dem_z_cat))
                _s_val = float(transform.s.detach().cpu())
                _tz_val = float(transform.t[2].detach().cpu())
                print(f"[GeoV3][Z-diag]  seg {segment_input.segment_id}: "
                      f"pred_z_med={_pred_z_med:+.2f}m  dem_z_med={_dem_z_med:+.2f}m  "
                      f"s={_s_val:.3f}  t_z={_tz_val:+.3f}m  K={_K_tot} frames={_z_frame_count} "
                        f"dz_lim={_dz_limit:.1f}m std_lim={_z_std_limit:.1f}m "
                        f"enabled={_dem_z_enabled} direct={_direct_enabled} "
                        f"pgo_edge={_pgo_edges_enabled} diag_only={_z_diag_only}")
                _quality_ok = True
                _z_reject_reason = None
                if not _dem_z_enabled:
                    _quality_ok = False
                    _z_reject_reason = "disabled"
                elif _z_frame_count < _z_min_frames:
                    _quality_ok = False
                    _z_reject_reason = f"frames<{_z_min_frames}"
                elif _K_tot < _z_min_points:
                    _quality_ok = False
                    _z_reject_reason = f"pts<{_z_min_points}"
                elif _dz_std > _z_std_limit:
                    _quality_ok = False
                    _z_reject_reason = f"std>{_z_std_limit:.1f}m"
                elif abs(_dz) > _dz_limit:
                    _quality_ok = False
                    _z_reject_reason = f"abs_dz>{_dz_limit:.1f}m"
                _dem_z_stats["quality_ok"] = bool(_quality_ok)

                if not (_quality_ok and _pgo_edges_enabled):
                    road_pts_model_pgo = None
                    road_dem_z_pgo = None
                    if not _quality_ok:
                        dem_z_sample_source = "none"
                    elif _z_diag_only:
                        dem_z_sample_source = "diagnostic_road"
                    elif not _pgo_edges_enabled:
                        dem_z_sample_source = "road_direct_only"

                if _quality_ok and _direct_enabled:
                    _new_t = transform.t.detach().clone()
                    _new_t[2] = _new_t[2] + _dz
                    transform.t = _new_t
                    world_enu_final = transform.model_to_world(model_final)
                    _dem_z_stats["direct_applied"] = True
                    _dem_z_stats["reason"] = "applied"
                    print(f"[GeoV3][Z-correct] seg {segment_input.segment_id}: "
                          f"dz={_dz:+.3f}m std={_dz_std:.3f}m from "
                          f"{_z_frame_count} frames / {_K_tot} {_z_sample_label} pts "
                          f"[dem-aligned, pgo_edge={_pgo_edges_enabled}]")
                else:
                    if _quality_ok:
                        if _z_diag_only:
                            _z_reject_reason = "diagnostic_only"
                        elif not _direct_enabled:
                            _z_reject_reason = "direct_disabled"
                        else:
                            _z_reject_reason = "direct_skipped"
                    _dem_z_stats["reason"] = str(_z_reject_reason or "rejected")
                    print(f"[GeoV3][Z-correct] seg {segment_input.segment_id}: "
                          f"dz={_dz:+.3f}m std={_dz_std:.3f}m skipped "
                          f"reason={_z_reject_reason} "
                          f"[dem-aligned, pgo_edge={_pgo_edges_enabled and _quality_ok}]")
            else:
                _dem_z_stats["reason"] = f"insufficient_points<{_K_tot}"
                print(f"[GeoV3][Z-correct] seg {segment_input.segment_id}: "
                      f"insufficient {_z_sample_label} pts ({_K_tot}), skipping Z correction")
        else:
            _dem_z_stats["reason"] = "no_correspondences"
            if _nonbld_pts_model_list:
                _nonbld_K = int(sum(int(_pts.shape[0]) for _pts in _nonbld_pts_model_list))
                print(f"[GeoV3][Z-correct] seg {segment_input.segment_id}: "
                      f"no road correspondences, skipping Z correction "
                      f"(non-building diagnostics: {_nonbld_frame_count} frames / {_nonbld_K} pts)")
            else:
                print(f"[GeoV3][Z-correct] seg {segment_input.segment_id}: "
                      f"no road correspondences, skipping Z correction")

        # Camera center DEM+AGL correction. This is the primary Z signal for
        # oblique UAVScene: it constrains camera height directly and does not
        # require road samples. AGL comes from the method's first-frame anchor,
        # not from GT diagnostic fields or drifting render-pose estimates.
        _camera_agl_ref = 435.0
        if geo_elev is not None:
            try:
                _ground_alt_ref = geo_elev.elevation_at_lonlat(geo_elev.lon0, geo_elev.lat0)
                _camera_agl_ref = max(50.0, float(geo_elev.alt0) - float(_ground_alt_ref))
            except Exception:
                pass
        _camera_z_config_enabled = (
            not bool(getattr(config, "map_anchor_only", False))
            and (
                bool(getattr(config, "dem_z_camera_correct_enable", True))
                or bool(getattr(config, "dem_z_camera_pgo_edges_enable", True))
            )
        )
        _cam_z_stats: Dict[str, Any] = {
            "enabled": bool(_camera_z_config_enabled),
            "applied": False,
            "n": 0,
            "agl_ref_m": float(_camera_agl_ref),
        }
        _dem_z_enabled_cam = (
            bool(getattr(config, "dem_z_correct_enable", True))
            and not bool(getattr(config, "map_anchor_only", False))
        ) if config is not None else True
        _z_diag_only_cam = bool(getattr(config, "dem_z_diagnostic_only", False)) if config is not None else False
        _cam_direct_enabled = (
            _dem_z_enabled_cam
            and not _z_diag_only_cam
            and bool(getattr(config, "dem_z_direct_correct_enable", True))
            and bool(getattr(config, "dem_z_camera_correct_enable", True))
        )
        _cam_pgo_enabled = (
            _dem_z_enabled_cam
            and not _z_diag_only_cam
            and bool(getattr(config, "dem_z_camera_pgo_edges_enable", True))
        )
        _cam_model_samples: List[np.ndarray] = []
        _cam_target_z_samples: List[float] = []
        _cam_pred_z_samples: List[float] = []
        _cam_ground_z_samples: List[float] = []
        if _dem_z_enabled_cam and _camera_z_config_enabled and geo_obs:
            try:
                _world_cam_np = world_enu_final.detach().cpu().float().numpy()
                _model_cam_np = model_final.detach().cpu().float().numpy()
                _dom_points_camera_z = (
                    _dom_points_mode
                    and bool(getattr(config, "dom_point_bootstrap_agl_z_recenter_enable", True))
                )
                _camera_z_max_pitch_deg = max(
                    0.0,
                    float(getattr(config, "dom_point_bootstrap_camera_recenter_max_pitch_deg", 12.0)),
                )
                for _fi_cam, _corr_cam in sorted(geo_obs.items(), key=lambda kv: int(kv[0])):
                    if _corr_cam is None:
                        continue
                    _idx_cam = int(_fi_cam)
                    if _idx_cam < 0 or _idx_cam >= int(_world_cam_np.shape[0]):
                        continue
                    _pred_cam = np.asarray(_world_cam_np[_idx_cam], dtype=np.float64).reshape(-1)
                    if _pred_cam.size < 3 or not np.all(np.isfinite(_pred_cam[:3])):
                        continue
                    if _dom_points_camera_z:
                        _crop_info_cam = _corr_cam.get("dom_crop_info") or {}
                        _render_pitch_cam = (
                            _finite_float_optional(_crop_info_cam.get("render_pitch_deg"))
                            if isinstance(_crop_info_cam, dict) else None
                        )
                        if _render_pitch_cam is None or abs(float(_render_pitch_cam)) > _camera_z_max_pitch_deg:
                            continue
                        _ground_z_cam, _ground_source_cam = _ground_z_from_corr_for_ray(_corr_cam)
                        _agl_cam = _dom_point_agl_from_corr(_corr_cam)
                        if _ground_z_cam is None or _agl_cam is None:
                            continue
                        _target_z_cam = float(_ground_z_cam) + float(_agl_cam)
                    else:
                        _ground_z_cam = _dem_ground_z_at_enu_xy(
                            _pred_cam[:2], segment_input.project_fn, geo_elev, dom_transform)
                        if _ground_z_cam is None or not np.isfinite(float(_ground_z_cam)):
                            continue
                        _target_z_cam = float(_ground_z_cam) + float(_camera_agl_ref)
                    if not np.isfinite(_target_z_cam):
                        continue
                    _cam_model_samples.append(np.asarray(_model_cam_np[_idx_cam], dtype=np.float32).copy())
                    _cam_target_z_samples.append(_target_z_cam)
                    _cam_pred_z_samples.append(float(_pred_cam[2]))
                    _cam_ground_z_samples.append(float(_ground_z_cam))
            except Exception as _e:
                _cam_z_stats["reason"] = f"sample_failed:{_e}"

        _cam_z_n = int(len(_cam_target_z_samples))
        _cam_z_stats["n"] = _cam_z_n
        if _cam_z_n > 0:
            _cam_target_z_arr = np.asarray(_cam_target_z_samples, dtype=np.float64)
            _cam_pred_z_arr = np.asarray(_cam_pred_z_samples, dtype=np.float64)
            _cam_ground_z_arr = np.asarray(_cam_ground_z_samples, dtype=np.float64)
            _cam_dz_arr = _cam_target_z_arr - _cam_pred_z_arr
            _cam_dz = float(np.median(_cam_dz_arr))
            _cam_dz_std = float(np.std(_cam_dz_arr))
            _cam_z_stats.update({
                "dz_m": _cam_dz,
                "std_m": _cam_dz_std,
                "target_z_med_m": float(np.median(_cam_target_z_arr)),
                "pred_z_med_m": float(np.median(_cam_pred_z_arr)),
                "ground_z_med_m": float(np.median(_cam_ground_z_arr)),
                "direct_enabled": bool(_cam_direct_enabled),
                "pgo_edge_enabled": bool(_cam_pgo_enabled),
            })
            if _cam_pgo_enabled:
                camera_z_pts_model_pgo = np.stack(_cam_model_samples, axis=0).astype(np.float32)
                camera_z_target_pgo = _cam_target_z_arr.astype(np.float64)
                camera_z_sample_source = "dom_frame_center_agl" if _dom_points_mode else "camera_dem_agl"
            if _cam_direct_enabled:
                _new_t = transform.t.detach().clone()
                _new_t[2] = _new_t[2] + _cam_dz
                transform.t = _new_t
                world_enu_final = transform.model_to_world(model_final)
                _cam_z_stats["applied"] = True
                print(
                    f"[GeoV3][Z-camera-dem] seg {segment_input.segment_id}: "
                    f"dz={_cam_dz:+.3f}m std={_cam_dz_std:.3f}m n={_cam_z_n} "
                    f"pred_z_med={float(np.median(_cam_pred_z_arr)):+.2f}m "
                    f"target_z_med={float(np.median(_cam_target_z_arr)):+.2f}m "
                    f"ground_z_med={float(np.median(_cam_ground_z_arr)):+.2f}m "
                    f"agl_ref={float(_camera_agl_ref):.1f}m pgo_edge={_cam_pgo_enabled} applied=True"
                )
            else:
                _reason = "diagnostic_only" if _z_diag_only_cam else "direct_disabled"
                _cam_z_stats["reason"] = _reason
                print(
                    f"[GeoV3][Z-camera-dem] seg {segment_input.segment_id}: "
                    f"dz={_cam_dz:+.3f}m std={_cam_dz_std:.3f}m n={_cam_z_n} "
                    f"target_z_med={float(np.median(_cam_target_z_arr)):+.2f}m "
                    f"agl_ref={float(_camera_agl_ref):.1f}m skipped reason={_reason} "
                    f"pgo_edge={_cam_pgo_enabled}"
                )
        elif _camera_z_config_enabled:
            _cam_z_stats["reason"] = "no_dem_camera_samples"
            print(
                f"[GeoV3][Z-camera-dem] seg {segment_input.segment_id}: "
                f"no camera DEM samples, skipping camera Z correction"
            )
        else:
            _cam_z_stats["reason"] = "disabled"

        # Raw render+fixedR PnP-Z is opt-in only. In oblique near-planar scenes
        # it is much weaker than the ground-footprint XY signal, so it should
        # not silently become a hard camera-Z correction.
        _pnp_z_stats: Dict[str, Any] = {
            "enabled": bool(getattr(config, "pnp_camera_z_correct_enable", True)),
            "applied": False,
            "n": 0,
        }
        _allow_raw_pnp_z = bool(getattr(config, "allow_raw_pnp_camera_targets", False))
        if bool(getattr(config, "pnp_camera_z_correct_enable", True)) and not _allow_raw_pnp_z:
            _pnp_z_stats.update({"enabled": False, "reason": "raw_pnp_target_disabled"})
            print(
                f"[GeoV3][Z-camera] seg {segment_input.segment_id}: "
                f"skipping raw fixedR PnP-Z (allow_raw_pnp_camera_targets=False)"
            )
        elif bool(getattr(config, "pnp_camera_z_correct_enable", True)) and geo_obs:
            _pnp_dz_vals: List[float] = []
            for _fi_z, _corr_z in geo_obs.items():
                if _corr_z is None or int(_fi_z) < 0 or int(_fi_z) >= int(model_final.shape[0]):
                    continue
                _target_z_enu, _target_z_source = _geo_camera_target_enu(
                    _corr_z,
                    prefer_ground_signal=False,
                    prefer_pnp=True,
                    allow_raw_pnp=True,
                )
                if _target_z_source != "fixedR_pnp" or _target_z_enu is None or len(_target_z_enu) < 3:
                    continue
                with torch.no_grad():
                    _pred_cam_z = float(
                        transform.model_to_world(model_final[int(_fi_z):int(_fi_z) + 1])[0, 2]
                        .detach().cpu().item()
                    )
                _target_cam_z = float(_target_z_enu[2])
                if np.isfinite(_target_cam_z) and np.isfinite(_pred_cam_z):
                    _pnp_dz_vals.append(_target_cam_z - _pred_cam_z)

            _pnp_z_n = int(len(_pnp_dz_vals))
            _pnp_z_stats["n"] = _pnp_z_n
            _pnp_z_min = int(getattr(config, "pnp_camera_z_correct_min_frames", 2))
            if _pnp_z_n >= _pnp_z_min:
                _pnp_dz_arr = np.asarray(_pnp_dz_vals, dtype=np.float64)
                _pnp_dz = float(np.median(_pnp_dz_arr))
                _pnp_dz_std = float(np.std(_pnp_dz_arr))
                _pnp_max_abs = float(getattr(config, "pnp_camera_z_correct_max_abs_m", 80.0))
                _pnp_max_std = float(getattr(config, "pnp_camera_z_correct_max_std_m", 30.0))
                _pnp_accept = abs(_pnp_dz) <= _pnp_max_abs and _pnp_dz_std <= _pnp_max_std
                _pnp_z_stats.update({
                    "dz_m": _pnp_dz,
                    "std_m": _pnp_dz_std,
                    "max_abs_m": _pnp_max_abs,
                    "max_std_m": _pnp_max_std,
                    "accepted": bool(_pnp_accept),
                })
                if _pnp_accept:
                    _new_t = transform.t.detach().clone()
                    _new_t[2] = _new_t[2] + _pnp_dz
                    transform.t = _new_t
                    world_enu_final = transform.model_to_world(model_final)
                    _pnp_z_stats["applied"] = True
                    if bool(getattr(config, "pnp_camera_z_skip_dem_edges", True)):
                        road_pts_model_pgo = None
                        road_dem_z_pgo = None
                        dem_z_sample_source = "skipped_camera_pnp_z"
                        _pnp_z_stats["dem_edges_skipped"] = True
                    print(
                        f"[GeoV3][Z-camera] seg {segment_input.segment_id}: "
                        f"dz={_pnp_dz:+.3f}m std={_pnp_dz_std:.3f}m "
                        f"n={_pnp_z_n} source=fixedR_pnp applied=True"
                    )
                else:
                    _reason = "std" if _pnp_dz_std > _pnp_max_std else "abs_dz"
                    _pnp_z_stats["reason"] = _reason
                    print(
                        f"[GeoV3][Z-camera] seg {segment_input.segment_id}: "
                        f"dz={_pnp_dz:+.3f}m std={_pnp_dz_std:.3f}m "
                        f"n={_pnp_z_n} skipped reason={_reason}"
                    )
            else:
                _pnp_z_stats["reason"] = f"insufficient_frames<{_pnp_z_min}"
                print(
                    f"[GeoV3][Z-camera] seg {segment_input.segment_id}: "
                    f"insufficient fixedR_pnp frames ({_pnp_z_n}/{_pnp_z_min}), skipping"
                )
        transform_diag["pnp_camera_z_correct"] = _pnp_z_stats

        _cam_z_diag: Dict[str, Any] = {
            "enabled": bool(getattr(config, "camera_z_diagnostic_enable", True)),
            "n": 0,
        }
        if bool(getattr(config, "camera_z_diagnostic_enable", True)) and geo_obs:
            _pred_gt_z_errs: List[float] = []
            _pred_agl_vals: List[float] = []
            _agl_ref_errs: List[float] = []
            _dem_plus_agl_z_errs: List[float] = []
            _ground_z_vals: List[float] = []
            _ground_sources: Dict[str, int] = {}
            _world_np_for_z_diag = world_enu_final.detach().cpu().numpy()
            for _fi_zd, _corr_zd in geo_obs.items():
                if _corr_zd is None or int(_fi_zd) < 0 or int(_fi_zd) >= len(_world_np_for_z_diag):
                    continue
                _pred_cam = np.asarray(_world_np_for_z_diag[int(_fi_zd)], dtype=np.float64).reshape(-1)
                if _pred_cam.size < 3 or not np.all(np.isfinite(_pred_cam[:3])):
                    continue
                _pred_z = float(_pred_cam[2])
                _ground_z = _dem_ground_z_at_enu_xy(
                    _pred_cam[:2], segment_input.project_fn, geo_elev, dom_transform)
                _ground_source = "dem_at_pred_xy"
                if _ground_z is None and (
                    _corr_zd.get("enu_positions_road") is not None
                    or _corr_zd.get("enu_positions") is not None
                ):
                    _ground_z = _geo_ground_z_mean(_corr_zd, prefer_road=True)
                    _ground_source = "matched_sample_mean"
                _gt_z = _corr_zd.get("camera_z_gt_diag")
                if _gt_z is None:
                    _gt_z = _corr_zd.get("camera_z_gt")
                if _gt_z is not None and np.isfinite(float(_gt_z)):
                    _pred_gt_z_errs.append(_pred_z - float(_gt_z))
                if _ground_z is not None and np.isfinite(float(_ground_z)):
                    _ground_z = float(_ground_z)
                    _ground_z_vals.append(_ground_z)
                    _pred_agl = _pred_z - _ground_z
                    _pred_agl_vals.append(_pred_agl)
                    _ground_sources[_ground_source] = _ground_sources.get(_ground_source, 0) + 1
                    _agl_ref = _corr_zd.get("gt_agl_diag")
                    if _agl_ref is None:
                        _agl_ref = _corr_zd.get("gt_agl")
                    if _agl_ref is not None and np.isfinite(float(_agl_ref)):
                        _agl_err = _pred_agl - float(_agl_ref)
                        _agl_ref_errs.append(_agl_err)
                        _dem_plus_agl_z_errs.append(_pred_z - (_ground_z + float(_agl_ref)))
            _cam_z_diag.update({
                "n": int(max(len(_pred_gt_z_errs), len(_pred_agl_vals))),
                "pred_vs_gt_z_err_m": _fmt_signed_summary(_pred_gt_z_errs),
                "pred_agl_m": _fmt_unsigned_summary(_pred_agl_vals),
                "pred_agl_minus_ref_m": _fmt_signed_summary(_agl_ref_errs),
                "pred_z_minus_dem_plus_ref_agl_m": _fmt_signed_summary(_dem_plus_agl_z_errs),
                "ground_z_m": _fmt_unsigned_summary(_ground_z_vals),
                "ground_sources": dict(_ground_sources),
            })
            print(
                f"[GeoV3][Z-cam-diag] seg {segment_input.segment_id}: "
                f"pred_gt_z_err({_cam_z_diag['pred_vs_gt_z_err_m']}) "
                f"pred_agl({_cam_z_diag['pred_agl_m']}) "
                f"agl_err({_cam_z_diag['pred_agl_minus_ref_m']}) "
                f"dem+agl_z_err({_cam_z_diag['pred_z_minus_dem_plus_ref_agl_m']}) "
                f"ground_z({_cam_z_diag['ground_z_m']}) "
                f"sources={_ground_sources}"
            )
        transform_diag["camera_z_diagnostic"] = _cam_z_diag

        # The DEM-Z stage includes road-point alignment plus camera DEM/AGL
        # correction and raw-PnP-Z diagnostics.  Keep it separate from the
        # inexpensive output/serialization work below.
        _record_phase("dem_z_s", _dem_z_t0)
        _post_decode_t0 = time.perf_counter()

        # Convert sampled pts3d from model space → ENU for footprint vis
        _pts3d_model_all = torch.cat(_pts3d_model_list, dim=0) if _pts3d_model_list else None
        pts3d_enu_sampled = (
            transform.model_to_world(_pts3d_model_all).detach().cpu().numpy()
            if _pts3d_model_all is not None else None
        )

        # Convert full per-frame pts3d (M_k) → ENU. Stays on CPU as float32.
        # Shape: (N_seg, H, W, 3). Frames left as a python list to avoid an
        # extra GB-scale stack if H/W vary; eval scripts can stack later.
        # Also save the M_k (model-space) version so post-PGO can remap with
        # refined T_k (saves ~76 MB / 32-frame seg, kept on CPU as float32).
        pts3d_enu_full: Optional[np.ndarray] = None
        pts3d_conf_full: Optional[np.ndarray] = None
        pts3d_frame_indices: Optional[np.ndarray] = None
        pts3d_model_full: Optional[np.ndarray] = None
        if _export_full_pts3d and len(_pts3d_model_full_list) > 0:
            _full_dev = transform.t.device
            _enu_chunks: List[np.ndarray] = []
            _model_chunks: List[np.ndarray] = []
            for _pf_cpu in _pts3d_model_full_list:
                _H, _W, _ = _pf_cpu.shape
                _flat_cpu_np = _pf_cpu.numpy().astype(np.float32)
                _model_chunks.append(_flat_cpu_np.reshape(_H, _W, 3))
                _flat = _pf_cpu.reshape(-1, 3).to(_full_dev)
                _enu_flat = transform.model_to_world(_flat).detach().cpu().float().numpy()
                _enu_chunks.append(_enu_flat.reshape(_H, _W, 3))
            pts3d_enu_full = np.stack(_enu_chunks, axis=0).astype(np.float32)
            pts3d_model_full = np.stack(_model_chunks, axis=0).astype(np.float32)
            pts3d_conf_full = np.stack(
                [c.numpy().astype(np.float32) for c in _pts3d_conf_full_list], axis=0
            )
            pts3d_frame_indices = np.arange(
                int(segment_input.start_frame),
                int(segment_input.start_frame) + int(num_frames),
                dtype=np.int64,
            )

        if (
            _dom_points_mode
            and _dom_point_ground_ray_stage_enabled(config, "pgo")
            and ground_ray_cache
            and len(post_ttt_ray_dirs_model) == int(model_final.shape[0])
        ):
            _gr_centers: List[torch.Tensor] = []
            _gr_dirs: List[torch.Tensor] = []
            _gr_targets: List[torch.Tensor] = []
            for _fi_gr, _entry_gr in sorted(ground_ray_cache.items()):
                _fi_i = int(_fi_gr)
                if _fi_i < 0 or _fi_i >= int(model_final.shape[0]):
                    continue
                _target_t = _entry_gr.get("target_enu") if isinstance(_entry_gr, dict) else None
                if _target_t is None:
                    continue
                _dir_t = post_ttt_ray_dirs_model[_fi_i].to(device=model_final.device, dtype=model_final.dtype)
                _target_t = _target_t.to(device=model_final.device, dtype=model_final.dtype)
                if not (torch.isfinite(_dir_t).all() and torch.isfinite(_target_t).all()):
                    continue
                _gr_centers.append(model_final[_fi_i].detach().float())
                _gr_dirs.append(F.normalize(_dir_t.detach().float(), dim=0, eps=1e-8))
                _gr_targets.append(_target_t.detach().float())
            if _gr_centers:
                _gr_centers_t = torch.stack(_gr_centers, dim=0).to(device=model_final.device, dtype=model_final.dtype)
                _gr_dirs_t = torch.stack(_gr_dirs, dim=0).to(device=model_final.device, dtype=model_final.dtype)
                _gr_targets_t = torch.stack(_gr_targets, dim=0).to(device=model_final.device, dtype=model_final.dtype)
                _gr_centers_enu = transform.model_to_world(_gr_centers_t)
                _gr_dirs_enu = _gr_dirs_t @ transform.R.to(device=model_final.device, dtype=model_final.dtype).T
                _gr_pred_xy = []
                for _i_gr in range(int(_gr_centers_t.shape[0])):
                    _gr_pred_xy.append(_ground_ray_intersection_xy_torch(
                        _gr_centers_enu[_i_gr],
                        _gr_dirs_enu[_i_gr],
                        _gr_targets_t[_i_gr, 2],
                        float(getattr(config, "dom_point_ground_ray_min_abs_dir_z", 0.05)),
                    ))
                _gr_pred_xy_t = torch.stack(_gr_pred_xy, dim=0)
                _gr_xy_err = torch.linalg.norm(_gr_pred_xy_t - _gr_targets_t[:, :2], dim=1)
                ground_ray_centers_model_pgo = _gr_centers_t.detach().cpu().float().numpy().astype(np.float32)
                ground_ray_dirs_model_pgo = _gr_dirs_t.detach().cpu().float().numpy().astype(np.float32)
                ground_ray_targets_enu_pgo = _gr_targets_t.detach().cpu().float().numpy().astype(np.float32)
                ground_ray_edge_stats.update({
                    "pgo_edges": int(ground_ray_centers_model_pgo.shape[0]),
                    "xy_median_m": float(_gr_xy_err.median().detach().cpu()),
                    "xy_mean_m": float(_gr_xy_err.mean().detach().cpu()),
                })
                print(
                    f"[GeoV3][GroundRay-PGO] seg {segment_input.segment_id}: "
                    f"rays={ground_ray_edge_stats['pgo_edges']} "
                    f"xy_med={ground_ray_edge_stats['xy_median_m']:.2f}m "
                    f"xy_mean={ground_ray_edge_stats['xy_mean_m']:.2f}m"
                )
            else:
                ground_ray_edge_stats.update({"pgo_edges": 0, "reason": "no_valid_post_rays"})
        canonical_diag = {"canonicalized": True, **transform_diag}
        propagation_diag = {"propagated": True}

        # Capture post-TTT overlap positions in M_0 space for next segment's Umeyama.
        # seg 0: M_k = M_0, so post-TTT positions are already in M_0.
        # seg k>0: M_k → ENU (via T_k) → M_0 (via T_0⁻¹).
        post_ttt_overlap_m0: Optional[np.ndarray] = None
        _t0 = transform_diag.get("t0_transform")
        if overlap_len > 0 and len(model_final) >= overlap_len:
            overlap_mk = model_final[-overlap_len:]  # last overlap_len frames in M_k
            if _t0 is not None and _t0 is not transform:
                # seg k>0: M_k → ENU → M_0
                overlap_enu = transform.model_to_world(overlap_mk)
                overlap_m0 = _t0.world_to_model(overlap_enu)
                post_ttt_overlap_m0 = overlap_m0.cpu().float().numpy()
            else:
                # seg 0: M_k IS M_0 (T_0 = transform itself)
                post_ttt_overlap_m0 = overlap_mk.cpu().float().numpy()
            print(f"[GeoV3][PostTTT-Overlap] seg {segment_input.segment_id}: "
                  f"captured {overlap_len} post-TTT positions in M_0 for next segment")

    _record_phase("post_decode_s", _post_decode_t0)
    _phase_timing_s["segment_total_s"] = float(max(0.0, time.perf_counter() - _segment_t0))

    geo_cache = GeoObservationCache(correspondences=geo_obs, metadata={"geo_consist_stride": geo_consist_stride, "geo_consist_max_corr": geo_consist_max_corr})
    submap_state = SubmapState(segment_id=segment_input.segment_id, model_space=model_final.detach().cpu().numpy(), transform=transform, anchor_world_xyz=world_enu_final[0].detach().cpu().numpy() if len(world_enu_final) else None, source_summary=segment_input.prev_summary)
    runtime_state.ttt_outputs["segment_input"] = {"segment_id": segment_input.segment_id, "start_frame": segment_input.start_frame, "end_frame": segment_input.end_frame, "num_frames": num_frames}
    runtime_state.loss_history.append(loss_diag["ttt_losses"]["total_loss"])
    runtime_state.ttt_outputs.update({"loss_diag": loss_diag, "canonical_diag": canonical_diag, "propagation_diag": propagation_diag, "world_xyz_init": world_xyz_init.detach().cpu().numpy(), "world_xyz_final": world_enu_final.detach().cpu().numpy(), "model_xyz_init": model_final.detach().cpu().numpy(), "model_xyz_final": model_final.detach().cpu().numpy(), "delta_model": np.zeros_like(model_final.detach().cpu().numpy())})
    # boundary_end_enu: ENU of the frame that becomes the NEXT segment's first frame.
    # With overlap, that is num_frames - overlap_len (not the last frame).
    _boundary_idx = max(0, num_frames - overlap_len) if overlap_len > 0 else num_frames - 1

    # Per-frame R_c2w in model space (3x3). Multiply by T_k.R externally to get
    # R_c2w_enu (PGO-aware). Store for dom_points so the next overlap can inherit full extrinsics.
    per_frame_R_c2w_model_np: Optional[np.ndarray] = None
    if _export_full_pts3d or dom_points_mode:
        try:
            from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
            _R_list: List[np.ndarray] = []
            for _pose_fi in per_frame_pose:
                _extri, _ = pose_encoding_to_extri_intri(
                    _pose_fi, image_size_hw=(1, 1), build_intrinsics=False)
                _R_w2c = _extri[..., :3, :3]
                if _R_w2c.dim() > 2:
                    _R_w2c = _R_w2c.reshape(-1, 3, 3)[0]
                _R_c2w_m = _R_w2c.detach().cpu().float().numpy().T  # (3, 3)
                _R_list.append(_R_c2w_m)
            if _R_list:
                per_frame_R_c2w_model_np = np.stack(_R_list, axis=0).astype(np.float32)
        except Exception as _e:
            print(f"[GeoV3][per-frame R] seg {segment_input.segment_id}: failed ({_e})")

    # End-of-segment predicted pitch (Plan A' gate input for next segment).
    _end_pred_pitch_deg: Optional[float] = None
    try:
        _end_pred_pitch_deg = _pose_to_pitch_deg(per_frame_pose[-1])
    except Exception:
        _end_pred_pitch_deg = None

    diagnostics: Dict[str, Any] = {"segment_id": segment_input.segment_id, "start_frame": segment_input.start_frame, "end_frame": segment_input.end_frame, "num_frames": num_frames, "map_observation_stats": map_observation_stats, "bootstrap_quality": bootstrap_quality_diag, **propagation_diag, **canonical_diag, **loss_diag, "last_pose": per_frame_pose[-1].detach().cpu().numpy(), "boundary_start_pose": per_frame_pose[0].detach().cpu().numpy(), "boundary_end_pose": per_frame_pose[-1].detach().cpu().numpy(), "boundary_start_enu": world_enu_final[0].detach().cpu().numpy(), "boundary_end_enu": world_enu_final[_boundary_idx].detach().cpu().numpy(), "post_ttt_overlap_m0": post_ttt_overlap_m0, "pts3d_enu_sampled": pts3d_enu_sampled, "pts3d_enu_full": pts3d_enu_full, "pts3d_model_full": pts3d_model_full, "pts3d_conf_full": pts3d_conf_full, "pts3d_frame_indices": pts3d_frame_indices, "visual_ground_points_model_by_frame": _visual_ground_points_model_by_frame, "visual_ground_dem_z_by_frame": _visual_ground_dem_z_by_frame, "visual_ground_source_by_frame": _visual_ground_source_by_frame, "per_frame_R_c2w_model": per_frame_R_c2w_model_np, "transform_R": transform.R.detach().cpu().float().numpy().astype(np.float32), "overlap_dom_targets_enu": overlap_dom_targets_enu, "road_pts_model": road_pts_model_pgo, "road_dem_z_enu": road_dem_z_pgo, "dem_z_sample_source": dem_z_sample_source, "dem_z_direct_stats": _dem_z_stats, "camera_z_pts_model": camera_z_pts_model_pgo, "camera_z_target_enu": camera_z_target_pgo, "camera_z_sample_source": camera_z_sample_source, "camera_z_correct_stats": _cam_z_stats, "dom_pts_model": dom_pts_model_pgo, "dom_pts_enu": dom_pts_enu_pgo, "dom_point_edge_stats": dom_point_edge_stats, "dom_point_camera_targets_enu": dom_point_camera_targets_enu, "dom_point_camera_stats": dom_point_camera_stats, "ground_ray_centers_model": ground_ray_centers_model_pgo, "ground_ray_dirs_model": ground_ray_dirs_model_pgo, "ground_ray_targets_enu": ground_ray_targets_enu_pgo, "ground_ray_edge_stats": ground_ray_edge_stats, "point_ttt_stats": point_ttt_stats, "point_srt_stats": point_srt_stats, "phase_timing_s": dict(_phase_timing_s), "end_pred_pitch_deg": _end_pred_pitch_deg}
    return SegmentOutput(trajectory_global=world_enu_final.detach().cpu().numpy(), trajectory_local=model_final.detach().cpu().numpy(), diagnostics=diagnostics, transform=transform, submap_state=submap_state, geo_cache=geo_cache)
