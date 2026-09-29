"""Plan A：段级位姿图优化（PGO，segment-level pose graph optimization）。

这是一个后处理步骤，用以下约束精修各段的 Sim3 变换 ``T_k``：

- **segment-0 首帧的一元先验**（硬约束）：``T_0(p_0^{M_0}) = origin_enu``。
- **仅重叠帧的 DOM 一元边**：对每个重叠帧 ``f``，
  ``T_k(p_{f}^{M_k}) ≈ dom_target_f^{ENU}``。
- **重叠一致性二元边**：对相邻两段（``k`` 与 ``k+1``）共享的帧 ``f``，
  ``T_k(p_f^{M_k}) ≈ T_{k+1}(p_f^{M_{k+1}})``。

每个 ``T_k`` 用 7 自由度的 Sim3 参数化：
``(omega_x, omega_y, omega_z, t_x, t_y, t_z, log_s)``，旋转用 Rodrigues 公式。

本模块**不触碰**神经网络与 TTT 流程：它消费一个已完成的 ``GlobalChainState``，
返回一组新的 ``LocalModelTransform``（每段一个），用于重新拼装全局轨迹。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .canonicalization import LocalModelTransform
from .state import GlobalChainState, SegmentSummary


# -----------------------------------------------------------------------------
# Sim3 numpy helpers
# -----------------------------------------------------------------------------

def _rodrigues(omega: np.ndarray) -> np.ndarray:
    """Rodrigues 公式：so(3) 的 3 维向量 → 3×3 旋转矩阵。

    omega 的方向是转轴、模长是转角 theta。theta 近 0 时直接返回单位阵以避免除零。
    """
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = omega / theta          # 单位转轴
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]], dtype=np.float64)   # k 的反对称矩阵 [k]_×
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) 对数映射：3×3 旋转矩阵 → so(3) 的 3 维向量（_rodrigues 的逆）。

    由迹得到转角 theta=arccos((tr(R)-1)/2)，由反对称部分得到转轴方向。
    theta 近 0 时返回零向量。结果满足 ||.|| <= pi（真正的测地距离）。
    """
    tr = float(np.trace(R))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)   # 数值裁剪，防 arccos 越界
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
                 dtype=np.float64)
    return w * (theta / (2.0 * np.sin(theta)))


def _pack_sim3(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """(R, t, s) → 7 维向量 [omega(3), t(3), log_s]。尺度取对数以便无约束优化。"""
    return np.concatenate([_log_so3(R), np.asarray(t, dtype=np.float64).reshape(3),
                           [np.log(max(float(s), 1e-8))]]).astype(np.float64)


def _unpack_sim3(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """7 维向量 → (R, t, s)（_pack_sim3 的逆）。"""
    R = _rodrigues(v[0:3])
    t = v[3:6].astype(np.float64)
    s = float(np.exp(v[6]))     # log_s -> s
    return R, t, s


def _apply_sim3(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """把 7 维 Sim3 作用到 (N, 3) 点上：ENU = s * R @ p + t（行向量形式 s * p @ R.T + t）。"""
    R, t, s = _unpack_sim3(v)
    return s * (pts @ R.T) + t[None, :]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class PGOConfig:
    """全局 PGO 各类边 / 先验的权重与上限配置。权重为 0 通常表示关闭该类边。"""
    w_first_frame: float = 1e3      # segment-0 首帧硬锚点权重
    w_overlap_dom_xy: float = 5.0   # 重叠帧 DOM XY 一元边权重
    w_overlap_dom_z: float = 1.0    # 重叠帧 DOM Z 一元边权重（更弱；DEM Z 噪声大）
    w_overlap_consist: float = 10.0 # 相邻段重叠一致性二元边权重
    w_z_dem: float = 5.0            # 逐段路面点 DEM Z 一元边权重（仅 Z）
    w_camera_agl: float = 5.0       # 相机中心 Z 朝 DEM(相机 XY)+AGL 的权重
    w_point_xy: float = 0.0         # 直接 DOM/DEM 点 XY 一元边权重（0 关闭）
    w_point_z: float = 0.0          # 直接 DOM/DEM 点 Z 一元边权重（0 关闭）
    w_ground_ray_xy: float = 0.0    # 画面中心地面射线 XY 一元边权重（0 关闭）
    ground_ray_min_abs_dir_z: float = 0.05   # 射线方向 z 分量最小绝对值（防除零）
    w_scale: float = 50.0           # 相邻段 log(s) 一致性权重（0 关闭）
    fix_rotation: bool = False      # 固定各段地图 bootstrap 的 R，仅优化其余变量
    fix_scale: bool = False         # 固定各段独立地图尺度；同时移除跨段 scale edge
    fix_map_registered_xy: bool = False  # 固定成功绝对地图注册段的 tx/ty；tz 仍由 DEM/overlap 统一优化
    max_iter: int = 200             # scipy least_squares 的 max_nfev 系数
    verbose: bool = True            # 是否打印求解日志


# -----------------------------------------------------------------------------
# Edge extraction
# -----------------------------------------------------------------------------

@dataclass
class _DomEdge:
    """重叠帧 DOM 一元边：T_seg(该帧 model 位置) 应落在 DOM+DEM 给出的 ENU 目标上。"""
    seg_idx: int       # 在 chain.summaries 中的段下标
    local_fi: int      # 该段内的局部帧下标
    target_enu: np.ndarray  # (3,) 来自 DOM+DEM 的 ENU XYZ 目标


@dataclass
class _ConsistEdge:
    """重叠一致性二元边：同一共享帧在相邻两段下变换出的 ENU 位置应一致。"""
    seg_a: int         # 较早的段下标
    seg_b: int         # 较晚的段下标
    local_fi_a: int    # 共享帧在 seg_a 中的局部下标
    local_fi_b: int    # 共享帧在 seg_b 中的局部下标


@dataclass
class _ScaleEdge:
    """相邻段 log-scale 一致性边。

    残差为 ``(log_s_b - log_s_a) * w_scale``。这是一条软先验：相邻子图的
    model->ENU 尺度应当一致（同一相机、同一场景）。
    """
    seg_a: int
    seg_b: int


@dataclass
class _DemZEdge:
    """逐段路面点 DEM-Z 一元边（仅约束 Z）。

    把 T_k(pts_model)[:, 2] 拉向 DEM 地面 Z（ENU 下）。每个路面点贡献一个标量
    残差。由于相邻段共享同一 DEM，跨段聚合后会产生软的段间 Z 一致性。
    """
    seg_idx: int
    pts_model: np.ndarray   # (K, 3) M_k 空间下的路面点
    dem_z_enu: np.ndarray   # (K,)   对应的 DEM 地面 Z（ENU）


@dataclass
class _CameraZEdge:
    """逐段相机中心 DEM+AGL 一元边。

    把相机中心拉向"跟随 DEM 的高度目标"，通常为 DEM(相机 xy) + 锚点 AGL。
    与路面 DEM-Z 不同，它直接约束相机高度，且能覆盖没有路面采样的帧。
    """
    seg_idx: int
    pts_model: np.ndarray       # (K, 3) M_k 下的相机中心
    target_z_enu: np.ndarray    # (K,) ENU 下的相机 Z 目标


@dataclass
class _PointEdge:
    """逐段直接 DOM/DEM 点一元边。

    把匹配到的预测 model-space 点拉向其对应的 DOM/DEM ENU XYZ 目标。与 PnP/H
    类边不同，它把约束保持在**点级别**，而不是塌缩成单个相机位姿。
    """
    seg_idx: int
    pts_model: np.ndarray       # (K, 3) 在 query 像素处采样的 M_k 点
    target_enu: np.ndarray      # (K, 3) DOM/DEM ENU 目标


@dataclass
class _GroundRayEdge:
    """逐段画面中心地面射线一元边。

    DOM 画面中心是一个地面点。对每个相机中心 c 与光轴方向 d，把射线
    c + lambda·d 与平面 z=target_enu.z 求交，并约束交点 XY 等于 target_enu.xy。
    """
    seg_idx: int
    centers_model: np.ndarray   # (K, 3) M_k 下的相机中心
    dirs_model: np.ndarray      # (K, 3) M_k 下的光轴方向
    target_enu: np.ndarray      # (K, 3) DOM 画面中心地面目标


def _signed_safe_denominator_np(values: np.ndarray, min_abs: float) -> np.ndarray:
    """把分母的绝对值钳到 >= min_abs（保留原符号），避免后续除法被接近 0 的值放大。"""
    min_abs = max(float(min_abs), 1e-8)
    arr = np.asarray(values, dtype=np.float64)
    sign = np.where(arr >= 0.0, 1.0, -1.0)
    return np.where(np.abs(arr) < min_abs, sign * min_abs, arr)


def _ground_ray_xy_residual_np(
    sim3: np.ndarray,
    centers_model: np.ndarray,
    dirs_model: np.ndarray,
    target_enu: np.ndarray,
    min_abs_dir_z: float,
) -> np.ndarray:
    """计算地面射线 XY 残差：相机射线与平面 z=target_z 的交点 XY 减去目标 XY。

    步骤：用 sim3 把相机中心与光轴方向变换到 ENU；以 lambda 沿射线推进到目标
    高度处，求交点 XY；返回 (交点 XY - 目标 XY)。denom 做了带符号保护以防除零。
    """
    R, t, s = _unpack_sim3(sim3)
    centers = s * (centers_model @ R.T) + t[None, :]   # 相机中心 -> ENU
    dirs = dirs_model @ R.T                             # 光轴方向 -> ENU
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-8)               # 单位化
    denom = _signed_safe_denominator_np(dirs[:, 2], min_abs_dir_z)  # 防除零的 dir_z
    lam = (target_enu[:, 2] - centers[:, 2]) / denom    # 推进到目标高度所需的 lambda
    pred_xy = centers[:, :2] + lam[:, None] * dirs[:, :2]  # 交点 XY
    return pred_xy - target_enu[:, :2]


def _summarize_point_residuals(params_flat: np.ndarray,
                               point_edges: List[_PointEdge]) -> Dict[str, float]:
    """仅用于诊断：统计点边在当前参数下的**未加权**残差（RMSE / 中位数 / p90 等）。"""
    if not point_edges:
        return {"n_points": 0}
    params = np.asarray(params_flat, dtype=np.float64).reshape(-1, 7)
    xy_parts: List[np.ndarray] = []
    z_parts: List[np.ndarray] = []
    for edge in point_edges:
        if edge.seg_idx < 0 or edge.seg_idx >= len(params):
            continue
        pts = np.asarray(edge.pts_model, dtype=np.float64)
        tgt = np.asarray(edge.target_enu, dtype=np.float64)
        if pts.ndim != 2 or tgt.ndim != 2 or pts.shape != tgt.shape or pts.shape[1] != 3:
            continue
        enu = _apply_sim3(params[edge.seg_idx], pts)
        diff = enu - tgt
        finite = np.isfinite(diff).all(axis=1)
        if not np.any(finite):
            continue
        diff = diff[finite]
        xy_parts.append(np.linalg.norm(diff[:, :2], axis=1))
        z_parts.append(diff[:, 2])
    if not xy_parts:
        return {"n_points": 0}
    xy = np.concatenate(xy_parts, axis=0)
    z = np.concatenate(z_parts, axis=0)
    return {
        "n_points": int(xy.shape[0]),
        "xy_rmse_m": float(np.sqrt(np.mean(xy ** 2))),
        "xy_mean_m": float(np.mean(xy)),
        "xy_median_m": float(np.median(xy)),
        "xy_p90_m": float(np.percentile(xy, 90.0)),
        "xy_max_m": float(np.max(xy)),
        "z_abs_mean_m": float(np.mean(np.abs(z))),
        "z_abs_median_m": float(np.median(np.abs(z))),
    }


def _summarize_ground_ray_residuals(
    params_flat: np.ndarray,
    ground_ray_edges: List[_GroundRayEdge],
    min_abs_dir_z: float,
) -> Dict[str, float]:
    """仅用于诊断：统计地面射线 XY 残差的**未加权**分布。"""
    if not ground_ray_edges:
        return {"n_rays": 0}
    params = np.asarray(params_flat, dtype=np.float64).reshape(-1, 7)
    parts: List[np.ndarray] = []
    for edge in ground_ray_edges:
        if edge.seg_idx < 0 or edge.seg_idx >= len(params):
            continue
        diff = _ground_ray_xy_residual_np(
            params[edge.seg_idx],
            edge.centers_model,
            edge.dirs_model,
            edge.target_enu,
            min_abs_dir_z,
        )
        finite = np.isfinite(diff).all(axis=1)
        if np.any(finite):
            parts.append(np.linalg.norm(diff[finite], axis=1))
    if not parts:
        return {"n_rays": 0}
    xy = np.concatenate(parts, axis=0)
    return {
        "n_rays": int(xy.shape[0]),
        "xy_rmse_m": float(np.sqrt(np.mean(xy ** 2))),
        "xy_mean_m": float(np.mean(xy)),
        "xy_median_m": float(np.median(xy)),
        "xy_p90_m": float(np.percentile(xy, 90.0)),
        "xy_max_m": float(np.max(xy)),
    }


def _aggregate_point_edge_source_stats(chain: GlobalChainState) -> Dict[str, int]:
    """汇总各段由 GeoV3 写入的 point-edge 来源计数器（用于诊断点边构成）。"""
    out: Dict[str, int] = {}
    for summary in chain.summaries:
        meta = summary.metadata if isinstance(summary.metadata, dict) else {}
        stats = meta.get("dom_point_edge_stats")
        if not isinstance(stats, dict):
            continue
        for key, value in stats.items():
            if isinstance(value, (bool, str)):
                continue
            try:
                out[key] = out.get(key, 0) + int(value)
            except Exception:
                pass
    return out


def _extract_edges(chain: GlobalChainState) -> Tuple[List[_DomEdge], List[_ConsistEdge], List[_DemZEdge], List[_CameraZEdge], List[_PointEdge], List[_GroundRayEdge], List[_ScaleEdge]]:
    """从已完成的 GlobalChainState 中抽取 PGO 所需的全部边。

    返回七类边的列表（按返回顺序）：
        DOM 一元边、重叠一致性二元边、DEM-Z 边、相机-Z 边、
        直接点边、地面射线边、相邻段尺度边。
    其中一致性边通过"全局帧 -> [(段, 局部帧)]"的映射，找出被多段共享的帧来构造。
    """
    dom_edges: List[_DomEdge] = []
    consist_edges: List[_ConsistEdge] = []
    dem_z_edges: List[_DemZEdge] = []
    camera_z_edges: List[_CameraZEdge] = []
    point_edges: List[_PointEdge] = []
    ground_ray_edges: List[_GroundRayEdge] = []
    scale_edges: List[_ScaleEdge] = []

    summaries = chain.summaries
    # Map: global_fi → list of (seg_idx, local_fi) for binary consistency
    frame_to_segs: Dict[int, List[Tuple[int, int]]] = {}

    for k, summary in enumerate(summaries):
        traj_local = summary.trajectory_local
        if traj_local is None or len(traj_local) == 0:
            continue
        start = int(summary.start_frame)

        # Register every frame for potential binary consistency
        for local_idx in range(len(traj_local)):
            gf = start + local_idx
            frame_to_segs.setdefault(gf, []).append((k, local_idx))

        # Unary DOM edges (only for overlap frames that have DOM targets)
        overlap_targets = summary.overlap_dom_targets_enu
        if overlap_targets:
            for global_fi, enu_xyz in overlap_targets.items():
                local_fi = int(global_fi) - start
                if local_fi < 0 or local_fi >= len(traj_local):
                    continue
                dom_edges.append(_DomEdge(
                    seg_idx=k,
                    local_fi=local_fi,
                    target_enu=np.asarray(enu_xyz, dtype=np.float64),
                ))

        meta = getattr(summary, "metadata", None) or {}
        dom_point_camera_targets = meta.get("dom_point_camera_targets_enu")
        if dom_point_camera_targets:
            for local_fi_raw, target_enu in dict(dom_point_camera_targets).items():
                try:
                    local_fi = int(local_fi_raw)
                except (TypeError, ValueError):
                    continue
                if local_fi < 0 or local_fi >= len(traj_local):
                    continue
                target_arr = np.asarray(target_enu, dtype=np.float64).reshape(-1)
                if target_arr.size < 3 or not np.isfinite(target_arr[:3]).all():
                    continue
                dom_edges.append(_DomEdge(
                    seg_idx=k,
                    local_fi=local_fi,
                    target_enu=target_arr[:3].copy(),
                ))

        # Per-segment DEM-Z unary edges (from road pts subsampled at inference time).
        # Stored under summary.metadata by GeoV3Runner._build_summary catch-all.
        rpm = meta.get("road_pts_model")
        rdz = meta.get("road_dem_z_enu")
        if rpm is not None and rdz is not None:
            rpm_np = np.asarray(rpm, dtype=np.float64)
            rdz_np = np.asarray(rdz, dtype=np.float64).reshape(-1)
            if (rpm_np.ndim == 2 and rpm_np.shape[1] == 3
                    and rdz_np.ndim == 1 and len(rdz_np) == rpm_np.shape[0]
                    and len(rdz_np) > 0):
                dem_z_edges.append(_DemZEdge(
                    seg_idx=k,
                    pts_model=rpm_np,
                    dem_z_enu=rdz_np,
                ))

        czm = meta.get("camera_z_pts_model")
        czt = meta.get("camera_z_target_enu")
        if czm is not None and czt is not None:
            czm_np = np.asarray(czm, dtype=np.float64)
            czt_np = np.asarray(czt, dtype=np.float64).reshape(-1)
            if (czm_np.ndim == 2 and czm_np.shape[1] == 3
                    and czt_np.ndim == 1 and len(czt_np) == czm_np.shape[0]
                    and len(czt_np) > 0):
                camera_z_edges.append(_CameraZEdge(
                    seg_idx=k,
                    pts_model=czm_np,
                    target_z_enu=czt_np,
                ))

        dpm = meta.get("dom_pts_model")
        dpe = meta.get("dom_pts_enu")
        if dpm is not None and dpe is not None:
            dpm_np = np.asarray(dpm, dtype=np.float64)
            dpe_np = np.asarray(dpe, dtype=np.float64)
            if (dpm_np.ndim == 2 and dpm_np.shape[1] == 3
                    and dpe_np.ndim == 2 and dpe_np.shape[1] == 3
                    and dpm_np.shape[0] == dpe_np.shape[0]
                    and dpm_np.shape[0] > 0):
                point_edges.append(_PointEdge(
                    seg_idx=k,
                    pts_model=dpm_np,
                    target_enu=dpe_np,
                ))

        grc = meta.get("ground_ray_centers_model")
        grd = meta.get("ground_ray_dirs_model")
        grt = meta.get("ground_ray_targets_enu")
        if grc is not None and grd is not None and grt is not None:
            grc_np = np.asarray(grc, dtype=np.float64)
            grd_np = np.asarray(grd, dtype=np.float64)
            grt_np = np.asarray(grt, dtype=np.float64)
            if (grc_np.ndim == 2 and grc_np.shape[1] == 3
                    and grd_np.ndim == 2 and grd_np.shape[1] == 3
                    and grt_np.ndim == 2 and grt_np.shape[1] == 3
                    and grc_np.shape[0] == grd_np.shape[0] == grt_np.shape[0]
                    and grc_np.shape[0] > 0):
                finite = (
                    np.isfinite(grc_np).all(axis=1)
                    & np.isfinite(grd_np).all(axis=1)
                    & np.isfinite(grt_np).all(axis=1)
                    & (np.linalg.norm(grd_np, axis=1) > 1e-8)
                )
                if np.any(finite):
                    ground_ray_edges.append(_GroundRayEdge(
                        seg_idx=k,
                        centers_model=grc_np[finite],
                        dirs_model=grd_np[finite],
                        target_enu=grt_np[finite],
                    ))

    # Binary consistency: any frame appearing in 2+ segments
    for gf, occurrences in frame_to_segs.items():
        if len(occurrences) < 2:
            continue
        occurrences.sort(key=lambda x: x[0])
        for i in range(len(occurrences) - 1):
            a_seg, a_fi = occurrences[i]
            b_seg, b_fi = occurrences[i + 1]
            consist_edges.append(_ConsistEdge(
                seg_a=a_seg, seg_b=b_seg,
                local_fi_a=a_fi, local_fi_b=b_fi,
            ))

    # Adjacent-segment scale consistency (one edge per neighbouring pair).
    for k in range(len(summaries) - 1):
        if summaries[k].trajectory_local is None or summaries[k + 1].trajectory_local is None:
            continue
        scale_edges.append(_ScaleEdge(seg_a=k, seg_b=k + 1))

    return dom_edges, consist_edges, dem_z_edges, camera_z_edges, point_edges, ground_ray_edges, scale_edges


# -----------------------------------------------------------------------------
# Core solver
# -----------------------------------------------------------------------------

def _build_jac_sparsity(
    N: int,
    model_pts: List[np.ndarray],
    dom_edges: List[_DomEdge],
    consist_edges: List[_ConsistEdge],
    dem_z_edges: List[_DemZEdge],
    camera_z_edges: List[_CameraZEdge],
    point_edges: List[_PointEdge],
    ground_ray_edges: List[_GroundRayEdge],
    scale_edges: List[_ScaleEdge],
):
    """构建 Jacobian 的稀疏结构 (M, 7N)。

    行布局**必须**与 ``_residual_fn`` 完全一致，否则稀疏模式会错位：
      [seg-0 锚点 (3，若存在)] + [DOM 一元 (各 3)] +
            [一致性二元 (各 3)] + [DEM-Z 一元 (每个路面点 1)] +
            [相机-Z 一元 (每个采样相机 1)] +
            [点一元 (每个匹配点 3)] +
                        [地面射线一元 (每条射线 2)] +
      [尺度二元 (各 1)]

    其中：每个 Sim3 占 7 个参数列，第 k 段的列偏移为 k*7；尺度边只依赖 log_s 列
    （列偏移 +6）。M==0 或 N==0 时返回 None。
    """
    has_anchor = (
        N > 0
        and model_pts[0] is not None
        and len(model_pts[0]) > 0
    )
    n_dem_rows = sum(int(len(e.dem_z_enu)) for e in dem_z_edges)
    n_camera_z_rows = sum(int(len(e.target_z_enu)) for e in camera_z_edges)
    n_point_rows = sum(3 * int(e.pts_model.shape[0]) for e in point_edges)
    n_ground_ray_rows = sum(2 * int(e.centers_model.shape[0]) for e in ground_ray_edges)
    M = ((3 if has_anchor else 0)
         + 3 * len(dom_edges)
         + 3 * len(consist_edges)
         + n_dem_rows
         + n_camera_z_rows
         + n_point_rows
            + n_ground_ray_rows
         + len(scale_edges))
    if M == 0 or N == 0:
        return None
    sp = lil_matrix((M, 7 * N), dtype=np.uint8)
    row = 0
    if has_anchor:
        sp[row:row + 3, 0:7] = 1
        row += 3
    for e in dom_edges:
        c0 = e.seg_idx * 7
        sp[row:row + 3, c0:c0 + 7] = 1
        row += 3
    for e in consist_edges:
        ca = e.seg_a * 7
        cb = e.seg_b * 7
        sp[row:row + 3, ca:ca + 7] = 1
        sp[row:row + 3, cb:cb + 7] = 1
        row += 3
    for e in dem_z_edges:
        c0 = e.seg_idx * 7
        k = int(len(e.dem_z_enu))
        sp[row:row + k, c0:c0 + 7] = 1
        row += k
    for e in camera_z_edges:
        c0 = e.seg_idx * 7
        k = int(len(e.target_z_enu))
        sp[row:row + k, c0:c0 + 7] = 1
        row += k
    for e in point_edges:
        c0 = e.seg_idx * 7
        k3 = 3 * int(e.pts_model.shape[0])
        sp[row:row + k3, c0:c0 + 7] = 1
        row += k3
    for e in ground_ray_edges:
        c0 = e.seg_idx * 7
        k2 = 2 * int(e.centers_model.shape[0])
        sp[row:row + k2, c0:c0 + 7] = 1
        row += k2
    for e in scale_edges:
        # Scale residual depends only on log_s columns (col offset 6).
        sp[row, e.seg_a * 7 + 6] = 1
        sp[row, e.seg_b * 7 + 6] = 1
        row += 1
    return sp.tocsr()


def _build_initial_params(chain: GlobalChainState) -> np.ndarray:
    """把所有段的当前 transform 打包成扁平的 7N 向量，作为优化初值。

    某段没有 transform 时用单位 Sim3（R=I, t=0, s=1）占位。
    """
    params = []
    for summary in chain.summaries:
        T = summary.transform
        if T is None:
            # Fallback: identity
            params.append(_pack_sim3(np.eye(3), np.zeros(3), 1.0))
            continue
        R = T.R.detach().cpu().numpy().astype(np.float64)
        t = T.t.detach().cpu().numpy().astype(np.float64)
        s = float(T.s.detach().cpu().item())
        params.append(_pack_sim3(R, t, s))
    return np.concatenate(params).astype(np.float64)


def _residual_fn(
    params_flat: np.ndarray,
    model_pts: List[np.ndarray],   # per-segment (N, 3) local-space trajectories
    dom_edges: List[_DomEdge],
    consist_edges: List[_ConsistEdge],
    dem_z_edges: List[_DemZEdge],
    camera_z_edges: List[_CameraZEdge],
    point_edges: List[_PointEdge],
    ground_ray_edges: List[_GroundRayEdge],
    scale_edges: List[_ScaleEdge],
    origin_enu: np.ndarray,        # (3,) ENU target for seg-0 first frame
    cfg: PGOConfig,
) -> np.ndarray:
    """为 scipy.optimize.least_squares 计算残差向量。

    把 7N 参数还原为每段的 Sim3，按与 ``_build_jac_sparsity`` 完全相同的顺序逐类
    拼接残差：seg-0 首帧硬锚点、DOM 一元、重叠一致性二元、DEM-Z、相机-Z、
    直接点、地面射线、相邻段尺度。各类残差均已乘上对应权重。
    """
    N = len(model_pts)
    params = params_flat.reshape(N, 7)
    residuals: List[np.ndarray] = []

    # 1) Segment-0 first-frame hard anchor
    if N > 0 and model_pts[0] is not None and len(model_pts[0]) > 0:
        p0 = model_pts[0][0:1]                   # (1, 3)
        enu0 = _apply_sim3(params[0], p0)[0]     # (3,)
        residuals.append((enu0 - origin_enu) * cfg.w_first_frame)

    # 2) Unary DOM edges
    for edge in dom_edges:
        p = model_pts[edge.seg_idx][edge.local_fi:edge.local_fi + 1]
        enu = _apply_sim3(params[edge.seg_idx], p)[0]
        diff = enu - edge.target_enu
        w = np.array([cfg.w_overlap_dom_xy, cfg.w_overlap_dom_xy, cfg.w_overlap_dom_z])
        residuals.append(diff * w)

    # 3) Binary overlap consistency
    for edge in consist_edges:
        pa = model_pts[edge.seg_a][edge.local_fi_a:edge.local_fi_a + 1]
        pb = model_pts[edge.seg_b][edge.local_fi_b:edge.local_fi_b + 1]
        enu_a = _apply_sim3(params[edge.seg_a], pa)[0]
        enu_b = _apply_sim3(params[edge.seg_b], pb)[0]
        residuals.append((enu_a - enu_b) * cfg.w_overlap_consist)

    # 4) DEM-Z unary edges (Z only). Pulls every road pt's predicted Z toward
    #    DEM ground Z. Aggregating across segments yields soft Z-consistency
    #    along the full chain without explicit neighbour pairing.
    for edge in dem_z_edges:
        enu = _apply_sim3(params[edge.seg_idx], edge.pts_model)   # (K, 3)
        residuals.append((enu[:, 2] - edge.dem_z_enu) * cfg.w_z_dem)

    # 5) Camera center DEM+AGL unary edges (Z only).
    if cfg.w_camera_agl > 0.0:
        for edge in camera_z_edges:
            enu = _apply_sim3(params[edge.seg_idx], edge.pts_model)
            residuals.append((enu[:, 2] - edge.target_z_enu) * cfg.w_camera_agl)

    # 6) Direct DOM/DEM point unary edges.
    if cfg.w_point_xy > 0.0 or cfg.w_point_z > 0.0:
        w_pt = np.array([cfg.w_point_xy, cfg.w_point_xy, cfg.w_point_z], dtype=np.float64)
        for edge in point_edges:
            enu = _apply_sim3(params[edge.seg_idx], edge.pts_model)
            residuals.append(((enu - edge.target_enu) * w_pt).reshape(-1))

    # 7) Image-center ground-ray unary edges.
    if cfg.w_ground_ray_xy > 0.0:
        for edge in ground_ray_edges:
            diff_xy = _ground_ray_xy_residual_np(
                params[edge.seg_idx],
                edge.centers_model,
                edge.dirs_model,
                edge.target_enu,
                cfg.ground_ray_min_abs_dir_z,
            )
            residuals.append((diff_xy * cfg.w_ground_ray_xy).reshape(-1))

    # 8) Adjacent-segment log-scale consistency (1 residual per pair).
    if cfg.w_scale > 0.0:
        for edge in scale_edges:
            log_s_a = float(params[edge.seg_a, 6])
            log_s_b = float(params[edge.seg_b, 6])
            residuals.append(np.array([(log_s_b - log_s_a) * cfg.w_scale], dtype=np.float64))

    if not residuals:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(residuals).astype(np.float64)


def _map_registered_segment_mask(chain: GlobalChainState) -> np.ndarray:
    """Identify segments whose final transform came from accepted map points.

    ``map_registration_accepted`` also includes overlap handoff for historical
    compatibility, so it is too broad for this purpose.  A hard absolute-map
    node requires both an accepted point bootstrap and selection of that
    bootstrap as the segment's primary transform.
    """
    mask = np.zeros(len(chain.summaries), dtype=bool)
    for segment_index, summary in enumerate(chain.summaries):
        meta = getattr(summary, "metadata", None) or {}
        attempt = meta.get("point_bootstrap_attempt")
        source = str(meta.get("transform_source") or "")
        primary = bool(meta.get("point_bootstrap_primary", False))
        accepted = isinstance(attempt, dict) and bool(attempt.get("accepted", False))
        if accepted and primary and source in {
            "dom_point_bootstrap_primary",
            "dom_point_bootstrap",
        }:
            mask[segment_index] = True
    return mask


def _pgo_free_parameter_mask(
    num_segments: int,
    cfg: PGOConfig,
    fixed_xy_segments: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the optimizable coordinates of the packed per-segment Sim(3)s."""
    mask = np.ones(max(0, int(num_segments)) * 7, dtype=bool)
    fixed_t = (
        np.asarray(fixed_xy_segments, dtype=bool).reshape(-1)
        if fixed_xy_segments is not None
        else np.zeros(max(0, int(num_segments)), dtype=bool)
    )
    if fixed_t.size != max(0, int(num_segments)):
        raise ValueError("fixed_xy_segments must match num_segments")
    for segment_index in range(max(0, int(num_segments))):
        offset = segment_index * 7
        if bool(getattr(cfg, "fix_rotation", False)):
            mask[offset:offset + 3] = False
        if fixed_t[segment_index]:
            mask[offset + 3:offset + 5] = False
        if bool(getattr(cfg, "fix_scale", False)):
            mask[offset + 6] = False
    return mask


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def refine_transforms_with_pgo(
    chain: GlobalChainState,
    origin_enu: Optional[np.ndarray] = None,
    cfg: Optional[PGOConfig] = None,
) -> Tuple[List[LocalModelTransform], Dict[str, float]]:
    """对一条已完成的链路（chain）运行 Plan-A 全局 PGO。

    流程：收集各段 model-space 轨迹 -> 抽取各类边 -> 按权重置零关闭对应边 ->
    用 scipy.least_squares（trf + 稀疏 Jacobian）一次性优化全部段的 7N 个 Sim3 参数。

    返回：
        new_transforms: 精修后的 LocalModelTransform 列表，与 chain.summaries 一一对应；
        stats:          求解诊断（初始/最终代价、迭代数、各类边数等）。
    """
    cfg = cfg or PGOConfig()
    origin_enu = (np.asarray(origin_enu, dtype=np.float64).reshape(3)
                  if origin_enu is not None else np.zeros(3, dtype=np.float64))

    # Collect per-segment local trajectories
    model_pts: List[np.ndarray] = []
    for summary in chain.summaries:
        if summary.trajectory_local is None:
            model_pts.append(np.zeros((0, 3), dtype=np.float64))
        else:
            model_pts.append(np.asarray(summary.trajectory_local, dtype=np.float64))

    (dom_edges, consist_edges, dem_z_edges, camera_z_edges, point_edges,
     ground_ray_edges, scale_edges) = _extract_edges(chain)
    if cfg.w_overlap_dom_xy <= 0.0 and cfg.w_overlap_dom_z <= 0.0:
        dom_edges = []
    if cfg.w_z_dem <= 0.0:
        dem_z_edges = []
    if cfg.w_camera_agl <= 0.0:
        camera_z_edges = []
    if cfg.w_point_xy <= 0.0 and cfg.w_point_z <= 0.0:
        point_edges = []
    if cfg.w_ground_ray_xy <= 0.0:
        ground_ray_edges = []
    scale_edges_dropped = 0
    if bool(getattr(cfg, "fix_scale", False)):
        # Each segment has its own model-space normalization. Equality between
        # numeric model->ENU scales is therefore not a valid physical prior.
        scale_edges_dropped = int(len(scale_edges))
        scale_edges = []

    x0 = _build_initial_params(chain)
    fixed_xy_segments = (
        _map_registered_segment_mask(chain)
        if bool(getattr(cfg, "fix_map_registered_xy", False))
        else np.zeros(len(chain.summaries), dtype=bool)
    )
    free_parameter_mask = _pgo_free_parameter_mask(
        len(chain.summaries), cfg, fixed_xy_segments)
    x0_free = x0[free_parameter_mask]
    max_nfev = max(1, int(cfg.max_iter))
    n_dem_total = sum(int(len(e.dem_z_enu)) for e in dem_z_edges)
    n_camera_z_total = sum(int(len(e.target_z_enu)) for e in camera_z_edges)
    n_point_total = sum(int(e.pts_model.shape[0]) for e in point_edges)
    n_ground_ray_total = sum(int(e.centers_model.shape[0]) for e in ground_ray_edges)
    print(
        f"[PGO] start: segments={len(chain.summaries)} "
        f"vars={len(x0_free)}/{len(x0)} "
        f"dom_edges={len(dom_edges)} consist_edges={len(consist_edges)} "
        f"dem_z_rows={n_dem_total} camera_z_rows={n_camera_z_total} "
        f"point_pts={n_point_total} ground_rays={n_ground_ray_total} "
        f"scale_edges={len(scale_edges)} dropped_scale_edges={scale_edges_dropped} "
        f"fix_R={bool(cfg.fix_rotation)} fix_s={bool(cfg.fix_scale)} "
        f"fixed_map_xy={int(fixed_xy_segments.sum())} "
        f"max_nfev={max_nfev} verbose={bool(cfg.verbose)}"
    )

    if cfg.verbose:
        dom_per_seg = [0] * len(chain.summaries)
        for e in dom_edges:
            dom_per_seg[e.seg_idx] += 1
        dem_per_seg = [0] * len(chain.summaries)
        for e in dem_z_edges:
            dem_per_seg[e.seg_idx] = int(len(e.dem_z_enu))
        camera_z_per_seg = [0] * len(chain.summaries)
        for e in camera_z_edges:
            camera_z_per_seg[e.seg_idx] = int(len(e.target_z_enu))
        point_per_seg = [0] * len(chain.summaries)
        for e in point_edges:
            point_per_seg[e.seg_idx] = int(e.pts_model.shape[0])
        ground_ray_per_seg = [0] * len(chain.summaries)
        for e in ground_ray_edges:
            ground_ray_per_seg[e.seg_idx] = int(e.centers_model.shape[0])
        s_init = [float(np.exp(x0.reshape(-1, 7)[k, 6]))
                  for k in range(len(chain.summaries))]
        print(f"[PGO] segments={len(chain.summaries)}  "
              f"dom_edges={len(dom_edges)}  consist_edges={len(consist_edges)}  "
              f"dem_z_edges={len(dem_z_edges)} (rows={n_dem_total})  "
              f"camera_z_edges={len(camera_z_edges)} (rows={n_camera_z_total}, "
              f"w={cfg.w_camera_agl})  "
              f"point_edges={len(point_edges)} (pts={n_point_total}, "
              f"w=({cfg.w_point_xy},{cfg.w_point_z}))  "
              f"ground_ray_edges={len(ground_ray_edges)} (rays={n_ground_ray_total}, "
              f"w={cfg.w_ground_ray_xy})  "
              f"scale_edges={len(scale_edges)} (w={cfg.w_scale})")
        print(f"[PGO] DOM edges per segment: {dom_per_seg}")
        print(f"[PGO] DEM-Z rows per segment: {dem_per_seg}")
        print(f"[PGO] camera-Z rows per segment: {camera_z_per_seg}")
        print(f"[PGO] point rows per segment: {point_per_seg}")
        print(f"[PGO] ground-ray rows per segment: {ground_ray_per_seg}")
        print(f"[PGO] s_k (pre-PGO): " + ", ".join(f"{v:.2f}" for v in s_init))

    point_residual_pre = _summarize_point_residuals(x0, point_edges)
    ground_ray_residual_pre = _summarize_ground_ray_residuals(
        x0, ground_ray_edges, cfg.ground_ray_min_abs_dir_z)
    point_edge_source_stats = _aggregate_point_edge_source_stats(chain)
    r0 = _residual_fn(x0, model_pts, dom_edges, consist_edges, dem_z_edges,
                   camera_z_edges, point_edges, ground_ray_edges, scale_edges, origin_enu, cfg)
    init_cost = float(0.5 * np.sum(r0 * r0))

    fixed_map_segments = np.flatnonzero(fixed_xy_segments).astype(int).tolist()
    if x0_free.size == 0:
        print("[PGO] all transform coordinates fixed - returning initial transforms")
        return _unpack_transforms(x0, chain), {
            "init_cost": init_cost,
            "final_cost": init_cost,
            "num_dom_edges": len(dom_edges),
            "num_consist_edges": len(consist_edges),
            "num_dem_z_edges": len(dem_z_edges),
            "num_camera_z_edges": len(camera_z_edges),
            "num_point_edges": len(point_edges),
            "num_ground_ray_edges": len(ground_ray_edges),
            "num_scale_edges": len(scale_edges),
            "num_scale_edges_dropped": int(scale_edges_dropped),
            "fix_rotation": bool(cfg.fix_rotation),
            "fix_scale": bool(cfg.fix_scale),
            "fix_map_registered_xy": bool(cfg.fix_map_registered_xy),
            "fixed_map_xy_segments": fixed_map_segments,
            "num_fixed_map_xy_segments": len(fixed_map_segments),
            "num_free_parameters": 0,
            "iterations": 0,
            "point_residual_pre": point_residual_pre,
            "point_residual_post": point_residual_pre,
            "ground_ray_residual_pre": ground_ray_residual_pre,
            "ground_ray_residual_post": ground_ray_residual_pre,
            "point_edge_source_stats": point_edge_source_stats,
        }

    if (len(dom_edges) == 0 and len(consist_edges) == 0
            and len(dem_z_edges) == 0 and len(point_edges) == 0
            and len(camera_z_edges) == 0 and len(ground_ray_edges) == 0
            and len(scale_edges) == 0):
        print("[PGO] no edges - skipping optimization, returning initial transforms")
        return _unpack_transforms(x0, chain), {"init_cost": init_cost,
                                                "final_cost": init_cost,
                                                "num_dom_edges": 0,
                                                "num_consist_edges": 0,
                                                "num_dem_z_edges": 0,
                                                "num_camera_z_edges": 0,
                                                "num_point_edges": 0,
                                                "num_ground_ray_edges": 0,
                                                "num_scale_edges": 0,
                                                "num_scale_edges_dropped": int(scale_edges_dropped),
                                                "fix_rotation": bool(cfg.fix_rotation),
                                                "fix_scale": bool(cfg.fix_scale),
                                                "fix_map_registered_xy": bool(cfg.fix_map_registered_xy),
                                                "fixed_map_xy_segments": fixed_map_segments,
                                                "num_fixed_map_xy_segments": len(fixed_map_segments),
                                                "num_free_parameters": int(free_parameter_mask.sum()),
                                                "iterations": 0,
                                                "point_residual_pre": point_residual_pre,
                                                "point_residual_post": point_residual_pre,
                                                "ground_ray_residual_pre": ground_ray_residual_pre,
                                                "ground_ray_residual_post": ground_ray_residual_pre,
                                                "point_edge_source_stats": point_edge_source_stats}

    jac_sparsity = _build_jac_sparsity(
        N=len(chain.summaries),
        model_pts=model_pts,
        dom_edges=dom_edges,
        consist_edges=consist_edges,
        dem_z_edges=dem_z_edges,
        camera_z_edges=camera_z_edges,
        point_edges=point_edges,
        ground_ray_edges=ground_ray_edges,
        scale_edges=scale_edges,
    )
    if jac_sparsity is not None:
        jac_sparsity = jac_sparsity[:, free_parameter_mask]
    if cfg.verbose and jac_sparsity is not None:
        nnz = int(jac_sparsity.nnz)
        total = int(jac_sparsity.shape[0]) * int(jac_sparsity.shape[1])
        density = nnz / max(total, 1)
        print(f"[PGO] jac_sparsity shape={jac_sparsity.shape}  nnz={nnz}  "
              f"density={density:.2e}")

    def _expand_free_parameters(values: np.ndarray) -> np.ndarray:
        expanded = x0.copy()
        expanded[free_parameter_mask] = values
        return expanded

    def _residual_from_free_parameters(values: np.ndarray) -> np.ndarray:
        return _residual_fn(
            _expand_free_parameters(values),
            model_pts,
            dom_edges,
            consist_edges,
            dem_z_edges,
            camera_z_edges,
            point_edges,
            ground_ray_edges,
            scale_edges,
            origin_enu,
            cfg,
        )

    result = least_squares(
        _residual_from_free_parameters, x0_free,
        method="trf",
        jac_sparsity=jac_sparsity,
        max_nfev=max_nfev,
        verbose=2 if cfg.verbose else 0,
    )
    x_final = _expand_free_parameters(result.x)
    final_cost = float(0.5 * np.sum(result.fun * result.fun))
    point_residual_post = _summarize_point_residuals(x_final, point_edges)
    ground_ray_residual_post = _summarize_ground_ray_residuals(
        x_final, ground_ray_edges, cfg.ground_ray_min_abs_dir_z)

    if cfg.verbose:
        print(f"[PGO] init_cost={init_cost:.3f}  final_cost={final_cost:.3f}  "
              f"nfev={result.nfev}  status={result.status}")
        if point_residual_pre.get("n_points", 0) > 0:
            print(
                "[PGO][PointResidual] "
                f"pre_xy_rmse={point_residual_pre.get('xy_rmse_m', 0.0):.3f}m "
                f"post_xy_rmse={point_residual_post.get('xy_rmse_m', 0.0):.3f}m "
                f"pre_xy_med={point_residual_pre.get('xy_median_m', 0.0):.3f}m "
                f"post_xy_med={point_residual_post.get('xy_median_m', 0.0):.3f}m "
                f"n={point_residual_post.get('n_points', 0)}"
            )
        if ground_ray_residual_pre.get("n_rays", 0) > 0:
            print(
                "[PGO][GroundRayResidual] "
                f"pre_xy_rmse={ground_ray_residual_pre.get('xy_rmse_m', 0.0):.3f}m "
                f"post_xy_rmse={ground_ray_residual_post.get('xy_rmse_m', 0.0):.3f}m "
                f"pre_xy_med={ground_ray_residual_pre.get('xy_median_m', 0.0):.3f}m "
                f"post_xy_med={ground_ray_residual_post.get('xy_median_m', 0.0):.3f}m "
                f"n={ground_ray_residual_post.get('n_rays', 0)}"
            )
        if point_edge_source_stats:
            print(f"[PGO][PointSource] {point_edge_source_stats}")
        s_post = [float(np.exp(x_final.reshape(-1, 7)[k, 6]))
                  for k in range(len(chain.summaries))]
        print(f"[PGO] s_k (post-PGO): " + ", ".join(f"{v:.2f}" for v in s_post))

    stats = {
        "init_cost": init_cost,
        "final_cost": final_cost,
        "num_dom_edges": len(dom_edges),
        "num_consist_edges": len(consist_edges),
        "num_dem_z_edges": len(dem_z_edges),
        "num_camera_z_edges": len(camera_z_edges),
        "num_point_edges": len(point_edges),
        "num_ground_ray_edges": len(ground_ray_edges),
        "num_scale_edges": len(scale_edges),
        "num_scale_edges_dropped": int(scale_edges_dropped),
        "fix_rotation": bool(cfg.fix_rotation),
        "fix_scale": bool(cfg.fix_scale),
        "fix_map_registered_xy": bool(cfg.fix_map_registered_xy),
        "fixed_map_xy_segments": fixed_map_segments,
        "num_fixed_map_xy_segments": len(fixed_map_segments),
        "num_free_parameters": int(free_parameter_mask.sum()),
        "iterations": int(result.nfev),
        "max_nfev": int(max_nfev),
        "point_residual_pre": point_residual_pre,
        "point_residual_post": point_residual_post,
        "ground_ray_residual_pre": ground_ray_residual_pre,
        "ground_ray_residual_post": ground_ray_residual_post,
        "point_edge_source_stats": point_edge_source_stats,
    }
    return _unpack_transforms(x_final, chain), stats


def _unpack_transforms(params_flat: np.ndarray,
                        chain: GlobalChainState) -> List[LocalModelTransform]:
    """把扁平的 7N 参数还原成精修后的 LocalModelTransform 列表。

    尽量沿用原 transform 的 device / dtype / anchor；原 transform 缺失时退回 CPU /
    float32 / 零锚点。
    """
    N = len(chain.summaries)
    params = params_flat.reshape(N, 7)
    out: List[LocalModelTransform] = []
    for k, summary in enumerate(chain.summaries):
        R, t, s = _unpack_sim3(params[k])
        # Prefer device/dtype of the original transform if available
        if summary.transform is not None:
            dev = summary.transform.s.device
            dtype = summary.transform.s.dtype
            anchor = summary.transform.anchor_world_xyz
        else:
            dev, dtype = torch.device("cpu"), torch.float32
            anchor = torch.zeros(3, device=dev, dtype=dtype)
        new_T = LocalModelTransform(
            anchor_world_xyz=anchor,
            R=torch.tensor(R, device=dev, dtype=dtype),
            s=torch.tensor(s, device=dev, dtype=dtype),
            t=torch.tensor(t, device=dev, dtype=dtype),
        )
        out.append(new_T)
    return out


# -----------------------------------------------------------------------------
# Utility: re-assemble global trajectory with refined transforms
# -----------------------------------------------------------------------------

def assemble_trajectory(
    chain: GlobalChainState,
    transforms: List[LocalModelTransform],
    num_frames: int,
) -> np.ndarray:
    """用精修后的各段 T_k 重新拼装出最终 (num_frames, 3) ENU 轨迹。

    逻辑与 GeoV3Runner._dedupe_by_frame_index 一致，区别是 transform 由外部传入
    （即 PGO 精修后的结果）。重叠帧上后段覆盖前段，缺帧用 hold-last 填充。
    """
    frame_map: Dict[int, np.ndarray] = {}
    for summary, T in zip(chain.summaries, transforms):
        traj_local = summary.trajectory_local
        if traj_local is None or len(traj_local) == 0:
            continue
        pts = torch.tensor(np.asarray(traj_local, dtype=np.float32),
                           device=T.s.device, dtype=T.s.dtype)
        traj_enu = T.model_to_world(pts).detach().cpu().numpy()
        start = int(summary.start_frame)
        for local_idx, point in enumerate(traj_enu):
            fi = start + local_idx
            if 0 <= fi < num_frames:
                frame_map[fi] = np.asarray(point, dtype=np.float32)

    if not frame_map:
        return np.zeros((0, 3), dtype=np.float32)
    ordered = []
    last = None
    for fi in range(num_frames):
        if fi in frame_map:
            last = frame_map[fi]
            ordered.append(last)
        elif last is not None:
            ordered.append(last.copy())
        else:
            ordered.append(np.zeros(3, dtype=np.float32))
    return np.stack(ordered, axis=0).astype(np.float32)


# =============================================================================
# Segment-level PGO (in-segment, replaces the Adam fused step)
# =============================================================================

@dataclass
class SegmentPGOConfig:
    """段内 PGO 步骤的配置（启用后取代 Adam fused 步骤）。"""
    enable: bool = False
    w_dom_xy: float = 5.0       # DOM 一元 XY 权重
    w_dom_z: float = 2.0        # DOM 一元 Z 权重
    w_consist: float = 8.0      # 与上段重叠的一致性（XY）权重（调用方提供时生效）
    w_consist_z: float = 16.0   # 一致性（Z）权重——单独且更高，用于抑制逐段 z 漂移
    w_delta_reg: float = 2.0    # ||delta_i|| 的 L2 先验
    w_pose_rot: float = 50.0    # 旋转的测地 SO(3) 先验（相对 omega_init 的偏移）
    w_pose_trans: float = 2.0   # (t - t_init) 平移先验
    yaw_only_rotation: bool = False     # 旋转是否仅限 yaw（绕 Z）
    use_point_edges: bool = False       # 是否使用直接点边
    w_point_xy: float = 0.5             # 点边 XY 权重
    w_point_z: float = 0.1              # 点边 Z 权重
    use_ground_ray: bool = False        # 是否使用地面射线边
    w_ground_ray_xy: float = 2.0        # 地面射线 XY 权重
    ground_ray_min_abs_dir_z: float = 0.05  # 射线方向 z 分量最小绝对值（防除零）
    freeze_delta: bool = True   # 为 True 时只优化 6-DoF 位姿（不优化逐点 delta）
    max_iter: int = 20          # LM/TRF 最大迭代数
    delta_clip: float = 5.0     # 把最终 delta_i 截断到 [-clip, clip] 米
    verbose: bool = False       # 是否打印日志


def refine_segment_pgo(
    model_pts: np.ndarray,                          # (N, 3) M_k space, current camera centers
    transform_init: LocalModelTransform,            # T_k initial (R, t; s is frozen)
    dom_targets_enu: Dict[int, np.ndarray],         # {local_fi: (3,) ENU camera-center target}
    prev_M0_overlap: Optional[np.ndarray],          # (overlap_len, 3) in M_0; None for seg 0
    t0_transform: Optional[LocalModelTransform],    # T_0; None for seg 0 (same as transform_init)
    overlap_len: int,
    seg_id: int,
    cfg: SegmentPGOConfig,
    point_model: Optional[np.ndarray] = None,       # (K, 3) M_k matched points
    point_targets_enu: Optional[np.ndarray] = None, # (K, 3) DOM/DEM ENU targets
    ground_ray_dirs_model: Optional[Dict[int, np.ndarray]] = None,
    ground_ray_targets_enu: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[LocalModelTransform, np.ndarray, Dict[str, float]]:
    """段内 scipy 最小二乘精修 (omega, t, delta)。

    尺度 s 被**冻结**为 transform_init.s（Umeyama 已给出最优尺度，否则 LM 会震荡）。
    优化变量 = [omega(3), t(3), delta(3*N)]，共 6 + 3N 维（freeze_delta 时仅 6 维）。

    残差项：
      1. 每个 fi 的 DOM 一元（dom_targets_enu）：
            r = (T_k(p_i + delta_i) - target_i^ENU) * [w_xy, w_xy, w_z]
      2. 一致性（seg_id >= 1，i 取 [0, overlap_len)）：
            target_i = T_0(prev_M0_overlap[i])
            r = (T_k(p_i + delta_i) - target_i) * w_consist
        3. 启用时的直接点一元：
            r = (T_k(x_j) - X_j^ENU) * [w_point_xy, w_point_xy, w_point_z]
        4. 启用时的地面射线一元：
            r = intersect_xy(T_k(c_i), R_k d_i, target_z) - target_xy
        5. delta L2 先验：r = delta_i * w_delta_reg

    返回：(refined_transform, delta(N,3), stats)
    """
    N = int(model_pts.shape[0])
    s_frozen = float(transform_init.s.detach().cpu().item())

    # Initial omega, t from transform_init
    R0 = transform_init.R.detach().cpu().numpy().astype(np.float64)
    t0 = transform_init.t.detach().cpu().numpy().astype(np.float64)
    omega0 = _log_so3(R0)
    freeze_delta = bool(cfg.freeze_delta)
    yaw_only_rotation = bool(getattr(cfg, "yaw_only_rotation", False))
    pose_var_dim = 4 if yaw_only_rotation else 6
    n_vars = pose_var_dim if freeze_delta else pose_var_dim + 3 * N
    if yaw_only_rotation:
        pose_init = np.concatenate([np.zeros(1, dtype=np.float64), t0]).astype(np.float64)
    else:
        pose_init = np.concatenate([omega0, t0]).astype(np.float64)
    if freeze_delta:
        x0 = pose_init
    else:
        x0 = np.concatenate([pose_init, np.zeros(3 * N)]).astype(np.float64)

    pts_np = np.asarray(model_pts, dtype=np.float64)  # (N, 3)

    # Pre-compute consist targets in ENU (T_0 · prev_M0_overlap)
    consist_targets: Optional[np.ndarray] = None  # (overlap_len, 3)
    consist_n = 0
    if (
        seg_id >= 1
        and prev_M0_overlap is not None
        and t0_transform is not None
        and overlap_len > 0
    ):
        L = min(int(overlap_len), int(prev_M0_overlap.shape[0]), N)
        if L > 0:
            with torch.no_grad():
                _pts = torch.tensor(prev_M0_overlap[:L], dtype=t0_transform.s.dtype,
                                    device=t0_transform.s.device)
                _enu = t0_transform.model_to_world(_pts).detach().cpu().numpy()
            consist_targets = _enu.astype(np.float64)
            consist_n = L

    dom_items = sorted(dom_targets_enu.items())
    if cfg.w_dom_xy <= 0.0 and cfg.w_dom_z <= 0.0:
        dom_items = []
    dom_n = sum(1 for fi, _ in dom_items if 0 <= int(fi) < N)
    point_np = None
    point_tgt_np = None
    point_n = 0
    if cfg.use_point_edges and point_model is not None and point_targets_enu is not None:
        _pm = np.asarray(point_model, dtype=np.float64)
        _pt = np.asarray(point_targets_enu, dtype=np.float64)
        if (_pm.ndim == 2 and _pm.shape[1] == 3
                and _pt.ndim == 2 and _pt.shape[1] == 3
                and _pm.shape[0] == _pt.shape[0]
                and _pm.shape[0] > 0):
            _finite = np.isfinite(_pm).all(axis=1) & np.isfinite(_pt).all(axis=1)
            if np.any(_finite):
                point_np = _pm[_finite]
                point_tgt_np = _pt[_finite]
                point_n = int(point_np.shape[0])

    ground_items: List[Tuple[int, np.ndarray, np.ndarray]] = []
    if cfg.use_ground_ray and ground_ray_dirs_model and ground_ray_targets_enu:
        for fi_raw, dir_model in sorted(ground_ray_dirs_model.items()):
            try:
                fi = int(fi_raw)
            except (TypeError, ValueError):
                continue
            if fi < 0 or fi >= N or fi not in ground_ray_targets_enu:
                continue
            dir_np = np.asarray(dir_model, dtype=np.float64).reshape(-1)
            tgt_np = np.asarray(ground_ray_targets_enu[fi], dtype=np.float64).reshape(-1)
            if dir_np.size < 3 or tgt_np.size < 3:
                continue
            if not (np.isfinite(dir_np[:3]).all() and np.isfinite(tgt_np[:3]).all()):
                continue
            norm = float(np.linalg.norm(dir_np[:3]))
            if norm <= 1e-8:
                continue
            ground_items.append((fi, dir_np[:3] / norm, tgt_np[:3].copy()))
    ground_ray_n = int(len(ground_items))

    if dom_n == 0 and consist_n == 0 and point_n == 0 and ground_ray_n == 0:
        if cfg.verbose:
            print(
                f"[GeoV3][SegPGO] seg {seg_id}: no edges "
                f"(dom={dom_n}, consist={consist_n}, points={point_n}, "
                f"ground_rays={ground_ray_n}); skipping"
            )
        return transform_init, np.zeros((N, 3), dtype=np.float32), {
            "dom_n": 0, "consist_n": 0, "point_n": 0, "ground_ray_n": 0,
            "init_cost": 0.0, "final_cost": 0.0, "nfev": 0,
            "delta_max": 0.0, "delta_mean": 0.0, "t_delta": 0.0, "omega_delta": 0.0,
        }

    def _residuals(x: np.ndarray) -> np.ndarray:
        if yaw_only_rotation:
            yaw_delta = float(x[0])
            cos_yaw = float(np.cos(yaw_delta))
            sin_yaw = float(np.sin(yaw_delta))
            yaw_rotation = np.array([
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            R = yaw_rotation @ R0
            t = x[1:4]
        else:
            omega = x[0:3]
            R = _rodrigues(omega)
            t = x[3:6]
        if freeze_delta:
            delta = np.zeros((N, 3), dtype=np.float64)
        else:
            delta = x[pose_var_dim:].reshape(N, 3)
        # T_k(p + delta) = s * R @ (p + delta) + t
        pts_enu = s_frozen * ((pts_np + delta) @ R.T) + t[None, :]   # (N, 3)

        residuals: List[np.ndarray] = []

        # 1) DOM unary
        if dom_n > 0:
            w = np.array([cfg.w_dom_xy, cfg.w_dom_xy, cfg.w_dom_z], dtype=np.float64)
            for fi, target in dom_items:
                fi = int(fi)
                if fi < 0 or fi >= N:
                    continue
                tgt = np.asarray(target, dtype=np.float64).reshape(3)
                residuals.append((pts_enu[fi] - tgt) * w)

        # 2) Consist (XY/Z weights split: Z gets higher weight to suppress
        # per-segment z drift while keeping XY behaviour identical to the
        # legacy single-w_consist path when w_consist_z == w_consist.)
        if consist_n > 0 and consist_targets is not None:
            w_c = np.array([cfg.w_consist, cfg.w_consist, cfg.w_consist_z],
                           dtype=np.float64)
            for i in range(consist_n):
                residuals.append((pts_enu[i] - consist_targets[i]) * w_c)

        # 3) Direct point correspondences: matched predicted 3D points should
        # land on their DOM/DEM ENU targets under the same segment transform.
        if point_n > 0 and point_np is not None and point_tgt_np is not None:
            point_enu = s_frozen * (point_np @ R.T) + t[None, :]
            w_p = np.array([cfg.w_point_xy, cfg.w_point_xy, cfg.w_point_z], dtype=np.float64)
            residuals.append(((point_enu - point_tgt_np) * w_p).reshape(-1))

        # 4) Image-center ground ray: camera center plus optical-axis direction
        # should hit the DOM frame-center ground point on the horizontal plane.
        if ground_ray_n > 0 and cfg.w_ground_ray_xy > 0.0:
            gr_parts: List[np.ndarray] = []
            for fi, dir_model, target in ground_items:
                center = pts_enu[fi]
                direction = dir_model @ R.T
                direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
                denom = _signed_safe_denominator_np(
                    np.asarray([direction[2]], dtype=np.float64),
                    cfg.ground_ray_min_abs_dir_z,
                )[0]
                lam = (target[2] - center[2]) / denom
                pred_xy = center[:2] + lam * direction[:2]
                gr_parts.append((pred_xy - target[:2]) * cfg.w_ground_ray_xy)
            if gr_parts:
                residuals.append(np.concatenate(gr_parts).astype(np.float64))

        # 5) pose prior — geodesic SO(3) distance instead of Euclidean diff in
        # omega coords. The Euclidean form (omega - omega0) admits 2pi-wrap
        # solutions that represent the same rotation but pay only the prior
        # cost (observed |omega_delta|~7rad in airzoo_real / Japan logs).
        # log_so3(R @ R0^T) is bounded to ||.|| <= pi and is a true geodesic.
        R_err = R @ R0.T
        omega_err = _log_so3(R_err)
        residuals.append(omega_err * cfg.w_pose_rot)
        residuals.append((t - t0) * cfg.w_pose_trans)

        # 6) delta L2 prior
        if not freeze_delta:
            residuals.append((delta * cfg.w_delta_reg).reshape(-1))

        return np.concatenate(residuals).astype(np.float64)

    init_res = _residuals(x0)
    init_cost = float(0.5 * np.sum(init_res * init_res))

    result = least_squares(
        _residuals, x0,
        method="trf",
        max_nfev=cfg.max_iter * max(len(x0), 1),
        x_scale="jac",
        verbose=2 if cfg.verbose else 0,
    )
    final_cost = float(0.5 * np.sum(result.fun * result.fun))

    if yaw_only_rotation:
        yaw_delta_f = float(result.x[0])
        cos_yaw_f = float(np.cos(yaw_delta_f))
        sin_yaw_f = float(np.sin(yaw_delta_f))
        yaw_rotation_f = np.array([
            [cos_yaw_f, -sin_yaw_f, 0.0],
            [sin_yaw_f, cos_yaw_f, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        R_f = yaw_rotation_f @ R0
        t_f = result.x[1:4]
    else:
        omega_f = result.x[0:3]
        R_f = _rodrigues(omega_f)
        t_f = result.x[3:6]
    if freeze_delta:
        delta_f = np.zeros((N, 3), dtype=np.float64)
    else:
        delta_f = result.x[pose_var_dim:].reshape(N, 3)
        if cfg.delta_clip > 0:
            delta_f = np.clip(delta_f, -cfg.delta_clip, cfg.delta_clip)

    dev = transform_init.s.device
    dtype = transform_init.s.dtype
    refined = LocalModelTransform(
        anchor_world_xyz=transform_init.anchor_world_xyz,
        R=torch.tensor(R_f, device=dev, dtype=dtype),
        s=torch.tensor(s_frozen, device=dev, dtype=dtype),
        t=torch.tensor(t_f, device=dev, dtype=dtype),
    )

    # Geodesic SO(3) distance for the log: ||log(R_f @ R0^T)|| <= pi.
    # Euclidean ||omega_f - omega0|| is misleading because of 2pi wrap-around.
    omega_geo = _log_so3(R_f @ R0.T)
    stats = {
        "dom_n": int(dom_n),
        "consist_n": int(consist_n),
        "point_n": int(point_n),
        "ground_ray_n": int(ground_ray_n),
        "init_cost": init_cost,
        "final_cost": final_cost,
        "nfev": int(result.nfev),
        "delta_max": float(np.max(np.abs(delta_f))),
        "delta_mean": float(np.mean(np.linalg.norm(delta_f, axis=-1))),
        "t_delta": float(np.linalg.norm(t_f - t0)),
        "omega_delta": float(np.linalg.norm(omega_geo)),
        "yaw_only_rotation": bool(yaw_only_rotation),
    }
    if yaw_only_rotation:
        stats["yaw_delta"] = float(abs(yaw_delta_f))
    return refined, delta_f.astype(np.float32), stats
