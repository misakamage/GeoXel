"""Plan A: segment-level pose graph optimization (PGO).

Post-processing step that refines per-segment Sim3 transforms ``T_k`` using:

- **Unary prior** on segment-0 first frame: ``T_0(p_0^{M_0}) = origin_enu`` (hard).
- **Unary DOM edges** for overlap frames only:
  ``T_k(p_{f}^{M_k}) ≈ dom_target_f^{ENU}`` for each overlap frame ``f``.
- **Binary overlap-consistency edges**: for a frame ``f`` shared between two
  adjacent segments (``k`` and ``k+1``),
  ``T_k(p_f^{M_k}) ≈ T_{k+1}(p_f^{M_{k+1}})``.

Each ``T_k`` is parameterized as 7 DoF Sim3:
``(omega_x, omega_y, omega_z, t_x, t_y, t_z, log_s)``. Rotation uses Rodrigues.

The module does not touch the neural network or the TTT pipeline. It consumes
a completed ``GlobalChainState`` and returns a new list of ``LocalModelTransform``
objects (one per segment) that can be used to re-assemble the global trajectory.
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
    """Rodrigues: so(3) 3-vector → 3x3 rotation matrix."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = omega / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _log_so3(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → so(3) 3-vector."""
    tr = float(np.trace(R))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
                 dtype=np.float64)
    return w * (theta / (2.0 * np.sin(theta)))


def _pack_sim3(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """(R, t, s) → 7-vector [omega(3), t(3), log_s]."""
    return np.concatenate([_log_so3(R), np.asarray(t, dtype=np.float64).reshape(3),
                           [np.log(max(float(s), 1e-8))]]).astype(np.float64)


def _unpack_sim3(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """7-vector → (R, t, s)."""
    R = _rodrigues(v[0:3])
    t = v[3:6].astype(np.float64)
    s = float(np.exp(v[6]))
    return R, t, s


def _apply_sim3(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply Sim3 to (N, 3) points: ENU = s * R @ p + t."""
    R, t, s = _unpack_sim3(v)
    return s * (pts @ R.T) + t[None, :]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class PGOConfig:
    w_first_frame: float = 1e3      # segment-0 first-frame hard anchor
    w_overlap_dom_xy: float = 5.0   # overlap DOM XY unary
    w_overlap_dom_z: float = 1.0    # overlap DOM Z unary (weaker; DEM Z noise)
    w_overlap_consist: float = 10.0 # binary overlap consistency between segments
    w_z_dem: float = 5.0            # per-segment road-pt DEM Z unary (Z only)
    w_camera_agl: float = 5.0       # camera center Z toward DEM(camera XY)+AGL
    w_point_xy: float = 0.0         # direct DOM/DEM point XY unary (0 disables)
    w_point_z: float = 0.0          # direct DOM/DEM point Z unary (0 disables)
    w_ground_ray_xy: float = 0.0    # image-center ground-ray XY unary (0 disables)
    ground_ray_min_abs_dir_z: float = 0.05
    w_scale: float = 50.0           # adjacent-segment log(s) consistency (0 disables)
    max_iter: int = 200             # direct scipy least_squares max_nfev
    verbose: bool = True


# -----------------------------------------------------------------------------
# Edge extraction
# -----------------------------------------------------------------------------

@dataclass
class _DomEdge:
    seg_idx: int       # index into chain.summaries
    local_fi: int      # local frame index within that segment
    target_enu: np.ndarray  # (3,) ENU XYZ from DOM+DEM


@dataclass
class _ConsistEdge:
    seg_a: int         # earlier segment index
    seg_b: int         # later segment index
    local_fi_a: int    # local idx of shared frame in seg_a
    local_fi_b: int    # local idx of shared frame in seg_b


@dataclass
class _ScaleEdge:
    """Adjacent-segment log-scale consistency edge.

    Residual: ``(log_s_b - log_s_a) * w_scale``. Soft prior that the model→ENU
    scale is consistent across neighbouring submaps (same camera, same scene).
    """
    seg_a: int
    seg_b: int


@dataclass
class _DemZEdge:
    """Per-segment road-pt DEM Z unary edge.

    Pulls T_k(pts_model)[:, 2] toward DEM ground Z (in ENU). Z-only residual
    (one scalar per road point). Aggregating across segments produces a soft
    inter-segment Z consistency since neighbouring segments share the DEM.
    """
    seg_idx: int
    pts_model: np.ndarray   # (K, 3) M_k space road points
    dem_z_enu: np.ndarray   # (K,)   DEM ground Z in ENU


@dataclass
class _CameraZEdge:
    """Per-segment camera center DEM+AGL unary edge.

    Pulls camera centers toward a DEM-following altitude target, typically
    DEM(camera_xy) + anchor AGL. Unlike road DEM-Z, this constrains camera
    height directly and also covers frames with no road samples.
    """
    seg_idx: int
    pts_model: np.ndarray       # (K, 3) M_k camera centers
    target_z_enu: np.ndarray    # (K,) ENU camera Z target


@dataclass
class _PointEdge:
    """Per-segment direct DOM/DEM point unary edge.

    Pulls matched predicted model-space points toward their corresponding
    DOM/DEM ENU XYZ targets. Unlike PnP/H edges, this keeps the constraints at
    the point level instead of collapsing them into a single camera pose.
    """
    seg_idx: int
    pts_model: np.ndarray       # (K, 3) M_k points sampled at query pixels
    target_enu: np.ndarray      # (K, 3) DOM/DEM ENU targets


@dataclass
class _GroundRayEdge:
    """Per-segment image-center ground-ray unary edge.

    The DOM frame center is a ground point. For each camera center c and
    optical-axis direction d, intersect c + lambda d with target_enu.z and
    constrain that intersection XY to target_enu.xy.
    """
    seg_idx: int
    centers_model: np.ndarray   # (K, 3) M_k camera centers
    dirs_model: np.ndarray      # (K, 3) optical-axis directions in M_k
    target_enu: np.ndarray      # (K, 3) DOM frame-center ground targets


def _signed_safe_denominator_np(values: np.ndarray, min_abs: float) -> np.ndarray:
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
    R, t, s = _unpack_sim3(sim3)
    centers = s * (centers_model @ R.T) + t[None, :]
    dirs = dirs_model @ R.T
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-8)
    denom = _signed_safe_denominator_np(dirs[:, 2], min_abs_dir_z)
    lam = (target_enu[:, 2] - centers[:, 2]) / denom
    pred_xy = centers[:, :2] + lam[:, None] * dirs[:, :2]
    return pred_xy - target_enu[:, :2]


def _summarize_point_residuals(params_flat: np.ndarray,
                               point_edges: List[_PointEdge]) -> Dict[str, float]:
    """Unweighted point-edge residual statistics for diagnostics only."""
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
    """Unweighted ground-ray XY residual statistics for diagnostics only."""
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
    """Sum per-segment point-edge source counters stored by GeoV3."""
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
    """Collect overlap DOM unary edges, binary consistency edges, per-seg DEM-Z edges, adjacent scale edges."""
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
    """Build sparsity pattern of the Jacobian (M, 7N).

    Row layout MUST mirror ``_residual_fn`` exactly:
      [seg-0 anchor (3, if present)] + [DOM unary (3 each)] +
            [consist binary (3 each)] + [DEM-Z unary (1 per road pt)] +
            [camera-Z unary (1 per sampled camera)] +
            [point unary (3 per matched point)] +
                        [ground-ray unary (2 per ray)] +
      [scale binary (1 each)]
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
    """Pack all segment transforms into a flat 7N vector."""
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
    """Compute residuals for scipy.optimize.least_squares."""
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


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def refine_transforms_with_pgo(
    chain: GlobalChainState,
    origin_enu: Optional[np.ndarray] = None,
    cfg: Optional[PGOConfig] = None,
) -> Tuple[List[LocalModelTransform], Dict[str, float]]:
    """Run Plan-A PGO over a completed chain.

    Returns:
        new_transforms: list of refined LocalModelTransform, parallel to chain.summaries
        stats: solver diagnostics (initial/final cost, iterations, edge counts)
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

    x0 = _build_initial_params(chain)
    max_nfev = max(1, int(cfg.max_iter))
    n_dem_total = sum(int(len(e.dem_z_enu)) for e in dem_z_edges)
    n_camera_z_total = sum(int(len(e.target_z_enu)) for e in camera_z_edges)
    n_point_total = sum(int(e.pts_model.shape[0]) for e in point_edges)
    n_ground_ray_total = sum(int(e.centers_model.shape[0]) for e in ground_ray_edges)
    print(
        f"[PGO] start: segments={len(chain.summaries)} vars={len(x0)} "
        f"dom_edges={len(dom_edges)} consist_edges={len(consist_edges)} "
        f"dem_z_rows={n_dem_total} camera_z_rows={n_camera_z_total} "
        f"point_pts={n_point_total} ground_rays={n_ground_ray_total} "
        f"scale_edges={len(scale_edges)} "
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
    if cfg.verbose and jac_sparsity is not None:
        nnz = int(jac_sparsity.nnz)
        total = int(jac_sparsity.shape[0]) * int(jac_sparsity.shape[1])
        density = nnz / max(total, 1)
        print(f"[PGO] jac_sparsity shape={jac_sparsity.shape}  nnz={nnz}  "
              f"density={density:.2e}")

    result = least_squares(
        _residual_fn, x0,
        args=(model_pts, dom_edges, consist_edges, dem_z_edges, camera_z_edges,
              point_edges, ground_ray_edges, scale_edges, origin_enu, cfg),
        method="trf",
        jac_sparsity=jac_sparsity,
        max_nfev=max_nfev,
        verbose=2 if cfg.verbose else 0,
    )
    final_cost = float(0.5 * np.sum(result.fun * result.fun))
    point_residual_post = _summarize_point_residuals(result.x, point_edges)
    ground_ray_residual_post = _summarize_ground_ray_residuals(
        result.x, ground_ray_edges, cfg.ground_ray_min_abs_dir_z)

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
        s_post = [float(np.exp(result.x.reshape(-1, 7)[k, 6]))
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
        "iterations": int(result.nfev),
        "max_nfev": int(max_nfev),
        "point_residual_pre": point_residual_pre,
        "point_residual_post": point_residual_post,
        "ground_ray_residual_pre": ground_ray_residual_pre,
        "ground_ray_residual_post": ground_ray_residual_post,
        "point_edge_source_stats": point_edge_source_stats,
    }
    return _unpack_transforms(result.x, chain), stats


def _unpack_transforms(params_flat: np.ndarray,
                        chain: GlobalChainState) -> List[LocalModelTransform]:
    """Convert flat 7N params → list of refined LocalModelTransform."""
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
    """Produce the final (num_frames, 3) ENU trajectory using refined T_k's.

    Mirrors GeoV3Runner._dedupe_by_frame_index but takes external transforms.
    Later segments overwrite earlier ones on overlapping frame indices.
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
    """Configuration for in-segment PGO step."""
    enable: bool = False
    w_dom_xy: float = 5.0       # DOM unary XY weight
    w_dom_z: float = 2.0        # DOM unary Z weight
    w_consist: float = 8.0      # consist (XY) with prev overlap when caller supplies it
    w_consist_z: float = 16.0   # consist (Z) — separate, higher weight to suppress per-seg z drift
    w_delta_reg: float = 2.0    # ||delta_i|| L2 prior
    w_pose_rot: float = 50.0    # geodesic SO(3) prior on rotation (delta from omega_init)
    w_pose_trans: float = 2.0   # (t - t_init) pose prior
    yaw_only_rotation: bool = False
    use_point_edges: bool = False
    w_point_xy: float = 0.5
    w_point_z: float = 0.1
    use_ground_ray: bool = False
    w_ground_ray_xy: float = 2.0
    ground_ray_min_abs_dir_z: float = 0.05
    freeze_delta: bool = True   # if True, only optimize 6-DoF pose (no delta)
    max_iter: int = 20          # max LM/TRF iterations
    delta_clip: float = 5.0     # clamp final delta_i to [-clip, clip] meters
    verbose: bool = False


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
    """In-segment scipy LS refinement of (omega, t, delta).

    s is FROZEN at transform_init.s (Umeyama already optimal; otherwise LM
    oscillates). Variables = [omega(3), t(3), delta(3*N)] = 6 + 3N dims.

    Residuals:
      1. DOM unary per fi in dom_targets_enu:
            r = (T_k(p_i + delta_i) - target_i^ENU) * [w_xy, w_xy, w_z]
      2. Consist (seg_id >= 1, i in [0, overlap_len)):
            target_i = T_0(prev_M0_overlap[i])
            r = (T_k(p_i + delta_i) - target_i) * w_consist
        3. direct point unary when enabled:
            r = (T_k(x_j) - X_j^ENU) * [w_point_xy, w_point_xy, w_point_z]
        4. ground-ray unary when enabled:
            r = intersect_xy(T_k(c_i), R_k d_i, target_z) - target_xy
        5. delta L2 prior:  r = delta_i * w_delta_reg

    Returns: (refined_transform, delta(N,3), stats)
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
