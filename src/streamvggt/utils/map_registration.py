"""External DOM/DEM map-registration pass for baseline artifacts.

The baseline model is run first and remains untouched.  This module consumes
its frame-indexed pointmaps, obtains DOM/DEM correspondences at a fixed stride,
fits a robust Sim(3) per registration segment, and evaluates directly in ENU.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SRC = os.path.join(_ROOT, "src")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from eval_japan_dom import load_dom_tiff, make_project_fn, make_inverse_project_fn
from streamvggt.utils.dom_matching import GeoElevationQuery, get_frame_dom_correspondences, load_roma
from streamvggt.utils.uavscene_pose import sampleinfo_c2w_rotation


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if len(src) < 4 or src.shape != dst.shape:
        raise ValueError("at least four paired 3D points are required")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    var = float((X * X).sum() / len(X))
    if var < 1e-12:
        raise np.linalg.LinAlgError("degenerate source points")
    U, sv, Vt = np.linalg.svd((Y.T @ X) / len(X))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float((sv * np.diag(D)).sum() / var)
    t = mu_d - s * (R @ mu_s)
    return R, t, s


def _fit_ransac(src: np.ndarray, dst: np.ndarray, thresh: float = 15.0,
                iters: int = 256, seed: int = 23) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, float]], np.ndarray, Dict[str, Any]]:
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    finite = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[finite], dst[finite]
    stats: Dict[str, Any] = {"num_points": int(len(src)), "accepted": False}
    if len(src) < 4:
        stats["reason"] = "too_few_points"
        return None, np.zeros(len(src), dtype=bool), stats
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(max(1, int(iters))):
        try:
            idx = rng.choice(len(src), 4, replace=False)
            fit = _umeyama(src[idx], dst[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        R, t, s = fit
        err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        mask = np.isfinite(err) & (err <= float(thresh))
        score = (int(mask.sum()), -float(np.median(err[mask])) if mask.any() else -np.inf)
        if best is None or score > best[0]:
            best = (score, mask)
    if best is None or int(best[1].sum()) < 4:
        stats["reason"] = "ransac_failed"
        return None, best[1] if best is not None else np.zeros(len(src), dtype=bool), stats
    mask = best[1]
    try:
        R, t, s = _umeyama(src[mask], dst[mask])
    except (ValueError, np.linalg.LinAlgError):
        stats["reason"] = "refit_failed"
        return None, mask, stats
    if not (np.isfinite(R).all() and np.isfinite(t).all() and np.isfinite(s) and float(s) > 0.0):
        stats.update({"reason": "invalid_sim3_scale", "scale": float(s) if np.isfinite(s) else float("nan")})
        return None, mask, stats
    err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    mask = np.isfinite(err) & (err <= float(thresh))
    stats.update({"accepted": bool(mask.sum() >= 4), "reason": "ok" if mask.sum() >= 4 else "refit_too_few_inliers",
                  "inliers": int(mask.sum()), "inlier_ratio": float(mask.mean()),
                  "median_xy_residual_m": float(np.median(np.linalg.norm(((s * (R @ src.T)).T + t - dst)[:, :2][mask], axis=1)) if mask.any() else float("nan")),
                  "median_residual_m": float(np.median(err[mask])) if mask.any() else float("nan"), "scale": float(s)})
    return (R, t, s) if stats["accepted"] else None, mask, stats


def _sample_map_points(pointmap: np.ndarray, pixels: np.ndarray, query_w: float, query_h: float) -> np.ndarray:
    pm = np.asarray(pointmap, dtype=np.float64)
    if pm.ndim == 4:
        pm = pm[0]
    if pm.ndim != 3 or pm.shape[-1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    h, w = pm.shape[:2]
    xy = np.asarray(pixels, dtype=np.float64).copy()
    # Pointmaps can be resized by a baseline; map original query pixels into
    # their actual raster coordinates before nearest-neighbour sampling.
    xy[:, 0] *= (w - 1) / max(float(query_w - 1), 1.0)
    xy[:, 1] *= (h - 1) / max(float(query_h - 1), 1.0)
    u = np.clip(np.rint(xy[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.rint(xy[:, 1]).astype(int), 0, h - 1)
    return pm[v, u, :3]


def _natural_key(value: str) -> List[Any]:
    """Sort image names in the same human/numeric order as the evaluators."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def _frame_records(image_paths: List[str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Resolve stable artifact and original frame indices for diagnostics.

    New artifacts may carry ``original_frame_indices`` directly.  Older
    artifacts are reconstructed from numeric AirZoo names or the natural
    ordering of the camera directory used by the UAVScenes evaluator.
    """
    explicit = metadata.get("original_frame_indices")
    explicit = list(explicit) if isinstance(explicit, list) else []
    by_path: Dict[str, int] = {}
    if not explicit or len(explicit) != len(image_paths):
        candidate_cache: Dict[Tuple[str, bool], List[str]] = {}
        for path in image_paths:
            parent = os.path.dirname(path)
            base = os.path.basename(path)
            # AirZoo has RGB/depth pairs in one directory; the evaluator uses
            # the RGB ``*_0`` files as its frame list.
            rgb_only = base.endswith("_0.png")
            cache_key = (parent, rgb_only)
            if cache_key not in candidate_cache:
                candidates = []
                for pattern in ("*.jpg", "*.jpeg", "*.png"):
                    candidates.extend(glob.glob(os.path.join(parent, pattern)))
                if rgb_only:
                    candidates = [p for p in candidates if os.path.basename(p).endswith("_0.png")]
                candidate_cache[cache_key] = sorted(
                    set(candidates), key=lambda p: _natural_key(os.path.basename(p)))
            candidates = candidate_cache[cache_key]
            try:
                by_path[os.path.abspath(path)] = candidates.index(path)
            except ValueError:
                by_path[os.path.abspath(path)] = -1

    records: List[Dict[str, Any]] = []
    for local_idx, path in enumerate(image_paths):
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        numeric_stem = stem[:-2] if stem.endswith("_0") else stem
        original_idx: Any = None
        if explicit and len(explicit) == len(image_paths):
            original_idx = explicit[local_idx]
        elif numeric_stem.isdigit():
            # AirZoo's filename is the original frame id.
            original_idx = int(numeric_stem)
        else:
            original_idx = by_path.get(os.path.abspath(path), -1)
        records.append({
            "local_frame_index": int(local_idx),
            "original_frame_index": int(original_idx) if original_idx is not None else None,
            "image_path": str(path),
            "image_name": base,
        })
    return records


def _frame_ref(records: List[Dict[str, Any]], local_idx: Optional[int]) -> Optional[Dict[str, Any]]:
    if local_idx is None or local_idx < 0 or local_idx >= len(records):
        return None
    item = records[int(local_idx)]
    return {
        "local_frame_index": int(item["local_frame_index"]),
        "original_frame_index": item.get("original_frame_index"),
        "image_name": item.get("image_name"),
    }


def _project_to_so3(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError("rotation contains non-finite values")
    U, _, Vt = np.linalg.svd(matrix)
    result = U @ Vt
    if np.linalg.det(result) < 0:
        U[:, -1] *= -1.0
        result = U @ Vt
    return result


def _resolve_first_frame_camera(metadata: Dict[str, Any], image_path: str) -> Tuple[np.ndarray, Dict[str, float]]:
    rotation = metadata.get("anchor_R_c2w_enu")
    intrinsics = metadata.get("camera_intrinsics")
    image_size = metadata.get("camera_image_size")

    seq_root = os.path.dirname(os.path.dirname(os.path.abspath(image_path)))
    sampleinfo_path = os.path.join(seq_root, "sampleinfos_interpolated.json")
    # Always prefer the authoritative UAVScenes record when it is available.
    # Artifacts exported before the c2w convention fix cached a transposed
    # anchor rotation, so trusting that cache would keep old forward passes
    # unusable for corrected map registration.
    if os.path.exists(sampleinfo_path):
        with open(sampleinfo_path) as sampleinfo_file:
            sampleinfos = json.load(sampleinfo_file)
        image_name = os.path.basename(image_path)
        first_info = next(
            (item for item in sampleinfos
             if item.get("OriginalImageName") == image_name),
            None,
        )
        if first_info is None:
            raise ValueError(f"first frame is missing from {sampleinfo_path}: {image_name}")
        rotation = sampleinfo_c2w_rotation(first_info)
        intrinsics = first_info["P3x3"]
        image_size = [first_info["Width"], first_info["Height"]]
    elif rotation is None or intrinsics is None:
        raise ValueError(
            "camera orientation/intrinsics are missing from both artifact "
            f"metadata and {sampleinfo_path}")

    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    if image_size is None:
        from PIL import Image
        with Image.open(image_path) as image:
            image_size = image.size
    width, height = map(int, image_size[:2])
    camera_fov = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "orig_w": width,
        "orig_h": height,
    }
    return _project_to_so3(np.asarray(rotation, dtype=np.float64)), camera_fov


def _estimate_model_ground_distance(
    first_pointmap: np.ndarray,
    first_center: np.ndarray,
    first_rotation: np.ndarray,
) -> float:
    points = np.asarray(first_pointmap, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(first_center, dtype=np.float64).reshape(3)
    forward = _project_to_so3(first_rotation)[:, 2]
    delta = points - center.reshape(1, 3)
    depth = delta @ forward
    finite = np.isfinite(delta).all(axis=1) & np.isfinite(depth) & (depth > 1e-5)
    depth = depth[finite]
    if len(depth) < 100:
        raise ValueError("too few finite positive-depth points for map scale bootstrap")
    lo, hi = np.percentile(depth, [10.0, 90.0])
    trimmed = depth[(depth >= lo) & (depth <= hi)]
    estimate = float(np.median(trimmed if len(trimmed) else depth))
    if not np.isfinite(estimate) or estimate <= 1e-5:
        raise ValueError("invalid model ground-distance estimate")
    return estimate


def _ground_z_at_enu(
    center_enu: np.ndarray,
    project_fn,
    geo_elev: Optional[GeoElevationQuery],
    dom_transform,
    fallback_agl: float,
) -> float:
    fallback = float(center_enu[2] - fallback_agl)
    if geo_elev is None:
        return fallback
    try:
        import torch
        xyz = torch.tensor([[float(center_enu[0]), float(center_enu[1]), 0.0]])
        uv = np.asarray(project_fn(xyz), dtype=np.float64)[0]
        _, _, ground_z = geo_elev.enu_3d_from_dom(
            float(uv[0]), float(uv[1]), dom_transform)
        if np.isfinite(ground_z):
            return float(ground_z)
    except Exception:
        pass
    return fallback


def register_artifact(artifact_dir: str, dom_dir: str, map_stride: int = 4,
                      segment_length: int = 32, overlap: int = 8,
                      max_corr: int = 512, device: str = "cuda",
                      crop_size_m: float = 250.0, thresh_m: float = 15.0,
                      freeze_after_failure: bool = True) -> Dict[str, Any]:
    meta_path = os.path.join(artifact_dir, "metadata.json")
    traj_path = os.path.join(artifact_dir, "trajectory.npz")
    pm_path = os.path.join(artifact_dir, "frame_pointmaps.npz")
    with open(meta_path) as f:
        metadata = json.load(f)
    image_paths = list(metadata.get("image_paths", []))
    if not image_paths or not os.path.exists(pm_path):
        raise FileNotFoundError("map-registration requires metadata.json and frame_pointmaps.npz")
    traj = np.load(traj_path)
    raw_centers = np.asarray(traj["pred_xyz"], dtype=np.float64)
    centers = raw_centers - raw_centers[0]
    rot_path = os.path.join(artifact_dir, "pred_rotations.npz")
    rotations = np.asarray(np.load(rot_path)["pred_R_c2w"] if os.path.exists(rot_path)
                           else np.tile(np.eye(3), (len(centers), 1, 1)), dtype=np.float64)
    pms = np.asarray(np.load(pm_path)["frame_pointmaps"], dtype=np.float32)
    if len(pms) != len(centers) or len(rotations) != len(centers) or len(image_paths) != len(centers):
        raise ValueError(
            "artifact frame counts disagree: "
            f"images={len(image_paths)} poses={len(centers)} rotations={len(rotations)} pointmaps={len(pms)}")
    # The trajectory is expressed relative to its first model pose below;
    # pointmaps must receive the identical translation before pixel sampling.
    pms = pms - raw_centers[0].reshape(1, 1, 1, 3).astype(np.float32)
    dom_np, dom_transform, dom_crs = load_dom_tiff(os.path.join(dom_dir, "dom.tif"))
    import torch
    dom_cpu = torch.from_numpy(dom_np).permute(2, 0, 1).contiguous()
    anchor = np.asarray(traj["gt_enu"][0], dtype=np.float64) if "gt_enu" in traj else np.zeros(3)
    anchor_meta = metadata.get("anchor_lonlatalt") or metadata.get("anchor")
    if anchor_meta is None:
        raise ValueError("artifact metadata must contain anchor_lonlatalt for DOM projection")
    lon0, lat0, alt0 = map(float, anchor_meta[:3])
    project_fn = make_project_fn(dom_transform, dom_crs, lon0, lat0, alt0)
    inv_project_fn = make_inverse_project_fn(dom_transform, dom_crs, lon0, lat0)
    geo_elev = None
    dem_path = os.path.join(dom_dir, "dem.tif")
    if os.path.exists(dem_path):
        geo_elev = GeoElevationQuery(dem_path, os.path.join(dom_dir, "building_mask.tif"), lon0, lat0, alt0,
                                     road_mask_path=os.path.join(dom_dir, "road_mask.tif"))
    anchor_R_c2w_enu, camera_fov = _resolve_first_frame_camera(metadata, image_paths[0])
    try:
        pred_R0 = _project_to_so3(rotations[0])
        initial_rotation_valid = True
    except (ValueError, np.linalg.LinAlgError):
        # A baseline may not export rotations for every frame.  Keep the
        # artifact auditable and let the first map query freeze registration,
        # rather than aborting the whole external pass.
        pred_R0 = np.eye(3, dtype=np.float64)
        initial_rotation_valid = False
    initial_R = anchor_R_c2w_enu @ pred_R0.T
    if geo_elev is not None:
        ground_alt = float(geo_elev.elevation_at_lonlat(lon0, lat0))
        anchor_agl = max(10.0, float(alt0) - ground_alt)
    else:
        anchor_agl = 100.0
    model_ground_distance = _estimate_model_ground_distance(pms[0], centers[0], pred_R0)
    initial_scale = float(anchor_agl / model_ground_distance)
    if not np.isfinite(initial_scale) or initial_scale <= 0.0:
        raise ValueError(
            "invalid bootstrap scale: "
            f"anchor_agl={anchor_agl!r}, model_ground_distance={model_ground_distance!r}")
    initial_t = anchor - initial_scale * (initial_R @ centers[0])
    initial_transform = (initial_R, initial_t, initial_scale)
    camera_fov["altitude"] = float(anchor_agl)
    roma = load_roma(device=device)
    diagnostics: List[Dict[str, Any]] = []
    frame_records = _frame_records(image_paths, metadata)
    frame_diagnostics: List[Dict[str, Any]] = []
    registered = centers.copy()
    previous = None
    map_active = True
    first_match_failure: Optional[Dict[str, Any]] = None
    first_registration_failure: Optional[Dict[str, Any]] = None
    first_fallback: Optional[Dict[str, Any]] = None
    last_successful_match: Optional[Dict[str, Any]] = None
    fallback_mask = np.zeros(len(centers), dtype=bool)
    total_match_attempts = 0
    total_match_successes = 0
    total_match_failures = 0
    for seg_start in range(0, len(centers), max(1, int(segment_length) - int(overlap))):
        seg_end = min(len(centers), seg_start + int(segment_length))
        segment_map_active_before = bool(map_active)
        indices = list(range(seg_start, seg_end, max(1, int(map_stride))))
        if seg_end - 1 not in indices:
            indices.append(seg_end - 1)
        src_chunks, dst_chunks = [], []
        segment_match_failure = None
        segment_match_attempts = 0
        segment_match_successes = 0
        segment_match_failures = 0
        if map_active:
            for fi in indices:
                frame_diag: Dict[str, Any] = {
                    **frame_records[fi],
                    "registration_segment_start": int(seg_start),
                    "registration_segment_end": int(seg_end),
                    "map_match_attempted": True,
                    "map_match_available": False,
                    "num_correspondences": 0,
                    "num_valid_point_pairs": 0,
                    "fallback_used": False,
                }
                segment_match_attempts += 1
                total_match_attempts += 1
                crop_transform = previous if previous is not None else initial_transform
                crop_R, crop_t, crop_s = crop_transform
                approx_center = (crop_s * (crop_R @ centers[fi])) + crop_t
                try:
                    pred_R_c2w = _project_to_so3(rotations[fi])
                except (ValueError, np.linalg.LinAlgError):
                    frame_diag["map_match_failure_reason"] = "invalid_pred_rotation"
                    segment_match_failure = frame_diag
                    segment_match_failures += 1
                    total_match_failures += 1
                    frame_diagnostics.append(frame_diag)
                    break
                crop_R_c2w = crop_R @ pred_R_c2w
                ground_z = _ground_z_at_enu(
                    approx_center, project_fn, geo_elev, dom_transform, anchor_agl)
                up_enu = crop_R_c2w @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
                query_yaw_deg = float(np.degrees(np.arctan2(up_enu[0], up_enu[1])))
                corr = get_frame_dom_correspondences(
                    image_paths[fi], dom_cpu,
                    approx_center,
                    project_fn, inv_project_fn, roma,
                    crop_size_m=float(crop_size_m), device=device, max_correspondences=int(max_corr),
                    geo_elev=geo_elev, dom_transform=dom_transform,
                    camera_fov=camera_fov,
                    pred_pose_crop={
                        "cam_center_enu": approx_center,
                        "R_c2w": crop_R_c2w,
                        "ground_z": ground_z,
                        "crop_scale": 1.3,
                        "min_half_size_m": 2.0,
                        "max_half_size_m": 600.0,
                        "max_center_shift_m": 2000.0,
                        "strict_footprint": True,
                        "max_hit_dist_m": 2500.0,
                        "disable_render": True,
                    },
                    query_yaw_deg=query_yaw_deg)
                if not corr:
                    frame_diag["map_match_failure_reason"] = "no_correspondences"
                    segment_match_failure = frame_diag
                    segment_match_failures += 1
                    total_match_failures += 1
                    frame_diagnostics.append(frame_diag)
                    break
                frame_diag["map_match_available"] = True
                frame_diag["num_correspondences"] = int(len(corr.get("frame_pixels", [])))
                pts = _sample_map_points(pms[fi], corr.get("frame_pixels"), corr.get("query_W", 1), corr.get("query_H", 1))
                enu = np.asarray(corr.get("enu_positions"), dtype=np.float64)
                n = min(len(pts), len(enu))
                if n > 0 and enu.ndim == 2 and enu.shape[1] >= 3:
                    valid = np.isfinite(pts[:n]).all(axis=1) & np.isfinite(enu[:n, :3]).all(axis=1)
                    pts_valid, enu_valid = pts[:n][valid], enu[:n, :3][valid]
                else:
                    pts_valid = np.empty((0, 3), dtype=np.float64)
                    enu_valid = np.empty((0, 3), dtype=np.float64)
                frame_diag["num_valid_point_pairs"] = int(len(pts_valid))
                if len(pts_valid) < 4:
                    frame_diag["map_match_failure_reason"] = "too_few_point_pairs"
                    segment_match_failure = frame_diag
                    segment_match_failures += 1
                    total_match_failures += 1
                    frame_diagnostics.append(frame_diag)
                    break
                src_chunks.append(pts_valid)
                dst_chunks.append(enu_valid)
                segment_match_successes += 1
                total_match_successes += 1
                last_successful_match = _frame_ref(frame_records, fi)
                frame_diagnostics.append(frame_diag)
        else:
            # Once map registration is lost, do not even query RoMa/DOM again.
            # This is deliberately one-way for a reproducible map-only protocol.
            for fi in indices:
                frame_diagnostics.append({
                    **frame_records[fi],
                    "registration_segment_start": int(seg_start),
                    "registration_segment_end": int(seg_end),
                    "map_match_attempted": False,
                    "map_match_available": False,
                    "num_correspondences": 0,
                    "num_valid_point_pairs": 0,
                    "fallback_used": True,
                    "fallback_reason": "map_registration_frozen_after_first_failure",
                })

        if segment_match_failure is not None:
            if first_match_failure is None:
                first_match_failure = {
                    **_frame_ref(frame_records, segment_match_failure["local_frame_index"]),
                    "reason": segment_match_failure.get("map_match_failure_reason", "map_match_failed"),
                }
            if freeze_after_failure:
                map_active = False
                fit, _, stats = None, np.empty(0, dtype=bool), {
                    "accepted": False,
                    "reason": "map_match_failed_freeze",
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                }
            else:
                fit, _, stats = (None, np.empty(0, dtype=bool), {
                    "accepted": False, "reason": "map_match_failed", "inliers": 0, "inlier_ratio": 0.0
                })
        elif not map_active:
            fit, _, stats = None, np.empty(0, dtype=bool), {
                "accepted": False,
                "reason": "map_registration_frozen_after_first_failure",
                "inliers": 0,
                "inlier_ratio": 0.0,
            }
        elif src_chunks:
            fit, _, stats = _fit_ransac(np.concatenate(src_chunks), np.concatenate(dst_chunks), thresh=thresh_m, seed=23 + seg_start)
        else:
            fit, stats = None, {"accepted": False, "reason": "no_correspondences", "inliers": 0, "inlier_ratio": 0.0}
        prior = previous if previous is not None else initial_transform
        prior_center = prior[2] * (prior[0] @ centers[seg_start]) + prior[1]
        if fit is not None:
            fit_center = fit[2] * (fit[0] @ centers[seg_start]) + fit[1]
            scale_ratio = float(fit[2] / max(prior[2], 1e-12))
            center_shift = float(np.linalg.norm((fit_center - prior_center)[:2]))
            stats.update({"prior_scale": float(prior[2]), "scale_ratio_to_prior": scale_ratio,
                          "center_shift_from_prior_m": center_shift})
            if not (np.isfinite(scale_ratio) and 0.25 <= scale_ratio <= 4.0):
                stats.update({"accepted": False, "reason": "implausible_scale_vs_map_prior"})
                fit = None
            elif center_shift > max(50.0, float(crop_size_m)):
                stats.update({"accepted": False, "reason": "implausible_center_shift_vs_map_prior"})
                fit = None
        if fit is None and segment_map_active_before and first_registration_failure is None:
            first_registration_failure = {
                **_frame_ref(frame_records, seg_start),
                "reason": stats.get("reason", "registration_failed"),
            }
            if freeze_after_failure:
                map_active = False
        source = "current_map_registration" if fit is not None else (
            "frozen_previous_transform" if previous is not None and not map_active else
            ("previous_transform" if previous is not None else "initial_map_prior"))
        if fit is None:
            fit = prior
        R, t, s = fit
        registered[seg_start:seg_end] = (s * (R @ centers[seg_start:seg_end].T)).T + t
        if source == "current_map_registration":
            previous = fit
        fallback_used = source != "current_map_registration"
        if fallback_used:
            fallback_mask[seg_start:seg_end] = True
            if first_fallback is None:
                first_fallback = {
                    **_frame_ref(frame_records, seg_start),
                    "reason": stats.get("reason", "map_registration_fallback"),
                }
        stats.update({"segment_start": seg_start, "segment_end": seg_end, "transform_source": source,
                      "map_registration_accepted": source == "current_map_registration", "scale": float(s),
                      "bootstrap_scale": initial_scale, "anchor_agl_m": anchor_agl,
                      "model_ground_distance": model_ground_distance,
                      "map_registration_active_before": segment_map_active_before,
                      "map_registration_active_after": bool(map_active),
                      "map_match_attempts": int(segment_match_attempts),
                      "map_match_successes": int(segment_match_successes),
                      "map_match_failures": int(segment_match_failures),
                      "first_match_failure_in_segment": _frame_ref(frame_records, segment_match_failure["local_frame_index"]) if segment_match_failure else None})
        diagnostics.append(stats)
    gt = np.asarray(traj["gt_enu"], dtype=np.float64)[:len(registered)]
    diff = registered[:len(gt)] - gt
    errors = np.linalg.norm(diff, axis=1) if len(gt) else np.empty(0)
    fallback_count = int(fallback_mask.sum())
    longest_fallback = 0
    run = 0
    for flag in fallback_mask:
        run = run + 1 if flag else 0
        longest_fallback = max(longest_fallback, run)
    summary = {
        "freeze_after_failure": bool(freeze_after_failure),
        "map_registration_frozen": bool(not map_active),
        "first_match_failure_frame": first_match_failure,
        "first_registration_failure_frame": first_registration_failure,
        "first_fallback_frame": first_fallback,
        "last_successful_match_frame": last_successful_match,
        "fallback_frame_count": fallback_count,
        "longest_consecutive_fallback": int(longest_fallback),
        "map_match_attempts": int(total_match_attempts),
        "map_match_successes": int(total_match_successes),
        "map_match_failures": int(total_match_failures),
        "initial_rotation_valid": bool(initial_rotation_valid),
    }
    return {"pred_xyz_registered": registered, "gt_enu": gt,
            "ate_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("nan"),
            "ate_xy_rmse": float(np.sqrt(np.mean(np.sum(diff[:, :2] ** 2, axis=1)))) if len(diff) else float("nan"),
            "diagnostics": diagnostics, "frame_diagnostics": frame_diagnostics,
            "map_registration_summary": summary, "n_frames": int(len(registered)),
            "protocol": "map_registration_direct_enu"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--dom_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map_stride", type=int, default=4)
    parser.add_argument("--segment_length", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--max_corr", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    recovery_group = parser.add_mutually_exclusive_group()
    recovery_group.add_argument("--freeze_after_failure", dest="freeze_after_failure",
                                action="store_true", default=True,
                                help="Freeze map registration after the first failed match/fit (default)")
    recovery_group.add_argument("--allow_map_recovery", dest="freeze_after_failure",
                                action="store_false",
                                help="Allow map queries after a failed segment (legacy diagnostic mode)")
    args = parser.parse_args()
    result = register_artifact(
        args.artifact_dir, args.dom_dir, args.map_stride, args.segment_length,
        args.overlap, args.max_corr, args.device,
        freeze_after_failure=bool(args.freeze_after_failure),
    )
    pred = np.asarray(result.pop("pred_xyz_registered"), dtype=np.float32)
    gt = np.asarray(result.get("gt_enu", np.empty((0, 3))), dtype=np.float32)
    frame_diagnostics = result.get("frame_diagnostics", [])
    original_indices = np.full(len(pred), -1, dtype=np.int64)
    for item in frame_diagnostics:
        local_idx = int(item.get("local_frame_index", -1))
        original_idx = item.get("original_frame_index")
        if 0 <= local_idx < len(original_indices) and original_idx is not None:
            original_indices[local_idx] = int(original_idx)
    np.savez_compressed(os.path.join(os.path.dirname(args.output), "trajectory_registered.npz"),
                        pred_xyz=pred, gt_enu=gt,
                        local_frame_index=np.arange(len(pred), dtype=np.int64),
                        original_frame_index=original_indices)
    with open(args.output, "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}, f, indent=2)


if __name__ == "__main__":
    main()
