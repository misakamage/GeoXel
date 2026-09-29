"""Post-PGO camera-only vertical scale correction.

The first camera height and the DEM provide the only absolute vertical anchor.
All later segment scales and offsets are solved jointly from shared overlap
frames. Dense points are deliberately left untouched because their final PGO
transforms have already aligned them to the DEM.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


_MIN_POSITIVE_AGL_M = 1e-6
_MAX_LOG_SCALE_DELTA = 4.0


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _transform_points(points: np.ndarray, transform: Any) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    rotation = _to_numpy(transform.R).astype(np.float64).reshape(3, 3)
    scale = float(_to_numpy(transform.s).reshape(()))
    translation = _to_numpy(transform.t).astype(np.float64).reshape(3)
    return scale * (points @ rotation.T) + translation


def build_vertical_scale_evidence(
    chain: Any,
    transforms: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Build post-PGO camera Z and translation-invariant visual AGL evidence."""
    evidence: List[Dict[str, Any]] = []
    summaries = list(getattr(chain, "summaries", []))
    for segment_index, (summary, transform) in enumerate(zip(summaries, transforms)):
        trajectory_local = np.asarray(summary.trajectory_local, dtype=np.float64)
        if trajectory_local.ndim != 2 or trajectory_local.shape[1] != 3:
            trajectory_local = np.zeros((0, 3), dtype=np.float64)
        camera_enu = (
            _transform_points(trajectory_local, transform)
            if len(trajectory_local)
            else np.zeros((0, 3), dtype=np.float64)
        )

        metadata = summary.metadata if isinstance(summary.metadata, dict) else {}
        ground_by_frame = metadata.get("visual_ground_points_model_by_frame") or {}
        dem_by_frame = metadata.get("visual_ground_dem_z_by_frame") or {}
        source_by_frame = metadata.get("visual_ground_source_by_frame") or {}
        visual_agl: Dict[int, float] = {}
        dem_ground_z: Dict[int, float] = {}
        sources: Dict[int, str] = {}
        start_frame = int(summary.start_frame)

        for local_key, ground_model_raw in dict(ground_by_frame).items():
            try:
                local_index = int(local_key)
            except (TypeError, ValueError):
                continue
            if local_index < 0 or local_index >= len(camera_enu):
                continue
            ground_model = np.asarray(ground_model_raw, dtype=np.float64)
            if ground_model.ndim != 2 or ground_model.shape[1] != 3:
                continue
            finite_model = np.isfinite(ground_model).all(axis=1)
            if not np.any(finite_model):
                continue
            ground_enu = _transform_points(ground_model[finite_model], transform)
            finite_ground_z = np.isfinite(ground_enu[:, 2])
            if not np.any(finite_ground_z):
                continue

            global_index = start_frame + local_index
            height = float(
                camera_enu[local_index, 2]
                - np.median(ground_enu[finite_ground_z, 2])
            )
            if np.isfinite(height):
                visual_agl[global_index] = height

            dem_raw = dem_by_frame.get(local_key, dem_by_frame.get(local_index))
            if dem_raw is not None:
                dem_values = np.asarray(dem_raw, dtype=np.float64).reshape(-1)
                dem_values = dem_values[np.isfinite(dem_values)]
                if len(dem_values):
                    dem_ground_z[global_index] = float(np.median(dem_values))
            sources[global_index] = str(
                source_by_frame.get(local_key, source_by_frame.get(local_index, "unknown"))
            )

        evidence.append({
            "segment_index": int(segment_index),
            "start_frame": start_frame,
            "camera_enu": camera_enu,
            "visual_agl": visual_agl,
            "dem_ground_z": dem_ground_z,
            "sources": sources,
        })
    return evidence


def _camera_z_at(evidence: Dict[str, Any], global_index: int) -> float | None:
    camera = np.asarray(evidence.get("camera_enu"), dtype=np.float64)
    local_index = int(global_index) - int(evidence.get("start_frame", 0))
    if camera.ndim != 2 or camera.shape[1] < 3 or not (0 <= local_index < len(camera)):
        return None
    value = float(camera[local_index, 2])
    return value if np.isfinite(value) else None


def _build_overlap_observations(
    evidence: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    frame_to_segments: DefaultDict[int, List[int]] = defaultdict(list)
    for segment_index, item in enumerate(evidence):
        camera = np.asarray(item.get("camera_enu"), dtype=np.float64)
        start_frame = int(item.get("start_frame", 0))
        if camera.ndim != 2:
            continue
        for local_index in range(len(camera)):
            frame_to_segments[start_frame + local_index].append(segment_index)

    pair_frames: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    for global_index, segment_indices in frame_to_segments.items():
        for left, right in combinations(sorted(set(segment_indices)), 2):
            pair_frames[(left, right)].append(global_index)

    scale_observations: List[Dict[str, Any]] = []
    pose_observations: List[Dict[str, Any]] = []
    for (left, right), shared_frames in sorted(pair_frames.items()):
        log_ratios: List[float] = []
        scale_frames: List[int] = []
        pose_rows: List[Tuple[int, float, float]] = []
        left_agl = evidence[left].get("visual_agl", {})
        right_agl = evidence[right].get("visual_agl", {})
        for global_index in sorted(shared_frames):
            z_left = _camera_z_at(evidence[left], global_index)
            z_right = _camera_z_at(evidence[right], global_index)
            if z_left is not None and z_right is not None:
                pose_rows.append((int(global_index), z_left, z_right))

            h_left = left_agl.get(global_index)
            h_right = right_agl.get(global_index)
            if h_left is None or h_right is None:
                continue
            h_left = float(h_left)
            h_right = float(h_right)
            if not (
                np.isfinite(h_left)
                and np.isfinite(h_right)
                and h_left > _MIN_POSITIVE_AGL_M
                and h_right > _MIN_POSITIVE_AGL_M
            ):
                continue
            # alpha_right / alpha_left = h_left / h_right.
            log_ratios.append(float(np.log(h_left / h_right)))
            scale_frames.append(int(global_index))

        if log_ratios:
            ratio_array = np.asarray(log_ratios, dtype=np.float64)
            ratio_median = float(np.median(ratio_array))
            scale_observations.append({
                "left": int(left),
                "right": int(right),
                "log_ratio": ratio_median,
                "num_frames": int(len(ratio_array)),
                "mad": float(np.median(np.abs(ratio_array - ratio_median))),
                "frames": scale_frames,
            })
        if pose_rows:
            pose_observations.append({
                "left": int(left),
                "right": int(right),
                "rows": pose_rows,
            })
    return scale_observations, pose_observations


def _residual_summary(values: np.ndarray) -> Dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "rmse": 0.0, "median_abs": 0.0, "max_abs": 0.0}
    return {
        "count": int(len(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "median_abs": float(np.median(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))),
    }


def estimate_vertical_affines(
    evidence: Sequence[Dict[str, Any]],
    first_camera_z: float,
    *,
    huber_delta: float = 0.25,
    log_smooth_weight: float = 0.25,
    log_reference_weight: float = 0.25,
    pose_weight: float = 1.0,
    max_nfev: int = 1000,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """Jointly estimate ``z_corrected = alpha_k * z + beta_k`` for all segments.

    Segment zero is fixed by the first-frame GT camera Z and matched DEM AGL.
    Later segments use robust overlap AGL ratios and overlap camera continuity.
    A weak reference-scale prior prevents a single bad ratio from producing a
    permanent multiplicative step in every following segment.
    """
    if not evidence:
        return [], {"applied": False, "reason": "no_segments"}

    first = evidence[0]
    first_global = int(first.get("start_frame", 0))
    first_agl = first.get("visual_agl", {}).get(first_global)
    first_dem_z = first.get("dem_ground_z", {}).get(first_global)
    first_camera = np.asarray(first.get("camera_enu"), dtype=np.float64)
    if (
        first_camera.ndim != 2
        or first_camera.shape[1] < 3
        or len(first_camera) == 0
        or first_agl is None
        or first_dem_z is None
    ):
        return [], {
            "applied": False,
            "reason": "missing_first_frame_visual_ground_or_dem",
            "first_frame": first_global,
        }

    first_camera_z = float(first_camera_z)
    first_agl = float(first_agl)
    first_dem_z = float(first_dem_z)
    gt_agl = first_camera_z - first_dem_z
    if not (
        np.isfinite(first_camera_z)
        and np.isfinite(gt_agl)
        and np.isfinite(first_agl)
        and gt_agl > 0.0
        and first_agl > _MIN_POSITIVE_AGL_M
    ):
        return [], {
            "applied": False,
            "reason": "invalid_first_frame_agl",
            "gt_agl_m": float(gt_agl),
            "visual_agl_m": float(first_agl),
        }

    num_segments = len(evidence)
    alpha0 = float(gt_agl / first_agl)
    a0 = float(np.log(alpha0))
    beta0 = float(first_camera_z - alpha0 * first_camera[0, 2])
    scale_observations, pose_observations = _build_overlap_observations(evidence)

    if num_segments == 1:
        affines = [(alpha0, beta0)]
        return affines, {
            "applied": True,
            "solver": "fixed_first_segment_only",
            "num_segments": 1,
            "first_frame": first_global,
            "first_gt_agl_m": float(gt_agl),
            "first_visual_agl_m": float(first_agl),
            "first_dem_ground_z_m": float(first_dem_z),
            "segments": [{
                "segment_index": 0,
                "alpha": alpha0,
                "beta_m": beta0,
                "alpha_source": "first_frame_gt_dem_agl",
                "overlap_agl_frames": 0,
                "overlap_pose_frames": 0,
            }],
        }

    huber_delta = max(float(huber_delta), 1e-6)
    log_smooth_weight = max(float(log_smooth_weight), 0.0)
    log_reference_weight = max(float(log_reference_weight), 0.0)
    pose_weight = max(float(pose_weight), 0.0)

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = np.empty(num_segments, dtype=np.float64)
        beta = np.empty(num_segments, dtype=np.float64)
        a[0], beta[0] = a0, beta0
        a[1:] = x[:num_segments - 1]
        beta[1:] = x[num_segments - 1:]
        return a, beta

    a_init = np.full(num_segments, a0, dtype=np.float64)
    beta_init = np.full(num_segments, beta0, dtype=np.float64)
    for segment_index in range(1, num_segments):
        previous = evidence[segment_index - 1]
        current = evidence[segment_index]
        previous_start = int(previous.get("start_frame", 0))
        current_start = int(current.get("start_frame", 0))
        previous_camera = np.asarray(previous.get("camera_enu"), dtype=np.float64)
        current_camera = np.asarray(current.get("camera_enu"), dtype=np.float64)
        shared = sorted(
            set(range(previous_start, previous_start + len(previous_camera)))
            .intersection(range(current_start, current_start + len(current_camera)))
        )
        candidates = []
        for global_index in shared:
            z_previous = _camera_z_at(previous, global_index)
            z_current = _camera_z_at(current, global_index)
            if z_previous is not None and z_current is not None:
                candidates.append(
                    np.exp(a_init[segment_index - 1]) * z_previous
                    + beta_init[segment_index - 1]
                    - np.exp(a_init[segment_index]) * z_current
                )
        if candidates:
            beta_init[segment_index] = float(np.median(candidates))
        elif len(previous_camera) and len(current_camera):
            previous_last = (
                np.exp(a_init[segment_index - 1]) * float(previous_camera[-1, 2])
                + beta_init[segment_index - 1]
            )
            beta_init[segment_index] = float(
                previous_last - np.exp(a_init[segment_index]) * float(current_camera[0, 2])
            )
        else:
            beta_init[segment_index] = beta_init[segment_index - 1]

    x0 = np.concatenate([a_init[1:], beta_init[1:]])

    def residuals(x: np.ndarray) -> np.ndarray:
        a, beta = unpack(x)
        parts: List[float] = []
        for observation in scale_observations:
            left = observation["left"]
            right = observation["right"]
            parts.append(float(a[right] - a[left] - observation["log_ratio"]))
        if pose_weight > 0.0:
            for observation in pose_observations:
                left = observation["left"]
                right = observation["right"]
                rows = observation["rows"]
                row_weight = np.sqrt(pose_weight / max(len(rows), 1))
                alpha_left = np.exp(a[left])
                alpha_right = np.exp(a[right])
                for _, z_left, z_right in rows:
                    parts.append(float(row_weight * (
                        alpha_right * z_right + beta[right]
                        - alpha_left * z_left - beta[left]
                    )))
        if log_smooth_weight > 0.0:
            weight = np.sqrt(log_smooth_weight)
            parts.extend(float(weight * (a[k] - a[k - 1])) for k in range(1, num_segments))
        if log_reference_weight > 0.0:
            weight = np.sqrt(log_reference_weight)
            parts.extend(float(weight * (a[k] - a0)) for k in range(1, num_segments))
        return np.asarray(parts, dtype=np.float64)

    lower = np.full_like(x0, -np.inf)
    upper = np.full_like(x0, np.inf)
    lower[:num_segments - 1] = a0 - _MAX_LOG_SCALE_DELTA
    upper[:num_segments - 1] = a0 + _MAX_LOG_SCALE_DELTA
    initial_residual = residuals(x0)
    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        method="trf",
        loss="huber",
        f_scale=huber_delta,
        max_nfev=max(int(max_nfev), 1),
    )
    a_final, beta_final = unpack(result.x)
    alpha_final = np.exp(a_final)
    affines = [
        (float(alpha_final[k]), float(beta_final[k]))
        for k in range(num_segments)
    ]

    agl_counts = [0] * num_segments
    pose_counts = [0] * num_segments
    for observation in scale_observations:
        agl_counts[observation["left"]] += int(observation["num_frames"])
        agl_counts[observation["right"]] += int(observation["num_frames"])
    for observation in pose_observations:
        count = int(len(observation["rows"]))
        pose_counts[observation["left"]] += count
        pose_counts[observation["right"]] += count

    segment_stats = []
    for segment_index, (alpha, beta) in enumerate(affines):
        segment_stats.append({
            "segment_index": int(segment_index),
            "alpha": float(alpha),
            "beta_m": float(beta),
            "log_alpha": float(a_final[segment_index]),
            "alpha_source": (
                "first_frame_gt_dem_agl" if segment_index == 0
                else "global_robust_overlap_solve"
            ),
            "overlap_agl_frames": int(agl_counts[segment_index]),
            "overlap_pose_frames": int(pose_counts[segment_index]),
        })

    return affines, {
        "applied": True,
        "solver": "global_robust_log_scale_offset_least_squares",
        "loss": "huber",
        "huber_delta": float(huber_delta),
        "log_smooth_weight": float(log_smooth_weight),
        "log_reference_weight": float(log_reference_weight),
        "pose_weight": float(pose_weight),
        "num_segments": int(num_segments),
        "num_scale_pair_edges": int(len(scale_observations)),
        "num_scale_overlap_frames": int(sum(o["num_frames"] for o in scale_observations)),
        "num_pose_pair_edges": int(len(pose_observations)),
        "num_pose_overlap_frames": int(sum(len(o["rows"]) for o in pose_observations)),
        "first_frame": int(first_global),
        "first_gt_agl_m": float(gt_agl),
        "first_visual_agl_m": float(first_agl),
        "first_dem_ground_z_m": float(first_dem_z),
        "solve_success": bool(result.success),
        "solve_status": int(result.status),
        "solve_message": str(result.message),
        "solve_cost": float(result.cost),
        "solve_optimality": float(result.optimality),
        "solve_nfev": int(result.nfev),
        "initial_residual": _residual_summary(initial_residual),
        "final_residual": _residual_summary(residuals(result.x)),
        "scale_observations": scale_observations,
        "segments": segment_stats,
    }


def apply_vertical_scale_propagation(
    chain: Any,
    transforms: Sequence[Any],
    trajectory: np.ndarray,
    first_camera_z: float,
    *,
    huber_delta: float = 0.25,
    log_smooth_weight: float = 0.25,
    log_reference_weight: float = 0.25,
    pose_weight: float = 1.0,
    max_nfev: int = 1000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Correct only camera Z; preserve XY and all DEM-aligned point arrays."""
    evidence = build_vertical_scale_evidence(chain, transforms)
    affines, stats = estimate_vertical_affines(
        evidence,
        first_camera_z=first_camera_z,
        huber_delta=huber_delta,
        log_smooth_weight=log_smooth_weight,
        log_reference_weight=log_reference_weight,
        pose_weight=pose_weight,
        max_nfev=max_nfev,
    )
    original = np.asarray(trajectory)
    if not affines:
        return np.asarray(original, dtype=np.float32).copy(), stats

    corrected = np.asarray(original, dtype=np.float64).copy()
    summaries = list(getattr(chain, "summaries", []))
    diagnostics = list(getattr(chain, "diagnostics", []))
    for segment_index, (summary, segment_evidence, affine) in enumerate(
        zip(summaries, evidence, affines)
    ):
        alpha, beta = affine
        start_frame = int(summary.start_frame)
        camera = np.asarray(segment_evidence["camera_enu"], dtype=np.float64)
        # Later segments overwrite overlap frames, matching assemble_trajectory.
        for local_index, camera_z in enumerate(camera[:, 2]):
            global_index = start_frame + local_index
            if 0 <= global_index < len(corrected):
                corrected[global_index, 2] = alpha * float(camera_z) + beta

        correction = stats["segments"][segment_index]
        metadata = summary.metadata if isinstance(summary.metadata, dict) else {}
        metadata["vertical_scale_correction"] = correction
        if segment_index >= len(diagnostics):
            continue
        wrapped = diagnostics[segment_index]
        diag = wrapped.get("diag") if isinstance(wrapped, dict) else None
        if not isinstance(diag, dict):
            diag = wrapped if isinstance(wrapped, dict) else {}
        diag["vertical_scale_correction"] = correction

    alpha_values = np.asarray([alpha for alpha, _ in affines], dtype=np.float64)
    stats["alpha_min"] = float(np.min(alpha_values))
    stats["alpha_max"] = float(np.max(alpha_values))
    stats["alpha_last"] = float(alpha_values[-1])
    stats["point_arrays_corrected"] = 0
    stats["dense_points_policy"] = "preserve_dem_aligned_geometry"
    stats["camera_components_corrected"] = ["z"]
    stats["uses_gt_after_first_frame"] = False
    stats["xy_unchanged"] = bool(np.array_equal(corrected[:, :2], original[:, :2]))
    return corrected.astype(np.float32), stats
