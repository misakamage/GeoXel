"""
Trajectory chaining utilities for windowed inference.

Provides pose-encoding-to-camera-center conversion, simple window chaining,
and DOM-anchor-corrected chaining (Sim2 + linear interpolation).
"""

import numpy as np
import torch


def pose_enc_to_camera_centers(pose_encs):
    """
    Convert pose encodings [N, 9] to camera centers in world coordinates [N, 3].

    pose_enc: [tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w]
    The extrinsic [R|t] is camera-from-world, so camera center = -R^T @ t.
    """
    from streamvggt.utils.rotation import quat_to_mat

    T = pose_encs[:, :3]
    quat = pose_encs[:, 3:7]

    R = quat_to_mat(quat)  # [N, 3, 3]
    centers = -torch.bmm(R.transpose(1, 2), T.unsqueeze(-1)).squeeze(-1)  # [N, 3]
    return centers


def chain_window_centers(all_window_centers):
    """
    Chain camera centers from multiple windows into a single global trajectory.

    Each window's first frame is treated as the origin. We chain by accumulating:
    the last camera center of window k becomes the origin offset for window k+1.

    Args:
        all_window_centers: list of (W_i, 3) np arrays, camera centers per window.

    Returns:
        global_centers: (N, 3) concatenated
    """
    if not all_window_centers:
        return np.zeros((0, 3), dtype=np.float32)
    if len(all_window_centers) == 1:
        return np.asarray(all_window_centers[0], dtype=np.float32)

    global_centers = []
    previous_global_end = None
    for i, wc in enumerate(all_window_centers):
        wc = np.asarray(wc, dtype=np.float32)
        if len(wc) == 0:
            continue
        # Each independent inference window has its own origin at frame 0.
        # The next window must therefore be translated by the previous global
        # endpoint minus the current window origin, not by wc[-1] itself.
        offset = np.zeros(3, dtype=np.float32) if previous_global_end is None else previous_global_end - wc[0]
        wc_global = wc + offset
        global_centers.append(wc_global)
        previous_global_end = wc_global[-1]

    return np.concatenate(global_centers, axis=0)


def chain_window_centers_with_overlap(all_window_centers, overlap=0):
    """Chain independent windows from their shared-frame Sim(3) handoff.

    This is a vision-only operation: the first window defines the output
    coordinate frame and each later window is aligned only to overlapping model
    predictions.  No GT, DOM, DEM, or map observation is used.
    """
    windows = [np.asarray(w, dtype=np.float64) for w in all_window_centers if len(w)]
    if not windows:
        return np.zeros((0, 3), dtype=np.float32)
    result = windows[0].copy()
    overlap = max(0, int(overlap))
    for current in windows[1:]:
        n = min(overlap, len(current), len(result))
        if n >= 3:
            source, target = current[:n], result[-n:]
            mu_source, mu_target = source.mean(0), target.mean(0)
            X, Y = source - mu_source, target - mu_target
            U, singular, Vt = np.linalg.svd((X.T @ Y) / float(n))
            # A nearly straight flight path cannot determine a 3-D rotation
            # from the overlap.  Translation-only handoff is deterministic
            # and avoids an arbitrary SVD reflection in that case.
            if float(singular[1]) < 1e-6:
                aligned = current + (target[-1] - current[n - 1])
                result = np.concatenate([result, aligned[n:]], axis=0)
                continue
            D = np.eye(3)
            if np.linalg.det(Vt.T @ U.T) < 0.0:
                D[-1, -1] = -1.0
            R = Vt.T @ D @ U.T
            var_source = float(np.sum(X * X) / float(n))
            scale = float(np.sum(singular * np.diag(D)) / max(var_source, 1e-12))
            if not np.isfinite(scale) or scale <= 1e-6 or not np.isfinite(R).all():
                scale, R = 1.0, np.eye(3)
            translation = mu_target - scale * (R @ mu_source)
            aligned = scale * (current @ R.T) + translation
        else:
            aligned = current + (result[-1] - current[0])
        result = np.concatenate([result, aligned[n:] if n > 0 else aligned], axis=0)
    return result.astype(np.float32)


def chain_window_centers_with_dom_anchors(all_window_centers, dom_enu_anchors):
    """Fix chaining drift using DOM-derived absolute ENU positions.

    Strategy:
      1. Chain normally in model coordinates (same as baseline)
      2. Estimate 2D similarity (scale, rotation, translation) from anchor
         correspondences:  model_xy → s * R @ model_xy + t ≈ enu_xy
      3. Transform full trajectory to approximate ENU
      4. Compute residual drift at each anchor frame
      5. Linearly interpolate drift corrections across all frames

    This preserves intra-window trajectory shape while correcting the
    cumulative drift between windows.

    Returns: (N, 3) drift-corrected trajectory (in approximate ENU).
    """
    # Step 1: Normal chaining in model coordinates
    chained = chain_window_centers(all_window_centers)
    N = len(chained)

    # Map window index → frame index in the chained trajectory
    frame_offsets = []
    cum = 0
    for wc in all_window_centers:
        frame_offsets.append(cum)
        cum += len(wc)

    # Collect valid anchor correspondences
    anchor_frames = []
    anchor_enu = []
    for w, anc in enumerate(dom_enu_anchors):
        if anc is not None and w < len(frame_offsets):
            fi = frame_offsets[w]
            if fi < N:
                anchor_frames.append(fi)
                anchor_enu.append(anc)  # (2,) array [E, N]

    K = len(anchor_frames)
    if K < 2:
        print("[DOM-drift] <2 valid anchors, falling back to normal chaining")
        return chained

    anchor_frames = np.array(anchor_frames)
    anchor_enu = np.array(anchor_enu)        # (K, 2)
    anchor_model = chained[anchor_frames, :2]  # (K, 2)

    # Step 2: Estimate 2D similarity (Umeyama) from model → ENU at anchor points
    mu_m = anchor_model.mean(axis=0)
    mu_e = anchor_enu.mean(axis=0)
    dm = anchor_model - mu_m
    de = anchor_enu - mu_e

    sigma_m2 = np.sum(dm ** 2) / K
    H_ = dm.T @ de / K          # (2, 2)
    U, S, Vt = np.linalg.svd(H_)
    d = np.linalg.det(Vt.T @ U.T)
    D_mat = np.diag([1.0, np.sign(d)])
    R = Vt.T @ D_mat @ U.T
    s = np.sum(S * np.diag(D_mat)) / (sigma_m2 + 1e-12)
    t = mu_e - s * R @ mu_m

    angle_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    print(f"[DOM-drift] Sim2 from {K} anchors: scale={s:.2f}, "
          f"rotation={angle_deg:.1f}°, t=({t[0]:.1f}, {t[1]:.1f})")

    # Step 3: Compute residual drift at anchors (in ENU), then convert to model space
    approx_enu_xy = s * (chained[:, :2] @ R.T) + t   # (N, 2)
    approx_at_anchors = approx_enu_xy[anchor_frames]   # (K, 2)
    residuals_enu = anchor_enu - approx_at_anchors      # (K, 2)

    # Convert ENU residuals back to model space: delta_model = R^T @ delta_enu / s
    residuals_model = (residuals_enu @ R) / s           # (K, 2)

    res_norms_enu = [np.linalg.norm(residuals_enu[i]) for i in range(K)]
    print(f"[DOM-drift] Residuals at anchors (m): "
          + ", ".join(f"f{anchor_frames[i]}={res_norms_enu[i]:.2f}" for i in range(K)))

    # Step 4: Linearly interpolate model-space residuals across all frames
    interpolated = np.zeros((N, 2))
    for dim in range(2):
        interpolated[:, dim] = np.interp(
            np.arange(N), anchor_frames, residuals_model[:, dim])

    # Step 5: Correct trajectory in model space (Z untouched, Umeyama handles alignment)
    corrected = chained.copy()
    corrected[:, :2] += interpolated

    return corrected
