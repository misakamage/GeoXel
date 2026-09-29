"""
DOM-RANSAC anchor matching: match query frames to a Digital Orthophoto Map
(DOM) using RoMa dense matching + RANSAC homography to obtain absolute ENU
positions.

Extended with DEM elevation queries, building mask filtering, and PnP-based
6-DoF pose estimation (migrated from LoGeR backward_ttt.py).
"""

import os
import numpy as np
import torch


def _seed_opencv_ransac(cv2, operation: int) -> None:
    """Make each OpenCV RANSAC solve independent of prior call order."""
    cv2.setRNGSeed(0x475600 + int(operation))


# ======================================================================
#  RoMa outdoor loader
# ======================================================================


def load_roma(device="cuda", variant="full", **kwargs):
    """Load RoMa outdoor model.

    `variant` is accepted for forward-compatibility with caller sites
    (currently only "full" outdoor model is wired up).
    """
    from romatch import roma_outdoor
    return roma_outdoor(device=device, use_custom_corr=False)


# ======================================================================
#  WGS84 coordinate helpers (from LoGeR)
# ======================================================================
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2


def _geodetic_to_ecef(lon, lat, alt):
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    N = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat_r) ** 2)
    x = (N + alt) * np.cos(lat_r) * np.cos(lon_r)
    y = (N + alt) * np.cos(lat_r) * np.sin(lon_r)
    z = (N * (1 - _WGS84_E2) + alt) * np.sin(lat_r)
    return x, y, z


def geodetic_to_enu(lon, lat, alt, lon0, lat0, alt0):
    x, y, z = _geodetic_to_ecef(lon, lat, alt)
    x0, y0, z0 = _geodetic_to_ecef(lon0, lat0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    sl, cl = np.sin(np.radians(lat0)), np.cos(np.radians(lat0))
    sn, cn = np.sin(np.radians(lon0)), np.cos(np.radians(lon0))
    e = -sn * dx + cn * dy
    n = -sl * cn * dx - sl * sn * dy + cl * dz
    u = cl * cn * dx + cl * sn * dy + sl * dz
    return e, n, u


# ======================================================================
#  DEM + Building Mask helpers (from LoGeR)
# ======================================================================
class GeoElevationQuery:
    """Query DEM elevation, building mask and road safe-zone for DOM pixel coordinates."""

    def __init__(self, dem_tif_path, building_mask_path, lon0, lat0, alt0,
                 road_mask_path=None):
        import rasterio
        self.lon0, self.lat0, self.alt0 = lon0, lat0, alt0

        self.dem_src = rasterio.open(dem_tif_path)
        self.dem_data = self.dem_src.read(1)

        if building_mask_path is not None and os.path.exists(building_mask_path):
            self.bld_src = rasterio.open(building_mask_path)
            self.bld_data = self.bld_src.read(1)
            print(f"  [GeoElev] Building mask: {self.bld_src.width}x{self.bld_src.height}, "
                  f"building={(self.bld_data > 0).sum()/self.bld_data.size*100:.1f}%")
        else:
            self.bld_src = None
            self.bld_data = None

        # road_mask: 255 = road safe-zone (ground, no occlusion, good for DOM matching)
        if road_mask_path is not None and os.path.exists(road_mask_path):
            self.road_src = rasterio.open(road_mask_path)
            self.road_data = self.road_src.read(1)
            print(f"  [GeoElev] Road mask: {self.road_src.width}x{self.road_src.height}, "
                  f"road_zone={(self.road_data > 0).sum()/self.road_data.size*100:.1f}%")
        else:
            self.road_src = None
            self.road_data = None

        print(f"  [GeoElev] DEM: {self.dem_src.width}x{self.dem_src.height}, "
              f"elev=[{self.dem_data.min():.1f}, {self.dem_data.max():.1f}]m")

        # Pre-build a projected-CRS → WGS84 transformer so that enu_3d_from_dom
        # correctly converts DOM/DEM pixel coordinates (which are in a projected
        # CRS such as UTM, NOT in lon/lat degrees) before calling geodetic_to_enu.
        # Without this, rasterio.transform.xy returns e.g. (500000, 3900000) which
        # geodetic_to_enu treats as degrees, producing ECEF / ENU errors of ~10^7 m.
        self._crs_to_wgs84 = None
        try:
            import pyproj
            _dem_crs = self.dem_src.crs
            if _dem_crs is not None and not _dem_crs.is_geographic:
                _wgs84 = pyproj.CRS("EPSG:4326")
                self._crs_to_wgs84 = pyproj.Transformer.from_crs(
                    _dem_crs, _wgs84, always_xy=True)
                print(f"  [GeoElev] CRS={_dem_crs.to_epsg()}, built projected→WGS84 transformer")
            else:
                print(f"  [GeoElev] CRS is already geographic, no CRS conversion needed")
        except Exception as _e:
            print(f"  [GeoElev] WARN: could not build CRS transformer: {_e}")

    def elevation_at_lonlat(self, lon, lat):
        import rasterio
        row, col = rasterio.transform.rowcol(self.dem_src.transform, lon, lat)
        row = max(0, min(row, self.dem_data.shape[0] - 1))
        col = max(0, min(col, self.dem_data.shape[1] - 1))
        return float(self.dem_data[row, col])

    def is_building_at_dom(self, dom_col, dom_row):
        if self.bld_data is None:
            return False
        r = int(round(dom_row))
        c = int(round(dom_col))
        if 0 <= r < self.bld_data.shape[0] and 0 <= c < self.bld_data.shape[1]:
            return self.bld_data[r, c] > 0
        return False

    def is_road_at_dom(self, dom_col, dom_row):
        """Return True when the DOM pixel falls inside the road safe-zone."""
        if self.road_data is None:
            return False
        r = int(round(dom_row))
        c = int(round(dom_col))
        if 0 <= r < self.road_data.shape[0] and 0 <= c < self.road_data.shape[1]:
            return self.road_data[r, c] > 0
        return False

    def enu_3d_from_dom(self, dom_col, dom_row, dom_transform):
        import rasterio
        # rasterio.transform.xy returns coordinates in the DOM's CRS, which is
        # typically a projected CRS (e.g. UTM) — NOT lon/lat in degrees.
        crs_x, crs_y = rasterio.transform.xy(
            dom_transform, int(round(dom_row)), int(round(dom_col)))
        # DEM elevation query works in the same projected CRS space.
        elev = self.elevation_at_lonlat(crs_x, crs_y)
        # Convert projected CRS → WGS84 lon/lat (degrees) before geodetic_to_enu.
        if self._crs_to_wgs84 is not None:
            lon, lat = self._crs_to_wgs84.transform(crs_x, crs_y)
        else:
            lon, lat = crs_x, crs_y  # already geographic (degrees)
        e, n, u = geodetic_to_enu(lon, lat, elev, self.lon0, self.lat0, self.alt0)
        return float(e), float(n), float(u)

    def close(self):
        self.dem_src.close()
        if self.bld_src is not None:
            self.bld_src.close()
        if self.road_src is not None:
            self.road_src.close()


# ======================================================================
#  PnP solver (from LoGeR)
# ======================================================================
def solve_pnp_frame(pts_2d, pts_3d_enu, fx, fy, cx, cy,
                    method=None, reproj_thresh=8.0):
    """Solve PnP: 2D pixel <-> 3D ENU correspondences -> 6-DoF pose.

    Returns:
        R_c2w: (3,3) camera-to-world rotation, or None
        t_world: (3,) camera position in ENU, or None
        n_inliers: int
    """
    import cv2
    if len(pts_2d) < 6:
        return None, None, 0

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    if method is None:
        method = cv2.SOLVEPNP_ITERATIVE

    _seed_opencv_ransac(cv2, 1)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d_enu.astype(np.float64),
        pts_2d.astype(np.float64),
        camera_matrix,
        distCoeffs=None,
        flags=method,
        reprojectionError=reproj_thresh,
        iterationsCount=2000,
        confidence=0.999,
    )

    if not success or inliers is None or len(inliers) < 4:
        return None, None, 0 if inliers is None else len(inliers)

    n_inl = len(inliers)

    pts_3d_inl = pts_3d_enu[inliers.ravel()]
    pts_2d_inl = pts_2d[inliers.ravel()]
    rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
        pts_3d_inl.astype(np.float64),
        pts_2d_inl.astype(np.float64),
        camera_matrix, None, rvec, tvec,
    )

    R_cv, _ = cv2.Rodrigues(rvec_ref)
    t_world = -R_cv.T @ tvec_ref.ravel()
    R_c2w = R_cv.T

    return R_c2w, t_world, n_inl


def _project_points_with_pose(pts_3d_enu, R_c2w, cam_center_enu,
                              fx, fy, cx, cy):
    pts = np.asarray(pts_3d_enu, dtype=np.float64)
    R = np.asarray(R_c2w, dtype=np.float64)
    C = np.asarray(cam_center_enu, dtype=np.float64).reshape(3)
    pts_cam = (pts - C.reshape(1, 3)) @ R
    z = pts_cam[:, 2]
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(z) & (z > 1e-6)
    uv[valid, 0] = fx * pts_cam[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * pts_cam[valid, 1] / z[valid] + cy
    return uv, z


def solve_pnp_fixed_rotation_center(
    pts_2d,
    pts_3d_enu,
    fx,
    fy,
    cx,
    cy,
    R_c2w_prior,
    center_prior_enu,
    prior_weight=25.0,
    huber_m=25.0,
    max_iters=5,
):
    """Estimate camera center with the render pose rotation fixed.

    The pose crop/render already provides the current model pose prior.  For
    oblique render crops we therefore solve only for a local camera-center
    correction from the matched query pixels and DOM/DEM 3D points, instead of
    re-running an unconstrained 6-DoF planar PnP.
    """
    pts_2d = np.asarray(pts_2d, dtype=np.float64)
    pts_3d = np.asarray(pts_3d_enu, dtype=np.float64)
    R = np.asarray(R_c2w_prior, dtype=np.float64).reshape(3, 3)
    C0 = np.asarray(center_prior_enu, dtype=np.float64).reshape(3)
    if len(pts_2d) < 6 or pts_3d.ndim != 2 or pts_3d.shape[1] < 3:
        return None, None, 0, {"reason": "too_few_points"}

    K_inv = np.linalg.inv(np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64))
    pix_h = np.concatenate(
        [pts_2d[:, :2], np.ones((len(pts_2d), 1), dtype=np.float64)], axis=1)
    rays_cam = pix_h @ K_inv.T
    rays_world = rays_cam @ R.T
    ray_norm = np.linalg.norm(rays_world, axis=1)
    valid = (np.isfinite(ray_norm) & (ray_norm > 1e-9)
             & np.all(np.isfinite(pts_3d[:, :3]), axis=1))
    if int(valid.sum()) < 6:
        return None, None, 0, {"reason": "too_few_valid_rays"}

    X = pts_3d[valid, :3]
    d = rays_world[valid] / ray_norm[valid, None]
    I3 = np.eye(3, dtype=np.float64)
    weights = np.ones(len(X), dtype=np.float64)
    C = C0.copy()
    prior_w = max(float(prior_weight), 1e-6)
    huber = max(float(huber_m), 1e-6)

    for _ in range(max(int(max_iters), 1)):
        H = prior_w * I3
        b = prior_w * C0
        for wi, Xi, di in zip(weights, X, d):
            P = I3 - np.outer(di, di)
            H += float(wi) * P
            b += float(wi) * (P @ Xi)
        try:
            C = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            C = np.linalg.lstsq(H, b, rcond=None)[0]
        diff = C.reshape(1, 3) - X
        along = np.sum(diff * d, axis=1, keepdims=True)
        perp = diff - along * d
        residual_m = np.linalg.norm(perp, axis=1)
        weights = np.minimum(1.0, huber / np.maximum(residual_m, 1e-6))

    uv, depth = _project_points_with_pose(X, R, C, fx, fy, cx, cy)
    err_px = np.linalg.norm(uv - pts_2d[valid, :2], axis=1)
    depth_valid = np.isfinite(depth) & (depth > 1e-6)
    err_valid = np.isfinite(err_px) & depth_valid
    n_good = int((err_valid & (err_px < 25.0)).sum())
    if n_good < 4:
        n_good = int(err_valid.sum())

    delta = C - C0
    diag = {
        "method": "fixed_rotation_center",
        "n_valid": int(valid.sum()),
        "n_good": int(n_good),
        "positive_depth_ratio": float(depth_valid.mean()) if len(depth_valid) else 0.0,
        "median_ray_residual_m": float(np.nanmedian(residual_m)) if len(X) else np.nan,
        "median_reproj_px": float(np.nanmedian(err_px[err_valid])) if err_valid.any() else np.nan,
        "delta_from_prior_m": delta.astype(np.float64).tolist(),
        "delta_xy_m": float(np.linalg.norm(delta[:2])),
        "delta_z_m": float(delta[2]),
    }
    if not np.all(np.isfinite(C)):
        return None, None, 0, {**diag, "reason": "nonfinite_center"}
    return R, C, n_good, diag


def _save_match_vis(query_pil, dom_crop_pil, pts_query, pts_dom_crop,
                    mask, H_mat, n_inliers, save_path):
    """Draw matching visualization: query | DOM crop with inlier lines."""
    import cv2
    query_np = np.array(query_pil)
    dom_np = np.array(dom_crop_pil)

    target_h = 600
    q_h, q_w = query_np.shape[:2]
    d_h, d_w = dom_np.shape[:2]
    q_scale = target_h / max(q_h, 1)
    d_scale = target_h / max(d_h, 1)
    q_resized = cv2.resize(query_np, (max(1, int(q_w * q_scale)), target_h))
    d_resized = cv2.resize(dom_np, (max(1, int(d_w * d_scale)), target_h))

    gap = 20
    canvas_w = q_resized.shape[1] + gap + d_resized.shape[1]
    canvas = np.ones((target_h, canvas_w, 3), dtype=np.uint8) * 40
    q_end = q_resized.shape[1]
    canvas[:, :q_end] = q_resized
    d_start = q_end + gap
    canvas[:, d_start:] = d_resized

    if mask is None:
        inlier_mask = np.zeros((len(pts_query),), dtype=bool)
    else:
        inlier_mask = mask.ravel().astype(bool)
    inlier_idx = np.where(inlier_mask)[0]
    outlier_idx = np.where(~inlier_mask)[0]
    rng = np.random.RandomState(42)

    out_show = rng.choice(outlier_idx, min(50, len(outlier_idx)), replace=False) if len(outlier_idx) > 0 else []
    for i in out_show:
        p1 = (int(pts_query[i, 0] * q_scale), int(pts_query[i, 1] * q_scale))
        p2 = (d_start + int(pts_dom_crop[i, 0] * d_scale), int(pts_dom_crop[i, 1] * d_scale))
        cv2.line(canvas, p1, p2, (80, 80, 80), 1)

    in_show = rng.choice(inlier_idx, min(200, len(inlier_idx)), replace=False) if len(inlier_idx) > 0 else []
    for i in in_show:
        p1 = (int(pts_query[i, 0] * q_scale), int(pts_query[i, 1] * q_scale))
        p2 = (d_start + int(pts_dom_crop[i, 0] * d_scale), int(pts_dom_crop[i, 1] * d_scale))
        cv2.line(canvas, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (0, 200, 255), -1)
        cv2.circle(canvas, p2, 3, (0, 200, 255), -1)

    if H_mat is not None:
        cx, cy = query_np.shape[1] / 2.0, query_np.shape[0] / 2.0
        ctr_in_crop = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float64), H_mat)[0, 0]
        ctr_vis = (d_start + int(ctr_in_crop[0] * d_scale), int(ctr_in_crop[1] * d_scale))
        cv2.drawMarker(canvas, ctr_vis, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)

    label = f"Inliers: {n_inliers}/{len(pts_query)}"
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, "Query", (10, target_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(canvas, "DOM crop", (d_start + 10, target_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if isinstance(save_path, (list, tuple)):
        save_paths = [p for p in save_path if p is not None]
    else:
        save_paths = [save_path] if save_path is not None else []
    for _save_path in save_paths:
        os.makedirs(os.path.dirname(_save_path), exist_ok=True)
        ok = cv2.imwrite(_save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"[DOM-VIS] saved={ok} path={_save_path} inliers={n_inliers}/{len(pts_query)}")


def match_frame_to_dom(query_image_path, dom_image_cpu, approx_enu,
                       project_fn, inv_project_fn, roma_model,
                       crop_size_m=250, device="cuda",
                       save_vis=None):
    """Match a query frame to the DOM via RoMa + RANSAC homography.

    Args:
        query_image_path: path to the query frame.
        dom_image_cpu: (3, H_dom, W_dom) uint8 tensor on CPU.
        approx_enu: approximate ENU [E, N, U] for cropping.
        project_fn: ENU → DOM pixel mapping.
        inv_project_fn: DOM pixel → ENU mapping.
        roma_model: loaded RoMa model.
        crop_size_m: half-size of DOM crop in meters.
        device: torch device.

    Returns:
        np.array([E, N]) absolute ENU position, or None on failure.
    """
    import cv2
    from PIL import Image as PILImage

    # 1. Project approximate ENU to DOM pixel for crop centre
    approx_xyz = torch.tensor([[approx_enu[0], approx_enu[1], 0.0]])
    center_uv = project_fn(approx_xyz).numpy()[0]  # (col, row)
    center_col, center_row = center_uv[0], center_uv[1]

    # Estimate DOM pixels per metre
    offset_xyz = torch.tensor([[approx_enu[0] + 1.0, approx_enu[1], 0.0]])
    offset_uv = project_fn(offset_xyz).numpy()[0]
    px_per_m = abs(offset_uv[0] - center_col)
    crop_half_px = int(crop_size_m * px_per_m)

    _, H_dom, W_dom = dom_image_cpu.shape
    col0 = max(0, int(center_col) - crop_half_px)
    col1 = min(W_dom, int(center_col) + crop_half_px)
    row0 = max(0, int(center_row) - crop_half_px)
    row1 = min(H_dom, int(center_row) + crop_half_px)

    if col1 - col0 < 100 or row1 - row0 < 100:
        return None

    # 2. Extract DOM crop as PIL image
    dom_crop_np = dom_image_cpu[:, row0:row1, col0:col1].permute(1, 2, 0).numpy()
    dom_crop_pil = PILImage.fromarray(dom_crop_np)
    crop_H, crop_W = dom_crop_np.shape[0], dom_crop_np.shape[1]

    query_pil = PILImage.open(query_image_path).convert("RGB")
    query_W, query_H = query_pil.size

    # 3. RoMa matching (TinyRoMa: warp=(H,W,4), certainty=(H,W))
    warp, certainty = roma_model.match(query_pil, dom_crop_pil)
    if warp.dim() == 4:  # original RoMa: (1, H, W*2, 4)
        W_out = warp.shape[2] // 2
        a_coords = warp[0, :, :W_out, :2].reshape(-1, 2).cpu().numpy()
        b_coords = warp[0, :, :W_out, 2:4].reshape(-1, 2).cpu().numpy()
        c_flat = certainty[0, :, :W_out].reshape(-1).cpu().numpy()
    else:  # TinyRoMa: (H, W, 4)
        a_coords = warp[:, :, :2].reshape(-1, 2).cpu().numpy()
        b_coords = warp[:, :, 2:4].reshape(-1, 2).cpu().numpy()
        c_flat = certainty.reshape(-1).cpu().numpy()

    K = 5000
    top_idx = np.argsort(c_flat)[-K:]

    # Normalised [-1, 1] → original pixel coords
    pts_query = np.stack([
        (a_coords[top_idx, 0] + 1) / 2 * (query_W - 1),
        (a_coords[top_idx, 1] + 1) / 2 * (query_H - 1),
    ], axis=-1).astype(np.float64)

    pts_dom_crop = np.stack([
        (b_coords[top_idx, 0] + 1) / 2 * (crop_W - 1),
        (b_coords[top_idx, 1] + 1) / 2 * (crop_H - 1),
    ], axis=-1).astype(np.float64)

    # 4. RANSAC homography (query → DOM crop)
    _seed_opencv_ransac(cv2, 2)
    H_mat, mask = cv2.findHomography(pts_query, pts_dom_crop, cv2.RANSAC, 5.0)
    if H_mat is None:
        return None
    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < 20:
        return None

    # --- Visualization ---
    if save_vis is not None:
        print(f"[DOM-VIS] writing visualization to {save_vis}")
        _save_match_vis(query_pil, dom_crop_pil, pts_query, pts_dom_crop,
                        mask, H_mat, n_inliers, save_vis)

    # 5. Map query image centre through homography → DOM crop position
    cx, cy = query_W / 2.0, query_H / 2.0
    center_in_crop = cv2.perspectiveTransform(
        np.array([[[cx, cy]]], dtype=np.float64), H_mat
    )[0, 0]

    dom_col = col0 + center_in_crop[0]
    dom_row = row0 + center_in_crop[1]

    # 6. DOM pixel → ENU
    e, n = inv_project_fn(dom_col, dom_row)
    return np.array([e, n])


def compute_dom_anchors(image_paths, dom_image_cpu, gt_enu, project_fn,
                        inv_project_fn, window_size, device,
                        save_vis_dir=None):
    """Match window-start frames to DOM for absolute ENU positioning.

    Loads RoMa, matches each window's first frame, then frees VRAM.
    Uses gt_enu only as approximate crop centres (not for training).

    Returns:
        list of (np.array([E, N]) or None) per window.
    """
    N = len(image_paths)
    num_windows = (N + window_size - 1) // window_size
    boundary_indices = [w * window_size for w in range(num_windows)]

    print("[DOM-RANSAC] Loading RoMa outdoor model...")
    roma_model = load_roma(device=device)
    print(f"[DOM-RANSAC] Matching {len(boundary_indices)} boundary frames to DOM...")

    anchors = []
    for w, bi in enumerate(boundary_indices):
        approx_enu = gt_enu[bi]
        vis_path = None
        if save_vis_dir:
            vis_path = os.path.join(save_vis_dir, f"match_window{w:02d}_frame{bi:04d}.jpg")
        result = match_frame_to_dom(
            image_paths[bi], dom_image_cpu, approx_enu,
            project_fn, inv_project_fn, roma_model,
            device=device, save_vis=vis_path,
        )
        if result is not None:
            err = np.linalg.norm(result - gt_enu[bi, :2])
            print(f"  Window {w}: frame {bi} → ENU ({result[0]:.1f}, {result[1]:.1f}) m "
                  f"(GT: {gt_enu[bi, 0]:.1f}, {gt_enu[bi, 1]:.1f}), err={err:.1f} m")
        else:
            print(f"  Window {w}: frame {bi} → FAILED")
        anchors.append(result)

    del roma_model
    torch.cuda.empty_cache()
    n_ok = sum(a is not None for a in anchors)
    print(f"[DOM-RANSAC] Done: {n_ok}/{len(anchors)} anchors matched")
    return anchors


def _ray_ground_intersection(cam_center_enu, R_c2w, K_inv, pixel, ground_z=0.0):
    """Intersect a camera ray with the ground plane Z=ground_z in ENU.

    Args:
        cam_center_enu: (3,) camera position in ENU
        R_c2w: (3,3) camera-to-world rotation matrix
        K_inv: (3,3) inverse camera intrinsic matrix
        pixel: (2,) pixel coordinate (col, row)
        ground_z: ground elevation in ENU (default 0)

    Returns:
        (2,) ENU (E, N) ground intersection, or None if ray is parallel to ground.
    """
    # Ray direction in world (ENU) coordinates
    ray_cam = K_inv @ np.array([pixel[0], pixel[1], 1.0])
    ray_world = R_c2w @ ray_cam  # (3,)
    # Intersect with plane Z = ground_z
    dz = ray_world[2]
    if abs(dz) < 1e-8:
        return None  # parallel to ground
    t = (ground_z - cam_center_enu[2]) / dz
    if t < 0:
        return None  # behind camera
    hit = cam_center_enu + t * ray_world
    return hit[:2]


def render_dom_to_camera_view(
    dom_image_cpu,
    cam_enu,
    R_c2w,
    K,
    target_W,
    target_H,
    geo_elev,
    dom_transform,
    ground_z=0.0,
    max_horiz_dist_m=600.0,
):
    """Render nadir DOM as oblique perspective view via backward ray casting.

    For each output pixel (u,v): cast ray → ground plane Z=ground_z → ENU
    ground hit → DOM pixel → bilinear sample.  Flat-ground assumption
    (terrain at ground_z); good enough for RoMa dense matching.

    Returns:
        (rendered_rgb, src_col_map, src_row_map) or None on failure.
          rendered_rgb  : (H, W, 3) uint8 numpy array
          src_col_map   : (H, W) float32, source DOM col (NaN = no hit)
          src_row_map   : (H, W) float32, source DOM row (NaN = no hit)
    """
    try:
        import pyproj
        _dom_crs = geo_elev.dem_src.crs
        if _dom_crs is not None and not _dom_crs.is_geographic:
            _wgs84 = pyproj.CRS("EPSG:4326")
            _wgs84_to_crs = pyproj.Transformer.from_crs(
                _wgs84, _dom_crs, always_xy=True)
        else:
            _wgs84_to_crs = None
    except Exception as _e:
        print(f"[render_dom] WARN: could not build WGS84→CRS transformer: {_e}")
        _wgs84_to_crs = None

    # ── Step 1: backward ray cast for every output pixel ─────────────────────
    N_pix = target_H * target_W
    v_flat = (np.arange(N_pix, dtype=np.float64) // target_W)  # row
    u_flat = (np.arange(N_pix, dtype=np.float64) % target_W)   # col

    K_inv = np.linalg.inv(K)
    p_cam = K_inv @ np.vstack([u_flat, v_flat, np.ones(N_pix)])  # (3, N)
    ray_enu = R_c2w @ p_cam  # (3, N)

    # Intersect with flat ground plane Z = ground_z
    dz = ray_enu[2, :]
    valid = dz < -1e-6  # ray must point downward
    t_arr = np.where(valid, (float(ground_z) - float(cam_enu[2])) / dz, np.inf)
    E_hit = float(cam_enu[0]) + t_arr * ray_enu[0, :]
    N_hit = float(cam_enu[1]) + t_arr * ray_enu[1, :]
    # Hard cap: discard near-horizontal rays to prevent unbounded footprints
    horiz_sq = ((E_hit - float(cam_enu[0]))**2
                + (N_hit - float(cam_enu[1]))**2)
    valid &= np.isfinite(horiz_sq) & (horiz_sq < max_horiz_dist_m**2)

    if valid.sum() < 100:
        return None

    # ── Step 2: ENU ground hit → DOM pixel (fully vectorized) ────────────────
    E_v = E_hit[valid]
    N_v = N_hit[valid]

    # ENU → WGS84 lon/lat (linear approx; error < 1 m over < 5 km)
    lat_ref_rad = np.radians(float(geo_elev.lat0))
    m_per_deg_lat = (111132.92 - 559.82 * np.cos(2 * lat_ref_rad)
                     + 1.175 * np.cos(4 * lat_ref_rad))
    m_per_deg_lon = (111412.84 * np.cos(lat_ref_rad)
                     - 93.5 * np.cos(3 * lat_ref_rad))
    lons_v = float(geo_elev.lon0) + E_v / m_per_deg_lon
    lats_v = float(geo_elev.lat0) + N_v / m_per_deg_lat

    # WGS84 → DOM CRS
    if _wgs84_to_crs is not None:
        crs_xs, crs_ys = _wgs84_to_crs.transform(lons_v, lats_v)
        crs_xs = np.asarray(crs_xs, dtype=np.float64)
        crs_ys = np.asarray(crs_ys, dtype=np.float64)
    else:
        crs_xs, crs_ys = lons_v, lats_v

    # CRS → DOM pixel via vectorized inverse affine
    inv_t = ~dom_transform
    dom_cols_v = inv_t.a * crs_xs + inv_t.b * crs_ys + inv_t.c
    dom_rows_v = inv_t.d * crs_xs + inv_t.e * crs_ys + inv_t.f

    # ── Step 3: filter to DOM bounds + bilinear sample ────────────────────────
    _, H_dom, W_dom = dom_image_cpu.shape
    in_dom = ((dom_cols_v >= 0) & (dom_cols_v < W_dom - 1) &
              (dom_rows_v >= 0) & (dom_rows_v < H_dom - 1))
    if in_dom.sum() < 100:
        return None

    dc = dom_cols_v[in_dom]
    dr = dom_rows_v[in_dom]
    c0_i = dc.astype(np.int32)
    r0_i = dr.astype(np.int32)
    wc = (dc - c0_i).astype(np.float32)
    wr = (dr - r0_i).astype(np.float32)
    c1_i = np.minimum(c0_i + 1, W_dom - 1)
    r1_i = np.minimum(r0_i + 1, H_dom - 1)

    dom_np = dom_image_cpu.numpy().astype(np.float32)  # (3, H_dom, W_dom)
    px_color = (dom_np[:, r0_i, c0_i] * ((1 - wr) * (1 - wc)) +
                dom_np[:, r0_i, c1_i] * ((1 - wr) * wc) +
                dom_np[:, r1_i, c0_i] * (wr * (1 - wc)) +
                dom_np[:, r1_i, c1_i] * (wr * wc))  # (3, M)
    px_color = np.clip(px_color, 0, 255).astype(np.uint8)

    # ── Step 4: fill rendered canvas ─────────────────────────────────────────
    rendered = np.zeros((target_H, target_W, 3), dtype=np.uint8)
    src_col_map = np.full((target_H, target_W), np.nan, dtype=np.float32)
    src_row_map = np.full((target_H, target_W), np.nan, dtype=np.float32)

    valid_flat_idx = np.where(valid)[0]
    final_flat_idx = valid_flat_idx[in_dom]
    pix_r = final_flat_idx // target_W
    pix_c = final_flat_idx % target_W

    rendered[pix_r, pix_c] = px_color.T  # (M, 3)
    src_col_map[pix_r, pix_c] = dc.astype(np.float32)
    src_row_map[pix_r, pix_c] = dr.astype(np.float32)

    fill_frac = float(in_dom.sum()) / max(float(valid.sum()), 1.0)
    print(f"[render_dom] fill={fill_frac*100:.1f}%"
          f" ({in_dom.sum()}/{valid.sum()} px), ground_z={ground_z:.1f}m")
    return rendered, src_col_map, src_row_map


def get_frame_dom_correspondences(query_image_path, dom_image_cpu, approx_enu,
                                   project_fn, inv_project_fn, roma_model,
                                   crop_size_m=250, device="cuda",
                                   min_inliers=20, top_k=5000,
                                   max_correspondences=500,
                                   save_vis=None,
                                   save_vis_extra=None,
                                   camera_fov=None,
                                   geo_elev=None,
                                   dom_transform=None,
                                   pred_pose_crop=None,
                                   query_yaw_deg=None,
                                   compute_pnp=True,
                                   **kwargs):
    # NOTE(2026-04-29): query_yaw_deg used to pre-rotate the DOM crop so its
    # "up" direction aligns with the query image "up" before RoMa matching.
    # This restores the heading-aware crop behavior that was lost in the
    # 2026-04-28 revert and avoids RoMa having to do hard rotation matching.
    if kwargs:
        try:
            print(f"[dom_matching] WARN: get_frame_dom_correspondences ignored kwargs={list(kwargs.keys())}")
        except Exception:
            pass
    """Get dense pixel→ENU correspondences for a frame via RoMa + RANSAC.

    Args:
        camera_fov: dict with keys 'fx', 'fy', 'orig_w', 'orig_h', 'altitude'
                    for precise FOV-matched crop. If None, uses crop_size_m square.
        geo_elev: GeoElevationQuery instance for DEM height + building filter.
                  If None, returns 2D ENU (Z=0) without building filtering.
        dom_transform: rasterio transform for DOM→lon/lat conversion (needed
                       by geo_elev.enu_3d_from_dom).
        pred_pose_crop: dict with keys 'cam_center_enu' (3,), 'R_c2w' (3,3),
                and optional 'ground_z', 'crop_scale',
                'min_half_size_m', 'max_half_size_m',
                'max_center_shift_m'. When provided, the crop center is
                the image-center ray's ground intersection, clamped near
                approx_enu, and the crop size is a bounded FOV-derived
                rectangle. Requires camera_fov for intrinsics.

    Returns dict with 'frame_pixels', 'enu_positions' (K,2 or K,3),
    'n_inliers', 'query_W', 'query_H', 'pnp_result' (optional), or None.
    """
    import cv2
    from PIL import Image as PILImage

    # --- Crop center & bounds computation ---
    _use_pose_crop = False
    _pose_half_w_m = None
    _pose_half_h_m = None
    _pose_center_shift_m = None
    _pose_agl_m = None
    _direct_rotated_crop = False
    _direct_crop_axis_right = None
    _direct_crop_axis_up = None
    _direct_crop_center_uv = None
    _direct_crop_right_uv = None
    _direct_crop_down_uv = None
    _direct_crop_center_px = None
    _pose_crop_failure_reason = None
    _pose_center_ray_z = None
    _pose_center_ray_t = None
    _pose_cam_center_enu = None
    _pose_center_ground_enu = None
    _pose_crop_center_ground_enu = None
    _pose_ground_z = None
    if pred_pose_crop is not None and camera_fov is not None:
        _cam_enu = np.asarray(pred_pose_crop['cam_center_enu'], dtype=np.float64)
        _R_c2w = np.asarray(pred_pose_crop['R_c2w'], dtype=np.float64)
        _pose_cam_center_enu = _cam_enu[:3].copy()
        _fx = camera_fov['fx']
        _fy = camera_fov['fy']
        _W = camera_fov['orig_w']
        _H = camera_fov['orig_h']
        _cx = float(camera_fov.get('cx', _W / 2.0))
        _cy = float(camera_fov.get('cy', _H / 2.0))
        _K = np.array([[_fx, 0, _cx], [0, _fy, _cy], [0, 0, 1]], dtype=np.float64)
        _K_inv = np.linalg.inv(_K)
        _strict_footprint = bool(pred_pose_crop.get('strict_footprint', False))

        _ground_z = float(pred_pose_crop.get('ground_z', 0.0))
        _pose_ground_z = float(_ground_z)

        # Image center → ground
        try:
            _center_ray_cam = _K_inv @ np.array([_cx, _cy, 1.0], dtype=np.float64)
            _center_ray_world = _R_c2w @ _center_ray_cam
            _pose_center_ray_z = float(_center_ray_world[2])
            if abs(_pose_center_ray_z) > 1e-9:
                _pose_center_ray_t = float((_ground_z - float(_cam_enu[2])) / _pose_center_ray_z)
        except Exception:
            _pose_crop_failure_reason = 'center_ray_invalid'
        _center_gnd = _ray_ground_intersection(_cam_enu, _R_c2w, _K_inv,
                                                np.array([_cx, _cy]), _ground_z)
        if _center_gnd is not None:
            _pose_center_ground_enu = np.asarray(_center_gnd[:2], dtype=np.float64).copy()
        if _center_gnd is not None:
            _use_pose_crop = True
            _approx_xy = np.asarray(approx_enu[:2], dtype=np.float64)
            _center_shift = np.asarray(_center_gnd, dtype=np.float64) - _approx_xy
            _pose_center_shift_m = float(np.linalg.norm(_center_shift))
            _max_center_shift_m = float(pred_pose_crop.get('max_center_shift_m', crop_size_m))
            if np.isfinite(_max_center_shift_m) and _max_center_shift_m > 0 and _pose_center_shift_m > _max_center_shift_m:
                _center_gnd = _approx_xy + _center_shift * (_max_center_shift_m / max(_pose_center_shift_m, 1e-6))
                _pose_center_shift_m = float(_max_center_shift_m)

            _pose_agl = abs(float(_cam_enu[2]) - _ground_z)
            if not np.isfinite(_pose_agl) or _pose_agl < 1.0:
                _pose_agl = float(camera_fov.get('altitude', crop_size_m))
            _pose_agl_m = float(_pose_agl)
            _crop_scale = float(pred_pose_crop.get('crop_scale', 2.5))
            _min_half_m = float(pred_pose_crop.get('min_half_size_m', min(crop_size_m, 200.0)))
            _max_half_m = float(pred_pose_crop.get('max_half_size_m', max(crop_size_m, _min_half_m)))
            # Strict mode samples query corners too, so the crop is the actual
            # query footprint instead of the historical loose edge-midpoint box.
            _hit_dist_cfg = pred_pose_crop.get('max_hit_dist_m', None)
            if _hit_dist_cfg is not None and np.isfinite(float(_hit_dist_cfg)):
                _max_hit_dist_m = float(_hit_dist_cfg)
            else:
                _max_hit_dist_m = min(6.0 * _pose_agl_m, 2.0 * _max_half_m)
            _has_query_yaw = (
                query_yaw_deg is not None
                and np.isfinite(float(query_yaw_deg))
            )
            if _strict_footprint or _has_query_yaw:
                _edge_pixels = [
                    np.array([0.0, 0.0]),
                    np.array([float(_W), 0.0]),
                    np.array([float(_W), float(_H)]),
                    np.array([0.0, float(_H)]),
                    np.array([0.0, _cy]),
                    np.array([float(_W), _cy]),
                    np.array([_cx, 0.0]),
                    np.array([_cx, float(_H)]),
                ]
            else:
                _edge_pixels = [
                    np.array([0.0,  _cy]),   # left edge midpoint
                    np.array([_W,   _cy]),   # right edge midpoint
                    np.array([_cx,  0.0]),   # top edge midpoint
                    np.array([_cx,  _H]),    # bottom edge midpoint
                ]
            _valid_hits = []
            for _pix in _edge_pixels:
                _hit = _ray_ground_intersection(_cam_enu, _R_c2w, _K_inv, _pix, _ground_z)
                if _hit is None:
                    continue
                if float(np.linalg.norm(_hit - _cam_enu[:2])) > _max_hit_dist_m:
                    continue
                _valid_hits.append(_hit)
            if len(_valid_hits) >= 2:
                _hits_arr = np.array(_valid_hits)
                _query_aligned_bbox = False
                if _has_query_yaw:
                    _yaw_rad = np.deg2rad(float(query_yaw_deg))
                    _axis_up = np.array([np.sin(_yaw_rad), np.cos(_yaw_rad)], dtype=np.float64)
                    _axis_right = np.array([np.cos(_yaw_rad), -np.sin(_yaw_rad)], dtype=np.float64)
                    _aligned = np.stack([
                        _hits_arr[:, :2] @ _axis_right,
                        _hits_arr[:, :2] @ _axis_up,
                    ], axis=1)
                    _min_xy = _aligned.min(axis=0)
                    _max_xy = _aligned.max(axis=0)
                    _center_aligned = 0.5 * (_min_xy + _max_xy)
                    _center_gnd = _axis_right * _center_aligned[0] + _axis_up * _center_aligned[1]
                    _center_shift = _center_gnd - _approx_xy
                    _pose_center_shift_m = float(np.linalg.norm(_center_shift))
                    _half_right = max(0.5 * float(_max_xy[0] - _min_xy[0]) * _crop_scale, _min_half_m)
                    _half_up = max(0.5 * float(_max_xy[1] - _min_xy[1]) * _crop_scale, _min_half_m)
                    _query_aspect = float(_W) / max(float(_H), 1e-6)
                    if np.isfinite(_query_aspect) and _query_aspect > 0:
                        _cur_aspect = _half_right / max(_half_up, 1e-6)
                        if _cur_aspect < _query_aspect:
                            _half_right = _half_up * _query_aspect
                        else:
                            _half_up = _half_right / _query_aspect
                    _pose_half_w_m = _half_right
                    _pose_half_h_m = _half_up
                    _direct_rotated_crop = True
                    _direct_crop_axis_right = _axis_right
                    _direct_crop_axis_up = _axis_up
                    _query_aligned_bbox = True
                if not _query_aligned_bbox:
                    if _strict_footprint:
                        _min_e, _max_e = float(_hits_arr[:, 0].min()), float(_hits_arr[:, 0].max())
                        _min_n, _max_n = float(_hits_arr[:, 1].min()), float(_hits_arr[:, 1].max())
                        _center_gnd = np.array([0.5 * (_min_e + _max_e), 0.5 * (_min_n + _max_n)], dtype=np.float64)
                        _center_shift = _center_gnd - _approx_xy
                        _pose_center_shift_m = float(np.linalg.norm(_center_shift))
                        _half_e = 0.5 * (_max_e - _min_e)
                        _half_n = 0.5 * (_max_n - _min_n)
                    else:
                        # Bounding box half-radii relative to the crop center
                        _half_e = float(np.max(np.abs(_hits_arr[:, 0] - _center_gnd[0])))
                        _half_n = float(np.max(np.abs(_hits_arr[:, 1] - _center_gnd[1])))
                    _pose_half_w_m = max(_half_e * _crop_scale, _min_half_m)
                    _pose_half_h_m = max(_half_n * _crop_scale, _min_half_m)
            else:
                # Fallback: nadir formula (exact equivalent for pitch=0)
                _fov_w_m = float(_W) / max(float(_fx), 1e-6) * _pose_agl_m
                _fov_h_m = float(_H) / max(float(_fy), 1e-6) * _pose_agl_m
                _pose_half_w_m = max(0.5 * _fov_w_m * _crop_scale, _min_half_m)
                _pose_half_h_m = max(0.5 * _fov_h_m * _crop_scale, _min_half_m)
            if np.isfinite(_max_half_m) and _max_half_m > 0:
                _pose_half_w_m = min(_pose_half_w_m, _max_half_m)
                _pose_half_h_m = min(_pose_half_h_m, _max_half_m)

            _pose_crop_center_ground_enu = np.asarray(_center_gnd[:2], dtype=np.float64).copy()

            # Project the bounded center-ray crop center to DOM pixels.
            _center_xyz = torch.tensor([[_center_gnd[0], _center_gnd[1], _ground_z]])
            _center_uv = project_fn(_center_xyz)
            if torch.is_tensor(_center_uv):
                _center_uv = _center_uv.detach().cpu().numpy()
            _center_uv = np.asarray(_center_uv, dtype=np.float64)[0]
            center_col, center_row = _center_uv[0], _center_uv[1]
        elif _pose_crop_failure_reason is None:
            if _pose_center_ray_z is None or not np.isfinite(_pose_center_ray_z):
                _pose_crop_failure_reason = 'center_ray_invalid'
            elif abs(_pose_center_ray_z) <= 1e-9:
                _pose_crop_failure_reason = 'center_ray_parallel'
            elif _pose_center_ray_t is not None and _pose_center_ray_t < 0.0:
                _pose_crop_failure_reason = 'center_ray_behind'
            else:
                _pose_crop_failure_reason = 'center_ray_no_hit'

    if not _use_pose_crop:
        # Fallback: nadir-based crop
        approx_xyz = torch.tensor([[approx_enu[0], approx_enu[1], 0.0]])
        center_uv = project_fn(approx_xyz).numpy()[0]
        center_col, center_row = center_uv[0], center_uv[1]

    # px_per_m: DOM resolution (pixels per meter), independent of crop center
    _ref_xyz = torch.tensor([[approx_enu[0], approx_enu[1], 0.0]])
    _ref_uv = project_fn(_ref_xyz).numpy()[0]
    offset_xyz = torch.tensor([[approx_enu[0] + 1.0, approx_enu[1], 0.0]])
    offset_uv = project_fn(offset_xyz).numpy()[0]
    px_per_m = abs(offset_uv[0] - _ref_uv[0])

    if _use_pose_crop:
        half_w_px = int(round(float(_pose_half_w_m) * px_per_m))
        half_h_px = int(round(float(_pose_half_h_m) * px_per_m))
        _, H_dom, W_dom = dom_image_cpu.shape
        col0 = max(0, int(center_col) - half_w_px)
        col1 = min(W_dom, int(center_col) + half_w_px)
        row0 = max(0, int(center_row) - half_h_px)
        row1 = min(H_dom, int(center_row) + half_h_px)
    else:
        # Compute crop bounds: camera-FOV-matched rectangle or square fallback
        if camera_fov is not None:
            fov_w_m = camera_fov['orig_w'] / camera_fov['fx'] * camera_fov['altitude']
            fov_h_m = camera_fov['orig_h'] / camera_fov['fy'] * camera_fov['altitude']
            half_w_px = int(round(fov_w_m / 2.0 * px_per_m))
            half_h_px = int(round(fov_h_m / 2.0 * px_per_m))
        else:
            half_w_px = int(crop_size_m * px_per_m)
            half_h_px = half_w_px

        _, H_dom, W_dom = dom_image_cpu.shape
        col0 = max(0, int(center_col) - half_w_px)
        col1 = min(W_dom, int(center_col) + half_w_px)
        row0 = max(0, int(center_row) - half_h_px)
        row1 = min(H_dom, int(center_row) + half_h_px)

    if col1 - col0 < 100 or row1 - row0 < 100:
        return None

    # --- Pose render: synthesize perspective view from pose ------------------
    # When full pose info is available, render the nadir DOM as a perspective
    # view that matches the query camera.  Near-nadir frames are allowed too;
    # a hard pitch threshold creates frame-to-frame crop discontinuities around
    # the threshold and makes the fallback pose crop look like a larger-altitude
    # orthographic crop.
    _use_render = False
    _render_src_col_map = None
    _render_src_row_map = None
    _pitch_deg_render = None
    _render_failure_reason = None
    _render_valid_ratio = None
    if (_use_pose_crop and geo_elev is not None and dom_transform is not None
            and camera_fov is not None
            and not bool(pred_pose_crop.get('disable_render', False))):
        _fwd_enu = _R_c2w @ np.array([0.0, 0.0, 1.0])
        # pitch from nadir = angle between optical axis (+z_cam in ENU) and
        # straight-down direction (-U = [0,0,-1] in ENU).
        # arccos(-fwd_z): nadir → 0°, horizontal → 90°.
        _pitch_deg_render = float(np.degrees(
            np.arccos(np.clip(-float(_fwd_enu[2]), -1.0, 1.0))))
        _max_dist_render = max(
            float(pred_pose_crop.get('max_center_shift_m', 400.0)) + 150.0,
            6.0 * float(_pose_agl_m),
            800.0,
        )
        _render_result = render_dom_to_camera_view(
            dom_image_cpu, _cam_enu, _R_c2w, _K,
            int(_W), int(_H),
            geo_elev, dom_transform,
            ground_z=_ground_z,
            max_horiz_dist_m=_max_dist_render,
        )
        if _render_result is not None:
            _render_np, _render_src_col_map, _render_src_row_map = _render_result
            _render_valid = (
                np.isfinite(_render_src_col_map)
                & np.isfinite(_render_src_row_map)
            )
            _render_valid_ratio = float(_render_valid.mean()) if _render_valid.size else 0.0
            _min_valid_ratio = float(pred_pose_crop.get('render_min_valid_ratio', 0.30))
            if _render_valid_ratio >= _min_valid_ratio:
                dom_crop_np = _render_np
                _use_render = True
                M_rot_inv = None
                rot_angle_ccw = 0.0
                rot_inner_offset_col = 0.0
                rot_inner_offset_row = 0.0
                crop_offset_col = 0
                crop_offset_row = 0
            else:
                _render_failure_reason = f'render_low_valid_ratio:{_render_valid_ratio:.3f}<{_min_valid_ratio:.3f}'
        else:
            _render_failure_reason = 'render_none'
    elif _use_pose_crop:
        _render_failure_reason = 'render_disabled_or_missing_inputs'

    if not _use_render:
        # --- Yaw-aware crop: rotate DOM crop so "up" aligns with query "up" ---
        # query_yaw_deg is the angle of query image "up" measured CW from north
        # (defaults.py computes it from R_model->enu @ R_c2w_model[fi] @ [0,-1,0]
        # via atan2(up_enu[0], up_enu[1])).
        # DOM image coord: x->east(col), y->south(row), so DOM "up" = north.
        # cv2.getRotationMatrix2D angle convention: positive=CCW in image coords.
        # Rotate the north-up DOM crop by the query-up yaw so the DOM crop's
        # image-up coincides with query-up.
        _apply_yaw = (
            query_yaw_deg is not None
            and np.isfinite(float(query_yaw_deg))
            and abs(float(query_yaw_deg)) > 1.0
        )
        rot_inner_offset_col = 0.0
        rot_inner_offset_row = 0.0
        if _direct_rotated_crop:
            _, H_dom, W_dom = dom_image_cpu.shape
            core_w = max(1, int(round(float(_pose_half_w_m) * 2.0 * px_per_m)))
            core_h = max(1, int(round(float(_pose_half_h_m) * 2.0 * px_per_m)))
            if core_w < 100 or core_h < 100:
                return None
            _basis_xyz = torch.tensor([
                [_center_gnd[0], _center_gnd[1], _ground_z],
                [_center_gnd[0] + _direct_crop_axis_right[0], _center_gnd[1] + _direct_crop_axis_right[1], _ground_z],
                [_center_gnd[0] - _direct_crop_axis_up[0], _center_gnd[1] - _direct_crop_axis_up[1], _ground_z],
            ], dtype=torch.float64)
            _basis_uv = project_fn(_basis_xyz)
            if torch.is_tensor(_basis_uv):
                _basis_uv = _basis_uv.detach().cpu().numpy()
            _basis_uv = np.asarray(_basis_uv, dtype=np.float64)
            _direct_crop_center_uv = _basis_uv[0]
            _direct_crop_right_uv = _basis_uv[1] - _basis_uv[0]
            _direct_crop_down_uv = _basis_uv[2] - _basis_uv[0]
            _direct_crop_center_px = np.array([0.5 * (core_w - 1), 0.5 * (core_h - 1)], dtype=np.float64)
            _xs_m = (np.arange(core_w, dtype=np.float32) - float(_direct_crop_center_px[0])) / float(px_per_m)
            _ys_m = (np.arange(core_h, dtype=np.float32) - float(_direct_crop_center_px[1])) / float(px_per_m)
            _grid_x_m, _grid_y_m = np.meshgrid(_xs_m, _ys_m)
            _map_col = (_direct_crop_center_uv[0]
                        + _grid_x_m * _direct_crop_right_uv[0]
                        + _grid_y_m * _direct_crop_down_uv[0]).astype(np.float32)
            _map_row = (_direct_crop_center_uv[1]
                        + _grid_x_m * _direct_crop_right_uv[1]
                        + _grid_y_m * _direct_crop_down_uv[1]).astype(np.float32)
            _dom_np = dom_image_cpu.permute(1, 2, 0).numpy()
            dom_crop_np = cv2.remap(
                _dom_np, _map_col, _map_row,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            M_rot_inv = None
            rot_angle_ccw = float(query_yaw_deg) if query_yaw_deg is not None else 0.0
            crop_offset_col = 0
            crop_offset_row = 0
            col0 = max(0, int(np.floor(float(np.nanmin(_map_col)))))
            col1 = min(W_dom, int(np.ceil(float(np.nanmax(_map_col)))))
            row0 = max(0, int(np.floor(float(np.nanmin(_map_row)))))
            row1 = min(H_dom, int(np.ceil(float(np.nanmax(_map_row)))))
        elif _apply_yaw:
            EXPAND = 1.5  # enlarge bbox so rotated content has no border in inner core
            cx_px = 0.5 * (col0 + col1)
            cy_px = 0.5 * (row0 + row1)
            half_w_cur = 0.5 * (col1 - col0)
            half_h_cur = 0.5 * (row1 - row0)
            half_w_exp = half_w_cur * EXPAND
            half_h_exp = half_h_cur * EXPAND
            _, H_dom, W_dom = dom_image_cpu.shape
            col0_exp = max(0, int(round(cx_px - half_w_exp)))
            col1_exp = min(W_dom, int(round(cx_px + half_w_exp)))
            row0_exp = max(0, int(round(cy_px - half_h_exp)))
            row1_exp = min(H_dom, int(round(cy_px + half_h_exp)))
            if col1_exp - col0_exp < 100 or row1_exp - row0_exp < 100:
                return None
            dom_crop_exp = dom_image_cpu[
                :, row0_exp:row1_exp, col0_exp:col1_exp
            ].permute(1, 2, 0).numpy()
            eh, ew = dom_crop_exp.shape[:2]
            rot_angle_ccw = float(query_yaw_deg)  # map north-up DOM to query-up
            rot_center = (float(cx_px - col0_exp), float(cy_px - row0_exp))
            M_rot = cv2.getRotationMatrix2D(
                rot_center, rot_angle_ccw, 1.0)
            dom_crop_rot = cv2.warpAffine(
                dom_crop_exp, M_rot, (ew, eh),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            M_rot_inv = cv2.invertAffineTransform(M_rot)  # rotated px -> expanded crop px
            core_w = max(1, int(col1 - col0))
            core_h = max(1, int(row1 - row0))
            inner_x0 = int(round(rot_center[0] - 0.5 * core_w))
            inner_y0 = int(round(rot_center[1] - 0.5 * core_h))
            inner_x0 = max(0, min(max(0, ew - core_w), inner_x0))
            inner_y0 = max(0, min(max(0, eh - core_h), inner_y0))
            inner_x1 = min(ew, inner_x0 + core_w)
            inner_y1 = min(eh, inner_y0 + core_h)
            if inner_x1 - inner_x0 < 100 or inner_y1 - inner_y0 < 100:
                return None
            dom_crop_np = dom_crop_rot[inner_y0:inner_y1, inner_x0:inner_x1]
            rot_inner_offset_col = float(inner_x0)
            rot_inner_offset_row = float(inner_y0)
            crop_offset_col = col0_exp
            crop_offset_row = row0_exp
        else:
            dom_crop_np = dom_image_cpu[:, row0:row1, col0:col1].permute(1, 2, 0).numpy()
            M_rot_inv = None
            rot_angle_ccw = 0.0
            crop_offset_col = col0
            crop_offset_row = row0

    dom_crop_pil = PILImage.fromarray(dom_crop_np)
    crop_H, crop_W = dom_crop_np.shape[0], dom_crop_np.shape[1]

    query_pil = PILImage.open(query_image_path).convert("RGB")
    query_W, query_H = query_pil.size

    warp, certainty = roma_model.match(query_pil, dom_crop_pil)
    # TinyRoMa: warp=(H,W,4), certainty=(H,W); RoMa: (1,H,W*2,4), (1,H,W*2)
    if warp.dim() == 4:
        W_out = warp.shape[2] // 2
        a_coords = warp[0, :, :W_out, :2].reshape(-1, 2).cpu().numpy()
        b_coords = warp[0, :, :W_out, 2:4].reshape(-1, 2).cpu().numpy()
        c_flat = certainty[0, :, :W_out].reshape(-1).cpu().numpy()
    else:
        a_coords = warp[:, :, :2].reshape(-1, 2).cpu().numpy()
        b_coords = warp[:, :, 2:4].reshape(-1, 2).cpu().numpy()
        c_flat = certainty.reshape(-1).cpu().numpy()

    top_idx = np.argsort(c_flat)[-top_k:]

    pts_query = np.stack([
        (a_coords[top_idx, 0] + 1) / 2 * (query_W - 1),
        (a_coords[top_idx, 1] + 1) / 2 * (query_H - 1),
    ], axis=-1).astype(np.float64)

    pts_dom_crop = np.stack([
        (b_coords[top_idx, 0] + 1) / 2 * (crop_W - 1),
        (b_coords[top_idx, 1] + 1) / 2 * (crop_H - 1),
    ], axis=-1).astype(np.float64)

    _seed_opencv_ransac(cv2, 3)
    H_mat, mask = cv2.findHomography(pts_query, pts_dom_crop, cv2.RANSAC, 5.0)
    if H_mat is None:
        return None
    inlier_mask = mask.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    if n_inliers < min_inliers:
        return None

    # --- Visualization ---
    _save_vis_paths = []
    if save_vis is not None:
        _save_vis_paths.append(save_vis)
    if save_vis_extra is not None:
        if isinstance(save_vis_extra, (list, tuple)):
            _save_vis_paths.extend([p for p in save_vis_extra if p is not None])
        else:
            _save_vis_paths.append(save_vis_extra)
    if _save_vis_paths:
        _save_match_vis(query_pil, dom_crop_pil, pts_query, pts_dom_crop,
                        mask, H_mat, n_inliers, _save_vis_paths)

    inlier_query = pts_query[inlier_mask]
    inlier_dom_crop = pts_dom_crop[inlier_mask]

    # DOM crop pixel → absolute DOM pixel (account for render / yaw rotation)
    if _use_render:
        # Rendered view: map rendered pixel (u,v) → source DOM pixel via src maps
        _rc_H, _rc_W = _render_src_col_map.shape
        _dom_cols_list = []
        _dom_rows_list = []
        _valid_render_idx = []
        for _ri in range(len(inlier_dom_crop)):
            _ur = int(round(np.clip(inlier_dom_crop[_ri, 0], 0, _rc_W - 1)))
            _vr = int(round(np.clip(inlier_dom_crop[_ri, 1], 0, _rc_H - 1)))
            _sc = _render_src_col_map[_vr, _ur]
            _sr = _render_src_row_map[_vr, _ur]
            if np.isnan(_sc) or np.isnan(_sr):
                continue  # this rendered pixel has no DOM backing
            _dom_cols_list.append(float(_sc))
            _dom_rows_list.append(float(_sr))
            _valid_render_idx.append(_ri)
        if len(_valid_render_idx) < min_inliers:
            return None
        _valid_render_idx = np.array(_valid_render_idx)
        inlier_query = inlier_query[_valid_render_idx]
        dom_cols = np.array(_dom_cols_list, dtype=np.float64)
        dom_rows = np.array(_dom_rows_list, dtype=np.float64)
    elif _direct_crop_center_uv is not None:
        _x_m = (inlier_dom_crop[:, 0] - _direct_crop_center_px[0]) / float(px_per_m)
        _y_m = (inlier_dom_crop[:, 1] - _direct_crop_center_px[1]) / float(px_per_m)
        dom_cols = (_direct_crop_center_uv[0]
                    + _x_m * _direct_crop_right_uv[0]
                    + _y_m * _direct_crop_down_uv[0])
        dom_rows = (_direct_crop_center_uv[1]
                    + _x_m * _direct_crop_right_uv[1]
                    + _y_m * _direct_crop_down_uv[1])
    elif M_rot_inv is not None:
        inlier_dom_rot = inlier_dom_crop.copy()
        inlier_dom_rot[:, 0] += rot_inner_offset_col
        inlier_dom_rot[:, 1] += rot_inner_offset_row
        _ones = np.ones((len(inlier_dom_crop), 1), dtype=np.float64)
        _h = np.concatenate([inlier_dom_rot, _ones], axis=1)  # (K, 3)
        _unrot = _h @ M_rot_inv.T  # (K, 2) in expanded crop pixels
        dom_cols = crop_offset_col + _unrot[:, 0]
        dom_rows = crop_offset_row + _unrot[:, 1]
    else:
        dom_cols = crop_offset_col + inlier_dom_crop[:, 0]
        dom_rows = crop_offset_row + inlier_dom_crop[:, 1]

    # Build correspondences with optional building filter + DEM elevation
    is_road_arr = None  # road membership per kept correspondence
    if geo_elev is not None and dom_transform is not None:
        # 3D ENU with building filtering
        keep_idx = []
        enu_3d_list = []
        is_road_list = []
        n_bld_filtered = 0
        for k in range(len(dom_cols)):
            if geo_elev.is_building_at_dom(dom_cols[k], dom_rows[k]):
                n_bld_filtered += 1
                continue
            ej, nj, uj = geo_elev.enu_3d_from_dom(
                dom_cols[k], dom_rows[k], dom_transform)
            keep_idx.append(k)
            enu_3d_list.append([ej, nj, uj])
            is_road_list.append(geo_elev.is_road_at_dom(dom_cols[k], dom_rows[k]))

        if len(keep_idx) < min_inliers:
            return None

        keep_idx = np.array(keep_idx)
        inlier_query = inlier_query[keep_idx]
        enu_positions = np.array(enu_3d_list)  # (K, 3)
        is_road_arr = np.array(is_road_list, dtype=bool)
        n_inliers = len(keep_idx)
    else:
        # 2D ENU fallback (no DEM/building mask)
        enu_positions = np.zeros((len(dom_cols), 2))
        for k in range(len(dom_cols)):
            e, n = inv_project_fn(dom_cols[k], dom_rows[k])
            enu_positions[k] = [e, n]
        n_bld_filtered = 0

    # Subsample if too many
    if n_inliers > max_correspondences:
        sel = np.random.RandomState(42).choice(
            n_inliers, max_correspondences, replace=False)
        inlier_query = inlier_query[sel]
        enu_positions = enu_positions[sel]
        if is_road_arr is not None:
            is_road_arr = is_road_arr[sel]
        n_inliers = max_correspondences

    enu_positions_road = None
    frame_pixels_road = None
    n_road_inliers = 0
    if is_road_arr is not None and len(is_road_arr) == len(inlier_query):
        road_mask = is_road_arr.astype(bool)
        n_road_inliers = int(road_mask.sum())
        if n_road_inliers >= 8:
            enu_positions_road = enu_positions[road_mask]
            frame_pixels_road = inlier_query[road_mask]

    # Also compute frame center ENU via same homography (zero extra cost)
    cx, cy = query_W / 2.0, query_H / 2.0
    center_in_crop = cv2.perspectiveTransform(
        np.array([[[cx, cy]]], dtype=np.float64), H_mat
    )[0, 0]
    if _use_render:
        _fc_ur = int(round(np.clip(
            center_in_crop[0], 0, _render_src_col_map.shape[1] - 1)))
        _fc_vr = int(round(np.clip(
            center_in_crop[1], 0, _render_src_col_map.shape[0] - 1)))
        _fc_sc = _render_src_col_map[_fc_vr, _fc_ur]
        _fc_sr = _render_src_row_map[_fc_vr, _fc_ur]
        if not (np.isnan(_fc_sc) or np.isnan(_fc_sr)):
            fc_e, fc_n = inv_project_fn(float(_fc_sc), float(_fc_sr))
        else:
            fc_e, fc_n = float(approx_enu[0]), float(approx_enu[1])
    elif _direct_crop_center_uv is not None:
        _x_m = (center_in_crop[0] - _direct_crop_center_px[0]) / float(px_per_m)
        _y_m = (center_in_crop[1] - _direct_crop_center_px[1]) / float(px_per_m)
        fc_col = (_direct_crop_center_uv[0]
                  + _x_m * _direct_crop_right_uv[0]
                  + _y_m * _direct_crop_down_uv[0])
        fc_row = (_direct_crop_center_uv[1]
                  + _x_m * _direct_crop_right_uv[1]
                  + _y_m * _direct_crop_down_uv[1])
        fc_e, fc_n = inv_project_fn(fc_col, fc_row)
    elif M_rot_inv is not None:
        _pt = np.array([
            center_in_crop[0] + rot_inner_offset_col,
            center_in_crop[1] + rot_inner_offset_row,
            1.0,
        ], dtype=np.float64)
        _pt_un = M_rot_inv @ _pt
        fc_col = crop_offset_col + _pt_un[0]
        fc_row = crop_offset_row + _pt_un[1]
        fc_e, fc_n = inv_project_fn(fc_col, fc_row)
    else:
        fc_col = crop_offset_col + center_in_crop[0]
        fc_row = crop_offset_row + center_in_crop[1]
        fc_e, fc_n = inv_project_fn(fc_col, fc_row)

    # Heading from partial affine (4 DOF: rotation+scale+tx+ty)
    inlier_mask_h = mask.ravel().astype(bool)
    pts_q_inl = pts_query[inlier_mask_h]
    pts_d_inl = pts_dom_crop[inlier_mask_h]
    _seed_opencv_ransac(cv2, 4)
    A_part, _ = cv2.estimateAffinePartial2D(
        pts_q_inl, pts_d_inl, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    heading_deg = None
    estimated_agl = None
    estimated_agl_source = None
    if A_part is not None:
        _h_rot = float(np.degrees(np.arctan2(A_part[1, 0], A_part[0, 0])))
        heading_deg = _h_rot - rot_angle_ccw
        # Estimate per-frame AGL from affine scale only for metric DOM crops.
        # In render mode the crop pixels are camera-view pixels, not DOM px/m;
        # using DOM px_per_m there turns an image scale near 1 into fx/px_per_m.
        affine_scale = float(np.sqrt(A_part[0, 0]**2 + A_part[1, 0]**2))
        if px_per_m > 0 and camera_fov is not None:
            if _use_render:
                if _pose_agl_m is not None and np.isfinite(float(_pose_agl_m)):
                    estimated_agl = float(_pose_agl_m)
                    estimated_agl_source = 'render_pose_agl'
            else:
                estimated_agl = float(camera_fov['fx']) * affine_scale / px_per_m
                estimated_agl_source = 'affine_metric_crop'

    # PnP if we have 3D ENU correspondences + camera intrinsics.
    # For oblique rendered crops, keep the render pose rotation fixed and solve
    # only for the local camera-center correction around the current model pose.
    # Non-render paths keep the historical free-PnP behavior unchanged.
    pnp_result = None
    pnp_debug = None
    pnp_rotation_check = None
    if bool(compute_pnp) and enu_positions.shape[1] == 3 and camera_fov is not None:
        fx = camera_fov['fx']
        fy = camera_fov['fy']
        cx_cam = float(camera_fov.get('cx', camera_fov['orig_w'] / 2.0))
        cy_cam = float(camera_fov.get('cy', camera_fov['orig_h'] / 2.0))
        if _use_render and _use_pose_crop and pred_pose_crop is not None:
            _prior_weight = max(25.0, min(100.0, 0.05 * float(len(enu_positions))))
            _huber_m = max(15.0, min(60.0, 0.10 * float(_pose_agl_m or 250.0)))
            R_c2w, t_world, n_pnp_inl, pnp_debug = solve_pnp_fixed_rotation_center(
                inlier_query, enu_positions, fx, fy, cx_cam, cy_cam,
                _R_c2w, _cam_enu, prior_weight=_prior_weight,
                huber_m=_huber_m)
            R_alt, t_alt, n_alt_inl, alt_debug = solve_pnp_fixed_rotation_center(
                inlier_query, enu_positions, fx, fy, cx_cam, cy_cam,
                _R_c2w.T, _cam_enu, prior_weight=_prior_weight,
                huber_m=_huber_m)
            if alt_debug is not None:
                pnp_rotation_check = {
                    'mode': 'transpose_R_c2w',
                    't_world': None if t_alt is None else t_alt.astype(np.float64).tolist(),
                    'n_inliers': int(n_alt_inl),
                    'diagnostics': alt_debug,
                }
                if pnp_debug is not None:
                    pnp_debug['rotation_check_transpose'] = {
                        'mode': pnp_rotation_check['mode'],
                        'n_inliers': int(n_alt_inl),
                        'diagnostics': alt_debug,
                    }
                print(
                    "[dom_matching][pnp-fixedR-rotcheck] "
                    f"cur(n={int(n_pnp_inl)}, "
                    f"reproj={pnp_debug.get('median_reproj_px', float('nan')) if pnp_debug else float('nan'):.1f}px, "
                    f"ray={pnp_debug.get('median_ray_residual_m', float('nan')) if pnp_debug else float('nan'):.1f}m, "
                    f"dxy={pnp_debug.get('delta_xy_m', float('nan')) if pnp_debug else float('nan'):.1f}m, "
                    f"dz={pnp_debug.get('delta_z_m', float('nan')) if pnp_debug else float('nan'):.1f}m) "
                    f"rt(n={int(n_alt_inl)}, "
                    f"reproj={alt_debug.get('median_reproj_px', float('nan')):.1f}px, "
                    f"ray={alt_debug.get('median_ray_residual_m', float('nan')):.1f}m, "
                    f"dxy={alt_debug.get('delta_xy_m', float('nan')):.1f}m, "
                    f"dz={alt_debug.get('delta_z_m', float('nan')):.1f}m)"
                )
            _accept = False
            if R_c2w is not None and t_world is not None and pnp_debug is not None:
                _max_delta_xy = max(100.0, min(
                    400.0, float(pred_pose_crop.get('max_center_shift_m', 400.0))))
                _max_delta_z = max(50.0, min(200.0, 0.50 * float(_pose_agl_m or 250.0)))
                _max_ray_res = max(60.0, min(150.0, 0.30 * float(_pose_agl_m or 250.0)))
                _max_reproj_px = float(pred_pose_crop.get('max_pnp_reproj_px', 40.0))
                _med_ray = float(pnp_debug.get('median_ray_residual_m', np.inf))
                _med_reproj = float(pnp_debug.get('median_reproj_px', np.inf))
                _pos_ratio = float(pnp_debug.get('positive_depth_ratio', 0.0))
                _delta_xy = float(pnp_debug.get('delta_xy_m', np.inf))
                _delta_z = abs(float(pnp_debug.get('delta_z_m', np.inf)))
                _accept = (
                    np.isfinite(_med_ray)
                    and np.isfinite(_med_reproj)
                    and _pos_ratio >= 0.50
                    and _med_reproj <= _max_reproj_px
                    and _delta_xy <= _max_delta_xy
                    and _delta_z <= _max_delta_z
                    and _med_ray <= _max_ray_res
                    and int(n_pnp_inl) >= 8
                )
                pnp_debug.update({
                    'accepted': bool(_accept),
                    'max_delta_xy_m': float(_max_delta_xy),
                    'max_delta_z_m': float(_max_delta_z),
                    'max_ray_residual_m': float(_max_ray_res),
                    'max_reproj_px': float(_max_reproj_px),
                    'prior_weight': float(_prior_weight),
                    'huber_m': float(_huber_m),
                })
            if _accept:
                print(
                    "[dom_matching][pnp-fixedR] accept "
                    f"n={int(n_pnp_inl)}/{int(pnp_debug.get('n_valid', 0))} "
                    f"pos={pnp_debug.get('positive_depth_ratio', 0.0):.2f} "
                    f"reproj={pnp_debug.get('median_reproj_px', float('nan')):.1f}px "
                    f"ray={pnp_debug.get('median_ray_residual_m', float('nan')):.1f}m "
                    f"dxy={pnp_debug.get('delta_xy_m', float('nan')):.1f}m "
                    f"dz={pnp_debug.get('delta_z_m', float('nan')):.1f}m"
                )
                pnp_result = {
                    'R_c2w': R_c2w,
                    't_world': t_world,
                    'n_inliers': n_pnp_inl,
                    'method': 'fixed_rotation_center',
                    'diagnostics': pnp_debug,
                }
            elif pnp_debug is not None:
                print(
                    "[dom_matching][pnp-fixedR] reject "
                    f"pos={pnp_debug.get('positive_depth_ratio', 0.0):.2f} "
                    f"reproj={pnp_debug.get('median_reproj_px', float('nan')):.1f}px "
                    f"ray={pnp_debug.get('median_ray_residual_m', float('nan')):.1f}m "
                    f"dxy={pnp_debug.get('delta_xy_m', float('nan')):.1f}m "
                    f"dz={pnp_debug.get('delta_z_m', float('nan')):.1f}m"
                )
        else:
            R_c2w, t_world, n_pnp_inl = solve_pnp_frame(
                inlier_query, enu_positions, fx, fy, cx_cam, cy_cam)
            if R_c2w is not None:
                pnp_result = {
                    'R_c2w': R_c2w,
                    't_world': t_world,
                    'n_inliers': n_pnp_inl,
                }

    result = {
        'frame_pixels': inlier_query,      # [K, 2] (col, row) in query image
        'enu_positions': enu_positions,     # [K, 2] or [K, 3] absolute ENU
        'is_road': is_road_arr,            # [K] bool or None; True = road surface
        'n_road_inliers': n_road_inliers,
        'building_filter_applied': bool(
            geo_elev is not None
            and dom_transform is not None
            and getattr(geo_elev, 'bld_data', None) is not None
        ),
        'frame_center_enu': np.array([fc_e, fc_n]),  # (2,) frame center in ENU
        'n_inliers': n_inliers,
        'pnp_disabled': bool(not compute_pnp),
        'n_building_filtered': n_bld_filtered,
        'query_W': query_W,
        'query_H': query_H,
        'heading_deg': heading_deg,
        'estimated_agl': estimated_agl,    # per-frame AGL from affine scale (m)
        'estimated_agl_source': estimated_agl_source,
        'dom_crop_info': {
            'mode': 'render' if _use_render else ('strict_pose' if (_use_pose_crop and bool(pred_pose_crop.get('strict_footprint', False))) else ('pose' if _use_pose_crop else 'nadir')),
            'strict_footprint': bool(pred_pose_crop.get('strict_footprint', False)) if pred_pose_crop is not None else False,
            'query_aligned_bbox': bool(_direct_rotated_crop),
            'col0': int(col0),
            'col1': int(col1),
            'row0': int(row0),
            'row1': int(row1),
            'yaw_applied_deg': float(rot_angle_ccw),
            'pose_half_w_m': None if _pose_half_w_m is None else float(_pose_half_w_m),
            'pose_half_h_m': None if _pose_half_h_m is None else float(_pose_half_h_m),
            'pose_center_shift_m': _pose_center_shift_m,
            'pose_agl_m': _pose_agl_m,
            'pose_crop_failure_reason': _pose_crop_failure_reason,
            'pose_center_ray_z': _pose_center_ray_z,
            'pose_center_ray_t': _pose_center_ray_t,
            'pose_cam_center_enu': None if _pose_cam_center_enu is None else _pose_cam_center_enu.tolist(),
            'pose_center_ground_enu': None if _pose_center_ground_enu is None else _pose_center_ground_enu.tolist(),
            'pose_image_center_ground_enu': None if _pose_center_ground_enu is None else _pose_center_ground_enu.tolist(),
            'pose_crop_center_ground_enu': None if _pose_crop_center_ground_enu is None else _pose_crop_center_ground_enu.tolist(),
            'pose_ground_z': _pose_ground_z,
            'render_failure_reason': None if _use_render else _render_failure_reason,
            'render_pitch_deg': _pitch_deg_render,
            'render_valid_ratio': _render_valid_ratio,
        },
    }
    if pnp_result is not None:
        result['pnp_result'] = pnp_result
    elif pnp_debug is not None:
        result['pnp_debug'] = pnp_debug
    if pnp_rotation_check is not None:
        result['pnp_rotation_check'] = pnp_rotation_check
    if enu_positions_road is not None and frame_pixels_road is not None:
        result['enu_positions_road'] = enu_positions_road
        result['frame_pixels_road'] = frame_pixels_road
    return result


def compute_dense_correspondences(image_paths, dom_image_cpu, gt_enu,
                                   project_fn, inv_project_fn,
                                   stride=5, device="cuda",
                                   crop_size_m=250, max_correspondences=500,
                                   save_vis_dir=None,
                                   camera_fov=None,
                                   geo_elev=None,
                                   dom_transform=None):
    """Pre-compute RoMa pixel↔ENU correspondences for selected frames.

    Loads RoMa once, matches every `stride`-th frame, then frees VRAM.
    Uses gt_enu only as approximate crop centres (not for training).

    Args:
        geo_elev: GeoElevationQuery for DEM + building mask (optional).
        dom_transform: rasterio transform for DOM (needed by geo_elev).

    Returns:
        dict: global_frame_idx → correspondence dict (or None for failures)
    """
    N = len(image_paths)
    selected = sorted(set(list(range(0, N, stride)) + [N - 1]))  # always include last frame

    print(f"[DOM-Geo] Loading RoMa outdoor model...")
    roma_model = load_roma(device=device)
    print(f"[DOM-Geo] Matching {len(selected)}/{N} frames (stride={stride})...")

    correspondences = {}
    dense_anchors = {}  # frame_idx → (2,) ENU position of frame center
    n_ok = 0
    for fi in selected:
        vis_path = None
        if save_vis_dir:
            vis_path = os.path.join(save_vis_dir, f"roma_match_f{fi:04d}.jpg")
        corr = get_frame_dom_correspondences(
            image_paths[fi], dom_image_cpu, gt_enu[fi],
            project_fn, inv_project_fn, roma_model,
            crop_size_m=crop_size_m, device=device,
            max_correspondences=max_correspondences,
            save_vis=vis_path,
            camera_fov=camera_fov,
            geo_elev=geo_elev,
            dom_transform=dom_transform,
        )
        if corr is not None:
            correspondences[fi] = corr
            if 'frame_center_enu' in corr:
                dense_anchors[fi] = corr['frame_center_enu']
            n_ok += 1
        else:
            print(f"  Frame {fi}: FAILED")

    del roma_model
    torch.cuda.empty_cache()
    if n_ok > 0:
        avg_corr = np.mean([c['n_inliers'] for c in correspondences.values()])
        print(f"[DOM-Geo] Done: {n_ok}/{len(selected)} frames matched "
              f"(avg {avg_corr:.0f} corr/frame), "
              f"{len(dense_anchors)} dense anchors")
    else:
        print(f"[DOM-Geo] Done: 0/{len(selected)} frames matched")
    return correspondences, dense_anchors
