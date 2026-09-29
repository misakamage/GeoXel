"""
Geo-Supervised Test-Time Training (Geo-TTT) for StreamVGGT.

想法5 — Cross-Modal TTT + 想法2 — DOM Confidence Gating

Core idea: pre-compute SIFT feature matches between camera images and DOM,
then use the matched correspondences as differentiable reprojection loss
to update poseLN_modulation at test time.

Why SIFT reprojection works where NCC fails:
  - At 435m altitude, 1m shift changes <0.3% of pixels → NCC is FLAT.
  - SIFT finds 100s of cross-domain keypoint matches (verified: 776 inliers).
  - Each match gives a DIRECT 2D position constraint with strong gradient.
  - Gradient: d(reproj_error)/d(poseLN) flows through:
    reproj_error → DOM_UV → world_xyz → camera_pose → poseLN_modulation

Architecture:
  Pre-compute (once per window):
    1. Aggregator → tokens (frozen, detached)
    2. Initial camera_head → poses → world positions
    3. For each frame: crop DOM at predicted position → SIFT match camera↔DOM
    4. Convert match positions to absolute DOM UV coordinates

  TTT loop (K gradient steps):
    1. camera_head(tokens) → pose → world_xyz (grad ON for poseLN)
    2. For each match: back-project camera pixel to ground → DOM UV (differentiable)
    3. Reprojection loss = mean ||predicted_DOM_UV - matched_DOM_UV||²
    4. Confidence gating: num_matches per frame modulates loss weight
    5. Backward → update poseLN_modulation
"""

import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List


@dataclass
class GeoTTTConfig:
    """Configuration for Geo-supervised Test-Time Training."""

    # === Optimization ===
    lr: float = 5e-5
    num_steps: int = 5
    warmup_frames: int = 2

    # === Adaptable parameters ===
    adapt_poseLN: bool = True
    adapt_pose_branch: bool = False

    # === Loss ===
    reproj_weight: float = 1.0         # weight for reprojection loss
    reg_weight: float = 0.01           # L2 reg toward pretrained
    loss_clamp_px: float = 50.0        # clamp individual match error (DOM pixels)

    # === SIFT matching ===
    sift_max_features: int = 2000
    sift_match_ratio: float = 0.75     # Lowe's ratio test threshold
    sift_ransac_thresh: float = 5.0    # RANSAC reprojection threshold (pixels)
    min_matches: int = 10              # skip frame if fewer matches

    # === Confidence gating (想法2) ===
    use_confidence_gating: bool = True
    confidence_ref_matches: int = 200  # frames with this many matches get full weight

    # === Camera / DOM geometry ===
    camera_fx: float = 1931.7
    camera_fy: float = 1931.7
    camera_orig_w: int = 1600
    camera_orig_h: int = 1200
    base_altitude_m: float = 435.0
    dom_gsd_m: float = 0.123

    # === Accumulation ===
    accumulate_params: bool = True


class GeoTTT:
    """
    Geo-supervised TTT using SIFT feature match reprojection loss.
    """

    def __init__(self, config: GeoTTTConfig, camera_head: nn.Module):
        self.config = config
        self._pretrained_state = {}
        self._save_pretrained(camera_head)

    # ------------------------------------------------------------------
    #  Parameter management
    # ------------------------------------------------------------------

    def _save_pretrained(self, camera_head: nn.Module):
        for name, param in camera_head.named_parameters():
            if self._should_adapt(name):
                self._pretrained_state[name] = param.data.clone()

    def _should_adapt(self, param_name: str) -> bool:
        cfg = self.config
        if cfg.adapt_poseLN and "poseLN_modulation" in param_name:
            return True
        if cfg.adapt_pose_branch and "pose_branch" in param_name:
            return True
        return False

    def reset_to_pretrained(self, camera_head: nn.Module):
        with torch.no_grad():
            for name, param in camera_head.named_parameters():
                if name in self._pretrained_state:
                    param.data.copy_(self._pretrained_state[name].to(param.device))

    # ------------------------------------------------------------------
    #  Geometry helpers
    # ------------------------------------------------------------------

    def _compute_ground_footprint(self, altitude_m: float) -> Tuple[float, float]:
        cfg = self.config
        ground_w = cfg.camera_orig_w / cfg.camera_fx * altitude_m
        ground_h = cfg.camera_orig_h / cfg.camera_fy * altitude_m
        return ground_w, ground_h

    # ------------------------------------------------------------------
    #  SIFT feature matching
    # ------------------------------------------------------------------

    def _sift_match_frame(
        self,
        cam_img_np: np.ndarray,
        dom_crop_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SIFT match between camera image and DOM crop.

        Args:
            cam_img_np: [H, W, 3] uint8 camera image
            dom_crop_np: [H, W, 3] uint8 DOM crop (same size as camera)

        Returns:
            cam_pts: [K, 2] matched keypoint positions in camera image (x, y)
            dom_pts: [K, 2] matched keypoint positions in DOM crop (x, y)
        """
        cfg = self.config
        sift = cv2.SIFT_create(nfeatures=cfg.sift_max_features)

        cam_gray = cv2.cvtColor(cam_img_np, cv2.COLOR_RGB2GRAY)
        dom_gray = cv2.cvtColor(dom_crop_np, cv2.COLOR_RGB2GRAY)

        kp1, des1 = sift.detectAndCompute(cam_gray, None)
        kp2, des2 = sift.detectAndCompute(dom_gray, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return np.zeros((0, 2)), np.zeros((0, 2))

        # FLANN-based matching
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < cfg.sift_match_ratio * n.distance:
                    good.append(m)

        if len(good) < cfg.min_matches:
            return np.zeros((0, 2)), np.zeros((0, 2))

        cam_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dom_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        # RANSAC geometric verification
        _, mask = cv2.findHomography(cam_pts, dom_pts, cv2.RANSAC,
                                     cfg.sift_ransac_thresh)
        if mask is None:
            return np.zeros((0, 2)), np.zeros((0, 2))

        mask = mask.ravel().astype(bool)
        return cam_pts[mask], dom_pts[mask]

    def _precompute_matches(
        self,
        model,
        frames: List[dict],
        agg_outputs: List[torch.Tensor],
        dom_image: torch.Tensor,
        project_fn: Callable,
        get_world_xyz_fn: Callable,
        heading_fn: Callable,
        dtype,
    ) -> List[dict]:
        """
        Pre-compute SIFT matches for all frames.

        For each frame:
          1. Get initial pose from camera_head (no grad)
          2. Crop DOM at predicted position (heading-rotated)
          3. Run SIFT matching camera ↔ DOM crop
          4. Convert DOM crop keypoints to absolute DOM pixel coords
          5. Store matched camera pixel positions and DOM pixel positions

        Returns:
            List of dicts per frame with:
              - cam_pts_model: [K, 2] camera keypoints in model image coords
              - dom_uv_abs: [K, 2] matched DOM positions in full DOM pixel coords
              - num_matches: int
        """
        cfg = self.config
        camera_head = model.camera_head
        device = next(camera_head.parameters()).device

        ground_w, ground_h = self._compute_ground_footprint(cfg.base_altitude_m)
        cam_crop_w_px = int(round(ground_w / cfg.dom_gsd_m))
        cam_crop_h_px = int(round(ground_h / cfg.dom_gsd_m))
        B_dom, C_dom, H_dom, W_dom = dom_image.shape

        all_matches = []

        with torch.no_grad():
            for i, tokens in enumerate(agg_outputs):
                token_input = [tokens.to(device)]
                past_kv_cam = [None] * camera_head.trunk_depth
                pose_enc_list, _ = camera_head(
                    token_input,
                    past_key_values_camera=past_kv_cam,
                    use_cache=True,
                )
                camera_pose = pose_enc_list[-1][:, 0, :].float()
                world_xyz = get_world_xyz_fn(camera_pose)  # [1, 3]

                # Project to DOM coordinates
                dom_uv_center = project_fn(world_xyz).float()  # [1, 2] col, row

                # DOM crop: camera-footprint-sized, heading-rotated
                heading_rad = heading_fn(i) if heading_fn else 0.0
                cx, cy = dom_uv_center[0, 0].item(), dom_uv_center[0, 1].item()

                # Build rotated crop grid
                half_w = cam_crop_w_px / 2.0
                half_h = cam_crop_h_px / 2.0

                # Grid of DOM pixel offsets from center (heading-rotated)
                cos_h = math.cos(heading_rad)
                sin_h = math.sin(heading_rad)

                # Camera image pixel → DOM pixel offset (with rotation)
                # The camera x-axis → sin/cos(heading) in DOM
                # Use uniform grid for crop extraction
                xs = np.linspace(-half_w, half_w, cam_crop_w_px)
                ys = np.linspace(-half_h, half_h, cam_crop_h_px)
                gx, gy = np.meshgrid(xs, ys)
                if abs(heading_rad) > 1e-6:
                    rx = gx * cos_h - gy * sin_h
                    ry = gx * sin_h + gy * cos_h
                else:
                    rx, ry = gx, gy

                dom_cols = (cx + rx).astype(int)
                dom_rows = (cy + ry).astype(int)

                # Clamp to DOM bounds
                dom_cols = np.clip(dom_cols, 0, W_dom - 1)
                dom_rows = np.clip(dom_rows, 0, H_dom - 1)

                # Extract DOM crop as numpy
                dom_np = dom_image[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                dom_crop_np = dom_np[dom_rows, dom_cols]  # [crop_h, crop_w, 3]
                dom_crop_np = (dom_crop_np * 255).clip(0, 255).astype(np.uint8)

                # Camera image as numpy
                cam_img = frames[i]["img"]  # [1, 3, H, W]
                cam_np = cam_img[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                cam_np = (cam_np * 255).clip(0, 255).astype(np.uint8)

                # Resize DOM crop to camera image size for SIFT matching
                cam_H, cam_W = cam_np.shape[:2]
                dom_crop_resized = cv2.resize(dom_crop_np, (cam_W, cam_H))

                # SIFT matching
                cam_pts, dom_pts_crop = self._sift_match_frame(cam_np, dom_crop_resized)

                if len(cam_pts) < cfg.min_matches:
                    all_matches.append({
                        "cam_pts_model": np.zeros((0, 2)),
                        "dom_uv_abs": np.zeros((0, 2)),
                        "num_matches": 0,
                    })
                    continue

                # Convert DOM crop pixel coords back to absolute DOM coords
                # dom_pts_crop is in resized crop coords [0, cam_W] × [0, cam_H]
                # Scale to original crop coords
                dom_pts_orig = dom_pts_crop.copy()
                dom_pts_orig[:, 0] *= cam_crop_w_px / cam_W
                dom_pts_orig[:, 1] *= cam_crop_h_px / cam_H

                # Convert from crop-relative to absolute DOM coords
                # Each crop pixel (u, v) corresponds to DOM pixel:
                # dom_col = cx + (u - half_w) * cos_h - (v - half_h) * sin_h
                # dom_row = cy + (u - half_w) * sin_h + (v - half_h) * cos_h
                du = dom_pts_orig[:, 0] - half_w
                dv = dom_pts_orig[:, 1] - half_h
                abs_col = cx + du * cos_h - dv * sin_h
                abs_row = cy + du * sin_h + dv * cos_h

                dom_uv_abs = np.stack([abs_col, abs_row], axis=-1)  # [K, 2]

                all_matches.append({
                    "cam_pts_model": cam_pts,      # [K, 2] in model img coords
                    "dom_uv_abs": dom_uv_abs,      # [K, 2] in full DOM pixel coords
                    "num_matches": len(cam_pts),
                })

        total_matches = sum(m["num_matches"] for m in all_matches)
        frames_with_matches = sum(1 for m in all_matches if m["num_matches"] > 0)
        print(f"    [GeoTTT] SIFT matches: {total_matches} total, "
              f"{frames_with_matches}/{len(frames)} frames with matches")
        return all_matches

    # ------------------------------------------------------------------
    #  Differentiable reprojection loss
    # ------------------------------------------------------------------

    def _reprojection_loss(
        self,
        world_xyz: torch.Tensor,
        match_info: dict,
        proj_J_E: torch.Tensor,
        proj_J_N: torch.Tensor,
        proj_uv0: torch.Tensor,
        cam_H: int,
        cam_W: int,
        heading_rad: float,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute differentiable reprojection loss from SIFT matches.

        For each match (u_cam, v_cam) ↔ (u_dom, v_dom):
          1. Back-project camera pixel to ground:
             ground = camera_center + pixel_offset(u_cam, v_cam)
          2. Project ground to DOM UV:
             pred_dom_uv = uv0 + ground_E * J_E + ground_N * J_N
          3. Error = ||pred_dom_uv - target_dom_uv||²

        Args:
            world_xyz: [B, 3] camera center in ENU (differentiable!)
            match_info: dict with cam_pts_model, dom_uv_abs
            proj_J_E, proj_J_N: affine Jacobians [1, 2]
            proj_uv0: DOM UV at ENU origin [1, 2]
            cam_H, cam_W: model image spatial dimensions
            heading_rad: camera heading in radians

        Returns:
            loss: scalar (mean squared reprojection error in DOM pixels)
            n_matches: number of matches used
        """
        cfg = self.config
        device = world_xyz.device

        cam_pts = match_info["cam_pts_model"]     # [K, 2] numpy
        dom_uv_target = match_info["dom_uv_abs"]  # [K, 2] numpy
        n_matches = len(cam_pts)

        if n_matches == 0:
            return torch.tensor(0.0, device=device), 0

        # Convert to tensors
        cam_pts_t = torch.tensor(cam_pts, dtype=torch.float32, device=device)
        target_uv = torch.tensor(dom_uv_target, dtype=torch.float32, device=device)

        # Camera intrinsics in model image coordinates
        fx_m = cfg.camera_fx * cam_W / cfg.camera_orig_w
        fy_m = cfg.camera_fy * cam_H / cfg.camera_orig_h
        cx_m = cam_W / 2.0
        cy_m = cam_H / 2.0
        alt = cfg.base_altitude_m

        # Back-project camera pixels to ground offsets (meters)
        dx_m = (cam_pts_t[:, 0] - cx_m) / fx_m * alt  # [K]
        dy_m = (cam_pts_t[:, 1] - cy_m) / fy_m * alt  # [K]

        # Apply heading rotation: camera → ENU
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        dE = dx_m * cos_h - dy_m * sin_h  # [K]
        dN = dx_m * sin_h + dy_m * cos_h  # [K]

        # Ground point = camera center + offset (differentiable!)
        ground_E = world_xyz[0, 0] + dE  # [K], grad flows through world_xyz
        ground_N = world_xyz[0, 1] + dN  # [K]

        # Project ground points to DOM UV (differentiable via affine)
        pred_uv_col = proj_uv0[0, 0] + ground_E * proj_J_E[0, 0] + ground_N * proj_J_N[0, 0]
        pred_uv_row = proj_uv0[0, 1] + ground_E * proj_J_E[0, 1] + ground_N * proj_J_N[0, 1]
        pred_uv = torch.stack([pred_uv_col, pred_uv_row], dim=-1)  # [K, 2]

        # Reprojection error (L2 in DOM pixels, clamped)
        errors = (pred_uv - target_uv).pow(2).sum(dim=-1)  # [K]
        errors = errors.clamp(max=cfg.loss_clamp_px ** 2)
        loss = errors.mean()

        return loss, n_matches

    # ------------------------------------------------------------------
    #  Main TTT loop
    # ------------------------------------------------------------------

    def adapt_window(
        self,
        model,
        frames: List[dict],
        dom_image: torch.Tensor,
        project_fn: Callable,
        anchor_world_xyz: torch.Tensor,
        get_world_xyz_fn: Callable,
        heading_fn: Callable = None,
        dtype=torch.bfloat16,
    ) -> dict:
        """
        Perform Geo-TTT adaptation at the start of a window.

        Pipeline:
          1. Pre-compute aggregator outputs (frozen)
          2. Pre-compute SIFT matches: camera ↔ DOM for each frame
          3. TTT gradient loop:
             a. camera_head → pose → world_xyz (poseLN has grad)
             b. Reprojection loss from SIFT matches (differentiable)
             c. Confidence gating by match count
             d. Backward → update poseLN_modulation
        """
        cfg = self.config
        camera_head = model.camera_head
        aggregator = model.aggregator
        device = next(camera_head.parameters()).device

        if len(frames) < cfg.warmup_frames:
            return {"skipped": True, "reason": "too_few_frames"}

        # --- Freeze all except adapted params ---
        for name, param in camera_head.named_parameters():
            param.requires_grad = self._should_adapt(name)
        for param in aggregator.parameters():
            param.requires_grad = False

        adapted_params = [p for p in camera_head.parameters() if p.requires_grad]
        if not adapted_params:
            return {"skipped": True, "reason": "no_params_to_adapt"}

        n_adapted = sum(p.numel() for p in adapted_params)
        print(f"  [GeoTTT] Adapting {len(adapted_params)} param groups "
              f"({n_adapted / 1e6:.1f}M params), lr={cfg.lr}, "
              f"steps={cfg.num_steps}")

        optimizer = torch.optim.Adam(adapted_params, lr=cfg.lr)

        # --- Pre-compute aggregator outputs (frozen, only last layer) ---
        agg_outputs = []
        past_kv = [None] * aggregator.depth
        with torch.no_grad():
            for i, frame in enumerate(frames):
                images = frame["img"].unsqueeze(0)
                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images,
                        past_key_values=past_kv,
                        use_cache=True,
                        past_frame_idx=i,
                    )
                if isinstance(agg_out, tuple) and len(agg_out) == 3:
                    tokens, psi, past_kv = agg_out
                else:
                    tokens, psi = agg_out
                if isinstance(tokens, list):
                    agg_outputs.append(tokens[-1].detach())
                else:
                    agg_outputs.append(tokens.detach())
        del past_kv
        torch.cuda.empty_cache()

        # --- Pre-compute differentiable affine ENU→DOM projection ---
        with torch.no_grad():
            uv0 = project_fn(torch.zeros(1, 3, device=device)).float()
            uv_x = project_fn(torch.tensor([[1.0, 0.0, 0.0]], device=device)).float()
            uv_y = project_fn(torch.tensor([[0.0, 1.0, 0.0]], device=device)).float()
            proj_J_E = (uv_x - uv0).detach().to(device)
            proj_J_N = (uv_y - uv0).detach().to(device)

        proj_uv0 = uv0.detach().to(device)

        cam_H, cam_W = frames[0]["img"].shape[-2], frames[0]["img"].shape[-1]

        # --- Pre-compute SIFT matches ---
        all_matches = self._precompute_matches(
            model, frames, agg_outputs,
            dom_image, project_fn, get_world_xyz_fn,
            heading_fn, dtype,
        )

        total_matches = sum(m["num_matches"] for m in all_matches)
        if total_matches < cfg.min_matches:
            print(f"    [GeoTTT] Too few matches ({total_matches}), skipping TTT")
            for param in camera_head.parameters():
                param.requires_grad = False
            return {"skipped": True, "reason": "too_few_matches",
                    "total_matches": total_matches}

        # --- Cast tokens to float32 ---
        for idx in range(len(agg_outputs)):
            agg_outputs[idx] = agg_outputs[idx].float()

        # ====== Geo-TTT gradient loop ======
        losses = []
        n_total_frames = len(agg_outputs)

        with torch.cuda.amp.autocast(enabled=False):
            for step in range(cfg.num_steps):
                optimizer.zero_grad()
                step_loss_sum = 0.0
                step_matches_sum = 0

                for i, tokens in enumerate(agg_outputs):
                    match_info = all_matches[i]
                    if match_info["num_matches"] == 0:
                        continue

                    # camera_head forward (poseLN has grad)
                    past_kv_cam = [None] * camera_head.trunk_depth
                    pose_enc_list, _ = camera_head(
                        [tokens],
                        past_key_values_camera=past_kv_cam,
                        use_cache=True,
                    )
                    camera_pose = pose_enc_list[-1][:, 0, :]  # [1, 9]

                    # Differentiable: pose → world ENU
                    world_xyz = get_world_xyz_fn(camera_pose)  # [1, 3]

                    heading_rad = heading_fn(i) if heading_fn else 0.0

                    # Reprojection loss (differentiable through world_xyz)
                    reproj_loss, n_matches = self._reprojection_loss(
                        world_xyz, match_info,
                        proj_J_E, proj_J_N, proj_uv0,
                        cam_H, cam_W, heading_rad,
                    )

                    if step == 0 and i == 0:
                        print(f"    [GeoTTT] pose.grad={camera_pose.requires_grad}, "
                              f"xyz.grad={world_xyz.requires_grad}, "
                              f"reproj_loss={reproj_loss.item():.2f} px², "
                              f"n_matches={n_matches}")

                    # Confidence gating (想法2): weight by match count
                    if cfg.use_confidence_gating:
                        gate = min(1.0, n_matches / cfg.confidence_ref_matches)
                    else:
                        gate = 1.0

                    weighted_loss = cfg.reproj_weight * reproj_loss * gate

                    # Backward per frame
                    (weighted_loss / n_total_frames).backward()
                    step_loss_sum += reproj_loss.item()
                    step_matches_sum += n_matches

                # L2 regularization toward pretrained
                if cfg.reg_weight > 0:
                    reg = torch.tensor(0.0, device=device)
                    for name, param in camera_head.named_parameters():
                        if name in self._pretrained_state:
                            diff = param - self._pretrained_state[name].to(device)
                            reg = reg + diff.pow(2).sum()
                    (cfg.reg_weight * reg).backward()

                # Gradient step
                grad_norms = []
                for p in adapted_params:
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm().item())
                    else:
                        grad_norms.append(0.0)
                total_grad = sum(grad_norms)

                n_frames_with_matches = sum(
                    1 for m in all_matches if m["num_matches"] > 0
                )
                avg_loss = (step_loss_sum / n_frames_with_matches
                            if n_frames_with_matches > 0 else 0.0)

                print(f"    [GeoTTT step {step}] "
                      f"reproj_err={avg_loss:.2f} px², "
                      f"matches={step_matches_sum}, "
                      f"grad_norm={total_grad:.6e}, "
                      f"n_grad={sum(1 for g in grad_norms if g > 0)}/{len(grad_norms)}")

                torch.nn.utils.clip_grad_norm_(adapted_params, max_norm=1.0)
                optimizer.step()
                losses.append(avg_loss)

        # Cleanup & re-freeze
        for param in camera_head.parameters():
            param.requires_grad = False

        return {
            "skipped": False,
            "num_steps": cfg.num_steps,
            "losses": losses,
            "total_matches": total_matches,
            "confidences": [total_matches / len(frames)],  # avg matches per frame
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
            "final_confidence": total_matches / len(frames),
        }
