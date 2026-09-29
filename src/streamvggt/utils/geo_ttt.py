"""
Geo-Supervised Test-Time Training for StreamVGGT — KV-Cache TTT.

想法5: Cross-Modal TTT (DOM as external geo-supervision target)
想法2: DOM-Conditioned Confidence Gating

=== Core Innovation (TTT3R-inspired) ===
TTT3R treats state as "fast weights" updated by gradient descent:
    S_t = S_{t-1} - beta_t * grad_S L(S_{t-1}, X_t)

We extend this to StreamVGGT's camera head KV-cache with EXTERNAL
geo-supervision from DOM (Digital Orthophoto Map):
    KV_cam = KV_cam - eta * grad_KV L_dom

The camera head's trunk blocks use KV-cache to attend to historical
frames' camera tokens. By optimizing these KV entries using DOM
reprojection loss, we refine the spatial representation that determines
how current frames relate to past frames — improving pose accuracy.

=== Architecture ===
Phase 1: Aggregator forward (frozen) -> aggregated tokens per frame
Phase 2: Camera head forward (frozen) -> build camera KV-cache + initial poses
Phase 3: SIFT feature matching: camera image <-> DOM crop
Phase 4: Make camera KV entries into differentiable leaf tensors
Phase 5: TTT gradient loop:
    - Re-run camera trunk using external KV-cache (gradient-enabled)
    - SIFT reprojection loss: world_xyz -> DOM UV -> compare with matched points
    - Backprop gradients to KV entries + poseLN parameters
    - Update KV (manual SGD) and poseLN (Adam)
Phase 6: Re-compute all poses using adapted KV + poseLN

=== Why KV-cache TTT works where previous approaches failed ===
- NCC landscape is flat at 435m altitude (1m shift = <0.3% pixel change)
- poseLN-only adaptation: 12.6M params with ~200 constraints = underdetermined
- KV-cache TTT: directly modifies historical attention context, structured
  gradient, combined with poseLN for both representation and modulation control
"""

import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List

from streamvggt.heads.head_act import activate_pose


def _modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


@dataclass
class GeoTTTConfig:
    """Configuration for Geo-supervised Test-Time Training."""

    # === KV-cache TTT ===
    kv_lr: float = 1e-1
    num_steps: int = 3
    kv_update_layers: str = "all"  # "all", "last_half", "last"

    # === poseLN adaptation (complementary) ===
    adapt_poseLN: bool = True
    poseLN_lr: float = 5e-5

    # === Loss ===
    reg_weight: float = 0.01
    loss_clamp_px: float = 50.0

    # === SIFT matching ===
    sift_max_features: int = 2000
    sift_match_ratio: float = 0.75
    sift_ransac_thresh: float = 5.0
    min_matches: int = 10

    # === Confidence gating (想法2) ===
    use_confidence_gating: bool = True
    confidence_ref_matches: int = 200

    # === Camera / DOM geometry ===
    camera_fx: float = 1931.7
    camera_fy: float = 1931.7
    camera_orig_w: int = 1600
    camera_orig_h: int = 1200
    base_altitude_m: float = 435.0
    dom_gsd_m: float = 0.123

    # === Misc ===
    warmup_frames: int = 2
    accumulate_params: bool = False


class GeoTTT:
    """
    Geo-supervised TTT: update camera head KV-cache + poseLN using DOM
    reprojection loss. Treats KV-cache as TTT3R-style fast weights.
    """

    def __init__(self, config: GeoTTTConfig, camera_head: nn.Module):
        self.config = config
        self._pretrained_poseLN = {}
        for name, param in camera_head.named_parameters():
            if "poseLN_modulation" in name:
                self._pretrained_poseLN[name] = param.data.clone()
        self._adapted_cam_kv = None
        self._agg_outputs = None

    def reset_poseLN(self, camera_head: nn.Module):
        """Reset poseLN to pretrained weights."""
        with torch.no_grad():
            for name, param in camera_head.named_parameters():
                if name in self._pretrained_poseLN:
                    param.data.copy_(self._pretrained_poseLN[name].to(param.device))

    def _get_update_layer_indices(self, trunk_depth: int) -> List[int]:
        cfg = self.config
        if cfg.kv_update_layers == "all":
            return list(range(trunk_depth))
        elif cfg.kv_update_layers == "last_half":
            return list(range(trunk_depth // 2, trunk_depth))
        elif cfg.kv_update_layers == "last":
            return [trunk_depth - 1]
        return list(range(trunk_depth))

    # ------------------------------------------------------------------
    #  Phase 2: Build camera KV-cache
    # ------------------------------------------------------------------

    def _build_cam_kv_cache(
        self,
        camera_head: nn.Module,
        agg_outputs: List[torch.Tensor],
    ) -> Tuple[list, List[torch.Tensor]]:
        """
        Run camera head for all frames -> build KV-cache + initial poses.

        Returns:
            cam_kv: list of (K, V) per trunk block
            initial_poses: list of [1, 9] pose tensors
        """
        cam_kv = [None] * camera_head.trunk_depth
        initial_poses = []

        with torch.no_grad():
            for tokens in agg_outputs:
                pose_enc_list, cam_kv = camera_head(
                    [tokens],
                    past_key_values_camera=cam_kv,
                    use_cache=True,
                )
                pose = pose_enc_list[-1][:, 0, :].float()
                initial_poses.append(pose)

        return cam_kv, initial_poses

    # ------------------------------------------------------------------
    #  Phase 5: Custom trunk forward with external KV (supports grad)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_trunk_with_kv(
        camera_head: nn.Module,
        pose_tokens: torch.Tensor,
        cam_kv_cache: list,
        frame_idx: int,
        num_iters: int = 4,
    ) -> torch.Tensor:
        """
        Re-run camera head trunk for one frame using external KV-cache.

        This is the core of KV-cache TTT: the gradient flows from
        the attention output through the KV-cache leaf tensors.

        Gradient path:
            loss -> camera_pose -> pose_branch -> trunk_norm -> x
            -> trunk_block attention using F.scaled_dot_product_attention
            -> K_full = cat(past_K_slice, current_K) -> past_K_slice
            -> KV-cache leaf tensor (gradient computed!)

        Args:
            camera_head: CameraHead module (poseLN may have grad)
            pose_tokens: [B, 1, C] normalized camera token
            cam_kv_cache: list of (K, V) per trunk block
                K shape: [B, heads, 4*N_frames, 1, head_dim]
            frame_idx: which frame (0-indexed)
            num_iters: number of refinement iterations (default 4)

        Returns:
            activated_pose: [B, 1, 9]
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None

        for iter_idx in range(num_iters):
            if pred_pose_enc is None:
                module_input = camera_head.embed_pose(
                    camera_head.empty_pose_tokens.expand(B, S, -1)
                )
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = camera_head.embed_pose(pred_pose_enc)

            shift, scale, gate = camera_head.poseLN_modulation(
                module_input
            ).chunk(3, dim=-1)

            x = gate * _modulate(
                camera_head.adaln_norm(pose_tokens), shift, scale
            )
            x = x + pose_tokens

            # Causal KV: use entries from all past frames + current
            # frame's previous iterations
            entry_idx = frame_idx * num_iters + iter_idx

            for block_idx in range(camera_head.trunk_depth):
                kv_entry = cam_kv_cache[block_idx]
                if kv_entry is not None and entry_idx > 0:
                    k_full, v_full = kv_entry
                    past_kv = (
                        k_full[:, :, :entry_idx].contiguous(),
                        v_full[:, :, :entry_idx].contiguous(),
                    )
                else:
                    past_kv = None

                # Block forward: attention uses past_kv (grad flows!)
                # Returned new_kv is discarded (read-only pass)
                x, _ = camera_head.trunk[block_idx](
                    x, past_key_values=past_kv, use_cache=True
                )

            pred_pose_enc_delta = camera_head.pose_branch(
                camera_head.trunk_norm(x)
            )

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

        activated_pose = activate_pose(
            pred_pose_enc,
            trans_act=camera_head.trans_act,
            quat_act=camera_head.quat_act,
            fl_act=camera_head.fl_act,
        )
        return activated_pose

    # ------------------------------------------------------------------
    #  SIFT matching
    # ------------------------------------------------------------------

    def _sift_match_frame(
        self,
        cam_img_np: np.ndarray,
        dom_crop_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SIFT match between camera image and DOM crop."""
        cfg = self.config
        sift = cv2.SIFT_create(nfeatures=cfg.sift_max_features)

        cam_gray = cv2.cvtColor(cam_img_np, cv2.COLOR_RGB2GRAY)
        dom_gray = cv2.cvtColor(dom_crop_np, cv2.COLOR_RGB2GRAY)

        kp1, des1 = sift.detectAndCompute(cam_gray, None)
        kp2, des2 = sift.detectAndCompute(dom_gray, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return np.zeros((0, 2)), np.zeros((0, 2))

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

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

        _, mask = cv2.findHomography(
            cam_pts, dom_pts, cv2.RANSAC, cfg.sift_ransac_thresh
        )
        if mask is None:
            return np.zeros((0, 2)), np.zeros((0, 2))

        mask = mask.ravel().astype(bool)
        return cam_pts[mask], dom_pts[mask]

    def _precompute_matches(
        self,
        frames: List[dict],
        initial_poses: List[torch.Tensor],
        dom_image: torch.Tensor,
        project_fn: Callable,
        get_world_xyz_fn: Callable,
        heading_fn: Callable,
    ) -> List[dict]:
        """
        Pre-compute SIFT matches for all frames using initial poses.

        For each frame:
          1. Use initial pose -> world_xyz -> DOM center
          2. Extract heading-rotated DOM crop at camera footprint scale
          3. SIFT match camera image <-> DOM crop
          4. Convert match coords to absolute DOM pixel coords
        """
        cfg = self.config
        ground_w = cfg.camera_orig_w / cfg.camera_fx * cfg.base_altitude_m
        ground_h = cfg.camera_orig_h / cfg.camera_fy * cfg.base_altitude_m
        cam_crop_w_px = int(round(ground_w / cfg.dom_gsd_m))
        cam_crop_h_px = int(round(ground_h / cfg.dom_gsd_m))
        _, C_dom, H_dom, W_dom = dom_image.shape

        dom_np_raw = dom_image[0].permute(1, 2, 0).cpu().numpy()
        # Handle both uint8 and float DOM tensor formats
        if dom_np_raw.dtype == np.uint8:
            dom_np_u8 = dom_np_raw
        else:
            dom_np_u8 = (dom_np_raw * 255).clip(0, 255).astype(np.uint8)

        all_matches = []

        with torch.no_grad():
            for i, pose in enumerate(initial_poses):
                world_xyz = get_world_xyz_fn(pose)
                dom_uv_center = project_fn(world_xyz).float()
                cx = dom_uv_center[0, 0].item()
                cy = dom_uv_center[0, 1].item()

                heading_rad = heading_fn(i) if heading_fn else 0.0
                cos_h = math.cos(heading_rad)
                sin_h = math.sin(heading_rad)
                half_w = cam_crop_w_px / 2.0
                half_h = cam_crop_h_px / 2.0

                xs = np.linspace(-half_w, half_w, cam_crop_w_px)
                ys = np.linspace(-half_h, half_h, cam_crop_h_px)
                gx, gy = np.meshgrid(xs, ys)
                if abs(heading_rad) > 1e-6:
                    rx = gx * cos_h - gy * sin_h
                    ry = gx * sin_h + gy * cos_h
                else:
                    rx, ry = gx, gy

                dom_cols = np.clip((cx + rx).astype(int), 0, W_dom - 1)
                dom_rows = np.clip((cy + ry).astype(int), 0, H_dom - 1)
                dom_crop_np = dom_np_u8[dom_rows, dom_cols]

                cam_img = frames[i]["img"]
                cam_np = cam_img[0].permute(1, 2, 0).cpu().numpy()
                cam_np = (cam_np * 255).clip(0, 255).astype(np.uint8)
                cam_H, cam_W = cam_np.shape[:2]

                dom_crop_resized = cv2.resize(dom_crop_np, (cam_W, cam_H))
                cam_pts, dom_pts_crop = self._sift_match_frame(
                    cam_np, dom_crop_resized
                )

                if len(cam_pts) < cfg.min_matches:
                    all_matches.append({
                        "cam_pts": np.zeros((0, 2)),
                        "dom_uv_abs": np.zeros((0, 2)),
                        "num_matches": 0,
                        "heading_rad": heading_rad,
                    })
                    continue

                # Convert DOM crop coords -> absolute DOM coords
                dom_pts_orig = dom_pts_crop.copy()
                dom_pts_orig[:, 0] *= cam_crop_w_px / cam_W
                dom_pts_orig[:, 1] *= cam_crop_h_px / cam_H

                du = dom_pts_orig[:, 0] - half_w
                dv = dom_pts_orig[:, 1] - half_h
                abs_col = cx + du * cos_h - dv * sin_h
                abs_row = cy + du * sin_h + dv * cos_h
                dom_uv_abs = np.stack([abs_col, abs_row], axis=-1)

                all_matches.append({
                    "cam_pts": cam_pts,
                    "dom_uv_abs": dom_uv_abs,
                    "num_matches": len(cam_pts),
                    "heading_rad": heading_rad,
                })

        total = sum(m["num_matches"] for m in all_matches)
        n_with = sum(1 for m in all_matches if m["num_matches"] > 0)
        print(f"    [GeoTTT] SIFT: {total} matches, "
              f"{n_with}/{len(frames)} frames with matches")
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
    ) -> Tuple[torch.Tensor, int]:
        """
        Differentiable reprojection loss from SIFT matches.

        For each match (u_cam, v_cam) <-> (u_dom, v_dom):
          1. Back-project camera pixel to ground
          2. Apply heading rotation -> ENU offset
          3. ground_point = camera_center + offset (differentiable!)
          4. Project to DOM UV via affine (differentiable!)
          5. Error = ||predicted_DOM_UV - matched_DOM_UV||^2
        """
        cfg = self.config
        device = world_xyz.device

        cam_pts = match_info["cam_pts"]
        dom_uv_target = match_info["dom_uv_abs"]
        heading_rad = match_info["heading_rad"]
        n_matches = len(cam_pts)

        if n_matches == 0:
            return torch.tensor(0.0, device=device), 0

        cam_pts_t = torch.tensor(cam_pts, dtype=torch.float32, device=device)
        target_uv = torch.tensor(dom_uv_target, dtype=torch.float32, device=device)

        fx_m = cfg.camera_fx * cam_W / cfg.camera_orig_w
        fy_m = cfg.camera_fy * cam_H / cfg.camera_orig_h
        cx_m = cam_W / 2.0
        cy_m = cam_H / 2.0
        alt = cfg.base_altitude_m

        dx_m = (cam_pts_t[:, 0] - cx_m) / fx_m * alt
        dy_m = (cam_pts_t[:, 1] - cy_m) / fy_m * alt

        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        # Camera axes for nadir-looking camera at heading θ:
        #   image-right → (cos θ, -sin θ) in (E, N)
        #   image-down  → (-sin θ, -cos θ) in (E, N)
        # So ground offset = dx_m*(cos θ, -sin θ) + dy_m*(-sin θ, -cos θ)
        dE = dx_m * cos_h - dy_m * sin_h
        dN = -dx_m * sin_h - dy_m * cos_h

        ground_E = world_xyz[0, 0] + dE
        ground_N = world_xyz[0, 1] + dN

        pred_col = (proj_uv0[0, 0]
                    + ground_E * proj_J_E[0, 0]
                    + ground_N * proj_J_N[0, 0])
        pred_row = (proj_uv0[0, 1]
                    + ground_E * proj_J_E[0, 1]
                    + ground_N * proj_J_N[0, 1])
        pred_uv = torch.stack([pred_col, pred_row], dim=-1)

        # Use per-match L2 distance (pixels), then Smooth L1 (Huber) for robustness.
        # Huber loss: L2-like for small errors, L1-like for large errors.
        # This ensures gradient never goes to zero even for large errors.
        per_match_dist = torch.sqrt((pred_uv - target_uv).pow(2).sum(dim=-1) + 1e-6)
        loss = F.smooth_l1_loss(
            per_match_dist,
            torch.zeros_like(per_match_dist),
            beta=cfg.loss_clamp_px,
            reduction='mean',
        )

        return loss, n_matches

    # ------------------------------------------------------------------
    #  Main TTT loop — KV-cache gradient update
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
        Geo-supervised TTT: update camera KV-cache + poseLN.

        Implements the TTT3R formula on StreamVGGT's camera head:
            KV_cam <- KV_cam - eta * grad_KV L_dom

        where L_dom is SIFT reprojection loss against DOM.

        The gradient flows through:
            L_dom -> world_xyz -> camera_pose -> pose_branch
            -> trunk blocks -> F.scaled_dot_product_attention(Q, K, V)
            -> KV-cache leaf tensors
        """
        cfg = self.config
        camera_head = model.camera_head
        aggregator = model.aggregator
        device = next(camera_head.parameters()).device

        if len(frames) < cfg.warmup_frames:
            return {"skipped": True, "reason": "too_few_frames"}

        # Reset poseLN to pretrained weights at the start of each window
        # to prevent drift across windows (unless accumulate is enabled)
        if not cfg.accumulate_params:
            self.reset_poseLN(camera_head)

        # ==========================================================
        #  Phase 1: Aggregator forward (frozen)
        # ==========================================================
        print(f"  [GeoTTT] Phase 1: Aggregator for {len(frames)} frames...")
        agg_outputs = []
        past_kv_agg = [None] * aggregator.depth
        with torch.no_grad():
            for i, frame in enumerate(frames):
                images = frame["img"].unsqueeze(0)
                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images,
                        past_key_values=past_kv_agg,
                        use_cache=True,
                        past_frame_idx=i,
                    )
                if isinstance(agg_out, tuple) and len(agg_out) == 3:
                    tokens, psi, past_kv_agg = agg_out
                else:
                    tokens, psi = agg_out
                last_tok = tokens[-1] if isinstance(tokens, list) else tokens
                agg_outputs.append(last_tok.detach())
        del past_kv_agg
        torch.cuda.empty_cache()

        # Store for inference
        self._agg_outputs = agg_outputs

        # ==========================================================
        #  Phase 2: Build camera KV-cache + initial poses
        # ==========================================================
        print(f"  [GeoTTT] Phase 2: Building camera KV-cache...")
        cam_kv_cache, initial_poses = self._build_cam_kv_cache(
            camera_head, agg_outputs
        )

        # ==========================================================
        #  Phase 3: SIFT matching
        # ==========================================================
        print(f"  [GeoTTT] Phase 3: SIFT matching...")
        all_matches = self._precompute_matches(
            frames, initial_poses, dom_image,
            project_fn, get_world_xyz_fn, heading_fn,
        )

        total_matches = sum(m["num_matches"] for m in all_matches)
        if total_matches < cfg.min_matches:
            print(f"    [GeoTTT] Too few matches ({total_matches}), skipping")
            return {"skipped": True, "reason": "too_few_matches",
                    "total_matches": total_matches}

        # ==========================================================
        #  Phase 4: Make KV entries differentiable leaf tensors
        # ==========================================================
        update_layers = self._get_update_layer_indices(camera_head.trunk_depth)
        kv_leaves = []
        kv_originals = []

        for block_idx in range(camera_head.trunk_depth):
            if cam_kv_cache[block_idx] is not None:
                k_old, v_old = cam_kv_cache[block_idx]
                if block_idx in update_layers:
                    k_new = k_old.detach().float().requires_grad_(True)
                    v_new = v_old.detach().float().requires_grad_(True)
                    cam_kv_cache[block_idx] = (k_new, v_new)
                    kv_leaves.extend([k_new, v_new])
                    kv_originals.extend([k_new.data.clone(), v_new.data.clone()])
                else:
                    cam_kv_cache[block_idx] = (
                        k_old.detach().float(), v_old.detach().float()
                    )

        n_kv = sum(p.numel() for p in kv_leaves)
        print(f"    [GeoTTT] KV leaves: {len(kv_leaves)} tensors, "
              f"{n_kv/1e6:.2f}M params (layers {update_layers})")

        # Enable grad for poseLN
        for name, param in camera_head.named_parameters():
            param.requires_grad = (
                "poseLN_modulation" in name and cfg.adapt_poseLN
            )
        poseLN_params = [
            p for p in camera_head.parameters() if p.requires_grad
        ]
        poseLN_optimizer = None
        if poseLN_params:
            poseLN_optimizer = torch.optim.Adam(
                poseLN_params, lr=cfg.poseLN_lr
            )
            n_poseLN = sum(p.numel() for p in poseLN_params)
            print(f"    [GeoTTT] poseLN: {n_poseLN/1e6:.1f}M params")

        # Use Adam for KV-cache too (handles vanishing gradients through deep network)
        kv_optimizer = torch.optim.Adam(kv_leaves, lr=cfg.kv_lr)

        # Precompute affine ENU -> DOM projection
        with torch.no_grad():
            uv0 = project_fn(torch.zeros(1, 3, device=device)).float()
            uv_x = project_fn(
                torch.tensor([[1.0, 0.0, 0.0]], device=device)
            ).float()
            uv_y = project_fn(
                torch.tensor([[0.0, 1.0, 0.0]], device=device)
            ).float()
            proj_J_E = (uv_x - uv0).detach()
            proj_J_N = (uv_y - uv0).detach()
            proj_uv0 = uv0.detach()

        cam_H = frames[0]["img"].shape[-2]
        cam_W = frames[0]["img"].shape[-1]

        # Cast aggregated tokens to float32
        for idx in range(len(agg_outputs)):
            agg_outputs[idx] = agg_outputs[idx].float()

        # ==========================================================
        #  Phase 5: TTT gradient loop
        # ==========================================================
        losses = []
        n_total = len(frames)

        print(f"  [GeoTTT] Phase 5: TTT loop ({cfg.num_steps} steps, "
              f"kv_lr={cfg.kv_lr}, poseLN_lr={cfg.poseLN_lr})...")

        with torch.cuda.amp.autocast(enabled=False):
            for step in range(cfg.num_steps):
                # Zero gradients
                if poseLN_optimizer:
                    poseLN_optimizer.zero_grad()
                kv_optimizer.zero_grad()

                step_loss = 0.0
                step_matches = 0
                frames_with_loss = 0

                for i in range(n_total):
                    match_info = all_matches[i]
                    if match_info["num_matches"] == 0:
                        continue

                    tokens = agg_outputs[i]
                    # Extract and normalize camera token
                    pose_tokens = camera_head.token_norm(tokens[:, :, 0])

                    # Run trunk with external KV (grad enabled)
                    activated_pose = self._run_trunk_with_kv(
                        camera_head, pose_tokens, cam_kv_cache, i
                    )

                    camera_pose = activated_pose[:, 0, :]  # [1, 9]
                    world_xyz = get_world_xyz_fn(camera_pose)

                    reproj_loss, n_matches = self._reprojection_loss(
                        world_xyz, match_info,
                        proj_J_E, proj_J_N, proj_uv0,
                        cam_H, cam_W,
                    )

                    if step == 0 and i == 0:
                        print(f"    [GeoTTT] reproj_loss={reproj_loss.item():.2f} px2, "
                              f"n_matches={n_matches}, "
                              f"pose.grad={camera_pose.requires_grad}, "
                              f"xyz.grad={world_xyz.requires_grad}")

                    # Confidence gating (想法2)
                    gate = (
                        min(1.0, n_matches / cfg.confidence_ref_matches)
                        if cfg.use_confidence_gating else 1.0
                    )

                    (reproj_loss * gate / n_total).backward()

                    step_loss += reproj_loss.item()
                    step_matches += n_matches
                    frames_with_loss += 1

                # L2 regularization
                if cfg.reg_weight > 0:
                    reg = torch.tensor(0.0, device=device)
                    # poseLN reg
                    if cfg.adapt_poseLN:
                        for name, param in camera_head.named_parameters():
                            if name in self._pretrained_poseLN:
                                diff = param - self._pretrained_poseLN[name].to(device)
                                reg = reg + diff.pow(2).sum()
                    # KV reg
                    for idx, p in enumerate(kv_leaves):
                        reg = reg + (p - kv_originals[idx].to(device)).pow(2).sum()
                    (cfg.reg_weight * reg).backward()

                # Update poseLN (Adam)
                if poseLN_optimizer:
                    torch.nn.utils.clip_grad_norm_(
                        poseLN_params, max_norm=1.0
                    )
                    poseLN_optimizer.step()

                # Update KV-cache (Adam — handles vanishing gradients)
                kv_grad_norm = 0.0
                kv_updated = 0
                for p in kv_leaves:
                    if p.grad is not None:
                        g = p.grad.norm().item()
                        kv_grad_norm += g
                        if g > 0:
                            kv_updated += 1
                kv_optimizer.step()

                avg_loss = step_loss / max(frames_with_loss, 1)
                losses.append(avg_loss)

                poseLN_grad = sum(
                    p.grad.norm().item()
                    for p in poseLN_params if p.grad is not None
                ) if poseLN_params else 0.0

                print(f"    [GeoTTT step {step}] "
                      f"reproj_err={avg_loss:.2f} px2, "
                      f"kv_grad={kv_grad_norm:.4e}, "
                      f"kv_updated={kv_updated}/{len(kv_leaves)}, "
                      f"poseLN_grad={poseLN_grad:.4e}, "
                      f"matches={step_matches}")

        # ==========================================================
        #  Phase 6: Store adapted state
        # ==========================================================
        self._adapted_cam_kv = []
        for block_idx in range(camera_head.trunk_depth):
            if cam_kv_cache[block_idx] is not None:
                k, v = cam_kv_cache[block_idx]
                self._adapted_cam_kv.append(
                    (k.detach(), v.detach())
                )
            else:
                self._adapted_cam_kv.append(None)

        # Freeze all
        for param in camera_head.parameters():
            param.requires_grad = False

        torch.cuda.empty_cache()

        return {
            "skipped": False,
            "num_steps": cfg.num_steps,
            "losses": losses,
            "total_matches": total_matches,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
            "final_confidence": total_matches / len(frames),
            "confidences": [total_matches / len(frames)],
        }

    # ------------------------------------------------------------------
    #  Inference with adapted state
    # ------------------------------------------------------------------

    def compute_adapted_poses(
        self,
        camera_head: nn.Module,
    ) -> List[torch.Tensor]:
        """
        Re-compute poses for all frames using adapted KV + poseLN.

        Uses the stored KV-cache and aggregated tokens from adapt_window().
        No re-running of the aggregator needed.
        """
        cam_kv = self._adapted_cam_kv
        agg_outputs = self._agg_outputs

        if cam_kv is None or agg_outputs is None:
            raise RuntimeError(
                "Must call adapt_window() before compute_adapted_poses()"
            )

        poses = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            for i, tokens in enumerate(agg_outputs):
                tokens_f = tokens.float()
                pose_tokens = camera_head.token_norm(tokens_f[:, :, 0])
                activated_pose = self._run_trunk_with_kv(
                    camera_head, pose_tokens, cam_kv, i
                )
                poses.append(activated_pose[:, 0, :])

        return poses
