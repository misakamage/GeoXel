"""
DOM State Manager for StreamVGGT.

Ported from TTT3R's dom_state_feat approach:
- Maintains a fixed-size world-anchor state from DOM (Digital Orthophoto Map) crops.
- Updates slowly (EMA) to prevent drift while avoiding single-frame noise.
- Fuses with aggregated tokens before heads for world-coordinate-aware inference.
- Provides conservative XY translation correction on pose output.

All operations are parameter-free (inference-only, no training required).
"""

import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class DOMStateConfig:
    """Configuration for DOM state management."""

    # --- State update ---
    update_rate: float = 0.1          # EMA rate for dom_state update (slow anchor)
    min_confidence: float = 0.3       # Minimum similarity to accept DOM candidate
    fusion_strength: float = 0.25     # How strongly dom_state influences tokens

    # --- DOM crop ---
    crop_size: int = 224              # DOM crop size in pixels (for state candidate)
    context_size: int = 448           # Larger context for local search

    # --- Pose XY correction ---
    pose_xy_blend_strength: float = 0.1   # Weight of DOM XY correction
    pose_xy_max_step_m: float = 5.0       # Max single-frame XY correction in meters
    search_radius_px: int = 12            # Local search radius in DOM pixels
    search_step_px: int = 4               # Local search grid step
    search_min_confidence: float = 0.25   # Minimum confidence for search result
    jacobian_step_m: float = 1.0          # Finite-difference step for UV→world Jacobian

    # --- Camera intrinsics (original image resolution) ---
    camera_fx: float = 1931.7             # Focal length in pixels (x)
    camera_fy: float = 1931.7             # Focal length in pixels (y)
    camera_orig_w: int = 1600             # Original image width before preprocessing
    camera_orig_h: int = 1200             # Original image height before preprocessing
    base_altitude_m: float = 435.0        # RTK altitude of first frame (above ground)

    # --- IPM-based XY correction ---
    dom_gsd_m: float = 0.123              # DOM ground sample distance (meters/pixel)
    ipm_search_m: float = 700.0           # Search region (meters) allows ±(search-tpl)/2 correction
    ipm_use_edges: bool = True            # Match on edge maps instead of RGB
    ncc_search_scale: float = 2.0         # NCC search region = ncc_search_scale × camera footprint

    # --- Feedback ---
    feedback_enabled: bool = True
    feedback_alpha: float = 0.05          # Strength of XY correction feedback into state
    feedback_schedule_onset: int = 3      # Frame index to start enabling feedback
    feedback_schedule_k: float = 0.5      # Sigmoid slope for onset


class DOMState:
    """
    Lightweight container for DOM world-anchor state.
    No learnable parameters — all operations are inference-only.
    """

    def __init__(self, config: DOMStateConfig = None):
        self.config = config or DOMStateConfig()
        self.state: Optional[torch.Tensor] = None       # [B, N, C]
        self.reliability: Optional[torch.Tensor] = None  # [B]
        self.frame_idx: int = 0

    def reset(self):
        self.state = None
        self.reliability = None
        self.frame_idx = 0


class DOMStateManager:
    """
    Manages the DOM state lifecycle for StreamVGGT streaming inference.

    Lifecycle per frame:
        1. crop_and_encode_dom()    → DOM crop features (dom_candidate)
        2. compute_reliability()    → cosine similarity between query and DOM
        3. update_state()           → slow EMA update of dom_state
        4. fuse_to_tokens()         → fuse dom_state into aggregated tokens
        5. compute_xy_correction()  → optional pose XY correction from local search
        6. apply_feedback()         → write XY correction back into state
    """

    def __init__(self, config: DOMStateConfig = None):
        self.config = config or DOMStateConfig()

    # ------------------------------------------------------------------ #
    #  1. Initialize state from first frame
    # ------------------------------------------------------------------ #
    def init_state(
        self,
        dom_state: DOMState,
        initial_tokens: torch.Tensor,
        dom_image: torch.Tensor,
        world_xyz: torch.Tensor,
        project_fn: Callable,
        encode_fn: Callable,
        camera_hw: Tuple[int, int] = (392, 518),
    ) -> DOMState:
        """
        Initialize dom_state from the first frame's DOM crop.

        The DOM crop covers the same ground area as the camera footprint
        (ipm_template_m) so that DINOv2 patches have matching spatial scales.

        Args:
            dom_state: DOMState container to initialize.
            initial_tokens: Aggregated tokens of the first frame [B, P, C].
            dom_image: Full DOM image [B, 3, H_dom, W_dom].
            world_xyz: World coordinates of the first camera [B, 3].
            project_fn: world_xyz → DOM UV coordinate [B, 2].
            encode_fn: DOM crop [B, 3, crop_h, crop_w] → features [B, N, C].
            camera_hw: (H, W) of the preprocessed camera image.
        """
        crop = self._crop_dom_at_camera_scale(
            dom_image, world_xyz, project_fn, camera_hw,
        )
        dom_features = encode_fn(crop)  # [B, N, C]
        dom_state.state = dom_features.clone()
        dom_state.reliability = torch.ones(
            dom_features.shape[0], device=dom_features.device, dtype=dom_features.dtype
        )
        dom_state.frame_idx = 0
        return dom_state

    # ------------------------------------------------------------------ #
    #  2. Build candidate from current frame's DOM crop
    # ------------------------------------------------------------------ #
    def build_candidate(
        self,
        dom_image: torch.Tensor,
        world_xyz: torch.Tensor,
        query_tokens: torch.Tensor,
        project_fn: Callable,
        encode_fn: Callable,
        camera_hw: Tuple[int, int] = (392, 518),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a DOM state candidate from the current estimated position.

        Returns:
            candidate: DOM crop features [B, N, C].
            reliability: Cosine similarity gate [B].
        """
        crop = self._crop_dom_at_camera_scale(
            dom_image, world_xyz, project_fn, camera_hw,
        )
        candidate = encode_fn(crop)  # [B, N, C]
        reliability = self._compute_reliability(query_tokens, candidate)
        return candidate, reliability

    # ------------------------------------------------------------------ #
    #  3. Update state (slow EMA)
    # ------------------------------------------------------------------ #
    def update_state(
        self,
        dom_state: DOMState,
        candidate: torch.Tensor,
        reliability: torch.Tensor,
    ) -> DOMState:
        """
        Slowly update dom_state via EMA, gated by reliability.

        dom_state = dom_state + α * reliability * (candidate - dom_state)
        """
        if dom_state.state is None:
            dom_state.state = candidate.clone()
            dom_state.reliability = reliability.clone()
            return dom_state

        alpha = (
            reliability[:, None, None] * self.config.update_rate
        ).clamp(0.0, 1.0)

        dom_state.state = dom_state.state + alpha * (candidate - dom_state.state)
        dom_state.reliability = reliability.clone()
        dom_state.frame_idx += 1
        return dom_state

    # ------------------------------------------------------------------ #
    #  4. Fuse dom_state into aggregated tokens
    # ------------------------------------------------------------------ #
    def fuse_to_tokens(
        self,
        tokens: torch.Tensor,
        dom_state: DOMState,
        patch_start_idx: int = 0,
    ) -> torch.Tensor:
        """
        Fuse dom_state into aggregated tokens before they reach the heads.

        Only modifies PATCH tokens (skipping camera + register special tokens)
        to avoid corrupting the special token semantics.

        tokens shape: [B, S, P, 2C] or list of such tensors.
        dom_state shape: [B, N, C].
        patch_start_idx: index where patch tokens begin (skip camera/register tokens).
        """
        if dom_state.state is None or dom_state.reliability is None:
            return tokens

        reliability = dom_state.reliability
        alpha = (
            reliability * self.config.fusion_strength
        ).clamp(0.0, 1.0)

        # Compute dom_state global summary: mean pool → [B, C]
        dom_global = dom_state.state.mean(dim=1)  # [B, C]

        if isinstance(tokens, list):
            fused = []
            for t in tokens:
                fused.append(self._fuse_single(t, dom_global, alpha, patch_start_idx))
            return fused
        else:
            return self._fuse_single(tokens, dom_global, alpha, patch_start_idx)

    def _fuse_single(
        self,
        tokens: torch.Tensor,
        dom_global: torch.Tensor,
        alpha: torch.Tensor,
        patch_start_idx: int = 0,
    ) -> torch.Tensor:
        """Fuse dom_global into patch tokens AND camera token (index 0).

        The camera head only reads token index 0 (camera token), so we
        propagate the mean patch-fusion shift to the camera token as well.
        """
        C_dom = dom_global.shape[-1]
        C_tok = tokens.shape[-1]

        if C_dom * 2 == C_tok:
            # DOM features (C) → expand to 2C by replicating into both halves
            dom_expanded = torch.cat([dom_global, dom_global], dim=-1)  # [B, 2C]
        elif C_dom == C_tok:
            dom_expanded = dom_global
        else:
            return tokens

        result = tokens.clone()

        if tokens.dim() == 4:
            B, S, P, CC = tokens.shape
            patch_tokens = tokens[:, :, patch_start_idx:, :]  # [B, S, P_patch, 2C]
            patch_mean = patch_tokens.mean(dim=2, keepdim=True)  # [B, S, 1, 2C]
            alpha_b = alpha[:, None, None, None]
            dom_b = dom_expanded[:, None, None, :]
            shift = alpha_b * (dom_b - patch_mean)
            result[:, :, patch_start_idx:, :] = patch_tokens + shift
            # Propagate mean shift to camera token so camera_head sees DOM info
            mean_shift = shift.mean(dim=2, keepdim=True)  # [B, S, 1, 2C]
            result[:, :, 0:1, :] = result[:, :, 0:1, :] + mean_shift
        elif tokens.dim() == 3:
            patch_tokens = tokens[:, patch_start_idx:, :]
            patch_mean = patch_tokens.mean(dim=1, keepdim=True)
            alpha_b = alpha[:, None, None]
            dom_b = dom_expanded[:, None, :]
            shift = alpha_b * (dom_b - patch_mean)
            result[:, patch_start_idx:, :] = patch_tokens + shift
            mean_shift = shift.mean(dim=1, keepdim=True)
            result[:, 0:1, :] = result[:, 0:1, :] + mean_shift

        return result

    # ------------------------------------------------------------------ #
    #  5. Image-space DOM NCC for XY correction (nadir-to-nadir)
    # ------------------------------------------------------------------ #
    def compute_xy_correction(
        self,
        dom_image: torch.Tensor,
        world_xyz: torch.Tensor,
        camera_image: torch.Tensor,
        heading_rad: float,
        project_fn: Callable,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Estimate XY translation correction by matching the camera image against
        a DOM search region in image space using edge-based FFT NCC.

        Both camera and DOM are nadir aerial images, so direct pixel-level
        matching is appropriate.  We work at **camera GSD** (downsample DOM
        to match camera resolution) so both images have the same ground scale.

        Steps:
          1. Crop a large DOM region (search_m in meters) at DOM resolution.
          2. Downsample DOM crop to camera GSD.
          3. Rotate DOM to align with camera heading.
          4. Camera image is the template (already at camera GSD).
          5. Sobel edge maps for robust matching.
          6. FFT-NCC → peak offset in camera pixels → world meters.

        Args:
            dom_image: Full DOM [B, 3, H_dom, W_dom] (may be on CPU).
            world_xyz: Current estimated position [B, 3] (on GPU).
            camera_image: Preprocessed camera image [B, 3, H_cam, W_cam] (on GPU).
            heading_rad: Camera heading in radians (clockwise from North).
            project_fn: world_xyz [B, 3] → DOM UV [B, 2] in pixels.

        Returns:
            delta_xy: World XY correction [B, 2] in meters, or None.
            confidence: NCC peak confidence [B].
        """
        cfg = self.config
        target_device = world_xyz.device

        B, _, H_cam, W_cam = camera_image.shape

        # Camera GSD from intrinsics and altitude
        altitude = cfg.base_altitude_m + world_xyz[:, 2].mean().item()
        ground_w, _ = self._compute_ground_footprint(altitude)
        cam_gsd = ground_w / W_cam  # meters per preprocessed camera pixel
        search_m = cfg.ipm_search_m

        # 1. Crop DOM region at DOM resolution
        dom_crop_px = int(round(search_m / cfg.dom_gsd_m))
        dom_crop = self._crop_dom_patch(
            dom_image, world_xyz, project_fn, dom_crop_px,
        )  # [B, 3, dom_crop_px, dom_crop_px] on target_device

        # 2. Downsample DOM crop to camera GSD
        search_work_px = int(round(search_m / cam_gsd))
        dom_work = F.interpolate(
            dom_crop.float(), size=(search_work_px, search_work_px),
            mode="bilinear", align_corners=False,
        )  # [B, 3, search_work_px, search_work_px]

        # 3. Rotate DOM to align with camera heading
        #    heading_rad is clockwise from North; DOM is North-up.
        if abs(heading_rad) > 0.01:
            cos_h = math.cos(heading_rad)
            sin_h = math.sin(heading_rad)
            theta = torch.tensor(
                [[cos_h, -sin_h, 0], [sin_h, cos_h, 0]],
                dtype=torch.float32, device=target_device,
            ).unsqueeze(0).expand(B, -1, -1)
            rot_grid = F.affine_grid(theta, dom_work.shape, align_corners=False)
            dom_work = F.grid_sample(
                dom_work, rot_grid, mode="bilinear",
                padding_mode="zeros", align_corners=False,
            )

        # 4. Camera image is the template (at camera GSD)
        cam_tpl = camera_image.float()  # [B, 3, H_cam, W_cam]

        # 5. Edge maps for robust matching
        if cfg.ipm_use_edges:
            dom_edges = self._sobel_edge_map(dom_work)  # [B, 1, H, W]
            cam_edges = self._sobel_edge_map(cam_tpl)   # [B, 1, h, w]
        else:
            dom_edges = dom_work.mean(dim=1, keepdim=True)
            cam_edges = cam_tpl.mean(dim=1, keepdim=True)

        # 6. FFT NCC
        ncc_map = self._fft_ncc(dom_edges, cam_edges)  # [B, H_out, W_out]

        # 7. Find peak
        ncc_flat = ncc_map.reshape(B, -1)
        peak_vals, peak_idxs = ncc_flat.max(dim=-1)
        peak_rows = peak_idxs // ncc_map.shape[-1]
        peak_cols = peak_idxs % ncc_map.shape[-1]

        center_row = (ncc_map.shape[-2] - 1) / 2.0
        center_col = (ncc_map.shape[-1] - 1) / 2.0

        offset_col = peak_cols.float() - center_col
        offset_row = peak_rows.float() - center_row

        # 8. Sub-pixel refinement (in camera pixels)
        best_offset = torch.stack([offset_col, offset_row], dim=-1)
        refined = self._subpixel_refine(ncc_map, peak_rows, peak_cols, best_offset)

        # Convert camera-pixel offset back to DOM-pixel offset for UV→world
        # accounting for heading rotation
        dom_scale = cfg.dom_gsd_m / cam_gsd  # DOM pixels per camera pixel
        if abs(heading_rad) > 0.01:
            cos_h = math.cos(-heading_rad)  # inverse rotation
            sin_h = math.sin(-heading_rad)
            uv_dx = refined[:, 0] * cos_h - refined[:, 1] * sin_h
            uv_dy = refined[:, 0] * sin_h + refined[:, 1] * cos_h
            uv_offset = torch.stack([uv_dx, uv_dy], dim=-1) * dom_scale
        else:
            uv_offset = refined * dom_scale

        # 9. Confidence
        confidence = peak_vals.clamp(0.0, 1.0)
        confidence = torch.where(
            confidence >= cfg.search_min_confidence,
            confidence,
            torch.zeros_like(confidence),
        )

        if confidence.max() <= 0.0:
            return None, confidence

        # 10. Convert DOM pixel offset to world meters
        delta_xy = self._uv_offset_to_world_delta(
            world_xyz, uv_offset, project_fn, cfg.jacobian_step_m,
        )

        # 11. Clamp step size
        if cfg.pose_xy_max_step_m > 0:
            delta_norm = delta_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scale = torch.minimum(
                torch.ones_like(delta_norm),
                torch.full_like(delta_norm, cfg.pose_xy_max_step_m) / delta_norm,
            )
            delta_xy = delta_xy * scale

        delta_xy = delta_xy * confidence[:, None]
        return delta_xy, confidence

    # ------------------------------------------------------------------ #
    #  5b. NCC snap-back (direct cam_gsd conversion, no Jacobian)
    # ------------------------------------------------------------------ #
    def compute_ncc_snapback(
        self,
        dom_image: torch.Tensor,
        world_xyz: torch.Tensor,
        camera_image: torch.Tensor,
        heading_rad: float,
        project_fn: Callable,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Estimate XY correction by NCC matching, converting pixel offset to
        meters via camera GSD directly (no Jacobian).

        The search region is ``ncc_search_scale × camera footprint`` so that
        the DOM crop always covers a sensible neighbourhood regardless of
        absolute trajectory length.

        Offset convention (DOM is North-up, y increases downward):
            delta_east  =  offset_col  × cam_gsd
            delta_north = -offset_row  × cam_gsd

        When heading != 0 the DOM is rotated to align with the camera before
        NCC, so the resulting offset is in the camera frame; we rotate it back
        to ENU before returning.

        Returns:
            delta_xy: [B, 2] ENU (east, north) correction in metres, or None.
            confidence: [B] NCC peak value.
        """
        cfg = self.config
        target_device = world_xyz.device
        B, _, H_cam, W_cam = camera_image.shape

        # Camera GSD
        altitude = cfg.base_altitude_m + world_xyz[:, 2].mean().item()
        ground_w, ground_h = self._compute_ground_footprint(altitude)
        cam_gsd = ground_w / W_cam  # metres per preprocessed camera pixel

        # Search region = ncc_search_scale × camera footprint (use the larger dim)
        search_m = max(ground_w, ground_h) * cfg.ncc_search_scale

        # 1. Square DOM crop at DOM resolution
        dom_crop_px = int(round(search_m / cfg.dom_gsd_m))
        dom_crop = self._crop_dom_patch(
            dom_image, world_xyz, project_fn, dom_crop_px,
        )  # [B, 3, dom_crop_px, dom_crop_px]

        # 2. Downsample to camera GSD
        search_work_px = int(round(search_m / cam_gsd))
        dom_work = F.interpolate(
            dom_crop.float(), size=(search_work_px, search_work_px),
            mode="bilinear", align_corners=False,
        )

        # 3. Rotate DOM to align with camera heading
        if abs(heading_rad) > 0.01:
            cos_h = math.cos(heading_rad)
            sin_h = math.sin(heading_rad)
            theta = torch.tensor(
                [[cos_h, -sin_h, 0], [sin_h, cos_h, 0]],
                dtype=torch.float32, device=target_device,
            ).unsqueeze(0).expand(B, -1, -1)
            rot_grid = F.affine_grid(theta, dom_work.shape, align_corners=False)
            dom_work = F.grid_sample(
                dom_work, rot_grid, mode="bilinear",
                padding_mode="zeros", align_corners=False,
            )

        # 4. Edge maps
        cam_tpl = camera_image.float()
        if cfg.ipm_use_edges:
            dom_edges = self._sobel_edge_map(dom_work)
            cam_edges = self._sobel_edge_map(cam_tpl)
        else:
            dom_edges = dom_work.mean(dim=1, keepdim=True)
            cam_edges = cam_tpl.mean(dim=1, keepdim=True)

        # 5. FFT NCC
        ncc_map = self._fft_ncc(dom_edges, cam_edges)  # [B, H_out, W_out]

        # 6. Peak
        ncc_flat = ncc_map.reshape(B, -1)
        peak_vals, peak_idxs = ncc_flat.max(dim=-1)
        peak_rows = peak_idxs // ncc_map.shape[-1]
        peak_cols = peak_idxs % ncc_map.shape[-1]

        center_row = (ncc_map.shape[-2] - 1) / 2.0
        center_col = (ncc_map.shape[-1] - 1) / 2.0

        offset_col = peak_cols.float() - center_col
        offset_row = peak_rows.float() - center_row

        best_offset = torch.stack([offset_col, offset_row], dim=-1)
        refined = self._subpixel_refine(ncc_map, peak_rows, peak_cols, best_offset)
        # refined: [B, 2] = (offset_col, offset_row) in camera-GSD pixels

        # 7. Confidence gate
        confidence = peak_vals.clamp(0.0, 1.0)
        confidence = torch.where(
            confidence >= cfg.search_min_confidence,
            confidence,
            torch.zeros_like(confidence),
        )
        if confidence.max() <= 0.0:
            return None, confidence

        # 8. Direct cam_gsd conversion → camera-frame delta
        #    offset_col → east-ish, offset_row → south (DOM y-down)
        dx_cam = refined[:, 0] * cam_gsd   # metres, right in camera view
        dy_cam = -refined[:, 1] * cam_gsd  # metres, forward (North if heading=0)

        # 9. Rotate camera-frame delta back to ENU
        if abs(heading_rad) > 0.01:
            cos_h = math.cos(-heading_rad)
            sin_h = math.sin(-heading_rad)
            delta_east  = dx_cam * cos_h - dy_cam * sin_h
            delta_north = dx_cam * sin_h + dy_cam * cos_h
        else:
            delta_east = dx_cam
            delta_north = dy_cam

        delta_xy = torch.stack([delta_east, delta_north], dim=-1)  # [B, 2]

        # 10. Clamp step
        if cfg.pose_xy_max_step_m > 0:
            delta_norm = delta_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scale = torch.minimum(
                torch.ones_like(delta_norm),
                torch.full_like(delta_norm, cfg.pose_xy_max_step_m) / delta_norm,
            )
            delta_xy = delta_xy * scale

        delta_xy = delta_xy * confidence[:, None]
        return delta_xy, confidence

    # ------------------------------------------------------------------ #
    #  6. State feedback — write XY correction back into state
    # ------------------------------------------------------------------ #
    def apply_feedback(
        self,
        dom_state: DOMState,
        corrected_world_xyz: Optional[torch.Tensor],
        confidence: torch.Tensor,
        dom_image: torch.Tensor,
        project_fn: Callable,
        encode_fn: Callable,
        camera_hw: Tuple[int, int] = (392, 518),
    ) -> DOMState:
        """
        Write DOM XY correction feedback by re-cropping DOM at the corrected
        position and blending the new crop into state.

        This keeps feedback in the proper feature space (DOM crop features)
        instead of injecting arbitrary sinusoidal signals.
        """
        if not self.config.feedback_enabled:
            return dom_state
        if corrected_world_xyz is None or dom_state.state is None:
            return dom_state

        cfg = self.config
        device = dom_state.state.device

        # Sigmoid schedule
        frame_t = torch.tensor(float(dom_state.frame_idx), device=device)
        schedule = torch.sigmoid(
            (frame_t - cfg.feedback_schedule_onset) * cfg.feedback_schedule_k
        )

        gate = (cfg.feedback_alpha * confidence * schedule).clamp(0.0, 1.0)

        # Re-crop DOM at corrected position (camera-scale) and encode
        crop = self._crop_dom_at_camera_scale(
            dom_image, corrected_world_xyz, project_fn, camera_hw,
        )
        corrected_features = encode_fn(crop)  # [B, N, C]

        # Blend corrected features into state (stays in feature space)
        dom_state.state = dom_state.state + gate[:, None, None] * (
            corrected_features - dom_state.state
        )

        return dom_state

    # ------------------------------------------------------------------ #
    #  Blend XY correction into pose output
    # ------------------------------------------------------------------ #
    def blend_pose_xy(
        self,
        pose_params: torch.Tensor,
        delta_xy_world: Optional[torch.Tensor],
        confidence: torch.Tensor,
        dom_state: DOMState,
    ) -> torch.Tensor:
        """
        Conservatively blend DOM XY correction into the pose output.

        IMPORTANT: pose_params [B, 9] = [tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w]
        where T = (tx, ty, tz) is the translation in camera-from-world extrinsic,
        NOT the camera center in world coordinates.

        delta_xy_world is in ENU world space. We must transform it:
            T_cam = R @ c_world + t  =>  delta_T = R @ delta_c_world
        where R is the rotation from pose_params.

        Args:
            pose_params: Raw pose encoding from camera_head [B, 9].
            delta_xy_world: World XY correction [B, 2] in ENU meters, or None.
            confidence: Search confidence [B].
            dom_state: Current DOM state for schedule.

        Returns:
            Corrected pose_params [B, 9].
        """
        if delta_xy_world is None:
            return pose_params

        cfg = self.config
        device = pose_params.device

        # Frame schedule
        frame_t = torch.tensor(float(dom_state.frame_idx), device=device)
        schedule = torch.sigmoid(
            (frame_t - cfg.feedback_schedule_onset) * cfg.feedback_schedule_k
        )

        blend_w = (
            cfg.pose_xy_blend_strength * confidence * schedule
        ).clamp(0.0, 1.0)

        # Extract rotation from pose_params to transform world delta to camera space
        quat = pose_params[:, 3:7]  # [B, 4]
        from streamvggt.utils.rotation import quat_to_mat
        R = quat_to_mat(quat)  # [B, 3, 3]

        # World XY correction → 3D world vector (Z=0)
        delta_world_3d = torch.zeros(delta_xy_world.shape[0], 3,
                                     device=device, dtype=delta_xy_world.dtype)
        delta_world_3d[:, 0] = delta_xy_world[:, 0]
        delta_world_3d[:, 1] = delta_xy_world[:, 1]

        # Transform to camera space: delta_t_cam = R @ delta_c_world
        # Note: camera center c = -R^T @ t, so t = -R @ c
        # When c shifts by delta_c, t shifts by delta_t = -R @ delta_c
        delta_t_cam = -torch.bmm(R, delta_world_3d.unsqueeze(-1)).squeeze(-1)  # [B, 3]

        corrected = pose_params.clone()
        corrected[:, 0] = corrected[:, 0] + blend_w * delta_t_cam[:, 0]
        corrected[:, 1] = corrected[:, 1] + blend_w * delta_t_cam[:, 1]
        corrected[:, 2] = corrected[:, 2] + blend_w * delta_t_cam[:, 2]

        return corrected

    # ================================================================== #
    #  Internal helpers
    # ================================================================== #
    def _crop_dom_patch(
        self,
        dom_image: torch.Tensor,
        world_xyz: torch.Tensor,
        project_fn: Callable,
        crop_size: int,
    ) -> torch.Tensor:
        """
        Crop a square patch from dom_image centered at the projected world_xyz.

        Supports cross-device: dom_image may be on CPU (to save GPU memory)
        while world_xyz is on GPU.  The returned crop is on world_xyz's device.

        Args:
            dom_image: [B, 3, H, W] full DOM (may be on CPU).
            world_xyz: [B, 3] world coordinates (target device).
            project_fn: world_xyz [B, 3] → UV [B, 2] in DOM pixels.
            crop_size: Target crop size.

        Returns:
            Cropped patch [B, 3, crop_size, crop_size] on world_xyz's device.
        """
        target_device = world_xyz.device
        dom_dev = dom_image.device
        B, C, H, W = dom_image.shape
        uv = project_fn(world_xyz).to(dom_dev)  # [B, 2] on dom device

        half = (crop_size - 1) / 2.0

        # Pre-crop a bounding box in the original dtype to avoid full-image float conversion
        cx = uv[0, 0].item()
        cy = uv[0, 1].item()
        margin = int(half) + 2
        x0 = max(int(cx - margin), 0)
        y0 = max(int(cy - margin), 0)
        x1 = min(int(cx + margin) + 1, W)
        y1 = min(int(cy + margin) + 1, H)
        dom_crop_region = dom_image[:, :, y0:y1, x0:x1].float()
        if dom_image.dtype == torch.uint8:
            dom_crop_region = dom_crop_region / 255.0

        # Adjust UV for the cropped region
        uv_local = uv.clone()
        uv_local[:, 0] -= x0
        uv_local[:, 1] -= y0

        crop_H, crop_W = dom_crop_region.shape[2], dom_crop_region.shape[3]

        ys = torch.linspace(-half, half, crop_size, device=dom_dev, dtype=torch.float32)
        xs = torch.linspace(-half, half, crop_size, device=dom_dev, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [crop, crop, 2]

        pixel_grid = base_grid[None] + uv_local[:, None, None, :]  # [B, crop, crop, 2]

        # Normalize to [-1, 1] for grid_sample
        grid_norm_x = (pixel_grid[..., 0] / max(crop_W - 1, 1)) * 2.0 - 1.0
        grid_norm_y = (pixel_grid[..., 1] / max(crop_H - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_norm_x, grid_norm_y], dim=-1)  # [B, crop, crop, 2]

        crop = F.grid_sample(
            dom_crop_region,
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return crop.to(target_device)

    def _compute_ground_footprint(self, altitude_m: float) -> Tuple[float, float]:
        """Compute camera ground coverage (width, height) in meters from intrinsics."""
        cfg = self.config
        ground_w = cfg.camera_orig_w / cfg.camera_fx * altitude_m
        ground_h = cfg.camera_orig_h / cfg.camera_fy * altitude_m
        return ground_w, ground_h

    def _crop_dom_at_camera_scale(
        self,
        dom_image: torch.Tensor,
        world_xyz: torch.Tensor,
        project_fn: Callable,
        camera_hw: Tuple[int, int] = (392, 518),
    ) -> torch.Tensor:
        """
        Crop DOM at camera-footprint scale and resize to camera input size.

        The crop rectangle is computed dynamically from camera intrinsics
        and altitude, matching the camera's ground footprint so that DINOv2
        patches cover the same ground area for both camera and DOM.

        Args:
            dom_image: [B, 3, H, W] full DOM.
            world_xyz: [B, 3] world coordinates.
            project_fn: world_xyz → DOM UV [B, 2].
            camera_hw: (H, W) of preprocessed camera image.

        Returns:
            crop: [B, 3, H_cam, W_cam] on world_xyz's device.
        """
        H_cam, W_cam = camera_hw
        target_device = world_xyz.device
        dom_dev = dom_image.device
        B, C, H_dom, W_dom = dom_image.shape

        # Per-frame altitude = base + ENU z
        altitude = self.config.base_altitude_m + world_xyz[:, 2].mean().item()
        ground_w, ground_h = self._compute_ground_footprint(altitude)

        # DOM pixels corresponding to ground footprint
        dom_crop_w = int(round(ground_w / self.config.dom_gsd_m))
        dom_crop_h = int(round(ground_h / self.config.dom_gsd_m))

        # Project center position to DOM UV
        uv = project_fn(world_xyz).to(dom_dev)  # [B, 2]  (col, row)

        # Pre-crop bounding box in original dtype to avoid full-image float conversion
        cx = uv[0, 0].item()
        cy = uv[0, 1].item()
        margin_w = int(dom_crop_w / 2) + 2
        margin_h = int(dom_crop_h / 2) + 2
        x0 = max(int(cx - margin_w), 0)
        y0 = max(int(cy - margin_h), 0)
        x1 = min(int(cx + margin_w) + 1, W_dom)
        y1 = min(int(cy + margin_h) + 1, H_dom)
        dom_region = dom_image[:, :, y0:y1, x0:x1].float()
        if dom_image.dtype == torch.uint8:
            dom_region = dom_region / 255.0
        region_H, region_W = dom_region.shape[2], dom_region.shape[3]

        # Adjust UV for cropped region
        uv_local = uv.clone().float()
        uv_local[:, 0] -= x0
        uv_local[:, 1] -= y0

        # Build rectangular grid centered on UV
        half_w = (dom_crop_w - 1) / 2.0
        half_h = (dom_crop_h - 1) / 2.0
        xs = torch.linspace(-half_w, half_w, dom_crop_w, device=dom_dev, dtype=torch.float32)
        ys = torch.linspace(-half_h, half_h, dom_crop_h, device=dom_dev, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [dom_crop_h, dom_crop_w, 2]

        pixel_grid = base_grid[None] + uv_local[:, None, None, :]  # [B, H, W, 2]

        # Normalize to [-1, 1] for grid_sample
        grid_norm_x = (pixel_grid[..., 0] / max(region_W - 1, 1)) * 2.0 - 1.0
        grid_norm_y = (pixel_grid[..., 1] / max(region_H - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_norm_x, grid_norm_y], dim=-1)

        crop = F.grid_sample(
            dom_region, grid.float(),
            mode="bilinear", padding_mode="border", align_corners=True,
        )  # [B, 3, dom_crop_h, dom_crop_w]

        # Resize to camera input size
        crop_resized = F.interpolate(
            crop.float(), size=(H_cam, W_cam),
            mode="bilinear", align_corners=False,
        )
        return crop_resized.to(target_device)

    def _compute_reliability(
        self,
        query_tokens: torch.Tensor,
        dom_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity between query and DOM global features."""
        q = query_tokens.mean(dim=1)  # [B, C_q]
        d = dom_tokens.mean(dim=1)    # [B, C_d]
        # Aggregator output is 2C (frame+global); DOM features are C.
        # Use the first C dims (frame-attention half) for fair comparison.
        if q.shape[-1] != d.shape[-1]:
            d_dim = d.shape[-1]
            q = q[..., :d_dim]
        q_global = F.normalize(q, dim=-1)
        d_global = F.normalize(d, dim=-1)
        sim = F.cosine_similarity(q_global, d_global, dim=-1)
        reliability = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
        reliability = torch.where(
            reliability >= self.config.min_confidence,
            reliability,
            torch.zeros_like(reliability),
        )
        return reliability

    def _sobel_edge_map(self, img: torch.Tensor) -> torch.Tensor:
        """Compute Sobel edge magnitude from an RGB image.

        Args:
            img: [B, C, H, W] float image.

        Returns:
            edges: [B, 1, H, W] edge magnitude.
        """
        # Convert to grayscale
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img[:, 0:1]

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=gray.dtype, device=gray.device,
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=gray.dtype, device=gray.device,
        ).reshape(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        return (gx.pow(2) + gy.pow(2)).sqrt()

    def _fft_ncc(
        self,
        context: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-channel Normalized Cross-Correlation (NCC) via FFT.

        Works for both RGB images (C=3) and high-dimensional feature maps (C=1024).
        Cross-correlation and local statistics are summed across channels.

        Args:
            context: Search region [B, C, H_ctx, W_ctx].
            template: Template patch [B, C, H_tpl, W_tpl].

        Returns:
            ncc_map: [B, H_ctx - H_tpl + 1, W_ctx - W_tpl + 1], values in [-1, 1].
        """
        B, C, H_ctx, W_ctx = context.shape
        _, _, H_tpl, W_tpl = template.shape

        n = H_tpl * W_tpl

        # Template statistics per channel, then sum
        tpl_mean = template.mean(dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
        tpl_centered = template - tpl_mean
        # Sum of squared deviations across all channels and spatial positions
        tpl_energy = tpl_centered.pow(2).sum(dim=1, keepdim=True).sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)  # [B, 1, 1, 1]

        # FFT size
        fft_h = 1 << (H_ctx + H_tpl - 2).bit_length()
        fft_w = 1 << (W_ctx + W_tpl - 2).bit_length()

        # For high-channel features, process in chunks to avoid huge FFT memory
        chunk_size = min(C, 64)
        out_h = H_ctx - H_tpl + 1
        out_w = W_ctx - W_tpl + 1
        r0, c0 = H_tpl - 1, W_tpl - 1

        cross_corr_sum = torch.zeros(B, 1, out_h, out_w, device=context.device, dtype=context.dtype)
        local_var_sum = torch.zeros(B, 1, out_h, out_w, device=context.device, dtype=context.dtype)

        # Ones kernel for local sum (shared across channels)
        ones_kernel = torch.ones(1, 1, H_tpl, W_tpl, device=context.device, dtype=context.dtype)
        ones_fft = torch.fft.rfft2(ones_kernel, s=(fft_h, fft_w))

        for c_start in range(0, C, chunk_size):
            c_end = min(c_start + chunk_size, C)
            ctx_chunk = context[:, c_start:c_end]   # [B, chunk, H, W]
            tpl_chunk = tpl_centered[:, c_start:c_end]

            # Cross-correlation via FFT
            tpl_flipped = tpl_chunk.flip(dims=(-2, -1))
            ctx_fft = torch.fft.rfft2(ctx_chunk, s=(fft_h, fft_w))
            tpl_fft = torch.fft.rfft2(tpl_flipped, s=(fft_h, fft_w))
            xcorr_full = torch.fft.irfft2(ctx_fft * tpl_fft, s=(fft_h, fft_w))
            # Sum across channels in this chunk, extract valid region
            xcorr = xcorr_full[:, :, r0:r0 + out_h, c0:c0 + out_w].sum(dim=1, keepdim=True)
            cross_corr_sum = cross_corr_sum + xcorr

            # Local variance per channel: E[X^2] - E[X]^2
            local_sum_full = torch.fft.irfft2(ctx_fft * ones_fft.expand_as(ctx_fft), s=(fft_h, fft_w))
            local_sum = local_sum_full[:, :, r0:r0 + out_h, c0:c0 + out_w]
            local_mean = local_sum / n

            ctx_sq_fft = torch.fft.rfft2(ctx_chunk.pow(2), s=(fft_h, fft_w))
            local_sq_sum_full = torch.fft.irfft2(ctx_sq_fft * ones_fft.expand_as(ctx_sq_fft), s=(fft_h, fft_w))
            local_sq_sum = local_sq_sum_full[:, :, r0:r0 + out_h, c0:c0 + out_w]

            lvar = (local_sq_sum / n - local_mean.pow(2)).clamp_min(0.0)
            local_var_sum = local_var_sum + (lvar * n).sum(dim=1, keepdim=True)

        local_std_sum = local_var_sum.sqrt().clamp_min(1e-8)  # [B, 1, out_h, out_w]
        ncc = cross_corr_sum / (local_std_sum * tpl_energy)
        ncc = ncc.squeeze(1).clamp(-1.0, 1.0)
        return ncc

    def _subpixel_refine(
        self,
        ncc_map: torch.Tensor,
        peak_rows: torch.Tensor,
        peak_cols: torch.Tensor,
        best_offset: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sub-pixel refinement of NCC peak via parabolic interpolation.

        Fits a parabola to the 3 neighbors along each axis and computes
        the sub-pixel peak location.

        Args:
            ncc_map: [B, H, W] NCC correlation map.
            peak_rows: [B] integer row indices of peaks.
            peak_cols: [B] integer column indices of peaks.
            best_offset: [B, 2] integer-pixel offsets to refine.

        Returns:
            Refined [B, 2] sub-pixel offsets.
        """
        B, H, W = ncc_map.shape
        refined = best_offset.clone()

        for b in range(B):
            r, c = int(peak_rows[b]), int(peak_cols[b])

            # Refine along columns (x direction)
            if 0 < c < W - 1:
                left = ncc_map[b, r, c - 1]
                center = ncc_map[b, r, c]
                right = ncc_map[b, r, c + 1]
                denom = 2.0 * (2.0 * center - left - right)
                if abs(denom) > 1e-8:
                    refined[b, 0] = best_offset[b, 0] + (left - right) / denom

            # Refine along rows (y direction)
            if 0 < r < H - 1:
                top = ncc_map[b, r - 1, c]
                center = ncc_map[b, r, c]
                bottom = ncc_map[b, r + 1, c]
                denom = 2.0 * (2.0 * center - top - bottom)
                if abs(denom) > 1e-8:
                    refined[b, 1] = best_offset[b, 1] + (top - bottom) / denom

        return refined

    def _uv_offset_to_world_delta(
        self,
        world_xyz: torch.Tensor,
        uv_offset: torch.Tensor,
        project_fn: Callable,
        eps: float,
    ) -> torch.Tensor:
        """Convert UV pixel offset to world XY delta via finite-difference Jacobian."""
        base_uv = project_fn(world_xyz)  # [B, 2]

        # Perturb X
        x_perturbed = world_xyz.clone()
        x_perturbed[:, 0] += eps
        uv_x = project_fn(x_perturbed)

        # Perturb Y
        y_perturbed = world_xyz.clone()
        y_perturbed[:, 1] += eps
        uv_y = project_fn(y_perturbed)

        # Jacobian: d(uv) / d(world_xy) — [B, 2, 2]
        J_col0 = (uv_x - base_uv) / eps  # [B, 2]
        J_col1 = (uv_y - base_uv) / eps  # [B, 2]
        jacobian = torch.stack([J_col0, J_col1], dim=-1)  # [B, 2, 2]

        # Invert Jacobian with condition number check
        jacobian_d = jacobian.double()
        cond = torch.linalg.cond(jacobian_d)
        # If ill-conditioned, fall back to pseudo-inverse
        if (cond > 1e6).any():
            logger.warning("DOM Jacobian ill-conditioned (cond=%.1e), using pseudo-inverse", cond.max().item())
            jacobian_inv = torch.linalg.pinv(jacobian_d).to(world_xyz.dtype)
        else:
            jacobian_inv = torch.linalg.inv(jacobian_d).to(world_xyz.dtype)

        # Apply: delta_world_xy = J_inv @ uv_offset
        delta_xy = torch.bmm(
            jacobian_inv, uv_offset.unsqueeze(-1)
        ).squeeze(-1)  # [B, 2]

        return delta_xy
