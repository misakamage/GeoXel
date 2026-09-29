"""
Test-Time Adaptation (TTA) for StreamVGGT camera_head using DOM reference.

Instead of inference-only DOM correction (which proved ineffective due to
LayerNorm dilution and post-hoc nature), TTA fine-tunes a small subset of
camera_head parameters using a self-supervised NCC loss between the camera
image and the DOM crop at the predicted pose.

Key design choices:
  - Only fine-tune `poseLN_modulation` (~12M params): controls how the pose
    encoding modulates the camera token via AdaLN (shift, scale, gate).
  - Loss: negative NCC between camera edge map and DOM edge map at predicted pose.
  - No GT pose needed — the DOM itself is the supervision signal.
  - Applied at the start of each window (K=3-5 gradient steps).
  - All other parameters (aggregator, trunk blocks, pose_branch) are frozen.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List


@dataclass
class TTAConfig:
    """Configuration for test-time adaptation."""

    # Optimization
    lr: float = 1e-4                 # Learning rate for TTA steps
    num_steps: int = 3               # Gradient steps per window
    warmup_frames: int = 2           # Minimum frames before first TTA (need reference)

    # Which parameters to adapt
    adapt_poseLN: bool = True        # poseLN_modulation (shift/scale/gate generator)
    adapt_pose_branch: bool = False  # pose_branch MLP (final linear layers)
    adapt_embed_pose: bool = False   # embed_pose (pose → token embedding)

    # Loss
    use_edge_ncc: bool = True        # Use Sobel edge maps for NCC (more robust)
    ncc_weight: float = 1.0          # Weight of NCC loss
    reg_weight: float = 0.01         # L2 regularization to prevent drift from pretrained

    # DOM matching
    camera_fx: float = 1931.7
    camera_fy: float = 1931.7
    camera_orig_w: int = 1600
    camera_orig_h: int = 1200
    base_altitude_m: float = 435.0
    dom_gsd_m: float = 0.123

    # Schedule
    tta_interval: int = 1            # Apply TTA every N windows (1 = every window)
    accumulate_params: bool = True   # Carry adapted params across windows


class CameraHeadTTA:
    """
    Test-time adaptation manager for camera_head.

    Usage:
        tta = CameraHeadTTA(config, camera_head)
        # At each window:
        tta.adapt_window(camera_head, frames, aggregator, dom_image, ...)
        # Then run normal inference with the adapted camera_head
    """

    def __init__(self, config: TTAConfig, camera_head: nn.Module):
        self.config = config
        # Store the original pretrained parameters for regularization
        self._pretrained_state = {}
        self._save_pretrained(camera_head)

    def _save_pretrained(self, camera_head: nn.Module):
        """Save a copy of the parameters we'll adapt, for regularization."""
        for name, param in camera_head.named_parameters():
            if self._should_adapt(name):
                self._pretrained_state[name] = param.data.clone()

    def _should_adapt(self, param_name: str) -> bool:
        """Decide if a parameter should be adapted based on config."""
        cfg = self.config
        if cfg.adapt_poseLN and "poseLN_modulation" in param_name:
            return True
        if cfg.adapt_pose_branch and "pose_branch" in param_name:
            return True
        if cfg.adapt_embed_pose and "embed_pose" in param_name:
            return True
        return False

    def _compute_ground_footprint(self, altitude_m: float) -> Tuple[float, float]:
        """Camera ground coverage (width, height) in meters."""
        cfg = self.config
        ground_w = cfg.camera_orig_w / cfg.camera_fx * altitude_m
        ground_h = cfg.camera_orig_h / cfg.camera_fy * altitude_m
        return ground_w, ground_h

    def _crop_dom_at_pose(
        self,
        dom_image: torch.Tensor,
        world_xyz: torch.Tensor,
        project_fn: Callable,
        camera_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Crop DOM at camera-footprint scale for a given world position.
        Returns [B, 3, H_cam, W_cam].
        """
        H_cam, W_cam = camera_hw
        target_device = world_xyz.device
        dom_dev = dom_image.device
        B, C, H_dom, W_dom = dom_image.shape

        altitude = self.config.base_altitude_m + world_xyz[:, 2].mean().item()
        ground_w, ground_h = self._compute_ground_footprint(altitude)

        dom_crop_w = int(round(ground_w / self.config.dom_gsd_m))
        dom_crop_h = int(round(ground_h / self.config.dom_gsd_m))

        uv = project_fn(world_xyz).to(dom_dev)

        half_w = (dom_crop_w - 1) / 2.0
        half_h = (dom_crop_h - 1) / 2.0
        xs = torch.linspace(-half_w, half_w, dom_crop_w, device=dom_dev, dtype=dom_image.dtype)
        ys = torch.linspace(-half_h, half_h, dom_crop_h, device=dom_dev, dtype=dom_image.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)

        pixel_grid = base_grid[None] + uv[:, None, None, :]
        grid_norm_x = (pixel_grid[..., 0] / max(W_dom - 1, 1)) * 2.0 - 1.0
        grid_norm_y = (pixel_grid[..., 1] / max(H_dom - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_norm_x, grid_norm_y], dim=-1)

        crop = F.grid_sample(
            dom_image.float(), grid.float(),
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        crop_resized = F.interpolate(
            crop.float(), size=(H_cam, W_cam),
            mode="bilinear", align_corners=False,
        )
        return crop_resized.to(target_device)

    @staticmethod
    def _sobel_edge_map(img: torch.Tensor) -> torch.Tensor:
        """Compute Sobel edge magnitude [B, 1, H, W]."""
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img[:, 0:1]
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
        return (gx.pow(2) + gy.pow(2) + 1e-6).sqrt()

    @staticmethod
    def _ncc_loss(img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Negative NCC between two images of the same size.
        img_a, img_b: [B, 1, H, W]
        Returns scalar loss (higher = worse alignment, range [-1, 1]).
        """
        B = img_a.shape[0]
        a = img_a.reshape(B, -1).float()
        b = img_b.reshape(B, -1).float()
        a_mean = a.mean(dim=-1, keepdim=True)
        b_mean = b.mean(dim=-1, keepdim=True)
        a_c = a - a_mean
        b_c = b - b_mean
        numer = (a_c * b_c).sum(dim=-1)
        denom = (a_c.pow(2).sum(dim=-1).clamp_min(1e-8).sqrt() * b_c.pow(2).sum(dim=-1).clamp_min(1e-8).sqrt())
        ncc = numer / denom  # [B], range [-1, 1]
        return -ncc.mean()  # Negative: minimize → maximize correlation

    def _compute_tta_loss(
        self,
        camera_image: torch.Tensor,
        dom_crop: torch.Tensor,
        camera_head: nn.Module,
    ) -> torch.Tensor:
        """
        Compute TTA loss: NCC between camera edges and DOM edges + regularization.
        """
        cfg = self.config

        # Edge maps
        if cfg.use_edge_ncc:
            cam_edges = self._sobel_edge_map(camera_image.float())
            dom_edges = self._sobel_edge_map(dom_crop.float())
        else:
            cam_edges = camera_image.float().mean(dim=1, keepdim=True)
            dom_edges = dom_crop.float().mean(dim=1, keepdim=True)

        # NCC loss
        loss = cfg.ncc_weight * self._ncc_loss(cam_edges, dom_edges)

        # L2 regularization toward pretrained weights
        if cfg.reg_weight > 0:
            reg = torch.tensor(0.0, device=loss.device)
            for name, param in camera_head.named_parameters():
                if name in self._pretrained_state:
                    reg = reg + (param - self._pretrained_state[name].to(param.device)).pow(2).sum()
            loss = loss + cfg.reg_weight * reg

        return loss

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
        Perform TTA at the start of a window.

        Steps:
          1. Run forward pass (frozen aggregator) to get initial poses
          2. For each TTA step:
             a. Use predicted pose to get world XYZ
             b. Crop DOM at that position
             c. Compute NCC loss (camera edges vs DOM edges)
             d. Backprop through camera_head only
             e. Update adapted parameters

        Args:
            model: StreamVGGT model
            frames: List of frame dicts with "img" key
            dom_image: Full DOM [B, 3, H_dom, W_dom]
            project_fn: world_xyz → DOM UV
            anchor_world_xyz: First frame's world position [B, 3]
            get_world_xyz_fn: pose_params → world_xyz
            heading_fn: frame_idx → heading_rad
            dtype: compute dtype

        Returns:
            dict with diagnostics (losses, etc.)
        """
        cfg = self.config
        camera_head = model.camera_head
        aggregator = model.aggregator
        device = next(camera_head.parameters()).device

        if len(frames) < cfg.warmup_frames:
            return {"skipped": True, "reason": "too_few_frames"}

        # --- Freeze everything except adapted params ---
        for name, param in camera_head.named_parameters():
            param.requires_grad = self._should_adapt(name)

        # Also make sure aggregator is fully frozen
        for param in aggregator.parameters():
            param.requires_grad = False

        # --- Optimizer for adapted params only ---
        adapted_params = [p for p in camera_head.parameters() if p.requires_grad]
        if not adapted_params:
            return {"skipped": True, "reason": "no_params_to_adapt"}

        optimizer = torch.optim.Adam(adapted_params, lr=cfg.lr)

        # --- Pre-compute aggregator outputs (frozen, no grad) ---
        # Run aggregator for all frames to get tokens (this is cached / deterministic)
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
                # Detach tokens so no gradient flows to aggregator
                if isinstance(tokens, list):
                    tokens_detached = [t.detach() for t in tokens]
                else:
                    tokens_detached = tokens.detach()
                agg_outputs.append(tokens_detached)

        # --- Pre-compute differentiable affine ENU → DOM pixel projection ---
        with torch.no_grad():
            uv0 = project_fn(torch.zeros(1, 3, device=device)).float()
            uv_x = project_fn(torch.tensor([[1.0, 0.0, 0.0]], device=device)).float()
            uv_y = project_fn(torch.tensor([[0.0, 1.0, 0.0]], device=device)).float()
            proj_J_E = (uv_x - uv0).detach()    # d(UV)/d(East): [1, 2]
            proj_J_N = (uv_y - uv0).detach()    # d(UV)/d(North): [1, 2]

            # Get anchor UV (where the window starts in the DOM)
            anchor_uv = project_fn(anchor_world_xyz.unsqueeze(0) if anchor_world_xyz.dim() == 1 else anchor_world_xyz).float().detach()

        # --- Extract a regional DOM patch and put on GPU ---
        # Crop a large region (3× camera ground footprint) around the anchor position.
        # This fits on GPU (~200-500MB) while enabling fully on-GPU differentiable grid_sample.
        B_dom, C_dom, H_dom, W_dom = dom_image.shape
        ground_w, ground_h = self._compute_ground_footprint(cfg.base_altitude_m)
        cam_crop_w_px = int(round(ground_w / cfg.dom_gsd_m))
        cam_crop_h_px = int(round(ground_h / cfg.dom_gsd_m))

        # Regional patch: 4× the camera footprint to allow for pose drift
        region_scale = 4.0
        patch_w = int(cam_crop_w_px * region_scale)
        patch_h = int(cam_crop_h_px * region_scale)

        # Center of patch in DOM pixels
        anchor_col = int(anchor_uv[0, 0].item())
        anchor_row = int(anchor_uv[0, 1].item())

        # Clamp to DOM bounds
        r0 = max(0, anchor_row - patch_h // 2)
        r1 = min(H_dom, r0 + patch_h)
        c0 = max(0, anchor_col - patch_w // 2)
        c1 = min(W_dom, c0 + patch_w)

        # Extract patch [1, 3, patch_h, patch_w] and move to GPU
        dom_patch = dom_image[:, :, r0:r1, c0:c1].float().to(device)
        patch_H, patch_W = dom_patch.shape[2], dom_patch.shape[3]

        # Offset: UV in full DOM → UV in patch
        patch_origin = torch.tensor([[c0, r0]], dtype=torch.float32, device=device)

        # Camera output size
        cam_H, cam_W = frames[0]["img"].shape[-2], frames[0]["img"].shape[-1]

        # Base grid for camera-footprint crop
        half_w = (cam_crop_w_px - 1) / 2.0
        half_h = (cam_crop_h_px - 1) / 2.0
        xs = torch.linspace(-half_w, half_w, cam_crop_w_px, device=device, dtype=torch.float32)
        ys = torch.linspace(-half_h, half_h, cam_crop_h_px, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([gx, gy], dim=-1)  # [crop_h, crop_w, 2]

        # Pre-compute camera edge maps (constant, no grad)
        cam_edges_list = []
        with torch.no_grad():
            for frame in frames:
                cam_edges_list.append(self._sobel_edge_map(frame["img"].float()))

        # The anchor UV offset applied to get UV relative to the anchor
        proj_uv0 = uv0.detach().to(device)  # UV at ENU origin

        # --- TTA gradient steps (in float32 for stable gradients) ---
        # Disable autocast: outer context may have bfloat16 autocast enabled,
        # but TTA needs float32 for reliable gradient computation.
        losses = []
        n_total = len(agg_outputs)

        # Cast tokens to float32 for TTA gradient loop
        for idx in range(len(agg_outputs)):
            if isinstance(agg_outputs[idx], list):
                agg_outputs[idx] = [t.float() for t in agg_outputs[idx]]
            else:
                agg_outputs[idx] = agg_outputs[idx].float()

        with torch.cuda.amp.autocast(enabled=False):
          for step in range(cfg.num_steps):
            optimizer.zero_grad()
            step_loss_sum = 0.0
            n_frames_used = 0

            for i, tokens in enumerate(agg_outputs):
                token_input = tokens if isinstance(tokens, list) else [tokens]

                # camera_head forward WITH gradients on poseLN_modulation
                past_kv_cam_single = [None] * camera_head.trunk_depth
                pose_enc_list, _ = camera_head(
                    token_input,
                    past_key_values_camera=past_kv_cam_single,
                    use_cache=True,
                )
                camera_pose = pose_enc_list[-1][:, 0, :]  # [B, 9]

                # Differentiable: pose → camera center → ENU world coords
                world_xyz = get_world_xyz_fn(camera_pose)  # [B, 3], grad ON

                if step == 0 and i == 0:
                    print(f"    [TTA debug] camera_pose.requires_grad={camera_pose.requires_grad}, "
                          f"world_xyz.requires_grad={world_xyz.requires_grad}, "
                          f"camera_pose.dtype={camera_pose.dtype}, world_xyz={world_xyz[0].tolist()}")

                # Differentiable affine: ENU → DOM pixel UV
                full_uv = (proj_uv0
                           + world_xyz[:, 0:1] * proj_J_E.to(device)
                           + world_xyz[:, 1:2] * proj_J_N.to(device))  # [B, 2]

                # Convert to patch-relative UV
                patch_uv = full_uv - patch_origin  # [B, 2], grad ON

                # Build sampling grid in patch coordinates
                pixel_grid = base_grid[None] + patch_uv[:, None, None, :]
                norm_x = (pixel_grid[..., 0] / max(patch_W - 1, 1)) * 2.0 - 1.0
                norm_y = (pixel_grid[..., 1] / max(patch_H - 1, 1)) * 2.0 - 1.0
                grid = torch.stack([norm_x, norm_y], dim=-1)

                # Differentiable crop from GPU patch
                dom_crop = F.grid_sample(
                    dom_patch, grid.float(),
                    mode="bilinear", padding_mode="border", align_corners=True,
                )
                dom_crop = F.interpolate(
                    dom_crop, size=(cam_H, cam_W),
                    mode="bilinear", align_corners=False,
                )

                # NCC loss on edge maps (fully differentiable)
                dom_edges = self._sobel_edge_map(dom_crop)
                ncc_loss = cfg.ncc_weight * self._ncc_loss(
                    cam_edges_list[i].to(device), dom_edges,
                )

                # Backward per frame to bound memory
                (ncc_loss / n_total).backward()
                step_loss_sum += ncc_loss.item()
                n_frames_used += 1

            # L2 regularization toward pretrained (once per step)
            if cfg.reg_weight > 0:
                reg = torch.tensor(0.0, device=device)
                for name, param in camera_head.named_parameters():
                    if name in self._pretrained_state:
                        reg = reg + (param - self._pretrained_state[name].to(param.device)).pow(2).sum()
                (cfg.reg_weight * reg).backward()

            if n_frames_used > 0:
                # Debug: print gradient norms
                grad_norms = []
                for p in adapted_params:
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm().item())
                    else:
                        grad_norms.append(0.0)
                total_grad = sum(grad_norms)
                print(f"    [TTA step {step}] loss={step_loss_sum / n_frames_used:.6f}, "
                      f"grad_norm={total_grad:.6e}, n_params_with_grad={sum(1 for g in grad_norms if g > 0)}/{len(grad_norms)}")

                torch.nn.utils.clip_grad_norm_(adapted_params, max_norm=1.0)
                optimizer.step()
                losses.append(step_loss_sum / n_frames_used)

        # Free regional patch
        del dom_patch

        # --- Re-freeze all params after TTA ---
        for param in camera_head.parameters():
            param.requires_grad = False

        return {
            "skipped": False,
            "num_steps": cfg.num_steps,
            "losses": losses,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
        }

    def reset_to_pretrained(self, camera_head: nn.Module):
        """Reset adapted parameters to pretrained values."""
        with torch.no_grad():
            for name, param in camera_head.named_parameters():
                if name in self._pretrained_state:
                    param.data.copy_(self._pretrained_state[name].to(param.device))


# ======================================================================
#  Post-hoc pose refinement using DOM NCC
# ======================================================================

def refine_poses_with_dom(
    camera_images: List[torch.Tensor],
    initial_centers_enu: torch.Tensor,
    dom_image: torch.Tensor,
    project_fn: Callable,
    config: TTAConfig,
    headings_rad: Optional[List[float]] = None,
    num_steps: int = 20,
    lr: float = 0.05,
    use_edges: bool = True,
) -> torch.Tensor:
    """
    Refine camera positions by optimizing per-frame 2D offsets to maximize
    NCC between camera images and heading-rotated DOM crops.

    Args:
        camera_images: List of [1, 3, H, W] camera image tensors.
        initial_centers_enu: [N, 3] initial camera centers in ENU coordinates.
        dom_image: [1, 3, H_dom, W_dom] DOM image (may be on CPU).
        project_fn: ENU [B, 3] -> DOM pixel UV [B, 2].
        config: TTAConfig with camera/DOM params.
        headings_rad: Per-frame heading angle in radians (from North, CW positive).
        num_steps: Number of optimization steps.
        lr: Learning rate for the 2D offsets.
        use_edges: Use Sobel edge maps (True) or raw grayscale (False).

    Returns:
        refined_centers_enu: [N, 3] refined camera centers in ENU (numpy).
        diagnostics: dict with losses and offsets info.
    """
    N = len(camera_images)
    device = camera_images[0].device

    # --- Pre-compute affine projection ENU → DOM pixel ---
    with torch.no_grad():
        uv0 = project_fn(torch.zeros(1, 3, device=device)).float()
        uv_x = project_fn(torch.tensor([[1.0, 0.0, 0.0]], device=device)).float()
        uv_y = project_fn(torch.tensor([[0.0, 1.0, 0.0]], device=device)).float()
        J_E = (uv_x - uv0).detach()
        J_N = (uv_y - uv0).detach()

    # --- Camera ground footprint in DOM pixels ---
    gsd = config.dom_gsd_m
    alt = config.base_altitude_m
    ground_w = alt * config.camera_orig_w / config.camera_fx
    ground_h = alt * config.camera_orig_h / config.camera_fy
    crop_w_px = int(round(ground_w / gsd))
    crop_h_px = int(round(ground_h / gsd))
    cam_H, cam_W = camera_images[0].shape[-2], camera_images[0].shape[-1]

    # --- Extract regional DOM patch ---
    centers_tensor = initial_centers_enu.float().to(device)
    with torch.no_grad():
        all_uv = (uv0
                   + centers_tensor[:, 0:1] * J_E
                   + centers_tensor[:, 1:2] * J_N)
        center_uv = all_uv.mean(dim=0)

    B_dom, C_dom, H_dom, W_dom = dom_image.shape
    patch_margin = 3.0
    patch_w = int(crop_w_px * (2 * patch_margin + 1))
    patch_h = int(crop_h_px * (2 * patch_margin + 1))

    anchor_col = int(center_uv[0].item())
    anchor_row = int(center_uv[1].item())
    r0 = max(0, anchor_row - patch_h // 2)
    r1 = min(H_dom, r0 + patch_h)
    c0 = max(0, anchor_col - patch_w // 2)
    c1 = min(W_dom, c0 + patch_w)

    dom_patch = dom_image[:, :, r0:r1, c0:c1].float().to(device)
    patch_H, patch_W = dom_patch.shape[2], dom_patch.shape[3]
    patch_origin = torch.tensor([[c0, r0]], dtype=torch.float32, device=device)

    # --- Pre-compute per-frame rotated base grids ---
    half_w = (crop_w_px - 1) / 2.0
    half_h = (crop_h_px - 1) / 2.0
    xs = torch.linspace(-half_w, half_w, crop_w_px, device=device, dtype=torch.float32)
    ys = torch.linspace(-half_h, half_h, crop_h_px, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base_xy = torch.stack([gx, gy], dim=-1)  # [crop_h, crop_w, 2]

    # Build per-frame rotated grids based on heading
    rotated_grids = []
    for i in range(N):
        if headings_rad is not None and i < len(headings_rad):
            theta = headings_rad[i]
        else:
            theta = 0.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # Rotate the base grid by heading angle
        # In DOM pixel space: x = East, y = South (row increases downward)
        # Heading is from North (up) clockwise
        rx = base_xy[..., 0] * cos_t - base_xy[..., 1] * sin_t
        ry = base_xy[..., 0] * sin_t + base_xy[..., 1] * cos_t
        rotated_grids.append(torch.stack([rx, ry], dim=-1))

    # --- Camera feature maps (constant) ---
    cam_features = []
    with torch.no_grad():
        for img in camera_images:
            if use_edges:
                cam_features.append(CameraHeadTTA._sobel_edge_map(img.float()))
            else:
                gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
                cam_features.append(gray.float())

    # --- Learnable 2D offsets (East, North) per frame ---
    offsets = torch.zeros(N, 2, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([offsets], lr=lr)

    losses_history = []

    with torch.cuda.amp.autocast(enabled=False):
        for step in range(num_steps):
            optimizer.zero_grad()
            total_loss = 0.0

            for i in range(N):
                # Adjusted center: initial + offset
                center_i = centers_tensor[i:i+1].clone()
                center_i[:, 0:1] = center_i[:, 0:1] + offsets[i:i+1, 0:1]
                center_i[:, 1:2] = center_i[:, 1:2] + offsets[i:i+1, 1:2]

                # ENU → DOM pixel UV (differentiable through offsets)
                full_uv = (uv0.to(device)
                           + center_i[:, 0:1] * J_E.to(device)
                           + center_i[:, 1:2] * J_N.to(device))
                patch_uv = full_uv - patch_origin

                # Build sampling grid (rotated by heading)
                pixel_grid = rotated_grids[i][None] + patch_uv[:, None, None, :]
                norm_x = (pixel_grid[..., 0] / max(patch_W - 1, 1)) * 2.0 - 1.0
                norm_y = (pixel_grid[..., 1] / max(patch_H - 1, 1)) * 2.0 - 1.0
                grid = torch.stack([norm_x, norm_y], dim=-1)

                dom_crop = F.grid_sample(
                    dom_patch, grid.float(),
                    mode="bilinear", padding_mode="border", align_corners=True,
                )
                dom_crop = F.interpolate(
                    dom_crop, size=(cam_H, cam_W),
                    mode="bilinear", align_corners=False,
                )

                # Feature extraction from DOM crop
                if use_edges:
                    dom_feat = CameraHeadTTA._sobel_edge_map(dom_crop)
                else:
                    dom_feat = 0.299 * dom_crop[:, 0:1] + 0.587 * dom_crop[:, 1:2] + 0.114 * dom_crop[:, 2:3]

                ncc = CameraHeadTTA._ncc_loss(cam_features[i], dom_feat)
                (ncc / N).backward()
                total_loss += ncc.item()

            # Smooth regularization: neighboring offsets should be similar
            if N > 1:
                smooth_reg = ((offsets[1:] - offsets[:-1]).pow(2).sum()) * 0.1
                smooth_reg.backward()

            optimizer.step()
            losses_history.append(total_loss / N)

    # Apply offsets to centers
    with torch.no_grad():
        refined = initial_centers_enu.clone()
        refined[:, 0] += offsets[:, 0].cpu()
        refined[:, 1] += offsets[:, 1].cpu()

    offset_magnitudes = offsets.detach().cpu().norm(dim=-1)
    return refined, {
        "losses": losses_history,
        "mean_offset_m": float(offset_magnitudes.mean()),
        "max_offset_m": float(offset_magnitudes.max()),
    }
