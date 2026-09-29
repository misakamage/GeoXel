"""
Geo-Supervised Test-Time Training v2 — Dense DOM Re-rendering Loss on
Aggregator KV-cache.

Gradient path (fully differentiable):
  L_dom  →  |sampled_DOM − cam_img|
         →  F.grid_sample(DOM, grid)
         →  grid = affine(pts3d_xy)
         →  pts3d = point_head(agg_tokens)
         →  agg_tokens from global_blocks(tokens, KV)
         →  KV-cache  (leaf tensors, updated by Adam)
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List


@dataclass
class GeoTTTv2Config:
    """Configuration for Geo-TTT v2."""

    # === Mode ===
    adapt_mode: str = "token"  # "token" | "kv" | "pts3d_residual" | "lora" | "geo_consist" | "geo_consist_token" | "pose_refine" | "pose_ttt" | "pose_ttt_token" | "pose_ttt_rerun23"

    # === Optimization ===
    kv_lr: float = 1e-2
    num_steps: int = 3
    kv_update_layers: str = "all"  # "all" | "last_half" | "last_quarter" | "dpt" | "last_layer" | "4,11,17,23"

    # === Loss ===
    reg_weight: float = 0.01
    loss_type: str = "l1"          # "l1" | "mse" | "smooth_l1"
    conf_weighted: bool = True

    # === Frame selection ===
    warmup_frames: int = 2
    max_frames_for_loss: int = 8

    # === Pose preservation ===
    cam_token_weight: float = 10.0  # camera-token consistency weight

    # === DOM crop ===
    dom_crop_padding_px: int = 500

    # === LoRA (CAPA-style) ===
    lora_rank: int = 4
    lora_alpha: float = None       # default: 2 * rank
    lora_target_blocks: str = "all"  # "all" | "frame" | "global"
    lora_target_layers: str = "qkv"  # "qkv" | "qkv_proj" | "all"
    lora_lr: float = 1e-3
    lora_batch_ratio: float = 0.25  # fraction of frames per optimization step
    lora_sequence_share: bool = True  # share LoRA params across all windows

    # === Geometric consistency (Test3R-style) ===
    geo_consist_stride: int = 5       # match every N frames to DOM
    geo_consist_crop_m: float = 250.0 # DOM crop half-size in meters
    geo_consist_max_corr: int = 500   # max correspondences per matched frame

    # === Camera intrinsics (for FOV-matched DOM crop) ===
    camera_fx: float = None
    camera_fy: float = None
    camera_orig_w: int = None
    camera_orig_h: int = None
    base_altitude_m: float = None

    # === pose_ttt loss mode ===
    pose_ttt_loss: str = "mse"  # "mse" | "dense_reproj" | "both"
    dense_reproj_weight: float = 0.5    # weight for dense reproj loss
    dense_reproj_z_weight: float = 0.1  # relative weight for Z (DEM) vs XY (DOM)
    dense_reproj_huber_xy: float = 5.0  # Huber delta for XY (meters)
    dense_reproj_huber_z: float = 10.0  # Huber delta for Z (meters)
    mse_weight: float = 1.0             # weight for MSE center loss (scale up to balance with dense)
    heading_weight: float = 0.0         # weight for heading (rotation) loss from DOM (0 = disabled)
    max_grad_norm: float = 0.0          # gradient clipping (0 = disabled)

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "GeoTTTv2Config":
        """Load config from YAML file. CLI overrides take precedence."""
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        # Only keep keys that are actual dataclass fields
        import dataclasses
        valid = {fd.name for fd in dataclasses.fields(cls)}
        cfg_dict = {k: v for k, v in raw.items() if k in valid}
        cfg_dict.update({k: v for k, v in overrides.items()
                         if k in valid and v is not None})
        return cls(**cfg_dict)


# ======================================================================
#  Model→ENU alignment: scale + 2D rotation from camera centers vs GT
# ======================================================================

def compute_model_to_enu_transform(
    model_centers: np.ndarray,   # [N, 3] camera centers in model coords
    gt_enu_local: np.ndarray,    # [N, 3] GT ENU relative to anchor
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute scale *s*, 2×2 rotation *R*, and 2-vec translation *t*
    such that:
        gt_enu_local_xy ≈ s * R @ model_xy + t
    Uses Procrustes on the XY components (ignoring Z).
    Returns (s, R_2x2, t_2).
    """
    m = model_centers[:, :2]   # [N, 2]
    g = gt_enu_local[:, :2]    # [N, 2]
    m_mean = m.mean(axis=0)
    g_mean = g.mean(axis=0)
    m_c = m - m_mean
    g_c = g - g_mean
    # cross-covariance
    H = m_c.T @ g_c            # [2, 2]
    U, S, Vt = np.linalg.svd(H)
    # rotation
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, np.sign(d)])
    R = Vt.T @ D @ U.T        # [2, 2]
    # scale
    scale = np.sum(S * np.diag(D)) / (np.sum(m_c ** 2) + 1e-12)
    # translation
    t = g_mean - scale * R @ m_mean
    return float(scale), R, t


def ransac_sim2(
    src: np.ndarray,
    dst: np.ndarray,
    reproj_thresh: float = 5.0,
    max_iters: int = 2000,
    confidence: float = 0.999,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate 2D similarity (s, R, t) from src→dst with RANSAC.

    Uses cv2.estimateAffinePartial2D which fits [sR | t] with RANSAC.

    Args:
        src, dst: (N, 2) point correspondences.
        reproj_thresh: inlier threshold in dst units (metres for ENU).

    Returns:
        (scale, R_2x2, t_2, inlier_mask_bool)
    """
    M, inlier_mask = cv2.estimateAffinePartial2D(
        src.reshape(-1, 1, 2).astype(np.float64),
        dst.reshape(-1, 1, 2).astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh,
        maxIters=max_iters,
        confidence=confidence,
    )
    if M is None:
        # Fallback to plain Procrustes
        src_3d = np.column_stack([src, np.zeros(len(src))])
        dst_3d = np.column_stack([dst, np.zeros(len(dst))])
        s, R, t = compute_model_to_enu_transform(src_3d, dst_3d)
        return s, R, t, np.ones(len(src), dtype=bool)

    # Extract scale, rotation, translation from [[a, -b, tx], [b, a, ty]]
    a, b = M[0, 0], M[1, 0]
    scale = np.sqrt(a * a + b * b)
    cos_t = a / scale
    sin_t = b / scale
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    t = np.array([M[0, 2], M[1, 2]])
    mask = inlier_mask.ravel().astype(bool)
    return float(scale), R, t, mask


# ======================================================================
#  Differentiable affine projection:  pts3d_local_xy  →  DOM pixel UV
# ======================================================================

def build_differentiable_projection(
    project_fn: Callable,
    enu_offset: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Numerically linearise *project_fn* at the window-anchor position so
    that the full mapping  pts3d_local_xy → DOM_pixel_(col, row)  becomes
    a simple affine:   dom_uv = A @ pts_xy + b.
    """
    ref = torch.tensor([[enu_offset[0], enu_offset[1], 0.0]], dtype=torch.float64)
    dx  = torch.tensor([[enu_offset[0] + 1.0, enu_offset[1], 0.0]], dtype=torch.float64)
    dy  = torch.tensor([[enu_offset[0], enu_offset[1] + 1.0, 0.0]], dtype=torch.float64)

    uv_ref = project_fn(ref).double()
    uv_dx  = project_fn(dx).double()
    uv_dy  = project_fn(dy).double()

    A = torch.tensor(
        [[(uv_dx[0, 0] - uv_ref[0, 0]).item(), (uv_dy[0, 0] - uv_ref[0, 0]).item()],
         [(uv_dx[0, 1] - uv_ref[0, 1]).item(), (uv_dy[0, 1] - uv_ref[0, 1]).item()]],
        dtype=torch.float32, device=device,
    )
    b = torch.tensor(
        [uv_ref[0, 0].item(), uv_ref[0, 1].item()],
        dtype=torch.float32, device=device,
    )
    return A, b


# ======================================================================
#  DOM crop utility
# ======================================================================

def crop_dom_for_window(
    dom_image_cpu: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    pts3d_list: List[torch.Tensor],
    padding: int = 500,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Crop the full DOM to the region visible from this window and move it
    to GPU as float32.  Returns the crop, adjusted bias, and crop dims.
    """
    _, _, H_dom, W_dom = dom_image_cpu.shape
    A_cpu, b_cpu = A.cpu().float(), b.cpu().float()

    all_col_min, all_col_max = float(W_dom), 0.0
    all_row_min, all_row_max = float(H_dom), 0.0

    with torch.no_grad():
        for pts3d in pts3d_list:
            xy = pts3d[..., :2].cpu().float()
            uv = torch.einsum('ij,...j->...i', A_cpu, xy) + b_cpu
            all_col_min = min(all_col_min, uv[..., 0].min().item())
            all_col_max = max(all_col_max, uv[..., 0].max().item())
            all_row_min = min(all_row_min, uv[..., 1].min().item())
            all_row_max = max(all_row_max, uv[..., 1].max().item())

    col0 = max(0, int(all_col_min) - padding)
    col1 = min(W_dom, int(all_col_max) + padding)
    row0 = max(0, int(all_row_min) - padding)
    row1 = min(H_dom, int(all_row_max) + padding)

    dom_crop = dom_image_cpu[:, :, row0:row1, col0:col1].float() / 255.0
    dom_crop = dom_crop.to(device)

    b_crop = b.clone()
    b_crop[0] -= col0
    b_crop[1] -= row0
    return dom_crop, b_crop, row1 - row0, col1 - col0


# ======================================================================
#  GeoTTTv2
# ======================================================================

class GeoTTTv2:

    def __init__(self, config: GeoTTTv2Config):
        self.config = config
        self._global_sim2 = None  # scale from first window, shared by all

    # helper -----------------------------------------------------------
    def _update_layers(self, depth: int) -> List[int]:
        c = self.config.kv_update_layers
        if c == "all":
            return list(range(depth))
        if c == "last_half":
            return list(range(depth // 2, depth))
        if c == "last_quarter":
            return list(range(3 * depth // 4, depth))
        if c == "dpt":
            # Only the 4 layers used as DPT skip-connections
            return [4, 11, 17, 23]
        if c == "last_layer":
            return [depth - 1]
        # comma-separated indices: "4,11,17,23"
        if "," in c:
            return [int(x) for x in c.split(",")]
        return list(range(depth))

    # ------------------------------------------------------------------
    #  Phase 1: frozen forward
    # ------------------------------------------------------------------

    def _frozen_forward(self, model, frames, dtype=torch.bfloat16):
        """Frozen streaming forward — caches KV-cache and initial pts3d."""
        aggregator = model.aggregator
        camera_head = model.camera_head
        depth = aggregator.depth

        past_kv = [None] * depth
        past_kv_cam = [None] * camera_head.trunk_depth
        per_frame_pts = []
        per_frame_pose = []
        per_frame_cam_token = []   # camera token for consistency loss
        psi = 5

        with torch.no_grad():
            for i, frame in enumerate(frames):
                images = frame["img"].unsqueeze(0)
                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images, past_key_values=past_kv,
                        use_cache=True, past_frame_idx=i,
                    )
                if isinstance(agg_out, tuple) and len(agg_out) == 3:
                    agg_tokens, psi, past_kv = agg_out
                else:
                    agg_tokens, psi = agg_out

                with torch.cuda.amp.autocast(enabled=False):
                    pose_enc, past_kv_cam = camera_head(
                        agg_tokens, past_key_values_camera=past_kv_cam,
                        use_cache=True,
                    )
                    per_frame_pose.append(
                        pose_enc[-1][:, 0, :].detach().float()
                    )
                    # Cache camera token for consistency loss
                    cam_tok = agg_tokens[-1][:, :, 0:1].detach().float()
                    per_frame_cam_token.append(cam_tok)
                    pts3d, _ = model.point_head(
                        agg_tokens, images=images, patch_start_idx=psi,
                    )
                    per_frame_pts.append(pts3d[:, 0].detach().float())

        del past_kv_cam
        torch.cuda.empty_cache()
        return past_kv, per_frame_pts, per_frame_pose, per_frame_cam_token, psi

    # ------------------------------------------------------------------
    #  Phase 2: detach KV → leaves
    # ------------------------------------------------------------------
    @staticmethod
    def _setup_kv_leaves(past_kv, depth, update_layers):
        """Detach selected KV entries and make them leaf tensors.
        KV per layer: (K, V) with shape [B, heads, N_frames, P, dim] (5-D).
        """
        kv_leaves, kv_originals = [], []
        for j in range(depth):
            if past_kv[j] is not None:
                k, v = past_kv[j]
                if j in update_layers:
                    k = k.detach().float().requires_grad_(True)
                    v = v.detach().float().requires_grad_(True)
                    past_kv[j] = (k, v)
                    kv_leaves.extend([k, v])
                    kv_originals.extend([k.data.clone(), v.data.clone()])
                else:
                    past_kv[j] = (k.detach(), v.detach())
        return past_kv, kv_leaves, kv_originals

    # ------------------------------------------------------------------
    #  per-frame loss (core differentiable path)
    # ------------------------------------------------------------------
    def _frame_loss(self, model, frame, fi, leaf_kv, depth,
                    A, b_crop, dom_crop, crop_h, crop_w,
                    cam_token_orig, dtype, device):
        """
        Re-run aggregator for frame *fi* using KV[:, :, :fi] (entries
        from prior frames).  Then pts3d → affine → grid_sample DOM → loss.
        """
        cfg = self.config
        aggregator = model.aggregator

        # --- trim KV to entries before this frame (5-D: B, H, frames, P, D) ---
        kv_trim = []
        for j in range(depth):
            if leaf_kv[j] is not None and fi > 0:
                k, v = leaf_kv[j]
                kv_trim.append((k[:, :, :fi], v[:, :, :fi]))
            else:
                kv_trim.append(None)

        images = frame["img"].unsqueeze(0)               # [1,1,3,H,W]
        with torch.cuda.amp.autocast(dtype=dtype):
            agg_out = aggregator(
                images, past_key_values=kv_trim,
                use_cache=True, past_frame_idx=fi,
            )
        agg_tokens = agg_out[0]
        psi = agg_out[1]

        with torch.cuda.amp.autocast(enabled=False):
            agg_f = [t.float() for t in agg_tokens]
            pts3d, pts3d_conf = model.point_head(
                agg_f, images=images, patch_start_idx=psi,
            )
        pts3d = pts3d[:, 0]               # [B, H, W, 3]
        pts3d_conf = pts3d_conf[:, 0]     # [B, H, W]

        # --- differentiable projection ---
        pts_xy = pts3d[..., :2]
        dom_uv = torch.einsum('ij,...j->...i', A, pts_xy) + b_crop

        grid_x = (dom_uv[..., 0] / max(crop_w - 1, 1)) * 2.0 - 1.0
        grid_y = (dom_uv[..., 1] / max(crop_h - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        sampled = F.grid_sample(
            dom_crop, grid.float(), mode="bilinear",
            padding_mode="border", align_corners=True,
        )                                                  # [B, 3, Hpt, Wpt]

        _, _, Hpt, Wpt = sampled.shape
        cam = F.interpolate(
            frame["img"][:1].float().to(device),
            size=(Hpt, Wpt), mode="bilinear", align_corners=False,
        )

        if cfg.loss_type == "l1":
            px = (sampled - cam).abs().mean(dim=1)
        elif cfg.loss_type == "mse":
            px = (sampled - cam).pow(2).mean(dim=1)
        else:
            px = F.smooth_l1_loss(sampled, cam, reduction='none').mean(dim=1)

        in_bounds = (grid[..., 0].abs() < 1) & (grid[..., 1].abs() < 1)
        px = px * in_bounds.float()

        if cfg.conf_weighted:
            cw = pts3d_conf.detach().clamp(min=1.0)
            cw = (cw - 1.0) / (cw.max() + 1e-8)
            px = px * cw

        dom_loss = px.sum() / in_bounds.float().sum().clamp(min=1.0)

        # --- camera-token consistency loss ---
        if cfg.cam_token_weight > 0 and cam_token_orig is not None:
            cam_token_new = agg_tokens[-1][:, :, 0:1].float()
            cam_tok_loss = (cam_token_new - cam_token_orig.to(device)).abs().mean()
            return dom_loss + cfg.cam_token_weight * cam_tok_loss
        return dom_loss

    # ------------------------------------------------------------------
    #  adapt_window  (main entry point)
    # ------------------------------------------------------------------
    def adapt_window(self, model, frames, dom_image_cpu, project_fn,
                     enu_offset, dtype=torch.bfloat16, gt_enu_window=None,
                     correspondences=None,
                     roma_model=None, image_paths=None,
                     inv_project_fn=None, geo_elev=None,
                     dom_transform=None):
        cfg = self.config
        if cfg.adapt_mode == "geo_consist":
            return self._adapt_window_geo_consist(
                model, frames, dom_image_cpu, project_fn, enu_offset, dtype,
                gt_enu_window=gt_enu_window, correspondences=correspondences)
        if cfg.adapt_mode == "geo_consist_token":
            return self._adapt_window_geo_consist_token(
                model, frames, dom_image_cpu, project_fn, enu_offset, dtype,
                gt_enu_window=gt_enu_window, correspondences=correspondences)
        if cfg.adapt_mode == "pose_refine":
            return self._adapt_window_pose_refine(
                model, frames, dom_image_cpu, project_fn, enu_offset, dtype,
                gt_enu_window=gt_enu_window, correspondences=correspondences)
        if cfg.adapt_mode == "pose_ttt":
            return self._adapt_window_pose_ttt(
                model, frames, dom_image_cpu, project_fn, enu_offset, dtype,
                gt_enu_window=gt_enu_window, correspondences=correspondences,
                roma_model=roma_model, image_paths=image_paths,
                inv_project_fn=inv_project_fn, geo_elev=geo_elev,
                dom_transform=dom_transform)
        if cfg.adapt_mode == "pose_ttt_token":
            return self._adapt_window_pose_ttt_token(
                model, frames, dom_image_cpu, project_fn, enu_offset, dtype,
                gt_enu_window=gt_enu_window, correspondences=correspondences,
                roma_model=roma_model, image_paths=image_paths,
                inv_project_fn=inv_project_fn, geo_elev=geo_elev,
                dom_transform=dom_transform)
        if cfg.adapt_mode == "pose_ttt_rerun23":
            return self._adapt_window_pose_ttt_rerun23(
                model, frames, dom_image_cpu, project_fn, enu_offset, dtype,
                gt_enu_window=gt_enu_window, correspondences=correspondences)
        if cfg.adapt_mode == "lora":
            return self._adapt_window_lora(
                model, frames, dom_image_cpu, project_fn, enu_offset, dtype,
                gt_enu_window=gt_enu_window)
        if cfg.adapt_mode == "token":
            return self._adapt_window_token(
                model, frames, dom_image_cpu, project_fn, enu_offset, dtype)
        if cfg.adapt_mode == "pts3d_residual":
            return self._adapt_window_pts3d(
                model, frames, dom_image_cpu, project_fn, enu_offset, dtype,
                gt_enu_window=gt_enu_window)
        return self._adapt_window_kv(
            model, frames, dom_image_cpu, project_fn, enu_offset, dtype)

    # ------------------------------------------------------------------
    #  LoRA mode (CAPA-style: PEFT on ViT encoder attention)
    # ------------------------------------------------------------------
    def _adapt_window_lora(self, model, frames, dom_image_cpu, project_fn,
                           enu_offset, dtype=torch.bfloat16, gt_enu_window=None):
        """Adapt window using LoRA on aggregator attention layers.

        Memory-efficient KV-cache streaming approach:
          Phase 1: frozen streaming → build KV-cache + alignment + DOM crop
          Phase 2: inject LoRA adapters
          Phase 3: per-frame forward with frozen KV-cache context,
                   gradient accumulation across mini-batch frames

        Each selected frame is forwarded individually through the LoRA-adapted
        aggregator, using the frozen KV-cache from Phase 1 as context for
        previous frames.  This avoids the O(S²·P²) memory of batched global
        attention while still letting gradients flow to LoRA {A, B}.
        """
        from streamvggt.utils.lora import inject_lora, remove_lora

        cfg = self.config
        device = next(model.parameters()).device
        N = len(frames)
        if N < cfg.warmup_frames:
            return {"skipped": True, "reason": "too_few_frames"}

        # Remove stale LoRA from previous window so Phase 1 uses clean model
        if hasattr(self, '_lora_injected') and self._lora_injected:
            remove_lora(model)
            self._lora_injected = False
            self._lora_params = None

        # === Phase 1: frozen streaming forward (clean model) ===
        print(f"  [GeoTTTv2-LoRA] Phase 1: frozen forward ({N} frames)...")
        past_kv_full, pts_list, pose_list, _, psi = self._frozen_forward(
            model, frames, dtype=dtype,
        )

        # === differentiable affine (ENU meters → DOM pixels) ===
        A, b = build_differentiable_projection(project_fn, enu_offset, device)

        # === Model→ENU alignment ===
        if gt_enu_window is not None and len(gt_enu_window) == N:
            from streamvggt.utils.rotation import quat_to_mat
            pose_stack = torch.cat(pose_list, dim=0)  # [N, 9]
            T_cam = pose_stack[:, :3]
            quat = pose_stack[:, 3:7]
            R_cam = quat_to_mat(quat)
            model_centers = -torch.bmm(
                R_cam.transpose(1, 2), T_cam.unsqueeze(-1)
            ).squeeze(-1).cpu().numpy()  # [N, 3]

            gt_local = gt_enu_window - enu_offset
            s_m2e, R_m2e, t_m2e = compute_model_to_enu_transform(
                model_centers, gt_local
            )

            sR = torch.tensor(s_m2e * R_m2e, dtype=torch.float32, device=device)
            t_vec = torch.tensor(t_m2e, dtype=torch.float32, device=device)
            b = A @ t_vec + b
            A = A @ sR

            print(f"    Model→ENU: scale={s_m2e:.2f}, "
                  f"t=({t_m2e[0]:.2f}, {t_m2e[1]:.2f}) m")
        else:
            print("    WARNING: no gt_enu_window — using raw affine (WILL BE WRONG)")

        # === DOM crop ===
        dom_crop, b_crop, crop_h, crop_w = crop_dom_for_window(
            dom_image_cpu, A, b, pts_list,
            padding=cfg.dom_crop_padding_px, device=device,
        )
        print(f"    DOM crop: {crop_h}×{crop_w}, "
              f"{dom_crop.nelement()*4/1e6:.0f} MB on GPU")

        del pts_list, pose_list
        torch.cuda.empty_cache()

        # === Phase 2: inject fresh LoRA adapters (one set per window) ===
        lora_params, num_params = inject_lora(
            model,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            target_blocks=cfg.lora_target_blocks,
            target_layers=cfg.lora_target_layers,
        )
        self._lora_params = lora_params
        self._lora_injected = True

        # Freeze everything except LoRA
        for p in model.parameters():
            p.requires_grad_(False)
        for p in lora_params:
            p.requires_grad_(True)

        optimizer = torch.optim.AdamW(lora_params, lr=cfg.lora_lr)

        # === Phase 3: KV-cache streaming optimization ===
        #
        # Detach the frozen KV-cache — no gradients through past context.
        # past_kv_full[layer] = (K, V) with K shape [B, H, N_frames, P, dim]
        past_kv_frozen = []
        for kv in past_kv_full:
            if kv is not None:
                k, v = kv
                past_kv_frozen.append((k.detach(), v.detach()))
            else:
                past_kv_frozen.append(None)
        del past_kv_full
        torch.cuda.empty_cache()

        aggregator = model.aggregator
        depth = aggregator.depth
        batch_size = max(2, int(N * cfg.lora_batch_ratio))
        print(f"  [GeoTTTv2-LoRA] TTT ({cfg.num_steps} steps, "
              f"batch={batch_size}/{N} frames, lr={cfg.lora_lr})")

        losses = []
        for step in range(cfg.num_steps):
            optimizer.zero_grad()

            # Mini-batch: random subset of frames
            fi_sel = sorted(
                np.random.choice(N, size=min(batch_size, N), replace=False).tolist()
            )

            step_loss = 0.0
            n_ok = 0

            for fi in fi_sel:
                # Slice frozen KV-cache to frames 0..fi-1
                if fi == 0:
                    past_kv_i = [None] * depth
                else:
                    past_kv_i = []
                    for kv in past_kv_frozen:
                        if kv is not None:
                            k, v = kv
                            past_kv_i.append((k[:, :, :fi], v[:, :, :fi]))
                        else:
                            past_kv_i.append(None)

                # Single-frame forward through LoRA-adapted aggregator
                images_i = frames[fi]["img"].unsqueeze(0)  # [B=1, S=1, 3, H, W]

                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images_i, past_key_values=past_kv_i,
                        use_cache=True, past_frame_idx=fi,
                    )
                agg_tokens = agg_out[0]
                psi_out = agg_out[1]

                with torch.cuda.amp.autocast(enabled=False):
                    agg_f = [t.float() for t in agg_tokens]
                    pts3d, pts3d_conf = model.point_head(
                        agg_f, images=images_i, patch_start_idx=psi_out,
                    )
                # pts3d: [B=1, S=1, H, W, 3]

                # DOM rendering loss for this frame
                pts_frame = pts3d[:, 0]  # [1, H, W, 3]
                pts_xy = pts_frame[..., :2]
                dom_uv = torch.einsum('ij,...j->...i', A, pts_xy) + b_crop

                grid_x = (dom_uv[..., 0] / max(crop_w - 1, 1)) * 2.0 - 1.0
                grid_y = (dom_uv[..., 1] / max(crop_h - 1, 1)) * 2.0 - 1.0
                grid = torch.stack([grid_x, grid_y], dim=-1)

                sampled = F.grid_sample(
                    dom_crop, grid.float(), mode="bilinear",
                    padding_mode="border", align_corners=True,
                )

                _, _, Hpt, Wpt = sampled.shape
                cam = F.interpolate(
                    frames[fi]["img"][:1].float().to(device),
                    size=(Hpt, Wpt), mode="bilinear", align_corners=False,
                )

                if cfg.loss_type == "l1":
                    px = (sampled - cam).abs().mean(dim=1)
                else:
                    px = (sampled - cam).pow(2).mean(dim=1)

                in_bounds = (grid[..., 0].abs() < 1) & (grid[..., 1].abs() < 1)
                px = px * in_bounds.float()

                if cfg.conf_weighted:
                    cw = pts3d_conf[:, 0].detach().clamp(min=1.0)
                    cw = (cw - 1.0) / (cw.max() + 1e-8)
                    px = px * cw

                frame_loss = px.sum() / in_bounds.float().sum().clamp(min=1.0)

                # Backward with gradient accumulation (divide by batch for avg)
                (frame_loss / batch_size).backward()
                step_loss += frame_loss.item()
                n_ok += 1

                # Free computation graph for this frame
                del agg_out, agg_tokens, agg_f, pts3d, pts3d_conf
                del pts_frame, pts_xy, dom_uv, grid, sampled, cam, px, frame_loss
                del past_kv_i
                torch.cuda.empty_cache()

            avg_loss = step_loss / max(n_ok, 1)

            # LoRA regularization (L2 on LoRA params)
            if cfg.reg_weight > 0:
                reg = sum(p.pow(2).sum() for p in lora_params)
                (cfg.reg_weight * reg).backward()

            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)

            g = sum(p.grad.norm().item() for p in lora_params if p.grad is not None)
            optimizer.step()

            losses.append(avg_loss)
            if step % 10 == 0 or step == cfg.num_steps - 1:
                print(f"    step {step:3d}: loss={avg_loss:.5f} |grad|={g:.4f}")

        del past_kv_frozen, dom_crop
        torch.cuda.empty_cache()

        self._per_frame_params = None

        return {
            "skipped": False,
            "losses": losses,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
        }

    # ------------------------------------------------------------------
    #  LoRA mode: compute refined poses with adapted model
    # ------------------------------------------------------------------
    def compute_adapted_poses_lora(self, model, frames, dtype=torch.bfloat16):
        """Re-run streaming inference with LoRA-adapted model to get refined poses."""
        aggregator = model.aggregator
        camera_head = model.camera_head
        depth = aggregator.depth

        past_kv = [None] * depth
        past_kv_cam = [None] * camera_head.trunk_depth
        poses = []

        with torch.no_grad():
            for i, frame in enumerate(frames):
                images = frame["img"].unsqueeze(0)
                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images, past_key_values=past_kv,
                        use_cache=True, past_frame_idx=i,
                    )
                if isinstance(agg_out, tuple) and len(agg_out) == 3:
                    agg_tokens, psi, past_kv = agg_out
                else:
                    agg_tokens, psi = agg_out

                with torch.cuda.amp.autocast(enabled=False):
                    pose_enc, past_kv_cam = camera_head(
                        agg_tokens, past_key_values_camera=past_kv_cam,
                        use_cache=True,
                    )
                    poses.append(pose_enc[-1][:, 0, :].float())

        del past_kv, past_kv_cam
        return poses

    # ------------------------------------------------------------------
    #  KV-cache mode
    # ------------------------------------------------------------------
    def _adapt_window_kv(self, model, frames, dom_image_cpu, project_fn,
                         enu_offset, dtype=torch.bfloat16):
        cfg = self.config
        device = next(model.parameters()).device
        depth = model.aggregator.depth
        N = len(frames)
        if N < cfg.warmup_frames:
            return {"skipped": True, "reason": "too_few_frames"}

        # === Phase 1 ===
        print(f"  [GeoTTTv2] Phase 1: frozen forward ({N} frames)...")
        past_kv, pts_list, pose_list, cam_tok_list, psi = self._frozen_forward(
            model, frames, dtype=dtype,
        )

        # === differentiable affine ===
        A, b = build_differentiable_projection(project_fn, enu_offset, device)

        # === DOM crop ===
        dom_crop, b_crop, crop_h, crop_w = crop_dom_for_window(
            dom_image_cpu, A, b, pts_list,
            padding=cfg.dom_crop_padding_px, device=device,
        )
        print(f"    DOM crop: {crop_h}×{crop_w}, "
              f"{dom_crop.nelement()*4/1e6:.0f} MB on GPU")

        # === Phase 2 ===
        update_layers = self._update_layers(depth)
        leaf_kv, kv_leaves, kv_originals = self._setup_kv_leaves(
            past_kv, depth, update_layers,
        )
        n_p = sum(p.numel() for p in kv_leaves)
        print(f"    KV leaves: {len(kv_leaves)} tensors, {n_p/1e6:.1f}M params")

        optimizer = torch.optim.Adam(kv_leaves, lr=cfg.kv_lr)

        # frame selection (skip frame 0 — has no KV context)
        max_f = min(N, cfg.max_frames_for_loss)
        fi_sel = np.unique(np.linspace(1, N - 1, max_f, dtype=int)).tolist()

        for p in model.parameters():
            p.requires_grad_(False)

        # === Phase 3–4 ===
        losses = []
        print(f"  [GeoTTTv2] Phase 3-4: TTT ({cfg.num_steps} steps, "
              f"{len(fi_sel)} frames, lr={cfg.kv_lr})")

        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            s_loss, n_ok = 0.0, 0
            for fi in fi_sel:
                loss = self._frame_loss(
                    model, frames[fi], fi, leaf_kv, depth,
                    A, b_crop, dom_crop, crop_h, crop_w,
                    cam_tok_list[fi], dtype, device,
                )
                (loss / len(fi_sel)).backward()
                s_loss += loss.item()
                n_ok += 1
                del loss

            if cfg.reg_weight > 0:
                reg = sum(
                    (p - o.to(device)).pow(2).mean()
                    for p, o in zip(kv_leaves, kv_originals)
                )
                (cfg.reg_weight * reg).backward()

            g = sum(p.grad.norm().item() for p in kv_leaves if p.grad is not None)
            optimizer.step()

            avg = s_loss / max(n_ok, 1)
            losses.append(avg)
            print(f"    step {step}: loss={avg:.4f}, grad={g:.2e}")

        del dom_crop
        for p in model.parameters():
            p.requires_grad_(False)
        torch.cuda.empty_cache()

        self._leaf_kv = leaf_kv
        self._depth = depth

        return {
            "skipped": False,
            "losses": losses,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
        }

    # ------------------------------------------------------------------
    #  Phase 5: re-run with adapted KV → refined poses
    # ------------------------------------------------------------------
    def compute_adapted_poses(self, model, frames, dtype=torch.bfloat16):
        leaf_kv = self._leaf_kv
        depth = self._depth
        aggregator = model.aggregator
        camera_head = model.camera_head

        past_kv_cam = [None] * camera_head.trunk_depth
        poses = []

        with torch.no_grad():
            for i, frame in enumerate(frames):
                images = frame["img"].unsqueeze(0)
                kv_for = []
                for j in range(depth):
                    if leaf_kv[j] is not None and i > 0:
                        k, v = leaf_kv[j]
                        kv_for.append((k[:, :, :i].detach(), v[:, :, :i].detach()))
                    else:
                        kv_for.append(None)

                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images, past_key_values=kv_for,
                        use_cache=True, past_frame_idx=i,
                    )
                agg_tokens = agg_out[0]
                with torch.cuda.amp.autocast(enabled=False):
                    pose_enc, past_kv_cam = camera_head(
                        agg_tokens, past_key_values_camera=past_kv_cam,
                        use_cache=True,
                    )
                    poses.append(pose_enc[-1][:, 0, :].float())
                del kv_for
        return poses

    def compute_adapted_pts3d(self, model, frames, dtype=torch.bfloat16,
                              stride=1, kv_override=None):
        """Re-run with adapted KV to collect pts3d from point_head.

        Args:
            model: StreamVGGT model.
            frames: list of frame dicts with 'img' key.
            dtype: precision for aggregator.
            stride: collect pts3d every `stride` frames (1 = all).
            kv_override: if provided, use this KV instead of self._leaf_kv
                         (e.g. original pre-TTT KV for baseline).

        Returns:
            pts3d_list: list of dicts per frame (or None for skipped frames).
                Each dict: {'pts3d': [H,W,3] float32, 'conf': [H,W] float32,
                            'rgb': [H,W,3] float32}
        """
        leaf_kv = kv_override if kv_override is not None else self._leaf_kv
        depth = self._depth
        aggregator = model.aggregator
        point_head = model.point_head

        pts3d_list = []

        with torch.no_grad():
            for i, frame in enumerate(frames):
                if i % stride != 0:
                    pts3d_list.append(None)
                    continue

                images = frame["img"].unsqueeze(0)
                kv_for = []
                for j in range(depth):
                    if leaf_kv[j] is not None and i > 0:
                        k, v = leaf_kv[j]
                        kv_for.append((k[:, :, :i].detach(),
                                       v[:, :, :i].detach()))
                    else:
                        kv_for.append(None)

                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images, past_key_values=kv_for,
                        use_cache=True, past_frame_idx=i,
                    )
                agg_tokens = agg_out[0]
                psi = agg_out[1] if isinstance(agg_out, tuple) and len(agg_out) >= 2 else aggregator.patch_start_idx

                with torch.cuda.amp.autocast(enabled=False):
                    agg_f = [t.float() for t in agg_tokens]
                    pts3d, pts3d_conf = point_head(
                        agg_f, images=images, patch_start_idx=psi,
                    )

                # pts3d: [1, 1, H, W, 3] → [H, W, 3]
                pts_hw3 = pts3d[0, 0].cpu().float()
                conf_hw = pts3d_conf[0, 0].cpu().float() if pts3d_conf is not None else None
                # RGB from input image: frame["img"] is [1, 3, H_img, W_img]
                img_3hw = frame["img"][0].cpu().float()  # [3, H_img, W_img]
                # Resize image to pts3d resolution if needed
                H_p, W_p = pts_hw3.shape[:2]
                H_img, W_img = img_3hw.shape[1], img_3hw.shape[2]
                if H_img != H_p or W_img != W_p:
                    img_3hw = torch.nn.functional.interpolate(
                        img_3hw.unsqueeze(0), size=(H_p, W_p),
                        mode='bilinear', align_corners=False,
                    )[0]
                rgb_hw3 = img_3hw.permute(1, 2, 0)  # [H, W, 3]

                pts3d_list.append({
                    'pts3d': pts_hw3,
                    'conf': conf_hw,
                    'rgb': rgb_hw3,
                })
                del kv_for, agg_tokens, agg_f

        return pts3d_list

    # ==================================================================
    #  TOKEN mode — optimise aggregated tokens directly (bypass softmax)
    # ==================================================================

    def _frozen_forward_with_tokens(self, model, frames, dtype=torch.bfloat16):
        """Frozen streaming forward — caches per-frame aggregated tokens
        (at the DPT skip layers) plus poses for later use."""
        aggregator = model.aggregator
        camera_head = model.camera_head
        depth = aggregator.depth

        dpt_layers = list(model.point_head.intermediate_layer_idx)
        psi = aggregator.patch_start_idx

        past_kv = [None] * depth
        past_kv_cam = [None] * camera_head.trunk_depth
        # per_frame_tokens: list[dict[layer_idx -> Tensor]]
        per_frame_tokens = []   # one dict per frame
        per_frame_all_tokens = []  # full agg_tokens list per frame
        per_frame_pose = []

        with torch.no_grad():
            for i, frame in enumerate(frames):
                images = frame["img"].unsqueeze(0)
                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images, past_key_values=past_kv,
                        use_cache=True, past_frame_idx=i,
                    )
                if isinstance(agg_out, tuple) and len(agg_out) == 3:
                    agg_tokens, psi_out, past_kv = agg_out
                else:
                    agg_tokens, psi_out = agg_out
                psi = psi_out

                with torch.cuda.amp.autocast(enabled=False):
                    pose_enc, past_kv_cam = camera_head(
                        agg_tokens, past_key_values_camera=past_kv_cam,
                        use_cache=True,
                    )
                    per_frame_pose.append(
                        pose_enc[-1][:, 0, :].detach().float()
                    )

                # Cache tokens at DPT layers (detached, float32)
                tok_dict = {}
                for li in dpt_layers:
                    tok_dict[li] = agg_tokens[li].detach().float()
                per_frame_tokens.append(tok_dict)

                # Cache full agg_tokens for later (all layers, detached)
                per_frame_all_tokens.append(
                    [t.detach() for t in agg_tokens]
                )

        del past_kv, past_kv_cam
        torch.cuda.empty_cache()
        return per_frame_tokens, per_frame_all_tokens, per_frame_pose, psi

    def _token_frame_loss(self, model, frame, fi, token_leaves,
                          all_tokens_orig, psi,
                          A, b_crop, dom_crop, crop_h, crop_w, device):
        """
        Compute DOM re-rendering loss using the *leaf* token dict (with
        gradient), falling back to original detached tokens for non-DPT
        layers.  No aggregator forward needed — just DPT + affine + grid_sample.
        """
        cfg = self.config
        dpt_layers = set(token_leaves.keys())
        n_layers = len(all_tokens_orig)

        # Build agg_tokens list: use leaf tokens at DPT layers, frozen elsewhere
        agg_tokens = []
        for li in range(n_layers):
            if li in dpt_layers:
                agg_tokens.append(token_leaves[li])
            else:
                agg_tokens.append(all_tokens_orig[li])

        images = frame["img"].unsqueeze(0)  # [1,1,3,H,W]

        with torch.cuda.amp.autocast(enabled=False):
            agg_f = [t.float() for t in agg_tokens]
            pts3d, pts3d_conf = model.point_head(
                agg_f, images=images, patch_start_idx=psi,
            )
        pts3d = pts3d[:, 0]           # [B, H, W, 3]
        pts3d_conf = pts3d_conf[:, 0]

        # --- differentiable projection ---
        pts_xy = pts3d[..., :2]
        dom_uv = torch.einsum('ij,...j->...i', A, pts_xy) + b_crop

        grid_x = (dom_uv[..., 0] / max(crop_w - 1, 1)) * 2.0 - 1.0
        grid_y = (dom_uv[..., 1] / max(crop_h - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        sampled = F.grid_sample(
            dom_crop, grid.float(), mode="bilinear",
            padding_mode="border", align_corners=True,
        )

        _, _, Hpt, Wpt = sampled.shape
        cam = F.interpolate(
            frame["img"][:1].float().to(device),
            size=(Hpt, Wpt), mode="bilinear", align_corners=False,
        )

        if cfg.loss_type == "l1":
            px = (sampled - cam).abs().mean(dim=1)
        elif cfg.loss_type == "mse":
            px = (sampled - cam).pow(2).mean(dim=1)
        else:
            px = F.smooth_l1_loss(sampled, cam, reduction='none').mean(dim=1)

        in_bounds = (grid[..., 0].abs() < 1) & (grid[..., 1].abs() < 1)
        px = px * in_bounds.float()

        if cfg.conf_weighted:
            cw = pts3d_conf.detach().clamp(min=1.0)
            cw = (cw - 1.0) / (cw.max() + 1e-8)
            px = px * cw

        return px.sum() / in_bounds.float().sum().clamp(min=1.0)

    def _adapt_window_token(self, model, frames, dom_image_cpu, project_fn,
                            enu_offset, dtype=torch.bfloat16):
        cfg = self.config
        device = next(model.parameters()).device
        N = len(frames)
        if N < cfg.warmup_frames:
            return {"skipped": True, "reason": "too_few_frames"}

        # === Phase 1: frozen forward ===
        print(f"  [GeoTTTv2-token] Phase 1: frozen forward ({N} frames)...")
        (per_frame_tokens, per_frame_all_tokens,
         per_frame_pose, psi) = self._frozen_forward_with_tokens(
            model, frames, dtype=dtype,
        )

        dpt_layers = list(model.point_head.intermediate_layer_idx)

        # === differentiable affine ===
        A, b = build_differentiable_projection(project_fn, enu_offset, device)

        # === DOM crop (use frozen pts3d projected from cached tokens) ===
        # Compute pts3d from frozen tokens for crop sizing
        pts_list = []
        with torch.no_grad():
            for fi in range(N):
                agg_f = [per_frame_all_tokens[fi][li].float()
                         for li in range(len(per_frame_all_tokens[fi]))]
                images = frames[fi]["img"].unsqueeze(0)
                pts3d, _ = model.point_head(agg_f, images=images, patch_start_idx=psi)
                pts_list.append(pts3d[:, 0].float())

        dom_crop, b_crop, crop_h, crop_w = crop_dom_for_window(
            dom_image_cpu, A, b, pts_list,
            padding=cfg.dom_crop_padding_px, device=device,
        )
        del pts_list
        print(f"    DOM crop: {crop_h}×{crop_w}, "
              f"{dom_crop.nelement()*4/1e6:.0f} MB on GPU")

        # === Phase 2: make DPT-layer tokens into leaves ===
        all_leaves = []       # flat list for optimizer
        all_originals = []    # for regularisation
        per_frame_leaf_tokens = []  # per-frame dict of leaf tensors

        for fi in range(N):
            tok_leaves = {}
            for li in dpt_layers:
                t = per_frame_tokens[fi][li].to(device).requires_grad_(True)
                tok_leaves[li] = t
                all_leaves.append(t)
                all_originals.append(per_frame_tokens[fi][li].to(device).clone())
            per_frame_leaf_tokens.append(tok_leaves)

        n_p = sum(p.numel() for p in all_leaves)
        print(f"    Token leaves: {len(all_leaves)} tensors, {n_p/1e6:.1f}M params "
              f"({len(dpt_layers)} DPT layers × {N} frames)")

        optimizer = torch.optim.Adam(all_leaves, lr=cfg.kv_lr)

        # frame selection
        max_f = min(N, cfg.max_frames_for_loss)
        fi_sel = list(range(N)) if N <= max_f else \
            np.unique(np.linspace(0, N - 1, max_f, dtype=int)).tolist()

        for p in model.parameters():
            p.requires_grad_(False)

        # === Phase 3-4: TTT ===
        losses = []
        print(f"  [GeoTTTv2-token] Phase 3-4: TTT ({cfg.num_steps} steps, "
              f"{len(fi_sel)} frames, lr={cfg.kv_lr})")

        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            s_loss, n_ok = 0.0, 0

            for fi in fi_sel:
                loss = self._token_frame_loss(
                    model, frames[fi], fi,
                    per_frame_leaf_tokens[fi],
                    per_frame_all_tokens[fi],
                    psi, A, b_crop, dom_crop, crop_h, crop_w, device,
                )
                (loss / len(fi_sel)).backward()
                s_loss += loss.item()
                n_ok += 1
                del loss

            if cfg.reg_weight > 0:
                reg = sum(
                    (p - o).pow(2).mean()
                    for p, o in zip(all_leaves, all_originals)
                )
                (cfg.reg_weight * reg).backward()

            g = sum(p.grad.norm().item() for p in all_leaves if p.grad is not None)
            optimizer.step()

            avg = s_loss / max(n_ok, 1)
            losses.append(avg)
            print(f"    step {step}: loss={avg:.4f}, grad={g:.2e}")

        del dom_crop
        for p in model.parameters():
            p.requires_grad_(False)
        torch.cuda.empty_cache()

        # Store adapted tokens + originals for Phase 5
        self._per_frame_leaf_tokens = per_frame_leaf_tokens
        self._per_frame_all_tokens = per_frame_all_tokens
        self._per_frame_pose = per_frame_pose
        self._psi = psi

        return {
            "skipped": False,
            "losses": losses,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
        }

    def compute_adapted_poses_token(self, model, frames, dtype=torch.bfloat16):
        """Phase 5 for token mode: use ORIGINAL poses (camera tokens untouched)."""
        return self._per_frame_pose

    # ==================================================================
    #  POSE_TTT mode — KV-cache TTT with direct camera center loss
    # ==================================================================

    def _adapt_window_pose_ttt(self, model, frames, dom_image_cpu,
                               project_fn, enu_offset,
                               dtype=torch.bfloat16,
                               gt_enu_window=None,
                               correspondences=None,
                               roma_model=None, image_paths=None,
                               inv_project_fn=None, geo_elev=None,
                               dom_transform=None):
        """Adapt Aggregator KV-cache using direct camera center loss.

        Uses RoMa-derived per-frame ENU positions as supervision to optimize
        the Aggregator's KV-cache. Gradient path:
          KV → Aggregator → agg_tokens → CameraHead → pose_enc → centers → loss

        After TTT, re-runs full inference with adapted KV to get improved poses.
        Produces direct ENU trajectories (baseline + TTT) via Sim2 transform.
        """
        from streamvggt.utils.trajectory_chain import pose_enc_to_camera_centers

        N = len(frames)
        cfg = self.config
        device = frames[0]["img"].device
        aggregator = model.aggregator
        camera_head = model.camera_head
        depth = aggregator.depth

        if N < cfg.warmup_frames:
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True, "reason": "too_few_frames"}

        # ====== Phase 1: Frozen forward — build KV caches ======
        past_kv = [None] * depth
        past_kv_cam = [None] * camera_head.trunk_depth
        per_frame_pose = []

        with torch.no_grad():
            for i, frame in enumerate(frames):
                images = frame["img"].unsqueeze(0)
                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images, past_key_values=past_kv,
                        use_cache=True, past_frame_idx=i,
                    )
                if isinstance(agg_out, tuple) and len(agg_out) == 3:
                    agg_tokens, psi, past_kv = agg_out
                else:
                    agg_tokens, psi = agg_out

                with torch.cuda.amp.autocast(enabled=False):
                    pose_enc, past_kv_cam = camera_head(
                        agg_tokens, past_key_values_camera=past_kv_cam,
                        use_cache=True,
                    )
                    per_frame_pose.append(
                        pose_enc[-1][:, 0, :].detach().float()
                    )

        self._per_frame_pose = per_frame_pose
        pose_encs = torch.cat(per_frame_pose, dim=0)
        centers_model = pose_enc_to_camera_centers(pose_encs).cpu().numpy()  # (N, 3)

        # ====== Phase 1.5: Streaming DOM matching (if needed) ======
        if correspondences is None and roma_model is not None and image_paths is not None:
            print(f"  [pose_ttt] Streaming DOM matching ({N} frames)...")
            _dom_for_match = dom_image_cpu.squeeze(0) if dom_image_cpu.dim() == 4 else dom_image_cpu
            correspondences = self._streaming_dom_match(
                centers_model, image_paths, enu_offset,
                roma_model, _dom_for_match, project_fn, inv_project_fn,
                cfg, device=str(device),
                geo_elev=geo_elev, dom_transform=dom_transform,
            )

        # ====== Phase 2: ENU targets + Sim2 ======
        cx_img = (cfg.camera_orig_w or 1600) / 2.0
        cy_img = (cfg.camera_orig_h or 1200) / 2.0

        optimal_enu = {}
        optimal_enu_3d = {}
        residuals = {}
        matched_frames = []
        n_pnp_used = 0

        if correspondences:
            for fi, corr in correspondences.items():
                pixels = corr['frame_pixels']
                enu_abs = corr['enu_positions']
                K = len(pixels)
                if K < 4:
                    continue
                pnp = corr.get('pnp_result')
                if pnp is not None and pnp.get('success'):
                    optimal_enu[fi] = pnp['t_world'][:2].copy()
                    optimal_enu_3d[fi] = pnp['t_world'].copy()
                    residuals[fi] = np.full(K, pnp.get('reproj_err', 1.0))
                    n_pnp_used += 1
                    continue
                ones = np.ones((K, 1))
                design = np.column_stack([pixels, ones])
                params_E, _, _, _ = np.linalg.lstsq(design, enu_abs[:, 0], rcond=None)
                params_N, _, _, _ = np.linalg.lstsq(design, enu_abs[:, 1], rcond=None)
                optimal_enu[fi] = np.array([
                    params_E[0] * cx_img + params_E[1] * cy_img + params_E[2],
                    params_N[0] * cx_img + params_N[1] * cy_img + params_N[2],
                ])
                pred_E = design @ params_E
                pred_N = design @ params_N
                res_mag = np.sqrt((pred_E - enu_abs[:, 0])**2 + (pred_N - enu_abs[:, 1])**2)
                residuals[fi] = res_mag
                med_res = np.median(res_mag)
                inlier_mask = res_mag < 3 * med_res + 1.0
                if inlier_mask.sum() >= 4:
                    design_in = design[inlier_mask]
                    p_E, _, _, _ = np.linalg.lstsq(design_in, enu_abs[inlier_mask, 0], rcond=None)
                    p_N, _, _, _ = np.linalg.lstsq(design_in, enu_abs[inlier_mask, 1], rcond=None)
                    optimal_enu[fi] = np.array([
                        p_E[0] * cx_img + p_E[1] * cy_img + p_E[2],
                        p_N[0] * cx_img + p_N[1] * cy_img + p_N[2],
                    ])
            matched_frames = sorted(optimal_enu.keys())

        if n_pnp_used > 0:
            print(f"  [pose_ttt] PnP used for {n_pnp_used}/{len(correspondences)} frames")

        # --- Step 2b: Build Sim2 correspondences ---
        if matched_frames:
            src_pts = np.array([centers_model[fi, :2] for fi in matched_frames])
            dst_pts = np.array([optimal_enu[fi] - enu_offset[:2] for fi in matched_frames])
        else:
            src_pts = np.zeros((0, 2))
            dst_pts = np.zeros((0, 2))

        if len(src_pts) == 0:
            print("  [pose_ttt] No DOM matches, skipping")
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True}

        # Estimate Sim2
        n_total_pts = len(src_pts)
        if n_total_pts >= 4:
            s_m2e, R_m2e, t_m2e, _inlier_mask = ransac_sim2(
                src_pts, dst_pts, reproj_thresh=10.0, max_iters=2000)
            n_inliers = int(_inlier_mask.sum())
            print(f"  [pose_ttt] Sim2 (RANSAC): s={s_m2e:.4f}, "
                  f"{n_inliers}/{n_total_pts} inliers "
                  f"({len(matched_frames)} DOM frames)")
        elif n_total_pts >= 2:
            from streamvggt.utils.trajectory_chain import compute_model_to_enu_transform
            s_m2e, R_m2e, t_m2e = compute_model_to_enu_transform(
                np.vstack([src_pts, np.zeros((len(src_pts), 1))]),
                np.column_stack([dst_pts, np.zeros(len(dst_pts))]))
            print(f"  [pose_ttt] Sim2 (Procrustes, {n_total_pts} pts): s={s_m2e:.4f}")
        else:
            s_m2e, R_m2e, t_m2e = 1.0, np.eye(2), np.zeros(2)
            print(f"  [pose_ttt] Sim2 fallback (identity)")

        if abs(s_m2e) < 1e-8:
            print(f"  [pose_ttt] Degenerate Sim2 (scale={s_m2e:.2e}), skipping TTT")
            self._leaf_kv = past_kv
            self._depth = depth
            self._per_frame_pose = per_frame_pose
            self._geo_token_corrections = np.zeros((N, 2))
            self._per_frame_params = None
            self._centroids = None
            for p in model.parameters():
                p.requires_grad_(False)
            return {
                "skipped": False,
                "losses": [0.0],
                "n_matched": len(matched_frames),
                "optimal_enu": {fi: optimal_enu[fi].copy() for fi in matched_frames},
                "optimal_enu_3d": {fi: optimal_enu_3d[fi].copy() for fi in matched_frames if fi in optimal_enu_3d},
                "fit_residuals": {fi: float(np.median(residuals[fi]))
                                  for fi in matched_frames if fi in residuals},
            }
        if not matched_frames:
            print(f"  [pose_ttt] No dense targets, skipping TTT")
            self._leaf_kv = past_kv
            self._depth = depth
            self._per_frame_pose = per_frame_pose
            self._geo_token_corrections = np.zeros((N, 2))
            self._per_frame_params = None
            self._centroids = None

            bl_enu = np.zeros((N, 2))
            for fi in range(N):
                bl_enu[fi] = s_m2e * R_m2e @ centers_model[fi, :2] + t_m2e
            bl_enu_abs = bl_enu + enu_offset[:2]

            if hasattr(self, '_streaming_last_enu'):
                self._streaming_last_enu = bl_enu_abs[-1].copy()

            return {
                "skipped": False,
                "losses": [],
                "n_matched": 0,
                "optimal_enu": {},
                "optimal_enu_3d": {},
                "fit_residuals": {},
                "baseline_centers": centers_model.copy(),
                "ttt_enu_trajectory": bl_enu_abs.copy(),
                "baseline_enu_trajectory": bl_enu_abs.copy(),
            }

        sR_inv = np.linalg.inv(s_m2e * R_m2e)
        target_model_xy = {}
        for fi in matched_frames:
            enu_local = optimal_enu[fi] - enu_offset[:2]
            target_model_xy[fi] = (enu_local - t_m2e) @ sR_inv.T

        target_tensor = torch.zeros(N, 2, device=device, dtype=torch.float32)
        for fi in matched_frames:
            target_tensor[fi] = torch.tensor(target_model_xy[fi],
                                             dtype=torch.float32, device=device)

        # ====== Phase 3: KV-cache TTT with camera center loss ======
        # Save original (pre-TTT) KV for baseline comparison
        self._original_past_kv = []
        for j in range(depth):
            if past_kv[j] is not None:
                k, v = past_kv[j]
                self._original_past_kv.append(
                    (k.detach().clone(), v.detach().clone()))
            else:
                self._original_past_kv.append(None)

        update_layers = self._update_layers(depth)
        leaf_kv, kv_leaves, kv_originals = self._setup_kv_leaves(
            past_kv, depth, update_layers)
        n_p = sum(p.numel() for p in kv_leaves)

        for p in model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(kv_leaves, lr=cfg.kv_lr)

        # Select frames for loss
        max_f = min(len(matched_frames), cfg.max_frames_for_loss)
        if len(matched_frames) > max_f:
            sel_idx = np.unique(np.linspace(0, len(matched_frames) - 1,
                                            max_f, dtype=int))
            fi_sel = [matched_frames[i] for i in sel_idx]
        else:
            fi_sel = matched_frames

        # Exclude frame 0 (no KV context → KV leaves not in graph → backward fails)
        fi_sel = [fi for fi in fi_sel if fi > 0]
        if not fi_sel:
            if len(matched_frames) > 1:
                fi_sel = matched_frames[1:2]
            else:
                # Only frame 0 matched — cannot do TTT (no KV context)
                print("  [pose_ttt] Only frame 0 matched, skipping TTT")
                self._leaf_kv = past_kv
                self._depth = depth
                self._per_frame_pose = per_frame_pose
                self._geo_token_corrections = np.zeros((N, 2))
                self._per_frame_params = None
                self._centroids = None
                for p in model.parameters():
                    p.requires_grad_(False)
                return {
                    "skipped": False,
                    "losses": [0.0],
                    "n_matched": len(matched_frames),
                    "optimal_enu": {fi: optimal_enu[fi].copy() for fi in matched_frames},
                    "optimal_enu_3d": {fi: optimal_enu_3d[fi].copy() for fi in matched_frames if fi in optimal_enu_3d},
                    "fit_residuals": {fi: float(np.median(residuals[fi]))
                                      for fi in matched_frames if fi in residuals},
                }

        _loss_mode = cfg.pose_ttt_loss  # "mse" | "dense_reproj" | "both"
        _do_mse = _loss_mode in ("mse", "both")
        _do_dense = _loss_mode in ("dense_reproj", "both")

        # Pre-compute dense reproj targets in model space
        _dense_targets = {}  # fi → (M, 5) tensor [u, v, x_enu_local, y_enu_local, z_enu]
        if _do_dense:
            for fi in fi_sel:
                corr = correspondences.get(fi)
                if corr is None:
                    continue
                px = corr['frame_pixels']  # (K, 2) in original image coords
                enu = corr['enu_positions']  # (K, 2) or (K, 3) absolute ENU
                K = len(px)
                if K < 4:
                    continue
                # Store ENU local (subtract enu_offset); compare in real-world space
                # Avoids stale-target issue: pts3d → ENU via forward Sim2 in TTT loop
                enu_local = enu.copy()
                enu_local[:, :2] = enu_local[:, :2] - enu_offset[:2]
                target_enu = np.zeros((K, 3))
                target_enu[:, :2] = enu_local[:, :2]
                if enu.shape[1] >= 3:
                    target_enu[:, 2] = enu_local[:, 2]
                _dense_targets[fi] = torch.tensor(
                    np.column_stack([px, target_enu]),
                    dtype=torch.float32, device=device)  # (K, 5): [u, v, x_enu, y_enu, z_enu]

        if _do_dense:
            n_dense_fi = sum(1 for fi in fi_sel if fi in _dense_targets)
            print(f"  [pose_ttt] Dense reproj: {n_dense_fi} frames with targets")

        print(f"  [pose_ttt] {len(matched_frames)} matched, "
              f"{len(fi_sel)} loss frames, KV={n_p/1e6:.1f}M params, "
              f"lr={cfg.kv_lr}, steps={cfg.num_steps}, "
              f"loss={_loss_mode}")

        point_head = model.point_head if _do_dense else None

        # Precompute Sim2 forward transform tensors (model space → ENU local)
        # Forward: enu_xy = pts_model_xy @ sR_fwd.T + t_fwd
        sR_fwd = torch.tensor(s_m2e * R_m2e, dtype=torch.float32, device=device)  # (2, 2)
        t_fwd = torch.tensor(t_m2e, dtype=torch.float32, device=device)            # (2,)
        s_fwd = float(s_m2e)

        losses = []
        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            step_loss = 0.0
            step_dense_loss = 0.0
            n_ok = 0
            n_dense_ok = 0

            for fi in fi_sel:
                # Trim aggregator KV to frames before fi
                kv_trim = []
                for j in range(depth):
                    if leaf_kv[j] is not None and fi > 0:
                        k, v = leaf_kv[j]
                        kv_trim.append((k[:, :, :fi], v[:, :, :fi]))
                    else:
                        kv_trim.append(None)

                images = frames[fi]["img"].unsqueeze(0)
                with torch.cuda.amp.autocast(dtype=dtype):
                    agg_out = aggregator(
                        images, past_key_values=kv_trim,
                        use_cache=True, past_frame_idx=fi,
                    )
                agg_tokens = agg_out[0]

                frame_loss = torch.tensor(0.0, device=device)

                # --- MSE center loss ---
                if _do_mse:
                    cam_kv_trim = []
                    for j in range(camera_head.trunk_depth):
                        if past_kv_cam[j] is not None and fi > 0:
                            ck, cv = past_kv_cam[j]
                            cam_kv_trim.append((ck[:, :, :fi].detach(),
                                                cv[:, :, :fi].detach()))
                        else:
                            cam_kv_trim.append(None)

                    with torch.cuda.amp.autocast(enabled=False):
                        agg_tokens_f = [t.float() for t in agg_tokens]
                        pose_enc_new, _ = camera_head(
                            agg_tokens_f,
                            past_key_values_camera=cam_kv_trim,
                            use_cache=True,
                        )
                    pose_enc_fi = pose_enc_new[-1][:, 0, :]
                    center_new = pose_enc_to_camera_centers(pose_enc_fi)
                    mse_loss = F.mse_loss(center_new[0, :2], target_tensor[fi])
                    frame_loss = frame_loss + mse_loss
                    step_loss += mse_loss.item()
                    n_ok += 1
                    del pose_enc_new, center_new

                # --- Dense reproj loss on pts3d ---
                if _do_dense and fi in _dense_targets:
                    psi_fi = agg_out[1] if isinstance(agg_out, tuple) and len(agg_out) >= 2 else psi
                    with torch.cuda.amp.autocast(enabled=False):
                        agg_f = [t.float() for t in agg_tokens]
                        pts3d, pts3d_conf = point_head(
                            agg_f, images=images, patch_start_idx=psi_fi,
                        )
                    # pts3d: [1, 1, Hp, Wp, 3] → [1, 3, Hp, Wp] for grid_sample
                    pts3d_hw = pts3d[:, 0]  # [1, Hp, Wp, 3]
                    pts3d_chw = pts3d_hw.permute(0, 3, 1, 2).float()  # [1, 3, Hp, Wp]

                    corr_t = _dense_targets[fi]  # (K, 5): u, v, x_enu, y_enu, z_enu
                    img_w = cfg.camera_orig_w or 1600
                    img_h = cfg.camera_orig_h or 1200
                    # Normalize pixel coords to [-1, 1] for grid_sample
                    grid_x = (corr_t[:, 0] / (img_w - 1) * 2 - 1).float()
                    grid_y = (corr_t[:, 1] / (img_h - 1) * 2 - 1).float()
                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # [1,1,K,2]

                    sampled = F.grid_sample(
                        pts3d_chw, grid, align_corners=True,
                        mode='bilinear', padding_mode='border')  # [1,3,1,K]
                    pts_sampled = sampled.squeeze(0).squeeze(1).T  # (K, 3)

                    target_enu_t = corr_t[:, 2:5]  # (K, 3) ENU local

                    # Forward Sim2: model → ENU local (loss in real-world metres)
                    # enu_xy = pts_model_xy @ sR_fwd.T + t_fwd
                    enu_pred_xy = pts_sampled[:, :2] @ sR_fwd.T + t_fwd  # (K, 2)

                    # Check valid (positive depth in model space)
                    valid = pts_sampled[:, 2] > 0.01
                    if valid.sum() >= 3:
                        p_xy = enu_pred_xy[valid]
                        t_xy = target_enu_t[valid, :2]
                        xy_loss = F.huber_loss(p_xy, t_xy,
                                               delta=cfg.dense_reproj_huber_xy)
                        dense_loss = xy_loss
                        if corr_t.shape[1] >= 5 and target_enu_t[:, 2].abs().max() > 0.01:
                            enu_pred_z = pts_sampled[valid, 2] * s_fwd  # (K_valid,)
                            t_z = target_enu_t[valid, 2]
                            z_loss = F.huber_loss(enu_pred_z, t_z,
                                                  delta=cfg.dense_reproj_huber_z)
                            dense_loss = dense_loss + cfg.dense_reproj_z_weight * z_loss
                        frame_loss = frame_loss + cfg.dense_reproj_weight * dense_loss
                        step_dense_loss += dense_loss.item()
                        n_dense_ok += 1

                    del pts3d, pts3d_conf, pts3d_hw, pts3d_chw

                (frame_loss / len(fi_sel)).backward()
                del frame_loss, agg_tokens
                if _do_mse:
                    del agg_tokens_f

            # L2 regularization on KV changes
            if cfg.reg_weight > 0:
                reg = sum(
                    (p - o.to(device)).pow(2).mean()
                    for p, o in zip(kv_leaves, kv_originals)
                )
                (cfg.reg_weight * reg).backward()

            # Gradient clipping
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(kv_leaves, cfg.max_grad_norm)

            g = sum(p.grad.norm().item() for p in kv_leaves if p.grad is not None)

            # Per-layer gradient diagnostic (first step of first window only)
            if step == 0 and not hasattr(self, '_layer_grad_printed'):
                self._layer_grad_printed = True
                print("  [diag] Per-layer KV gradient norms:")
                for li in range(len(kv_leaves) // 2):
                    gk = kv_leaves[2*li].grad.norm().item() if kv_leaves[2*li].grad is not None else 0
                    gv = kv_leaves[2*li+1].grad.norm().item() if kv_leaves[2*li+1].grad is not None else 0
                    layer_idx = update_layers[li] if li < len(update_layers) else li
                    print(f"    layer {layer_idx:2d}: K_grad={gk:.2e}, V_grad={gv:.2e}")

            optimizer.step()

            avg = step_loss / max(n_ok, 1)
            avg_dense = step_dense_loss / max(n_dense_ok, 1)
            losses.append(avg + avg_dense)
            if _do_dense:
                print(f"    step {step}: mse={avg:.4f}, dense={avg_dense:.4f}, grad={g:.2e}")
            else:
                print(f"    step {step}: pose_loss={avg:.4f}, grad={g:.2e}")

        # ====== Phase 4: Re-run with adapted KV → improved poses ======
        self._leaf_kv = leaf_kv
        self._depth = depth
        adapted_poses = self.compute_adapted_poses(model, frames, dtype=dtype)
        self._per_frame_pose = adapted_poses

        adapted_encs = torch.cat([p for p in adapted_poses], dim=0)
        adapted_centers = pose_enc_to_camera_centers(adapted_encs).cpu().numpy()
        delta_model = np.linalg.norm(adapted_centers[:, :2] - centers_model[:, :2], axis=1)
        delta_enu = delta_model * s_m2e
        print(f"  [pose_ttt] Adaptation effect: center shift "
              f"mean={np.mean(delta_enu):.2f}m, max={np.max(delta_enu):.2f}m (ENU)")

        # ====== Per-layer KV signal attenuation diagnostics ======
        kv_layer_diag = {}
        with torch.no_grad():
            for li_idx, li in enumerate(update_layers):
                if leaf_kv[li] is None or self._original_past_kv[li] is None:
                    continue
                k_new, v_new = leaf_kv[li]
                k_orig, v_orig = self._original_past_kv[li]
                k_orig = k_orig.to(k_new.device)
                v_orig = v_orig.to(v_new.device)
                # Delta norms
                dk = (k_new - k_orig).norm().item()
                dv = (v_new - v_orig).norm().item()
                # Relative change (signal attenuation proxy)
                k_rel = dk / (k_orig.norm().item() + 1e-8)
                v_rel = dv / (v_orig.norm().item() + 1e-8)
                kv_layer_diag[li] = {
                    'dk_abs': dk, 'dv_abs': dv,
                    'dk_rel': k_rel, 'dv_rel': v_rel,
                    'k_norm': k_orig.norm().item(),
                    'v_norm': v_orig.norm().item(),
                }
        if kv_layer_diag:
            print(f"  [diag] Per-layer KV signal change after TTT ({len(kv_layer_diag)} layers):")
            print(f"  {'Layer':>6}  {'|ΔK|':>10}  {'|ΔK|/|K|':>10}  "
                  f"{'|ΔV|':>10}  {'|ΔV|/|V|':>10}  {'|K|':>10}  {'|V|':>10}")
            for li in sorted(kv_layer_diag.keys()):
                d = kv_layer_diag[li]
                print(f"  {li:6d}  {d['dk_abs']:10.4f}  {d['dk_rel']:10.6f}  "
                      f"{d['dv_abs']:10.4f}  {d['dv_rel']:10.6f}  "
                      f"{d['k_norm']:10.2f}  {d['v_norm']:10.2f}")

        # ====== Build per-frame ENU trajectories ======
        # TTT trajectory: adapted model centers → ENU via Sim2
        ttt_enu = np.zeros((N, 2))
        for fi in range(N):
            ttt_enu[fi] = s_m2e * R_m2e @ adapted_centers[fi, :2] + t_m2e
        ttt_enu_abs = ttt_enu + enu_offset[:2]

        # Baseline trajectory: original model centers → ENU via same Sim2
        bl_enu = np.zeros((N, 2))
        for fi in range(N):
            bl_enu[fi] = s_m2e * R_m2e @ centers_model[fi, :2] + t_m2e
        bl_enu_abs = bl_enu + enu_offset[:2]

        if hasattr(self, '_streaming_last_enu'):
            self._streaming_last_enu = ttt_enu_abs[-1].copy()

        for p in model.parameters():
            p.requires_grad_(False)
        torch.cuda.empty_cache()

        self._geo_token_corrections = np.zeros((N, 2))
        self._per_frame_params = None
        self._centroids = None

        return {
            "skipped": False,
            "losses": losses,
            "n_matched": len(matched_frames),
            "optimal_enu": {fi: optimal_enu[fi].copy() for fi in matched_frames},
            "optimal_enu_3d": {fi: optimal_enu_3d[fi].copy() for fi in matched_frames if fi in optimal_enu_3d},
            "fit_residuals": {fi: float(np.median(residuals[fi]))
                              for fi in matched_frames if fi in residuals},
            "baseline_centers": centers_model.copy(),
            "ttt_enu_trajectory": ttt_enu_abs.copy(),
            "baseline_enu_trajectory": bl_enu_abs.copy(),
            "kv_layer_diag": kv_layer_diag,
        }

    # ==================================================================
    #  Streaming DOM matching — use predicted poses as crop centres
    # ==================================================================

    def _streaming_dom_match(self, centers_model, image_paths, enu_offset,
                             roma_model, dom_image_cpu, project_fn,
                             inv_project_fn, cfg, device="cuda",
                             geo_elev=None, dom_transform=None,
                             save_vis_dir=None):
        """Streaming per-frame DOM matching using predicted poses as crop centres.

        Frame 0: crop centred at enu_offset (first-frame GPS).
        Once ≥4 anchors accumulated: estimate running Sim2 (model→ENU).
        Remaining frames: crop centred at Sim2(model_center).

        Uses self._streaming_sim2 to carry Sim2 across windows so that
        window-1+ frame-0 can immediately use a good crop centre.

        Returns:
            correspondences: dict  fi → corr dict (same format as
                             get_frame_dom_correspondences output).
        """
        from streamvggt.utils.dom_matching import get_frame_dom_correspondences

        N = len(image_paths)

        # Camera FOV dict for FOV-matched crop
        camera_fov = None
        if cfg.camera_fx is not None and cfg.base_altitude_m is not None:
            camera_fov = {
                'fx': cfg.camera_fx, 'fy': cfg.camera_fy,
                'orig_w': cfg.camera_orig_w, 'orig_h': cfg.camera_orig_h,
                'altitude': cfg.base_altitude_m,
            }

        stride = cfg.geo_consist_stride
        selected = sorted(set(list(range(0, N, stride)) + [N - 1]))

        # Initialise running Sim2 from previous window (if available).
        # Only scale and rotation carry across windows; translation is
        # window-local (depends on enu_offset) and must be re-estimated.
        if not hasattr(self, '_streaming_sim2_sR'):
            self._streaming_sim2_sR = None
        if self._streaming_sim2_sR is not None:
            s_est, R_est = self._streaming_sim2_sR
            # t is unknown for this window — will be set once first anchor arrives
            t_est = np.zeros(2)
            sim2_ready = False  # need ≥1 anchor to compute t for new window
            sR_from_prev = True
        else:
            s_est, R_est, t_est = 1.0, np.eye(2), np.zeros(2)
            sim2_ready = False
            sR_from_prev = False

        # Last known absolute ENU — fallback crop centre before Sim2 is ready
        if not hasattr(self, '_streaming_last_enu'):
            self._streaming_last_enu = enu_offset[:2].copy()
        last_enu = self._streaming_last_enu.copy()

        correspondences = {}
        n_ok = 0

        for fi in selected:
            # --- Determine approximate crop centre (absolute ENU) ---
            if fi == 0 and not sim2_ready:
                # Very first window, first frame: use GPS
                approx_enu = enu_offset.copy()
            elif sim2_ready:
                # Project model-space centre → ENU via running Sim2
                c_model = centers_model[fi, :2]
                approx_xy = s_est * R_est @ c_model + t_est + enu_offset[:2]
                approx_enu = np.array([approx_xy[0], approx_xy[1],
                                       enu_offset[2] if len(enu_offset) > 2 else 0.0])
            else:
                # Sim2 not yet ready — use last matched ENU
                approx_enu = np.array([last_enu[0], last_enu[1],
                                       enu_offset[2] if len(enu_offset) > 2 else 0.0])

            vis_path = None
            if save_vis_dir:
                vis_path = os.path.join(save_vis_dir, f"roma_match_f{fi:04d}.jpg")

            corr = get_frame_dom_correspondences(
                image_paths[fi], dom_image_cpu, approx_enu,
                project_fn, inv_project_fn, roma_model,
                crop_size_m=cfg.geo_consist_crop_m, device=device,
                max_correspondences=cfg.geo_consist_max_corr,
                save_vis=vis_path,
                camera_fov=camera_fov,
                geo_elev=geo_elev,
                dom_transform=dom_transform,
            )

            if corr is not None:
                correspondences[fi] = corr
                last_enu = corr['frame_center_enu'].copy()
                n_ok += 1

                # Update running Sim2 when enough anchors
                matched = sorted(correspondences.keys())
                if sR_from_prev and len(matched) >= 1 and not sim2_ready:
                    # Have s,R from previous window — compute t from first anchor
                    src = np.array([centers_model[matched[0], :2]])
                    dst = np.array([correspondences[matched[0]]['frame_center_enu']
                                    - enu_offset[:2]])
                    t_est = dst[0] - s_est * R_est @ src[0]
                    sim2_ready = True
                if len(matched) >= 4:
                    src = np.array([centers_model[f, :2] for f in matched])
                    dst = np.array([correspondences[f]['frame_center_enu']
                                    - enu_offset[:2] for f in matched])
                    s_est, R_est, t_est, _ = ransac_sim2(
                        src, dst, reproj_thresh=10.0, max_iters=2000)
                    sim2_ready = True
                    sR_from_prev = False  # now using fresh estimates

        # Persist (s, R) for next window; t is window-local
        self._streaming_sim2_sR = (s_est, R_est) if sim2_ready else None
        self._streaming_last_enu = last_enu.copy()

        if n_ok > 0:
            avg_corr = np.mean([c['n_inliers'] for c in correspondences.values()])
            print(f"  [streaming-DOM] {n_ok}/{len(selected)} frames matched "
                  f"(avg {avg_corr:.0f} corr/frame), sim2_ready={sim2_ready}")
        else:
            print(f"  [streaming-DOM] 0/{len(selected)} frames matched")

        return correspondences

    # ==================================================================
    #  POSE_TTT_TOKEN mode — per-frame token TTT with ENU dense reproj
    # ==================================================================

    def _adapt_window_pose_ttt_token(self, model, frames, dom_image_cpu,
                                     project_fn, enu_offset,
                                     dtype=torch.bfloat16,
                                     gt_enu_window=None,
                                     correspondences=None,
                                     roma_model=None,
                                     image_paths=None,
                                     inv_project_fn=None,
                                     geo_elev=None,
                                     dom_transform=None):
        """Per-frame token TTT with ENU dense reproj loss.

        Unlike pose_ttt (KV-cache TTT), this mode optimises per-frame DPT
        agg_tokens as independent leaves.  Each frame's gradient is isolated
        — no cross-frame contamination through shared KV.

        Gradient path (per frame fi):
          token_leaf[fi] (DPT layers)
            → PointHead → pts3d
            → Sim2 forward (model→ENU)
            → Huber loss vs ENU GT from DOM correspondences

        If correspondences is None and roma_model is provided, streaming DOM
        matching is performed using predicted model-space centres as crop
        positions (only first-frame GPS required).
        """
        from streamvggt.utils.trajectory_chain import pose_enc_to_camera_centers

        N = len(frames)
        cfg = self.config
        device = frames[0]["img"].device

        if N < cfg.warmup_frames:
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True, "reason": "too_few_frames"}

        # ====== Phase 1: Frozen forward — cache per-frame tokens ======
        print(f"  [pose_ttt_token] Phase 1: frozen forward ({N} frames)...")
        (per_frame_tokens, per_frame_all_tokens,
         per_frame_pose, psi) = self._frozen_forward_with_tokens(
            model, frames, dtype=dtype,
        )

        self._per_frame_pose = per_frame_pose
        pose_encs = torch.cat(per_frame_pose, dim=0)
        centers_model = pose_enc_to_camera_centers(pose_encs).cpu().numpy()

        # ====== Phase 1.5: Streaming DOM matching (if needed) ======
        if correspondences is None and roma_model is not None and image_paths is not None:
            print(f"  [pose_ttt_token] Streaming DOM matching ({N} frames)...")
            save_vis_dir = None  # could be wired from config if needed
            # dom_image_cpu may be (1,3,H,W) from run_inference; squeeze to (3,H,W)
            _dom_for_match = dom_image_cpu.squeeze(0) if dom_image_cpu.dim() == 4 else dom_image_cpu
            correspondences = self._streaming_dom_match(
                centers_model, image_paths, enu_offset,
                roma_model, _dom_for_match, project_fn, inv_project_fn,
                cfg, device=str(device),
                geo_elev=geo_elev, dom_transform=dom_transform,
                save_vis_dir=save_vis_dir,
            )

        # ====== Phase 1.75: Pre-compute frozen CameraHead KV (once) ======
        # Avoids O(N²) KV rebuild per TTT step inside Phase 3.
        camera_head = model.camera_head
        frozen_cam_kv_full = [None] * camera_head.trunk_depth
        with torch.no_grad():
            for _fi in range(N):
                with torch.amp.autocast('cuda', enabled=False):
                    _, frozen_cam_kv_full = camera_head(
                        per_frame_all_tokens[_fi],
                        past_key_values_camera=frozen_cam_kv_full,
                        use_cache=True,
                    )

        # ====== Phase 2: ENU targets + Sim2 ======
        cx_img = (cfg.camera_orig_w or 1600) / 2.0
        cy_img = (cfg.camera_orig_h or 1200) / 2.0

        optimal_enu = {}
        optimal_enu_3d = {}
        residuals = {}
        matched_frames = []
        n_pnp_used = 0

        if correspondences:
            for fi, corr in correspondences.items():
                pixels = corr['frame_pixels']
                enu_abs = corr['enu_positions']
                K = len(pixels)
                if K < 4:
                    continue
                pnp = corr.get('pnp_result')
                if pnp is not None and pnp.get('success'):
                    optimal_enu[fi] = pnp['t_world'][:2].copy()
                    optimal_enu_3d[fi] = pnp['t_world'].copy()
                    residuals[fi] = np.full(K, pnp.get('reproj_err', 1.0))
                    n_pnp_used += 1
                    continue
                ones = np.ones((K, 1))
                design = np.column_stack([pixels, ones])
                params_E, _, _, _ = np.linalg.lstsq(design, enu_abs[:, 0], rcond=None)
                params_N, _, _, _ = np.linalg.lstsq(design, enu_abs[:, 1], rcond=None)
                optimal_enu[fi] = np.array([
                    params_E[0] * cx_img + params_E[1] * cy_img + params_E[2],
                    params_N[0] * cx_img + params_N[1] * cy_img + params_N[2],
                ])
                pred_E = design @ params_E
                pred_N = design @ params_N
                res_mag = np.sqrt((pred_E - enu_abs[:, 0])**2 + (pred_N - enu_abs[:, 1])**2)
                residuals[fi] = res_mag
                med_res = np.median(res_mag)
                inlier_mask = res_mag < 3 * med_res + 1.0
                if inlier_mask.sum() >= 4:
                    design_in = design[inlier_mask]
                    p_E, _, _, _ = np.linalg.lstsq(design_in, enu_abs[inlier_mask, 0], rcond=None)
                    p_N, _, _, _ = np.linalg.lstsq(design_in, enu_abs[inlier_mask, 1], rcond=None)
                    optimal_enu[fi] = np.array([
                        p_E[0] * cx_img + p_E[1] * cy_img + p_E[2],
                        p_N[0] * cx_img + p_N[1] * cy_img + p_N[2],
                    ])
            matched_frames = sorted(optimal_enu.keys())

        # --- Step 2b: Build Sim2 correspondences ---
        if matched_frames:
            src_pts = np.array([centers_model[fi, :2] for fi in matched_frames])
            dst_pts = np.array([optimal_enu[fi] - enu_offset[:2] for fi in matched_frames])
        else:
            src_pts = np.zeros((0, 2))
            dst_pts = np.zeros((0, 2))

        if len(src_pts) == 0:
            print("  [pose_ttt_token] No DOM matches, skipping")
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True}

        n_total_pts = len(src_pts)
        if n_total_pts >= 4:
            # RANSAC Sim2 for robustness against DOM outliers
            s_m2e, R_m2e, t_m2e, inlier_mask = ransac_sim2(
                src_pts, dst_pts, reproj_thresh=10.0, max_iters=2000)
            n_inliers = int(inlier_mask.sum())
            print(f"  [pose_ttt_token] Sim2 from anchors (RANSAC): "
                  f"s={s_m2e:.4f}, {n_inliers}/{n_total_pts} inliers "
                  f"({len(matched_frames)} DOM frames)")
            # Only keep DOM inlier frames for loss targets
            dom_inlier_mask = inlier_mask[:len(matched_frames)]
            matched_frames_for_loss = [
                fi for fi, ok in zip(matched_frames, dom_inlier_mask) if ok]
            n_dom_outliers = len(matched_frames) - len(matched_frames_for_loss)
            if n_dom_outliers > 0:
                print(f"  [pose_ttt_token] Filtered {n_dom_outliers} DOM outlier frames "
                      f"from loss targets ({len(matched_frames_for_loss)} kept)")
        elif n_total_pts >= 2:
            # Too few for RANSAC, use Procrustes
            src_3d = np.column_stack([src_pts, np.zeros(n_total_pts)])
            dst_3d = np.column_stack([dst_pts, np.zeros(n_total_pts)])
            s_m2e, R_m2e, t_m2e = compute_model_to_enu_transform(src_3d, dst_3d)
            print(f"  [pose_ttt_token] Sim2 from anchors (Procrustes): "
                  f"s={s_m2e:.4f}, {n_total_pts} points "
                  f"({len(matched_frames)} DOM frames)")
            matched_frames_for_loss = list(matched_frames)
        else:
            # Single anchor: cannot estimate rotation/scale
            s_m2e, R_m2e, t_m2e = 1.0, np.eye(2), np.zeros(2)
            print(f"  [pose_ttt_token] WARNING: only {n_total_pts} "
                  f"anchor(s), using identity Sim2")
            matched_frames_for_loss = list(matched_frames)

        print(f"  [pose_ttt_token] Sim2 scale this window: s={s_m2e:.4f}")

        if abs(s_m2e) < 1e-8:
            print(f"  [pose_ttt_token] Degenerate Sim2 (scale={s_m2e:.2e}), skipping")
            self._per_frame_pose = per_frame_pose
            self._per_frame_all_tokens = per_frame_all_tokens
            self._psi = psi
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True}

        # ====== Phase 3: Per-frame token TTT with ENU dense reproj ======
        dpt_layers = list(model.point_head.intermediate_layer_idx)
        point_head = model.point_head

        # Setup per-frame token leaves (DPT layers only)
        all_leaves = []
        all_originals = []
        per_frame_leaf_tokens = []

        for fi in range(N):
            tok_leaves = {}
            for li in dpt_layers:
                t = per_frame_tokens[fi][li].to(device).requires_grad_(True)
                tok_leaves[li] = t
                all_leaves.append(t)
                all_originals.append(per_frame_tokens[fi][li].to(device).clone())
            per_frame_leaf_tokens.append(tok_leaves)

        n_p = sum(p.numel() for p in all_leaves)

        for p in model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(all_leaves, lr=cfg.kv_lr)

        # Select frames for loss (only RANSAC-inlier matched frames)
        fi_sel = list(matched_frames_for_loss)

        # Pre-compute dense reproj targets in ENU local space
        _dense_targets = {}
        for fi in fi_sel:
            corr = correspondences.get(fi)
            if corr is None:
                continue
            px = corr['frame_pixels']
            enu = corr['enu_positions']
            K = len(px)
            if K < 4:
                continue
            enu_local = enu.copy()
            enu_local[:, :2] = enu_local[:, :2] - enu_offset[:2]
            target_enu = np.zeros((K, 3))
            target_enu[:, :2] = enu_local[:, :2]
            if enu.shape[1] >= 3:
                target_enu[:, 2] = enu_local[:, 2]
            _dense_targets[fi] = torch.tensor(
                np.column_stack([px, target_enu]),
                dtype=torch.float32, device=device)

        fi_sel = [fi for fi in fi_sel if fi in _dense_targets]
        if not fi_sel:
            # No dense reproj targets — skip TTT, produce baseline ENU via Sim2.
            print(f"  [pose_ttt_token] No dense targets — "
                  f"producing baseline-only trajectory via Sim2")
            self._per_frame_pose = per_frame_pose
            self._per_frame_all_tokens = per_frame_all_tokens
            self._psi = psi
            self._geo_token_corrections = np.zeros((N, 2))

            # Build ENU trajectory from frozen-forward centers + Sim2
            bl_enu = np.zeros((N, 2))
            for fi in range(N):
                c_model = centers_model[fi, :2]
                bl_enu[fi] = s_m2e * R_m2e @ c_model + t_m2e
            bl_enu_abs = bl_enu + enu_offset[:2]

            if hasattr(self, '_streaming_last_enu'):
                self._streaming_last_enu = bl_enu_abs[-1].copy()

            return {
                "skipped": False,
                "losses": [],
                "n_matched": 0,
                "optimal_enu": {},
                "optimal_enu_3d": {},
                "fit_residuals": {},
                "baseline_centers": centers_model.copy(),
                "ttt_enu_trajectory": bl_enu_abs.copy(),
                "baseline_enu_trajectory": bl_enu_abs.copy(),
            }

        # Sim2 forward transform tensors (model → ENU)
        sR_fwd = torch.tensor(s_m2e * R_m2e, dtype=torch.float32, device=device)
        t_fwd = torch.tensor(t_m2e, dtype=torch.float32, device=device)
        s_fwd = float(s_m2e)

        # MSE center targets in ENU local space (NOT model space!)
        # Computing MSE in ENU amplifies gradient by sR_fwd factor (~250x)
        target_center_enu = {}
        for fi in matched_frames_for_loss:
            enu_local = optimal_enu[fi] - enu_offset[:2]
            target_center_enu[fi] = torch.tensor(
                enu_local, dtype=torch.float32, device=device)

        # Heading (rotation) targets from DOM affine fit
        target_heading_dir = {}
        if cfg.heading_weight > 0:
            R_fwd_2x2 = torch.tensor(R_m2e, dtype=torch.float32, device=device)
            for fi in matched_frames:
                corr = correspondences.get(fi)
                if corr is None:
                    continue
                hdeg = corr.get('heading_deg')
                if hdeg is None:
                    continue
                hrad = np.radians(hdeg)
                target_heading_dir[fi] = torch.tensor(
                    [np.cos(hrad), np.sin(hrad)],
                    dtype=torch.float32, device=device)
            if target_heading_dir:
                print(f"  [pose_ttt_token] Heading targets: {len(target_heading_dir)} frames")

        print(f"  [pose_ttt_token] Dense reproj: {len(fi_sel)} frames with targets")
        print(f"  [pose_ttt_token] {len(matched_frames)} matched, "
              f"{len(fi_sel)} loss frames, tokens={n_p/1e6:.1f}M params, "
              f"lr={cfg.kv_lr}, steps={cfg.num_steps}")

        losses = []
        phase3_centers_enu = {}  # diagnostic: record Phase 3 final-step poses
        for step in range(cfg.num_steps):
            is_last_step = (step == cfg.num_steps - 1)
            optimizer.zero_grad()
            step_dense_loss = 0.0
            step_mse_loss = 0.0
            step_heading_loss = 0.0
            n_dense_ok = 0
            n_mse_ok = 0
            n_heading_ok = 0

            for fi in fi_sel:
                # Build agg_tokens: use leaf at DPT layers, frozen elsewhere
                n_layers = len(per_frame_all_tokens[fi])
                agg_tokens = []
                for li in range(n_layers):
                    if li in per_frame_leaf_tokens[fi]:
                        agg_tokens.append(per_frame_leaf_tokens[fi][li])
                    else:
                        agg_tokens.append(per_frame_all_tokens[fi][li])

                images = frames[fi]["img"].unsqueeze(0)
                frame_loss = torch.tensor(0.0, device=device)

                # --- MSE center loss (camera token gradient) ---
                if fi in target_center_enu:
                    # Slice pre-computed frozen KV to frames < fi (O(1) vs O(N))
                    cam_kv_trim = [
                        (kv[0][:, :, :fi], kv[1][:, :, :fi]) if kv is not None else None
                        for kv in frozen_cam_kv_full
                    ]

                    with torch.amp.autocast('cuda', enabled=False):
                        agg_f = [t.float() for t in agg_tokens]
                        pose_enc_new, _ = camera_head(
                            agg_f,
                            past_key_values_camera=cam_kv_trim,
                            use_cache=True,
                        )
                    pose_enc_fi = pose_enc_new[-1][:, 0, :]
                    center_new = pose_enc_to_camera_centers(pose_enc_fi)
                    # Forward Sim2: model → ENU (differentiable)
                    center_enu = center_new[0, :2] @ sR_fwd.T + t_fwd
                    mse_loss = F.mse_loss(center_enu, target_center_enu[fi])
                    frame_loss = frame_loss + cfg.mse_weight * mse_loss
                    step_mse_loss += mse_loss.item()
                    n_mse_ok += 1

                    # Record Phase 3 final-step pose for diagnostic
                    if is_last_step:
                        phase3_centers_enu[fi] = center_enu.detach().cpu().numpy().copy()

                    # --- Heading (rotation) loss (reuse pose_enc_fi) ---
                    if cfg.heading_weight > 0 and fi in target_heading_dir:
                        from streamvggt.utils.rotation import quat_to_mat
                        quat = pose_enc_fi[:, 3:7]  # (1, 4) XYZW scalar-last
                        R_c2w = quat_to_mat(quat)    # (1, 3, 3)
                        # Camera right in world (model space), XY components
                        cam_right_model = R_c2w.transpose(1, 2)[:, :2, 0]  # (1, 2)
                        # Apply Sim2 rotation to get ENU direction
                        cam_right_enu = cam_right_model @ R_fwd_2x2.T  # (1, 2)
                        cam_right_enu_norm = F.normalize(cam_right_enu, dim=-1)
                        heading_loss = F.mse_loss(
                            cam_right_enu_norm.squeeze(0), target_heading_dir[fi])
                        frame_loss = frame_loss + cfg.heading_weight * heading_loss
                        step_heading_loss += heading_loss.item()
                        n_heading_ok += 1

                    del pose_enc_new, center_new

                # --- Dense reproj loss (patch token gradient) ---
                if fi in _dense_targets:
                    with torch.amp.autocast('cuda', enabled=False):
                        agg_f2 = [t.float() for t in agg_tokens]
                        pts3d, pts3d_conf = point_head(
                            agg_f2, images=images, patch_start_idx=psi,
                        )

                    pts3d_hw = pts3d[:, 0]
                    pts3d_chw = pts3d_hw.permute(0, 3, 1, 2).float()

                    corr_t = _dense_targets[fi]
                    img_w = cfg.camera_orig_w or 1600
                    img_h = cfg.camera_orig_h or 1200
                    grid_x = (corr_t[:, 0] / (img_w - 1) * 2 - 1).float()
                    grid_y = (corr_t[:, 1] / (img_h - 1) * 2 - 1).float()
                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)

                    sampled = F.grid_sample(
                        pts3d_chw, grid, align_corners=True,
                        mode='bilinear', padding_mode='border')
                    pts_sampled = sampled.squeeze(0).squeeze(1).T

                    target_enu_t = corr_t[:, 2:5]
                    enu_pred_xy = pts_sampled[:, :2] @ sR_fwd.T + t_fwd

                    valid = pts_sampled[:, 2] > 0.01
                    if valid.sum() >= 3:
                        p_xy = enu_pred_xy[valid]
                        t_xy = target_enu_t[valid, :2]
                        xy_loss = F.huber_loss(p_xy, t_xy,
                                               delta=cfg.dense_reproj_huber_xy)
                        dense_loss = xy_loss
                        if target_enu_t[:, 2].abs().max() > 0.01:
                            enu_pred_z = pts_sampled[valid, 2] * s_fwd
                            t_z = target_enu_t[valid, 2]
                            z_loss = F.huber_loss(enu_pred_z, t_z,
                                                  delta=cfg.dense_reproj_huber_z)
                            dense_loss = dense_loss + cfg.dense_reproj_z_weight * z_loss
                        frame_loss = frame_loss + cfg.dense_reproj_weight * dense_loss
                        step_dense_loss += dense_loss.item()
                        n_dense_ok += 1

                    del pts3d, pts3d_conf, pts3d_hw, pts3d_chw

                if frame_loss.grad_fn is not None:
                    (frame_loss / len(fi_sel)).backward()
                del frame_loss

            # L2 regularization
            if cfg.reg_weight > 0:
                reg = sum(
                    (p - o).pow(2).mean()
                    for p, o in zip(all_leaves, all_originals)
                )
                (cfg.reg_weight * reg).backward()

            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(all_leaves, cfg.max_grad_norm)

            g = sum(p.grad.norm().item() for p in all_leaves if p.grad is not None)
            optimizer.step()

            avg_dense = step_dense_loss / max(n_dense_ok, 1)
            avg_mse = step_mse_loss / max(n_mse_ok, 1)
            avg_heading = step_heading_loss / max(n_heading_ok, 1)
            losses.append(avg_dense + avg_mse + avg_heading)
            print(f"    step {step}: mse={avg_mse:.4f}, dense={avg_dense:.4f}, "
                  f"heading={avg_heading:.4f}, grad={g:.2e}")

        # ====== Phase 4: Re-run CameraHead with adapted tokens ======
        # Update per_frame_all_tokens with adapted leaves
        for fi in range(N):
            for li in dpt_layers:
                per_frame_all_tokens[fi][li] = per_frame_leaf_tokens[fi][li].detach()
        adapted_poses = []
        with torch.no_grad():
            past_kv_cam = [None] * camera_head.trunk_depth
            for fi in range(N):
                agg_tokens = per_frame_all_tokens[fi]
                with torch.cuda.amp.autocast(enabled=False):
                    pose_enc, past_kv_cam = camera_head(
                        agg_tokens, past_key_values_camera=past_kv_cam,
                        use_cache=True,
                    )
                    adapted_poses.append(pose_enc[-1][:, 0, :].float())

        self._per_frame_pose = adapted_poses
        self._per_frame_all_tokens = per_frame_all_tokens
        self._per_frame_leaf_tokens = per_frame_leaf_tokens
        self._psi = psi

        # Print adaptation effect
        adapted_encs = torch.cat(adapted_poses, dim=0)
        adapted_centers = pose_enc_to_camera_centers(adapted_encs).cpu().numpy()
        delta_model = np.linalg.norm(adapted_centers[:, :2] - centers_model[:, :2], axis=1)
        delta_enu = delta_model * s_m2e
        print(f"  [pose_ttt_token] Adaptation effect: center shift "
              f"mean={np.mean(delta_enu):.2f}m, max={np.max(delta_enu):.2f}m (ENU)")

        # ====== Build per-frame ENU trajectories ======
        # --- TTT trajectory (Phase 3 optimized) ---
        # Matched frames: use Phase 3 final center_enu
        # Unmatched: use adapted model-space center + Sim2 transform
        ttt_enu_trajectory = np.zeros((N, 2))
        for fi in range(N):
            c_model = adapted_centers[fi, :2]
            c_enu = s_m2e * R_m2e @ c_model + t_m2e
            ttt_enu_trajectory[fi] = c_enu
        for fi, p3_enu in phase3_centers_enu.items():
            ttt_enu_trajectory[fi] = p3_enu
        ttt_enu_trajectory_abs = ttt_enu_trajectory + enu_offset[:2]

        # --- Baseline trajectory (Phase 1 centers + same Sim2, no TTT) ---
        baseline_enu_trajectory = np.zeros((N, 2))
        for fi in range(N):
            c_model = centers_model[fi, :2]
            c_enu = s_m2e * R_m2e @ c_model + t_m2e
            baseline_enu_trajectory[fi] = c_enu
        baseline_enu_trajectory_abs = baseline_enu_trajectory + enu_offset[:2]

        n_matched = len(phase3_centers_enu)
        n_total = N
        print(f"  [pose_ttt_token] ENU trajectory: {n_matched}/{n_total} frames "
              f"from Phase 3, {n_total - n_matched} from Sim2 transform")

        # ====== DIAGNOSTIC: Phase 3 vs Phase 4 pose comparison ======
        if phase3_centers_enu:
            phase4_centers_enu = {}
            for fi in phase3_centers_enu:
                c_model = adapted_centers[fi, :2]
                c_enu = c_model @ (s_m2e * R_m2e).T + t_m2e
                phase4_centers_enu[fi] = c_enu
            print(f"  [DIAGNOSTIC] Phase 3 vs Phase 4 pose comparison ({len(phase3_centers_enu)} frames):")
            diffs = []
            for fi in sorted(phase3_centers_enu.keys()):
                p3 = phase3_centers_enu[fi]
                p4 = phase4_centers_enu[fi]
                tgt = target_center_enu[fi].cpu().numpy()
                d34 = np.linalg.norm(p3 - p4)
                d3t = np.linalg.norm(p3 - tgt)
                d4t = np.linalg.norm(p4 - tgt)
                diffs.append(d34)
                if fi < 5 or fi % 10 == 0:  # print a subset
                    print(f"    frame {fi}: P3-P4 diff={d34:.3f}m, P3-target={d3t:.3f}m, P4-target={d4t:.3f}m")
            print(f"    Phase3-Phase4 diff: mean={np.mean(diffs):.3f}m, max={np.max(diffs):.3f}m")

        for p in model.parameters():
            p.requires_grad_(False)
        torch.cuda.empty_cache()

        self._geo_token_corrections = np.zeros((N, 2))
        self._per_frame_params = None
        self._centroids = None

        # Update streaming state with TTT-optimized last-frame ENU
        # (better than raw DOM match centre for next window's enu_offset)
        if hasattr(self, '_streaming_last_enu'):
            self._streaming_last_enu = ttt_enu_trajectory_abs[-1].copy()

        return {
            "skipped": False,
            "losses": losses,
            "n_matched": len(matched_frames),
            "optimal_enu": {fi: optimal_enu[fi].copy() for fi in matched_frames},
            "optimal_enu_3d": {fi: optimal_enu_3d[fi].copy() for fi in matched_frames if fi in optimal_enu_3d},
            "fit_residuals": {fi: float(np.median(residuals[fi]))
                              for fi in matched_frames if fi in residuals},
            "baseline_centers": centers_model.copy(),
            "ttt_enu_trajectory": ttt_enu_trajectory_abs.copy(),
            "baseline_enu_trajectory": baseline_enu_trajectory_abs.copy(),
        }

    # ==================================================================
    #  POSE_TTT_RERUN23 — leaf at layer 22, re-run frame_block[23]
    # ==================================================================

    def _adapt_window_pose_ttt_rerun23(self, model, frames, dom_image_cpu,
                                       project_fn, enu_offset,
                                       dtype=torch.bfloat16,
                                       gt_enu_window=None,
                                       correspondences=None):
        """TTT with leaf = global_out[22], re-running frame_block[23].

        Key insight: frame_block[23] is a per-frame full self-attention that
        mixes camera token (idx 0) with patch tokens (idx 5+).  By placing
        the trainable leaf BEFORE this layer, both CameraHead (pose) and
        PointHead (pts3d) receive gradient from a single shared loss through
        the attention's Q/K/V mixing.

        Gradient path (per frame fi):
          leaf_fi (global_out_22, C-dim)
            → frame_block[23]  (camera ↔ patch self-attention)
            → concat with frozen global_out_23 → new agg_tokens[23]
            → CameraHead → pose → MSE center loss
            → PointHead  → pts3d → dense reproj loss
          Both losses backprop through the SAME attention layer.
        """
        from streamvggt.utils.trajectory_chain import pose_enc_to_camera_centers

        N = len(frames)
        cfg = self.config
        device = frames[0]["img"].device

        if N < cfg.warmup_frames:
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True, "reason": "too_few_frames"}

        # ====== Phase 1: Frozen forward — cache per-frame tokens ======
        print(f"  [rerun23] Phase 1: frozen forward ({N} frames)...")
        (per_frame_tokens, per_frame_all_tokens,
         per_frame_pose, psi) = self._frozen_forward_with_tokens(
            model, frames, dtype=dtype,
        )

        self._per_frame_pose = per_frame_pose
        pose_encs = torch.cat(per_frame_pose, dim=0)
        centers_model = pose_enc_to_camera_centers(pose_encs).cpu().numpy()

        # ====== Phase 2: DOM matching → ENU targets + Sim2 ======
        if not correspondences:
            print("  [rerun23] No correspondences, skipping")
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True}

        if gt_enu_window is not None and len(gt_enu_window) >= 3:
            gt_local = gt_enu_window - gt_enu_window[0]
            s_m2e, R_m2e, t_m2e = compute_model_to_enu_transform(
                centers_model, gt_local)
        else:
            s_m2e, R_m2e, t_m2e = 1.0, np.eye(2), np.zeros(2)

        cx_img = (cfg.camera_orig_w or 1600) / 2.0
        cy_img = (cfg.camera_orig_h or 1200) / 2.0

        optimal_enu = {}
        optimal_enu_3d = {}
        residuals = {}
        n_pnp_used = 0
        for fi, corr in correspondences.items():
            pixels = corr['frame_pixels']
            enu_abs = corr['enu_positions']
            K = len(pixels)
            if K < 4:
                continue
            pnp = corr.get('pnp_result')
            if pnp is not None and pnp.get('success'):
                optimal_enu[fi] = pnp['t_world'][:2].copy()
                optimal_enu_3d[fi] = pnp['t_world'].copy()
                residuals[fi] = np.full(K, pnp.get('reproj_err', 1.0))
                n_pnp_used += 1
                continue
            ones = np.ones((K, 1))
            design = np.column_stack([pixels, ones])
            params_E, _, _, _ = np.linalg.lstsq(design, enu_abs[:, 0], rcond=None)
            params_N, _, _, _ = np.linalg.lstsq(design, enu_abs[:, 1], rcond=None)
            optimal_enu[fi] = np.array([
                params_E[0] * cx_img + params_E[1] * cy_img + params_E[2],
                params_N[0] * cx_img + params_N[1] * cy_img + params_N[2],
            ])
            pred_E = design @ params_E
            pred_N = design @ params_N
            res_mag = np.sqrt((pred_E - enu_abs[:, 0])**2 + (pred_N - enu_abs[:, 1])**2)
            residuals[fi] = res_mag
            med_res = np.median(res_mag)
            inlier_mask = res_mag < 3 * med_res + 1.0
            if inlier_mask.sum() >= 4:
                design_in = design[inlier_mask]
                p_E, _, _, _ = np.linalg.lstsq(design_in, enu_abs[inlier_mask, 0], rcond=None)
                p_N, _, _, _ = np.linalg.lstsq(design_in, enu_abs[inlier_mask, 1], rcond=None)
                optimal_enu[fi] = np.array([
                    p_E[0] * cx_img + p_E[1] * cy_img + p_E[2],
                    p_N[0] * cx_img + p_N[1] * cy_img + p_N[2],
                ])

        if not optimal_enu:
            print("  [rerun23] No RoMa matches, skipping")
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True}

        matched_frames = sorted(optimal_enu.keys())

        if abs(s_m2e) < 1e-8:
            print(f"  [rerun23] Degenerate Sim2 (scale={s_m2e:.2e}), skipping")
            self._per_frame_pose = per_frame_pose
            self._per_frame_all_tokens = per_frame_all_tokens
            self._psi = psi
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True}

        # ====== Phase 3: Setup leaves + re-run infrastructure ======
        aggregator = model.aggregator
        frame_block_23 = aggregator.frame_blocks[23]
        camera_head = model.camera_head
        point_head = model.point_head
        dpt_layers = list(point_head.intermediate_layer_idx)  # [4, 11, 17, 23]

        # Determine C from the stored tokens (concat_inter has 2C)
        two_C = per_frame_all_tokens[0][22].shape[-1]
        C = two_C // 2

        # Compute RoPE position embeddings (same for all frames)
        images0 = frames[0]["img"].unsqueeze(0)  # [1,1,3,H,W]
        _, _, _, H_img, W_img = images0.shape
        patch_size = aggregator.patch_size
        rope_pos = None
        if aggregator.rope is not None:
            rope_pos = aggregator.position_getter(
                1, H_img // patch_size, W_img // patch_size,
                device=device,
            )
            if psi > 0:
                rope_pos = rope_pos + 1
                pos_special = torch.zeros(1, psi, 2, device=device, dtype=rope_pos.dtype)
                rope_pos = torch.cat([pos_special, rope_pos], dim=1)

        # Create per-frame leaves: global half of concat_inter[22]
        all_leaves = []
        all_originals = []
        per_frame_leaf22 = {}  # fi -> leaf tensor (1, P, C)

        for fi in range(N):
            # global half = last C dims of concat_inter[22]
            global_22 = per_frame_all_tokens[fi][22][:, 0, :, C:].detach().float()
            # shape: (1, P, C)
            leaf = global_22.clone().to(device).requires_grad_(True)
            per_frame_leaf22[fi] = leaf
            all_leaves.append(leaf)
            all_originals.append(global_22.to(device).clone())

        n_p = sum(p.numel() for p in all_leaves)
        P = per_frame_all_tokens[0][22].shape[2]

        # Frozen global_out[23] for each frame (to concat with recomputed frame half)
        per_frame_global23_frozen = {}
        for fi in range(N):
            per_frame_global23_frozen[fi] = (
                per_frame_all_tokens[fi][23][:, 0, :, C:].detach().float().to(device)
            )  # (1, P, C)

        for p in model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam(all_leaves, lr=cfg.kv_lr)

        # Pre-compute dense reproj targets in ENU local space
        fi_sel = [fi for fi in matched_frames]
        _dense_targets = {}
        for fi in fi_sel:
            corr = correspondences.get(fi)
            if corr is None:
                continue
            px = corr['frame_pixels']
            enu = corr['enu_positions']
            K_corr = len(px)
            if K_corr < 4:
                continue
            enu_local = enu.copy()
            enu_local[:, :2] = enu_local[:, :2] - enu_offset[:2]
            target_enu = np.zeros((K_corr, 3))
            target_enu[:, :2] = enu_local[:, :2]
            if enu.shape[1] >= 3:
                target_enu[:, 2] = enu_local[:, 2]
            _dense_targets[fi] = torch.tensor(
                np.column_stack([px, target_enu]),
                dtype=torch.float32, device=device)

        fi_sel = [fi for fi in fi_sel if fi in _dense_targets]
        if not fi_sel:
            print("  [rerun23] No dense targets, skipping TTT")
            self._per_frame_pose = per_frame_pose
            self._per_frame_all_tokens = per_frame_all_tokens
            self._psi = psi
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True}

        # Sim2 transforms
        sR_fwd = torch.tensor(s_m2e * R_m2e, dtype=torch.float32, device=device)
        t_fwd = torch.tensor(t_m2e, dtype=torch.float32, device=device)
        s_fwd = float(s_m2e)

        # MSE center targets in ENU local space (NOT model space!)
        target_center_enu = {}
        for fi in matched_frames:
            enu_local = optimal_enu[fi] - enu_offset[:2]
            target_center_enu[fi] = torch.tensor(
                enu_local, dtype=torch.float32, device=device)

        # Heading (rotation) targets from DOM affine fit
        target_heading_dir = {}
        if cfg.heading_weight > 0:
            R_fwd_2x2 = torch.tensor(R_m2e, dtype=torch.float32, device=device)
            for fi in matched_frames:
                corr = correspondences.get(fi)
                if corr is None:
                    continue
                hdeg = corr.get('heading_deg')
                if hdeg is None:
                    continue
                hrad = np.radians(hdeg)
                target_heading_dir[fi] = torch.tensor(
                    [np.cos(hrad), np.sin(hrad)],
                    dtype=torch.float32, device=device)
            if target_heading_dir:
                print(f"  [rerun23] Heading targets: {len(target_heading_dir)} frames")

        print(f"  [rerun23] {len(fi_sel)} loss frames, {len(matched_frames)} matched, "
              f"leaf_params={n_p/1e6:.1f}M (layer22 global), P={P}, C={C}, "
              f"lr={cfg.kv_lr}, steps={cfg.num_steps}")

        # ====== Phase 3: TTT loop ======
        losses = []
        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            step_dense_loss = 0.0
            step_mse_loss = 0.0
            step_heading_loss = 0.0
            n_dense_ok = 0
            n_mse_ok = 0
            n_heading_ok = 0

            for fi in fi_sel:
                # -- Re-run frame_block[23] with leaf --
                leaf_fi = per_frame_leaf22[fi]  # (1, P, C)
                with torch.amp.autocast('cuda', enabled=False):
                    frame_out_23 = frame_block_23(
                        leaf_fi.float(), pos=rope_pos,
                    )  # (1, P, C) — camera↔patch mixed!

                # Build new concat_inter[23] = cat(frame_new, global_frozen)
                frame_half = frame_out_23.unsqueeze(0)  # (1, 1, P, C)
                global_half = per_frame_global23_frozen[fi].unsqueeze(0)  # (1, 1, P, C)
                new_concat_23 = torch.cat([frame_half, global_half], dim=-1)  # (1, 1, P, 2C)

                # Build agg_tokens list for heads
                agg_tokens = []
                for li in range(24):
                    if li == 23:
                        agg_tokens.append(new_concat_23)
                    else:
                        agg_tokens.append(per_frame_all_tokens[fi][li])

                images = frames[fi]["img"].unsqueeze(0)
                frame_loss = torch.tensor(0.0, device=device)

                # --- MSE center loss (via CameraHead) ---
                if fi in target_center_enu:
                    # Build camera KV from frozen frames < fi
                    cam_kv_trim = [None] * camera_head.trunk_depth
                    with torch.no_grad():
                        for prev_fi in range(fi):
                            prev_agg = per_frame_all_tokens[prev_fi]
                            with torch.amp.autocast('cuda', enabled=False):
                                _, cam_kv_trim = camera_head(
                                    prev_agg,
                                    past_key_values_camera=cam_kv_trim,
                                    use_cache=True,
                                )

                    with torch.amp.autocast('cuda', enabled=False):
                        agg_f = [t.float() for t in agg_tokens]
                        pose_enc_new, _ = camera_head(
                            agg_f,
                            past_key_values_camera=cam_kv_trim,
                            use_cache=True,
                        )
                    pose_enc_fi = pose_enc_new[-1][:, 0, :]
                    center_new = pose_enc_to_camera_centers(pose_enc_fi)
                    # Forward Sim2: model → ENU (differentiable)
                    center_enu = center_new[0, :2] @ sR_fwd.T + t_fwd
                    mse_loss = F.mse_loss(center_enu, target_center_enu[fi])
                    frame_loss = frame_loss + cfg.mse_weight * mse_loss
                    step_mse_loss += mse_loss.item()
                    n_mse_ok += 1

                    # --- Heading (rotation) loss (reuse pose_enc_fi) ---
                    if cfg.heading_weight > 0 and fi in target_heading_dir:
                        from streamvggt.utils.rotation import quat_to_mat
                        quat = pose_enc_fi[:, 3:7]  # (1, 4) XYZW scalar-last
                        R_c2w = quat_to_mat(quat)    # (1, 3, 3)
                        cam_right_model = R_c2w.transpose(1, 2)[:, :2, 0]  # (1, 2)
                        cam_right_enu = cam_right_model @ R_fwd_2x2.T  # (1, 2)
                        cam_right_enu_norm = F.normalize(cam_right_enu, dim=-1)
                        heading_loss = F.mse_loss(
                            cam_right_enu_norm.squeeze(0), target_heading_dir[fi])
                        frame_loss = frame_loss + cfg.heading_weight * heading_loss
                        step_heading_loss += heading_loss.item()
                        n_heading_ok += 1

                    del pose_enc_new, center_new

                # --- Dense reproj loss (via PointHead) ---
                if fi in _dense_targets:
                    with torch.amp.autocast('cuda', enabled=False):
                        agg_f2 = [t.float() for t in agg_tokens]
                        pts3d, pts3d_conf = point_head(
                            agg_f2, images=images, patch_start_idx=psi,
                        )
                    pts3d_hw = pts3d[:, 0]
                    pts3d_chw = pts3d_hw.permute(0, 3, 1, 2).float()

                    corr_t = _dense_targets[fi]
                    img_w = cfg.camera_orig_w or 1600
                    img_h = cfg.camera_orig_h or 1200
                    grid_x = (corr_t[:, 0] / (img_w - 1) * 2 - 1).float()
                    grid_y = (corr_t[:, 1] / (img_h - 1) * 2 - 1).float()
                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)

                    sampled = F.grid_sample(
                        pts3d_chw, grid, align_corners=True,
                        mode='bilinear', padding_mode='border')
                    pts_sampled = sampled.squeeze(0).squeeze(1).T

                    target_enu_t = corr_t[:, 2:5]
                    enu_pred_xy = pts_sampled[:, :2] @ sR_fwd.T + t_fwd

                    valid = pts_sampled[:, 2] > 0.01
                    if valid.sum() >= 3:
                        p_xy = enu_pred_xy[valid]
                        t_xy = target_enu_t[valid, :2]
                        xy_loss = F.huber_loss(p_xy, t_xy,
                                               delta=cfg.dense_reproj_huber_xy)
                        dense_loss = xy_loss
                        if target_enu_t[:, 2].abs().max() > 0.01:
                            enu_pred_z = pts_sampled[valid, 2] * s_fwd
                            t_z = target_enu_t[valid, 2]
                            z_loss = F.huber_loss(enu_pred_z, t_z,
                                                  delta=cfg.dense_reproj_huber_z)
                            dense_loss = dense_loss + cfg.dense_reproj_z_weight * z_loss
                        frame_loss = frame_loss + cfg.dense_reproj_weight * dense_loss
                        step_dense_loss += dense_loss.item()
                        n_dense_ok += 1

                    del pts3d, pts3d_conf, pts3d_hw, pts3d_chw

                (frame_loss / len(fi_sel)).backward()
                del frame_loss

            # L2 regularization
            if cfg.reg_weight > 0:
                reg = sum(
                    (p - o).pow(2).mean()
                    for p, o in zip(all_leaves, all_originals)
                )
                (cfg.reg_weight * reg).backward()

            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(all_leaves, cfg.max_grad_norm)

            g = sum(p.grad.norm().item() for p in all_leaves if p.grad is not None)
            optimizer.step()

            avg_dense = step_dense_loss / max(n_dense_ok, 1)
            avg_mse = step_mse_loss / max(n_mse_ok, 1)
            avg_heading = step_heading_loss / max(n_heading_ok, 1)
            losses.append(avg_dense + avg_mse + avg_heading)
            print(f"    step {step}: mse={avg_mse:.6f}, dense={avg_dense:.4f}, "
                  f"heading={avg_heading:.4f}, grad={g:.2e}")

        # ====== Phase 4: Re-run all frames through frame_block[23] + CameraHead ======
        adapted_poses = []
        with torch.no_grad():
            # First: recompute concat_inter[23] for all frames using adapted leaves
            for fi in range(N):
                leaf_fi = per_frame_leaf22[fi].detach()
                with torch.amp.autocast('cuda', enabled=False):
                    frame_out_23 = frame_block_23(leaf_fi.float(), pos=rope_pos)
                frame_half = frame_out_23.unsqueeze(0)  # (1, 1, P, C)
                global_half = per_frame_global23_frozen[fi].unsqueeze(0)
                per_frame_all_tokens[fi][23] = torch.cat(
                    [frame_half, global_half], dim=-1)

            # Then: stream through CameraHead
            past_kv_cam = [None] * camera_head.trunk_depth
            for fi in range(N):
                agg_tokens = per_frame_all_tokens[fi]
                with torch.amp.autocast('cuda', enabled=False):
                    pose_enc, past_kv_cam = camera_head(
                        agg_tokens, past_key_values_camera=past_kv_cam,
                        use_cache=True,
                    )
                    adapted_poses.append(pose_enc[-1][:, 0, :].float())

        self._per_frame_pose = adapted_poses
        self._per_frame_all_tokens = per_frame_all_tokens
        self._psi = psi

        # Print adaptation effect
        adapted_encs = torch.cat(adapted_poses, dim=0)
        adapted_centers = pose_enc_to_camera_centers(adapted_encs).cpu().numpy()
        delta_model = np.linalg.norm(adapted_centers[:, :2] - centers_model[:, :2], axis=1)
        delta_enu = delta_model * s_m2e
        print(f"  [rerun23] Adaptation effect: center shift "
              f"mean={np.mean(delta_enu):.2f}m, max={np.max(delta_enu):.2f}m (ENU)")

        for p in model.parameters():
            p.requires_grad_(False)
        torch.cuda.empty_cache()

        self._geo_token_corrections = np.zeros((N, 2))
        self._per_frame_params = None
        self._centroids = None

        return {
            "skipped": False,
            "losses": losses,
            "n_matched": len(matched_frames),
            "optimal_enu": {fi: optimal_enu[fi].copy() for fi in matched_frames},
            "optimal_enu_3d": {fi: optimal_enu_3d[fi].copy() for fi in matched_frames if fi in optimal_enu_3d},
            "fit_residuals": {fi: float(np.median(residuals[fi]))
                              for fi in matched_frames if fi in residuals},
            "baseline_centers": centers_model.copy(),
        }

    # ==================================================================
    #  POSE_REFINE mode — direct camera center correction via affine fit
    # ==================================================================

    def _adapt_window_pose_refine(self, model, frames, dom_image_cpu,
                                  project_fn, enu_offset,
                                  dtype=torch.bfloat16,
                                  gt_enu_window=None,
                                  correspondences=None):
        """Directly correct per-frame camera centers using RoMa correspondences.

        For each frame with RoMa matches (pixel -> ENU), fit a robust affine
        transform from image pixels to ENU coordinates. The camera center is
        then the ENU position corresponding to the principal point (cx, cy).

        This bypasses any neural network optimization — it's a pure geometric
        closed-form correction that directly addresses XY trajectory drift.

        Corrections for unmatched frames are interpolated from matched neighbors.
        """
        from streamvggt.utils.trajectory_chain import pose_enc_to_camera_centers

        N = len(frames)
        cfg = self.config
        device = frames[0]["img"].device

        # Phase 1: frozen forward to get initial poses
        per_frame_pose = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                output = model.inference(frames)
            for res in output.ress:
                per_frame_pose.append(res["camera_pose"].cpu().float())
        self._per_frame_pose = per_frame_pose
        del output
        torch.cuda.empty_cache()

        pose_encs = torch.cat(per_frame_pose, dim=0)
        centers_model = pose_enc_to_camera_centers(pose_encs).numpy()  # (N, 3)

        # If no correspondences, skip
        if not correspondences:
            print("  [pose_refine] No correspondences, skipping")
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True}

        # Phase 2: Sim2 alignment model -> ENU
        if gt_enu_window is not None and len(gt_enu_window) >= 3:
            gt_local = gt_enu_window - gt_enu_window[0]
            s_m2e, R_m2e, t_m2e = compute_model_to_enu_transform(
                centers_model, gt_local)
        else:
            s_m2e, R_m2e, t_m2e = 1.0, np.eye(2), np.zeros(2)

        # Convert model centers to ENU for comparison
        centers_enu_xy = s_m2e * (centers_model[:, :2] @ R_m2e.T) + t_m2e
        # Add window offset to get absolute ENU
        centers_enu_abs = centers_enu_xy + enu_offset[:2]

        # Phase 3: for each matched frame, compute optimal camera center
        # using affine fit from pixel correspondences -> ENU
        cx_img = (cfg.camera_orig_w or 1600) / 2.0  # principal point col
        cy_img = (cfg.camera_orig_h or 1200) / 2.0  # principal point row

        optimal_enu = {}  # fi -> (E, N) absolute ENU
        residuals = {}    # fi -> per-correspondence residual magnitudes
        for fi, corr in correspondences.items():
            pixels = corr['frame_pixels']      # (K, 2) col, row in original image
            enu_abs = corr['enu_positions']     # (K, 2) absolute ENU
            K = len(pixels)

            if K < 4:
                continue

            # Method 1: Use pre-computed frame_center_enu from homography
            # (already computed in get_frame_dom_correspondences via
            #  perspectiveTransform of principal point through homography)
            if 'frame_center_enu' in corr:
                hom_center = corr['frame_center_enu']  # (2,)

            # Method 2: Robust affine fit pixel -> ENU, evaluate at (cx, cy)
            # [E, N] = A @ [col, row, 1]
            ones = np.ones((K, 1))
            design = np.column_stack([pixels, ones])  # (K, 3)
            params_E, _, _, _ = np.linalg.lstsq(design, enu_abs[:, 0], rcond=None)
            params_N, _, _, _ = np.linalg.lstsq(design, enu_abs[:, 1], rcond=None)

            affine_center_E = params_E[0] * cx_img + params_E[1] * cy_img + params_E[2]
            affine_center_N = params_N[0] * cx_img + params_N[1] * cy_img + params_N[2]

            # Compute fit residuals to assess quality
            pred_E = design @ params_E
            pred_N = design @ params_N
            res_mag = np.sqrt((pred_E - enu_abs[:, 0])**2 + (pred_N - enu_abs[:, 1])**2)
            residuals[fi] = res_mag

            # Use affine fit (more robust with many points)
            optimal_enu[fi] = np.array([affine_center_E, affine_center_N])

            # Also do RANSAC-robust version: use inlier subset
            # Fit residual-based outlier rejection (3-sigma)
            med_res = np.median(res_mag)
            inlier_mask = res_mag < 3 * med_res + 1.0  # at least 1m tolerance
            if inlier_mask.sum() >= 4:
                design_in = design[inlier_mask]
                p_E, _, _, _ = np.linalg.lstsq(design_in, enu_abs[inlier_mask, 0], rcond=None)
                p_N, _, _, _ = np.linalg.lstsq(design_in, enu_abs[inlier_mask, 1], rcond=None)
                optimal_enu[fi] = np.array([
                    p_E[0] * cx_img + p_E[1] * cy_img + p_E[2],
                    p_N[0] * cx_img + p_N[1] * cy_img + p_N[2],
                ])

        if not optimal_enu:
            print("  [pose_refine] No frames matched, skipping")
            self._geo_token_corrections = np.zeros((N, 2))
            return {"skipped": True}

        # Phase 4: Store optimal ENU positions for post-chain dense anchor
        # correction. Do NOT convert to model-space corrections here — that
        # causes Sim2 inconsistency when the chaining Sim2 differs from the
        # GT-based Sim2 used inside this window.
        matched_frames = sorted(optimal_enu.keys())

        # Print diagnostics (corrections relative to Sim2-projected centers)
        for fi in matched_frames:
            d = optimal_enu[fi] - centers_enu_abs[fi]
            res = residuals.get(fi)
            res_str = f"fit_res={np.median(res):.2f}m" if res is not None else ""
            print(f"    f{fi:3d}: RoMa→ENU=({optimal_enu[fi][0]:+.1f}, "
                  f"{optimal_enu[fi][1]:+.1f}), delta=({d[0]:+.2f}, "
                  f"{d[1]:+.2f})m, |d|={np.linalg.norm(d):.2f}m {res_str}")

        # No model-space corrections — dense anchors handle everything post-chain
        self._geo_token_corrections = np.zeros((N, 2))
        self._per_frame_params = None
        self._centroids = None

        return {
            "skipped": False,
            "losses": [float(np.mean([np.linalg.norm(optimal_enu[fi] - centers_enu_abs[fi])
                                      for fi in matched_frames]))],
            "n_matched": len(matched_frames),
            "optimal_enu": {fi: optimal_enu[fi].copy() for fi in matched_frames},
            "fit_residuals": {fi: float(np.median(residuals[fi]))
                              for fi in matched_frames if fi in residuals},
        }

    # ==================================================================
    #  GEO_CONSIST_TOKEN mode — geometric loss through DPT tokens
    # ==================================================================

    def _adapt_window_geo_consist_token(self, model, frames, dom_image_cpu,
                                        project_fn, enu_offset,
                                        dtype=torch.bfloat16,
                                        gt_enu_window=None,
                                        correspondences=None):
        """Adapt DPT-layer tokens using geometric consistency loss.

        Combines:
        - Token-mode mechanics: DPT tokens as learnable, point_head re-run
        - geo_consist supervision: RoMa-matched pixel↔ENU correspondences

        Loss = |Sim2(pts3d_from_adapted_tokens[pixel_xy]) - enu_target|
        Gradient flows: loss → Sim2(linear) → grid_sample → pts3d → point_head → DPT tokens
        """
        cfg = self.config
        device = next(model.parameters()).device
        self._geo_token_corrections = None  # reset from previous window
        N = len(frames)
        if N < cfg.warmup_frames:
            return {"skipped": True, "reason": "too_few_frames"}

        if not correspondences:
            print("  [GeoTTTv2-geo_consist_token] No correspondences, skipping")
            return {"skipped": True, "reason": "no_correspondences"}

        # === Phase 1: frozen forward → cache tokens + initial pts3d ===
        print(f"  [GeoTTTv2-geo_consist_token] Phase 1: frozen forward ({N} frames)...")
        (per_frame_tokens, per_frame_all_tokens,
         per_frame_pose, psi) = self._frozen_forward_with_tokens(
            model, frames, dtype=dtype,
        )
        dpt_layers = list(model.point_head.intermediate_layer_idx)

        # Compute initial pts3d from frozen tokens (for Sim2 alignment)
        pts_list = []
        with torch.no_grad():
            for fi in range(N):
                agg_f = [per_frame_all_tokens[fi][li].float()
                         for li in range(len(per_frame_all_tokens[fi]))]
                images = frames[fi]["img"].unsqueeze(0)
                pts3d, _ = model.point_head(agg_f, images=images,
                                            patch_start_idx=psi)
                pts_list.append(pts3d[:, 0].cpu().float())  # [1, H, W, 3]
        torch.cuda.empty_cache()

        # === Phase 2: Sim2 alignment from correspondences ===
        all_model_xy = []
        all_enu_local = []
        corr_grids = {}
        for fi_local, corr in correspondences.items():
            if fi_local < 0 or fi_local >= N:
                continue
            query_W = corr['query_W']
            query_H = corr['query_H']
            pixels = corr['frame_pixels']
            enu_abs = corr['enu_positions']
            K_corr = len(pixels)

            u_norm = pixels[:, 0] / max(query_W - 1, 1)
            v_norm = pixels[:, 1] / max(query_H - 1, 1)
            grid = torch.tensor(
                np.stack([2.0 * u_norm - 1.0, 2.0 * v_norm - 1.0], axis=-1),
                dtype=torch.float32,
            ).reshape(1, 1, K_corr, 2)

            pts3d_frozen = pts_list[fi_local]
            pts_xy_img = pts3d_frozen[..., :2].permute(0, 3, 1, 2)
            sampled = F.grid_sample(
                pts_xy_img, grid.to(pts3d_frozen.device), mode='bilinear',
                align_corners=True, padding_mode='border',
            )
            model_xy = sampled[0, :, 0, :].T.cpu().numpy()

            enu_local = enu_abs - enu_offset[:2]
            all_model_xy.append(model_xy)
            all_enu_local.append(enu_local)
            corr_grids[fi_local] = (grid, enu_abs, query_W, query_H)

        if not all_model_xy:
            print("  [GeoTTTv2-geo_consist_token] No valid targets, skipping")
            return {"skipped": True, "reason": "no_valid_targets"}

        all_model_xy = np.concatenate(all_model_xy, axis=0)
        all_enu_local = np.concatenate(all_enu_local, axis=0)
        n_total = len(all_model_xy)

        s_m2e, R_m2e, t_m2e, keep = ransac_sim2(
            all_model_xy, all_enu_local,
            reproj_thresh=8.0, max_iters=3000,
        )
        n_inliers = int(keep.sum())
        aligned_enu = s_m2e * (all_model_xy @ R_m2e.T) + t_m2e
        all_err = np.linalg.norm(aligned_enu - all_enu_local, axis=1)
        inlier_err = all_err[keep]
        print(f"    RANSAC Sim2: scale={s_m2e:.2f}, "
              f"{n_total} pairs → {n_inliers} inliers "
              f"({100*n_inliers/n_total:.0f}%), "
              f"inlier residual={np.mean(inlier_err):.2f}m "
              f"(med={np.median(inlier_err):.2f}m)")

        if n_inliers < 10:
            print("  [GeoTTTv2-geo_consist_token] Too few RANSAC inliers, skipping")
            return {"skipped": True, "reason": "too_few_inliers"}

        # Refit on inliers
        inlier_model = all_model_xy[keep]
        inlier_enu = all_enu_local[keep]
        model_3d = np.column_stack([inlier_model, np.zeros(n_inliers)])
        enu_3d = np.column_stack([inlier_enu, np.zeros(n_inliers)])
        s_m2e, R_m2e, t_m2e = compute_model_to_enu_transform(model_3d, enu_3d)

        aligned2 = s_m2e * (inlier_model @ R_m2e.T) + t_m2e
        err2 = np.linalg.norm(aligned2 - inlier_enu, axis=1)
        med_refined = np.median(err2)
        print(f"    Refined: scale={s_m2e:.2f}, "
              f"residual med={med_refined:.2f}m, max={np.max(err2):.2f}m")

        max_residual_m = 15.0
        if med_refined > max_residual_m:
            print(f"  [GeoTTTv2-geo_consist_token] Alignment residual too high "
                  f"({med_refined:.1f}m > {max_residual_m}m), skipping")
            self._per_frame_pose = per_frame_pose
            self._per_frame_params = None
            self._centroids = None
            return {"skipped": False, "losses": [0.0],
                    "loss_improvement": 0.0, "quality_skip": True}

        # Filter per-frame correspondences to RANSAC inliers
        offset = 0
        corr_grids_filtered = {}
        for fi_local in sorted(corr_grids.keys()):
            grid, enu_abs, query_W, query_H = corr_grids[fi_local]
            n_this = len(enu_abs)
            frame_keep = keep[offset:offset + n_this]
            offset += n_this
            if frame_keep.sum() < 3:
                continue
            g = grid.reshape(-1, 2)[frame_keep]
            corr_grids_filtered[fi_local] = (
                g.reshape(1, 1, -1, 2),
                enu_abs[frame_keep], query_W, query_H)
        corr_grids = corr_grids_filtered

        # Forward Sim2 transform: model_xy → ENU (for loss in metric space)
        sR_fwd = torch.tensor(
            s_m2e * R_m2e,
            dtype=torch.float32, device=device,
        )
        t_fwd = torch.tensor(t_m2e, dtype=torch.float32, device=device)
        # Inverse for post-TTT corrections
        sR_inv_np = np.linalg.inv(s_m2e * R_m2e)

        # Build per-frame geometric targets in ENU coordinates (metres)
        geo_targets = {}
        total_corr = 0
        for fi_local, (grid, enu_abs, query_W, query_H) in corr_grids.items():
            enu_local = enu_abs - enu_offset[:2]
            enu_target = torch.tensor(enu_local, dtype=torch.float32,
                                      device=device)
            geo_targets[fi_local] = (grid.to(device), enu_target)
            total_corr += len(enu_abs)

        n_matched = len(geo_targets)

        del pts_list
        torch.cuda.empty_cache()

        # === Phase 3: make DPT-layer tokens learnable ===
        all_leaves = []
        all_originals = []
        per_frame_leaf_tokens = []
        for fi in range(N):
            tok_leaves = {}
            for li in dpt_layers:
                t = per_frame_tokens[fi][li].to(device).requires_grad_(True)
                tok_leaves[li] = t
                all_leaves.append(t)
                all_originals.append(
                    per_frame_tokens[fi][li].to(device).clone())
            per_frame_leaf_tokens.append(tok_leaves)

        n_p = sum(p.numel() for p in all_leaves)
        print(f"    Token leaves: {len(all_leaves)} tensors, "
              f"{n_p/1e6:.1f}M params "
              f"({len(dpt_layers)} DPT layers × {N} frames)")
        print(f"    Targets: {n_matched} frames, "
              f"{total_corr} correspondences")

        optimizer = torch.optim.Adam(all_leaves, lr=cfg.kv_lr)

        for p in model.parameters():
            p.requires_grad_(False)

        # === Phase 4: TTT optimisation ===
        losses = []
        print(f"  [GeoTTTv2-geo_consist_token] TTT "
              f"({cfg.num_steps} steps, lr={cfg.kv_lr})")

        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            s_loss, n_ok = 0.0, 0

            for fi in range(N):
                if fi not in geo_targets:
                    continue

                grid, enu_target = geo_targets[fi]

                # Build agg_tokens: leaf at DPT layers, frozen elsewhere
                n_layers = len(per_frame_all_tokens[fi])
                dpt_set = set(dpt_layers)
                agg_tokens = []
                for li in range(n_layers):
                    if li in dpt_set:
                        agg_tokens.append(per_frame_leaf_tokens[fi][li])
                    else:
                        agg_tokens.append(per_frame_all_tokens[fi][li])

                images = frames[fi]["img"].unsqueeze(0)
                with torch.cuda.amp.autocast(enabled=False):
                    agg_f = [t.float() for t in agg_tokens]
                    pts3d, _ = model.point_head(
                        agg_f, images=images, patch_start_idx=psi,
                    )

                pts_frame = pts3d[:, 0]  # [1, H, W, 3]
                pts_xy_img = pts_frame[..., :2].permute(0, 3, 1, 2)

                sampled = F.grid_sample(
                    pts_xy_img, grid, mode='bilinear',
                    align_corners=True, padding_mode='border',
                )
                sampled_xy = sampled[0, :, 0, :].T  # [K, 2]

                # Loss in ENU (metric) space — ~266× larger gradients
                pred_enu = sampled_xy @ sR_fwd.T + t_fwd

                if cfg.loss_type == "l1":
                    loss = (pred_enu - enu_target).abs().mean()
                elif cfg.loss_type == "mse":
                    loss = (pred_enu - enu_target).pow(2).mean()
                else:
                    loss = F.smooth_l1_loss(pred_enu, enu_target)

                (loss / n_matched).backward()
                s_loss += loss.item()
                n_ok += 1
                del loss, pts3d, pts_frame, pts_xy_img, sampled, sampled_xy
                del pred_enu, agg_tokens, agg_f

            if cfg.reg_weight > 0:
                reg = sum(
                    (p - o).pow(2).mean()
                    for p, o in zip(all_leaves, all_originals)
                )
                (cfg.reg_weight * reg).backward()

            g = sum(p.grad.norm().item()
                    for p in all_leaves if p.grad is not None)
            optimizer.step()

            avg = s_loss / max(n_ok, 1)
            losses.append(avg)
            if step % max(1, cfg.num_steps // 5) == 0 \
                    or step == cfg.num_steps - 1:
                print(f"    step {step:3d}: loss={avg:.4f}m, "
                      f"|grad|={g:.6f}")

        for p in model.parameters():
            p.requires_grad_(False)

        # === Phase 5: compute per-frame corrections from adapted pts3d ===
        corrections = np.zeros((N, 2))  # model-space XY corrections
        matched_frames = sorted(geo_targets.keys())
        with torch.no_grad():
            for fi in matched_frames:
                grid, enu_target = geo_targets[fi]
                # Build adapted agg_tokens
                n_layers = len(per_frame_all_tokens[fi])
                dpt_set = set(dpt_layers)
                agg_tokens = []
                for li in range(n_layers):
                    if li in dpt_set:
                        agg_tokens.append(per_frame_leaf_tokens[fi][li])
                    else:
                        agg_tokens.append(per_frame_all_tokens[fi][li])

                images = frames[fi]["img"].unsqueeze(0)
                agg_f = [t.float() for t in agg_tokens]
                pts3d, _ = model.point_head(
                    agg_f, images=images, patch_start_idx=psi)

                pts_frame = pts3d[:, 0]
                pts_xy_img = pts_frame[..., :2].permute(0, 3, 1, 2)
                sampled = F.grid_sample(
                    pts_xy_img, grid, mode='bilinear',
                    align_corners=True, padding_mode='border')
                adapted_model_xy = sampled[0, :, 0, :].T.cpu().numpy()

                # Residual in ENU
                adapted_enu = adapted_model_xy @ (s_m2e * R_m2e).T + t_m2e
                enu_target_np = enu_target.cpu().numpy()
                residual_enu = enu_target_np - adapted_enu  # [K, 2]

                # Median correction → model space
                med_corr_enu = np.median(residual_enu, axis=0)
                corrections[fi] = sR_inv_np @ med_corr_enu

        # Interpolate corrections for unmatched frames
        if len(matched_frames) >= 2:
            for dim in range(2):
                vals = np.array([corrections[fi, dim] for fi in matched_frames])
                corrections[:, dim] = np.interp(
                    np.arange(N),
                    matched_frames,
                    vals,
                )
        elif len(matched_frames) == 1:
            corrections[:] = corrections[matched_frames[0]]

        corr_enu_mag = np.linalg.norm(
            corrections @ (s_m2e * R_m2e).T, axis=1)
        print(f"    Per-frame corrections: "
              f"mean={np.mean(corr_enu_mag):.2f}m, "
              f"max={np.max(corr_enu_mag):.2f}m (ENU)")

        torch.cuda.empty_cache()

        # Store adapted tokens + corrections for Phase 5
        self._per_frame_leaf_tokens = per_frame_leaf_tokens
        self._per_frame_all_tokens = per_frame_all_tokens
        self._per_frame_pose = per_frame_pose
        self._psi = psi
        self._per_frame_params = None
        self._centroids = None
        self._geo_token_corrections = corrections  # (N, 2) model-space

        return {
            "skipped": False,
            "losses": losses,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
        }

    # ==================================================================
    #  PTS3D_RESIDUAL mode — optimise a per-frame XY residual on pts3d
    # ==================================================================

    def _adapt_window_pts3d(self, model, frames, dom_image_cpu, project_fn,
                            enu_offset, dtype=torch.bfloat16, gt_enu_window=None):
        cfg = self.config
        device = next(model.parameters()).device
        N = len(frames)
        if N < cfg.warmup_frames:
            return {"skipped": True, "reason": "too_few_frames"}

        # === Phase 1: frozen forward → cache pts3d, poses ===
        print(f"  [GeoTTTv2-pts3d] Phase 1: frozen forward ({N} frames)...")
        _, pts_list, pose_list, _, psi = self._frozen_forward(
            model, frames, dtype=dtype,
        )

        # === differentiable affine (ENU meters → DOM pixels) ===
        A, b = build_differentiable_projection(project_fn, enu_offset, device)

        # === Model→ENU alignment (corrects ~264× scale mismatch) ===
        if gt_enu_window is not None and len(gt_enu_window) == N:
            from streamvggt.utils.rotation import quat_to_mat
            pose_stack = torch.cat(pose_list, dim=0)  # [N, 9]
            T_cam = pose_stack[:, :3]
            quat = pose_stack[:, 3:7]
            R_cam = quat_to_mat(quat)
            model_centers = -torch.bmm(
                R_cam.transpose(1, 2), T_cam.unsqueeze(-1)
            ).squeeze(-1).cpu().numpy()  # [N, 3]

            # GT positions relative to window anchor (enu_offset)
            gt_local = gt_enu_window - enu_offset  # [N, 3]

            s_m2e, R_m2e, t_m2e = compute_model_to_enu_transform(
                model_centers, gt_local
            )

            # Corrected affine:
            #   enu_local_xy = s * R @ model_xy + t
            #   dom_uv = A @ enu_local_xy + b = (A @ sR) @ model_xy + (A @ t + b)
            sR = torch.tensor(s_m2e * R_m2e, dtype=torch.float32, device=device)
            t_vec = torch.tensor(t_m2e, dtype=torch.float32, device=device)
            b = A @ t_vec + b   # b_new = A_orig @ t + b_orig  (BEFORE overwriting A)
            A = A @ sR           # A_new = A_orig @ sR

            print(f"    Model→ENU: scale={s_m2e:.2f}, "
                  f"t=({t_m2e[0]:.2f}, {t_m2e[1]:.2f}) m")
        else:
            s_m2e, R_m2e = None, None
            print("    WARNING: no gt_enu_window — using raw affine (WILL BE WRONG)")

        # === DOM crop ===
        dom_crop, b_crop, crop_h, crop_w = crop_dom_for_window(
            dom_image_cpu, A, b, pts_list,
            padding=cfg.dom_crop_padding_px, device=device,
        )
        print(f"    DOM crop: {crop_h}×{crop_w}, "
              f"{dom_crop.nelement()*4/1e6:.0f} MB on GPU")

        # === Learnable: per-frame rigid body motion (rotation + translation) ===
        # 3 params per frame: theta (rotation), tx, ty (translation)
        # Applied relative to pts3d centroid:
        #   pts_corrected = R(theta) @ (pts_xy - centroid) + centroid + [tx, ty]
        params_per_frame = []
        centroids = []
        for fi in range(N):
            p = torch.zeros(3, device=device, requires_grad=True)   # [theta, tx, ty]
            params_per_frame.append(p)
            centroids.append(pts_list[fi][..., :2].mean(dim=(0, 1, 2)).detach().to(device))

        print(f"    Per-frame rigid body: {N} frames × 3 = {N * 3} params")

        optimizer = torch.optim.Adam(params_per_frame, lr=cfg.kv_lr)

        fi_sel = list(range(N))

        # === TTT loop ===
        losses = []
        print(f"  [GeoTTTv2-pts3d] TTT ({cfg.num_steps} steps, "
              f"{len(fi_sel)} frames, lr={cfg.kv_lr})")

        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            s_loss, n_ok = 0.0, 0

            for fi in fi_sel:
                pts3d = pts_list[fi].to(device)  # [1, H, W, 3], detached
                p = params_per_frame[fi]
                theta, tx, ty = p[0], p[1], p[2]
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                # 2D rotation matrix
                R_2d = torch.stack([
                    torch.stack([cos_t, -sin_t]),
                    torch.stack([sin_t,  cos_t]),
                ])  # [2, 2]
                # Apply rigid transform relative to centroid
                c = centroids[fi]
                pts_xy = pts3d[..., :2] - c
                pts_xy = torch.einsum('ij,...j->...i', R_2d, pts_xy)
                pts_xy = pts_xy + c + torch.stack([tx, ty])

                dom_uv = torch.einsum('ij,...j->...i', A, pts_xy) + b_crop
                grid_x = (dom_uv[..., 0] / max(crop_w - 1, 1)) * 2.0 - 1.0
                grid_y = (dom_uv[..., 1] / max(crop_h - 1, 1)) * 2.0 - 1.0
                grid = torch.stack([grid_x, grid_y], dim=-1)

                sampled = F.grid_sample(
                    dom_crop, grid.float(), mode="bilinear",
                    padding_mode="border", align_corners=True,
                )

                _, _, Hpt, Wpt = sampled.shape
                cam = F.interpolate(
                    frames[fi]["img"][:1].float().to(device),
                    size=(Hpt, Wpt), mode="bilinear", align_corners=False,
                )

                if cfg.loss_type == "l1":
                    px = (sampled - cam).abs().mean(dim=1)
                else:
                    px = (sampled - cam).pow(2).mean(dim=1)

                in_bounds = (grid[..., 0].abs() < 1) & (grid[..., 1].abs() < 1)
                px = px * in_bounds.float()
                loss = px.sum() / in_bounds.float().sum().clamp(min=1.0)

                (loss / len(fi_sel)).backward()
                s_loss += loss.item()
                n_ok += 1
                del loss

            if cfg.reg_weight > 0:
                reg = sum(p.pow(2).sum() for p in params_per_frame)
                (cfg.reg_weight * reg).backward()

            g = sum(p.grad.norm().item() for p in params_per_frame if p.grad is not None)
            optimizer.step()

            avg = s_loss / max(n_ok, 1)
            losses.append(avg)
            print(f"    step {step}: loss={avg:.4f}, grad={g:.2e}")

        del dom_crop
        torch.cuda.empty_cache()

        # Store per-frame corrections for pose update
        self._per_frame_pose = pose_list
        self._per_frame_params = [p.detach() for p in params_per_frame]  # [theta, tx, ty]
        self._centroids = centroids
        self._model_to_enu_scale = s_m2e
        self._model_to_enu_R = R_m2e

        for fi in range(min(5, N)):
            p = params_per_frame[fi].detach()
            theta_deg = float(p[0]) * 180.0 / 3.14159
            print(f"    Frame {fi}: theta={theta_deg:+.3f}°, "
                  f"tx={float(p[1]):+.6f}, ty={float(p[2]):+.6f}")

        return {
            "skipped": False,
            "losses": losses,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
        }

    # ==================================================================
    #  GEO_CONSIST mode — Test3R-style geometric consistency with DOM
    # ==================================================================

    def _adapt_window_geo_consist(self, model, frames, dom_image_cpu, project_fn,
                                   enu_offset, dtype=torch.bfloat16,
                                   gt_enu_window=None, correspondences=None):
        """Adapt window using Test3R-style geometric consistency with DOM.

        Instead of photometric DOM re-rendering loss, uses direct geometric
        correspondences: RoMa-matched pixels should map to correct absolute
        ENU positions via the predicted 3D points.

        Loss = |pts3d_corrected[pixel] - target_model[pixel]|
        where target_model = inverse(model→ENU transform) of DOM-derived ENU.
        """
        cfg = self.config
        device = next(model.parameters()).device
        N = len(frames)
        if N < cfg.warmup_frames:
            return {"skipped": True, "reason": "too_few_frames"}

        if not correspondences:
            print("  [GeoTTTv2-geo_consist] No correspondences, skipping")
            return {"skipped": True, "reason": "no_correspondences"}

        # === Phase 1: frozen forward ===
        print(f"  [GeoTTTv2-geo_consist] Phase 1: frozen forward ({N} frames)...")
        _, pts_list, pose_list, _, psi = self._frozen_forward(
            model, frames, dtype=dtype)

        # === Phase 2: Model→ENU alignment from correspondences ===
        # Instead of aligning camera centers to GT (poor for scene points),
        # align the pts3d at correspondence pixels directly to DOM ENU.
        all_model_xy = []
        all_enu_local = []
        corr_grids = {}   # fi_local → (grid, enu_abs, query_W, query_H)
        for fi_local, corr in correspondences.items():
            if fi_local < 0 or fi_local >= N:
                continue

            query_W = corr['query_W']
            query_H = corr['query_H']
            pixels = corr['frame_pixels']   # [K, 2] (col, row)
            enu_abs = corr['enu_positions']  # [K, 2] (E, N)
            K = len(pixels)

            # Normalize pixel coords for grid_sample [-1, 1]
            u_norm = pixels[:, 0] / max(query_W - 1, 1)
            v_norm = pixels[:, 1] / max(query_H - 1, 1)
            grid = torch.tensor(
                np.stack([2.0 * u_norm - 1.0, 2.0 * v_norm - 1.0], axis=-1),
                dtype=torch.float32,
            ).reshape(1, 1, K, 2)

            # Sample frozen pts3d at correspondence pixels
            pts3d = pts_list[fi_local]  # [1, H, W, 3]
            pts_xy_img = pts3d[..., :2].permute(0, 3, 1, 2)  # [1, 2, H, W]
            sampled = F.grid_sample(
                pts_xy_img, grid.to(pts3d.device), mode='bilinear',
                align_corners=True, padding_mode='border',
            )  # [1, 2, 1, K]
            model_xy = sampled[0, :, 0, :].T.cpu().numpy()  # [K, 2]

            enu_local = enu_abs - enu_offset[:2]
            all_model_xy.append(model_xy)
            all_enu_local.append(enu_local)
            corr_grids[fi_local] = (grid, enu_abs, query_W, query_H)

        if not all_model_xy:
            print("  [GeoTTTv2-geo_consist] No valid targets, skipping")
            return {"skipped": True, "reason": "no_valid_targets"}

        all_model_xy = np.concatenate(all_model_xy, axis=0)
        all_enu_local = np.concatenate(all_enu_local, axis=0)
        n_total = len(all_model_xy)

        # --- RANSAC Sim2: robust to high outlier rates (>50%) ---
        s_m2e, R_m2e, t_m2e, keep = ransac_sim2(
            all_model_xy, all_enu_local,
            reproj_thresh=8.0,  # 8m inlier threshold
            max_iters=3000,
        )
        n_inliers = int(keep.sum())
        aligned_enu = s_m2e * (all_model_xy @ R_m2e.T) + t_m2e
        all_err = np.linalg.norm(aligned_enu - all_enu_local, axis=1)
        inlier_err = all_err[keep]
        print(f"    RANSAC Sim2: scale={s_m2e:.2f}, "
              f"{n_total} pairs → {n_inliers} inliers ({100*n_inliers/n_total:.0f}%), "
              f"inlier residual={np.mean(inlier_err):.2f}m "
              f"(med={np.median(inlier_err):.2f}m, max={np.max(inlier_err):.2f}m)")

        if n_inliers < 10:
            print("  [GeoTTTv2-geo_consist] Too few RANSAC inliers, skipping")
            return {"skipped": True, "reason": "too_few_inliers"}

        # Refit on inliers only for precision
        inlier_model = all_model_xy[keep]
        inlier_enu = all_enu_local[keep]
        model_3d = np.column_stack([inlier_model, np.zeros(n_inliers)])
        enu_3d = np.column_stack([inlier_enu, np.zeros(n_inliers)])
        s_m2e, R_m2e, t_m2e = compute_model_to_enu_transform(model_3d, enu_3d)

        aligned2 = s_m2e * (inlier_model @ R_m2e.T) + t_m2e
        err2 = np.linalg.norm(aligned2 - inlier_enu, axis=1)
        med_refined = np.median(err2)
        print(f"    Refined on inliers: scale={s_m2e:.2f}, "
              f"residual={np.mean(err2):.2f}m "
              f"(med={med_refined:.2f}m, max={np.max(err2):.2f}m)")

        # Quality gate
        max_residual_m = 15.0
        if med_refined > max_residual_m:
            print(f"  [GeoTTTv2-geo_consist] Alignment residual too high "
                  f"({med_refined:.1f}m > {max_residual_m}m), skipping corrections")
            self._per_frame_pose = pose_list
            self._per_frame_params = None
            self._centroids = None
            return {"skipped": False, "losses": [0.0],
                    "loss_improvement": 0.0, "quality_skip": True}

        # Filter per-frame correspondence grids to keep only RANSAC inliers
        offset = 0
        corr_grids_filtered = {}
        for fi_local in sorted(corr_grids.keys()):
            grid, enu_abs, query_W, query_H = corr_grids[fi_local]
            n_this = len(enu_abs)
            frame_keep = keep[offset:offset + n_this]
            offset += n_this
            if frame_keep.sum() < 3:
                continue
            g = grid.reshape(-1, 2)[frame_keep]
            corr_grids_filtered[fi_local] = (
                g.reshape(1, 1, -1, 2),
                enu_abs[frame_keep],
                query_W, query_H)
        corr_grids = corr_grids_filtered

        sR = torch.tensor(s_m2e * R_m2e, dtype=torch.float32, device=device)
        t_vec = torch.tensor(t_m2e, dtype=torch.float32, device=device)
        sR_inv = torch.inverse(sR)  # [2, 2]

        # === Phase 3: Per-frame learnable rigid body corrections ===
        params_per_frame = []
        centroids = []
        for fi in range(N):
            p = torch.zeros(3, device=device, requires_grad=True)
            params_per_frame.append(p)
            centroids.append(
                pts_list[fi][..., :2].mean(dim=(0, 1, 2)).detach().to(device))

        # === Build per-frame geometric targets in MODEL coordinates ===
        geo_targets = {}
        total_corr = 0
        for fi_local, (grid, enu_abs, query_W, query_H) in corr_grids.items():
            enu_local = enu_abs - enu_offset[:2]
            enu_local_t = torch.tensor(
                enu_local, dtype=torch.float32, device=device)
            target_model = torch.einsum(
                'ij,kj->ki', sR_inv, enu_local_t - t_vec)  # [K, 2]

            geo_targets[fi_local] = (grid.to(device), target_model)
            total_corr += len(enu_abs)

        n_matched = len(geo_targets)
        print(f"    Targets: {n_matched} frames, "
              f"{total_corr} correspondences, "
              f"{N * 3} params")

        optimizer = torch.optim.Adam(params_per_frame, lr=cfg.kv_lr)

        # === Phase 4: TTT optimization ===
        losses = []
        print(f"  [GeoTTTv2-geo_consist] TTT "
              f"({cfg.num_steps} steps, lr={cfg.kv_lr})")

        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            s_loss, n_ok = 0.0, 0

            for fi in range(N):
                if fi not in geo_targets:
                    continue

                grid, target_model = geo_targets[fi]
                pts3d = pts_list[fi].to(device)  # [1, H, W, 3]
                p = params_per_frame[fi]
                theta, tx, ty = p[0], p[1], p[2]

                # Per-frame rigid correction on XY
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                R_2d = torch.stack([
                    torch.stack([cos_t, -sin_t]),
                    torch.stack([sin_t,  cos_t]),
                ])
                c = centroids[fi]
                pts_xy = pts3d[..., :2] - c
                pts_xy = torch.einsum('ij,...j->...i', R_2d, pts_xy)
                pts_xy = pts_xy + c + torch.stack([tx, ty])

                # Sample corrected pts at correspondence locations
                pts_xy_img = pts_xy.permute(0, 3, 1, 2)  # [1, 2, H, W]
                sampled = F.grid_sample(
                    pts_xy_img, grid, mode='bilinear',
                    align_corners=True, padding_mode='border',
                )  # [1, 2, 1, K]
                sampled_xy = sampled[0, :, 0, :].T  # [K, 2]

                # Geometric consistency loss (model coords)
                if cfg.loss_type == "l1":
                    loss = (sampled_xy - target_model).abs().mean()
                elif cfg.loss_type == "mse":
                    loss = (sampled_xy - target_model).pow(2).mean()
                else:
                    loss = F.smooth_l1_loss(sampled_xy, target_model)

                (loss / n_matched).backward()
                s_loss += loss.item()
                n_ok += 1
                del loss

            if cfg.reg_weight > 0:
                reg = sum(p.pow(2).sum() for p in params_per_frame)
                (cfg.reg_weight * reg).backward()

            g = sum(
                p.grad.norm().item()
                for p in params_per_frame if p.grad is not None
            )
            optimizer.step()

            avg = s_loss / max(n_ok, 1)
            losses.append(avg)
            enu_err = avg * s_m2e  # approximate ENU error
            if step % max(1, cfg.num_steps // 5) == 0 \
                    or step == cfg.num_steps - 1:
                print(f"    step {step:3d}: loss={avg:.6f} "
                      f"(~{enu_err:.2f}m), |grad|={g:.4f}")

        # Store results for pose extraction + corrections
        self._per_frame_pose = pose_list
        self._per_frame_params = [
            p.detach() for p in params_per_frame]
        self._centroids = centroids
        self._model_to_enu_scale = s_m2e
        self._model_to_enu_R = R_m2e

        # Show corrections for matched frames (not just first 5)
        matched_fis = sorted(geo_targets.keys())
        for fi in matched_fis[:8]:
            p = params_per_frame[fi].detach()
            theta_deg = float(p[0]) * 180.0 / 3.14159
            enu_shift = s_m2e * np.sqrt(
                float(p[1])**2 + float(p[2])**2)
            print(f"    Frame {fi}: theta={theta_deg:+.3f}°, "
                  f"tx={float(p[1]):+.6f}, ty={float(p[2]):+.6f} "
                  f"(~{enu_shift:.2f}m)")

        return {
            "skipped": False,
            "losses": losses,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0.0,
            "pts3d_data": {
                "pts3d_list": [p.cpu() for p in pts_list],
                "sim2": (s_m2e, R_m2e.copy(), t_m2e.copy()),
                "per_frame_params": [p.detach().cpu().numpy() for p in params_per_frame],
                "centroids": [c.cpu().numpy() for c in centroids],
            },
        }

    # ==================================================================
    #  Per-frame corrections extraction
    # ==================================================================

    def compute_per_frame_corrections(self, centers):
        """Extract per-frame XY corrections from stored rigid-body params.

        Converts the learned (theta, tx, ty) per frame into XY displacement
        vectors in model coordinates. Returns zero-mean corrections to avoid
        shifting the global trajectory.

        Args:
            centers: (N, 3) camera centers in model coordinates for one window.

        Returns:
            corrections: (N, 2) XY corrections in model coordinates.
        """
        N = len(centers)
        corrections = np.zeros((N, 2))

        # geo_consist_token: pre-computed corrections from adapted pts3d
        if (hasattr(self, '_geo_token_corrections')
                and self._geo_token_corrections is not None):
            n_corr = min(N, len(self._geo_token_corrections))
            corrections[:n_corr] = self._geo_token_corrections[:n_corr]
            return corrections

        if (self.config.adapt_mode not in ("pts3d_residual", "geo_consist")
                or not hasattr(self, '_per_frame_params')
                or self._per_frame_params is None):
            return corrections

        n_corr = min(N, len(self._per_frame_params))
        has_centroids = (hasattr(self, '_centroids')
                         and self._centroids is not None
                         and len(self._centroids) >= n_corr)

        # Model→ENU scale for magnitude cap
        s_m2e = getattr(self, '_model_to_enu_scale', None)

        for fi in range(n_corr):
            p = self._per_frame_params[fi].cpu().numpy()
            theta, tx, ty = float(p[0]), float(p[1]), float(p[2])
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            # Use pts3d centroid as rotation center (same as optimization)
            if has_centroids:
                c = self._centroids[fi].cpu().numpy()[:2]
            else:
                c = centers[:n_corr, :2].mean(axis=0)
            dx = centers[fi, 0] - c[0]
            dy = centers[fi, 1] - c[1]
            # corrected = R(cam - c) + c + [tx, ty]  (same direction as optimization)
            new_x = cos_t * dx - sin_t * dy + c[0] + tx
            new_y = sin_t * dx + cos_t * dy + c[1] + ty
            corrections[fi] = [new_x - centers[fi, 0],
                                new_y - centers[fi, 1]]

        # Per-frame magnitude cap (in model coords, corresponding to ~10m ENU)
        max_enu_m = 10.0
        if s_m2e is not None and s_m2e > 0:
            max_model = max_enu_m / s_m2e
            for fi in range(n_corr):
                mag = np.linalg.norm(corrections[fi])
                if mag > max_model:
                    corrections[fi] *= max_model / mag

        # Zero-mean to avoid affecting global trajectory placement
        corrections -= corrections.mean(axis=0)
        return corrections

    # ==================================================================
    #  Full inference pipeline
    # ==================================================================

    def run_inference(self, model, image_paths, device, dtype, dom_image_tensor,
                      project_fn, gt_enu,
                      window_size=32, inv_project_fn=None,
                      use_dom_anchors=True, load_frames_fn=None,
                      output_dir=None, export_pts3d=False,
                      pts3d_stride=5, pts3d_conf_thresh=1.0,
                      pts3d_voxel_size=0.5,
                      wgs84_anchor=None,
                      geo_elev=None, dom_transform=None):
        """Run GeoTTT v2 full inference pipeline.

        For each window:
          1. Load frames, adapt with DOM rendering loss
          2. Extract refined poses and per-frame corrections
          3. Chain windows with optional DOM anchor correction

        Args:
            model: StreamVGGT model.
            image_paths: list of image file paths.
            device: torch device.
            dtype: torch dtype for mixed precision.
            dom_image_tensor: (3, H, W) uint8 tensor on CPU.
            project_fn: ENU → DOM pixel mapping.
            gt_enu: (N, 3) GT ENU positions (for model→ENU alignment).
            window_size: frames per window.
            inv_project_fn: DOM pixel → ENU mapping (for DOM anchors).
            use_dom_anchors: whether to use DOM-RANSAC anchors for drift correction.
            load_frames_fn: callable(image_paths, device) → list of frame dicts.
            geo_elev: GeoElevationQuery instance for DEM/building queries.
            dom_transform: rasterio Affine geotransform of DOM.

        Returns:
            global_centers: (N, 3) trajectory in approximate ENU.
            all_diagnostics: list of per-window diagnostics dicts.
        """
        from streamvggt.utils.trajectory_chain import (
            pose_enc_to_camera_centers, chain_window_centers,
            chain_window_centers_with_dom_anchors,
        )
        from streamvggt.utils.dom_matching import compute_dom_anchors

        cfg = self.config
        N = len(image_paths)
        if cfg.adapt_mode == "lora":
            print(f"[GeoTTTv2-LoRA] Running on {N} frames (window={window_size}, "
                  f"rank={cfg.lora_rank}, steps={cfg.num_steps}, "
                  f"lr={cfg.lora_lr}, batch_ratio={cfg.lora_batch_ratio}, "
                  f"seq_share={cfg.lora_sequence_share})")
        else:
            print(f"[GeoTTTv2] Running on {N} frames (window={window_size}, "
                  f"kv_lr={cfg.kv_lr}, steps={cfg.num_steps}, "
                  f"layers={cfg.kv_update_layers}, reg={cfg.reg_weight})")

        all_window_centers = []
        all_window_corrections = []
        all_diagnostics = []
        all_pts3d_windows = []  # per-window pts3d data (only if export_pts3d)
        all_baseline_pts3d_windows = []  # per-window baseline (no-TTT) pts3d
        all_enu_offsets = []    # per-window ENU offsets
        all_window_ranges = []  # (start, end) global frame indices per window

        num_windows = (N + window_size - 1) // window_size

        # Reset per-layer gradient diagnostic flag
        if hasattr(self, '_layer_grad_printed'):
            del self._layer_grad_printed

        # === DOM-RANSAC anchors (computed BEFORE model uses GPU) ===
        # Skip for pose_ttt/pose_refine: dense RoMa correspondences fully replace boundary anchors
        dom_enu_anchors = None
        if use_dom_anchors and inv_project_fn is not None \
                and cfg.adapt_mode not in ("pose_refine", "pose_ttt", "pose_ttt_token", "pose_ttt_rerun23"):
            vis_dir = os.path.join("eval_results", "japan_dom", "match_vis")
            dom_enu_anchors = compute_dom_anchors(
                image_paths, dom_image_tensor, gt_enu, project_fn, inv_project_fn,
                window_size=window_size, device=device,
                save_vis_dir=vis_dir)

        # === Dense correspondences ===
        # pose_ttt and pose_ttt_token use streaming DOM matching (per-window,
        # using predicted poses as crop centres).  Other modes use batch pre-computation.
        dense_corr = None
        dense_anchors = None  # frame_idx → (2,) ENU
        _streaming_roma_model = None  # held for streaming mode

        _use_streaming = (cfg.adapt_mode in ("pose_ttt", "pose_ttt_token")
                          and inv_project_fn is not None)

        if _use_streaming:
            # Load RoMa once — will be passed into each window's adapt call
            from streamvggt.utils.dom_matching import load_roma
            print(f"[streaming-DOM] Loading RoMa outdoor model for streaming matching...")
            _streaming_roma_model = load_roma(device=device)
            # Reset streaming state
            self._streaming_sim2_sR = None
            self._streaming_last_enu = gt_enu[0, :2].copy()
        elif cfg.adapt_mode in ("geo_consist", "geo_consist_token", "pose_refine",
                                "pose_ttt_rerun23") \
                and inv_project_fn is not None:
            from streamvggt.utils.dom_matching import compute_dense_correspondences
            _roma_vis_dir = os.path.join(output_dir, "roma_vis") if output_dir else None
            # Build camera_fov dict for FOV-matched crop
            _camera_fov = None
            if cfg.camera_fx is not None and cfg.base_altitude_m is not None:
                _camera_fov = {
                    'fx': cfg.camera_fx, 'fy': cfg.camera_fy,
                    'orig_w': cfg.camera_orig_w, 'orig_h': cfg.camera_orig_h,
                    'altitude': cfg.base_altitude_m,
                }
            dense_corr, dense_anchors = compute_dense_correspondences(
                image_paths, dom_image_tensor, gt_enu,
                project_fn, inv_project_fn,
                stride=cfg.geo_consist_stride, device=device,
                crop_size_m=cfg.geo_consist_crop_m,
                max_correspondences=cfg.geo_consist_max_corr,
                save_vis_dir=_roma_vis_dir,
                camera_fov=_camera_fov,
                geo_elev=geo_elev,
                dom_transform=dom_transform,
            )

        dom_img = dom_image_tensor.unsqueeze(0)  # [1, 3, H, W] uint8, CPU

        for w in range(num_windows):
            start = w * window_size
            end = min(start + window_size, N)
            window_paths = image_paths[start:end]
            all_window_ranges.append((start, end))

            frames = load_frames_fn(window_paths, device)

            # ENU offset: first window uses GPS of frame 0;
            # subsequent windows use TTT-ENU estimate from previous window's
            # last frame (truly streaming — no per-frame GT needed).
            if w == 0:
                enu_offset = gt_enu[0].copy()
            elif _use_streaming and hasattr(self, '_streaming_last_enu'):
                # Propagate from streaming state (absolute ENU of last matched)
                enu_offset = np.array([self._streaming_last_enu[0],
                                       self._streaming_last_enu[1],
                                       gt_enu[0, 2] if gt_enu.shape[1] > 2 else 0.0])
            else:
                enu_offset = gt_enu[start].copy()

            # --- extract window-local correspondences ---
            window_corr = None
            if dense_corr is not None:
                window_corr = {}
                for fi_global, corr in dense_corr.items():
                    if start <= fi_global < end:
                        window_corr[fi_global - start] = corr

            # --- adapt ---
            diag = self.adapt_window(
                model, frames, dom_img, project_fn,
                enu_offset, dtype=dtype,
                gt_enu_window=gt_enu[start:end],
                correspondences=window_corr,
                roma_model=_streaming_roma_model,
                image_paths=window_paths,
                inv_project_fn=inv_project_fn,
                geo_elev=geo_elev,
                dom_transform=dom_transform,
            )

            if diag.get("skipped"):
                # Fallback: normal inference
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=dtype):
                        output = model.inference(frames)
                window_pose_encs = []
                for res in output.ress:
                    window_pose_encs.append(res["camera_pose"].cpu().float())
                window_pose_encs = torch.cat(window_pose_encs, dim=0)
            else:
                # Phase 5: refined poses
                if cfg.adapt_mode == "lora":
                    adapted_poses = self.compute_adapted_poses_lora(model, frames, dtype=dtype)
                elif cfg.adapt_mode in ("token", "pts3d_residual", "geo_consist",
                                        "geo_consist_token", "pose_refine",
                                        "pose_ttt", "pose_ttt_token",
                                        "pose_ttt_rerun23"):
                    adapted_poses = self.compute_adapted_poses_token(model, frames, dtype=dtype)
                else:
                    adapted_poses = self.compute_adapted_poses(model, frames, dtype=dtype)
                window_pose_encs = torch.cat([p.cpu().float() for p in adapted_poses], dim=0)

            centers = pose_enc_to_camera_centers(window_pose_encs).numpy()

            # Extract per-frame corrections from learned rigid-body params
            window_corrections = self.compute_per_frame_corrections(centers)

            all_window_centers.append(centers)
            all_window_corrections.append(window_corrections)
            diag["enu_offset"] = enu_offset.copy()
            all_diagnostics.append(diag)
            all_enu_offsets.append(enu_offset.copy())

            # Collect pts3d if requested (reuses adapted KV from Phase 4)
            if export_pts3d and cfg.adapt_mode == "pose_ttt" \
                    and not diag.get("skipped") and hasattr(self, '_leaf_kv'):
                pts3d_window = self.compute_adapted_pts3d(
                    model, frames, dtype=dtype, stride=pts3d_stride)
                all_pts3d_windows.append(pts3d_window)
                # Also collect baseline (original KV, no TTT) pts3d
                if hasattr(self, '_original_past_kv'):
                    baseline_window = self.compute_adapted_pts3d(
                        model, frames, dtype=dtype, stride=pts3d_stride,
                        kv_override=self._original_past_kv)
                    all_baseline_pts3d_windows.append(baseline_window)
                else:
                    all_baseline_pts3d_windows.append(None)
            else:
                all_pts3d_windows.append(None)
                all_baseline_pts3d_windows.append(None)

            del frames
            torch.cuda.empty_cache()

            # (RoMa kept alive for all windows — freed after the loop)

            # For LoRA mode without sequence sharing, remove after each window
            if (cfg.adapt_mode == "lora" and not cfg.lora_sequence_share
                    and hasattr(self, '_lora_injected') and self._lora_injected):
                from streamvggt.utils.lora import remove_lora
                remove_lora(model)
                self._lora_injected = False
                self._lora_params = None

            if (w + 1) % 5 == 0 or w == num_windows - 1:
                losses_str = ""
                if isinstance(diag, dict) and "losses" in diag:
                    losses_str = f", losses: {[f'{l:.4f}' for l in diag['losses']]}"
                print(f"  Window {w + 1}/{num_windows} done "
                      f"(frames {start}-{end - 1}{losses_str})")

        # === Cleanup LoRA after all windows ===
        if (cfg.adapt_mode == "lora"
                and hasattr(self, '_lora_injected') and self._lora_injected):
            from streamvggt.utils.lora import remove_lora
            remove_lora(model)
            self._lora_injected = False
            self._lora_params = None

        # === Cleanup streaming RoMa model ===
        if _streaming_roma_model is not None:
            del _streaming_roma_model
            torch.cuda.empty_cache()
            print(f"[streaming-DOM] RoMa model freed")

        # === Build output trajectory ===
        pts3d_enu_result = None

        if cfg.adapt_mode == "pose_ttt_token":
            # ---- Direct ENU output (no chaining / no post-processing) ----
            # Collect per-window TTT-ENU and Baseline-ENU trajectories
            ttt_enu_all = []
            bl_enu_all = []
            has_direct = True
            for w, diag in enumerate(all_diagnostics):
                if isinstance(diag, dict) and 'ttt_enu_trajectory' in diag:
                    ttt_enu_all.append(diag['ttt_enu_trajectory'])
                    bl_enu_all.append(diag.get('baseline_enu_trajectory',
                                               diag['ttt_enu_trajectory']))
                else:
                    has_direct = False
                    break

            if has_direct and ttt_enu_all:
                ttt_enu_global = np.concatenate(ttt_enu_all, axis=0)[:N]
                bl_enu_global = np.concatenate(bl_enu_all, axis=0)[:N]
                # Pad to (N, 3) for compatibility with caller (add Z=0)
                global_centers = np.column_stack([
                    ttt_enu_global, np.zeros(N)])

                # Print direct comparison: Baseline vs TTT (no chaining)
                if gt_enu is not None and len(gt_enu) > 0:
                    gt_arr = np.array(gt_enu)
                    n_eval = min(len(ttt_enu_global), len(gt_arr))

                    # Baseline-ENU direct
                    bl_err = np.linalg.norm(
                        bl_enu_global[:n_eval, :2] - gt_arr[:n_eval, :2], axis=1)
                    bl_rmse = np.sqrt(np.mean(bl_err**2))
                    bl_mean = np.mean(bl_err)
                    bl_med = np.median(bl_err)
                    bl_max = np.max(bl_err)

                    # TTT-ENU direct
                    ttt_err = np.linalg.norm(
                        ttt_enu_global[:n_eval, :2] - gt_arr[:n_eval, :2], axis=1)
                    ttt_rmse = np.sqrt(np.mean(ttt_err**2))
                    ttt_mean = np.mean(ttt_err)
                    ttt_med = np.median(ttt_err)
                    ttt_max = np.max(ttt_err)

                    pct = (1 - ttt_rmse / bl_rmse) * 100 if bl_rmse > 1e-6 else 0.0

                    print(f"\n{'='*60}")
                    print(f"  DIRECT ENU EVALUATION (no chaining)")
                    print(f"{'='*60}")
                    print(f"  {'Metric':<18} {'Baseline':<12} {'TTT':<12} {'Improv.'}")
                    print(f"  {'-'*54}")
                    print(f"  {'XY RMSE':<18} {bl_rmse:<12.4f} {ttt_rmse:<12.4f} {pct:+.1f}%")
                    print(f"  {'XY Mean':<18} {bl_mean:<12.4f} {ttt_mean:<12.4f}")
                    print(f"  {'XY Median':<18} {bl_med:<12.4f} {ttt_med:<12.4f}")
                    print(f"  {'XY Max':<18} {bl_max:<12.4f} {ttt_max:<12.4f}")
                    print(f"  {'Last frame XY':<18} {bl_err[-1]:<12.4f} {ttt_err[-1]:<12.4f}")
                    print(f"{'='*60}\n")
            else:
                # Fallback if no direct ENU available
                global_centers = self._chain_and_correct(
                    all_window_centers, all_window_corrections, dom_enu_anchors)

        elif cfg.adapt_mode in ("pose_refine", "pose_ttt", "pose_ttt_rerun23"):
            # ---- Legacy post-processing path for other modes ----
            dense_anchors_enu = {}
            dense_anchors_enu_3d = {}
            for w, diag in enumerate(all_diagnostics):
                start_idx = w * window_size
                if isinstance(diag, dict) and 'optimal_enu' in diag:
                    for fi_local, enu in diag['optimal_enu'].items():
                        dense_anchors_enu[start_idx + fi_local] = enu
                if isinstance(diag, dict) and 'optimal_enu_3d' in diag:
                    for fi_local, enu3d in diag['optimal_enu_3d'].items():
                        dense_anchors_enu_3d[start_idx + fi_local] = enu3d

            global_centers, sim2_per_window = self._chain_with_dense_anchors(
                all_window_centers, dense_anchors_enu, window_size,
                dense_anchors_enu_3d=dense_anchors_enu_3d)
            self._sim2_per_window = sim2_per_window

            # Export pts3d to PLY if requested
            if export_pts3d and output_dir and any(p is not None for p in all_pts3d_windows):
                result = self._export_pts3d_ply(
                    all_pts3d_windows, sim2_per_window, all_enu_offsets,
                    all_window_centers, window_size, output_dir,
                    pts3d_stride=pts3d_stride,
                    conf_thresh=pts3d_conf_thresh,
                    voxel_size=pts3d_voxel_size,
                    wgs84_anchor=wgs84_anchor,
                    output_filename="pts3d_enu.ply",
                )
                if result is not None:
                    pts3d_enu_result = {'points': result[0], 'colors': result[1]}
        else:
            global_centers = self._chain_and_correct(
                all_window_centers, all_window_corrections, dom_enu_anchors)

        return global_centers, all_diagnostics, pts3d_enu_result

    def _export_pts3d_ply(self, all_pts3d_windows, sim2_per_window,
                          all_enu_offsets, all_window_centers, window_size,
                          output_dir, pts3d_stride=5, conf_thresh=1.0,
                          voxel_size=0.5, wgs84_anchor=None,
                          output_filename="pts3d_enu.ply"):
        """Transform model-space pts3d to ENU using per-window Sim2, export PLY.

        For each window with a valid Sim2:
          XY: pts3d_enu_xy = s * (pts3d_model_xy @ R.T) + t + interp_res[fi] + enu_offset[:2] - enu_offsets[0][:2]
          Z:  pts3d_enu_z  = s * (pts3d_model_z - cam_center_z) + enu_offset[2] - enu_offsets[0][2]

        Note: Since each window has its own enu_offset (GPS of first frame),
        we normalize all points relative to enu_offsets[0] so the output is
        in a single ENU frame centered at the first frame.

        If wgs84_anchor is provided as (lon0, lat0, alt0), also exports:
          - pts3d_wgs84.csv  (lon,lat,alt,r,g,b — importable in QGIS)
          - pts3d_wgs84.geojson (GeoJSON FeatureCollection with EPSG:4326)

        Args:
            all_pts3d_windows: list of per-window pts3d lists (or None).
            sim2_per_window: list of Sim2 dicts (or None) per window.
            all_enu_offsets: list of [3] enu_offset arrays per window.
            all_window_centers: list of (W_i, 3) model-space center arrays.
            window_size: frames per window.
            output_dir: directory for output PLY file.
            pts3d_stride: stride used during collection.
            conf_thresh: confidence threshold for filtering.
            voxel_size: voxel downsample size in meters (0 = no downsample).
            wgs84_anchor: (lon0, lat0, alt0) tuple for WGS84 conversion.
        """
        all_points = []
        all_colors = []
        enu_origin = all_enu_offsets[0].copy()  # reference ENU origin
        n_windows_used = 0

        # === Global Sim2 consensus: unified scale & rotation across windows ===
        per_w_scales = []
        per_w_angles = []
        sim2_windows_idx = []
        for w_i, sim2 in enumerate(sim2_per_window):
            if sim2 is None or sim2.get('type') != 'sim2':
                continue
            per_w_scales.append(sim2['s'])
            per_w_angles.append(np.arctan2(sim2['R'][1, 0], sim2['R'][0, 0]))
            sim2_windows_idx.append(w_i)

        if per_w_scales:
            s_global = float(np.median(per_w_scales))
            theta_global = float(np.median(per_w_angles))
            R_global = np.array([
                [np.cos(theta_global), -np.sin(theta_global)],
                [np.sin(theta_global),  np.cos(theta_global)],
            ])
            print(f"[pts3d export] Global Sim2 consensus: "
                  f"s={s_global:.1f} (range {min(per_w_scales):.1f}-"
                  f"{max(per_w_scales):.1f}), "
                  f"\u03b8={np.degrees(theta_global):.2f}\u00b0 "
                  f"(range {np.degrees(min(per_w_angles)):.2f}\u00b0-"
                  f"{np.degrees(max(per_w_angles)):.2f}\u00b0)")

            # Re-fit per-window translation & residuals with global s, R
            n_refit = 0
            for w_i in sim2_windows_idx:
                sim2 = sim2_per_window[w_i]
                if 'anchor_model_xy' not in sim2:
                    continue
                anchor_model = sim2['anchor_model_xy']
                anchor_enu = sim2['anchor_enu_xy']
                fi_local = sim2['anchor_fi']
                W_i = len(all_window_centers[w_i])

                # Re-fit t: minimize ||anchor_enu - s_global * R_global @ anchor_model - t||^2
                approx = s_global * (anchor_model @ R_global.T)
                t_new = np.mean(anchor_enu - approx, axis=0)

                # Recompute residuals at anchor points
                residuals = anchor_enu - (approx + t_new)
                interp_res = np.zeros((W_i, 2))
                for dim in range(2):
                    interp_res[:, dim] = np.interp(
                        np.arange(W_i), fi_local, residuals[:, dim])

                sim2['s'] = s_global
                sim2['R'] = R_global.copy()
                sim2['t'] = t_new.copy()
                sim2['interp_res'] = interp_res.copy()
                n_refit += 1

            print(f"[pts3d export] Re-fitted {n_refit}/{len(sim2_windows_idx)} "
                  f"windows with global s={s_global:.1f}, "
                  f"\u03b8={np.degrees(theta_global):.2f}\u00b0")

        for w, pts3d_win in enumerate(all_pts3d_windows):
            if pts3d_win is None or sim2_per_window[w] is None:
                continue
            sim2 = sim2_per_window[w]
            if sim2['type'] != 'sim2':
                continue

            s = sim2['s']
            R = sim2['R']
            t = sim2['t']
            interp_res = sim2['interp_res']  # [W_i, 2]
            enu_off = all_enu_offsets[w]
            cam_centers = all_window_centers[w]  # [W_i, 3] model space

            n_windows_used += 1

            for fi, frame_data in enumerate(pts3d_win):
                if frame_data is None:
                    continue

                pts_hw3 = frame_data['pts3d'].numpy()   # [H, W, 3]
                conf_hw = frame_data['conf'].numpy() if frame_data['conf'] is not None else None
                rgb_hw3 = frame_data['rgb'].numpy()      # [H, W, 3]

                H, W = pts_hw3.shape[:2]
                pts_flat = pts_hw3.reshape(-1, 3)       # [N, 3]
                rgb_flat = rgb_hw3.reshape(-1, 3)       # [N, 3]

                # Confidence filter
                if conf_hw is not None and conf_thresh > 0:
                    conf_flat = conf_hw.reshape(-1)
                    mask = conf_flat > conf_thresh
                else:
                    mask = np.ones(len(pts_flat), dtype=bool)

                if mask.sum() == 0:
                    continue

                pts_valid = pts_flat[mask].astype(np.float64)
                rgb_valid = rgb_flat[mask]

                # XY: apply unified global Sim2 + per-window t + residual
                xy_model = pts_valid[:, :2]
                xy_enu = s * (xy_model @ R.T) + t + interp_res[fi]

                # Z: scale relative to camera center (s is now global)
                cam_z = cam_centers[fi, 2]
                z_enu = s * (pts_valid[:, 2] - cam_z)

                pts_enu = np.column_stack([xy_enu, z_enu])
                all_points.append(pts_enu.astype(np.float32))

                # Clamp colors to [0, 1]
                rgb_valid = np.clip(rgb_valid, 0.0, 1.0)
                all_colors.append(rgb_valid.astype(np.float32))

        if not all_points:
            print("[pts3d export] No valid pts3d data to export")
            return None

        combined_pts = np.concatenate(all_points, axis=0)
        combined_rgb = np.concatenate(all_colors, axis=0)
        print(f"[pts3d export] {len(combined_pts)} points from "
              f"{n_windows_used} windows")
        # Diagnostic: coordinate ranges
        mn = combined_pts.min(axis=0)
        mx = combined_pts.max(axis=0)
        md = np.median(combined_pts, axis=0)
        print(f"[pts3d export] Bbox: X=[{mn[0]:.1f},{mx[0]:.1f}] "
              f"Y=[{mn[1]:.1f},{mx[1]:.1f}] Z=[{mn[2]:.1f},{mx[2]:.1f}]")
        print(f"[pts3d export] Median: ({md[0]:.1f}, {md[1]:.1f}, {md[2]:.1f})")
        print(f"[pts3d export] ENU origin: ({enu_origin[0]:.1f}, {enu_origin[1]:.1f}, {enu_origin[2]:.1f})")

        # Voxel downsampling
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(combined_pts)
            pcd.colors = o3d.utility.Vector3dVector(combined_rgb)

            if voxel_size > 0:
                pcd_down = pcd.voxel_down_sample(voxel_size)
                print(f"[pts3d export] Voxel downsample ({voxel_size}m): "
                      f"{len(combined_pts)} → {len(pcd_down.points)} points")
                pcd = pcd_down
                # Update arrays for WGS84 export
                combined_pts = np.asarray(pcd.points).astype(np.float64)
                combined_rgb = np.asarray(pcd.colors).astype(np.float32)

            ply_path = os.path.join(output_dir, output_filename)
            o3d.io.write_point_cloud(ply_path, pcd)
            print(f"[pts3d export] Saved ENU point cloud to {ply_path}")
        except ImportError:
            # Fallback: save as NPZ
            npz_name = output_filename.replace('.ply', '.npz')
            npz_path = os.path.join(output_dir, npz_name)
            np.savez_compressed(npz_path, points=combined_pts,
                                colors=combined_rgb)
            print(f"[pts3d export] open3d not available, saved NPZ to {npz_path}")

        # === WGS84 export for QGIS ===
        if wgs84_anchor is not None:
            self._export_pts3d_wgs84(combined_pts, combined_rgb,
                                     wgs84_anchor, enu_origin, output_dir)

        return combined_pts, combined_rgb

    def _export_pts3d_wgs84(self, pts_enu, colors, wgs84_anchor, enu_origin,
                            output_dir):
        """Export ENU point cloud as WGS84 CSV + GeoJSON for QGIS.

        Args:
            pts_enu: (N, 3) ENU points relative to enu_origin.
            colors: (N, 3) RGB in [0, 1].
            wgs84_anchor: (lon0, lat0, alt0) — WGS84 origin of the ENU system.
            enu_origin: (3,) — ENU offset of enu_origin relative to anchor.
            output_dir: output directory.
        """
        import json

        lon0, lat0, alt0 = wgs84_anchor
        # Points are in ENU relative to enu_origin.
        # enu_origin itself is gt_enu[0] which is (0,0,0) relative to anchor.
        # So absolute ENU = pts_enu + enu_origin.
        e_abs = pts_enu[:, 0] + enu_origin[0]
        n_abs = pts_enu[:, 1] + enu_origin[1]
        u_abs = pts_enu[:, 2] + enu_origin[2]

        # ENU → WGS84 (same math as eval_japan_dom.enu_to_geodetic)
        lat0r = np.radians(lat0)
        lon0r = np.radians(lon0)
        sin_lat0 = np.sin(lat0r)
        cos_lat0 = np.cos(lat0r)
        sin_lon0 = np.sin(lon0r)
        cos_lon0 = np.cos(lon0r)

        dx = -sin_lon0 * e_abs - sin_lat0 * cos_lon0 * n_abs + cos_lat0 * cos_lon0 * u_abs
        dy =  cos_lon0 * e_abs - sin_lat0 * sin_lon0 * n_abs + cos_lat0 * sin_lon0 * u_abs
        dz =                      cos_lat0 * n_abs             + sin_lat0 * u_abs

        # WGS84 constants
        _a = 6378137.0
        _f = 1.0 / 298.257223563
        _e2 = 2 * _f - _f ** 2
        _b = _a * (1 - _f)

        # Anchor ECEF
        N_a = _a / np.sqrt(1 - _e2 * np.sin(lat0r) ** 2)
        x0 = (N_a + alt0) * np.cos(lat0r) * np.cos(lon0r)
        y0 = (N_a + alt0) * np.cos(lat0r) * np.sin(lon0r)
        z0 = (N_a * (1 - _e2) + alt0) * np.sin(lat0r)

        x = x0 + dx
        y = y0 + dy
        z = z0 + dz

        # ECEF → geodetic (Bowring iterative)
        p = np.sqrt(x ** 2 + y ** 2)
        lon = np.degrees(np.arctan2(y, x))
        # Initial estimate
        theta = np.arctan2(z * _a, p * _b)
        lat_rad = np.arctan2(
            z + (_a ** 2 - _b ** 2) / _b * np.sin(theta) ** 3,
            p - _e2 * _a * np.cos(theta) ** 3,
        )
        for _ in range(3):
            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)
            N_iter = _a / np.sqrt(1 - _e2 * sin_lat ** 2)
            lat_rad = np.arctan2(z + _e2 * N_iter * sin_lat, p)
        lat = np.degrees(lat_rad)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        N_final = _a / np.sqrt(1 - _e2 * sin_lat ** 2)
        alt = np.where(
            np.abs(cos_lat) > 1e-10,
            p / cos_lat - N_final,
            np.abs(z) / np.abs(sin_lat) - N_final * (1 - _e2),
        )

        n_pts = len(lon)

        # --- CSV for QGIS (Add Delimited Text Layer → EPSG:4326) ---
        csv_path = os.path.join(output_dir, "pts3d_wgs84.csv")
        with open(csv_path, "w") as f:
            f.write("lon,lat,alt,r,g,b\n")
            # Subsample if too many points for CSV
            step = max(1, n_pts // 500000)
            n_written = 0
            for i in range(0, n_pts, step):
                r, g, b = int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255)
                f.write(f"{lon[i]:.8f},{lat[i]:.8f},{alt[i]:.2f},{r},{g},{b}\n")
                n_written += 1
        print(f"[pts3d export] WGS84 CSV: {csv_path} ({n_written} points)")
        print(f"  QGIS: Layer → Add Delimited Text Layer → {csv_path}")
        print(f"  Set X=lon, Y=lat, CRS=EPSG:4326")

        # --- GeoJSON for QGIS (drag-and-drop) ---
        # Subsample more aggressively for GeoJSON (file size)
        step_gj = max(1, n_pts // 100000)
        features = []
        for i in range(0, n_pts, step_gj):
            r, g, b = int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon[i]), float(lat[i]), float(alt[i])],
                },
                "properties": {"r": r, "g": g, "b": b},
            })
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
            },
            "features": features,
        }
        gj_path = os.path.join(output_dir, "pts3d_wgs84.geojson")
        with open(gj_path, "w") as f:
            json.dump(geojson, f)
        print(f"[pts3d export] WGS84 GeoJSON: {gj_path} ({len(features)} points)")
        print(f"  QGIS: drag {gj_path} into map canvas")

    def _chain_and_correct(self, all_window_centers, all_window_corrections,
                           dom_enu_anchors):
        """Chain windows and apply per-frame corrections.

        When DOM anchors are available:
          1. Chain UNCORRECTED centers with DOM anchors first
             (anchor positions must be accurate for Sim2 estimation)
          2. Apply per-frame corrections AFTER chaining
             (corrections are zero-meaned per window, only refine local shape)

        Without DOM anchors:
          1. Chain normally
          2. Apply per-frame corrections after chaining
        """
        from streamvggt.utils.trajectory_chain import (
            chain_window_centers, chain_window_centers_with_dom_anchors,
        )

        has_corrections = any(
            np.abs(wc).max() > 1e-10 for wc in all_window_corrections
        )

        # Apply corrections in MODEL coords BEFORE chaining (so Sim2
        # alignment converts both centers and corrections to ENU together).
        if has_corrections:
            corrected_centers = []
            for centers, corrections in zip(all_window_centers,
                                            all_window_corrections):
                c = centers.copy()
                c[:, 0] += corrections[:, 0]
                c[:, 1] += corrections[:, 1]
                corrected_centers.append(c)
        else:
            corrected_centers = all_window_centers

        if dom_enu_anchors is not None and sum(a is not None for a in dom_enu_anchors) >= 2:
            global_centers = chain_window_centers_with_dom_anchors(
                corrected_centers, dom_enu_anchors)

            n_anchors = sum(a is not None for a in dom_enu_anchors)
            corr_tag = " + pre-chain corrections" if has_corrections else ""
            print(f"[GeoTTTv2] DOM-anchored chaining ({n_anchors} anchors{corr_tag})")
        else:
            global_centers = chain_window_centers(corrected_centers)

        return global_centers

    def _chain_with_dense_anchors(self, all_window_centers, dense_anchors_enu,
                                  window_size, dense_anchors_enu_3d=None):
        """Per-window independent Sim2/Sim3 alignment to absolute ENU.

        When 3D ENU anchors (from PnP) are available, uses 3D Umeyama (Sim3).
        Otherwise falls back to 2D Umeyama (Sim2). Each window independently
        fits its own transform from model-space to ENU:
          - Eliminates cumulative drift from inter-window chaining
          - Allows each window to adapt to local scale/rotation independently
          - Uses residual interpolation within each window for fine correction
          - RANSAC for robust fitting when K ≥ 6

        Args:
            all_window_centers: list of (W_i, 3) arrays in model space per window
            dense_anchors_enu: dict {global_frame_idx: (E, N) absolute ENU}
            window_size: int, frames per window
            dense_anchors_enu_3d: dict {global_frame_idx: (E, N, U) absolute ENU}
                from PnP. If provided and enough 3D anchors exist, Sim3 is used.

        Returns:
            global_centers: (N, 3) trajectory in absolute ENU.
            sim2_per_window: list of dicts per window with Sim2/Sim3 params.
        """
        if dense_anchors_enu_3d is None:
            dense_anchors_enu_3d = {}

        num_windows = len(all_window_centers)
        N = sum(len(wc) for wc in all_window_centers)
        global_centers = np.zeros((N, 3))
        sim2_per_window = [None] * num_windows

        if not dense_anchors_enu:
            from streamvggt.utils.trajectory_chain import chain_window_centers
            print("[pose_refine] No dense anchors, falling back to chain")
            return chain_window_centers(all_window_centers), sim2_per_window

        def fit_sim2_2d(model_pts, enu_pts):
            """Fit 2D similarity: enu ≈ s * R @ model + t (Umeyama)."""
            n = len(model_pts)
            mu_m = model_pts.mean(axis=0)
            mu_e = enu_pts.mean(axis=0)
            dm = model_pts - mu_m
            de = enu_pts - mu_e
            sigma_m2 = np.sum(dm ** 2) / n
            H = dm.T @ de / n
            U, S, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            D_mat = np.diag([1.0, np.sign(d)])
            R = Vt.T @ D_mat @ U.T
            s = np.sum(S * np.diag(D_mat)) / (sigma_m2 + 1e-12)
            t = mu_e - s * R @ mu_m
            return s, R, t

        def fit_sim3_3d(model_pts, enu_pts):
            """Fit 3D similarity: enu ≈ s * R @ model + t (Umeyama 3D)."""
            n = len(model_pts)
            mu_m = model_pts.mean(axis=0)
            mu_e = enu_pts.mean(axis=0)
            dm = model_pts - mu_m
            de = enu_pts - mu_e
            sigma_m2 = np.sum(dm ** 2) / n
            H = dm.T @ de / n  # (3, 3)
            U, S, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            D_mat = np.diag([1.0, 1.0, np.sign(d)])
            R = Vt.T @ D_mat @ U.T  # (3, 3)
            s = np.sum(S * np.diag(D_mat)) / (sigma_m2 + 1e-12)
            t = mu_e - s * R @ mu_m
            return s, R, t

        def fit_sim3_ransac(model_pts, enu_pts, n_iter=200, thresh_m=5.0):
            """RANSAC Sim3: select best inlier set, refit on inliers."""
            n = len(model_pts)
            best_inliers = None
            best_count = 0
            rng = np.random.RandomState(42)
            min_sample = min(4, n)
            for _ in range(n_iter):
                idx = rng.choice(n, min_sample, replace=False)
                try:
                    s, R, t = fit_sim3_3d(model_pts[idx], enu_pts[idx])
                except Exception:
                    continue
                if abs(s) < 1e-8:
                    continue
                pred = s * (model_pts @ R.T) + t
                err = np.linalg.norm(pred - enu_pts, axis=1)
                inliers = err < thresh_m
                cnt = inliers.sum()
                if cnt > best_count:
                    best_count = cnt
                    best_inliers = inliers
            if best_inliers is not None and best_count >= 4:
                s, R, t = fit_sim3_3d(model_pts[best_inliers], enu_pts[best_inliers])
                return s, R, t, best_inliers
            # Fallback: fit on all
            s, R, t = fit_sim3_3d(model_pts, enu_pts)
            return s, R, t, np.ones(n, dtype=bool)

        per_window_stats = []

        for w in range(num_windows):
            if all_window_ranges is not None and w < len(all_window_ranges):
                start = all_window_ranges[w][0]
            else:
                start = w * window_size
            wc = all_window_centers[w]  # (W_i, 3) model space
            W_i = len(wc)

            # Gather anchors for this window (2D and 3D)
            window_anchors = {}
            window_anchors_3d = {}
            for fi_global, enu in dense_anchors_enu.items():
                if start <= fi_global < start + W_i:
                    window_anchors[fi_global - start] = enu
            for fi_global, enu3d in dense_anchors_enu_3d.items():
                if start <= fi_global < start + W_i:
                    window_anchors_3d[fi_global - start] = enu3d

            fi_local = np.array(sorted(window_anchors.keys())) if window_anchors else np.array([], dtype=int)
            fi_local_3d = np.array(sorted(window_anchors_3d.keys())) if window_anchors_3d else np.array([], dtype=int)
            K = len(fi_local)
            K3d = len(fi_local_3d)

            if K >= 3:
                # Check if model centers have sufficient spread
                anchor_model_xy = wc[fi_local, :2]
                model_spread = np.linalg.norm(
                    anchor_model_xy - anchor_model_xy.mean(0), axis=1).max()

                if model_spread > 1e-3:
                    # --- Try Sim3 (3D) when enough PnP anchors ---
                    if K3d >= 4:
                        anchor_model_3d = wc[fi_local_3d]  # (K3d, 3)
                        anchor_enu_3d = np.array([window_anchors_3d[fi] for fi in fi_local_3d])

                        if K3d >= 6:
                            s, R, t, inliers = fit_sim3_ransac(
                                anchor_model_3d, anchor_enu_3d)
                        else:
                            s, R, t = fit_sim3_3d(anchor_model_3d, anchor_enu_3d)
                            inliers = np.ones(K3d, dtype=bool)

                        if abs(s) > 1e-8:
                            # Transform all frames via Sim3
                            enu_all = s * (wc @ R.T) + t  # (W_i, 3)

                            # Residual correction at 3D anchor points
                            approx = s * (anchor_model_3d @ R.T) + t
                            residuals_3d = anchor_enu_3d - approx

                            # Interpolate residuals to non-anchor frames
                            interp_res = np.zeros((W_i, 3))
                            for dim in range(3):
                                interp_res[:, dim] = np.interp(
                                    np.arange(W_i), fi_local_3d, residuals_3d[:, dim])
                            enu_all += interp_res

                            global_centers[start:start+W_i] = enu_all

                            sim2_per_window[w] = {
                                'type': 'sim3',
                                's': s, 'R': R.copy(), 't': t.copy(),
                                'interp_res': interp_res[:, :2].copy(),  # 2D for pts3d export compat
                                'anchor_model_xy': anchor_model_xy.copy(),
                                'anchor_enu_xy': np.array([window_anchors[fi] for fi in fi_local]),
                                'anchor_fi': fi_local.copy(),
                                'R_3d': R.copy(), 't_3d': t.copy(),
                                'interp_res_3d': interp_res.copy(),
                            }

                            res_norms = np.linalg.norm(residuals_3d, axis=1)
                            n_inliers = int(inliers.sum()) if inliers is not None else K3d
                            per_window_stats.append(
                                f"w{w}:Sim3(K={K3d},inl={n_inliers},s={s:.1f},"
                                f"res={np.mean(res_norms):.2f}m)")
                            continue

                    # --- Fallback: Sim2 (2D) ---
                    anchor_enu_2d = np.array([window_anchors[fi] for fi in fi_local])
                    s, R, t = fit_sim2_2d(anchor_model_xy, anchor_enu_2d)

                    if abs(s) > 1e-8:
                        enu_xy = s * (wc[:, :2] @ R.T) + t

                        approx = s * (anchor_model_xy @ R.T) + t
                        residuals = anchor_enu_2d - approx

                        interp_res = np.zeros((W_i, 2))
                        for dim in range(2):
                            interp_res[:, dim] = np.interp(
                                np.arange(W_i), fi_local, residuals[:, dim])
                        enu_xy += interp_res

                        global_centers[start:start+W_i, :2] = enu_xy
                        global_centers[start:start+W_i, 2] = wc[:, 2]

                        sim2_per_window[w] = {
                            'type': 'sim2',
                            's': s, 'R': R.copy(), 't': t.copy(),
                            'interp_res': interp_res.copy(),
                            'anchor_model_xy': anchor_model_xy.copy(),
                            'anchor_enu_xy': anchor_enu_2d.copy(),
                            'anchor_fi': fi_local.copy(),
                        }

                        res_norms = np.linalg.norm(residuals, axis=1)
                        per_window_stats.append(
                            f"w{w}:Sim2(K={K},s={s:.1f},res={np.mean(res_norms):.2f}m)")
                        continue

                # Degenerate model scale: interpolate from optimal_enu directly
                anchor_enu_2d = np.array([window_anchors[fi] for fi in fi_local])
                all_enu = np.zeros((W_i, 2))
                for dim in range(2):
                    all_enu[:, dim] = np.interp(
                        np.arange(W_i), fi_local, anchor_enu_2d[:, dim])
                global_centers[start:start+W_i, :2] = all_enu
                global_centers[start:start+W_i, 2] = wc[:, 2]
                per_window_stats.append(f"w{w}:interp(K={K},degen)")

            elif K >= 1:
                anchor_enu = np.array([window_anchors[fi] for fi in fi_local])
                all_enu = np.zeros((W_i, 2))
                for dim in range(2):
                    all_enu[:, dim] = np.interp(
                        np.arange(W_i), fi_local, anchor_enu[:, dim])
                global_centers[start:start+W_i, :2] = all_enu
                global_centers[start:start+W_i, 2] = wc[:, 2]
                per_window_stats.append(f"w{w}:interp(K={K})")

            else:
                if start > 0:
                    global_centers[start:start+W_i, :2] = global_centers[start-1, :2]
                global_centers[start:start+W_i, 2] = wc[:, 2]
                per_window_stats.append(f"w{w}:no_anchors")

        # Summary
        n_sim3 = sum(1 for s in per_window_stats if 'Sim3' in s)
        n_sim2 = sum(1 for s in per_window_stats if 'Sim2' in s)
        n_interp = sum(1 for s in per_window_stats if 'interp' in s)
        n_none = sum(1 for s in per_window_stats if 'no_anchors' in s)
        print(f"  [pose_refine] Per-window alignment: {n_sim3} Sim3, "
              f"{n_sim2} Sim2, {n_interp} interp, {n_none} no_anchors "
              f"/ {num_windows} total")
        if per_window_stats:
            print(f"  [pose_refine] Details: {', '.join(per_window_stats[:10])}"
                  + ("..." if len(per_window_stats) > 10 else ""))

        return global_centers, sim2_per_window
