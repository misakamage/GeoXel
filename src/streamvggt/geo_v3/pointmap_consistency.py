"""Opt-in camera/point token consistency. No PnP and no weight adaptation.

The map Sim3 is fixed throughout this diagnostic. Only cached patch tokens and
the final camera token are optimized. Intrinsics come from dataset metadata,
not the camera head's predicted field of view; provenance is recorded, not
treated as independent proof that the dataset calibration is correct.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri


def crop_pixel_transform(original_hw, model_hw):
    """Pixel-center affine for load_fn's single-image, width-518 crop mode."""
    h, w = (int(x) for x in original_hw)
    if min(h, w) <= 0:
        raise ValueError("Invalid original image dimensions")
    resized_h = round(h * (518 / w) / 14) * 14
    expected = (min(resized_h, 518), 518)
    if tuple(model_hw) != expected:
        raise ValueError(f"Unsupported preprocessing: expected {expected}, got {tuple(model_hw)}")
    sx, sy = 518 / w, resized_h / h
    crop_top = max(0, (resized_h - 518) // 2)
    return np.array([[sx, 0, (sx - 1) / 2],
                     [0, sy, (sy - 1) / 2 - crop_top], [0, 0, 1]], dtype=np.float64)


def calibrated_frame_geometry(segment_input, fi, frame):
    path = str(segment_input.image_paths[segment_input.start_frame + fi])
    with Image.open(path) as image:
        w, h = image.size
    metadata = segment_input.metadata or {}
    by_path = metadata.get("camera_fov_by_path") or {}
    fov = by_path.get(path, by_path.get(Path(path).name))
    if fov is not None:
        if int(fov.get("orig_w", w)) != w or int(fov.get("orig_h", h)) != h:
            raise ValueError("Camera calibration resolution does not match the query image")
        intrinsic = np.array([[fov["fx"], 0, fov["cx"]],
                              [0, fov["fy"], fov["cy"]], [0, 0, 1]], dtype=float)
        source = "camera_fov_by_path"
    else:
        intrinsic = (metadata.get("airzoo_intrinsics") or {}).get((h, w))
        source = f"airzoo_intrinsics[{h},{w}]"
    if intrinsic is None:
        raise ValueError(f"No explicit calibration for {path} ({w}x{h}); refusing focal-length fallback")
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1])):
        raise ValueError("Invalid calibration matrix")
    affine = crop_pixel_transform((h, w), frame["img"].shape[-2:])
    return affine @ intrinsic, affine, {
        "source": source, "original_hw": [h, w],
        "model_hw": list(frame["img"].shape[-2:]),
        "K_original": intrinsic.tolist(), "K_model": (affine @ intrinsic).tolist(),
        "pixel_affine": affine.tolist(),
    }


def prepare_token_leaves(all_tokens, layers, device):
    """Unlike the legacy helper, never alias cached originals in this branch."""
    layers = sorted(set(layers) | {len(all_tokens[0]) - 1})
    per_frame, leaves, originals = [], [], []
    for tokens in all_tokens:
        frame_leaves = {}
        for layer in layers:
            if tokens[layer] is None:
                raise ValueError(f"Required PointMap token layer {layer} was not cached")
            original = tokens[layer].detach().to(device=device, dtype=torch.float32)
            leaf = original.clone().requires_grad_(True)
            frame_leaves[layer] = leaf
            leaves.append(leaf)
            originals.append(original)
        per_frame.append(frame_leaves)
    return per_frame, leaves, originals


@contextmanager
def frozen_weights(model):
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def sample_points(points, pixels, image_hw):
    """Bilinear sampling at model-input pixels, including noninteger pixels."""
    while points.ndim > 3:
        points = points[0]
    h, w = image_hw
    grid = torch.stack((2 * pixels[:, 0] / (w - 1) - 1,
                        2 * pixels[:, 1] / (h - 1) - 1), dim=-1)
    return F.grid_sample(points.permute(2, 0, 1)[None], grid[None, None],
                         align_corners=True)[0, :, 0].T


def project_points(pose, points, intrinsic):
    extrinsic, _ = pose_encoding_to_extri_intri(pose.reshape(1, 1, -1), build_intrinsics=False)
    extrinsic = extrinsic[0, 0]
    camera = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    homogeneous = camera @ intrinsic.T
    pixels = homogeneous[:, :2] / camera[:, 2:3].clamp_min(1e-5)
    center = -(extrinsic[:, :3].T @ extrinsic[:, 3])
    return pixels, camera[:, 2], center, extrinsic[:, :3]


def _huber(error, delta):
    return F.huber_loss(error, torch.zeros_like(error), delta=delta)


def run_joint_ttt(*, model, frames, per_frame_all_tokens, per_frame_leaf_tokens,
                  transform, optimizer, num_steps, psi, segment_input,
                  point_teacher_cache, all_leaves, all_originals, config,
                  overlap_target_cache=None, **unused):
    # Import lazily: defaults dispatches here only for the explicit joint mode.
    from .defaults import (_build_agg_tokens, _build_frozen_camera_kv_full,
                           _slice_camera_kv_prefix, _forward_pose_and_pts,
                           _forward_pose_cached)

    diag = {"mode": "joint", "accepted": False, "steps": 0,
            "camera_lr": config.pointmap_camera_lr,
            "patch_lr": optimizer.param_groups[0]["lr"],
            "reprojection_weight": config.pointmap_reprojection_weight,
            "holdout_scope": "TTT-only; these correspondences may have participated in map registration"}
    result = {"ttt_losses": {"total_loss": 0.0}, "pointmap_consistency": diag}
    if not point_teacher_cache:
        diag["reason"] = "no_map_teacher; skipped_unanchored_adaptation"
        return result

    device = frames[0]["img"].device
    last_layer = len(per_frame_all_tokens[0]) - 1
    cameras = [tokens[last_layer][..., :1, :].detach().float().clone().requires_grad_(True)
               for tokens in per_frame_all_tokens]
    camera_optimizer = torch.optim.Adam(cameras, lr=config.pointmap_camera_lr)
    n = len(frames)
    calibration, grid, teachers = {}, {}, {}
    for fi, frame in enumerate(frames):
        intrinsic, affine, provenance = calibrated_frame_geometry(segment_input, fi, frame)
        calibration[fi] = torch.as_tensor(intrinsic, device=device, dtype=torch.float32)
        diag.setdefault("calibration", {})[str(fi)] = provenance
        h, w = frame["img"].shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(8, h - 8, 32, device=device),
                                torch.arange(8, w - 8, 32, device=device), indexing="ij")
        grid[fi] = torch.stack([xx.flatten(), yy.flatten()], -1).float()
        teacher = point_teacher_cache.get(fi)
        if teacher is None:
            continue
        raw = np.asarray(teacher["pixels"])
        if (teacher["query_W"], teacher["query_H"]) != tuple(provenance["original_hw"][::-1]):
            raise ValueError("Map correspondences and calibration use different source image sizes")
        pixels = raw @ affine[:2, :2].T + affine[:2, 2]
        pixels = torch.as_tensor(pixels, device=device, dtype=torch.float32)
        target = teacher["target_enu"].detach().to(device=device, dtype=torch.float32)
        valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] <= w - 1)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] <= h - 1)
                 & torch.isfinite(target).all(dim=-1))
        pixels, target = pixels[valid], target[valid]
        if len(pixels) >= 5:
            held = torch.arange(len(pixels), device=device) % 5 == 0
            teachers[fi] = (pixels, target, held)
    if not teachers:
        diag["reason"] = "no_valid_map_pixels"
        return result

    frozen_kv = _build_frozen_camera_kv_full(model, per_frame_all_tokens)

    def tokens_for(fi):
        tokens = _build_agg_tokens(per_frame_all_tokens, per_frame_leaf_tokens, fi)
        tokens[last_layer] = torch.cat([cameras[fi], tokens[last_layer][..., 1:, :]], dim=-2)
        return tokens

    def prefix(fi):
        return _slice_camera_kv_prefix(frozen_kv, fi, n)

    def camera_state(fi):
        pose = _forward_pose_cached(model, tokens_for(fi), prefix(fi))
        _, _, center, rotation = project_points(pose, grid[fi].new_zeros((1, 3)), calibration[fi])
        return center, rotation

    overlap_ids = sorted((overlap_target_cache or {}).keys())
    overlap_ids = [fi for fi in overlap_ids if 0 <= fi < n]
    overlap_relative = None
    if len(overlap_ids) > 1:
        targets = torch.stack([overlap_target_cache[fi].detach().float() for fi in overlap_ids])
        overlap_relative = targets - targets.mean(0)
    scale = transform.s.detach().float()
    original_by_id = {id(leaf): original for leaf, original in zip(all_leaves, all_originals)}
    initial_cameras = [camera.detach().clone() for camera in cameras]

    def restore():
        with torch.no_grad():
            for leaf, original in zip(all_leaves, all_originals):
                leaf.copy_(original)
            for camera, original in zip(cameras, initial_cameras):
                camera.copy_(original)

    with frozen_weights(model):
        try:
            with torch.no_grad():
                state = [camera_state(fi) for fi in range(n)]
                initial_centers = torch.stack([s[0] for s in state])
                initial_rotations = torch.stack([s[1] for s in state])

            def evaluate(backward=False):
                # Detached neighbor states avoid retaining a segment-sized DPT graph.
                with torch.no_grad():
                    centers = torch.stack([camera_state(fi)[0] for fi in range(n)])
                overlap_mean = centers[overlap_ids].mean(0) if overlap_ids else None
                sums = dict(total=0.0, map=0.0, reprojection=0.0, relative_motion=0.0)
                pixel_errors, held_errors, train_errors = [], [], []
                front_counts = [0, 0]
                for fi, frame in enumerate(frames):
                    pose, points, _ = _forward_pose_and_pts(model, frame, tokens_for(fi), psi, prefix(fi))
                    samples = sample_points(points, grid[fi], frame["img"].shape[-2:])
                    pixels, depths, center, rotation = project_points(pose, samples, calibration[fi])
                    if not torch.isfinite(pixels).all() or not torch.isfinite(center).all():
                        raise FloatingPointError("Nonfinite camera/point output")
                    front = depths > 1e-4
                    front_counts[0] += int(front.sum())
                    front_counts[1] += len(front)
                    if int(front.sum()) < len(front) * 0.9:
                        raise FloatingPointError("More than 10% of projection samples are behind camera")
                    reproj = _huber(pixels[front] - grid[fi][front], 5.0)
                    pixel_errors.append(torch.linalg.vector_norm(pixels[front] - grid[fi][front], dim=-1).detach().cpu())
                    map_loss = points.new_zeros(())
                    if fi in teachers:
                        px, target, held = teachers[fi]
                        mapped = transform.model_to_world(sample_points(points, px, frame["img"].shape[-2:]))
                        error = mapped - target
                        map_loss = _huber(error[~held], 2.0)
                        held_errors.append(error[held].detach().cpu())
                        train_errors.append(error[~held].detach().cpu())
                    motion = center.new_zeros(())
                    for neighbor in (fi - 1, fi + 1):
                        if 0 <= neighbor < n:
                            delta = center - centers[neighbor] - (initial_centers[fi] - initial_centers[neighbor])
                            motion = motion + 0.5 * _huber(scale * delta, 2.0)
                    if overlap_relative is not None and fi in overlap_ids:
                        error = center - overlap_mean - overlap_relative[overlap_ids.index(fi)]
                        motion = motion + _huber(scale * error, 2.0)
                    regularization = sum(
                        ((leaf - original_by_id[id(leaf)]).float().square().mean()
                         / original_by_id[id(leaf)].float().square().mean().clamp_min(1e-6))
                        for leaf in per_frame_leaf_tokens[fi].values())
                    regularization = regularization + (cameras[fi] - initial_cameras[fi]).square().mean()
                    loss = (map_loss / len(teachers)
                            + (config.pointmap_reprojection_weight * reproj + 0.01 * motion
                               + 0.01 * (rotation - initial_rotations[fi]).square().mean()
                               + 0.001 * regularization) / n)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite PointMap consistency loss")
                    if backward:
                        loss.backward()
                    sums["total"] += float(loss.detach())
                    sums["map"] += float(map_loss.detach()) / len(teachers)
                    sums["reprojection"] += float(reproj.detach()) / n
                    sums["relative_motion"] += float(motion.detach()) / n
                errors = torch.cat(held_errors)
                training = torch.cat(train_errors)
                sums.update(reprojection_median_px=float(torch.cat(pixel_errors).median()),
                            front_fraction=front_counts[0] / front_counts[1],
                            ttt_holdout_xy_median_m=float(torch.linalg.vector_norm(errors[:, :2], dim=-1).median()),
                            ttt_holdout_z_median_m=float(errors[:, 2].abs().median()),
                            train_xy_median_m=float(torch.linalg.vector_norm(training[:, :2], dim=-1).median()),
                            camera_centers_model=centers.cpu().tolist())
                return sums

            with torch.no_grad():
                diag["before"] = evaluate()
            history = []
            for step in range(num_steps):
                optimizer.zero_grad(set_to_none=True)
                camera_optimizer.zero_grad(set_to_none=True)
                metrics = evaluate(backward=True)
                params = all_leaves + cameras
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError("Nonfinite PointMap token gradient")
                if step == 0:
                    diag["camera_gradient_norm"] = float(sum(p.grad.float().square().sum() for p in cameras if p.grad is not None).sqrt())
                    diag["patch_gradient_norm"] = float(sum(p.grad[..., psi:, :].float().square().sum() for p in all_leaves if p.grad is not None).sqrt())
                optimizer.step()
                camera_optimizer.step()
                history.append({key: metrics[key] for key in ("total", "map", "reprojection", "relative_motion")})
                diag["steps"] = step + 1
                print(f"[PointMapJoint] seg={segment_input.segment_id} step={step+1}/{num_steps} "
                      f"loss={metrics['total']:.4f} reproj={metrics['reprojection_median_px']:.2f}px", flush=True)
            with torch.no_grad():
                diag["candidate"] = evaluate()
            diag["history"] = history
            # Numerical/descent guard only; held-out points are not used for model selection.
            diag["accepted"] = diag["candidate"]["total"] < diag["before"]["total"]
            diag["reason"] = "objective_decreased" if diag["accepted"] else "objective_not_decreased; rolled_back"
            if not diag["accepted"]:
                restore()
            with torch.no_grad():
                for fi, camera in enumerate(cameras):
                    per_frame_leaf_tokens[fi][last_layer][..., :1, :].copy_(camera)
            diag["after"] = diag["candidate"] if diag["accepted"] else diag["before"]
            result["ttt_losses"]["total_loss"] = diag["after"]["total"]
        except FloatingPointError as exc:
            restore()
            diag["reason"] = f"{exc}; rolled_back"
            diag["after"] = diag.get("before")
        except BaseException:
            restore()
            raise
        finally:
            optimizer.zero_grad(set_to_none=True)
            camera_optimizer.zero_grad(set_to_none=True)
    return result
