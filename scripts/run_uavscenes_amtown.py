#!/usr/bin/env python3
"""Run the public GeoXel pipeline on one UAVScenes sequence."""

from __future__ import annotations

import argparse
import errno
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geoxel.geo import geodetic_to_enu, load_geospatial_maps
from geoxel.inference import load_checkpoint


def _parse_pose_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) < 7:
                continue
            try:
                lon, lat, alt = (float(value) for value in fields[1:4])
            except ValueError as exc:
                raise ValueError(f"invalid GPS values at {path}:{line_number}") from exc
            rows.append({"image": fields[0], "lon": lon, "lat": lat, "alt": alt})
    if not rows:
        raise ValueError(f"no pose rows found in {path}")
    return rows


def _parse_ttt_layers(value: str):
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("TTT layers must be comma-separated integers") from exc
    if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("TTT layers must be unique nonnegative integers")
    return layers


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_preprocessed_image(load_fn, path: Path, retries: int = 6):
    """Load one network-backed dataset image, retrying transient EAGAIN errors."""
    for attempt in range(retries):
        try:
            return load_fn([str(path)], mode="crop")
        except BlockingIOError as exc:
            if exc.errno != errno.EAGAIN or attempt + 1 >= retries:
                raise
            delay = min(8.0, 0.5 * (2 ** attempt))
            print(
                f"Transient image read EAGAIN for {path.name}; "
                f"retrying in {delay:.1f}s ({attempt + 1}/{retries - 1})",
                flush=True,
            )
            time.sleep(delay)


def _select_frames(cam_dir: Path, pose_path: Path, start_frame: int, stride: int, num_frames: int):
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"camera image directory not found: {cam_dir}")
    rows = _parse_pose_file(pose_path)
    available = {path.name for path in cam_dir.iterdir() if path.is_file()}
    rows = [row for row in rows if row["image"] in available]
    if not rows:
        raise ValueError(f"no pose-listed images found in {cam_dir}")
    if start_frame < 0 or stride < 1 or num_frames < 0:
        raise ValueError("start-frame and num-frames must be >= 0; stride must be >= 1")

    rows = rows[start_frame::stride]
    if num_frames > 0:
        rows = rows[:num_frames]
    if not rows:
        raise ValueError(
            f"no frames selected (start_frame={start_frame}, stride={stride}, cam_dir={cam_dir})"
        )
    return [cam_dir / row["image"] for row in rows], rows


def _evaluate_trajectory(trajectory: np.ndarray, selected_rows, origin):
    gps = np.asarray(
        [[row["lon"], row["lat"], row["alt"]] for row in selected_rows],
        dtype=np.float64,
    )
    predicted = np.asarray(trajectory, dtype=np.float64)
    if predicted.shape != gps.shape or not np.isfinite(predicted).all():
        raise ValueError("trajectory and selected GPS rows must be finite matching Nx3 arrays")
    gt = geodetic_to_enu(gps[:, 0], gps[:, 1], gps[:, 2], origin)
    error = predicted - predicted[0] + gt[0] - gt
    return {
        "alignment_protocol": "first_pose_translation_only_no_scale",
        "ate_rmse_m": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
        "xy_rmse_m": float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))),
        "z_rmse_m": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "last_frame_xy_err_m": float(np.linalg.norm(error[-1, :2])),
        "last_frame_z_err_m": float(error[-1, 2]),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Run GeoXel on a UAVScenes sequence")
    parser.add_argument("--uavscenes-root", type=Path, required=True,
                        help="dataset root containing images/ and poses/")
    parser.add_argument("--sequence", default="interval1_AMtown01")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="local StreamVGGT checkpoint")
    parser.add_argument("--output", type=Path, default=Path("outputs/uavscenes_amtown"))
    parser.add_argument("--dom", type=Path, default=None, help="override DOM GeoTIFF path")
    parser.add_argument("--dem", type=Path, default=None, help="override DEM GeoTIFF path")
    parser.add_argument("--building-mask", type=Path, default=None)
    parser.add_argument("--road-mask", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-frame", type=int, default=200,
                        help="zero-based row index after missing images are filtered")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=200,
                        help="0 processes all remaining frames")
    parser.add_argument(
        "--max-frames-in-memory", type=int, default=512,
        help="Refuse oversized runs before loading images; 0 disables this guard",
    )
    parser.add_argument("--segment-length", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--ttt-steps", type=int, default=15)
    parser.add_argument("--ttt-lr", type=float, default=1e-4)
    parser.add_argument("--ttt-layers", type=_parse_ttt_layers, default=_parse_ttt_layers("17,23"))
    parser.add_argument("--dom-point-quality-min-road-frame-ratio", type=float, default=0.0)
    parser.add_argument(
        "--dom-point-camera-teacher-pgo-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dem-z-direct-correct", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dem-z-road-direct-correct", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seg-pgo-w-dom-xy", type=float, default=5.0)
    parser.add_argument("--seg-pgo-w-dom-z", type=float, default=2.0)
    parser.add_argument("--seg-pgo-w-point-xy", type=float, default=0.5)
    parser.add_argument("--seg-pgo-w-point-z", type=float, default=0.1)
    parser.add_argument("--pgo-w-first-frame", type=float, default=1000.0)
    parser.add_argument("--pgo-w-overlap-dom-xy", type=float, default=5.0)
    parser.add_argument("--pgo-w-overlap-dom-z", type=float, default=1.0)
    parser.add_argument("--pgo-w-overlap-consist", type=float, default=10.0)
    parser.add_argument("--pgo-w-point-xy", type=float, default=0.5)
    parser.add_argument("--pgo-w-point-z", type=float, default=0.0)
    parser.add_argument("--pgo-w-z-dem", type=float, default=5.0)
    parser.add_argument("--pgo-w-camera-agl", type=float, default=5.0)
    parser.add_argument("--pgo-w-scale", type=float, default=50.0)
    parser.add_argument("--pgo-max-iter", type=int, default=200)
    parser.add_argument("--roma-variant", choices=("full", "tiny"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-roma", action="store_true")
    parser.add_argument("--no-post-pgo", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and frame selection only")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stride < 1 or args.start_frame < 0 or args.num_frames < 0:
        raise SystemExit("start-frame and num-frames must be >= 0; stride must be >= 1")
    if args.segment_length < 1 or not 0 <= args.overlap < args.segment_length:
        raise SystemExit("overlap must be >= 0 and smaller than segment-length")
    if args.anchor_stride < 1 or args.ttt_steps < 0 or args.ttt_lr <= 0:
        raise SystemExit("anchor-stride must be positive, ttt-steps nonnegative, and ttt-lr positive")
    if not 0.0 <= args.dom_point_quality_min_road_frame_ratio <= 1.0:
        raise SystemExit("dom-point-quality-min-road-frame-ratio must be in [0, 1]")
    if args.pgo_max_iter < 1:
        raise SystemExit("pgo-max-iter must be positive")
    if args.max_frames_in_memory < 0:
        raise SystemExit("max-frames-in-memory must be nonnegative")
    _set_seed(args.seed)

    root = args.uavscenes_root.expanduser()
    sequence_dir = root / "images" / args.sequence
    cam_dir = sequence_dir / "interval1_CAM"
    dom_dir = sequence_dir / "interval1_DOM"
    pose_path = root / "poses" / f"{args.sequence}.txt"
    sampleinfo_path = sequence_dir / "sampleinfos_interpolated.json"
    dom_path = args.dom or dom_dir / "dom.tif"
    dem_path = args.dem or dom_dir / "dem.tif"
    checkpoint = args.checkpoint.expanduser()

    for path, label in ((pose_path, "pose file"), (sampleinfo_path, "sample-info JSON"),
                        (checkpoint, "checkpoint"),
                        (dom_path, "DOM raster"), (dem_path, "DEM raster")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    image_paths, selected_rows = _select_frames(
        cam_dir, pose_path, args.start_frame, args.stride, args.num_frames
    )
    # Only the selected first-frame GPS defines the ENU origin. Later pose rows
    # are not supplied to GeoXel inference.
    origin = (selected_rows[0]["lon"], selected_rows[0]["lat"], selected_rows[0]["alt"])
    print(f"Sequence: {args.sequence}; selected frames: {len(image_paths)}")
    print(f"First selected frame: {image_paths[0].name}; ENU origin: {origin}")
    if args.max_frames_in_memory and len(image_paths) > args.max_frames_in_memory:
        raise SystemExit(
            f"selected {len(image_paths)} frames, exceeding --max-frames-in-memory "
            f"{args.max_frames_in_memory}; use --num-frames 200 (the AMtown preset) "
            "or explicitly pass --max-frames-in-memory 0 for a full-sequence run"
        )

    with sampleinfo_path.open("r", encoding="utf-8") as stream:
        sampleinfos = json.load(stream)
    if isinstance(sampleinfos, dict):
        sampleinfos = list(sampleinfos.values())
    first_info = next(
        (item for item in sampleinfos if item.get("OriginalImageName") == image_paths[0].name),
        None,
    )
    if first_info is None:
        raise ValueError(f"first selected frame is missing from {sampleinfo_path}: {image_paths[0].name}")
    from streamvggt.utils.uavscene_pose import sampleinfo_c2w_rotation

    first_rotation = sampleinfo_c2w_rotation(first_info).astype(np.float32)
    intrinsic = np.asarray(first_info["P3x3"], dtype=np.float64).reshape(3, 3)
    if not np.isfinite(intrinsic).all():
        raise ValueError(f"non-finite camera intrinsics for {image_paths[0].name}")
    building_mask = args.building_mask or dom_dir / "building_mask.tif"
    road_mask = args.road_mask or dom_dir / "road_mask.tif"
    for path, label in ((building_mask, "building mask"), (road_mask, "road mask")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.dry_run:
        print(f"Last selected frame: {image_paths[-1].name}")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu for a CPU run")

    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images

    frames = []
    for path in image_paths:
        image = _load_preprocessed_image(load_and_preprocess_images, path)
        frames.append({"img": image})

    maps = load_geospatial_maps(
        dom_path,
        dem_path,
        origin,
        building_mask_path=building_mask,
        road_mask_path=road_mask,
    )
    model = StreamVGGT()
    checkpoint_info = load_checkpoint(model, checkpoint, device, strict=True)
    from streamvggt.geo_v3 import (
        GeoV3Config, PGOConfig, assemble_trajectory, refine_transforms_with_pgo,
    )
    from streamvggt.utils.dom_matching import load_roma

    cfg = GeoV3Config(
        segment_length=args.segment_length,
        overlap=args.overlap,
        anchor_stride=args.anchor_stride,
        ttt_adaptation_type="token",
        geo_target_mode="dom_points",
        geo_ttt_v2_mode="token",
        geo_ttt_v2_steps=args.ttt_steps,
        geo_ttt_v2_kv_lr=args.ttt_lr,
        geo_ttt_v2_reg_weight=0.01,
        geo_ttt_v2_loss="mse",
        geo_consist_stride=4,
        geo_consist_max_corr=512,
        dense_reproj_weight=0.5,
        dense_reproj_z_weight=0.0,
        cam_token_weight=10.0,
        max_grad_norm=1.0,
        prefer_pnp_camera_targets=False,
        allow_raw_pnp_camera_targets=False,
        dem_z_direct_correct_enable=args.dem_z_direct_correct,
        dem_z_camera_correct_enable=False,
        dem_z_camera_pgo_edges_enable=True,
        dem_z_road_direct_correct_enable=args.dem_z_road_direct_correct,
        seg_pgo_enable=True,
        seg_pgo_w_dom_xy=args.seg_pgo_w_dom_xy,
        seg_pgo_w_dom_z=args.seg_pgo_w_dom_z,
        seg_pgo_w_consist=8.0,
        seg_pgo_use_overlap_consistency=True,
        seg_pgo_w_delta_reg=0.5,
        seg_pgo_use_point_edges=True,
        seg_pgo_w_point_xy=args.seg_pgo_w_point_xy,
        seg_pgo_w_point_z=args.seg_pgo_w_point_z,
        seg_pgo_max_iter=50,
        seg_pgo_delta_clip=5.0,
        dom_point_edges_enable=True,
        dom_point_edges_road_only=True,
        dom_point_edges_max=512,
        dom_point_quality_min_road_frame_ratio=args.dom_point_quality_min_road_frame_ratio,
        dom_point_camera_teacher_pgo_enable=args.dom_point_camera_teacher_pgo_enable,
        dom_point_z_nonbuilding_fallback_enable=True,
        dom_point_bootstrap_enable=True,
        dom_point_prefer_point_bootstrap_transform=True,
        dom_point_bootstrap_max_points=4096,
        dom_point_bootstrap_anchor_align=True,
        dom_point_bootstrap_camera_recenter_enable=True,
        dom_point_bootstrap_camera_recenter_max_pitch_deg=8.0,
        dom_point_bootstrap_camera_recenter_xy_weight=1.0,
        dom_point_bootstrap_camera_recenter_max_apply_xy_m=80.0,
        dom_point_bootstrap_agl_z_recenter_enable=True,
        dom_point_bootstrap_agl_z_weight=0.5,
        dom_point_ttt_enable=True,
        dom_point_ttt_pose_teacher_source="transform",
        dom_point_ttt_road_only=True,
        dom_point_camera_teacher_enable=True,
        dom_point_camera_teacher_source="frame_center",
        dom_point_ttt_camera_teacher_source="point_rays",
        ttt_layers=args.ttt_layers,
    )
    paths = [str(path) for path in image_paths]
    camera_fov = {
        path: {
            "fx": float(intrinsic[0, 0]), "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]), "cy": float(intrinsic[1, 2]),
        }
        for path in paths
    }
    args.output.mkdir(parents=True, exist_ok=True)
    roma_model = (
        load_roma(device=str(device), variant=args.roma_variant)
        if not args.disable_roma else None
    )
    result = model.inference_with_geo_v3(
        frames=[{**frame, "img": frame["img"].to(device)} for frame in frames],
        dom_image=maps.dom_image,
        project_fn=maps.project_fn,
        anchor_world_xyz=torch.zeros(3, device=device, dtype=torch.float32),
        get_world_xyz_fn=None,
        image_paths=paths,
        inv_project_fn=maps.inv_project_fn,
        geo_elev=maps.geo_elev,
        dom_transform=maps.dom_transform,
        roma_model=roma_model,
        save_vis_dir=str(args.output / "matches"),
        geo_v3_config=cfg,
        geo_v3_metadata={
            "dataset": "uavscene",
            "airzoo_intrinsics": {},
            "airzoo_default_intrinsic": intrinsic,
            "camera_fov_by_path": camera_fov,
        },
        heading_fn=None,
        gt_first_frame_rotation=first_rotation,
        dtype=(
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16 if device.type == "cuda" else torch.float32
        ),
    )

    trajectory = np.asarray(result.geo_v3_trajectory, dtype=np.float32)
    pgo_stats = None
    if not args.no_post_pgo:
        pgo_cfg = PGOConfig(
            w_first_frame=args.pgo_w_first_frame,
            w_overlap_dom_xy=args.pgo_w_overlap_dom_xy,
            w_overlap_dom_z=args.pgo_w_overlap_dom_z,
            w_overlap_consist=args.pgo_w_overlap_consist,
            w_point_xy=args.pgo_w_point_xy,
            w_point_z=args.pgo_w_point_z,
            w_z_dem=args.pgo_w_z_dem,
            w_camera_agl=args.pgo_w_camera_agl,
            w_scale=args.pgo_w_scale,
            max_iter=args.pgo_max_iter,
            verbose=False,
        )
        transforms, pgo_stats = refine_transforms_with_pgo(
            result.geo_v3_chain, origin_enu=np.zeros(3, dtype=np.float64), cfg=pgo_cfg
        )
        trajectory = np.asarray(
            assemble_trajectory(result.geo_v3_chain, transforms, len(image_paths)),
            dtype=np.float32,
        )
    if trajectory.shape != (len(image_paths), 3):
        raise RuntimeError(
            f"GeoXel returned trajectory shape {trajectory.shape}; expected ({len(image_paths)}, 3)"
        )
    evaluation = _evaluate_trajectory(trajectory, selected_rows, origin)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "trajectory_enu.npy", trajectory)
    metadata = {
        "dataset": "UAVScenes",
        "sequence": args.sequence,
        "num_frames": len(image_paths),
        "frame_names": [path.name for path in image_paths],
        "origin_lon_lat_alt": list(origin),
        "start_frame": args.start_frame,
        "stride": args.stride,
        "requested_num_frames": args.num_frames,
        "checkpoint": checkpoint.name,
        "checkpoint_load": checkpoint_info,
        "segment_length": args.segment_length,
        "overlap": args.overlap,
        "anchor_stride": args.anchor_stride,
        "ttt_steps": args.ttt_steps,
        "ttt_lr": args.ttt_lr,
        "ttt_layers": args.ttt_layers,
        "dom_point_quality_min_road_frame_ratio": (
            args.dom_point_quality_min_road_frame_ratio
        ),
        "dom_point_camera_teacher_pgo_enable": args.dom_point_camera_teacher_pgo_enable,
        "dem_z_direct_correct_enable": args.dem_z_direct_correct,
        "dem_z_road_direct_correct_enable": args.dem_z_road_direct_correct,
        "seg_pgo_w_dom_xy": args.seg_pgo_w_dom_xy,
        "seg_pgo_w_dom_z": args.seg_pgo_w_dom_z,
        "seg_pgo_w_point_xy": args.seg_pgo_w_point_xy,
        "seg_pgo_w_point_z": args.seg_pgo_w_point_z,
        "pgo_weights": {
            "first_frame": args.pgo_w_first_frame,
            "overlap_dom_xy": args.pgo_w_overlap_dom_xy,
            "overlap_dom_z": args.pgo_w_overlap_dom_z,
            "overlap_consist": args.pgo_w_overlap_consist,
            "point_xy": args.pgo_w_point_xy,
            "point_z": args.pgo_w_point_z,
            "dem_z": args.pgo_w_z_dem,
            "camera_agl": args.pgo_w_camera_agl,
            "scale": args.pgo_w_scale,
            "max_iter": args.pgo_max_iter,
        },
        "roma": args.roma_variant if not args.disable_roma else None,
        "post_pgo": not args.no_post_pgo,
        "seed": args.seed,
        "pgo_stats": pgo_stats,
        "evaluation": evaluation,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(
            metadata, indent=2,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray)
            else value.item() if isinstance(value, np.generic) else str(value),
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(trajectory)} poses to {args.output / 'trajectory_enu.npy'}")
    print(
        f"[EvalProtocol] first-pose translation only, no scale | "
        f"ATE={evaluation['ate_rmse_m']:.3f} m  "
        f"XY={evaluation['xy_rmse_m']:.3f} m  "
        f"Z={evaluation['z_rmse_m']:.3f} m  "
        f"last XY={evaluation['last_frame_xy_err_m']:.3f} m",
        flush=True,
    )


if __name__ == "__main__":
    main()
