"""Command-line entry point for a public GeoXel run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .geo import load_geospatial_maps
from .inference import load_checkpoint, load_image_sequence, run_geoxel


def _triple(text: str):
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return tuple(values)


def build_parser():
    p = argparse.ArgumentParser(description="GeoXel geo-referenced reconstruction")
    p.add_argument("--images", required=True, help="directory containing ordered video frames")
    p.add_argument("--dom", required=True, help="DOM GeoTIFF")
    p.add_argument("--dem", required=True, help="DEM GeoTIFF")
    p.add_argument("--checkpoint", required=True, help="local StreamVGGT checkpoint")
    p.add_argument("--origin-lon-lat-alt", type=_triple, required=True, metavar="LON,LAT,ALT")
    p.add_argument("--anchor-enu", type=_triple, default=(0.0, 0.0, 0.0), metavar="E,N,U")
    p.add_argument("--building-mask", default=None)
    p.add_argument("--road-mask", default=None)
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--segment-length", type=int, default=32)
    p.add_argument("--overlap", type=int, default=8)
    p.add_argument("--anchor-stride", type=int, default=4)
    p.add_argument("--ttt-steps", type=int, default=15)
    p.add_argument("--ttt-lr", type=float, default=1e-4)
    p.add_argument("--disable-roma", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    from streamvggt.models.streamvggt import StreamVGGT

    frames, paths = load_image_sequence(args.images, stride=args.stride, max_frames=args.max_frames)
    maps = load_geospatial_maps(
        args.dom, args.dem, args.origin_lon_lat_alt,
        building_mask_path=args.building_mask,
        road_mask_path=args.road_mask,
    )
    model = StreamVGGT()
    checkpoint_info = load_checkpoint(model, args.checkpoint, device)
    result = run_geoxel(
        model, frames, paths, maps, device=device, anchor_enu=args.anchor_enu,
        segment_length=args.segment_length, overlap=args.overlap,
        anchor_stride=args.anchor_stride, ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr,
        use_roma=not args.disable_roma, save_vis_dir=out / "matches",
    )
    trajectory = np.asarray(result.geo_v3_trajectory, dtype=np.float32)
    np.save(out / "trajectory_enu.npy", trajectory)
    metadata = {
        "num_frames": int(len(trajectory)),
        "checkpoint": str(Path(args.checkpoint).name),
        "checkpoint_load": checkpoint_info,
        "origin_lon_lat_alt": list(args.origin_lon_lat_alt),
        "anchor_enu": list(args.anchor_enu),
        "segment_length": args.segment_length,
        "overlap": args.overlap,
        "anchor_stride": args.anchor_stride,
        "ttt_steps": args.ttt_steps,
        "ttt_lr": args.ttt_lr,
        "roma": not args.disable_roma,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(trajectory)} poses to {out / 'trajectory_enu.npy'}")


if __name__ == "__main__":
    main()
