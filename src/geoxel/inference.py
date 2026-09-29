"""Small, path-independent helpers for public GeoXel inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .geo import GeoXelMap, load_geospatial_maps


def load_image_sequence(image_dir: str | Path, *, stride: int = 1, max_frames: int = 0, mode: str = "crop"):
    """Load an image directory as the frame dictionaries expected by StreamVGGT."""
    from streamvggt.utils.load_fn import load_and_preprocess_images

    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory not found: {root}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in extensions)
    paths = paths[::max(1, int(stride))]
    if max_frames > 0:
        paths = paths[: int(max_frames)]
    if not paths:
        raise ValueError(f"no supported images found in {root}")
    frames = []
    for path in paths:
        image = load_and_preprocess_images([str(path)], mode=mode)[0]
        frames.append({"img": image})
    return frames, [str(p) for p in paths]


def load_checkpoint(model, checkpoint: str | Path, device: torch.device, *, strict: bool = False):
    """Load a local StreamVGGT checkpoint with common wrapper formats."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint object: {type(state).__name__}")
    state = {str(k).removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    model.to(device).eval()
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def run_geoxel(
    model,
    frames,
    image_paths,
    maps: GeoXelMap,
    *,
    device: str | torch.device = "cuda",
    anchor_enu=(0.0, 0.0, 0.0),
    segment_length: int = 32,
    overlap: int = 8,
    anchor_stride: int = 4,
    ttt_steps: int = 15,
    ttt_lr: float = 1e-4,
    use_roma: bool = True,
    save_vis_dir: Optional[str | Path] = None,
):
    """Run the paper's GeoXel segment/TTT/PGO pipeline."""
    from streamvggt.geo_v3 import GeoV3Config
    from streamvggt.utils.dom_matching import load_roma
    device = torch.device(device)
    model = model.to(device).eval()
    frames = [{**frame, "img": frame["img"].to(device)} for frame in frames]
    anchor = torch.tensor(anchor_enu, dtype=torch.float32, device=device)
    roma_model = load_roma(device=str(device)) if use_roma else None

    cfg = GeoV3Config(
        segment_length=int(segment_length),
        overlap=int(overlap),
        anchor_stride=int(anchor_stride),
        geo_ttt_v2_steps=int(ttt_steps),
        geo_ttt_v2_kv_lr=float(ttt_lr),
        ttt_layers=[17, 23],
        geo_target_mode="dom_points",
    )
    output = model.inference_with_geo_v3(
        frames=frames,
        dom_image=maps.dom_image,
        project_fn=maps.project_fn,
        anchor_world_xyz=anchor,
        # Keep the core runner's established fallback for converting pose
        # encodings to world coordinates.
        get_world_xyz_fn=None,
        image_paths=image_paths,
        inv_project_fn=maps.inv_project_fn,
        geo_elev=maps.geo_elev,
        dom_transform=maps.dom_transform,
        roma_model=roma_model,
        save_vis_dir=str(save_vis_dir) if save_vis_dir else None,
        geo_v3_config=cfg,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    return output
