# GeoXel

GeoXel is a geo-referenced 3D reconstruction pipeline for long-duration aerial
video. It combines the StreamVGGT visual geometry model with Digital
Orthophoto Maps (DOM), Digital Elevation Models (DEM), optional semantic masks,
test-time token adaptation, and pose-graph optimization (PGO).

This repository is the public source release. Checkpoints, benchmark data,
DOM/DEM rasters, RoMa weights, and generated results are intentionally excluded
and must be obtained separately under their own licenses.

## Contents

```text
src/streamvggt/       StreamVGGT backbone and GeoXel geometry pipeline
src/geoxel/            Public map I/O, checkpoint helpers, and Python API
scripts/run_geoxel.py  Generic image-directory command-line entry point
scripts/run_uavscenes_amtown.py
                       UAVScenes AMtown evaluation entry point
tests/                 Public helper and coordinate-conversion tests
```

## Requirements

- Linux is recommended.
- Python 3.10 or newer.
- A CUDA-compatible PyTorch and torchvision installation for GPU inference.
  PyTorch 2.4 or newer is recommended for current Transformers/RoMa releases.
- A local StreamVGGT checkpoint.
- `rasterio`, `pyproj`, and `romatch` for DOM/DEM registration and RoMa
  matching.

Install PyTorch for the target CUDA version first, then install GeoXel:

```bash
python -m pip install -e '.[maps]'
```

For the complete optional tool set:

```bash
python -m pip install -e '.[full]'
```

Install `pytest` separately to verify the public helpers:

```bash
PYTHONPATH=src python -m pytest -q
```

## Generic Usage

The generic CLI expects an image directory, a DOM GeoTIFF, a DEM GeoTIFF, and a
WGS-84 origin for the local ENU coordinate frame.

```bash
PYTHONPATH=src python scripts/run_geoxel.py \
  --images /path/to/frames \
  --dom /path/to/dom.tif \
  --dem /path/to/dem.tif \
  --checkpoint /path/to/streamvggt.pth \
  --origin-lon-lat-alt 121.4737,31.2304,5.0 \
  --anchor-enu 0,0,0 \
  --device cuda:0 \
  --output outputs/example
```

The command writes `trajectory_enu.npy` and `metadata.json`. Match
visualizations, when enabled, are written under `outputs/example/matches`.
Use `--disable-roma` for a smoke test without RoMa map matching.

The generic entry point uses 32-frame segments, 8-frame overlap, 15 token-TTT
steps, and a token learning rate of `1e-4` by default. Use `--max-frames` for a
small smoke run.

## UAVScenes: AMtown

The AMtown script reads the standard UAVScenes layout:

```text
UAVSCENES_ROOT/
|-- images/interval1_AMtown01/interval1_CAM/
|-- images/interval1_AMtown01/interval1_DOM/{dom,dem,building_mask,road_mask}.tif
|-- images/interval1_AMtown01/sampleinfos_interpolated.json
`-- poses/interval1_AMtown01.txt
```

To reproduce the 200-frame configuration used for debugging:

```bash
cd /path/to/geoxel
PYTHONPATH=src python -u scripts/run_uavscenes_amtown.py \
  --uavscenes-root /path/to/uavscene \
  --sequence interval1_AMtown01 \
  --checkpoint /path/to/checkpoints.pth \
  --device cuda:0 \
  --start-frame 200 --stride 5 --num-frames 200 \
  --segment-length 32 --overlap 8 --anchor-stride 1 \
  --ttt-layers 17,23 --ttt-steps 15 --ttt-lr 1e-4 \
  --seed 0 \
  --output outputs/uavscenes_amtown_200
```

To process every available frame after frame 200 at stride 5, set
`--num-frames 0` and disable the in-memory guard:

```bash
PYTHONPATH=src python -u scripts/run_uavscenes_amtown.py \
  --uavscenes-root /path/to/uavscene \
  --sequence interval1_AMtown01 \
  --checkpoint /path/to/checkpoints.pth \
  --device cuda:0 \
  --start-frame 200 --stride 5 --num-frames 0 \
  --max-frames-in-memory 0 \
  --segment-length 32 --overlap 8 --anchor-stride 1 \
  --ttt-layers 17,23 --ttt-steps 15 --ttt-lr 1e-4 \
  --seed 0 \
  --output outputs/uavscenes_amtown_from200_stride5
```

This command selects 2,549 frames in the current AMtown dataset. The script
loads all selected images before inference; `--max-frames-in-memory 0` only
disables the frame-count guard and does not reduce memory use. Use `--dry-run`
to check paths and frame selection without loading the model.

The shell wrapper exposes the same options through environment variables:

```bash
UAVSCENES_ROOT=/path/to/uavscene \
CHECKPOINT=/path/to/checkpoints.pth \
NUM_FRAMES=0 MAX_FRAMES_IN_MEMORY=0 \
bash scripts/run_uavscenes_amtown.sh
```

The script prints the selected frame count and, after inference, the evaluation
metrics directly to the terminal. It also saves them in `metadata.json`:

```text
[EvalProtocol] first-pose translation only, no scale | ATE=... m XY=... m Z=... m last XY=... m
```

The evaluation protocol translates the prediction so its first position matches
the first GPS position. It does not fit a global scale or rotation. `ATE` is the
3D RMSE, while `XY` and `Z` report horizontal and vertical RMSE separately.
Only the first selected GPS position anchors inference; later GPS positions
are used to compute evaluation metrics after inference.

## Outputs

Each run writes:

- `trajectory_enu.npy`: predicted camera centers in the local ENU frame;
- `metadata.json`: frame selection, checkpoint loading, configuration, PGO
  statistics, and (for UAVScenes) terminal evaluation metrics;
- `matches/`: optional DOM matching visualizations and diagnostics.

Do not commit checkpoints, rasters, datasets, or generated results. The
repository `.gitignore` contains patterns for the common cases.

## Data and Reproducibility

GeoXel was evaluated on UAVScenes, SynthCity-6, and UAVD4L-2yr. Download these
datasets and any DOM, DEM, or OSM layers from their official sources and follow
their licenses. Record the dataset version, map CRS, ENU origin, frame stride,
segment settings, checkpoint, and random seed alongside any reported result.

## Citation

If GeoXel contributes to published work, cite the accompanying GeoXel paper
and the upstream StreamVGGT work. A machine-readable citation stub is provided
in [`CITATION.cff`](CITATION.cff).

## License and Notices

GeoXel source is distributed with the repository license. The copied
StreamVGGT implementation remains subject to
[`STREAMVGGT-LICENSE.txt`](STREAMVGGT-LICENSE.txt). See [`NOTICE.md`](NOTICE.md)
for attribution and excluded third-party assets.
