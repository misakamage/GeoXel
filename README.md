# GeoXel

**Geographic Per-pixel 3D Mapping from Aerial Video in the Wild**

[Project Page](https://nudt-sawlab.github.io/geoxel/) · [Code](https://github.com/nudt-sawlab/geoxel)

![GeoXel overview and AMtown trajectory comparison](assets/teaser.png)

GeoXel reconstructs geographically aligned camera trajectories and 3D geometry
from long aerial videos. It combines a StreamVGGT backbone with orthophoto and
elevation maps, test-time token adaptation, and pose-graph optimization.

This repository contains the public implementation accompanying the GeoXel
paper, *GeoXel: Geographic Per-pixel 3D Mapping from Aerial Video in the Wild*.

## Overview



The public UAVScenes entry point is
[`scripts/run_uavscenes_amtown.sh`](scripts/run_uavscenes_amtown.sh). It runs
the AMtown configuration with 32-frame segments, 8-frame overlap, RoMa map
matching, token adaptation on layers 17 and 23, and pose-graph refinement.
The Python entry point accepts additional configuration options.

## Repository Structure

```text
assets/                         Teaser and pipeline figures
docs/                           GitHub Pages project website
src/streamvggt/                StreamVGGT backbone and GeoXel pipeline
src/geoxel/                    Map I/O and public Python helpers
scripts/run_uavscenes_amtown.sh  UAVScenes AMtown entry point
scripts/run_uavscenes_amtown.py  UAVScenes runner and evaluation
scripts/run_geoxel.py           Image-directory command-line entry point
tests/                          Public helper tests
```

## Environment

Linux, Python 3.10 or newer, and a CUDA GPU are recommended. Install a
CUDA-compatible PyTorch and torchvision pair first. PyTorch 2.4 or newer is
recommended for current Transformers and RoMa releases.

```bash
conda create -n geoxel python=3.11 -y
conda activate geoxel
# Install PyTorch and torchvision for your CUDA version first.
python -m pip install -e '.[maps]'
```

The `maps` extra installs the DOM/DEM and RoMa dependencies used by the AMtown
runner. See [`pyproject.toml`](pyproject.toml) for the full dependency list.

## Quick Start

Place a compatible StreamVGGT checkpoint on disk and prepare the UAVScenes
dataset in the layout below. From the repository root, run the 200-frame AMtown
configuration:

```bash
UAVSCENES_ROOT=/path/to/uavscene \
CHECKPOINT=/path/to/checkpoints.pth \
bash scripts/run_uavscenes_amtown.sh
```

The default run starts at frame index 200, samples every fifth frame, and
processes 200 frames. Set `DRY_RUN=1` to validate paths and frame selection
without running inference. The script prints evaluation metrics in the terminal
and writes its results to `outputs/uavscenes_amtown/`.

## UAVScenes AMtown

The runner expects the standard UAVScenes sequence structure:

```text
uavscene/
|-- images/interval1_AMtown01/
|   |-- interval1_CAM/*.jpg
|   |-- interval1_DOM/{dom,dem,building_mask,road_mask}.tif
|   `-- sampleinfos_interpolated.json
`-- poses/interval1_AMtown01.txt
```

To process every available AMtown frame from index 200 at stride 5:

```bash
UAVSCENES_ROOT=/path/to/uavscene \
CHECKPOINT=/path/to/checkpoints.pth \
NUM_FRAMES=0 MAX_FRAMES_IN_MEMORY=0 \
OUTPUT_DIR=outputs/uavscenes_amtown_from200_stride5 \
bash scripts/run_uavscenes_amtown.sh
```

The runner loads all selected images into memory before inference;
`MAX_FRAMES_IN_MEMORY=0` disables its frame-count guard and does not reduce
memory usage. The first selected GPS position provides the geographic origin.
Later GPS positions are used for evaluation, not as inference inputs.

## Acknowledgments

GeoXel builds on StreamVGGT and uses ideas or code from VGGT, DUSt3R, CroCo,
and RoMa. We thank the authors of those projects. See [`NOTICE.md`](NOTICE.md)
and the notices in the source tree for attribution.

## Citation

If this repository helps your research, please cite the accompanying paper,
*GeoXel: Geographic Per-pixel 3D Mapping from Aerial Video in the Wild*.
Bibliographic details will be added when the paper is publicly available.
Software citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

This repository is released under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 license. See [`LICENSE`](LICENSE),
[`STREAMVGGT-LICENSE.txt`](STREAMVGGT-LICENSE.txt), and [`NOTICE.md`](NOTICE.md)
for the applicable terms and upstream notices.
