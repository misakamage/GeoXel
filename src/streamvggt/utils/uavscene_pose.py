"""UAVScenes camera-pose convention helpers.

``sampleinfos_interpolated.json`` stores ``T4x4`` as camera-to-world.  Keep
that convention in one dependency-light helper so evaluation, point-cloud
construction, and external map registration cannot silently disagree.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def sampleinfo_c2w(sampleinfo_or_pose: Any) -> np.ndarray:
    """Return a validated 4x4 camera-to-world matrix from UAVScenes metadata."""
    value = sampleinfo_or_pose
    if isinstance(value, Mapping):
        if "T4x4" not in value:
            raise KeyError("UAVScenes sample-info record is missing T4x4")
        value = value["T4x4"]
    pose = np.asarray(value, dtype=np.float64)
    if pose.size != 16:
        raise ValueError(f"UAVScenes T4x4 must contain 16 values, got shape {pose.shape}")
    pose = pose.reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError("UAVScenes T4x4 contains non-finite values")
    return pose.copy()


def sampleinfo_c2w_rotation(sampleinfo_or_pose: Any) -> np.ndarray:
    """Return ``R_c2w`` without transposing the dataset matrix."""
    return sampleinfo_c2w(sampleinfo_or_pose)[:3, :3].copy()
