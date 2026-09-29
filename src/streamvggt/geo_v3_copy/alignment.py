from typing import Any, Tuple
import numpy as np


def canonicalize_geo_targets_to_model_space(
    geo_targets: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Map geo-space XY targets into local model space via inverse Sim2."""
    sR = s * R
    if abs(np.linalg.det(sR)) < 1e-12:
        raise ValueError("degenerate Sim2")
    inv = np.linalg.inv(sR)
    return (geo_targets - t) @ inv.T


def apply_model_to_geo(
    model_xy: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Map model-space XY into geo-space via Sim2."""
    return s * (model_xy @ R.T) + t
