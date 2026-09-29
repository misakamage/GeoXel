"""Controlled map-input perturbations used by rebuttal robustness studies."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def shift_geotransform_enu(
    geotransform: Any,
    crs: Any,
    anchor_lon: float,
    anchor_lat: float,
    east_m: float = 0.0,
    north_m: float = 0.0,
) -> Tuple[Any, Dict[str, float]]:
    """Move the DOM georeference by a requested local-ENU translation.

    Pixel values are unchanged. A positive east/north shift means that the
    same DOM pixel is declared to lie east/north of its original location.
    """
    east_m = _require_finite("east_m", east_m)
    north_m = _require_finite("north_m", north_m)
    diag = {
        "dom_shift_e_m": east_m,
        "dom_shift_n_m": north_m,
        "crs_shift_x": 0.0,
        "crs_shift_y": 0.0,
    }
    if east_m == 0.0 and north_m == 0.0:
        return geotransform, diag

    import pyproj
    from affine import Affine

    lat_rad = np.radians(float(anchor_lat))
    m_per_deg_lat = (
        111132.92 - 559.82 * np.cos(2.0 * lat_rad)
        + 1.175 * np.cos(4.0 * lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3.0 * lat_rad)
    )
    shifted_lon = float(anchor_lon) + east_m / max(float(m_per_deg_lon), 1e-9)
    shifted_lat = float(anchor_lat) + north_m / max(float(m_per_deg_lat), 1e-9)
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:4326"), pyproj.CRS(crs), always_xy=True
    )
    x0, y0 = transformer.transform(float(anchor_lon), float(anchor_lat))
    x1, y1 = transformer.transform(shifted_lon, shifted_lat)
    dx, dy = float(x1 - x0), float(y1 - y0)

    if hasattr(geotransform, "a"):
        shifted = Affine(
            geotransform.a,
            geotransform.b,
            geotransform.c + dx,
            geotransform.d,
            geotransform.e,
            geotransform.f + dy,
        )
    else:
        # GDAL tuple: (origin_x, pixel_x, rot_x, origin_y, rot_y, pixel_y).
        values = tuple(geotransform)
        if len(values) != 6:
            raise ValueError("geotransform must be rasterio Affine or GDAL 6-tuple")
        shifted = (
            values[0] + dx,
            values[1],
            values[2],
            values[3] + dy,
            values[4],
            values[5],
        )
    diag.update({"crs_shift_x": dx, "crs_shift_y": dy})
    return shifted, diag


def degrade_dom_resolution(
    dom_image: np.ndarray,
    factor: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reduce DOM texture resolution while preserving its grid/georeference.

    The image is area-downsampled and bilinearly restored to its original
    dimensions. Thus only visual information content changes; downstream DOM
    pixel coordinates, masks, and the DEM remain on the original grid.
    """
    factor = _require_finite("factor", factor)
    if factor < 1.0:
        raise ValueError(f"factor must be >= 1, got {factor}")
    image = np.asarray(dom_image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"DOM image must have shape (H,W,C>=3), got {image.shape}")
    height, width = int(image.shape[0]), int(image.shape[1])
    low_width = max(1, int(round(width / factor)))
    low_height = max(1, int(round(height / factor)))
    diag: Dict[str, Any] = {
        "dom_resolution_factor": factor,
        "native_width": width,
        "native_height": height,
        "degraded_width": low_width,
        "degraded_height": low_height,
        "effective_factor_x": float(width / low_width),
        "effective_factor_y": float(height / low_height),
    }
    if factor == 1.0 or (low_width == width and low_height == height):
        return image, diag

    from PIL import Image

    rgb = np.clip(image[..., :3], 0, 255).astype(np.uint8, copy=False)
    pil = Image.fromarray(rgb, mode="RGB")
    resampling = getattr(Image, "Resampling", Image)
    low = pil.resize((low_width, low_height), resample=resampling.BOX)
    restored = low.resize((width, height), resample=resampling.BILINEAR)
    return np.asarray(restored, dtype=np.uint8), diag


def apply_dem_elevation_bias(geo_elev: Any, bias_m: float) -> Dict[str, float]:
    """Apply a uniform vertical bias to every DEM lookup in-place."""
    bias_m = _require_finite("bias_m", bias_m)
    if geo_elev is None:
        if bias_m != 0.0:
            raise ValueError("a non-zero DEM bias requires an available DEM")
        return {"dem_elevation_bias_m": 0.0}
    data = np.asarray(geo_elev.dem_data)
    before_min = float(np.nanmin(data))
    before_max = float(np.nanmax(data))
    if bias_m != 0.0:
        geo_elev.dem_data = data.astype(np.float32, copy=True) + bias_m
    geo_elev.elevation_bias_m = bias_m
    return {
        "dem_elevation_bias_m": bias_m,
        "dem_min_before_m": before_min,
        "dem_max_before_m": before_max,
        "dem_min_after_m": before_min + bias_m,
        "dem_max_after_m": before_max + bias_m,
    }
