"""Geospatial I/O helpers for GeoXel.

GeoXel optimizes in a local East-North-Up (ENU) frame. Raster maps may use any
projected CRS supported by rasterio/pyproj; this module handles the conversion
between ENU coordinates and DOM pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch


_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F**2


def _geodetic_to_ecef(lon, lat, alt):
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    sin_lat = np.sin(lat)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    return (
        (n + alt) * np.cos(lat) * np.cos(lon),
        (n + alt) * np.cos(lat) * np.sin(lon),
        (n * (1.0 - _WGS84_F) ** 2 + alt) * np.sin(lat),
    )


def geodetic_to_enu(lon, lat, alt, origin):
    """Convert WGS-84 lon/lat/alt to ENU metres around ``origin``."""
    lon0, lat0, alt0 = map(float, origin)
    x, y, z = _geodetic_to_ecef(lon, lat, alt)
    x0, y0, z0 = _geodetic_to_ecef(lon0, lat0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    lo, la = np.radians(lon0), np.radians(lat0)
    e = -np.sin(lo) * dx + np.cos(lo) * dy
    n = -np.sin(la) * np.cos(lo) * dx - np.sin(la) * np.sin(lo) * dy + np.cos(la) * dz
    u = np.cos(la) * np.cos(lo) * dx + np.cos(la) * np.sin(lo) * dy + np.sin(la) * dz
    return np.stack([e, n, u], axis=-1)


def enu_to_geodetic(enu, origin):
    """Convert ENU points (metres) to WGS-84 lon/lat/alt."""
    xyz = np.asarray(enu, dtype=np.float64)
    shape = xyz.shape[:-1]
    xyz = xyz.reshape(-1, 3)
    lon0, lat0, alt0 = map(float, origin)
    lo, la = np.radians(lon0), np.radians(lat0)
    e, n, u = xyz.T
    dx = -np.sin(lo) * e - np.sin(la) * np.cos(lo) * n + np.cos(la) * np.cos(lo) * u
    dy = np.cos(lo) * e - np.sin(la) * np.sin(lo) * n + np.cos(la) * np.sin(lo) * u
    dz = np.cos(la) * n + np.sin(la) * u
    x0, y0, z0 = _geodetic_to_ecef(lon0, lat0, alt0)
    x, y, z = x0 + dx, y0 + dy, z0 + dz
    b = _WGS84_A * (1.0 - _WGS84_F)
    ep2 = (_WGS84_A**2 - b**2) / b**2
    p = np.sqrt(x**2 + y**2)
    lon = np.degrees(np.arctan2(y, x))
    theta = np.arctan2(z * _WGS84_A, p * b)
    lat_r = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3,
        p - _WGS84_E2 * _WGS84_A * np.cos(theta) ** 3,
    )
    n_radius = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat_r) ** 2)
    alt = p / np.cos(lat_r) - n_radius
    return np.stack([lon, np.degrees(lat_r), alt], axis=-1).reshape(shape + (3,))


@dataclass
class GeoXelMap:
    """Loaded DOM/DEM maps and coordinate conversion functions."""

    dom_image: torch.Tensor  # [3, H, W], uint8 on CPU
    dom_transform: object
    dom_crs: object
    project_fn: Callable
    inv_project_fn: Callable
    geo_elev: object
    origin: Tuple[float, float, float]
    dem_path: Path


def _to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = np.repeat(array[None], 3, axis=0)
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    array = array[:3].astype(np.float32)
    if np.nanmax(array) <= 1.0:
        array *= 255.0
    finite = np.isfinite(array)
    if not finite.all():
        fill = np.nanmedian(array[finite]) if finite.any() else 0.0
        array = np.nan_to_num(array, nan=float(fill), posinf=255.0, neginf=0.0)
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def load_geospatial_maps(
    dom_path: str | Path,
    dem_path: str | Path,
    origin: Tuple[float, float, float],
    building_mask_path: Optional[str | Path] = None,
    road_mask_path: Optional[str | Path] = None,
) -> GeoXelMap:
    """Load DOM/DEM GeoTIFFs and return GeoXel-compatible map callbacks."""
    import rasterio
    import pyproj

    dom_path, dem_path = Path(dom_path), Path(dem_path)
    if not dom_path.is_file():
        raise FileNotFoundError(f"DOM raster not found: {dom_path}")
    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM raster not found: {dem_path}")

    with rasterio.open(dom_path) as src:
        dom = _to_rgb_uint8(src.read())
        transform, crs = src.transform, src.crs
        height, width = src.height, src.width
    if crs is None:
        raise ValueError("DOM GeoTIFF must declare a CRS")

    wgs84 = pyproj.CRS("EPSG:4326")
    to_map = pyproj.Transformer.from_crs(wgs84, crs, always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs(crs, wgs84, always_xy=True)
    lat_ref_rad = np.radians(float(origin[1]))
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_ref_rad) + 1.175 * np.cos(4 * lat_ref_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_ref_rad) - 93.5 * np.cos(3 * lat_ref_rad)

    def project_fn(world_xyz):
        device, dtype = world_xyz.device, world_xyz.dtype
        xyz = world_xyz.detach().cpu().float().numpy()
        lon = float(origin[0]) + xyz[:, 0] / m_per_deg_lon
        lat = float(origin[1]) + xyz[:, 1] / m_per_deg_lat
        x, y = to_map.transform(lon, lat)
        rows, cols = [], []
        for map_x, map_y in zip(x, y):
            row, col = rasterio.transform.rowcol(transform, map_x, map_y)
            rows.append(row)
            cols.append(col)
        uv = np.stack([np.asarray(cols, dtype=np.float64),
                       np.asarray(rows, dtype=np.float64)], axis=-1)
        return torch.tensor(uv, device=device, dtype=dtype)

    def inv_project_fn(dom_col, dom_row):
        x, y = rasterio.transform.xy(transform, int(round(dom_row)), int(round(dom_col)))
        lon, lat = to_wgs84.transform(x, y)
        e = (lon - float(origin[0])) * m_per_deg_lon
        n = (lat - float(origin[1])) * m_per_deg_lat
        return float(e), float(n)

    from streamvggt.utils.dom_matching import GeoElevationQuery

    geo_elev = GeoElevationQuery(
        str(dem_path),
        str(building_mask_path) if building_mask_path else None,
        float(origin[0]), float(origin[1]), float(origin[2]),
        road_mask_path=str(road_mask_path) if road_mask_path else None,
    )
    return GeoXelMap(
        dom_image=torch.from_numpy(dom),
        dom_transform=transform,
        dom_crs=crs,
        project_fn=project_fn,
        inv_project_fn=inv_project_fn,
        geo_elev=geo_elev,
        origin=tuple(map(float, origin)),
        dem_path=dem_path,
    )
