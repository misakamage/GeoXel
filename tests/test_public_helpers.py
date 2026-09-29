import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from geoxel.geo import enu_to_geodetic, geodetic_to_enu, load_geospatial_maps
from scripts.run_uavscenes_amtown import _evaluate_trajectory
from streamvggt.geo_v3.segment import segment_sequence


def test_segment_sequence_covers_frames():
    ranges = segment_sequence(65, 32, 8)
    assert ranges[0].start == 0
    assert ranges[-1].end == 65
    assert all(a.end > a.start for a in ranges)


def test_enu_roundtrip():
    origin = (121.4737, 31.2304, 5.0)
    points = np.array([[0.0, 0.0, 0.0], [20.0, -4.0, 12.0]])
    geo = enu_to_geodetic(points, origin)
    recovered = geodetic_to_enu(geo[:, 0], geo[:, 1], geo[:, 2], origin)
    np.testing.assert_allclose(recovered, points, atol=1e-4)


def test_amtown_evaluation_aligns_first_pose_without_scale():
    origin = (44.82436351242829, 39.91950879258584, 1115.2109222674)
    gps = enu_to_geodetic(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), origin)
    rows = [dict(zip(("lon", "lat", "alt"), point)) for point in gps]
    gt = geodetic_to_enu(gps[:, 0], gps[:, 1], gps[:, 2], origin)
    predicted = gt + np.array([100.0, -50.0, 20.0])
    predicted[1] += [3.0, 0.0, 4.0]

    result = _evaluate_trajectory(predicted, rows, origin)

    assert result["alignment_protocol"] == "first_pose_translation_only_no_scale"
    np.testing.assert_allclose(result["ate_rmse_m"], np.sqrt(25.0 / 2.0))
    np.testing.assert_allclose(result["xy_rmse_m"], np.sqrt(9.0 / 2.0))
    np.testing.assert_allclose(result["z_rmse_m"], np.sqrt(16.0 / 2.0))
    np.testing.assert_allclose(result["last_frame_xy_err_m"], 3.0)
    np.testing.assert_allclose(result["last_frame_z_err_m"], 4.0)


def test_map_callbacks_use_original_pixel_convention(tmp_path):
    lon0, lat0, alt0 = 44.82436351242829, 39.91950879258584, 1115.2109222674
    transform = from_origin(lon0 - 0.001, lat0 + 0.001, 1e-5, 1e-5)
    dom_path, dem_path = tmp_path / "dom.tif", tmp_path / "dem.tif"
    for path, bands, dtype in ((dom_path, 3, "uint8"), (dem_path, 1, "float32")):
        with rasterio.open(
            path, "w", driver="GTiff", height=200, width=200, count=bands,
            dtype=dtype, crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(np.zeros((bands, 200, 200), dtype=dtype))

    maps = load_geospatial_maps(dom_path, dem_path, (lon0, lat0, alt0))
    points = torch.tensor([[0.0, 0.0, 0.0], [24.0, -11.0, 300.0]], dtype=torch.float32)
    projected = maps.project_fn(points)
    assert projected.dtype == points.dtype
    assert torch.equal(projected[:, :], maps.project_fn(points * torch.tensor([1.0, 1.0, 0.0])))

    lat_rad = np.radians(lat0)
    metres_per_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    metres_per_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    for point, pixel in zip(points.numpy(), projected.numpy()):
        row, col = rasterio.transform.rowcol(
            transform, lon0 + point[0] / metres_per_lon, lat0 + point[1] / metres_per_lat,
        )
        np.testing.assert_array_equal(pixel, [col, row])

    fractional_pixel = (105.3, 103.7)
    snapped_x, snapped_y = rasterio.transform.xy(transform, 104, 105)
    expected = ((snapped_x - lon0) * metres_per_lon,
                (snapped_y - lat0) * metres_per_lat)
    np.testing.assert_allclose(maps.inv_project_fn(*fractional_pixel), expected, atol=1e-9)
