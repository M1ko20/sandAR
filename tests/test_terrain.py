import numpy as np
import pytest

from sandbox.config import TerrainSettings
from sandbox.sensor.simulated import SimulatedSensor
from sandbox.terrain.calibration import KinectCalibration, fit_plane
from sandbox.terrain.processor import TerrainProcessor, fill_holes


@pytest.fixture
def sim():
    return SimulatedSensor(seed=1)


def test_fit_plane_recovers_plane_with_outliers():
    rng = np.random.default_rng(0)
    n = np.array([0.05, -0.03, -1.0])
    n /= np.linalg.norm(n)
    pts = rng.uniform(-0.5, 0.5, (5000, 3))
    pts[:, 2] = 0
    pts = pts - n * ((pts @ n) + 1.2)[:, None]  # rovina h = n·p + 1.2 = 0
    pts[:300] += n * 0.1  # odlehlé body (ruka, okraj)
    normal, offset = fit_plane(pts)
    assert offset == pytest.approx(1.2, abs=1e-3)
    assert abs(normal @ n) == pytest.approx(1.0, abs=1e-4)


def test_calibration_from_flat_depth_matches_ground_truth(sim):
    _, offset, corners = sim.ground_truth()
    sim.flatten()
    cal = KinectCalibration.from_depth(sim.intrinsics, sim.render_depth(noise=True), corners)
    assert cal.offset == pytest.approx(offset, abs=0.005)
    assert cal.size[0] == pytest.approx(1.0, abs=0.01)
    assert cal.size[1] == pytest.approx(0.75, abs=0.01)


def test_box_pixel_roundtrip(sim):
    normal, offset, corners = sim.ground_truth()
    cal = KinectCalibration.from_plane(sim.intrinsics, normal, offset, corners)
    px = cal.box_to_pixel(np.array([0.2, 0.7]), np.array([0.1, 0.6]), np.array([0.05, -0.03]))
    cam = cal.box_to_camera(np.array([0.2, 0.7]), np.array([0.1, 0.6]), np.array([0.05, -0.03]))
    x, y, h = cal.camera_to_box(cam)
    np.testing.assert_allclose(x, [0.2, 0.7], atol=1e-6)
    np.testing.assert_allclose(y, [0.1, 0.6], atol=1e-6)
    np.testing.assert_allclose(h, [0.05, -0.03], atol=1e-6)
    np.testing.assert_allclose(cal.corners_px[0], cal.box_to_pixel(0.0, 0.0)[...], atol=1e-6)
    assert px.shape == (2, 2)


def test_calibration_serialization(sim):
    normal, offset, corners = sim.ground_truth()
    cal = KinectCalibration.from_plane(sim.intrinsics, normal, offset, corners)
    cal2 = KinectCalibration.from_dict(cal.to_dict())
    np.testing.assert_allclose(cal.plane_to_box, cal2.plane_to_box)
    assert cal2.size == pytest.approx(cal.size)


def test_processor_reconstructs_terrain(sim):
    normal, offset, corners = sim.ground_truth()
    cal = KinectCalibration.from_plane(sim.intrinsics, normal, offset, corners)
    proc = TerrainProcessor(cal, TerrainSettings(grid_width=128))
    for i in range(5):
        hm = proc.process(sim.render_depth(noise=True), seq=i)
    gx, gy = hm.cell_centers()
    truth = sim.true_height(gx, gy)
    err = hm.heights - truth
    inner = (slice(4, -4), slice(4, -4))
    assert np.sqrt(np.mean(err[inner] ** 2)) < 0.006  # < 6 mm RMS
    assert proc.last_coverage > 0.95


def test_processor_ignores_passing_hand(sim):
    normal, offset, corners = sim.ground_truth()
    cal = KinectCalibration.from_plane(sim.intrinsics, normal, offset, corners)
    sim.flatten()
    proc = TerrainProcessor(cal, TerrainSettings(grid_width=128))
    flat = sim.render_depth(noise=False)
    proc.process(flat)
    hand = flat.copy()
    hand[150:260, 200:300] -= 0.10  # ruka 10 cm nad pískem
    for _ in range(4):  # krátký průjezd rukou
        hm = proc.process(hand)
    assert np.abs(hm.heights).max() < 0.01
    assert hm.hands is not None and hm.hands.any()
    for _ in range(30):  # ruka zmizí
        hm = proc.process(flat)
    assert not hm.hands.any()


def test_processor_accepts_lasting_change(sim):
    normal, offset, corners = sim.ground_truth()
    cal = KinectCalibration.from_plane(sim.intrinsics, normal, offset, corners)
    sim.flatten()
    settings = TerrainSettings(grid_width=128)
    proc = TerrainProcessor(cal, settings)
    proc.process(sim.render_depth(noise=False))
    sim.set_sand(lambda x, y: 0.06 * np.exp(-((x - 0.5) ** 2 + (y - 0.4) ** 2) / 0.01))
    for _ in range(settings.stable_frames * 3):
        hm = proc.process(sim.render_depth(noise=True))
    assert hm.sample(0.5, 0.4) == pytest.approx(0.06, abs=0.008)


def test_fill_holes():
    v = np.ones((20, 20), np.float32) * 3
    valid = np.ones_like(v, bool)
    valid[5:10, 5:10] = False
    out = fill_holes(np.where(valid, v, 0), valid)
    np.testing.assert_allclose(out, 3, atol=1e-5)


def test_detect_box_corners_in_simulation(sim):
    from sandbox.terrain.calibration import detect_box_corners

    _, _, truth = sim.ground_truth()
    sim.flatten()
    corners = detect_box_corners(sim.intrinsics, sim.render_depth(noise=True))
    assert corners is not None
    np.testing.assert_allclose(corners, truth, atol=4)  # do 4 px


def test_detect_box_corners_on_real_kinect_frame():
    """Skutečný snímek z Kinectu v2 nad pískovištěm (11. 11. 2026)."""
    from pathlib import Path

    from sandbox.sensor.base import Intrinsics
    from sandbox.terrain.calibration import detect_box_corners

    data = np.load(Path(__file__).parent / "data" / "kinect2_flat.npz")
    w, h, fx, fy, cx, cy = data["intr"]
    intr = Intrinsics(int(w), int(h), fx, fy, cx, cy)
    depth = data["depth_mm"].astype(np.float32) / 1000
    corners = detect_box_corners(intr, depth)
    assert corners is not None
    cal = KinectCalibration.from_depth(intr, depth, corners)
    assert 1.5 < cal.size[0] < 1.9 and 0.95 < cal.size[1] < 1.2  # bedna cca 173 x 109 cm
    assert cal.offset == pytest.approx(1.39, abs=0.03)
