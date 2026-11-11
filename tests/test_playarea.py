import numpy as np

from sandbox.projector.calibration import ProjectorCalibration
from sandbox.sensor.simulated import SimulatedSensor
from sandbox.terrain.calibration import KinectCalibration
from sandbox.terrain.playarea import PlayArea, crop, estimate_margins, visible_rect


def _kc():
    sim = SimulatedSensor(seed=1)
    normal, offset, corners = sim.ground_truth()
    return KinectCalibration.from_plane(sim.intrinsics, normal, offset, corners)


def test_estimate_margins_finds_jumpy_wall_band_but_ignores_gentle_sand():
    gh, gw, size = 100, 160, (1.6, 1.0)  # buňka 1 cm
    h = np.zeros((gh, gw), np.float32)
    h += np.linspace(0, 0.03, gw)[None, :]  # pozvolný svah písku – není stěna
    rng = np.random.default_rng(0)
    h[:, :6] = rng.uniform(0.05, 0.12, (gh, 6))  # x=0: odrazy u stěny 6 cm
    h[:3, :] = 0.10  # y=0: úkos 3 cm
    h[:, -1] += 0.08  # x=W: 1 cm
    m = estimate_margins(h, np.ones_like(h, bool), size, safety=0.0)
    np.testing.assert_allclose(m, (0.06, 0.03, 0.01, 0.0), atol=0.011)


def test_visible_rect_stays_inside_projector_image():
    kc = _kc()
    W, D = kc.size
    # projektor pokrývá jen střed bedny (obraz je menší) a je mírně pootočený
    pc = ProjectorCalibration((1280, 800), np.array([[-200, -150], [1500, -100], [1450, 980], [-180, 940]], float))
    x0, y0, x1, y1 = visible_rect(kc, pc, PlayArea((0.02, 0.02, 0.02, 0.02), frame_inset_px=0))
    pts = pc.box_to_projector(np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1]), np.zeros(4), kc.size)
    assert np.all(pts >= -1.5) and np.all(pts[:, 0] <= 1281.5) and np.all(pts[:, 1] <= 801.5)
    assert x1 - x0 > 0.5 * W and y1 - y0 > 0.5 * D  # nezmenšil zbytečně


def test_crop_keeps_physical_mapping():
    kc = _kc()
    W, D = kc.size
    pc = ProjectorCalibration((1280, 800), np.array([[30, 20], [1250, 25], [1240, 790], [40, 780]], float),
                              (W / 2, D / 2), 1.3)
    rect = (0.07, 0.05, W - 0.08, D - 0.04)
    kc2, pc2 = crop(kc, pc, rect)
    assert pc2 is not None
    np.testing.assert_allclose(kc2.size, (rect[2] - rect[0], rect[3] - rect[1]), atol=2e-3)
    x, y, h = 0.4, 0.3, 0.06  # stejný fyzický bod (i s výškou) -> stejný pixel Kinectu i projektoru
    np.testing.assert_allclose(kc.box_to_pixel(x, y, h), kc2.box_to_pixel(x - rect[0], y - rect[1], h), atol=0.3)
    a = pc.box_to_projector(np.array(x), np.array(y), np.array(h), kc.size)
    b = pc2.box_to_projector(np.array(x - rect[0]), np.array(y - rect[1]), np.array(h), kc2.size)
    np.testing.assert_allclose(a, b, atol=1.0)
