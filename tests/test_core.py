import time

import numpy as np

from sandbox.attract import AttractController
from sandbox.config import AttractSettings, TerrainSettings
from sandbox.projector.calibration import ProjectorCalibration
from sandbox.terrain.heightmap import HeightMap


def _hm(h=None, hand_points=()):
    heights = np.zeros((60, 100), np.float32) if h is None else h
    return HeightMap(heights, (1.0, 0.6), np.zeros(heights.shape, bool), hand_points=list(hand_points))


def test_attract_starts_after_idle_and_stops_on_hand():
    a = AttractController(AttractSettings(idle_seconds=0.05, cycle_seconds=10))
    a.observe(_hm())
    assert not a.want_switch()
    time.sleep(0.07)
    assert a.want_switch()  # první přepnutí hned po začátku klidu
    assert a.active and not a.want_switch()  # další až po cycle_seconds
    a.observe(_hm(hand_points=[(0.5, 0.3)]))
    assert not a.active and not a.want_switch()


def test_attract_terrain_change_counts_as_activity():
    a = AttractController(AttractSettings(idle_seconds=0.05))
    a.observe(_hm())
    time.sleep(0.07)
    h = np.zeros((60, 100), np.float32)
    h[10:30, 10:40] = 0.05  # někdo postavil kopec
    a.observe(_hm(h))
    assert a.idle_for < 0.05 and not a.want_switch()


def test_hand_points_are_blob_centers_in_meters():
    from sandbox.sensor.simulated import SimulatedSensor
    from sandbox.terrain import KinectCalibration, TerrainProcessor

    sim = SimulatedSensor(seed=1)

    normal, offset, corners = sim.ground_truth()
    cal = KinectCalibration.from_plane(sim.intrinsics, normal, offset, corners)
    proc = TerrainProcessor(cal, TerrainSettings())
    hands = np.zeros((proc.gh, proc.gw), bool)
    hands[10:20, 30:40] = True  # větší
    hands[40:44, 80:84] = True  # menší
    pts = proc.hand_points(hands)
    cw, ch = proc.size[0] / proc.gw, proc.size[1] / proc.gh
    assert len(pts) == 2
    np.testing.assert_allclose(pts[0], (35 * cw, 15 * ch), atol=1e-6)  # největší první
    assert proc.hand_points(np.zeros_like(hands)) == []


def test_fit_projector_recovers_known_projector_with_parallax():
    from sandbox.autocal import fit_projector

    box = (1.6, 1.0)
    truth = ProjectorCalibration((1280, 800), np.array([[1200, 40], [60, 20], [90, 780], [1250, 760]], float),
                                 (0.7, 0.4), 1.4)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0.1, 1.5, 80), rng.uniform(0.1, 0.9, 80)
    h = rng.uniform(-0.03, 0.15, 80)
    proj = truth.box_to_projector(x, y, h, box)
    proj += rng.normal(0, 0.3, proj.shape)  # šum měření
    cal, rms = fit_projector(np.column_stack([x, y, h]), proj, box, (1280, 800), (0.8, 0.5))
    assert rms < 1.0
    assert abs(cal.height - 1.4) < 0.1
    np.testing.assert_allclose(cal.corners, truth.corners, atol=3)
