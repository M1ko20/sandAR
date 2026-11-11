import json

import numpy as np
import pytest

from sandbox.config import Settings
from sandbox.games import discover_games
from sandbox.games.mapa import find_peaks
from sandbox.games.targets import BUILTIN, evaluate
from sandbox.games.voda import WaterSim
from sandbox.projector.calibration import ProjectorCalibration


def test_games_are_discovered():
    ids = [g.id for g in discover_games()]
    assert ids[:5] == ["topo", "mapa", "vyska", "predloha", "voda"]


def test_water_conserves_volume_and_flows_downhill():
    gh, gw = 60, 80
    x = np.linspace(0, 1, gw)[None, :].repeat(gh, 0)
    terrain = (0.1 * x).astype(np.float32)  # svah klesající doleva
    sim = WaterSim((gh, gw), 0.005)
    sim.depth[:, -10:] = 0.01
    v0 = sim.volume
    for _ in range(300):
        sim.advance(terrain, 1 / 30)
    assert sim.volume == pytest.approx(v0, rel=1e-3)
    assert sim.depth[:, :10].sum() > sim.depth[:, -10:].sum()  # voda stekla dolů
    assert np.all(sim.depth >= 0)


def test_water_drain_removes_water():
    sim = WaterSim((30, 30), 0.005)
    sim.depth[:] = 0.01
    for _ in range(100):
        sim.advance(np.zeros((30, 30), np.float32), 1 / 30, drain=True)
    assert sim.volume < 0.5 * 0.01 * 30 * 30 * 0.005**2


@pytest.mark.parametrize("target", BUILTIN, ids=lambda t: t.name)
def test_targets_have_requested_relief_and_zero_mean(target):
    grid = target.grid((96, 128), (1.0, 0.75), 0.10)
    assert grid.shape == (96, 128)
    assert abs(grid.mean()) < 1e-4
    assert grid.max() - grid.min() == pytest.approx(0.10, rel=1e-3)


def test_score_perfect_flat_and_offset():
    target = BUILTIN[0].grid((96, 128), (1.0, 0.75), 0.10)
    perfect, _ = evaluate(target + 0.03, target, 0.015)  # posun hladiny se nepočítá
    assert perfect.score == pytest.approx(100, abs=0.1)
    flat, diff = evaluate(np.zeros_like(target), target, 0.015)
    assert flat.score < 5
    half, _ = evaluate(target * 0.5, target, 0.015)
    assert 40 < half.score < 60
    assert diff.shape == target.shape


def test_find_peaks_finds_two_hills():
    y, x = np.mgrid[0:100, 0:150]
    h = (0.08 * np.exp(-((x - 40) ** 2 + (y - 50) ** 2) / 300)
         + 0.05 * np.exp(-((x - 110) ** 2 + (y - 50) ** 2) / 300)).astype(np.float32)
    peaks = find_peaks(h, 0.0, 0.01, radius=8)
    assert [(p[0], p[1]) for p in peaks] == [(50, 40), (50, 110)]


def test_projector_parallax_moves_high_points_away_from_projector():
    pc = ProjectorCalibration.default((1280, 800))
    pc.position, pc.height = (0.5, 0.375), 1.2
    flat = pc.box_to_projector(0.9, 0.375, 0.0, (1.0, 0.75))
    high = pc.box_to_projector(0.9, 0.375, 0.1, (1.0, 0.75))
    assert high[0] > flat[0]  # kopec napravo od projektoru se kreslí dál od středu
    assert pc.box_to_projector(0.0, 0.0, 0.0, (1.0, 0.75)) == pytest.approx([0, 0], abs=1e-6)


def test_settings_merge_and_roundtrip():
    s = Settings.from_dict({"map": {"contour_interval": 0.02}, "display": {"monitor": "BenQ"}})
    assert s.map.contour_interval == 0.02 and s.display.monitor == "BenQ"
    assert Settings.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    with pytest.raises(ValueError):
        Settings.from_dict({"map": {"neexistuje": 1}})
