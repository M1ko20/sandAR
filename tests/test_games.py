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
    assert ids[:5] == ["topo", "mapa", "vyska", "krajina", "voda"]


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
def test_targets_have_relief_base_and_main_part(target):
    grid, mask = target.layout((96, 128), (1.0, 0.75), 0.10)
    assert grid.shape == mask.shape == (96, 128)
    assert max(grid.max(), -grid.min() / 0.6) == pytest.approx(0.10, rel=1e-3)
    assert np.median(grid[~mask]) == pytest.approx(0.0, abs=1e-3)  # okolí = okolní písek
    assert 0.08 < mask.mean() < 0.6  # hodnotí se hlavní část, ne celá bedna


def test_only_main_part_is_scored():
    grid, mask = BUILTIN[0].layout((96, 128), (1.0, 0.75), 0.10)
    messy = grid + np.where(mask, 0, 0.05 * np.sin(np.arange(128) / 3)[None, :]).astype(np.float32)
    assert evaluate(messy, grid, 0.015, mask=mask)[0].score == pytest.approx(100, abs=0.5)
    assert evaluate(messy, grid, 0.015)[0].score < 90  # bez masky by nepořádek okolo vadil


def test_score_perfect_flat_and_offset():
    target = BUILTIN[0].grid((96, 128), (1.0, 0.75), 0.10)
    perfect, _ = evaluate(target + 0.03, target, 0.015)  # posun hladiny se nepočítá
    assert perfect.score == pytest.approx(100, abs=0.1)
    flat, diff = evaluate(np.zeros_like(target), target, 0.015)
    assert flat.score < 5
    half, _ = evaluate(target * 0.5, target, 0.015)
    assert 40 < half.score < 75  # odchylka do tolerance se nepočítá -> o něco víc než polovina
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


def test_main_target_reaches_100_within_tolerance():
    """Hlavní předloha jde postavit na 100 %: chyby do tolerance (ruka, šum Kinectu) se nepočítají."""
    assert BUILTIN[0].name == "Hora se sedlem"
    grid, mask = BUILTIN[0].layout((96, 160), (1.62, 0.97), 0.10)
    rough = grid + np.random.default_rng(1).uniform(-0.012, 0.012, grid.shape).astype(np.float32)
    assert evaluate(rough, grid, 0.015, mask=mask)[0].score == pytest.approx(100, abs=0.01)
    assert grid.min() >= -1e-6  # nic se nekope pod okolní písek


def test_treasure_is_found_when_dug_to_the_bottom():
    """Poklad nikdy není pod dnem bedny: kdo vykope až na dno, najde ho."""
    from sandbox.games.poklad import Treasure, update_reveals

    heights = np.zeros((40, 60), np.float32)
    tr = Treasure(0.5, 0.3, z=-0.06)  # „zakopaný“ hlouběji, než sahá písek
    heights[:, :] = -0.02  # dno bedny: níž to nejde
    assert update_reveals([tr], heights, (1.0, 0.6), 1.0, 0.0) == []
    found = update_reveals([tr], heights, (1.0, 0.6), 1.0, 0.0, bottom=-0.02)
    assert found == [tr]


def test_audio_mixer_loops_fade_and_one_shots_end():
    """Mixer: smyčka sleduje hlasitost a po přestání volání dozní, jednorázový zvuk skončí."""
    from sandbox.audio import Mixer

    m = Mixer()
    m.master, m._g = 1.0, 1.0
    tone = (0.5 * np.sin(np.arange(48000) / 48000 * 2 * np.pi * 440)).astype(np.float32)
    m.play(tone[:4800], 1.0, 0.0)
    for _ in range(30):
        m.loop("t", tone, 0.5, -0.5)
        out = m.render(1024)
    assert np.abs(out).max() > 0.1
    assert np.sqrt((out[:, 0] ** 2).mean()) > np.sqrt((out[:, 1] ** 2).mean())  # panorama vlevo
    assert m.active == 1  # jednorázový zvuk dohrál, zbývá smyčka
    m._loops["t"].last = 0.0  # mód ji přestal volat
    for _ in range(40):
        out = m.render(1024)
    assert m.active == 0 and np.abs(out).max() < 1e-3
