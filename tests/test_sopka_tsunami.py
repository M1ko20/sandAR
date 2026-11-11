import numpy as np
import pytest

from sandbox.games.sopka import DURATION as SOPKA_DURATION
from sandbox.games.sopka import Particles, disc_triangles, find_vent
from sandbox.games.tsunami import (edge_slices, inward_flux, lowest_edge, place_towns, sea_fill, sea_level_for,
                                   velocity)
from sandbox.sim.lava import LavaSim, blur3, downsample
from sandbox.sim.water import WaterSim


def cone(gh=60, gw=80, cy=30, cx=40, height=0.08, radius=30):
    y, x = np.mgrid[0:gh, 0:gw]
    r = np.hypot(y - cy, x - cx)
    return (height * np.clip(1 - r / radius, 0, None)).astype(np.float32)


# --- láva -----------------------------------------------------------------------------
def test_downsample_and_blur():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    d = downsample(a, 2)
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx((0 + 1 + 4 + 5) / 4)
    assert downsample(np.zeros((5, 7), np.float32), 2).shape == (2, 3)
    b = blur3(np.ones((5, 6), np.float32))
    assert b.shape == (5, 6) and np.allclose(b, 1)


def test_lava_conserves_volume_and_flows_downhill():
    terrain = cone()
    sim = LavaSim(terrain.shape, 0.01)
    sim.erupt(30, 40, 2.5, 2e-5, 1.0)
    v0 = sim.volume
    assert v0 == pytest.approx(2e-5, rel=1e-4)
    for _ in range(90):
        sim.advance(terrain, 1 / 30)
    assert sim.volume == pytest.approx(v0, rel=1e-3)
    assert np.all(sim.thickness >= 0)
    assert 0 <= sim.temp.min() and sim.temp.max() <= 1
    covered = sim.thickness > 1e-4
    ys, xs = np.nonzero(covered)
    assert np.hypot(ys - 30, xs - 40).max() > 6  # rozteklo se po svazích
    assert sim.thickness[30, 40] < 0.05


def test_lava_follows_slope_not_uphill():
    gh, gw = 40, 80
    x = np.linspace(0, 1, gw)[None, :].repeat(gh, 0)
    terrain = (0.15 * x).astype(np.float32)  # svah klesá doleva
    sim = LavaSim((gh, gw), 0.01)
    for _ in range(60):
        sim.erupt(20, 40, 2.0, 1e-5, 1 / 30)
        sim.advance(terrain, 1 / 30)
    left = sim.thickness[:, :40].sum()
    right = sim.thickness[:, 41:].sum()
    assert left > 3 * right


def test_cold_lava_does_not_move_and_hot_lava_cools():
    terrain = cone()
    sim = LavaSim(terrain.shape, 0.01)
    sim.thickness[25:35, 35:45] = 0.01
    sim.temp[25:35, 35:45] = 0.05  # pod teplotou tuhnutí
    before = sim.thickness.copy()
    for _ in range(30):
        sim.advance(terrain, 1 / 30)
    assert np.allclose(sim.thickness, before)

    sim.clear()
    sim.erupt(30, 40, 3, 2e-5, 1.0)
    t0 = sim.temp.max()
    for _ in range(30 * 40):
        sim.advance(terrain, 1 / 30)
    assert sim.temp.max() < t0
    assert sim.temp.max() < sim.solid  # po 40 s je všechno ztuhlé
    assert not sim.molten.any()


def test_thin_hot_film_stops_on_flat_ground():
    """Mez kluzu: tenká vrstva na rovině neteče do nekonečna (láva tvoří jazyky s čelem)."""
    terrain = np.zeros((40, 40), np.float32)
    sim = LavaSim(terrain.shape, 0.01, cooling=0, thin_cooling=0)
    sim.erupt(20, 20, 3, 2e-5, 1.0)
    for _ in range(300):
        sim.advance(terrain, 1 / 30)
    ys, xs = np.nonzero(sim.thickness > 1e-6)
    assert np.hypot(ys - 20, xs - 20).max() < 18


def test_lava_window_matches_full_step():
    terrain = cone()
    a = LavaSim(terrain.shape, 0.01)
    a.erupt(30, 40, 2.5, 2e-5, 1.0)
    b = LavaSim(terrain.shape, 0.01)
    b.thickness[:] = a.thickness
    b.temp[:] = a.temp
    n = int(np.ceil((1 / 30) * a.mobility / 0.15))
    for _ in range(10):
        a.advance(terrain, 1 / 30)
        for _ in range(n):
            b.step(terrain, 1 / 30 / n)
    assert np.allclose(a.thickness, b.thickness, atol=1e-7)
    assert np.allclose(a.temp, b.temp, atol=1e-5)


# --- sopka --------------------------------------------------------------------------
def test_find_vent_highest_peak_near_hand_and_ignores_hands():
    h = cone(80, 120, 40, 30, 0.10, 20) + cone(80, 120, 40, 90, 0.06, 20)
    iy, ix, rel = find_vent(h)
    assert (iy, ix) == (40, 30) and rel > 0.05
    iy, ix, _ = find_vent(h, near=(40, 95), radius=15)
    assert abs(iy - 40) <= 1 and abs(ix - 90) <= 1
    hands = np.zeros(h.shape, bool)
    hands[30:50, 20:40] = True  # ruka zakrývá vyšší vrchol
    iy, ix, _ = find_vent(h, hands)
    assert abs(ix - 90) <= 2


def test_particles_expire_and_discs():
    p = Particles()
    p.spawn([[0.1, 0.1], [0.2, 0.2]], [[0.1, 0], [0, 0.1]], [0.5, 2.0], 1)
    p.update(1.0, np.zeros(2, np.float32))
    assert len(p) == 1
    verts, cols = disc_triangles(np.array([[10.0, 10.0]]), np.array([5.0]), np.array([[1, 0, 0, 1.0]]), segments=8)
    assert verts.shape == (24, 2) and cols.shape == (24, 4)
    assert sum(SOPKA_DURATION.values()) > 30


# --- tsunami --------------------------------------------------------------------------
def test_edges_sea_level_and_fill():
    gh, gw = 40, 60
    x = np.linspace(0, 1, gw)[None, :].repeat(gh, 0)
    terrain = (0.1 * x - 0.02).astype(np.float32)  # nejníž vlevo
    terrain[15:25, 40:50] = -0.05  # vnitrozemská prohlubeň (není spojená s mořem)
    assert lowest_edge(terrain) == 0
    sea = sea_level_for(terrain, 0)
    fill = sea_fill(terrain, sea, 0)
    assert fill[:, 0].min() > 0  # u okraje moře je voda
    assert fill[20, 45] == 0  # jezero ve vnitrozemí moře nezalije
    assert np.all(fill >= 0)
    r, c = edge_slices(2, 3)
    assert np.zeros((gh, gw))[r, c].shape == (gh, 3)
    r, c = edge_slices(1, 2)
    assert np.zeros((gh, gw))[r, c].shape == (2, gw)


def test_towns_are_on_dry_land_away_from_sea():
    gh, gw = 60, 100
    x = np.linspace(0, 1, gw)[None, :].repeat(gh, 0)
    terrain = (0.1 * x - 0.02).astype(np.float32)
    sea = sea_level_for(terrain, 0)
    towns = place_towns(terrain, sea, 0, np.random.default_rng(1))
    assert len(towns) >= 4
    for ty, tx in towns:
        assert terrain[ty, tx] > sea + 0.008
        assert tx > 0.18 * gw


def _run_wave(terrain):
    sim = WaterSim(terrain.shape, 0.02)
    sea = -0.01
    sim.depth[:] = sea_fill(terrain, sea, 0)
    rows, cols = edge_slices(0, 3)
    g = 9.81 * 0.35
    for i in range(30 * 5):
        if i < 30 * 3:
            sim.depth[rows, cols] = np.maximum(sea + 0.08 - terrain[rows, cols], 0)
            inward_flux(sim, 0)[rows, cols] = 0.5 * sim.depth[rows, cols] * sim.cell
        sim.advance(terrain, 1 / 30, g)
        sim.depth[edge_slices(2, 1)] = 0
    return sim


def test_wave_floods_lowland_but_not_behind_high_wall():
    terrain = np.zeros((40, 80), np.float32)
    terrain[:, :10] = -0.03  # moře vlevo
    open_sim = _run_wave(terrain)
    walled = terrain.copy()
    walled[:, 40:44] = 0.25  # vysoká hráz přes celou bednu
    wall_sim = _run_wave(walled)
    assert open_sim.depth[:, 50:70].mean() > 0.005
    assert wall_sim.depth[:, 50:70].max() < 1e-4
    assert velocity(open_sim).max() > 0


def test_velocity_zero_for_still_water():
    sim = WaterSim((20, 20), 0.01)
    sim.depth[:] = 0.02
    sim.advance(np.zeros((20, 20), np.float32), 1 / 30)
    assert velocity(sim).max() < 1e-3


# --- integrace s RenderHost (bez okna) --------------------------------------------------
@pytest.fixture(scope="module")
def host():
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - stroj bez OpenGL
        pytest.skip(f"OpenGL není k dispozici: {exc}")
    from sandbox.config import Settings
    from sandbox.render.host import RenderHost

    h = RenderHost(ctx, Settings())
    h.configure((1.0, 0.6), (60, 100))
    return h


def _heightmap(hands=False):
    from sandbox.terrain.heightmap import HeightMap

    terrain = cone(60, 100, 30, 50, 0.08, 25) - 0.01
    hm = HeightMap(terrain, (1.0, 0.6))
    if hands:
        hm.hands = np.zeros(terrain.shape, bool)
        hm.hands[5:10, 5:10] = True
        hm.hand_points = [(0.5, 0.3)]
    return hm


def test_volcano_cycle_erupts_and_resets(host):
    from sandbox.games.sopka import CALM, VolcanoGame

    hm = _heightmap()
    host.update_heightmap(hm)
    game = VolcanoGame(host)
    game.start()
    for _ in range(30 * 12):  # první klid je krátký -> probouzení -> erupce
        game.update(hm, 1 / 30)
    assert game.eruptions == 1
    assert game.sim is not None and game.sim.volume > 0
    assert abs(game.vent_m[0] - 0.5) < 0.03 and abs(game.vent_m[1] - 0.3) < 0.03
    host.begin_canvas()
    game.render()
    assert game.status()
    game.reset()
    assert game.phase == CALM and game.sim.volume == 0
    game.stop()


def test_volcano_hand_wakes_volcano(host):
    from sandbox.games.sopka import RUMBLE, VolcanoGame

    hm = _heightmap(hands=True)
    game = VolcanoGame(host)
    game.start()
    for _ in range(40):
        game.update(hm, 1 / 30)
    assert game.phase == RUMBLE
    game.stop()


def test_tsunami_cycle_floods_and_reports(host):
    from sandbox.games.tsunami import CALM, RESULT, TsunamiGame

    hm = _heightmap()
    host.update_heightmap(hm)
    game = TsunamiGame(host)
    game.start()
    assert game.key("t")
    for _ in range(30 * 16):
        game.update(hm, 1 / 30)
    assert game.waves == 1
    assert game.phase == RESULT
    assert game.last_result is not None and game.last_result[0] > 5
    host.begin_canvas()
    game.render()
    assert any("zaplaveno" in s for s in game.status())
    game.reset()
    assert game.phase == CALM
    game.stop()
