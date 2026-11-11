import math

import numpy as np
import pytest

from sandbox.sim import ecosystem as eco
from sandbox.sim.water import WaterSim

SIZE = (1.0, 0.75)
SHAPE = (60, 80)
CELL = (SIZE[0] / SHAPE[1], SIZE[1] / SHAPE[0])


def grid():
    gh, gw = SHAPE
    x, y = np.meshgrid((np.arange(gw) + 0.5) * CELL[0], (np.arange(gh) + 0.5) * CELL[1])
    return x, y


def test_biomes_follow_height_and_slope():
    p = eco.BiomeParams()
    h = np.array([[-0.05, p.water_level + 0.003, 0.0, 0.03, 0.06, 0.1]], np.float32)
    flat = np.zeros_like(h)
    assert eco.classify_biomes(h, flat, p).tolist() == [[eco.WATER, eco.BEACH, eco.MEADOW, eco.FOREST, eco.ROCK,
                                                          eco.SNOW]]
    steep = np.full_like(h, 0.8)
    b = eco.classify_biomes(np.array([[0.0]], np.float32), steep[:, :1], p)
    assert b[0, 0] == eco.ROCK  # strmá stěna je skála i v nížině


def test_slope_map_of_plane():
    x, _ = grid()
    s = eco.slope_map((0.2 * x).astype(np.float32), CELL)
    assert np.allclose(s, 0.2, atol=1e-4)


def test_priority_flood_fills_pit_to_rim():
    x, y = grid()
    r = np.hypot(x - 0.5, y - 0.375)
    h = np.where(r < 0.15, -0.03, 0.0) + 0.02 * np.clip(r - 0.15, 0, 1)  # jáma uprostřed
    filled = eco.priority_flood(h)
    assert np.all(filled >= h - 1e-12)
    pit = r < 0.1
    rim = min(h[0].min(), h[-1].min(), h[:, 0].min(), h[:, -1].min())  # nejnižší přeliv (okraj bedny)
    assert filled[pit].min() == pytest.approx(rim, abs=1e-3)
    assert np.ptp(filled[pit]) < 1e-3  # rovná hladina jezera
    # z každé buňky vede cesta k okraji
    rec = eco.d8_receivers(filled)
    path = eco.trace_downhill(rec, int(np.argmin(np.where(pit, h, 1).ravel())))
    iy, ix = divmod(path[-1], SHAPE[1])
    assert iy in (0, SHAPE[0] - 1) or ix in (0, SHAPE[1] - 1)


def test_mfd_accumulation_collects_rain_in_valley():
    x, y = grid()
    h = (0.15 * np.abs(x - 0.5) + 0.1 * (0.75 - y)).astype(np.float32)  # údolí ve tvaru V, spád k y = 0.75
    filled = eco.priority_flood(h)
    acc = eco.mfd_accumulation(eco.mfd_fractions(filled), np.ones(SHAPE))
    row = acc[-2]
    assert abs(int(np.argmax(row)) - SHAPE[1] // 2) <= 1  # nejvíc vody teče středem údolí
    # vnitřní buňky nic neztrácí: tok dna údolí ~ součet všech buněk nad ním
    assert row.max() > 0.3 * SHAPE[0] * SHAPE[1] / 2


def test_hydrology_river_in_valley_and_lake_in_pit():
    gh, gw = 128, 192
    x, y = np.meshgrid((np.arange(gw) + 0.5) / gw * 1.0, (np.arange(gh) + 0.5) / gh * 0.75)
    h = 0.12 * np.abs(x - 0.5) + 0.06 * (0.75 - y)  # hory po stranách, údolí uprostřed
    h -= 0.04 * np.exp(-((x - 0.25) ** 2 + (y - 0.2) ** 2) / 0.002)  # jáma -> jezero
    hy = eco.analyze_hydrology(h.astype(np.float32), 2, -0.5, 1.0 / gw, rain_from=0.0, rain_span=0.05,
                               river_threshold=6.0)
    ch, cw = hy.coarse.shape
    assert hy.river[ch - 6, cw // 2 - 1:cw // 2 + 2].max() > 0.35  # řeka dole v údolí
    assert hy.river[ch - 6, :5].max() == 0  # na svahu ne
    assert hy.flow_dir[1, ch - 6, cw // 2] > 0.7  # teče po proudu (+y)
    lake_y, lake_x = int(0.2 / 0.75 * ch), int(0.25 * cw)
    assert hy.lake[lake_y, lake_x] > 0.01
    tex = hy.texture_data(-0.5)
    assert tex.shape == (ch, cw, 4) and tex.dtype == np.float32
    assert hy.wet_mask((gh, gw)).shape == (gh, gw)


def forest_world():
    h = np.full(SHAPE, 0.03, np.float32)  # celé území je les
    slope = np.zeros(SHAPE, np.float32)
    biome = eco.classify_biomes(h, slope, eco.BiomeParams())
    return h, slope, biome


def test_trees_grow_and_die_when_buried_or_flooded():
    h, slope, biome = forest_world()
    wet = np.zeros(SHAPE, bool)
    veg = eco.Vegetation(SIZE, max_trees=80, seed=1)
    for _ in range(90):
        veg.spawn(1 / 30, h, biome, wet, slope)
        veg.step(1 / 30)
    n = len(veg)
    assert 30 <= n <= 80
    r0 = veg.radii().copy()
    for _ in range(60):
        veg.step(1 / 30)
    assert np.all(veg.radii() >= r0 - 1e-9)  # rostou
    d = np.hypot(veg.pos[:, None, 0] - veg.pos[None, :, 0], veg.pos[:, None, 1] - veg.pos[None, :, 1])
    np.fill_diagonal(d, 1)
    assert d.min() > 0.5 * veg.radius  # koruny se nepřekrývají
    # levou polovinu zasypeme, pravou zaplaví řeka
    h2 = h.copy()
    h2[:, :SHAPE[1] // 2] += 0.03
    wet[:, SHAPE[1] // 2:] = True
    killed = veg.check(h2, biome, wet)
    assert killed == n
    for _ in range(int(veg.DIE_TIME * 30) + 2):
        veg.step(1 / 30)
    assert len(veg) == 0


def test_trees_do_not_spawn_in_water():
    h = np.full(SHAPE, -0.05, np.float32)
    slope = np.zeros(SHAPE, np.float32)
    biome = eco.classify_biomes(h, slope, eco.BiomeParams())
    veg = eco.Vegetation(SIZE, seed=2)
    for _ in range(60):
        veg.spawn(1 / 30, h, biome, np.zeros(SHAPE, bool), slope)
    assert len(veg) == 0


def test_deer_stay_on_land_and_flee_from_hand():
    h, slope, biome = forest_world()
    biome = biome.copy()
    biome[:, : SHAPE[1] // 3] = eco.WATER  # vlevo jezero
    biome[:, SHAPE[1] // 3: SHAPE[1] // 2] = eco.MEADOW
    wet = np.zeros(SHAPE, bool)
    a = eco.Animals(SIZE, n_deer=6, n_birds=3, seed=3)
    for _ in range(300):
        a.step(1 / 30, biome, wet)
    iy, ix = eco._cells(a.deer, SHAPE, SIZE)
    assert np.all(a.allowed(biome, wet, iy, ix))
    pos, head = a.birds()
    assert pos.shape == (3, 2) and np.all(np.isfinite(head))
    # ruka těsně vedle srnky -> srnka utíká pryč
    hx, hy = a.deer[0] + [0.03, 0.0]
    before = a.deer[0].copy()
    for _ in range(10):
        a.step(1 / 30, biome, wet, [(hx, hy)])
    assert a.fleeing[0] or np.hypot(*(a.deer[0] - [hx, hy])) > 0.18
    assert np.hypot(*(a.deer[0] - [hx, hy])) >= np.hypot(*(before - [hx, hy])) - 1e-6


def slope_world():
    """Svah klesající od horního okraje (y = 0) dolů, s mělkou nížinou vpravo dole."""
    x, y = grid()
    h = 0.08 * (1 - y / SIZE[1]) - 0.02 * np.exp(-((x - 0.7) ** 2 + (y - 0.6) ** 2) / 0.02)
    return h.astype(np.float32)


def test_flood_source_on_high_edge_and_town_downstream():
    h = slope_world()
    rng = np.random.default_rng(0)
    sources = eco.flood_source_candidates(h, SIZE, rng, factor=1, n=3)
    assert sources[0][1] < 0.2  # nejlepší pramen je u vysokého horního okraje
    flow = 0.004 * SIZE[0] * SIZE[1] / 20
    planner = eco.FloodPlanner(h, SIZE, sources, flow, 20.0, factor=1, infiltration=0.0002)
    while not planner.step(100):
        pass
    res = planner.choose(rng, 0.08, -1.0)
    assert res is not None
    src, town = res
    assert town[1] > src[1] + 0.2  # město je po proudu
    gx, gy = int(town[0] / CELL[0]), int(town[1] / CELL[1])
    arr = planner.arrival[planner.sources.index(src)]
    assert np.isfinite(arr[gy, gx])  # voda tam bez hrází doteče


def test_town_layout_and_flood_detection():
    rng = np.random.default_rng(4)
    houses = eco.layout_houses((0.5, 0.4), 10, 0.025, SIZE, rng)
    assert len(houses) == 10 and houses[0].church
    for i, a in enumerate(houses):
        assert 0 < a.x < SIZE[0] and 0 < a.y < SIZE[1]
        for b in houses[i + 1:]:
            assert math.hypot(a.x - b.x, a.y - b.y) > 0.025 * 1.3
    depth = np.zeros(SHAPE, np.float32)
    assert not eco.flooded_houses(depth, houses, SIZE).any()
    h0 = houses[3]
    iy, ix = eco._cells(np.array([[h0.x, h0.y]]), SHAPE, SIZE)
    depth[iy[0], ix[0]] = 0.01
    flooded = eco.flooded_houses(depth, houses, SIZE)
    assert flooded[3] and flooded.sum() <= 2


def test_add_source_adds_exact_volume_and_sim_keeps_it():
    sim = WaterSim(SHAPE, CELL[0])
    eco.add_source(sim.depth, SIZE, (0.5, 0.1), 0.04, 1e-4)
    assert sim.volume == pytest.approx(1e-4, rel=1e-3)
    for _ in range(30):
        sim.advance(slope_world(), 1 / 30)
    assert sim.volume == pytest.approx(1e-4, rel=1e-2)
    eco.soak(sim.depth, 1.0)
    assert sim.volume == 0


@pytest.fixture(scope="module")
def host():
    moderngl = pytest.importorskip("moderngl")
    from sandbox.config import Settings
    from sandbox.render.host import RenderHost

    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - bez GPU / EGL
        pytest.skip(f"OpenGL není k dispozici: {exc}")
    h = RenderHost(ctx, Settings())
    return h


@pytest.mark.parametrize("game_id", ["ekosystem", "mesto"])
def test_games_run_headless(host, game_id):
    from sandbox.games import discover_games
    from sandbox.terrain.heightmap import HeightMap

    cls = {c.id: c for c in discover_games()}[game_id]
    assert cls.category in ("zážitek", "soutěž") and cls.intro and cls.keys
    h = slope_world()
    hm = HeightMap(h, SIZE, hands=np.zeros(SHAPE, bool), hand_points=[(0.5, 0.4)])
    host.update_heightmap(hm)
    game = cls(host)
    game.start()
    for _ in range(90):
        game.update(hm, 1 / 30)
    host.begin_canvas()
    game.render()
    assert game.status()
    game.reset()
    game.update(hm, 1 / 30)
    game.render()
    game.stop()


def test_mesto_round_flow(host):
    from sandbox.games.mesto import FLOOD, PREP, RESULT, TownGame
    from sandbox.terrain.heightmap import HeightMap

    hm = HeightMap(slope_world(), SIZE)
    host.update_heightmap(hm)
    g = TownGame(host)
    g.start()
    g.rng = np.random.default_rng(5)
    for _ in range(30):
        g.update(hm, 1 / 30)
    assert g.phase == PREP and len(g.houses) == g.N_HOUSES
    assert g.key("enter")
    assert g.phase == FLOOD
    for _ in range(int(g.FLOOD_TIME * 20) + 5):
        g.update(hm, 1 / 20)
        if g.phase != FLOOD:
            break
    assert g.phase == RESULT and g.last_score is not None
    assert g.last_score[1] == g.N_HOUSES
    assert g.last_score[0] < g.N_HOUSES  # bez hrází voda do města doteče
    first_town = g.town
    for _ in range(int(g.RESULT_TIME * 20) + 40):
        g.update(hm, 1 / 20)
    assert g.phase == PREP and g.town is not None and g.town != first_town  # nové kolo, město jinde
    g.stop()
