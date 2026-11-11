"""Hry pro dva hráče: Vodní válka, Král kopce, Potopa ostrovů, Modeluj podle předlohy."""

import numpy as np
import pytest

from sandbox.config import Settings
from sandbox.games import discover_games, versus as vs
from sandbox.games.kral import KingOfHillGame
from sandbox.games.modeluj import ModelRaceGame
from sandbox.games.potopa import IslandFloodGame
from sandbox.games.targets import BUILTIN, evaluate, evaluate_shape, half_grid
from sandbox.games.valka import WaterWarGame, mirrored_villages
from sandbox.terrain.heightmap import HeightMap

SIZE = (1.73, 1.09)
SHAPE = (100, 160)


class FakeTexture:
    def __init__(self, size):
        self.size = size

    def write(self, data):
        pass

    def release(self):
        pass


class FakeHost:
    """Jen to, co hry potřebují v update() – bez OpenGL."""

    def __init__(self):
        self.settings = Settings()
        self.canvas_size = (1280, 806)
        self.box_size = SIZE
        self.messages = []
        self.heightmap = None

    def notify(self, text, seconds=3.0):
        self.messages.append(text)

    def program(self, src):
        return FakeTexture((0, 0))

    def float_texture(self, size, data=None):
        return FakeTexture(size)


def hm_of(heights, hands=None):
    return HeightMap(heights=np.asarray(heights, np.float32), size=SIZE, hands=hands)


def bump(cx, cy, h, s=8.0):
    y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return (h * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))).astype(np.float32)


def run(game, hm, seconds, dt=1 / 30, until=None):
    for _ in range(int(round(seconds / dt))):
        game.update(hm, dt)
        if until is not None and until(game):
            return


def test_new_games_are_discovered_after_existing_ones():
    ids = [g.id for g in discover_games()]
    assert ids[:5] == ["topo", "mapa", "vyska", "predloha", "voda"]
    assert ids[-4:] == ["valka", "kral", "potopa", "modeluj"]
    assert len(ids) <= 19  # všechny mají klávesu (1–9, 0, Shift+1–9)


# --- společné ------------------------------------------------------------------------------
def test_zone_masks_split_long_side_with_neutral_strip():
    a = vs.zone_mask(SHAPE, 0, border=2, neutral=3)
    b = vs.zone_mask(SHAPE, 1, border=2, neutral=3)
    assert not (a & b).any()
    assert a[:, :77].any() and not a[:, 77:].any()
    assert b[:, 83:].any() and not b[:, :83].any()
    assert a.sum() == b.sum()


def test_zone_peak_ignores_hands():
    h = bump(40, 50, 0.08) + bump(120, 50, 0.05)
    hands = np.zeros(SHAPE, bool)
    hands[45:55, 35:45] = True
    left = vs.zone_mask(SHAPE, 0)
    iy, ix, v = vs.zone_peak(h, None, left)
    assert (iy, ix) == (50, 40) and v == pytest.approx(0.08, abs=0.005)
    assert vs.zone_peak(h, hands, left)[2] < 0.05


# --- Modeluj podle předlohy --------------------------------------------------------------------
def test_half_targets_face_each_player():
    saddle = BUILTIN[2]  # Dva vrcholy a sedlo: vyšší vrchol vlevo z pohledu hráče
    left = half_grid(saddle, (SHAPE[0], SHAPE[1] // 2), SIZE, 0.1, 0)
    right = half_grid(saddle, (SHAPE[0], SHAPE[1] // 2), SIZE, 0.1, 1)
    np.testing.assert_allclose(right, left[::-1, ::-1])
    # hráč 0 stojí u x = 0, jeho levá ruka míří k y = 0 -> vyšší vrchol v horní polovině
    assert left[: SHAPE[0] // 2].max() > left[SHAPE[0] // 2:].max()


def test_shape_scoring_is_much_more_lenient():
    target = BUILTIN[0].grid(SHAPE, SIZE, 0.10)
    half = target * 0.5 + 0.02
    strict, _ = evaluate(half, target, 0.015)
    mild, _ = evaluate_shape(half, target)
    assert strict.score < 55 and mild.score > 80
    assert evaluate_shape(target, target)[0].score == pytest.approx(100, abs=0.5)
    assert evaluate_shape(np.zeros_like(target), target)[0].score < 5
    noisy = target + np.random.default_rng(0).normal(0, 0.006, target.shape).astype(np.float32)
    assert evaluate_shape(noisy, target)[0].score > evaluate(noisy, target, 0.015)[0].score


def test_solo_finishes_with_rough_copy():
    g = ModelRaceGame(FakeHost())
    g.start()
    g.players = 1
    g.reset()
    flat = hm_of(np.zeros(SHAPE))
    g.update(flat, 1 / 30)
    g.start_round()
    rough = hm_of(g.grid * 0.6 + np.random.default_rng(1).normal(0, 0.004, SHAPE))
    run(g, rough, 3.0)
    assert g.state == "výsledek" and g.winner == 0


def test_duel_winner_is_player_with_matching_half():
    g = ModelRaceGame(FakeHost())
    g.start()
    assert g.players == 2
    g.update(hm_of(np.zeros(SHAPE)), 1 / 30)
    g.start_round()
    terrain = np.zeros(SHAPE, np.float32)
    terrain[:, SHAPE[1] // 2:] = g.grid[:, SHAPE[1] // 2:]  # modří to mají přesně
    run(g, hm_of(terrain), 3.0)
    assert g.state == "výsledek" and g.winner == 1


# --- Král kopce --------------------------------------------------------------------------------
def test_king_of_hill_points_go_to_higher_zone():
    g = KingOfHillGame(FakeHost())
    g.start()
    g.start_round()
    run(g, hm_of(bump(40, 50, 0.06) + bump(120, 50, 0.09)), 10)
    assert g.leader == 1
    assert g.points[1] == pytest.approx(10, abs=0.2) and g.points[0] == 0
    run(g, hm_of(bump(40, 50, 0.10) + bump(120, 50, 0.09)), 55)
    assert g.state == "výsledek" and g.winner == 0  # 50 s jako král > 10 s


def test_king_of_hill_ignores_pile_on_the_divider():
    g = KingOfHillGame(FakeHost())
    g.start()
    # úzká hromádka přímo na čáře (svahy nezasahují za neutrální pás) se nepočítá nikomu
    g.update(hm_of(bump(80, 50, 0.12, s=1.2) + bump(40, 50, 0.04) + bump(120, 50, 0.03)), 1 / 30)
    assert g.leader == 0


# --- Potopa ostrovů --------------------------------------------------------------------------
def test_lower_island_sinks_first():
    g = IslandFloodGame(FakeHost())
    g.start()
    hm = hm_of(bump(40, 50, 0.05, s=10) + bump(120, 50, 0.10, s=10))
    g.start_build()
    g.start_rise(hm)
    assert g.sea < 0.05 - 0.01  # na začátku má ostrov každý
    run(g, hm, 120, until=lambda g: g.state == "výsledek")
    assert g.state == "výsledek" and g.winner == 1 and g.sunk == [True, False]


# --- Vodní válka -----------------------------------------------------------------------------
def test_villages_are_mirrored():
    left, right = mirrored_villages(SIZE, 5, 0.04, np.random.default_rng(3))
    assert len(left) == len(right) == 5
    for a, b in zip(left, right):
        assert a.x < SIZE[0] / 2 < b.x
        assert (a.x + b.x, a.y + b.y) == pytest.approx(SIZE)


def test_water_war_floods_village_in_the_lower_half():
    g = WaterWarGame(FakeHost())
    g.start()
    x = np.linspace(-1, 1, SHAPE[1], dtype=np.float32)
    terrain = np.repeat((0.03 * x)[None, :], SHAPE[0], 0)  # celé pískoviště se svažuje k červeným
    hm = hm_of(terrain)
    g.update(hm, 1 / 30)
    g._start_flood()
    run(g, hm, 120, dt=1 / 20, until=lambda g: g.phase == "výsledek")
    assert g.phase == "výsledek"
    assert g.dry(0) == 0 and g.dry(1) == 5 and g.winner == 1
    assert g.score == [0, 1]
