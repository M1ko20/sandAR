import math

import numpy as np
import pytest

from sandbox.games import discover_games
from sandbox.games.kulicka import Sparks, ball_in_hole, pick_spot
from sandbox.games.poklad import hint_for, place_treasures, surface_level, update_reveals
from sandbox.sim.ball import Ball, BallPhysics, SlopeField, kick, step

CELL = 0.007
GH, GW = 100, 160
SIZE = (GW * CELL, GH * CELL)


def _grid():
    y, x = np.mgrid[0:GH, 0:GW]
    return (x + 0.5) * CELL, (y + 0.5) * CELL


def _run(balls, heights, seconds, phys=None):
    field = SlopeField(heights, CELL)
    for _ in range(int(seconds * 60)):
        step(balls, field, 1 / 60, phys)


# --- fyzika kuličky --------------------------------------------------------------------
def test_slope_field_gradient_matches_plane():
    X, Y = _grid()
    h = (0.1 * X - 0.05 * Y).astype(np.float32)
    gx, gy = SlopeField(h, CELL).gradient(0.5, 0.3)
    assert float(gx) == pytest.approx(0.1, abs=1e-3)
    assert float(gy) == pytest.approx(-0.05, abs=1e-3)


def test_ball_rolls_downhill():
    X, _ = _grid()
    h = (0.15 * X).astype(np.float32)  # svah stoupá doprava
    b = Ball(0.8, 0.35)
    _run([b], h, 0.5)
    assert b.x < 0.8 and b.vx < 0
    assert b.y == pytest.approx(0.35, abs=1e-3)


def test_ball_stays_on_flat_and_gentle_slope():
    X, _ = _grid()
    for h in (np.zeros((GH, GW), np.float32), (0.02 * X).astype(np.float32)):  # 2 % svah valivý odpor udrží
        b = Ball(0.5, 0.35)
        _run([b], h, 2.0)
        assert (b.x, b.y) == pytest.approx((0.5, 0.35), abs=1e-6)


def test_ball_settles_in_pit():
    X, Y = _grid()
    h = (-0.06 * np.exp(-((X - 0.56) ** 2 + (Y - 0.35) ** 2) / (2 * 0.12**2))).astype(np.float32)
    b = Ball(0.40, 0.30)
    _run([b], h, 30.0)
    assert math.hypot(b.x - 0.56, b.y - 0.35) < 0.03
    assert b.speed < 0.01


def test_ball_bounces_off_walls_and_stays_inside():
    b = Ball(0.05, 0.35, vx=-1.0, radius=0.02)
    phys = BallPhysics(damping=0.0, rolling_resistance=0.0)
    _run([b], np.zeros((GH, GW), np.float32), 0.2, phys)
    assert b.vx > 0 and b.x >= b.radius
    for _ in range(5):
        kick(b, b.x + 0.01, b.y - 0.01, 3.0)
        _run([b], np.zeros((GH, GW), np.float32), 0.5)
        assert b.radius <= b.x <= SIZE[0] - b.radius and b.radius <= b.y <= SIZE[1] - b.radius


def test_kick_pushes_away_from_hand():
    b = Ball(0.5, 0.3)
    kick(b, 0.45, 0.3, 0.8)
    assert b.vx == pytest.approx(0.8) and b.vy == pytest.approx(0.0)


def test_ball_in_hole_needs_slow_ball():
    assert ball_in_hole(Ball(0.5, 0.3, vx=0.2), (0.52, 0.3), 0.05, 0.9)
    assert not ball_in_hole(Ball(0.5, 0.3, vx=2.0), (0.52, 0.3), 0.05, 0.9)
    assert not ball_in_hole(Ball(0.5, 0.3), (0.6, 0.3), 0.05, 0.9)


def test_pick_spot_respects_margin_and_distance():
    rng = np.random.default_rng(1)
    h = np.zeros((GH, GW), np.float32)
    for _ in range(20):
        x, y = pick_spot(h, SIZE, rng, margin=0.1, away_from=(0.2, 0.35), min_dist=0.4)
        assert 0.1 <= x <= SIZE[0] - 0.1 and 0.1 <= y <= SIZE[1] - 0.1
        assert math.hypot(x - 0.2, y - 0.35) >= 0.4


def test_sparks_expire():
    s = Sparks()
    s.burst(0.5, 0.3, 50, 1.0, [(1, 1, 0)], life=1.0, rng=np.random.default_rng(0))
    assert len(s) == 50
    for _ in range(40):
        s.update(1 / 30)
    assert len(s) == 0


# --- poklad ----------------------------------------------------------------------------
def _hill():
    X, Y = _grid()
    return (0.10 * np.exp(-((X - 0.56) ** 2 + (Y - 0.35) ** 2) / (2 * 0.2**2))).astype(np.float32)


def test_treasures_are_buried_under_high_sand_and_spread_out():
    h = _hill()
    rng = np.random.default_rng(5)
    trs = place_treasures(h, SIZE, 5, rng, cover=(0.025, 0.04), min_dist=0.15)
    assert len(trs) == 5
    median = float(np.median(h))
    for i, tr in enumerate(trs):
        surf = surface_level(h, SIZE, tr.x, tr.y)
        assert 0.025 - 1e-6 <= surf - tr.z <= 0.04 + 1e-6  # musí se kopat
        assert surf >= median  # spíš na vyvýšených místech
        assert 0.1 <= tr.x <= SIZE[0] - 0.1 and 0.1 <= tr.y <= SIZE[1] - 0.1
        for other in trs[i + 1:]:
            assert math.hypot(tr.x - other.x, tr.y - other.y) >= 0.15 - 1e-9


def test_treasures_avoid_forbidden_cells():
    h = _hill()
    forbid = np.zeros_like(h, bool)
    forbid[:, : GW // 2] = True
    trs = place_treasures(h, SIZE, 4, np.random.default_rng(2), forbid=forbid)
    assert all(tr.x > SIZE[0] / 2 - CELL for tr in trs)


def test_digging_reveals_treasure_after_hold_time():
    h = _hill()
    trs = place_treasures(h, SIZE, 3, np.random.default_rng(3))
    tr = trs[0]
    assert update_reveals(trs, h, SIZE, 1.0, 0.0) == []  # nekopáno
    X, Y = _grid()
    dug = np.where(np.hypot(X - tr.x, Y - tr.y) < 0.05, tr.z - 0.005, h).astype(np.float32)
    assert update_reveals(trs, dug, SIZE, 0.1, 1.0) == []  # musí vydržet (šum)
    found = update_reveals(trs, dug, SIZE, 0.1, 1.1) + update_reveals(trs, dug, SIZE, 0.15, 1.25)
    assert found == [tr] and tr.found and not trs[1].found


def test_single_noisy_cell_does_not_reveal():
    h = _hill()
    trs = place_treasures(h, SIZE, 1, np.random.default_rng(4))
    tr = trs[0]
    noisy = h.copy()
    noisy[int(tr.y / CELL), int(tr.x / CELL)] = -0.2  # díra v datech Kinectu
    for k in range(10):
        update_reveals(trs, noisy, SIZE, 0.1, k * 0.1)
    assert not tr.found


def test_hint_words_get_hotter():
    words = [hint_for(d)[0] for d in (0.5, 0.3, 0.15, 0.05)]
    assert words == ["Zima", "Přihořívá", "Teplo", "Hoří!"]


def test_games_registered():
    games = {g.id: g for g in discover_games()}
    assert games["kulicka"].category == "hra" and games["kulicka"].order == 60
    assert games["poklad"].category == "soutěž" and games["poklad"].order == 65
    for gid in ("kulicka", "poklad"):
        assert games[gid].intro and games[gid].keys
