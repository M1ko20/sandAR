import datetime as dt
import json
import random

import cv2
import numpy as np
import pytest

from sandbox.config import Settings
from sandbox.games.hora import HighestMountainGame, Records, extreme_point
from sandbox.terrain.heightmap import HeightMap


# --- pomocníci -------------------------------------------------------------------------
class FakeTexture:
    def __init__(self, size):
        self.size = size
        self.filter = None
        self.released = False

    def release(self):
        self.released = True


class FakeCtx:
    NEAREST = LINEAR = 0

    def texture(self, size, components, data=None, **kw):
        return FakeTexture(size)


class FakeHost:
    """Jen to, co hry potřebují v update() – bez OpenGL."""

    def __init__(self):
        self.settings = Settings()
        self.canvas_size = (1280, 777)
        self.ctx = FakeCtx()
        self.messages = []
        self.time = 0.0

    def altitude(self, h):
        m = self.settings.map
        return m.base_altitude + h * m.vertical_scale

    def notify(self, text, seconds=3.0):
        self.messages.append(text)

    def program(self, src):
        return FakeTexture((0, 0))


def hill(gh=96, gw=160, h=0.08, cx=40, cy=48, s=200.0):
    y, x = np.mgrid[0:gh, 0:gw]
    return (h * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s)).astype(np.float32)


def heightmap(heights, hands=None):
    return HeightMap(heights=heights, size=(1.78, 1.08), hands=hands)


def test_extreme_point_ignores_hands_and_border():
    h = hill(cx=40) + hill(h=0.12, cx=110)
    iy, ix, v = extreme_point(h, None)
    assert abs(ix - 110) <= 1 and v > 0.1
    hands = np.zeros_like(h, bool)
    hands[30:70, 95:125] = True  # ruka nad vyšším kopcem -> počítá se ten druhý
    iy, ix, v = extreme_point(h, hands)
    assert abs(ix - 40) <= 1 and 0.07 < v < 0.09
    pit = -hill(h=0.05, cx=80)
    pit[:, :2] = -0.2  # okraj se nepočítá
    iy, ix, v = extreme_point(pit, None, pit=True)
    assert abs(ix - 80) <= 1 and v < -0.04


def test_records_top5_record_and_atomic(tmp_path):
    path = tmp_path / "records.json"
    r = Records(path)
    now = dt.datetime(2026, 11, 11, 10, 0)
    assert r.submit("hora", 0.05, 550, now) == (True, 1)
    assert r.submit("hora", 0.03, 450, now) == (False, 2)
    assert r.submit("hora", 0.08, 700, now) == (True, 1)
    for s in (0.01, 0.02, 0.025):
        r.submit("hora", s, 300 + s * 5000, now)
    assert r.submit("hora", 0.001, 305, now) == (False, None)  # mimo TOP 5
    r2 = Records(path)
    assert r2.record("hora")["alt"] == 700
    assert [e["alt"] for e in r2.today("hora", "2026-11-11")] == [700, 550, 450, 425, 400]
    assert r2.today("hora", "2026-11-12") == [] and r2.record("propast") is None
    assert not list(tmp_path.glob("*.tmp"))
    path.write_text("{rozbité")
    assert Records(path).record("hora") is None  # poškozený soubor hru neshodí


def test_mountain_round_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(HighestMountainGame, "records_path", tmp_path / "records.json")
    game = HighestMountainGame(FakeHost())
    game.start()
    hands = np.zeros((96, 160), bool)
    hands[70:90, 120:150] = True
    for _ in range(30):  # ruka nad pískem -> kolo začne
        game.update(heightmap(hill(h=0.02), hands), 1 / 30)
    assert game.state == "round"
    for _ in range(30 * 58):
        game.update(heightmap(hill(h=0.06)), 1 / 30)
    assert game.state == "round"
    for _ in range(30 * 3):
        game.update(heightmap(hill(h=0.06)), 1 / 30)
    assert game.state == "result" and game.is_record and game.rank == 1
    assert game.final == pytest.approx(0.06, abs=0.005)
    assert game.confetti is not None
    data = json.loads((tmp_path / "records.json").read_text())
    assert data["hora"]["record"]["alt"] == pytest.approx(600, abs=25)
    for _ in range(30 * 13):
        game.update(heightmap(hill(h=0.06)), 1 / 30)
    assert game.state == "ready"
    # druhé kolo horší -> není rekord, 2. místo
    game.key("enter")
    for _ in range(30 * 61):
        game.update(heightmap(hill(h=0.04)), 1 / 30)
    assert game.state == "result" and not game.is_record and game.rank == 2
    assert any("rekord" in s.lower() for s in game.status())


def test_pit_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(HighestMountainGame, "records_path", tmp_path / "records.json")
    game = HighestMountainGame(FakeHost())
    game.start()
    assert game.key("v") and game.variant == "propast"
    game.key("space")
    pit = hill(h=0.05) - hill(h=0.04, cx=110)
    for _ in range(30 * 61):
        game.update(heightmap(pit), 1 / 30)
    assert game.state == "result" and game.final == pytest.approx(-0.04, abs=0.005)
    rec = Records(tmp_path / "records.json").record("propast")
    assert rec["score"] == pytest.approx(0.04, abs=0.005) and rec["alt"] == pytest.approx(100, abs=25)
    assert game.key("x") is True
    game.key("v")
    assert game.variant == "hora" and game.state == "ready"
