import datetime as dt
import json
import random
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from sandbox import share
from sandbox.config import Settings
from sandbox.games.hora import HighestMountainGame, Records, extreme_point
from sandbox.games.otisk import FORMS, FootprintGame, flattest_spot, landscape_name, render_map_image
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


# --- sdílení -----------------------------------------------------------------------------
@pytest.mark.parametrize("name,ok", [
    ("abc123.png", True), ("/abc123.html", True), ("abc_x-1.jpg", True),
    ("../config/settings.json", False), ("..%2Fsecret.png", False), ("a/b.png", False),
    ("abc.py", False), (".hidden.png", False), ("", False), ("x" * 70 + ".png", False),
])
def test_sanitize_name(name, ok):
    assert (share.sanitize_name(name) is not None) == ok


def test_qr_image_is_scannable_dark_on_light():
    url = "http://192.168.10.42:8765/k3x9qa"
    img = share.qr_image(url, 400)
    assert img.shape == (400, 400, 4) and img.dtype == np.uint8
    assert img[0, 0, 0] == 255 and img[..., 3].min() == 255  # bílý okraj, neprůhledné
    assert (img[..., 0] == 0).mean() > 0.2  # tmavé moduly
    text, *_ = cv2.QRCodeDetector().detectAndDecode(np.ascontiguousarray(img[..., :3]))
    assert text == url


def test_local_ip_is_ipv4():
    parts = share.local_ip().split(".")
    assert len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts)


def test_save_share_writes_png_and_escaped_page(tmp_path):
    img = np.zeros((20, 30, 3), np.uint8)
    img[..., 0] = 255
    sid = share.save_share(img, "Hory <b>&</b>", "podtitul", directory=tmp_path)
    assert (tmp_path / f"{sid}.png").is_file()
    page = (tmp_path / f"{sid}.html").read_text(encoding="utf-8")
    assert "Hory &lt;b&gt;&amp;&lt;/b&gt;" in page and f'src="{sid}.png"' in page
    back = cv2.imread(str(tmp_path / f"{sid}.png"))
    assert back.shape == (20, 30, 3) and back[0, 0, 2] == 255  # RGB -> BGR správně
    with pytest.raises(ValueError):
        share.save_share(img, "x", directory=tmp_path, share_id="../evil")


def test_prune_keeps_newest(tmp_path):
    import os
    for i in range(5):
        sid = share.save_share(np.zeros((4, 4, 3), np.uint8), "x", directory=tmp_path, share_id=f"s{i}")
        os.utime(tmp_path / f"{sid}.png", (1000 + i, 1000 + i))
    share.prune(tmp_path, keep=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s3.html", "s3.png", "s4.html", "s4.png"]


def test_share_server_serves_only_shares(tmp_path):
    (tmp_path / "shares").mkdir()
    (tmp_path / "secret.txt").write_text("tajné")
    sid = share.save_share(np.zeros((8, 8, 3), np.uint8), "Test", directory=tmp_path / "shares")
    srv = share.ShareServer(tmp_path / "shares", port=28765)
    base = f"http://127.0.0.1:{srv.port}"

    def get(path):
        try:
            with urllib.request.urlopen(base + path, timeout=5) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, b"", e.headers

    try:
        code, body, _ = get(f"/{sid}")
        assert code == 200 and b"<img" in body
        code, body, headers = get(f"/{sid}.png")
        assert code == 200 and body[:4] == b"\x89PNG" and headers["Content-Type"] == "image/png"
        code, _, headers = get(f"/{sid}.png?download=1")
        assert code == 200 and "attachment" in headers["Content-Disposition"]
        assert get("/")[0] == 200
        for bad in ["/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/secret.txt", "/nope.png"]:
            assert get(bad)[0] == 404, bad
        # obsazený port -> další volný
        srv2 = share.ShareServer(tmp_path / "shares", port=srv.port)
        assert srv2.port != srv.port
        srv2.stop()
    finally:
        srv.stop()


# --- otisk ---------------------------------------------------------------------------------
def test_landscape_name_matches_terrain():
    m = Settings().map
    rng = random.Random(1)
    mountains = landscape_name(hill(h=0.12), m.water_level, rng)
    assert mountains.split()[0] in FORMS["hory"] and len(mountains.split()) >= 3
    sea = np.full((50, 50), -0.05, np.float32)
    sea[20:30, 20:30] = 0.03
    assert landscape_name(sea, m.water_level, rng).split()[0] in sum([f.split() for f in FORMS["voda"]], [])
    assert landscape_name(np.zeros((40, 40), np.float32), m.water_level, rng).split()[0] in FORMS["rovina"]
    assert landscape_name(hill(), 0.0, random.Random(5)) == landscape_name(hill(), 0.0, random.Random(5))


def test_render_map_image():
    img = render_map_image(hill(), (1.78, 1.08), Settings().map, "Pohoří Zlatého Draka",
                           when=dt.datetime(2026, 11, 11), width=900)
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[1] == 900 and img.shape[2] == 3
    assert img.shape[0] > 900 * 1.08 / 1.78  # mapa + nadpis + zápatí
    assert img.std() > 20  # není prázdný


def test_flattest_spot_avoids_hill_and_hands():
    h = hill(cx=40, cy=48, s=400)
    iy, ix = flattest_spot(h, None, (30, 40))
    assert ix > 80  # rovina je vpravo od kopce
    hands = np.zeros_like(h, bool)
    hands[:, 100:] = True
    iy, ix = flattest_spot(h, hands, (30, 40))
    assert ix + 20 <= 100  # okno se nepřekrývá s rukama


def test_footprint_auto_capture_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(share, "SHARES_DIR", tmp_path)
    monkeypatch.setattr(share, "ensure_server", lambda: None)
    game = FootprintGame(FakeHost())
    game.start()
    hm = heightmap(hill())
    for _ in range(30 * 5):  # 5 s klidu – ještě nefotí
        game.update(hm, 1 / 30)
    assert game.state == "idle"
    for _ in range(30 * 4):  # po 8 s klidu bez rukou začne odpočet
        game.update(hm, 1 / 30)
    assert game.state == "countdown"
    for _ in range(30 * 4):
        game.update(hm, 1 / 30)
    assert game.state == "show" and game.share_id
    assert (tmp_path / f"{game.share_id}.png").is_file() and (tmp_path / f"{game.share_id}.html").is_file()
    assert game.url.endswith("/" + game.share_id) and game.name
    x, y, w, h = game.card
    assert w >= game.qr_px and 0 <= x and x + w <= 1280 and 0 <= y and y + h <= 777
    for _ in range(30 * 26):  # QR zmizí a stejnou krajinu už znovu nefotí
        game.update(hm, 1 / 30)
    assert game.state == "idle" and not game.dirty
    for _ in range(30 * 20):
        game.update(hm, 1 / 30)
    assert game.state == "idle"
    # změna terénu -> po klidu nová fotka
    game.update(heightmap(hill(cx=110)), 1 / 30)
    for _ in range(30 * 9):
        game.update(heightmap(hill(cx=110)), 1 / 30)
    assert game.state == "countdown"
    game.stop()


def test_footprint_waits_while_hands_and_key_triggers(tmp_path, monkeypatch):
    monkeypatch.setattr(share, "SHARES_DIR", tmp_path)
    monkeypatch.setattr(share, "ensure_server", lambda: None)
    game = FootprintGame(FakeHost())
    game.start()
    hands = np.zeros((96, 160), bool)
    hands[10:30, 10:30] = True
    hm = heightmap(hill(), hands)
    for _ in range(30 * 10):  # ruka nad pískem -> čeká (do STABLE_FORCE)
        game.update(hm, 1 / 30)
    assert game.state == "idle"
    assert game.key("p") and game.state == "countdown"
    assert game.key("f") and game.state == "countdown"
    assert not game.key("q")
    game.reset()
    assert game.state == "idle" and game.dirty


# --- nejvyšší hora ----------------------------------------------------------------------------
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
