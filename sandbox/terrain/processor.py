"""Převod hloubkových snímků na stabilní výškovou mapu.

Kroky pro každý snímek:

1. pixely -> 3D body -> (x, y, h) v souřadnicích bedny (přesná perspektiva Kinectu),
2. rozprostření do mřížky (průměr bodů v buňce),
3. vyřazení rukou: pixely, které detektor rukou (``hands.py``) označí jako paži/ruku, se do
   výšky vůbec nezapočítají – písek pod rukou zůstane, jak byl,
4. časový filtr: malé změny = šum. Ty se berou z klouzavého okna posledních N snímků
   (průměr bez nejvyšší a nejnižší hodnoty) a změny pod ``hysteresis`` se ignorují,
   aby se vrstevnice na klidném písku nechvěly. Velké změny se přijmou až když
   vydrží několik snímků (ruka, která jen projíždí, se do terénu nepropíše) –
   okno je nezdržuje, po přijetí se na novou výšku přenastaví,
5. lehké prostorové vyhlazení.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..config import HandSettings, TerrainSettings
from .calibration import KinectCalibration
from .hands import Hand, HandDetector
from .heightmap import HeightMap


def fill_holes(values: np.ndarray, valid: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Doplní neplatné buňky z okolí (normalizovaná konvoluce v rostoucích měřítkách)."""
    out = np.where(valid, values, 0).astype(np.float32)
    weight = valid.astype(np.float32)
    missing = ~valid
    sigma = 1.0
    while missing.any() and sigma < max(values.shape):
        num = cv2.GaussianBlur(out * weight, (0, 0), sigma)
        den = cv2.GaussianBlur(weight, (0, 0), sigma)
        ok = missing & (den > 1e-3)
        out[ok] = num[ok] / den[ok]
        weight[ok] = 1.0
        missing &= ~ok
        sigma *= 2
    out[missing] = default
    return out


def _replicate_edges(a: np.ndarray, ex: int, ey: int) -> None:
    """Okrajový pás nahradí nejbližšími vnitřními hodnotami (na místě) a podél stěny ho
    rozmaže – jinak by kopírování jednoho sloupce kreslilo pruhy kolmo ke stěně."""
    if ey:
        a[:ey, :] = a[ey, :]
        a[-ey:, :] = a[-ey - 1, :]
    if ex:
        a[:, :ex] = a[:, ex:ex + 1]
        a[:, -ex:] = a[:, -ex - 1:-ex]
    if ex or ey:
        blur = cv2.GaussianBlur(a, (0, 0), 3.0)
        if ey:
            a[:ey, :] = blur[:ey, :]
            a[-ey:, :] = blur[-ey:, :]
        if ex:
            a[:, :ex] = blur[:, :ex]
            a[:, -ex:] = blur[:, -ex:]


class TerrainProcessor:
    def __init__(self, calibration: KinectCalibration, settings: TerrainSettings,
                 hand_settings: HandSettings | None = None,
                 outer: tuple[float, float, float, float] | None = None) -> None:
        self.cal = calibration
        self.s = settings
        hs = hand_settings if hand_settings is not None else HandSettings()
        self.hand_detector = HandDetector(calibration, hs, outer) if hs.enabled else None
        self._last_t: float | None = None
        W, D = calibration.size
        self.size = (W, D)
        self.gw = int(settings.grid_width)
        self.gh = max(8, int(round(self.gw * D / W)))

        rays = calibration.intrinsics.rays().reshape(-1, 3).astype(np.float64)
        # Pracujeme jen s pixely, které mohou dopadnout do bedny (ušetří ~polovinu práce).
        keep = np.zeros(len(rays), bool)
        rn = rays @ calibration.normal
        margin = 0.05
        for hh in (settings.min_height, 0.0, settings.max_height + 0.1):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (hh - calibration.offset) / rn
            x, y, _ = calibration.camera_to_box(rays * t[:, None])
            keep |= (t > 0) & (x > -margin) & (x < W + margin) & (y > -margin) & (y < D + margin)
        self.pixel_index = np.flatnonzero(keep)
        r = rays[keep]
        self._rn = (r @ calibration.normal).astype(np.float32)
        self._ra = (r @ calibration.ex).astype(np.float32)
        self._rb = (r @ calibration.ey).astype(np.float32)
        self._oa = float(calibration.origin @ calibration.ex)
        self._ob = float(calibration.origin @ calibration.ey)
        self._H = calibration.plane_to_box.astype(np.float32)

        cw, ch = W / self.gw, D / self.gh
        self.edge = (min(self.gw // 4, int(np.ceil(settings.edge_margin / cw))),
                     min(self.gh // 4, int(np.ceil(settings.edge_margin / ch))))
        self.state: np.ndarray | None = None
        self._cand = np.zeros((self.gh, self.gw), np.float32)
        self._count = np.zeros((self.gh, self.gw), np.int16)
        self.window = max(1, int(settings.temporal_window))
        self._buf: np.ndarray | None = None  # (N, GH, GW) poslední měření, kruhově
        self._buf_i = 0
        self.last_coverage = 0.0
        self.last_ms = 0.0

    def reset(self) -> None:
        self.state = None
        self._buf = None
        if self.hand_detector is not None:
            self.hand_detector.reset()

    def _window_filter(self, meas: np.ndarray, use: np.ndarray) -> np.ndarray:
        """Klouzavý průměr posledních N měření buňky bez nejvyšší a nejnižší hodnoty.

        Buňka, kde ``use`` není, převezme do okna svou předchozí hodnotu."""
        n = self.window
        if n < 3:
            return meas
        buf = self._buf
        if buf is None:
            buf = self._buf = np.repeat(np.where(use, meas, self.state)[None], n, axis=0)
            self._buf_i = 0
        else:
            prev = buf[self._buf_i]
            self._buf_i = (self._buf_i + 1) % n
            buf[self._buf_i] = np.where(use, meas, prev)
        return (buf.sum(axis=0) - buf.max(axis=0) - buf.min(axis=0)) / (n - 2)

    # ------------------------------------------------------------------------
    def measure(self, depth: np.ndarray, hand_px: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Jeden snímek -> (průměrná výška v buňce, platnost, maska buněk s rukou).

        ``hand_px`` = pixely ruky (z detektoru): do výšky písku se nepočítají, jen do masky rukou."""
        z = depth.reshape(-1)[self.pixel_index]
        ok = z > 0
        hand = None
        if hand_px is not None:
            hand = hand_px.reshape(-1)[self.pixel_index][ok]
        z = z[ok]
        h = z * self._rn[ok] + self.cal.offset
        a = z * self._ra[ok] - self._oa
        b = z * self._rb[ok] - self._ob
        H = self._H
        w = H[2, 0] * a + H[2, 1] * b + H[2, 2]
        x = (H[0, 0] * a + H[0, 1] * b + H[0, 2]) / w
        y = (H[1, 0] * a + H[1, 1] * b + H[1, 2]) / w

        W, D = self.size
        ix = np.floor(x * (self.gw / W)).astype(np.int32)
        iy = np.floor(y * (self.gh / D)).astype(np.int32)
        inside = (ix >= 0) & (ix < self.gw) & (iy >= 0) & (iy < self.gh)
        cell = iy[inside] * self.gw + ix[inside]
        h = h[inside]

        n = self.gw * self.gh
        sand = (h >= self.s.min_height) & (h <= self.s.max_height)
        if hand is not None:
            hand = hand[inside]
            high = np.bincount(cell[hand], minlength=n)
            sand &= ~hand
        else:
            high = np.bincount(cell[h > self.s.max_height], minlength=n)
        counts = np.bincount(cell[sand], minlength=n)
        sums = np.bincount(cell[sand], weights=h[sand], minlength=n)

        valid = counts > 0
        mean = np.zeros(n, np.float32)
        mean[valid] = sums[valid] / counts[valid]
        shape = (self.gh, self.gw)
        return mean.reshape(shape), valid.reshape(shape), (high >= (1 if hand is not None else 2)).reshape(shape)

    def process(self, depth: np.ndarray, seq: int = 0, timestamp: float | None = None) -> HeightMap:
        t0 = time.perf_counter()
        now = time.monotonic() if timestamp is None else timestamp
        dt = 1 / 30 if self._last_t is None else float(np.clip(now - self._last_t, 1e-3, 0.5))
        self._last_t = now
        hand_list: list[Hand] = []
        hand_px = None
        if self.hand_detector is not None:
            hand_list, hand_px = self.hand_detector.process(depth, dt, self.state)
        meas, valid, high = self.measure(depth, hand_px)
        self.last_coverage = float(valid.mean())
        s = self.s

        ex, ey = self.edge
        if ex or ey:  # u stěn bedny neměříme písek, ale stěnu
            valid[:ey, :] = valid[self.gh - ey:, :] = False
            valid[:, :ex] = valid[:, self.gw - ex:] = False
            high[:ey, :] = high[self.gh - ey:, :] = False
            high[:, :ex] = high[:, self.gw - ex:] = False

        if self.state is None:
            self.state = fill_holes(meas, valid)
            self._cand[:] = self.state
            self._count[:] = 0

        state, cand, count = self.state, self._cand, self._count
        diff = meas - state
        small = valid & (np.abs(diff) < s.noise_threshold)
        # Do okna jdou jen malé změny – ruka ani zákmit se do průměru nedostane.
        filt = self._window_filter(meas, small)
        fdiff = filt - state
        move = small & (np.abs(fdiff) > s.hysteresis)  # mrtvé pásmo proti chvění vrstevnic
        state[move] += s.smoothing * fdiff[move]
        count[small] = 0

        big = valid & ~small
        same = big & (np.abs(meas - cand) < s.noise_threshold * 1.5)
        cand[same] += 0.5 * (meas[same] - cand[same])
        count[same] += 1
        fresh = big & ~same
        cand[fresh] = meas[fresh]
        count[fresh] = 1

        # Náhlé zvýšení o víc než hand_margin je spíš ruka – chceme delší potvrzení.
        needed = np.where(cand - state > s.hand_margin, 2 * s.stable_frames, s.stable_frames)
        commit = count >= needed
        state[commit] = cand[commit]
        count[commit] = 0
        if self._buf is not None and commit.any():
            self._buf[:, commit] = cand[commit]  # okno začne od nové výšky

        _replicate_edges(state, ex, ey)
        if self.hand_detector is not None:
            hands = high
        else:  # bez detektoru: ruka = co je vysoko nebo náhle vystoupí nad písek
            hands = high | (valid & (meas - state > s.hand_margin) & ~commit)
        if hands.any():
            hands = cv2.dilate(hands.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        points = [h.pos for h in hand_list] if self.hand_detector is not None else self.hand_points(hands)

        out = cv2.GaussianBlur(state, (0, 0), s.spatial_sigma) if s.spatial_sigma > 0 else state.copy()
        self.last_ms = (time.perf_counter() - t0) * 1000
        return HeightMap(out, self.size, hands, hand_points=points, hand_list=hand_list, seq=seq,
                         timestamp=time.monotonic())

    def hand_points(self, hands: np.ndarray, min_cells: int = 12) -> list[tuple[float, float]]:
        """Středy souvislých oblastí rukou (v metrech), od největší."""
        if not hands.any():
            return []
        n, _, stats, cent = cv2.connectedComponentsWithStats(hands.astype(np.uint8), connectivity=8)
        cw, ch = self.size[0] / self.gw, self.size[1] / self.gh
        blobs = [(int(stats[i, cv2.CC_STAT_AREA]), (float(cent[i, 0] + 0.5) * cw, float(cent[i, 1] + 0.5) * ch))
                 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_cells]
        return [p for _, p in sorted(blobs, reverse=True)]
