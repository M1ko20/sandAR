"""Simulovaný Kinect nad virtuálním pískovištěm.

Slouží k vývoji a testování her bez hardwaru. Simulace vrhá paprsky z
virtuální kamery (s mírným náklonem, jako v reálu) na výškové pole, které obsahuje
písek, okraje bedny a stůl kolem. Do snímku přidává šum podobný Kinectu.
Písek jde „hrabat“ myší v operátorském okně.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from .base import KINECT2_DEFAULT_INTRINSICS, DepthSensor


def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class SimulatedSensor(DepthSensor):
    name = "sim"
    supports_interaction = True

    BOX_SIZE = (1.0, 0.75)  # [m] vnitřní rozměr bedny
    WALL_HEIGHT = 0.07  # [m] okraj nad úrovní uhlazeného písku
    WALL_THICKNESS = 0.025
    FLOOR_HEIGHT = -0.25  # stůl kolem bedny
    SAND_LIMITS = (-0.12, 0.20)
    RES = 0.004  # [m] rozlišení výškového pole

    def __init__(self, seed: int = 7, fps: float = 30.0, noise: float = 0.0015, camera_height: float = 1.15) -> None:
        super().__init__()
        self.intrinsics = KINECT2_DEFAULT_INTRINSICS
        self.fps = fps
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

        W, D = self.BOX_SIZE
        self.camera_pos = np.array([W / 2 + 0.03, D / 2 - 0.02, camera_height])
        # kamera míří dolů: osa x obrazu = +x světa, osa y obrazu = +y světa, optická osa = -z
        base = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]]).T
        self.cam_to_world = _rotation(math.radians(2.5), math.radians(-1.5), math.radians(1.2)) @ base

        self._x0, self._y0 = -0.45, -0.45
        self._nx = int(round((W + 0.9) / self.RES))
        self._ny = int(round((D + 0.9) / self.RES))
        xs = self._x0 + (np.arange(self._nx) + 0.5) * self.RES
        ys = self._y0 + (np.arange(self._ny) + 0.5) * self.RES
        self._gx, self._gy = np.meshgrid(xs, ys)
        t = self.WALL_THICKNESS
        self._inside = (self._gx >= 0) & (self._gx <= W) & (self._gy >= 0) & (self._gy <= D)
        outer = (self._gx >= -t) & (self._gx <= W + t) & (self._gy >= -t) & (self._gy <= D + t)
        self._static = np.where(outer, self.WALL_HEIGHT, self.FLOOR_HEIGHT).astype(np.float32)
        self.sand = np.zeros((self._ny, self._nx), np.float32)
        self._composite = self._static.copy()
        self.randomize()

        rays = self.intrinsics.rays().reshape(-1, 3).astype(np.float64)
        self._dirs = rays @ self.cam_to_world.T  # směry ve světě, z-složka kamery = 1

    # --- tvar písku ---------------------------------------------------------
    def flatten(self) -> None:
        with self._lock:
            self.sand[:] = 0
            self._rebuild()

    def randomize(self, n_hills: int = 6) -> None:
        W, D = self.BOX_SIZE
        sand = np.zeros_like(self.sand)
        for _ in range(n_hills):
            cx, cy = self._rng.uniform(0.1, W - 0.1), self._rng.uniform(0.1, D - 0.1)
            r = self._rng.uniform(0.06, 0.18)
            a = self._rng.uniform(-0.06, 0.11)
            sand += a * np.exp(-((self._gx - cx) ** 2 + (self._gy - cy) ** 2) / (2 * r * r))
        sand -= sand[self._inside].mean()  # objem písku se nemění
        with self._lock:
            self.sand[:] = sand
            self._rebuild()

    def set_sand(self, fn) -> None:
        """Nastaví písek funkcí fn(x, y) -> výška [m] (pro testy)."""
        with self._lock:
            self.sand[:] = fn(self._gx, self._gy)
            self._rebuild()

    def _rebuild(self) -> None:
        lo, hi = self.SAND_LIMITS
        np.clip(self.sand, lo, hi, out=self.sand)
        self._composite = np.where(self._inside, self.sand, self._static).astype(np.float32)

    def true_height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Skutečná výška písku v souřadnicích bedny (pro testy přesnosti)."""
        ix = np.clip(((np.asarray(x) - self._x0) / self.RES).astype(int), 0, self._nx - 1)
        iy = np.clip(((np.asarray(y) - self._y0) / self.RES).astype(int), 0, self._ny - 1)
        return self.sand[iy, ix]

    # --- kalibrace podle známé geometrie ------------------------------------
    def ground_truth(self) -> tuple[np.ndarray, float, np.ndarray]:
        """(normála, offset, rohy bedny v pixelech) – ideální kalibrace simulace."""
        M = self.cam_to_world
        normal = M.T @ np.array([0.0, 0.0, 1.0])
        offset = float(self.camera_pos[2])
        W, D = self.BOX_SIZE
        corners_w = np.array([[0, 0, 0], [W, 0, 0], [W, D, 0], [0, D, 0]], float)
        corners_c = (corners_w - self.camera_pos) @ M  # = M^T (p - C)
        return normal, offset, self.intrinsics.project(corners_c)

    # --- interakce -------------------------------------------------------------
    def interact(self, u: float, v: float, amount: float, radius: float = 0.05) -> None:
        x, y = self._pixel_to_world(u, v)
        sigma = radius / 2
        with self._lock:
            r2 = (self._gx - x) ** 2 + (self._gy - y) ** 2
            bump = amount * np.exp(-r2 / (2 * sigma * sigma))
            if amount < 0:  # odhrabaný písek se nahrne kolem (objem zůstává)
                ring = np.exp(-r2 / (2 * (2.2 * sigma) ** 2)) - np.exp(-r2 / (2 * sigma * sigma))
                bump -= ring * (bump.sum() / max(ring.sum(), 1e-9))
            self.sand += bump * self._inside
            self._rebuild()

    def _pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        intr = self.intrinsics
        d = self.cam_to_world @ np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
        C = self.camera_pos
        h = 0.0
        for _ in range(4):
            t = (h - C[2]) / d[2]
            p = C + t * d
            h = float(self._sample(np.array([p[0]]), np.array([p[1]]), self._composite)[0])
        return float(p[0]), float(p[1])

    def _sample(self, x: np.ndarray, y: np.ndarray, field: np.ndarray) -> np.ndarray:
        ix = ((x - self._x0) / self.RES).astype(np.int32)
        iy = ((y - self._y0) / self.RES).astype(np.int32)
        ok = (ix >= 0) & (ix < self._nx) & (iy >= 0) & (iy < self._ny)
        out = np.full(x.shape, self.FLOOR_HEIGHT, np.float32)
        out[ok] = field[iy[ok], ix[ok]]
        return out

    # --- snímání -------------------------------------------------------------
    def render_depth(self, noise: bool = True) -> np.ndarray:
        composite = self._composite
        C, d = self.camera_pos, self._dirs
        h = np.zeros(len(d), np.float32)
        for _ in range(4):  # zpřesňování průsečíku paprsku s výškovým polem
            t = (h - C[2]) / d[:, 2]
            h = self._sample(C[0] + t * d[:, 0], C[1] + t * d[:, 1], composite)
        t = (h - C[2]) / d[:, 2]
        # U hran (stěna bedny) iterace osciluje – tam průsečík dohledáme bisekcí.
        bad = np.flatnonzero(np.abs(self._sample(C[0] + t * d[:, 0], C[1] + t * d[:, 1], composite) - h) > 0.002)
        if len(bad):
            db = d[bad]
            lo = (self.SAND_LIMITS[1] + 0.05 - C[2]) / db[:, 2]  # nad nejvyšším bodem
            hi = (self.FLOOR_HEIGHT - 0.01 - C[2]) / db[:, 2]  # pod stolem
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                above = C[2] + mid * db[:, 2] > self._sample(C[0] + mid * db[:, 0], C[1] + mid * db[:, 1], composite)
                lo = np.where(above, mid, lo)
                hi = np.where(above, hi, mid)
            t[bad] = hi
        depth = t.astype(np.float32)
        if noise:
            depth += self._rng.normal(0, self.noise, depth.shape).astype(np.float32) * depth**2
            depth[self._rng.random(depth.shape) < 0.003] = 0  # výpadky měření
        return depth.reshape(self.intrinsics.height, self.intrinsics.width)

    def _read(self) -> np.ndarray | None:
        t0 = time.monotonic()
        depth = self.render_depth(noise=self.noise > 0)
        time.sleep(max(0.0, 1.0 / self.fps - (time.monotonic() - t0)))
        return depth
