"""Mělká voda metodou „virtuálních trubek“ (Mei et al. 2007) – sdílená simulace pro módy.

Mezi sousedními buňkami teče voda úměrně rozdílu hladin (terén + voda). Stabilní,
zachovává objem, běží v numpy na mřížce výškové mapy (~256×156) v reálném čase.
"""

from __future__ import annotations

import math

import numpy as np


class WaterSim:
    def __init__(self, shape: tuple[int, int], cell: float) -> None:
        self.cell = cell
        self.depth = np.zeros(shape, np.float32)
        self.fl = np.zeros(shape, np.float32)  # odtok doleva [m³/s]
        self.fr = np.zeros(shape, np.float32)
        self.ft = np.zeros(shape, np.float32)  # nahoru (−y)
        self.fb = np.zeros(shape, np.float32)  # dolů (+y)

    @property
    def volume(self) -> float:
        """Objem vody [m³]."""
        return float(self.depth.sum()) * self.cell * self.cell

    def velocity(self, min_depth: float = 0.002, max_speed: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
        """Rychlost vody v buňkách (vx, vy) [m/s] z toků v trubkách (průměr přítoku a odtoku)."""
        fl, fr, ft, fb = self.fl, self.fr, self.ft, self.fb
        qx = fr - fl
        qy = fb - ft
        qx[:, 1:] += fr[:, :-1]  # přítok zleva teče doprava
        qx[:, :-1] -= fl[:, 1:]
        qy[1:, :] += fb[:-1, :]
        qy[:-1, :] -= ft[1:, :]
        k = 0.5 / (self.cell * np.maximum(self.depth, min_depth))
        vx = np.clip(qx * k, -max_speed, max_speed)
        vy = np.clip(qy * k, -max_speed, max_speed)
        dry = self.depth < 1e-4
        vx[dry] = 0
        vy[dry] = 0
        return vx.astype(np.float32), vy.astype(np.float32)

    def clear(self) -> None:
        for a in (self.depth, self.fl, self.fr, self.ft, self.fb):
            a[:] = 0

    def advance(self, terrain: np.ndarray, frame_dt: float, gravity: float = 9.81, drain: bool = False) -> None:
        """Posune simulaci o frame_dt; rozdělí ho na stabilní podkroky (CFL)."""
        max_depth = float(self.depth.max())
        wave_speed = math.sqrt(gravity * max(max_depth, 0.002))
        n = int(min(12, max(1, math.ceil(frame_dt / (0.4 * self.cell / wave_speed)))))
        for _ in range(n):
            self.step(terrain, frame_dt / n, gravity, drain)

    def step(self, terrain: np.ndarray, dt: float, gravity: float = 9.81, drain: bool = False,
             damping: float = 0.995) -> None:
        d = self.depth
        H = terrain + d
        k = dt * gravity * self.cell
        fl, fr, ft, fb = self.fl, self.fr, self.ft, self.fb
        fr[:, :-1] = np.maximum(0, fr[:, :-1] * damping + k * (H[:, :-1] - H[:, 1:]))
        fl[:, 1:] = np.maximum(0, fl[:, 1:] * damping + k * (H[:, 1:] - H[:, :-1]))
        fb[:-1, :] = np.maximum(0, fb[:-1, :] * damping + k * (H[:-1, :] - H[1:, :]))
        ft[1:, :] = np.maximum(0, ft[1:, :] * damping + k * (H[1:, :] - H[:-1, :]))
        fr[:, -1] = fl[:, 0] = fb[-1, :] = ft[0, :] = 0  # stěny bedny

        area = self.cell * self.cell
        out = fl + fr + ft + fb
        scale = np.minimum(1.0, d * area / np.maximum(out * dt, 1e-12))
        for f in (fl, fr, ft, fb):
            f *= scale
        out *= scale

        inflow = np.zeros_like(d)
        inflow[:, 1:] += fr[:, :-1]
        inflow[:, :-1] += fl[:, 1:]
        inflow[1:, :] += fb[:-1, :]
        inflow[:-1, :] += ft[1:, :]
        d += dt * (inflow - out) / area
        np.maximum(d, 0, out=d)
        if drain:
            d[0, :] = d[-1, :] = 0
            d[:, 0] = d[:, -1] = 0
