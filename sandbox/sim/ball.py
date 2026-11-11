"""Kulička kutálející se po písku – sdílená fyzika pro módy (minigolf, kuličková dráha…).

Model: koule valící se bez prokluzu po ploše h(x, y). Zrychlení z gravitace je
``a = −(5/7) · g · ∇h / (1 + |∇h|²)`` (5/7 = podíl pro valící se plnou kouli),
k tomu valivý odpor (konstantní zpomalení proti pohybu – kulička se zastaví
v důlku i na mírném svahu) a slabé tlumení úměrné rychlosti. Od stěn bedny se
odráží. Sklon se počítá jednou za novou výškovou mapu (``SlopeField``) z mírně
vyhlazených výšek, aby šum Kinectu kuličkou necukal, a vzorkuje se bilineárně.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

ROLL_FACTOR = 5.0 / 7.0  # valící se plná koule


@dataclass
class Ball:
    x: float  # [m] souřadnice bedny
    y: float
    vx: float = 0.0  # [m/s]
    vy: float = 0.0
    radius: float = 0.02  # [m] jen pro odraz od stěn a kreslení
    kick_cooldown: float = 0.0  # [s] do dalšího kopnutí rukou
    still_time: float = 0.0  # [s] jak dlouho se kulička skoro nehýbe
    trail: list[tuple[float, float]] = field(default_factory=list)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass
class BallPhysics:
    gravity: float = 9.81 * 0.6  # zpomalená gravitace – na malém pískovišti je hra čitelnější
    rolling_resistance: float = 0.035  # koeficient valivého odporu (·g = zpomalení)
    damping: float = 0.35  # [1/s] tlumení úměrné rychlosti
    restitution: float = 0.55  # odraz od stěny (0 = lepí, 1 = pružný)
    max_speed: float = 1.6  # [m/s]
    max_substeps: int = 16


class SlopeField:
    """Sklon terénu (dh/dx, dh/dy) na mřížce výškové mapy s bilineárním vzorkováním."""

    def __init__(self, heights: np.ndarray, cell_size: tuple[float, float] | float, smooth_cells: float = 1.5) -> None:
        if isinstance(cell_size, (int, float)):
            cell_size = (float(cell_size), float(cell_size))
        self.cell = (float(cell_size[0]), float(cell_size[1]))
        h = np.asarray(heights, np.float32)
        if smooth_cells > 0:
            h = cv2.GaussianBlur(h, (0, 0), smooth_cells, borderType=cv2.BORDER_REPLICATE)
        self.heights = h
        gy, gx = np.gradient(h, self.cell[1], self.cell[0])
        self.gx = gx.astype(np.float32)
        self.gy = gy.astype(np.float32)
        gh, gw = h.shape
        self.size = (gw * self.cell[0], gh * self.cell[1])

    def _bilinear(self, grid: np.ndarray, x, y):
        gh, gw = grid.shape
        fx = np.clip(np.asarray(x, np.float64) / self.cell[0] - 0.5, 0, gw - 1.001)
        fy = np.clip(np.asarray(y, np.float64) / self.cell[1] - 0.5, 0, gh - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        a = grid[iy, ix] * (1 - tx) + grid[iy, ix + 1] * tx
        b = grid[iy + 1, ix] * (1 - tx) + grid[iy + 1, ix + 1] * tx
        return a * (1 - ty) + b * ty

    def gradient(self, x, y):
        """Bilineárně interpolovaný sklon v bodě/bodech [m] → (gx, gy)."""
        return self._bilinear(self.gx, x, y), self._bilinear(self.gy, x, y)

    def height(self, x, y):
        return self._bilinear(self.heights, x, y)


def step(balls: list[Ball], slope: SlopeField, dt: float, phys: BallPhysics | None = None) -> None:
    """Posune kuličky o dt (rozdělí na podkroky, aby kulička nepřeskočila buňku)."""
    phys = phys or BallPhysics()
    if not balls or dt <= 0:
        return
    cell = min(slope.cell)
    W, D = slope.size
    for b in balls:
        b.kick_cooldown = max(0.0, b.kick_cooldown - dt)
        n = int(min(phys.max_substeps, max(1, math.ceil(max(b.speed, 0.3) * dt / (0.5 * cell)))))
        h = dt / n
        for _ in range(n):
            gx, gy = slope.gradient(b.x, b.y)
            gx, gy = float(gx), float(gy)
            k = ROLL_FACTOR * phys.gravity / (1.0 + gx * gx + gy * gy)
            ax, ay = -k * gx, -k * gy
            v = b.speed
            stop = phys.rolling_resistance * phys.gravity
            if v < 1e-4:
                # klidové tření: svah, který valivý odpor neudrží, kuličku rozjede
                a = math.hypot(ax, ay)
                if a <= stop:
                    b.vx = b.vy = 0.0
                    continue
                ax -= ax / a * stop
                ay -= ay / a * stop
                b.vx, b.vy = ax * h, ay * h
            else:
                b.vx += (ax - phys.damping * b.vx) * h
                b.vy += (ay - phys.damping * b.vy) * h
                # valivý odpor – nesmí pohyb otočit, jen zastavit
                nv = math.hypot(b.vx, b.vy)
                dec = stop * h
                if nv <= dec:
                    b.vx = b.vy = 0.0
                else:
                    f = (nv - dec) / nv
                    b.vx *= f
                    b.vy *= f
            s = b.speed
            if s > phys.max_speed:
                b.vx *= phys.max_speed / s
                b.vy *= phys.max_speed / s
            b.x += b.vx * h
            b.y += b.vy * h
            _bounce(b, W, D, phys.restitution)
        b.still_time = b.still_time + dt if b.speed < 0.02 else 0.0


def _bounce(b: Ball, W: float, D: float, e: float) -> None:
    r = b.radius
    if b.x < r:
        b.x, b.vx = r, abs(b.vx) * e
    elif b.x > W - r:
        b.x, b.vx = W - r, -abs(b.vx) * e
    if b.y < r:
        b.y, b.vy = r, abs(b.vy) * e
    elif b.y > D - r:
        b.y, b.vy = D - r, -abs(b.vy) * e


def kick(b: Ball, from_x: float, from_y: float, speed: float = 0.9) -> None:
    """Odkopne kuličku směrem od bodu (např. od ruky)."""
    dx, dy = b.x - from_x, b.y - from_y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        ang = np.random.uniform(0, 2 * math.pi)
        dx, dy, d = math.cos(ang), math.sin(ang), 1.0
    b.vx += dx / d * speed
    b.vy += dy / d * speed
    b.still_time = 0.0
