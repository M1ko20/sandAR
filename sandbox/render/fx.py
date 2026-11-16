"""Drobné efekty sdílené módy: déšť z ruky (mráček + dopadající kapky)."""

from __future__ import annotations

import math

import numpy as np


class RainFx:
    """Déšť pod rukama, jak ho vidí člověk shora.

    * mráček na dlani – projektor ho promítne přímo na hřbet ruky,
    * dopady kapek kolem ruky – kroužky, které se rozbíhají a mizí (pod samotnou rukou
      je stín, tam by je nikdo neviděl).
    Souřadnice v metrech bedny, kreslí se do plátna přes ``host.painter``.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.drops = np.zeros((0, 3), np.float32)  # x, y, stáří [s]
        self.t = 0.0

    LIFE = 0.55  # [s] jak dlouho je vidět kroužek po dopadu

    def update(self, dt: float, sources: list[tuple[float, float, float]]) -> None:
        """sources = [(x, y, intenzita 0..1)] – kde právě prší."""
        self.t += dt
        if len(self.drops):
            self.drops[:, 2] += dt
            self.drops = self.drops[self.drops[:, 2] < self.LIFE]
        for x, y, k in sources:
            n = self.rng.poisson(70 * k * dt)
            if not n:
                continue
            ang = self.rng.uniform(0, 2 * math.pi, n)
            r = np.sqrt(self.rng.uniform(0.25, 1.0, n)) * 0.11  # mezikruží 5,5–11 cm (pod rukou je stín)
            new = np.stack([x + r * np.cos(ang), y + r * np.sin(ang), np.zeros(n)], 1).astype(np.float32)
            self.drops = np.vstack([self.drops, new])[-600:]

    def clear(self) -> None:
        self.drops = self.drops[:0]

    def draw(self, host, clouds: list[tuple[float, float, float]]) -> None:
        p = host.painter
        cw, ch = host.canvas_size
        sx, sy = cw / host.box_size[0], ch / host.box_size[1]
        if len(self.drops):
            k = self.drops[:, 2] / self.LIFE
            seg = 14
            ang = np.linspace(0, 2 * math.pi, seg + 1)
            cs, sn = np.cos(ang), np.sin(ang)
            r = (0.004 + 0.022 * np.sqrt(k)) * sx
            w = np.maximum(1.5, 3.5 * (1 - k)) * (sx / 800)
            cx, cy = self.drops[:, 0] * sx, self.drops[:, 1] * sy
            outer = np.stack([cx[:, None] + (r + w)[:, None] * cs, cy[:, None] + (r + w)[:, None] * sn], -1)
            inner = np.stack([cx[:, None] + r[:, None] * cs, cy[:, None] + r[:, None] * sn], -1)
            a, b, c, d = outer[:, :-1], outer[:, 1:], inner[:, 1:], inner[:, :-1]
            quads = np.stack([a, b, c, a, c, d], 2).reshape(-1, 2)
            alpha = (0.9 * (1 - k) ** 1.5)
            col = np.repeat(np.stack([0.85 + 0 * k, 0.95 + 0 * k, 1.0 + 0 * k, alpha], 1), seg * 6, 0)
            p.triangles(quads, col)
        for x, y, k in clouds:
            self._cloud(p, x * sx, y * sy, 0.075 * sx, k)

    def _cloud(self, p, x: float, y: float, r: float, k: float) -> None:
        """Mráček z několika překrývajících se kruhů, jemně se vlní."""
        t = self.t
        blobs = [(-0.55, 0.15, 0.55), (0.0, -0.2, 0.7), (0.55, 0.1, 0.6), (0.15, 0.35, 0.5), (-0.25, 0.35, 0.45)]
        a = 0.85 * k
        for dx, dy, s in blobs:  # tmavší spodek = objem
            p.circle(x + dx * r + r * 0.08, y + dy * r + r * 0.12, s * r * (1 + 0.04 * math.sin(t * 2 + dx * 5)),
                     (0.45, 0.55, 0.70, a), segments=24)
        for dx, dy, s in blobs:
            p.circle(x + dx * r, y + dy * r, s * r * 0.92 * (1 + 0.04 * math.sin(t * 2 + dx * 5)),
                     (0.92, 0.96, 1.0, a), segments=24)
