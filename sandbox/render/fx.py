"""Efekty sdílené módy: déšť z ruky (mráček + padající kapky) a textury vody pro ``water_shade``."""

from __future__ import annotations

import math

import numpy as np


class RainFx:
    """Déšť pod rukama, jak ho vidí člověk shora.

    * mráček na dlani – projektor ho promítne přímo na hřbet ruky,
    * kapky: padají z mráčku (světlá čárka letí ven od ruky – shora tak vypadá pád v perspektivě),
      dopadnou, rozběhne se kroužek a odstříknou drobné kapičky.
    Souřadnice v metrech bedny, kreslí se do plátna přes ``host.painter``.
    """

    FALL = 0.22  # [s] let kapky
    SPLASH = 0.5  # [s] kroužek po dopadu
    RATE = 110.0  # [kapek/s] z jednoho mráčku

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        # x0, y0 (start u mráčku), x1, y1 (dopad), stáří [s], úhel odstřiku
        self.drops = np.zeros((0, 6), np.float32)
        self.t = 0.0

    def update(self, dt: float, sources: list[tuple[float, float, float]]) -> None:
        """sources = [(x, y, intenzita 0..1)] – kde právě prší."""
        self.t += dt
        if len(self.drops):
            self.drops[:, 4] += dt
            self.drops = self.drops[self.drops[:, 4] < self.FALL + self.SPLASH]
        for x, y, k in sources:
            n = self.rng.poisson(self.RATE * k * dt)
            if not n:
                continue
            ang = self.rng.uniform(0, 2 * math.pi, n)
            r1 = np.sqrt(self.rng.uniform(0.3, 1.0, n)) * 0.13  # dopad 7–13 cm od dlaně (pod rukou je stín)
            r0 = r1 * 0.45
            new = np.stack([x + r0 * np.cos(ang), y + r0 * np.sin(ang), x + r1 * np.cos(ang), y + r1 * np.sin(ang),
                            np.zeros(n), self.rng.uniform(0, 2 * math.pi, n)], 1).astype(np.float32)
            self.drops = np.vstack([self.drops, new])[-900:]

    def clear(self) -> None:
        self.drops = self.drops[:0]

    def draw(self, host, clouds: list[tuple[float, float, float]]) -> None:
        p = host.painter
        cw, ch = host.canvas_size
        sx, sy = cw / host.box_size[0], ch / host.box_size[1]
        px = sx / 800  # měřítko čar podle rozlišení plátna
        d = self.drops
        if len(d):
            age = d[:, 4]
            falling = age < self.FALL
            if falling.any():
                f = d[falling]
                k = (f[:, 4] / self.FALL)[:, None]
                head = f[:, 0:2] + (f[:, 2:4] - f[:, 0:2]) * k
                tail = f[:, 0:2] + (f[:, 2:4] - f[:, 0:2]) * np.maximum(k - 0.35, 0)
                self._segments(p, head * [sx, sy], tail * [sx, sy], 2.6 * px, (0.80, 0.93, 1.0, 0.95))
            sp = ~falling
            if sp.any():
                q = d[sp]
                k = (q[:, 4] - self.FALL) / self.SPLASH
                cx, cy = q[:, 2] * sx, q[:, 3] * sy
                self._rings(p, cx, cy, (0.003 + 0.020 * np.sqrt(k)) * sx, np.maximum(1.2, 3.2 * (1 - k)) * px,
                            0.85 * (1 - k) ** 1.5)
                # odstřik: 3 kapičky letí ven a padají zpět
                early = k < 0.5
                if early.any():
                    kk = k[early] / 0.5
                    base = q[early, 5]
                    pts = []
                    for j in range(3):
                        a = base + j * 2.1
                        r = (0.008 + 0.022 * np.sin(kk * math.pi * 0.5)) * sx
                        pts.append(np.stack([cx[early] + r * np.cos(a), cy[early] + r * np.sin(a)], 1))
                    pts = np.concatenate(pts)
                    s = 1.3 * px
                    alpha = np.tile(0.9 * (1 - kk), 3)
                    quads = np.stack([pts + [-s, -s], pts + [s, -s], pts + [s, s], pts + [-s, -s], pts + [s, s],
                                      pts + [-s, s]], 1).reshape(-1, 2)
                    cols = np.repeat(np.stack([np.full_like(alpha, 0.9), np.full_like(alpha, 0.97),
                                               np.ones_like(alpha), alpha], 1), 6, 0)
                    p.triangles(quads, cols)
        for x, y, k in clouds:
            self._cloud(p, x * sx, y * sy, 0.075 * sx, k)

    @staticmethod
    def _segments(p, a: np.ndarray, b: np.ndarray, w: float, color) -> None:
        """Mnoho úseček (čárek) najednou jako tenké čtyřúhelníky."""
        d = a - b
        n = np.hypot(d[:, 0], d[:, 1])[:, None]
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(n, 1e-6) * (w / 2)
        quads = np.stack([a + nrm, a - nrm, b - nrm, a + nrm, b - nrm, b + nrm], 1).reshape(-1, 2)
        p.triangles(quads, color)

    @staticmethod
    def _rings(p, cx, cy, r, w, alpha, seg: int = 14) -> None:
        ang = np.linspace(0, 2 * math.pi, seg + 1)
        cs, sn = np.cos(ang), np.sin(ang)
        outer = np.stack([cx[:, None] + (r + w)[:, None] * cs, cy[:, None] + (r + w)[:, None] * sn], -1)
        inner = np.stack([cx[:, None] + r[:, None] * cs, cy[:, None] + r[:, None] * sn], -1)
        a, b, c, d = outer[:, :-1], outer[:, 1:], inner[:, 1:], inner[:, :-1]
        quads = np.stack([a, b, c, a, c, d], 2).reshape(-1, 2)
        col = np.repeat(np.stack([np.full_like(alpha, 0.85), np.full_like(alpha, 0.95), np.ones_like(alpha), alpha], 1),
                        seg * 6, 0)
        p.triangles(quads, col)

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


class WaterLayer:
    """Textury pro sdílený shader vody (``water_shade`` v glsl.py): hloubka + rychlost proudu.

    V shaderu: ``uniform sampler2D u_water; uniform sampler2D u_vel;`` a
    ``water_shade(c, uv, max(sample_bicubic(u_water, uv), 0.0), texture(u_vel, uv).rg, pěna)``.
    """

    def __init__(self, host) -> None:
        self.host = host
        self.depth = None
        self.vel = None

    def write(self, sim) -> None:
        import moderngl

        from ..sim.lava import blur3

        gh, gw = sim.depth.shape
        if self.depth is None or self.depth.size != (gw, gh):
            self.release()
            self.depth = self.host.float_texture((gw, gh))
            self.vel = self.host.ctx.texture((gw, gh), 2, dtype="f4")
            self.vel.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.vel.repeat_x = self.vel.repeat_y = False
        self.depth.write(np.ascontiguousarray(sim.depth, np.float32).tobytes())
        vx, vy = sim.velocity()
        self.vel.write(np.dstack([blur3(vx), blur3(vy)]).astype(np.float32).tobytes())

    @property
    def ready(self) -> bool:
        return self.depth is not None

    def uniforms(self, unit: int = 1) -> dict:
        return {"u_water": (self.depth, unit), "u_vel": (self.vel, unit + 1)}

    def release(self) -> None:
        for t in (self.depth, self.vel):
            if t is not None:
                t.release()
        self.depth = self.vel = None
