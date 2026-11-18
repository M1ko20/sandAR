"""Kreslení zvířat ekosystému shora (siluety z elips a trojúhelníků).

Každý druh je složený z několika dílů v místních souřadnicích (x = dopředu, y = doprava,
jednotka = délka těla). Všechna zvířata se sbírají do jednoho pole trojúhelníků a kreslí
jedním voláním, takže i 60 zvířat stojí zlomek milisekundy.
"""

from __future__ import annotations

import math

import numpy as np

from ..sim.fauna import Fauna, Herd


_UNIT: dict[int, np.ndarray] = {}


def _ellipse(cx: float, cy: float, rx: float, ry: float, seg: int = 12) -> np.ndarray:
    u = _UNIT.get(seg)
    if u is None:  # jednotkový kruh jako vějíř trojúhelníků (střed, okraj, okraj)
        a = np.linspace(0, 2 * math.pi, seg + 1)
        ring = np.stack([np.cos(a), np.sin(a)], 1)
        u = _UNIT[seg] = np.stack([np.zeros((seg, 2)), ring[:-1], ring[1:]], 1).reshape(-1, 2)
    return u * (rx, ry) + (cx, cy)


def _poly(*pts) -> np.ndarray:
    """Konvexní mnohoúhelník jako vějíř trojúhelníků."""
    p = np.array(pts, float)
    return np.stack([np.repeat(p[:1], len(p) - 2, 0), p[1:-1], p[2:]], 1).reshape(-1, 2)


SHADOW = (0.0, 0.0, 0.0, 0.35)


def _parts(key: str, t: float) -> list[tuple[np.ndarray, tuple]]:
    """Díly siluety druhu (stín se přidá automaticky). ``t`` = fáze animace."""
    if key == "srnka":
        return [(_ellipse(0, 0, 0.5, 0.22), (0.66, 0.36, 0.13, 1)),
                (_ellipse(0.58, 0, 0.16, 0.12, 10), (0.55, 0.28, 0.09, 1)),
                (_ellipse(0.62, -0.1, 0.06, 0.04, 6), (0.40, 0.20, 0.07, 1)),
                (_ellipse(0.62, 0.1, 0.06, 0.04, 6), (0.40, 0.20, 0.07, 1)),
                (_ellipse(-0.45, 0, 0.1, 0.1, 8), (1, 1, 1, 1))]
    if key == "zajic":
        ear = 0.05 * math.sin(t * 2)
        return [(_ellipse(0, 0, 0.45, 0.3), (0.62, 0.52, 0.40, 1)),
                (_ellipse(0.45, 0, 0.2, 0.17, 10), (0.55, 0.45, 0.34, 1)),
                (_ellipse(0.3, -0.12 - ear, 0.28, 0.06, 8), (0.48, 0.38, 0.28, 1)),
                (_ellipse(0.3, 0.12 + ear, 0.28, 0.06, 8), (0.48, 0.38, 0.28, 1)),
                (_ellipse(-0.45, 0, 0.12, 0.12, 8), (1, 1, 1, 1))]
    if key == "liska":
        sw = 0.15 * math.sin(t * 1.5)
        return [(_poly((-0.45, 0), (-1.05, -0.2 + sw), (-1.15, sw), (-1.05, 0.2 + sw)), (0.95, 0.45, 0.10, 1)),
                (_ellipse(-1.08, sw, 0.1, 0.1, 8), (1, 1, 1, 1)),
                (_ellipse(0, 0, 0.48, 0.2), (0.95, 0.42, 0.08, 1)),
                (_poly((0.4, -0.17), (0.85, 0), (0.4, 0.17)), (0.90, 0.38, 0.06, 1)),
                (_ellipse(0.82, 0, 0.05, 0.05, 6), (0.1, 0.05, 0.02, 1)),
                (_poly((0.42, -0.18), (0.55, -0.28), (0.58, -0.12)), (0.25, 0.12, 0.04, 1)),
                (_poly((0.42, 0.18), (0.55, 0.28), (0.58, 0.12)), (0.25, 0.12, 0.04, 1))]
    if key == "medved":
        return [(_ellipse(0, 0, 0.5, 0.36), (0.36, 0.22, 0.12, 1)),
                (_ellipse(0.5, 0, 0.2, 0.2, 10), (0.30, 0.18, 0.10, 1)),
                (_ellipse(0.6, -0.17, 0.07, 0.07, 6), (0.25, 0.14, 0.07, 1)),
                (_ellipse(0.6, 0.17, 0.07, 0.07, 6), (0.25, 0.14, 0.07, 1)),
                (_ellipse(0.68, 0, 0.08, 0.07, 6), (0.62, 0.48, 0.32, 1))]
    if key == "kamzik":
        return [(_ellipse(0, 0, 0.48, 0.22), (0.50, 0.38, 0.26, 1)),
                (_ellipse(-0.1, 0, 0.3, 0.08, 8), (0.25, 0.18, 0.12, 1)),
                (_ellipse(0.55, 0, 0.15, 0.12, 10), (0.92, 0.88, 0.80, 1)),
                (_poly((0.6, -0.06), (0.78, -0.12), (0.62, 0.0)), (0.08, 0.06, 0.05, 1)),
                (_poly((0.6, 0.06), (0.78, 0.12), (0.62, 0.0)), (0.08, 0.06, 0.05, 1))]
    if key == "krab":
        leg = 0.08 * math.sin(t * 6)
        parts = [(_ellipse(0, 0, 0.32, 0.45), (0.92, 0.22, 0.10, 1))]
        for side in (-1, 1):
            for k in (-0.2, 0.0, 0.2):
                parts.append((_poly((k, side * 0.3), (k + 0.05 + leg, side * 0.62), (k - 0.05 + leg, side * 0.62)),
                              (0.75, 0.15, 0.06, 1)))
            parts.append((_ellipse(0.42, side * 0.32, 0.16, 0.1, 8), (0.98, 0.35, 0.15, 1)))
        parts.append((_ellipse(0.25, -0.1, 0.04, 0.04, 5), (0, 0, 0, 1)))
        parts.append((_ellipse(0.25, 0.1, 0.04, 0.04, 5), (0, 0, 0, 1)))
        return parts
    if key == "zelva":
        f = 0.06 * math.sin(t * 2)
        parts = []
        for sx, sy in ((0.28, 0.3), (0.28, -0.3), (-0.28, 0.3), (-0.28, -0.3)):
            parts.append((_ellipse(sx + f * np.sign(sx), sy * 1.15, 0.13, 0.08, 8), (0.42, 0.62, 0.30, 1)))
        parts += [(_ellipse(0.55, 0, 0.13, 0.1, 8), (0.45, 0.65, 0.32, 1)),
                  (_ellipse(0, 0, 0.45, 0.36), (0.30, 0.42, 0.16, 1)),
                  (_ellipse(0, 0, 0.18, 0.14, 8), (0.45, 0.58, 0.22, 1)),
                  (_ellipse(0.24, 0.18, 0.1, 0.08, 6), (0.45, 0.58, 0.22, 1)),
                  (_ellipse(-0.24, -0.18, 0.1, 0.08, 6), (0.45, 0.58, 0.22, 1)),
                  (_ellipse(0.24, -0.18, 0.1, 0.08, 6), (0.45, 0.58, 0.22, 1)),
                  (_ellipse(-0.24, 0.18, 0.1, 0.08, 6), (0.45, 0.58, 0.22, 1))]
        return parts
    if key == "ryba":
        wag = 0.12 * math.sin(t * 8)
        return [(_poly((-0.35, 0), (-0.75, -0.25 + wag), (-0.75, 0.25 + wag)), (1.0, 0.55, 0.12, 0.9)),
                (_ellipse(0, 0, 0.45, 0.17), (1.0, 0.65, 0.18, 0.95)),
                (_ellipse(0.2, 0, 0.18, 0.1, 8), (1.0, 0.85, 0.5, 0.95))]
    if key == "kachna":
        return [(_ellipse(-0.05, 0, 0.45, 0.28), (0.55, 0.42, 0.30, 1)),
                (_ellipse(-0.1, 0, 0.3, 0.2), (0.75, 0.68, 0.58, 1)),
                (_ellipse(0.38, 0, 0.16, 0.14, 10), (0.05, 0.45, 0.20, 1)),
                (_poly((0.5, -0.06), (0.72, 0), (0.5, 0.06)), (1.0, 0.65, 0.1, 1))]
    if key == "ptak":
        flap = 0.45 + 0.35 * math.sin(t * 9)
        return [(_poly((0.15, 0), (-0.25 - 0.3 * flap, -0.55), (-0.1, 0)), (0.05, 0.05, 0.08, 1)),
                (_poly((0.15, 0), (-0.25 - 0.3 * flap, 0.55), (-0.1, 0)), (0.05, 0.05, 0.08, 1)),
                (_ellipse(0.0, 0, 0.18, 0.07, 6), (0.95, 0.95, 0.95, 1))]
    if key == "orel":
        tip = 0.06 * math.sin(t * 2)
        return [(_poly((0.1, -0.08), (-0.05, -0.62 - tip), (-0.25, -0.55 - tip), (-0.15, -0.05)), (0.30, 0.18, 0.08, 1)),
                (_poly((0.1, 0.08), (-0.05, 0.62 + tip), (-0.25, 0.55 + tip), (-0.15, 0.05)), (0.30, 0.18, 0.08, 1)),
                (_poly((-0.25, -0.08), (-0.5, -0.14), (-0.5, 0.14), (-0.25, 0.08)), (0.92, 0.9, 0.85, 1)),
                (_ellipse(-0.05, 0, 0.25, 0.09, 8), (0.36, 0.22, 0.10, 1)),
                (_ellipse(0.25, 0, 0.08, 0.07, 8), (0.98, 0.98, 0.95, 1)),
                (_poly((0.31, -0.03), (0.4, 0), (0.31, 0.03)), (1.0, 0.8, 0.1, 1))]
    if key == "motyl":
        s = 0.35 + 0.65 * abs(math.sin(t * 4))
        col = (1.0, 0.85, 0.15, 1) if int(t) % 2 else (0.95, 0.45, 0.85, 1)
        return [(_ellipse(0.1, -0.35 * s, 0.25, 0.3 * s, 8), col), (_ellipse(0.1, 0.35 * s, 0.25, 0.3 * s, 8), col),
                (_ellipse(-0.2, -0.25 * s, 0.18, 0.2 * s, 8), col), (_ellipse(-0.2, 0.25 * s, 0.18, 0.2 * s, 8), col),
                (_ellipse(0, 0, 0.32, 0.06, 6), (0.1, 0.08, 0.05, 1))]
    return [(_ellipse(0, 0, 0.5, 0.25), (1, 1, 1, 1))]


SCALE = 1.5  # zvířata větší než skutečná délka v metrech – ať jsou ze 2 m vidět
HALO = 1.08  # světlý lem kolem siluety (zvíře se odliší od lesa, vody i skal)
HALO_COLOR = (1.0, 0.97, 0.86, 0.55)
FLYING_SHADOW = {"ptak": 0.9, "orel": 1.3, "motyl": 0.35}


def draw_fauna(host, fauna: Fauna, clock: float) -> None:
    cw, ch = host.canvas_size
    sx, sy = cw / host.box_size[0], ch / host.box_size[1]
    verts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    order = ["ryba", "kachna", "zelva", "krab", "zajic", "srnka", "liska", "kamzik", "medved", "motyl", "ptak", "orel"]
    for key in order:
        h: Herd | None = fauna.herds.get(key)
        if h is None:
            continue
        L = h.sp.length * SCALE * sx
        lift = FLYING_SHADOW.get(key, 0.15)  # letci mají stín dál (výš nad zemí)
        for i in range(len(h.pos)):
            vis = float(h.visible[i])
            if vis < 0.03:
                continue
            x, y = h.pos[i, 0] * sx, h.pos[i, 1] * sy
            c, s = math.cos(h.heading[i]), math.sin(h.heading[i])
            rot = np.array([[c, s], [-s, c]]) * L
            parts = _parts(key, float(h.phase[i]) + clock * (1 if h.sp.flier else 0))
            body = np.concatenate([p for p, _ in parts])
            m = len(body)
            # stín, světlý lem (silueta zvětšená kolem středu těla), pak tělo – vše jedním polem
            verts.append(body @ rot + [x + L * lift, y + L * lift * 1.3])
            verts.append(body @ (rot * HALO) + [x, y])
            verts.append(body @ rot + [x, y])
            c = np.empty((3 * m, 4), np.float32)
            c[:m] = (0, 0, 0, SHADOW[3] * vis)
            c[m:2 * m] = (*HALO_COLOR[:3], HALO_COLOR[3] * vis * (0.6 if key == "ryba" else 1.0))
            c[2 * m:] = np.repeat(np.array([col for _, col in parts], np.float32), [len(p) for p, _ in parts], 0)
            c[2 * m:, 3] *= vis
            cols.append(c)
    if verts:
        host.painter.triangles(np.concatenate(verts).astype(np.float32), np.concatenate(cols).astype(np.float32))
