"""Společné věci pro hry dvou hráčů (Vodní válka, Král kopce, Potopa ostrovů, Modeluj podle předlohy).

Bedna se dělí čárou napříč přes střed delší strany. Hráč 0 stojí u levé kratší strany
plátna (x = 0) a má levou půlku, hráč 1 stojí u pravé kratší strany a má pravou půlku.
Texty pro hráče se kreslí otočené k němu (``rotation`` v ``Painter.text``), aby je
přečetl z místa, kde stojí.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

NAMES = ("Červení", "Modří")
COLORS = ((1.0, 0.32, 0.22, 1.0), (0.30, 0.66, 1.0, 1.0))
DARK = ((0.35, 0.05, 0.02, 1.0), (0.02, 0.12, 0.35, 1.0))
ROTATION = (90.0, -90.0)  # text čitelný od levé / pravé kratší strany


def half_cols(gw: int, player: int) -> slice:
    """Sloupce mřížky, které patří hráči (levá / pravá půlka)."""
    return slice(0, gw // 2) if player == 0 else slice(gw // 2, gw)


def player_of_x(x: float, width: float) -> int:
    return 0 if x < width / 2 else 1


def zone_mask(shape: tuple[int, int], player: int, border: int = 0, neutral: int = 0) -> np.ndarray:
    """Buňky hráčovy půlky bez okraje bedny (``border``) a bez pásu u dělicí čáry (``neutral``)."""
    gh, gw = shape
    m = np.zeros(shape, bool)
    cols = half_cols(gw, player)
    m[:, cols] = True
    if border:
        m[:border, :] = m[gh - border:, :] = False
        m[:, :border] = m[:, gw - border:] = False
    if neutral:
        mid = gw // 2
        m[:, max(0, mid - neutral):mid + neutral] = False
    return m


def zone_peak(heights: np.ndarray, hands: np.ndarray | None, zone: np.ndarray) -> tuple[int, int, float] | None:
    """Nejvyšší místo v zóně mimo ruce: (iy, ix, výška [m]) nebo None.

    Výšky se lehce vyhladí, ať nevyhraje jeden zašuměný pixel.
    """
    h = cv2.GaussianBlur(np.asarray(heights, np.float32), (0, 0), 1.0)
    valid = zone.copy()
    if hands is not None and hands.any():
        valid &= cv2.dilate(hands.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    if not valid.any():
        return None
    k = int(np.argmax(np.where(valid, h, -np.inf)))
    iy, ix = divmod(k, h.shape[1])
    return iy, ix, float(h[iy, ix])


# --- kreslení --------------------------------------------------------------------------------
def anchor(canvas: tuple[int, int], player: int, dist: float, along: float = 0.5) -> tuple[float, float]:
    """Bod na plátně ve vzdálenosti ``dist`` (podíl šířky plátna) od hráčovy strany.

    ``along`` = poloha podél jeho strany zleva doprava z pohledu hráče (0..1).
    """
    cw, ch = canvas
    if player == 0:
        return cw * dist, ch * along
    return cw * (1 - dist), ch * (1 - along)


def player_text(painter, canvas: tuple[int, int], player: int, text: str, dist: float, size: int,
                color: Sequence[float] | None = None, along: float = 0.5, **kw) -> tuple[float, float]:
    """Text otočený k hráči, se středem ``dist`` od jeho strany plátna."""
    x, y = anchor(canvas, player, dist, along)
    kw.setdefault("bold", True)
    kw.setdefault("outline", max(2, size // 12))
    return painter.text(text, x, y, size=size, anchor="mc", color=color or COLORS[player],
                        rotation=ROTATION[player], **kw)


def text_size(canvas: tuple[int, int], scale: float = 1.0) -> int:
    """Velikost písma čitelná ze 2 m (alespoň výška plátna / 30)."""
    return max(16, int(canvas[1] / 22 * scale))


def draw_divider(painter, canvas: tuple[int, int], neutral_px: float = 0.0, frame: bool = True) -> None:
    """Výrazná dělicí čára přes střed delší strany (+ volitelně neutrální pás) a barevné rámečky půlek."""
    cw, ch = canvas
    mid = cw / 2
    if neutral_px > 0:
        painter.rect(mid - neutral_px, 0, 2 * neutral_px, ch, (0, 0, 0, 0.35))
        step = ch / 24
        for k in range(25):  # šikmé šrafy neutrálního pásu
            y = k * step
            painter.polyline([(mid - neutral_px, y), (mid + neutral_px, y + step * 0.8)], (1, 1, 1, 0.25),
                             width=max(2, ch / 300))
    lw = max(6.0, ch / 90)
    painter.rect(mid - lw, 0, 2 * lw, ch, (0, 0, 0, 0.85))
    painter.rect(mid - lw * 0.45, 0, lw * 0.9, ch, (1, 1, 1, 1))
    if frame:
        fw = max(4.0, ch / 140)
        for player in (0, 1):
            x0 = 0 if player == 0 else mid + lw
            x1 = mid - lw if player == 0 else cw
            painter.rect_outline(x0 + fw / 2, fw / 2, x1 - x0 - fw, ch - fw, COLORS[player], width=fw)


def draw_crown(painter, x: float, y: float, r: float, rotation: float, color=(1.0, 0.85, 0.15, 1.0)) -> None:
    """Korunka (pro vítěze / krále) otočená stejně jako text hráče."""
    pts = np.array([[-1, 0.55], [1, 0.55], [1, -0.45], [0.5, 0.05], [0, -0.7], [-0.5, 0.05], [-1, -0.45]],
                   np.float32) * r
    a = np.radians(rotation)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    pts = pts @ rot.T + np.array([x, y], np.float32)
    tris = [pts[i] for t in ((0, 1, 2), (0, 2, 3), (0, 3, 5), (0, 5, 6), (3, 4, 5)) for i in t]
    outline = (pts - np.array([x, y])) * 1.25 + np.array([x, y])
    painter.triangles(np.array([outline[i] for t in ((0, 1, 2), (0, 2, 3), (0, 3, 5), (0, 5, 6), (3, 4, 5))
                                for i in t]), (0.2, 0.1, 0.0, 1))
    painter.triangles(np.array(tris), color)
