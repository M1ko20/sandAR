"""Provoz bez obsluhy: úvodní slajd módu, attract mode a automatický reset.

* Po výběru módu se na písek na pár sekund promítne jeho název a jedna věta „jak hrát“.
* Aktivita = ruce nad pískem nebo změna terénu (nebo klávesa operátora).
* Když je pískoviště `idle_seconds` v klidu, zapne se attract mode: módy s ``attract = True``
  se samy střídají po `cycle_seconds` a každý začne s čistým světem (``reset()``).
  Jakmile někdo sáhne do písku, střídání se zastaví a mód zůstane.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from .config import AttractSettings

if TYPE_CHECKING:
    from .games import Game
    from .render.host import RenderHost
    from .terrain.heightmap import HeightMap

CATEGORY_LABEL = {
    "vizualizace": "VIZUALIZACE",
    "simulace": "SIMULACE",
    "soutěž": "SOUTĚŽ",
    "hra": "HRA",
    "zážitek": "ZÁŽITEK",
    "dva hráči": "HRA PRO DVA HRÁČE",
}


# barva kategorie (štítek, záře názvu, linky úvodního slajdu)
CATEGORY_COLOR = {
    "vizualizace": (0.35, 0.85, 1.0),
    "simulace": (0.30, 0.75, 1.0),
    "soutěž": (1.0, 0.80, 0.20),
    "hra": (0.40, 1.0, 0.50),
    "zážitek": (1.0, 0.55, 0.20),
    "dva hráči": (1.0, 0.35, 0.55),
}


class AttractController:
    def __init__(self, settings: AttractSettings) -> None:
        self.s = settings
        self.last_activity = time.monotonic()
        self.active = False
        self._next_switch = 0.0
        self._prev_heights: np.ndarray | None = None
        self._intro_game: type[Game] | None = None
        self._intro_until = 0.0
        self._intro_start = 0.0

    # --- aktivita ------------------------------------------------------------------------
    def poke(self) -> None:
        """Někdo něco dělá (klávesa operátora, ruka, změna terénu)."""
        self.last_activity = time.monotonic()
        self.active = False

    def observe(self, hm: HeightMap) -> None:
        if hm.hand_points:
            self.poke()
        h = hm.heights
        if self._prev_heights is None or self._prev_heights.shape != h.shape:
            self._prev_heights = h.copy()
            return
        changed = float(np.mean(np.abs(h - self._prev_heights) > self.s.change_threshold))
        if changed > self.s.change_fraction:  # viditelná změna terénu
            self.poke()
            self._prev_heights = h.copy()

    @property
    def idle_for(self) -> float:
        return time.monotonic() - self.last_activity

    def want_switch(self) -> bool:
        """True = attract mode chce přepnout na další mód (volá app)."""
        if not self.s.enabled:
            return False
        now = time.monotonic()
        if not self.active:
            if self.idle_for >= self.s.idle_seconds:
                self.active = True
                self._next_switch = now  # hned první přepnutí (a reset světa)
            else:
                return False
        if now >= self._next_switch:
            self._next_switch = now + self.s.cycle_seconds
            return True
        return False

    # --- úvodní slajd --------------------------------------------------------------------
    def show_intro(self, game_cls: type[Game]) -> None:
        self._intro_game = game_cls
        self._intro_start = time.monotonic()
        self._intro_until = self._intro_start + self.s.intro_seconds

    def draw_intro(self, host: RenderHost) -> None:
        """Filmový úvodní slajd: pás se rozevře od středu, světelná linka přejede, název se rozzáří."""
        g = self._intro_game
        now = time.monotonic()
        if g is None or now >= self._intro_until:
            return
        draw_intro_slide(host, g, now - self._intro_start, self._intro_until - now)


def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def draw_intro_slide(host: RenderHost, g: type[Game], t_in: float, t_out: float) -> None:
    """Úvodní slajd v čase t_in od začátku a t_out do konce [s]."""
    from .render import hud
    from .render.text import get_font

    open_k = _ease(t_in / 0.45) * _ease(t_out / 0.45)  # rozevření / zavření pásu
    if open_k <= 0.01:
        return
    p = host.painter
    cw, ch = host.canvas_size
    accent = CATEGORY_COLOR.get(g.category, (0.45, 0.8, 1.0))
    band_h = ch * 0.38 * open_k
    cy = ch / 2
    y0, y1 = cy - band_h / 2, cy + band_h / 2
    # pás: tmavé sklo s měkkými okraji nahoře a dole (přechod do průhledna)
    fade = ch * 0.07
    dark = (0.01, 0.02, 0.05)
    verts: list[tuple[float, float]] = []
    cols: list[tuple[float, float, float, float]] = []

    def quad(ya: float, yb: float, aa: float, ab: float) -> None:
        verts.extend([(0, ya), (cw, ya), (cw, yb), (0, ya), (cw, yb), (0, yb)])
        cols.extend([(*dark, aa), (*dark, aa), (*dark, ab), (*dark, aa), (*dark, ab), (*dark, ab)])

    quad(y0 - fade, y0, 0.0, 0.88)
    quad(y0, y1, 0.88, 0.88)
    quad(y1, y1 + fade, 0.88, 0.0)
    p.triangles(np.array(verts, np.float32), np.array(cols, np.float32))
    # svítící linky na okrajích pásu + přejíždějící světlo
    line = max(3.0, ch / 150)
    for yy in (y0, y1 - line):
        p.rect(0, yy, cw, line, (*accent, 0.35 * open_k))
    sx = -cw * 0.2 + _ease(t_in / 1.2) * cw * 1.4
    for k, w in ((1.0, cw * 0.10), (0.5, cw * 0.24)):
        p.rect(sx - w / 2, y0, w, line, (*accent, 0.9 * k * open_k))
        p.rect(cw - sx - w / 2, y1 - line, w, line, (*accent, 0.9 * k * open_k))
    if open_k < 0.6:
        return
    a = _ease((t_in - 0.25) / 0.5) * _ease(t_out / 0.45)
    # kategorie jako barevný štítek
    label = CATEGORY_LABEL.get(g.category, g.category.upper())
    lsz = int(ch / 26)
    lw = get_font(lsz, True).getlength(label) + lsz * 1.4
    ly = y0 + band_h * 0.17
    p.rect(cw / 2 - lw / 2, ly - lsz * 0.72, lw, lsz * 1.44, (*accent, 0.92 * a))
    p.text(label, cw / 2, ly, size=lsz, anchor="mc", bold=True, color=(0.02, 0.03, 0.06, 1), alpha=a)
    # název: velký, s barevným halo (posunuté průsvitné vrstvy)
    tsz = hud.fit(g.title, ch / 8.5, cw * 0.9)
    ty = y0 + band_h * 0.46
    rise = (1 - _ease((t_in - 0.2) / 0.6)) * tsz * 0.35
    for r, al in ((tsz * 0.10, 0.10), (tsz * 0.05, 0.18)):
        for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)):
            p.text(g.title, cw / 2 + dx, ty + rise + dy, size=tsz, anchor="mc", bold=True,
                   color=(*accent, 1), alpha=al * a)
    p.text(g.title, cw / 2, ty + rise, size=tsz, anchor="mc", bold=True, outline=max(2, tsz // 18),
           alpha=a)
    # podtržení, které vyroste ze středu
    uw = cw * 0.34 * _ease((t_in - 0.35) / 0.6)
    p.rect(cw / 2 - uw / 2, ty + tsz * 0.64, uw, line, (*accent, 0.95 * a))
    how = g.intro or g.description
    if how:
        b = _ease((t_in - 0.6) / 0.5) * _ease(t_out / 0.45)
        hsz = hud.fit(how, ch / 22, cw * 0.94)
        p.text(how, cw / 2, y0 + band_h * 0.82, size=hsz, anchor="mc", bold=True,
               color=(1.0, 0.93, 0.6, 1), outline=max(2, hsz // 12), alpha=b)
