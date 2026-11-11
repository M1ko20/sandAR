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
        g = self._intro_game
        now = time.monotonic()
        if g is None or now >= self._intro_until:
            return
        t_in = now - self._intro_start
        t_out = self._intro_until - now
        alpha = min(1.0, t_in / 0.4, t_out / 0.6)
        if alpha <= 0:
            return
        p = host.painter
        cw, ch = host.canvas_size
        band_h = ch * 0.30
        y0 = ch / 2 - band_h / 2
        p.rect(0, y0, cw, band_h, (0.02, 0.03, 0.06, 0.78 * alpha))
        # světelná linka, která přejede přes pás
        sweep = (t_in * 0.9) % 1.0
        p.rect(cw * sweep - cw * 0.08, y0, cw * 0.16, 3, (0.4, 0.8, 1.0, 0.8 * alpha))
        p.rect(cw * (1 - sweep) - cw * 0.08, y0 + band_h - 3, cw * 0.16, 3, (0.4, 0.8, 1.0, 0.8 * alpha))
        label = CATEGORY_LABEL.get(g.category, g.category.upper())
        p.text(label, cw / 2, y0 + band_h * 0.17, size=max(14, ch // 32), anchor="mc", bold=True,
               color=(0.45, 0.8, 1.0, 1), alpha=alpha)
        p.text(g.title, cw / 2, y0 + band_h * 0.45, size=max(24, ch // 11), anchor="mc",
               bold=True, outline=3, alpha=alpha)
        how = g.intro or g.description
        if how:
            p.text(how, cw / 2, y0 + band_h * 0.78, size=max(16, ch // 24), anchor="mc",
                   color=(1.0, 0.92, 0.55, 1), outline=2, alpha=alpha)
