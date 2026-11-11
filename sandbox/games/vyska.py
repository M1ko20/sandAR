"""Výšková mapa: odstíny šedi nebo teplotní škála, s číselnou stupnicí."""

from __future__ import annotations

import numpy as np

from .base import Game

FRAG = """
uniform int u_gray;
uniform vec2 u_range;      // min, max výšky pro škálu
uniform vec4 u_legend;
void main() {
    vec2 uv = v_uv;
    float t;
    if (uv.x >= u_legend.x && uv.x <= u_legend.z && uv.y >= u_legend.y && uv.y <= u_legend.w) {
        t = 1.0 - (uv.y - u_legend.y) / (u_legend.w - u_legend.y);
    } else {
        t = (height_at(uv) - u_range.x) / max(u_range.y - u_range.x, 1e-4);
    }
    vec3 c = u_gray == 1 ? vec3(clamp(t, 0.0, 1.0)) : heat_color(t);
    f_color = vec4(c, 1.0);
}
"""


class HeightGame(Game):
    id = "vyska"
    title = "Výšková mapa"
    description = "Výška písku jako barva nebo odstín šedi (jak ji vidí počítač)."
    order = 30
    category = "vizualizace"
    attract = False  # pro ukázky nudné, je hlavně pro ladění
    intro = "Takhle vidí pískoviště počítač – každá barva je jiná výška."
    keys = [("G", "šedá / barevná"), ("A", "automatický rozsah")]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.gray = False
        self.auto = True
        self.range = (self.host.settings.map.elevation_min, self.host.settings.map.elevation_max)

    def stop(self) -> None:
        self.prog.release()

    def update(self, hm, dt: float) -> None:
        if self.auto:
            lo, hi = np.percentile(hm.heights, [1, 99])
            if hi - lo < 0.02:
                mid = (hi + lo) / 2
                lo, hi = mid - 0.01, mid + 0.01
            a = min(1.0, dt * 2)
            self.range = (self.range[0] + a * (lo - self.range[0]), self.range[1] + a * (hi - self.range[1]))
        else:
            m = self.host.settings.map
            self.range = (m.elevation_min, m.elevation_max)

    def render(self) -> None:
        legend = (0.955, 0.08, 0.985, 0.92)
        self.host.draw_fullscreen(self.prog, u_gray=int(self.gray), u_range=self.range, u_legend=legend)
        cw, ch = self.host.canvas_size
        size = max(12, ch // 45)
        for k in range(5):
            t = k / 4
            h = self.range[1] - t * (self.range[1] - self.range[0])
            y = (legend[1] + t * (legend[3] - legend[1])) * ch
            self.host.painter.text(f"{h * 100:+.1f} cm", legend[0] * cw - 8, y, size=size, anchor="mr",
                                   color=(1, 1, 1, 1), outline=2)

    def key(self, name: str, shift: bool = False) -> bool:
        if name == "g":
            self.gray = not self.gray
        elif name == "a":
            self.auto = not self.auto
            self.host.notify("Automatický rozsah " + ("zapnut" if self.auto else "vypnut"))
        else:
            return False
        return True

    def status(self) -> list[str]:
        return [f"Rozsah {self.range[0] * 100:+.1f} … {self.range[1] * 100:+.1f} cm"]
