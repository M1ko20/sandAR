"""Výšková mapa: odstíny šedi nebo teplotní škála, s číselnou stupnicí.

Barevně sytá žhavá škála (fialová → červená → žlutá → bílá) s izoliniemi a reliéfem.
"""

from __future__ import annotations

import numpy as np

from ..render import hud
from .base import Game

FRAG = """
uniform int u_gray;
uniform vec2 u_range;      // min, max výšky pro škálu
uniform vec4 u_legend;

// žhavá škála termokamery: indigo → fialová → červená → oranžová → žlutá → bílá
const int VYSKA_N = 7;
const float VYSKA_T[VYSKA_N] = float[](0.0, 0.16, 0.34, 0.52, 0.70, 0.86, 1.0);
const vec3 VYSKA_C[VYSKA_N] = vec3[](
    vec3(0.10, 0.06, 0.55), vec3(0.42, 0.02, 0.72), vec3(0.85, 0.05, 0.45), vec3(1.00, 0.28, 0.05),
    vec3(1.00, 0.62, 0.00), vec3(1.00, 0.92, 0.20), vec3(1.00, 1.00, 0.92));

vec3 vyska_iron(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c = VYSKA_C[0];
    for (int i = 1; i < VYSKA_N; ++i) {
        float k = clamp((t - VYSKA_T[i - 1]) / (VYSKA_T[i] - VYSKA_T[i - 1]), 0.0, 1.0);
        c = mix(c, VYSKA_C[i], k);
    }
    return c;
}

void main() {
    vec2 uv = v_uv;
    float t;
    bool legend = uv.x >= u_legend.x && uv.x <= u_legend.z && uv.y >= u_legend.y && uv.y <= u_legend.w;
    if (legend) {
        t = 1.0 - (uv.y - u_legend.y) / (u_legend.w - u_legend.y);
    } else {
        t = (height_at(uv) - u_range.x) / max(u_range.y - u_range.x, 1e-4);
    }
    vec3 c;
    if (u_gray == 1) {
        c = vec3(clamp(t, 0.0, 1.0));
    } else {
        c = vyska_iron(t);
        // izolinie: 10 pásem přes rozsah (tmavé – kreslí tvar jako vrstevnice), mimo rozsah žádné
        float g = t * 10.0;
        float d = abs(fract(g) - 0.5);
        float w = max(fwidth(g), 1e-4);
        float iso = smoothstep(0.5 - 1.8 * w, 0.5 - 0.6 * w, d);
        // u krajů stupnice ne: plochý vršek/dno leží na izolinii a celý by se zbarvil
        iso *= smoothstep(0.02, 0.05, t) * (1.0 - smoothstep(0.93, 0.97, t));
        if (!legend) c = mix(c, c * vec3(0.35, 0.25, 0.55), iso * 0.8);
    }
    if (legend) { f_color = vec4(c, 1.0); return; }
    // reliéf, ať jsou tvary čitelné i v ploché barvě
    c *= mix(0.70, 1.15, hillshade(uv, 2.5));
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
        host = self.host
        legend = (0.955, 0.08, 0.985, 0.92)
        host.draw_fullscreen(self.prog, u_gray=int(self.gray), u_range=self.range, u_legend=legend)
        cw, ch = host.canvas_size
        p = host.painter
        size = hud.size((cw, ch), 0.7)
        for k in range(5):
            tt = k / 4
            h = self.range[1] - tt * (self.range[1] - self.range[0])
            y = (legend[1] + tt * (legend[3] - legend[1])) * ch
            hud.label(p, f"{h * 100:+.1f} cm".replace(".", ","), legend[0] * cw - 8, y, size,
                      anchor="mr", bg=(0, 0, 0, 0.4))

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
