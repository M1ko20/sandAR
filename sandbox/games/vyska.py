"""Výšková mapa: odstíny šedi nebo teplotní škála, s číselnou stupnicí.

Vzhled „termokamera / sci-fi skener“: sytá žhavá škála (fialová → červená → žlutá → bílá),
zářící izolinie, mřížka po 10 cm, přes písek běží paprsek skeneru a zaměřovače ukazují
nejvyšší a nejnižší místo i ruce nad pískem (s výškou dlaně).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..render import hud
from .base import Game

FRAG = """
uniform int u_gray;
uniform vec2 u_range;      // min, max výšky pro škálu
uniform vec4 u_legend;
uniform float u_beam;      // poloha paprsku skeneru (uv.x)

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
    vec2 pm = uv * u_box;
    // mřížka po 10 cm (měřítko skeneru)
    vec2 gq = abs(fract(pm / 0.10 + 0.5) - 0.5) * 0.10 * u_canvas / u_box;   // vzdálenost v px
    float grid = 1.0 - smoothstep(1.0, 2.2, min(gq.x, gq.y));
    vec3 hud_c = u_gray == 1 ? vec3(0.2, 0.8, 1.0) : vec3(0.35, 0.95, 1.0);
    c = mix(c, hud_c, grid * 0.22);
    // paprsek skeneru: jasná čára + dosvit za ní (rozzáří barvy, kudy prošel)
    float dx = (uv.x - u_beam) * u_box.x;        // [m], záporné = paprsek už prošel
    float core = exp(-abs(dx) / 0.004);
    float trail = dx < 0.0 ? exp(dx / 0.12) : 0.0;
    c = c * (1.0 + 0.45 * trail) + hud_c * trail * 0.10;
    c = mix(c, vec3(0.85, 1.0, 1.0), core * 0.85);
    // v místě paprsku zesílí izolinie (skener „čte“ výšky)
    f_color = vec4(c, 1.0);
}
"""

BEAM_PERIOD = 4.5  # [s] přejezd paprsku přes bednu


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
        self.extremes: list[tuple[float, float, float, bool]] = []  # (x, y [m], výška, je max)
        self.hands: list[tuple[float, float, float]] = []
        self._next_ext = 0.0

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
        self.hands = [(h.x, h.y, h.height) for h in hm.hand_list]
        now = self.host.time
        if now >= self._next_ext:
            self._next_ext = now + 0.25
            self.extremes = self._find_extremes(hm)

    @staticmethod
    def _find_extremes(hm) -> list[tuple[float, float, float, bool]]:
        h = cv2.GaussianBlur(hm.heights, (0, 0), 2.0)
        gh, gw = h.shape
        bx, by = max(2, gw // 12), max(2, gh // 8)
        inner = h[by:-by, bx:-bx]
        if hm.hands is not None and hm.hands.shape == h.shape:
            near = cv2.dilate(hm.hands.astype(np.uint8), np.ones((11, 11), np.uint8))[by:-by, bx:-bx] > 0
            inner = np.where(near, np.nan, inner)
        if np.all(np.isnan(inner)):
            return []
        cw_m, ch_m = hm.cell_size
        out = []
        for fn, is_max in ((np.nanargmax, True), (np.nanargmin, False)):
            iy, ix = np.unravel_index(fn(inner), inner.shape)
            out.append(((ix + bx + 0.5) * cw_m, (iy + by + 0.5) * ch_m, float(inner[iy, ix]), is_max))
        return out

    def _beam(self) -> float:
        k = (self.host.time % BEAM_PERIOD) / BEAM_PERIOD
        return -0.1 + 1.25 * k

    def render(self) -> None:
        host = self.host
        legend = (0.955, 0.08, 0.985, 0.92)
        host.draw_fullscreen(self.prog, u_gray=int(self.gray), u_range=self.range, u_legend=legend,
                             u_beam=self._beam())
        host.glow(0.6, 0.93)
        cw, ch = host.canvas_size
        p = host.painter
        size = hud.size((cw, ch), 0.7)
        cyan = (0.4, 0.95, 1.0, 1.0)
        self._corners(p, cw, ch, size, cyan)
        hud.label(p, "HLOUBKOVÝ SKEN", size * 0.9, size * 0.9, size, anchor="tl", color=cyan,
                  bg=(0, 0.04, 0.1, 0.55))
        t = host.time
        for x, y, h, is_max in self.extremes:
            px, py = host.box_to_canvas(x, y)
            col = (1.0, 1.0, 0.6, 1) if is_max else (0.55, 0.85, 1.0, 1)
            self._reticle(p, px, py, size * 1.1, col, t * (1.2 if is_max else -1.2))
            txt = ("MAX " if is_max else "MIN ") + f"{h * 100:+.1f} cm".replace(".", ",")
            anchor = "ml" if px < cw * 0.75 else "mr"
            ox = size * 1.6 if anchor == "ml" else -size * 1.6
            hud.label(p, txt, px + ox, py, size, anchor=anchor, color=col, bg=(0, 0.02, 0.08, 0.6))
        for x, y, hh in self.hands:
            px, py = host.box_to_canvas(x, y)
            pulse = 1 + 0.08 * math.sin(t * 8)
            self._reticle(p, px, py, size * 1.8 * pulse, (0.3, 1.0, 0.5, 1), t * 2.5, brackets=True)
            hud.label(p, f"RUKA {hh * 100:.0f} cm", px, py - size * 2.6, size, color=(0.4, 1.0, 0.55, 1),
                      bg=(0, 0.06, 0.02, 0.6))
        for k in range(5):
            tt = k / 4
            h = self.range[1] - tt * (self.range[1] - self.range[0])
            y = (legend[1] + tt * (legend[3] - legend[1])) * ch
            hud.label(p, f"{h * 100:+.1f} cm".replace(".", ","), legend[0] * cw - 8, y, size,
                      anchor="mr", bg=(0, 0, 0, 0.4))

    @staticmethod
    def _reticle(p, x: float, y: float, r: float, col, ang: float, brackets: bool = False) -> None:
        """Zaměřovač: čtyři otáčející se oblouky + kříž (jeden balík trojúhelníků)."""
        w = max(4.0, r * 0.12)
        tris = []
        for q in range(4):
            a0 = ang + q * math.pi / 2 + 0.25
            a1 = a0 + (1.0 if not brackets else 0.7)
            seg = np.linspace(a0, a1, 8)
            outer = np.stack([x + r * np.cos(seg), y + r * np.sin(seg)], 1)
            inner = np.stack([x + (r - w) * np.cos(seg), y + (r - w) * np.sin(seg)], 1)
            for i in range(len(seg) - 1):
                tris += [outer[i], outer[i + 1], inner[i + 1], outer[i], inner[i + 1], inner[i]]
        L, g = r * 0.55, r * 0.18
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = x + dx * g, y + dy * g
            bx, by = x + dx * L, y + dy * L
            nx, ny = -dy * w * 0.4, dx * w * 0.4
            tris += [[ax + nx, ay + ny], [bx + nx, by + ny], [bx - nx, by - ny],
                     [ax + nx, ay + ny], [bx - nx, by - ny], [ax - nx, ay - ny]]
        tris = np.array(tris, np.float32)
        shadow = tris + np.array([2, 2], np.float32)
        p.triangles(shadow, (0, 0, 0, 0.6))
        p.triangles(tris, col)

    @staticmethod
    def _corners(p, cw: int, ch: int, size: float, col) -> None:
        """Rohové závorky HUD (rámeček hledáčku)."""
        L, w, m = size * 2.2, max(5.0, size * 0.22), size * 0.35
        tris = []
        for sx, sy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            x = m if sx == 0 else cw - m
            y = m if sy == 0 else ch - m
            dx = 1 if sx == 0 else -1
            dy = 1 if sy == 0 else -1
            for (x0, y0, x1, y1) in ((x, y, x + dx * L, y + dy * w), (x, y, x + dx * w, y + dy * L)):
                a, b = sorted((x0, x1)), sorted((y0, y1))
                tris += [[a[0], b[0]], [a[1], b[0]], [a[1], b[1]], [a[0], b[0]], [a[1], b[1]], [a[0], b[1]]]
        p.triangles(np.array(tris, np.float32), col)

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
