"""Skutečná krajina: letecký snímek skutečného místa v ČR promítnutý na písek.

Data jsou z bezplatných služeb ČÚZK (Ortofoto ČR + výškový model DMR 5G), stahují se předem
(``python -m sandbox ortofoto``, viz ``sandbox/ortofoto.py``) a mód běží offline.

Fotka se stínuje podle skutečného tvaru písku, takže vypadá trojrozměrně. Úkol je postavit
z písku reliéf toho místa (Říp, Macocha, Pražský hrad…): barevná nápověda ukazuje, kde přidat
a kde ubrat, hodnotí se tvar (mírně – výška kopců se smí lišit). Když krajina sedí, nápověda
zmizí a zůstane čistá fotka na správně tvarovaném terénu.
"""

from __future__ import annotations

import math
import time

import cv2
import moderngl
import numpy as np

from ..ortofoto import Landscape, crop_to_aspect, load_all
from ..render import hud
from .base import Game
from .targets import Score, evaluate_shape

FRAG = """
uniform sampler2D u_photo;
uniform sampler2D u_target;
uniform int u_hint;
uniform int u_lines;
uniform float u_tol;
uniform float u_offset;
uniform float u_scale;
uniform float u_reveal;   // 0..1: úspěch – nápověda zmizí, fotka se rozzáří
uniform float u_since;    // [s] od úspěchu (záblesk, světelný přejezd, jiskry); < 0 = není úspěch

float krajina_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// zlaté jiskry stoupající nad krajinou při odhalení (velké kulaté body, ne šum)
float krajina_sparks(vec2 pm, float t) {
    float cell = 0.06;
    vec2 q = pm + vec2(0.0, t * 0.035);            // stoupají (k menšímu y)
    vec2 id = floor(q / cell);
    vec2 f = q / cell - id;
    float r = krajina_hash(id);
    vec2 c = vec2(0.2 + 0.6 * krajina_hash(id + 7.1), 0.2 + 0.6 * krajina_hash(id + 3.7));
    float tw = 0.5 + 0.5 * sin(t * (2.0 + 3.0 * r) + r * 40.0);
    float d = length((f - c) * cell);              // [m]
    return step(0.70, r) * tw * (1.0 - smoothstep(0.004, 0.008, d));
}

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;
    float h = height_at(uv);
    vec3 c = pow(texture(u_photo, uv).rgb, vec3(0.76));   // zvednuté stíny (tmavý les na drti zaniká)
    // barevné podání: sytější, kontrastnější fotka (projekce na bílé drti jinak vybledne)
    float lum0 = dot(c, vec3(0.3, 0.55, 0.15));
    float sat = 1.25 + 0.25 * u_reveal;
    c = max(mix(vec3(lum0), c, sat), 0.0);
    c = mix(c, c * c * (3.0 - 2.0 * c), 0.45 + 0.25 * u_reveal);   // S-křivka
    // reliéf písku dává fotce hloubku: teplé slunce na svazích, chladné stíny v údolích
    float shade = hillshade(uv, 3.2 + 1.3 * u_reveal);   // při odhalení reliéf „vystoupí“
    vec3 light = mix(vec3(0.55, 0.60, 0.85), vec3(1.38, 1.24, 1.02), shade);
    c *= light;
    float cav = cavity(uv);
    c *= 1.0 - clamp(cav / 0.010, 0.0, 1.0) * 0.22;                // hlubší údolí tmavší
    c += vec3(1.0, 0.85, 0.55) * pow(shade, 5.0) * 0.22;           // zlatý lesk na hřebenech
    float t = sample_bicubic(u_target, uv) * u_scale + u_offset;
    float d = h - t;
    if (u_hint == 1) {
        float k = clamp((abs(d) - 0.6 * u_tol) / (2.0 * u_tol), 0.0, 1.0) * (1.0 - u_reveal);
        k *= 0.88 + 0.12 * sin(u_time * 2.4);                     // nápověda pomalu „dýchá“
        vec3 tint = d < 0.0 ? vec3(1.0, 0.12, 0.06) : vec3(0.10, 0.35, 1.0);
        float lum = dot(c, vec3(0.3, 0.55, 0.15));
        c = mix(c, tint * (0.8 + 0.5 * lum), k * 0.6);
    }
    if (u_lines == 1) {                                  // vrstevnice skutečného terénu
        float line = contour_line(t, u_contour_interval, 1.6);
        c = mix(c, vec3(1.0, 0.97, 0.85), line * 0.55 * (1.0 - u_reveal));
    }
    // mírná vinětace – pozornost do středu (při odhalení zesílí jako ve filmu)
    vec2 vq = (uv - 0.5) * vec2(1.0, 1.2);
    c *= 1.0 - (0.18 + 0.12 * u_reveal) * smoothstep(0.25, 0.75, dot(vq, vq) * 2.0);
    c *= 1.0 + 0.15 * u_reveal;
    if (u_since >= 0.0) {
        // záblesk při odhalení
        float flash = exp(-u_since * 4.0) * smoothstep(0.0, 0.06, u_since);
        c = mix(c, vec3(1.0, 0.97, 0.88), flash * 0.75);
        // světelný přejezd šikmo přes krajinu (jako odlesk na fotce)
        float s = uv.x * 0.8 + (1.0 - uv.y) * 0.45;
        float sc = -0.3 + u_since * 0.9;
        float band = exp(-pow((s - sc) / 0.045, 2.0)) * (1.0 - smoothstep(1.4, 2.4, u_since));
        c += vec3(1.0, 0.86, 0.55) * band * (0.6 + 0.3 * shade);
        // zlaté jiskry
        float sp = krajina_sparks(pm, u_time) * u_reveal * (0.45 + 0.55 * exp(-u_since / 3.0));
        c = mix(c, vec3(1.0, 0.88, 0.45), sp);
    }
    f_color = vec4(c, 1.0);
}
"""

LEGEND = [((1.0, 0.15, 0.08, 1), "přidej písek"), ((0.15, 0.35, 1.0, 1), "uber")]


class LandscapeGame(Game):
    id = "krajina"
    title = "Skutečná krajina"
    description = "Letecký snímek skutečného místa v Česku. Postav z písku jeho kopce a údolí."
    order = 40
    category = "zážitek"
    intro = "Na písku je letecký snímek skutečného místa – postav jeho kopce a údolí! Červená = přidej, modrá = uber."
    keys = [
        ("← / →", "jiné místo"),
        ("M", "barevná nápověda zap/vyp"),
        ("V", "vrstevnice skutečného terénu"),
    ]

    SUCCESS = 70.0  # [%] mírné hodnocení tvaru
    RELIEF = 0.12  # [m] skutečné převýšení se promítne na tolik cm písku
    TOL = 0.02  # [m] tolerance barevné nápovědy

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.places: list[Landscape] = load_all()
        self.index = getattr(self, "index", 0) % max(1, len(self.places))
        self.hint = getattr(self, "hint", True)
        self.lines = getattr(self, "lines", False)
        self.photo_tex: moderngl.Texture | None = None
        self.target_tex: moderngl.Texture | None = None
        self.grid: np.ndarray | None = None
        self.result: Score | None = None
        self.success_since: float | None = None
        self.best = 0.0
        self._next_eval = 0.0
        self._built_for: tuple | None = None

    def stop(self) -> None:
        self.prog.release()
        self._release()

    def _release(self) -> None:
        for tex in (self.photo_tex, self.target_tex):
            if tex is not None:
                tex.release()
        self.photo_tex = self.target_tex = None

    def reset(self) -> None:
        self._built_for = None
        self.success_since = None
        self.best = 0.0

    @property
    def place(self) -> Landscape | None:
        return self.places[self.index] if self.places else None

    def _build(self, hm) -> None:
        place = self.place
        self._release()
        self._built_for = (self.index, hm.shape, hm.size)
        self.result, self.success_since, self.best = None, None, 0.0
        if place is None:
            self.grid = None
            return
        aspect = hm.size[0] / hm.size[1]
        photo = np.ascontiguousarray(crop_to_aspect(place.photo, aspect))
        ph, pw = photo.shape[:2]
        self.photo_tex = self.host.ctx.texture((pw, ph), 3, photo.tobytes())
        self.photo_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.photo_tex.build_mipmaps()
        self.photo_tex.repeat_x = self.photo_tex.repeat_y = False
        dem = crop_to_aspect(place.dem, aspect)
        gh, gw = hm.shape
        dem = cv2.resize(dem, (gw, gh), interpolation=cv2.INTER_AREA)
        dem = cv2.GaussianBlur(dem, (0, 0), 1.2)  # jemné detaily (stromy, domy) písek neudrží
        span = max(float(dem.max() - dem.min()), 1.0)
        self.grid = ((dem - dem.min()) / span * self.RELIEF).astype(np.float32)
        self.grid -= float(self.grid.mean())
        self.target_tex = self.host.float_texture((gw, gh), self.grid)

    # --- logika -------------------------------------------------------------------
    def update(self, hm, dt: float) -> None:
        if self._built_for != (self.index, hm.shape, hm.size):
            self._build(hm)
        if self.grid is None:
            return
        now = time.monotonic()
        if now < self._next_eval:
            return
        self._next_eval = now + 0.3
        self.result, _ = evaluate_shape(hm.heights, self.grid, sigma=2.5)
        self.best = max(self.best, self.result.score)
        if self.result.score >= self.SUCCESS:
            if self.success_since is None:
                self.success_since = now
        elif self.result.score < self.SUCCESS - 8:  # hystereze, ať úspěch neblikne
            self.success_since = None

    # --- kreslení -----------------------------------------------------------------
    def render(self) -> None:
        p = self.host.painter
        canvas = self.host.canvas_size
        cw, ch = canvas
        size = hud.size(canvas)
        place = self.place
        if place is None or self.photo_tex is None:
            hud.label(p, "Skutečná krajina: chybí stažená data", cw / 2, ch / 2 - size, size)
            hud.label(p, "Obsluha: python -m sandbox ortofoto (potřebuje internet)", cw / 2, ch / 2 + size,
                      hud.size(canvas, 0.75), bold=False)
            return
        reveal, since = 0.0, -1.0
        if self.success_since is not None:
            since = time.monotonic() - self.success_since
            reveal = min(1.0, since / 1.5)
        r = self.result
        self.host.draw_fullscreen(self.prog, u_photo=(self.photo_tex, 1), u_target=(self.target_tex, 2),
                                  u_hint=int(self.hint), u_lines=int(self.lines), u_tol=self.TOL,
                                  u_offset=r.offset if r else 0.0, u_scale=r.scale if r else 1.0,
                                  u_reveal=float(reveal), u_since=float(since))
        if since >= 0:
            # filmový moment: záře, zatmívací pruhy nahoře a dole (cinemascope)
            self.host.glow(0.3 + 0.8 * math.exp(-since * 1.5), 0.86)
            k = min(1.0, since / 0.9)
            k = 1 - (1 - k) ** 3
            bar_h = ch * 0.085 * k
            p.rect(0, 0, cw, bar_h, (0, 0, 0, 0.88))
            p.rect(0, ch - bar_h, cw, bar_h, (0, 0, 0, 0.88))
        small = hud.size(canvas, 0.75)
        hud.label(p, place.name, size * 0.3, size * 0.25, size, anchor="tl", color=hud.YELLOW)
        hud.label(p, f"{place.min_m:.0f}–{place.max_m:.0f} m n. m. · výřez " + f"{place.width_m / 1000:.1f}".replace(".", ",") + " km",
                  size * 0.3, size * 1.75, small, anchor="tl", bold=False)
        self._north(p, cw - size * 0.9, size * 1.0, size * 0.6)
        p.text(place.source, cw - 6, ch - 4, size=max(11, ch // 60), anchor="br", color=(1, 1, 1, 0.8), outline=2,
               tiny=True)
        if self.hint and reveal < 1.0:
            hud.legend(p, LEGEND, size * 0.3, ch - small * 0.3, small, anchor="bl")
        if r is None:
            return
        right, bottom = cw - size * 0.3, ch - size * 0.9
        if self.success_since is not None:
            hud.label(p, f"Výborně! Takhle vypadá {place.name} doopravdy", cw / 2, ch / 2, int(size * 1.15),
                      color=hud.GREEN, alpha=reveal)
            hud.label(p, place.note, cw / 2, ch / 2 + size * 1.5, small, bold=False, alpha=reveal)
        hud.label(p, f"Shoda {r.score:.0f} %", right, bottom, size, anchor="br",
                  color=hud.GREEN if self.success_since else hud.WHITE)
        bw = cw * 0.22
        hud.bar(p, right - bw, bottom - size * 2.0, bw, size * 0.3, r.score / self.SUCCESS,
                hud.GREEN if self.success_since else hud.YELLOW)

    @staticmethod
    def _north(p, x: float, y: float, r: float) -> None:
        """Šipka na sever (nahoře je sever – tak jsou stažené snímky)."""
        pts = np.array([[0, -1.0], [0.55, 0.7], [0, 0.35], [-0.55, 0.7]]) * r + [x, y]
        p.triangles(np.array([pts[0], pts[1], pts[2], pts[0], pts[2], pts[3]]) + [2, 3], (0, 0, 0, 0.6))
        p.triangles(np.array([pts[0], pts[1], pts[2]]), (1, 1, 1, 1))
        p.triangles(np.array([pts[0], pts[2], pts[3]]), (0.75, 0.8, 0.9, 1))
        p.text("S", x, y + r * 0.95, size=int(r * 1.1), anchor="tc", bold=True, color=(1, 1, 1, 1), outline=2)

    # --- ovládání ------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name in ("left", "right") and self.places:
            self.index = (self.index + (1 if name == "right" else -1)) % len(self.places)
            self.host.notify(f"Místo: {self.places[self.index].name}")
        elif name == "m":
            self.hint = not self.hint
            self.host.notify("Nápověda zapnutá" if self.hint else "Nápověda vypnutá")
        elif name == "v":
            self.lines = not self.lines
            self.host.notify("Vrstevnice skutečného terénu " + ("zapnuté" if self.lines else "vypnuté"))
        else:
            return False
        return True

    def status(self) -> list[str]:
        if not self.places:
            return ["Chybí data: python -m sandbox ortofoto"]
        lines = [f"Místo {self.index + 1}/{len(self.places)}: {self.place.name}"]
        if self.result:
            lines.append(f"Shoda {self.result.score:.0f} % (nejlépe {self.best:.0f} %), měřítko ×{self.result.scale:.2f}")
        return lines
