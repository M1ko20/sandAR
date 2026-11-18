"""Barevná topografická mapa s vrstevnicemi – klasický režim AR pískoviště.

Vzhled: sytá hypsometrická škála (pobřeží → nížiny → hory → sníh), plastický reliéf s teplým
světlem a chladnými stíny, živá hladina vody s příbojem, vrstevnice s popisky výšek a jemná
animace – stíny mraků plují přes krajinu a co pár sekund vystoupá od moře k vrcholům zářící
„výšková vlna“ (ukazuje, co je stejně vysoko).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..render import hud
from .base import Game

MAX_LABELS = 10

FRAG = """
uniform int u_show_contours;
uniform int u_shade;
uniform vec4 u_legend;   // obdélník legendy v uv (x0, y0, x1, y1), x0 < 0 = vypnuto
uniform float u_scan;    // výška zářící vlny [m] (< -1 = vypnuto)
uniform float u_scan_a;  // síla vlny 0..1
uniform int u_nlabels;
uniform vec4 u_labels[10];  // popisky vrstevnic: střed uv, úhel [rad], polovina délky [m]

// Hypsometrická škála – syté barvy (bílá drť je jasná, pastely by vybledly)
const int TOPO_N = 9;
const float TOPO_T[TOPO_N] = float[](0.0, 0.07, 0.18, 0.31, 0.44, 0.58, 0.72, 0.86, 1.0);
const vec3 TOPO_C[TOPO_N] = vec3[](
    vec3(0.05, 0.55, 0.42),   // pobřežní nížina
    vec3(0.10, 0.66, 0.22),   // zelená
    vec3(0.52, 0.82, 0.16),   // světle zelená
    vec3(0.98, 0.88, 0.22),   // žlutá
    vec3(1.00, 0.60, 0.12),   // oranžová
    vec3(0.86, 0.30, 0.12),   // cihlová
    vec3(0.60, 0.20, 0.30),   // vínová skála
    vec3(0.62, 0.48, 0.70),   // fialová skála
    vec3(1.00, 1.00, 1.00));  // sníh

vec3 topo_land(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c = TOPO_C[0];
    for (int i = 1; i < TOPO_N; ++i) {
        float k = clamp((t - TOPO_T[i - 1]) / (TOPO_T[i] - TOPO_T[i - 1]), 0.0, 1.0);
        c = mix(c, TOPO_C[i], k);
    }
    return c;
}

float topo_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float topo_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(topo_hash(i), topo_hash(i + vec2(1, 0)), u.x),
               mix(topo_hash(i + vec2(0, 1)), topo_hash(i + vec2(1, 1)), u.x), u.y);
}
// stíny mraků: velké měkké skvrny (desítky cm), pomalu plují
float topo_clouds(vec2 pm) {
    vec2 q = pm * 2.2 + vec2(u_time * 0.035, u_time * 0.012);
    float n = 0.6 * topo_noise(q) + 0.4 * topo_noise(q * 1.9 + 4.7);
    return smoothstep(0.52, 0.72, n);
}

// popisek vrstevnice: v obdélníku kolem textu se vrstevnice přeruší (jako na tištěné mapě)
float topo_label_gap(vec2 uv) {
    vec2 pm = uv * u_box;
    float gap = 0.0;
    for (int i = 0; i < 10; ++i) {
        if (i >= u_nlabels) break;
        vec4 L = u_labels[i];
        vec2 d = pm - L.xy * u_box;
        float ca = cos(L.z), sa = sin(L.z);
        vec2 r = vec2(ca * d.x + sa * d.y, -sa * d.x + ca * d.y);
        vec2 q = abs(r) - vec2(L.w, 0.016);
        gap = max(gap, 1.0 - smoothstep(0.0, 0.004, max(q.x, q.y)));
    }
    return gap;
}

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;
    float h;
    bool legend = u_legend.x >= 0.0 && uv.x >= u_legend.x && uv.x <= u_legend.z
                  && uv.y >= u_legend.y && uv.y <= u_legend.w;
    if (legend) {
        float k = (uv.y - u_legend.y) / (u_legend.w - u_legend.y);
        h = mix(u_elev_max, u_elev_min, k);
    } else {
        h = height_at(uv);
    }
    bool water = h < u_water_level;
    float t = (h - u_water_level) / max(u_elev_max - u_water_level, 1e-4);
    vec3 c = topo_land(t);
    // pláž: úzký pruh písku u vody
    float beach = 1.0 - smoothstep(0.0, 0.004, h - u_water_level);
    c = mix(c, vec3(1.0, 0.86, 0.52), beach * 0.85);

    float hs = 0.62;
    if (u_shade == 1 && !legend) {
        hs = hillshade(uv, 3.0);
        // teplé světlo na osvětlených svazích, chladné modrofialové stíny – plastický reliéf
        vec3 lit = mix(vec3(0.42, 0.46, 0.78), vec3(1.16, 1.08, 0.92), hs);
        c *= lit;
        c += vec3(1.0, 0.95, 0.8) * pow(hs, 6.0) * 0.18;     // odlesk na hřbetech
    }
    if (!legend) {
        // stíny plujících mraků (jemné, velké)
        float cl = topo_clouds(pm);
        c *= mix(1.0, 0.80, cl);
    }

    if (u_show_contours == 1) {
        vec2 cl = contours(h);
        float gap = legend ? 0.0 : topo_label_gap(uv);
        float under = water ? 0.35 : 1.0;   // pod vodou slabší
        cl *= (1.0 - gap);
        c = mix(c, c * 0.42, cl.x * 0.75 * under);
        c = mix(c, vec3(0.10, 0.05, 0.03), cl.y * 0.95 * under);
    }

    if (water && !legend) {
        float depth = u_water_level - h;
        // příboj: vlny běží od hloubky ke břehu
        float surf = 0.5 + 0.5 * sin(depth / 0.0045 * 6.2832 + u_time * 2.2);
        float foam = smoothstep(0.75, 0.95, surf) * (1.0 - smoothstep(0.003, 0.012, depth)) * 0.8;
        vec2 vel = vec2(0.03, 0.012) + 0.02 * vec2(sin(u_time * 0.3 + pm.y * 4.0), cos(u_time * 0.25 + pm.x * 3.0));
        vec4 w = water_shade(c, uv, depth * 0.8 + 0.001, vel, foam);
        c = w.rgb;
        // hluboká voda: syté modré odstupňování podle hloubky (čitelná batymetrie)
        float d01 = clamp(depth / max(u_water_level - u_elev_min, 1e-4), 0.0, 1.0);
        c = mix(c, mix(vec3(0.0, 0.55, 0.95), vec3(0.02, 0.12, 0.55), d01), 0.35);
    } else if (water) {
        c = mix(vec3(0.25, 0.75, 1.0), vec3(0.02, 0.12, 0.55),
                clamp((u_water_level - h) / max(u_water_level - u_elev_min, 1e-4), 0.0, 1.0));
    }

    // pobřeží: výrazná tmavomodrá linka
    float shore = 1.0 - smoothstep(0.0, 1.8 * fwidth(h), abs(h - u_water_level));
    c = mix(c, vec3(0.02, 0.18, 0.55), shore * 0.9);

    // zářící výšková vlna – vystoupá od moře k vrcholům
    if (u_scan > -1.0 && !legend) {
        float dw = abs(h - u_scan);
        float wpx = max(fwidth(h), 2.5e-4);
        float core = 1.0 - smoothstep(1.5 * wpx, 2.8 * wpx, dw);
        float halo = exp(-dw / max(5.0 * wpx, 0.0012));
        float below = smoothstep(0.0, 0.02, u_scan - h) * (1.0 - smoothstep(0.0, 0.035, u_scan - h));
        vec3 sc = vec3(0.35, 1.0, 1.0);
        c = mix(c, c + sc * (core * 0.75 + halo * 0.25 + below * 0.10), u_scan_a);
    }
    f_color = vec4(c, 1.0);
}
"""

INTERVALS = [0.005, 0.01, 0.02]
SCAN_PERIOD = 14.0  # [s] jak často vystoupá výšková vlna
SCAN_RISE = 6.0  # [s] jak dlouho stoupá


def contour_labels(hm, levels: list[float], old: list[tuple] | None, limit: int = MAX_LABELS,
                   min_dist: float = 0.22, margin: float = 0.06) -> list[tuple]:
    """Místa pro popisky vrstevnic: [(x_m, y_m, úhel_rad, hladina)].

    Popisek leží na vrstevnici, natočený podél ní (text nikdy vzhůru nohama), mimo okraj a ruce.
    Staré popisky se ponechají, dokud vrstevnice pořád vede jejich místem (žádné poskakování).
    """
    hgt = hm.heights
    gh, gw = hgt.shape
    cw_m, ch_m = hm.cell_size
    sm = cv2.GaussianBlur(hgt, (0, 0), 1.5)
    gy, gx = np.gradient(sm, ch_m, cw_m)
    slope = np.hypot(gx, gy)
    ok = np.ones_like(sm, bool)
    bx, by = max(1, int(gw * margin)), max(1, int(gh * margin * 1.5))
    ok[:by] = ok[-by:] = False
    ok[:, :bx] = ok[:, -bx:] = False
    if hm.hands is not None and hm.hands.shape == ok.shape:
        ok &= ~cv2.dilate(hm.hands.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    # popisky jen na mírném svahu: ve strmém by text překryl sousední vrstevnice
    ok &= (slope > 0.06) & (slope < 0.55)

    def cell(x, y):
        return min(gh - 1, max(0, int(y / ch_m))), min(gw - 1, max(0, int(x / cw_m)))

    def angle(iy, ix):
        a = math.atan2(-gx[iy, ix], gy[iy, ix])  # tečna = kolmice ke spádu
        if a > math.pi / 2:
            a -= math.pi
        elif a < -math.pi / 2:
            a += math.pi
        return a

    out: list[tuple] = []
    for x, y, a, lv in old or []:
        iy, ix = cell(x, y)
        if lv in levels and ok[iy, ix] and abs(sm[iy, ix] - lv) < 0.0025:
            out.append((x, y, angle(iy, ix), lv))
    for lv in levels:
        if len(out) >= limit:
            break
        s = sm - lv
        edge = np.zeros_like(ok)
        edge[:, :-1] |= (s[:, :-1] * s[:, 1:]) <= 0
        edge[:-1, :] |= (s[:-1, :] * s[1:, :]) <= 0
        cand = edge & ok
        ys, xs = np.nonzero(cand[::3, ::3])
        if len(ys) == 0:
            continue
        ys, xs = ys * 3, xs * 3
        # od nejmírnějšího svahu (popisek se tam nejlépe vejde)
        order = np.argsort(slope[ys, xs])
        for i in order[:400]:
            iy, ix = int(ys[i]), int(xs[i])
            x, y = (ix + 0.5) * cw_m, (iy + 0.5) * ch_m
            if all((x - px) ** 2 + (y - py) ** 2 > min_dist**2 for px, py, _, _ in out):
                out.append((x, y, angle(iy, ix), lv))
                if len(out) >= limit or sum(1 for o in out if o[3] == lv) >= 3:
                    break
    return out[:limit]


def major_levels(m) -> list[float]:
    """Výšky hlavních vrstevnic (stejně jako GLSL ``contours``: (5k + 0,5) × interval)."""
    ci, me = m.contour_interval, max(1, m.major_every)
    k0 = math.floor((m.elevation_min / ci - 0.5) / me) - 1
    k1 = math.ceil((m.elevation_max / ci - 0.5) / me) + 1
    return [round(ci * (me * k + 0.5), 6) for k in range(k0, k1 + 1)]


class TopoGame(Game):
    id = "topo"
    title = "Topografická mapa"
    description = "Barvy podle nadmořské výšky, vrstevnice a stínování reliéfu."
    order = 10
    category = "vizualizace"
    intro = "Tvaruj krajinu rukama – barvy a vrstevnice ukazují nadmořskou výšku."
    keys = [
        ("K", "vrstevnice zap/vyp"),
        ("S", "stínování zap/vyp"),
        ("L", "legenda výšek"),
        ("↑ / ↓", "hladina vody"),
        ("← / →", "hustota vrstevnic"),
    ]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.show_contours = True
        self.shade = True
        self.legend = False
        self.labels: list[tuple] = []
        self._next_labels = 0.0
        self._lo, self._hi = -0.02, 0.08

    def stop(self) -> None:
        self.prog.release()

    def update(self, hm, dt: float) -> None:
        now = self.host.time
        if now < self._next_labels:
            return
        self._next_labels = now + 0.7
        m = self.host.settings.map
        lo, hi = np.percentile(hm.heights, [2, 99.5])
        self._lo, self._hi = float(lo), float(hi)
        levels = [lv for lv in major_levels(m) if m.water_level < lv]
        self.labels = contour_labels(hm, levels, self.labels) if self.show_contours else []

    def _scan(self) -> tuple[float, float]:
        """Výška a síla zářící vlny (stoupá od hladiny k nejvyššímu místu)."""
        ph = self.host.time % SCAN_PERIOD
        if ph > SCAN_RISE:
            return -10.0, 0.0
        k = ph / SCAN_RISE
        lo = max(self._lo, self.host.settings.map.water_level)
        hi = max(self._hi, lo + 0.01)
        k = k * k * (3 - 2 * k)
        a = min(1.0, ph / 0.6) * min(1.0, (SCAN_RISE - ph) / 0.8)
        return lo + (hi + 0.004 - lo) * k, a

    def render(self) -> None:
        host = self.host
        legend = (0.955, 0.08, 0.985, 0.92) if self.legend else (-1.0, 0, 0, 0)
        scan, scan_a = self._scan()
        bw, bh = host.box_size
        labs = []
        size = hud.size(host.canvas_size, 0.62)
        cw, ch = host.canvas_size
        for x, y, a, lv in self.labels:
            text = f"{host.altitude(lv):.0f}"
            half = (len(text) * 0.62 * size * 0.5 + size * 0.35) / cw * bw
            labs.append((x / bw, y / bh, a, half))
        flat = [v for L in labs for v in L] + [0.0] * (4 * (MAX_LABELS - len(labs)))
        prog = self.prog
        try:
            prog["u_labels"].write(np.array(flat, np.float32).tobytes())
        except KeyError:
            pass
        host.draw_fullscreen(prog, u_show_contours=int(self.show_contours), u_shade=int(self.shade),
                             u_legend=legend, u_scan=float(scan), u_scan_a=float(scan_a), u_nlabels=len(labs))
        host.glow(0.45 * max(scan_a, 0.25), 0.85)
        p = host.painter
        if self.show_contours:
            for x, y, a, lv in self.labels:
                px, py = host.box_to_canvas(x, y)
                p.text(f"{host.altitude(lv):.0f}", px, py, size=size, anchor="mc", bold=True,
                       color=(0.12, 0.06, 0.02, 1), outline=max(2, size // 9), outline_color=(1, 0.97, 0.86, 1),
                       rotation=math.degrees(a))
        if self.legend:
            self._legend_labels(legend)

    def _legend_labels(self, rect) -> None:
        host, m = self.host, self.host.settings.map
        cw, ch = host.canvas_size
        step = m.contour_interval * m.major_every
        h = (int(m.elevation_min / step)) * step
        while h <= m.elevation_max + 1e-9:
            k = (m.elevation_max - h) / (m.elevation_max - m.elevation_min)
            y = (rect[1] + k * (rect[3] - rect[1])) * ch
            hud.label(host.painter, f"{host.altitude(h):.0f} m", rect[0] * cw - 8, y, hud.size((cw, ch), 0.7),
                      anchor="mr", bg=(0, 0, 0, 0.4))
            h += step

    def key(self, name: str, shift: bool = False) -> bool:
        m = self.host.settings.map
        if name == "k":
            self.show_contours = not self.show_contours
            self._next_labels = 0.0
        elif name == "s":
            self.shade = not self.shade
        elif name == "l":
            self.legend = not self.legend
        elif name in ("up", "down"):
            m.water_level += 0.005 if name == "up" else -0.005
            m.water_level = min(max(m.water_level, m.elevation_min), m.elevation_max)
            self._next_labels = 0.0
            self.host.notify(f"Hladina vody: {self.host.altitude(m.water_level):.0f} m n. m.")
        elif name in ("left", "right"):
            i = min(range(len(INTERVALS)), key=lambda j: abs(INTERVALS[j] - m.contour_interval))
            i = max(0, min(len(INTERVALS) - 1, i + (1 if name == "right" else -1)))
            m.contour_interval = INTERVALS[i]
            self.labels = []
            self._next_labels = 0.0
            meters = m.contour_interval * m.vertical_scale
            self.host.notify(f"Vrstevnice po {m.contour_interval * 100:.1f} cm (= {meters:.0f} m)")
        else:
            return False
        return True

    def status(self) -> list[str]:
        m = self.host.settings.map
        return [f"Vrstevnice po {m.contour_interval * 100:.1f} cm = {m.contour_interval * m.vertical_scale:.0f} m",
                f"Hladina vody {self.host.altitude(m.water_level):.0f} m n. m."]
