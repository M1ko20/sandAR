"""Turistická mapa: papír, hnědé vrstevnice, vodní plochy a kóty vrcholů.

Učí číst mapu – žáci vidí, jak vypadá kopec, sedlo nebo údolí ve vrstevnicích.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..render import hud
from .base import Game
from .topo import contour_labels, major_levels

FRAG = """
uniform int u_shade;
uniform int u_nlabels;
uniform vec4 u_labels[10];  // popisky vrstevnic: střed uv, úhel [rad], polovina délky [m]

float mapa_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float mapa_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mapa_hash(i), mapa_hash(i + vec2(1, 0)), u.x),
               mix(mapa_hash(i + vec2(0, 1)), mapa_hash(i + vec2(1, 1)), u.x), u.y);
}
float mapa_label_gap(vec2 uv) {
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
    float h = height_at(v_uv);
    bool water = h < u_water_level;
    float t = clamp((h - u_water_level) / max(u_elev_max - u_water_level, 1e-4), 0.0, 1.0);
    // papír s jemným výškovým tónem (nížiny zelenkavé, hory okrové) – jako tištěná turistická mapa
    vec3 c = mix(vec3(0.90, 0.97, 0.78), vec3(1.0, 0.96, 0.84), smoothstep(0.0, 0.25, t));
    c = mix(c, vec3(0.98, 0.84, 0.62), smoothstep(0.35, 0.8, t));
    c = mix(c, vec3(1.0, 0.98, 0.95), smoothstep(0.85, 1.0, t));
    // lesy: velké zelené plochy s tmavším lemem (pevné v krajině, na svazích a kopcích)
    float fn = 0.65 * mapa_noise(pm * 5.0 + 3.1) + 0.35 * mapa_noise(pm * 11.0 + 9.7);
    float fv = fn - 0.55 + 0.25 * (1.0 - smoothstep(0.05, 0.12, t)) + 0.3 * smoothstep(0.75, 0.95, t);
    float fw = max(fwidth(fv), 1e-5);
    float forest = 1.0 - smoothstep(-fw, fw, fv);
    float fedge = 1.0 - smoothstep(1.0 * fw, 2.5 * fw, abs(fv));
    if (!water) {
        c = mix(c, vec3(0.66, 0.88, 0.52), forest * 0.9);
        c = mix(c, vec3(0.30, 0.62, 0.28), fedge * 0.8);
    }
    // stínování reliéfu (švýcarský styl: světlo zleva shora, stíny do modrofialova)
    if (u_shade == 1) {
        float hs = hillshade(v_uv, 2.5);
        c *= mix(vec3(0.62, 0.64, 0.80), vec3(1.06, 1.04, 0.98), hs);
    }
    vec2 cl = vec2(contour_line(h, u_contour_interval, 2.4), contours(h).y) * (1.0 - mapa_label_gap(uv));
    vec3 brown = vec3(0.66, 0.32, 0.08);
    if (water) {
        float depth = u_water_level - h;
        c = mix(vec3(0.58, 0.84, 1.0), vec3(0.30, 0.62, 0.95), clamp(depth / 0.05, 0.0, 1.0));
        // modré linky souběžné se břehem (klasická kresba vody), pomalu se sunou ke břehu
        float g = depth / 0.005 + u_time * 0.12;
        float d = abs(fract(g) - 0.5);
        float w = max(fwidth(g), 1e-5);
        float rip = 1.0 - smoothstep(1.0 * w, 2.0 * w, 0.5 - d);
        c = mix(c, vec3(0.20, 0.50, 0.88), rip * 0.55 * (1.0 - smoothstep(0.004, 0.03, depth)));
    } else {
        c = mix(c, brown, cl.x * 0.85);
        c = mix(c, brown * 0.72, cl.y);
    }
    float shore = 1.0 - smoothstep(1.0 * fwidth(h), 2.6 * fwidth(h), abs(h - u_water_level));
    c = mix(c, vec3(0.05, 0.30, 0.75), shore);
    f_color = vec4(c, 1.0);
}
"""


def find_peaks(heights: np.ndarray, min_height: float, prominence: float, radius: int, limit: int = 8):
    """Lokální maxima výškové mapy: [(iy, ix, výška)] seřazené od nejvyššího."""
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    local_max = cv2.dilate(heights, kernel)
    # výraznost vrcholu měříme ve větším okolí (široký kopec klesá pozvolna)
    k3 = 6 * radius + 1
    local_min = cv2.erode(heights, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k3, k3)))
    mask = (heights >= local_max - 1e-6) & (heights > min_height) & (heights - local_min > prominence)
    border = radius // 2
    mask[:border, :] = mask[-border:, :] = False
    mask[:, :border] = mask[:, -border:] = False
    ys, xs = np.nonzero(mask)
    order = np.argsort(-heights[ys, xs])
    peaks: list[tuple[int, int, float]] = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all((y - py) ** 2 + (x - px) ** 2 > radius**2 for py, px, _ in peaks):
            peaks.append((y, x, float(heights[y, x])))
        if len(peaks) >= limit:
            break
    return peaks


PEAK_NAMES = ["Kozí vrch", "Liščí hora", "Javorník", "Strážná", "Vysoký kámen", "Medvědí skála",
              "Orlí vrch", "Sokolka", "Hradiště", "Žalý", "Kamenec", "Bukovec", "Klínovec", "Ještěd", "Radhošť",
              "Čertova skála"]


class MapGame(Game):
    id = "mapa"
    title = "Turistická mapa"
    description = "Vrstevnice jako na papírové mapě, kóty vrcholů a vodní plochy."
    order = 20
    category = "vizualizace"
    intro = "Postav hory a údolí a sleduj, jak z nich vzniká turistická mapa."
    keys = [("V", "kóty vrcholů zap/vyp"), ("S", "stínování zap/vyp"), ("↑ / ↓", "hladina vody")]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.show_peaks = True
        self.shade = True
        self.peaks: list[tuple[int, int, float]] = []
        self._born: list[tuple[float, float, float]] = []  # (ix, iy, čas objevení) – animace kóty
        self.labels: list[tuple] = []
        self._next_peak_update = 0.0
        self._next_labels = 0.0
        self._next_peak_sound = 0.0

    def stop(self) -> None:
        self.prog.release()

    def update(self, hm, dt: float) -> None:
        now = self.host.time
        m = self.host.settings.map
        if now >= self._next_labels:
            self._next_labels = now + 0.7
            levels = [lv for lv in major_levels(m) if m.water_level < lv]
            self.labels = contour_labels(hm, levels, self.labels, limit=8)
        if now < self._next_peak_update:
            return
        self._next_peak_update = now + 0.3
        gw = hm.shape[1]
        self.peaks = find_peaks(hm.heights, m.water_level + m.contour_interval, 1.5 * m.contour_interval,
                                radius=max(4, gw // 16))
        r2 = (max(4, gw // 16)) ** 2
        born = []
        new_x = None
        for iy, ix, _ in self.peaks:
            prev = [b for b in self._born if (b[0] - ix) ** 2 + (b[1] - iy) ** 2 < r2]
            born.append((ix, iy, prev[0][2] if prev else now))
            if not prev and self._born:
                new_x = (ix + 0.5) / gw
        # mapa je tichá: jen jemné ťuknutí, když se objeví nová kóta (nejvýš jednou za 2 s)
        if new_x is not None and now >= self._next_peak_sound:
            self._next_peak_sound = now + 2.0
            self.sfx.play("tick", 0.35, x=new_x)
        self._born = born

    # --- kreslení -----------------------------------------------------------------
    def render(self) -> None:
        host = self.host
        bw, bh = host.box_size
        cw, ch = host.canvas_size
        lsize = hud.size(host.canvas_size, 0.62)
        labs = []
        for x, y, a, lv in self.labels:
            text = f"{host.altitude(lv):.0f}"
            half = (len(text) * 0.62 * lsize * 0.5 + lsize * 0.35) / cw * bw
            labs.append((x / bw, y / bh, a, half))
        flat = [v for L in labs for v in L] + [0.0] * (4 * (10 - len(labs)))
        try:
            self.prog["u_labels"].write(np.array(flat, np.float32).tobytes())
        except KeyError:
            pass
        host.draw_fullscreen(self.prog, u_shade=int(self.shade), u_nlabels=len(labs))
        p = host.painter
        brown = (0.50, 0.22, 0.04, 1)
        paper = (1.0, 0.97, 0.86, 1)
        for x, y, a, lv in self.labels:
            px, py = host.box_to_canvas(x, y)
            p.text(f"{host.altitude(lv):.0f}", px, py, size=lsize, anchor="mc", bold=True, color=brown,
                   outline=max(2, lsize // 9), outline_color=paper, rotation=math.degrees(a))
        self._frame(p, cw, ch)
        if self.show_peaks:
            self._peaks(p)

    def _peaks(self, p) -> None:
        host = self.host
        size = hud.size(host.canvas_size, 0.72)
        now = host.time
        ink = (0.12, 0.06, 0.02, 1)
        paper = (1.0, 0.97, 0.86, 1)
        tris, cols = [], []
        texts = []
        for i, (iy, ix, h) in enumerate(self.peaks):
            born = next((b[2] for b in self._born if b[0] == ix and b[1] == iy), now - 1)
            k = min(1.0, (now - born) / 0.45)
            pop = 1 + 0.6 * math.sin(k * math.pi) * (1 - k) + 0.0  # kóta „vyskočí“
            x, y = host.grid_to_canvas(ix, iy)
            s = size * 0.5 * pop * (0.3 + 0.7 * min(1.0, k * 2))
            # černý trojúhelník s papírovým lemem
            o = s + 4
            tris += [[x, y - o], [x + o * 1.05, y + o * 0.75], [x - o * 1.05, y + o * 0.75]]
            cols += [paper] * 3
            tris += [[x, y - s], [x + s, y + s * 0.7], [x - s, y + s * 0.7]]
            cols += [ink] * 3
            if k < 1:  # rozbíhající se kroužek při objevení nového vrcholu
                r = size * (0.6 + 2.2 * k)
                p.circle(x, y, r, (0.80, 0.10, 0.05, 1 - k), width=4)
            name = PEAK_NAMES[(ix // 24 * 7 + iy // 24 * 13) % len(PEAK_NAMES)] if i == 0 else None
            texts.append((x + s + 6, y, f"{host.altitude(h):.0f}", name))
        if tris:
            p.triangles(np.array(tris, np.float32), np.array(cols, np.float32))
        for x, y, alt, name in texts:
            if name:
                p.text(name, x, y - size * 0.55, size=size, anchor="ml", bold=True, color=ink, outline=3,
                       outline_color=paper)
                p.text(alt, x, y + size * 0.55, size=size, anchor="ml", color=(0.55, 0.08, 0.02, 1), outline=3,
                       outline_color=paper, bold=True)
            else:
                p.text(alt, x, y, size=size, anchor="ml", bold=True, color=ink, outline=3, outline_color=paper)

    def _frame(self, p, cw: int, ch: int) -> None:
        """Rám tištěné mapy: černobílá kilometrová lišta po okraji, měřítko a sever."""
        m = self.host.settings.map
        bw = self.host.box_size[0]
        t = max(10.0, ch / 70)  # šířka lišty
        p.rect(0, 0, cw, t, (0.08, 0.06, 0.04, 1))
        p.rect(0, ch - t, cw, t, (0.08, 0.06, 0.04, 1))
        p.rect(0, 0, t, ch, (0.08, 0.06, 0.04, 1))
        p.rect(cw - t, 0, t, ch, (0.08, 0.06, 0.04, 1))
        # bílé úseky každý „kilometr“ (střídavě), jako na okraji topografické mapy
        km_px = 1000.0 / m.vertical_scale / bw * cw
        seg = km_px / 2 if km_px > 120 else km_px
        verts = []
        i = 0.0
        k = 0
        while i < max(cw, ch):
            if k % 2 == 0:
                a, b = i, min(i + seg, max(cw, ch))
                inset = t * 0.3
                for (x0, y0, x1, y1) in ((a, inset, b, t - inset), (a, ch - t + inset, b, ch - inset)):
                    if x0 < cw:
                        x1 = min(x1, cw - t)
                        x0 = max(x0, t)
                        if x1 > x0:
                            verts += [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]
                for (x0, y0, x1, y1) in ((inset, a, t - inset, b), (cw - t + inset, a, cw - inset, b)):
                    if y0 < ch:
                        y1 = min(y1, ch - t)
                        y0 = max(y0, t)
                        if y1 > y0:
                            verts += [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]
            i += seg
            k += 1
        if verts:
            p.triangles(np.array(verts, np.float32), (1.0, 0.97, 0.86, 1))
        # měřítko vlevo dole: 0 – 500 – 1000 m
        size = hud.size((cw, ch), 0.62)
        x0, y0 = t + size * 0.6, ch - t - size * 0.9
        L = km_px
        p.rect(x0 - size * 0.4, y0 - size * 1.55, L + size * 2.4, size * 2.2, (1.0, 0.97, 0.86, 0.92))
        p.rect(x0, y0 - 6, L, 12, (0.1, 0.06, 0.03, 1))
        p.rect(x0 + L / 4 + 2, y0 - 4, L / 4 - 2, 8, (1.0, 0.97, 0.86, 1))
        p.rect(x0 + 3 * L / 4 + 2, y0 - 4, L / 4 - 4, 8, (1.0, 0.97, 0.86, 1))
        ink = (0.12, 0.06, 0.02, 1)
        p.text("0", x0, y0 - 10, size=size, anchor="bc", color=ink, bold=True)
        p.text("1 km", x0 + L, y0 - 10, size=size, anchor="bc", color=ink, bold=True)
        # severka vpravo nahoře
        x, y, r = cw - t - size * 1.1, t + size * 1.3, size * 0.8
        pts = np.array([[0, -1.0], [0.5, 0.65], [0, 0.3], [-0.5, 0.65]]) * r + [x, y]
        p.circle(x, y + r * 0.05, r * 1.25, (1.0, 0.97, 0.86, 0.92))
        p.triangles(np.array([pts[0], pts[1], pts[2]], np.float32), (0.75, 0.08, 0.03, 1))
        p.triangles(np.array([pts[0], pts[2], pts[3]], np.float32), (0.12, 0.06, 0.02, 1))

    def key(self, name: str, shift: bool = False) -> bool:
        m = self.host.settings.map
        if name == "v":
            self.show_peaks = not self.show_peaks
        elif name == "s":
            self.shade = not self.shade
        elif name in ("up", "down"):
            m.water_level += 0.005 if name == "up" else -0.005
            self._next_labels = 0.0
            self.sfx.play("bubble", 0.3, pitch=1.1 if name == "up" else 0.9)
            self.host.notify(f"Hladina vody: {self.host.altitude(m.water_level):.0f} m n. m.")
        else:
            return False
        return True

    def status(self) -> list[str]:
        if not self.peaks:
            return ["Žádný výrazný vrchol"]
        return [f"Nejvyšší vrchol: {self.host.altitude(self.peaks[0][2]):.0f} m n. m.",
                f"Počet vrcholů: {len(self.peaks)}"]
