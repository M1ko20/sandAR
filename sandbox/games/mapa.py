"""Turistická mapa: papír, hnědé vrstevnice, vodní plochy a kóty vrcholů.

Učí číst mapu – žáci vidí, jak vypadá kopec, sedlo nebo údolí ve vrstevnicích.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Game

FRAG = """
uniform int u_shade;
void main() {
    float h = height_at(v_uv);
    vec3 paper = vec3(0.97, 0.95, 0.87);
    vec3 c = paper;
    if (u_shade == 1) c *= mix(0.80, 1.05, hillshade(v_uv, 2.0));
    bool water = h < u_water_level;
    if (water) c = mix(vec3(0.62, 0.82, 0.95), vec3(0.45, 0.70, 0.92),
                       clamp((u_water_level - h) / 0.05, 0.0, 1.0));
    vec2 cl = contours(h);
    vec3 brown = vec3(0.62, 0.36, 0.14);
    if (!water) {
        c = mix(c, brown, cl.x * 0.9);
        c = mix(c, brown * 0.75, cl.y);
    } else {
        c = mix(c, vec3(0.25, 0.50, 0.80), cl.x * 0.35);
    }
    float shore = 1.0 - smoothstep(0.0, 1.8 * fwidth(h), abs(h - u_water_level));
    c = mix(c, vec3(0.10, 0.35, 0.75), shore);
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
        self.shade = False
        self.peaks: list[tuple[int, int, float]] = []
        self._next_peak_update = 0.0

    def stop(self) -> None:
        self.prog.release()

    def update(self, hm, dt: float) -> None:
        if self.host.time < self._next_peak_update:
            return
        self._next_peak_update = self.host.time + 0.3
        m = self.host.settings.map
        gw = hm.shape[1]
        self.peaks = find_peaks(hm.heights, m.water_level + m.contour_interval, 1.5 * m.contour_interval,
                                radius=max(4, gw // 16))

    def render(self) -> None:
        self.host.draw_fullscreen(self.prog, u_shade=int(self.shade))
        if not self.show_peaks:
            return
        p = self.host.painter
        size = max(14, self.host.canvas_size[1] // 32)
        for iy, ix, h in self.peaks:
            x, y = self.host.grid_to_canvas(ix, iy)
            s = size * 0.45
            p.triangles(np.array([[x, y - s], [x + s, y + s * 0.7], [x - s, y + s * 0.7]]), (0.15, 0.08, 0.02, 1))
            p.text(f"{self.host.altitude(h):.0f}", x + s + 3, y, size=size, anchor="ml",
                   color=(0.15, 0.08, 0.02, 1), outline=2, outline_color=(0.97, 0.95, 0.87, 1))

    def key(self, name: str, shift: bool = False) -> bool:
        m = self.host.settings.map
        if name == "v":
            self.show_peaks = not self.show_peaks
        elif name == "s":
            self.shade = not self.shade
        elif name in ("up", "down"):
            m.water_level += 0.005 if name == "up" else -0.005
            self.host.notify(f"Hladina vody: {self.host.altitude(m.water_level):.0f} m n. m.")
        else:
            return False
        return True

    def status(self) -> list[str]:
        if not self.peaks:
            return ["Žádný výrazný vrchol"]
        return [f"Nejvyšší vrchol: {self.host.altitude(self.peaks[0][2]):.0f} m n. m.",
                f"Počet vrcholů: {len(self.peaks)}"]
