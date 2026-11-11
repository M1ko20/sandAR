"""Barevná topografická mapa s vrstevnicemi – klasický režim AR pískoviště."""

from __future__ import annotations

from .base import Game

FRAG = """
uniform int u_show_contours;
uniform int u_shade;
uniform vec4 u_legend;   // obdélník legendy v uv (x0, y0, x1, y1), x0 < 0 = vypnuto

void main() {
    vec2 uv = v_uv;
    float h;
    bool legend = u_legend.x >= 0.0 && uv.x >= u_legend.x && uv.x <= u_legend.z
                  && uv.y >= u_legend.y && uv.y <= u_legend.w;
    if (legend) {
        float k = (uv.y - u_legend.y) / (u_legend.w - u_legend.y);
        h = mix(u_elev_max, u_elev_min, k);
    } else {
        h = height_at(uv);
    }
    vec3 c = elevation_color(h);
    if (u_shade == 1 && !legend) c *= mix(0.62, 1.12, hillshade(uv, 2.5));
    if (u_show_contours == 1) {
        vec2 cl = contours(h);
        float under = h < u_water_level ? 0.45 : 1.0;   // pod vodou slabší
        c = mix(c, c * 0.35, cl.x * 0.85 * under);
        c = mix(c, vec3(0.06), cl.y * under);
    }
    // pobřeží
    float shore = 1.0 - smoothstep(0.0, 1.5 * fwidth(h), abs(h - u_water_level));
    c = mix(c, vec3(0.05, 0.15, 0.45), shore * 0.8);
    f_color = vec4(c, 1.0);
}
"""

INTERVALS = [0.005, 0.01, 0.02]


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

    def stop(self) -> None:
        self.prog.release()

    def render(self) -> None:
        legend = (0.955, 0.08, 0.985, 0.92) if self.legend else (-1.0, 0, 0, 0)
        self.host.draw_fullscreen(self.prog, u_show_contours=int(self.show_contours), u_shade=int(self.shade),
                                  u_legend=legend)
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
            host.painter.text(f"{host.altitude(h):.0f} m", rect[0] * cw - 8, y, size=max(12, ch // 45),
                              anchor="mr", color=(1, 1, 1, 1), outline=2)
            h += step

    def key(self, name: str, shift: bool = False) -> bool:
        m = self.host.settings.map
        if name == "k":
            self.show_contours = not self.show_contours
        elif name == "s":
            self.shade = not self.shade
        elif name == "l":
            self.legend = not self.legend
        elif name in ("up", "down"):
            m.water_level += 0.005 if name == "up" else -0.005
            m.water_level = min(max(m.water_level, m.elevation_min), m.elevation_max)
            self.host.notify(f"Hladina vody: {self.host.altitude(m.water_level):.0f} m n. m.")
        elif name in ("left", "right"):
            i = min(range(len(INTERVALS)), key=lambda j: abs(INTERVALS[j] - m.contour_interval))
            i = max(0, min(len(INTERVALS) - 1, i + (1 if name == "right" else -1)))
            m.contour_interval = INTERVALS[i]
            meters = m.contour_interval * m.vertical_scale
            self.host.notify(f"Vrstevnice po {m.contour_interval * 100:.1f} cm (= {meters:.0f} m)")
        else:
            return False
        return True

    def status(self) -> list[str]:
        m = self.host.settings.map
        return [f"Vrstevnice po {m.contour_interval * 100:.1f} cm = {m.contour_interval * m.vertical_scale:.0f} m",
                f"Hladina vody {self.host.altitude(m.water_level):.0f} m n. m."]
