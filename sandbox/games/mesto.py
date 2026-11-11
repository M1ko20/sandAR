"""Záchrana města: v nížině stojí městečko, z vyvýšeného okraje přijde povodeň.

Návštěvníci mají chvilku na stavbu hrází a kanálů, které vodu odvedou jinam.
Pramen povodně je na nejvyšším okraji bedny, město v rovinaté nížině po proudu,
kam by voda opravdu stekla (odtok D8 po skutečném terénu). Proud vody počítá
WaterSim. Dům, ke kterému se dostane voda, je zatopený. Na konci se ukáže,
kolik domů se podařilo zachránit, a hra začne znovu s městem jinde.
"""

from __future__ import annotations

import math

import numpy as np

from ..sim import ecosystem as eco
from ..sim.water import WaterSim
from .base import Game

FRAG = """
uniform sampler2D u_water;
uniform sampler2D u_danger;   // předpověď: kam doteče voda (1 = zaplaveno)
uniform int u_show_danger;
uniform float u_clock;
void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    vec3 c = land_color((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4));
    c = mix(c, vec3(dot(c, vec3(0.3, 0.55, 0.15))), 0.25);   // trochu tlumené, ať vynikne voda a domy
    c *= mix(0.55, 1.12, hillshade(uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.5, cl.x * 0.5);
    c = mix(c, c * 0.3, cl.y * 0.7);

    if (u_show_danger == 1) {
        // šrafování „sem doteče voda“ – pohyblivé modré pruhy
        float dz = smoothstep(0.3, 0.7, texture(u_danger, uv).r);
        vec2 p = uv * u_box;
        float stripe = step(0.5, fract((p.x + p.y) * 22.0 - u_clock * 0.8));
        c = mix(c, mix(vec3(0.20, 0.55, 1.0), vec3(0.55, 0.85, 1.0), stripe), dz * 0.55);
    }
    float w = texture(u_water, uv).r;
    if (w > 0.0003) {
        float a = smoothstep(0.0003, 0.003, w);
        float d01 = clamp(w / 0.025, 0.0, 1.0);
        vec3 wc = mix(vec3(0.30, 0.78, 1.0), vec3(0.02, 0.18, 0.75), d01);
        float s = sin(uv.x * u_box.x * 80.0 + u_clock * 3.1) * sin(uv.y * u_box.y * 70.0 - u_clock * 2.6);
        wc += 0.10 * s * (1.0 - d01);
        // pěna na čele vlny (mělká voda)
        wc = mix(wc, vec3(0.92, 0.98, 1.0), (1.0 - smoothstep(0.0006, 0.0025, w)) * 0.6);
        c = mix(c, wc, a * 0.95);
    }
    f_color = vec4(c, 1.0);
}
"""

PLAN, PREP, FLOOD, RESULT = "plánování", "příprava", "povodeň", "výsledek"


class TownGame(Game):
    id = "mesto"
    title = "Záchrana města"
    description = "Z hor přijde povodeň. Postav hráze a kanály a zachraň co nejvíc domů."
    order = 82
    category = "soutěž"
    intro = "Za chvíli přijde povodeň – postav z písku hráze a kanály, ať voda obteče městečko!"
    keys = [
        ("Enter", "spustit povodeň hned / další kolo"),
        ("M", "nové město jinde"),
        ("↑ / ↓", "síla povodně"),
        ("Backspace", "začít znovu"),
    ]

    PREP_TIME = 25.0  # [s] na stavbu hrází
    FLOOD_TIME = 30.0  # [s] délka povodně
    RESULT_TIME = 10.0  # [s] zobrazení výsledku
    N_HOUSES = 10
    FLOOD_DEPTH = 0.003  # [m] voda, která zatopí dům
    INFILTRATION = 0.0003  # [m/s] vsakování do písku (bedna má stěny, voda neodtéká)

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.sim: WaterSim | None = None
        self.tex = None
        self.rng = np.random.default_rng()
        self.strength = 1.0
        self.clock = 0.0
        self.houses: list[eco.House] = []
        self.source: tuple[float, float] | None = None
        self.town: tuple[float, float] | None = None
        self.prev_town: tuple[float, float] | None = None
        self.forecast: eco.FloodPlanner | None = None
        self.danger_tex = None
        self.has_danger = False
        self.phase = PREP
        self.phase_t = 0.0
        self.rounds = 0
        self.last_score: tuple[int, int] | None = None
        self._need_town = True
        self.planner: eco.FloodPlanner | None = None

    def stop(self) -> None:
        self.prog.release()
        for tex in (self.tex, self.danger_tex):
            if tex is not None:
                tex.release()
        self.tex = self.danger_tex = None

    def reset(self) -> None:
        """Nové kolo: nové město (jinde než minule), suchá krajina, odpočet."""
        self._need_town = True

    # --- logika -----------------------------------------------------------------------------
    def _house_size(self) -> float:
        return 0.022 * self.host.box_size[0]

    def _flow(self, hm) -> float:
        """Průtok pramene [m³/s] – úměrný ploše bedny (za povodeň ~ 5 mm vody na celé pískoviště)."""
        return 0.005 * hm.size[0] * hm.size[1] / self.FLOOD_TIME * self.strength

    def _new_round(self, hm) -> None:
        """Začne plánovat nové kolo: zkušební povodně z několika pramenů (po kouscích, viz _plan)."""
        if self.sim is not None:
            self.sim.clear()
        factor = max(1, hm.shape[1] // 64)
        sources = eco.flood_source_candidates(hm.heights, hm.size, self.rng, factor=factor, n=3)
        self.planner = eco.FloodPlanner(hm.heights, hm.size, sources, self._flow(hm), self.FLOOD_TIME * 0.8,
                                        factor=factor, threshold=self.FLOOD_DEPTH, infiltration=self.INFILTRATION)
        self.houses = []
        self.source = self.town = None
        self.phase, self.phase_t = PLAN, 0.0
        self.forecast = None
        self.has_danger = False
        self._need_town = False

    def _plan(self, hm) -> None:
        assert self.planner is not None
        if not self.planner.step(40):
            return
        house = self._house_size()
        town_r = house * 4.2
        wl = self.host.settings.map.water_level
        res = self.planner.choose(self.rng, town_r, wl, avoid=self.prev_town)
        if res is None:  # voda nikam nedoteče (např. rovný písek) -> město po proudu podle D8
            source = self.planner.sources[0]
            town = eco.place_town(hm.heights, hm.size, wl, self.rng, source, town_r, avoid=self.prev_town,
                                  factor=max(1, hm.shape[1] // 64))
        else:
            source, town = res
        self.source, self.town = source, town
        self.prev_town = town
        self.houses = eco.layout_houses(town, self.N_HOUSES, house, hm.size, self.rng)
        self.planner = None
        self.phase, self.phase_t = PREP, 0.0

    def _update_danger(self, hm) -> None:
        """Předpověď zaplavené oblasti – zkušební povodeň po současném terénu, počítaná po kouscích.

        Když návštěvníci staví hráze, předpověď se každých pár sekund obnoví.
        """
        if self.forecast is None:
            assert self.source is not None
            factor = max(1, hm.shape[1] // 64)
            self.forecast = eco.FloodPlanner(hm.heights, hm.size, [self.source], self._flow(hm), self.FLOOD_TIME,
                                             factor=factor, threshold=self.FLOOD_DEPTH,
                                             infiltration=self.INFILTRATION)
        if self.forecast.step(12):
            danger = np.isfinite(self.forecast.arrival[0]).astype(np.float32)
            if self.danger_tex is None or self.danger_tex.size != (danger.shape[1], danger.shape[0]):
                if self.danger_tex is not None:
                    self.danger_tex.release()
                self.danger_tex = self.host.float_texture((danger.shape[1], danger.shape[0]))
            self.danger_tex.write(np.ascontiguousarray(danger).tobytes())
            self.has_danger = True
            self.forecast = None

    def update(self, hm, dt: float) -> None:
        if self.sim is None or self.sim.depth.shape != hm.shape:
            self.sim = WaterSim(hm.shape, hm.cell_size[0])
            if self.tex is not None:
                self.tex.release()
            self.tex = self.host.float_texture((hm.shape[1], hm.shape[0]))
            self._need_town = True
        if self._need_town:
            self._new_round(hm)
        dt = min(dt, 1 / 20)
        self.clock += dt
        self.phase_t += dt
        sim = self.sim
        if self.phase == PLAN:
            self._plan(hm)
        elif self.phase == PREP:
            self._update_danger(hm)
            if self.phase_t >= self.PREP_TIME:
                self.phase, self.phase_t = FLOOD, 0.0
                self.host.notify("Povodeň!")
        elif self.phase == FLOOD:
            assert self.source is not None
            eco.add_source(sim.depth, hm.size, self.source, 0.04, self._flow(hm) * dt)
            wet = eco.flooded_houses(sim.depth, self.houses, hm.size, self.FLOOD_DEPTH)
            for h, w in zip(self.houses, wet):
                if w and not h.flooded:
                    h.flooded, h.flooded_at = True, self.clock
            if self.phase_t >= self.FLOOD_TIME or all(h.flooded for h in self.houses):
                self._finish()
        elif self.phase == RESULT and self.phase_t >= self.RESULT_TIME:
            self._new_round(hm)
        sim.advance(hm.heights, dt, 9.81)
        # vsakování; po povodni voda opadá rychleji
        eco.soak(sim.depth, (self.INFILTRATION if self.phase == FLOOD else 0.0005) * dt)
        assert self.tex is not None
        self.tex.write(sim.depth.tobytes())

    def _finish(self) -> None:
        saved = sum(not h.flooded for h in self.houses)
        self.last_score = (saved, len(self.houses))
        self.rounds += 1
        self.phase, self.phase_t = RESULT, 0.0

    # --- kreslení ------------------------------------------------------------------------------
    def render(self) -> None:
        if self.tex is None:
            return
        show = self.phase == PREP and self.has_danger and self.danger_tex is not None
        self.host.draw_fullscreen(self.prog, u_water=(self.tex, 1), u_clock=float(self.clock),
                                  u_danger=(self.danger_tex if show else self.tex, 2), u_show_danger=int(show))
        sx, sy = self._px()
        self._draw_source(sx, sy)
        self._draw_houses(sx, sy)
        self._draw_hud()

    def _px(self) -> tuple[float, float]:
        cw, ch = self.host.canvas_size
        bw, bd = self.host.box_size
        return cw / bw, ch / bd

    def _draw_source(self, sx: float, sy: float) -> None:
        if self.source is None:
            return
        p = self.host.painter
        x, y = self.source[0] * sx, self.source[1] * sy
        r = 0.04 * sx
        pulse = 0.5 + 0.5 * math.sin(self.clock * 5)
        if self.phase == PREP:
            for k in range(3):  # rozbíhající se kruhy
                f = (self.clock * 0.6 + k / 3) % 1.0
                p.circle(x, y, r * (0.5 + 1.2 * f), (0.3, 0.8, 1.0, 1.0 - f), width=0.006 * sx)
        p.circle(x, y, r * 0.55, (0.05, 0.35, 0.95, 0.9))
        p.circle(x, y, r * (0.55 + 0.1 * pulse), (1, 1, 1, 0.9), width=0.004 * sx)
        # kapka
        d = r * 0.32
        p.triangles(np.array([[x, y - d * 1.5], [x + d * 0.75, y], [x - d * 0.75, y]]), (1, 1, 1, 1))
        p.circle(x, y + d * 0.15, d * 0.78, (1, 1, 1, 1), segments=16)
        if self.phase == PREP:
            size = self._text_size(0.9)
            # popisek ke středu bedny, ať nevyleze z plátna
            cw, ch = self.host.canvas_size
            ty = y + r * 1.6 if y < ch / 2 else y - r * 1.6
            anchor = ("t" if y < ch / 2 else "b") + ("l" if x < cw * 0.25 else "r" if x > cw * 0.75 else "c")
            p.text("Odtud přijde voda!", x, ty, size=size, anchor=anchor, bold=True, color=(0.6, 0.9, 1, 1),
                   outline=3, outline_color=(0, 0.05, 0.2, 1))

    def _draw_houses(self, sx: float, sy: float) -> None:
        p = self.host.painter
        for h in self.houses:
            self._draw_house(p, h, sx, sy)

    def _draw_house(self, p, h: eco.House, sx: float, sy: float) -> None:
        cx, cy = h.x * sx, h.y * sy
        c, s = math.cos(h.angle), math.sin(h.angle)
        L, W = h.length * sx / 2, h.width * sy / 2

        def tr(pts):
            pts = np.asarray(pts, float)
            return np.stack([cx + pts[:, 0] * c - pts[:, 1] * s, cy + pts[:, 0] * s + pts[:, 1] * c], 1)

        flooded = h.flooded
        since = self.clock - h.flooded_at
        if flooded:
            roof_a, roof_b = (0.35, 0.45, 0.75, 1), (0.22, 0.30, 0.60, 1)
        elif h.church:
            roof_a, roof_b = (0.95, 0.75, 0.25, 1), (0.75, 0.52, 0.12, 1)
        else:
            roof_a, roof_b = (1.0, 0.30, 0.12, 1), (0.72, 0.16, 0.06, 1)
        # stín a obrys
        p.triangles(tr([[-L, -W], [L, -W], [L, W], [-L, -W], [L, W], [-L, W]]) + [W * 0.35, W * 0.5], (0, 0, 0, 0.45))
        o = 1.12
        p.triangles(tr([[-L * o, -W * o], [L * o, -W * o], [L * o, W * o], [-L * o, -W * o], [L * o, W * o],
                        [-L * o, W * o]]), (0.1, 0.05, 0.02, 1))
        # dvě poloviny sedlové střechy (světlejší je ta ke světlu)
        p.triangles(tr([[-L, -W], [L, -W], [L, 0], [-L, -W], [L, 0], [-L, 0]]), roof_a)
        p.triangles(tr([[-L, 0], [L, 0], [L, W], [-L, 0], [L, W], [-L, W]]), roof_b)
        p.polyline(tr([[-L, 0], [L, 0]]), (0.25, 0.08, 0.02, 1), width=max(1.5, W * 0.15))
        if h.church:
            t = W * 0.95  # věž s křížem
            tx = L + t * 0.9
            p.triangles(tr([[tx - t, -t], [tx + t, -t], [tx + t, t], [tx - t, -t], [tx + t, t], [tx - t, t]]),
                        (0.1, 0.05, 0.02, 1))
            t2 = t * 0.82
            p.triangles(tr([[tx, 0], [tx - t2, -t2], [tx + t2, -t2]]), roof_a)
            p.triangles(tr([[tx, 0], [tx + t2, -t2], [tx + t2, t2]]), roof_a)
            p.triangles(tr([[tx, 0], [tx + t2, t2], [tx - t2, t2]]), roof_b)
            p.triangles(tr([[tx, 0], [tx - t2, t2], [tx - t2, -t2]]), roof_b)
            p.polyline(tr([[tx - t * 0.5, 0], [tx + t * 0.5, 0]]), (1, 1, 1, 1), width=max(2, t * 0.2))
            p.polyline(tr([[tx, -t * 0.5], [tx, t * 0.5]]), (1, 1, 1, 1), width=max(2, t * 0.2))
        else:
            p.triangles(tr([[L * 0.35, -W * 0.75], [L * 0.6, -W * 0.75], [L * 0.6, -W * 0.4], [L * 0.35, -W * 0.75],
                            [L * 0.6, -W * 0.4], [L * 0.35, -W * 0.4]]), (0.3, 0.12, 0.08, 1))  # komín
        R = max(L, W) * 1.5
        if flooded:
            # červený blikající kruh + šplouchnutí
            blink = 0.55 + 0.45 * math.sin(self.clock * 7)
            p.circle(cx, cy, R, (1.0, 0.1, 0.1, blink), width=max(3, R * 0.14))
            if since < 1.5:
                f = since / 1.5
                p.circle(cx, cy, R * (1 + 1.5 * f), (0.8, 0.95, 1, 1 - f), width=max(2, R * 0.12))
        elif self.phase == RESULT:
            p.circle(cx, cy, R, (0.2, 1.0, 0.35, 0.95), width=max(3, R * 0.14))

    def _text_size(self, scale: float = 1.0) -> int:
        ch = self.host.canvas_size[1]
        return max(16, int(ch / 24 * scale))

    def _draw_hud(self) -> None:
        p = self.host.painter
        cw, ch = self.host.canvas_size
        size = self._text_size()
        n = len(self.houses)
        saved = sum(not h.flooded for h in self.houses)
        if self.phase == PLAN:
            p.text("Hledám místo pro nové městečko…", cw / 2, size * 0.5, size=size, anchor="tc", bold=True,
                   color=(1, 1, 1, 1), outline=3)
        elif self.phase == PREP:
            left = max(0, math.ceil(self.PREP_TIME - self.phase_t))
            p.text(f"Povodeň za {left} s! Postav hráze a ochraň město.", cw / 2, size * 0.5, size=size, anchor="tc",
                   bold=True, color=(1, 0.95, 0.3, 1), outline=3)
        elif self.phase == FLOOD:
            left = max(0, math.ceil(self.FLOOD_TIME - self.phase_t))
            p.text(f"Povodeň! Ještě {left} s · domy v suchu: {saved}/{n}", cw / 2, size * 0.5, size=size,
                   anchor="tc", bold=True, color=(0.55, 0.9, 1, 1), outline=3)
        elif self.phase == RESULT and self.last_score is not None:
            s, total = self.last_score
            big = int(size * 1.7)
            pct = 100 * s / max(total, 1)
            if pct >= 80:
                verdict, col = "Skvělá práce, město je v bezpečí!", (0.35, 1.0, 0.45, 1)
            elif pct >= 40:
                verdict, col = "Dobrý pokus – zkuste hráz postavit dřív.", (1.0, 0.9, 0.3, 1)
            else:
                verdict, col = "Voda byla silnější. Zkuste to znovu!", (1.0, 0.45, 0.35, 1)
            p.rect(0, ch / 2 - big * 1.4, cw, big * 2.9, (0, 0, 0, 0.6))
            p.text(f"Zachránili jste {s}/{total} {self._domu(total)}!", cw / 2, ch / 2 - big * 0.2, size=big,
                   anchor="bc", bold=True, color=col, outline=4)
            p.text(verdict, cw / 2, ch / 2 + big * 0.2, size=size, anchor="tc", bold=True, color=(1, 1, 1, 1),
                   outline=3)

    @staticmethod
    def _domu(n: int) -> str:
        return "dům" if n == 1 else "domy" if 2 <= n <= 4 else "domů"

    # --- ovládání ---------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name in ("enter", "return", "space"):
            if self.phase == PLAN:
                return True
            if self.phase == PREP:
                self.phase, self.phase_t = FLOOD, 0.0
                self.host.notify("Povodeň!")
            elif self.phase == RESULT:
                self._need_town = True
            else:
                self._finish()
        elif name in ("m", "backspace"):
            self._need_town = True
            self.host.notify("Nové město")
        elif name in ("up", "down"):
            self.strength = float(np.clip(self.strength * (1.25 if name == "up" else 0.8), 0.3, 4.0))
            self.host.notify(f"Síla povodně ×{self.strength:.2f}")
        else:
            return False
        return True

    def status(self) -> list[str]:
        n = len(self.houses)
        flooded = sum(h.flooded for h in self.houses)
        lines = [f"Fáze: {self.phase} ({self.phase_t:.0f} s)", f"Domy: {n - flooded}/{n} v suchu",
                 f"Síla povodně ×{self.strength:.2f}"]
        if self.sim is not None:
            lines.append(f"Voda: {self.sim.volume * 1000:.2f} l")
        if self.last_score is not None:
            lines.append(f"Minulé kolo: {self.last_score[0]}/{self.last_score[1]} (kol: {self.rounds})")
        return lines
