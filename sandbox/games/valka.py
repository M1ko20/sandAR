"""Vodní válka (přehrady) – hra pro dva hráče.

Uprostřed bedny vyvěrá pramen, každý hráč má u své kratší strany vesnici (5 domků).
Hráči stavějí hráze, aby voda netekla k nim, a kopou koryta, aby tekla k soupeři.
Prohrává ten, komu se zatopí všechny domky; vyhrává ten, komu se vesnice zatopí
jako poslední. Pramen postupně sílí, takže kolo nemůže trvat věčně; kdyby do
``FLOOD_MAX`` nikdo nepřišel o všechno, vyhrává ten, kdo má víc suchých domků.

Vesnice jsou stejné, jen otočené o 180° (každý hráč má stejné podmínky ze svého pohledu).
Proud vody počítá ``WaterSim`` po skutečném terénu.
"""

from __future__ import annotations

import math

import numpy as np

from ..sim import ecosystem as eco
from ..sim.water import WaterSim
from . import versus as vs
from .base import Game
from .mesto import draw_house

FRAG = """
uniform sampler2D u_water;
uniform float u_clock;
void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    vec3 c = land_color((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4));
    c = mix(c, vec3(dot(c, vec3(0.3, 0.55, 0.15))), 0.25);
    c *= mix(0.55, 1.12, hillshade(uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.5, cl.x * 0.5);
    c = mix(c, c * 0.3, cl.y * 0.7);
    float w = texture(u_water, uv).r;
    if (w > 0.0003) {
        float a = smoothstep(0.0003, 0.003, w);
        float d01 = clamp(w / 0.025, 0.0, 1.0);
        vec3 wc = mix(vec3(0.30, 0.78, 1.0), vec3(0.02, 0.18, 0.75), d01);
        float s = sin(uv.x * u_box.x * 80.0 + u_clock * 3.1) * sin(uv.y * u_box.y * 70.0 - u_clock * 2.6);
        wc += 0.10 * s * (1.0 - d01);
        wc = mix(wc, vec3(0.92, 0.98, 1.0), (1.0 - smoothstep(0.0006, 0.0025, w)) * 0.6);
        c = mix(c, wc, a * 0.95);
    }
    f_color = vec4(c, 1.0);
}
"""

PREP, FLOOD, RESULT = "příprava", "povodeň", "výsledek"


def mirrored_villages(size: tuple[float, float], n: int, house: float, rng: np.random.Generator,
                      inset: float = 0.2) -> list[list[eco.House]]:
    """Dvě stejné vesnice u kratších stran, druhá otočená o 180° kolem středu bedny."""
    W, D = size
    left = eco.layout_houses((inset * W, D / 2), n, house, size, rng)
    right = [eco.House(W - h.x, D - h.y, h.angle + math.pi, h.length, h.width, church=h.church) for h in left]
    return [left, right]


class WaterWarGame(Game):
    id = "valka"
    title = "Vodní válka"
    description = "Dva hráči: uprostřed vyvěrá voda. Stav hráze k sobě, kopej koryta k soupeři. Kdo vydrží déle?"
    order = 94
    category = "soutěž"
    intro = "Uprostřed vyvěrá voda! Stav hráze ke své vesnici a kopej koryta k soupeři."
    keys = [("Enter", "pustit vodu hned / další kolo"), ("↑ / ↓", "síla pramene"), ("Backspace", "nové kolo")]

    PREP_TIME = 20.0  # [s] na stavbu hrází
    FLOOD_MAX = 120.0  # [s] nejdelší povodeň
    RESULT_TIME = 10.0
    N_HOUSES = 5
    FLOOD_DEPTH = 0.003  # [m] voda, která zatopí dům
    INFILTRATION = 0.00003  # [m/s] vsakování během povodně
    BASE_FLOW = 0.00006  # [m³/s na m² bedny] průtok pramene na začátku
    RAMP = 20.0  # [s] za jak dlouho průtok vzroste o původní hodnotu
    SPRING_R = 0.045  # [m]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.sim: WaterSim | None = None
        self.tex = None
        self.rng = np.random.default_rng()
        self.strength = getattr(self, "strength", 1.0)
        self.clock = 0.0
        self.villages: list[list[eco.House]] = [[], []]
        self.winner: int | None = None  # None + RESULT = remíza
        self.score = [0, 0]
        self._new_round_pending = True
        self.phase, self.phase_t = PREP, 0.0

    def stop(self) -> None:
        self.prog.release()
        if self.tex is not None:
            self.tex.release()
            self.tex = None

    def reset(self) -> None:
        self._new_round_pending = True

    # --- logika ------------------------------------------------------------------------------
    def spring(self, size: tuple[float, float]) -> tuple[float, float]:
        return size[0] / 2, size[1] / 2

    def flow(self, size: tuple[float, float], t: float) -> float:
        """Průtok pramene [m³/s] v čase t od začátku povodně – postupně sílí."""
        return self.BASE_FLOW * size[0] * size[1] * (1 + t / self.RAMP) * self.strength

    def dry(self, player: int) -> int:
        return sum(not h.flooded for h in self.villages[player])

    def _new_round(self, hm) -> None:
        if self.sim is not None:
            self.sim.clear()
        self.villages = mirrored_villages(hm.size, self.N_HOUSES, 0.022 * hm.size[0], self.rng)
        self.winner = None
        self.phase, self.phase_t = PREP, 0.0
        self._new_round_pending = False

    def update(self, hm, dt: float) -> None:
        if self.sim is None or self.sim.depth.shape != hm.shape:
            self.sim = WaterSim(hm.shape, hm.cell_size[0])
            if self.tex is not None:
                self.tex.release()
            self.tex = self.host.float_texture((hm.shape[1], hm.shape[0]))
            self._new_round_pending = True
        if self._new_round_pending:
            self._new_round(hm)
        dt = min(dt, 1 / 20)
        self.clock += dt
        self.phase_t += dt
        sim = self.sim
        if self.phase == PREP and self.phase_t >= self.PREP_TIME:
            self._start_flood()
        elif self.phase == FLOOD:
            eco.add_source(sim.depth, hm.size, self.spring(hm.size), self.SPRING_R,
                           self.flow(hm.size, self.phase_t) * dt)
            for village in self.villages:
                wet = eco.flooded_houses(sim.depth, village, hm.size, self.FLOOD_DEPTH)
                for h, w in zip(village, wet):
                    if w and not h.flooded:
                        h.flooded, h.flooded_at = True, self.clock
            self._check_end()
        elif self.phase == RESULT and self.phase_t >= self.RESULT_TIME:
            self._new_round(hm)
        sim.advance(hm.heights, dt, 9.81)
        eco.soak(sim.depth, (self.INFILTRATION if self.phase == FLOOD else 0.0006) * dt)
        assert self.tex is not None
        self.tex.write(sim.depth.tobytes())

    def _start_flood(self) -> None:
        self.phase, self.phase_t = FLOOD, 0.0
        self.host.notify("Pramen se otevřel!")

    def _check_end(self) -> None:
        dry = [self.dry(0), self.dry(1)]
        if min(dry) > 0 and self.phase_t < self.FLOOD_MAX:
            return
        if dry[0] == dry[1]:
            self.winner = None
        else:
            self.winner = 0 if dry[0] > dry[1] else 1
        if self.winner is not None:
            self.score[self.winner] += 1
        self.phase, self.phase_t = RESULT, 0.0

    # --- kreslení ------------------------------------------------------------------------------
    def render(self) -> None:
        if self.tex is None:
            return
        self.host.draw_fullscreen(self.prog, u_water=(self.tex, 1), u_clock=float(self.clock))
        p = self.host.painter
        canvas = self.host.canvas_size
        vs.draw_divider(p, canvas)
        cw, ch = canvas
        bw, bd = self.host.box_size
        sx, sy = cw / bw, ch / bd
        self._draw_spring(p, sx, sy)
        for player, village in enumerate(self.villages):
            if village:
                vx = sum(h.x for h in village) / len(village) * sx
                vy = sum(h.y for h in village) / len(village) * sy
                p.circle(vx, vy, 0.11 * sx, vs.COLORS[player][:3] + (0.9,), width=max(4, ch / 150))
            for h in village:
                draw_house(p, h, sx, sy, self.clock, saved_ring=self.phase == RESULT,
                           ring_color=vs.COLORS[player])
        for player in (0, 1):
            self._hud(p, canvas, player)

    def _draw_spring(self, p, sx: float, sy: float) -> None:
        bw, bd = self.host.box_size
        x, y = bw / 2 * sx, bd / 2 * sy
        r = self.SPRING_R * sx
        if self.phase == PREP:
            for k in range(3):
                f = (self.clock * 0.6 + k / 3) % 1.0
                p.circle(x, y, r * (0.6 + 1.4 * f), (0.3, 0.8, 1.0, 1.0 - f), width=max(3, 0.006 * sx))
        pulse = 0.5 + 0.5 * math.sin(self.clock * 5)
        p.circle(x, y, r * 0.6, (0.05, 0.35, 0.95, 0.95))
        p.circle(x, y, r * (0.6 + 0.1 * pulse), (1, 1, 1, 0.9), width=max(2, 0.004 * sx))
        d = r * 0.3
        p.triangles(np.array([[x, y - d * 1.5], [x + d * 0.75, y], [x - d * 0.75, y]]), (1, 1, 1, 1))
        p.circle(x, y + d * 0.15, d * 0.78, (1, 1, 1, 1), segments=16)

    def _hud(self, p, canvas, player: int) -> None:
        size = vs.text_size(canvas)
        small = vs.text_size(canvas, 0.75)
        n = len(self.villages[player])
        dry = self.dry(player)
        if self.phase == PREP:
            left = max(0, math.ceil(self.PREP_TIME - self.phase_t))
            vs.player_text(p, canvas, player, f"Voda poteče za {left} s!", 0.035, size, color=(1, 0.95, 0.3, 1))
            vs.player_text(p, canvas, player, "Stav hráz ke své vesnici", 0.085, small, color=(1, 1, 1, 1))
        elif self.phase == FLOOD:
            vs.player_text(p, canvas, player, f"Suché domky: {dry}/{n}", 0.035, size)
            vs.player_text(p, canvas, player, "Pošli vodu k soupeři!", 0.085, small, color=(0.6, 0.9, 1, 1))
        elif self.phase == RESULT:
            if self.winner is None:
                text, col = "Remíza!", (1, 0.95, 0.3, 1)
            elif self.winner == player:
                text, col = "Vyhráli jste!", (0.35, 1.0, 0.45, 1)
            else:
                text, col = "Vesnice je pod vodou!", (1.0, 0.45, 0.35, 1)
            vs.player_text(p, canvas, player, text, 0.04, int(size * 1.3), color=col,
                           background=(0, 0, 0, 0.6), padding=size * 0.25)
            vs.player_text(p, canvas, player, f"Suché domky: {dry}/{n}", 0.10, small, color=(1, 1, 1, 1))
        if self.phase != RESULT:
            vs.player_text(p, canvas, player, f"Výhry {self.score[player]}", 0.035, small, along=0.9)

    # --- ovládání ---------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name in ("enter", "return", "space"):
            if self.phase == PREP:
                self._start_flood()
            else:
                self._new_round_pending = True
        elif name == "backspace":
            self._new_round_pending = True
        elif name in ("up", "down"):
            self.strength = float(np.clip(self.strength * (1.25 if name == "up" else 0.8), 0.3, 4.0))
            self.host.notify(f"Síla pramene ×{self.strength:.2f}")
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Fáze: {self.phase} ({self.phase_t:.0f} s)",
                 f"Suché domky: červení {self.dry(0)}, modří {self.dry(1)}",
                 f"Výhry: červení {self.score[0]}, modří {self.score[1]}",
                 f"Síla pramene ×{self.strength:.2f}"]
        if self.sim is not None:
            lines.append(f"Voda: {self.sim.volume * 1000:.1f} l")
        return lines
