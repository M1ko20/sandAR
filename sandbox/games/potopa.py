"""Potopa ostrovů – hra pro dva hráče.

Každý hráč si ve své půlce bedny postaví ostrov. Pak začne pomalu (a čím dál
rychleji) stoupat hladina moře. Písek se bere odkudkoli, klidně i od soupeře.
Ostrov je souš nad hladinou v hráčově půlce; když je menší než ``MIN_AREA``
(déle než ``SINK_HOLD``), hráč prohrál. Vyhrává poslední ostrov nad hladinou.
Kdyby moře došlo až k ``MAX_SEA`` (výš už Kinect písek nerozezná od ruky),
vyhrává vyšší ostrov.
"""

from __future__ import annotations

import math

import numpy as np

from . import versus as vs
from .base import Game

FRAG = """
uniform float u_sea;
uniform float u_clock;
uniform float u_active;   // 0 = moře se ještě nekreslí (stavba)

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x),
               mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;
    float h = height_at(uv);
    // pevnina: barvy od pláže nahoru, relativně k hladině (při stavbě podle běžné výškové škály)
    float sea = u_active > 0.5 ? u_sea : u_elev_min - 0.01;
    float e = (h - sea) / max(u_elev_max - sea, 0.02);
    vec3 c = land_color(clamp(e, 0.0, 1.0));
    c *= mix(0.55, 1.15, hillshade(uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.45, cl.y * 0.6 + cl.x * 0.25);
    float d = sea - h;   // hloubka moře
    if (d > 0.0 && u_active > 0.5) {
        float d01 = clamp(d / 0.05, 0.0, 1.0);
        vec3 wc = mix(vec3(0.15, 0.75, 1.0), vec3(0.02, 0.15, 0.55), d01);
        float rip = vnoise(pm * 40.0 + u_clock * vec2(0.8, 0.5)) - vnoise(pm * 40.0 - u_clock * vec2(0.6, 0.7));
        wc += 0.08 * rip;
        float a = smoothstep(0.0, 0.002, d);
        c = mix(c, wc, a);
    }
    // příboj na pobřeží
    float shore = 1.0 - smoothstep(0.0, 0.004, abs(d));
    float n = vnoise(pm * 70.0 + u_clock * 1.5);
    c = mix(c, vec3(0.95, 0.98, 1.0), u_active * shore * (0.45 + 0.4 * n));
    f_color = vec4(c, 1.0);
}
"""

READY, BUILD, RISE, RESULT = "čeká", "stavba", "potopa", "výsledek"


class IslandFloodGame(Game):
    id = "potopa"
    title = "Potopa ostrovů"
    description = "Dva hráči: hladina moře stoupá, udrž svůj ostrov nad vodou. Vyhrává poslední ostrov."
    order = 96
    category = "soutěž"
    intro = "Moře bude stoupat! Postav si ve své půlce ostrov a udrž ho nad vodou."
    keys = [("Enter / mezerník", "začít / další kolo"), ("↑ / ↓", "rychlost stoupání moře")]

    BUILD_TIME = 20.0  # [s]
    RESULT_TIME = 12.0
    HANDS_TO_START = 0.5
    RISE_SPEED = 0.0015  # [m/s] na začátku
    RISE_ACCEL = 0.00004  # [m/s²] moře stoupá čím dál rychleji
    START_PERCENTILE = 35.0  # hladina na začátku potopy = tento percentil výšek písku
    MIN_AREA = 0.0020  # [m²] ostrov menší než ~4,5 × 4,5 cm už se nepočítá
    SINK_HOLD = 1.0  # [s] jak dlouho musí být ostrov pod limitem, aby se počítal jako potopený
    MAX_SEA = 0.19  # [m]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.speed = getattr(self, "speed", 1.0)
        self.score = getattr(self, "score", [0, 0])
        self.reset()

    def stop(self) -> None:
        self.prog.release()

    def reset(self) -> None:
        self.clock = 0.0
        self.state, self.state_t = READY, 0.0
        self.hand_time = 0.0
        self.sea = -1.0
        self.area = [0.0, 0.0]
        self.peaks: list[tuple[int, int, float] | None] = [None, None]
        self.below = [0.0, 0.0]  # jak dlouho je ostrov pod limitem
        self.sunk = [False, False]
        self.winner: int | None = None
        self._zones: dict = {}

    # --- logika ------------------------------------------------------------------------------
    def zones(self, hm) -> list[np.ndarray]:
        key = (hm.shape, hm.size)
        if key not in self._zones:
            border = max(2, hm.shape[1] // 40)
            self._zones = {key: [vs.zone_mask(hm.shape, p, border) for p in (0, 1)]}
        return self._zones[key]

    def island_area(self, heights: np.ndarray, zone: np.ndarray, cell: tuple[float, float]) -> float:
        """Plocha souše nad hladinou v zóně [m²]."""
        return float(np.count_nonzero(zone & (heights > self.sea))) * cell[0] * cell[1]

    def update(self, hm, dt: float) -> None:
        self.clock += dt
        self.state_t += dt
        hands = hm.hands is not None and int(hm.hands.sum()) > 12
        self.hand_time = self.hand_time + dt if hands else 0.0
        zones = self.zones(hm)
        self.peaks = [vs.zone_peak(hm.heights, hm.hands, z) for z in zones]

        if self.state == READY:
            if self.hand_time >= self.HANDS_TO_START:
                self.start_build()
        elif self.state == BUILD:
            if self.state_t >= self.BUILD_TIME:
                self.start_rise(hm)
        elif self.state == RISE:
            t = self.state_t
            self.sea += (self.RISE_SPEED + self.RISE_ACCEL * t) * self.speed * dt
            self.sea = min(self.sea, self.MAX_SEA)
            for p in (0, 1):
                self.area[p] = self.island_area(hm.heights, zones[p], hm.cell_size)
                self.below[p] = self.below[p] + dt if self.area[p] < self.MIN_AREA else 0.0
                self.sunk[p] = self.below[p] >= self.SINK_HOLD
            if any(self.sunk) or self.sea >= self.MAX_SEA:
                self._finish()
        elif self.state == RESULT and self.state_t >= self.RESULT_TIME:
            self.state, self.state_t = READY, 0.0
            self.hand_time = 0.0

    def start_build(self) -> None:
        self.state, self.state_t = BUILD, 0.0
        self.sunk = [False, False]
        self.below = [0.0, 0.0]
        self.winner = None
        self.sea = -1.0
        self.host.notify("Stavte ostrovy!")

    def start_rise(self, hm) -> None:
        inner = np.zeros(hm.shape, bool)
        for z in self.zones(hm):
            inner |= z
        self.sea = float(np.percentile(hm.heights[inner], self.START_PERCENTILE))
        peaks = [vs.zone_peak(hm.heights, None, z) for z in self.zones(hm)]
        if all(pk is not None for pk in peaks):  # na začátku má ostrov každý (aspoň kousek)
            self.sea = min(self.sea, min(pk[2] for pk in peaks) - 0.015)
        self.state, self.state_t = RISE, 0.0
        self.host.notify("Moře stoupá!")

    def _finish(self) -> None:
        if self.sunk[0] != self.sunk[1]:
            self.winner = 1 if self.sunk[0] else 0
        else:  # oba najednou nebo moře na maximu: vyhrává vyšší ostrov
            a, b = (pk[2] if pk is not None else -1.0 for pk in self.peaks)
            self.winner = None if abs(a - b) < 0.003 else (0 if a > b else 1)
        if self.winner is not None:
            self.score[self.winner] += 1
        self.state, self.state_t = RESULT, 0.0

    # --- kreslení ------------------------------------------------------------------------------
    def render(self) -> None:
        active = self.state in (RISE, RESULT)
        self.host.draw_fullscreen(self.prog, u_sea=float(self.sea), u_clock=float(self.clock),
                                  u_active=1.0 if active else 0.0)
        p = self.host.painter
        canvas = self.host.canvas_size
        vs.draw_divider(p, canvas)
        hm = self.host.heightmap
        if hm is not None and self.state != READY:
            for player, peak in enumerate(self.peaks):
                if peak is not None and not self.sunk[player]:
                    self._flag(p, hm, player, peak)
        for player in (0, 1):
            self._hud(p, canvas, player)

    def _flag(self, p, hm, player: int, peak) -> None:
        iy, ix, _ = peak
        gh, gw = hm.shape
        cw, ch = self.host.canvas_size
        x, y = (ix + 0.5) / gw * cw, (iy + 0.5) / gh * ch
        s = ch * 0.045
        away = 1 if player == 0 else -1  # „nahoru“ z pohledu hráče = směrem od něj
        side = 1 if player == 0 else -1  # „doprava“ z pohledu hráče
        pole = s * 1.7
        top = np.array([x + away * pole, y])
        low = np.array([x + away * pole * 0.55, y])
        wave = math.sin(self.clock * 6) * s * 0.1
        tip = np.array([x + away * (pole * 0.78 + wave), y + side * s * 1.1])
        p.circle(x, y, s * 0.25, (0, 0, 0, 0.6))
        p.polyline([(x, y), tuple(top)], (0.1, 0.06, 0.02, 1), width=max(4, s * 0.14))
        flag = np.array([top, low, tip])
        c = flag.mean(0)
        p.triangles((flag - c) * 1.2 + c, (0, 0, 0, 1))
        p.triangles(flag, vs.COLORS[player])

    def _hud(self, p, canvas, player: int) -> None:
        size = vs.text_size(canvas)
        small = vs.text_size(canvas, 0.75)
        if self.state == READY:
            vs.player_text(p, canvas, player, "Sáhni do písku a začneme!", 0.035, size, color=(1, 1, 1, 1),
                           background=(0, 0, 0, 0.5), padding=size * 0.2)
            vs.player_text(p, canvas, player, "Pak bude stoupat moře", 0.095, small, color=(0.6, 0.9, 1, 1))
        elif self.state == BUILD:
            left = max(0, math.ceil(self.BUILD_TIME - self.state_t))
            vs.player_text(p, canvas, player, f"Moře začne stoupat za {left} s!", 0.035, size,
                           color=(1, 0.95, 0.3, 1))
            vs.player_text(p, canvas, player, "Postav si co nejvyšší ostrov", 0.095, small, color=(1, 1, 1, 1))
        elif self.state == RISE:
            peak = self.peaks[player]
            above = (peak[2] - self.sea) if peak is not None else 0.0
            danger = above < 0.01 or self.below[player] > 0
            txt = f"Ostrov {max(0.0, above) * 100:.1f} cm nad vodou".replace(".", ",")
            vs.player_text(p, canvas, player, txt, 0.035, size,
                           color=(1, 0.3, 0.25, 1) if danger else vs.COLORS[player])
            vs.player_text(p, canvas, player, "Přidej písek – moře stoupá!" if danger else "Moře stoupá…",
                           0.095, small, color=(1, 1, 1, 1))
        elif self.state == RESULT:
            if self.winner is None:
                text, col = "Remíza!", (1, 0.95, 0.3, 1)
            elif self.winner == player:
                text, col = "Váš ostrov vydržel!", (0.35, 1.0, 0.45, 1)
            else:
                text, col = "Ostrov je pod vodou!", (1.0, 0.45, 0.35, 1)
            vs.player_text(p, canvas, player, text, 0.045, int(size * 1.2), color=col,
                           background=(0, 0, 0, 0.6), padding=size * 0.25)
            vs.player_text(p, canvas, player, f"Výhry {self.score[player]} : {self.score[1 - player]}", 0.115,
                           small, color=(1, 1, 1, 1))

    # --- ovládání ---------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name in ("enter", "return", "space"):
            hm = self.host.heightmap
            if self.state == BUILD and hm is not None:
                self.start_rise(hm)
            else:
                self.start_build()
        elif name in ("up", "down"):
            self.speed = float(np.clip(self.speed * (1.25 if name == "up" else 0.8), 0.3, 4.0))
            self.host.notify(f"Rychlost moře ×{self.speed:.2f}")
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Stav: {self.state} ({self.state_t:.0f} s)", f"Rychlost moře ×{self.speed:.2f}",
                 f"Výhry: červení {self.score[0]}, modří {self.score[1]}"]
        if self.state == RISE:
            lines.append(f"Hladina {self.sea * 100:.1f} cm")
            lines.append(f"Ostrovy: {self.area[0] * 1e4:.0f} cm² / {self.area[1] * 1e4:.0f} cm²")
        return lines
