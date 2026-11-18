"""Král kopce – hra pro dva hráče.

Bedna je čarou rozdělená na dvě půlky (přes střed delší strany), každý hráč stojí
u své kratší strany. Dokud má hráč ve své půlce nejvyšší vrchol, přibývají mu body
(1 bod za sekundu). Písek je ale jen jeden – hráči si ho berou navzájem.
Pás u dělicí čáry je neutrální (kopec přes čáru se nepočítá nikomu).

Kolo trvá 60 s a začne samo, jakmile někdo sáhne do písku (nebo Enter).
"""

from __future__ import annotations

import math

import numpy as np

from . import versus as vs
from .base import Game

FRAG = vs.ARENA_GLSL + """
void main() {
    float h = height_at(v_uv);
    vec3 c = elevation_color(h);
    c *= mix(0.62, 1.12, hillshade(v_uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.35, cl.x * 0.8);
    c = mix(c, vec3(0.06), cl.y);
    c = vs_arena(c, v_uv, 0.28);
    f_color = vec4(c, 1.0);
}
"""

READY, ROUND, RESULT = "čeká", "kolo", "výsledek"


class KingOfHillGame(Game):
    id = "kral"
    title = "Král kopce"
    description = "Dva hráči: body dostává ten, kdo má ve své půlce nejvyšší vrchol. Písek si berete navzájem!"
    order = 95
    category = "dva hráči"
    intro = "Postav ve své půlce nejvyšší kopec! Kdo je zrovna nejvýš, sbírá body."
    keys = [("Enter / mezerník", "nové kolo hned"), ("X", "ukončit kolo")]

    ROUND_TIME = 60.0
    RESULT_TIME = 12.0
    HANDS_TO_START = 0.5  # [s]
    TIE = 0.003  # [m] menší rozdíl vrcholů = nikdo nevede
    NEUTRAL = 0.03  # [m] neutrální pás na každou stranu od dělicí čáry

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.reset()

    def stop(self) -> None:
        self.prog.release()

    def reset(self) -> None:
        self.t = 0.0
        self.state, self.state_t = READY, 0.0
        self.hand_time = 0.0
        self.points = [0.0, 0.0]
        self.peaks: list[tuple[int, int, float] | None] = [None, None]
        self.leader: int | None = None
        self.winner: int | None = None
        self._zones: dict = {}
        self.arena = vs.Arena()
        self.lead_flash: tuple[int, float] | None = None  # (nový král, čas) – záblesk při převzetí koruny

    # --- logika ------------------------------------------------------------------------------
    def zones(self, hm) -> list[np.ndarray]:
        key = (hm.shape, hm.size)
        if key not in self._zones:
            gh, gw = hm.shape
            border = max(2, gw // 40)
            neutral = max(1, int(round(self.NEUTRAL / hm.cell_size[0])))
            self._zones = {key: [vs.zone_mask(hm.shape, p, border, neutral) for p in (0, 1)]}
        return self._zones[key]

    def update(self, hm, dt: float) -> None:
        self.t += dt
        self.state_t += dt
        hands = hm.has_hands
        self.hand_time = self.hand_time + dt if hands else 0.0
        self.peaks = [vs.zone_peak(hm.heights, hm.hands, z) for z in self.zones(hm)]
        prev_leader = self.leader
        a, b = self.peaks
        if a is None or b is None:
            self.leader = 0 if b is None and a is not None else 1 if a is None and b is not None else None
        elif abs(a[2] - b[2]) <= self.TIE:
            self.leader = None
        else:
            self.leader = 0 if a[2] > b[2] else 1
        if self.leader is not None and self.leader != prev_leader and self.state == ROUND:
            self.lead_flash = (self.leader, self.t)
        self.arena.update(dt, self.host.canvas_size)
        self.arena.phase(self.state, shock=self.state == ROUND)

        if self.state == READY:
            if self.hand_time >= self.HANDS_TO_START:
                self.start_round()
        elif self.state == ROUND:
            if self.leader is not None:
                self.points[self.leader] += dt
            if self.state_t >= self.ROUND_TIME:
                self._finish()
        elif self.state == RESULT and self.state_t >= self.RESULT_TIME:
            self.state, self.state_t = READY, 0.0
            self.hand_time = 0.0
            self.arena.clear_win()

    def start_round(self) -> None:
        self.state, self.state_t = ROUND, 0.0
        self.points = [0.0, 0.0]
        self.winner = None
        self.arena.clear_win()
        self.host.notify("Král kopce – start!")

    def _finish(self) -> None:
        a, b = (int(v) for v in self.points)
        self.winner = None if a == b else (0 if a > b else 1)
        self.state, self.state_t = RESULT, 0.0
        self.arena.win(self.winner, self.host.canvas_size)

    # --- kreslení ------------------------------------------------------------------------------
    def render(self) -> None:
        power = (0.55, 0.55)
        if self.state == ROUND and self.leader is not None:
            power = (1.0, 0.25) if self.leader == 0 else (0.25, 1.0)
        urgent = self.state == ROUND and self.ROUND_TIME - self.state_t <= 10
        alarm = 0.45 if urgent else 0.0
        self.host.draw_fullscreen(self.prog, **self.arena.uniforms(self.host.box_size, power, (alarm, alarm),
                                                                   neutral=self.NEUTRAL))
        p = self.host.painter
        canvas = self.host.canvas_size
        hm = self.host.heightmap
        if hm is not None:
            for player, peak in enumerate(self.peaks):
                if peak is not None:
                    self._marker(p, hm, player, peak)
        self.arena.draw(p)
        if self.state == ROUND:
            for player in (0, 1):
                vs.draw_countdown(p, canvas, player, self.ROUND_TIME - self.state_t, self.ROUND_TIME)
        for player in (0, 1):
            self._hud(p, canvas, player)

    def _marker(self, p, hm, player: int, peak) -> None:
        iy, ix, h = peak
        gh, gw = hm.shape
        cw, ch = self.host.canvas_size
        x, y = (ix + 0.5) / gw * cw, (iy + 0.5) / gh * ch
        r = ch * 0.03
        king = self.leader == player
        col = vs.COLORS[player]
        phase = (self.t * 1.2) % 1.0
        if king:  # zlatá záře a druhá vlna – král je vidět přes celou bednu
            vs.draw_glow_disc(p, x, y, r * (3.2 + 0.4 * math.sin(self.t * 4)), (1.0, 0.8, 0.2, 0.55))
            ph2 = (phase + 0.5) % 1.0
            p.circle(x, y, r * (1 + 3.0 * ph2), (1.0, 0.85, 0.25, 1 - ph2), width=max(5, ch / 120))
        else:
            vs.draw_glow_disc(p, x, y, r * 2.2, col[:3] + (0.35,))
        p.circle(x, y, r * (1 + 2.0 * phase), col[:3] + (1 - phase,), width=max(5, ch / 120))
        if self.lead_flash is not None and self.lead_flash[0] == player:
            ft = self.t - self.lead_flash[1]
            if ft < 1.2:  # nový král: rozpínající se zlatý prstenec
                p.circle(x, y, r * (1.5 + 9 * ft), (1.0, 0.9, 0.4, 1 - ft / 1.2), width=max(8, ch / 80))
        p.circle(x, y, r * 0.62, (0, 0, 0, 1))
        p.circle(x, y, r * 0.48, (1.0, 0.85, 0.2, 1) if king else col)
        rot = vs.ROTATION[player]
        # korunka „nad“ vrcholem z pohledu hráče (směrem od něj), výška pod ním (směrem k hráči)
        away = 1 if player == 0 else -1
        if king:
            bob = 1 + 0.08 * math.sin(self.t * 5)
            vs.draw_crown(p, x + away * r * 2.0, y, r * 1.2 * bob, rot)
        size = vs.text_size(self.host.canvas_size, 0.8)
        cm = f"{h * 100:.1f}".replace(".", ",") + " cm"
        p.text(cm, x - away * (r + size * 0.75), y, size=size, anchor="mc", bold=True,
               color=(1, 0.85, 0.15, 1) if king else (1, 1, 1, 1), outline=3, rotation=rot)

    def _hud(self, p, canvas, player: int) -> None:
        size = vs.text_size(canvas)
        small = vs.text_size(canvas, 0.75)
        pts = int(self.points[player])
        if self.state == READY:
            vs.player_text(p, canvas, player, "Sáhni do písku a začneme!", 0.035, size, color=(1, 1, 1, 1),
                           background=(0, 0, 0, 0.5), padding=size * 0.2)
            vs.player_text(p, canvas, player, "Postav ve své půlce nejvyšší kopec", 0.095, small,
                           color=(1, 0.95, 0.3, 1))
        elif self.state == ROUND:
            left = max(0, math.ceil(self.ROUND_TIME - self.state_t))
            urgent = left <= 10
            vs.player_text(p, canvas, player, f"{vs.NAMES[player]}: {pts} b.", 0.035, size)
            msg = "Jsi král kopce!" if self.leader == player else "Přestav soupeře!" if self.leader is not None \
                else "Kopce jsou stejně vysoké!"
            vs.player_text(p, canvas, player, f"{msg}  ·  zbývá {left} s", 0.095, small,
                           color=(1, 0.3, 0.25, 1) if urgent else (1, 1, 1, 1))
        elif self.state == RESULT:
            if self.winner is None:
                text, col = "Remíza!", (1, 0.95, 0.3, 1)
            elif self.winner == player:
                text, col = "Vyhráli jste!", (0.35, 1.0, 0.45, 1)
            else:
                text, col = "Tentokrát ne…", (1.0, 0.45, 0.35, 1)
            vs.player_text(p, canvas, player, text, 0.045, int(size * 1.3), color=col,
                           background=(0, 0, 0, 0.6), padding=size * 0.25)
            other = 1 - player
            vs.player_text(p, canvas, player, f"Body {pts} : {int(self.points[other])}", 0.115, small,
                           color=(1, 1, 1, 1))

    # --- ovládání ---------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name in ("enter", "return", "space"):
            self.start_round()
        elif name == "x":
            if self.state == ROUND:
                self.state, self.state_t = READY, 0.0
                self.hand_time = 0.0
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Stav: {self.state} ({self.state_t:.0f} s)",
                 f"Body: červení {self.points[0]:.0f}, modří {self.points[1]:.0f}"]
        for player, peak in enumerate(self.peaks):
            if peak is not None:
                lines.append(f"Vrchol {vs.NAMES[player]}: {peak[2] * 100:.1f} cm")
        return lines
