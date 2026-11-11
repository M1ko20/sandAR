"""Modeluj podle předlohy – závod pro dva hráče, nebo klidné modelování pro jednoho.

Na písek se promítne cílová vrstevnicová mapa (kopec, údolí, sedlo, …) a projektor
průběžně ukazuje procento shody.

* **2 hráči** – každý má svou půlku (čára přes střed delší strany) se stejnou
  předlohou otočenou k sobě. Vyhrává ten, kdo první udrží shodu ``DUEL_SUCCESS``
  po dobu ``HOLD`` s. Hodnotí se přísně (jako „Postav podle předlohy“).
  Když to do ``DUEL_LIMIT`` nikdo nedá, vyhrává vyšší shoda.
* **1 hráč** – celá bedna, mírné hodnocení ``evaluate_shape``: počítá se hlavně tvar
  (výška kopce se smí lišit), drobné nerovnosti se vyhladí a stačí ``SOLO_SUCCESS``.
  Barevná nápověda (červená = přidej, modrá = uber) je zapnutá.

Klávesa P přepíná 1/2 hráče, M nápovědu, ←/→ předlohu.
"""

from __future__ import annotations

import math

import numpy as np

from . import versus as vs
from .base import Game
from .targets import BUILTIN, Score, Target, evaluate, evaluate_shape, half_grid

FRAG = """
uniform sampler2D u_target;
uniform int u_hint;
uniform int u_split;
uniform float u_tol;
uniform vec2 u_offset;    // posun hladiny (levá půlka / celé, pravá půlka)
uniform vec2 u_success;   // 0..1 zelené orámování (levá / celé, pravá)

void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    int right = (u_split == 1 && uv.x > 0.5) ? 1 : 0;
    float off = right == 1 ? u_offset.y : u_offset.x;
    float t = sample_bicubic(u_target, uv) + off;
    float d = h - t;
    vec3 c = vec3(0.97, 0.95, 0.87) * mix(0.72, 1.05, hillshade(uv, 2.5));
    // vrstevnice současného písku (tenké, šedé) – pro srovnání s předlohou
    float cur = contour_line(h, u_contour_interval, 1.2);
    c = mix(c, vec3(0.45), cur * 0.55);
    if (u_hint == 1) {
        float k = clamp((abs(d) - 0.5 * u_tol) / (2.0 * u_tol), 0.0, 1.0);
        vec3 tint = d < 0.0 ? vec3(1.0, 0.12, 0.08) : vec3(0.10, 0.35, 1.0);
        c = mix(c, tint, k * 0.75);
        float good = 1.0 - smoothstep(0.5 * u_tol, u_tol, abs(d));
        c = mix(c, vec3(0.25, 0.85, 0.35), good * 0.22);
    }
    // vrstevnice předlohy (silné, hnědé, hlavní ještě silnější)
    float tl = contour_line(t, u_contour_interval, 2.6);
    float tm = contour_line(t, u_contour_interval * float(u_major_every), 4.0);
    c = mix(c, vec3(0.50, 0.25, 0.06), max(tl, tm));
    // orámování při úspěchu
    float s = right == 1 ? u_success.y : u_success.x;
    float x0 = u_split == 1 ? (right == 1 ? 0.5 : 0.0) : 0.0;
    float x1 = u_split == 1 ? (right == 1 ? 1.0 : 0.5) : 1.0;
    float edge = min(min(uv.x - x0, x1 - uv.x) * u_box.x, min(uv.y, 1.0 - uv.y) * u_box.y);
    c = mix(c, vec3(0.2, 1.0, 0.3), s * (1.0 - smoothstep(0.02, 0.04, edge)));
    f_color = vec4(c, 1.0);
}
"""

READY, ROUND, RESULT = "čeká", "kolo", "výsledek"


class ModelRaceGame(Game):
    id = "modeluj"
    title = "Modeluj podle předlohy"
    description = "Podle promítnutých vrstevnic vymodeluj terén. Dva hráči závodí, jeden hráč má mírné hodnocení."
    order = 97
    category = "soutěž"
    intro = "Hnědé čáry jsou vrstevnice předlohy – vymodeluj z písku stejný terén!"
    keys = [("P", "1 / 2 hráči"), ("← / →", "jiná předloha"), ("M", "barevná nápověda"),
            ("Enter", "začít / další předloha")]

    DUEL_SUCCESS = 80.0  # [%] přísné hodnocení
    DUEL_LIMIT = 180.0  # [s]
    SOLO_SUCCESS = 60.0  # [%] mírné hodnocení
    SOLO_TOL = 0.03  # [m] tolerance barevné nápovědy pro jednoho hráče
    HOLD = 1.5  # [s] jak dlouho musí shoda vydržet
    RESULT_TIME = 10.0
    HANDS_TO_START = 0.5
    EVAL_EVERY = 0.25  # [s]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.players = getattr(self, "players", 2)
        self.index = getattr(self, "index", 0)
        self.hint = getattr(self, "hint", None)  # None = podle počtu hráčů
        self.targets: list[Target] = list(BUILTIN)
        self.tex = None
        self.grid: np.ndarray | None = None
        self.reset()

    def stop(self) -> None:
        self.prog.release()
        if self.tex is not None:
            self.tex.release()
            self.tex = None

    def reset(self) -> None:
        self.t = 0.0
        self.state, self.state_t = READY, 0.0
        self.hand_time = 0.0
        self.scores: list[Score | None] = [None, None]
        self.good_since: list[float | None] = [None, None]
        self.winner: int | None = None
        self.finish_time = 0.0
        self._next_eval = 0.0
        self.grid = None

    @property
    def target(self) -> Target:
        return self.targets[self.index % len(self.targets)]

    @property
    def show_hint(self) -> bool:
        return self.players == 1 if self.hint is None else self.hint

    @property
    def success(self) -> float:
        return self.SOLO_SUCCESS if self.players == 1 else self.DUEL_SUCCESS

    # --- předloha ------------------------------------------------------------------------------
    def build_grid(self, shape: tuple[int, int], box: tuple[float, float]) -> np.ndarray:
        relief = self.host.settings.challenge.target_relief
        if self.players == 1:
            return self.target.grid(shape, box, relief)
        gh, gw = shape
        half = gw // 2
        grid = np.zeros(shape, np.float32)
        grid[:, :half] = half_grid(self.target, (gh, half), box, relief, 0)
        grid[:, half:] = half_grid(self.target, (gh, gw - half), box, relief, 1)
        return grid

    def _ensure_grid(self, hm) -> None:
        if self.grid is not None and self.grid.shape == hm.shape:
            return
        self.grid = self.build_grid(hm.shape, hm.size)
        if self.tex is not None:
            self.tex.release()
        self.tex = self.host.float_texture((hm.shape[1], hm.shape[0]), self.grid)

    def evaluate(self, heights: np.ndarray) -> list[Score | None]:
        assert self.grid is not None
        if self.players == 1:
            return [evaluate_shape(heights, self.grid)[0], None]
        tol = self.host.settings.challenge.tolerance
        gw = heights.shape[1]
        out: list[Score | None] = []
        for p in (0, 1):
            cols = vs.half_cols(gw, p)
            out.append(evaluate(heights[:, cols], self.grid[:, cols], tol, border=0.06)[0])
        return out

    # --- logika ------------------------------------------------------------------------------
    def update(self, hm, dt: float) -> None:
        self._ensure_grid(hm)
        self.t += dt
        self.state_t += dt
        hands = hm.hands is not None and int(hm.hands.sum()) > 12
        self.hand_time = self.hand_time + dt if hands else 0.0
        if self.t >= self._next_eval:
            self._next_eval = self.t + self.EVAL_EVERY
            self.scores = self.evaluate(hm.heights)
        if self.state == READY:
            if self.hand_time >= self.HANDS_TO_START:
                self.start_round()
        elif self.state == ROUND:
            for p in range(self.players):
                s = self.scores[p]
                if s is not None and s.score >= self.success:
                    if self.good_since[p] is None:
                        self.good_since[p] = self.state_t
                else:
                    self.good_since[p] = None
            done = [p for p in range(self.players)
                    if self.good_since[p] is not None and self.state_t - self.good_since[p] >= self.HOLD]
            if done:
                self._finish(done[0] if len(done) == 1 else self._better())
            elif self.players == 2 and self.state_t >= self.DUEL_LIMIT:
                self._finish(self._better())
        elif self.state == RESULT and self.state_t >= self.RESULT_TIME:
            self.next_target()

    def _better(self) -> int | None:
        a, b = (s.score if s is not None else 0.0 for s in self.scores)
        return None if abs(a - b) < 1.0 else (0 if a > b else 1)

    def start_round(self) -> None:
        self.state, self.state_t = ROUND, 0.0
        self.good_since = [None, None]
        self.winner = None

    def _finish(self, winner: int | None) -> None:
        self.winner = winner
        self.finish_time = self.state_t
        self.state, self.state_t = RESULT, 0.0

    def next_target(self, step: int = 1) -> None:
        self.index = (self.index + step) % len(self.targets)
        self.grid = None
        self.state, self.state_t = READY, 0.0
        self.hand_time = 0.0
        self.scores = [None, None]

    # --- kreslení ------------------------------------------------------------------------------
    def render(self) -> None:
        if self.tex is None:
            return
        tol = self.SOLO_TOL if self.players == 1 else self.host.settings.challenge.tolerance
        offs = [s.offset if s is not None else 0.0 for s in self.scores]
        glow = [0.0, 0.0]
        for p in range(self.players):
            if self.state == RESULT and (self.winner == p or (self.players == 1 and self.winner is not None)):
                glow[p] = 0.6 + 0.4 * math.sin(self.state_t * 6)
            elif self.good_since[p] is not None and self.state == ROUND:
                glow[p] = 0.5
        self.host.draw_fullscreen(self.prog, u_target=(self.tex, 1), u_hint=int(self.show_hint),
                                  u_split=int(self.players == 2), u_tol=float(tol), u_offset=tuple(offs),
                                  u_success=tuple(glow))
        p = self.host.painter
        canvas = self.host.canvas_size
        if self.players == 2:
            vs.draw_divider(p, canvas)
            for player in (0, 1):
                self._hud_duel(p, canvas, player)
        else:
            self._hud_solo(p, canvas)

    def _hud_duel(self, p, canvas, player: int) -> None:
        size = vs.text_size(canvas)
        small = vs.text_size(canvas, 0.75)
        s = self.scores[player]
        pct = f"{s.score:.0f} %" if s is not None else "–"
        if self.state == READY:
            vs.player_text(p, canvas, player, f"Předloha: {self.target.name}", 0.035, size, color=(1, 1, 1, 1),
                           background=(0, 0, 0, 0.55), padding=size * 0.2)
            vs.player_text(p, canvas, player, "Sáhni do písku a závod začne!", 0.095, small,
                           color=(1, 0.95, 0.3, 1), outline=3)
        elif self.state == ROUND:
            vs.player_text(p, canvas, player, f"Shoda {pct}", 0.035, size, background=(0, 0, 0, 0.55),
                           padding=size * 0.2)
            secs = int(self.state_t)
            vs.player_text(p, canvas, player, f"{self.target.name} · {secs // 60}:{secs % 60:02d}", 0.095, small,
                           color=(1, 1, 1, 1), background=(0, 0, 0, 0.45), padding=small * 0.2)
        elif self.state == RESULT:
            if self.winner is None:
                text, col = "Remíza!", (1, 0.95, 0.3, 1)
            elif self.winner == player:
                text, col = "Vyhráli jste!", (0.35, 1.0, 0.45, 1)
            else:
                text, col = "Soupeř byl rychlejší", (1.0, 0.45, 0.35, 1)
            vs.player_text(p, canvas, player, text, 0.045, int(size * 1.2), color=col,
                           background=(0, 0, 0, 0.65), padding=size * 0.25)
            vs.player_text(p, canvas, player, f"Shoda {pct} · čas {self.finish_time:.0f} s", 0.115, small,
                           color=(1, 1, 1, 1), background=(0, 0, 0, 0.5), padding=small * 0.2)

    def _hud_solo(self, p, canvas) -> None:
        cw, ch = canvas
        size = vs.text_size(canvas)
        s = self.scores[0]
        p.text(f"Předloha: {self.target.name}", cw / 2, size * 0.5, size=size, anchor="tc", bold=True,
               color=(1, 1, 1, 1), outline=3, background=(0, 0, 0, 0.5), padding=size * 0.2)
        if self.state == READY:
            msg = "Sáhni do písku a začni modelovat podle hnědých čar!"
            p.text(msg, cw / 2, ch - size * 0.6, size=int(size * 0.85), anchor="bc", bold=True,
                   color=(1, 0.95, 0.3, 1), outline=3, background=(0, 0, 0, 0.5), padding=size * 0.2)
        elif self.state == ROUND and s is not None:
            secs = int(self.state_t)
            near = s.score >= self.success * 0.75
            text = f"Shoda {s.score:.0f} %  ·  {secs // 60}:{secs % 60:02d}"
            p.text(text, cw / 2, ch - size * 0.6, size=size, anchor="bc", bold=True,
                   color=(0.4, 1.0, 0.5, 1) if near else (1, 1, 1, 1), outline=3,
                   background=(0, 0, 0, 0.5), padding=size * 0.2)
            # ukazatel: jak daleko je hráč od cíle
            bw, bh = cw * 0.4, size * 0.35
            x0, y0 = (cw - bw) / 2, ch - size * 2.3
            k = float(np.clip(s.score / self.success, 0, 1))
            p.rect(x0, y0, bw, bh, (0, 0, 0, 0.6))
            p.rect(x0, y0, bw * k, bh, (0.3, 0.9, 0.4, 1) if k >= 1 else (1, 0.8, 0.2, 1))
        elif self.state == RESULT:
            big = int(size * 1.5)
            p.rect(0, ch / 2 - big * 1.2, cw, big * 2.5, (0, 0, 0, 0.6))
            p.text("Výborně, sedí to!", cw / 2, ch / 2 - big * 0.1, size=big, anchor="bc", bold=True,
                   color=(0.35, 1.0, 0.45, 1), outline=4)
            p.text(f"Hotovo za {self.finish_time:.0f} s · shoda {s.score:.0f} %" if s else "Hotovo!",
                   cw / 2, ch / 2 + big * 0.2, size=size, anchor="tc", bold=True, color=(1, 1, 1, 1), outline=3)

    # --- ovládání ---------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name == "p":
            self.players = 1 if self.players == 2 else 2
            self.hint = None  # nápověda zase podle počtu hráčů
            self.reset()
            self.host.notify("Jeden hráč (mírné hodnocení)" if self.players == 1 else "Dva hráči – závod")
        elif name in ("left", "right"):
            self.next_target(1 if name == "right" else -1)
            self.host.notify(f"Předloha: {self.target.name}")
        elif name == "m":
            self.hint = not self.show_hint
            self.host.notify("Nápověda zapnutá" if self.hint else "Nápověda vypnutá")
        elif name in ("enter", "return", "space"):
            if self.state == ROUND:
                return True
            if self.state == RESULT:
                self.next_target()
            self.start_round()
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Hráči: {self.players} · předloha: {self.target.name}",
                 f"Stav: {self.state} ({self.state_t:.0f} s) · cíl {self.success:.0f} %"]
        for p in range(self.players):
            s = self.scores[p]
            if s is not None:
                who = vs.NAMES[p] if self.players == 2 else "Shoda"
                lines.append(f"{who}: {s.score:.0f} % (RMSE {s.rmse * 100:.1f} cm)")
        return lines
