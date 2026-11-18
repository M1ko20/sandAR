"""Modeluj podle předlohy – závod pro dva hráče, nebo klidné modelování pro jednoho.

Na písek se promítne cílová vrstevnicová mapa (kopec, údolí, sedlo, …) a projektor
průběžně ukazuje procento shody.

* **2 hráči** – každý má svou půlku (čára přes střed delší strany) se stejnou
  předlohou otočenou k sobě. Vyhrává ten, kdo první udrží shodu ``DUEL_SUCCESS``
  po dobu ``HOLD`` s. Hodnotí se přísně (jako „Postav podle předlohy“).
  Když to do ``DUEL_LIMIT`` nikdo nedá, vyhrává vyšší shoda.
* **1 hráč** – celá bedna, mírné hodnocení ``evaluate_shape``: počítá se hlavně tvar
  (výška kopce se smí lišit), drobné nerovnosti se vyhladí a stačí ``SOLO_SUCCESS``.

Oba režimy vypadají stejně: barevná nápověda (červená = přidej, modrá = uber, zelená = sedí)
je zapnutá a hodnotí se jen hlavní část předlohy v čárkovaném rámečku (zbytek je ztmavený).
Klávesa P přepíná 1/2 hráče, M nápovědu, ←/→ předlohu.
"""

from __future__ import annotations

import math

import numpy as np

import cv2

from ..render import hud
from . import predloha_fx as fx
from . import versus as vs
from .base import Game
from .targets import BUILTIN, Score, Target, evaluate, evaluate_shape, half_layout

FRAG = vs.ARENA_GLSL + fx.HINT_GLSL + """
uniform sampler2D u_target;
uniform sampler2D u_mask;
uniform sampler2D u_halo;
uniform int u_hint;
uniform int u_split;
uniform float u_tol;
uniform vec2 u_offset;    // posun hladiny (levá půlka / celé, pravá půlka)
uniform vec2 u_success;   // 0..1 zelené orámování (levá / celé, pravá)
uniform vec2 u_cel;       // [s] od úspěchu (levá / celé, pravá), < 0 = nic
uniform vec4 u_center;    // uv středu hlavní části (levá / celé: xy, pravá: zw)

const float PRED_CELL = 0.075;  // [m] mřížka symbolů + / −

void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    int right = (u_split == 1 && uv.x > 0.5) ? 1 : 0;
    float off = right == 1 ? u_offset.y : u_offset.x;
    float t = sample_bicubic(u_target, uv) + off;
    float m = texture(u_mask, uv).r;
    float d = h - t;
    float shade = hillshade(uv, 2.5);
    vec3 c = vec3(0.90, 0.88, 0.80) * mix(0.72, 1.05, shade);
    // vrstevnice současného písku (tenké, šedé) – pro srovnání s předlohou
    float cur = contour_line(h, u_contour_interval, 1.2);
    c = mix(c, vec3(0.45), cur * 0.55);
    if (u_hint == 1) {
        c = pred_hint(c, d, u_tol);
    }
    // vrstevnice předlohy (silné, hnědé, hlavní ještě silnější)
    float tl = contour_line(t, u_contour_interval, 2.6);
    float tm = contour_line(t, u_contour_interval * float(u_major_every), 4.0);
    c = mix(c, vec3(0.30, 0.13, 0.02), max(tl, tm) * m);
    if (u_hint == 1) {  // + / − tam, kde je chyba největší
        vec2 cu = pred_cell(uv, PRED_CELL);
        if (u_split == 1 && (cu.x > 0.5) != (right == 1)) cu.x = right == 1 ? 0.5001 : 0.4999;
        float dc = height_at(cu) - (sample_bicubic(u_target, cu) + off);
        c = pred_symbol(c, uv, cu, dc, texture(u_mask, cu).r, u_tol);
    }
    // okolí hlavní části je ztmavené – tam na tvaru nezáleží
    vec3 outside = vec3(0.05, 0.08, 0.14) * mix(0.6, 1.5, shade);
    c = mix(outside, c, smoothstep(0.35, 0.65, m));
    c = pred_holo(c, uv, m, texture(u_halo, uv).r);
    float cel = right == 1 ? u_cel.y : u_cel.x;
    vec2 cen = right == 1 ? u_center.zw : u_center.xy;
    c = pred_celebrate(c, uv, m, cel, cen);
    // orámování při úspěchu
    float s = right == 1 ? u_success.y : u_success.x;
    float x0 = u_split == 1 ? (right == 1 ? 0.5 : 0.0) : 0.0;
    float x1 = u_split == 1 ? (right == 1 ? 1.0 : 0.5) : 1.0;
    float edge = min(min(uv.x - x0, x1 - uv.x) * u_box.x, min(uv.y, 1.0 - uv.y) * u_box.y);
    c = mix(c, vec3(0.2, 1.0, 0.3), s * (1.0 - smoothstep(0.02, 0.04, edge)));
    c += vec3(0.2, 1.0, 0.3) * s * 0.5 * exp(-edge / 0.05);
    // dva hráči: aréna (záře půlek jen mimo hlavní část, ať nápověda zůstane čistá)
    if (u_split == 1) {
        float free = 1.0 - smoothstep(0.2, 0.5, m);
        c = vs_arena_k(c, uv, 0.25 * free, free);
    }
    f_color = vec4(c, 1.0);
}
"""

READY, ROUND, RESULT = "čeká", "kolo", "výsledek"
LEGEND = [((1.0, 0.15, 0.08, 1), "přidej"), ((0.15, 0.35, 1.0, 1), "uber"), ((0.2, 0.85, 0.35, 1), "sedí")]


class ModelRaceGame(Game):
    id = "modeluj"
    title = "Modeluj podle předlohy"
    description = "Závod dvou hráčů: každý má svou půlku se stejnou předlohou. Kdo ji první vymodeluje, vyhrává."
    order = 97
    category = "dva hráči"
    intro = "V rámečku vymodeluj tvar z předlohy: červená = přidej, modrá = uber, zelená = sedí!"
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
        self.hint = getattr(self, "hint", True)  # barevná nápověda – výchozí zapnutá pro 1 i 2 hráče
        self.targets: list[Target] = list(BUILTIN)
        self.tex = None
        self.mask_tex = None
        self.halo_tex = None
        self.centers = (0.5, 0.5, 0.5, 0.5)
        self.arena = vs.Arena(self)
        self.confetti = vs.Confetti()
        self._offs = [fx.Smooth(5.0), fx.Smooth(5.0)]
        self._shown = [fx.Smooth(5.0), fx.Smooth(5.0)]
        self.grid: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.reset()

    def stop(self) -> None:
        self.prog.release()
        for tex in (self.tex, self.mask_tex, self.halo_tex):
            if tex is not None:
                tex.release()
        self.tex = self.mask_tex = self.halo_tex = None

    def reset(self) -> None:
        self.t = 0.0
        self.state, self.state_t = READY, 0.0
        self.hand_time = 0.0
        self.scores: list[Score | None] = [None, None]
        self.good_since: list[float | None] = [None, None]
        self._chime_t = [-99.0, -99.0]
        self.winner: int | None = None
        self.finish_time = 0.0
        self._next_eval = 0.0
        self.grid = None

    @property
    def target(self) -> Target:
        return self.targets[self.index % len(self.targets)]

    @property
    def show_hint(self) -> bool:
        return bool(self.hint)

    @property
    def success(self) -> float:
        return self.SOLO_SUCCESS if self.players == 1 else self.DUEL_SUCCESS

    # --- předloha ------------------------------------------------------------------------------
    def build_grid(self, shape: tuple[int, int], box: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """(výšky předlohy, maska hlavní části) – pro dva hráče stejná předloha otočená ke každému."""
        relief = self.host.settings.challenge.target_relief
        if self.players == 1:
            return self.target.layout(shape, box, relief)
        gh, gw = shape
        half = gw // 2
        grid = np.zeros(shape, np.float32)
        mask = np.zeros(shape, bool)
        grid[:, :half], mask[:, :half] = half_layout(self.target, (gh, half), box, relief, 0)
        grid[:, half:], mask[:, half:] = half_layout(self.target, (gh, gw - half), box, relief, 1)
        return grid, mask

    def _ensure_grid(self, hm) -> None:
        if self.grid is not None and self.grid.shape == hm.shape:
            return
        self.grid, self.mask = self.build_grid(hm.shape, hm.size)
        for tex in (self.tex, self.mask_tex, self.halo_tex):
            if tex is not None:
                tex.release()
        self.tex = self.host.float_texture((hm.shape[1], hm.shape[0]), self.grid)
        soft = cv2.GaussianBlur(self.mask.astype(np.float32), (0, 0), 1.5)
        self.mask_tex = self.host.float_texture((hm.shape[1], hm.shape[0]), soft)
        cell = hm.size[0] / hm.shape[1]
        self.halo_tex = self.host.float_texture((hm.shape[1], hm.shape[0]), fx.halo_mask(self.mask, cell))
        if self.players == 1:
            self.centers = fx.mask_center(self.mask) * 2
        else:
            gw, half = hm.shape[1], hm.shape[1] // 2
            lx, ly = fx.mask_center(self.mask[:, :half])
            rx, ry = fx.mask_center(self.mask[:, half:])
            self.centers = (lx * half / gw, ly, (half + rx * (gw - half)) / gw, ry)

    def evaluate(self, heights: np.ndarray) -> list[Score | None]:
        assert self.grid is not None
        assert self.mask is not None
        if self.players == 1:
            return [evaluate_shape(heights, self.grid, mask=self.mask)[0], None]
        tol = self.host.settings.challenge.tolerance
        gw = heights.shape[1]
        out: list[Score | None] = []
        for p in (0, 1):
            cols = vs.half_cols(gw, p)
            out.append(evaluate(heights[:, cols], self.grid[:, cols], tol, border=0.06, mask=self.mask[:, cols])[0])
        return out

    # --- logika ------------------------------------------------------------------------------
    def update(self, hm, dt: float) -> None:
        self._ensure_grid(hm)
        self.t += dt
        self.state_t += dt
        self.arena.update(dt, self.host.canvas_size)
        self.arena.phase(self.state, shock=self.state == ROUND and self.players == 2)
        if self.state == ROUND and self.players == 2:
            self.arena.countdown(self.DUEL_LIMIT - self.state_t)
        self.confetti.update(dt)
        hands = hm.has_hands
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
                        if self.t - self._chime_t[p] > 3.0:  # shoda dosažena – vydrž! (ne při kolísání)
                            self._chime_t[p] = self.t
                            x = (0.25 + 0.5 * p) if self.players == 2 else None
                            self.sfx.play("chime", 0.7, x=x)
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
        self.arena.clear_win()

    def _finish(self, winner: int | None) -> None:
        self.winner = winner
        self.finish_time = self.state_t
        self.state, self.state_t = RESULT, 0.0
        canvas = self.host.canvas_size
        if self.players == 2:
            self.arena.win(winner, canvas)
        else:  # jeden hráč: konfety z hlavní části předlohy
            if winner is not None:
                self.sfx.play("tada", 1.0)
            self.confetti.burst(self.centers[0] * canvas[0], self.centers[1] * canvas[1], 220, fx.CONFETTI,
                                canvas[1] * 1.2)

    def next_target(self, step: int = 1) -> None:
        self.index = (self.index + step) % len(self.targets)
        self.grid = None
        self.state, self.state_t = READY, 0.0
        self.hand_time = 0.0
        self.scores = [None, None]
        self.arena.clear_win()

    # --- kreslení ------------------------------------------------------------------------------
    def render(self) -> None:
        if self.tex is None:
            return
        tol = self.SOLO_TOL if self.players == 1 else self.host.settings.challenge.tolerance
        offs = [self._offs[i](s.offset if s is not None else 0.0, self.t) for i, s in enumerate(self.scores)]
        glow = [0.0, 0.0]
        cel = [-1.0, -1.0]
        for p in range(self.players):
            if self.state == RESULT and (self.winner == p or (self.players == 1 and self.winner is not None)):
                glow[p] = 0.6 + 0.4 * math.sin(self.state_t * 6)
                cel[p] = self.state_t
            elif self.good_since[p] is not None and self.state == ROUND:
                glow[p] = 0.5 + 0.2 * math.sin(self.t * 8)  # drží shodu – rámeček „nabíjí“
        power = [0.5, 0.5]
        if self.players == 2 and self.state == ROUND:
            for i, s in enumerate(self.scores):
                power[i] = 0.3 + 0.7 * float(np.clip((s.score if s else 0.0) / self.success, 0, 1))
        self.host.draw_fullscreen(self.prog, u_target=(self.tex, 1), u_mask=(self.mask_tex, 2),
                                  u_halo=(self.halo_tex, 3), u_hint=int(self.show_hint),
                                  u_split=int(self.players == 2), u_tol=float(tol), u_offset=tuple(offs),
                                  u_success=tuple(glow), u_cel=tuple(cel), u_center=self.centers,
                                  **self.arena.uniforms(self.host.box_size, power))
        p = self.host.painter
        canvas = self.host.canvas_size
        self.arena.draw(p)
        self.confetti.draw(p)
        if self.players == 2:
            if self.state == ROUND:
                for player in (0, 1):
                    vs.draw_countdown(p, canvas, player, self.DUEL_LIMIT - self.state_t, self.DUEL_LIMIT)
            for player in (0, 1):
                self._hud_duel(p, canvas, player)
        else:
            self._hud_solo(p, canvas)
        if self.show_hint and self.state != RESULT:
            cw, ch = canvas
            sz = hud.size(canvas, 0.75)
            if self.players == 2:  # legenda uprostřed u dlouhé strany, čitelná pro oba
                hud.legend(p, LEGEND, cw / 2, ch - sz * 0.3, sz, anchor="bc")
            else:
                hud.legend(p, LEGEND, sz * 0.4, ch - sz * 0.3, sz, anchor="bl")

    def _hud_duel(self, p, canvas, player: int) -> None:
        size = vs.text_size(canvas)
        small = vs.text_size(canvas, 0.75)
        s = self.scores[player]
        pct = f"{self._shown[player](s.score, self.t):.0f} %" if s is not None else "–"
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
        """Jeden hráč – stejné rozložení jako „Postav podle předlohy“: název nahoře, legenda vlevo
        dole, shoda vpravo dole."""
        cw, ch = canvas
        size = hud.size(canvas)
        s = self.scores[0]
        hud.label(p, f"Předloha: {self.target.name}", cw / 2, size * 0.25, size, anchor="tc")
        right, bottom = cw - size * 0.4, ch - size * 0.3
        if self.state == READY:
            hud.label(p, "Sáhni do písku a začni!", right, bottom, hud.size(canvas, 0.85), anchor="br",
                      color=hud.YELLOW)
        elif self.state == ROUND and s is not None:
            secs = int(self.state_t)
            near = s.score >= self.success * 0.75
            hud.label(p, f"Shoda {s.score:.0f} %  ·  {secs // 60}:{secs % 60:02d}", right, bottom, size, anchor="br",
                      color=hud.GREEN if near else hud.WHITE)
            bw = cw * 0.25
            k = float(np.clip(s.score / self.success, 0, 1))
            hud.bar(p, right - bw, bottom - size * 2.0, bw, size * 0.32, k, hud.GREEN if k >= 1 else hud.YELLOW)
        elif self.state == RESULT:
            big = int(size * 1.5)
            p.rect(0, ch / 2 - big * 1.2, cw, big * 2.5, (0, 0, 0, 0.65))
            hud.label(p, "Výborně, sedí to!", cw / 2, ch / 2 - big * 0.1, big, anchor="bc", color=hud.GREEN, bg=None)
            hud.label(p, f"Hotovo za {self.finish_time:.0f} s · shoda {s.score:.0f} %" if s else "Hotovo!",
                      cw / 2, ch / 2 + big * 0.2, size, anchor="tc", bg=None)

    # --- ovládání ---------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name == "p":
            self.players = 1 if self.players == 2 else 2
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
