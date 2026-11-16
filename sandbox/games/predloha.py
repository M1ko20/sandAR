"""Postav podle předlohy – úkol s automatickým vyhodnocením shody.

Hodnotí se jen **hlavní část** předlohy (kopec, kráter, koryto + 5 cm okolo) – zbytek bedny
je ztmavený a na jeho tvaru nezáleží. Tak úkol zvládne i jeden člověk za pár minut.

Režimy:
* Nápověda – červená = přidej písek, modrá = uber písek, zelená = sedí; čáry = vrstevnice předlohy.
* Mapa     – na písku jsou jen vrstevnice předlohy (jako mapa). Žák staví podle
             mapy a klávesou Enter si nechá terén vyhodnotit.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from ..render import hud
from .base import Game
from .targets import Score, Target, all_targets, evaluate, save_capture

log = logging.getLogger(__name__)

FRAG = """
uniform sampler2D u_target;
uniform sampler2D u_mask;   // 1 = hlavní část (hodnotí se), 0 = okolí
uniform int u_mode;        // 0 nápověda, 1 mapa, 2 mapa + odhalení chyb
uniform float u_tol;
uniform float u_offset;
uniform float u_success;   // 0..1 zelené orámování při úspěchu

void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    float t = sample_bicubic(u_target, uv) + u_offset;
    float m = texture(u_mask, uv).r;
    float d = h - t;
    float shade = hillshade(uv, 2.0);
    vec3 c;
    if (u_mode == 0) {
        c = vec3(0.80, 0.78, 0.72) * mix(0.70, 1.08, shade);
    } else {
        c = vec3(0.97, 0.95, 0.87) * mix(0.85, 1.03, shade);
    }
    if (u_mode != 1) {
        float k = clamp((abs(d) - 0.5 * u_tol) / (2.0 * u_tol), 0.0, 1.0);
        vec3 tint = d < 0.0 ? vec3(1.0, 0.10, 0.05) : vec3(0.08, 0.30, 1.0);
        c = mix(c, tint, k * 0.9);
        float good = 1.0 - smoothstep(0.5 * u_tol, u_tol, abs(d));
        c = mix(c, vec3(0.15, 0.85, 0.30), good * 0.55);
    }
    // vrstevnice předlohy
    float line = contour_line(t, u_contour_interval, 2.4);
    vec3 ink = u_mode == 0 ? vec3(0.02) : vec3(0.55, 0.30, 0.10);
    c = mix(c, ink, line * m);
    // okolí hlavní části: ztmavené, na tvaru nezáleží
    vec3 outside = vec3(0.10, 0.11, 0.14) * mix(0.6, 1.4, shade);
    c = mix(outside, c, smoothstep(0.35, 0.65, m));
    // čárkovaný obrys hlavní části
    float edge = 1.0 - smoothstep(0.0, 2.0 * fwidth(m) + 1e-4, abs(m - 0.5));
    float dash = step(0.45, fract((uv.x * u_box.x + uv.y * u_box.y) * 18.0 - u_time * 0.4));
    c = mix(c, vec3(1.0, 0.95, 0.6), edge * dash);
    // orámování při úspěchu
    float frame = min(min(uv.x, 1.0 - uv.x) * u_box.x, min(uv.y, 1.0 - uv.y) * u_box.y);
    c = mix(c, vec3(0.2, 1.0, 0.3), u_success * (1.0 - smoothstep(0.015, 0.03, frame)));
    f_color = vec4(c, 1.0);
}
"""

MODES = ["Nápověda", "Mapa"]

LEGEND = [((1.0, 0.15, 0.08, 1), "přidej písek"), ((0.15, 0.35, 1.0, 1), "uber"), ((0.2, 0.85, 0.35, 1), "sedí")]


class ChallengeGame(Game):
    id = "predloha"
    title = "Postav podle předlohy"
    description = "Postav tvar podle předlohy v označeném místě. Barvy ukazují, kde přidat a kde ubrat písek."
    order = 88
    category = "soutěž"
    intro = "V čárkovaném rámečku postav tvar z předlohy: červená = přidej, modrá = uber, zelená = sedí."
    keys = [
        ("← / →", "jiná předloha"),
        ("M", "režim nápověda / mapa"),
        ("Enter", "vyhodnotit (v režimu mapa)"),
        ("↑ / ↓", "převýšení předlohy"),
        ("U", "uložit současný písek jako předlohu"),
    ]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.targets: list[Target] = all_targets()
        self.index = getattr(self, "index", 0) % len(self.targets)
        self.mode = getattr(self, "mode", 0)
        self.target_grid: np.ndarray | None = None
        self.target_mask: np.ndarray | None = None
        self.target_tex = None
        self.mask_tex = None
        self.result: Score | None = None
        self.best = 0.0
        self.reveal_until = 0.0
        self.success_since: float | None = None
        self._next_eval = 0.0
        self.started = time.monotonic()

    def stop(self) -> None:
        self.prog.release()
        for tex in (self.target_tex, self.mask_tex):
            if tex is not None:
                tex.release()
        self.target_tex = self.mask_tex = None

    def reset(self) -> None:
        self.target_grid = None  # znovu sestavit (nové hodnocení, čas od nuly)

    @property
    def target(self) -> Target:
        return self.targets[self.index]

    def _build_target(self, shape: tuple[int, int]) -> None:
        cfg = self.host.settings.challenge
        try:
            grid, mask = self.target.layout(shape, self.host.box_size, cfg.target_relief)
        except Exception as exc:
            log.exception("Předlohu %s nelze načíst", self.target.name)
            self.host.notify(f"Předlohu nelze načíst: {exc}")
            grid, mask = np.zeros(shape, np.float32), np.ones(shape, bool)
        self.target_grid, self.target_mask = grid, mask
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)
        for tex in (self.target_tex, self.mask_tex):
            if tex is not None:
                tex.release()
        self.target_tex = self.host.float_texture((shape[1], shape[0]), grid)
        self.mask_tex = self.host.float_texture((shape[1], shape[0]), soft)
        self.result = None
        self.best = 0.0
        self.success_since = None
        self.started = time.monotonic()

    def _select(self, step: int) -> None:
        self.index = (self.index + step) % len(self.targets)
        self.target_grid = None
        self.host.notify(f"Předloha: {self.target.name}")

    # --- logika -------------------------------------------------------------------
    def update(self, hm, dt: float) -> None:
        if self.target_grid is None or self.target_grid.shape != hm.shape:
            self._build_target(hm.shape)
        now = time.monotonic()
        if now < self._next_eval:
            return
        self._next_eval = now + 0.25
        cfg = self.host.settings.challenge
        assert self.target_grid is not None
        self.result, _ = evaluate(hm.heights, self.target_grid, cfg.tolerance, cfg.border, self.target_mask)
        self.best = max(self.best, self.result.score)
        if self.result.score >= cfg.success_score:
            if self.success_since is None:
                self.success_since = now
        else:
            self.success_since = None

    def _show_score(self) -> bool:
        return self.mode == 0 or time.monotonic() < self.reveal_until

    def render(self) -> None:
        if self.target_tex is None:
            return
        mode = self.mode
        if mode == 1 and time.monotonic() < self.reveal_until:
            mode = 2
        success = 0.0
        if self.success_since is not None and self._show_score():
            success = 0.6 + 0.4 * np.sin((time.monotonic() - self.success_since) * 6)
        cfg = self.host.settings.challenge
        self.host.draw_fullscreen(self.prog, u_target=(self.target_tex, 1), u_mask=(self.mask_tex, 2), u_mode=mode,
                                  u_tol=cfg.tolerance, u_offset=self.result.offset if self.result else 0.0,
                                  u_success=float(success))
        self._labels(mode)

    def _labels(self, mode: int) -> None:
        p = self.host.painter
        canvas = self.host.canvas_size
        cw, ch = canvas
        size = hud.size(canvas)
        small = hud.size(canvas, 0.8)
        hud.label(p, f"Předloha: {self.target.name}", cw / 2, size * 0.25, size, anchor="tc")
        if mode != 1:
            hud.legend(p, LEGEND, size * 0.4, ch - size * 0.35, small, anchor="bl")
        if self.result is None:
            return
        cfg = self.host.settings.challenge
        if self._show_score():
            ok = self.success_since is not None
            text = f"Výborně! Shoda {self.result.score:.0f} %" if ok else f"Shoda {self.result.score:.0f} %"
            hud.label(p, text, cw - size * 0.4, ch - size * 0.35, int(size * (1.25 if ok else 1.0)), anchor="br",
                      color=hud.GREEN if ok else hud.WHITE)
            bw, bh = cw * 0.22, size * 0.32
            hud.bar(p, cw - size * 0.4 - bw, ch - size * 2.3, bw, bh, self.result.score / cfg.success_score,
                    hud.GREEN if ok else hud.YELLOW)
        elif self.mode == 1:
            hud.label(p, "Stavěj podle vrstevnic · Enter = vyhodnotit", cw / 2, ch - size * 0.35, small,
                      anchor="bc")

    def key(self, name: str, shift: bool = False) -> bool:
        cfg = self.host.settings.challenge
        if name in ("left", "right"):
            self._select(1 if name == "right" else -1)
        elif name == "m":
            self.mode = (self.mode + 1) % len(MODES)
            self.host.notify(f"Režim: {MODES[self.mode]}")
        elif name in ("enter", "return"):
            self.reveal_until = time.monotonic() + 8.0
            self._next_eval = 0.0
        elif name in ("up", "down"):
            cfg.target_relief = float(np.clip(cfg.target_relief + (0.01 if name == "up" else -0.01), 0.02, 0.25))
            self.target_grid = None
            self.host.notify(f"Převýšení předlohy {cfg.target_relief * 100:.0f} cm")
        elif name == "u":
            hm = self.host.heightmap
            if hm is not None:
                path = save_capture(hm.heights)
                self.targets = all_targets()
                self.index = next((i for i, t in enumerate(self.targets) if t.path == path), self.index)
                self.target_grid = None
                self.host.notify(f"Uloženo jako předloha: {path.name}")
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Předloha {self.index + 1}/{len(self.targets)}: {self.target.name}", f"Režim: {MODES[self.mode]}"]
        if self.result:
            r = self.result
            lines += [f"Shoda {r.score:.0f} % (nejlépe {self.best:.0f} %)",
                      f"V toleranci {r.within:.0f} % hlavní části · RMSE {r.rmse * 100:.1f} cm",
                      f"Čas {time.monotonic() - self.started:.0f} s"]
        return lines
