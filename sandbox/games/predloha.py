"""Postav podle předlohy – úkol s automatickým vyhodnocením shody.

Režimy:
* Nápověda – červená = přidej písek, modrá = uber písek, černé čáry = vrstevnice předlohy.
* Mapa     – na písku jsou jen vrstevnice předlohy (jako mapa). Žák staví podle
             mapy a klávesou Enter si nechá terén vyhodnotit.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .base import Game
from .targets import Score, Target, all_targets, evaluate, save_capture

log = logging.getLogger(__name__)

FRAG = """
uniform sampler2D u_target;
uniform int u_mode;        // 0 nápověda, 1 mapa, 2 mapa + odhalení chyb
uniform float u_tol;
uniform float u_offset;
uniform float u_success;   // 0..1 zelené orámování při úspěchu

void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    float t = sample_bicubic(u_target, uv) + u_offset;
    float d = h - t;
    vec3 c;
    if (u_mode == 0) {
        float lum = dot(elevation_color(h), vec3(0.3, 0.55, 0.15));
        c = vec3(lum) * 0.55 + 0.2;
        c *= mix(0.75, 1.1, hillshade(uv, 2.0));
    } else {
        c = vec3(0.97, 0.95, 0.87) * mix(0.85, 1.03, hillshade(uv, 2.0));
    }
    if (u_mode != 1) {
        float k = clamp((abs(d) - 0.5 * u_tol) / (2.0 * u_tol), 0.0, 1.0);
        vec3 tint = d < 0.0 ? vec3(1.0, 0.12, 0.08) : vec3(0.10, 0.35, 1.0);
        c = mix(c, tint, k * 0.85);
        float good = 1.0 - smoothstep(0.5 * u_tol, u_tol, abs(d));
        c = mix(c, vec3(0.25, 0.85, 0.35), good * 0.18);
    }
    // vrstevnice předlohy
    float line = contour_line(t, u_contour_interval, 2.2);
    vec3 ink = u_mode == 0 ? vec3(0.02) : vec3(0.55, 0.30, 0.10);
    c = mix(c, ink, line);
    // orámování při úspěchu
    float edge = min(min(uv.x, 1.0 - uv.x) * u_box.x, min(uv.y, 1.0 - uv.y) * u_box.y);
    c = mix(c, vec3(0.2, 1.0, 0.3), u_success * (1.0 - smoothstep(0.015, 0.03, edge)));
    f_color = vec4(c, 1.0);
}
"""

MODES = ["Nápověda", "Mapa"]


class ChallengeGame(Game):
    id = "predloha"
    title = "Postav podle předlohy"
    description = "Postav terén podle předlohy, program ukazuje, kde přidat a kde ubrat písek."
    order = 40
    category = "soutěž"
    intro = "Červená = přidej, modrá = uber. Postav krajinu podle předlohy!"
    keys = [
        ("← / →", "jiná předloha"),
        ("M", "režim nápověda / mapa"),
        ("Enter", "vyhodnotit"),
        ("↑ / ↓", "převýšení předlohy"),
        ("U", "uložit současný písek jako předlohu"),
    ]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.targets: list[Target] = all_targets()
        self.index = getattr(self, "index", 0) % len(self.targets)
        self.mode = getattr(self, "mode", 0)
        self.target_grid: np.ndarray | None = None
        self.target_tex = None
        self.result: Score | None = None
        self.best = 0.0
        self.reveal_until = 0.0
        self.success_since: float | None = None
        self._next_eval = 0.0
        self.started = time.monotonic()

    def stop(self) -> None:
        self.prog.release()
        if self.target_tex is not None:
            self.target_tex.release()
            self.target_tex = None

    @property
    def target(self) -> Target:
        return self.targets[self.index]

    def _build_target(self, shape: tuple[int, int]) -> None:
        cfg = self.host.settings.challenge
        try:
            grid = self.target.grid(shape, self.host.box_size, cfg.target_relief)
        except Exception as exc:
            log.exception("Předlohu %s nelze načíst", self.target.name)
            self.host.notify(f"Předlohu nelze načíst: {exc}")
            grid = np.zeros(shape, np.float32)
        self.target_grid = grid
        if self.target_tex is not None:
            self.target_tex.release()
        self.target_tex = self.host.float_texture((shape[1], shape[0]), grid)
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
        self.result, _ = evaluate(hm.heights, self.target_grid, cfg.tolerance, cfg.border)
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
        self.host.draw_fullscreen(self.prog, u_target=(self.target_tex, 1), u_mode=mode, u_tol=cfg.tolerance,
                                  u_offset=self.result.offset if self.result else 0.0, u_success=float(success))
        self._labels()

    def _labels(self) -> None:
        p = self.host.painter
        cw, ch = self.host.canvas_size
        size = max(16, ch // 26)
        p.text(f"Předloha: {self.target.name}", cw / 2, size * 0.6, size=size, anchor="tc", bold=True,
               color=(1, 1, 1, 1), outline=3)
        if self.result is None:
            return
        if self._show_score():
            ok = self.success_since is not None
            text = f"Výborně! Shoda {self.result.score:.0f} %" if ok else f"Shoda {self.result.score:.0f} %"
            color = (0.3, 1.0, 0.4, 1) if ok else (1, 1, 1, 1)
            p.text(text, cw / 2, ch - size * 0.6, size=int(size * (1.4 if ok else 1.0)), anchor="bc", bold=True,
                   color=color, outline=3)
        elif self.mode == 1:
            p.text("Stavěj podle vrstevnic · Enter = vyhodnotit", cw / 2, ch - size * 0.6, size=int(size * 0.8),
                   anchor="bc", color=(0.35, 0.2, 0.08, 1), outline=2, outline_color=(0.97, 0.95, 0.87, 1))

    def key(self, name: str, shift: bool = False) -> bool:
        cfg = self.host.settings.challenge
        if name in ("left", "right"):
            self._select(1 if name == "right" else -1)
        elif name == "m":
            self.mode = (self.mode + 1) % len(MODES)
            self.host.notify(f"Režim: {MODES[self.mode]}")
        elif name == "enter":
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
                      f"V toleranci {r.within:.0f} % plochy · RMSE {r.rmse * 100:.1f} cm",
                      f"Čas {time.monotonic() - self.started:.0f} s"]
        return lines
