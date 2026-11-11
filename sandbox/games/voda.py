"""Voda: déšť, stékání ze svahů, jezera a povodně.

Simulace mělké vody metodou „virtuálních trubek“ (Mei et al. 2007): mezi
sousedními buňkami teče voda úměrně rozdílu hladin. Déšť padá tam, kam žák
podrží ruku nad pískem (Kinect ruku vidí), nebo všude po stisku R.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Game


class WaterSim:
    def __init__(self, shape: tuple[int, int], cell: float) -> None:
        self.cell = cell
        self.depth = np.zeros(shape, np.float32)
        self.fl = np.zeros(shape, np.float32)  # odtok doleva [m³/s]
        self.fr = np.zeros(shape, np.float32)
        self.ft = np.zeros(shape, np.float32)  # nahoru (−y)
        self.fb = np.zeros(shape, np.float32)  # dolů (+y)

    @property
    def volume(self) -> float:
        """Objem vody [m³]."""
        return float(self.depth.sum()) * self.cell * self.cell

    def clear(self) -> None:
        for a in (self.depth, self.fl, self.fr, self.ft, self.fb):
            a[:] = 0

    def advance(self, terrain: np.ndarray, frame_dt: float, gravity: float = 9.81, drain: bool = False) -> None:
        """Posune simulaci o frame_dt; rozdělí ho na stabilní podkroky (CFL)."""
        max_depth = float(self.depth.max())
        wave_speed = math.sqrt(gravity * max(max_depth, 0.002))
        n = int(min(12, max(1, math.ceil(frame_dt / (0.4 * self.cell / wave_speed)))))
        for _ in range(n):
            self.step(terrain, frame_dt / n, gravity, drain)

    def step(self, terrain: np.ndarray, dt: float, gravity: float = 9.81, drain: bool = False,
             damping: float = 0.995) -> None:
        d = self.depth
        H = terrain + d
        k = dt * gravity * self.cell
        fl, fr, ft, fb = self.fl, self.fr, self.ft, self.fb
        fr[:, :-1] = np.maximum(0, fr[:, :-1] * damping + k * (H[:, :-1] - H[:, 1:]))
        fl[:, 1:] = np.maximum(0, fl[:, 1:] * damping + k * (H[:, 1:] - H[:, :-1]))
        fb[:-1, :] = np.maximum(0, fb[:-1, :] * damping + k * (H[:-1, :] - H[1:, :]))
        ft[1:, :] = np.maximum(0, ft[1:, :] * damping + k * (H[1:, :] - H[:-1, :]))
        fr[:, -1] = fl[:, 0] = fb[-1, :] = ft[0, :] = 0  # stěny bedny

        area = self.cell * self.cell
        out = fl + fr + ft + fb
        scale = np.minimum(1.0, d * area / np.maximum(out * dt, 1e-12))
        for f in (fl, fr, ft, fb):
            f *= scale
        out *= scale

        inflow = np.zeros_like(d)
        inflow[:, 1:] += fr[:, :-1]
        inflow[:, :-1] += fl[:, 1:]
        inflow[1:, :] += fb[:-1, :]
        inflow[:-1, :] += ft[1:, :]
        d += dt * (inflow - out) / area
        np.maximum(d, 0, out=d)
        if drain:
            d[0, :] = d[-1, :] = 0
            d[:, 0] = d[:, -1] = 0


FRAG = """
uniform sampler2D u_water;
void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    vec3 c = land_color((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4));
    c *= mix(0.6, 1.1, hillshade(uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.45, cl.x * 0.6);
    c = mix(c, c * 0.25, cl.y * 0.8);

    float w = texture(u_water, uv).r;
    if (w > 0.0003) {
        float a = smoothstep(0.0003, 0.005, w);
        vec3 wc = mix(vec3(0.40, 0.75, 1.0), vec3(0.02, 0.16, 0.55), clamp(w / 0.04, 0.0, 1.0));
        float s = sin(uv.x * u_box.x * 70.0 + u_time * 2.3) * sin(uv.y * u_box.y * 63.0 - u_time * 1.9);
        wc += 0.06 * s * (1.0 - clamp(w / 0.04, 0.0, 1.0));
        c = mix(c, wc, a * 0.92);
    }
    f_color = vec4(c, 1.0);
}
"""


class WaterGame(Game):
    id = "voda"
    title = "Voda a povodně"
    description = "Podrž ruku nad pískem a začne pršet. Voda stéká do údolí a tvoří jezera."
    order = 50
    keys = [
        ("ruka", "déšť pod rukou"),
        ("R", "déšť všude zap/vyp"),
        ("P", "povodeň (+1 cm vody)"),
        ("X", "vysušit"),
        ("D", "odtok na okrajích"),
        ("↑ / ↓", "rychlost toku"),
    ]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.sim: WaterSim | None = None
        self.tex = None
        self.raining = False
        self.drain = False

    def stop(self) -> None:
        self.prog.release()
        if self.tex is not None:
            self.tex.release()
            self.tex = None

    def update(self, hm, dt: float) -> None:
        if self.sim is None or self.sim.depth.shape != hm.shape:
            self.sim = WaterSim(hm.shape, hm.cell_size[0])
            if self.tex is not None:
                self.tex.release()
            self.tex = self.host.float_texture((hm.shape[1], hm.shape[0]))
        cfg = self.host.settings.water
        dt = min(dt, 1 / 20)
        if self.raining:
            self.sim.depth += cfg.rain_rate * dt
        if hm.hands is not None and cfg.hand_rain_rate > 0:
            self.sim.depth[hm.hands] += cfg.hand_rain_rate * dt
        self.sim.advance(hm.heights, dt, 9.81 * cfg.flow_speed, self.drain)
        if cfg.evaporation > 0:
            np.maximum(self.sim.depth - cfg.evaporation * dt, 0, out=self.sim.depth)
        assert self.tex is not None
        self.tex.write(self.sim.depth.tobytes())

    def render(self) -> None:
        if self.tex is None:
            return
        self.host.draw_fullscreen(self.prog, u_water=(self.tex, 1))

    def key(self, name: str, shift: bool = False) -> bool:
        cfg = self.host.settings.water
        if name == "r":
            self.raining = not self.raining
            self.host.notify("Déšť " + ("zapnut" if self.raining else "vypnut"))
        elif name == "p" and self.sim is not None:
            self.sim.depth += 0.01
        elif name in ("x", "backspace") and self.sim is not None:
            self.sim.clear()
        elif name == "d":
            self.drain = not self.drain
            self.host.notify("Odtok na okrajích " + ("zapnut" if self.drain else "vypnut"))
        elif name in ("up", "down"):
            cfg.flow_speed = float(np.clip(cfg.flow_speed * (1.5 if name == "up" else 1 / 1.5), 0.05, 5))
            self.host.notify(f"Rychlost toku ×{cfg.flow_speed:.2f}")
        else:
            return False
        return True

    def status(self) -> list[str]:
        if self.sim is None:
            return []
        return [f"Voda: {self.sim.volume * 1000:.2f} l", "Déšť: " + ("ano" if self.raining else "jen pod rukou"),
                "Odtok: " + ("ano" if self.drain else "ne")]
