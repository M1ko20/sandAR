"""Voda: déšť, stékání ze svahů, jezera a povodně.

Simulace mělké vody metodou „virtuálních trubek“ (Mei et al. 2007): mezi
sousedními buňkami teče voda úměrně rozdílu hladin. Z každé dlaně zvednuté nad písek
(aspoň ~25 cm) prší – na hřbetu ruky se promítne mráček, kolem dopadají kapky. Ruka, která
modeluje v písku, neprší, aby šlo kopat koryta. Klávesou R prší všude.
"""

from __future__ import annotations

import moderngl
import numpy as np

from ..render.fx import RainFx
from ..sim.lava import blur3
from ..sim.water import WaterSim
from .base import Game

__all__ = ["WaterGame", "WaterSim"]


FRAG = """
uniform sampler2D u_water;
uniform sampler2D u_vel;     // rychlost vody [m/s] (rg)
void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    vec3 c = land_color((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4));
    c *= mix(0.6, 1.1, hillshade(uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.45, cl.x * 0.6);
    c = mix(c, c * 0.25, cl.y * 0.8);

    float w = max(sample_bicubic(u_water, uv), 0.0);
    vec2 v = texture(u_vel, uv).rg;
    // pěna tam, kde voda rychle stéká mělkým korytem (bystřina)
    float foam = clamp((length(v) - 0.10) / 0.25, 0.0, 1.0) * (1.0 - smoothstep(0.002, 0.010, w)) * 0.55;
    vec4 r = water_shade(c, uv, w, v, foam);
    f_color = vec4(r.rgb, 1.0);
}
"""


class WaterGame(Game):
    id = "voda"
    title = "Voda a povodně"
    description = "Zvedni ruku vysoko nad písek a začne z ní pršet. Voda stéká do údolí a tvoří jezera."
    order = 50
    category = "simulace"
    intro = "Zvedni ruku vysoko nad písek (aspoň 25 cm) – začne z ní pršet. Kam poteče voda?"
    keys = [
        ("ruka", "zvednutá dlaň = déšť (při modelování neprší)"),
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
        self.vel_tex = None
        self.raining = False
        self.drain = False
        self.fx = RainFx()
        self.clouds: list[tuple[float, float, float]] = []
        self._cloud_k: dict[int, float] = {}
        self._xy = None
        self.t = 0.0
        self.last_rain = -99.0

    def stop(self) -> None:
        self.prog.release()
        for tex in (self.tex, self.vel_tex):
            if tex is not None:
                tex.release()
        self.tex = self.vel_tex = None

    def update(self, hm, dt: float) -> None:
        if self.sim is None or self.sim.depth.shape != hm.shape:
            self.sim = WaterSim(hm.shape, hm.cell_size[0])
            for tex in (self.tex, self.vel_tex):
                if tex is not None:
                    tex.release()
            self.tex = self.host.float_texture((hm.shape[1], hm.shape[0]))
            self.vel_tex = self.host.ctx.texture((hm.shape[1], hm.shape[0]), 2, dtype="f4")
            self.vel_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.vel_tex.repeat_x = self.vel_tex.repeat_y = False
            self._xy = hm.cell_centers()
        cfg = self.host.settings.water
        dt = min(dt, 1 / 20)
        self.t += dt
        if self.raining:
            self.sim.depth += cfg.rain_rate * dt
            self.last_rain = self.t
        self._hand_rain(hm, dt, cfg.hand_rain_rate)
        self.sim.advance(hm.heights, dt, 9.81 * cfg.flow_speed, self.drain)
        if cfg.evaporation > 0:
            np.maximum(self.sim.depth - cfg.evaporation * dt, 0, out=self.sim.depth)
        assert self.tex is not None and self.vel_tex is not None
        self.tex.write(self.sim.depth.tobytes())
        vx, vy = self.sim.velocity()
        self.vel_tex.write(np.dstack([blur3(vx), blur3(vy)]).astype(np.float32).tobytes())

    def _hand_rain(self, hm, dt: float, rate: float) -> None:
        """Ze zvednuté dlaně prší (Gaussův mráček ~ 5 cm). Mráček se plynule rozsvítí / zhasne."""
        assert self._xy is not None and self.sim is not None
        X, Y = self._xy
        alive = set()
        sources = []
        for hand in hm.hand_list:
            raining = hand.raised
            k = self._cloud_k.get(hand.id, 0.0)
            k = min(1.0, k + dt * 4) if raining else max(0.0, k - dt * 4)
            self._cloud_k[hand.id] = k
            alive.add(hand.id)
            if k > 0:
                sources.append((hand.x, hand.y, k))
        for hid in list(self._cloud_k):
            if hid not in alive:
                del self._cloud_k[hid]
        if rate > 0:
            for x, y, k in sources:
                r2 = (X - x) ** 2 + (Y - y) ** 2
                near = r2 < 0.15 ** 2
                self.sim.depth[near] += (rate * dt * k) * np.exp(-r2[near] / (2 * 0.05 ** 2))
        if sources:
            self.last_rain = self.t
        self.clouds = sources
        self.fx.update(dt, sources)

    def render(self) -> None:
        if self.tex is None:
            return
        self.host.draw_fullscreen(self.prog, u_water=(self.tex, 1), u_vel=(self.vel_tex, 2))
        self.fx.draw(self.host, self.clouds)
        if self.t - self.last_rain > 8.0 and self.sim is not None and self.sim.volume < 0.002:
            p = self.host.painter
            cw, ch = self.host.canvas_size
            size = max(18, ch // 24)
            p.text("Zvedni ruku vysoko nad písek – začne z ní pršet", cw / 2, ch - size * 0.7, size=size, anchor="bc",
                   bold=True, color=(1, 1, 1, 1), outline=max(2, size // 10), background=(0, 0.05, 0.15, 0.55),
                   padding=size * 0.3)

    def reset(self) -> None:
        if self.sim is not None:
            self.sim.clear()
        self.raining = False
        self.fx.clear()

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
