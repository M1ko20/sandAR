"""Živá sopka: postav z písku kopec a sleduj, jak z něj vytéká láva.

Program najde nejvyšší vrchol (nebo vrchol, nad který návštěvník podrží ruku)
a nechá ho vybuchnout. Láva teče podle skutečného sklonu písku (viskózní
simulace ``sandbox.sim.lava``), chladne z bílé přes oranžovou a tmavě červenou
až v černou krustu s žhnoucími puklinami a zastaví se.

Cyklus běží sám: klid → probouzení → erupce → chladnutí → (vyblednutí) → klid.
"""

from __future__ import annotations

import math

import numpy as np

from ..sim.lava import LavaSim, blur3, downsample
from .base import Game

FRAG = """
uniform sampler2D u_lava;
uniform sampler2D u_temp;
uniform vec2 u_vent;      // kráter v uv
uniform float u_glow;     // 0..1 záře kráteru
uniform float u_solid;    // teplota tuhnutí
uniform float u_fade;     // 1 = láva viditelná, 0 = zmizela

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x),
               mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
// vzdálenost k hraně Voronoiovy buňky (pukliny v krustě)
float crack_dist(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float d1 = 8.0, d2 = 8.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            vec2 g = vec2(x, y);
            vec2 o = g + hash22(i + g) * 0.85 - f;
            float d = dot(o, o);
            if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
        }
    return sqrt(d2) - sqrt(d1);
}

vec3 hot_color(float t) {   // t = 0 (tuhnutí) .. 1 (čerstvá láva)
    vec3 c = mix(vec3(0.35, 0.02, 0.0), vec3(0.95, 0.10, 0.0), smoothstep(0.0, 0.3, t));
    c = mix(c, vec3(1.0, 0.45, 0.0), smoothstep(0.25, 0.55, t));
    c = mix(c, vec3(1.0, 0.85, 0.15), smoothstep(0.55, 0.8, t));
    c = mix(c, vec3(1.0, 1.0, 0.8), smoothstep(0.82, 1.0, t));
    return c;
}

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;   // metry
    float h = height_at(uv);

    // --- sopečná krajina: tmavá, ale s čitelným reliéfem ---
    float e = clamp((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4), 0.0, 1.0);
    vec3 c = mix(vec3(0.10, 0.24, 0.16), vec3(0.36, 0.25, 0.18), smoothstep(0.15, 0.5, e));
    c = mix(c, vec3(0.50, 0.45, 0.43), smoothstep(0.5, 0.95, e));
    c *= mix(0.45, 1.45, hillshade(uv, 3.0));
    vec2 cl = contours(h);
    c = mix(c, vec3(0.85, 0.30, 0.10), cl.y * 0.55 + cl.x * 0.25);

    // --- záře kráteru ---
    float rv = length((uv - u_vent) * u_box);
    float glow = u_glow * exp(-rv * rv / (0.06 * 0.06));
    c += vec3(1.0, 0.28, 0.02) * glow * 0.9;
    c = mix(c, vec3(1.0, 0.9, 0.5), u_glow * (1.0 - smoothstep(0.012, 0.025, rv)));

    // --- láva ---
    float th = sample_bicubic(u_lava, uv);
    float a = smoothstep(0.0004, 0.0016, th) * u_fade;
    if (a > 0.001) {
        float T = texture(u_temp, uv).r;
        float tn = clamp((T - u_solid) / (1.0 - u_solid), 0.0, 1.0);
        // tekoucí povrch: šum unášený po svahu
        vec2 g = height_grad(uv);
        vec2 flow = -g / max(length(g), 0.05);
        vec2 perp = vec2(-flow.y, flow.x);
        vec2 q = vec2(dot(pm, flow), dot(pm, perp));
        float n = vnoise(vec2(q.x * 30.0 - u_time * 1.2 * tn, q.y * 70.0)) * 0.65
                + vnoise(pm * 55.0 + u_time * 0.4) * 0.35;
        // tenké okraje a čela proudů jsou chladnější (tmavší)
        float tv = tn * mix(0.45, 1.0, smoothstep(0.0008, 0.008, th));
        vec3 hot = hot_color(tv * (0.75 + 0.45 * n));
        // krusta: desky (Voronoi) s žhnoucími puklinami; čím chladnější, tím víc krusty
        float cd = crack_dist(pm * 38.0 + vec2(3.1, 7.7));
        float crust = smoothstep(0.62, 0.15, tn);
        float plate = smoothstep(0.04, 0.16, cd);
        vec3 rock = vec3(0.07, 0.055, 0.05) * mix(0.7, 1.3, hillshade(uv, 3.0)) * (0.8 + 0.4 * n);
        float heat = smoothstep(0.02, u_solid + 0.12, T);          // zbytkové teplo v puklinách
        vec3 crack = hot_color(0.25 * heat) * heat * (0.75 + 0.25 * sin(u_time * 3.0 + n * 6.0));
        vec3 lc = mix(hot, mix(crack, rock, plate), crust);
        if (T <= u_solid) lc = mix(crack, rock, plate);
        c = mix(c, lc, a);
        // horká láva svítí do okolí i přes okraj
        c += vec3(1.0, 0.3, 0.0) * 0.25 * a * tn;
    }
    f_color = vec4(c, 1.0);
}
"""

# fáze cyklu: (název, délka [s])
CALM, RUMBLE, ERUPT, COOL, FADE = "klid", "probouzení", "erupce", "chladnutí", "vyhasínání"
DURATION = {CALM: 14.0, RUMBLE: 4.0, ERUPT: 15.0, COOL: 18.0, FADE: 4.0}
NEXT = {CALM: RUMBLE, RUMBLE: ERUPT, ERUPT: COOL, COOL: FADE, FADE: CALM}
FIRST_CALM = 3.0  # po spuštění/resetu se sopka probudí brzy (attract mode, snapshot)
DS = 2  # zmenšení mřížky pro simulaci lávy


def find_vent(heights: np.ndarray, hands: np.ndarray | None = None, near: tuple[float, float] | None = None,
              radius: float = 0.0, border: int = 3) -> tuple[int, int, float]:
    """Kráter = nejvyšší místo uhlazeného terénu (iy, ix, výška nad mediánem).

    near=(iy, ix) a radius [buňky]: hledá jen v kruhu kolem bodu (vrchol pod rukou).
    Buňky s rukou se ignorují (Kinect tam písek nevidí).
    """
    s = blur3(blur3(heights))
    score = s.copy()
    if hands is not None and hands.shape == s.shape:
        score[hands] = -np.inf
    if border > 0:
        score[:border, :] = score[-border:, :] = -np.inf
        score[:, :border] = score[:, -border:] = -np.inf
    if near is not None:
        gh, gw = s.shape
        y, x = np.ogrid[:gh, :gw]
        score[(y - near[0]) ** 2 + (x - near[1]) ** 2 > radius**2] = -np.inf
    idx = int(np.argmax(score))
    if not np.isfinite(score.flat[idx]):
        return s.shape[0] // 2, s.shape[1] // 2, 0.0
    iy, ix = divmod(idx, s.shape[1])
    return iy, ix, float(s[iy, ix] - np.median(s))


class Particles:
    """Jednoduché částice (kouř, žhavé úlomky) v metrech bedny."""

    def __init__(self) -> None:
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.age = np.zeros(0, np.float32)
        self.life = np.zeros(0, np.float32)
        self.kind = np.zeros(0, np.int8)  # 0 kouř, 1 úlomek

    def __len__(self) -> int:
        return len(self.age)

    def clear(self) -> None:
        self.__init__()  # type: ignore[misc]

    def spawn(self, pos, vel, life, kind: int) -> None:
        n = len(life)
        if n == 0:
            return
        self.pos = np.vstack([self.pos, np.asarray(pos, np.float32).reshape(n, 2)])
        self.vel = np.vstack([self.vel, np.asarray(vel, np.float32).reshape(n, 2)])
        self.age = np.concatenate([self.age, np.zeros(n, np.float32)])
        self.life = np.concatenate([self.life, np.asarray(life, np.float32)])
        self.kind = np.concatenate([self.kind, np.full(n, kind, np.int8)])

    def update(self, dt: float, wind: np.ndarray) -> None:
        if not len(self):
            return
        smoke = self.kind == 0
        self.vel[smoke] += (wind - self.vel[smoke]) * min(1.0, 0.8 * dt)
        self.vel[~smoke] *= max(0.0, 1.0 - 1.8 * dt)
        self.pos += self.vel * dt
        self.age += dt
        keep = self.age < self.life
        if not keep.all():
            self.pos, self.vel, self.age = self.pos[keep], self.vel[keep], self.age[keep]
            self.life, self.kind = self.life[keep], self.kind[keep]


def disc_triangles(centers: np.ndarray, radii: np.ndarray, colors: np.ndarray, segments: int = 12):
    """Mnoho vyplněných kruhů najednou -> (vrcholy, barvy) pro Painter.triangles."""
    n = len(centers)
    ang = np.linspace(0, 2 * math.pi, segments + 1)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (S+1, 2)
    pts = centers[:, None, :] + radii[:, None, None] * ring[None]  # (n, S+1, 2)
    c = np.repeat(centers[:, None, :], segments, axis=1)
    tris = np.stack([c, pts[:, :-1], pts[:, 1:]], axis=2).reshape(-1, 2)
    cols = np.repeat(colors, segments * 3, axis=0)
    # střed kruhu plnou barvou, okraj průhledný -> měkké obláčky
    cols = cols.reshape(n, segments, 3, 4).copy()
    cols[:, :, 1:, 3] *= 0.15
    return tris, cols.reshape(-1, 4)


class VolcanoGame(Game):
    id = "sopka"
    title = "Živá sopka"
    description = "Postav sopku z písku. Z nejvyššího vrcholu vyteče žhavá láva a stéká po svazích dolů."
    order = 70
    category = "zážitek"
    intro = "Postav vysoký kopec – sopku – a podrž nad ním ruku: probudí se a vyteče z ní láva!"
    keys = [
        ("ruka", "ruka nad vrcholem probudí sopku"),
        ("E", "erupce hned"),
        ("C", "automatický cyklus zap/vyp"),
        ("X", "smazat lávu"),
        ("↑ / ↓", "síla erupce"),
    ]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.sim: LavaSim | None = None
        self.tex_lava = None
        self.tex_temp = None
        self.rough: np.ndarray | None = None
        self.particles = Particles()
        self.rng = np.random.default_rng()
        self.auto = getattr(self, "auto", True)
        self.power = getattr(self, "power", 1.0)
        self.vent = (0, 0)  # (iy, ix) v mřížce simulace
        self.vent_m = (0.5, 0.4)  # kráter v metrech
        self.vent_height = 0.0
        self.hand_time = 0.0
        self.eruptions = 0
        self.fade = 1.0
        self._set_phase(CALM, FIRST_CALM)

    def stop(self) -> None:
        self.prog.release()
        for t in (self.tex_lava, self.tex_temp):
            if t is not None:
                t.release()
        self.tex_lava = self.tex_temp = None

    def reset(self) -> None:
        if self.sim is not None:
            self.sim.clear()
        self.particles.clear()
        self.fade = 1.0
        self.hand_time = 0.0
        self._set_phase(CALM, FIRST_CALM)

    # --- fáze ------------------------------------------------------------------------
    def _set_phase(self, phase: str, duration: float | None = None) -> None:
        self.phase = phase
        self.phase_t = 0.0
        self.phase_len = DURATION[phase] if duration is None else duration
        if phase == RUMBLE:
            self.wind = self.rng.normal(0, 0.03, 2).astype(np.float32)
        if phase == FADE:
            self.fade = 1.0
        if phase == ERUPT:
            self.eruptions += 1

    def _wake(self, hm, near_m: tuple[float, float] | None = None) -> None:
        """Probudí sopku: zvolí kráter a přejde do fáze probouzení."""
        self._choose_vent(hm, near_m)
        self._set_phase(RUMBLE)

    def _choose_vent(self, hm, near_m: tuple[float, float] | None = None) -> None:
        gh, gw = hm.shape
        cw, ch = hm.cell_size
        near = None
        radius = 0.0
        if near_m is not None:
            near = (near_m[1] / ch, near_m[0] / cw)
            radius = 0.25 / cw
        iy, ix, rel = find_vent(hm.heights, hm.hands, near, radius, border=max(2, gw // 40))
        if near is not None and rel < 0.015:  # pod rukou není kopec -> nejvyšší vrchol celkově
            iy, ix, rel = find_vent(hm.heights, hm.hands, border=max(2, gw // 40))
        self.vent_height = rel
        self.vent_m = ((ix + 0.5) * cw, (iy + 0.5) * ch)
        self.vent = (iy / DS, ix / DS)

    # --- logika ----------------------------------------------------------------------
    def update(self, hm, dt: float) -> None:
        dt = min(dt, 1 / 20)
        shape = (hm.shape[0] // DS, hm.shape[1] // DS)
        if self.sim is None or self.sim.shape != shape:
            self.sim = LavaSim(shape, hm.cell_size[0] * DS)
            rng = np.random.default_rng(7)
            # drobná nerovnost terénu: láva se větví do jazyků místo dokonalých kruhů
            self.rough = blur3(rng.normal(0, 1, shape).astype(np.float32)) * 0.004
            for t in (self.tex_lava, self.tex_temp):
                if t is not None:
                    t.release()
            self.tex_lava = self.host.float_texture((shape[1], shape[0]))
            self.tex_temp = self.host.float_texture((shape[1], shape[0]))
            self._choose_vent(hm)
        sim = self.sim
        assert self.rough is not None

        self.phase_t += dt
        self._hand_trigger(hm, dt)
        if self.phase == CALM and self.phase_t < 0.05:
            self._choose_vent(hm)
        if self.phase_t >= self.phase_len:
            if self.phase == CALM:
                if self.auto:
                    self._wake(hm)
                else:
                    self.phase_t = 0.0
            else:
                self._set_phase(NEXT[self.phase])

        if self.phase == ERUPT:
            # pulzující výron (fontána), slábne ke konci erupce
            k = self.phase_t / self.phase_len
            pulse = 0.65 + 0.35 * math.sin(self.phase_t * 5.3) * math.sin(self.phase_t * 1.7 + 1)
            rate = 1.2e-4 * self.power * pulse * (1.0 - 0.6 * k)
            sim.erupt(self.vent[0], self.vent[1], 2.5, rate, dt)
        if self.phase == FADE:
            self.fade = max(0.0, 1.0 - self.phase_t / self.phase_len)
            if self.phase_t + dt >= self.phase_len:
                sim.clear()
                self.fade = 1.0
        if sim.thickness.any():
            terrain = downsample(hm.heights, DS)[: shape[0], : shape[1]] + self.rough
            sim.advance(terrain, dt)
        self._particles(dt)
        assert self.tex_lava is not None and self.tex_temp is not None
        self.tex_lava.write(sim.thickness.tobytes())
        self.tex_temp.write(sim.temp.tobytes())

    def _hand_trigger(self, hm, dt: float) -> None:
        """Ruka podržená nad kopcem (do 25 cm) v klidu sopku probudí."""
        if self.phase != CALM or not hm.hand_points:
            self.hand_time = 0.0
            return
        self.hand_time += dt
        if self.hand_time >= 1.0:
            self.hand_time = 0.0
            hx, hy = hm.hand_points[0]
            if self.sim is not None and self.phase == FADE:
                self.sim.clear()
            self.fade = 1.0
            self._wake(hm, (hx, hy))
            self.host.notify("Probudil(a) jsi sopku!")

    def _particles(self, dt: float) -> None:
        p = self.particles
        rng = self.rng
        vx, vy = self.vent_m
        smoke_rate = {CALM: 1.5, RUMBLE: 6.0, ERUPT: 6.0, COOL: 3.0, FADE: 1.0}[self.phase]
        n = rng.poisson(smoke_rate * dt)
        if n:
            ang = rng.uniform(0, 2 * math.pi, n)
            p.spawn(np.stack([vx + 0.01 * np.cos(ang), vy + 0.01 * np.sin(ang)], 1),
                    np.stack([0.04 * np.cos(ang), 0.04 * np.sin(ang)], 1), rng.uniform(2.5, 5.0, n), 0)
        ember_rate = {RUMBLE: 10.0 * self.phase_t / DURATION[RUMBLE], ERUPT: 60.0 * self.power}.get(self.phase, 0.0)
        n = rng.poisson(ember_rate * dt)
        if n:
            ang = rng.uniform(0, 2 * math.pi, n)
            sp = rng.uniform(0.08, 0.45, n) * (1.0 if self.phase == ERUPT else 0.4)
            p.spawn(np.stack([np.full(n, vx), np.full(n, vy)], 1), np.stack([sp * np.cos(ang), sp * np.sin(ang)], 1),
                    rng.uniform(0.4, 1.4, n), 1)
        if len(p) > 600:
            keep = slice(len(p) - 600, None)
            p.pos, p.vel, p.age, p.life, p.kind = p.pos[keep], p.vel[keep], p.age[keep], p.life[keep], p.kind[keep]
        p.update(dt, getattr(self, "wind", np.zeros(2, np.float32)))

    # --- kreslení --------------------------------------------------------------------
    def _glow(self) -> float:
        t = self.host.time
        if self.phase == CALM:
            return 0.12 + 0.06 * math.sin(t * 1.5)
        if self.phase == RUMBLE:
            k = self.phase_t / self.phase_len
            return 0.25 + 0.6 * k + 0.25 * math.sin(t * (6 + 14 * k))
        if self.phase == ERUPT:
            return 0.95 + 0.1 * math.sin(t * 9)
        if self.phase == COOL:
            return max(0.15, 0.8 * (1 - self.phase_t / self.phase_len))
        return 0.12

    def render(self) -> None:
        if self.tex_lava is None or self.sim is None:
            return
        bw, bh = self.host.box_size
        self.host.draw_fullscreen(self.prog, u_lava=(self.tex_lava, 1), u_temp=(self.tex_temp, 2),
                                  u_vent=(self.vent_m[0] / bw, self.vent_m[1] / bh), u_glow=float(self._glow()),
                                  u_solid=float(self.sim.solid), u_fade=float(self.fade))
        self._draw_particles()
        self._labels()

    def _draw_particles(self) -> None:
        p = self.particles
        if not len(p):
            return
        cw, ch = self.host.canvas_size
        bw, bh = self.host.box_size
        k = np.clip(p.age / p.life, 0, 1)
        centers = p.pos * np.array([cw / bw, ch / bh], np.float32)
        px_per_m = cw / bw
        cols = np.zeros((len(p), 4), np.float32)
        radii = np.zeros(len(p), np.float32)
        smoke = p.kind == 0
        ks = k[smoke]
        radii[smoke] = (0.012 + 0.05 * np.sqrt(ks)) * px_per_m
        grey = 0.55 - 0.25 * ks
        cols[smoke] = np.stack([grey, grey * 0.95, grey * 0.92, 0.22 * (1 - ks) * np.minimum(1, ks * 8)], 1)
        ke = k[~smoke]
        radii[~smoke] = (0.006 * (1 - 0.5 * ke)) * px_per_m
        cols[~smoke] = np.stack([np.ones_like(ke), 0.9 - 0.75 * ke, 0.35 * (1 - ke) ** 2, 1 - ke**3], 1)
        order = np.argsort(p.kind, kind="stable")  # kouř pod úlomky
        verts, colors = disc_triangles(centers[order], radii[order], cols[order])
        self.host.painter.triangles(verts, colors)

    def _labels(self) -> None:
        p = self.host.painter
        cw, ch = self.host.canvas_size
        size = max(18, ch // 22)
        small = max(16, ch // 28)
        remaining = max(0.0, self.phase_len - self.phase_t)
        top = None
        bottom = None
        color = (1.0, 0.85, 0.3, 1)
        if self.phase == CALM:
            if self.vent_height < 0.02:
                top = "Postav z písku vysokou sopku!"
            else:
                top = "Sopka spí…"
            if self.auto:
                bottom = f"Podrž ruku nad vrcholem – nebo počkej {remaining:.0f} s"
            else:
                bottom = "Podrž ruku nad vrcholem a sopka se probudí"
            color = (1, 1, 1, 1)
        elif self.phase == RUMBLE:
            top = "Sopka se probouzí…"
            color = (1.0, 0.55, 0.15, 1)
        elif self.phase == ERUPT:
            if self.phase_t < 3.0:
                top = "ERUPCE!"
                color = (1.0, 0.35, 0.1, 1) if int(self.phase_t * 4) % 2 else (1.0, 0.9, 0.3, 1)
                size = int(size * 1.5)
            else:
                top = "Láva teče z kopce dolů"
            bottom = "Rukama můžeš lávě postavit hráz nebo vykopat koryto"
        elif self.phase == COOL:
            top = "Láva chladne a tuhne na černý kámen"
            color = (1.0, 0.6, 0.3, 1)
        if top:
            p.text(top, cw / 2, size * 0.5, size=size, anchor="tc", bold=True, color=color, outline=4)
        if bottom:
            p.text(bottom, cw / 2, ch - small * 0.5, size=small, anchor="bc", color=(1, 1, 1, 1), outline=3)

    # --- ovládání ------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        hm = self.host.heightmap
        if name == "e":
            if hm is not None:
                self._choose_vent(hm)
            self.fade = 1.0
            self._set_phase(ERUPT)
            self.host.notify("Erupce!")
        elif name == "c":
            self.auto = not self.auto
            self.host.notify("Automatický cyklus " + ("zapnut" if self.auto else "vypnut"))
        elif name in ("x", "backspace"):
            self.reset()
        elif name in ("up", "down"):
            self.power = float(np.clip(self.power * (1.3 if name == "up" else 1 / 1.3), 0.3, 3.0))
            self.host.notify(f"Síla erupce ×{self.power:.2f}")
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Fáze: {self.phase} ({max(0.0, self.phase_len - self.phase_t):.0f} s)",
                 f"Kráter: x={self.vent_m[0]:.2f} m, y={self.vent_m[1]:.2f} m, převýšení {self.vent_height * 100:.1f} cm",
                 f"Automatický cyklus: {'ano' if self.auto else 'ne'} · síla ×{self.power:.2f} · erupcí {self.eruptions}"]
        if self.sim is not None:
            cell = self.sim.cell
            lines.append(f"Láva: {self.sim.volume * 1000:.2f} l, tekutá plocha "
                         f"{self.sim.molten.sum() * cell * cell * 1e4:.0f} cm²")
        return lines
