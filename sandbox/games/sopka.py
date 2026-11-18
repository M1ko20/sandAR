"""Živá sopka: postav z písku kopec a sleduj, jak z něj vytéká láva.

Program najde nejvyšší vrchol a označí ho jako sopku (kruh „SOPKA“ přímo na písku),
15 s odpočítává a pak sopka vybuchne. Místo sopky se na začátku kola zamkne (jakmile je
v bedně opravdový kopec) a do konce kola se nemění, ani když lidé hrabou a staví hráze. Láva teče podle skutečného sklonu písku (viskózní
simulace ``sandbox.sim.lava``), chladne z bílé přes oranžovou a tmavě červenou
až v černou krustu s žhnoucími puklinami a zastaví se.

Cyklus běží sám: klid → probouzení → erupce → chladnutí → (vyblednutí) → klid.

Pod sopkou stojí vesnice (v údolích, kam by láva opravdu tekla). Dům, ke kterému láva
doteče, shoří. Návštěvníci mají během probouzení a erupce čas postavit hráze nebo vykopat
koryto, které lávu odvede – na konci se ukáže, kolik vesnic zachránili.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..render import hud
from ..sim.ecosystem import House
from ..sim.lava import LavaSim, blur3, downsample
from .base import Game
from .mesto import draw_house

FRAG = """
uniform sampler2D u_lava;
uniform sampler2D u_temp;
uniform sampler2D u_heat;  // rozmazané teplo lávy -> záře do okolí
uniform vec2 u_vent;      // kráter v uv
uniform float u_glow;     // 0..1 záře kráteru
uniform float u_solid;    // teplota tuhnutí
uniform float u_fade;     // 1 = láva viditelná, 0 = zmizela
uniform float u_wake;     // 0..1 probouzení sopky (odpočet -> erupce)
uniform float u_pulse;    // 0..1 otřes (pulz světla)
uniform float u_shock;    // [s] od začátku erupce (< 0 = žádná tlaková vlna)
uniform float u_erupt;    // 0..1 síla fontány v kráteru

float sopka_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 sopka_hash2(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
float sopka_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(sopka_hash(i), sopka_hash(i + vec2(1, 0)), u.x),
               mix(sopka_hash(i + vec2(0, 1)), sopka_hash(i + vec2(1, 1)), u.x), u.y);
}
// Voronoi desky krusty: x = vzdálenost k hraně desky, y = náhodné číslo desky
vec2 sopka_plates(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float d1 = 8.0, d2 = 8.0;
    float id = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            vec2 g = vec2(x, y);
            vec2 o = g + sopka_hash2(i + g) * 0.85 - f;
            float d = dot(o, o);
            if (d < d1) { d2 = d1; d1 = d; id = sopka_hash(i + g + 0.37); } else if (d < d2) { d2 = d; }
        }
    return vec2(sqrt(d2) - sqrt(d1), id);
}

// žhavá láva: t = 0 (tuhne, tmavě rudá) .. 1 (čerstvá, žlutooranžová) – nikdy ne bílá
vec3 sopka_hot(float t) {
    vec3 c = mix(vec3(0.28, 0.02, 0.0), vec3(0.78, 0.06, 0.0), smoothstep(0.0, 0.3, t));
    c = mix(c, vec3(0.95, 0.24, 0.0), smoothstep(0.25, 0.55, t));
    c = mix(c, vec3(0.95, 0.45, 0.03), smoothstep(0.5, 0.8, t));
    c = mix(c, vec3(0.95, 0.66, 0.16), smoothstep(0.85, 1.05, t));
    return c;
}

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;   // metry
    float h = height_at(uv);
    float shade = hillshade(uv, 3.0);

    // --- sopečná krajina: olivová údolí, hnědé svahy, popelavé vrcholy; reliéf čitelný ---
    float e = clamp((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4), 0.0, 1.0);
    vec3 c = mix(vec3(0.17, 0.33, 0.15), vec3(0.46, 0.30, 0.18), smoothstep(0.15, 0.5, e));
    c = mix(c, vec3(0.58, 0.52, 0.50), smoothstep(0.55, 0.95, e));
    c *= mix(0.38, 1.45, shade);
    vec2 cl = contours(h);
    c = mix(c, c * 0.55, cl.x * 0.5);
    c = mix(c, vec3(0.95, 0.42, 0.12), cl.y * 0.45);
    // sopka se probouzí: krajina potemní a zčervená, ať vynikne žár
    c *= 1.0 - 0.28 * u_wake;
    c = mix(c, c * vec3(1.25, 0.75, 0.6), 0.5 * u_wake);

    vec2 dv = (uv - u_vent) * u_box;
    float rv = length(dv);
    // otřes: pulz světla od kráteru přes celou krajinu
    c *= 1.0 + 0.22 * u_pulse;
    c += vec3(0.35, 0.07, 0.0) * u_pulse * exp(-rv * rv / (0.22 * 0.22));

    // --- světlo lávy na okolní krajinu ---
    float hl = texture(u_heat, uv).r;
    c = c * (1.0 + vec3(1.4, 0.45, 0.08) * hl) + vec3(0.55, 0.13, 0.0) * hl;

    // --- kráter: žhnoucí jícen, pukliny se rozbíhají, jak se sopka probouzí ---
    float ang = atan(dv.y, dv.x);
    float NC = 9.0;
    float fa = ang / 6.28318 * NC + (sopka_noise(vec2(rv * 22.0, 3.0)) - 0.5) * 0.9
               + (sopka_noise(vec2(rv * 70.0, 9.0)) - 0.5) * 0.25;
    float crack_id = floor(fa + 0.5);
    float len = (0.035 + 0.22 * u_wake) * (0.55 + 0.45 * sopka_hash(vec2(crack_id, 1.7)));
    float arc = abs(fract(fa + 0.5) - 0.5) / NC * 6.28318 * max(rv, 0.004);   // [m] od osy pukliny
    float wdt = 0.0035 * (1.0 - 0.6 * smoothstep(0.0, len, rv));
    float crack = (1.0 - smoothstep(wdt, wdt + 0.003, arc)) * (1.0 - smoothstep(len * 0.6, len, rv))
                  * smoothstep(0.015, 0.03, rv) * smoothstep(0.05, 0.3, u_wake);
    float halo = exp(-arc * arc / (0.012 * 0.012)) * (1.0 - smoothstep(len * 0.4, len, rv))
                 * smoothstep(0.015, 0.03, rv) * smoothstep(0.05, 0.3, u_wake);
    float beat = 0.7 + 0.3 * u_glow + 0.35 * u_pulse;
    c += vec3(0.9, 0.18, 0.0) * halo * 0.6 * beat;
    c = mix(c, sopka_hot(0.55 + 0.3 * u_pulse), crack * beat);
    // jícen: tmavý okraj, uvnitř vře láva
    float rim = exp(-pow((rv - 0.034) / 0.009, 2.0));
    c = mix(c, vec3(0.10, 0.06, 0.05), rim * 0.55 * smoothstep(0.02, 0.2, u_glow));
    float pool = 1.0 - smoothstep(0.018, 0.03, rv);
    float churn = sopka_noise(pm * 60.0 + vec2(u_time * 0.7, -u_time * 0.5));
    c = mix(c, sopka_hot(0.25 + 0.65 * u_glow * (0.7 + 0.3 * churn)), pool * clamp(u_glow * 1.3, 0.0, 1.0));
    c += vec3(1.0, 0.30, 0.03) * u_glow * 0.55 * exp(-rv * rv / (0.07 * 0.07));

    // --- fontána: žhavé jazyky stříkají z kráteru ---
    if (u_erupt > 0.01) {
        float jets = sopka_noise(vec2(ang * 5.0, rv * 18.0 - u_time * 7.0));
        jets = smoothstep(0.45, 0.85, jets) * (1.0 - smoothstep(0.03, 0.03 + 0.10 * u_erupt, rv));
        float core = 1.0 - smoothstep(0.02, 0.045 + 0.015 * u_erupt, rv);
        c = mix(c, sopka_hot(0.9), clamp(core + jets, 0.0, 1.0) * u_erupt);
    }
    // tlaková vlna na začátku erupce
    if (u_shock >= 0.0 && u_shock < 1.6) {
        float R = 0.05 + u_shock * 0.45;
        float ring = exp(-pow((rv - R) / 0.03, 2.0)) * (1.0 - u_shock / 1.6);
        c = mix(c, vec3(1.0, 0.55, 0.15), ring * 0.8);
    }

    // --- láva ---
    float th = sample_bicubic(u_lava, uv);
    float a = smoothstep(0.0004, 0.0016, th) * u_fade;
    if (a > 0.001) {
        float T = texture(u_temp, uv).r;
        float tn = clamp((T - u_solid) / (1.0 - u_solid), 0.0, 1.0);
        // povrch teče po spádu: krusta (desky) unášená proudem, dvě fáze se střídají
        vec2 g = height_grad(uv);
        float slope = length(g);
        vec2 flow = -g / max(slope, 0.05);
        float spd = tn * (0.006 + 0.04 * clamp(slope / 0.4, 0.0, 1.0));
        const float P = 2.2;
        const float K = 34.0;   // desky ~3 cm
        float ph0 = fract(u_time / P), ph1 = fract(u_time / P + 0.5);
        float wgt = abs(ph0 * 2.0 - 1.0);
        vec2 s0 = sopka_plates((pm - flow * spd * ph0 * P) * K);
        vec2 s1 = sopka_plates((pm - flow * spd * ph1 * P) * K + vec2(5.3, 1.7));
        float n0 = sopka_noise((pm - flow * spd * ph0 * P) * 45.0);
        float n1 = sopka_noise((pm - flow * spd * ph1 * P) * 45.0 + 7.0);
        vec2 pl = mix(s0, s1, wgt);
        float n = mix(n0, n1, wgt);

        // tenké okraje a čela proudů jsou chladnější
        float tv = tn * mix(0.5, 1.0, smoothstep(0.0008, 0.008, th));
        vec3 molten = sopka_hot(tv * (0.68 + 0.36 * n));
        // horká láva: jen ostrůvky tmavé kůže mezi širokými žhavými kanály; chladná: souvislá krusta
        float gap = mix(0.05, 0.30, smoothstep(0.1, 0.9, tn));
        float cover = mix(1.05, 0.62, smoothstep(0.2, 0.95, tn));
        float skin = smoothstep(gap, gap + 0.07, pl.x) * smoothstep(cover, cover - 0.15, pl.y);
        vec3 rock = vec3(0.22, 0.19, 0.18) * mix(0.55, 1.35, shade) * (0.8 + 0.2 * smoothstep(0.05, 0.3, pl.x));
        vec3 crust = mix(rock, vec3(0.42, 0.07, 0.02), smoothstep(0.0, 0.8, tn) * 0.9);
        // zbytkové teplo v puklinách ztuhlé lávy
        float heat = smoothstep(0.02, u_solid + 0.12, T);
        vec3 ember = sopka_hot(0.28 * heat) * heat * (0.8 + 0.2 * sin(u_time * 2.0 + n * 6.0));
        vec3 glowc = mix(ember, molten, smoothstep(0.0, 0.12, tn));
        vec3 lc = mix(glowc, crust, skin);
        if (T <= u_solid) lc = mix(ember, rock, smoothstep(0.03, 0.09, pl.x));
        c = mix(c, lc, a);
    }
    f_color = vec4(c, 1.0);
}
"""

# fáze cyklu: (název, délka [s])
CALM, RUMBLE, ERUPT, COOL, FADE = "klid", "probouzení", "erupce", "chladnutí", "vyhasínání"
DURATION = {CALM: 12.0, RUMBLE: 3.0, ERUPT: 20.0, COOL: 15.0, FADE: 4.0}
COUNTDOWN = DURATION[CALM] + DURATION[RUMBLE]  # [s] od označení sopky po erupci (15 s)
NEXT = {CALM: RUMBLE, RUMBLE: ERUPT, ERUPT: COOL, COOL: FADE, FADE: CALM}
ERUPT_RATE = 2.6e-4  # [m³/s · m⁻²] výron lávy z kráteru (víc = delší proudy)
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


VILLAGE_NAMES = ["Popelov", "Žhavice", "Horní Lhota", "Dolany", "Kamenice", "Pod Sopkou", "Teplá", "Lávka"]


@dataclass
class Village:
    name: str
    x: float
    y: float
    houses: list[House]

    @property
    def burned(self) -> bool:
        return any(h.flooded for h in self.houses)


def place_villages(heights: np.ndarray, size: tuple[float, float], vent: tuple[float, float],
                   rng: np.random.Generator, n: int = 4, min_gap: float = 0.2) -> list[tuple[float, float]]:
    """Místa pro vesnice: rovinatá, níž než kráter, v dosahu lávy (20–60 cm od kráteru),
    pokud možno v údolích po spádu. Navzájem aspoň ``min_gap`` od sebe, ne u stěn."""
    gh, gw = heights.shape
    W, D = size
    f = max(1, gw // 64)
    coarse = downsample(heights, f)
    ch, cw = coarse.shape
    cx, cy = W / cw, D / ch
    gx, gy = np.meshgrid((np.arange(cw) + 0.5) * cx, (np.arange(ch) + 0.5) * cy)
    gyy, gxx = np.gradient(coarse, cy, cx)
    slope = np.hypot(gxx, gyy)
    d = np.hypot(gx - vent[0], gy - vent[1])
    vent_h = float(coarse[min(ch - 1, int(vent[1] / cy)), min(cw - 1, int(vent[0] / cx))])
    mx, my = 0.10, 0.17  # nahoře a dole jsou texty pro návštěvníky
    ok = (gx > mx) & (gx < W - mx) & (gy > my) & (gy < D - my) & (d > 0.2) & (d < 0.65)
    # nízko (údolí) a rovně je lepší, kousek náhody ať to není pokaždé stejné
    cost = (coarse - vent_h) / 0.03 + slope / 0.15 + np.abs(d - 0.35) / 0.15 + rng.random(coarse.shape) * 1.2
    cost = np.where(ok, cost, np.inf)
    out: list[tuple[float, float]] = []
    for _ in range(n):
        if not np.isfinite(cost).any():
            break
        k = int(np.argmin(cost))
        px, py = float(gx.flat[k]), float(gy.flat[k])
        out.append((px, py))
        cost[np.hypot(gx - px, gy - py) < min_gap] = np.inf
    return out


def build_village(name: str, x: float, y: float, rng: np.random.Generator, n_houses: int = 4) -> Village:
    houses = []
    for i in range(n_houses):
        ang = i / max(1, n_houses - 1) * 2 * math.pi + rng.uniform(-0.3, 0.3)
        r = 0.0 if i == 0 else rng.uniform(0.045, 0.055)
        houses.append(House(x + r * math.cos(ang), y + r * math.sin(ang), rng.uniform(0, math.pi), 0.034, 0.022,
                            church=(i == 0)))
    return Village(name, x, y, houses)


SMOKE, EMBER, BOMB, STEAM = 0, 1, 2, 3
GRAVITY = 9.81 * 0.35  # [m/s²] zpomalený pád úlomků (fontána je dobře vidět)


class Particles:
    """Částice v metrech bedny: kouř/popel, pára, žhavé úlomky a lávové bomby.

    Úlomky letí balisticky – ``z`` je výška nad pískem (vykreslí se větší, jak jsou blíž projektoru),
    po dopadu se zastaví a dohasínají."""

    def __init__(self) -> None:
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.age = np.zeros(0, np.float32)
        self.life = np.zeros(0, np.float32)
        self.kind = np.zeros(0, np.int8)
        self.z = np.zeros(0, np.float32)
        self.vz = np.zeros(0, np.float32)
        self.landed: np.ndarray = np.zeros((0, 2), np.float32)  # kde tento krok dopadly bomby

    def __len__(self) -> int:
        return len(self.age)

    def clear(self) -> None:
        self.__init__()  # type: ignore[misc]

    def spawn(self, pos, vel, life, kind: int, vz=None) -> None:
        n = len(life)
        if n == 0:
            return
        self.pos = np.vstack([self.pos, np.asarray(pos, np.float32).reshape(n, 2)])
        self.vel = np.vstack([self.vel, np.asarray(vel, np.float32).reshape(n, 2)])
        self.age = np.concatenate([self.age, np.zeros(n, np.float32)])
        self.life = np.concatenate([self.life, np.asarray(life, np.float32)])
        self.kind = np.concatenate([self.kind, np.full(n, kind, np.int8)])
        self.z = np.concatenate([self.z, np.zeros(n, np.float32)])
        vz = np.zeros(n, np.float32) if vz is None else np.asarray(vz, np.float32).reshape(n)
        self.vz = np.concatenate([self.vz, vz])

    def keep(self, keep) -> None:
        self.pos, self.vel, self.age, self.life = self.pos[keep], self.vel[keep], self.age[keep], self.life[keep]
        self.kind, self.z, self.vz = self.kind[keep], self.z[keep], self.vz[keep]

    def update(self, dt: float, wind: np.ndarray) -> None:
        self.landed = np.zeros((0, 2), np.float32)
        if not len(self):
            return
        gas = (self.kind == SMOKE) | (self.kind == STEAM)
        self.vel[gas] += (wind - self.vel[gas]) * min(1.0, 0.8 * dt)
        solid = ~gas
        air = solid & ((self.z > 0) | (self.vz > 0))
        ground = solid & ~air
        self.vel[air] *= max(0.0, 1.0 - 0.25 * dt)
        self.vel[ground] *= max(0.0, 1.0 - 9.0 * dt)
        self.vz[air] -= GRAVITY * dt
        self.z[air] += self.vz[air] * dt
        hit = air & (self.z <= 0)
        if hit.any():
            self.z[hit] = 0.0
            self.vz[hit] = 0.0
            self.landed = self.pos[hit & (self.kind == BOMB)].copy()
        self.pos += self.vel * dt
        self.age += dt
        keep = self.age < self.life
        if not keep.all():
            self.keep(keep)


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
    description = "Nejvyšší kopec je sopka. Po odpočtu z ní vyteče láva – ochraň vesnice hrázemi z písku."
    order = 60
    category = "zážitek"
    intro = "Nejvyšší kopec je sopka – za 15 s vybuchne! Hrázemi z písku ochraň vesnice pod ní."
    keys = [
        ("E", "erupce hned"),
        ("A", "automatický cyklus zap/vyp"),
        ("V", "vesnice zap/vyp"),
        ("X", "smazat lávu"),
        ("↑ / ↓", "síla erupce"),
    ]

    WALL_GAP = 0.10  # [m] kráter ne blíž ke stěně (u stěn bývá nahrnutý písek)
    LOCK_HEIGHT = 0.02  # [m] od jak vysokého kopce se místo sopky zamkne

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.sim: LavaSim | None = None
        self.tex_lava = None
        self.tex_temp = None
        self.tex_heat = None
        self._heat_on = False
        self._tremor_ph = 0.0
        self.pulse = 0.0
        self.rough: np.ndarray | None = None
        self.particles = Particles()
        self.rng = np.random.default_rng()
        self.auto = getattr(self, "auto", True)
        self.power = getattr(self, "power", 1.0)
        self.vent = (0, 0)  # (iy, ix) v mřížce simulace
        self.vent_m = (0.5, 0.4)  # kráter v metrech
        self.vent_height = 0.0
        self.eruptions = 0
        self.fade = 1.0
        self.show_villages = getattr(self, "show_villages", True)
        self.villages: list[Village] = []
        self._place_pending = True
        self.saved_last: tuple[int, int] | None = None
        self.clock = 0.0
        self._start_calm()

    def _start_calm(self) -> None:
        """Klid po spuštění/resetu: odpočet se prodlouží o úvodní slajd (ten zakryje střed písku)."""
        self._set_phase(CALM, DURATION[CALM] + self.host.settings.attract.intro_seconds)

    def stop(self) -> None:
        self.prog.release()
        for t in (self.tex_lava, self.tex_temp, self.tex_heat):
            if t is not None:
                t.release()
        self.tex_lava = self.tex_temp = self.tex_heat = None

    def reset(self) -> None:
        if self.sim is not None:
            self.sim.clear()
        self.particles.clear()
        self.fade = 1.0
        self.villages = []
        self._place_pending = True
        self.saved_last = None
        self._start_calm()

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
        if phase == FADE and self.villages:  # konec erupce: kolik vesnic přežilo
            self.saved_last = (sum(not v.burned for v in self.villages), len(self.villages))
        if phase == CALM:
            self._place_pending = True  # nový cyklus = nové vesnice (terén se mezitím změnil)
            self._vent_locked = False  # nový cyklus = sopka se znovu vybere

    def _wake(self, hm) -> None:
        """Probudí sopku na zamčeném místě (kráter se během kola nestěhuje)."""
        if not self._vent_locked:
            self._lock_vent(hm, force=True)
        self._set_phase(RUMBLE)
        if self._place_pending:
            self._place_villages(hm)

    def _lock_vent(self, hm, force: bool = False) -> None:
        """Na začátku kola: sopka = nejvyšší vrchol, a tam už zůstane až do konce kola
        (když lidé hrabou a staví hráze, nejvyšší místo se mění – sopka nesmí přeskakovat).
        Na rovném písku se ještě nezamyká – počká, až někdo postaví kopec."""
        old = self.vent_m
        self._choose_vent(hm)
        if force or self.vent_height >= self.LOCK_HEIGHT:
            self._vent_locked = True
        if not self.villages or math.dist(old, self.vent_m) > 0.12:
            self._place_pending = True

    def _place_villages(self, hm) -> None:
        self._place_pending = False
        self.saved_last = None
        if not self.show_villages:
            self.villages = []
            return
        spots = place_villages(hm.heights, hm.size, self.vent_m, self.rng)
        names = list(self.rng.permutation(VILLAGE_NAMES))
        self.villages = [build_village(names[i], x, y, self.rng) for i, (x, y) in enumerate(spots)]

    def _choose_vent(self, hm) -> None:
        """Kráter = nejvyšší vrchol (ne u stěn, kde se písek nahrne o okraj)."""
        gh, gw = hm.shape
        cw, ch = hm.cell_size
        border = max(2, int(round(self.WALL_GAP / cw)))
        iy, ix, rel = find_vent(hm.heights, hm.hands, border=border)
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
            for t in (self.tex_lava, self.tex_temp, self.tex_heat):
                if t is not None:
                    t.release()
            self.tex_lava = self.host.float_texture((shape[1], shape[0]))
            self.tex_temp = self.host.float_texture((shape[1], shape[0]))
            self.tex_heat = self.host.float_texture((shape[1], shape[0]), np.zeros(shape, np.float32))
            self._heat_on = False
            self._choose_vent(hm)
        sim = self.sim
        assert self.rough is not None

        self.phase_t += dt
        self.clock += dt
        if self.phase == CALM:
            if not self._vent_locked:
                self._vent_t = getattr(self, "_vent_t", 0.0) + dt
                if self.phase_t < 0.05 or self._vent_t > 0.5:
                    self._vent_t = 0.0
                    self._lock_vent(hm)
            if self._place_pending and self.phase_t > 0.5:
                self._place_villages(hm)  # v klidu podle aktuálního terénu, ať je vidět, co chránit
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
            rate = ERUPT_RATE * self.power * pulse * (1.0 - 0.45 * k)
            sim.erupt(self.vent[0], self.vent[1], 3.0, rate, dt)
        if self.phase == FADE:
            self.fade = max(0.0, 1.0 - self.phase_t / self.phase_len)
            if self.phase_t + dt >= self.phase_len:
                sim.clear()
                self.fade = 1.0
        if sim.thickness.any():
            terrain = downsample(hm.heights, DS)[: shape[0], : shape[1]] + self.rough
            sim.advance(terrain, dt)
        self._particles(dt)
        self._burn_houses(hm)
        self._tremor(dt)
        self._sound(dt)
        assert self.tex_lava is not None and self.tex_temp is not None and self.tex_heat is not None
        self.tex_lava.write(sim.thickness.tobytes())
        self.tex_temp.write(sim.temp.tobytes())
        if sim.thickness.any():  # žhavá láva svítí do okolí (rozmazané teplo, ~5 cm)
            hot = np.clip((sim.temp - sim.solid) / (1 - sim.solid), 0, 1) * np.clip(sim.thickness / 0.002, 0, 1)
            heat = cv2.GaussianBlur(hot, (0, 0), 0.05 / sim.cell) * (1.8 * self.fade)
            self.tex_heat.write(np.ascontiguousarray(heat, np.float32).tobytes())
            self._heat_on = True
        elif self._heat_on:
            self.tex_heat.write(np.zeros(sim.shape, np.float32).tobytes())
            self._heat_on = False

    def _sound(self, dt: float) -> None:
        """Zvuk podle fáze: dunění sílí k erupci, posledních 5 s pípá odpočet, výbuch, láva bublá,
        dopady bomb, syčení při chladnutí, na konci výsledek."""
        sfx = self.sfx
        x = self.vent_m[0] / self.host.box_size[0]
        prev = getattr(self, "_snd_phase", None)
        self._snd_phase = self.phase
        if prev is not None and prev != self.phase:
            if self.phase == ERUPT:
                sfx.play("boom", 1.0, x=x)
                if prev == RUMBLE:
                    sfx.play("go", 0.8)
            elif self.phase == COOL:
                sfx.play("hiss", 0.7, x=x)
            elif self.phase == FADE and self.saved_last is not None:
                s_, n_ = self.saved_last
                sfx.play("fanfare" if s_ == n_ else "chime" if s_ else "fail", 0.9)
        # odpočet: pípnutí při změně celé sekundy v posledních 5 s (5, 4, 3, 2, 1)
        left = self._countdown_left()
        if left is not None:
            last = getattr(self, "_snd_left", None)
            now = math.ceil(left)
            if last is not None and now != last and 1 <= now <= 5:
                sfx.play("beep", 0.8)
            self._snd_left = now
        else:
            self._snd_left = None
        # dunění země: od poloviny odpočtu sílí, při erupci drží, při chladnutí doznívá
        if left is not None:
            rumble = 0.9 * max(0.0, (self._wake_level() - 0.15) / 0.85)
        elif self.phase == ERUPT:
            rumble = 0.75 * (1.0 - 0.5 * self.phase_t / self.phase_len)
        elif self.phase == COOL:
            rumble = 0.3 * max(0.0, 1.0 - 2.0 * self.phase_t / self.phase_len)
        else:
            rumble = 0.0
        sfx.loop("rumble", rumble, x=x)
        # bublání lávy podle žhavé plochy (váženo teplotou – chladnoucí láva utichá)
        sim = self.sim
        if sim is not None and self.phase in (ERUPT, COOL, FADE):
            hot = float(np.clip((sim.temp - sim.solid) / (1 - sim.solid), 0, 1).sum()) * sim.cell * sim.cell
            sfx.loop("lava", min(1.0, hot / 0.15) * 0.85 * self.fade, x=x)
        # dopad lávové bomby: jen občas (ne každá), nejvýš jednou za 2 s
        landed = self.particles.landed
        if len(landed) and self.clock - getattr(self, "_thud_t", -9.0) > 2.0 and self.rng.random() < 0.5:
            self._thud_t = self.clock
            sfx.play("thud", 0.55, x=float(landed[0][0]) / self.host.box_size[0],
                     pitch=float(self.rng.uniform(0.85, 1.15)))

    def _countdown_left(self) -> float | None:
        """[s] do erupce (jen během odpočtu)."""
        if self.phase == CALM and self.auto:
            return self.phase_len - self.phase_t + DURATION[RUMBLE]
        if self.phase == RUMBLE:
            return self.phase_len - self.phase_t
        return None

    def _wake_level(self) -> float:
        """0..1 jak moc je sopka probuzená (pukliny, potemnění krajiny)."""
        left = self._countdown_left()
        if left is not None:
            return float(np.clip(1.0 - left / COUNTDOWN, 0.0, 1.0)) ** 1.3
        if self.phase == ERUPT:
            return 1.0
        if self.phase == COOL:
            return max(0.0, 1.0 - self.phase_t / self.phase_len)
        return 0.0

    def _tremor(self, dt: float) -> None:
        """Otřesy = pulzy světla; zrychlují se k erupci (nejvýš ~2 Hz, žádné blikání)."""
        left = self._countdown_left()
        amp, freq = 0.0, 1.0
        if left is not None and left < 8.0:
            u = 1.0 - left / 8.0
            amp, freq = 0.35 + 0.65 * u, 0.7 + 1.3 * u
        elif self.phase == ERUPT:
            amp, freq = 0.8 * (1.0 - 0.6 * self.phase_t / self.phase_len), 1.1
        self._tremor_ph += dt * freq
        self.pulse = amp * math.exp(-(self._tremor_ph % 1.0) * 5.0)

    def _burn_houses(self, hm) -> None:
        """Dům, pod kterým je horká láva, shoří."""
        sim = self.sim
        if sim is None or not self.villages or self.phase in (CALM, FADE):
            return
        sh, sw = sim.shape
        W, D = hm.size
        for v in self.villages:
            for h in v.houses:
                if h.flooded:
                    continue
                iy = min(sh - 1, max(0, int(h.y / D * sh)))
                ix = min(sw - 1, max(0, int(h.x / W * sw)))
                win = (slice(max(0, iy - 1), iy + 2), slice(max(0, ix - 1), ix + 2))
                if (sim.thickness[win] > 0.0012).any() and (sim.temp[win] > sim.solid).any():
                    h.flooded, h.flooded_at = True, self.clock
                    if self.clock - getattr(self, "_burn_snd", -9.0) > 0.5:  # víc domů naráz = jeden zvuk
                        self._burn_snd = self.clock
                        self.sfx.play("burn", 0.9, x=h.x / W, pitch=float(self.rng.uniform(0.9, 1.1)))

    def _particles(self, dt: float) -> None:
        p = self.particles
        rng = self.rng
        vx, vy = self.vent_m
        ph = self.phase
        # dým z kráteru (při erupci hustý sloup popela), v klidu jen pára
        smoke_rate = {CALM: 1.0, RUMBLE: 7.0, ERUPT: 14.0, COOL: 4.0, FADE: 1.0}[ph]
        n = rng.poisson(smoke_rate * dt)
        if n:
            ang = rng.uniform(0, 2 * math.pi, n)
            sp = rng.uniform(0.02, 0.07 if ph == ERUPT else 0.04, n)
            p.spawn(np.stack([vx + 0.012 * np.cos(ang), vy + 0.012 * np.sin(ang)], 1),
                    np.stack([sp * np.cos(ang), sp * np.sin(ang)], 1), rng.uniform(3.0, 6.0, n), SMOKE)
        if ph == CALM:
            n = rng.poisson(2.0 * dt)
            if n:
                ang = rng.uniform(0, 2 * math.pi, n)
                p.spawn(np.stack([vx + 0.01 * np.cos(ang), vy + 0.01 * np.sin(ang)], 1),
                        np.stack([0.03 * np.cos(ang), 0.03 * np.sin(ang)], 1), rng.uniform(2.0, 3.5, n), STEAM)
        # žhavé úlomky: při probouzení občas vyprskne, při erupci fontána
        if ph == RUMBLE:
            ember_rate = 6.0 + 30.0 * self.phase_t / DURATION[RUMBLE]
            speed, vz = (0.03, 0.15), (0.4, 1.0)
        elif ph == ERUPT:
            k = self.phase_t / self.phase_len
            ember_rate = 110.0 * self.power * (0.55 + 0.45 * self._fountain()) * (1.0 - 0.5 * k)
            speed, vz = (0.04, 0.32), (0.9, 2.1)
        else:
            ember_rate = 0.0
        n = rng.poisson(ember_rate * dt)
        if n:
            ang = rng.uniform(0, 2 * math.pi, n)
            sp = rng.uniform(*speed, n)
            p.spawn(np.stack([vx + 0.01 * np.cos(ang), vy + 0.01 * np.sin(ang)], 1),
                    np.stack([sp * np.cos(ang), sp * np.sin(ang)], 1), rng.uniform(1.3, 2.4, n), EMBER,
                    vz=rng.uniform(*vz, n))
        if ph == ERUPT:  # lávové bomby: velké kusy letí dál
            n = rng.poisson(5.0 * self.power * dt)
            if n:
                ang = rng.uniform(0, 2 * math.pi, n)
                sp = rng.uniform(0.12, 0.38, n)
                p.spawn(np.stack([np.full(n, vx), np.full(n, vy)], 1), np.stack([sp * np.cos(ang), sp * np.sin(ang)], 1),
                        rng.uniform(3.0, 4.0, n), BOMB, vz=rng.uniform(1.4, 2.3, n))
        # hořící domy kouří
        for v in self.villages:
            for h in v.houses:
                if h.flooded and self.clock - h.flooded_at < 8.0 and rng.random() < 2.5 * dt:
                    p.spawn([[h.x, h.y]], [[0.0, 0.0]], [rng.uniform(2.5, 4.0)], SMOKE)
        # chladnoucí láva dýmá
        sim = self.sim
        if sim is not None and ph in (ERUPT, COOL):
            n = rng.poisson(5.0 * dt)
            if n:
                cells = np.flatnonzero(sim.molten)
                if len(cells):
                    pick = rng.choice(cells, n)
                    sh, sw = sim.shape
                    iy, ix = np.divmod(pick, sw)
                    pos = np.stack([(ix + 0.5) * sim.cell, (iy + 0.5) * sim.cell], 1)
                    p.spawn(pos, np.zeros((n, 2)), rng.uniform(1.5, 2.5, n), STEAM)
        if len(p) > 900:
            p.keep(slice(len(p) - 900, None))
        p.update(dt, getattr(self, "wind", np.zeros(2, np.float32)))
        if len(p.landed):  # bomba dopadla: obláček prachu
            m = len(p.landed)
            ang = rng.uniform(0, 2 * math.pi, m)
            p.spawn(p.landed, np.stack([0.03 * np.cos(ang), 0.03 * np.sin(ang)], 1), rng.uniform(1.0, 1.8, m), SMOKE)

    def _fountain(self) -> float:
        """0..1 pulzování fontány v kráteru."""
        if self.phase != ERUPT:
            return 0.0
        return 0.5 + 0.5 * math.sin(self.phase_t * 5.3) * math.sin(self.phase_t * 1.7 + 1)

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

    def _erupt_level(self) -> float:
        if self.phase != ERUPT:
            return 0.0
        return (0.6 + 0.4 * self._fountain()) * (1.0 - 0.5 * self.phase_t / self.phase_len)

    def render(self) -> None:
        if self.tex_lava is None or self.sim is None:
            return
        bw, bh = self.host.box_size
        self.host.draw_fullscreen(self.prog, u_lava=(self.tex_lava, 1), u_temp=(self.tex_temp, 2),
                                  u_vent=(self.vent_m[0] / bw, self.vent_m[1] / bh), u_glow=float(self._glow()),
                                  u_solid=float(self.sim.solid), u_fade=float(self.fade), u_heat=(self.tex_heat, 3),
                                  u_wake=float(self._wake_level()), u_pulse=float(self.pulse),
                                  u_shock=float(self.phase_t if self.phase == ERUPT else -1.0),
                                  u_erupt=float(self._erupt_level()))
        self._draw_villages()
        self._draw_particles()
        self.host.glow(0.75, 0.6)
        self._draw_marker()
        self._labels()

    def _draw_marker(self) -> None:
        """Před erupcí: kde je sopka (kruh přímo na kráteru) – lidé vědí, odkud poteče láva."""
        if self.phase not in (CALM, RUMBLE):
            return
        p = self.host.painter
        canvas = self.host.canvas_size
        sx, sy = canvas[0] / self.host.box_size[0], canvas[1] / self.host.box_size[1]
        x, y = self.vent_m[0] * sx, self.vent_m[1] * sy
        t = self.host.time
        urgent = self.phase == RUMBLE
        r = 0.07 * sx
        # s otřesy (pulz) žlutá přechází do rudé – plynule, bez blikání
        u = min(1.0, self.pulse) if urgent else 0.0
        col = (1.0, 0.8 - 0.45 * u, 0.2 - 0.1 * u, 1)
        w = max(4.0, canvas[1] / 110)
        p.circle(x, y, r, (0, 0, 0, 0.6), width=w * 2.2)
        p.circle(x, y, r, col, width=w)
        ph = (t * (1.5 if urgent else 0.7)) % 1.0  # vlna od kráteru
        p.circle(x, y, r * (1 + 1.2 * ph), (col[0], col[1], col[2], 0.8 * (1 - ph)), width=w * 0.7)
        sz = hud.size(canvas, 0.7)
        above = y - r - sz * 0.2 > sz * 3.2  # nahoře je hlavní nápis – jinak popisek pod kruh
        hud.label(p, "SOPKA", x, y - r - sz * 0.2 if above else y + r + sz * 0.2, sz, color=col,
                  anchor="bc" if above else "tc")

    def _draw_villages(self) -> None:
        if not self.villages:
            return
        p = self.host.painter
        canvas = self.host.canvas_size
        sx, sy = canvas[0] / self.host.box_size[0], canvas[1] / self.host.box_size[1]
        sz = hud.size(canvas, 0.62)
        for v in self.villages:
            for h in v.houses:
                draw_house(p, h, sx, sy, self.clock, damage="fire")
            color = (1.0, 0.55, 0.4, 1) if v.burned else (1, 1, 1, 1)
            hud.label(p, v.name, v.x * sx, v.y * sy + 0.06 * sy, sz, color=color, anchor="tc",
                      bg=(0, 0, 0, 0.45))

    def _draw_particles(self) -> None:
        """Kouř, pára, žhavé úlomky se stopou a lávové bomby – vše jedním voláním painteru."""
        p = self.particles
        if not len(p):
            return
        cw, ch = self.host.canvas_size
        bw, bh = self.host.box_size
        scale = np.array([cw / bw, ch / bh], np.float32)
        ppm = cw / bw
        k = np.clip(p.age / p.life, 0, 1)
        centers = p.pos * scale
        verts, colors = [], []

        gas = (p.kind == SMOKE) | (p.kind == STEAM)
        gi = np.flatnonzero(gas)
        if len(gi):
            kk = k[gi]
            smoke = p.kind[gi] == SMOKE
            r = np.where(smoke, 0.016 + 0.085 * np.sqrt(kk), 0.010 + 0.04 * np.sqrt(kk)) * ppm
            grey = np.where(smoke, 0.40 - 0.16 * kk, 0.88)
            alpha = np.where(smoke, 0.45, 0.24) * (1 - kk) * np.minimum(1, kk * 6)
            col = np.stack([grey, grey * 0.96, grey * 0.94], 1)
            # sloup dýmu nad kráterem je podsvícený žárem
            d = np.hypot(*(p.pos[gi] - np.array(self.vent_m, np.float32)).T)
            lit = (min(1.0, self._glow()) * np.exp(-(d / 0.13) ** 2) * smoke)[:, None]
            col = col * (1 - 0.6 * lit) + lit * np.array([1.0, 0.33, 0.06]) * 0.85
            v, c = disc_triangles(centers[gi], r, np.column_stack([col, alpha]).astype(np.float32), 14)
            verts.append(v)
            colors.append(c)

        si = np.flatnonzero(~gas)
        if len(si):
            z, kk = p.z[si], k[si]
            bomb = p.kind[si] == BOMB
            air = z > 0
            # dopadlé dohasínají z oranžové do rudé
            heat = np.where(air, 1.0, np.clip((1 - kk) * 2.0, 0, 1))
            r = np.where(bomb, 0.011, 0.0058) * ppm * (1 + 1.4 * z)
            ember = np.stack([np.ones_like(heat), 0.15 + 0.5 * heat ** 1.5, 0.02 + 0.16 * heat ** 3,
                              np.clip(heat * 1.5, 0, 1)], 1).astype(np.float32)
            # stopa letících úlomků (protažená po směru letu)
            fly = np.flatnonzero(air)
            if len(fly):
                vel = p.vel[si][fly] * scale
                spd = np.linalg.norm(vel, axis=1, keepdims=True)
                dirv = vel / np.maximum(spd, 1e-6)
                nrm = np.stack([-dirv[:, 1], dirv[:, 0]], 1)
                head = centers[si][fly]
                rr = r[fly, None]
                tail = head - dirv * (spd * 0.10 + rr * 2)
                tri = np.stack([head + nrm * rr * 0.8, head - nrm * rr * 0.8, tail], 1).reshape(-1, 2)
                tc = np.repeat(ember[fly], 3, axis=0)
                tc[2::3, 3] = 0.0
                tc[0::3, 3] *= 0.75
                tc[1::3, 3] *= 0.75
                verts.append(tri)
                colors.append(tc)
            order = np.argsort(air, kind="stable")  # dopadlé pod letícími
            sel = si[order]
            # žhnoucí obal
            v, c = disc_triangles(centers[sel], r[order] * 1.35, ember[order], 10)
            verts.append(v)
            colors.append(c)
            # bomby: tmavé jádro s žhnoucím okrajem
            b = order[bomb[order]]
            if len(b):
                core = np.tile(np.array([[0.30, 0.09, 0.04, 1.0]], np.float32), (len(b), 1))
                core[:, 3] = np.clip(heat[b] * 1.5, 0, 0.9)  # dopadlá bomba splyne s lávou
                v, c = disc_triangles(centers[si][b], r[b] * 0.65, core, 10)
                verts.append(v)
                colors.append(c)
        if verts:
            self.host.painter.triangles(np.concatenate(verts), np.concatenate(colors))

    def _labels(self) -> None:
        p = self.host.painter
        canvas = self.host.canvas_size
        cw, ch = canvas
        size = hud.size(canvas)
        small = hud.size(canvas, 0.8)
        remaining = max(0.0, self.phase_len - self.phase_t)
        top = None
        bottom = None
        color = hud.YELLOW
        n_v = len(self.villages)
        safe = sum(not v.burned for v in self.villages)
        if self.phase == CALM:
            left = math.ceil(remaining + DURATION[RUMBLE])
            top = f"Sopka vybuchne za {left} s" if self.auto else "Sopka spí (erupce klávesou E)"
            color = hud.WHITE
            if self.saved_last is not None and self.phase_t < 6.0:
                s_, n_ = self.saved_last
                top = f"Zachránili jste {s_} z {n_} vesnic!" if s_ else "Láva zničila všechny vesnice…"
                color = hud.GREEN if s_ == n_ else hud.YELLOW
            if self.vent_height < 0.02:
                bottom = "Postav z písku vysoký kopec – nejvyšší vrchol je sopka"
            elif n_v:
                bottom = "Láva poteče ze sopky dolů – postav hráze, ať ochráníš vesnice"
            else:
                bottom = "Nejvyšší vrchol je sopka – odtud poteče láva"
        elif self.phase == RUMBLE:
            top = f"Sopka vybuchne za {math.ceil(remaining)} s!"
            color = (1.0, 0.55, 0.15, 1)
            bottom = "Rychle – hráze z písku kolem vesnic!" if n_v else None
        elif self.phase == ERUPT:
            if self.phase_t < 3.0:
                top = "ERUPCE!"
                u = 0.5 + 0.5 * math.cos(self.phase_t * 2 * math.pi * 1.5)
                color = (1.0, 0.35 + 0.55 * u, 0.1 + 0.2 * u, 1)
                size = int(size * 1.4)
            else:
                top = "Láva teče z kopce dolů"
            bottom = "Hrází z písku nebo korytem odveď lávu od vesnic" if n_v else \
                "Rukama můžeš lávě postavit hráz nebo vykopat koryto"
        elif self.phase == COOL:
            top = "Láva chladne a tuhne na černý kámen"
            color = (1.0, 0.6, 0.3, 1)
        if top:
            hud.label(p, top, cw / 2, size * 0.25, size, anchor="tc", color=color)
        if bottom:
            hud.label(p, bottom, cw / 2, ch - small * 0.3, small, anchor="bc")
        if n_v and self.phase in (RUMBLE, ERUPT, COOL):  # pod hlavním nápisem, ať se nepřekrývají
            hud.label(p, f"Vesnice v bezpečí: {safe} z {n_v}", cw / 2, size * 0.25 + size * 1.75, small,
                      anchor="tc", color=hud.GREEN if safe == n_v else (1.0, 0.55, 0.4, 1))

    # --- ovládání ------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        hm = self.host.heightmap
        if name == "e":
            if hm is not None and not self._vent_locked:
                self._lock_vent(hm, force=True)
            self.fade = 1.0
            self._set_phase(ERUPT)
            self.host.notify("Erupce!")
        elif name == "v":
            self.show_villages = not self.show_villages
            hm = self.host.heightmap
            if hm is not None:
                self._place_villages(hm)
            self.host.notify("Vesnice " + ("zapnuty" if self.show_villages else "vypnuty"))
        elif name == "a":
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
                 f"Automatický cyklus: {'ano' if self.auto else 'ne'} · síla ×{self.power:.2f} · erupcí {self.eruptions}",
                 f"Vesnice: {sum(not v.burned for v in self.villages)}/{len(self.villages)} v bezpečí"]
        if self.sim is not None:
            cell = self.sim.cell
            lines.append(f"Láva: {self.sim.volume * 1000:.2f} l, tekutá plocha "
                         f"{self.sim.molten.sum() * cell * cell * 1e4:.0f} cm²")
        return lines
