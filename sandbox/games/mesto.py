"""Záchrana města: v nížině stojí městečko, z vyvýšeného okraje přijde povodeň.

Návštěvníci mají chvilku na stavbu hrází a kanálů, které vodu odvedou jinam.
Pramen povodně je na nejvyšším okraji bedny, město v rovinaté nížině po proudu,
kam by voda opravdu stekla (odtok D8 po skutečném terénu). Proud vody počítá
WaterSim. Dům, ke kterému se dostane voda, je zatopený. Na konci se ukáže,
kolik domů se podařilo zachránit, a hra začne znovu s městem jinde.
"""

from __future__ import annotations

import math

import numpy as np

from ..render import hud
from ..sim import ecosystem as eco
from ..sim.water import WaterSim
from ..render.fx import RainFx, WaterLayer
from .base import Game
from .kulicka import Sparks
from .poklad import star_tris

FRAG = """
uniform sampler2D u_water;
uniform sampler2D u_vel;
uniform float u_clock;
uniform vec4 u_roads[12];     // ulice městečka: úsečky (uv a, uv b)
uniform int u_nroads;
uniform float u_alarm;        // 0..1 poslední vteřiny před povodní: rudé tepání okrajů
uniform float u_storm;        // 0..1 bouře během povodně (ztmavení, blesky)
uniform float u_flash;        // 0..1 záblesk blesku
uniform vec3 u_result;        // výsledek: barva rozzáření (0 = nic)

float mesto_seg(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
    return length(p - a - ab * t);
}

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;
    float h = height_at(uv);
    vec3 c = land_color((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4));
    c = mix(c, vec3(dot(c, vec3(0.3, 0.55, 0.15))), 0.15);   // trochu tlumené, ať vynikne voda a domy
    // ulice: světlá dlážděná cesta s tmavým lemem (pod vodou zmizí)
    float road = 1e3;
    for (int i = 0; i < u_nroads; ++i) {
        road = min(road, mesto_seg(pm, u_roads[i].xy * u_box, u_roads[i].zw * u_box));
    }
    c = mix(c, vec3(0.30, 0.20, 0.12), 1.0 - smoothstep(0.0075, 0.0095, road));
    c = mix(c, vec3(0.95, 0.86, 0.66), 1.0 - smoothstep(0.0045, 0.0065, road));
    c *= mix(0.52, 1.12, hillshade(uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.5, cl.x * 0.5);
    c = mix(c, c * 0.3, cl.y * 0.7);

    float w = max(sample_bicubic(u_water, uv), 0.0);
    vec2 v = texture(u_vel, uv).rg;
    float foam = clamp((length(v) - 0.10) / 0.25, 0.0, 1.0) * (1.0 - smoothstep(0.002, 0.010, w)) * 0.55;
    c = water_shade(c, uv, w, v, foam).rgb;
    // bouře: krajina potemní do modrošeda, blesk ji na okamžik ozáří
    c = mix(c, c * vec3(0.62, 0.70, 0.88), u_storm * 0.35);
    c += vec3(0.55, 0.65, 0.85) * u_flash * 0.45;
    // poplach: rudě tepající okraje
    vec2 e = min(pm, u_box - pm);
    float edge = 1.0 - smoothstep(0.0, 0.10, min(e.x, e.y));
    c = mix(c, vec3(1.0, 0.08, 0.05), edge * u_alarm * (0.55 + 0.45 * sin(u_clock * 9.0)));
    // výsledek: okraje se rozzáří barvou hodnocení
    c += u_result * edge * (0.5 + 0.3 * sin(u_clock * 3.0));
    f_color = vec4(c, 1.0);
}
"""

PLAN, PREP, FLOOD, RESULT = "plánování", "příprava", "povodeň", "výsledek"


def _unit_ring(segments: int) -> np.ndarray:
    ang = np.linspace(0, 2 * math.pi, segments + 1)
    return np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)


_RINGS = {n: _unit_ring(n) for n in (12, 16, 20, 24)}


def _disc_tris(x: float, y: float, r: float, segments: int) -> np.ndarray:
    ring = _RINGS[segments] * r + (x, y)
    c = np.broadcast_to(np.array((x, y), np.float32), (segments, 2))
    return np.stack([c, ring[:-1], ring[1:]], 1).reshape(-1, 2)


def _soft_disc(p, x: float, y: float, r: float, color, segments: int = 20) -> None:
    """Měkký kruh: střed barvou, okraj průhledný (světlo, kouř)."""
    col = np.tile(np.array([color, (*color[:3], 0.0), (*color[:3], 0.0)], np.float32), (segments, 1))
    p.triangles(_disc_tris(x, y, r, segments), col)


class _HouseBatch:
    """Trojúhelníky jednoho domu: místní souřadnice (otočí se najednou) + kruhy v plátně; jedno kreslení."""

    def __init__(self) -> None:
        self.local: list[tuple[float, float]] = []
        self.local_col: list[tuple[float, ...]] = []
        self.parts: list[tuple[int, np.ndarray, np.ndarray]] = []  # (pořadí, vrcholy plátna, barvy)

    def quad(self, x0, y0, x1, y1, color) -> None:
        self.local += [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]
        self.local_col += [color] * 6

    def tri(self, pts, color) -> None:
        self.local += pts
        self.local_col += [color] * len(pts)

    def line(self, a, b, color, width: float) -> None:
        """Úsečka v místních souřadnicích (jen vodorovná/svislá – stačí pro hřeben a kříž)."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / n * width / 2, dx / n * width / 2
        self.tri([(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny),
                  (a[0] + nx, a[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)], color)

    def _world(self, verts: np.ndarray, cols: np.ndarray) -> None:
        self.parts.append((len(self.local), verts, cols))

    def disc(self, x, y, r, color, segments: int = 16) -> None:
        v = _disc_tris(x, y, r, segments)
        self._world(v, np.broadcast_to(np.asarray(color, np.float32), (len(v), 4)))

    def soft(self, x, y, r, color, segments: int = 20) -> None:
        col = np.tile(np.array([color, (*color[:3], 0.0), (*color[:3], 0.0)], np.float32), (segments, 1))
        self._world(_disc_tris(x, y, r, segments), col)

    def ring(self, x, y, r, color, width: float, segments: int = 24) -> None:
        u = _RINGS[segments]
        o, i = u * (r + width / 2) + (x, y), u * (r - width / 2) + (x, y)
        v = np.stack([o[:-1], o[1:], i[1:], o[:-1], i[1:], i[:-1]], 1).reshape(-1, 2)
        self._world(v, np.broadcast_to(np.asarray(color, np.float32), (len(v), 4)))

    def flush(self, p, cx: float, cy: float, c: float, s: float) -> None:
        loc = np.asarray(self.local, np.float32).reshape(-1, 2)
        lw = np.stack([cx + loc[:, 0] * c - loc[:, 1] * s, cy + loc[:, 0] * s + loc[:, 1] * c], 1)
        lc = np.asarray(self.local_col, np.float32).reshape(-1, 4)
        verts, cols, pos = [], [], 0
        for at, v, col in self.parts:  # zachovat pořadí kreslení (světlo pod domem, plameny nad ním)
            verts.append(lw[pos:at]); cols.append(lc[pos:at])
            verts.append(v); cols.append(col)
            pos = at
        verts.append(lw[pos:]); cols.append(lc[pos:])
        p.triangles(np.concatenate(verts), np.concatenate(cols))


def draw_house(p, h: eco.House, sx: float, sy: float, clock: float, saved_ring: bool = False,
               ring_color=(0.2, 1.0, 0.35, 0.95), damage: str = "water") -> None:
    """Domek se sedlovou střechou (kostel s věží); zatopený bliká červeně, zachráněný má zelený kruh.

    Neporušený dům má kolem sebe teplé světlo z oken, svítící vikýře a z komína mu stoupá kouř.
    ``damage="fire"``: zasažený dům (láva) je ohořelý a chvíli hoří. Kreslí se jedním voláním painteru."""
    cx, cy = h.x * sx, h.y * sy
    c, s = math.cos(h.angle), math.sin(h.angle)
    L, W = h.length * sx / 2, h.width * sy / 2
    b = _HouseBatch()

    def to_world(x, y):
        return cx + x * c - y * s, cy + x * s + y * c

    flooded = h.flooded
    since = clock - h.flooded_at
    seed = (h.x * 37.1 + h.y * 91.7) % 6.283
    if flooded and damage == "fire":
        roof_a, roof_b = (0.20, 0.16, 0.14, 1), (0.11, 0.09, 0.08, 1)
    elif flooded:
        roof_a, roof_b = (0.35, 0.45, 0.75, 1), (0.22, 0.30, 0.60, 1)
    elif h.church:
        roof_a, roof_b = (1.0, 0.80, 0.28, 1), (0.78, 0.52, 0.12, 1)
    else:
        roof_a, roof_b = (1.0, 0.34, 0.14, 1), (0.74, 0.17, 0.06, 1)
    R = max(L, W) * 1.5
    if saved_ring and not flooded:  # zachráněný: zelená záře pod domem
        b.soft(cx, cy, R * 1.8, (*ring_color[:3], 0.3 + 0.15 * math.sin(clock * 4 + seed)))
    if not flooded:  # teplé světlo z oken rozlité kolem domu (živé městečko)
        b.soft(cx, cy, R * 1.9, (1.0, 0.78, 0.30, 0.42 * (0.9 + 0.1 * math.sin(clock * 3.1 + seed))))
    elif damage != "fire":  # pod vodou: vlnky se rozbíhají od domu
        for k in range(2):
            f = (clock * 0.7 + k * 0.5 + seed) % 1.0
            b.ring(cx, cy, R * (0.9 + 0.9 * f), (0.85, 0.95, 1.0, 0.55 * (1 - f)), max(2.0, R * 0.1))
    # stín a obrys
    sh_x, sh_y = W * 0.35, W * 0.5  # stín posunutý v plátně -> v místních souřadnicích
    lx, ly = sh_x * c + sh_y * s, -sh_x * s + sh_y * c
    b.quad(-L + lx, -W + ly, L + lx, W + ly, (0, 0, 0, 0.45))
    o = 1.12
    b.quad(-L * o, -W * o, L * o, W * o, (0.1, 0.05, 0.02, 1))
    # dvě poloviny sedlové střechy (světlejší je ta ke světlu), tmavší okap, hřeben
    b.quad(-L, -W, L, 0, roof_a)
    b.quad(-L, 0, L, W, roof_b)
    b.quad(-L, -W, L, -W * 0.78, (roof_a[0] * 0.8, roof_a[1] * 0.8, roof_a[2] * 0.8, 1))
    b.line((-L, 0), (L, 0), (0.25, 0.08, 0.02, 1), max(1.5, W * 0.15))
    if not flooded:  # svítící střešní okna (vikýře)
        ww = max(W * 0.28, 2.5)
        for fx in ((-0.45, 0.2) if not h.church else (-0.55, -0.1)):
            b.quad(L * fx, -W * 0.62, L * fx + ww, -W * 0.62 + ww, (1.0, 0.93, 0.45, 1))
    if h.church:
        t = W * 0.95  # věž s křížem
        tx = L + t * 0.9
        b.quad(tx - t, -t, tx + t, t, (0.1, 0.05, 0.02, 1))
        t2 = t * 0.82
        b.tri([(tx, 0), (tx - t2, -t2), (tx + t2, -t2), (tx, 0), (tx + t2, -t2), (tx + t2, t2)], roof_a)
        b.tri([(tx, 0), (tx + t2, t2), (tx - t2, t2), (tx, 0), (tx - t2, t2), (tx - t2, -t2)], roof_b)
        b.line((tx - t * 0.5, 0), (tx + t * 0.5, 0), (1, 1, 1, 1), max(2, t * 0.2))
        b.line((tx, -t * 0.5), (tx, t * 0.5), (1, 1, 1, 1), max(2, t * 0.2))
    else:
        b.quad(L * 0.35, -W * 0.75, L * 0.6, -W * 0.4, (0.3, 0.12, 0.08, 1))  # komín
        if not flooded:  # kouř z komína: obláčky stoupají a vítr je odnáší
            ch_x, ch_y = to_world(L * 0.47, -W * 0.58)
            for k in range(3):
                f = (clock * 0.35 + k / 3 + seed) % 1.0
                b.soft(ch_x + f * R * 1.6, ch_y - f * R * 1.1, R * (0.25 + 0.45 * f),
                       (0.92, 0.92, 0.95, 0.5 * (1 - f) * min(1.0, f * 6)), segments=12)
    if flooded and damage == "fire":
        # plameny: blikající oranžová záře, po chvíli jen doutná
        k = max(0.25, 1.0 - since / 6.0)
        for i, (dx, dy) in enumerate(((0, 0), (0.5, -0.3), (-0.5, 0.2))):
            fl = 0.6 + 0.4 * math.sin(clock * (9 + i * 3) + i)
            b.disc(cx + dx * L, cy + dy * W, R * 0.55 * fl, (1.0, 0.45 + 0.3 * fl, 0.05, 0.75 * k), 16)
            b.disc(cx + dx * L, cy + dy * W, R * 0.25 * fl, (1.0, 0.95, 0.5, 0.9 * k), 12)
    elif flooded:
        # červený blikající kruh + šplouchnutí
        blink = 0.55 + 0.45 * math.sin(clock * 7)
        b.ring(cx, cy, R, (1.0, 0.1, 0.1, blink), max(3, R * 0.14))
        if since < 1.5:
            f = since / 1.5
            b.ring(cx, cy, R * (1 + 1.5 * f), (0.8, 0.95, 1, 1 - f), max(2, R * 0.12))
    elif saved_ring:
        pulse = 0.5 + 0.5 * math.sin(clock * 4 + seed)
        b.ring(cx, cy, R * (1 + 0.08 * pulse), ring_color, max(3, R * 0.16))
    b.flush(p, cx, cy, c, s)


class TownGame(Game):
    id = "mesto"
    title = "Záchrana města"
    description = "Z hor přijde povodeň. Postav hráze a kanály a zachraň co nejvíc domů."
    order = 80
    category = "soutěž"
    intro = "Za chvíli přijde povodeň – postav z písku hráze a kanály, ať voda obteče městečko!"
    keys = [
        ("Enter", "spustit povodeň hned / další kolo"),
        ("M", "nové město jinde"),
        ("↑ / ↓", "síla povodně"),
        ("Backspace", "začít znovu"),
    ]

    PREP_TIME = 25.0  # [s] na stavbu hrází
    FLOOD_TIME = 30.0  # [s] délka povodně
    RESULT_TIME = 10.0  # [s] zobrazení výsledku
    N_HOUSES = 10
    FLOOD_DEPTH = 0.003  # [m] voda, která zatopí dům
    INFILTRATION = 0.0003  # [m/s] vsakování do písku (bedna má stěny, voda neodtéká)

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.sim: WaterSim | None = None
        self.tex = None
        self.rng = np.random.default_rng()
        self.strength = 1.0
        self.clock = 0.0
        self.houses: list[eco.House] = []
        self.source: tuple[float, float] | None = None
        self.town: tuple[float, float] | None = None
        self.prev_town: tuple[float, float] | None = None
        self.phase = PREP
        self.phase_t = 0.0
        self.rounds = 0
        self.last_score: tuple[int, int] | None = None
        self._need_town = True
        self.planner: eco.FloodPlanner | None = None
        # jen vzhled: bouře nad pramenem, ohňostroj, obyvatelé
        self.rain = RainFx(seed=1)
        self.sparks = Sparks()
        self.walk = 0.0
        self.flash = 0.0
        self._next_flash = 4.0
        self._next_fw = 0.0

    def stop(self) -> None:
        self.prog.release()
        if self.tex is not None:
            self.tex.release()
        self.tex = None

    def reset(self) -> None:
        """Nové kolo: nové město (jinde než minule), suchá krajina, odpočet."""
        self._need_town = True

    # --- logika -----------------------------------------------------------------------------
    def _house_size(self) -> float:
        return 0.022 * self.host.box_size[0]

    def _flow(self, hm) -> float:
        """Průtok pramene [m³/s] – úměrný ploše bedny (za povodeň ~ 5 mm vody na celé pískoviště)."""
        return 0.005 * hm.size[0] * hm.size[1] / self.FLOOD_TIME * self.strength

    def _new_round(self, hm) -> None:
        """Začne plánovat nové kolo: zkušební povodně z několika pramenů (po kouscích, viz _plan)."""
        if self.sim is not None:
            self.sim.clear()
        factor = max(1, hm.shape[1] // 64)
        sources = eco.flood_source_candidates(hm.heights, hm.size, self.rng, factor=factor, n=3)
        self.planner = eco.FloodPlanner(hm.heights, hm.size, sources, self._flow(hm), self.FLOOD_TIME * 0.8,
                                        factor=factor, threshold=self.FLOOD_DEPTH, infiltration=self.INFILTRATION)
        self.houses = []
        self.source = self.town = None
        self.phase, self.phase_t = PLAN, 0.0
        self._need_town = False

    def _plan(self, hm) -> None:
        assert self.planner is not None
        if not self.planner.step(40):
            return
        house = self._house_size()
        town_r = house * 4.2
        wl = self.host.settings.map.water_level
        res = self.planner.choose(self.rng, town_r, wl, avoid=self.prev_town)
        if res is None:  # voda nikam nedoteče (např. rovný písek) -> město po proudu podle D8
            source = self.planner.sources[0]
            town = eco.place_town(hm.heights, hm.size, wl, self.rng, source, town_r, avoid=self.prev_town,
                                  factor=max(1, hm.shape[1] // 64))
        else:
            source, town = res
        self.source, self.town = source, town
        self.prev_town = town
        self.houses = eco.layout_houses(town, self.N_HOUSES, house, hm.size, self.rng)
        self.planner = None
        self.phase, self.phase_t = PREP, 0.0

    def update(self, hm, dt: float) -> None:
        if self.sim is None or self.sim.depth.shape != hm.shape:
            self.sim = WaterSim(hm.shape, hm.cell_size[0])
            self.tex = self.tex or WaterLayer(self.host)
            self._need_town = True
        if self._need_town:
            self._new_round(hm)
        dt = min(dt, 1 / 20)
        self.clock += dt
        self.phase_t += dt
        sim = self.sim
        if self.phase == PLAN:
            self._plan(hm)
        elif self.phase == PREP:
            if self.phase_t >= self.PREP_TIME:
                self.phase, self.phase_t = FLOOD, 0.0
                self.host.notify("Povodeň!")
        elif self.phase == FLOOD:
            assert self.source is not None
            eco.add_source(sim.depth, hm.size, self.source, 0.04, self._flow(hm) * dt)
            wet = eco.flooded_houses(sim.depth, self.houses, hm.size, self.FLOOD_DEPTH)
            for h, w in zip(self.houses, wet):
                if w and not h.flooded:
                    h.flooded, h.flooded_at = True, self.clock
            if self.phase_t >= self.FLOOD_TIME or all(h.flooded for h in self.houses):
                self._finish()
        elif self.phase == RESULT and self.phase_t >= self.RESULT_TIME:
            self._new_round(hm)
        sim.advance(hm.heights, dt, 9.81)
        # vsakování; po povodni voda opadá rychleji
        eco.soak(sim.depth, (self.INFILTRATION if self.phase == FLOOD else 0.0005) * dt)
        assert self.tex is not None
        self.tex.write(sim)
        self._update_fx(hm, dt)

    def _update_fx(self, hm, dt: float) -> None:
        """Jen vzhled: déšť z bouřkového mraku nad pramenem, blesky, ohňostroj, chůze obyvatel."""
        storm = self._storm()
        self.rain.update(dt, [(self.source[0], self.source[1], storm)] if self.source and storm > 0 else [])
        self.walk += dt * (2.6 if self.phase == FLOOD else 1.0)
        self.flash = max(0.0, self.flash - dt * 5)
        if self.phase == FLOOD and self.clock >= self._next_flash:
            self.flash = 1.0
            self._next_flash = self.clock + float(self.rng.uniform(3.0, 7.0))
        self.sparks.update(dt)
        if (self.phase == RESULT and self.last_score and self.last_score[0] >= 0.8 * self.last_score[1]
                and self.town is not None and self.clock >= self._next_fw and self.phase_t < self.RESULT_TIME - 2):
            self._next_fw = self.clock + 0.45
            cols = [(1, 0.85, 0.2), (1, 0.3, 0.5), (0.3, 1, 1), (0.6, 1, 0.3), (1, 1, 1)]
            r = self._house_size() * 5
            self.sparks.burst(self.town[0] + float(self.rng.uniform(-r, r)), self.town[1] + float(self.rng.uniform(-r, r)),
                              60, 0.55, cols, life=1.5, rng=self.rng)

    def _storm(self) -> float:
        """0..1 síla bouře nad pramenem: chvíli před povodní se zatahuje, během povodně leje."""
        if self.phase == FLOOD:
            return 1.0
        if self.phase == PREP:
            return float(np.clip((self.phase_t - (self.PREP_TIME - 6.0)) / 6.0, 0.0, 1.0)) * 0.5
        return 0.0

    def _finish(self) -> None:
        saved = sum(not h.flooded for h in self.houses)
        self.last_score = (saved, len(self.houses))
        self.rounds += 1
        self.phase, self.phase_t = RESULT, 0.0

    # --- kreslení ------------------------------------------------------------------------------
    def render(self) -> None:
        if self.tex is None or not self.tex.ready:
            return
        bw, bd = self.host.box_size
        roads = np.zeros((12, 4), np.float32)
        if self.houses:
            c0 = self.houses[0]
            for i, h in enumerate(self.houses[1:13]):
                roads[i] = (c0.x / bw, c0.y / bd, h.x / bw, h.y / bd)
        try:
            self.prog["u_roads"].write(roads.tobytes())  # type: ignore[union-attr]
        except KeyError:
            pass
        alarm = 0.0
        if self.phase == PREP and self.PREP_TIME - self.phase_t <= 5:
            alarm = 0.8
        result = (0.0, 0.0, 0.0)
        if self.phase == RESULT and self.last_score is not None:
            pct = self.last_score[0] / max(self.last_score[1], 1)
            col = (0.25, 1.0, 0.35) if pct >= 0.8 else (1.0, 0.85, 0.2) if pct >= 0.4 else (1.0, 0.3, 0.2)
            k = min(1.0, self.phase_t * 2) * 0.6
            result = tuple(v * k for v in col)
        self.host.draw_fullscreen(self.prog, **self.tex.uniforms(1), u_clock=float(self.clock),
                                  u_nroads=max(0, min(12, len(self.houses) - 1)), u_alarm=alarm,
                                  u_storm=self._storm(), u_flash=float(self.flash), u_result=result)
        sx, sy = self._px()
        self._draw_source(sx, sy)
        self._draw_people(sx, sy)
        self._draw_houses(sx, sy)
        self.sparks.draw(self.host, size_px=self.host.canvas_size[1] / 60)
        if self.source is not None and self._storm() > 0:
            self.rain.draw(self.host, [(self.source[0], self.source[1], self._storm())])
        self.host.glow(0.8, 0.82)
        self._draw_hud()

    def _draw_people(self, sx: float, sy: float) -> None:
        """Obyvatelé chodí po ulicích mezi kostelem a domy; při povodni pobíhají, ze zatopených domů utečou."""
        if len(self.houses) < 2:
            return
        p = self.host.painter
        c0 = self.houses[0]
        pts, cols, rads = [], [], []
        palette = [(1, 0.3, 0.3), (0.3, 0.75, 1), (1, 0.9, 0.2), (0.9, 0.4, 1), (0.3, 1, 0.5), (1, 0.6, 0.15)]
        r = max(4.0, 0.0065 * sx)
        for i in range(14):
            h = self.houses[1 + i % (len(self.houses) - 1)]
            if c0.flooded and h.flooded:
                continue
            ph = i * 1.93
            f = 0.5 + 0.5 * math.sin(self.walk * (0.5 + 0.13 * (i % 4)) + ph)
            f = 0.15 + 0.7 * f
            if h.flooded:  # utíkají ke kostelu
                f *= max(0.0, 1.0 - (self.clock - h.flooded_at) * 0.8)
            elif c0.flooded:
                f = 0.15 + 0.7 * max(f, min(1.0, (self.clock - c0.flooded_at) * 0.8))
            x, y = c0.x + (h.x - c0.x) * f, c0.y + (h.y - c0.y) * f
            side = 0.006 if i % 2 else -0.006  # chodí po krajích ulice
            dx, dy = h.x - c0.x, h.y - c0.y
            n = math.hypot(dx, dy) or 1.0
            x, y = x - dy / n * side, y + dx / n * side
            jump = 1.0
            if self.phase == RESULT and not h.flooded:
                jump = 1.0 + 0.35 * abs(math.sin(self.clock * 7 + i))  # radost: poskakují
            pts.append((x * sx, y * sy))
            cols.append(palette[i % len(palette)])
            rads.append(r * jump)
        if not pts:
            return
        P, R_ = np.array(pts), np.array(rads)
        ang = np.linspace(0, 2 * math.pi, 13)
        ring = np.stack([np.cos(ang), np.sin(ang)], 1)

        def discs(centers, radii, colors):
            q = centers[:, None, :] + radii[:, None, None] * ring[None]
            cc = np.repeat(centers[:, None, :], 12, 1)
            tris = np.stack([cc, q[:, :-1], q[:, 1:]], 2).reshape(-1, 2)
            return tris, np.repeat(np.asarray(colors, np.float32), 36, 0)

        k = len(P)
        p.triangles(*discs(P + R_[:, None] * 0.35, R_ * 1.25, np.tile([0, 0, 0, 0.5], (k, 1))))  # stín
        p.triangles(*discs(P, R_ * 1.2, np.tile([0.1, 0.05, 0.02, 1], (k, 1))))
        p.triangles(*discs(P, R_, np.column_stack([np.array(cols), np.ones(k)])))
        p.triangles(*discs(P - R_[:, None] * 0.15, R_ * 0.5, np.tile([1.0, 0.85, 0.7, 1], (k, 1))))  # hlava

    def _px(self) -> tuple[float, float]:
        cw, ch = self.host.canvas_size
        bw, bd = self.host.box_size
        return cw / bw, ch / bd

    def _draw_source(self, sx: float, sy: float) -> None:
        if self.source is None:
            return
        p = self.host.painter
        x, y = self.source[0] * sx, self.source[1] * sy
        r = 0.04 * sx
        pulse = 0.5 + 0.5 * math.sin(self.clock * 5)
        if self.phase == PREP:
            for k in range(3):  # rozbíhající se kruhy
                f = (self.clock * 0.6 + k / 3) % 1.0
                p.circle(x, y, r * (0.5 + 1.2 * f), (0.3, 0.8, 1.0, 1.0 - f), width=0.006 * sx)
        elif self.phase == FLOOD:  # pramen chrlí vodu: rychlé bílé vlny
            _soft_disc(p, x, y, r * 2.2, (0.6, 0.9, 1.0, 0.55))
            for k in range(4):
                f = (self.clock * 1.4 + k / 4) % 1.0
                p.circle(x, y, r * (0.6 + 2.2 * f), (0.9, 0.97, 1.0, 0.9 * (1 - f)), width=0.008 * sx * (1 - f) + 2)
        p.circle(x, y, r * 0.55, (0.05, 0.35, 0.95, 0.9))
        p.circle(x, y, r * (0.55 + 0.1 * pulse), (1, 1, 1, 0.9), width=0.004 * sx)
        # kapka
        d = r * 0.32
        p.triangles(np.array([[x, y - d * 1.5], [x + d * 0.75, y], [x - d * 0.75, y]]), (1, 1, 1, 1))
        p.circle(x, y + d * 0.15, d * 0.78, (1, 1, 1, 1), segments=16)
        if self.phase == PREP:
            size = self._text_size(0.9)
            # popisek ke středu bedny, ať nevyleze z plátna
            cw, ch = self.host.canvas_size
            ty = y + r * 1.6 if y < ch / 2 else y - r * 1.6
            anchor = ("t" if y < ch / 2 else "b") + ("l" if x < cw * 0.25 else "r" if x > cw * 0.75 else "c")
            p.text("Odtud přijde voda!", x, ty, size=size, anchor=anchor, bold=True, color=(0.6, 0.9, 1, 1),
                   outline=3, outline_color=(0, 0.05, 0.2, 1))

    def _draw_houses(self, sx: float, sy: float) -> None:
        p = self.host.painter
        for h in self.houses:
            self._draw_house(p, h, sx, sy)

    def _draw_house(self, p, h: eco.House, sx: float, sy: float) -> None:
        draw_house(p, h, sx, sy, self.clock, saved_ring=self.phase == RESULT)

    def _text_size(self, scale: float = 1.0) -> int:
        ch = self.host.canvas_size[1]
        return max(16, int(ch / 24 * scale))

    def _draw_hud(self) -> None:
        p = self.host.painter
        cw, ch = self.host.canvas_size
        size = self._text_size()
        n = len(self.houses)
        saved = sum(not h.flooded for h in self.houses)
        if self.phase == PLAN:
            hud.label(p, "Hledám místo pro nové městečko…", cw / 2, size * 0.4, size, anchor="tc")
        elif self.phase == PREP:
            left = max(0, math.ceil(self.PREP_TIME - self.phase_t))
            hud.label(p, f"Povodeň za {left} s! Postav hráze a ochraň město.", cw / 2, size * 0.4, size,
                      anchor="tc", color=hud.YELLOW)
        elif self.phase == FLOOD:
            left = max(0, math.ceil(self.FLOOD_TIME - self.phase_t))
            hud.label(p, f"Povodeň! Ještě {left} s · domy v suchu: {saved}/{n}", cw / 2, size * 0.4, size,
                      anchor="tc", color=(0.55, 0.9, 1, 1))
        elif self.phase == RESULT and self.last_score is not None:
            s, total = self.last_score
            big = int(size * 1.7)
            pct = 100 * s / max(total, 1)
            if pct >= 80:
                verdict, col = "Skvělá práce, město je v bezpečí!", (0.35, 1.0, 0.45, 1)
            elif pct >= 40:
                verdict, col = "Dobrý pokus – zkuste hráz postavit dřív.", (1.0, 0.9, 0.3, 1)
            else:
                verdict, col = "Voda byla silnější. Zkuste to znovu!", (1.0, 0.45, 0.35, 1)
            p.rect(0, ch / 2 - big * 1.4, cw, big * 2.9, (0, 0, 0, 0.6))
            p.rect(0, ch / 2 - big * 1.4, cw, 5, col)
            p.rect(0, ch / 2 + big * 1.5 - 5, cw, 5, col)
            # hvězdy hodnocení (1–3), vyskakují postupně
            stars = 3 if pct >= 80 else 2 if pct >= 40 else 1 if s > 0 else 0
            sr = big * 0.75
            for k in range(3):
                appear = self.phase_t - 0.4 - 0.35 * k
                if appear < 0:
                    continue
                pop = 1.0 + 0.5 * math.exp(-appear * 6) * math.cos(appear * 20)
                x = cw / 2 + (k - 1) * sr * 2.6
                y = ch / 2 - big * 1.4 - sr * 1.15 - (sr * 0.25 if k == 1 else 0)
                rot = -math.pi / 2
                if k < stars:
                    _soft_disc(p, x, y, sr * 2.0 * pop, (1, 0.8, 0.2, 0.5))
                    p.triangles(star_tris(x, y, sr * 1.12 * pop, sr * 0.5 * pop, 5, rot), (0.35, 0.15, 0.0, 1))
                    p.triangles(star_tris(x, y, sr * pop, sr * 0.42 * pop, 5, rot), (1, 0.85, 0.15, 1))
                    p.triangles(star_tris(x - sr * 0.08, y - sr * 0.1, sr * 0.5 * pop, sr * 0.2 * pop, 5, rot),
                                (1, 1, 0.7, 1))
                else:
                    p.triangles(star_tris(x, y, sr * 1.12, sr * 0.5, 5, rot), (0.0, 0.0, 0.0, 0.7))
                    p.triangles(star_tris(x, y, sr, sr * 0.42, 5, rot), (0.35, 0.35, 0.4, 0.9))
            p.text(f"Zachránili jste {s}/{total} {self._domu(total)}!", cw / 2, ch / 2 - big * 0.2, size=big,
                   anchor="bc", bold=True, color=col, outline=4)
            p.text(verdict, cw / 2, ch / 2 + big * 0.2, size=size, anchor="tc", bold=True, color=(1, 1, 1, 1),
                   outline=3)

    @staticmethod
    def _domu(n: int) -> str:
        return "dům" if n == 1 else "domy" if 2 <= n <= 4 else "domů"

    # --- ovládání ---------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name in ("enter", "return", "space"):
            if self.phase == PLAN:
                return True
            if self.phase == PREP:
                self.phase, self.phase_t = FLOOD, 0.0
                self.host.notify("Povodeň!")
            elif self.phase == RESULT:
                self._need_town = True
            else:
                self._finish()
        elif name in ("m", "backspace"):
            self._need_town = True
            self.host.notify("Nové město")
        elif name in ("up", "down"):
            self.strength = float(np.clip(self.strength * (1.25 if name == "up" else 0.8), 0.3, 4.0))
            self.host.notify(f"Síla povodně ×{self.strength:.2f}")
        else:
            return False
        return True

    def status(self) -> list[str]:
        n = len(self.houses)
        flooded = sum(h.flooded for h in self.houses)
        lines = [f"Fáze: {self.phase} ({self.phase_t:.0f} s)", f"Domy: {n - flooded}/{n} v suchu",
                 f"Síla povodně ×{self.strength:.2f}"]
        if self.sim is not None:
            lines.append(f"Voda: {self.sim.volume * 1000:.2f} l")
        if self.last_score is not None:
            lines.append(f"Minulé kolo: {self.last_score[0]}/{self.last_score[1]} (kol: {self.rounds})")
        return lines
