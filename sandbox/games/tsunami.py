"""Mega tsunami: obří vlna přijde z moře a prožene se krajinou z písku.

Návštěvníci postaví pobřeží, údolí a hráze. Na nejnižší straně bedny je moře,
na pevnině stojí města. Každých ~30 s (nebo klávesou T) přijde po odpočtu obří
vlna: okraj bedny na straně moře se na pár sekund zvedne a voda se s hybností
vřítí dovnitř (mělká voda ``WaterSim``). Pěna je tam, kde je voda rychlá a mělká.
Pak voda odteče zpět do moře a ukáže se, kolik pevniny a měst zatopila.
"""

from __future__ import annotations

import math

import cv2
import moderngl
import numpy as np

from ..sim.lava import blur3, downsample
from ..sim.water import WaterSim
from ..render import hud
from ..render.text import get_font
from .base import Game

FRAG = """
uniform sampler2D u_state;   // r = hloubka vody, g = pěna, b = mokrý písek, a = zatopeno při vlně
uniform sampler2D u_vel;     // rg = rychlost vody [m/s], b = vzdutí hladiny nad moře [m], a = čelo vlny 0..1
uniform float u_sea;         // hladina moře [m]
uniform float u_flash;       // 0..1 poplach (plynulý pulz)
uniform vec4 u_edge;         // strana moře (levá, horní, pravá, dolní) – jedna jednička
uniform float u_show_flood;  // 0..1 vyznačit zatopenou pevninu
uniform float u_warn;        // 0..1 průběh varování (vlna roste na obzoru)

float tsu_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float tsu_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(tsu_hash(i), tsu_hash(i + vec2(1, 0)), u.x),
               mix(tsu_hash(i + vec2(0, 1)), tsu_hash(i + vec2(1, 1)), u.x), u.y);
}
// dvě pootočené vrstvy šumu – bez čtverečkové struktury hodnotového šumu
float tsu_noise2(vec2 p) {
    return 0.5 * (tsu_noise(p) + tsu_noise(mat2(0.8, -0.6, 0.6, 0.8) * p * 1.37 + 17.3));
}
// pěna unášená proudem (dvě fáze se střídají, ať se vzor netrhá); velké chuchvalce, ne zrnitý šum
float tsu_foam(vec2 pm, vec2 vel) {
    const float P = 1.4;
    float p0 = fract(u_time / P), p1 = fract(u_time / P + 0.5);
    float a = tsu_noise2((pm - vel * p0 * P) * 26.0) * 0.65 + tsu_noise2((pm - vel * p0 * P) * 9.0 + 3.1) * 0.35;
    float b = tsu_noise2((pm - vel * p1 * P) * 26.0 + 11.7) * 0.65 + tsu_noise2((pm - vel * p1 * P) * 9.0 + 8.3) * 0.35;
    return mix(a, b, abs(p0 * 2.0 - 1.0));
}

// bikubická interpolace všech 4 kanálů (hladké okraje vody i na hrubé mřížce simulace)
vec4 tsu_bicubic4(sampler2D tex, vec2 uv) {
    vec2 size = vec2(textureSize(tex, 0));
    vec2 inv = 1.0 / size;
    vec2 tc = uv * size - 0.5;
    vec2 f = fract(tc);
    tc -= f;
    vec4 xc = _cubic(f.x);
    vec4 yc = _cubic(f.y);
    vec4 c = tc.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xc.xz + xc.yw, yc.xz + yc.yw);
    vec4 off = (c + vec4(xc.yw, yc.yw) / s) * inv.xxyy;
    vec4 s0 = texture(tex, off.xz), s1 = texture(tex, off.yz), s2 = texture(tex, off.xw), s3 = texture(tex, off.yw);
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    return mix(mix(s3, s2, sx), mix(s1, s0, sx), sy);
}

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;
    float h = height_at(uv);
    vec4 st = max(tsu_bicubic4(u_state, uv), 0.0);
    float w = st.r;
    float foam = st.g;
    float wet = st.b;

    // vzdálenost od strany moře [m] a souřadnice podél pobřeží
    float d = 10.0, along = pm.y;
    if (u_edge.x > 0.5) d = pm.x;
    if (u_edge.y > 0.5) { d = pm.y; along = pm.x; }
    if (u_edge.z > 0.5) d = u_box.x - pm.x;
    if (u_edge.w > 0.5) { d = u_box.y - pm.y; along = pm.x; }

    // --- pevnina ---
    float e = (h - u_sea) / max(u_elev_max - u_sea, 1e-4);
    vec3 c = land_color(e);
    c *= mix(0.55, 1.2, hillshade(uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.45, cl.y * 0.6 + cl.x * 0.25);
    c = mix(c, c * vec3(0.45, 0.42, 0.40), wet * 0.7);            // mokrý písek po vlně

    // zatopená pevnina po vlně: bahnitá a rudě prosvícená, pruhy pomalu plují, žlutý okraj zátopy
    float fl = u_show_flood * smoothstep(0.25, 0.6, st.a);
    vec3 mud = c * vec3(0.55, 0.48, 0.42) + vec3(0.10, 0.04, 0.02);
    float stripe = smoothstep(0.18, 0.26, abs(fract((pm.x + pm.y) * 11.0 - u_time * 0.25) - 0.5));
    vec3 red = mix(vec3(0.95, 0.12, 0.08), vec3(1.0, 0.32, 0.18), stripe) * (0.92 + 0.08 * sin(u_time * 2.2));
    c = mix(c, mix(mud, red, 0.62), fl);
    float ga = st.a - 0.5;
    float border = 1.0 - smoothstep(1.4, 2.8, abs(ga) / max(fwidth(ga), 1e-4));
    c = mix(c, vec3(1.0, 0.85, 0.25), border * u_show_flood);

    // --- voda: sdílené vykreslení + velká pěna vlny ---
    vec4 vt = texture(u_vel, uv);
    vec2 vel = vt.rg;
    float eta = max(vt.b, 0.0);
    float front = vt.a;
    float n = tsu_foam(pm, vel);
    float f = smoothstep(0.45, 0.8, foam * (0.1 + 1.05 * n));
    vec4 wr = water_shade(c, uv, w, vel, f);
    vec3 col = mix(wr.rgb, wr.rgb * vec3(0.78, 0.85, 0.95), smoothstep(0.04, 0.22, w) * wr.a);  // širé moře tmavší

    // klidné moře: dlouhé vlny (vzdouvání) běží k pobřeží
    float swell = sin(d * 42.0 - u_time * 1.6 + tsu_noise(vec2(along * 6.0, 1.0)) * 3.0);
    col += vec3(0.05, 0.10, 0.12) * smoothstep(0.4, 1.0, swell) * smoothstep(0.01, 0.04, w) * (1.0 - u_show_flood);

    // vlna: vzdutá voda je tyrkysová a prosvětlená, její svah stíněný, hřeben bílý
    vec2 e2 = 1.0 / vec2(textureSize(u_vel, 0));
    float ex = texture(u_vel, uv + vec2(e2.x, 0.0)).b - texture(u_vel, uv - vec2(e2.x, 0.0)).b;
    float ey = texture(u_vel, uv + vec2(0.0, e2.y)).b - texture(u_vel, uv - vec2(0.0, e2.y)).b;
    vec2 sg = vec2(ex / (2.0 * e2.x * u_box.x), ey / (2.0 * e2.y * u_box.y));
    vec3 nrm = normalize(vec3(-sg * 5.0, 1.0));
    float lit = clamp(dot(nrm, normalize(vec3(-1.0, -1.2, 1.6))), 0.0, 1.0);
    sg *= smoothstep(0.03, 0.07, d);  // u okraje bedny simulace vlnu „vhání“ – tam nestínovat
    float body = smoothstep(0.004, 0.05, eta) * wr.a;
    col = mix(col, vec3(0.05, 0.66, 0.78), body * 0.6);
    col *= mix(1.0, 0.62 + 0.6 * lit, smoothstep(0.002, 0.02, eta) * wr.a);
    float crest = clamp(front * 1.8, 0.0, 1.0) * smoothstep(0.03, 0.06, d);  // ne na okraji bedny
    float lace = smoothstep(0.42, 0.75, crest * (0.1 + 1.05 * n));
    vec3 white = mix(vec3(0.55, 0.85, 0.95), vec3(0.97, 1.0, 1.0), smoothstep(0.45, 0.8, n));  // chuchvalce pěny
    col = mix(col, white, lace * wr.a);

    // varování: na obzoru roste vlna – tmavé údolí a bílý hřeben, který se zvedá
    if (u_warn > 0.0) {
        float r0 = 0.03 + 0.05 * u_warn;
        float sea = smoothstep(0.0005, 0.003, w);
        float ridge = exp(-pow((d - r0) / 0.022, 2.0));
        float trough = exp(-pow((d - r0 - 0.05) / 0.035, 2.0));
        float wob = tsu_noise2(vec2(along * 9.0, u_time * 0.8));
        col = mix(col, col * vec3(0.35, 0.5, 0.6), trough * u_warn * 0.7 * sea);
        col = mix(col, vec3(0.08, 0.70, 0.80), ridge * u_warn * sea * 0.8);
        col = mix(col, vec3(0.97, 1.0, 1.0), smoothstep(0.45, 0.9, ridge * (0.6 + 0.6 * wob)) * u_warn * sea);
    }
    c = col;

    // poplach: pás u moře s šipkami (chevrony) mířícími na pevninu + rudé okraje, plynulý pulz
    float band = 1.0 - smoothstep(0.05, 0.13, d);
    float chev = step(0.0, u_warn - 0.001) * step(0.5, fract((d + abs(fract(along * 3.5) - 0.5) * 0.18) * 7.0 - u_time * 0.9));
    vec3 alarm = mix(vec3(1.0, 0.06, 0.02), vec3(1.0, 0.78, 0.10), chev);
    float frame = 1.0 - smoothstep(0.0, 0.07, min(min(pm.x, u_box.x - pm.x), min(pm.y, u_box.y - pm.y)));
    c = mix(c, alarm, u_flash * band * 0.65);
    c = mix(c, vec3(1.0, 0.05, 0.02), u_flash * frame * 0.6);
    f_color = vec4(c, 1.0);
}
"""

CALM, WARN, WAVE, RECEDE, RESULT = "klid", "varování", "vlna", "ústup", "výsledek"
DURATION = {CALM: 22.0, WARN: 3.0, WAVE: 5.0, RECEDE: 7.0, RESULT: 7.0}
NEXT = {CALM: WARN, WARN: WAVE, WAVE: RECEDE, RECEDE: RESULT, RESULT: CALM}
FIRST_CALM = 4.0  # po spuštění/resetu přijde první vlna brzy
DS = 2  # zmenšení mřížky pro simulaci vody
EDGES = ["vlevo", "nahoře", "vpravo", "dole"]
WET = 0.002  # [m] od jaké hloubky je buňka zaplavená


def edge_slices(edge: int, width: int) -> tuple[slice, slice]:
    """Řezy (řádky, sloupce) pásu u dané strany bedny (0 levá, 1 horní, 2 pravá, 3 dolní)."""
    width = max(1, width)
    if edge == 0:
        return slice(None), slice(None, width)
    if edge == 1:
        return slice(None, width), slice(None)
    if edge == 2:
        return slice(None), slice(-width, None)
    return slice(-width, None), slice(None)


def lowest_edge(terrain: np.ndarray, frac: float = 0.12) -> int:
    """Strana bedny s nejnižším terénem – tam bude moře."""
    gh, gw = terrain.shape
    means = [float(terrain[edge_slices(e, int(round((gw if e % 2 == 0 else gh) * frac)))].mean()) for e in range(4)]
    return int(np.argmin(means))


def sea_level_for(terrain: np.ndarray, edge: int, frac: float = 0.12) -> float:
    """Hladina moře: tak, aby moře zalilo nižší polovinu pobřežního pásu (min. 1/8 strany)."""
    gh, gw = terrain.shape
    band = terrain[edge_slices(edge, int(round((gw if edge % 2 == 0 else gh) * frac)))]
    return float(np.percentile(band, 50)) + 0.004


def sea_fill(terrain: np.ndarray, sea: float, edge: int, width: int = 2) -> np.ndarray:
    """Hloubka moře: buňky pod hladinou spojené s okrajem moře (vnitrozemská jezera ne)."""
    mask = (terrain < sea).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=4)
    ids = np.unique(labels[edge_slices(edge, width)])
    ids = ids[ids > 0]
    if len(ids) == 0:
        return np.zeros_like(terrain, dtype=np.float32)
    connected = np.isin(labels, ids)
    return np.where(connected, sea - terrain, 0.0).astype(np.float32)


def inward_flux(sim: WaterSim, edge: int) -> np.ndarray:
    """Pole toku směrem od dané strany dovnitř bedny."""
    return (sim.fr, sim.fb, sim.fl, sim.ft)[edge]


def velocity(sim: WaterSim) -> np.ndarray:
    """Přibližná rychlost vody [m/s] v každé buňce (z toků v trubkách)."""
    gx = sim.fr - sim.fl
    gy = sim.fb - sim.ft
    q = np.sqrt(gx * gx + gy * gy)
    return q / (sim.cell * np.maximum(sim.depth, 0.004))


def place_towns(terrain: np.ndarray, sea: float, edge: int, rng: np.random.Generator, n_x: int = 4, n_y: int = 2,
                min_height: float = 0.008) -> list[tuple[int, int]]:
    """Města na pevnině: jedno v každém poli mřížky n_x × n_y, na nejvhodnějším místě.

    Vhodné = suché (nad mořem), rovné a ne u moře. Vrací [(iy, ix)].
    """
    gh, gw = terrain.shape
    gy, gx = np.gradient(terrain)
    slope = np.hypot(gx, gy)
    ok = terrain > sea + min_height
    margin = max(2, gw // 30)
    text = max(margin, gh // 7)  # nahoře a dole je text
    ok[:text, :] = ok[-text:, :] = False
    ok[:, :margin] = ok[:, -margin:] = False
    ok[edge_slices(edge, int(round((gw if edge % 2 == 0 else gh) * 0.18)))] = False
    towns = []
    for j in range(n_y):
        for i in range(n_x):
            ys = slice(j * gh // n_y, (j + 1) * gh // n_y)
            xs = slice(i * gw // n_x, (i + 1) * gw // n_x)
            cand = ok[ys, xs]
            if cand.sum() < 4:
                continue
            score = np.where(cand, -slope[ys, xs] + rng.uniform(0, 0.002, cand.shape), -np.inf)
            k = int(np.argmax(score))
            ty, tx = divmod(k, cand.shape[1])
            towns.append((ys.start + ty, xs.start + tx))
    return towns


def soft_discs(centers: np.ndarray, radii: np.ndarray, colors: np.ndarray, segments: int = 10):
    """Mnoho měkkých kruhů (plný střed, průhledný okraj) -> (vrcholy, barvy) pro Painter.triangles."""
    n = len(centers)
    ang = np.linspace(0, 2 * math.pi, segments + 1)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    pts = centers[:, None, :] + radii[:, None, None] * ring[None]
    c = np.repeat(centers[:, None, :], segments, axis=1)
    tris = np.stack([c, pts[:, :-1], pts[:, 1:]], axis=2).reshape(-1, 2)
    cols = np.repeat(colors, segments * 3, axis=0).reshape(n, segments, 3, 4).copy()
    cols[:, :, 1:, 3] *= 0.2
    return tris, cols.reshape(-1, 4)


class TsunamiGame(Game):
    id = "tsunami"
    title = "Mega tsunami"
    description = "Z moře přijde obří vlna a prožene se krajinou. Postav hráze a ochraň města!"
    order = 65
    category = "zážitek"
    intro = "Postav z písku hráze a kopce, ať obří vlna z moře nezaplaví města – vlna přijde každou půlminutu!"
    keys = [
        ("T", "vlna hned (odpočet)"),
        ("S", "moře na jiné straně"),
        ("C", "automatický cyklus zap/vyp"),
        ("↑ / ↓", "výška vlny"),
        ("X", "vysušit"),
    ]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.sim: WaterSim | None = None
        self.tex: moderngl.Texture | None = None
        self.vel_tex: moderngl.Texture | None = None
        self.rng = np.random.default_rng()
        self.auto = getattr(self, "auto", True)
        self.amplitude = getattr(self, "amplitude", 0.10)  # [m] výška vlny nad mořem
        self.edge_fixed: int | None = getattr(self, "edge_fixed", None)
        self.edge = 0
        self.sea = 0.0
        self.seafill: np.ndarray | None = None
        self.terrain: np.ndarray | None = None
        self.towns: list[tuple[int, int]] = []
        self.town_flooded: list[bool] = []
        self.waves = 0
        self.flooded_pct = 0.0
        self.last_result: tuple[float, int, int] | None = None
        self._refresh_t = 0.0
        self._planned = False
        self.spray = np.zeros((0, 6), np.float32)  # tříšť: x, y, vx, vy, věk, život [m, m/s, s]
        self.front = None
        self._set_phase(CALM, FIRST_CALM + self.host.settings.attract.intro_seconds)  # po úvodním slajdu

    def stop(self) -> None:
        self.prog.release()
        for tex in (self.tex, self.vel_tex):
            if tex is not None:
                tex.release()
        self.tex = self.vel_tex = None

    def reset(self) -> None:
        if self.sim is not None:
            self.sim.clear()
            self.wet[:] = 0
            self.foam[:] = 0
            self.flood[:] = 0
        self.seafill = None
        self.last_result = None
        self.flooded_pct = 0.0
        self._set_phase(CALM, FIRST_CALM)

    def _set_phase(self, phase: str, duration: float | None = None) -> None:
        self.phase = phase
        self.phase_t = 0.0
        self.phase_len = DURATION[phase] if duration is None else duration
        if phase == WAVE:
            self.waves += 1
            self._begin_wave()
        if phase == RESULT:
            self._finish_wave()
        if phase == CALM:
            self._planned = False  # nový cyklus -> nové pobřeží a města podle terénu

    # --- moře, města --------------------------------------------------------------
    def _plan_coast(self) -> None:
        """Jednou za cyklus (začátek klidu): strana moře, hladina a města.

        Pak se až do další vlny nemění – moře ani města nepřeskakují, když lidé stavějí."""
        assert self.terrain is not None
        t = self.terrain
        self.edge = lowest_edge(t) if self.edge_fixed is None else self.edge_fixed
        self.sea = sea_level_for(t, self.edge)
        rng = np.random.default_rng(self.waves * 7 + self.edge)
        self.towns = place_towns(t, self.sea, self.edge, rng)
        self.town_flooded = [False] * len(self.towns)
        self._planned = True
        self._refresh_coast()

    def _refresh_coast(self) -> None:
        """Moře podle aktuálního terénu (hráz u pobřeží moře zmenší) – strana a hladina zůstávají."""
        assert self.terrain is not None
        self.seafill = sea_fill(self.terrain, self.sea, self.edge)

    def _begin_wave(self) -> None:
        assert self.terrain is not None and self.sim is not None
        if not self._planned:
            self._plan_coast()
        self.land = (self.terrain > self.sea + 0.003) & (self.seafill <= 0)  # type: ignore[operator]
        self.flood[:] = 0
        self.town_flooded = [False] * len(self.towns)
        self.flooded_pct = 0.0

    def _finish_wave(self) -> None:
        saved = sum(1 for f in self.town_flooded if not f)
        self.last_result = (self.flooded_pct, saved, len(self.towns))

    # --- logika --------------------------------------------------------------------
    def update(self, hm, dt: float) -> None:
        dt = min(dt, 1 / 20)
        shape = (hm.shape[0] // DS, hm.shape[1] // DS)
        self.terrain = downsample(hm.heights, DS)[: shape[0], : shape[1]]
        if self.sim is None or self.sim.depth.shape != shape:
            self.sim = WaterSim(shape, hm.cell_size[0] * DS)
            self.foam = np.zeros(shape, np.float32)
            self.wet = np.zeros(shape, np.float32)
            self.flood = np.zeros(shape, np.float32)
            self.land = np.zeros(shape, bool)
            for tex in (self.tex, self.vel_tex):
                if tex is not None:
                    tex.release()
            self.tex = self.host.ctx.texture((shape[1], shape[0]), 4, dtype="f4")
            self.vel_tex = self.host.ctx.texture((shape[1], shape[0]), 4, dtype="f4")
            for tex in (self.tex, self.vel_tex):
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                tex.repeat_x = tex.repeat_y = False
            self.seafill = None
        sim = self.sim

        self.phase_t += dt
        self._sclock = getattr(self, "_sclock", 0.0) + dt  # hodiny pro rozestupy zvuků
        if self.phase_t >= self.phase_len:
            if self.phase == CALM and not self.auto:
                self.phase_t = 0.0
            else:
                self._set_phase(NEXT[self.phase])

        # v klidu sleduj, jak návštěvníci mění pobřeží (1× za sekundu)
        self._refresh_t -= dt
        if not self._planned:
            self._refresh_t = 1.0
            self._plan_coast()
        elif self.seafill is None or (self.phase in (CALM, WARN) and self._refresh_t <= 0):
            self._refresh_t = 1.0
            self._refresh_coast()
        assert self.seafill is not None

        terrain = self.terrain
        g = 9.81 * 0.35  # zpomalený záběr: vlna je dobře vidět
        rows, cols = edge_slices(self.edge, 3)
        if self.phase in (CALM, WARN, RESULT):
            # moře drží hladinu, vnitrozemská voda pomalu vysychá / vsakuje
            np.maximum(sim.depth - 0.004 * dt, self.seafill, out=sim.depth)
        elif self.phase == WAVE:
            k = self.phase_t / self.phase_len
            env = math.sin(math.pi * min(1.0, k * 1.15)) ** 0.7 if k < 0.87 else 0.0
            level = self.sea + self.amplitude * env
            band = terrain[rows, cols]
            sim.depth[rows, cols] = np.maximum(level - band, 0.0)
            speed = math.sqrt(g * max(self.amplitude, 0.01)) * env
            f = inward_flux(sim, self.edge)
            f[rows, cols] = speed * sim.depth[rows, cols] * sim.cell
        else:  # RECEDE – moře na okraji pohlcuje vracející se vodu
            sim.depth[rows, cols] = np.minimum(sim.depth[rows, cols], self.seafill[rows, cols])

        prev = sim.depth.copy()
        sim.advance(terrain, dt, g)
        if self.phase in (WAVE, RECEDE):
            # protější strana bedny vodu odvádí (vlna se od ní neodrazí zpět)
            sim.depth[edge_slices((self.edge + 2) % 4, 1)] = 0.0

        # pěna: rychlá a mělká voda + čelo vlny (hloubka rychle roste)
        v = velocity(sim)
        rise = np.clip(((sim.depth - prev) / max(dt, 1e-3) - 0.02) / 0.06, 0, 1)
        rise *= self.seafill <= 0  # čelo vlny = voda se valí na pevninu (ne celé vzdouvající se moře)
        shallow = np.clip(1.0 - sim.depth / 0.02, 0.0, 1.0)
        new_foam = np.clip((v - 0.18) / 0.3, 0, 1) * shallow
        # příboj: pruh vody u souše (na čele vlny silnější)
        wet = (sim.depth > 0.0015).astype(np.uint8)
        shore = (cv2.dilate(1 - wet, np.ones((3, 3), np.uint8)) > 0) & (wet > 0)
        surf = shore * np.clip(v / 0.2, 0.5, 1.0)
        new_foam = blur3(blur3(np.maximum(np.maximum(new_foam, rise), surf) * (sim.depth > 0.001)))
        self.front = blur3(blur3(rise))
        np.maximum(new_foam, self.foam * math.exp(-dt / 0.6), out=self.foam)
        self.foam *= sim.depth > 0.0008

        flooded_now = (sim.depth > WET) & (self.seafill <= 0)
        self.wet = np.maximum(self.wet * math.exp(-dt / 12.0), flooded_now.astype(np.float32))
        if self.phase in (WAVE, RECEDE):
            self.flood[flooded_now & self.land] = 1.0
            n_land = int(self.land.sum())
            self.flooded_pct = 100.0 * float(self.flood[self.land].sum()) / n_land if n_land else 0.0
            for i, (ty, tx) in enumerate(self.towns):
                if sim.depth[ty, tx] > 0.004:
                    if not self.town_flooded[i] and self._sclock - getattr(self, "_splash_t", -9.0) > 0.8:
                        self._splash_t = self._sclock  # víc měst naráz = jedno šplouchnutí
                        self.sfx.play("splash", 0.6, x=(tx + 0.5) / terrain.shape[1],
                                      pitch=float(self.rng.uniform(0.9, 1.1)))
                    self.town_flooded[i] = True

        assert self.tex is not None
        self.tex.write(np.dstack([sim.depth, self.foam, self.wet, self.flood]).astype(np.float32).tobytes())
        vx, vy = sim.velocity(max_speed=1.0)
        # vzdutí hladiny nad moře (stínování vlny) a čelo vlny (bílý hřeben)
        # (na moři hladina nad klidným mořem, na pevnině hloubka valící se vody)
        eta = np.where(self.seafill > 0, terrain + sim.depth - self.sea, sim.depth)
        eta = np.where(sim.depth > 0.0015, np.clip(eta, 0.0, 0.15), 0.0)
        self.vel_tex.write(np.dstack([blur3(vx), blur3(vy), blur3(blur3(eta.astype(np.float32))),
                                      self.front]).astype(np.float32).tobytes())
        self._update_spray(dt, rise, vx, vy)
        self._sound(v, flooded_now)

    def _sound(self, v: np.ndarray, flooded_now: np.ndarray) -> None:
        """Příboj v klidu, siréna při varování, náraz vlny, hukot vody na pevnině, výsledek."""
        sfx = self.sfx
        x = (0.1, 0.5, 0.9, 0.5)[self.edge]  # zvuk moře zní ze strany moře
        prev = getattr(self, "_snd_phase", None)
        self._snd_phase = self.phase
        if prev is not None and prev != self.phase:
            if self.phase == WARN:
                sfx.play("alarm", 0.9, x=x)
            elif self.phase == WAVE:
                sfx.play("crash", 1.0, x=x)
            elif self.phase == RESULT and self.last_result is not None:
                _, saved, total = self.last_result
                if total:
                    sfx.play("fanfare" if saved == total else "chime" if saved else "fail", 0.9)
        # odpočet: pípnutí 5 a 4 s před vlnou (pak už zní siréna varování)
        if self.phase == CALM and self.auto:
            now = math.ceil(self.phase_len - self.phase_t + DURATION[WARN])
            last = getattr(self, "_snd_left", None)
            if last is not None and now != last and now in (4, 5):
                sfx.play("beep", 0.8)
            self._snd_left = now
        else:
            self._snd_left = None
        sfx.loop("sea", {CALM: 0.35, WARN: 0.45, RESULT: 0.3}.get(self.phase, 0.0), x=x)
        if self.phase in (WAVE, RECEDE):
            # hukot podle toho, jak rychle se voda valí po pevnině
            k = self.phase_t / self.phase_len
            if self.phase == WAVE:  # vlna: naplno, ke konci mírně slábne
                roar = 0.95 * min(1.0, self.phase_t / 0.6) * (1.0 - 0.3 * k)
            else:  # ústup: podle toho, jak rychle voda odtéká z pevniny
                speed = float(v[flooded_now].mean()) if flooded_now.any() else 0.0
                roar = 0.8 * max(0.3, min(1.0, speed / 0.25)) * math.sqrt(max(0.0, 1.0 - k))
            sfx.loop("roar", roar, x=x)

    def _update_spray(self, dt: float, rise: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> None:
        """Tříšť z čela vlny: bílé kapky vylétají po směru proudu a mizí."""
        sp = self.spray
        if len(sp):
            sp[:, 0:2] += sp[:, 2:4] * dt
            sp[:, 2:4] *= max(0.0, 1.0 - 1.2 * dt)
            sp[:, 4] += dt
            sp = sp[sp[:, 4] < sp[:, 5]]
        if self.phase == WAVE and self.sim is not None:
            cells = np.flatnonzero(rise.ravel() > 0.7)
            n = min(len(cells), int(self.rng.poisson(90 * dt)))
            if n:
                pick = self.rng.choice(cells, n)
                gw = rise.shape[1]
                iy, ix = np.divmod(pick, gw)
                cell = self.sim.cell
                new = np.zeros((n, 6), np.float32)
                new[:, 0] = (ix + self.rng.random(n)) * cell
                new[:, 1] = (iy + self.rng.random(n)) * cell
                new[:, 2] = vx.ravel()[pick] * 1.3 + self.rng.normal(0, 0.06, n)
                new[:, 3] = vy.ravel()[pick] * 1.3 + self.rng.normal(0, 0.06, n)
                new[:, 5] = self.rng.uniform(0.5, 1.2, n)
                sp = np.vstack([sp, new])
        self.spray = sp[-250:]

    # --- kreslení --------------------------------------------------------------------
    def render(self) -> None:
        if self.tex is None:
            return
        flash = 0.0
        warn = 0.0
        if self.phase == WARN:  # poplach: plynulý pulz 1 Hz (žádné prudké blikání)
            flash = 0.6 + 0.4 * math.cos(self.phase_t * 2 * math.pi)
            warn = min(1.0, self.phase_t / self.phase_len * 1.2)
        elif self.phase == CALM and self.phase_len - self.phase_t < 5 and self.auto:
            flash = 0.25 * (0.5 + 0.5 * math.cos(self.phase_t * 2 * math.pi))
        show_flood = 0.0
        if self.phase == RESULT:
            show_flood = min(1.0, self.phase_t / 1.0)
        elif self.phase == RECEDE:
            show_flood = max(0.0, (self.phase_t - self.phase_len + 3) / 3)
        edge = tuple(1.0 if i == self.edge else 0.0 for i in range(4))
        self.host.draw_fullscreen(self.prog, u_state=(self.tex, 1), u_vel=(self.vel_tex, 2), u_sea=float(self.sea), u_flash=float(flash),
                                  u_edge=edge, u_show_flood=float(show_flood), u_warn=float(warn))
        self._draw_spray()
        self._draw_towns()
        self._draw_sea_label()
        self._labels()

    def _cell_to_canvas(self, iy: float, ix: float) -> tuple[float, float]:
        assert self.sim is not None
        gh, gw = self.sim.depth.shape
        cw, ch = self.host.canvas_size
        return (ix + 0.5) / gw * cw, (iy + 0.5) / gh * ch

    def _draw_spray(self) -> None:
        sp = self.spray
        if not len(sp):
            return
        cw, ch = self.host.canvas_size
        bw, bh = self.host.box_size
        k = np.clip(sp[:, 4] / sp[:, 5], 0, 1)
        centers = sp[:, :2] * np.array([cw / bw, ch / bh], np.float32)
        r = (0.006 + 0.012 * np.sqrt(k)) * cw / bw
        cols = np.zeros((len(sp), 4), np.float32)
        cols[:, :3] = (0.95, 0.99, 1.0)
        cols[:, 3] = 0.7 * (1 - k) ** 1.5
        verts, colors = soft_discs(centers, r, cols)
        self.host.painter.triangles(verts, colors)

    def _draw_towns(self) -> None:
        """Města: tři domky; vlna je zatopí (modře prosvítají pod vodou, kolem kruhy vln),
        po vlně zatopená = šedé trosky s rudým křížkem, zachráněná = zelený kruh."""
        p = self.host.painter
        cw, ch = self.host.canvas_size
        t = self.host.time
        s = cw / self.host.box_size[0] * 0.018  # domek 1,8 cm
        for i, ((ty, tx), flooded) in enumerate(zip(self.towns, self.town_flooded)):
            x, y = self._cell_to_canvas(ty, tx)
            under = self.phase == WAVE and flooded
            ruined = flooded and self.phase in (RECEDE, RESULT)
            if self.phase == RESULT and not flooded:  # zachráněné město září
                pulse = 0.5 + 0.5 * math.sin(t * 3.0 + i)
                p.circle(x, y, s * (2.4 + 0.3 * pulse), (0.2, 1.0, 0.35, 0.25 + 0.15 * pulse))
            p.circle(x, y, s * 1.95, (0, 0, 0, 0.45))
            if ruined:
                body, roof = (0.45, 0.45, 0.48, 1), (0.28, 0.27, 0.30, 1)
            elif under:
                body, roof = (0.45, 0.75, 0.95, 0.75), (0.25, 0.45, 0.85, 0.75)
            else:
                body, roof = (1.0, 0.97, 0.88, 1), (0.95, 0.15, 0.08, 1)
            tris, cols = [], []
            for j, (dx, dy) in enumerate(((-0.75, -0.45), (0.75, -0.45), (0.0, 0.65))):
                hx, hy = x + dx * s, y + dy * s
                if ruined:  # trosky: nakřivo a níž
                    hy += s * 0.15
                    hx += s * 0.12 * (j - 1)
                q = s * 0.5
                tris += [[hx - q, hy - q], [hx + q, hy - q], [hx + q, hy + q], [hx - q, hy - q], [hx + q, hy + q],
                         [hx - q, hy + q]]
                cols += [body] * 6
                if not ruined:
                    tris += [[hx - s * 0.62, hy - s * 0.45], [hx + s * 0.62, hy - s * 0.45], [hx, hy - s * 1.05]]
                    cols += [roof] * 3
                else:
                    tris += [[hx - s * 0.6, hy - s * 0.2], [hx + s * 0.1, hy - s * 0.75], [hx + s * 0.55, hy - s * 0.3]]
                    cols += [roof] * 3
            p.triangles(np.array(tris, np.float32), np.array(cols, np.float32))
            if under:  # kruhy na hladině nad zatopeným městem
                for ph in ((t * 0.9) % 1.0, (t * 0.9 + 0.5) % 1.0):
                    p.circle(x, y, s * (1.6 + 2.2 * ph), (0.9, 0.97, 1.0, 0.8 * (1 - ph)), width=s * 0.22)
            if ruined:
                r = s * 1.7
                pulse = 0.75 + 0.25 * math.sin(t * 2.5 + i)
                p.circle(x, y, s * 2.15, (1, 0.12, 0.08, pulse), width=s * 0.3)
                p.polyline([(x - r, y - r), (x + r, y + r)], (1, 0.1, 0.05, 1), width=s * 0.38)
                p.polyline([(x + r, y - r), (x - r, y + r)], (1, 0.1, 0.05, 1), width=s * 0.38)
            elif self.phase == RESULT:
                p.circle(x, y, s * 2.15, (0.2, 1.0, 0.35, 1), width=s * 0.3)
                # fajfka
                p.polyline([(x - s * 0.9, y + s * 0.1), (x - s * 0.25, y + s * 0.8), (x + s * 1.0, y - s * 0.8)],
                           (0.25, 1.0, 0.4, 1), width=s * 0.35)

    def _draw_sea_label(self) -> None:
        if self.phase not in (CALM, WARN):
            return
        p = self.host.painter
        cw, ch = self.host.canvas_size
        size = max(16, ch // 26)
        col = (0.6, 0.9, 1.0, 1) if self.phase == CALM else (1, 0.85, 0.2, 1)
        # šipky od moře dovnitř; při varování se opakovaně posouvají na pevninu
        n = 3
        shift = ((self.phase_t * 1.2) % 1.0) * size * 2.5 if self.phase == WARN else 0.0
        for i in range(n):
            f = (i + 1) / (n + 1)
            if self.edge == 0:
                a, d = (size * 0.8, f * ch), (1, 0)
            elif self.edge == 2:
                a, d = (cw - size * 0.8, f * ch), (-1, 0)
            elif self.edge == 1:
                a, d = (f * cw, size * 0.8), (0, 1)
            else:
                a, d = (f * cw, ch - size * 0.8), (0, -1)
            a = (a[0] + d[0] * shift, a[1] + d[1] * shift)
            self._arrow(a, d, size * 2.2, col)
        if self.edge in (1, 3):
            y = size * 0.3 if self.edge == 1 else ch - size * 1.4
            p.text("MOŘE", cw * 0.12, y, size=size, anchor="tc", bold=True, color=col, outline=3)
        else:
            x = size * 0.3 if self.edge == 0 else cw - size * 0.3
            p.text("MOŘE", x, ch * 0.12, size=size, anchor="tl" if self.edge == 0 else "tr", bold=True,
                   color=col, outline=3)

    def _arrow(self, start, direction, length: float, color) -> None:
        p = self.host.painter
        sx, sy = start
        dx, dy = direction
        ex, ey = sx + dx * length, sy + dy * length
        w = length * 0.12
        p.polyline([(sx, sy), (ex - dx * w * 2, ey - dy * w * 2)], color, width=w)
        nx, ny = -dy, dx
        p.triangles(np.array([[ex, ey], [ex - dx * w * 3 + nx * w * 2, ey - dy * w * 3 + ny * w * 2],
                              [ex - dx * w * 3 - nx * w * 2, ey - dy * w * 3 - ny * w * 2]]), color)

    def _warning_sign(self, x: float, y: float, s: float, color) -> None:
        p = self.host.painter
        p.triangles(np.array([[x, y - s * 0.62], [x - s * 0.62, y + s * 0.48], [x + s * 0.62, y + s * 0.48]]), (0, 0, 0, 1))
        p.triangles(np.array([[x, y - s * 0.5], [x - s * 0.5, y + s * 0.4], [x + s * 0.5, y + s * 0.4]]), color)
        p.text("!", x, y + s * 0.05, size=int(s * 0.65), anchor="mc", bold=True, color=(0, 0, 0, 1))

    def _labels(self) -> None:
        p = self.host.painter
        cw, ch = self.host.canvas_size
        size = max(18, ch // 22)
        small = max(16, ch // 28)
        remaining = max(0.0, self.phase_len - self.phase_t)
        top_y = size * 0.5 if self.edge != 1 else ch * 0.18
        bottom_y = ch - small * 0.5 if self.edge != 3 else ch * 0.84
        if self.phase == CALM:
            hud.label(p, "Postav hráze a ochraň města před vlnou!", cw / 2, top_y, size, anchor="tc")
            if self.auto:
                hud.label(p, f"Tsunami přijde za {math.ceil(remaining + DURATION[WARN])} s", cw / 2, bottom_y,
                          small, anchor="bc", color=hud.YELLOW)
        elif self.phase == WARN:
            n = max(1, math.ceil(remaining))
            text = f"TSUNAMI za {n}…"
            big = hud.fit("TSUNAMI za 8…", ch / 7, cw * 0.62)  # stejná velikost pro všechna čísla
            u = 0.5 + 0.5 * math.cos(self.phase_t * 2 * math.pi)  # plynule rudá <-> žlutá, 1 Hz
            col = (1, 0.2 + 0.7 * u, 0.1 + 0.1 * u, 1)
            sign = big * 1.1
            w = get_font(big, True).getlength(text)
            x0 = (cw - (sign + big * 0.3 + w)) / 2  # značka + mezera + text, celé na střed
            self._warning_sign(x0 + sign / 2, ch / 2, sign, (1, 0.85, 0.1, 1))
            p.text(text, x0 + sign + big * 0.3, ch / 2, size=big, anchor="ml", bold=True, color=col, outline=6)
        elif self.phase in (WAVE, RECEDE):
            title = "VLNA!" if self.phase == WAVE else "Voda ustupuje zpět do moře…"
            hud.label(p, title, cw / 2, top_y, int(size * (1.4 if self.phase == WAVE else 1.0)), anchor="tc")
            hit = sum(self.town_flooded)
            hud.label(p, f"Zaplaveno {self.flooded_pct:.0f} % pevniny · zatopená města: {hit}/{len(self.towns)}",
                      cw / 2, bottom_y, small, anchor="bc", color=hud.YELLOW)
        elif self.phase == RESULT and self.last_result is not None:
            pct, saved, total = self.last_result
            cy = size * 1.85  # nahoře, kam se města nestaví
            p.rect(cw * 0.12, cy - size * 1.75, cw * 0.76, size * 3.4, (0.0, 0.03, 0.08, 0.72))
            good_all = bool(total) and saved == total
            edge_col = (0.25, 1.0, 0.4, 1) if good_all else (1.0, 0.25, 0.15, 1)
            p.rect_outline(cw * 0.12, cy - size * 1.75, cw * 0.76, size * 3.4, edge_col, width=max(3.0, size * 0.12))
            p.text(f"Vlna zaplavila {pct:.0f} % pevniny", cw / 2, cy - size * 0.7, size=int(size * 1.3),
                   anchor="mc", bold=True, color=(1, 0.4, 0.3, 1), outline=5)
            if total:
                good = saved == total
                txt = "Všechna města přežila!" if good else f"Zachráněná města: {saved} z {total}"
                p.text(txt, cw / 2, cy + size * 0.8, size=size, anchor="mc", bold=True,
                       color=(0.3, 1, 0.4, 1) if saved else (1, 1, 1, 1), outline=5)
            items = [((1.0, 0.22, 0.12, 1), "zatopená pevnina")]
            if total:
                items += [((0.25, 1.0, 0.4, 1), "zachráněné město")] if saved else []
                items += [((1.0, 0.1, 0.05, 1), "zatopené město")] if saved < total else []
            hud.legend(p, items, cw / 2, bottom_y, small, anchor="bc")

    # --- ovládání ------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name == "t":
            if self.phase in (CALM, RESULT):
                self._set_phase(WARN)
        elif name == "s":
            self.edge_fixed = (self.edge + 1) % 4
            self.seafill = None
            self._planned = False
            if self.sim is not None:
                self.sim.clear()
            self._set_phase(CALM, max(FIRST_CALM, self.phase_len - self.phase_t) if self.phase == CALM else 8.0)
            self.host.notify(f"Moře je {EDGES[self.edge_fixed]}")
        elif name == "c":
            self.auto = not self.auto
            self.host.notify("Automatické vlny " + ("zapnuty" if self.auto else "vypnuty"))
        elif name in ("up", "down"):
            self.amplitude = float(np.clip(self.amplitude + (0.01 if name == "up" else -0.01), 0.02, 0.15))
            self.host.notify(f"Výška vlny {self.amplitude * 100:.0f} cm")
        elif name in ("x", "backspace"):
            self.reset()
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Fáze: {self.phase} ({max(0.0, self.phase_len - self.phase_t):.0f} s)",
                 f"Moře: {EDGES[self.edge]}{' (pevně)' if self.edge_fixed is not None else ''}, "
                 f"hladina {self.sea * 100:.1f} cm · vlna {self.amplitude * 100:.0f} cm",
                 f"Města: {len(self.towns)} · vln: {self.waves} · auto: {'ano' if self.auto else 'ne'}"]
        if self.last_result:
            pct, saved, total = self.last_result
            lines.append(f"Minulá vlna: zaplaveno {pct:.0f} %, zachráněno {saved}/{total} měst")
        if self.sim is not None:
            lines.append(f"Voda: {self.sim.volume * 1000:.1f} l")
        return lines
