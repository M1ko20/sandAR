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
from .base import Game

FRAG = """
uniform sampler2D u_state;   // r = hloubka vody, g = pěna, b = mokrý písek, a = zatopeno při vlně
uniform float u_sea;         // hladina moře [m]
uniform float u_flash;       // 0..1 červené varování
uniform vec4 u_edge;         // strana moře (levá, horní, pravá, dolní) – jedna jednička
uniform float u_show_flood;  // 0..1 vyznačit zatopenou pevninu

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x),
               mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}

// bikubická interpolace všech 4 kanálů (hladké okraje vody i na hrubé mřížce simulace)
vec4 bicubic4(sampler2D tex, vec2 uv) {
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

// dvě pootočené vrstvy šumu – bez čtverečkové struktury hodnotového šumu
float noise2(vec2 p) {
    return 0.5 * (vnoise(p) + vnoise(mat2(0.8, -0.6, 0.6, 0.8) * p * 1.37 + 17.3));
}

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;
    float h = height_at(uv);
    vec4 st = max(bicubic4(u_state, uv), 0.0);
    float w = st.r;
    float foam = st.g;
    float wet = st.b;

    // --- pevnina ---
    float e = (h - u_sea) / max(u_elev_max - u_sea, 1e-4);
    vec3 c = land_color(e);
    c *= mix(0.55, 1.2, hillshade(uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.45, cl.y * 0.6 + cl.x * 0.25);
    c = mix(c, c * vec3(0.45, 0.42, 0.40), wet * 0.7);            // mokrý písek po vlně
    // zatopená pevnina po vlně – červené šrafování
    float stripe = step(0.5, fract((pm.x + pm.y) * 28.0));
    c = mix(c, mix(c, vec3(1.0, 0.15, 0.1), 0.55 + 0.25 * stripe), u_show_flood * st.a);

    // --- voda ---
    if (w > 0.0008) {
        float a = smoothstep(0.0008, 0.004, w);
        float d01 = clamp(w / 0.05, 0.0, 1.0);
        vec3 wc = mix(vec3(0.10, 0.70, 1.0), vec3(0.03, 0.30, 0.85), d01);
        wc = mix(wc, vec3(0.02, 0.10, 0.45), smoothstep(0.05, 0.14, w));
        wc *= mix(0.8, 1.15, hillshade(uv, 2.5));
        float rip = noise2(pm * 45.0 + u_time * vec2(0.9, 0.6)) - noise2(pm * 45.0 - u_time * vec2(0.7, 0.8));
        wc += 0.10 * rip;
        // pěna: rozbitá bílá textura
        float n = noise2(pm * 90.0 + u_time * vec2(1.3, -0.9)) * 0.5 + noise2(pm * 30.0 - u_time * 0.5) * 0.5;
        float f = smoothstep(0.3, 0.7, foam * (0.5 + 0.8 * n));
        wc = mix(wc, vec3(0.97, 0.99, 1.0), f);
        c = mix(c, wc, a);
    }

    // --- varování: blikající pruh na straně moře ---
    float dl = uv.x * u_box.x, dt = uv.y * u_box.y, dr = (1.0 - uv.x) * u_box.x, db = (1.0 - uv.y) * u_box.y;
    float d = 10.0;
    if (u_edge.x > 0.5) d = dl;
    if (u_edge.y > 0.5) d = dt;
    if (u_edge.z > 0.5) d = dr;
    if (u_edge.w > 0.5) d = db;
    float band = 1.0 - smoothstep(0.03, 0.12, d);
    c = mix(c, vec3(1.0, 0.05, 0.02), u_flash * band * 0.85);
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


class TsunamiGame(Game):
    id = "tsunami"
    title = "Mega tsunami"
    description = "Z moře přijde obří vlna a prožene se krajinou. Postav hráze a ochraň města!"
    order = 72
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
        self._set_phase(CALM, FIRST_CALM)

    def stop(self) -> None:
        self.prog.release()
        if self.tex is not None:
            self.tex.release()
            self.tex = None

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

    # --- moře, města --------------------------------------------------------------
    def _refresh_coast(self) -> None:
        """Podle aktuálního terénu urči stranu moře, hladinu, moře a města."""
        assert self.terrain is not None
        t = self.terrain
        self.edge = lowest_edge(t) if self.edge_fixed is None else self.edge_fixed
        self.sea = sea_level_for(t, self.edge)
        self.seafill = sea_fill(t, self.sea, self.edge)
        rng = np.random.default_rng(self.waves * 7 + self.edge)
        self.towns = place_towns(t, self.sea, self.edge, rng)
        self.town_flooded = [False] * len(self.towns)

    def _begin_wave(self) -> None:
        assert self.terrain is not None and self.sim is not None
        if self.seafill is None:
            self._refresh_coast()
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
            if self.tex is not None:
                self.tex.release()
            self.tex = self.host.ctx.texture((shape[1], shape[0]), 4, dtype="f4")
            self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.tex.repeat_x = self.tex.repeat_y = False
            self.seafill = None
        sim = self.sim

        self.phase_t += dt
        if self.phase_t >= self.phase_len:
            if self.phase == CALM and not self.auto:
                self.phase_t = 0.0
            else:
                self._set_phase(NEXT[self.phase])

        # v klidu sleduj, jak návštěvníci mění pobřeží (1× za sekundu)
        self._refresh_t -= dt
        if self.seafill is None or (self.phase in (CALM, WARN) and self._refresh_t <= 0):
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
        shallow = np.clip(1.0 - sim.depth / 0.035, 0.0, 1.0)
        new_foam = np.clip((v - 0.08) / 0.25, 0, 1) * shallow
        # příboj: pruh vody u souše (na čele vlny silnější)
        wet = (sim.depth > 0.0015).astype(np.uint8)
        shore = (cv2.dilate(1 - wet, np.ones((3, 3), np.uint8)) > 0) & (wet > 0)
        surf = shore * np.clip(v / 0.2, 0.5, 1.0)
        new_foam = blur3(blur3(np.maximum(np.maximum(new_foam, rise), surf) * (sim.depth > 0.001)))
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
                    self.town_flooded[i] = True

        assert self.tex is not None
        self.tex.write(np.dstack([sim.depth, self.foam, self.wet, self.flood]).astype(np.float32).tobytes())

    # --- kreslení --------------------------------------------------------------------
    def render(self) -> None:
        if self.tex is None:
            return
        flash = 0.0
        if self.phase == WARN:
            flash = 0.5 + 0.5 * math.cos(self.phase_t * 2 * math.pi * 2)
        elif self.phase == CALM and self.phase_len - self.phase_t < 5 and self.auto:
            flash = 0.25 * (0.5 + 0.5 * math.cos(self.phase_t * 2 * math.pi))
        show_flood = 0.0
        if self.phase == RESULT:
            show_flood = min(1.0, self.phase_t / 1.0)
        elif self.phase == RECEDE:
            show_flood = max(0.0, (self.phase_t - self.phase_len + 3) / 3)
        edge = tuple(1.0 if i == self.edge else 0.0 for i in range(4))
        self.host.draw_fullscreen(self.prog, u_state=(self.tex, 1), u_sea=float(self.sea), u_flash=float(flash),
                                  u_edge=edge, u_show_flood=float(show_flood))
        self._draw_towns()
        self._draw_sea_label()
        self._labels()

    def _cell_to_canvas(self, iy: float, ix: float) -> tuple[float, float]:
        assert self.sim is not None
        gh, gw = self.sim.depth.shape
        cw, ch = self.host.canvas_size
        return (ix + 0.5) / gw * cw, (iy + 0.5) / gh * ch

    def _draw_towns(self) -> None:
        p = self.host.painter
        cw, ch = self.host.canvas_size
        s = cw / self.host.box_size[0] * 0.016  # domek 1,6 cm
        for (ty, tx), flooded in zip(self.towns, self.town_flooded):
            x, y = self._cell_to_canvas(ty, tx)
            wave_over = self.phase in (RECEDE, RESULT) or (self.phase == WAVE and flooded)
            if flooded and wave_over:
                body, roof = (0.35, 0.35, 0.38, 1), (0.25, 0.25, 0.28, 1)
            else:
                body, roof = (1.0, 0.97, 0.88, 1), (0.95, 0.15, 0.08, 1)
            p.circle(x, y, s * 1.9, (0, 0, 0, 0.45))
            for dx, dy in ((-0.75, -0.45), (0.75, -0.45), (0.0, 0.65)):
                hx, hy = x + dx * s, y + dy * s
                p.rect(hx - s * 0.5, hy - s * 0.5, s, s, body)
                p.triangles(np.array([[hx - s * 0.6, hy - s * 0.45], [hx + s * 0.6, hy - s * 0.45], [hx, hy - s * 1.0]]),
                            roof)
            if flooded and wave_over:
                r = s * 1.7
                p.polyline([(x - r, y - r), (x + r, y + r)], (1, 0.1, 0.05, 1), width=s * 0.35)
                p.polyline([(x + r, y - r), (x - r, y + r)], (1, 0.1, 0.05, 1), width=s * 0.35)
            elif self.phase == RESULT:
                p.circle(x, y, s * 2.1, (0.2, 1.0, 0.3, 1), width=s * 0.3)

    def _draw_sea_label(self) -> None:
        if self.phase not in (CALM, WARN):
            return
        p = self.host.painter
        cw, ch = self.host.canvas_size
        size = max(16, ch // 26)
        col = (0.6, 0.9, 1.0, 1) if self.phase == CALM else (1, 0.3, 0.2, 1)
        # šipky od moře dovnitř
        n = 3
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
            p.text("Postav hráze a ochraň města před vlnou!", cw / 2, top_y, size=size, anchor="tc", bold=True,
                   color=(1, 1, 1, 1), outline=4)
            if self.auto:
                p.text(f"Tsunami přijde za {remaining:.0f} s", cw / 2, bottom_y, size=small, anchor="bc",
                       color=(1, 0.85, 0.3, 1), outline=3)
        elif self.phase == WARN:
            n = max(1, math.ceil(remaining))
            big = int(ch / 7)
            on = (self.phase_t % 0.5) < 0.32
            col = (1, 0.2, 0.1, 1) if on else (1, 0.9, 0.2, 1)
            w, _ = p.text(f"TSUNAMI za {n}…", cw / 2 + big * 0.45, ch / 2, size=big, anchor="mc", bold=True,
                          color=col, outline=6)
            self._warning_sign(cw / 2 - w / 2 - big * 0.2, ch / 2, big * 1.1, (1, 0.85, 0.1, 1))
        elif self.phase in (WAVE, RECEDE):
            title = "VLNA!" if self.phase == WAVE else "Voda ustupuje zpět do moře…"
            p.text(title, cw / 2, top_y, size=int(size * (1.4 if self.phase == WAVE else 1.0)), anchor="tc",
                   bold=True, color=(1, 1, 1, 1), outline=4)
            hit = sum(self.town_flooded)
            p.text(f"Zaplaveno {self.flooded_pct:.0f} % pevniny · zatopená města: {hit}/{len(self.towns)}",
                   cw / 2, bottom_y, size=small, anchor="bc", color=(1, 0.85, 0.3, 1), outline=3)
        elif self.phase == RESULT and self.last_result is not None:
            pct, saved, total = self.last_result
            cy = size * 1.85  # nahoře, kam se města nestaví
            p.rect(cw * 0.12, cy - size * 1.75, cw * 0.76, size * 3.4, (0, 0, 0, 0.6))
            p.text(f"Vlna zaplavila {pct:.0f} % pevniny", cw / 2, cy - size * 0.7, size=int(size * 1.3),
                   anchor="mc", bold=True, color=(1, 0.4, 0.3, 1), outline=5)
            if total:
                good = saved == total
                txt = "Všechna města přežila!" if good else f"Zachráněná města: {saved} z {total}"
                p.text(txt, cw / 2, cy + size * 0.8, size=size, anchor="mc", bold=True,
                       color=(0.3, 1, 0.4, 1) if saved else (1, 1, 1, 1), outline=5)

    # --- ovládání ------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name == "t":
            if self.phase in (CALM, RESULT):
                self._set_phase(WARN)
        elif name == "s":
            self.edge_fixed = (self.edge + 1) % 4
            self.seafill = None
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
