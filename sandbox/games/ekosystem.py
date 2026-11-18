"""Živý ekosystém: na písku roste krajina, která reaguje na každou změnu terénu.

* biomy podle výšky a sklonu – moře a pláže v nížinách, louky, lesy na svazích,
  skály na strmých stěnách a nad hranicí lesa, sníh na vrcholcích,
* řeky stékají z hor údolími (akumulace toku D8), ve sníženinách vznikají jezera,
* stromy postupně vyrůstají a hynou, když je zasypeš, vykopeš nebo zatopíš,
* 12 druhů zvířat, každý ve svém prostředí (``sandbox/sim/fauna.py``): ryby a kachny ve vodě,
  krabi a želvy na pláži, srnky, zajíci a lišky na loukách, medvěd v lese, kamzíci na skalách,
  nad horami krouží orel, nad krajinou ptáci a motýli. Počet roste s velikostí prostředí,
  plachá zvířata utíkají před rukou.

Logika je v ``sandbox/sim/ecosystem.py`` a ``sandbox/sim/fauna.py``, tady je jen kreslení.
"""

from __future__ import annotations

import math

import cv2
import moderngl
import numpy as np

from ..render.fauna import draw_fauna
from ..sim import ecosystem as eco
from ..sim.fauna import Fauna
from .base import Game

FRAG = """
uniform sampler2D u_hydro;   // r = řeka, g/b = směr toku
uniform sampler2D u_lake;    // hladina jezera (mimo jezera pod terénem)
uniform float u_clock;
uniform float u_beach;
uniform float u_forest_h;
uniform float u_forest_s;
uniform float u_treeline;
uniform float u_snow;
uniform float u_rock_s;
uniform vec2 u_wind;         // směr větru (jednotkový)
uniform float u_golden;      // 0..1 teplé světlo (ráno / podvečer)

float eko_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float eko_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(eko_hash(i), eko_hash(i + vec2(1, 0)), f.x),
               mix(eko_hash(i + vec2(0, 1)), eko_hash(i + vec2(1, 1)), f.x), f.y);
}
float eko_fbm(vec2 p) {
    return 0.5 * eko_noise(p) + 0.3 * eko_noise(p * 2.07 + 3.1) + 0.2 * eko_noise(p * 4.13 + 7.7);
}

// třpyt sněhu: v buňce 25 mm jedna jiskra (měkký bod + paprsky), plynule zablýskne a zhasne
float eko_sparkle(vec2 p) {
    vec2 g = p * 40.0;
    vec2 id = floor(g);
    float r = eko_hash(id);
    if (r > 0.55) return 0.0;
    vec2 off = vec2(eko_hash(id + 7.1), eko_hash(id + 3.3)) * 0.5 + 0.25;
    vec2 d = fract(g) - off;
    float tw = pow(max(sin(u_clock * (1.3 + 2.0 * r) + r * 40.0), 0.0), 4.0);
    float core = exp(-dot(d, d) / 0.018);
    float rays = exp(-abs(d.x) / 0.06) * exp(-abs(d.y) / 0.28) + exp(-abs(d.y) / 0.06) * exp(-abs(d.x) / 0.28);
    return tw * (core + 0.6 * rays);
}

void main() {
    vec2 uv = v_uv;
    vec2 p = uv * u_box;               // metry
    float h = height_at(uv);
    float s = length(height_grad(uv));
    float n = eko_noise(p * 30.0) * 0.65 + eko_noise(p * 70.0) * 0.35;

    // vítr: poryvy putují krajinou, tráva se v nich leskne (světlejší rub stébel)
    float along = dot(p, u_wind);
    float gust = smoothstep(0.35, 0.75, eko_noise(p * 5.0 - u_wind * u_clock * 0.45));
    float wave = 0.5 + 0.5 * sin(along * 40.0 - u_clock * 2.4 + eko_noise(p * 10.0) * 3.0);
    float sway = wave * gust;

    vec3 meadow = mix(vec3(0.38, 0.80, 0.10), vec3(0.58, 0.90, 0.16), n);
    meadow = mix(meadow, vec3(0.80, 0.96, 0.42), sway * 0.55);
    vec3 forest = mix(vec3(0.04, 0.36, 0.09), vec3(0.09, 0.50, 0.14), n);
    forest = mix(forest, vec3(0.20, 0.62, 0.22), sway * 0.35);
    // skály: vrstvy hornin podle výšky (pruhy po vrstevnicích), teplé a studené odstíny
    float strata = 0.5 + 0.5 * sin(h * 700.0 + eko_noise(p * 18.0) * 4.0);
    vec3 rock = mix(vec3(0.46, 0.39, 0.35), vec3(0.68, 0.61, 0.55), n);
    rock = mix(rock, rock * vec3(0.82, 0.80, 0.86), smoothstep(0.55, 0.8, strata) * 0.8);
    vec3 beach = mix(vec3(1.0, 0.84, 0.40), vec3(1.0, 0.93, 0.62), n);

    float wl = u_water_level;
    float fo = max(smoothstep(u_forest_h - 0.003, u_forest_h + 0.003, h),
                   smoothstep(u_forest_s - 0.03, u_forest_s + 0.03, s));
    vec3 c = mix(meadow, forest, fo);
    float ro = max(smoothstep(u_treeline - 0.003, u_treeline + 0.003, h + 0.004 * (n - 0.5)),
                   smoothstep(u_rock_s - 0.06, u_rock_s + 0.06, s));
    c = mix(c, rock, ro);
    float sn = smoothstep(u_snow - 0.0025, u_snow + 0.0025, h + 0.004 * (eko_noise(p * 25.0) - 0.5));
    float be = 1.0 - smoothstep(wl + u_beach - 0.0015, wl + u_beach + 0.0015, h);
    c = mix(c, beach, be);
    float hs = hillshade(uv, 2.6);
    c *= mix(0.48, 1.20, hs);
    // sníh: modré stíny, zářivě bílá osvětlená strana, třpyt
    vec3 snow = mix(vec3(0.55, 0.68, 0.98), vec3(1.0, 1.0, 1.0), smoothstep(0.15, 0.85, hs));
    c = mix(c, snow, sn * (1.0 - be));
    c += vec3(0.9, 0.95, 1.0) * sn * eko_sparkle(p) * 0.9;
    // jemné vrstevnice (orientace v terénu)
    vec2 cl = contours(h);
    c = mix(c, c * 0.55, cl.y * 0.6);

    // voda: moře, jezera, řeky – sdílené vykreslení vody (proud po směru toku, průhledné mělčiny)
    vec4 hy = texture(u_hydro, uv);
    float sea = max(wl - h, 0.0);
    float depth = max(sea, max(sample_bicubic(u_lake, uv) - h, 0.0));
    float rv = sample_bicubic(u_hydro, uv);
    float river = smoothstep(0.30, 0.45, rv);
    vec2 dir = hy.gb;
    float L = length(dir);
    dir = L > 1e-3 ? dir / L : vec2(0.0);
    // řeka: koryto hloubky ~2–6 mm, proud tím rychlejší, čím je řeka mohutnější
    float rdepth = river * (0.0025 + 0.004 * clamp(rv, 0.0, 1.0));
    vec2 vel = dir * (0.08 + 0.12 * river) * (1.0 - smoothstep(0.004, 0.012, depth));
    // klidná voda se vlní ve větru
    vel += u_wind * 0.015 * smoothstep(0.004, 0.012, depth);
    // příboj: vlnky běží ke břehu a rozbíjejí se v bílou pěnu
    float surf = 0.0;
    if (sea > 0.0 && sea < 0.008) {
        float ph = sea * 900.0 + u_clock * 1.8 + eko_noise(p * 14.0) * 2.5;
        surf = smoothstep(0.75, 1.0, sin(ph)) * (1.0 - smoothstep(0.002, 0.008, sea));
    }
    vec4 wr = water_shade(c, uv, max(depth, rdepth), vel, surf * 0.8);
    c = wr.rgb;
    // peřeje: světlé jiskření na mohutných řekách
    float rap = smoothstep(0.6, 0.9, eko_noise(p * 60.0 - dir * u_clock * 3.0)) * river * smoothstep(0.5, 1.0, rv);
    c = mix(c, vec3(0.92, 0.98, 1.0), rap * 0.5 * (1.0 - smoothstep(0.004, 0.012, depth)));

    // stíny oblaků putují s větrem přes celou krajinu
    float cs = eko_fbm(p * 3.2 - u_wind * u_clock * 0.035);
    float shadow = smoothstep(0.52, 0.68, cs);
    c = mix(c, c * vec3(0.58, 0.64, 0.80), shadow * 0.65);
    // teplé světlo slunce (ráno / podvečer) – jen na osvětlených svazích
    c *= mix(vec3(1.0), vec3(1.10, 0.98, 0.82), u_golden * smoothstep(0.4, 0.9, hs) * (1.0 - shadow * 0.7));
    f_color = vec4(c, 1.0);
}
"""


def disc_triangles(cx: np.ndarray, cy: np.ndarray, r: np.ndarray, seg: int = 14, lobes: int = 0,
                   lobe_amp: float = 0.0, rot: np.ndarray | None = None) -> np.ndarray:
    """Trojúhelníky pro N kruhů najednou (N, seg*3, 2); lobes>0 = zubatý okraj (jehličnan)."""
    ang = np.linspace(0, 2 * np.pi, seg + 1)[None, :]
    if rot is not None:
        ang = ang + rot[:, None]
    rr = r[:, None] * (1 + lobe_amp * np.cos(lobes * ang)) if lobes else r[:, None] * np.ones_like(ang)
    ring = np.stack([cx[:, None] + rr * np.cos(ang), cy[:, None] + rr * np.sin(ang)], -1)  # (N, seg+1, 2)
    center = np.broadcast_to(np.stack([cx, cy], -1)[:, None, :], (len(cx), seg, 2))
    tri = np.stack([center, ring[:, :-1], ring[:, 1:]], 2)  # (N, seg, 3, 2)
    return tri.reshape(len(cx), seg * 3, 2)


class EcosystemGame(Game):
    id = "ekosystem"
    title = "Živý ekosystém"
    description = "Na písku roste živá krajina: řeky, jezera, lesy, sníh na horách a zvířata, každé ve svém domově."
    order = 55
    category = "zážitek"
    intro = "Postav hory a vykopej jezera – v každém prostředí se usadí jiná zvířata: ryby, kamzíci, srnky, medvěd…"
    keys = [
        ("R", "řeky zap/vyp"),
        ("A", "zvířata zap/vyp"),
        ("↑ / ↓", "sněžná čára"),
        ("← / →", "hladina moře"),
        ("Backspace", "nový svět (vykácet les)"),
    ]

    HYDRO_PERIOD = 0.35  # [s] přepočet řek a jezer
    CHECK_PERIOD = 0.2  # [s] kontrola stromů a biomů

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.params = eco.BiomeParams()
        self.hydro_tex = None
        self.lake_tex = None
        self.hydro: eco.Hydrology | None = None
        self.veg: eco.Vegetation | None = None
        self.animals: Fauna | None = None
        self.show_rivers = True
        self.show_animals = True
        self.clock = 0.0
        self._next_hydro = 0.0
        self._next_check = 0.0
        self.biome: np.ndarray | None = None
        self.slope: np.ndarray | None = None
        self.wet: np.ndarray | None = None
        self.heights: np.ndarray | None = None

    def stop(self) -> None:
        self.prog.release()
        for tex in (self.hydro_tex, self.lake_tex):
            if tex is not None:
                tex.release()
        self.hydro_tex = self.lake_tex = None

    def reset(self) -> None:
        if self.veg is not None:
            self.veg.reset()
        if self.animals is not None:
            self.animals.reset()

    # --- logika -------------------------------------------------------------------------------
    def _ensure(self, hm) -> None:
        if self.veg is None or self.veg.size != hm.size:
            # stromy a zvířata v metrech – na větší bedně jich je víc
            area = hm.size[0] * hm.size[1]
            self.veg = eco.Vegetation(hm.size, max_trees=int(200 * area) + 60, radius=0.017)
            self.animals = Fauna(hm.size)

    def _classify(self, hm) -> None:
        m = self.host.settings.map
        self.params.water_level = m.water_level
        smooth = cv2.GaussianBlur(hm.heights.astype(np.float32), (5, 5), 1.2)
        self.heights = smooth
        self.slope = eco.slope_map(smooth, hm.cell_size)
        self.biome = eco.classify_biomes(smooth, self.slope, self.params)

    def _analyze_water(self, hm) -> None:
        gw = hm.shape[1]
        factor = max(1, gw // 64)
        self.hydro = eco.analyze_hydrology(self.heights, factor, self.params.water_level, hm.cell_size[0],
                                           rain_from=self.params.forest_height * 0.6, river_threshold=8.0)
        data = self.hydro.texture_data(self.params.water_level)
        if not self.show_rivers:
            data[..., 0] = 0
        size = (data.shape[1], data.shape[0])
        if self.hydro_tex is None or self.hydro_tex.size != size:
            if self.hydro_tex is not None:
                self.hydro_tex.release()
                self.lake_tex.release()
            self.hydro_tex = self.host.ctx.texture(size, 4, dtype="f4")
            self.hydro_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.hydro_tex.repeat_x = self.hydro_tex.repeat_y = False
            self.lake_tex = self.host.float_texture(size)
        self.hydro_tex.write(np.ascontiguousarray(data).tobytes())
        self.lake_tex.write(np.ascontiguousarray(data[..., 3]).tobytes())
        self.wet = self.hydro.wet_mask(hm.shape) if self.show_rivers else np.zeros(hm.shape, bool)

    def update(self, hm, dt: float) -> None:
        self._ensure(hm)
        dt = min(dt, 0.1)
        self.clock += dt
        if self.biome is None or self.biome.shape != hm.shape or self.clock >= self._next_check:
            self._next_check = self.clock + self.CHECK_PERIOD
            self._classify(hm)
        if self.hydro is None or self.wet is None or self.wet.shape != hm.shape or self.clock >= self._next_hydro:
            self._next_hydro = self.clock + self.HYDRO_PERIOD
            self._analyze_water(hm)
            assert self.veg is not None and self.wet is not None
            self.veg.check(self.heights, self.biome, self.wet, hm.hands)
        assert self.veg is not None and self.animals is not None
        self.veg.step(dt)
        self.veg.spawn(dt, self.heights, self.biome, self.wet, self.slope)
        hands = [h.pos for h in hm.hand_list] + [h.tip for h in hm.hand_list] if hm.hand_list else hm.hand_points
        self.animals.step(dt, self.biome, self.wet, self._water(hm.shape), hands)

    def _water(self, shape: tuple[int, int]) -> np.ndarray:
        """Buňky pod vodou: moře + jezera (pro ryby a kachny)."""
        sea = self.biome == eco.WATER if self.biome is not None else np.zeros(shape, bool)
        if self.hydro is None or not self.show_rivers:
            return sea
        lake = cv2.resize((self.hydro.lake > 0.002).astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        return sea | lake

    # --- kreslení -------------------------------------------------------------------------------
    def render(self) -> None:
        if self.hydro_tex is None:
            return
        p = self.params
        wind = self._wind()
        golden = 0.5 + 0.5 * math.sin(self.clock * 2 * math.pi / 150.0)
        self.host.draw_fullscreen(self.prog, u_hydro=(self.hydro_tex, 1), u_lake=(self.lake_tex, 2), u_clock=float(self.clock),
                                  u_beach=p.beach, u_forest_h=p.forest_height, u_forest_s=p.forest_slope,
                                  u_treeline=p.treeline, u_snow=p.snow, u_rock_s=p.rock_slope,
                                  u_wind=(float(wind[0]), float(wind[1])), u_golden=float(golden))
        self._draw_trees(wind)
        if self.show_animals and self.animals is not None:
            draw_fauna(self.host, self.animals, self.clock)
        self.host.glow(0.55, 0.82)  # třpyt sněhu a vody, jasné koruny

    def _px(self) -> tuple[float, float]:
        cw, ch = self.host.canvas_size
        bw, bd = self.host.box_size
        return cw / bw, ch / bd

    def _wind(self) -> np.ndarray:
        """Směr větru – pomalu se stáčí (stíny oblaků, vlny trávy, kymácení stromů)."""
        a = 0.5 + 0.6 * math.sin(self.clock * 0.021)
        return np.array([math.cos(a), math.sin(a)])

    def _sway(self, pos: np.ndarray, wind: np.ndarray) -> np.ndarray:
        """Výchylka korun ve větru [m]: vlny poryvů běží lesem po směru větru."""
        t = self.clock
        along = pos @ wind
        gust = 0.55 + 0.45 * np.sin(pos[:, 0] * 6.0 + t * 0.7) * np.sin(pos[:, 1] * 5.0 - t * 0.5)
        wave = np.sin(along * 40.0 - t * 2.4) * 0.6 + np.sin(along * 23.0 - t * 1.6 + pos[:, 1] * 9.0) * 0.4
        amp = (0.0025 + 0.002 * wave) * gust  # 1–6 mm = 1–5 px na plátně
        return amp[:, None] * wind[None, :]

    def _draw_trees(self, wind: np.ndarray | None = None) -> None:
        veg = self.veg
        if veg is None or not len(veg):
            return
        sx, sy = self._px()
        sway = self._sway(veg.pos, wind if wind is not None else self._wind())
        x0 = veg.pos[:, 0] * sx
        y0 = veg.pos[:, 1] * sy
        x = x0 + sway[:, 0] * sx * 0.6
        y = y0 + sway[:, 1] * sy * 0.6
        tx = x0 + sway[:, 0] * sx  # vršek koruny se kymácí víc
        ty = y0 + sway[:, 1] * sy
        r = veg.radii() * sx
        n = len(x)
        con = veg.kind == eco.CONIFER
        # barvy: jehličnany tmavé, listnáče jasné; umírající hnědnou, utopené šednou
        base = np.where(con[:, None], [0.02, 0.30, 0.10, 1.0], [0.20, 0.62, 0.08, 1.0]).astype(np.float32)
        light = np.where(con[:, None], [0.10, 0.55, 0.22, 1.0], [0.55, 0.90, 0.25, 1.0]).astype(np.float32)
        k = np.clip(np.where(veg.dead >= 0, veg.dead / (veg.DIE_TIME * 0.5), 0), 0, 1)[:, None]
        dead_col = np.where(veg.cause[:, None] == 1, [0.45, 0.50, 0.55, 1.0], [0.55, 0.32, 0.10, 1.0])
        base = base * (1 - k) + dead_col * k
        light = light * (1 - k) + (dead_col * 1.3).clip(0, 1) * k
        p = self.host.painter
        seg = 14
        # stíny (vpravo dole, světlo jde zleva shora jako ve stínování reliéfu)
        sh = disc_triangles(x0 + r * 0.35, y0 + r * 0.45, r * 1.02, seg)
        p.triangles(sh.reshape(-1, 2), (0.0, 0.08, 0.0, 0.45))
        rot = (veg.pos[:, 0] * 997 % 1.0) * 6.28
        crown = np.empty((n, seg * 3, 2), np.float32)
        if con.any():
            crown[con] = disc_triangles(x[con], y[con], r[con], seg, lobes=7, lobe_amp=0.22, rot=rot[con])
        if (~con).any():
            crown[~con] = disc_triangles(x[~con], y[~con], r[~con], seg, lobes=3, lobe_amp=0.10, rot=rot[~con])
        mid = disc_triangles(tx - r * 0.12, ty - r * 0.14, r * 0.62, seg)
        top = disc_triangles(tx - r * 0.25, ty - r * 0.28, r * 0.28, 10)
        midc = base * 0.4 + light * 0.6
        verts = np.concatenate([crown, mid, top], 1).reshape(-1, 2)
        cols = np.concatenate([np.repeat(base[:, None], seg * 3, 1), np.repeat(midc[:, None], seg * 3, 1),
                               np.repeat(light[:, None], 30, 1)], 1).reshape(-1, 4)
        p.triangles(verts, cols)

    # --- ovládání --------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        m = self.host.settings.map
        if name == "r":
            self.show_rivers = not self.show_rivers
            self._next_hydro = 0.0
            self.host.notify("Řeky " + ("zapnuty" if self.show_rivers else "vypnuty"))
        elif name == "a":
            self.show_animals = not self.show_animals
            self.host.notify("Zvířata " + ("zapnuta" if self.show_animals else "vypnuta"))
        elif name in ("up", "down"):
            self.params.snow = float(np.clip(self.params.snow + (0.005 if name == "up" else -0.005), 0.02, 0.2))
            self.params.treeline = self.params.snow - 0.017
            self._next_check = self._next_hydro = 0.0
            self.host.notify(f"Sněžná čára {self.host.altitude(self.params.snow):.0f} m n. m.")
        elif name in ("left", "right"):
            m.water_level += 0.005 if name == "right" else -0.005
            self._next_check = self._next_hydro = 0.0
            self.host.notify(f"Hladina moře {self.host.altitude(m.water_level):.0f} m n. m.")
        elif name == "backspace":
            self.reset()
            self.host.notify("Nový svět")
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = []
        if self.biome is not None:
            frac = np.bincount(self.biome.ravel(), minlength=6) / self.biome.size * 100
            lines.append(" · ".join(f"{n} {f:.0f} %" for n, f in zip(eco.BIOME_NAMES, frac)))
        if self.veg is not None:
            lines.append(f"Stromy: {int(self.veg.alive.sum())} (uhynulo {self.veg.deaths})")
        if self.animals is not None:
            from ..sim.fauna import BY_KEY
            c = {k: v for k, v in self.animals.counts().items() if v}
            lines.append("Zvířata: " + (", ".join(f"{BY_KEY[k].name} {v}" for k, v in c.items()) or "žádná"))
        if self.hydro is not None:
            lines.append(f"Řeky: {int((self.hydro.river > 0).sum())} buněk · jezera: {int((self.hydro.lake > 0).sum())}")
        return lines
