"""Živý ekosystém: na písku roste krajina, která reaguje na každou změnu terénu.

* biomy podle výšky a sklonu – moře a pláže v nížinách, louky, lesy na svazích,
  skály na strmých stěnách a nad hranicí lesa, sníh na vrcholcích,
* řeky stékají z hor údolími (akumulace toku D8), ve sníženinách vznikají jezera,
* stromy postupně vyrůstají a hynou, když je zasypeš, vykopeš nebo zatopíš,
* srnky se pasou na loukách (utíkají před rukou), nad krajinou krouží ptáci.

Logika je v ``sandbox/sim/ecosystem.py``, tady je jen kreslení.
"""

from __future__ import annotations

import math

import cv2
import moderngl
import numpy as np

from ..sim import ecosystem as eco
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

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

void main() {
    vec2 uv = v_uv;
    vec2 p = uv * u_box;               // metry
    float h = height_at(uv);
    float s = length(height_grad(uv));
    float n = vnoise(p * 45.0) * 0.6 + vnoise(p * 140.0) * 0.4;

    vec3 meadow = mix(vec3(0.42, 0.85, 0.12), vec3(0.62, 0.92, 0.18), n);
    vec3 forest = mix(vec3(0.05, 0.40, 0.10), vec3(0.10, 0.52, 0.14), n);
    vec3 rock = mix(vec3(0.46, 0.40, 0.37), vec3(0.66, 0.60, 0.56), n);
    vec3 snow = vec3(0.96, 0.98, 1.0);
    vec3 beach = mix(vec3(1.0, 0.86, 0.42), vec3(1.0, 0.93, 0.60), n);

    float wl = u_water_level;
    float fo = max(smoothstep(u_forest_h - 0.003, u_forest_h + 0.003, h),
                   smoothstep(u_forest_s - 0.03, u_forest_s + 0.03, s));
    vec3 c = mix(meadow, forest, fo);
    float ro = max(smoothstep(u_treeline - 0.003, u_treeline + 0.003, h + 0.004 * (n - 0.5)),
                   smoothstep(u_rock_s - 0.06, u_rock_s + 0.06, s));
    c = mix(c, rock, ro);
    float sn = smoothstep(u_snow - 0.002, u_snow + 0.002, h + 0.006 * (n - 0.5));
    c = mix(c, snow, sn);
    c = mix(c, beach, 1.0 - smoothstep(wl + u_beach - 0.0015, wl + u_beach + 0.0015, h));
    c *= mix(0.50, 1.18, hillshade(uv, 2.5));
    // třpyt sněhu
    float tw = step(0.985, hash(floor(p * 260.0) + floor(u_clock * 3.0)));
    c += sn * tw * 0.5;
    // jemné vrstevnice (orientace v terénu)
    vec2 cl = contours(h);
    c = mix(c, c * 0.55, cl.y * 0.6);

    // voda: moře, jezera, řeky
    vec4 hy = texture(u_hydro, uv);
    float depth = max(wl - h, sample_bicubic(u_lake, uv) - h);
    float river = smoothstep(0.30, 0.42, sample_bicubic(u_hydro, uv));
    float wa = max(smoothstep(0.0, 0.003, depth), river);
    if (wa > 0.0) {
        float d01 = clamp(depth / 0.04, 0.0, 1.0);
        vec3 wc = mix(vec3(0.15, 0.62, 1.0), vec3(0.02, 0.14, 0.62), d01);
        float wave = sin(p.x * 70.0 + u_clock * 1.7 + 3.0 * vnoise(p * 20.0)) * sin(p.y * 60.0 - u_clock * 1.3);
        wc += 0.07 * wave * (1.0 - d01);
        // řeka: protáhlé světlé proužky unášené po proudu
        vec2 dir = hy.gb;
        float L = length(dir);
        dir = L > 1e-3 ? dir / L : vec2(1.0, 0.0);
        vec2 perp = vec2(-dir.y, dir.x);
        vec2 q = vec2(dot(p, dir) * 25.0 - u_clock * 2.2, dot(p, perp) * 160.0);
        float streak = smoothstep(0.55, 0.85, vnoise(q)) * 0.8 + smoothstep(0.6, 0.9, vnoise(q * 1.7 + 13.0)) * 0.5;
        vec3 rc = mix(vec3(0.05, 0.50, 1.0), vec3(0.75, 0.97, 1.0), clamp(streak, 0.0, 1.0));
        wc = mix(wc, rc, river * (1.0 - smoothstep(0.0, 0.006, depth)));
        c = mix(c, wc, wa);
    }
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
    description = "Na písku roste živá krajina: řeky, jezera, lesy, sníh na horách a zvířata."
    order = 80
    category = "zážitek"
    intro = "Postav hory a vykopej údolí – z hor potečou řeky, na svazích vyroste les a na loukách se budou pást srnky."
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
        self.animals: eco.Animals | None = None
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
            self.animals = eco.Animals(hm.size, n_deer=max(3, int(4 * area)), n_birds=4)

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
        self.animals.step(dt, self.biome, self.wet, hm.hand_points)

    # --- kreslení -------------------------------------------------------------------------------
    def render(self) -> None:
        if self.hydro_tex is None:
            return
        p = self.params
        self.host.draw_fullscreen(self.prog, u_hydro=(self.hydro_tex, 1), u_lake=(self.lake_tex, 2), u_clock=float(self.clock),
                                  u_beach=p.beach, u_forest_h=p.forest_height, u_forest_s=p.forest_slope,
                                  u_treeline=p.treeline, u_snow=p.snow, u_rock_s=p.rock_slope)
        self._draw_trees()
        if self.show_animals:
            self._draw_deer()
            self._draw_birds()

    def _px(self) -> tuple[float, float]:
        cw, ch = self.host.canvas_size
        bw, bd = self.host.box_size
        return cw / bw, ch / bd

    def _draw_trees(self) -> None:
        veg = self.veg
        if veg is None or not len(veg):
            return
        sx, sy = self._px()
        x = veg.pos[:, 0] * sx
        y = veg.pos[:, 1] * sy
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
        sh = disc_triangles(x + r * 0.35, y + r * 0.45, r * 1.02, seg)
        p.triangles(sh.reshape(-1, 2), (0.0, 0.08, 0.0, 0.45))
        rot = (veg.pos[:, 0] * 997 % 1.0) * 6.28
        crown = np.empty((n, seg * 3, 2), np.float32)
        if con.any():
            crown[con] = disc_triangles(x[con], y[con], r[con], seg, lobes=7, lobe_amp=0.22, rot=rot[con])
        if (~con).any():
            crown[~con] = disc_triangles(x[~con], y[~con], r[~con], seg, lobes=3, lobe_amp=0.10, rot=rot[~con])
        mid = disc_triangles(x - r * 0.12, y - r * 0.14, r * 0.62, seg)
        top = disc_triangles(x - r * 0.25, y - r * 0.28, r * 0.28, 10)
        midc = base * 0.4 + light * 0.6
        verts = np.concatenate([crown, mid, top], 1).reshape(-1, 2)
        cols = np.concatenate([np.repeat(base[:, None], seg * 3, 1), np.repeat(midc[:, None], seg * 3, 1),
                               np.repeat(light[:, None], 30, 1)], 1).reshape(-1, 4)
        p.triangles(verts, cols)

    def _draw_deer(self) -> None:
        a = self.animals
        if a is None or a.n_deer == 0:
            return
        sx, sy = self._px()
        p = self.host.painter
        L = 0.032 * sx  # délka těla v px
        for (x, y), hd, vis in zip(a.deer, a.heading, a.visible):
            if vis < 0.05:
                continue
            cx, cy = x * sx, y * sy
            c, s = math.cos(hd), math.sin(hd)
            # tělo = protáhlý šestiúhelník
            body = np.array([[1.0, 0], [0.5, 0.42], [-0.5, 0.42], [-1.0, 0], [-0.5, -0.42], [0.5, -0.42]]) * L * 0.5
            rot = np.array([[c, s], [-s, c]])
            pts = body @ rot + [cx, cy]
            tris = np.array([[pts[0], pts[i], pts[i + 1]] for i in range(1, 5)]).reshape(-1, 2)
            shadow = tris + [L * 0.15, L * 0.2]
            p.triangles(shadow, (0, 0, 0, 0.35 * vis))
            p.triangles(tris, (0.62, 0.34, 0.12, vis))
            hx, hy = cx + c * L * 0.62, cy + s * L * 0.62
            p.circle(hx, hy, L * 0.22, (0.50, 0.26, 0.08, vis), segments=10)
            p.circle(cx - c * L * 0.48, cy - s * L * 0.48, L * 0.13, (1, 1, 1, vis), segments=8)  # bílé zrcátko

    def _draw_birds(self) -> None:
        a = self.animals
        if a is None:
            return
        sx, sy = self._px()
        p = self.host.painter
        pos, head = a.birds()
        span = 0.04 * sx
        for k, ((x, y), hd) in enumerate(zip(pos, head)):
            cx, cy = x * sx, y * sy
            flap = 0.45 + 0.35 * math.sin(self.clock * 9 + k * 1.7)
            c, s = math.cos(hd), math.sin(hd)
            back = -flap
            wing = []
            for side in (-1, 1):
                wx = cx + (-s * side) * span * 0.5 + c * back * span * 0.35
                wy = cy + (c * side) * span * 0.5 + s * back * span * 0.35
                wing.append((wx, wy))
            line = [wing[0], (cx + c * span * 0.12, cy + s * span * 0.12), wing[1]]
            shadow = [(u + span * 0.6, v + span * 0.8) for u, v in line]
            p.polyline(shadow, (0, 0, 0, 0.35), width=span * 0.16)
            p.polyline(line, (0.05, 0.05, 0.08, 1), width=span * 0.2)
            p.polyline(line, (1, 1, 1, 1), width=span * 0.1)

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
        if self.hydro is not None:
            lines.append(f"Řeky: {int((self.hydro.river > 0).sum())} buněk · jezera: {int((self.hydro.lake > 0).sum())}")
        return lines
