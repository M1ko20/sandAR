"""Voda: déšť, stékání ze svahů, jezera a povodně.

Simulace mělké vody metodou „virtuálních trubek“ (Mei et al. 2007): mezi
sousedními buňkami teče voda úměrně rozdílu hladin. Z každé dlaně zvednuté nad písek
(aspoň ~25 cm) prší – na hřbetu ruky se promítne bouřkový mrak (občas z něj šlehne blesk),
kolem dopadají kapky a na hladině se šíří kroužky. Ruka, která modeluje v písku, neprší,
aby šlo kopat koryta. Klávesou R prší všude.

Vzhled: světlé „proudnice“ unášené proudem ukazují, kudy voda stéká; písek, po kterém
voda tekla, zůstane chvíli tmavý a mokrý; na soutocích se dělá pěna; mělčiny mají
světelné vlnky (kaustiky) a hladina se leskne.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

from ..sim.lava import blur3
from ..sim.water import WaterSim
from .base import Game

__all__ = ["WaterGame", "WaterSim", "StormFx"]

N_RIP = 32  # kroužky na hladině (dopady kapek, blesky)
N_STORM = 4  # bouřkové mraky (ruce)

FRAG = """
uniform sampler2D u_water;
uniform sampler2D u_vel;     // rychlost vody [m/s] (rg)
uniform sampler2D u_aux;     // r = mokrý písek 0..1, g = pěna (soutoky, peřeje)
uniform vec4 u_rip[32];      // kroužky: x, y [m], stáří [s], síla
uniform int u_nrip;
uniform vec4 u_storm[4];     // mraky: x, y [m], síla 0..1, -
uniform int u_nstorm;
uniform vec4 u_flash;        // blesk: x, y [m], jas, -

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;
    float h = height_at(uv);
    vec3 c = land_color((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4));
    c *= mix(0.55, 1.12, hillshade(uv, 2.6));
    vec2 cl = contours(h);
    c = mix(c, c * 0.45, cl.x * 0.6);
    c = mix(c, c * 0.25, cl.y * 0.8);

    vec2 aux = texture(u_aux, uv).rg;
    // mokrý písek: tmavší, sytější, do modra – stopa, kudy voda tekla
    float wet = clamp(aux.r, 0.0, 1.0);
    c = mix(c, c * vec3(0.66, 0.72, 0.86), wet * 0.65);

    // stín bouřkových mraků: krajina pod rukou ztmavne a zmodrá
    float shade = 0.0;
    for (int i = 0; i < u_nstorm; ++i) {
        float d = length(pm - u_storm[i].xy);
        shade = max(shade, u_storm[i].z * (1.0 - smoothstep(0.05, 0.30, d)));
    }
    c = mix(c, c * vec3(0.55, 0.62, 0.85), shade * 0.75);

    float w = max(sample_bicubic(u_water, uv), 0.0);
    vec2 v = texture(u_vel, uv).rg;
    float sp = length(v);
    // pěna: peřeje (rychlý mělký proud) + soutoky (spočteno v Pythonu)
    float foam = clamp((sp - 0.10) / 0.25, 0.0, 1.0) * (1.0 - smoothstep(0.002, 0.010, w)) * 0.55;
    foam = max(foam, clamp(aux.g, 0.0, 1.0) * 0.85);

    // kroužky na hladině po dopadu kapek: vlnka (světlý hřbet + tmavší důl za ním)
    float rip = 0.0;
    for (int i = 0; i < u_nrip; ++i) {
        vec4 r = u_rip[i];
        float d = length(pm - r.xy);
        float R = 0.006 + r.z * (0.07 + 0.05 * r.w);
        float x = (d - R) / (0.004 + 0.006 * r.w);
        float env = r.w * pow(max(1.0 - r.z / 1.1, 0.0), 1.5);
        rip += env * (exp(-x * x) - 0.55 * exp(-(x + 1.6) * (x + 1.6)));
    }

    vec4 wr = water_shade(c, uv, w, v, foam);
    vec3 col = wr.rgb;
    float a = wr.a;
    float calm = 1.0 - clamp(sp / 0.15, 0.0, 1.0);
    // kaustiky: světelné žilky na dně mělké vody
    float cn = abs(_wnoise(pm * 30.0 + vec2(u_time * 0.35, u_time * 0.2))
                 - _wnoise(pm * 30.0 * 1.27 + vec2(-u_time * 0.28, u_time * 0.31) + 5.3));
    float caus = (1.0 - smoothstep(0.0, 0.07, cn)) * smoothstep(0.002, 0.006, w) * (1.0 - smoothstep(0.012, 0.03, w));
    col += vec3(0.35, 0.55, 0.60) * caus * a * 0.55;
    // lesk hladiny: pomalu putující měkký odlesk oblohy na klidné hlubší vodě
    float sheen = smoothstep(0.55, 0.85, _wnoise(pm * vec2(5.0, 9.0) + vec2(u_time * 0.07, -u_time * 0.04)));
    col += vec3(0.20, 0.32, 0.42) * sheen * calm * smoothstep(0.005, 0.02, w) * a;
    // kroužky
    col += vec3(0.55, 0.80, 1.00) * clamp(rip, -0.4, 1.0) * a;

    // blesk: záblesk krajiny kolem místa úderu (a slabě všude)
    float fd = length(pm - u_flash.xy);
    col += vec3(0.55, 0.65, 0.95) * u_flash.z * (0.05 + 0.65 * exp(-fd * fd / (0.07 * 0.07)));
    f_color = vec4(col, 1.0);
}
"""


def _seg_quads(a: np.ndarray, b: np.ndarray, w) -> np.ndarray:
    """Úsečky a→b jako čtyřúhelníky (2 trojúhelníky) šířky w – (N*6, 2)."""
    d = b - a
    n = np.hypot(d[:, 0], d[:, 1])[:, None]
    w = np.asarray(w, np.float32).reshape(-1, 1) if np.ndim(w) else w
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(n, 1e-6) * (w / 2)
    return np.stack([a + nrm, a - nrm, b - nrm, a + nrm, b - nrm, b + nrm], 1).reshape(-1, 2)


def _discs(cx: np.ndarray, cy: np.ndarray, r: np.ndarray, seg: int = 16) -> np.ndarray:
    """N kruhů jako vějíře trojúhelníků – (N, seg*3, 2)."""
    ang = np.linspace(0, 2 * math.pi, seg + 1)[None, :]
    ring = np.stack([cx[:, None] + r[:, None] * np.cos(ang), cy[:, None] + r[:, None] * np.sin(ang)], -1)
    center = np.broadcast_to(np.stack([cx, cy], -1)[:, None, :], (len(cx), seg, 2))
    return np.stack([center, ring[:, :-1], ring[:, 1:]], 2).reshape(len(cx), seg * 3, 2)


def _rgba(col, n: int, alpha=None) -> np.ndarray:
    c = np.tile(np.asarray(col, np.float32), (n, 1))
    if alpha is not None:
        c[:, 3] = alpha
    return c


class StormFx:
    """Bouřka pod rukou, jak ji vidí člověk shora (souřadnice v metrech bedny).

    * bouřkový mrak na hřbetu ruky – tmavé chuchvalce, které se pomalu otáčejí, při blesku
      se rozsvítí zevnitř,
    * blesky: klikatý výboj z okraje mraku k zemi, dva záblesky, rázový kruh v místě úderu,
    * kapky: světlé čárky letí ven od ruky (shora tak vypadá pád), dopadnou, odstříknou
      a na hladině rozběhnou kroužek (ten kreslí shader – ``landed``).
    """

    FALL = 0.20  # [s] let kapky
    SPLASH = 0.45  # [s] odstřik po dopadu
    RATE = 120.0  # [kapek/s] z jednoho mraku
    BOLT_LIFE = 0.45  # [s]

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        # x0, y0, x1, y1, stáří, úhel odstřiku
        self.drops = np.zeros((0, 6), np.float32)
        self.landed = np.zeros((0, 2), np.float32)  # kapky, které právě dopadly (tento snímek)
        self.bolts: list[dict] = []
        self.strikes: list[tuple[float, float]] = []  # údery blesku v tomto snímku
        self._next_bolt: dict[int, float] = {}
        self.t = 0.0

    def clear(self) -> None:
        self.drops = self.drops[:0]
        self.bolts = []

    # --- logika --------------------------------------------------------------------------------
    def update(self, dt: float, sources: list[tuple[int, float, float, float]]) -> None:
        """sources = [(id, x, y, intenzita 0..1)] – kde právě prší."""
        self.t += dt
        rng = self.rng
        landed = []
        if len(self.drops):
            age0 = self.drops[:, 4].copy()
            self.drops[:, 4] += dt
            hit = (age0 < self.FALL) & (self.drops[:, 4] >= self.FALL)
            if hit.any():
                landed.append(self.drops[hit, 2:4])
            self.drops = self.drops[self.drops[:, 4] < self.FALL + self.SPLASH]
        self.landed = np.concatenate(landed) if landed else np.zeros((0, 2), np.float32)
        for _, x, y, k in sources:
            n = rng.poisson(self.RATE * k * dt)
            if not n:
                continue
            ang = rng.uniform(0, 2 * math.pi, n)
            r1 = np.sqrt(rng.uniform(0.25, 1.0, n)) * 0.14  # dopad 7–14 cm od dlaně (pod rukou je stín)
            r0 = r1 * 0.40
            new = np.stack([x + r0 * np.cos(ang), y + r0 * np.sin(ang), x + r1 * np.cos(ang), y + r1 * np.sin(ang),
                            np.zeros(n), rng.uniform(0, 2 * math.pi, n)], 1).astype(np.float32)
            self.drops = np.vstack([self.drops, new])[-1200:]
        # blesky: z plně rozvinutého mraku jednou za 1,6–4 s
        self.strikes = []
        ids = {s[0] for s in sources}
        for hid in list(self._next_bolt):
            if hid not in ids:
                del self._next_bolt[hid]
        for hid, x, y, k in sources:
            nxt = self._next_bolt.setdefault(hid, self.t + rng.uniform(0.6, 1.4))
            if k > 0.9 and self.t >= nxt:
                self._next_bolt[hid] = self.t + rng.uniform(1.6, 4.0)
                self.bolts.append(self._make_bolt(hid, x, y))
                self.strikes.append(self.bolts[-1]["end"])
        for b in self.bolts:
            b["age"] += dt
        self.bolts = [b for b in self.bolts if b["age"] < self.BOLT_LIFE]

    def _make_bolt(self, hid: int, x: float, y: float) -> dict:
        rng = self.rng
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.15, 0.24)
        start = np.array([x + 0.05 * math.cos(ang), y + 0.05 * math.sin(ang)])
        end = np.array([x + dist * math.cos(ang + rng.normal(0, 0.25)), y + dist * math.sin(ang + rng.normal(0, 0.25))])
        main = self._zigzag(start, end, 0.022)
        paths = [main]
        for _ in range(rng.integers(1, 3)):  # odbočky
            i = int(rng.integers(len(main) // 4, len(main) * 3 // 4))
            d = main[-1] - main[0]
            a = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
            L = np.hypot(*d) * rng.uniform(0.25, 0.45)
            paths.append(self._zigzag(main[i], main[i] + L * np.array([math.cos(a), math.sin(a)]), 0.010))
        return {"id": hid, "paths": paths, "end": (float(end[0]), float(end[1])), "age": 0.0}

    def _zigzag(self, a: np.ndarray, b: np.ndarray, amp: float, levels: int = 4) -> np.ndarray:
        pts = np.array([a, b], float)
        for _ in range(levels):
            mid = (pts[:-1] + pts[1:]) / 2
            d = pts[1:] - pts[:-1]
            n = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
            mid += n * self.rng.normal(0, amp, (len(mid), 1))
            out = np.empty((len(pts) + len(mid), 2))
            out[0::2], out[1::2] = pts, mid
            pts = out
            amp *= 0.55
        return pts

    @classmethod
    def bolt_intensity(cls, age: float) -> float:
        """Dva záblesky a dohasínání (bez prudkého stroboskopu)."""
        if age < 0.06:
            return 1.0
        if age < 0.12:
            return 0.35
        if age < 0.20:
            return 0.9
        return max(0.0, 0.9 * (1 - (age - 0.20) / (cls.BOLT_LIFE - 0.20))) ** 1.5

    def flash(self) -> tuple[float, float, float]:
        """Nejsilnější probíhající blesk: (x, y, jas) pro shader."""
        best = (0.0, 0.0, 0.0)
        for b in self.bolts:
            k = self.bolt_intensity(b["age"])
            if k > best[2]:
                best = (*b["end"], k)
        return best

    # --- kreslení ------------------------------------------------------------------------------
    def draw(self, host, clouds: list[tuple[int, float, float, float]]) -> None:
        p = host.painter
        cw, ch = host.canvas_size
        sx, sy = cw / host.box_size[0], ch / host.box_size[1]
        px = sx / 800  # 1 „bod“ při plátně 1280 px na 1,6 m
        verts: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        d = self.drops
        if len(d):
            age = d[:, 4]
            falling = age < self.FALL
            if falling.any():
                f = d[falling]
                k = (f[:, 4] / self.FALL)[:, None]
                head = (f[:, 0:2] + (f[:, 2:4] - f[:, 0:2]) * k) * [sx, sy]
                tail = (f[:, 0:2] + (f[:, 2:4] - f[:, 0:2]) * np.maximum(k - 0.5, 0)) * [sx, sy]
                q = _seg_quads(head, tail, 5.0 * px)
                verts.append(q)
                a = np.repeat(0.55 + 0.4 * k[:, 0], 6)
                cols.append(_rgba((0.78, 0.92, 1.0, 1.0), len(q), a))
            sp = ~falling
            if sp.any():
                s = d[sp]
                k = (s[:, 4] - self.FALL) / self.SPLASH
                cx, cy = s[:, 2] * sx, s[:, 3] * sy
                # kroužek po dopadu
                seg = 12
                ang = np.linspace(0, 2 * math.pi, seg + 1)
                rad = (0.004 + 0.016 * np.sqrt(k)) * sx
                ring = np.stack([cx[:, None] + rad[:, None] * np.cos(ang), cy[:, None] + rad[:, None] * np.sin(ang)], -1)
                q = _seg_quads(ring[:, :-1].reshape(-1, 2), ring[:, 1:].reshape(-1, 2),
                               np.repeat(np.maximum(2.5, 5.0 * (1 - k)) * px, seg))
                verts.append(q)
                cols.append(_rgba((0.85, 0.96, 1.0, 1.0), len(q), np.repeat(0.85 * (1 - k) ** 1.5, seg * 6)))
                # záblesk kapky v místě dopadu
                early = k < 0.35
                if early.any():
                    kk = k[early] / 0.35
                    disc = _discs(cx[early], cy[early], (5.0 - 2.0 * kk) * px, 8).reshape(-1, 2)
                    verts.append(disc)
                    cols.append(_rgba((0.95, 1.0, 1.0, 1.0), len(disc), np.repeat(0.9 * (1 - kk), 24)))
        if verts:
            p.triangles(np.concatenate(verts), np.concatenate(cols))
        for hid, x, y, k in clouds:
            lit = max((self.bolt_intensity(b["age"]) for b in self.bolts if b["id"] == hid), default=0.0)
            self._cloud(p, x * sx, y * sy, 0.08 * sx, k, lit * 0.6, hid)
        self._draw_bolts(p, sx, sy, px)  # blesk šlehá zpod okraje mraku – kreslí se navrch

    def _draw_bolts(self, p, sx: float, sy: float, px: float) -> None:
        verts: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        for b in self.bolts:
            k = self.bolt_intensity(b["age"])
            if k <= 0.02:
                continue
            for j, path in enumerate(b["paths"]):
                pts = path * [sx, sy]
                main = j == 0
                for w, col, al in ((30 if main else 16, (0.55, 0.40, 1.0), 0.30),
                                   (14 if main else 8, (0.60, 0.80, 1.0), 0.75),
                                   (6 if main else 4, (1.0, 1.0, 1.0), 1.0)):
                    q = _seg_quads(pts[:-1], pts[1:], w * px)
                    verts.append(q)
                    cols.append(_rgba((*col, al * k), len(q)))
                    # klouby, ať se čára netrhá
                    jd = _discs(pts[:, 0], pts[:, 1], np.full(len(pts), w * px / 2), 8).reshape(-1, 2)
                    verts.append(jd)
                    cols.append(_rgba((*col, al * k), len(jd)))
            # úder: zářící bod + rázový kruh
            ex, ey = b["end"][0] * sx, b["end"][1] * sy
            age = b["age"]
            core = _discs(np.array([ex]), np.array([ey]), np.array([(10 + 8 * k) * px]), 20).reshape(-1, 2)
            verts.append(core)
            cols.append(_rgba((1.0, 1.0, 1.0, 0.9 * k), len(core)))
            r0 = (0.01 + 0.09 * age / self.BOLT_LIFE) * sx
            ang = np.linspace(0, 2 * math.pi, 33)
            ring = np.stack([ex + r0 * np.cos(ang), ey + r0 * np.sin(ang)], 1)
            q = _seg_quads(ring[:-1], ring[1:], (7 * (1 - age / self.BOLT_LIFE) + 2) * px)
            verts.append(q)
            cols.append(_rgba((0.75, 0.90, 1.0, 0.8 * (1 - age / self.BOLT_LIFE)), len(q)))
        if verts:
            p.triangles(np.concatenate(verts), np.concatenate(cols))

    # chuchvalce mraku: úhel, vzdálenost (×r), velikost (×r)
    _PUFFS = np.array([(0.0, 0.0, 0.66), (0.4, 0.55, 0.50), (1.3, 0.62, 0.44), (2.2, 0.50, 0.52),
                       (3.1, 0.66, 0.42), (4.0, 0.55, 0.50), (4.9, 0.64, 0.45), (5.7, 0.52, 0.46),
                       (0.8, 0.98, 0.30), (1.9, 1.02, 0.26), (2.7, 0.95, 0.31), (3.6, 1.04, 0.25),
                       (4.5, 0.97, 0.29), (5.4, 1.00, 0.27)])

    def _cloud(self, p, x: float, y: float, r: float, k: float, lit: float, hid: int) -> None:
        """Bouřkový mrak: měkké chuchvalce (tmavé tělo, světlé vršky), pomalu se točí a dýchá;
        při blesku se rozsvítí zevnitř."""
        t = self.t
        P = self._PUFFS
        n = len(P)
        rot = t * 0.30 + hid * 1.7
        wob = 1 + 0.08 * np.sin(t * 1.3 + np.arange(n) * 1.9)
        ang = P[:, 0] + rot * np.where(P[:, 1] > 0.9, 1.0, 0.55)
        dist = P[:, 1] * r * (0.55 + 0.45 * k)
        cx = x + dist * np.cos(ang)
        cy = y + dist * np.sin(ang)
        rr = P[:, 2] * r * wob * (0.45 + 0.55 * k)
        flash = np.array([0.92, 0.92, 1.0])
        verts, cols = [], []

        def layer(lx, ly, lr, col, al, inner):
            c = np.array(col) * (1 - lit * 0.85) + flash * lit * 0.85
            v, cc = _soft_discs(lx, ly, lr, (*c, al * k), inner)
            verts.append(v)
            cols.append(cc)

        layer(np.array([x + r * 0.15]), np.array([y + r * 0.2]), np.array([r * 1.75]), (0.03, 0.05, 0.14), 0.55, 0.3)
        layer(cx + r * 0.08, cy + r * 0.11, rr * 1.12, (0.08, 0.10, 0.20), 0.7, 0.55)  # spodek
        layer(cx, cy, rr, (0.32, 0.37, 0.52), 1.0, 0.72)  # tělo
        layer(cx - rr * 0.22, cy - rr * 0.26, rr * 0.75, (0.50, 0.56, 0.72), 0.55, 0.3)  # vršky chuchvalců
        # jedno velké světlo pro celý mrak (shora zleva) – mrak působí jako objem, ne jako hrozen
        layer(np.array([x - r * 0.35]), np.array([y - r * 0.40]), np.array([r * 0.95]), (0.80, 0.85, 0.97), 0.55, 0.15)
        if lit > 0.02:  # blesk uvnitř mraku
            layer(np.array([x]), np.array([y]), np.array([r * 1.3]), (0.85, 0.80, 1.0), 0.8 * lit, 0.1)
        p.triangles(np.concatenate(verts), np.concatenate(cols))


def _soft_discs(cx: np.ndarray, cy: np.ndarray, r: np.ndarray, col, inner: float = 0.6, seg: int = 20):
    """N měkkých kruhů: plné do ``inner``×r, pak plynule do průhledna. Vrací (verts, colors)."""
    n = len(cx)
    ang = np.linspace(0, 2 * math.pi, seg + 1)[None, :]
    cs, sn = np.cos(ang), np.sin(ang)
    ri, ro = r[:, None] * inner, r[:, None]
    pi = np.stack([cx[:, None] + ri * cs, cy[:, None] + ri * sn], -1)
    po = np.stack([cx[:, None] + ro * cs, cy[:, None] + ro * sn], -1)
    c = np.broadcast_to(np.stack([cx, cy], -1)[:, None, :], (n, seg, 2))
    fan = np.stack([c, pi[:, :-1], pi[:, 1:]], 2).reshape(n, -1, 2)
    a, b, cc, d = pi[:, :-1], pi[:, 1:], po[:, 1:], po[:, :-1]
    ring = np.stack([a, b, cc, a, cc, d], 2).reshape(n, -1, 2)
    verts = np.concatenate([fan, ring], 1).reshape(-1, 2)
    solid = np.asarray(col, np.float32)
    clear = solid.copy()
    clear[3] = 0
    ring_cols = np.array([solid, solid, clear, solid, clear, clear], np.float32)
    per = np.concatenate([np.tile(solid, (seg * 3, 1)), np.tile(ring_cols, (seg, 1))])
    return verts, np.tile(per, (n, 1))


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

    WET_FADE = 14.0  # [s] za jak dlouho mokrý písek oschne
    N_STREAKS = 320  # světlé proudnice unášené vodou

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.sim: WaterSim | None = None
        self.tex = None
        self.vel_tex = None
        self.aux_tex = None
        self.raining = False
        self.drain = False
        self.fx = StormFx()
        self.clouds: list[tuple[int, float, float, float]] = []
        self._cloud_k: dict[int, float] = {}
        self._xy = None
        self.t = 0.0
        self.last_rain = -99.0
        self.wet: np.ndarray | None = None
        self.foam: np.ndarray | None = None
        self.vx = self.vy = None
        self.rip = np.zeros((0, 4), np.float32)  # x, y, stáří, síla
        self.streaks = np.zeros((0, 5), np.float32)  # x, y, stáří, život, náhodná fáze
        self.rng = np.random.default_rng(1)

    def stop(self) -> None:
        self.prog.release()
        for tex in (self.tex, self.vel_tex, self.aux_tex):
            if tex is not None:
                tex.release()
        self.tex = self.vel_tex = self.aux_tex = None

    def update(self, hm, dt: float) -> None:
        if self.sim is None or self.sim.depth.shape != hm.shape:
            self.sim = WaterSim(hm.shape, hm.cell_size[0])
            for tex in (self.tex, self.vel_tex, self.aux_tex):
                if tex is not None:
                    tex.release()
            self.tex = self.host.float_texture((hm.shape[1], hm.shape[0]))
            self.vel_tex = self.host.ctx.texture((hm.shape[1], hm.shape[0]), 2, dtype="f4")
            self.vel_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.vel_tex.repeat_x = self.vel_tex.repeat_y = False
            self.aux_tex = self.host.ctx.texture((hm.shape[1], hm.shape[0]), 2, dtype="f4")
            self.aux_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.aux_tex.repeat_x = self.aux_tex.repeat_y = False
            self._xy = hm.cell_centers()
            self.wet = np.zeros(hm.shape, np.float32)
            self.foam = np.zeros(hm.shape, np.float32)
            self.streaks = self.streaks[:0]
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
        assert self.tex is not None and self.vel_tex is not None and self.aux_tex is not None
        self.tex.write(self.sim.depth.tobytes())
        vx, vy = self.sim.velocity()
        self.vx, self.vy = blur3(vx), blur3(vy)
        self.vel_tex.write(np.dstack([self.vx, self.vy]).astype(np.float32).tobytes())
        self._surface(hm, dt)
        self.aux_tex.write(np.dstack([self.wet, self.foam]).astype(np.float32).tobytes())
        self._ripples(hm, dt)
        self._advect_streaks(hm, dt)

    # --- vzhled hladiny (jen vizuální stav) -------------------------------------------------------
    def _surface(self, hm, dt: float) -> None:
        """Mokrý písek (kudy tekla voda) a pěna na soutocích (proud se sbíhá)."""
        assert self.sim is not None and self.wet is not None and self.foam is not None
        d = self.sim.depth
        now = np.clip((d - np.float32(0.00005)) * np.float32(1 / 0.0006), 0, 1)
        np.maximum(self.wet - np.float32(dt / self.WET_FADE), now, out=self.wet)
        cx, cy = hm.cell_size
        vx, vy = self.vx, self.vy
        div = np.zeros_like(vx)  # centrální diference (okraje 0)
        div[:, 1:-1] = (vx[:, 2:] - vx[:, :-2]) * np.float32(0.5 / cx)
        div[1:-1, :] += (vy[2:, :] - vy[:-2, :]) * np.float32(0.5 / cy)
        sp = np.hypot(vx, vy)
        # soutok = proud se sbíhá (záporná divergence); u ústí do jezera se rychlá voda zastaví
        conv = np.clip((-div - np.float32(0.8)) * np.float32(1 / 3.0), 0, 1) * np.clip(sp * np.float32(1 / 0.06), 0, 1)
        conv *= d > 0.0004
        target = blur3(conv)
        self.foam += (target - self.foam) * min(1.0, dt * 3.0)

    def _ripples(self, hm, dt: float) -> None:
        """Kroužky na hladině: kapky dopadlé do vody, úder blesku do vody."""
        assert self.sim is not None
        gh, gw = hm.shape
        cx, cy = hm.cell_size
        rip = self.rip
        if len(rip):
            rip[:, 2] += dt
            rip = rip[rip[:, 2] < 1.1]
        new = []
        pts = self.fx.landed
        if len(pts):
            ix = np.clip((pts[:, 0] / cx).astype(int), 0, gw - 1)
            iy = np.clip((pts[:, 1] / cy).astype(int), 0, gh - 1)
            inwater = self.sim.depth[iy, ix] > 0.0015
            if inwater.any():
                q = pts[inwater]
                new.append(np.column_stack([q, np.zeros(len(q)), self.rng.uniform(0.45, 0.8, len(q))]))
        for x, y in self.fx.strikes:
            new.append(np.array([[x, y, 0.0, 1.6]]))
        if new:
            rip = np.vstack([rip] + new).astype(np.float32)
        # bez těch nejstarších, ale údery blesku přednostně
        if len(rip) > N_RIP:
            keep = np.argsort(rip[:, 2] - (rip[:, 3] > 1.2) * 10)[:N_RIP]
            rip = rip[keep]
        self.rip = rip

    def _advect_streaks(self, hm, dt: float) -> None:
        """Světlé proudnice: vznikají v tekoucí vodě a plují s proudem – je vidět, kudy a jak rychle teče."""
        assert self.sim is not None and self.vx is not None
        gh, gw = hm.shape
        cx, cy = hm.cell_size
        d = self.sim.depth
        s = self.streaks
        if len(s):
            ix = np.clip((s[:, 0] / cx).astype(int), 0, gw - 1)
            iy = np.clip((s[:, 1] / cy).astype(int), 0, gh - 1)
            v = np.stack([self.vx[iy, ix], self.vy[iy, ix]], 1)
            s[:, 0:2] += v * dt
            s[:, 2] += dt
            alive = (s[:, 2] < s[:, 3]) & (d[iy, ix] > 0.00008) & (np.hypot(v[:, 0], v[:, 1]) > 0.015)
            s = s[alive]
        # nové tam, kde voda teče
        free = self.N_STREAKS - len(s)
        if free > 0:
            sp = np.hypot(self.vx, self.vy)
            cand = np.flatnonzero((sp > 0.04) & (d > 0.0005))
            if len(cand):
                n = min(free, max(3, int(len(cand) * dt * 0.35)), 25)
                pick = self.rng.choice(cand, n)
                yy, xx = np.divmod(pick, gw)
                jit = self.rng.random((n, 2))
                new = np.column_stack([(xx + jit[:, 0]) * cx, (yy + jit[:, 1]) * cy, np.zeros(n),
                                       self.rng.uniform(0.8, 2.2, n), self.rng.random(n)]).astype(np.float32)
                s = np.vstack([s, new])
        self.streaks = s

    def _hand_rain(self, hm, dt: float, rate: float) -> None:
        """Ze zvednuté dlaně prší (Gaussův mráček ~ 5 cm). Mrak se plynule rozvine / rozplyne."""
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
                sources.append((hand.id, hand.x, hand.y, k))
        for hid in list(self._cloud_k):
            if hid not in alive:
                del self._cloud_k[hid]
        if rate > 0:
            for _, x, y, k in sources:
                r2 = (X - x) ** 2 + (Y - y) ** 2
                near = r2 < 0.15 ** 2
                self.sim.depth[near] += (rate * dt * k) * np.exp(-r2[near] / (2 * 0.05 ** 2))
        if sources:
            self.last_rain = self.t
        self.clouds = sources
        self.fx.update(dt, sources)

    # --- kreslení ------------------------------------------------------------------------------
    def render(self) -> None:
        if self.tex is None:
            return
        prog = self.prog
        rip = np.zeros((N_RIP, 4), np.float32)
        rip[:len(self.rip)] = self.rip[:N_RIP]
        storm = np.zeros((N_STORM, 4), np.float32)
        for i, (_, x, y, k) in enumerate(self.clouds[:N_STORM]):
            storm[i] = (x, y, k, 0)
        for name, arr in (("u_rip", rip), ("u_storm", storm)):
            try:
                prog[name].write(arr.tobytes())
            except KeyError:
                pass
        self.host.draw_fullscreen(prog, u_water=(self.tex, 1), u_vel=(self.vel_tex, 2), u_aux=(self.aux_tex, 3),
                                  u_nrip=int(min(len(self.rip), N_RIP)), u_nstorm=int(min(len(self.clouds), N_STORM)),
                                  u_flash=(*self.fx.flash(), 0.0))
        self._draw_streaks()
        self.fx.draw(self.host, self.clouds)
        flash = self.fx.flash()[2]
        self.host.glow(0.45 + 0.35 * flash, 0.80)
        if self.t - self.last_rain > 8.0 and self.sim is not None and self.sim.volume < 0.002:
            p = self.host.painter
            cw, ch = self.host.canvas_size
            size = max(18, ch // 24)
            p.text("Zvedni ruku vysoko nad písek – začne z ní pršet", cw / 2, ch - size * 0.7, size=size, anchor="bc",
                   bold=True, color=(1, 1, 1, 1), outline=max(2, size // 10), background=(0, 0.05, 0.15, 0.55),
                   padding=size * 0.3)

    def _draw_streaks(self) -> None:
        s = self.streaks
        if not len(s) or self.vx is None or self.sim is None:
            return
        cw, ch = self.host.canvas_size
        bw, bh = self.host.box_size
        sx, sy = cw / bw, ch / bh
        gh, gw = self.vx.shape
        ix = np.clip((s[:, 0] / bw * gw).astype(int), 0, gw - 1)
        iy = np.clip((s[:, 1] / bh * gh).astype(int), 0, gh - 1)
        v = np.stack([self.vx[iy, ix], self.vy[iy, ix]], 1)
        sp = np.hypot(v[:, 0], v[:, 1])
        L = np.clip(sp * 0.18, 0.006, 0.05)  # délka čárky [m] roste s rychlostí
        dirv = v / np.maximum(sp, 1e-6)[:, None]
        head = s[:, 0:2] * [sx, sy]
        tail = (s[:, 0:2] - dirv * L[:, None]) * [sx, sy]
        life = s[:, 2] / s[:, 3]
        fade = np.minimum(1.0, np.minimum(s[:, 2] / 0.25, (1 - life) / 0.3))
        deep = np.clip(self.sim.depth[iy, ix] / 0.02, 0, 1)
        alpha = fade * np.clip(sp / 0.10, 0.3, 1.0) * (0.55 - 0.25 * deep)
        w = (3.2 + 2.0 * np.clip(sp / 0.2, 0, 1)) * sx / 800
        q = _seg_quads(head, tail, np.repeat(w, 1))
        col = np.repeat(np.column_stack([np.full(len(s), 0.70), np.full(len(s), 0.95), np.ones(len(s)), alpha]), 6, 0)
        self.host.painter.triangles(q, col.astype(np.float32))

    def reset(self) -> None:
        if self.sim is not None:
            self.sim.clear()
        if self.wet is not None:
            self.wet[:] = 0
            self.foam[:] = 0
        self.streaks = self.streaks[:0]
        self.rip = self.rip[:0]
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
            if self.wet is not None:
                self.wet[:] = 0
                self.foam[:] = 0
            self.streaks = self.streaks[:0]
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
