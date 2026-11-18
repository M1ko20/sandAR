"""Hledání pokladu: pod pískem je zakopáno několik pokladů, návštěvníci kopou.

Každý poklad leží v absolutní výšce ``z`` (pár cm pod povrchem v místě, kde byl
písek při začátku kola vysoko). Když se povrch nad pokladem sníží až k ``z``
(Kinect vidí díru), poklad se odkryje: zlatá mince nebo drahokam s jiskrami.
Ruka nad pískem dostane nápovědu „zima / přihořívá / hoří“ a kroužek, jehož
poloměr je zhruba vzdálenost k nejbližšímu pokladu. Po nalezení všech pokladů
se ukáže vítězná obrazovka a začne nové kolo s novými místy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..render import hud
from .base import Game
from .kulicka import Sparks, fmt_time, ring_quads, soft_discs

FRAG = """
uniform sampler2D u_start;    // výšky na začátku kola
uniform vec4 u_tr[12];        // poklady: x, y (uv), blízko 0..1, nalezen 0/1
uniform int u_ntr;
uniform float u_victory;      // 0..1 vítězné rozzáření

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;
    float h = height_at(uv);
    float t = clamp((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4), 0.0, 1.0);
    // pouštní písek pokladové mapy: sytá okrová v dolících -> světlé zlatavé duny
    vec3 c = mix(vec3(0.82, 0.42, 0.10), vec3(1.0, 0.78, 0.36), smoothstep(0.10, 0.65, t));
    c = mix(c, vec3(1.0, 0.95, 0.70), smoothstep(0.70, 1.0, t));
    c *= mix(0.42, 1.22, hillshade(uv, 3.5));
    // vykopaná místa: tmavý vlhký písek do hnědočervena, se zlatavým leskem na dně
    float dug = texture(u_start, uv).r - h;
    float dk = smoothstep(0.004, 0.04, dug);
    c = mix(c, vec3(0.34, 0.09, 0.20), dk * 0.85);
    // nasypaná místa světlejší
    c = mix(c, vec3(1.0, 0.97, 0.80), smoothstep(0.006, 0.04, -dug) * 0.45);
    // vrstevnice jako na staré mapě (hnědý inkoust)
    vec2 cl = contours(h);
    c = mix(c, c * 0.55, cl.x * 0.5);
    c = mix(c, vec3(0.30, 0.10, 0.02), cl.y * 0.75);
    // připálené okraje mapy
    vec2 e = min(pm, u_box - pm);
    float edge = min(e.x, e.y);
    c *= mix(0.45, 1.0, smoothstep(0.0, 0.09, edge));
    c = mix(vec3(0.35, 0.12, 0.02), c, smoothstep(0.0, 0.035, edge) * 0.6 + 0.4);
    for (int i = 0; i < u_ntr; ++i) {
        vec4 tr = u_tr[i];
        vec2 d = (uv - tr.xy) * u_box;
        float r = length(d);
        if (tr.w > 0.5) {
            // nalezený poklad: zlatá záře a pomalu se točící paprsky
            float a = atan(d.y, d.x);
            float rays = pow(0.5 + 0.5 * cos(a * 9.0 + u_time * 0.7 + float(i)), 6.0);
            float rays2 = pow(0.5 + 0.5 * cos(a * 5.0 - u_time * 0.45 + float(i) * 2.0), 8.0);
            float fall = exp(-r / 0.07) * smoothstep(0.015, 0.03, r);
            c += vec3(1.0, 0.72, 0.15) * (0.55 * exp(-r * r / (0.04 * 0.04)) + 0.7 * max(rays, rays2) * fall);
        } else if (tr.z > 0.0) {
            // „už to skoro je“ – tepající zlatý třpyt v díře, kde zbývá jen trocha písku
            float beat = 0.5 + 0.5 * sin(u_time * (4.0 + 6.0 * tr.z));
            float tw = 0.5 + 0.5 * sin(u_time * 7.0 - r * 120.0);
            c += vec3(1.0, 0.80, 0.25) * tr.z * (0.35 + 0.45 * tw) * (0.6 + 0.4 * beat)
                 * exp(-r * r / (0.05 * 0.05));
        }
    }
    if (u_victory > 0.0) {
        // vítězství: celá mapa zezlátne
        c = mix(c, c * vec3(1.15, 1.0, 0.7) + vec3(0.18, 0.12, 0.0), u_victory * 0.5);
    }
    f_color = vec4(c, 1.0);
}
"""

MAX_TREASURES = 12
SAMPLE_R = 0.02  # [m] okolí, ze kterého se bere výška povrchu nad pokladem
NEAR_RANGE = 0.015  # [m] od této zbývající vrstvy se ukazuje třpyt


def star_tris(cx: float, cy: float, r_out: float, r_in: float, n: int = 5, rot: float = -math.pi / 2) -> np.ndarray:
    """Hvězda s n cípy jako trojúhelníky (vějíř ze středu)."""
    k = np.arange(2 * n)
    ang = rot + k * math.pi / n
    rad = np.where(k % 2 == 0, r_out, r_in)
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1)
    c = np.tile([cx, cy], (2 * n, 1))
    return np.stack([c, pts, np.roll(pts, -1, 0)], 1).reshape(-1, 2)


def ray_fan(cx: float, cy: float, r0: float, r1: float, n: int, half_width: float, rot: float,
            color) -> tuple[np.ndarray, np.ndarray]:
    """Paprsky světla ze středu (n klínů, ke konci průhledné) -> (vrcholy, barvy) pro Painter.triangles."""
    a = rot + np.arange(n) * 2 * math.pi / n
    base = np.stack([cx + r0 * np.cos(a), cy + r0 * np.sin(a)], 1)
    left = np.stack([cx + r1 * np.cos(a - half_width), cy + r1 * np.sin(a - half_width)], 1)
    right = np.stack([cx + r1 * np.cos(a + half_width), cy + r1 * np.sin(a + half_width)], 1)
    verts = np.stack([base, left, right], 1).reshape(-1, 2)
    col = np.asarray(color, np.float32)
    tip = col.copy()
    tip[3] = 0.0
    return verts, np.tile(np.stack([col, tip, tip]), (n, 1))


@dataclass
class Treasure:
    x: float  # [m]
    y: float
    z: float  # [m] absolutní výška pokladu (nad základní rovinou)
    kind: int = 0  # 0 mince, 1 drahokam, 2 truhla
    found: bool = False
    found_t: float = 0.0
    hold: float = 0.0  # [s] jak dlouho je povrch dost nízko (proti šumu)


def surface_level(heights: np.ndarray, size: tuple[float, float], x: float, y: float,
                  radius: float = SAMPLE_R) -> float:
    """Výška povrchu nad bodem: 25. percentil v okolí (díra vykopaná prstem stačí, šum ne)."""
    gh, gw = heights.shape
    cx, cy = x / size[0] * gw, y / size[1] * gh
    rx, ry = max(1, int(round(radius / size[0] * gw))), max(1, int(round(radius / size[1] * gh)))
    ix, iy = int(np.clip(cx, 0, gw - 1)), int(np.clip(cy, 0, gh - 1))
    patch = heights[max(0, iy - ry): iy + ry + 1, max(0, ix - rx): ix + rx + 1]
    return float(np.percentile(patch, 25))


def place_treasures(heights: np.ndarray, size: tuple[float, float], n: int, rng: np.random.Generator,
                    cover: tuple[float, float] = (0.015, 0.03), margin: float = 0.1, min_dist: float = 0.18,
                    floor: float = -0.07, forbid: np.ndarray | None = None,
                    bottom: float | None = None) -> list[Treasure]:
    """Rozmístí poklady na vyvýšená místa tak, aby se k nim muselo kopat ``cover`` metrů písku.

    ``bottom`` = nejnižší místo, kam se v bedně zatím dalo vykopat (dno) – poklad nikdy není níž,
    jinak by ho nešlo vykopat."""
    gh, gw = heights.shape
    W, D = size
    xs = (np.arange(gw) + 0.5) / gw * W
    ys = (np.arange(gh) + 0.5) / gh * D
    inner = ((xs[None, :] > margin) & (xs[None, :] < W - margin)) & ((ys[:, None] > margin) & (ys[:, None] < D - margin))
    ok = inner & (heights >= floor + cover[1])
    if forbid is not None:
        ok &= ~forbid
    if ok.sum() > 0:
        high = ok & (heights >= np.percentile(heights[ok], 40))  # spíš na kopcích – je co kopat
        if high.sum() >= 4 * n:
            ok = high
    cand = np.flatnonzero(ok if ok.any() else inner)
    rng.shuffle(cand)
    pts: list[tuple[float, float]] = []
    dist = min_dist
    while len(pts) < n and dist > 0.01:
        for c in cand:
            iy, ix = divmod(int(c), gw)
            x, y = float(xs[ix]), float(ys[iy])
            if all(math.hypot(x - px, y - py) >= dist for px, py in pts):
                pts.append((x, y))
                if len(pts) == n:
                    break
        dist *= 0.7
    out = []
    for i, (x, y) in enumerate(pts):
        z = surface_level(heights, size, x, y) - float(rng.uniform(*cover))
        if bottom is not None:
            z = max(z, bottom + 0.008)
        out.append(Treasure(x, y, z, kind=i % 3))
    return out


def update_reveals(treasures: list[Treasure], heights: np.ndarray, size: tuple[float, float], dt: float,
                   now: float, eps: float = 0.004, hold_time: float = 0.3,
                   bottom: float | None = None) -> list[Treasure]:
    """Odkryje poklady, nad kterými je povrch aspoň hold_time s níž než z + eps. Vrací nově nalezené.

    Kdo vykope až na dno bedny (``bottom`` = nejnižší místo, které kdy bylo vidět), poklad najde
    vždycky – i kdyby byl kvůli tenké vrstvě písku zakopaný „pod dnem“."""
    new = []
    for tr in treasures:
        if tr.found:
            continue
        target = tr.z if bottom is None else max(tr.z, bottom + 0.006)
        if surface_level(heights, size, tr.x, tr.y) <= target + eps:
            tr.hold += dt
            if tr.hold >= hold_time:
                tr.found, tr.found_t = True, now
                new.append(tr)
        else:
            tr.hold = 0.0
    return new


def hint_for(distance: float) -> tuple[str, tuple[float, float, float]]:
    """Slovní nápověda a barva podle vzdálenosti k pokladu [m]."""
    if distance < 0.10:
        return "Hoří!", (1.0, 0.15, 0.1)
    if distance < 0.22:
        return "Teplo", (1.0, 0.55, 0.1)
    if distance < 0.40:
        return "Přihořívá", (1.0, 0.95, 0.2)
    return "Zima", (0.35, 0.7, 1.0)


class TreasureGame(Game):
    id = "poklad"
    title = "Hledání pokladu"
    description = "Pod pískem jsou zakopané poklady. Kopej – ruka ti napoví, jestli jsi blízko."
    order = 75
    category = "soutěž"
    intro = "Pod pískem jsou zakopané poklady – hrabej a najdi je všechny co nejrychleji, ruka ti napoví, kde hledat."
    keys = [
        ("ruka", "nápověda zima / hoří"),
        ("N / Backspace", "nové kolo"),
        ("↑ / ↓", "počet pokladů"),
        ("H", "nápověda u ruky zap/vyp"),
        ("O", "odhalit poklady (operátor)"),
    ]

    VICTORY_TIME = 10.0  # [s] vítězná obrazovka, pak nové kolo

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.rng = np.random.default_rng()
        self.count = getattr(self, "count", 5)
        self.hints = getattr(self, "hints", True)
        self.sparks = Sparks()
        self.start_tex = None
        self.t = 0.0
        self.show_all = False
        self._new_round()

    def stop(self) -> None:
        self.prog.release()
        if self.start_tex is not None:
            self.start_tex.release()
            self.start_tex = None

    def reset(self) -> None:
        self._new_round()

    def _new_round(self) -> None:
        self.treasures: list[Treasure] = []
        self.need_place = True
        self.round_t0: float | None = None  # čas běží od první ruky nad pískem
        self.victory_t: float | None = None
        self.final_time = 0.0
        self.hand_hints: list[tuple[float, float, float]] = []
        self.bottom: float | None = getattr(self, "bottom", None)

    def _place(self, hm) -> None:
        self.treasures = place_treasures(hm.heights, hm.size, self.count, self.rng, forbid=hm.hands,
                                         bottom=self.bottom)
        if self.start_tex is not None:
            self.start_tex.release()
        self.start_tex = self.host.float_texture((hm.shape[1], hm.shape[0]), hm.heights)
        self.need_place = False

    @property
    def found(self) -> int:
        return sum(t.found for t in self.treasures)

    def update(self, hm, dt: float) -> None:
        dt = min(dt, 1 / 15)
        self.t += dt
        self.sparks.update(dt)
        if self.need_place:
            self._place(hm)
        if self.victory_t is not None:
            if self.t - self.victory_t > self.VICTORY_TIME:
                self._new_round()
            elif int((self.t - dt) / 0.35) != int(self.t / 0.35):  # ohňostroj
                W, D = hm.size
                cols = [(1, 0.85, 0.2), (1, 0.3, 0.5), (0.3, 1, 1), (0.6, 1, 0.3), (1, 1, 1)]
                fx = float(self.rng.uniform(0.15, 0.85))
                self.sparks.burst(fx * W, self.rng.uniform(0.2, 0.8) * D, 60, 0.7, cols,
                                  life=1.6, rng=self.rng)
                if self.t - self.victory_t < 4.0:  # ohňostroj slyšet jen na začátku oslavy
                    self.sfx.play("pop", 0.35, x=fx, pitch=float(self.rng.uniform(0.9, 1.15)))
            return
        if self.round_t0 is None and hm.hand_points:
            self.round_t0 = self.t
        # dno bedny: nejnižší místo, kam se kdy vykopalo (poklad pod ním by nešel najít)
        gh, gw = hm.shape
        low = float(np.percentile(hm.heights[gh // 10: -gh // 10 or None, gw // 10: -gw // 10 or None], 0.5))
        self.bottom = low if self.bottom is None else min(self.bottom, low)
        for tr in update_reveals(self.treasures, hm.heights, hm.size, dt, self.t, bottom=self.bottom):
            if self.round_t0 is None:
                self.round_t0 = self.t
            cols = [(1, 0.85, 0.15), (1, 1, 0.6), (1, 1, 1)] if tr.kind != 1 else [(0.3, 1, 1), (1, 0.3, 0.9), (1, 1, 1)]
            self.sparks.burst(tr.x, tr.y, 70, 0.6, cols, life=1.4, rng=self.rng)
            # zvuk nálezu: mince cinkne; drahokam zvoní výš a třpytí se, truhla je hlubší a třpytí se
            x = tr.x / hm.size[0]
            self.sfx.play("coin", 1.0, x=x, pitch=(1.0, 1.15, 0.85)[tr.kind % 3])
            if tr.kind in (1, 2):
                self.sfx.play("sparkle", 0.7, x=x)
        # jiskření nalezených
        for tr in self.treasures:
            if tr.found and self.rng.random() < dt * 1.5:
                self.sparks.burst(tr.x, tr.y, 6, 0.15, [(1, 1, 0.7), (1, 1, 1)], life=0.8, rng=self.rng)
        if self.treasures and self.found == len(self.treasures):
            self.victory_t = self.t
            self.sfx.play("fanfare", 1.0)
            self.final_time = self.t - (self.round_t0 if self.round_t0 is not None else self.t)
        # nápověda u rukou
        hidden = [(tr.x, tr.y) for tr in self.treasures if not tr.found]
        self.hand_hints = []
        if self.hints and hidden:
            hp = np.asarray(hidden)
            for hx, hy in hm.hand_points:
                d = float(np.min(np.hypot(hp[:, 0] - hx, hp[:, 1] - hy)))
                self.hand_hints.append((hx, hy, d))

    # --- kreslení ---------------------------------------------------------------------
    def _uniform_treasures(self) -> bytes:
        hm = self.host.heightmap
        arr = np.zeros((MAX_TREASURES, 4), np.float32)
        W, D = self.host.box_size
        for i, tr in enumerate(self.treasures[:MAX_TREASURES]):
            near = 0.0
            if not tr.found and hm is not None:
                remaining = surface_level(hm.heights, hm.size, tr.x, tr.y) - tr.z
                near = float(np.clip(1 - remaining / NEAR_RANGE, 0, 1))
            arr[i] = (tr.x / W, tr.y / D, near, 1.0 if (tr.found or self.show_all) else 0.0)
        return arr.tobytes()

    def render(self) -> None:
        if self.start_tex is None:
            return
        try:
            self.prog["u_tr"].write(self._uniform_treasures())  # type: ignore[union-attr]
        except KeyError:
            pass
        vk = min(1.0, (self.t - self.victory_t) * 1.5) if self.victory_t is not None else 0.0
        self.host.draw_fullscreen(self.prog, u_start=(self.start_tex, 1),
                                  u_ntr=min(len(self.treasures), MAX_TREASURES), u_victory=float(vk))
        host, p = self.host, self.host.painter
        cw, ch = host.canvas_size
        ppm = cw / host.box_size[0]
        fs = max(18, ch // 28)
        labels = []  # texty až po záři (zůstanou ostré)
        for i, tr in enumerate(self.treasures):
            if not (tr.found or self.show_all):
                continue
            x, y = host.box_to_canvas(tr.x, tr.y)
            since = self.t - tr.found_t if tr.found else 99.0
            pop = min(1.0, since * 3)
            scale = 0.035 * ppm * (1 + 0.6 * (1 - pop) + 0.08 * math.sin(self.t * 3 + tr.x * 9))
            if since < 2.5:  # odhalení: paprsky světla a rozbíhající se kruh
                k = since / 2.5
                p.triangles(*ray_fan(x, y, scale * 1.2, scale * (3.5 + 3 * min(1.0, since * 2)), 12, 0.12,
                                     self.t * 0.8, (1, 0.85, 0.3, 0.9 * (1 - k))))
                p.circle(x, y, scale * (1.2 + 5 * k), (1, 0.9, 0.3, 1 - k), width=max(5.0, scale * 0.25))
                labels.append((x, y - scale * 1.6 - 20 * k, 1 - k * 0.5))
            open_k = min(1.0, max(0.0, (since - 0.25) * 2.5)) if tr.found else 1.0
            self._icon(tr.kind, x, y, scale, alpha=1.0 if tr.found else 0.5, open_k=open_k)
            if tr.found:
                self._twinkle(x, y, scale, i)
        for hx, hy, d in self.hand_hints:
            labels.append(self._hint(hx, hy, d))
        self.sparks.draw(host, size_px=ch / 60)
        if self.victory_t is not None:
            self._victory_fx(cw, ch, vk)
        # bez host.glow: světlý písek by se celý rozzářil – záře kreslí měkké kruhy a shader
        for item in labels:
            if len(item) == 3:
                x, y, a = item
                p.text("POKLAD!", x, y, size=int(fs * 1.3), anchor="bc", bold=True, color=(1, 0.9, 0.2, 1),
                       outline=4, outline_color=(0.35, 0.1, 0.0, 1), alpha=a)
            else:
                word, col, x, y = item
                p.text(word, x, y, size=fs, anchor="bc", bold=True, color=(*col, 1), outline=4)
        self._hud(fs)

    def _icon(self, kind: int, x: float, y: float, r: float, alpha: float = 1.0, open_k: float = 1.0) -> None:
        host, p = self.host, self.host.painter
        a = alpha
        soft_discs(host, [(x, y)], r * 2.4, (1, 0.8, 0.25, 0.45 * a))  # zlatá aura
        soft_discs(host, [(x + r * 0.25, y + r * 0.35)], r * 1.4, (0.15, 0.03, 0.0, 0.5 * a))  # stín
        if kind == 1:  # drahokam (briliant shora): fazety + tabulka
            n = 8
            ang = np.arange(n + 1) * 2 * math.pi / n + math.pi / 8
            outer = np.stack([x + r * 1.15 * np.cos(ang), y + r * 1.15 * np.sin(ang)], 1)
            inner = np.stack([x + r * 0.55 * np.cos(ang), y + r * 0.55 * np.sin(ang)], 1)
            p.circle(x, y, r * 1.3, (0.0, 0.12, 0.3, a), segments=8)
            quads = np.stack([outer[:-1], outer[1:], inner[1:], outer[:-1], inner[1:], inner[:-1]], 1)
            shade = 0.55 + 0.45 * np.cos(ang[:-1] - 3.9 + self.t * 0.6)  # světlo po fazetách obíhá
            cols = np.stack([0.15 + 0.5 * shade, 0.55 + 0.45 * shade, np.ones(n), np.full(n, a)], 1)
            p.triangles(quads.reshape(-1, 2), np.repeat(cols, 6, 0))
            c = np.tile([x, y], (n, 1))
            p.triangles(np.stack([c, inner[:-1], inner[1:]], 1).reshape(-1, 2), (0.75, 1.0, 1.0, a))
            p.triangles(star_tris(x - r * 0.25, y - r * 0.25, r * 0.55, r * 0.1, 4), (1, 1, 1, 0.95 * a))
        elif kind == 2:  # truhla, víko se otevře
            w, bh = r * 2.6, r * 1.3
            top = y - bh * 0.25
            if open_k > 0:  # zlato uvnitř
                soft_discs(host, [(x, top)], r * 2.2 * open_k, (1, 0.85, 0.3, 0.7 * a * open_k))
            p.rect(x - w / 2 - 3, top - 3, w + 6, bh + 6, (0.2, 0.07, 0.0, a))
            p.rect(x - w / 2, top, w, bh, (0.58, 0.29, 0.08, a))
            p.rect(x - w / 2, top + bh * 0.45, w, bh * 0.14, (1, 0.78, 0.15, a))  # kování
            p.rect(x - w * 0.36, top, w * 0.1, bh, (0.85, 0.6, 0.12, a))
            p.rect(x + w * 0.26, top, w * 0.1, bh, (0.85, 0.6, 0.12, a))
            lid_h = r * 0.75 * (1 - 0.45 * open_k)
            lid_y = top - lid_h - r * 0.9 * open_k
            if open_k > 0:
                p.rect(x - w / 2, top - r * 0.35 * open_k, w, r * 0.35 * open_k + 2, (0.18, 0.06, 0.0, a))
                cx = x + np.array([-0.6, 0.0, 0.6, -0.3, 0.3]) * r
                cy = top - r * open_k * np.array([0.15, 0.3, 0.15, 0.45, 0.45])
                cent = np.stack([cx, cy], 1)
                soft_discs(host, cent, r * 0.42, (1, 0.85, 0.2, a), segments=14, edge=1.0)
                soft_discs(host, cent - r * 0.08, r * 0.2, (1, 1, 0.7, a), segments=10, edge=1.0)
            p.rect(x - w / 2 - 3, lid_y - 3, w + 6, lid_h + 6, (0.2, 0.07, 0.0, a))
            p.rect(x - w / 2, lid_y, w, lid_h, (0.78, 0.42, 0.12, a))
            p.rect(x - w / 2, lid_y + lid_h * 0.7, w, lid_h * 0.3, (1, 0.8, 0.2, a))
            p.rect(x - r * 0.22, top - r * 0.05, r * 0.44, r * 0.55, (1, 0.88, 0.25, a))  # zámek
        else:  # zlatá mince s hvězdou
            p.circle(x, y, r * 1.12, (0.45, 0.25, 0.0, a), segments=28)
            p.circle(x, y, r, (1.0, 0.72, 0.08, a), segments=28)
            p.circle(x, y, r * 0.78, (1.0, 0.86, 0.28, a), segments=28)
            p.triangles(star_tris(x, y, r * 0.62, r * 0.26, 5, -math.pi / 2 + 0.1 * math.sin(self.t)),
                        (1.0, 0.97, 0.65, a))
            p.circle(x - r * 0.42, y - r * 0.42, r * 0.17, (1, 1, 1, 0.9 * a), segments=12)

    def _twinkle(self, x: float, y: float, r: float, i: int) -> None:
        """Třpytky: čtyřcípé hvězdičky, které kolem pokladu vyskakují a mizí."""
        tris = []
        for j in range(3):
            ph = self.t * 0.8 + j / 3 + i * 0.37
            cyc, f = int(ph), ph % 1.0
            a = (cyc * 2.39996 + j * 2.1 + i) % (2 * math.pi)
            d = r * (1.0 + 0.6 * ((cyc * 7 + j * 3) % 5) / 4)
            sz = r * 0.9 * math.sin(f * math.pi)
            if sz > 1:
                tris.append(star_tris(x + d * math.cos(a), y + d * math.sin(a), sz, sz * 0.18, 4, self.t))
        if tris:
            self.host.painter.triangles(np.concatenate(tris), (1, 1, 0.85, 1))

    def _hint(self, hx: float, hy: float, d: float):
        host, p = self.host, self.host.painter
        ppm = host.canvas_size[0] / host.box_size[0]
        word, col = hint_for(d)
        x, y = host.box_to_canvas(hx, hy)
        heat = max(0.0, 1 - d / 0.4)
        freq = 2.0 + 8.0 * heat  # čím blíž, tím rychleji pulzuje
        pulse = 0.5 + 0.5 * math.sin(self.t * freq)
        r = max(0.05, min(d, 0.4)) * ppm
        soft_discs(host, [(x, y)], r * 1.05, (*col, 0.10 + 0.12 * pulse), segments=40, edge=0.6)
        p.circle(x, y, r, (*col, 0.85 + 0.15 * pulse), width=6 + 4 * pulse)
        # točící se „kompas“: čárky na vnějším kruhu, blízko pokladu se točí rychle
        rot = self.t * (0.4 + 3.0 * heat)
        quads = [ring_quads(x, y, r + 12, 7, rot + k * math.pi / 6, rot + k * math.pi / 6 + 0.25, 4)
                 for k in range(12)]
        p.triangles(np.concatenate(quads), (*col, 0.9))
        if heat > 0.45:  # horko: vlny se sbíhají k ruce
            for k in range(2):
                f = 1 - ((self.t * (0.8 + heat) + k / 2) % 1.0)
                p.circle(x, y, r * f, (*col, 0.8 * (1 - f)), width=5)
        return word, col, x, y - r - 14

    def _victory_fx(self, cw: int, ch: int, k: float) -> None:
        """Vítězství: obří zlaté paprsky za otevřenou truhlou uprostřed."""
        p = self.host.painter
        cx, cy = cw / 2, ch * 0.5
        p.triangles(*ray_fan(cx, cy, ch * 0.05, cw * 0.6, 16, 0.09, self.t * 0.3, (1, 0.8, 0.25, 0.45 * k)))
        p.rect(0, ch * 0.28, cw, ch * 0.44, (0.12, 0.03, 0.12, 0.72 * k))
        p.rect(0, ch * 0.28, cw, 6, (1, 0.8, 0.2, k))
        p.rect(0, ch * 0.72 - 6, cw, 6, (1, 0.8, 0.2, k))
        r = ch * 0.045
        for sx in (cw * 0.12, cw * 0.88):
            soft_discs(self.host, [(sx, ch * 0.5)], r * 3.5, (1, 0.8, 0.2, 0.6 * k))
            self._icon(2, sx, ch * 0.52, r, alpha=k, open_k=1.0)

    def _hud(self, fs: int) -> None:
        host, p = self.host, self.host.painter
        cw, ch = host.canvas_size
        big = max(fs, ch // 22)
        w, hh = hud.label(p, f"Poklady: {self.found} / {len(self.treasures)}", big * 0.5, big * 0.45, big,
                          color=(1, 0.88, 0.2, 1), anchor="tl")
        r = big * 0.36
        y = big * 0.45 + hh / 2
        for i, tr in enumerate(self.treasures):
            x = big * 0.5 + w + big * 0.9 + i * r * 3.0
            if tr.found:
                self._icon(tr.kind, x, y, r)
            else:
                p.circle(x, y, r * 1.1, (0.1, 0.03, 0.0, 0.85))
                p.circle(x, y, r * 1.1, (1, 0.8, 0.3, 0.8), width=3)
        elapsed = (self.final_time if self.victory_t is not None
                   else (self.t - self.round_t0 if self.round_t0 is not None else 0.0))
        hud.label(p, f"Čas: {fmt_time(elapsed)}", cw - big * 0.5, big * 0.45, big, anchor="tr")
        if self.victory_t is not None:
            k = min(1.0, (self.t - self.victory_t) * 2)
            pulse = 1 + 0.05 * math.sin(self.t * 6)
            p.text("Všechny poklady nalezeny!", cw / 2, ch * 0.42, size=int(ch / 12 * pulse), anchor="mc", bold=True,
                   color=(1, 0.88, 0.15, 1), outline=6, outline_color=(0.35, 0.08, 0.0, 1), alpha=k)
            p.text(f"Čas: {fmt_time(self.final_time)}", cw / 2, ch * 0.55, size=int(ch / 15), anchor="mc", bold=True,
                   color=(1, 1, 1, 1), outline=5, alpha=k)
            left = max(0, math.ceil(self.VICTORY_TIME - (self.t - self.victory_t)))
            p.text(f"Nové poklady se zakopou za {left} s", cw / 2, ch * 0.66, size=fs, anchor="mc",
                   color=(0.85, 0.9, 1, 1), outline=3, alpha=k)
        elif self.round_t0 is None:
            hud.label(p, "Kopej v písku a najdi poklady!", cw / 2, ch - big * 0.5, big, anchor="bc")

    # --- ovládání -------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name in ("n", "backspace"):
            self._new_round()
            self.host.notify("Nové kolo – poklady zakopány")
        elif name in ("up", "down"):
            self.count = int(np.clip(self.count + (1 if name == "up" else -1), 1, 9))
            self._new_round()
            self.host.notify(f"Počet pokladů: {self.count}")
        elif name == "h":
            self.hints = not self.hints
            self.host.notify("Nápověda " + ("zapnuta" if self.hints else "vypnuta"))
        elif name == "o":
            self.show_all = not self.show_all
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Nalezeno {self.found}/{len(self.treasures)}",
                 "Nápověda: " + ("ano" if self.hints else "ne")]
        if self.victory_t is not None:
            lines.append(f"Vítězství za {self.final_time:.0f} s")
        elif self.round_t0 is not None:
            lines.append(f"Čas kola {self.t - self.round_t0:.0f} s")
        hm = self.host.heightmap
        for i, tr in enumerate(self.treasures):
            if hm is not None and not tr.found:
                rest = surface_level(hm.heights, hm.size, tr.x, tr.y) - tr.z
                lines.append(f"#{i + 1} ({tr.x:.2f}, {tr.y:.2f}) zbývá {rest * 100:.1f} cm")
        return lines
