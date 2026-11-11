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

from .base import Game
from .kulicka import Sparks, fmt_time

FRAG = """
uniform sampler2D u_start;    // výšky na začátku kola
uniform vec4 u_tr[12];        // poklady: x, y (uv), blízko 0..1, nalezen 0/1
uniform int u_ntr;

void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    float t = clamp((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4), 0.0, 1.0);
    vec3 c = mix(vec3(0.95, 0.60, 0.18), vec3(1.0, 0.96, 0.62), smoothstep(0.15, 0.85, t));
    c *= mix(0.45, 1.25, hillshade(uv, 3.5));
    // vykopaná místa tmavnou do hnědočervené (vlhký písek)
    float dug = texture(u_start, uv).r - h;
    c = mix(c, vec3(0.38, 0.10, 0.22), smoothstep(0.004, 0.04, dug) * 0.9);
    // nasypaná místa světlejší
    c = mix(c, vec3(1.0, 0.97, 0.80), smoothstep(0.006, 0.04, -dug) * 0.5);
    vec2 cl = contours(h);
    c = mix(c, c * 0.55, cl.x * 0.5);
    c = mix(c, vec3(0.25, 0.10, 0.02), cl.y * 0.7);
    for (int i = 0; i < u_ntr; ++i) {
        vec4 tr = u_tr[i];
        float r = length((uv - tr.xy) * u_box);
        if (tr.w > 0.5) {
            c += vec3(1.0, 0.75, 0.15) * 0.45 * exp(-r * r / (0.035 * 0.035));  // záře nalezeného
        } else if (tr.z > 0.0) {
            // „už to skoro je“ – třpyt v díře, kde zbývá jen trocha písku
            float tw = 0.5 + 0.5 * sin(u_time * 7.0 + r * 160.0);
            c += vec3(1.0, 0.85, 0.3) * tr.z * 0.7 * tw * exp(-r * r / (0.045 * 0.045));
        }
    }
    f_color = vec4(c, 1.0);
}
"""

MAX_TREASURES = 12
SAMPLE_R = 0.02  # [m] okolí, ze kterého se bere výška povrchu nad pokladem
NEAR_RANGE = 0.015  # [m] od této zbývající vrstvy se ukazuje třpyt


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
                    cover: tuple[float, float] = (0.025, 0.04), margin: float = 0.1, min_dist: float = 0.18,
                    floor: float = -0.07, forbid: np.ndarray | None = None) -> list[Treasure]:
    """Rozmístí poklady na vyvýšená místa tak, aby se k nim muselo kopat ``cover`` metrů písku."""
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
        out.append(Treasure(x, y, z, kind=i % 3))
    return out


def update_reveals(treasures: list[Treasure], heights: np.ndarray, size: tuple[float, float], dt: float,
                   now: float, eps: float = 0.004, hold_time: float = 0.3) -> list[Treasure]:
    """Odkryje poklady, nad kterými je povrch aspoň hold_time s níž než z + eps. Vrací nově nalezené."""
    new = []
    for tr in treasures:
        if tr.found:
            continue
        if surface_level(heights, size, tr.x, tr.y) <= tr.z + eps:
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
    order = 65
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

    def _place(self, hm) -> None:
        self.treasures = place_treasures(hm.heights, hm.size, self.count, self.rng, forbid=hm.hands)
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
                self.sparks.burst(self.rng.uniform(0.15, 0.85) * W, self.rng.uniform(0.2, 0.8) * D, 60, 0.7, cols,
                                  life=1.6, rng=self.rng)
            return
        if self.round_t0 is None and hm.hand_points:
            self.round_t0 = self.t
        for tr in update_reveals(self.treasures, hm.heights, hm.size, dt, self.t):
            if self.round_t0 is None:
                self.round_t0 = self.t
            cols = [(1, 0.85, 0.15), (1, 1, 0.6), (1, 1, 1)] if tr.kind != 1 else [(0.3, 1, 1), (1, 0.3, 0.9), (1, 1, 1)]
            self.sparks.burst(tr.x, tr.y, 70, 0.6, cols, life=1.4, rng=self.rng)
        # jiskření nalezených
        for tr in self.treasures:
            if tr.found and self.rng.random() < dt * 1.5:
                self.sparks.burst(tr.x, tr.y, 6, 0.15, [(1, 1, 0.7), (1, 1, 1)], life=0.8, rng=self.rng)
        if self.treasures and self.found == len(self.treasures):
            self.victory_t = self.t
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
        self.host.draw_fullscreen(self.prog, u_start=(self.start_tex, 1),
                                  u_ntr=min(len(self.treasures), MAX_TREASURES))
        host = self.host
        cw, ch = host.canvas_size
        ppm = cw / host.box_size[0]
        fs = max(18, ch // 28)
        for tr in self.treasures:
            if tr.found or self.show_all:
                x, y = host.box_to_canvas(tr.x, tr.y)
                pop = min(1.0, (self.t - tr.found_t) * 3) if tr.found else 1.0
                scale = 0.035 * ppm * (1 + 0.6 * (1 - pop) + 0.08 * math.sin(self.t * 3 + tr.x * 9))
                self._icon(tr.kind, x, y, scale, alpha=1.0 if tr.found else 0.5)
                if tr.found and self.t - tr.found_t < 2.5:
                    p = host.painter
                    k = (self.t - tr.found_t) / 2.5
                    p.circle(x, y, scale * (1.2 + 4 * k), (1, 0.9, 0.3, 1 - k), width=6)
                    p.text("POKLAD!", x, y - scale * 1.4 - 20 * k, size=int(fs * 1.3), anchor="bc", bold=True,
                           color=(1, 0.9, 0.2, 1 - k * 0.5), outline=4)
        for hx, hy, d in self.hand_hints:
            self._hint(hx, hy, d, fs)
        self.sparks.draw(host, size_px=ch / 75)
        self._hud(fs)

    def _icon(self, kind: int, x: float, y: float, r: float, alpha: float = 1.0) -> None:
        p = self.host.painter
        p.circle(x, y, r * 1.9, (1, 0.85, 0.3, 0.25 * alpha))
        if kind == 1:  # drahokam
            pts = np.array([(x, y - r * 1.2), (x + r, y), (x, y + r * 1.2), (x, y - r * 1.2), (x, y + r * 1.2),
                            (x - r, y)])
            p.triangles(pts, (0.2, 0.95, 1.0, alpha))
            p.triangles(np.array([(x, y - r * 1.2), (x + r * 0.45, y), (x - r * 0.45, y)]), (0.85, 1, 1, alpha))
            p.polyline(pts[[0, 1, 2, 5]], (0, 0.25, 0.5, alpha), width=3, closed=True)
        elif kind == 2:  # truhla
            p.rect(x - r * 1.2, y - r * 0.5, r * 2.4, r * 1.3, (0.55, 0.28, 0.08, alpha))
            p.rect(x - r * 1.2, y - r * 1.0, r * 2.4, r * 0.55, (0.75, 0.40, 0.12, alpha))
            p.rect(x - r * 1.2, y - r * 0.55, r * 2.4, r * 0.15, (1, 0.8, 0.2, alpha))
            p.rect(x - r * 0.2, y - r * 0.6, r * 0.4, r * 0.5, (1, 0.85, 0.2, alpha))
            p.rect_outline(x - r * 1.2, y - r * 1.0, r * 2.4, r * 1.8, (0.25, 0.1, 0.0, alpha), width=3)
        else:  # mince
            p.circle(x, y, r, (1.0, 0.78, 0.1, alpha))
            p.circle(x, y, r * 0.72, (1.0, 0.92, 0.35, alpha))
            p.circle(x, y, r * 0.72, (0.75, 0.5, 0.0, alpha), width=3)
            p.circle(x, y, r, (0.6, 0.38, 0.0, alpha), width=3)
            p.circle(x - r * 0.3, y - r * 0.3, r * 0.18, (1, 1, 1, 0.9 * alpha))

    def _hint(self, hx: float, hy: float, d: float, fs: int) -> None:
        host, p = self.host, self.host.painter
        ppm = host.canvas_size[0] / host.box_size[0]
        word, col = hint_for(d)
        x, y = host.box_to_canvas(hx, hy)
        freq = 2.0 + 8.0 * max(0.0, 1 - d / 0.4)  # čím blíž, tím rychleji pulzuje
        pulse = 0.5 + 0.5 * math.sin(self.t * freq)
        r = max(0.05, min(d, 0.4)) * ppm
        p.circle(x, y, r, (*col, 0.8 + 0.2 * pulse), width=6 + 4 * pulse)
        p.circle(x, y, r, (*col, 0.10 * pulse))
        p.text(word, x, y - r - 6, size=fs, anchor="bc", bold=True, color=(*col, 1), outline=4)

    def _hud(self, fs: int) -> None:
        host, p = self.host, self.host.painter
        cw, ch = host.canvas_size
        big = max(fs, ch // 22)
        w, _ = p.text(f"Poklady: {self.found} / {len(self.treasures)}", big * 0.6, big * 0.4, size=big, bold=True,
                      color=(1, 0.9, 0.2, 1), outline=4)
        r = big * 0.32
        for i, tr in enumerate(self.treasures):
            x = big * 0.6 + w + big * 0.6 + i * r * 2.8
            if tr.found:
                self._icon(tr.kind, x, big * 0.95, r)
            else:
                p.circle(x, big * 0.95, r, (0.2, 0.1, 0.05, 0.8), width=3)
        elapsed = (self.final_time if self.victory_t is not None
                   else (self.t - self.round_t0 if self.round_t0 is not None else 0.0))
        p.text(f"Čas: {fmt_time(elapsed)}", cw - big * 0.6, big * 0.4, size=big, anchor="tr", bold=True,
               color=(1, 1, 1, 1), outline=4)
        if self.victory_t is not None:
            k = min(1.0, (self.t - self.victory_t) * 2)
            p.rect(0, ch * 0.28, cw, ch * 0.44, (0.05, 0.02, 0.15, 0.7 * k))
            p.text("Všechny poklady nalezeny!", cw / 2, ch * 0.42, size=int(ch / 12), anchor="mc", bold=True,
                   color=(1, 0.88, 0.15, k), outline=6)
            p.text(f"Čas: {fmt_time(self.final_time)}", cw / 2, ch * 0.55, size=int(ch / 15), anchor="mc", bold=True,
                   color=(1, 1, 1, k), outline=5)
            left = max(0, math.ceil(self.VICTORY_TIME - (self.t - self.victory_t)))
            p.text(f"Nové poklady se zakopou za {left} s", cw / 2, ch * 0.66, size=fs, anchor="mc",
                   color=(0.8, 0.9, 1, k), outline=3)
        elif self.round_t0 is None:
            p.text("Kopej v písku a najdi poklady!", cw / 2, ch - big * 0.4, size=big, anchor="bc", bold=True,
                   color=(1, 1, 1, 1), outline=4)

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
