"""Společné věci pro hry dvou hráčů (Vodní válka, Král kopce, Potopa ostrovů, Modeluj podle předlohy).

Bedna se dělí čárou napříč přes střed delší strany. Hráč 0 stojí u levé kratší strany
plátna (x = 0) a má levou půlku, hráč 1 stojí u pravé kratší strany a má pravou půlku.
Texty pro hráče se kreslí otočené k němu (``rotation`` v ``Painter.text``), aby je
přečetl z místa, kde stojí.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

NAMES = ("Červení", "Modří")
COLORS = ((1.0, 0.32, 0.22, 1.0), (0.30, 0.66, 1.0, 1.0))
DARK = ((0.35, 0.05, 0.02, 1.0), (0.02, 0.12, 0.35, 1.0))
ROTATION = (90.0, -90.0)  # text čitelný od levé / pravé kratší strany


def half_cols(gw: int, player: int) -> slice:
    """Sloupce mřížky, které patří hráči (levá / pravá půlka)."""
    return slice(0, gw // 2) if player == 0 else slice(gw // 2, gw)


def player_of_x(x: float, width: float) -> int:
    return 0 if x < width / 2 else 1


def zone_mask(shape: tuple[int, int], player: int, border: int = 0, neutral: int = 0) -> np.ndarray:
    """Buňky hráčovy půlky bez okraje bedny (``border``) a bez pásu u dělicí čáry (``neutral``)."""
    gh, gw = shape
    m = np.zeros(shape, bool)
    cols = half_cols(gw, player)
    m[:, cols] = True
    if border:
        m[:border, :] = m[gh - border:, :] = False
        m[:, :border] = m[:, gw - border:] = False
    if neutral:
        mid = gw // 2
        m[:, max(0, mid - neutral):mid + neutral] = False
    return m


def zone_peak(heights: np.ndarray, hands: np.ndarray | None, zone: np.ndarray) -> tuple[int, int, float] | None:
    """Nejvyšší místo v zóně mimo ruce: (iy, ix, výška [m]) nebo None.

    Výšky se lehce vyhladí, ať nevyhraje jeden zašuměný pixel.
    """
    h = cv2.GaussianBlur(np.asarray(heights, np.float32), (0, 0), 1.0)
    valid = zone.copy()
    if hands is not None and hands.any():
        valid &= cv2.dilate(hands.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    if not valid.any():
        return None
    k = int(np.argmax(np.where(valid, h, -np.inf)))
    iy, ix = divmod(k, h.shape[1])
    return iy, ix, float(h[iy, ix])


# --- kreslení --------------------------------------------------------------------------------
def anchor(canvas: tuple[int, int], player: int, dist: float, along: float = 0.5) -> tuple[float, float]:
    """Bod na plátně ve vzdálenosti ``dist`` (podíl šířky plátna) od hráčovy strany.

    ``along`` = poloha podél jeho strany zleva doprava z pohledu hráče (0..1).
    """
    cw, ch = canvas
    if player == 0:
        return cw * dist, ch * along
    return cw * (1 - dist), ch * (1 - along)


def player_text(painter, canvas: tuple[int, int], player: int, text: str, dist: float, size: int,
                color: Sequence[float] | None = None, along: float = 0.5, **kw) -> tuple[float, float]:
    """Text otočený k hráči, se středem ``dist`` od jeho strany plátna."""
    x, y = anchor(canvas, player, dist, along)
    kw.setdefault("bold", True)
    kw.setdefault("outline", max(2, size // 12))
    return painter.text(text, x, y, size=size, anchor="mc", color=color or COLORS[player],
                        rotation=ROTATION[player], **kw)


def text_size(canvas: tuple[int, int], scale: float = 1.0) -> int:
    """Velikost písma čitelná ze 2 m (alespoň výška plátna / 30)."""
    return max(16, int(canvas[1] / 22 * scale))


def draw_divider(painter, canvas: tuple[int, int], neutral_px: float = 0.0, frame: bool = True) -> None:
    """Výrazná dělicí čára přes střed delší strany (+ volitelně neutrální pás) a barevné rámečky půlek."""
    cw, ch = canvas
    mid = cw / 2
    if neutral_px > 0:
        painter.rect(mid - neutral_px, 0, 2 * neutral_px, ch, (0, 0, 0, 0.35))
        step = ch / 24
        for k in range(25):  # šikmé šrafy neutrálního pásu
            y = k * step
            painter.polyline([(mid - neutral_px, y), (mid + neutral_px, y + step * 0.8)], (1, 1, 1, 0.25),
                             width=max(2, ch / 300))
    lw = max(6.0, ch / 90)
    painter.rect(mid - lw, 0, 2 * lw, ch, (0, 0, 0, 0.85))
    painter.rect(mid - lw * 0.45, 0, lw * 0.9, ch, (1, 1, 1, 1))
    if frame:
        fw = max(4.0, ch / 140)
        for player in (0, 1):
            x0 = 0 if player == 0 else mid + lw
            x1 = mid - lw if player == 0 else cw
            painter.rect_outline(x0 + fw / 2, fw / 2, x1 - x0 - fw, ch - fw, COLORS[player], width=fw)


def draw_crown(painter, x: float, y: float, r: float, rotation: float, color=(1.0, 0.85, 0.15, 1.0)) -> None:
    """Korunka (pro vítěze / krále) otočená stejně jako text hráče."""
    pts = np.array([[-1, 0.55], [1, 0.55], [1, -0.45], [0.5, 0.05], [0, -0.7], [-0.5, 0.05], [-1, -0.45]],
                   np.float32) * r
    a = np.radians(rotation)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    pts = pts @ rot.T + np.array([x, y], np.float32)
    tris = [pts[i] for t in ((0, 1, 2), (0, 2, 3), (0, 3, 5), (0, 5, 6), (3, 4, 5)) for i in t]
    outline = (pts - np.array([x, y])) * 1.25 + np.array([x, y])
    painter.triangles(np.array([outline[i] for t in ((0, 1, 2), (0, 2, 3), (0, 3, 5), (0, 5, 6), (3, 4, 5))
                                for i in t]), (0.2, 0.1, 0.0, 1))
    painter.triangles(np.array(tris), color)


# --- aréna: záře půlek, energetická bariéra, rázové vlny, vítězství (GLSL) ---------------------
# Vkládá se do shaderu módu (před main) – mód na konci zavolá ``c = vs_arena(c, uv, tint);``
# (nebo ``vs_arena_k`` s útlumem záře stěn) a uniformy nastaví přes ``Arena.uniforms()``.
ARENA_GLSL = """
uniform float u_vs_neutral;   // polovina neutrálního pásu u dělicí čáry [m] (0 = žádný)
uniform vec2 u_vs_power;      // síla záře půlky červených / modrých (vedoucí svítí víc)
uniform vec2 u_vs_alarm;      // 0..1 nebezpečí / poslední sekundy (pulzující okraj půlky)
uniform float u_vs_winner;    // -1 nic, 0/1 vítěz, 2 remíza
uniform float u_vs_win_t;     // [s] od konce kola (záře vítěze), < 0 = neběží
uniform float u_vs_shock;     // [s] od startu fáze (rázová vlna od bariéry), < 0 = nic
uniform float u_vs_shock_x;   // [m] odkud se vlna šíří (x), standardně střed

const vec3 VS_RED = vec3(1.0, 0.20, 0.10);
const vec3 VS_BLUE = vec3(0.12, 0.50, 1.0);
const vec3 VS_GOLD = vec3(1.0, 0.80, 0.25);

float vs_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vs_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(vs_hash(i), vs_hash(i + vec2(1, 0)), u.x),
               mix(vs_hash(i + vec2(0, 1)), vs_hash(i + vec2(1, 1)), u.x), u.y);
}

vec3 vs_arena_k(vec3 c, vec2 uv, float tint, float rimk) {
    vec2 pm = uv * u_box;
    float W = u_box.x, D = u_box.y, t = u_time;
    float mid = 0.5 * W;
    int side = pm.x < mid ? 0 : 1;
    vec3 team = side == 0 ? VS_RED : VS_BLUE;
    float power = side == 0 ? u_vs_power.x : u_vs_power.y;
    float alarm = side == 0 ? u_vs_alarm.x : u_vs_alarm.y;
    float dx = abs(pm.x - mid);                  // vzdálenost od bariéry [m]
    float own = 1.0 - dx / mid;                  // 1 u bariéry, 0 u hráčovy kratší strany
    float lum = dot(c, vec3(0.3, 0.55, 0.15));

    // 1) půlka v barvě týmu (terén zůstává čitelný; voda zůstane modrá)
    float wet = smoothstep(0.08, 0.3, c.b - max(c.r, c.g));
    vec3 tcol = side == 0 ? vec3(1.0, 0.30, 0.18) * 1.5 : vec3(0.16, 0.42, 1.0) * 1.45;
    // modrá na teplém (žluto-oranžovém) terénu šedne – potřebuje silnější podíl, aby byly půlky vyrovnané
    float tk = side == 0 ? 1.0 : 1.7;
    c = mix(c, tcol * lum, clamp(tint * tk * (0.8 + 0.4 * power), 0.0, 0.85) * (1.0 - wet));

    // 2) záře podél stěn půlky (v nebezpečí pulzuje červeně)
    float e = min(min(pm.x, W - pm.x), min(pm.y, D - pm.y));
    float rim = exp(-e / 0.035);
    float flow = 0.8;
    float ap = alarm * (0.55 + 0.45 * sin(t * 9.0));
    vec3 rimc = mix(team, vec3(1.0, 0.15, 0.05), alarm * 0.6);
    c += rimc * rim * rimk * ((0.25 + 0.3 * power) * flow + 0.7 * ap);

    // 3) neutrální pás: šikmé pruhy
    if (u_vs_neutral > 0.0 && dx < u_vs_neutral) {
        float st = step(0.5, fract((pm.x + pm.y) / 0.045));
        c = c * 0.45 + vec3(0.9, 0.9, 1.0) * st * 0.16;
        c += team * 0.8 * exp(-(u_vs_neutral - dx) / 0.004);
    }

    // 4) dělicí čára: bílé jádro a záře v barvách týmů
    float plasma = vs_noise(vec2(dx * 35.0, pm.y * 7.0));
    float packets = 0.0;
    float core = exp(-pow(dx / 0.0065, 2.0));
    float inner = exp(-dx / (0.014 + 0.012 * plasma));
    float outer = exp(-dx / 0.08);
    c = mix(c, c * 0.55, outer * 0.5);                       // tmavší lem, aby bariéra vynikla
    c += team * (inner * (0.8 + 0.9 * plasma + 1.4 * packets) + outer * (0.25 + 0.35 * power));
    c = mix(c, vec3(1.0, 0.98, 0.95), clamp(core * (0.9 + 0.3 * packets), 0.0, 1.0));


    // 6) vítězství: paprsky od strany vítěze, poražený zešedne
    if (u_vs_win_t >= 0.0) {
        float fade = smoothstep(0.0, 0.8, u_vs_win_t);
        if (u_vs_winner > 1.5) {
            c += VS_GOLD * 0.18 * fade * (0.6 + 0.4 * sin(t * 3.0 + dx * 12.0));
        } else if (float(side) == u_vs_winner) {
            vec2 o = vec2(side == 0 ? -0.05 : W + 0.05, 0.5 * D);
            vec2 q = pm - o;
            float a = atan(q.y, abs(q.x));
            float rays = smoothstep(0.25, 0.9, sin(a * 9.0));
            c += mix(team, VS_GOLD, 0.45) * fade * (0.38 * rays + 0.18) * (0.3 + 0.7 * rimk);
        } else {
            c = mix(c, vec3(lum) * 0.8, 0.45 * fade);
        }
    }
    return c;
}
vec3 vs_arena(vec3 c, vec2 uv, float tint) { return vs_arena_k(c, uv, tint, 1.0); }
"""


class Arena:
    """Stav arénových efektů jednoho módu: rázová vlna, vítěz, konfety, odpočet.

    Mód volá ``phase(name)`` při změně fáze (spustí vlnu), ``win(player|None)`` na konci kola,
    ``update(dt)`` každý snímek a v ``render`` předá ``uniforms()`` do svého shaderu.

    Zvuky arény (stejné ve všech hrách pro dva): start kola ``whistle``, odpočet posledních 5 s
    ``beep`` (v poslední vteřině ``go``) přes ``countdown(left)``, výhra ``fanfare`` ze strany
    vítěze, remíza ``gong``. ``owner`` = mód (jeho ``sfx``); bez něj je aréna tichá.
    """

    def __init__(self, owner=None) -> None:
        self.owner = owner
        self._cd_prev: float | None = None
        self.t = 0.0
        self.shock = -1.0
        self.shock_x: float | None = None
        self.win_t = -1.0
        self.winner: int | None = None
        self.confetti = Confetti()
        self._phase: str | None = None
        self._fountain = 0.0

    def phase(self, name: str, shock: bool = True, x: float | None = None) -> None:
        """Ohlásí fázi; při změně spustí rázovou vlnu (od bariéry, nebo od ``x`` [m])."""
        if name == self._phase:
            return
        self._phase = name
        self._cd_prev = None
        if shock:
            self.shock, self.shock_x = 0.0, x
            self._play("whistle", 0.9)

    def _play(self, name: str, gain: float = 1.0, x: float | None = None, pitch: float = 1.0) -> None:
        sfx = getattr(self.owner, "sfx", None)
        if sfx is not None:
            sfx.play(name, gain=gain, x=x, pitch=pitch)

    def countdown(self, left: float, last: int = 5) -> None:
        """Volat každý snímek se zbývajícím časem [s] do události: ``beep`` při každé celé sekundě
        v posledních ``last`` s, v poslední vteřině ``go``. Každý zvuk zazní jen jednou."""
        prev, self._cd_prev = self._cd_prev, left
        if prev is None or left <= 0:
            return
        a, b = math.ceil(prev), math.ceil(left)
        if a != b and 1 <= b <= last:
            self._play("go" if b == 1 else "beep", 0.8)

    def win(self, player: int | None, canvas: tuple[int, int] | None = None) -> None:
        """Konec kola: ``player`` = vítěz (None = remíza). Vybuchnou konfety v barvě vítěze."""
        self.winner, self.win_t = player, 0.0
        if player is None:
            self._play("gong", 0.9)
        else:
            self._play("fanfare", 1.0, x=0.2 if player == 0 else 0.8)
        if canvas is not None:
            self.burst(canvas, player)

    def clear_win(self) -> None:
        self.winner, self.win_t = None, -1.0

    def burst(self, canvas: tuple[int, int], player: int | None, n: int = 160) -> None:
        cw, ch = canvas
        if player is None:
            for pl in (0, 1):
                x, y = anchor(canvas, pl, 0.22)
                self.confetti.burst(x, y, n // 2, (COLORS[pl], GOLD, (1, 1, 1, 1)), ch * 0.9)
        else:
            x, y = anchor(canvas, player, 0.22)
            self.confetti.burst(x, y, n, (COLORS[player], GOLD, (1, 1, 1, 1)), ch * 1.1)

    def update(self, dt: float, canvas: tuple[int, int] | None = None) -> None:
        self.t += dt
        if self.shock >= 0:
            self.shock = self.shock + dt if self.shock < 3.0 else -1.0
        if self.win_t >= 0:
            self.win_t += dt
            # vítězná fontána: menší dávky konfet ještě pár sekund
            if canvas is not None and self.win_t < 5.0 and self.winner is not None:
                self._fountain += dt
                if self._fountain > 0.35:
                    self._fountain = 0.0
                    x, y = anchor(canvas, self.winner, 0.12, float(np.random.uniform(0.2, 0.8)))
                    self.confetti.burst(x, y, 18, (COLORS[self.winner], GOLD), canvas[1] * 0.6)
        self.confetti.update(dt)

    def uniforms(self, box: tuple[float, float], power=(0.5, 0.5), alarm=(0.0, 0.0),
                 neutral: float = 0.0) -> dict:
        winner = -1.0 if self.win_t < 0 else (2.0 if self.winner is None else float(self.winner))
        sx = box[0] / 2 if self.shock_x is None else self.shock_x
        return dict(u_vs_neutral=float(neutral), u_vs_power=tuple(float(v) for v in power),
                    u_vs_alarm=tuple(float(np.clip(v, 0, 1)) for v in alarm), u_vs_winner=winner,
                    u_vs_win_t=float(self.win_t), u_vs_shock=float(self.shock), u_vs_shock_x=float(sx))

    def draw(self, painter) -> None:
        """Konfety (kreslit před texty)."""
        self.confetti.draw(painter)


GOLD = (1.0, 0.82, 0.2, 1.0)


class Confetti:
    """Konfety pohledem shora: vyletí do stran, zpomalí, třepetají se a dosednou (zmizí).

    Kousky jsou velké (≥ 10 px), aby byly na drti vidět; všechno jedním ``triangles``.
    """

    MAX = 900

    def __init__(self) -> None:
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.ang = np.zeros(0, np.float32)
        self.spin = np.zeros(0, np.float32)
        self.life = np.zeros(0, np.float32)
        self.age = np.zeros(0, np.float32)
        self.size = np.zeros(0, np.float32)
        self.col = np.zeros((0, 4), np.float32)

    def burst(self, x: float, y: float, n: int, colors, speed: float) -> None:
        rng = np.random.default_rng()
        a = rng.uniform(0, 2 * np.pi, n)
        v = speed * rng.uniform(0.25, 1.0, n) ** 0.7
        pal = np.asarray(colors, np.float32)
        self.pos = np.vstack([self.pos, np.tile([x, y], (n, 1)).astype(np.float32)])
        self.vel = np.vstack([self.vel, np.stack([np.cos(a) * v, np.sin(a) * v], 1).astype(np.float32)])
        self.ang = np.concatenate([self.ang, rng.uniform(0, np.pi, n).astype(np.float32)])
        self.spin = np.concatenate([self.spin, rng.uniform(-9, 9, n).astype(np.float32)])
        self.life = np.concatenate([self.life, rng.uniform(1.6, 3.2, n).astype(np.float32)])
        self.age = np.concatenate([self.age, np.zeros(n, np.float32)])
        self.size = np.concatenate([self.size, (speed / 60 * rng.uniform(0.75, 1.25, n)).astype(np.float32)])
        self.col = np.vstack([self.col, pal[rng.integers(0, len(pal), n)]])
        if len(self.age) > self.MAX:
            for k in ("pos", "vel", "ang", "spin", "life", "age", "size", "col"):
                setattr(self, k, getattr(self, k)[-self.MAX:])

    def update(self, dt: float) -> None:
        if not len(self.age):
            return
        self.age += dt
        self.pos += self.vel * dt
        self.vel *= np.float32(np.exp(-2.6 * dt))
        self.ang += self.spin * dt
        keep = self.age < self.life
        if not keep.all():
            for k in ("pos", "vel", "ang", "spin", "life", "age", "size", "col"):
                setattr(self, k, getattr(self, k)[keep])

    def draw(self, painter) -> None:
        n = len(self.age)
        if not n:
            return
        s = np.maximum(self.size, 9.0)
        flutter = 0.35 + 0.65 * np.abs(np.cos(self.ang * 1.7))  # třepetání = proměnlivá šířka
        hw, hh = s * 0.5 * flutter, s * 0.32
        ca, sa = np.cos(self.ang), np.sin(self.ang)
        ux, uy = np.stack([ca * hw, sa * hw], 1), np.stack([-sa * hh, ca * hh], 1)
        p = self.pos
        q = np.stack([p - ux - uy, p + ux - uy, p + ux + uy, p - ux - uy, p + ux + uy, p - ux + uy], 1)
        k = np.clip(self.age / self.life, 0, 1)
        col = self.col.copy()
        col[:, 3] *= np.clip((1 - k) * 3.0, 0, 1)
        col[:, :3] *= (0.75 + 0.25 * flutter)[:, None]  # natočení k projektoru = jas
        painter.triangles(q.reshape(-1, 2), np.repeat(col, 6, 0))


def draw_countdown(painter, canvas: tuple[int, int], player: int, left: float, total: float,
                   last: float = 5.0, dist: float = 0.36) -> None:
    """Odpočet v posledních ``last`` s: velké číslo s prstencem, které s každou sekundou „pulzne“."""
    if left <= 0 or left > last:
        return
    cw, ch = canvas
    x, y = anchor(canvas, player, dist)
    n = int(np.ceil(left))
    f = n - left  # 0 → 1 během sekundy
    pop = 1.0 + 0.35 * np.exp(-f * 6)
    r = ch * 0.085 * pop
    col = COLORS[player]
    hot = (1.0, 0.85, 0.2, 1.0) if n > 2 else (1.0, 0.3, 0.2, 1.0)
    painter.circle(x, y, r * 1.05, (0, 0, 0, 0.55))
    # prstenec ubývá se zbytkem sekundy
    seg = 48
    a0 = -np.pi / 2 + np.radians(ROTATION[player])
    ang = a0 + np.linspace(0, 2 * np.pi * (1 - f), seg + 1)
    w = max(8.0, ch / 70)
    ring = np.stack([x + r * np.cos(ang), y + r * np.sin(ang)], 1)
    painter.polyline(ring, hot, width=w)
    painter.circle(x, y, r * (1.0 + 0.6 * f), col[:3] + (0.8 * (1 - f),), width=w * 0.6)
    size = int(ch * 0.11 * pop)
    painter.text(str(n), x, y, size=size, anchor="mc", bold=True, color=hot, outline=max(3, size // 12),
                 rotation=ROTATION[player])


def draw_glow_disc(painter, x: float, y: float, r: float, color, segments: int = 32) -> None:
    """Měkká záře (kruh s průhledným okrajem) – jedno volání ``triangles``."""
    ang = np.linspace(0, 2 * np.pi, segments + 1)
    ring = np.stack([x + r * np.cos(ang), y + r * np.sin(ang)], 1)
    c = np.array([x, y], np.float32)
    v = np.stack([np.repeat(c[None], segments, 0), ring[:-1], ring[1:]], 1).reshape(-1, 2)
    col = np.tile(np.array([color, color[:3] + (0.0,), color[:3] + (0.0,)], np.float32), (segments, 1))
    painter.triangles(v, col)
