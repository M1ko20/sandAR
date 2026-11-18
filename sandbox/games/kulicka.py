"""Kulička (minigolf): kulička se kutálí po písku, návštěvníci staví dráhu do jamky.

Fyzika je v ``sandbox/sim/ball.py`` (sklon terénu z výškové mapy, valivý odpor,
odraz od stěn). Kulička se pohybuje jen po terénu; odpalování rukou (ruka u kuličky ji
odstrčí) se zapíná klávesou K. Když kulička dojede do jamky
dost pomalu, je to „jamka“: ohňostroj, započítá se čas a jamka se přesune jinam.

Modul obsahuje i drobné efekty (jiskry, svítící tečky), které používá i hra Poklad.
"""

from __future__ import annotations

import math

import numpy as np

from ..render import hud
from ..sim.ball import Ball, BallPhysics, SlopeField, kick, step
from .base import Game

FRAG = """
uniform vec2 u_hole;        // jamka (uv); x < 0 = zatím není
uniform vec2 u_pad;         // start (uv)
uniform vec4 u_balls[4];    // kuličky: uv, rychlost [m/s], 1 = existuje
uniform vec3 u_ballcol[4];
uniform float u_cel;        // [s] od poslední trefy (velké číslo = žádná oslava)
uniform vec2 u_cel_pos;     // uv poslední trefy

float kulicka_band(float x, float soft) {   // měkké pruhy 0/1 (bez ostrých hran -> žádný šum na drti)
    return smoothstep(-soft, soft, sin(x * 6.2831853));
}

void main() {
    vec2 uv = v_uv;
    vec2 pm = uv * u_box;
    float h = height_at(uv);
    float t = clamp((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4), 0.0, 1.0);
    // trávník: smaragd v dolících -> sytá zeleň -> světlá limetka na kopcích
    vec3 c = mix(vec3(0.0, 0.24, 0.16), vec3(0.06, 0.60, 0.19), smoothstep(0.0, 0.40, t));
    c = mix(c, vec3(0.56, 0.80, 0.12), smoothstep(0.40, 0.85, t));
    // posečený trávník: široké pruhy dvou směrů sekání (šachovnice jako na stadionu)
    float mow = kulicka_band(pm.x / 0.24, 0.35);
    float cross = kulicka_band(pm.y / 0.24, 0.35);
    c *= mix(0.94, 1.04, mow) * mix(0.97, 1.02, cross);
    // nejvyšší vršky = písečný bunkr
    float bunker = smoothstep(0.84, 0.92, t);
    c = mix(c, vec3(1.0, 0.85, 0.50), bunker);

    // green kolem jamky: jemně posečený, světlejší, se soustřednými pruhy a tmavším límcem
    if (u_hole.x >= 0.0) {
        float d = length((uv - u_hole) * u_box);
        float g = 1.0 - smoothstep(0.15, 0.165, d);
        vec3 green = vec3(0.16, 0.70, 0.28) * mix(0.93, 1.05, kulicka_band(d / 0.05, 0.5));
        c = mix(c, green, g * (1.0 - bunker));
        float collar = smoothstep(0.150, 0.162, d) * (1.0 - smoothstep(0.168, 0.185, d));
        c = mix(c, vec3(0.02, 0.32, 0.12), collar * 0.8);
    }
    c *= mix(0.5, 1.16, hillshade(uv, 3.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.55, cl.x * 0.45);
    c = mix(c, vec3(0.0, 0.10, 0.04), cl.y * 0.75);

    // světlo kuliček: barevná záře na trávě kolem každé kuličky
    for (int i = 0; i < 4; ++i) {
        vec4 b = u_balls[i];
        if (b.w < 0.5) continue;
        float d = length((uv - b.xy) * u_box);
        float k = 0.45 + 0.35 * clamp(b.z / 0.6, 0.0, 1.0);
        c += u_ballcol[i] * k * exp(-d * d / (0.055 * 0.055));
    }
    // oslava trefy: záblesk a zlatá rázová vlna přes celý trávník
    if (u_cel < 3.0) {
        float d = length((uv - u_cel_pos) * u_box);
        float r = u_cel * 0.75;
        float fade = 1.0 - u_cel / 3.0;
        c += vec3(1.0, 0.80, 0.25) * exp(-pow((d - r) / 0.035, 2.0)) * fade * 0.9;
        c += vec3(1.0, 0.95, 0.70) * exp(-pow((d - r * 0.6) / 0.02, 2.0)) * fade * 0.5;
        c += vec3(1.0, 0.85, 0.40) * exp(-d * d / 0.03) * max(0.0, 1.0 - u_cel * 1.4) * 0.45;
    }
    f_color = vec4(c, 1.0);
}
"""

BALL_COLORS = [(1.0, 0.95, 0.2), (1.0, 0.3, 0.85), (0.3, 0.95, 1.0), (1.0, 0.55, 0.1)]


# --- efekty sdílené s dalšími hrami ---------------------------------------------------
class Sparks:
    """Jiskry / konfety v metrech bedny – numpy pole, kreslí se jedním voláním."""

    def __init__(self, gravity: float = 0.0) -> None:
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)
        self.max_life = np.zeros(0, np.float32)
        self.color = np.zeros((0, 3), np.float32)
        self.gravity = gravity

    def __len__(self) -> int:
        return len(self.life)

    def burst(self, x: float, y: float, n: int, speed: float, colors, life: float = 1.5,
              rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        ang = rng.uniform(0, 2 * math.pi, n)
        sp = speed * np.sqrt(rng.uniform(0.05, 1.0, n))
        vel = np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1)
        cols = np.asarray(colors, np.float32).reshape(-1, 3)[rng.integers(0, len(colors), n)]
        lf = life * rng.uniform(0.6, 1.0, n)
        self.pos = np.vstack([self.pos, np.tile([x, y], (n, 1))]).astype(np.float32)
        self.vel = np.vstack([self.vel, vel]).astype(np.float32)
        self.life = np.concatenate([self.life, lf]).astype(np.float32)
        self.max_life = np.concatenate([self.max_life, lf]).astype(np.float32)
        self.color = np.vstack([self.color, cols]).astype(np.float32)

    def update(self, dt: float) -> None:
        if not len(self):
            return
        self.pos += self.vel * dt
        self.vel *= math.exp(-2.2 * dt)
        self.vel[:, 1] += self.gravity * dt
        self.life -= dt
        alive = self.life > 0
        if not alive.all():
            for name in ("pos", "vel", "life", "max_life", "color"):
                setattr(self, name, getattr(self, name)[alive])

    def draw(self, host, size_px: float = 10.0) -> None:
        if not len(self):
            return
        sx = host.canvas_size[0] / host.box_size[0]
        sy = host.canvas_size[1] / host.box_size[1]
        k = np.clip(self.life / np.maximum(self.max_life, 1e-3), 0, 1)
        r = (size_px * (0.4 + 0.6 * k))[:, None]
        c = np.stack([self.pos[:, 0] * sx, self.pos[:, 1] * sy], 1)
        # kosočtverec = 2 trojúhelníky
        up, right = np.array([0, -1.0]), np.array([1.0, 0])
        p_up, p_dn = c + up * r * 1.4, c - up * r * 1.4
        p_r, p_l = c + right * r, c - right * r
        tris = np.stack([p_up, p_r, p_dn, p_up, p_dn, p_l], 1).reshape(-1, 2)
        rgba = np.concatenate([self.color, k[:, None]], 1)
        host.painter.triangles(tris, np.repeat(rgba, 6, 0))


def glow_dot(host, x_px: float, y_px: float, r: float, color, intensity: float = 1.0) -> None:
    """Svítící tečka: měkká záře + jasné jádro."""
    p = host.painter
    cr, cg, cb = color[:3]
    for scale, alpha in ((3.2, 0.10), (2.2, 0.22), (1.5, 0.45)):
        p.circle(x_px, y_px, r * scale, (cr, cg, cb, alpha * intensity), segments=28)
    p.circle(x_px, y_px, r, (cr, cg, cb, 1.0), segments=28)
    p.circle(x_px, y_px, r * 0.5, (1, 1, 1, 0.95), segments=20)


def soft_discs(host, centers, radii, colors, segments: int = 24, edge: float = 0.0) -> None:
    """Mnoho měkkých kruhů jedním voláním: střed barvou ``colors`` (RGBA), okraj průhledný (× ``edge``)."""
    centers = np.asarray(centers, np.float32).reshape(-1, 2)
    n = len(centers)
    if n == 0:
        return
    radii = np.broadcast_to(np.asarray(radii, np.float32), (n,))
    colors = np.broadcast_to(np.asarray(colors, np.float32).reshape(-1, 4), (n, 4))
    ang = np.linspace(0, 2 * math.pi, segments + 1)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1)
    pts = centers[:, None, :] + radii[:, None, None] * ring[None]
    c = np.repeat(centers[:, None, :], segments, 1)
    tris = np.stack([c, pts[:, :-1], pts[:, 1:]], 2).reshape(-1, 2)
    cols = np.repeat(colors[:, None, :], segments * 3, 1).reshape(n, segments, 3, 4).copy()
    cols[:, :, 1:, 3] *= edge
    host.painter.triangles(tris, cols.reshape(-1, 4))


def ring_quads(cx: float, cy: float, r: float, width: float, a0: float, a1: float, segments: int = 12) -> np.ndarray:
    """Oblouk (úhly a0..a1 [rad]) jako trojúhelníky – pro čárkované kruhy skládané do jednoho volání."""
    ang = np.linspace(a0, a1, segments + 1)
    ro, ri = r + width / 2, r - width / 2
    o = np.stack([cx + ro * np.cos(ang), cy + ro * np.sin(ang)], 1)
    i = np.stack([cx + ri * np.cos(ang), cy + ri * np.sin(ang)], 1)
    return np.stack([o[:-1], o[1:], i[1:], o[:-1], i[1:], i[:-1]], 1).reshape(-1, 2)


def fading_trail(host, pts_px: np.ndarray, color, width: float) -> None:
    """Stopa: čára, která ke konci slábne a zužuje se."""
    n = len(pts_px)
    if n < 2:
        return
    a, b = pts_px[:-1], pts_px[1:]
    d = b - a
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    k = np.linspace(0.0, 1.0, n)  # 0 = nejstarší
    wa = (width * (0.2 + 0.8 * k[:-1]))[:, None] / 2
    wb = (width * (0.2 + 0.8 * k[1:]))[:, None] / 2
    quads = np.stack([a + nrm * wa, b + nrm * wb, b - nrm * wb, a + nrm * wa, b - nrm * wb, a - nrm * wa], 1)
    al_a, al_b = 0.8 * k[:-1] ** 1.5, 0.8 * k[1:] ** 1.5
    alpha = np.stack([al_a, al_b, al_b, al_a, al_b, al_a], 1).reshape(-1, 1)
    rgb = np.broadcast_to(np.asarray(color[:3], np.float32), (len(alpha), 3))
    host.painter.triangles(quads.reshape(-1, 2), np.concatenate([rgb, alpha], 1))


def fmt_time(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 60}:{s % 60:02d}"


# --- logika hry (bez OpenGL, testovatelná) -------------------------------------------
def pick_spot(heights: np.ndarray, size: tuple[float, float], rng: np.random.Generator, margin: float,
              away_from: tuple[float, float] | None = None, min_dist: float = 0.0,
              max_height: float | None = None, tries: int = 60) -> tuple[float, float]:
    """Náhodné místo v bedně (okraj ``margin``), dál než ``min_dist`` od bodu, pokud možno níž než max_height."""
    gh, gw = heights.shape
    W, D = size
    best, best_score = None, -1e9
    for _ in range(tries):
        x, y = rng.uniform(margin, W - margin), rng.uniform(margin, D - margin)
        score = 0.0
        if away_from is not None:
            d = math.hypot(x - away_from[0], y - away_from[1])
            score -= max(0.0, min_dist - d) * 10
        if max_height is not None:
            h = float(heights[min(int(y / D * gh), gh - 1), min(int(x / W * gw), gw - 1)])
            score -= max(0.0, h - max_height) * 10
        if score >= 0:
            return x, y
        if score > best_score:
            best, best_score = (x, y), score
    assert best is not None
    return best


def ball_in_hole(ball: Ball, hole: tuple[float, float], radius: float, max_speed: float) -> bool:
    """Kulička je v jamce, když je uvnitř a nejede moc rychle (jinak jamku „přeskočí“)."""
    return math.hypot(ball.x - hole[0], ball.y - hole[1]) < radius and ball.speed < max_speed


class MinigolfGame(Game):
    id = "kulicka"
    title = "Kulička (minigolf)"
    description = "Kulička se kutálí z kopce. Postav z písku dráhu, která ji dovede do jamky s vlajkou."
    order = 70
    category = "hra"
    intro = "Uprav písek tak, aby se svítící kulička sama skutálela do jamky s vlajkou!"
    keys = [
        ("K", "odpalování rukou zap/vyp (výchozí vyp)"),
        ("N", "nová kulička na start"),
        ("J", "přesunout jamku"),
        ("B", "počet kuliček 1–4"),
        ("↑ / ↓", "síla gravitace"),
        ("Backspace", "nová hra"),
    ]

    HOLE_R = 0.05  # [m] poloměr jamky (zachycení kuličky)
    CAPTURE_SPEED = 0.9  # [m/s] rychlejší kulička jamku přejede
    KICK_R = 0.09  # [m] ruka blíž než tohle kuličku odkopne
    STUCK_RESPAWN = 45.0  # [s] stojící kulička se vrátí na start

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.phys = BallPhysics()
        self.rng = np.random.default_rng()
        self.n_balls = getattr(self, "n_balls", 1)
        self.kicks = getattr(self, "kicks", False)  # ruka odstrčí kuličku jen po zapnutí (K)
        self.slope: SlopeField | None = None
        self._slope_key: object = None
        self.sparks = Sparks()
        self.t = 0.0
        self.balls: list[Ball] = []
        self.pad = (0.25, 0.5)
        self.hole: tuple[float, float] | None = None
        self.pending_spawn: list[float] = []
        self._new_game()

    def stop(self) -> None:
        self.prog.release()

    def reset(self) -> None:
        self._new_game()

    # --- herní logika ------------------------------------------------------------------
    def _new_game(self) -> None:
        self.holes_made = 0
        self.times: list[float] = []
        self.best: float | None = None
        self.celebrate_t = -99.0
        self.celebrate_pos = (0.0, 0.0)
        self.fireworks: list[tuple[float, float, float]] = []  # (čas, x, y) – další salvy oslavy
        self.balls = []
        self.hole = None  # umístí se při prvním update (potřebuje výšky)
        self.pending_spawn = [self.t] * self.n_balls
        self.hole_t0 = self.t

    def _place(self, hm) -> None:
        W, D = hm.size
        side = self.rng.integers(0, 2)
        x = self.rng.uniform(0.12, 0.3) * W
        self.pad = (x if side == 0 else W - x, self.rng.uniform(0.3, 0.7) * D)
        self._move_hole(hm)

    def _move_hole(self, hm) -> None:
        pad_h = hm.sample(*self.pad)
        self.hole = pick_spot(hm.heights, hm.size, self.rng, margin=0.16, away_from=self.pad,
                              min_dist=0.45 * hm.size[0], max_height=pad_h + 0.01)
        self.hole_t0 = self.t

    def _spawn(self) -> None:
        i = len(self.balls)
        off = 0.035 * i
        self.balls.append(Ball(self.pad[0] + off * math.cos(i * 2.1), self.pad[1] + off * math.sin(i * 2.1),
                               radius=0.022))

    def _score(self, ball: Ball) -> None:
        assert self.hole is not None
        took = self.t - self.hole_t0
        self.holes_made += 1
        self.times.append(took)
        self.best = took if self.best is None else min(self.best, took)
        self.celebrate_t = self.t
        self.celebrate_pos = self.hole
        cols = [(1, 0.9, 0.2), (1, 0.3, 0.6), (0.3, 1, 1), (1, 1, 1), (0.5, 1, 0.3)]
        self.sparks.burst(*self.hole, 140, 0.9, cols, life=2.0, rng=self.rng)
        hx, hy = self.hole
        for k in range(5):  # ohňostroj: další salvy kolem jamky
            a = self.rng.uniform(0, 2 * math.pi)
            r = self.rng.uniform(0.12, 0.25)
            self.fireworks.append((self.t + 0.3 + 0.35 * k, hx + r * math.cos(a), hy + r * math.sin(a)))
        self.balls.remove(ball)
        self.pending_spawn.append(self.t + 1.5)
        hm = self.host.heightmap
        if hm is not None:
            self._move_hole(hm)

    def update(self, hm, dt: float) -> None:
        dt = min(dt, 1 / 15)
        self.t += dt
        key = (id(hm.heights), hm.seq)
        if self.slope is None or key != self._slope_key:
            self.slope = SlopeField(hm.heights, hm.cell_size)
            self._slope_key = key
        if self.hole is None:
            self._place(hm)
        # nové kuličky
        due = [s for s in self.pending_spawn if s <= self.t]
        for _ in due:
            if len(self.balls) < self.n_balls:
                self._spawn()
        self.pending_spawn = [s for s in self.pending_spawn if s > self.t]
        # ruka odstrčí kuličku (jen v režimu odpalování)
        for b in self.balls if self.kicks else []:
            if b.kick_cooldown > 0:
                continue
            for hand in hm.hand_list:
                hx, hy = min((hand.pos, hand.tip), key=lambda q: math.hypot(b.x - q[0], b.y - q[1]))
                if math.hypot(b.x - hx, b.y - hy) < self.KICK_R:
                    kick(b, hx, hy, 0.8)
                    b.kick_cooldown = 1.2
                    break
        step(self.balls, self.slope, dt, self.phys)
        for b in list(self.balls):
            if not b.trail or math.hypot(b.x - b.trail[-1][0], b.y - b.trail[-1][1]) > 0.004:
                b.trail.append((b.x, b.y))
                del b.trail[:-50]
            if self.hole is not None and ball_in_hole(b, self.hole, self.HOLE_R, self.CAPTURE_SPEED):
                self._score(b)
            elif b.still_time > self.STUCK_RESPAWN:
                self.balls.remove(b)
                self.pending_spawn.append(self.t + 0.5)
        for fw in [f for f in self.fireworks if f[0] <= self.t]:
            self.fireworks.remove(fw)
            pal = [[(1, 0.85, 0.2), (1, 1, 0.7)], [(1, 0.3, 0.7), (1, 0.8, 0.9)], [(0.3, 1, 1), (0.8, 1, 1)],
                   [(0.5, 1, 0.3), (1, 1, 1)]][int(self.rng.integers(0, 4))]
            self.sparks.burst(fw[1], fw[2], 70, 0.6, pal, life=1.4, rng=self.rng)
        self.sparks.update(dt)

    # --- kreslení ---------------------------------------------------------------------
    def render(self) -> None:
        host = self.host
        bw, bd = host.box_size
        balls = np.zeros((4, 4), np.float32)
        bcols = np.zeros((4, 3), np.float32)
        for i, b in enumerate(self.balls[:4]):
            balls[i] = (b.x / bw, b.y / bd, b.speed, 1.0)
            bcols[i] = BALL_COLORS[i % len(BALL_COLORS)]
        for name, data in (("u_balls", balls), ("u_ballcol", bcols)):
            try:
                self.prog[name].write(data.tobytes())  # type: ignore[union-attr]
            except KeyError:
                pass
        hole = (self.hole[0] / bw, self.hole[1] / bd) if self.hole is not None else (-1.0, -1.0)
        host.draw_fullscreen(self.prog, u_hole=hole, u_pad=(self.pad[0] / bw, self.pad[1] / bd),
                             u_cel=float(min(self.t - self.celebrate_t, 99.0)),
                             u_cel_pos=(self.celebrate_pos[0] / bw, self.celebrate_pos[1] / bd))
        cw, ch = host.canvas_size
        ppm = cw / bw
        fs = max(18, ch // 28)
        self._draw_start(ppm, fs)
        if self.hole is not None:
            self._draw_hole(ppm, fs)
        self._draw_balls(ppm)
        self.sparks.draw(host, size_px=ch / 60)
        since = self.t - self.celebrate_t
        if since < 3.0:  # zlatá záře za nápisem JAMKA! (rozzáří se v bloomu)
            x, y = self._cel_text_pos()
            k = max(0.0, 1 - since / 3.0)
            soft_discs(host, [(x - ch * 0.12, y), (x, y), (x + ch * 0.12, y)], ch * 0.16, (1, 0.7, 0.1, 0.55 * k))
        host.glow(0.9, 0.8)
        self._hud(fs)

    def _draw_start(self, ppm: float, fs: int) -> None:
        host, p = self.host, self.host.painter
        sx, sy = host.box_to_canvas(*self.pad)
        r = 0.05 * ppm
        soft_discs(host, [(sx, sy)], r * 1.8, (0.2, 0.85, 1.0, 0.35))
        p.circle(sx, sy, r, (0.15, 0.55, 0.75, 0.55))
        # točící se čárkovaný kruh (odpaliště)
        rot = self.t * 0.8
        quads = [ring_quads(sx, sy, r * 1.25, max(5.0, r * 0.16), rot + k * math.pi / 4, rot + k * math.pi / 4 + 0.5,
                            6) for k in range(8)]
        p.triangles(np.concatenate(quads), (0.4, 0.95, 1.0, 0.95))
        p.circle(sx, sy, r, (0.6, 1.0, 1.0, 1.0), width=max(4.0, r * 0.1))
        p.text("START", sx, sy + r * 1.55, size=int(fs * 0.8), anchor="tc", bold=True,
               color=(0.5, 0.97, 1, 1), outline=3, outline_color=(0, 0.1, 0.2, 1))

    def _draw_hole(self, ppm: float, fs: int) -> None:
        host, p = self.host, self.host.painter
        t = self.t
        hx, hy = host.box_to_canvas(*self.hole)
        r = self.HOLE_R * ppm
        # výzva: zlaté kruhy se stahují do jamky
        for k in range(3):
            f = 1.0 - ((t * 0.55 + k / 3) % 1.0)
            p.circle(hx, hy, r * (1.15 + 2.2 * f), (1, 0.92, 0.45, 0.75 * (1 - f) * min(1.0, f * 4)),
                     width=max(4.0, r * 0.13))
        # jamka: stín, černý důlek, bílý lem
        soft_discs(host, [(hx + r * 0.15, hy + r * 0.2)], r * 1.6, (0, 0.05, 0, 0.55))
        p.circle(hx, hy, r * 1.08, (0.95, 1.0, 0.95, 1.0))
        p.circle(hx, hy, r * 0.92, (0.0, 0.0, 0.0, 1.0))
        p.circle(hx - r * 0.12, hy - r * 0.14, r * 0.7, (0.06, 0.09, 0.07, 1.0))
        # stín žerdi na trávu (světlo zleva shora)
        pole_h = r * 3.6
        p.polyline([(hx, hy), (hx + pole_h * 0.55, hy + pole_h * 0.25)], (0, 0.06, 0, 0.45), width=max(5.0, r * 0.16))
        # vlajka: vlnící se látka
        top = hy - pole_h
        n = 10
        u = np.linspace(0.0, 1.0, n + 1)
        fw, fh = r * 2.3, r * 1.25
        wave = r * 0.28 * u * np.sin(t * 5.5 - u * 4.0)
        xs = hx + u * fw * (1 - 0.06 * np.abs(np.sin(t * 5.5 - u * 4.0)))
        top_y = top + wave
        bot_y = top + fh * (1 - 0.18 * u) + wave
        shade = 0.78 + 0.22 * np.cos(t * 5.5 - u * 4.0)
        tl = np.stack([xs, top_y], 1)
        bl = np.stack([xs, bot_y], 1)
        verts = np.stack([tl[:-1], tl[1:], bl[1:], tl[:-1], bl[1:], bl[:-1]], 1).reshape(-1, 2)
        cs = np.stack([np.full(n + 1, 1.0) * shade, 0.12 * shade, 0.16 * shade, np.ones(n + 1)], 1)
        cols = np.stack([cs[:-1], cs[1:], cs[1:], cs[:-1], cs[1:], cs[:-1]], 1).reshape(-1, 4)
        p.triangles(verts, cols)
        # žerď s obrysem a zlatou kuličkou
        p.polyline([(hx, hy), (hx, top - r * 0.2)], (0.05, 0.05, 0.05, 1), width=max(9.0, r * 0.26))
        p.polyline([(hx, hy), (hx, top - r * 0.2)], (1, 1, 1, 1), width=max(5.0, r * 0.14))
        p.circle(hx, top - r * 0.25, max(5.0, r * 0.2), (1, 0.85, 0.2, 1), segments=16)
        mid = n // 2
        p.text(str(self.holes_made + 1), float(xs[mid]), float((top_y[mid] + bot_y[mid]) / 2), size=int(fs * 0.8),
               anchor="mc", bold=True, color=(1, 1, 1, 1), outline=3, outline_color=(0.4, 0, 0, 1))

    def _draw_balls(self, ppm: float) -> None:
        host, p = self.host, self.host.painter
        for i, b in enumerate(self.balls):
            col = BALL_COLORS[i % len(BALL_COLORS)]
            r = b.radius * ppm
            bx, by = host.box_to_canvas(b.x, b.y)
            if len(b.trail) > 1:  # kometa: široká barevná stopa + světlé jádro
                pts = np.array([host.box_to_canvas(x, y) for x, y in b.trail] + [(bx, by)])
                fading_trail(host, pts, col, r * 2.2)
                fading_trail(host, pts[len(pts) // 2:], (1, 1, 1), r * 0.7)
            soft_discs(host, [(bx + r * 0.45, by + r * 0.6)], r * 1.5, (0, 0.04, 0, 0.6))  # stín
            soft_discs(host, [(bx, by)], r * 3.4, (*col, 0.45))  # záře
            p.circle(bx, by, r * 1.12, (0.05, 0.05, 0.05, 0.9), segments=28)
            p.circle(bx, by, r, (*col, 1.0), segments=28)
            p.circle(bx - r * 0.12, by - r * 0.12, r * 0.72, (min(1, col[0] + 0.35), min(1, col[1] + 0.35),
                                                              min(1, col[2] + 0.35), 1.0), segments=24)
            p.circle(bx - r * 0.35, by - r * 0.38, r * 0.3, (1, 1, 1, 0.95), segments=16)

    def _cel_text_pos(self) -> tuple[float, float]:
        cw, ch = self.host.canvas_size
        x, y = self.host.box_to_canvas(*self.celebrate_pos)
        return float(np.clip(x, cw * 0.2, cw * 0.8)), float(np.clip(y - ch * 0.2, ch * 0.15, ch * 0.85))

    def _hud(self, fs: int) -> None:
        host, p = self.host, self.host.painter
        cw, ch = host.canvas_size
        big = max(fs, ch // 22)
        hud.label(p, f"Jamky: {self.holes_made}", big * 0.5, big * 0.45, big, color=(1, 0.92, 0.3, 1), anchor="tl")
        hud.label(p, f"Čas: {fmt_time(self.t - self.hole_t0)}", cw - big * 0.5, big * 0.45, big, anchor="tr")
        if self.best is not None:
            hud.label(p, f"Nejrychlejší: {fmt_time(self.best)}", cw - big * 0.5, big * 1.95, int(big * 0.7),
                      color=(0.7, 1, 0.7, 1), anchor="tr")
        since = self.t - self.celebrate_t
        if since < 3.0:
            x, y = self._cel_text_pos()
            pop = 1.0 + 0.35 * math.exp(-since * 4) * math.cos(since * 18)  # nápis „vyskočí“
            size = int(ch / 8 * min(1.0, 0.4 + since * 3) * pop)
            fade = min(1.0, (3.0 - since) * 2)
            p.text("JAMKA!", x, y, size=size, anchor="mc", bold=True, color=(1, 0.88, 0.1, 1), outline=max(5, size // 12),
                   outline_color=(0.45, 0.12, 0, 1), alpha=fade)
            p.text(f"za {self.times[-1]:.0f} s", x, y + size * 0.75, size=int(big * 0.9), anchor="mc", bold=True,
                   color=(1, 1, 1, 1), outline=4, alpha=fade)
        elif self.t - self.hole_t0 > 40 and int(self.t / 6) % 2 == 0:
            tip = "Tip: nasyp písek pod kuličku nebo ji odstrč rukou" if self.kicks else \
                "Tip: nasyp písek za kuličku nebo vykopej koryto k jamce"
            hud.label(p, tip, cw / 2, ch - big * 0.5, fs, anchor="bc")

    # --- ovládání -------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        hm = self.host.heightmap
        if name == "n":
            self.balls.clear()
            self.pending_spawn = [self.t] * self.n_balls
        elif name == "j" and hm is not None:
            self._move_hole(hm)
            self.host.notify("Jamka přesunuta")
        elif name == "k":
            self.kicks = not self.kicks
            self.host.notify("Odpalování rukou zapnuto" if self.kicks else "Kulička jen po terénu")
        elif name == "b":
            self.n_balls = self.n_balls % 4 + 1
            del self.balls[self.n_balls:]
            self.pending_spawn = [self.t] * (self.n_balls - len(self.balls))
            self.host.notify(f"Kuliček: {self.n_balls}")
        elif name in ("up", "down"):
            g = self.phys.gravity * (1.25 if name == "up" else 0.8)
            self.phys.gravity = float(np.clip(g, 1.0, 20.0))
            self.host.notify(f"Gravitace {self.phys.gravity / 9.81:.2f} g")
        elif name == "backspace":
            self.reset()
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Jamky: {self.holes_made}", f"Kuliček: {len(self.balls)}/{self.n_balls} · odpalování rukou: "
                 + ("ano" if self.kicks else "ne"),
                 f"Gravitace {self.phys.gravity / 9.81:.2f} g"]
        if self.times:
            lines.append(f"Poslední {self.times[-1]:.0f} s, nejlepší {self.best:.0f} s")
        for b in self.balls:
            lines.append(f"Kulička ({b.x:.2f}, {b.y:.2f}) m, {b.speed:.2f} m/s")
        return lines
