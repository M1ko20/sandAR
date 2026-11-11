"""Kulička (minigolf): kulička se kutálí po písku, návštěvníci staví dráhu do jamky.

Fyzika je v ``sandbox/sim/ball.py`` (sklon terénu z výškové mapy, valivý odpor,
odraz od stěn). Ruka nad kuličkou ji odkopne. Když kulička dojede do jamky
dost pomalu, je to „jamka“: ohňostroj, započítá se čas a jamka se přesune jinam.

Modul obsahuje i drobné efekty (jiskry, svítící tečky), které používá i hra Poklad.
"""

from __future__ import annotations

import math

import numpy as np

from ..sim.ball import Ball, BallPhysics, SlopeField, kick, step
from .base import Game

FRAG = """
void main() {
    vec2 uv = v_uv;
    float h = height_at(uv);
    // golfový trávník: tmavá zeleň v důlcích, světlá na kopcích
    float t = clamp((h - u_elev_min) / max(u_elev_max - u_elev_min, 1e-4), 0.0, 1.0);
    vec3 c = mix(vec3(0.02, 0.30, 0.22), vec3(0.35, 0.95, 0.20), smoothstep(0.0, 0.7, t));
    c = mix(c, vec3(1.0, 0.92, 0.45), smoothstep(0.6, 1.0, t) * 0.7);   // vršky do písková
    c *= mix(0.5, 1.3, hillshade(uv, 3.5));
    // pruhy posečeného trávníku – zvýrazní tvar terénu
    float stripe = step(0.5, fract(uv.x * u_box.x * 6.0));
    c *= mix(0.93, 1.05, stripe);
    vec2 cl = contours(h);
    c = mix(c, c * 0.35, cl.x * 0.7);
    c = mix(c, vec3(0.0, 0.08, 0.02), cl.y * 0.9);
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
    order = 60
    category = "hra"
    intro = "Uprav písek tak, aby se svítící kulička skutálela do jamky s vlajkou – rukou ji můžeš i odstrčit."
    keys = [
        ("ruka", "odstrčit kuličku"),
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
        self.hole = pick_spot(hm.heights, hm.size, self.rng, margin=0.12, away_from=self.pad,
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
        # ruka odkopne kuličku
        for b in self.balls:
            if b.kick_cooldown > 0:
                continue
            for hx, hy in hm.hand_points:
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
        self.sparks.update(dt)

    # --- kreslení ---------------------------------------------------------------------
    def render(self) -> None:
        self.host.draw_fullscreen(self.prog)
        host, p = self.host, self.host.painter
        cw, ch = host.canvas_size
        ppm = cw / host.box_size[0]
        t = self.t
        # start
        sx, sy = host.box_to_canvas(*self.pad)
        p.circle(sx, sy, 0.05 * ppm, (0.3, 0.9, 1.0, 0.9), width=4)
        p.circle(sx, sy, 0.05 * ppm, (0.3, 0.9, 1.0, 0.18))
        fs = max(18, ch // 28)
        p.text("START", sx, sy + 0.06 * ppm, size=int(fs * 0.8), anchor="tc", bold=True,
               color=(0.4, 0.95, 1, 1), outline=3)
        # jamka s vlajkou
        if self.hole is not None:
            hx, hy = host.box_to_canvas(*self.hole)
            r = self.HOLE_R * ppm
            pulse = 0.5 + 0.5 * math.sin(t * 4)
            p.circle(hx, hy, r * (1.6 + 0.5 * pulse), (1, 1, 1, 0.25 * (1 - pulse)), width=4)
            p.circle(hx, hy, r, (0.0, 0.0, 0.0, 1.0))
            p.circle(hx, hy, r, (1, 1, 1, 1), width=5)
            pole_top = hy - r * 3.2
            p.polyline([(hx, hy), (hx, pole_top)], (1, 1, 1, 1), width=5)
            wave = 6 * math.sin(t * 5)
            p.triangles(np.array([(hx, pole_top), (hx + r * 2.0, pole_top + r * 0.6 + wave * 0.3),
                                  (hx, pole_top + r * 1.2)]), (1.0, 0.1, 0.1, 1))
            p.text(str(self.holes_made + 1), hx + r * 0.7, pole_top + r * 0.55, size=int(fs * 0.7), anchor="mc",
                   bold=True, color=(1, 1, 1, 1))
        # kuličky se stopou
        for i, b in enumerate(self.balls):
            col = BALL_COLORS[i % len(BALL_COLORS)]
            if len(b.trail) > 1:
                pts = np.array([host.box_to_canvas(x, y) for x, y in b.trail] + [host.box_to_canvas(b.x, b.y)])
                fading_trail(host, pts, col, b.radius * ppm * 1.4)
            bx, by = host.box_to_canvas(b.x, b.y)
            glow_dot(host, bx, by, b.radius * ppm, col)
        self.sparks.draw(host, size_px=ch / 70)
        self._hud(fs)

    def _hud(self, fs: int) -> None:
        host, p = self.host, self.host.painter
        cw, ch = host.canvas_size
        big = max(fs, ch // 22)
        p.text(f"Jamky: {self.holes_made}", big * 0.6, big * 0.4, size=big, bold=True, color=(1, 0.95, 0.3, 1),
               outline=4)
        p.text(f"Čas: {fmt_time(self.t - self.hole_t0)}", cw - big * 0.6, big * 0.4, size=big,
               anchor="tr", bold=True, color=(1, 1, 1, 1), outline=4)
        if self.best is not None:
            p.text(f"Nejrychlejší: {fmt_time(self.best)}", cw - big * 0.6, big * 1.6, size=int(big * 0.7),
                   anchor="tr", color=(0.7, 1, 0.7, 1), outline=3)
        since = self.t - self.celebrate_t
        if since < 3.0:
            k = min(1.0, since * 4)
            x, y = host.box_to_canvas(*self.celebrate_pos)
            x = float(np.clip(x, cw * 0.2, cw * 0.8))
            y = float(np.clip(y - ch * 0.12, ch * 0.15, ch * 0.85))
            size = int(ch / 9 * (0.6 + 0.4 * k))
            p.text("JAMKA!", x, y, size=size, anchor="mc", bold=True, color=(1, 0.9, 0.1, 1), outline=6)
            p.text(f"za {self.times[-1]:.0f} s", x, y + size * 0.8, size=int(big * 0.9), anchor="mc", bold=True,
                   color=(1, 1, 1, 1), outline=4)
            ring = since * 0.6 * cw / host.box_size[0]
            cx, cy = host.box_to_canvas(*self.celebrate_pos)
            p.circle(cx, cy, ring, (1, 0.9, 0.2, max(0.0, 1 - since / 1.5)), width=8)
        elif self.t - self.hole_t0 > 40 and int(self.t / 6) % 2 == 0:
            p.text("Tip: nasyp písek pod kuličku nebo ji odstrč rukou", cw / 2, ch - big * 0.4, size=fs,
                   anchor="bc", bold=True, color=(1, 1, 1, 1), outline=4)

    # --- ovládání -------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        hm = self.host.heightmap
        if name == "n":
            self.balls.clear()
            self.pending_spawn = [self.t] * self.n_balls
        elif name == "j" and hm is not None:
            self._move_hole(hm)
            self.host.notify("Jamka přesunuta")
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
        lines = [f"Jamky: {self.holes_made}", f"Kuliček: {len(self.balls)}/{self.n_balls}",
                 f"Gravitace {self.phys.gravity / 9.81:.2f} g"]
        if self.times:
            lines.append(f"Poslední {self.times[-1]:.0f} s, nejlepší {self.best:.0f} s")
        for b in self.balls:
            lines.append(f"Kulička ({b.x:.2f}, {b.y:.2f}) m, {b.speed:.2f} m/s")
        return lines
