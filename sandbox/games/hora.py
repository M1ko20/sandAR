"""Nejvyšší hora – soutěž, kdo postaví nejvyšší horu (nebo vyhloubí nejhlubší propast).

Kolo trvá 30 s a začne samo, jakmile se někdo dotkne písku (nebo Enter).
Na písku svítí značka na nejvyšším místě a velké číslo v metrech nad mořem
(podle měřítka mapy: 1 cm písku = 50 m) i ve skutečných centimetrech.
Po konci kola se výsledek zapíše do ``config/records.json`` (rekord všech dob
a pět nejlepších dne). Překonaný rekord oslaví ohňostroj „NOVÝ REKORD!“.

Ruce nad pískem se nepočítají – buňky pod rukou (a kousek okolo) se ignorují.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..config import CONFIG_DIR, write_json
from .base import Game

log = logging.getLogger(__name__)

ROUND = 30.0  # [s] délka kola
RESULT = 12.0  # [s] jak dlouho svítí výsledek
FINAL_WINDOW = 1.5  # [s] výsledek = nejlepší hodnota z posledních chvil kola (kdyby ruka zakryla vrchol)
HANDS_TO_START = 0.5  # [s] jak dlouho musí být ruka nad pískem, aby začalo kolo
TOP_N = 5

VARIANTS = {
    # id: (název, výzva, popis veličiny)
    "hora": ("Nejvyšší hora", "Postav co nejvyšší horu!", "nejvyšší bod"),
    "propast": ("Nejhlubší propast", "Vyhlub co nejhlubší propast!", "nejnižší bod"),
}

FRAG = """
uniform int u_pit;
uniform float u_rec_h;     // výška rekordu [m] (hora: terén nad ní zlatě září; propast: pod ní)
uniform vec3 u_peak;       // vrchol / dno: uv, výška [m]; z < -1 = není
uniform float u_gold;      // 0..1 oslava rekordu
uniform float u_alarm;     // 0..1 posledních pár sekund kola
void main() {
    vec2 pm = v_uv * u_box;
    float h = height_at(v_uv);
    vec3 c = elevation_color(h);
    c *= mix(0.6, 1.12, hillshade(v_uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.35, cl.x * 0.8);
    c = mix(c, vec3(0.06), cl.y);
    float sgn = u_pit == 1 ? -1.0 : 1.0;
    // terén „přes rekord“ se rozzáří zlatě (vlní se), na výšce rekordu je zlatá vrstevnice
    if (u_rec_h > -0.9) {
        float above = (h - u_rec_h) * sgn;
        float ab = smoothstep(0.0, 0.003, above);
        float shimmer = 0.85 + 0.15 * sin(u_time * 5.0 - above * 900.0);
        c = mix(c, vec3(1.0, 0.80, 0.18) * shimmer, ab * 0.75);
        float w = max(fwidth(h), 1e-5);
        float line = 1.0 - smoothstep(1.5 * w, 3.0 * w, abs(h - u_rec_h));
        c = mix(c, vec3(1.0, 0.85, 0.25), line);
    }
    // vrcholová čepice: místa do 1 cm pod vrcholem v jeho okolí tepou světlem
    if (u_peak.z > -1.0) {
        float d = length((v_uv - u_peak.xy) * u_box);
        float below = (u_peak.z - h) * sgn;
        float cap = (1.0 - smoothstep(0.0, 0.012, below)) * exp(-d * d / (0.12 * 0.12));
        vec3 col = u_pit == 1 ? vec3(0.2, 0.85, 1.0) : vec3(1.0, 0.35, 0.12);
        c = mix(c, col, cap * (0.35 + 0.2 * sin(u_time * 4.0)));
        // oslava rekordu: bedna zezlátne
        c = mix(c, c * vec3(1.15, 1.05, 0.7) + vec3(0.1, 0.07, 0.0), u_gold * 0.4);
    }
    vec2 e = min(pm, u_box - pm);
    float edge = 1.0 - smoothstep(0.0, 0.09, min(e.x, e.y));
    c = mix(c, vec3(1.0, 0.1, 0.05), edge * u_alarm * (0.5 + 0.5 * sin(u_time * 12.0)));
    f_color = vec4(c, 1.0);
}
"""


class Fireworks:
    """Ohňostroj v pixelech plátna: rakety stoupají a vybuchnou do jisker (numpy, jedno kreslení)."""

    PALETTE = np.array([(1, 0.85, 0.15), (1, 0.3, 0.25), (0.25, 0.85, 1), (0.45, 1, 0.35), (1, 0.4, 0.95),
                        (1, 1, 1)], np.float32)

    def __init__(self) -> None:
        self.rng = np.random.default_rng()
        self.rockets = np.zeros((0, 6), np.float32)  # x, y, vy, cíl y, barva (index), 0
        self.sparks = np.zeros((0, 9), np.float32)  # x, y, vx, vy, život, max, r, g, b

    def launch(self, cw: float, ch: float, n: int = 1) -> None:
        r = np.column_stack([self.rng.uniform(0.1, 0.9, n) * cw, np.full(n, ch * 1.02),
                             -self.rng.uniform(0.9, 1.2, n) * ch, self.rng.uniform(0.15, 0.55, n) * ch,
                             self.rng.integers(0, len(self.PALETTE), n), np.zeros(n)]).astype(np.float32)
        self.rockets = np.vstack([self.rockets, r])

    def burst(self, x: float, y: float, color_i: int, ch: float, n: int = 90) -> None:
        ang = self.rng.uniform(0, 2 * np.pi, n)
        sp = ch * 0.45 * np.sqrt(self.rng.uniform(0.1, 1.0, n))
        col = np.where(self.rng.random(n)[:, None] < 0.75, self.PALETTE[int(color_i)], self.PALETTE[-1])
        life = self.rng.uniform(1.0, 1.8, n)
        new = np.column_stack([np.full(n, x), np.full(n, y), np.cos(ang) * sp, np.sin(ang) * sp, life, life,
                               col]).astype(np.float32)
        self.sparks = np.vstack([self.sparks, new])[-1500:]

    def update(self, dt: float, ch: float) -> list[float]:
        """Posune rakety a jiskry; vrací x [px] raket, které právě vybuchly (pro zvuk)."""
        exploded: list[float] = []
        if len(self.rockets):
            self.rockets[:, 1] += self.rockets[:, 2] * dt
            done = self.rockets[:, 1] <= self.rockets[:, 3]
            for r in self.rockets[done]:
                self.burst(float(r[0]), float(r[1]), int(r[4]), ch)
                exploded.append(float(r[0]))
            self.rockets = self.rockets[~done]
        if len(self.sparks):
            s = self.sparks
            s[:, 0:2] += s[:, 2:4] * dt
            s[:, 2:4] *= math.exp(-2.0 * dt)
            s[:, 3] += ch * 0.12 * dt
            s[:, 4] -= dt
            self.sparks = s[s[:, 4] > 0]
        return exploded

    def draw(self, p, ch: float) -> None:
        verts, cols = [], []
        size = ch / 75
        if len(self.rockets):  # raketa: světlá hlavička se stopou
            r = self.rockets
            x, y = r[:, 0], r[:, 1]
            w = size * 0.5
            verts.append(np.stack([np.stack([x - w, y], 1), np.stack([x + w, y], 1), np.stack([x, y + size * 6], 1)],
                                  1).reshape(-1, 2))
            c = np.repeat(np.array([[1, 0.95, 0.7, 1], [1, 0.95, 0.7, 1], [1, 0.5, 0.1, 0]], np.float32)[None],
                          len(r), 0)
            cols.append(c.reshape(-1, 4))
        if len(self.sparks):
            s = self.sparks
            k = np.clip(s[:, 4] / s[:, 5], 0, 1)
            rr = (size * (0.35 + 0.65 * k))[:, None]
            c = s[:, 0:2]
            up, right = np.array([0, -1.0]), np.array([1.0, 0])
            tris = np.stack([c + up * rr * 1.4, c + right * rr, c - up * rr * 1.4, c + up * rr * 1.4,
                             c - up * rr * 1.4, c - right * rr], 1).reshape(-1, 2)
            verts.append(tris)
            cols.append(np.repeat(np.column_stack([s[:, 6:9], k]), 6, 0))
        if verts:
            p.triangles(np.concatenate(verts), np.concatenate(cols))


# --- záznamy rekordů ---------------------------------------------------------------------
class Records:
    """Rekordy v JSON souboru. ``score`` je vždy „víc = lepší“ (výška hory / hloubka propasti v m)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            log.warning("Soubor rekordů %s je poškozený – začínám znovu", self.path)
            return {}

    def record(self, kind: str) -> dict | None:
        return self.data.get(kind, {}).get("record")

    def today(self, kind: str, day: str | None = None) -> list[dict]:
        day = day or _dt.date.today().isoformat()
        return list(self.data.get(kind, {}).get("days", {}).get(day, []))

    def submit(self, kind: str, score: float, altitude: float, now: _dt.datetime | None = None) -> tuple[bool, int | None]:
        """Zapíše výsledek kola. Vrací (nový rekord?, pořadí dne 1..TOP_N nebo None)."""
        now = now or _dt.datetime.now()
        entry = {"score": round(float(score), 4), "alt": round(float(altitude), 1),
                 "date": now.date().isoformat(), "time": now.strftime("%H:%M")}
        k = self.data.setdefault(kind, {})
        best = k.get("record")
        is_record = best is None or entry["score"] > float(best.get("score", -1e9))
        if is_record:
            k["record"] = entry
        days = k.setdefault("days", {})
        lst = days.setdefault(entry["date"], [])
        lst.append(entry)
        lst.sort(key=lambda e: -float(e["score"]))
        rank = lst.index(entry) + 1
        del lst[TOP_N:]
        try:
            write_json(self.path, self.data)  # atomicky: .tmp + přejmenování
        except OSError:
            log.exception("Rekordy se nepodařilo uložit")
        return is_record, (rank if rank <= TOP_N else None)


def extreme_point(heights: np.ndarray, hands: np.ndarray | None, pit: bool = False,
                  border: int | None = None) -> tuple[int, int, float] | None:
    """Nejvyšší (pit=True: nejnižší) místo terénu mimo ruce a okraj: (iy, ix, výška [m]).

    Okraj (výchozí 4 % šířky) se nepočítá – u stěn bedny Kinect měří nespolehlivě.
    """
    if border is None:
        border = max(2, heights.shape[1] // 25)
    h = cv2.GaussianBlur(np.asarray(heights, np.float32), (0, 0), 1.0)  # jeden šumivý pixel nevyhraje
    if pit:
        h = -h
    valid = np.zeros(h.shape, bool)
    valid[border:h.shape[0] - border, border:h.shape[1] - border] = True
    if hands is not None and hands.any():
        valid &= cv2.dilate(hands.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    if not valid.any():
        return None
    h = np.where(valid, h, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(h)), h.shape)
    v = float(h[iy, ix])
    return int(iy), int(ix), (-v if pit else v)


class HighestMountainGame(Game):
    id = "hora"
    title = "Nejvyšší hora"
    description = "Soutěž: kdo za minutu postaví nejvyšší horu? Rekordy dne a všech dob. V = nejhlubší propast."
    order = 85
    category = "soutěž"
    intro = "Máš 30 sekund – postav co nejvyšší horu a překonej dnešní rekord!"
    keys = [("Enter / mezerník", "nové kolo hned"), ("V", "nejvyšší hora / nejhlubší propast"),
            ("X", "ukončit kolo bez zápisu")]
    records_path: Path = CONFIG_DIR / "records.json"

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.records = Records(self.records_path)
        self.variant = getattr(self, "variant", "hora")
        self.reset()

    def stop(self) -> None:
        self.prog.release()

    def reset(self) -> None:
        self.t = 0.0
        self.state = "ready"  # ready | round | result
        self.state_t = 0.0
        self.hand_time = 0.0
        self.point: tuple[int, int, float] | None = None
        self.shown_xy: tuple[float, float] | None = None  # vyhlazená poloha značky na plátně
        self.recent: deque[tuple[float, float]] = deque()  # (čas, výška) za poslední chvíli kola
        self.round_best: float | None = None
        self.final: float | None = None
        self.final_xy: tuple[float, float] | None = None
        self.is_record = False
        self.rank: int | None = None
        self._next_measure = 0.0
        self.confetti: np.ndarray | None = None  # (N, 8): x, y, vx, vy, úhel, r, g, b
        self.fw = Fireworks()  # ohňostroj (jen vzhled)
        self._next_rocket = 0.0
        self._pop_t = -1.0  # poslední zvuk prasknutí ohňostroje
        self.gauge_v: float | None = None  # vyhlazená hodnota na výškové stupnici [m]

    # --- veličiny ------------------------------------------------------------------------
    @property
    def pit(self) -> bool:
        return self.variant == "propast"

    def score_of(self, h: float) -> float:
        """Výška [m] -> skóre (víc = lepší)."""
        return -h if self.pit else h

    def _alt(self, h: float) -> str:
        return f"{self.host.altitude(h):,.0f}".replace(",", " ") + " m n. m."

    def _cm(self, h: float) -> str:
        txt = f"{abs(h) * 100:.1f}".replace(".", ",")
        return f"{txt} cm {'pod' if h < 0 else 'nad'} rovinou"

    # --- logika -----------------------------------------------------------------------------
    def update(self, hm, dt: float) -> None:
        self.t += dt
        self.state_t += dt
        hands = hm.has_hands
        self.hand_time = self.hand_time + dt if hands else 0.0
        if self.t >= self._next_measure:  # stačí 10x za sekundu
            self._next_measure = self.t + 0.1
            self.point = extreme_point(hm.heights, hm.hands, self.pit)
            if self.point is not None:
                gh, gw = hm.shape
                cw, ch = self.host.canvas_size
                x, y = (self.point[1] + 0.5) / gw * cw, (self.point[0] + 0.5) / gh * ch
                if self.shown_xy is None or math.hypot(x - self.shown_xy[0], y - self.shown_xy[1]) > ch * 0.08:
                    self.shown_xy = (x, y)  # skok na jiný kopec
                else:
                    self.shown_xy = (self.shown_xy[0] + 0.5 * (x - self.shown_xy[0]),
                                     self.shown_xy[1] + 0.5 * (y - self.shown_xy[1]))
            if self.state == "round" and self.point is not None:
                v = self.point[2]
                self.recent.append((self.t, v))
                if self.round_best is None or self.score_of(v) > self.score_of(self.round_best):
                    self.round_best = v

        if self.point is not None:  # výšková stupnice plynule dojíždí k aktuální hodnotě
            target = self.score_of(self.final if self.state == "result" and self.final is not None
                                   else self.point[2])
            self.gauge_v = target if self.gauge_v is None else self.gauge_v + (target - self.gauge_v) * min(1.0, dt * 6)
        ch = self.host.canvas_size[1]
        if self.state == "result" and self.is_record and self.state_t < 8 and self.t >= self._next_rocket:
            self._next_rocket = self.t + 0.3
            self.fw.launch(self.host.canvas_size[0], ch, 2)
        exploded = self.fw.update(dt, ch)
        if exploded and self.t - self._pop_t >= 0.4 and not (self.state == "result" and self.state_t > 6):
            # prasknutí ohňostroje: nejvýš 2,5× za s a jen prvních 6 s oslavy (ať to neleze na nervy)
            self._pop_t = self.t
            cw = self.host.canvas_size[0]
            self.sfx.play("pop", 0.45, x=sum(exploded) / len(exploded) / cw,
                          pitch=0.9 + 0.25 * ((self.t * 7.3) % 1.0))

        if self.state == "ready":
            if self.hand_time >= HANDS_TO_START:
                self.start_round()
        elif self.state == "round":
            left, prev = ROUND - self.state_t, ROUND - (self.state_t - dt)
            if 0 < left and math.ceil(left) != math.ceil(prev) and math.ceil(left) <= 5:
                self.sfx.play("go" if math.ceil(left) == 1 else "beep", 0.8)  # odpočet posledních 5 s
            while self.recent and self.recent[0][0] < self.t - FINAL_WINDOW:
                self.recent.popleft()
            if self.state_t >= ROUND:
                self._finish()
        elif self.state == "result":
            if self.confetti is not None:
                self._step_confetti(dt)
            if self.state_t >= RESULT:
                self._set_state("ready")
                self.confetti = None
                self.hand_time = 0.0

    def _set_state(self, state: str) -> None:
        self.state, self.state_t = state, 0.0

    def start_round(self) -> None:
        self._set_state("round")
        self.recent.clear()
        self.round_best = None
        self.final = None
        self.confetti = None
        self.host.notify(VARIANTS[self.variant][1])
        self.sfx.play("whistle", 0.9)

    def _finish(self) -> None:
        if not self.recent and self.point is not None:
            self.recent.append((self.t, self.point[2]))
        self._set_state("result")
        self.sfx.play("gong", 0.9)
        if not self.recent:
            self.final = None
            return
        self.final = max((v for _, v in self.recent), key=self.score_of)
        self.final_xy = self.shown_xy
        self.is_record, self.rank = self.records.submit(self.variant, self.score_of(self.final),
                                                        self.host.altitude(self.final))
        x, y = self.final_xy or (self.host.canvas_size[0] / 2, self.host.canvas_size[1] / 2)
        if self.is_record:
            self._spawn_confetti()
            self.fw.burst(x, y, 0, self.host.canvas_size[1], 160)
            self.sfx.play("fanfare", 1.0, x=x / self.host.canvas_size[0])
        elif self.rank is not None:  # umístění v TOP 5 dne: jedna salva na vrcholu
            self.fw.burst(x, y, 2, self.host.canvas_size[1], 90)
            self.sfx.play("chime", 0.9, x=x / self.host.canvas_size[0])
        log.info("%s: %.1f cm (%s), rekord=%s, pořadí dne=%s", VARIANTS[self.variant][0], self.final * 100,
                 self._alt(self.final), self.is_record, self.rank)

    # --- oslava ---------------------------------------------------------------------------------
    def _spawn_confetti(self, n: int = 220) -> None:
        cw, ch = self.host.canvas_size
        rng = np.random.default_rng()
        x0, y0 = self.final_xy or (cw / 2, ch / 2)
        ang = rng.uniform(0, 2 * np.pi, n)
        speed = rng.uniform(0.25, 0.9, n) * ch
        palette = np.array([(1, 0.85, 0.1), (1, 0.3, 0.2), (0.2, 0.8, 1), (0.4, 1, 0.3), (1, 0.4, 0.9), (1, 1, 1)])
        cols = palette[rng.integers(0, len(palette), n)]
        self.confetti = np.column_stack([np.full(n, x0), np.full(n, y0), np.cos(ang) * speed, np.sin(ang) * speed,
                                         rng.uniform(0, np.pi, n), cols]).astype(np.float32)

    def _step_confetti(self, dt: float) -> None:
        c = self.confetti
        assert c is not None
        c[:, 0:2] += c[:, 2:4] * dt
        c[:, 2:4] *= 0.985  # odpor vzduchu
        c[:, 3] += self.host.canvas_size[1] * 0.25 * dt  # lehká „gravitace“ směrem k divákům
        c[:, 4] += 6 * dt
        if self.state_t > 3.5 and self.state_t < 6:  # druhá salva
            if not getattr(self, "_second", False):
                self._second = True
                old = c
                self._spawn_confetti(160)
                self.confetti = np.vstack([old, self.confetti])
        if self.state_t < 1:
            self._second = False

    def _draw_confetti(self) -> None:
        c = self.confetti
        if c is None or len(c) == 0:
            return
        s = max(6.0, self.host.canvas_size[1] / 60)
        ca, sa = np.cos(c[:, 4]) * s, np.sin(c[:, 4]) * s * 0.5
        x, y = c[:, 0], c[:, 1]
        p0 = np.stack([x - ca, y - sa], 1)
        p1 = np.stack([x + sa, y - ca * 0.5], 1)
        p2 = np.stack([x + ca, y + sa], 1)
        p3 = np.stack([x - sa, y + ca * 0.5], 1)
        verts = np.stack([p0, p1, p2, p0, p2, p3], 1).reshape(-1, 2)
        alpha = max(0.0, min(1.0, (RESULT - self.state_t) / 2))
        cols = np.repeat(np.column_stack([c[:, 5:8], np.full(len(c), alpha)]), 6, axis=0)
        self.host.painter.triangles(verts, cols)

    # --- kreslení ---------------------------------------------------------------------------------
    def render(self) -> None:
        cw, ch = self.host.canvas_size
        rec = self.records.record(self.variant)
        rec_h = 0.0 if rec is None else (-float(rec["score"]) if self.pit else float(rec["score"]))
        peak = (-5.0, -5.0, -5.0)
        xy, ph = None, None
        if self.state == "result" and self.final is not None and self.final_xy is not None:
            xy, ph = self.final_xy, self.final
        elif self.point is not None and self.shown_xy is not None:
            xy, ph = self.shown_xy, self.point[2]
        if xy is not None:
            peak = (xy[0] / cw, xy[1] / ch, ph)
        gold = 0.0
        if self.state == "result" and self.is_record:
            # záblesk na začátku, pak klidnější zlaté vlnění; na konci výsledku dohasne
            gold = (0.35 + 0.65 * max(0.0, 1 - self.state_t / 2.0)) * max(0.0, min(1.0, (RESULT - self.state_t) / 2))
        alarm = 0.0
        if self.state == "round" and ROUND - self.state_t <= 5:
            alarm = 0.6
        self.host.draw_fullscreen(self.prog, u_pit=int(self.pit), u_rec_h=float(rec_h) if rec else -1.0,
                                  u_peak=peak, u_gold=gold, u_alarm=alarm)
        p = self.host.painter
        big, mid, small = int(ch / 9), max(int(ch / 18), 18), max(int(ch / 28), 15)
        name, call, _ = VARIANTS[self.variant]

        # značka na vrcholu / dně (tvary), ohňostroj, pak záře; texty až po ní
        if self.state == "result" and self.final is not None:
            self._marker(self.final_xy, self.final, big, small, gold=self.is_record, label=False)
        elif self.point is not None and self.shown_xy is not None:
            self._marker(self.shown_xy, self.point[2], int(ch / 12), small, gold=False, label=False)
        self._gauge(cw, ch, small, rec)
        if self.state == "result" and self.is_record:
            self._draw_confetti()
        self.fw.draw(p, ch)
        self.host.glow(0.85, 0.8)
        if self.state != "result" and self.point is not None and self.shown_xy is not None:
            self._marker(self.shown_xy, self.point[2], int(ch / 12), small, gold=False, shapes=False)

        # horní lišta: název + časomíra
        bar_h = mid * 1.6
        p.rect(0, 0, cw, bar_h, (0, 0.02, 0.08, 0.72))
        p.rect(0, bar_h - 3, cw, 3, (1, 0.85, 0.2, 0.9))
        p.text(name, small * 0.6, bar_h / 2, size=mid, anchor="ml", bold=True, color=(1, 0.85, 0.2, 1))
        if self.state == "round":
            left = max(0.0, ROUND - self.state_t)
            urgent = left <= 10
            pulse = 1 + (0.12 * math.sin(self.t * 10) if urgent else 0)
            secs = int(math.ceil(left))
            if urgent:  # červená tabulka časomíry
                p.rect(cw / 2 - mid * 1.9, 0, mid * 3.8, bar_h, (0.75, 0.05, 0.05, 0.65 + 0.3 * abs(math.sin(self.t * 5))))
            p.text(f"{secs // 60}:{secs % 60:02d}",
                   cw / 2, bar_h / 2, size=int(mid * 1.25 * pulse), anchor="mc", bold=True,
                   color=(1, 1, 1, 1) if urgent else (1, 1, 1, 1), outline=3)
            k = left / ROUND
            p.rect(0, bar_h - 6, cw * k, 6, (1, 0.25, 0.2, 1) if urgent else (0.3, 0.9, 0.4, 1))
            if left <= 5:
                p.text("Ruce pryč – měřím!", cw / 2, bar_h + mid * 0.9, size=mid, anchor="mc", bold=True,
                       color=(1, 1, 1, 1), outline=3, background=(0.75, 0.1, 0.1, 0.85), padding=mid * 0.25)
        elif self.state == "ready":
            p.text(call, cw / 2, bar_h / 2, size=mid, anchor="mc", bold=True, color=(1, 1, 1, 1))
        rec_txt = f"Rekord: {rec['alt']:,.0f} m n. m.".replace(",", " ") if rec else "Rekord: zatím žádný"
        p.text(rec_txt, cw - small * 0.6, bar_h / 2, size=small, anchor="mr", bold=True, color=(1, 0.85, 0.2, 1))

        self._leaderboard(cw, ch, small)

        if self.state == "ready":
            msg = "Dotkni se písku a kolo začne – máš 30 sekund!"
            p.text(msg, cw / 2, ch - mid * 1.0, size=mid, anchor="mc", bold=True, color=(1, 1, 1, 1),
                   outline=3, background=(0, 0, 0, 0.55), padding=mid * 0.3)
        elif self.state == "result":
            self._render_result(cw, ch, big, mid)

    def _gauge(self, cw: int, ch: int, small: int, rec) -> None:
        """Výšková stupnice u levého okraje (jako ve sportovním přenosu): sloupec teď, rekord, dnešní maximum."""
        p = self.host.painter
        top_y, bot_y = ch * 0.17, ch * 0.86
        x0, w = small * 0.5, max(14.0, ch * 0.022)
        cur = self.gauge_v
        best_today = self.records.today(self.variant)
        vals = [0.08, cur or 0.0]
        if rec:
            vals.append(float(rec["score"]))
        if best_today:
            vals.append(float(best_today[0]["score"]))
        vmax = max(vals) * 1.15
        sc = self.host.settings.map.vertical_scale

        def y_of(v: float) -> float:
            return bot_y - max(0.0, min(1.0, v / vmax)) * (bot_y - top_y)

        p.rect(x0 - 4, top_y - 4, w + 8, bot_y - top_y + 8, (0, 0.02, 0.08, 0.7))
        # ryska po 100 / 200 / 500 m (nadmořská výška), popisky max. 6
        step = next(s for s in (100, 200, 500, 1000, 2000) if vmax * sc / s <= 6)
        ticks = np.arange(step, vmax * sc, step)
        for a in ticks:
            y = y_of(a / sc)
            p.rect(x0 + w, y - 1.5, w * 0.6, 3, (1, 1, 1, 0.9))
            alt = self.host.altitude(-a / sc if self.pit else a / sc)
            p.text(f"{alt:,.0f}".replace(",", " "), x0 + w * 1.8, y, size=small, anchor="ml", bold=True,
                   color=(1, 1, 1, 0.95), outline=3)
        if cur is not None and cur > 0:  # sloupec: zelená -> žlutá -> červená, nahoře svítí
            y = y_of(cur)
            f = min(1.0, cur / vmax)
            c_top = (1.0, max(0.2, 1.0 - f), 0.1, 1) if f > 0.5 else (0.3 + 1.4 * f, 1.0, 0.25, 1)
            verts = np.array([[x0, bot_y], [x0 + w, bot_y], [x0 + w, y], [x0, bot_y], [x0 + w, y], [x0, y]])
            cols = np.array([(0.2, 0.8, 0.3, 1), (0.2, 0.8, 0.3, 1), c_top, (0.2, 0.8, 0.3, 1), c_top, c_top])
            p.triangles(verts, cols)
            p.circle(x0 + w / 2, y, w * 0.9, (c_top[0], c_top[1], c_top[2], 0.5))
            p.triangles(np.array([[x0 + w + 2, y], [x0 + w + w, y - w * 0.6], [x0 + w + w, y + w * 0.6]]), (1, 1, 1, 1))
        if best_today:
            y = y_of(float(best_today[0]["score"]))
            p.rect(x0 - 6, y - 2.5, w + 12, 5, (0.85, 0.9, 1.0, 1))
        if rec:  # rekord: zlatá čára s hvězdou
            y = y_of(float(rec["score"]))
            p.rect(x0 - 8, y - 4, w + 16, 8, (1, 0.8, 0.15, 1))
            star = []
            for k in range(10):
                a = -math.pi / 2 + k * math.pi / 5
                rr = w * (0.95 if k % 2 == 0 else 0.4)
                star.append((x0 + w / 2 + rr * math.cos(a), y + rr * math.sin(a)))
            star = np.array(star)
            c = np.array([x0 + w / 2, y])
            p.triangles(np.stack([np.tile(c, (10, 1)), star, np.roll(star, -1, 0)], 1).reshape(-1, 2), (1, 0.9, 0.3, 1))

    def _marker(self, xy, h: float, size: int, small: int, gold: bool, label: bool = True,
                shapes: bool = True) -> None:
        if xy is None:
            return
        p = self.host.painter
        cw, ch = self.host.canvas_size
        x, y = xy
        col = (1, 0.85, 0.15, 1) if gold else (1, 0.2, 0.15, 1)
        r = ch * 0.03
        if shapes:
            # paprsky světla (vrchol = hvězda přenosu), pulzující kruh
            n, rot = 12, self.t * 0.6
            a = rot + np.arange(n) * 2 * math.pi / n
            r0, r1, hw = r * 1.1, r * (3.2 if gold else 2.4), 0.11
            base = np.stack([x + r0 * np.cos(a), y + r0 * np.sin(a)], 1)
            lft = np.stack([x + r1 * np.cos(a - hw), y + r1 * np.sin(a - hw)], 1)
            rgt = np.stack([x + r1 * np.cos(a + hw), y + r1 * np.sin(a + hw)], 1)
            rc = np.array([(*col[:3], 0.8), (*col[:3], 0.0), (*col[:3], 0.0)], np.float32)
            p.triangles(np.stack([base, lft, rgt], 1).reshape(-1, 2), np.tile(rc, (n, 1)))
            phase = (self.t * 1.2) % 1.0
            p.circle(x, y, r * (1 + 2.5 * phase), (1, 1, 1, 1 - phase), width=4)  # pulzující kruh
            p.circle(x, y, r * 1.05, (0, 0, 0, 0.55))
            s = r * 0.9
            if self.pit:
                tri = [[x, y + s], [x + s, y - s * 0.8], [x - s, y - s * 0.8]]
            else:
                tri = [[x, y - s], [x + s, y + s * 0.8], [x - s, y + s * 0.8]]
            p.triangles(np.array(tri), (0, 0, 0, 1))
            p.triangles(np.array(tri) * 0.75 + np.array([x, y]) * 0.25, col)
            if not self.pit:  # sněhová čepička na trojúhelníku
                cap = np.array(tri) * 0.75 + np.array([x, y]) * 0.25
                apex = cap[0]
                p.triangles(np.array([apex, apex + (cap[1] - apex) * 0.38, apex + (cap[2] - apex) * 0.38]),
                            (1, 1, 1, 1))
            if gold:  # rekord: zlaté kruhy se šíří od vrcholu
                for k in range(3):
                    ph = (self.t * 0.8 + k / 3) % 1.0
                    p.circle(x, y, r * (1.5 + 9 * ph), (1, 0.85, 0.15, 1 - ph), width=max(4, ch / 120))
        if not label:
            return
        # popisek na tu stranu, kde je víc místa
        right = x < cw * 0.6
        ax = x + r * 1.8 if right else x - r * 1.8
        anchor = "l" if right else "r"
        ty = min(max(y, size * 1.5), ch - size * 1.2)
        w, hh = p.text(self._alt(h), ax, ty, size=size, anchor="b" + anchor, bold=True, color=col,
                       outline=max(3, size // 14), outline_color=(0, 0, 0, 1))
        p.text(self._cm(h), ax, ty + size * 0.1, size=small, anchor="t" + anchor, bold=True,
               color=(1, 1, 1, 1), outline=3, outline_color=(0, 0, 0, 1))

    def _leaderboard(self, cw: int, ch: int, small: int) -> None:
        p = self.host.painter
        top = self.records.today(self.variant)
        x = cw - small * 0.6
        y = small * 2.6 + ch * 0.02
        lines = ["Dnešní TOP 5"] + [f"{i + 1}. {e['alt']:,.0f} m · {e['time']}".replace(",", " ")
                                    for i, e in enumerate(top)]
        if not top:
            lines.append("buď první!")
        width = max(p.text_cache.get(s, small, (255, 255, 255, 255), True, 0, (0, 0, 0, 255)).size[0] for s in lines)
        p.rect(x - width - small * 0.5, y - small * 0.4, width + small, small * 1.35 * len(lines) + small * 0.5,
               (0, 0, 0, 0.6))
        for i, s in enumerate(lines):
            hl = self.state == "result" and self.rank is not None and i == self.rank
            col = (1, 0.85, 0.2, 1) if i == 0 or hl else (1, 1, 1, 1)
            p.text(s, x, y + i * small * 1.35, size=small, anchor="tr", bold=True, color=col)

    def _render_result(self, cw: int, ch: int, big: int, mid: int) -> None:
        p = self.host.painter
        if self.final is None:
            p.text("Kolo skončilo", cw / 2, ch / 2, size=big, anchor="mc", bold=True, outline=4)
            return
        # nápis na opačnou polovinu, než je vrchol (ať ho nezakryje)
        py = self.final_xy[1] if self.final_xy else ch / 2
        up, down = big * 0.7, big * 0.95 + mid * 1.7  # rozsah nápisů nad a pod středem cy
        cy = py - ch * 0.07 - down  # raději nad vrchol…
        if cy - up < mid * 1.7:
            cy = py + ch * 0.07 + up  # …jinak pod něj
        cy = min(max(cy, mid * 1.7 + up), ch - down)
        if self.is_record:
            self._draw_confetti()
            pulse = 1 + 0.08 * math.sin(self.state_t * 8)
            hue = (self.state_t * 2) % 1.0
            col = (1, 0.85 + 0.15 * math.sin(hue * 6.28), 0.15, 1)
            p.text("NOVÝ REKORD!", cw / 2, cy, size=int(big * 1.25 * pulse), anchor="mc", bold=True,
                   color=col, outline=max(4, big // 12), outline_color=(0.4, 0.05, 0.0, 1))
        else:
            title = f"{self.rank}. místo dnes!" if self.rank else "Konec kola!"
            p.text(title, cw / 2, cy, size=big, anchor="mc", bold=True, color=(1, 1, 1, 1),
                   outline=max(4, big // 12))
        p.text(f"{VARIANTS[self.variant][2].capitalize()}: {self._alt(self.final)}", cw / 2, cy + big * 0.95,
               size=mid, anchor="mc", bold=True, color=(1, 1, 1, 1), outline=3, background=(0, 0, 0, 0.6),
               padding=mid * 0.25)
        p.text(f"({self._cm(self.final)})", cw / 2, cy + big * 0.95 + mid * 1.25, size=int(mid * 0.7),
               anchor="mc", bold=True, color=(1, 1, 1, 1), outline=3)

    # --- ovládání ---------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name in ("enter", "space"):
            self.start_round()
        elif name == "v":
            self.variant = "propast" if self.variant == "hora" else "hora"
            self.reset()
            self.host.notify(VARIANTS[self.variant][0])
        elif name == "x":
            if self.state == "round":
                self._set_state("ready")
                self.hand_time = 0.0
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Varianta: {VARIANTS[self.variant][0]}, stav: {self.state}"]
        if self.state == "round":
            lines.append(f"Zbývá {max(0.0, ROUND - self.state_t):.0f} s")
        if self.point is not None:
            lines.append(f"Teď: {self._alt(self.point[2])} ({self.point[2] * 100:+.1f} cm)")
        rec = self.records.record(self.variant)
        if rec:
            lines.append(f"Rekord: {rec['alt']:.0f} m n. m. ({rec['date']} {rec['time']})")
        return lines
