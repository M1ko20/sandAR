"""Nejvyšší hora – soutěž, kdo postaví nejvyšší horu (nebo vyhloubí nejhlubší propast).

Kolo trvá 60 s a začne samo, jakmile se někdo dotkne písku (nebo Enter).
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

ROUND = 60.0  # [s] délka kola
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
void main() {
    float h = height_at(v_uv);
    vec3 c = elevation_color(h);
    c *= mix(0.62, 1.12, hillshade(v_uv, 2.5));
    vec2 cl = contours(h);
    c = mix(c, c * 0.35, cl.x * 0.8);
    c = mix(c, vec3(0.06), cl.y);
    f_color = vec4(c, 1.0);
}
"""


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
    order = 92
    category = "soutěž"
    intro = "Máš 60 sekund – postav co nejvyšší horu a překonej dnešní rekord!"
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
        hands = hm.hands is not None and int(hm.hands.sum()) > 12
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

        if self.state == "ready":
            if self.hand_time >= HANDS_TO_START:
                self.start_round()
        elif self.state == "round":
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

    def _finish(self) -> None:
        if not self.recent and self.point is not None:
            self.recent.append((self.t, self.point[2]))
        self._set_state("result")
        if not self.recent:
            self.final = None
            return
        self.final = max((v for _, v in self.recent), key=self.score_of)
        self.final_xy = self.shown_xy
        self.is_record, self.rank = self.records.submit(self.variant, self.score_of(self.final),
                                                        self.host.altitude(self.final))
        if self.is_record:
            self._spawn_confetti()
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
        self.host.draw_fullscreen(self.prog, u_pit=int(self.pit))
        p = self.host.painter
        big, mid, small = int(ch / 9), max(int(ch / 18), 18), max(int(ch / 28), 15)
        name, call, _ = VARIANTS[self.variant]

        # značka na vrcholu / dně
        if self.state == "result" and self.final is not None:
            self._marker(self.final_xy, self.final, big, small, gold=self.is_record, label=False)
        elif self.point is not None and self.shown_xy is not None:
            self._marker(self.shown_xy, self.point[2], int(ch / 12), small, gold=False)

        # horní lišta: název + časomíra
        bar_h = mid * 1.6
        p.rect(0, 0, cw, bar_h, (0, 0, 0, 0.6))
        p.text(name, small * 0.6, bar_h / 2, size=mid, anchor="ml", bold=True, color=(1, 0.85, 0.2, 1))
        if self.state == "round":
            left = max(0.0, ROUND - self.state_t)
            urgent = left <= 10
            pulse = 1 + (0.12 * math.sin(self.t * 10) if urgent else 0)
            secs = int(math.ceil(left))
            p.text(f"{secs // 60}:{secs % 60:02d}",
                   cw / 2, bar_h / 2, size=int(mid * 1.25 * pulse), anchor="mc", bold=True,
                   color=(1, 0.25, 0.2, 1) if urgent else (1, 1, 1, 1), outline=3)
            k = left / ROUND
            p.rect(0, bar_h - 6, cw * k, 6, (1, 0.25, 0.2, 1) if urgent else (0.3, 0.9, 0.4, 1))
            if left <= 5:
                p.text("Ruce pryč – měřím!", cw / 2, bar_h + mid * 0.9, size=mid, anchor="mc", bold=True,
                       color=(1, 1, 1, 1), outline=3, background=(0.75, 0.1, 0.1, 0.85), padding=mid * 0.25)
        elif self.state == "ready":
            p.text(call, cw / 2, bar_h / 2, size=mid, anchor="mc", bold=True, color=(1, 1, 1, 1))
        rec = self.records.record(self.variant)
        rec_txt = f"Rekord: {rec['alt']:,.0f} m n. m.".replace(",", " ") if rec else "Rekord: zatím žádný"
        p.text(rec_txt, cw - small * 0.6, bar_h / 2, size=small, anchor="mr", bold=True, color=(1, 0.85, 0.2, 1))

        self._leaderboard(cw, ch, small)

        if self.state == "ready":
            msg = "Dotkni se písku a kolo začne – máš 60 sekund!"
            p.text(msg, cw / 2, ch - mid * 1.0, size=mid, anchor="mc", bold=True, color=(1, 1, 1, 1),
                   outline=3, background=(0, 0, 0, 0.55), padding=mid * 0.3)
        elif self.state == "result":
            self._render_result(cw, ch, big, mid)

    def _marker(self, xy, h: float, size: int, small: int, gold: bool, label: bool = True) -> None:
        if xy is None:
            return
        p = self.host.painter
        cw, ch = self.host.canvas_size
        x, y = xy
        col = (1, 0.85, 0.15, 1) if gold else (1, 0.2, 0.15, 1)
        r = ch * 0.03
        phase = (self.t * 1.2) % 1.0
        p.circle(x, y, r * (1 + 2.5 * phase), (1, 1, 1, 1 - phase), width=4)  # pulzující kruh
        s = r * 0.9
        if self.pit:
            tri = [[x, y + s], [x + s, y - s * 0.8], [x - s, y - s * 0.8]]
        else:
            tri = [[x, y - s], [x + s, y + s * 0.8], [x - s, y + s * 0.8]]
        p.triangles(np.array(tri), (0, 0, 0, 1))
        p.triangles(np.array(tri) * 0.75 + np.array([x, y]) * 0.25, col)
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
