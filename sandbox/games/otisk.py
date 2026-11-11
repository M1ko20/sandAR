"""Digitální otisk – fotka postavené krajiny jako mapa do mobilu.

Návštěvník postaví krajinu. Když ji chvíli nikdo nemění a ruce jsou pryč
(nebo operátor stiskne F), proběhne odpočet 3-2-1, „cvak“ a program v Pythonu
(numpy + OpenCV + Pillow, bez GPU) vykreslí pěknou mapu: barvy nadmořských výšek,
stínování, vrstevnice, vymyšlené jméno krajiny, nejvyšší bod, datum a rámeček.
Mapa se uloží do ``shares/`` a na písek se promítne velký QR kód – po naskenování
si ji návštěvník stáhne do telefonu (musí být na stejné Wi-Fi, viz ``sandbox/share.py``).
"""

from __future__ import annotations

import datetime as _dt
import logging
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import share
from ..render.text import get_font
from .base import Game
from .mapa import find_peaks

log = logging.getLogger(__name__)

FRAG = """
uniform float u_dim;     // 1 = plný jas, menší = ztlumit (pod QR kódem)
void main() {
    float h = height_at(v_uv);
    vec3 c = elevation_color(h);
    c *= mix(0.62, 1.12, hillshade(v_uv, 2.5));
    vec2 cl = contours(h);
    float under = h < u_water_level ? 0.45 : 1.0;
    c = mix(c, c * 0.35, cl.x * 0.85 * under);
    c = mix(c, vec3(0.06), cl.y * under);
    f_color = vec4(c * u_dim, 1.0);
}
"""

# --- jméno krajiny ---------------------------------------------------------------------
# "Pohoří" + "Zlatého Draka" – druhá část je ve 2. pádě, takže sedí ke každému rodu.
FORMS = {
    "hory": ["Pohoří", "Velehory", "Masiv", "Hřbet", "Štíty"],
    "kopce": ["Vrchovina", "Pahorkatina", "Kopce", "Vrchy", "Hřebeny"],
    "voda": ["Souostroví", "Ostrovy", "Zátoka", "Jezerní kraj", "Pobřeží"],
    "rovina": ["Nížina", "Pláně", "Údolí", "Kraj", "Roviny"],
    "propast": ["Kaňon", "Rokle", "Propasti", "Soutěsky"],
}
OF_WHAT = [
    "Zlatého Draka", "Ztraceného Krále", "Tančících Medvědů", "Moudré Sovy", "Stříbrného Vlka",
    "Spícího Obra", "Divokých Koz", "Zpívajících Skal", "Bludných Kamenů", "Ohnivého Lišáka",
    "Modrého Měsíce", "Tří Čarodějnic", "Věčného Sněhu", "Bručících Jezevců", "Šťastných Žab",
    "Statečné Princezny", "Kouzelných Sušenek", "Rychlých Šneků", "Zapomenutých Pirátů", "Polední Mlhy",
    "Létajících Koloběžek", "Hladového Trolla", "Smějících se Veverek", "Křišťálové Jeskyně",
    "Ospalých Netopýrů", "Duhového Jednorožce", "Starého Vodníka", "Tajemné Mapy",
]

# barevná škála souše – stejná jako v glsl.LIB (land_color)
LAND_T = np.array([0.0, 0.18, 0.38, 0.55, 0.72, 0.88, 1.0], np.float32)
LAND_C = np.array([(0.16, 0.50, 0.24), (0.42, 0.70, 0.32), (0.80, 0.84, 0.45), (0.95, 0.80, 0.42),
                   (0.80, 0.52, 0.28), (0.58, 0.36, 0.24), (0.97, 0.97, 0.97)], np.float32)

COUNTDOWN = 3.0  # [s] 3-2-1
FLASH = 0.5  # [s] bílý záblesk
SHOW = 25.0  # [s] jak dlouho svítí QR kód
STABLE_AUTO = 8.0  # [s] klidu po změně terénu -> automatická fotka
HANDS_GONE = 1.5  # [s] bez rukou
STABLE_FORCE = 15.0  # [s] klidu i s „rukou“ (třeba hračka položená na písku)
FOOTER = "AR pískoviště · Den otevřených dveří"


def landscape_name(heights: np.ndarray, water_level: float, rng: random.Random | None = None) -> str:
    """Vymyslí jméno krajiny podle jejího charakteru (hory, voda, rovina…)."""
    rng = rng or random.Random()
    h = heights[np.isfinite(heights)]
    if h.size == 0:
        return "Neznámá Země"
    relief = float(np.percentile(h, 99.5))
    low = float(np.percentile(h, 0.5))
    water = float(np.mean(h < water_level))
    if water > 0.35:
        kind = "voda"
    elif relief > 0.07:
        kind = "hory"
    elif water_level - low > 0.04 and relief < 0.04:
        kind = "propast"
    elif relief > 0.03:
        kind = "kopce"
    else:
        kind = "rovina"
    return f"{rng.choice(FORMS[kind])} {rng.choice(OF_WHAT)}"


def _land_color(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return np.stack([np.interp(t, LAND_T, LAND_C[:, k]) for k in range(3)], axis=-1)


def _fmt_alt(meters: float) -> str:
    return f"{meters:,.0f}".replace(",", " ")  # 1 024 – úzká mezera mezi tisíci


def render_map_image(heights: np.ndarray, box_size: tuple[float, float], map_settings, name: str,
                     when: _dt.datetime | None = None, width: int = 1800, footer: str = FOOTER) -> np.ndarray:
    """Stylizovaná mapa terénu jako RGB obrázek (uint8). Čistě CPU, ~0.2 s."""
    m = map_settings
    when = when or _dt.datetime.now()
    hts = np.nan_to_num(np.asarray(heights, np.float32))
    gh, gw = hts.shape
    pad = int(width * 0.035)  # rámeček
    head = int(width * 0.085)  # pás s názvem
    foot = int(width * 0.07)  # pás s údaji
    mw = width - 2 * pad
    mh = int(round(mw * box_size[1] / box_size[0]))
    total_h = pad + head + mh + foot + pad

    # --- terén: barvy a stínování v menším rozlišení (jsou hladké), vrstevnice v plném ---
    smooth = cv2.GaussianBlur(hts, (0, 0), 0.8)
    big = cv2.resize(smooth, (mw, mh), interpolation=cv2.INTER_CUBIC)
    sw, sh = max(8, mw // 3), max(8, mh // 3)
    low = cv2.resize(smooth, (sw, sh), interpolation=cv2.INTER_CUBIC)
    lwater = low < m.water_level
    rgb = _land_color((low - m.water_level) / max(m.elevation_max - m.water_level, 1e-4)).astype(np.float32)
    depth = np.clip((m.water_level - low) / max(m.water_level - m.elevation_min, 1e-4), 0, 1)[..., None]
    wcol = (1 - depth) * np.float32([0.35, 0.68, 0.98]) + depth * np.float32([0.03, 0.14, 0.48])
    rgb = np.where(lwater[..., None], wcol, rgb).astype(np.float32)
    # stínování (světlo od severozápadu), sklon počítaný v metrech
    gy, gx = np.gradient(low)
    ex = 2.5
    nx, ny = -gx / (box_size[0] / sw) * ex, -gy / (box_size[1] / sh) * ex
    light = np.float32([-1.0, -1.2, 1.6]) / np.float32(np.linalg.norm([-1.0, -1.2, 1.6]))
    shade = np.clip((nx * light[0] + ny * light[1] + light[2]) / np.sqrt(nx * nx + ny * ny + 1), 0, 1)
    k = np.where(lwater, 0.92 + 0.08 * shade, 0.62 + 0.5 * shade).astype(np.float32)[..., None]
    rgb = cv2.resize(np.clip(rgb * k, 0, 1).astype(np.float32), (mw, mh), interpolation=cv2.INTER_LINEAR)
    water = big < m.water_level

    # vrstevnice – hladké čáry z OpenCV (antialiasing), hlavní silnější
    lines_minor = np.zeros((mh, mw), np.uint8)
    lines_major = np.zeros((mh, mw), np.uint8)
    iv = m.contour_interval
    lo = int(np.floor(big.min() / iv - 0.5)) - 1
    hi = int(np.ceil(big.max() / iv - 0.5)) + 1
    for i in range(lo, hi + 1):
        level = (i + 0.5) * iv
        if level <= m.water_level:
            continue
        mask = (big > level).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        major = m.major_every > 1 and i % m.major_every == 0
        cv2.polylines(lines_major if major else lines_minor, cs, True, 255,
                      thickness=max(2, width // 600) if major else max(1, width // 1200), lineType=cv2.LINE_AA)
    a = lines_minor.astype(np.float32) * (0.55 * 0.65 / 255)
    rgb *= (1 - a)[..., None]
    b = (lines_major.astype(np.float32) * (0.85 / 255))[..., None]
    rgb = rgb * (1 - b) + np.float32([0.20, 0.12, 0.06]) * b
    # pobřeží
    shore = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    rgb[shore] = rgb[shore] * 0.3 + np.float32([0.05, 0.15, 0.45]) * 0.7

    map_img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # --- papír, rámeček a texty (Pillow) ---
    paper = (246, 240, 222)
    ink = (52, 36, 22)
    img = Image.new("RGB", (width, total_h), (34, 58, 74))
    d = ImageDraw.Draw(img)
    inner = pad // 3
    d.rectangle([inner, inner, width - inner - 1, total_h - inner - 1], fill=paper)
    d.rectangle([inner + 6, inner + 6, width - inner - 7, total_h - inner - 7], outline=(178, 140, 70), width=3)
    my = pad + head
    img.paste(Image.fromarray(map_img), (pad, my))
    d.rectangle([pad - 3, my - 3, pad + mw + 2, my + mh + 2], outline=ink, width=3)

    # nadpis
    title_font = get_font(int(head * 0.55), True)
    d.text((width // 2, pad + head * 0.42), name, font=title_font, fill=ink, anchor="mm")
    small = get_font(int(head * 0.2), False)
    d.text((width // 2, pad + head * 0.84), "mapa krajiny postavené v písku", font=small, fill=(120, 96, 60),
           anchor="mm")

    # vrcholy s kótami
    # jen opravdové kopce: výrazně nad okolní rovinou (ne okraj jámy)
    floor = max(m.water_level + iv, float(np.median(smooth)) + 2 * iv)
    peaks = find_peaks(smooth, floor, 1.5 * iv, radius=max(4, gw // 16), limit=5)
    label_font = get_font(max(14, width // 70), True)
    sx, sy = mw / gw, mh / gh
    top_alt = m.base_altitude + float(smooth.max()) * m.vertical_scale
    for j, (iy, ix, h) in enumerate(peaks):
        x, y = pad + (ix + 0.5) * sx, my + (iy + 0.5) * sy
        s = width / 110 * (1.3 if j == 0 else 1.0)
        d.polygon([(x, y - s), (x + s, y + s * 0.7), (x - s, y + s * 0.7)], fill=(200, 30, 30) if j == 0 else ink,
                  outline=(255, 255, 255))
        alt = m.base_altitude + h * m.vertical_scale
        right = x < pad + mw - width * 0.09  # u pravého okraje popisek vlevo od značky
        d.text((x + s * 1.3 if right else x - s * 1.3, y), f"{_fmt_alt(alt)} m", font=label_font, fill=ink,
               anchor="lm" if right else "rm", stroke_width=3, stroke_fill=paper)

    # severka
    cx, cy, r = pad + mw - width * 0.035, my + width * 0.045, width * 0.022
    d.polygon([(cx, cy - r), (cx + r * 0.45, cy + r * 0.6), (cx, cy + r * 0.25), (cx - r * 0.45, cy + r * 0.6)],
              fill=ink, outline=paper)
    d.text((cx, cy - r * 1.05), "S", font=get_font(int(r * 0.9), True), fill=ink, anchor="md",
           stroke_width=3, stroke_fill=paper)

    # měřítko: vodorovně stejné jako svisle (1 cm písku = vertical_scale cm ve skutečnosti)
    real_w_km = box_size[0] * m.vertical_scale / 1000
    nice = next((v for v in (0.5, 1, 2, 5, 10, 20, 50) if v >= real_w_km / 6), 1)
    bar = mw * nice / real_w_km
    bx, by = pad + width * 0.02, my + mh - width * 0.03
    seg = bar / 2
    for q in range(2):
        d.rectangle([bx + q * seg, by, bx + (q + 1) * seg, by + width * 0.008],
                    fill=ink if q == 0 else paper, outline=ink, width=2)
    d.text((bx + bar + 10, by + width * 0.004), f"{nice:g} km", font=label_font, fill=ink, anchor="lm",
           stroke_width=3, stroke_fill=paper)

    # zápatí
    fy = my + mh + foot * 0.52
    info_font = get_font(int(foot * 0.34), True)
    d.text((pad, fy), f"Nejvyšší bod: {_fmt_alt(top_alt)} m n. m.", font=info_font, fill=ink, anchor="lm")
    d.text((width - pad, fy), f"{when.day}. {when.month}. {when.year}", font=info_font, fill=ink, anchor="rm")
    d.text((width // 2, fy), footer, font=get_font(int(foot * 0.28), False), fill=(120, 96, 60), anchor="mm")
    return np.asarray(img)


def flattest_spot(heights: np.ndarray, hands: np.ndarray | None, win: tuple[int, int],
                  border: int = 2) -> tuple[int, int]:
    """Střed (iy, ix) okna win=(výška, šířka) v buňkách, kde je terén nejrovnější.

    Rovinatost = rozptyl výšek + sklon v okně; mírně preferujeme střed bedny a vyhýbáme se rukám.
    """
    gh, gw = heights.shape
    wh, ww = min(win[0], gh), min(win[1], gw)
    h = heights.astype(np.float32)
    gy, gx = np.gradient(h)
    rough = cv2.blur(gx * gx + gy * gy, (ww, wh)) * 50
    mean = cv2.blur(h, (ww, wh))
    var = np.maximum(cv2.blur(h * h, (ww, wh)) - mean * mean, 0)
    cost = var + rough
    if hands is not None and hands.any():
        cost = cost + cv2.blur(hands.astype(np.float32), (ww, wh)) * 1.0
    yy, xx = np.mgrid[0:gh, 0:gw]
    center = ((yy - gh / 2) / gh) ** 2 + ((xx - gw / 2) / gw) ** 2
    cost = cost + center * 2e-5
    # střed okna musí ležet tak, aby se celé vešlo
    y0, y1 = wh // 2 + border, gh - (wh - wh // 2) - border
    x0, x1 = ww // 2 + border, gw - (ww - ww // 2) - border
    if y1 < y0 or x1 < x0:
        return gh // 2, gw // 2
    sub = cost[y0:y1 + 1, x0:x1 + 1]
    iy, ix = np.unravel_index(int(np.argmin(sub)), sub.shape)
    return int(iy + y0), int(ix + x0)


class FootprintGame(Game):
    id = "otisk"
    title = "Digitální otisk"
    description = "Postav krajinu, program ji vyfotí jako mapu a přes QR kód si ji stáhneš do mobilu."
    order = 90
    category = "zážitek"
    intro = "Postav krajinu, dej ruce pryč a nech si ji vyfotit – mapu si stáhneš do mobilu přes QR kód."
    keys = [("F / P", "vyfotit hned (odpočet 3-2-1)"), ("X", "skrýt QR kód"), ("A", "automatické focení zap/vyp")]

    def start(self) -> None:
        self.prog = self.host.program(FRAG)
        self.qr_tex = None
        self.auto = getattr(self, "auto", True)
        self.server_ok = share.ensure_server() is not None
        self.ip = share.local_ip()
        self.reset()

    def stop(self) -> None:
        self.prog.release()
        self._drop_qr()

    def reset(self) -> None:
        self.t = 0.0
        self.state = "idle"  # idle | countdown | flash | show
        self.state_t = 0.0
        self.dirty = True  # krajina ještě není vyfocená
        self.last_change = 0.0
        self.last_hand = -1e9
        self._ref: np.ndarray | None = None
        self._next_check = 0.0
        self.share_id: str | None = None
        self.url = ""
        self.name = ""
        self.top_alt = 0.0
        self.card = (0.0, 0.0, 0.0, 0.0)
        self.error = ""
        self.shots = getattr(self, "shots", 0)
        self._drop_qr()

    def _drop_qr(self) -> None:
        if getattr(self, "qr_tex", None) is not None:
            self.qr_tex.release()
        self.qr_tex = None

    # --- logika --------------------------------------------------------------------------
    def update(self, hm, dt: float) -> None:
        self.t += dt
        self.state_t += dt
        if hm.hands is not None and int(hm.hands.sum()) > 12:
            self.last_hand = self.t
        if self.t >= self._next_check:  # změny terénu stačí hlídat 4x za sekundu
            self._next_check = self.t + 0.25
            self._watch_terrain(hm)

        if self.state == "idle":
            calm = self.t - self.last_change
            if self.auto and self.dirty and (
                    (calm >= STABLE_AUTO and self.t - self.last_hand >= HANDS_GONE) or calm >= STABLE_FORCE):
                self.trigger()
        elif self.state == "countdown" and self.state_t >= COUNTDOWN:
            self._capture(hm)
            self._set_state("flash")
        elif self.state == "flash" and self.state_t >= FLASH:
            self._set_state("show" if self.share_id else "idle")
        elif self.state == "show" and self.state_t >= SHOW:
            self._set_state("idle")

    def _set_state(self, state: str) -> None:
        self.state, self.state_t = state, 0.0

    def _watch_terrain(self, hm) -> None:
        cur = hm.heights[::4, ::4]
        if self._ref is None or self._ref.shape != cur.shape:
            self._ref = cur.copy()
            return
        diff = np.abs(cur - self._ref) > 0.008
        if hm.hands is not None:
            hands = cv2.dilate(hm.hands.astype(np.uint8), np.ones((9, 9), np.uint8))[::4, ::4] > 0
            diff &= ~hands
        if diff.mean() > 0.003:
            self._ref = cur.copy()
            self.last_change = self.t
            self.dirty = True

    def trigger(self) -> None:
        """Spustí odpočet 3-2-1 a fotku."""
        if self.state in ("countdown", "flash"):
            return
        self._set_state("countdown")

    def _capture(self, hm) -> None:
        t0 = time.perf_counter()
        m = self.host.settings.map
        heights = hm.heights.copy()
        self.name = landscape_name(heights, m.water_level)
        self.top_alt = self.host.altitude(float(cv2.GaussianBlur(heights, (0, 0), 0.8).max()))
        self.dirty = False
        self._ref = hm.heights[::4, ::4].copy()
        try:
            img = render_map_image(heights, hm.size, m, self.name)
            self.share_id = share.save_share(
                img, self.name, f"Nejvyšší bod {_fmt_alt(self.top_alt)} m n. m. · "
                f"{_dt.date.today().strftime('%d. %m. %Y')}")
            self.url = share.url_for(self.share_id)
            self.shots += 1
            self.error = ""
        except Exception as exc:  # fotka nesmí shodit hru
            log.exception("Otisk se nepodařil")
            self.share_id, self.error = None, str(exc)
            self.host.notify("Fotka se nepodařila")
            return
        self._layout_card(hm)
        log.info("Otisk %s (%s) uložen za %.0f ms: %s", self.share_id, self.name,
                 (time.perf_counter() - t0) * 1000, self.url)

    def _layout_card(self, hm) -> None:
        """Rozměr a poloha bílé karty s QR kódem: velká, na nejrovnějším místě písku."""
        cw, ch = self.host.canvas_size
        qr_px = int(ch * 0.50)
        self._drop_qr()
        rgba = share.qr_image(self.url, qr_px)
        self.qr_tex = self.host.ctx.texture((qr_px, qr_px), 4, np.ascontiguousarray(rgba).tobytes())
        self.qr_tex.filter = (self.host.ctx.NEAREST, self.host.ctx.NEAREST)
        self.qr_px = qr_px
        self.big = max(int(ch / 22), 18)
        self.small = max(int(ch / 30), 14)
        text_w = max(get_font(self.big, True).getlength(self.name),
                     get_font(self.small, True).getlength("Naskenuj a stáhni si svou krajinu"))
        card_w = min(cw - 8, max(qr_px, text_w) + 2 * self.small)
        card_h = qr_px + self.big * 1.6 + self.small * 3.4 + self.small
        gh, gw = hm.shape
        win = (int(np.ceil(card_h / ch * gh)), int(np.ceil(card_w / cw * gw)))
        iy, ix = flattest_spot(hm.heights, hm.hands, win)
        x, y = (ix + 0.5) / gw * cw - card_w / 2, (iy + 0.5) / gh * ch - card_h / 2
        x = min(max(x, 4), cw - card_w - 4)
        y = min(max(y, 4), ch - card_h - 4)
        self.card = (x, y, card_w, card_h)

    # --- kreslení ---------------------------------------------------------------------------
    def render(self) -> None:
        cw, ch = self.host.canvas_size
        p = self.host.painter
        dim = 0.45 if self.state == "show" else 1.0
        self.host.draw_fullscreen(self.prog, u_dim=dim)
        if self.state == "idle":
            self._render_idle(cw, ch)
        elif self.state == "countdown":
            n = int(np.ceil(COUNTDOWN - self.state_t))
            frac = (COUNTDOWN - self.state_t) % 1.0
            size = int(ch * (0.35 + 0.15 * frac))
            p.text(str(max(n, 1)), cw / 2, ch / 2, size=size, anchor="mc", bold=True,
                   color=(1, 1, 1, 1), outline=max(4, size // 18), outline_color=(0.05, 0.05, 0.1, 1))
            p.text("Pozor, fotím! Ruce pryč z písku", cw / 2, ch * 0.08, size=max(int(ch / 22), 18), anchor="mc",
                   bold=True, color=(1, 0.9, 0.3, 1), outline=3)
        elif self.state == "flash":
            a = max(0.0, 1.0 - self.state_t / FLASH)
            p.rect(0, 0, cw, ch, (1, 1, 1, a))
            self._camera_icon(cw / 2, ch / 2, ch * 0.18, (0.1, 0.1, 0.1, a))
        elif self.state == "show":
            self._render_card(cw, ch)

    def _render_idle(self, cw: int, ch: int) -> None:
        p = self.host.painter
        size = max(int(ch / 26), 16)
        calm = self.t - self.last_change
        if self.dirty and self.auto:
            msg = "Postav krajinu a dej ruce pryč – vyfotím ji!"
        else:
            msg = "Postav novou krajinu – vyfotím ji do mobilu!"
        p.text(msg, cw / 2, ch - size * 1.2, size=size, anchor="mc", bold=True, color=(1, 1, 1, 1),
               outline=3, background=(0, 0, 0, 0.55), padding=size * 0.35)
        # kroužek, jak dlouho už je klid (ukazuje, že se blíží fotka)
        if self.dirty and self.auto and calm > 1.0 and self.t - self.last_hand >= 0.5:
            k = min(1.0, calm / STABLE_AUTO)
            cx, cy, r = cw - size * 2.2, ch - size * 1.2, size * 0.9
            self._camera_icon(cx, cy, r * 0.9, (1, 1, 1, 0.9))
            ang = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi * k, max(2, int(60 * k)))
            p.polyline(np.stack([cx + r * 1.25 * np.cos(ang), cy + r * 1.25 * np.sin(ang)], 1),
                       (1, 0.85, 0.2, 1), width=max(3, size / 6))

    def _camera_icon(self, cx: float, cy: float, s: float, color) -> None:
        p = self.host.painter
        p.rect(cx - s, cy - s * 0.6, 2 * s, s * 1.3, color)
        p.rect(cx - s * 0.4, cy - s * 0.85, s * 0.8, s * 0.3, color)
        bg = (1, 1, 1, color[3]) if sum(color[:3]) < 1.5 else (0, 0, 0, color[3])
        p.circle(cx, cy + s * 0.05, s * 0.45, bg)
        p.circle(cx, cy + s * 0.05, s * 0.3, color)

    def _render_card(self, cw: int, ch: int) -> None:
        p = self.host.painter
        x, y, w, h = self.card
        p.rect(x - 6, y - 6, w + 12, h + 12, (0.95, 0.75, 0.15, 1))  # zlatý okraj karty
        p.rect(x, y, w, h, (1, 1, 1, 1))
        cx = x + w / 2
        ty = y + self.small * 0.4
        p.text(self.name, cx, ty, size=self.big, anchor="tc", bold=True, color=(0.1, 0.1, 0.15, 1))
        qy = ty + self.big * 1.45
        if self.qr_tex is not None:
            p.image(self.qr_tex, cx - self.qr_px / 2, qy, self.qr_px, self.qr_px)
        ly = qy + self.qr_px + self.small * 0.2
        p.text("Naskenuj a stáhni si svou krajinu", cx, ly, size=self.small, anchor="tc", bold=True,
               color=(0.1, 0.1, 0.15, 1))
        p.text(self.url.removeprefix("http://"), cx, ly + self.small * 1.35, size=self.small, anchor="tc",
               color=(0.3, 0.3, 0.35, 1))
        # ubývající proužek – za jak dlouho QR zmizí
        k = max(0.0, 1.0 - self.state_t / SHOW)
        p.rect(x, y + h - self.small * 0.45, w * k, self.small * 0.45, (0.95, 0.6, 0.1, 1))

    # --- ovládání ------------------------------------------------------------------------------
    def key(self, name: str, shift: bool = False) -> bool:
        if name in ("f", "p"):  # F se simulací zarovná písek – proto i P
            self.trigger()
        elif name == "x":
            if self.state == "show":
                self._set_state("idle")
        elif name == "a":
            self.auto = not self.auto
            self.host.notify("Automatické focení " + ("zapnuto" if self.auto else "vypnuto"))
        else:
            return False
        return True

    def status(self) -> list[str]:
        lines = [f"Stav: {self.state}" + (f" ({self.t - self.last_change:.0f} s klid)" if self.state == "idle" else ""),
                 f"Fotek: {self.shots}, auto {'zap' if self.auto else 'vyp'}"]
        if self.share_id:
            lines.append(f"Poslední: {self.name} – {self.url}")
        if not self.server_ok:
            lines.append("Server pro sdílení neběží!")
        elif self.ip.startswith("127."):
            lines.append("Pozor: počítač není v síti, QR kód v mobilu nepůjde otevřít")
        if self.error:
            lines.append(f"Chyba: {self.error}")
        return lines
