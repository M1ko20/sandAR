"""Živá krajina a povodně – logika bez kreslení (čisté numpy, sdílené módy „ekosystem“ a „mesto“).

* biomy podle výšky a sklonu (voda, pláž, louka, les, skály, sníh),
* hydrologie na zmenšené mřížce: vyplnění bezodtokých sníženin (= jezera),
  vícesměrná akumulace toku MFD (= řeky stékající z hor do údolí), cesta vody D8,
* vegetace: stromy vyrůstají, kde se jim daří, a hynou, když se terén pod nimi změní,
* zvířata: srnky se pasou na loukách a vyhýbají se vodě, ptáci krouží nad krajinou,
* město: zdroj povodně na vyvýšeném okraji a město v nížině po proudu.

Souřadnice: mřížka ``heights[iy, ix]``, pozice v metrech (x podél šířky bedny, y podél hloubky).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np

WATER, BEACH, MEADOW, FOREST, ROCK, SNOW = range(6)
BIOME_NAMES = ["voda", "pláž", "louka", "les", "skály", "sníh"]

# 8 sousedů (dy, dx) a jejich vzdálenost v buňkách
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_NB_DIST = [math.hypot(dy, dx) for dy, dx in _NB]


# --- biomy -----------------------------------------------------------------------------
@dataclass
class BiomeParams:
    water_level: float = -0.02  # [m] hladina moře (pod ní je voda)
    beach: float = 0.006  # [m] pás pláže nad hladinou
    forest_height: float = 0.014  # [m] od této výšky roste les
    forest_slope: float = 0.14  # [-] od tohoto sklonu roste les i níž
    treeline: float = 0.055  # [m] horní hranice lesa, výš jsou skály
    snow: float = 0.072  # [m] sněžná čára
    rock_slope: float = 0.55  # [-] strmé svahy jsou skalnaté


def slope_map(heights: np.ndarray, cell: tuple[float, float]) -> np.ndarray:
    """Sklon terénu (|grad h|, bezrozměrný) – cell = (šířka, hloubka) buňky v metrech."""
    gy, gx = np.gradient(heights.astype(np.float32), cell[1], cell[0])
    return np.hypot(gx, gy)


def classify_biomes(heights: np.ndarray, slope: np.ndarray, p: BiomeParams) -> np.ndarray:
    """Biom každé buňky (uint8, konstanty WATER..SNOW)."""
    b = np.full(heights.shape, MEADOW, np.uint8)
    b[(heights >= p.forest_height) | (slope >= p.forest_slope)] = FOREST
    b[(heights >= p.treeline) | (slope >= p.rock_slope)] = ROCK
    b[heights >= p.snow] = SNOW
    b[heights < p.water_level + p.beach] = BEACH
    b[heights < p.water_level] = WATER
    return b


# --- hydrologie ----------------------------------------------------------------------------
def downsample(heights: np.ndarray, factor: int) -> np.ndarray:
    """Zmenší mřížku průměrováním bloků factor×factor."""
    if factor <= 1:
        return heights.astype(np.float32)
    gh, gw = heights.shape
    return cv2.resize(heights.astype(np.float32), (max(2, gw // factor), max(2, gh // factor)),
                      interpolation=cv2.INTER_AREA)


def priority_flood(h: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Vyplní bezodtoké sníženiny (Barnes 2014, priority flood + epsilon).

    Okraje bedny jsou odtoky. Výsledek je všude >= h a z každé buňky vede
    cesta ostře klesající k okraji – v sníženinách vznikne hladina jezera.
    """
    gh, gw = h.shape
    filled = h.astype(np.float64).ravel().tolist()
    done = [False] * (gh * gw)
    heap: list[tuple[float, int]] = []
    for y in range(gh):
        for x in (range(gw) if y in (0, gh - 1) else (0, gw - 1)):
            i = y * gw + x
            done[i] = True
            heap.append((filled[i], i))
    heapq.heapify(heap)
    offs = [(dy, dx, dy * gw + dx) for dy, dx in _NB]
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        z, i = pop(heap)
        y, x = divmod(i, gw)
        for dy, dx, d in offs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < gh and 0 <= nx < gw:
                j = i + d
                if not done[j]:
                    done[j] = True
                    nz = filled[j]
                    if nz <= z + eps:
                        nz = z + eps
                        filled[j] = nz
                    push(heap, (nz, j))
    return np.asarray(filled, np.float64).reshape(gh, gw)


def d8_receivers(filled: np.ndarray) -> np.ndarray:
    """Index (ploše) souseda s nejprudším spádem; −1 = odtok (okraj nebo bez spádu)."""
    gh, gw = filled.shape
    pad = np.pad(filled, 1, constant_values=np.inf)
    best = np.zeros((gh, gw))
    idx = np.full((gh, gw), -1, np.int64)
    iy, ix = np.mgrid[0:gh, 0:gw]
    for (dy, dx), dist in zip(_NB, _NB_DIST):
        nb = pad[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw]
        drop = (filled - nb) / dist
        better = drop > best
        best = np.where(better, drop, best)
        idx = np.where(better, (iy + dy) * gw + (ix + dx), idx)
    border = np.zeros((gh, gw), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    idx[border] = -1
    return idx.ravel()


def trace_downhill(receivers: np.ndarray, start: int, limit: int = 10000) -> list[int]:
    """Cesta vody z buňky start až k odtoku (ploché indexy)."""
    path = [start]
    i = start
    for _ in range(limit):
        i = int(receivers[i])
        if i < 0:
            break
        path.append(i)
    return path


@dataclass
class Hydrology:
    coarse: np.ndarray  # zmenšené výšky
    filled: np.ndarray  # výšky s vyplněnými jezery
    receivers: np.ndarray  # D8 (plochý index, −1 odtok)
    acc: np.ndarray  # akumulace toku
    lake: np.ndarray  # hloubka jezera [m] (0 = není jezero)
    river: np.ndarray  # síla řeky 0..1 (0 = bez řeky)
    flow_dir: np.ndarray  # (2, gh, gw) jednotkový směr toku (x, y)
    factor: int

    def texture_data(self, water_level: float) -> np.ndarray:
        """RGBA pro shader: r = řeka, g/b = směr toku, a = hladina jezera (jinde pod terénem)."""
        lake = self.lake > 0
        near = cv2.dilate(lake.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        level = np.where(near, self.filled, self.coarse - 0.03)
        level = np.where(self.coarse < water_level, self.coarse - 0.03, level)  # moře řeší shader sám
        return np.dstack([self.river, self.flow_dir[0], self.flow_dir[1], level]).astype(np.float32)

    def wet_mask(self, shape: tuple[int, int], river_min: float = 0.3) -> np.ndarray:
        """Maska (v plném rozlišení), kde je řeka nebo jezero – tam stromy nerostou a zvířata nechodí."""
        wet = ((self.river >= river_min) | (self.lake > 0)).astype(np.uint8)
        return cv2.resize(wet, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def mfd_fractions(filled: np.ndarray, exponent: float = 4.0) -> np.ndarray:
    """Vícesměrný odtok (MFD, Quinn/Freeman): podíl vody do každého z 8 sousedů, (8, gh, gw).

    Na vypuklém svahu se voda rozlévá, v údolí se slévá – řeky tak vznikají v údolích,
    ne v umělých paprscích jako u čistého D8. Okrajové buňky jsou odtoky (vše 0).
    """
    gh, gw = filled.shape
    pad = np.pad(filled, 1, mode="edge")
    frac = np.zeros((8, gh, gw))
    for k, ((dy, dx), dist) in enumerate(zip(_NB, _NB_DIST)):
        drop = (filled - pad[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw]) / dist
        frac[k] = np.maximum(drop, 0.0) ** exponent
    total = frac.sum(0)
    frac /= np.where(total > 0, total, 1.0)
    frac[:, 0, :] = frac[:, -1, :] = 0
    frac[:, :, 0] = frac[:, :, -1] = 0
    return frac


def mfd_accumulation(frac: np.ndarray, weights: np.ndarray, keep: np.ndarray | None = None,
                     max_iter: int | None = None) -> np.ndarray:
    """Akumulace toku po MFD – opakované přelévání (Jacobi), přesné po délce nejdelší cesty."""
    gh, gw = weights.shape
    w = weights.astype(np.float64)
    acc = w.copy()
    for _ in range(max_iter or 2 * (gh + gw)):
        out = acc * (1.0 if keep is None else keep)
        new = w.copy()
        for k, (dy, dx) in enumerate(_NB):
            f = frac[k] * out
            # buňka (y, x) posílá do (y + dy, x + dx)
            ys, yd = (slice(0, gh - dy), slice(dy, gh)) if dy >= 0 else (slice(-dy, gh), slice(0, gh + dy))
            xs, xd = (slice(0, gw - dx), slice(dx, gw)) if dx >= 0 else (slice(-dx, gw), slice(0, gw + dx))
            new[yd, xd] += f[ys, xs]
        if np.allclose(new, acc, rtol=1e-6, atol=1e-9):
            acc = new
            break
        acc = new
    return acc


def analyze_hydrology(heights: np.ndarray, factor: int, water_level: float, cell: float, rain_from: float = 0.008,
                      rain_span: float = 0.04, river_threshold: float = 6.0, lake_min: float = 0.004,
                      flat_slope: float = 0.05, flat_loss: float = 0.12, exponent: float = 4.0) -> Hydrology:
    """Jezera a řeky pro výškovou mapu.

    Déšť (a tání sněhu) padá hlavně na vyvýšená místa: váha buňky roste od výšky
    ``rain_from`` do ``rain_from + rain_span``. Řeka je tam, kde se sejde víc než
    ``river_threshold`` vah, jezero tam, kde je sníženina hlubší než ``lake_min``.
    Na rovině (sklon pod ``flat_slope``) se z řeky vsakuje až ``flat_loss`` vody na buňku.
    ``cell`` = velikost buňky původní mřížky [m].
    """
    coarse = downsample(heights, factor)
    filled = priority_flood(coarse)
    rec = d8_receivers(filled)
    frac = mfd_fractions(filled, exponent)
    weights = np.clip((coarse - rain_from) / rain_span, 0.0, 1.5)
    gh, gw = coarse.shape
    drop = filled.ravel() - np.where(rec >= 0, filled.ravel()[np.maximum(rec, 0)], filled.ravel())
    slope = (drop / (cell * factor)).reshape(gh, gw)
    keep = 1.0 - flat_loss * np.clip(1.0 - slope / flat_slope, 0.0, 1.0)
    acc = mfd_accumulation(frac, weights, keep)
    depth = filled - coarse
    lake = np.where(depth > lake_min, depth, 0.0)
    sea = coarse < water_level
    ratio = np.maximum(acc, 1e-9) / river_threshold
    river = np.where(ratio >= 1, 0.35 + 0.65 * np.clip(np.log(ratio) / np.log(25), 0, 1), 0)
    river[(lake > 0) | sea] = 0
    # směr toku = vážený průměr směrů odtoku, vyhlazený, aby se textura v zatáčkách nelámala
    dx = sum(frac[k] * (ddx / d) for k, ((_, ddx), d) in enumerate(zip(_NB, _NB_DIST))).astype(np.float32)
    dy = sum(frac[k] * (ddy / d) for k, ((ddy, _), d) in enumerate(zip(_NB, _NB_DIST))).astype(np.float32)
    wgt = river.astype(np.float32) + 1e-3
    dx = cv2.blur(dx * wgt, (3, 3))
    dy = cv2.blur(dy * wgt, (3, 3))
    n = np.maximum(np.hypot(dx, dy), 1e-6)
    return Hydrology(coarse, filled, rec, acc, lake.astype(np.float32), river.astype(np.float32),
                     np.stack([dx / n, dy / n]), factor)


# --- vegetace ----------------------------------------------------------------------------------
CONIFER, DECIDUOUS = 0, 1
# pravděpodobnost uchycení semínka podle biomu (WATER..SNOW)
SPAWN_CHANCE = np.array([0.0, 0.0, 0.05, 1.0, 0.05, 0.0])
# rozestup korun podle biomu (louka = jen řídké solitéry)
SPACING = np.array([1.0, 1.0, 3.0, 0.85, 1.6, 1.0])


def _cells(pos: np.ndarray, shape: tuple[int, int], size: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    gh, gw = shape
    ix = np.clip((pos[:, 0] / size[0] * gw).astype(int), 0, gw - 1)
    iy = np.clip((pos[:, 1] / size[1] * gh).astype(int), 0, gh - 1)
    return iy, ix


class Vegetation:
    """Stromy jako pole (pozice, velikost, druh…) – rostou, hynou, nově se uchycují."""

    GROW_TIME = 7.0  # [s] z semínka do plné velikosti
    DIE_TIME = 1.4  # [s] jak dlouho trvá uschnutí / spláchnutí
    MAX_CHANGE = 0.012  # [m] změna terénu, kterou strom nepřežije (zasypání / vykopání)

    def __init__(self, size: tuple[float, float], max_trees: int = 320, radius: float = 0.016,
                 spawn_rate: float = 22.0, seed: int | None = None) -> None:
        self.size = size
        self.max_trees = max_trees
        self.radius = radius  # [m] průměrný poloměr koruny dospělého stromu
        self.spawn_rate = spawn_rate  # [stromů/s] když je kde růst
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.pos = np.zeros((0, 2), np.float32)
        self.max_r = np.zeros(0, np.float32)
        self.age = np.zeros(0, np.float32)
        self.kind = np.zeros(0, np.int8)
        self.h0 = np.zeros(0, np.float32)
        self.dead = np.zeros(0, np.float32)  # < 0 živý, jinak čas od úhynu
        self.cause = np.zeros(0, np.int8)  # 0 sucho/zasypání, 1 utopení
        self._spawn_budget = 0.0
        self.deaths = 0

    def __len__(self) -> int:
        return len(self.pos)

    @property
    def alive(self) -> np.ndarray:
        return self.dead < 0

    def radii(self) -> np.ndarray:
        """Aktuální poloměr koruny (roste s věkem, při úhynu se zmenšuje)."""
        g = np.clip(self.age / self.GROW_TIME, 0, 1)
        g = 0.18 + 0.82 * (1 - (1 - g) ** 2)
        shrink = np.where(self.dead < 0, 1.0, np.clip(1 - self.dead / self.DIE_TIME, 0, 1))
        return self.max_r * g * (0.4 + 0.6 * shrink)

    def _keep(self, keep: np.ndarray) -> None:
        for name in ("pos", "max_r", "age", "kind", "h0", "dead", "cause"):
            setattr(self, name, getattr(self, name)[keep])

    def step(self, dt: float) -> None:
        self.age += dt
        self.dead[self.dead >= 0] += dt
        if np.any(self.dead > self.DIE_TIME):
            self._keep(self.dead <= self.DIE_TIME)

    def check(self, heights: np.ndarray, biome: np.ndarray, wet: np.ndarray, hands: np.ndarray | None = None) -> int:
        """Zabije stromy, pod kterými se terén změnil, zaplavila je voda nebo jsou v nevhodném biomu."""
        if not len(self):
            return 0
        iy, ix = _cells(self.pos, heights.shape, self.size)
        h = heights[iy, ix]
        b = biome[iy, ix]
        drowned = wet[iy, ix] | (b == WATER)
        bad = drowned | (np.abs(h - self.h0) > self.MAX_CHANGE) | (b == SNOW) | (b == BEACH)
        if hands is not None:
            bad &= ~hands[iy, ix]  # pod rukou Kinect písek nevidí
        bad &= self.alive
        n = int(bad.sum())
        if n:
            self.dead[bad] = 0.0
            self.cause[bad] = drowned[bad].astype(np.int8)
            self.deaths += n
        return n

    def spawn(self, dt: float, heights: np.ndarray, biome: np.ndarray, wet: np.ndarray, slope: np.ndarray) -> int:
        """Nové stromy tam, kde se jim daří (a kde ještě není hustý les)."""
        self._spawn_budget = min(self._spawn_budget + self.spawn_rate * dt, 6.0)
        n_try = int(self._spawn_budget * 6) + 1
        gh, gw = heights.shape
        iy = self.rng.integers(0, gh, n_try)
        ix = self.rng.integers(0, gw, n_try)
        chance = SPAWN_CHANCE[biome[iy, ix]] * ~wet[iy, ix]
        ok = self.rng.random(n_try) < chance
        added = 0
        for y, x in zip(iy[ok], ix[ok]):
            if self._spawn_budget < 1 or len(self) >= self.max_trees:
                break
            px = (x + self.rng.random()) / gw * self.size[0]
            py = (y + self.rng.random()) / gh * self.size[1]
            h = float(heights[y, x])
            r = self.radius * float(self.rng.uniform(0.75, 1.25))
            if biome[y, x] == ROCK:
                r *= 0.7
            if len(self):
                d = np.hypot(self.pos[:, 0] - px, self.pos[:, 1] - py)
                if np.any(d < SPACING[biome[y, x]] * (self.max_r + r)):
                    continue
            p_con = 0.15 + 0.8 * float(np.clip((h - 0.01) / 0.035 + slope[y, x] - 0.2, 0, 1))
            kind = CONIFER if self.rng.random() < p_con else DECIDUOUS
            self.pos = np.vstack([self.pos, np.array([[px, py]], np.float32)])
            self.max_r = np.append(self.max_r, np.float32(r))
            self.age = np.append(self.age, np.float32(0))
            self.kind = np.append(self.kind, np.int8(kind))
            self.h0 = np.append(self.h0, np.float32(h))
            self.dead = np.append(self.dead, np.float32(-1))
            self.cause = np.append(self.cause, np.int8(0))
            self._spawn_budget -= 1
            added += 1
        return added


# --- zvířata ------------------------------------------------------------------------------------
GRAZE_BIOMES = (MEADOW, FOREST)


class Animals:
    """Srnky (chodí po loukách, vyhýbají se vodě, utíkají před rukou) a krouživí ptáci."""

    def __init__(self, size: tuple[float, float], n_deer: int = 6, n_birds: int = 4, seed: int | None = None) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.n_deer, self.n_birds = n_deer, n_birds
        self.reset()

    def reset(self) -> None:
        rng = self.rng
        w, d = self.size
        n = self.n_deer
        self.deer = rng.random((n, 2)) * [w, d]
        self.heading = rng.uniform(0, 2 * np.pi, n)
        self.walking = rng.random(n) < 0.5
        self.timer = rng.uniform(0.5, 3.0, n)
        self.speed = np.zeros(n)
        self.misplaced = np.full(n, 99.0)  # jak dlouho je srnka na špatném místě (99 = přemístit)
        self.visible = np.zeros(n)  # 0..1 prolínání při objevení
        self.fleeing = np.zeros(n, bool)
        m = self.n_birds
        self.bird_center = rng.random((m, 2)) * [w * 0.6, d * 0.6] + [w * 0.2, d * 0.2]
        self.bird_angle = rng.uniform(0, 2 * np.pi, m)
        self.bird_radius = rng.uniform(0.08, 0.16, m) * max(w, d)
        self.bird_omega = rng.choice([-1, 1], m) * rng.uniform(0.35, 0.6, m)
        self.bird_drift = rng.uniform(0, 2 * np.pi, m)

    @staticmethod
    def allowed(biome: np.ndarray, wet: np.ndarray, iy, ix) -> np.ndarray:
        b = biome[iy, ix]
        return ((b == MEADOW) | (b == FOREST)) & ~wet[iy, ix]

    def birds(self) -> tuple[np.ndarray, np.ndarray]:
        """Pozice ptáků (m, 2) a jejich směr letu [rad]."""
        a = self.bird_angle
        p = self.bird_center + self.bird_radius[:, None] * np.stack([np.cos(a), np.sin(a)], 1)
        head = a + np.sign(self.bird_omega) * np.pi / 2
        return p, head

    def step(self, dt: float, biome: np.ndarray, wet: np.ndarray, hand_points=()) -> None:
        rng = self.rng
        w, d = self.size
        shape = biome.shape
        # ptáci: kroužení kolem pomalu bloudících středů
        self.bird_angle += self.bird_omega * dt
        self.bird_drift += rng.normal(0, 0.4, self.n_birds) * dt
        self.bird_center += 0.03 * dt * np.stack([np.cos(self.bird_drift), np.sin(self.bird_drift)], 1)
        lo = np.array([w * 0.15, d * 0.15])
        hi = np.array([w * 0.85, d * 0.85])
        out = (self.bird_center < lo) | (self.bird_center > hi)
        self.bird_drift[out.any(1)] += np.pi * 0.5
        self.bird_center = np.clip(self.bird_center, lo, hi)

        if self.n_deer == 0:
            return
        iy, ix = _cells(self.deer, shape, self.size)
        here_ok = self.allowed(biome, wet, iy, ix)
        self.misplaced = np.where(here_ok, 0.0, self.misplaced + dt)
        # srnka v vodě / na skále moc dlouho -> objeví se jinde na louce
        lost = self.misplaced > 1.5
        if np.any(lost):
            good = np.argwhere(self.allowed(biome, wet, slice(None), slice(None)))
            meadow = good[biome[good[:, 0], good[:, 1]] == MEADOW] if len(good) else good
            pool = meadow if len(meadow) > 20 else good
            for k in np.nonzero(lost)[0]:
                if not len(pool):
                    break
                y, x = pool[rng.integers(len(pool))]
                self.deer[k] = [(x + 0.5) / shape[1] * w, (y + 0.5) / shape[0] * d]
                self.misplaced[k] = 0
                self.visible[k] = 0
                self.heading[k] = rng.uniform(0, 2 * np.pi)
        visible_target = np.where(self.misplaced > 0.5, 0.0, 1.0)
        self.visible += np.clip(visible_target - self.visible, -dt * 2, dt * 2)

        # chování: pastva / chůze / útěk před rukou
        self.timer -= dt
        switch = self.timer <= 0
        self.walking[switch] = ~self.walking[switch]
        self.timer[switch] = np.where(self.walking[switch], rng.uniform(1.5, 4.0, switch.sum()),
                                      rng.uniform(1.0, 3.5, switch.sum()))
        self.heading += rng.normal(0, 0.8, self.n_deer) * dt * self.walking
        self.fleeing[:] = False
        for hx, hy in hand_points:
            dx, dy = self.deer[:, 0] - hx, self.deer[:, 1] - hy
            near = np.hypot(dx, dy) < 0.18
            self.heading[near] = np.arctan2(dy[near], dx[near])
            self.fleeing |= near
        speed = np.where(self.fleeing, 0.16, np.where(self.walking, 0.045, 0.0))
        self.speed += (speed - self.speed) * min(1.0, dt * 4)
        # dívej se kousek dopředu; před vodou / skálou / okrajem se otoč
        look = 0.04
        ahead = self.deer + look * np.stack([np.cos(self.heading), np.sin(self.heading)], 1)
        inside = (ahead[:, 0] > 0.02) & (ahead[:, 0] < w - 0.02) & (ahead[:, 1] > 0.02) & (ahead[:, 1] < d - 0.02)
        ay, ax = _cells(ahead, shape, self.size)
        blocked = ~(inside & self.allowed(biome, wet, ay, ax)) & here_ok & (self.speed > 0.005)
        if np.any(blocked):
            self.heading[blocked] += rng.uniform(np.pi * 0.6, np.pi * 1.4, blocked.sum())
            self.speed[blocked] *= 0.3
        move = ~blocked
        self.deer[move] += (self.speed[move] * dt)[:, None] * np.stack(
            [np.cos(self.heading[move]), np.sin(self.heading[move])], 1)
        self.deer[:, 0] = np.clip(self.deer[:, 0], 0.01, w - 0.01)
        self.deer[:, 1] = np.clip(self.deer[:, 1], 0.01, d - 0.01)


# --- město a povodeň ------------------------------------------------------------------------------
@dataclass
class House:
    x: float
    y: float
    angle: float  # [rad] natočení hřebene střechy
    length: float  # [m]
    width: float  # [m]
    church: bool = False
    flooded: bool = False
    flooded_at: float = 0.0


def flood_source_candidates(heights: np.ndarray, size: tuple[float, float], rng: np.random.Generator,
                            inset: float = 0.12, factor: int = 4, n: int = 3) -> list[tuple[float, float]]:
    """Kandidáti na pramen povodně: vyvýšená místa u okraje, ze kterých voda teče daleko do bedny.

    Pro body podél okrajů se spočítá cesta vody (D8 po terénu s vyplněnými sníženinami);
    vyhrává dlouhá cesta z vysokého místa (s trochou náhody). Vrací n nejlepších, různě daleko od sebe.
    """
    w, d = size
    pts = []
    for t in np.linspace(0.12, 0.88, 10):
        pts += [(t * w, inset), (t * w, d - inset), (inset, t * d), (w - inset, t * d)]
    pts = np.array(pts)
    coarse = downsample(heights, factor)
    ch, cw = coarse.shape
    rec = d8_receivers(priority_flood(coarse))
    iy, ix = _cells(pts, coarse.shape, size)
    smooth = cv2.blur(coarse, (3, 3))
    diag = math.hypot(w, d)
    score = np.empty(len(pts))
    for k, (y, x) in enumerate(zip(iy, ix)):
        path = np.array(trace_downhill(rec, int(y * cw + x)))
        px, py = (path % cw + 0.5) * w / cw, (path // cw + 0.5) * d / ch
        reach = float(np.max(np.hypot(px - pts[k, 0], py - pts[k, 1])))  # jak daleko voda doteče
        score[k] = reach / diag + (smooth[y, x] - float(np.median(coarse))) / 0.05
    score += rng.normal(0, 0.08, len(pts))
    chosen: list[tuple[float, float]] = []
    for k in np.argsort(-score):
        x, y = pts[k]
        if all(math.hypot(x - cx, y - cy) > 0.25 * diag for cx, cy in chosen):
            chosen.append((float(x), float(y)))
        if len(chosen) >= n:
            break
    return chosen


def choose_flood_source(heights: np.ndarray, size: tuple[float, float], rng: np.random.Generator,
                        inset: float = 0.12, factor: int = 4) -> tuple[float, float]:
    """Nejlepší kandidát na pramen povodně (viz flood_source_candidates)."""
    return flood_source_candidates(heights, size, rng, inset, factor, n=1)[0]


class FloodPlanner:
    """Zkušební povodně na zmenšené mřížce: kam a za jak dlouho voda z pramene doteče.

    Pro každý kandidátní pramen běží WaterSim (dt = 0.1 s) – po kouscích, aby se
    nezasekl snímek (``step``). Výsledek (``choose``) je pramen a místo pro město,
    kam voda bez zásahu opravdu dorazí zhruba v polovině povodně.
    """

    def __init__(self, heights: np.ndarray, size: tuple[float, float], sources: list[tuple[float, float]],
                 flow: float, duration: float, factor: int = 4, dt: float = 0.1, threshold: float = 0.003,
                 infiltration: float = 0.0, drain: bool = False) -> None:
        from .water import WaterSim

        self.size = size
        self.sources = sources
        self.coarse = downsample(heights, factor)
        self.flow, self.duration, self.dt, self.threshold = flow, duration, dt, threshold
        self.infiltration, self.drain = infiltration, drain
        cell = size[0] / self.coarse.shape[1]
        self.sims = [WaterSim(self.coarse.shape, cell) for _ in sources]
        self.arrival = [np.full(self.coarse.shape, np.inf) for _ in sources]
        self.t = [0.0 for _ in sources]
        self.k = 0

    @property
    def done(self) -> bool:
        return self.k >= len(self.sources)

    def step(self, n_steps: int = 50) -> bool:
        """Spočítá dalších n_steps kroků; vrací True, když je hotovo."""
        for _ in range(n_steps):
            if self.done:
                break
            k = self.k
            sim = self.sims[k]
            add_source(sim.depth, self.size, self.sources[k], 0.04, self.flow * self.dt)
            sim.advance(self.coarse, self.dt, 9.81, drain=self.drain)
            soak(sim.depth, self.infiltration * self.dt)
            self.t[k] += self.dt
            arr = self.arrival[k]
            arr[(sim.depth > self.threshold) & np.isinf(arr)] = self.t[k]
            if self.t[k] >= self.duration:
                self.k += 1
        return self.done

    def choose(self, rng: np.random.Generator, town_radius: float, water_level: float,
               avoid: tuple[float, float] | None = None) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """(pramen, střed města) nebo None, když voda nikam rozumně nedoteče."""
        w, d = self.size
        ch, cw = self.coarse.shape
        cell = (w / cw, d / ch)
        gx, gy = np.meshgrid((np.arange(cw) + 0.5) * cell[0], (np.arange(ch) + 0.5) * cell[1])
        slope = cv2.blur(slope_map(self.coarse, cell), (3, 3))
        flat_h = cv2.blur(self.coarse, (3, 3))
        margin = town_radius + 0.03
        # nahoře je navíc místo pro nápis s odpočtem
        inside = (gx > margin) & (gx < w - margin) & (gy > margin + 0.08 * d) & (gy < d - margin)
        low = np.percentile(self.coarse, 5)
        diag = math.hypot(w, d)
        best = None
        for src, arr in zip(self.sources, self.arrival):
            d_src = np.hypot(gx - src[0], gy - src[1])
            t = arr / self.duration
            cost = (slope / 0.08 + 3.0 * np.abs(t - 0.45) + (flat_h - low) / 0.03
                    - 0.5 * np.clip((flat_h - water_level) / 0.01, -2, 0)  # ne do nejhlubší díry
                    + rng.random(arr.shape) * 0.6)
            if avoid is not None:
                cost += 2.0 * (np.hypot(gx - avoid[0], gy - avoid[1]) < 0.2 * diag)
            for level, ok in enumerate((inside & (t < 0.75) & (d_src > 0.3 * diag),
                                        inside & np.isfinite(t) & (d_src > 0.15 * diag))):
                if not ok.any():
                    continue
                c = np.where(ok, cost, np.inf)
                i = int(np.argmin(c))
                # pramen s více možnostmi je lepší (voda teče daleko po souši)
                total = float(c.flat[i]) - 0.02 * float(ok.sum()) + 10.0 * level
                if best is None or total < best[0]:
                    best = (total, src, (float(gx.flat[i]), float(gy.flat[i])))
                break
        if best is None:
            return None
        return best[1], best[2]


def place_town(heights: np.ndarray, size: tuple[float, float], water_level: float, rng: np.random.Generator,
               source: tuple[float, float], town_radius: float, avoid: tuple[float, float] | None = None,
               factor: int = 4) -> tuple[float, float]:
    """Střed města: rovinatá nížina po proudu od pramene (kam povodeň opravdu doteče)."""
    w, d = size
    coarse = downsample(heights, factor)
    ch, cw = coarse.shape
    cell = (w / cw, d / ch)
    filled = priority_flood(coarse)
    rec = d8_receivers(filled)
    sy, sx = _cells(np.array([source]), coarse.shape, size)
    path = np.array(trace_downhill(rec, int(sy[0] * cw + sx[0])))
    px = (path % cw + 0.5) * cell[0]
    py = (path // cw + 0.5) * cell[1]
    gx, gy = np.meshgrid((np.arange(cw) + 0.5) * cell[0], (np.arange(ch) + 0.5) * cell[1])
    d_path = np.min(np.hypot(gx[..., None] - px, gy[..., None] - py), axis=2)
    d_src = np.hypot(gx - source[0], gy - source[1])
    slope = cv2.blur(slope_map(coarse, cell), (3, 3))
    flat_h = cv2.blur(coarse, (3, 3))
    diag = math.hypot(w, d)
    margin = town_radius + 0.03
    inside = (gx > margin) & (gx < w - margin) & (gy > margin) & (gy < d - margin)
    dry = flat_h > water_level + 0.006
    far = (d_src > 0.3 * diag) & (d_src < 0.9 * diag)
    cost = ((flat_h - np.percentile(coarse, 5)) / 0.02 + slope / 0.08 + d_path / 0.05
            + rng.random(coarse.shape) * 0.8)
    if avoid is not None:
        cost += 3.0 * (np.hypot(gx - avoid[0], gy - avoid[1]) < 0.25 * diag)
    for mask in (inside & dry & far & (d_path < 0.08), inside & dry & far, inside & dry, inside):
        if mask.any():
            c = np.where(mask, cost, np.inf)
            iy, ix = np.unravel_index(int(np.argmin(c)), c.shape)
            return float(gx[iy, ix]), float(gy[iy, ix])
    return w / 2, d / 2


def layout_houses(center: tuple[float, float], n: int, house: float, size: tuple[float, float],
                  rng: np.random.Generator) -> list[House]:
    """Rozmístí n domů (první je kostel) kolem středu města, bez překryvů a uvnitř bedny."""
    cx, cy = center
    base = float(rng.uniform(0, np.pi))
    houses = [House(cx, cy, base, house * 1.5, house * 0.85, church=True)]
    spacing = house * 1.35
    for i in range(800):
        if len(houses) >= n:
            break
        r = house * rng.uniform(1.7, 2.0 + 0.2 * len(houses) + i * 0.01)
        a = rng.uniform(0, 2 * np.pi)
        x, y = cx + r * np.cos(a), cy + r * np.sin(a)
        if not (house < x < size[0] - house and house < y < size[1] - house):
            continue
        if any(math.hypot(x - h.x, y - h.y) < spacing * (1.25 if h.church else 1.0) for h in houses):
            continue
        ang = base + (np.pi / 2) * int(rng.integers(0, 2)) + rng.normal(0, 0.12)
        houses.append(House(float(x), float(y), float(ang), house * rng.uniform(0.9, 1.15), house * 0.7))
    return houses


def house_footprint(h: House, n: int = 3) -> np.ndarray:
    """Body (n×n) pokrývající půdorys domu, (n², 2) v metrech."""
    u = np.linspace(-0.45, 0.45, n)
    a, b = np.meshgrid(u * h.length, u * h.width)
    ca, sa = math.cos(h.angle), math.sin(h.angle)
    return np.stack([h.x + a.ravel() * ca - b.ravel() * sa, h.y + a.ravel() * sa + b.ravel() * ca], 1)


def flooded_houses(depth: np.ndarray, houses: list[House], size: tuple[float, float],
                   threshold: float = 0.003) -> np.ndarray:
    """Které domy stojí ve vodě hlubší než threshold (bool pro každý dům)."""
    out = np.zeros(len(houses), bool)
    for k, h in enumerate(houses):
        iy, ix = _cells(house_footprint(h), depth.shape, size)
        out[k] = bool(np.max(depth[iy, ix]) > threshold)
    return out


def soak(depth: np.ndarray, amount: float) -> None:
    """Vsakování: z každé mokré buňky ubere amount [m] vody."""
    if amount > 0:
        np.maximum(depth - amount, 0, out=depth)


def add_source(depth: np.ndarray, size: tuple[float, float], source: tuple[float, float], radius: float,
               volume: float) -> None:
    """Přidá objem vody [m³] rovnoměrně do kruhu kolem pramene."""
    gh, gw = depth.shape
    cw, ch = size[0] / gw, size[1] / gh
    x0, x1 = int(max(0, (source[0] - radius) / cw)), int(min(gw, (source[0] + radius) / cw + 1))
    y0, y1 = int(max(0, (source[1] - radius) / ch)), int(min(gh, (source[1] + radius) / ch + 1))
    gx = (np.arange(x0, x1) + 0.5) * cw
    gy = (np.arange(y0, y1) + 0.5) * ch
    mask = np.hypot(gx[None, :] - source[0], gy[:, None] - source[1]) <= radius
    n = int(mask.sum())
    if n:
        depth[y0:y1, x0:x1][mask] += volume / (n * cw * ch)
