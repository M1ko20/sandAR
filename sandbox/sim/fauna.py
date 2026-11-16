"""Zvířata živého ekosystému – každý druh žije jen tam, kde se mu daří.

Druh má své prostředí (biomy, voda), hustotu (kolik kusů na m² vhodné plochy), rychlost
a povahu (utíká před rukou / je zvědavý / je mu to jedno). Když prostředí zmizí (zasypané
jezero, srovnaná hora), zvířata se ztratí; když vznikne, objeví se. Počet se řídí velikostí
vhodné plochy, takže velké jezero má hejno ryb, malé jen pár.

Pozemní a vodní zvířata chodí/plavou, před nevhodným terénem se otočí. Letci (ptáci, orel,
motýli) se terénem omezují jen tím, nad čím krouží.

Logika je čisté numpy (testovatelná), kreslení je v ``sandbox/render/fauna.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ecosystem import BEACH, FOREST, MEADOW, ROCK, SNOW, WATER, _cells

# prostředí (masky z biomů, vlhka a vody)
LAND_GREEN = "zeleň"  # louka a les (ne v řece)
MEADOW_ONLY = "louka"
FOREST_ONLY = "les"
MOUNTAIN = "hory"  # skály a sníh
BEACH_ONLY = "pláž"
WATER_ANY = "voda"  # moře, jezera, řeky
WATER_OPEN = "hladina"  # moře a jezera (ne úzké řeky)
AIR_LOW = "nad krajinou"
AIR_HIGH = "nad horami"


@dataclass
class Species:
    key: str
    name: str  # česky, pro obsluhu
    habitat: str
    density: float  # [kusů / m² vhodné plochy]
    max_count: int
    length: float  # [m] délka těla (kreslení)
    walk: float  # [m/s]
    flee: float  # [m/s] při útěku před rukou (0 = neutíká)
    flee_radius: float = 0.16
    turn: float = 0.8  # náhodné zatáčení [rad/s]
    rest: tuple[float, float] = (1.0, 3.5)  # jak dlouho stojí / pase se
    move: tuple[float, float] = (1.5, 4.0)  # jak dlouho jde
    flier: bool = False
    min_area: float = 0.004  # [m²] menší prostředí = žádné zvíře


SPECIES: list[Species] = [
    Species("srnka", "srnky", LAND_GREEN, 5.0, 6, 0.036, 0.045, 0.17),
    Species("zajic", "zajíci", MEADOW_ONLY, 6.0, 6, 0.024, 0.06, 0.24, turn=1.6, rest=(0.6, 2.0), move=(0.4, 1.2)),
    Species("liska", "lišky", LAND_GREEN, 1.5, 2, 0.036, 0.055, 0.12, flee_radius=0.12),
    Species("medved", "medvědi", FOREST_ONLY, 1.2, 1, 0.055, 0.03, 0.0, rest=(3.0, 7.0), min_area=0.03),
    Species("kamzik", "kamzíci", MOUNTAIN, 14.0, 4, 0.03, 0.04, 0.14, turn=1.2, min_area=0.006),
    Species("krab", "krabi", BEACH_ONLY, 25.0, 5, 0.018, 0.03, 0.08, turn=2.5, rest=(0.5, 2.0), move=(0.5, 1.5)),
    Species("zelva", "želvy", BEACH_ONLY, 6.0, 2, 0.034, 0.012, 0.0, rest=(2.0, 6.0), move=(3.0, 6.0),
            min_area=0.01),
    Species("ryba", "ryby", WATER_ANY, 40.0, 14, 0.022, 0.07, 0.2, turn=1.4, rest=(0.2, 0.6), move=(2.0, 5.0)),
    Species("kachna", "kachny", WATER_OPEN, 10.0, 5, 0.026, 0.025, 0.1, rest=(1.5, 4.0), min_area=0.01),
    Species("ptak", "ptáci", AIR_LOW, 3.0, 4, 0.04, 0.0, 0.0, flier=True),
    Species("orel", "orli", AIR_HIGH, 5.0, 1, 0.07, 0.0, 0.0, flier=True, min_area=0.01),
    Species("motyl", "motýli", MEADOW_ONLY, 6.0, 8, 0.016, 0.06, 0.0, turn=3.0, rest=(0.1, 0.3), move=(1.0, 3.0),
            flier=True),
]
BY_KEY = {s.key: s for s in SPECIES}


def habitat_masks(biome: np.ndarray, wet: np.ndarray, water: np.ndarray) -> dict[str, np.ndarray]:
    """Masky prostředí. ``wet`` = řeky a jezera, ``water`` = buňky pod hladinou (moře + jezera)."""
    river = wet & ~water
    green = ((biome == MEADOW) | (biome == FOREST)) & ~wet
    return {
        LAND_GREEN: green,
        MEADOW_ONLY: (biome == MEADOW) & ~wet,
        FOREST_ONLY: (biome == FOREST) & ~wet,
        MOUNTAIN: ((biome == ROCK) | (biome == SNOW)) & ~wet,
        BEACH_ONLY: (biome == BEACH) & ~wet,
        WATER_ANY: water | river | (biome == WATER),
        WATER_OPEN: water | (biome == WATER),
        AIR_LOW: green,
        AIR_HIGH: (biome == ROCK) | (biome == SNOW),
    }


@dataclass
class Herd:
    """Všichni jedinci jednoho druhu."""

    sp: Species
    pos: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    timer: np.ndarray
    walking: np.ndarray
    visible: np.ndarray  # 0..1 (prolínání při objevení / zmizení)
    misplaced: np.ndarray  # jak dlouho je na nevhodném místě
    phase: np.ndarray  # náhodná fáze (animace, kroužení)
    active: int = 0  # kolik jedinců má teď prostředí (zbytek je skrytý)
    fleeing: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    center: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # letci: střed kroužení
    radius: np.ndarray = field(default_factory=lambda: np.zeros(0))


class Fauna:
    def __init__(self, size: tuple[float, float], seed: int | None = None, species: list[Species] | None = None) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.species = species if species is not None else SPECIES
        self.herds: dict[str, Herd] = {}
        self.reset()

    def reset(self) -> None:
        rng = self.rng
        w, d = self.size
        self.herds = {}
        for sp in self.species:
            n = sp.max_count
            h = Herd(sp, rng.random((n, 2)) * [w, d], rng.uniform(0, 2 * np.pi, n), np.zeros(n),
                     rng.uniform(0.2, 2.0, n), rng.random(n) < 0.5, np.zeros(n), np.full(n, 99.0),
                     rng.uniform(0, 2 * np.pi, n), fleeing=np.zeros(n, bool))
            if sp.flier:
                h.center = h.pos.copy()
                h.radius = rng.uniform(0.07, 0.15, n) * max(w, d) if sp.key != "motyl" else np.zeros(n)
            self.herds[sp.key] = h

    def counts(self) -> dict[str, int]:
        return {k: int((h.visible > 0.5).sum()) for k, h in self.herds.items()}

    # ------------------------------------------------------------------------------------
    def step(self, dt: float, biome: np.ndarray, wet: np.ndarray, water: np.ndarray, hands=()) -> None:
        """Jeden krok. ``hands`` = body rukou [(x, y), …] v metrech."""
        masks = habitat_masks(biome, wet, water)
        cell_area = (self.size[0] / biome.shape[1]) * (self.size[1] / biome.shape[0])
        hands = np.asarray(list(hands), float).reshape(-1, 2)
        for h in self.herds.values():
            mask = masks[h.sp.habitat]
            area = float(mask.sum()) * cell_area
            h.active = 0 if area < h.sp.min_area else int(min(h.sp.max_count, math.ceil(area * h.sp.density)))
            if h.sp.flier and h.sp.key != "motyl":
                self._circle(h, dt, mask)
            else:
                self._walk(h, dt, mask, hands)

    def _relocate(self, h: Herd, k: int, mask: np.ndarray) -> bool:
        good = np.argwhere(mask)
        if not len(good):
            return False
        gh, gw = mask.shape
        y, x = good[self.rng.integers(len(good))]
        w, d = self.size
        h.pos[k] = [(x + self.rng.random()) / gw * w, (y + self.rng.random()) / gh * d]
        h.misplaced[k] = 0
        h.visible[k] = 0
        h.heading[k] = self.rng.uniform(0, 2 * np.pi)
        return True

    def _walk(self, h: Herd, dt: float, mask: np.ndarray, hands: np.ndarray) -> None:
        sp, rng = h.sp, self.rng
        n = len(h.pos)
        w, d = self.size
        shape = mask.shape
        iy, ix = _cells(h.pos, shape, self.size)
        here_ok = mask[iy, ix]
        h.misplaced = np.where(here_ok, 0.0, h.misplaced + dt)
        idx = np.arange(n)
        for k in np.nonzero((h.misplaced > 1.5) & (idx < h.active))[0]:
            self._relocate(h, k, mask)
        want = np.where((h.misplaced > 0.5) | (idx >= h.active), 0.0, 1.0)
        h.visible += np.clip(want - h.visible, -dt * 2, dt * 2)

        h.timer -= dt
        sw = h.timer <= 0
        h.walking[sw] = ~h.walking[sw]
        h.timer[sw] = np.where(h.walking[sw], rng.uniform(*sp.move, sw.sum()), rng.uniform(*sp.rest, sw.sum()))
        h.heading += rng.normal(0, sp.turn, n) * dt * h.walking
        h.fleeing[:] = False
        if sp.flee > 0 and len(hands):
            dx = h.pos[:, 0, None] - hands[None, :, 0]
            dy = h.pos[:, 1, None] - hands[None, :, 1]
            dist = np.hypot(dx, dy)
            j = np.argmin(dist, axis=1)
            near = dist[idx, j] < sp.flee_radius
            h.heading[near] = np.arctan2(dy[idx, j], dx[idx, j])[near]
            h.fleeing |= near
        target = np.where(h.fleeing, sp.flee, np.where(h.walking, sp.walk, 0.0))
        if sp.key == "zajic":  # zajíc skáče: rychlost pulzuje
            target = target * (0.4 + 1.2 * (np.sin(h.phase * 3 + idx) > 0.3))
        h.phase += dt * (2.0 + 6.0 * (h.speed > 0.01))
        h.speed += (target - h.speed) * min(1.0, dt * 4)
        look = max(0.03, sp.length)
        ahead = h.pos + look * np.stack([np.cos(h.heading), np.sin(h.heading)], 1)
        inside = (ahead[:, 0] > 0.02) & (ahead[:, 0] < w - 0.02) & (ahead[:, 1] > 0.02) & (ahead[:, 1] < d - 0.02)
        ay, ax = _cells(ahead, shape, self.size)
        blocked = ~(inside & mask[ay, ax]) & here_ok & (h.speed > 0.003)
        if np.any(blocked):
            h.heading[blocked] += rng.uniform(np.pi * 0.6, np.pi * 1.4, blocked.sum())
            h.speed[blocked] *= 0.3
        move = ~blocked
        h.pos[move] += (h.speed[move] * dt)[:, None] * np.stack([np.cos(h.heading[move]), np.sin(h.heading[move])], 1)
        h.pos[:, 0] = np.clip(h.pos[:, 0], 0.01, w - 0.01)
        h.pos[:, 1] = np.clip(h.pos[:, 1], 0.01, d - 0.01)

    def _circle(self, h: Herd, dt: float, mask: np.ndarray) -> None:
        """Letci krouží kolem středu, který pomalu bloudí nad jejich prostředím."""
        rng = self.rng
        n = len(h.pos)
        w, d = self.size
        idx = np.arange(n)
        omega = np.where(idx % 2 == 0, 1.0, -1.0) * (0.5 if h.sp.key == "ptak" else 0.3)
        h.phase += omega * dt
        h.heading = h.phase + np.sign(omega) * np.pi / 2
        drift = h.timer  # timer = směr bloudění středu
        drift += rng.normal(0, 0.4, n) * dt
        h.center += 0.03 * dt * np.stack([np.cos(drift), np.sin(drift)], 1)
        lo, hi = np.array([w * 0.15, d * 0.15]), np.array([w * 0.85, d * 0.85])
        out = ((h.center < lo) | (h.center > hi)).any(1)
        drift[out] += np.pi * 0.5
        h.center = np.clip(h.center, lo, hi)
        # střed kroužení nad vhodným prostředím (orel nad horami)
        cy, cx = _cells(h.center, mask.shape, self.size)
        bad = ~mask[cy, cx]
        h.misplaced = np.where(bad, h.misplaced + dt, 0.0)
        for k in np.nonzero((h.misplaced > 3.0) & (idx < h.active))[0]:
            if self._relocate(h, k, mask):
                h.center[k] = h.pos[k]
        h.pos = h.center + h.radius[:, None] * np.stack([np.cos(h.phase), np.sin(h.phase)], 1)
        want = np.where(idx < h.active, 1.0, 0.0)
        h.visible += np.clip(want - h.visible, -dt, dt)
