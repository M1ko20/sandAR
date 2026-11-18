"""Předlohy terénu pro úkol „postav podle předlohy“.

Zdroje předloh:

* vestavěné tvary (kopec, sopka, sedlo, údolí, …),
* obrázky PNG/JPG ve složce ``targets/`` (šedotónová výšková mapa, bílá = nejvýš),
* soubory ``.npy`` ve složce ``targets/`` – výšky v metrech, vznikají klávesou
  „U“ (uložit aktuální písek jako předlohu).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import TARGETS_DIR


@dataclass
class Target:
    name: str
    generator: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None  # (X, Y) -> -1..1, 0 = okolní písek
    path: Path | None = None
    absolute: bool = False  # True = výšky jsou v metrech (nepřeškálovávat)

    def layout(self, shape: tuple[int, int], box_size: tuple[float, float], relief: float
               ) -> tuple[np.ndarray, np.ndarray]:
        """(výšky [m], maska hlavní části). Výška 0 = okolní písek, hodnotí se jen maska.

        Hlavní část = kde se předloha liší od okolí (kopec, kráter, koryto) + okraj ``MASK_MARGIN``,
        aby se počítal i správný přechod do okolí. Zbytek bedny je jedno, jak vypadá."""
        gh, gw = shape
        if self.generator is not None:
            aspect = box_size[0] / box_size[1]
            X, Y = np.meshgrid((np.arange(gw) + 0.5) / gw * aspect, (np.arange(gh) + 0.5) / gh)
            t = self.generator(X - aspect / 2, Y - 0.5).astype(np.float32)
            # kopce do plného převýšení, prohlubně nejvýš do 60 % (písek nemusí být tak hluboký)
            peak = max(float(t.max()), -float(t.min()) / 0.6)
            t = t / peak * relief if peak > 0 else np.zeros_like(t)
        else:
            assert self.path is not None
            if self.path.suffix == ".npy":
                t = np.load(self.path).astype(np.float32)
            else:
                img = cv2.imread(str(self.path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                if img is None:
                    raise ValueError(f"Nelze načíst {self.path}")
                t = img.astype(np.float32)
            t = cv2.resize(t, (gw, gh), interpolation=cv2.INTER_AREA)
            t = cv2.GaussianBlur(t, (0, 0), 1.0)
            t -= float(np.median(t))  # nejčastější úroveň = okolní písek
            if not self.absolute:
                span = float(np.abs(t).max())
                t = t / span * relief if span > 0 else np.zeros_like(t)
        return t.astype(np.float32), feature_mask(t, box_size)

    def grid(self, shape: tuple[int, int], box_size: tuple[float, float], relief: float) -> np.ndarray:
        return self.layout(shape, box_size, relief)[0]


MASK_MARGIN = 0.05  # [m] hodnocený okraj kolem tvaru předlohy


def feature_mask(t: np.ndarray, box_size: tuple[float, float], margin: float = MASK_MARGIN) -> np.ndarray:
    """Hlavní část předlohy: kde se výška liší od okolí aspoň o 6 % převýšení, + okraj ``margin``."""
    gh, gw = t.shape
    peak = float(np.abs(t).max())
    if peak <= 0:
        return np.ones(t.shape, bool)
    core = (np.abs(t) > 0.06 * peak).astype(np.uint8)
    r = max(1, int(round(margin / (box_size[0] / gw))))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.dilate(core, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)  # bez děr mezi blízkými tvary
    return m.astype(bool)


def _gauss(X, Y, cx, cy, s):
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))


def _smooth_step(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _volcano(X, Y):
    r = np.hypot(X, Y)
    cone = np.clip(1 - r / 0.36, 0, None) ** 1.3
    return cone - 0.55 * _gauss(X, Y, 0, 0, 0.05)


def _ridge(X, Y):
    d = X * 0.6 - Y * 0.8  # vzdálenost od úhlopříčky
    along = X * 0.8 + Y * 0.6
    return np.exp(-(d ** 2) / (2 * 0.07 ** 2)) * (0.75 + 0.25 * np.cos(along * 9)) * _smooth_step(0.42, 0.3, np.abs(along))


def _island(X, Y):
    island = _gauss(X, Y, 0, 0, 0.15)
    bay = _gauss(X, Y, 0.14, 0.06, 0.06)
    return np.clip(island - 0.8 * bay, 0, None)


def _lake(X, Y):
    return -0.6 * _smooth_step(0.24, 0.08, np.hypot(X * 0.8, Y))


def _crater(X, Y):
    r = np.hypot(X, Y)
    rim = np.exp(-((r - 0.2) ** 2) / (2 * 0.05 ** 2))
    return rim - 0.6 * _smooth_step(0.17, 0.05, r)


def _river(X, Y):
    centre = 0.1 * np.sin(X * 9)
    trench = -0.6 * np.exp(-((Y - centre) ** 2) / (2 * 0.045 ** 2))
    return trench * _smooth_step(0.48, 0.36, np.abs(X))


def _mesa(X, Y):
    return _smooth_step(0.24, 0.17, np.hypot(X * 0.75, Y))


def _pyramid(X, Y):
    return np.clip(1 - np.maximum(np.abs(X), np.abs(Y)) / 0.24, 0, None)


def _horseshoe(X, Y):
    r = np.hypot(X, Y)
    ring = np.exp(-((r - 0.2) ** 2) / (2 * 0.055 ** 2))
    return ring * _smooth_step(0.08, -0.12, Y)  # otevřená strana „dole“


def _castle(X, Y):
    r = np.hypot(X, Y)
    hill = _smooth_step(0.13, 0.07, r)
    moat = -0.55 * np.exp(-((r - 0.2) ** 2) / (2 * 0.035 ** 2))
    return hill + moat


def _heart(X, Y):
    lobes = (np.hypot(X + 0.1, Y + 0.06) < 0.12) | (np.hypot(X - 0.1, Y + 0.06) < 0.12)
    tip = (Y > -0.06) & (Y < 0.24) & (np.abs(X) < (0.24 - Y) * 0.78)
    flat = (lobes | tip).astype(np.float32)
    step = float(abs(X[0, 1] - X[0, 0])) if X.shape[1] > 1 else 0.01
    return cv2.GaussianBlur(flat, (0, 0), 0.03 / step)


def _terraces(X, Y):
    r = np.hypot(X * 0.9, Y)
    return (_smooth_step(0.30, 0.27, r) + _smooth_step(0.20, 0.17, r) + _smooth_step(0.10, 0.07, r)) / 3


def _three_hills(X, Y):
    return _gauss(X, Y, -0.24, 0.05, 0.08) + 0.75 * _gauss(X, Y, 0.0, -0.06, 0.08) + 0.5 * _gauss(X, Y, 0.22, 0.05, 0.08)


def _canyon(X, Y):
    walls = 0.8 * _smooth_step(0.30, 0.22, np.abs(Y)) * _smooth_step(0.40, 0.3, np.abs(X))
    cut = 1.3 * np.exp(-(Y - 0.05 * np.sin(X * 8)) ** 2 / (2 * 0.04 ** 2)) * _smooth_step(0.45, 0.3, np.abs(X))
    return walls - cut


def _massif(X, Y):
    """Hlavní předloha: hora s vedlejším vrcholem, spojené hřbetem se sedlem (jako skutečné pohoří).

    Velké a hladké tvary (svahy do ~25°, drť je udrží), žádné prohlubně pod okolní písek –
    postaví se jen nahrnutím písku a dá se udělat na 100 %."""
    main = _gauss(X, Y, -0.20, -0.03, 0.14)
    side = 0.62 * _gauss(X, Y, 0.21, 0.05, 0.12)
    ax, ay, bx, by = -0.20, -0.03, 0.21, 0.05
    t = np.clip(((X - ax) * (bx - ax) + (Y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2), 0, 1)
    ridge = 0.42 * np.exp(-((X - ax - t * (bx - ax)) ** 2 + (Y - ay - t * (by - ay)) ** 2) / (2 * 0.075 ** 2))
    p = 4.0  # měkké maximum – plynulé napojení hřbetu na vrcholy
    return (main ** p + side ** p + ridge ** p) ** (1 / p)


BUILTIN = [
    Target("Hora se sedlem", _massif),
    Target("Kopec", lambda X, Y: _gauss(X, Y, 0, 0, 0.13)),
    Target("Sopka s kráterem", _volcano),
    Target("Dva vrcholy a sedlo", lambda X, Y: _gauss(X, Y, -0.2, 0, 0.1) + 0.8 * _gauss(X, Y, 0.2, 0.02, 0.1)),
    Target("Horský hřbet", _ridge),
    Target("Ostrov se zálivem", _island),
    Target("Jezero", _lake),
    Target("Kráter", _crater),
    Target("Řeka", _river),
    Target("Stolová hora", _mesa),
    Target("Pyramida", _pyramid),
    Target("Podkova", _horseshoe),
    Target("Hrad s příkopem", _castle),
    Target("Srdce", _heart),
    Target("Terasy", _terraces),
    Target("Tři kopce", _three_hills),
    Target("Kaňon", _canyon),
]


def all_targets(directory: Path = TARGETS_DIR) -> list[Target]:
    targets = list(BUILTIN)
    if directory.is_dir():
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                targets.append(Target(p.stem.replace("_", " "), path=p))
            elif p.suffix == ".npy":
                targets.append(Target(p.stem.replace("_", " "), path=p, absolute=True))
    return targets


def save_capture(heights: np.ndarray, directory: Path = TARGETS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / time.strftime("vlastni_%Y-%m-%d_%H%M%S.npy")
    np.save(path, heights.astype(np.float32))
    return path


def _excess(x: np.ndarray, tol: float) -> np.ndarray:
    """Odchylka nad toleranci (co je v toleranci, je přesná shoda)."""
    return np.maximum(np.abs(x) - tol, 0.0)


@dataclass
class Score:
    score: float  # 0..100, 100 = přesná shoda
    within: float  # podíl buněk v toleranci [%]
    rmse: float  # [m]
    offset: float  # [m] posun hladiny (nepočítá se do chyby)
    scale: float = 1.0  # natažení předlohy (mírné hodnocení: písek = scale·předloha + offset)


def _region(shape: tuple[int, int], border: float, mask: np.ndarray | None) -> np.ndarray:
    gh, gw = shape
    by, bx = int(gh * border), int(gw * border)
    r = np.zeros(shape, bool)
    r[by:gh - by or None, bx:gw - bx or None] = True
    if mask is not None and mask.any():
        r &= mask
    return r


def evaluate(current: np.ndarray, target: np.ndarray, tolerance: float, border: float = 0.05,
             mask: np.ndarray | None = None) -> tuple[Score, np.ndarray]:
    """Porovná terén s předlohou v oblasti ``mask`` (hlavní část předlohy; None = celá bedna).

    Skóre = 1 − E / E(rovný písek), kde E = RMS odchylky nad toleranci: uhlazený písek má 0 %,
    terén všude v toleranci (celá hlavní část zelená) 100 %. Celkový posun výšky se odečte
    (množství písku žák neovlivní).
    Vrací (skóre, rozdíl current − target po odečtení posunu).
    """
    region = _region(target.shape, border, mask)
    diff = current - target
    offset = float(np.median(diff[region]))
    diff = diff - offset
    d = diff[region]
    rmse = float(np.sqrt(np.mean(d**2)))
    t = target[region]
    err = float(np.sqrt(np.mean(_excess(d, tolerance) ** 2)))
    baseline = float(np.sqrt(np.mean(_excess(t - np.median(t), tolerance) ** 2))) or 1e-6
    score = float(np.clip(1 - err / baseline, 0, 1) * 100)
    within = float(np.mean(np.abs(d) <= tolerance) * 100)
    return Score(score, within, rmse, offset), diff.astype(np.float32)


def evaluate_shape(current: np.ndarray, target: np.ndarray, border: float = 0.05, sigma: float = 2.0,
                   scale: tuple[float, float] = (0.6, 1.4), mask: np.ndarray | None = None,
                   tolerance: float = 0.01) -> tuple[Score, np.ndarray]:
    """Mírné hodnocení (pro jednoho hráče): počítá hlavně tvar, ne přesnou výšku.

    Oba povrchy se vyhladí (drobné hrbolky nevadí) a předloha se smí natáhnout nebo
    zmenšit v rozsahu ``scale`` – kopec poloviční výšky tak dostane většinu bodů.
    Skóre = 1 − E / E(rovný písek), E = RMS odchylky nad ``tolerance``: rovný písek 0 %,
    správný tvar 100 %.
    Vrací (skóre, rozdíl current − (a·target + posun)) jako ``evaluate``.
    """
    region = _region(target.shape, border, mask)
    c = cv2.GaussianBlur(np.asarray(current, np.float32), (0, 0), sigma)[region]
    t = cv2.GaussianBlur(np.asarray(target, np.float32), (0, 0), sigma)[region]
    cm, tm = float(c.mean()), float(t.mean())
    c0, t0 = c - cm, t - tm
    tt = float(np.sum(t0 * t0)) or 1e-12
    a = float(np.clip(np.sum(c0 * t0) / tt, *scale))
    resid = c0 - a * t0
    rmse = float(np.sqrt(np.mean(resid**2)))
    err = float(np.sqrt(np.mean(_excess(resid, tolerance) ** 2)))
    baseline = float(np.sqrt(np.mean(_excess(a * t0, tolerance) ** 2))) or 1e-6
    score = float(np.clip(1 - err / baseline, 0, 1) * 100)
    offset = cm - a * tm
    diff = (current - (a * target + offset)).astype(np.float32)
    within = float(np.mean(np.abs(resid) <= 0.03) * 100)
    return Score(score, within, rmse, offset, a), diff


def half_layout(target: Target, shape: tuple[int, int], box_size: tuple[float, float], relief: float,
                player: int) -> tuple[np.ndarray, np.ndarray]:
    """Jako ``half_grid``, ale vrací (výšky, maska hlavní části)."""
    gh, gw = shape
    own, mask = target.layout((gw, gh), (box_size[1], box_size[0] / 2), relief)

    def turn(a: np.ndarray) -> np.ndarray:
        left = np.ascontiguousarray(a[::-1, :].T)
        return left if player == 0 else np.ascontiguousarray(left[::-1, ::-1])

    return turn(own), turn(mask)


def half_grid(target: Target, shape: tuple[int, int], box_size: tuple[float, float], relief: float,
              player: int) -> np.ndarray:
    """Předloha pro půlku bedny (``shape`` = mřížka půlky), otočená k hráči u kratší strany.

    Předloha se spočítá z pohledu hráče (šířka = kratší strana bedny, hloubka = půlka delší)
    a otočí se do souřadnic plátna. Hráč 1 dostane stejnou předlohu otočenou o 180°.
    """
    gh, gw = shape
    own = target.grid((gw, gh), (box_size[1], box_size[0] / 2), relief)  # řádek 0 = nejdál od hráče
    left = np.ascontiguousarray(own[::-1, :].T)  # hráč 0 stojí u x = 0, jeho pravá ruka míří k +y
    return left if player == 0 else np.ascontiguousarray(left[::-1, ::-1])
