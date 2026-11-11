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
    generator: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None  # (X, Y) -> 0..1
    path: Path | None = None
    absolute: bool = False  # True = výšky jsou v metrech (nepřeškálovávat)

    def grid(self, shape: tuple[int, int], box_size: tuple[float, float], relief: float) -> np.ndarray:
        """Výšky předlohy v metrech na mřížce, se střední hodnotou 0 (objem písku je daný)."""
        gh, gw = shape
        if self.generator is not None:
            aspect = box_size[0] / box_size[1]
            X, Y = np.meshgrid((np.arange(gw) + 0.5) / gw * aspect, (np.arange(gh) + 0.5) / gh)
            t = self.generator(X - aspect / 2, Y - 0.5).astype(np.float32)
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
        if not self.absolute:
            span = float(t.max() - t.min())
            t = (t - t.min()) / span * relief if span > 0 else np.zeros_like(t)
        return (t - t.mean()).astype(np.float32)


def _gauss(X, Y, cx, cy, s):
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))


def _volcano(X, Y):
    r = np.hypot(X, Y)
    cone = np.clip(1 - r / 0.42, 0, None) ** 1.3
    return cone - 0.55 * _gauss(X, Y, 0, 0, 0.055)


def _valley(X, Y):
    river = 0.12 * np.sin(X * 4.5)
    return np.clip(np.abs(Y - river) / 0.45, 0, 1) ** 0.8


def _ridge(X, Y):
    d = (X * 0.6 - Y * 0.8)  # vzdálenost od úhlopříčky
    return np.exp(-(d**2) / (2 * 0.1**2)) * (0.75 + 0.25 * np.cos(X * 3))


def _island(X, Y):
    island = _gauss(X, Y, 0, 0, 0.2)
    bay = _gauss(X, Y, 0.18, 0.08, 0.08)
    return np.clip(island - 0.8 * bay, 0, None)


def _basin(X, Y):
    return 1 - _gauss(X, Y, 0, 0, 0.18) - 0.3 * _gauss(X, Y, 0.0, 0.0, 0.5)


BUILTIN = [
    Target("Kopec", lambda X, Y: _gauss(X, Y, 0, 0, 0.2)),
    Target("Sopka s kráterem", _volcano),
    Target("Dva vrcholy a sedlo", lambda X, Y: _gauss(X, Y, -0.3, 0, 0.15) + 0.8 * _gauss(X, Y, 0.3, 0.02, 0.15)),
    Target("Údolí s řekou", _valley),
    Target("Horský hřbet", _ridge),
    Target("Ostrov se zálivem", _island),
    Target("Jezero v kotlině", _basin),
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


@dataclass
class Score:
    score: float  # 0..100, 100 = přesná shoda
    within: float  # podíl buněk v toleranci [%]
    rmse: float  # [m]
    offset: float  # [m] posun hladiny (nepočítá se do chyby)


def evaluate(current: np.ndarray, target: np.ndarray, tolerance: float, border: float = 0.05) -> tuple[Score, np.ndarray]:
    """Porovná terén s předlohou.

    Skóre = 1 − RMSE / RMSE(rovný písek), takže uhlazený písek má 0 %, přesná kopie
    100 %. Celkový posun výšky se odečte (množství písku žák neovlivní).
    Vrací (skóre, rozdíl current − target po odečtení posunu).
    """
    gh, gw = target.shape
    by, bx = int(gh * border), int(gw * border)
    inner = (slice(by, gh - by or None), slice(bx, gw - bx or None))
    diff = current - target
    offset = float(np.median(diff[inner]))
    diff = diff - offset
    d = diff[inner]
    rmse = float(np.sqrt(np.mean(d**2)))
    t = target[inner]
    baseline = float(np.sqrt(np.mean((t - t.mean()) ** 2))) or 1e-6
    score = float(np.clip(1 - rmse / baseline, 0, 1) * 100)
    within = float(np.mean(np.abs(d) <= tolerance) * 100)
    return Score(score, within, rmse, offset), diff.astype(np.float32)
