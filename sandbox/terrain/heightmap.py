"""Výšková mapa – jediný datový objekt, který hry dostávají od senzorové části."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .hands import Hand


@dataclass
class HeightMap:
    """Výšky písku nad základní rovinou v metrech na pravidelné mřížce.

    ``heights[iy, ix]`` je výška buňky se středem v
    ``x = (ix + 0.5) * cell_w``, ``y = (iy + 0.5) * cell_h``.
    Souřadnice: počátek v rohu 1 bedny, x k rohu 2, y k rohu 4 (viz kalibrace).
    """

    heights: np.ndarray  # float32 (GH, GW)
    size: tuple[float, float]  # rozměr bedny (šířka x, hloubka y) v metrech
    hands: np.ndarray | None = None  # bool (GH, GW) – nad buňkou je ruka/předmět
    hand_points: list[tuple[float, float]] = field(default_factory=list)  # středy dlaní [m, souř. bedny]
    hand_list: list["Hand"] = field(default_factory=list)  # sledované ruce (id, dlaň, špička, výška, rychlost)
    seq: int = 0
    timestamp: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def has_hands(self) -> bool:
        """Je nad pískem aspoň jedna ruka?"""
        if self.hand_list:
            return True
        return self.hands is not None and int(self.hands.sum()) > 12

    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape  # type: ignore[return-value]

    @property
    def cell_size(self) -> tuple[float, float]:
        gh, gw = self.heights.shape
        return self.size[0] / gw, self.size[1] / gh

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        gh, gw = self.heights.shape
        cw, ch = self.cell_size
        return np.meshgrid((np.arange(gw) + 0.5) * cw, (np.arange(gh) + 0.5) * ch)

    def sample(self, x: float, y: float) -> float:
        gh, gw = self.heights.shape
        ix = int(np.clip(x / self.size[0] * gw, 0, gw - 1))
        iy = int(np.clip(y / self.size[1] * gh, 0, gh - 1))
        return float(self.heights[iy, ix])
