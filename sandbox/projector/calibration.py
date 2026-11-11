"""Kalibrace projektoru vůči pískovišti.

Uživatel přetáhne 4 úchyty v obraze projektoru do rohů bedny (na úrovni
uhlazeného písku). Tím vznikne homografie „souřadnice bedny -> pixely projektoru“
pro rovinu h = 0.

Projektor ale svítí z jednoho bodu, takže kopec vysoký h se promítne posunutý.
To korigujeme paralaxou: bod (x, y, h) leží na stejném paprsku projektoru jako
bod roviny  G = P + (P - C) · h / (Hp - h),  kde C je průmět projektoru do roviny a
Hp jeho výška nad pískem. Výchozí poloha = poloha Kinectu (bývají vedle sebe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ProjectorCalibration:
    resolution: tuple[int, int] = (1280, 800)
    corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))  # px, pořadí rohů 1..4
    position: tuple[float, float] | None = None  # průmět projektoru do roviny [m, souř. bedny]
    height: float = 0.0  # [m] nad pískem; 0 = bez korekce paralaxy

    @classmethod
    def default(cls, resolution: tuple[int, int], inset: float = 0.0) -> "ProjectorCalibration":
        w, h = resolution
        dx, dy = w * inset, h * inset
        corners = np.array([[dx, dy], [w - dx, dy], [w - dx, h - dy], [dx, h - dy]], float)
        return cls(tuple(resolution), corners)

    def homography(self, box_size: tuple[float, float]) -> np.ndarray:
        W, D = box_size
        rect = np.array([[0, 0], [W, 0], [W, D], [0, D]], np.float32)
        return cv2.getPerspectiveTransform(rect, self.corners.astype(np.float32)).astype(float)

    def ground_point(self, x, y, h, box_size: tuple[float, float]):
        """Korekce paralaxy: bod písku (x, y, h) -> bod roviny se stejným pixelem projektoru."""
        if self.height <= 0:
            return np.asarray(x, float), np.asarray(y, float)
        cx, cy = self.position if self.position else (box_size[0] / 2, box_size[1] / 2)
        k = np.asarray(h, float) / np.maximum(self.height - np.asarray(h, float), 1e-3)
        return x + (x - cx) * k, y + (y - cy) * k

    def box_to_projector(self, x, y, h, box_size: tuple[float, float]) -> np.ndarray:
        gx, gy = self.ground_point(np.asarray(x, float), np.asarray(y, float), h, box_size)
        H = self.homography(box_size)
        w = H[2, 0] * gx + H[2, 1] * gy + H[2, 2]
        return np.stack([(H[0, 0] * gx + H[0, 1] * gy + H[0, 2]) / w,
                         (H[1, 0] * gx + H[1, 1] * gy + H[1, 2]) / w], axis=-1)

    def scaled_to(self, resolution: tuple[int, int]) -> "ProjectorCalibration":
        """Přepočet na jiné rozlišení projektoru (rohy jsou v pixelech)."""
        if tuple(resolution) == tuple(self.resolution):
            return self
        sx = resolution[0] / self.resolution[0]
        sy = resolution[1] / self.resolution[1]
        return ProjectorCalibration(tuple(resolution), self.corners * [sx, sy], self.position, self.height)

    def to_dict(self) -> dict:
        return {
            "resolution": list(self.resolution),
            "corners_px": self.corners.tolist(),
            "position_m": list(self.position) if self.position else None,
            "height_m": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectorCalibration":
        pos = d.get("position_m")
        return cls(tuple(d["resolution"]), np.array(d["corners_px"], float),
                   tuple(pos) if pos else None, float(d.get("height_m", 0.0)))
