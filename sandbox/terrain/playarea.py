"""Hrací plocha: část bedny, kde je opravdu písek a kam projektor dosvítí.

Kalibrace Kinectu najde rohy bedny podle stěn. Stěny mají ale úkos a Kinect u nich
měří falešné výšky (odrazy), a projektor nemusí pokrýt celou bednu. Kdyby plátno
sahalo až ke stěnám, mapy, texty i hry by se promítaly na stěny nebo mimo obraz.

Proto z celé bedny vyřízneme obdélník:
* od každé stěny odsadíme okraj (odhadne se z hloubky: kde výška přestane „skákat“),
* a zmenšíme ho tak, aby ležel celý uvnitř obrazu projektoru.

Výsledkem jsou nové (oříznuté) kalibrace Kinectu i projektoru – zbytek aplikace
pak nic dalšího řešit nemusí: plátno = viditelná hrací plocha.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ..projector.calibration import ProjectorCalibration
from .calibration import KinectCalibration

log = logging.getLogger(__name__)


@dataclass
class PlayArea:
    """Okraje od stěn [m] v souřadnicích celé bedny: x=0, y=0, x=W, y=D."""

    margins: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    frame_inset_px: float = 6.0  # odstup od okraje obrazu projektoru

    def to_dict(self) -> dict:
        return {"margins_m": [round(float(m), 4) for m in self.margins], "frame_inset_px": self.frame_inset_px}

    @classmethod
    def from_dict(cls, d: dict) -> "PlayArea":
        m = d.get("margins_m", [0, 0, 0, 0])
        return cls(tuple(float(v) for v in m), float(d.get("frame_inset_px", 6.0)))  # type: ignore[arg-type]


def estimate_margins(heights: np.ndarray, valid: np.ndarray, size: tuple[float, float],
                     jump: float = 0.012, safety: float = 0.01, max_margin: float = 0.15
                     ) -> tuple[float, float, float, float]:
    """Okraje od stěn z výškové mapy celé bedny (bez vyhlazení a bez doplňování okrajů).

    Pro každou stranu se vezme medián výšky podél stěny (prostředních 60 %) podle
    vzdálenosti od stěny. U stěny (úkos, odrazy ToF) výška mezi sousedními buňkami skáče
    o centimetry, písek se mění pozvolna – okraj končí za posledním skokem > `jump`
    (nebo za chybějícími daty), plus `safety`."""
    h = np.where(valid, heights, np.nan).astype(np.float32)
    gh, gw = h.shape
    cw, ch = size[0] / gw, size[1] / gh
    n_x, n_y = int(np.ceil(max_margin / cw)), int(np.ceil(max_margin / ch))
    mid_r, mid_c = slice(int(gh * 0.2), int(gh * 0.8)), slice(int(gw * 0.2), int(gw * 0.8))
    with _quiet():
        profiles = [
            (np.nanmedian(h[mid_r, :n_x], axis=0), cw),             # x = 0
            (np.nanmedian(h[:n_y, mid_c], axis=1), ch),             # y = 0
            (np.nanmedian(h[mid_r, -n_x:], axis=0)[::-1], cw),      # x = W
            (np.nanmedian(h[-n_y:, mid_c], axis=1)[::-1], ch),      # y = D
        ]
    out = []
    for prof, cell in profiles:
        bad = ~np.isfinite(prof)
        steps = np.abs(np.diff(prof)) > jump
        bad[:-1] |= np.nan_to_num(steps, nan=True).astype(bool)
        idx = np.flatnonzero(bad)
        m = (idx[-1] + 1) * cell + safety if len(idx) else safety
        out.append(float(min(m, max_margin)))
    return out[0], out[1], out[2], out[3]


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def visible_rect(kc: KinectCalibration, pc: ProjectorCalibration | None, area: PlayArea
                 ) -> tuple[float, float, float, float]:
    """Největší osově rovnoběžný obdélník (x0, y0, x1, y1) [m] uvnitř bedny bez okrajů
    a zároveň uvnitř obrazu projektoru (na úrovni písku h = 0)."""
    W, D = kc.size
    l, t, r, b = area.margins
    x0, y0, x1, y1 = l, t, W - r, D - b
    if pc is None or not np.any(pc.corners):
        return x0, y0, x1, y1
    # obraz projektoru v souřadnicích bedny (čtyřúhelník)
    Hinv = np.linalg.inv(pc.homography(kc.size))
    pw, ph = pc.resolution
    k = area.frame_inset_px
    frame = np.array([[[k, k]], [[pw - k, k]], [[pw - k, ph - k]], [[k, ph - k]]], np.float32)
    quad = cv2.perspectiveTransform(frame, Hinv)[:, 0].astype(float)
    center = quad.mean(axis=0)
    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]

    def side(a: np.ndarray, b_: np.ndarray, p: np.ndarray) -> float:
        e = b_ - a
        return float(e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]))

    def outside(p: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Hrany, za kterými bod leží (na opačné straně než těžiště)."""
        return [(a, b_) for a, b_ in edges if side(a, b_, p) * side(a, b_, center) < -1e-12]

    step = 0.002
    for _ in range(2000):
        corners = {"tl": np.array([x0, y0]), "tr": np.array([x1, y0]), "br": np.array([x1, y1]), "bl": np.array([x0, y1])}
        moved = False
        for name, p in corners.items():
            for a, b_ in outside(p):
                e = b_ - a
                if abs(e[1]) > abs(e[0]):  # spíš svislá hrana -> posunout levý/pravý okraj
                    if "l" in name:
                        x0 += step
                    else:
                        x1 -= step
                else:
                    if "t" in name:
                        y0 += step
                    else:
                        y1 -= step
                moved = True
        if not moved or x1 - x0 < 0.1 or y1 - y0 < 0.1:
            break
    return x0, y0, x1, y1


def crop(kc: KinectCalibration, pc: ProjectorCalibration | None, rect: tuple[float, float, float, float]
         ) -> tuple[KinectCalibration, ProjectorCalibration | None]:
    """Kalibrace, ve kterých je „bedna“ jen obdélník `rect` (souřadnice celé bedny)."""
    x0, y0, x1, y1 = rect
    xs, ys = np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1])
    corners_px = kc.box_to_pixel(xs, ys, np.zeros(4))
    kc2 = KinectCalibration.from_plane(kc.intrinsics, kc.normal, kc.offset, corners_px)
    if pc is None:
        return kc2, None
    proj = ProjectorCalibration(pc.resolution, pc.corners, pc.position, 0.0)  # rohy jen z roviny h = 0
    corners = proj.box_to_projector(xs, ys, np.zeros(4), kc.size)
    pos = (pc.position[0] - x0, pc.position[1] - y0) if pc.position else None
    return kc2, ProjectorCalibration(pc.resolution, np.asarray(corners, float), pos, pc.height)


def apply(kc: KinectCalibration | None, pc: ProjectorCalibration | None, area: PlayArea | None
          ) -> tuple[KinectCalibration | None, ProjectorCalibration | None]:
    if kc is None or area is None:
        return kc, pc
    rect = visible_rect(kc, pc, area)
    W, D = kc.size
    if rect == (0.0, 0.0, W, D):
        return kc, pc
    log.info("Hrací plocha %.0f × %.0f cm (z bedny %.0f × %.0f cm)", (rect[2] - rect[0]) * 100,
             (rect[3] - rect[1]) * 100, W * 100, D * 100)
    return crop(kc, pc, rect)
