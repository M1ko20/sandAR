"""Uložení a načtení kalibrace (Kinect + projektor) do config/calibration_<senzor>.json."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from .config import CONFIG_DIR, write_json
from .projector.calibration import ProjectorCalibration
from .sensor.base import Intrinsics
from .terrain.calibration import CalibrationError, KinectCalibration
from .terrain.playarea import PlayArea

log = logging.getLogger(__name__)


def calibration_path(sensor_name: str) -> Path:
    # Nahrávky pocházejí z Kinectu, takže sdílejí jeho kalibraci.
    key = "sim" if sensor_name == "sim" else "kinect2"
    return CONFIG_DIR / f"calibration_{key}.json"


def load_calibration(path: Path, intrinsics: Intrinsics | None = None
                     ) -> tuple[KinectCalibration | None, ProjectorCalibration | None]:
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Kalibraci %s nelze načíst: %s", path, exc)
        return None, None
    kinect = projector = None
    if data.get("kinect"):
        try:
            kd = data["kinect"]
            kinect = KinectCalibration.from_dict(kd)
            if intrinsics is not None and kinect.intrinsics != intrinsics:
                if (intrinsics.width, intrinsics.height) != (kinect.intrinsics.width, kinect.intrinsics.height):
                    log.warning("Kalibrace je pro jiné rozlišení senzoru – je potřeba kalibrovat znovu")
                    kinect = None
                else:  # stejný senzor, jen přesnější parametry z ovladače
                    kinect = KinectCalibration.from_plane(intrinsics, np.array(kd["normal"]), kd["offset"],
                                                          np.array(kd["corners_px"]))
        except (KeyError, CalibrationError, ValueError) as exc:
            log.error("Neplatná kalibrace Kinectu: %s", exc)
            kinect = None
    if data.get("projector"):
        try:
            projector = ProjectorCalibration.from_dict(data["projector"])
        except (KeyError, ValueError) as exc:
            log.error("Neplatná kalibrace projektoru: %s", exc)
    return kinect, projector


def save_calibration(path: Path, kinect: KinectCalibration | None, projector: ProjectorCalibration,
                     play_area: PlayArea | None = None) -> None:
    """Ukládá se vždy CELÁ bedna; hrací plocha (okraje od stěn) zvlášť – ořízne se až při načtení."""
    if play_area is None:
        play_area = load_play_area(path)
    write_json(path, {
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kinect": kinect.to_dict() if kinect else None,
        "projector": projector.to_dict(),
        "play_area": play_area.to_dict() if play_area else None,
    })
    log.info("Kalibrace uložena do %s", path)


def load_play_area(path: Path) -> PlayArea | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8")).get("play_area")
    except (OSError, json.JSONDecodeError):
        return None
    return PlayArea.from_dict(d) if d else None
