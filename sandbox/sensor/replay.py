"""Přehrávání nahraných hloubkových snímků (``python -m sandbox record``).

Hodí se k ladění her doma bez Kinectu, ale se skutečnými daty z pískoviště.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import DepthSensor, Intrinsics, SensorError


def save_recording(path: Path, frames: list[np.ndarray], intrinsics: Intrinsics, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = np.stack([np.round(f * 1000).astype(np.uint16) for f in frames])
    np.savez_compressed(path, depth_mm=depth_mm, fps=fps, **{f"intr_{k}": v for k, v in intrinsics.to_dict().items()})


class ReplaySensor(DepthSensor):
    name = "replay"

    def __init__(self, path: str, loop: bool = True) -> None:
        super().__init__()
        p = Path(path)
        if not p.is_file():
            raise SensorError(f"Nahrávka {p} neexistuje")
        data = np.load(p)
        self._frames = data["depth_mm"]
        self.fps = float(data["fps"])
        self.intrinsics = Intrinsics.from_dict({k[5:]: data[k] for k in data.files if k.startswith("intr_")})
        self.loop = loop
        self._i = 0

    def _read(self) -> np.ndarray | None:
        if self._i >= len(self._frames):
            if not self.loop:
                return None
            self._i = 0
        frame = self._frames[self._i].astype(np.float32) * 0.001
        self._i += 1
        time.sleep(1.0 / self.fps)
        return frame
