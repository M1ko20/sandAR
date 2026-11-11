"""Vlákno, které spojuje senzor a zpracování terénu.

Aplikace si od něj jen bere poslední hotovou výškovou mapu, takže vykreslování
(60 Hz) nikdy nečeká na Kinect (30 Hz) ani na výpočty.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..config import TerrainSettings
from ..sensor.base import DepthFrame, DepthSensor
from .calibration import KinectCalibration
from .heightmap import HeightMap
from .processor import TerrainProcessor

log = logging.getLogger(__name__)


class TerrainService:
    def __init__(self, sensor: DepthSensor, settings: TerrainSettings,
                 calibration: KinectCalibration | None = None) -> None:
        self.sensor = sensor
        self.settings = settings
        self._lock = threading.Lock()
        self._processor: TerrainProcessor | None = None
        self._heightmap: HeightMap | None = None
        self._depth: DepthFrame | None = None
        self.frozen = False
        self.sensor_fps = 0.0
        self._capture: dict | None = None
        self._running = True
        self.set_calibration(calibration)
        self._thread = threading.Thread(target=self._loop, name="terrain", daemon=True)
        self._thread.start()

    # --- veřejné API ---------------------------------------------------------
    @property
    def calibration(self) -> KinectCalibration | None:
        p = self._processor
        return p.cal if p else None

    @property
    def processor(self) -> TerrainProcessor | None:
        return self._processor

    def set_calibration(self, calibration: KinectCalibration | None) -> None:
        with self._lock:
            self._processor = TerrainProcessor(calibration, self.settings) if calibration else None
            self._heightmap = None

    def latest_heightmap(self) -> HeightMap | None:
        return self._heightmap

    def latest_depth(self) -> DepthFrame | None:
        return self._depth

    def start_capture(self, frames: int = 30) -> None:
        """Začne průměrovat snímky (pro kalibraci roviny)."""
        with self._lock:
            self._capture = {"needed": frames, "frames": []}

    def capture_progress(self) -> float:
        c = self._capture
        if c is None:
            return 0.0
        return len(c["frames"]) / c["needed"]

    def capture_result(self) -> np.ndarray | None:
        """Medián zachycených snímků, nebo None, pokud ještě není hotovo."""
        with self._lock:
            c = self._capture
            if c is None or len(c["frames"]) < c["needed"]:
                return None
            self._capture = None
        stack = np.stack(c["frames"]).astype(np.float32)
        stack[stack <= 0] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(stack, axis=0)
        return np.nan_to_num(med, nan=0.0).astype(np.float32)

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    # --- vlákno ------------------------------------------------------------------
    def _loop(self) -> None:
        last_seq = 0
        t_last = time.monotonic()
        while self._running:
            frame = self.sensor.wait_frame(last_seq, timeout=0.5)
            if frame is None:
                continue
            last_seq = frame.seq
            now = time.monotonic()
            self.sensor_fps = 0.9 * self.sensor_fps + 0.1 / max(now - t_last, 1e-3)
            t_last = now
            self._depth = frame
            with self._lock:
                if self._capture is not None and len(self._capture["frames"]) < self._capture["needed"]:
                    self._capture["frames"].append(frame.depth.copy())
                processor = self._processor
            if processor is None or self.frozen:
                continue
            try:
                hm = processor.process(frame.depth, frame.seq)
            except Exception:
                log.exception("Chyba při zpracování terénu")
                continue
            with self._lock:
                if processor is self._processor:  # mezitím se nezměnila kalibrace
                    self._heightmap = hm
