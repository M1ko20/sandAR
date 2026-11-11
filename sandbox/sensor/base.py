"""Společné rozhraní hloubkových senzorů.

Senzor dělá jedinou věc: dodává hloubkové snímky (metry, 0 = neplatné) a zná
vnitřní parametry své kamery. Nic neví o pískovišti ani o projektoru.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


class SensorError(RuntimeError):
    pass


@dataclass(frozen=True)
class Intrinsics:
    """Model dírkové kamery (snímek je už bez zkreslení objektivu)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def rays(self) -> np.ndarray:
        """Směrové vektory (x/z, y/z, 1) pro každý pixel, tvar (H, W, 3)."""
        u = (np.arange(self.width, dtype=np.float32) - self.cx) / self.fx
        v = (np.arange(self.height, dtype=np.float32) - self.cy) / self.fy
        rays = np.ones((self.height, self.width, 3), np.float32)
        rays[..., 0] = u[None, :]
        rays[..., 1] = v[:, None]
        return rays

    def project(self, points: np.ndarray) -> np.ndarray:
        """3D body v souřadnicích kamery (…, 3) -> pixely (…, 2)."""
        z = points[..., 2]
        u = points[..., 0] / z * self.fx + self.cx
        v = points[..., 1] / z * self.fy + self.cy
        return np.stack([u, v], axis=-1)

    def to_dict(self) -> dict:
        return dict(width=self.width, height=self.height, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]))


# Typické hodnoty Kinectu v2 (pro simulaci a jako záloha).
KINECT2_DEFAULT_INTRINSICS = Intrinsics(512, 424, 365.5, 365.5, 256.0, 212.0)


@dataclass
class DepthFrame:
    depth: np.ndarray  # float32 (H, W), metry, 0 = neplatné
    seq: int
    timestamp: float = field(default_factory=time.monotonic)


class DepthSensor(ABC):
    """Základ pro senzory. Snímky čte vlákno na pozadí, aplikace si bere nejnovější."""

    name = "senzor"

    def __init__(self) -> None:
        self.intrinsics: Intrinsics = KINECT2_DEFAULT_INTRINSICS
        self._frame: DepthFrame | None = None
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._running = False
        self._seq = 0
        self.error: str | None = None

    # --- rozhraní pro potomky -------------------------------------------------
    def _open(self) -> None:
        """Otevře zařízení a nastaví self.intrinsics. Volá se před spuštěním vlákna."""

    @abstractmethod
    def _read(self) -> np.ndarray | None:
        """Blokující čtení jednoho snímku (metry). None = konec streamu."""

    def _close(self) -> None:
        pass

    # --- veřejné API ----------------------------------------------------------
    def start(self) -> None:
        self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"sensor-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._close()
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._cond:
            self._cond.notify_all()

    def latest(self) -> DepthFrame | None:
        return self._frame

    def wait_frame(self, after_seq: int, timeout: float = 0.5) -> DepthFrame | None:
        """Počká na snímek novější než after_seq."""
        with self._cond:
            self._cond.wait_for(lambda: (self._frame is not None and self._frame.seq > after_seq) or not self._running,
                                timeout=timeout)
            frame = self._frame
        return frame if frame is not None and frame.seq > after_seq else None

    @property
    def running(self) -> bool:
        return self._running

    # Interakce s pískem (jen simulace, skutečný písek se hrabe rukama :-)
    supports_interaction = False

    def interact(self, u: float, v: float, amount: float, radius: float = 0.05) -> None:
        pass

    def _loop(self) -> None:
        while self._running:
            try:
                depth = self._read()
            except Exception as exc:  # senzor nesmí shodit aplikaci
                if not self._running:  # chyby při vypínání nejsou chyby
                    break
                if isinstance(exc, SensorError):
                    log.warning("%s: %s", self.name, exc)
                else:
                    log.exception("Chyba senzoru %s", self.name)
                self.error = str(exc)
                time.sleep(1.0)
                continue
            self.error = None
            if depth is None:
                break
            self._seq += 1
            with self._cond:
                self._frame = DepthFrame(depth, self._seq)
                self._cond.notify_all()
        self._running = False
        with self._cond:
            self._cond.notify_all()
