"""Kinect v2 přes podproces ``kinect2_bridge`` (C++ / libfreenect2).

Ovladač běží v samostatném procesu: když spadne nebo se Kinect odpojí, aplikace
dál běží a most se po chvíli spustí znovu.
"""

from __future__ import annotations

import logging
import os
import select
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..config import ROOT
from .base import DepthSensor, Intrinsics, SensorError

log = logging.getLogger(__name__)

HEADER = struct.Struct("<4sIII4f")
FRAME_HEADER = struct.Struct("<4sId")


def find_bridge(configured: str = "") -> Path | None:
    candidates = [
        configured,
        os.environ.get("KINECT2_BRIDGE", ""),
        str(ROOT / "build" / "kinect2_bridge" / "kinect2_bridge"),
        shutil.which("kinect2_bridge") or "",
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return Path(c)
    return None


class Kinect2Sensor(DepthSensor):
    name = "kinect2"
    FRAME_TIMEOUT = 15.0  # [s] bez snímku -> restart mostu (most sám končí po 10 s)
    START_TIMEOUT = 15.0  # [s] na spuštění streamu
    STALL_HINT = "Kinect neposílá data – odpoj na 10 s napájení Kinectu (adaptér) a zase ho zapoj"

    def __init__(self, bridge_path: str = "", pipeline: str = "gl") -> None:
        super().__init__()
        bridge = find_bridge(bridge_path)
        if bridge is None:
            raise SensorError(
                "Nenalezen kinect2_bridge. Spusť ./scripts/install_kinect2.sh "
                "(nebo nastav sensor.bridge_path / proměnnou KINECT2_BRIDGE)."
            )
        self.bridge = bridge
        self.pipeline = pipeline
        self._proc: subprocess.Popen | None = None
        self._buf: bytearray | None = None
        self._stderr_tail: list[str] = []
        self._restart_delay = 3.0

    def _spawn(self) -> None:
        self._kill()
        log.info("Spouštím %s --pipeline %s", self.bridge, self.pipeline)
        self._proc = subprocess.Popen(
            [str(self.bridge), "--pipeline", self.pipeline],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=self._pump_stderr, args=(self._proc,), daemon=True).start()
        ready, _, _ = select.select([self._proc.stdout], [], [], self.START_TIMEOUT)
        if not ready:
            self._kill()
            raise SensorError(f"Kinect se nespustil do {self.START_TIMEOUT:.0f} s. {self.STALL_HINT}")
        raw = self._read_exact(HEADER.size)
        if raw is None:
            self._proc.wait(timeout=2)
            tail = "\n".join(self._stderr_tail[-5:])
            raise SensorError(f"kinect2_bridge skončil bez dat (kód {self._proc.returncode}).\n{tail}")
        magic, version, w, h, fx, fy, cx, cy = HEADER.unpack(raw)
        if magic != b"K2BR" or version != 1:
            raise SensorError(f"Neočekávaná hlavička z kinect2_bridge: {magic!r} v{version}")
        self.intrinsics = Intrinsics(w, h, fx, fy, cx, cy)
        self._buf = bytearray(w * h * 4)
        log.info("Kinect v2 připojen: %dx%d fx=%.1f fy=%.1f", w, h, fx, fy)

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for line in iter(proc.stderr.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            self._stderr_tail = (self._stderr_tail + [text])[-20:]
            log.info("[bridge] %s", text)

    def _read_exact(self, n: int, into: bytearray | None = None) -> bytes | bytearray | None:
        proc = self._proc
        if proc is None or proc.stdout is None:  # most byl mezitím ukončen (vypínání)
            return None
        buf = into if into is not None else bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            try:
                k = proc.stdout.readinto(view[got:n])
            except (OSError, ValueError):  # roura zavřená při vypínání
                return None
            if not k:
                return None
            got += k
        return buf

    def _open(self, attempts: int = 5, delay: float = 2.0) -> None:
        # Kinect v2 se po ukončení předchozího programu pár sekund znovu přihlašuje na USB,
        # proto to zkusíme několikrát, než to vzdáme.
        for attempt in range(1, attempts + 1):
            try:
                self._spawn()
                return
            except SensorError as exc:
                if attempt == attempts:
                    raise
                log.warning("Kinect zatím neodpovídá (pokus %d/%d): %s", attempt, attempts, str(exc).splitlines()[0])
                time.sleep(delay)

    def _read(self) -> np.ndarray | None:
        if not self._running:
            return None
        if self._proc is None or self._proc.poll() is not None:
            if self._proc is not None:
                code = self._proc.returncode
                self._proc = None
                hint = self.STALL_HINT if code == 6 else f"most skončil (kód {code})"
                self.error = hint
                log.warning("%s – zkusím to znovu za %.0f s", hint, self._restart_delay)
            self._backoff()
            self._spawn()
        # Hlídač: když Kinect přestane posílat snímky, most restartujeme (s rostoucí pauzou,
        # aby se Kinect stihl po resetu znovu přihlásit na USB).
        proc = self._proc
        if proc is not None and proc.stdout is not None:
            ready, _, _ = select.select([proc.stdout], [], [], self.FRAME_TIMEOUT)
            if not ready:
                self._kill()
                raise SensorError(self.STALL_HINT)
        head = self._read_exact(FRAME_HEADER.size)
        if head is None:
            raise SensorError("kinect2_bridge přestal posílat data, restartuji")
        magic, _seq, _t = FRAME_HEADER.unpack(head)
        if magic != b"FRME":
            self._kill()
            raise SensorError("Rozbitý proud dat z kinect2_bridge, restartuji")
        assert self._buf is not None
        if self._read_exact(len(self._buf), self._buf) is None:
            raise SensorError("kinect2_bridge přestal posílat data, restartuji")
        depth_mm = np.frombuffer(self._buf, dtype="<f4").reshape(self.intrinsics.height, self.intrinsics.width)
        self._restart_delay = 3.0
        return depth_mm * np.float32(0.001)

    def _backoff(self) -> None:
        """Pauza před dalším pokusem (3 s, 6 s, 12 s … max 30 s), přerušitelná vypnutím."""
        until = time.monotonic() + self._restart_delay
        while self._running and time.monotonic() < until:
            time.sleep(0.2)
        self._restart_delay = min(30.0, self._restart_delay * 2)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _close(self) -> None:
        self._kill()
