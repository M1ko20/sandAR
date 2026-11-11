"""Hloubkové senzory (Kinect v2, simulace, přehrávání nahrávek)."""

from __future__ import annotations

import logging

from ..config import SensorSettings
from .base import DepthFrame, DepthSensor, Intrinsics, SensorError

log = logging.getLogger(__name__)

__all__ = ["DepthFrame", "DepthSensor", "Intrinsics", "SensorError", "create_sensor"]


def create_sensor(settings: SensorSettings) -> DepthSensor:
    """Vytvoří a spustí senzor podle nastavení.

    ``auto`` zkusí Kinect v2 a při neúspěchu přejde na simulaci, aby šlo
    pracovat i bez hardwaru.
    """
    kind = settings.kind
    if kind == "replay" or (kind == "auto" and settings.replay_file):
        from .replay import ReplaySensor

        sensor: DepthSensor = ReplaySensor(settings.replay_file)
    elif kind in ("kinect2", "auto"):
        from .kinect2 import Kinect2Sensor

        try:
            sensor = Kinect2Sensor(settings.bridge_path, settings.pipeline)
            sensor.start()
            return sensor
        except SensorError as exc:
            if kind == "kinect2":
                raise
            log.warning("Kinect v2 nedostupný (%s) – používám simulaci.", str(exc).splitlines()[0])
            from .simulated import SimulatedSensor

            sensor = SimulatedSensor()
    elif kind == "sim":
        from .simulated import SimulatedSensor

        sensor = SimulatedSensor()
    else:
        raise SensorError(f"Neznámý senzor: {kind}")
    sensor.start()
    return sensor
