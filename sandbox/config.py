"""Nastavení aplikace.

Výchozí hodnoty jsou zde v dataclassech. Uživatel je může přepsat v souboru
``config/settings.json`` – stačí uvést jen to, co chce změnit, např.::

    {"map": {"contour_interval": 0.02}, "display": {"monitor": "BenQ"}}
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("SANDBOX_HOME", Path(__file__).resolve().parents[1]))
CONFIG_DIR = ROOT / "config"
TARGETS_DIR = ROOT / "targets"
RECORDINGS_DIR = ROOT / "recordings"


@dataclass
class SensorSettings:
    kind: str = "auto"  # auto | kinect2 | sim | replay
    bridge_path: str = ""  # prázdné = build/kinect2_bridge/kinect2_bridge
    pipeline: str = "gl"  # gl | cpu  (libfreenect2 dekódování hloubky)
    replay_file: str = ""


@dataclass
class TerrainSettings:
    grid_width: int = 256  # počet buněk výškové mapy na šířku (výška podle poměru stran)
    min_height: float = -0.15  # [m] níž pod základní rovinou = chyba měření
    max_height: float = 0.22  # [m] výš = ruka / předmět, ne písek
    noise_threshold: float = 0.006  # [m] menší změny jsou šum
    stable_frames: int = 8  # kolik snímků musí změna vydržet, než ji přijmeme
    smoothing: float = 0.2  # časové vyhlazení šumu (0..1, víc = rychlejší reakce)
    temporal_window: int = 12  # [snímky] klouzavé okno na buňku (průměr bez min/max); 1 = vypnuto
    hysteresis: float = 0.0015  # [m] menší odchylky od zobrazené výšky se ignorují (klidné vrstevnice)
    spatial_sigma: float = 1.0  # [buňky] prostorové vyhlazení
    hand_margin: float = 0.03  # [m] o kolik nad pískem už je to ruka
    edge_margin: float = 0.05  # [m] pás u stěn bedny, kde Kinect vidí stěnu – výška se doplní zevnitř


@dataclass
class DisplaySettings:
    monitor: str = "auto"  # auto | index (0,1,..) | část názvu ("BenQ")
    windowed: bool = False
    operator_window: bool = True
    canvas_width: int = 1280  # rozlišení vykreslování mapy pískoviště
    mesh_resolution: int = 96  # hustota sítě pro korekci paralaxy
    max_fps: float = 60.0  # strop snímků na Waylandu (tam swap s vsync neblokuje)


@dataclass
class MapSettings:
    elevation_min: float = -0.08  # [m] barva nejnižšího místa
    elevation_max: float = 0.14  # [m] barva nejvyššího místa
    water_level: float = -0.02  # [m] hladina moře (níž = voda)
    contour_interval: float = 0.01  # [m] vrstevnice po 1 cm
    major_every: int = 5  # zesílená vrstevnice každá N-tá
    vertical_scale: float = 5000.0  # 1 cm písku = 50 m ve skutečnosti
    base_altitude: float = 300.0  # nadmořská výška základní roviny [m n. m.]


@dataclass
class ChallengeSettings:
    target_relief: float = 0.10  # [m] převýšení předlohy
    tolerance: float = 0.015  # [m] odchylka, která se počítá jako "sedí"
    success_score: float = 80.0  # [%] od kolika je úkol splněn
    border: float = 0.05  # okraj (podíl rozměru), který se nehodnotí


@dataclass
class WaterSettings:
    rain_rate: float = 0.004  # [m/s] déšť po celém pískovišti
    hand_rain_rate: float = 0.03  # [m/s] déšť pod rukou
    evaporation: float = 0.0004  # [m/s]
    flow_speed: float = 1.0  # násobek gravitace (rychlost toku)


@dataclass
class AttractSettings:
    enabled: bool = True
    idle_seconds: float = 90.0  # [s] bez rukou a změn terénu -> módy se samy střídají
    cycle_seconds: float = 45.0  # [s] délka jedné ukázky v attract mode
    intro_seconds: float = 5.0  # [s] úvodní slajd po výběru módu
    change_threshold: float = 0.01  # [m] změna výšky buňky, která se počítá jako „někdo hrabe“
    change_fraction: float = 0.004  # podíl změněných buněk, od kterého je to aktivita


@dataclass
class LogSettings:
    file: str = "logs/piskoviste.log"  # relativně k ROOT; prázdné = jen konzole
    max_bytes: int = 5_000_000
    backups: int = 5


@dataclass
class Settings:
    sensor: SensorSettings = field(default_factory=SensorSettings)
    terrain: TerrainSettings = field(default_factory=TerrainSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    map: MapSettings = field(default_factory=MapSettings)
    challenge: ChallengeSettings = field(default_factory=ChallengeSettings)
    water: WaterSettings = field(default_factory=WaterSettings)
    attract: AttractSettings = field(default_factory=AttractSettings)
    log: LogSettings = field(default_factory=LogSettings)
    start_game: str = "topo"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        settings = cls()
        _merge_into(settings, data)
        return settings


def _merge_into(obj: Any, data: dict[str, Any], path: str = "") -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise ValueError(f"Neznámé nastavení: {path}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Nastavení {path}{key} má být objekt")
            _merge_into(current, value, f"{path}{key}.")
        else:
            if isinstance(current, float) and isinstance(value, int):
                value = float(value)
            setattr(obj, key, value)


def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_DIR / "settings.json"
    if path.exists():
        return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return Settings()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
