"""Skutečná krajina: ortofoto a výškový model míst v ČR z bezplatných služeb ČÚZK.

* Ortofoto ČR – ArcGIS MapServer ``ORTOFOTO`` (barevný letecký snímek, ~0,25 m/px),
* DMR 5G – ArcGIS ImageServer ``dmr5g`` (digitální model reliéfu, výšky v m n. m.).

Obojí jsou otevřená data ČÚZK (CC BY 4.0, uvádí se „© ČÚZK“). Na akci často není internet,
proto se data stahují předem příkazem ``python -m sandbox ortofoto`` do ``data/ortofoto/``
a mód „Skutečná krajina“ pak běží offline.

Výřez se počítá v Mercatorově projekci (EPSG:3857) kolem středu místa; poměr stran 5 : 3
odpovídá bedně, mód si ho případně ořízne na přesný poměr hrací plochy.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT

log = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "ortofoto"
ORTO_URL = "https://ags.cuzk.cz/arcgis1/rest/services/ORTOFOTO/MapServer/export"
DMR_URL = "https://ags.cuzk.cz/arcgis2/rest/services/dmr5g/ImageServer/exportImage"
ASPECT = 5 / 3  # šířka : výška výřezu
PHOTO_W = 1600
DEM_W = 400
USER_AGENT = "sandbox-ortofoto/1.0"


@dataclass(frozen=True)
class Place:
    slug: str
    name: str
    lat: float
    lon: float
    width_m: float  # šířka výřezu ve skutečnosti
    note: str  # jedna věta pro návštěvníky


PLACES = [
    Place("rip", "Říp", 50.3867, 14.2878, 2600, "Osamělá hora, na kterou podle pověsti vystoupil praotec Čech."),
    Place("milesovka", "Milešovka", 50.5552, 13.9316, 2600, "Největrnější hora Česka – sopečný kužel v Českém středohoří."),
    Place("bezdez", "Bezděz", 50.5385, 14.7206, 2400, "Dva sopečné vrchy, na vyšším stojí hrad."),
    Place("macocha", "Propast Macocha", 49.3733, 16.7290, 1500, "138 m hluboká propast v Moravském krasu."),
    Place("lysa", "Lysá hora", 49.5458, 18.4474, 4500, "Nejvyšší hora Beskyd s hlubokými údolími kolem."),
    Place("praha", "Pražský hrad a Vltava", 50.0880, 14.3985, 3000, "Vltava se vine pod Hradčany a Petřínem."),
]


def _mercator(lat: float, lon: float) -> tuple[float, float]:
    R = 6378137.0
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def bbox_3857(place: Place) -> tuple[float, float, float, float]:
    """Výřez v EPSG:3857 (Mercator zvětšuje měřítko 1/cos φ – přepočet ze skutečných metrů)."""
    x, y = _mercator(place.lat, place.lon)
    k = 1 / math.cos(math.radians(place.lat))
    hw = place.width_m * k / 2
    hh = hw / ASPECT
    return x - hw, y - hh, x + hw, y + hh


def _get(url: str, params: dict, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        ctype = resp.headers.get("Content-Type", "")
    if "json" in ctype or data[:1] == b"{":
        raise RuntimeError(f"Služba vrátila chybu: {data[:200]!r}")
    return data


def download(place: Place, directory: Path = DATA_DIR) -> Path:
    """Stáhne ortofoto (JPEG) a výškový model (float32 .npy, m n. m.) jednoho místa."""
    out = directory / place.slug
    out.mkdir(parents=True, exist_ok=True)
    bbox = ",".join(f"{v:.2f}" for v in bbox_3857(place))
    ph = int(round(PHOTO_W / ASPECT))
    photo = _get(ORTO_URL, {"bbox": bbox, "bboxSR": 3857, "imageSR": 3857, "size": f"{PHOTO_W},{ph}",
                            "format": "jpg", "transparent": "false", "f": "image"})
    img = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Ortofoto se nepodařilo dekódovat")
    cv2.imwrite(str(out / "orto.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    dh = int(round(DEM_W / ASPECT))
    tif = _get(DMR_URL, {"bbox": bbox, "bboxSR": 3857, "imageSR": 3857, "size": f"{DEM_W},{dh}",
                         "format": "tiff", "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation",
                         "f": "image"})
    try:  # GeoTIFF má značky, kterým OpenCV nerozumí – jen by spamoval varováními
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    dem = cv2.imdecode(np.frombuffer(tif, np.uint8), cv2.IMREAD_UNCHANGED)
    if dem is None or dem.ndim != 2:
        raise RuntimeError("Výškový model se nepodařilo dekódovat")
    dem = dem.astype(np.float32)
    bad = (dem < -100) | (dem > 3000) | ~np.isfinite(dem)
    if bad.mean() > 0.2:
        raise RuntimeError("Výškový model má příliš mnoho děr (místo u hranice?)")
    if bad.any():
        dem[bad] = float(np.median(dem[~bad]))
    np.save(out / "dem.npy", dem)
    info = {"name": place.name, "note": place.note, "lat": place.lat, "lon": place.lon, "width_m": place.width_m,
            "height_m": place.width_m / ASPECT, "min_m": float(dem.min()), "max_m": float(dem.max()),
            "source": "© ČÚZK – Ortofoto ČR, DMR 5G (CC BY 4.0)"}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def download_all(directory: Path = DATA_DIR) -> list[str]:
    """Stáhne všechna přednastavená místa; vrací hlášení pro konzoli."""
    lines = []
    for place in PLACES:
        try:
            path = download(place, directory)
            info = json.loads((path / "info.json").read_text(encoding="utf-8"))
            lines.append(f"OK  {place.name}: {info['min_m']:.0f}–{info['max_m']:.0f} m n. m. -> {path}")
        except Exception as exc:  # jedno místo nesmí zastavit ostatní
            log.warning("%s: %s", place.name, exc)
            lines.append(f"CHYBA {place.name}: {exc}")
    return lines


@dataclass
class Landscape:
    slug: str
    name: str
    note: str
    photo: np.ndarray  # RGB uint8
    dem: np.ndarray  # m n. m.
    min_m: float
    max_m: float
    width_m: float
    source: str


def load_all(directory: Path = DATA_DIR) -> list[Landscape]:
    """Stažená místa (v pořadí PLACES, pak případná další)."""
    if not directory.is_dir():
        return []
    order = {p.slug: i for i, p in enumerate(PLACES)}
    out = []
    for d in sorted(directory.iterdir(), key=lambda p: (order.get(p.name, 99), p.name)):
        try:
            info = json.loads((d / "info.json").read_text(encoding="utf-8"))
            img = cv2.imread(str(d / "orto.jpg"), cv2.IMREAD_COLOR)
            dem = np.load(d / "dem.npy").astype(np.float32)
        except (OSError, ValueError, KeyError):
            continue
        if img is None:
            continue
        out.append(Landscape(d.name, info["name"], info.get("note", ""), cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dem,
                             float(info["min_m"]), float(info["max_m"]), float(info["width_m"]),
                             info.get("source", "© ČÚZK")))
    return out


def crop_to_aspect(a: np.ndarray, aspect: float) -> np.ndarray:
    """Středový výřez pole (H, W, …) na poměr stran šířka/výška = aspect."""
    h, w = a.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return a[:, x0:x0 + nw]
    nh = int(round(w / aspect))
    y0 = (h - nh) // 2
    return a[y0:y0 + nh]
