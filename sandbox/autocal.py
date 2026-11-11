"""Automatická kalibrace a kontrola projekce přes barevnou kameru Kinectu.

Projektor promítá značky, barevná kamera Kinectu (srovnaná s hloubkovou mřížkou
nástrojem ``kinect2_snap``) je najde a hloubka řekne, kde v prostoru leží.

* ``autocalibrate``: rohy bedny podle stěn (hloubka) + model projektoru z ~70 bodů
  (homografie roviny písku + výška a poloha projektoru pro korekci paralaxy).
* ``checkcal``: nezávislá kontrola uložené kalibrace – přesnost bodů v cm (i na kopcích,
  s korekcí paralaxy a bez ní) a fotky promítnuté mřížky a vrstevnic proti hloubce.

Kinect v tu chvíli nesmí používat jiný program (pískoviště musí být vypnuté).
"""

from __future__ import annotations

import logging
import struct
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration_store import calibration_path, load_calibration, load_play_area, save_calibration
from .config import TerrainSettings
from .terrain import playarea
from .config import ROOT, Settings
from .projector import ProjectorCalibration, ProjectorCompositor
from .render.host import RenderHost
from .render.window import Window, find_projector_monitor, poll_events, terminate_glfw
from .sensor.base import Intrinsics
from .terrain.calibration import CalibrationError, KinectCalibration, detect_box_corners_walls, unmirror_corners
from .terrain.heightmap import HeightMap
from .terrain.processor import TerrainProcessor

log = logging.getLogger(__name__)

SNAP_BIN = ROOT / "build" / "kinect2_bridge" / "kinect2_snap"
W_PX, H_PX = 512, 424


class ColorSnap:
    """Podproces kinect2_snap: na požádání pošle barevný snímek (BGR) a hloubku [m] na stejné mřížce."""

    def __init__(self, skip: int = 6) -> None:
        if not SNAP_BIN.exists():
            raise RuntimeError(f"Chybí {SNAP_BIN} – spusť ./scripts/install_kinect2.sh --bridge")
        self.proc = subprocess.Popen([str(SNAP_BIN), "--skip", str(skip)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        assert self.proc.stdout is not None
        head = self.proc.stdout.read(28)
        if len(head) != 28 or head[:4] != b"K2SN":
            raise RuntimeError("kinect2_snap se nespustil (běží jiný program s Kinectem?)")
        _, w, h, fx, fy, cx, cy = struct.unpack("<4sII4f", head)
        self.intrinsics = Intrinsics(w, h, fx, fy, cx, cy)

    def grab(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(b"c\n")
        self.proc.stdin.flush()
        if self.proc.stdout.read(4) != b"SNAP":
            raise RuntimeError("kinect2_snap přestal posílat data")
        bgr = np.frombuffer(self.proc.stdout.read(W_PX * H_PX * 3), np.uint8).reshape(H_PX, W_PX, 3).copy()
        depth = np.frombuffer(self.proc.stdout.read(W_PX * H_PX * 4), np.float32).reshape(H_PX, W_PX) / 1000.0
        return bgr, depth.astype(np.float32)

    def grab_depth_median(self, n: int = 15) -> np.ndarray:
        st = np.stack([self.grab()[1] for _ in range(n)])
        st[st <= 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nan_to_num(np.nanmedian(st, axis=0)).astype(np.float32)

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class PatternWindow:
    """Celoobrazovkové okno na projektoru pro testovací vzory."""

    def __init__(self, settings: Settings) -> None:
        mon = find_projector_monitor(settings.display.monitor)
        if mon is None:
            raise RuntimeError("Projektor (druhý monitor) nenalezen")
        self.win = Window("Pískoviště – kalibrace", (1280, 800), monitor=mon, vsync=False)
        self.win.set_cursor_visible(False)
        self.size = self.win.framebuffer_size
        self.host = RenderHost(self.win.ctx, settings)
        self.compositor = ProjectorCompositor(self.win.ctx, settings.display.mesh_resolution)

    def _settle(self, seconds: float) -> None:
        t = time.monotonic()
        while time.monotonic() - t < seconds:
            poll_events()
            time.sleep(0.02)

    def squares(self, rects: list[tuple[float, float, float, float]] = (), bg: float = 0.0,  # type: ignore[assignment]
                settle: float = 0.25) -> None:
        """Bílé obdélníky (x, y, w, h v pixelech projektoru) na pozadí šedi `bg`."""
        ctx = self.win.ctx
        for _ in range(3):  # pár snímků, ať kompozitor zobrazí nový obraz
            fb = self.win.begin()
            ctx.scissor = None
            fb.clear(bg, bg, bg, 1)
            for x, y, w, h in rects:
                ctx.scissor = (int(x), int(self.size[1] - y - h), int(w), int(h))
                fb.clear(1, 1, 1, 1)
            ctx.scissor = None
            self.win.swap()
            poll_events()
        self._settle(settle)

    def canvas_pattern(self, frag: str, cal: ProjectorCalibration, hm: HeightMap, settle: float = 0.6,
                       parallax: bool = True) -> None:
        """Vykreslí shader na plátno bedny a promítne ho stejnou cestou jako aplikace."""
        host = self.host
        host.update_heightmap(hm)
        prog = host.program(frag)
        if not parallax:
            cal = ProjectorCalibration(cal.resolution, cal.corners, cal.position, 0.0)
        for _ in range(3):
            self.win.make_current()
            host.begin_canvas()
            assert host.canvas is not None and host.canvas_texture is not None
            host.canvas.clear(0, 0, 0, 1)
            host.draw_fullscreen(prog)
            fb = self.win.begin()
            fb.clear(0, 0, 0, 1)
            self.compositor.draw(host.canvas_texture, host.height_texture, cal, host.box_size, self.size)
            self.win.swap()
            poll_events()
        prog.release()
        self._settle(settle)

    def close(self) -> None:
        self.win.close()
        terminate_glfw()


def _gray(bgr: np.ndarray) -> np.ndarray:
    return bgr.astype(np.float32).sum(axis=2)  # součet kanálů tlumí pruhy z barevného kola DLP


@dataclass
class Spot:
    proj: tuple[float, float]  # pixel projektoru
    uv: tuple[float, float]  # kde ho viděla kamera (pixel hloubkové mřížky)
    strength: float


def measure_spot(pw: PatternWindow, snap: ColorSnap, px: float, py: float, guess: tuple[float, float] | None,
                 mask: np.ndarray, half: int = 14, repeats: int = 2) -> Spot | None:
    """Promítne čtverec na (px, py) a najde ho v kameře (rozdíl proti černé, průměr `repeats` měření)."""
    acc = np.zeros((H_PX, W_PX), np.float32)
    for _ in range(repeats):
        pw.squares(settle=0.2)
        ref, _ = snap.grab()
        pw.squares([(px - half, py - half, 2 * half, 2 * half)], settle=0.2)
        img, _ = snap.grab()
        acc += _gray(img) - _gray(ref)
    diff = cv2.GaussianBlur(acc / repeats, (0, 0), 1.5)
    if guess is not None:
        u0, v0 = int(round(guess[0])), int(round(guess[1]))
        w = 16
        if not (w <= u0 < W_PX - w and w <= v0 < H_PX - w):
            return None
        sub = np.zeros_like(diff)
        sub[v0 - w:v0 + w + 1, u0 - w:u0 + w + 1] = 1
        search = diff * sub * mask
    else:
        search = diff * mask
    v, u = np.unravel_index(int(np.argmax(search)), search.shape)
    bg = float(np.median(diff[max(0, v - 40):v + 41, max(0, u - 40):u + 41]))
    peak = float(diff[v, u]) - bg
    if guess is None:  # bez předpovědi musí být vrchol jednoznačný
        far = search.copy()
        far[max(0, v - 12):v + 13, max(0, u - 12):u + 13] = 0
        if far.max() - bg > 0.7 * peak:
            return None
    if peak < 60 or not (4 <= u < W_PX - 4 and 4 <= v < H_PX - 4):
        return None
    c = np.clip(diff[v - 4:v + 5, u - 4:u + 5] - bg, 0, None)
    yy, xx = np.mgrid[v - 4:v + 5, u - 4:u + 5]
    return Spot((px, py), (float((c * xx).sum() / c.sum()), float((c * yy).sum() / c.sum())), peak)


def _point_3d(intr: Intrinsics, depth: np.ndarray, u: float, v: float) -> np.ndarray | None:
    iu, iv = int(round(u)), int(round(v))
    win = depth[max(0, iv - 1):iv + 2, max(0, iu - 1):iu + 2]
    win = win[win > 0]
    if not len(win):
        return None
    z = float(np.median(win))
    return np.array([(u - intr.cx) / intr.fx * z, (v - intr.cy) / intr.fy * z, z])


def fit_projector(box_pts: np.ndarray, proj_pts: np.ndarray, box_size: tuple[float, float],
                  resolution: tuple[int, int], kinect_pos: tuple[float, float]) -> tuple[ProjectorCalibration, float]:
    """Model projektoru: homografie roviny písku + korekce paralaxy (výška Hp, průmět C).

    Hledá (Hp, C) minimalizující chybu v pixelech projektoru, homografii pro každou
    volbu počítá přímo. Vrací kalibraci a RMS chybu [px]."""
    bx, by, bh = box_pts[:, 0], box_pts[:, 1], box_pts[:, 2]
    Q = proj_pts.astype(np.float32)

    def err(hp: float, cx: float, cy: float) -> tuple[float, np.ndarray]:
        if hp <= 0:
            gx, gy = bx, by
        else:
            k = bh / np.maximum(hp - bh, 1e-3)
            gx, gy = bx + (bx - cx) * k, by + (by - cy) * k
        G = np.column_stack([gx, gy]).astype(np.float32)
        H, _ = cv2.findHomography(G, Q, 0)
        if H is None:
            return 1e9, np.eye(3)
        e = np.linalg.norm(cv2.perspectiveTransform(G[:, None], H)[:, 0] - Q, axis=1)
        return float(np.sqrt((e ** 2).mean())), H

    best = (err(0, 0, 0)[0], 0.0, kinect_pos[0], kinect_pos[1])
    span, span_h, step_h, step_c = 1.0, 1.0, 0.1, 0.1
    center = (1.5, kinect_pos[0], kinect_pos[1])
    for _ in range(3):  # hrubé hledání -> jemné
        for hp in np.arange(max(0.5, center[0] - span_h), center[0] + span_h + 1e-9, step_h):
            for cx in np.arange(center[1] - span, center[1] + span + 1e-9, step_c):
                for cy in np.arange(center[2] - span, center[2] + span + 1e-9, step_c):
                    r = err(hp, cx, cy)[0]
                    if r < best[0]:
                        best = (r, float(hp), float(cx), float(cy))
        center = best[1:] if best[1] > 0 else center
        span, span_h, step_h, step_c = span / 4, span_h / 4, step_h / 4, step_c / 4
    rms, hp, cx, cy = best
    _, H = err(hp, cx, cy)
    W, D = box_size
    corners = cv2.perspectiveTransform(np.array([[[0, 0]], [[W, 0]], [[W, D]], [[0, D]]], np.float32), H)[:, 0]
    return ProjectorCalibration(resolution, corners.astype(float), (cx, cy) if hp > 0 else None, hp), rms


def _box_mask(corners_px: np.ndarray, grow: float = 1.06) -> np.ndarray:
    mask = np.zeros((H_PX, W_PX), np.uint8)
    c = corners_px.mean(axis=0)
    cv2.fillConvexPoly(mask, np.round(c + (corners_px - c) * grow).astype(np.int32), 1)
    return mask.astype(np.float32)


def autocalibrate(settings: Settings, grid: tuple[int, int] = (10, 7), progress=print) -> dict:
    """Celá kalibrace bez obsluhy. Uloží config/calibration_kinect2.json a vrátí souhrn."""
    snap = ColorSnap()
    pw = PatternWindow(settings)
    try:
        intr = snap.intrinsics
        pw.squares(settle=1.0)
        progress("Měřím bednu (hloubka)…")
        depth = snap.grab_depth_median(20)
        corners = detect_box_corners_walls(intr, depth)
        if corners is None:
            raise CalibrationError("Stěny bedny se v hloubce nepodařilo najít")
        kc = KinectCalibration.from_depth(intr, depth, corners)
        corners = unmirror_corners(intr, corners, kc.normal)  # obraz Kinectu je zrcadlený
        kc = KinectCalibration.from_depth(intr, depth, corners)
        W, D = kc.size
        progress(f"Bedna {W * 100:.0f} × {D * 100:.0f} cm, Kinect {kc.offset * 100:.0f} cm nad pískem")
        mask = _box_mask(corners)
        PW, PH = pw.size
        gx, gy = grid
        xs, ys = np.linspace(50, PW - 50, gx), np.linspace(50, PH - 50, gy)
        # 1) hrubé body bez předpovědi (každý třetí), 2) homografie projektor -> kamera, 3) zbytek s předpovědí
        spots: list[Spot] = []
        coarse = [(x, y) for y in ys[::2] for x in xs[::3]]
        for i, (x, y) in enumerate(coarse):
            s = measure_spot(pw, snap, x, y, None, mask)
            if s:
                spots.append(s)
            progress(f"Hledám projekci {i + 1}/{len(coarse)}")
        if len(spots) < 6:
            raise CalibrationError("Kamera nevidí promítané body – je projektor zapnutý a míří do bedny?")
        Hpk, inl = cv2.findHomography(np.array([s.proj for s in spots], np.float32),
                                      np.array([s.uv for s in spots], np.float32), cv2.RANSAC, 5.0)
        spots = [s for s, ok in zip(spots, inl.ravel()) if ok]
        done = {s.proj for s in spots}
        todo = [(x, y) for y in ys for x in xs if (x, y) not in done]
        for i, (x, y) in enumerate(todo):
            g = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), Hpk)[0, 0]
            s = measure_spot(pw, snap, x, y, (float(g[0]), float(g[1])), mask)
            if s:
                spots.append(s)
            if i % 5 == 0:
                progress(f"Měřím body {i + 1}/{len(todo)}")
        pw.squares(settle=0.5)
        depth = snap.grab_depth_median(10)
    finally:
        snap.close()
        pw.close()

    box, proj = [], []
    for s in spots:
        p = _point_3d(intr, depth, *s.uv)
        if p is None:
            continue
        x, y, h = kc.camera_to_box(p[None])
        box.append((x[0], y[0], h[0]))
        proj.append(s.proj)
    box_a, proj_a = np.array(box), np.array(proj)
    kx, ky, _ = kc.camera_position_box()
    pc, rms = fit_projector(box_a, proj_a, kc.size, (PW, PH), (kx, ky))
    area = playarea.PlayArea(estimate_play_margins(kc, depth))
    save_calibration(calibration_path("kinect2"), kc, pc, area)
    x0, y0, x1, y1 = playarea.visible_rect(kc, pc, area)
    return {"play_area_cm": (round((x1 - x0) * 100), round((y1 - y0) * 100)),
            "okraje_cm": [round(m * 100, 1) for m in area.margins],
            "box_cm": (W * 100, D * 100), "kinect_cm": kc.offset * 100, "points": len(box_a),
            "rms_px": rms, "projector_height_cm": pc.height * 100, "corners": pc.corners.round(1).tolist()}


# --- kontrola ----------------------------------------------------------------------

GRID_FRAG = """
void main() {
    vec2 p = v_uv * u_box;
    vec2 g = abs(fract(p / 0.1 + 0.5) - 0.5) * 0.1;
    vec2 fw = fwidth(p);
    float grid = max(1.0 - smoothstep(0.5 * fw.x, 1.5 * fw.x, g.x), 1.0 - smoothstep(0.5 * fw.y, 1.5 * fw.y, g.y));
    float edge = min(min(p.x, u_box.x - p.x), min(p.y, u_box.y - p.y));
    float e = 1.0 - smoothstep(0.004, 0.008, edge);
    f_color = vec4(vec3(max(grid, e)), 1.0);
}
"""

CONTOUR_FRAG = """
void main() {
    float h = height_at(v_uv);
    float c = contour_line(h, 0.02, 2.0);
    f_color = vec4(vec3(c), 1.0);
}
"""


def _heightmap_from_depth(kc: KinectCalibration, settings: Settings, depth: np.ndarray) -> HeightMap:
    proc = TerrainProcessor(kc, settings.terrain)
    hm = proc.process(depth)
    for _ in range(3):
        hm = proc.process(depth)
    return hm


def check_calibration(settings: Settings, out_dir: Path, n_points: int = 24, progress=print) -> dict:
    """Nezávislá kontrola: body na náhodných místech + fotky mřížky a vrstevnic."""
    kc, pc = load_calibration(calibration_path("kinect2"))
    if kc is None or pc is None:
        raise CalibrationError("Není uložená kalibrace – spusť autocalibrate")
    kc, pc = playarea.apply(kc, pc, load_play_area(calibration_path("kinect2")))  # jako aplikace
    assert kc is not None and pc is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = ColorSnap()
    pw = PatternWindow(settings)
    pc = pc.scaled_to(pw.size)
    intr = kc.intrinsics
    report: dict = {}
    try:
        pw.squares(settle=1.0)
        depth = snap.grab_depth_median(15)
        hm = _heightmap_from_depth(kc, settings, depth)
        rays = intr.rays()
        mask = _box_mask(np.asarray(kc.corners_px))

        # 1) body: cílové místo v bedně -> pixel projektoru (s paralaxou i bez) -> kde dopadl
        rng = np.random.default_rng(int(time.time()))
        W, D = kc.size
        rows = []
        for i in range(n_points):
            x, y = rng.uniform(0.12, W - 0.12), rng.uniform(0.1, D - 0.1)
            if i < n_points // 3:  # třetina bodů na nejvyšších místech (test paralaxy)
                gy, gx = np.unravel_index(int(np.argmax(hm.heights + rng.normal(0, 0.01, hm.shape))), hm.shape)
                cw, ch = hm.cell_size
                x, y = (gx + 0.5) * cw, (gy + 0.5) * ch
            h = hm.sample(x, y)
            res = {}
            for mode in ("paralaxa", "bez"):
                cal = pc if mode == "paralaxa" else ProjectorCalibration(pc.resolution, pc.corners, pc.position, 0.0)
                px, py = cal.box_to_projector(np.array(x), np.array(y), np.array(h), kc.size)
                guess = kc.box_to_pixel(np.array(x), np.array(y), np.array(h))
                s = measure_spot(pw, snap, float(px), float(py), (float(guess[0]), float(guess[1])), mask, half=12)
                if s is None:
                    continue
                p = _point_3d(intr, depth, *s.uv)
                if p is None:
                    continue
                mx, my, _ = kc.camera_to_box(p[None])
                res[mode] = float(np.hypot(mx[0] - x, my[0] - y) * 100)
            if "paralaxa" in res:
                rows.append((x, y, h, res.get("paralaxa"), res.get("bez")))
                progress(f"bod {i + 1}/{n_points}: výška {h * 100:4.1f} cm, chyba {res['paralaxa']:.1f} cm"
                         + (f" (bez paralaxy {res['bez']:.1f} cm)" if "bez" in res else ""))
        e = np.array([r[3] for r in rows])
        e0 = np.array([r[4] for r in rows if r[4] is not None])
        hs = np.array([r[2] for r in rows])
        high = hs > np.percentile(hs, 66)
        report["body"] = len(rows)
        report["chyba_median_cm"] = float(np.median(e))
        report["chyba_max_cm"] = float(e.max())
        report["chyba_kopce_cm"] = float(np.median(e[high])) if high.any() else None
        report["bez_paralaxy_median_cm"] = float(np.median(e0)) if len(e0) else None

        # 2) mřížka 10 cm + obrys bedny, proti stěnám z hloubky
        pw.squares(settle=0.4)
        dark, _ = snap.grab()
        pw.canvas_pattern(GRID_FRAG, pc, hm)
        lit, _ = snap.grab()
        img = _overlay(dark, lit)
        cv2.polylines(img, [np.round(np.asarray(kc.corners_px)).astype(np.int32)], True, (255, 255, 0), 1)
        cv2.imwrite(str(out_dir / "kontrola_mrizka.png"), cv2.resize(img, None, fx=2, fy=2,
                                                                       interpolation=cv2.INTER_NEAREST))

        # 3) vrstevnice po 2 cm: promítnuté (zeleně z kamery) vs. spočítané z hloubky (červeně)
        pw.canvas_pattern(CONTOUR_FRAG, pc, hm)
        lit, depth2 = snap.grab()
        img = _overlay(dark, lit)
        P = rays * depth[..., None]
        _, _, hpx = kc.camera_to_box(P.reshape(-1, 3))
        hpx = hpx.reshape(depth.shape)
        hpx = cv2.GaussianBlur(np.where(depth > 0, hpx, np.nan).astype(np.float32), (0, 0), 1.2)
        level = np.floor(hpx / 0.02 - 0.5)  # stejně jako contour_line v glsl: čáry v polovině intervalu
        edges = np.zeros(depth.shape, bool)
        edges[:, 1:] |= level[:, 1:] != level[:, :-1]
        edges[1:, :] |= level[1:, :] != level[:-1, :]
        edges &= (mask > 0) & np.isfinite(hpx)
        img[edges] = (0, 0, 255)
        cv2.imwrite(str(out_dir / "kontrola_vrstevnice.png"), cv2.resize(img, None, fx=2, fy=2,
                                                                          interpolation=cv2.INTER_NEAREST))
        # číselně: kolik promítnutých čar leží do 1 px od spočítaných
        lit_lines = _lines_mask(dark, lit) & (mask > 0)
        near = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        report["vrstevnice_shoda_%"] = float(100 * (lit_lines & near).sum() / max(lit_lines.sum(), 1))
        pw.squares(settle=0.1)
    finally:
        snap.close()
        pw.close()
    report["obrazky"] = str(out_dir)
    return report


def _lines_mask(dark: np.ndarray, lit: np.ndarray) -> np.ndarray:
    d = cv2.GaussianBlur(_gray(lit) - _gray(dark), (0, 0), 0.8)
    thr = max(60.0, float(np.percentile(d, 97)) * 0.5)
    return d > thr


def _overlay(dark: np.ndarray, lit: np.ndarray) -> np.ndarray:
    """Šedá fotka bedny + promítnuté čáry zeleně."""
    base = cv2.cvtColor(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) // 2
    lines = _lines_mask(dark, lit)
    base[lines] = (0, 255, 0)
    return base


def estimate_play_margins(kc: KinectCalibration, depth: np.ndarray) -> tuple[float, float, float, float]:
    """Okraje od stěn celé bedny z (průměrovaného) hloubkového snímku."""
    proc = TerrainProcessor(kc, TerrainSettings(edge_margin=0.0, spatial_sigma=0.0))
    meas, valid, _ = proc.measure(depth)
    return playarea.estimate_margins(meas, valid, kc.size)


def fit_area(settings: Settings, progress=print) -> dict:
    """Přeměří jen hrací plochu (okraje od stěn) nad uloženou kalibrací – bez promítání bodů."""
    path = calibration_path("kinect2")
    kc, pc = load_calibration(path)
    if kc is None or pc is None:
        raise CalibrationError("Není uložená kalibrace – spusť autocalibrate")
    snap = ColorSnap()
    try:
        progress("Měřím okraje u stěn…")
        depth = snap.grab_depth_median(20)
    finally:
        snap.close()
    area = playarea.PlayArea(estimate_play_margins(kc, depth))
    save_calibration(path, kc, pc, area)
    x0, y0, x1, y1 = playarea.visible_rect(kc, pc, area)
    return {"okraje_cm": [round(m * 100, 1) for m in area.margins],
            "play_area_cm": (round((x1 - x0) * 100), round((y1 - y0) * 100)),
            "bedna_cm": (round(kc.size[0] * 100), round(kc.size[1] * 100))}
