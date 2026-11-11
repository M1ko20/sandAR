"""Operátorské okno na notebooku: co vidí Kinect, co se promítá, stav a nápověda."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import moderngl
import numpy as np

from ..render.painter import Painter
from ..render.window import MOUSE_LEFT, MOUSE_RIGHT, Window

if TYPE_CHECKING:
    from ..app import SandboxApp

BG = (0.11, 0.12, 0.14, 1)
PANEL = (0.17, 0.18, 0.21, 1)
TEXT = (0.92, 0.93, 0.95, 1)
DIM = (0.62, 0.65, 0.70, 1)
ACCENT = (0.40, 0.75, 1.0, 1)
CORNER_COLORS = [(1.0, 0.3, 0.3, 1), (0.3, 1.0, 0.3, 1), (0.35, 0.6, 1.0, 1), (1.0, 0.9, 0.2, 1)]

GLOBAL_KEYS = [
    ("1–9 / Tab", "výběr hry"),
    ("C", "kalibrace"),
    ("H", "nápověda na písku"),
    ("Z", "zmrazit terén"),
    ("Esc 2×", "konec"),
]
SIM_KEYS = [("myš", "hrabání písku (L = přidat, P = ubrat)"), ("F / N", "uhladit / nový terén")]


class OperatorView:
    def __init__(self, app: "SandboxApp", window: Window) -> None:
        self.app = app
        self.win = window
        self.ctx = window.ctx
        self.painter = Painter(self.ctx)
        self.depth_tex: moderngl.Texture | None = None
        self._depth_seq = -1
        self._canvas_ext: moderngl.Texture | None = None
        self._canvas_glo = None
        self.depth_rect = (0.0, 0.0, 1.0, 1.0)
        self.canvas_rect = (0.0, 0.0, 1.0, 1.0)
        self._rays_n: np.ndarray | None = None
        self._rays_key = None

    @property
    def scale(self) -> float:
        fw, _ = self.win.framebuffer_size
        w, _ = self.win.size
        return fw / max(w, 1)

    # --- převody souřadnic ----------------------------------------------------------
    def depth_panel_to_pixel(self, x: float, y: float) -> tuple[float, float] | None:
        rx, ry, rw, rh = self.depth_rect
        if not (rx <= x <= rx + rw and ry <= y <= ry + rh):
            return None
        intr = self.app.sensor.intrinsics
        return (x - rx) / rw * intr.width, (y - ry) / rh * intr.height

    def pixel_to_depth_panel(self, u: float, v: float) -> tuple[float, float]:
        rx, ry, rw, rh = self.depth_rect
        intr = self.app.sensor.intrinsics
        return rx + u / intr.width * rw, ry + v / intr.height * rh

    def canvas_panel_to_box(self, x: float, y: float) -> tuple[float, float] | None:
        rx, ry, rw, rh = self.canvas_rect
        if not (rx <= x <= rx + rw and ry <= y <= ry + rh):
            return None
        W, D = self.app.host.box_size
        return (x - rx) / rw * W, (y - ry) / rh * D

    # --- obsah ---------------------------------------------------------------------------
    def _update_depth(self) -> None:
        frame = self.app.terrain.latest_depth()
        if frame is None or frame.seq == self._depth_seq:
            return
        self._depth_seq = frame.seq
        depth = frame.depth
        valid = depth > 0
        cal = self.app.terrain.calibration
        if cal is not None:
            key = (id(cal), depth.shape)
            if self._rays_key != key:
                self._rays_n = (cal.intrinsics.rays() @ cal.normal).astype(np.float32)
                self._rays_key = key
            h = depth * self._rays_n + cal.offset
            ts = self.app.settings.terrain
            t = (h - ts.min_height) / (ts.max_height - ts.min_height)
        else:
            if valid.any():
                lo, hi = np.percentile(depth[valid], [2, 98])
            else:
                lo, hi = 0.5, 2.0
            t = 1 - (depth - lo) / max(hi - lo, 1e-3)
        img = cv2.applyColorMap((np.clip(t, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[~valid] = 0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hh, ww = depth.shape
        if self.depth_tex is None or self.depth_tex.size != (ww, hh):
            if self.depth_tex is not None:
                self.depth_tex.release()
            self.depth_tex = self.ctx.texture((ww, hh), 3)
        self.depth_tex.write(np.ascontiguousarray(img).tobytes())

    def _canvas_texture(self) -> moderngl.Texture | None:
        tex = self.app.host.canvas_texture
        if tex is None:
            return None
        if self._canvas_glo != tex.glo:
            self._canvas_ext = self.ctx.external_texture(tex.glo, tex.size, 4, 0, "f1")
            self._canvas_glo = tex.glo
        return self._canvas_ext

    @staticmethod
    def _fit(x, y, w, h, aspect) -> tuple[float, float, float, float]:
        if w / h > aspect:
            nw = h * aspect
            return x + (w - nw) / 2, y, nw, h
        nh = w / aspect
        return x, y, w, nh

    # --- kreslení ---------------------------------------------------------------------------
    def render(self, target: moderngl.Framebuffer | None = None) -> None:
        if target is None:
            fb = self.win.begin()
            fw, fh = self.win.framebuffer_size
        else:
            fb = target
            fb.use()
            fw, fh = fb.size
            fb.viewport = (0, 0, fw, fh)
        fb.clear(*BG)
        s = self.scale
        p = self.painter
        p.begin((fw, fh), screen=True)
        app = self.app

        m = 14 * s
        header = 46 * s
        # Zápatí podle obsahu (kalibrační návod má 9 řádků a do pevných 250 px se nevešel).
        if app.calibration.active:
            lines = app.calibration.instructions()
        else:
            lines = [app.game.description] if app.game else []
            lines += app.game.status() if app.game else []
            if app.terrain.calibration is None:
                lines.append("Pískoviště není zkalibrované – stiskni C.")
        lines = lines[:10]
        n_msgs = len(app.host.active_messages()[-2:])
        footer = max(250.0, 2 * 14 + 26 + 23 * max(len(lines) - 1, 0) + 25 * n_msgs + 26) * s
        panel_h = max(60.0, fh - header - footer - 2 * m)
        panel_w = (fw - 3 * m) / 2
        intr = app.sensor.intrinsics
        self.depth_rect = self._fit(m, header + m, panel_w, panel_h, intr.width / intr.height)
        W, D = app.host.box_size
        self.canvas_rect = self._fit(2 * m + panel_w, header + m, panel_w, panel_h, W / D)

        # záhlaví
        p.rect(0, 0, fw, header, PANEL)
        title = app.game.title if app.game else "—"
        if app.calibration.active:
            title = "Kalibrace"
        p.text(f"Pískoviště · {title}", m, header / 2, size=int(20 * s), anchor="ml", bold=True, color=TEXT)
        info = (f"senzor: {app.sensor.name} {app.terrain.sensor_fps:4.1f} Hz · "
                f"vykreslování {app.fps:4.1f} Hz")
        proc = app.terrain.processor
        if proc is not None:
            info += f" · terén {proc.last_ms:4.1f} ms, pokrytí {proc.last_coverage * 100:3.0f} %"
        if app.terrain.frozen:
            info += " · ZMRAZENO"
        at = app.attract
        if at.active:
            info += " · ATTRACT MODE"
        elif app.settings.attract.enabled:
            info += f" · klid {at.idle_for:3.0f}/{app.settings.attract.idle_seconds:.0f} s"
        if app.sensor.error:  # např. zaseknutý Kinect – ať to obsluha vidí hned
            p.rect(0, header, fw, 30 * s, (0.55, 0.12, 0.10, 1))
            p.text(f"POZOR: {app.sensor.error}", m, header + 15 * s, size=int(15 * s), anchor="ml", bold=True,
                   color=TEXT)
        p.text(info, fw - m, header / 2, size=int(14 * s), anchor="mr", color=DIM)

        # panely
        self._update_depth()
        dx, dy, dw, dh = self.depth_rect
        if self.depth_tex is not None:
            p.image(self.depth_tex, dx, dy, dw, dh)
        else:
            p.rect(dx, dy, dw, dh, PANEL)
            p.text("Čekám na senzor…", dx + dw / 2, dy + dh / 2, size=int(18 * s), anchor="mc", color=DIM)
        p.text("Kinect (hloubka)", dx, dy - 4 * s, size=int(13 * s), anchor="bl", color=DIM)
        self._depth_overlay()

        cx, cy, cw, ch = self.canvas_rect
        tex = self._canvas_texture()
        if tex is not None:
            p.image(tex, cx, cy, cw, ch)
        p.rect_outline(cx, cy, cw, ch, (0.4, 0.42, 0.46, 1), 1.5 * s)
        p.text(f"Promítání (pohled shora, {W * 100:.0f} × {D * 100:.0f} cm)", cx, cy - 4 * s,
               size=int(13 * s), anchor="bl", color=DIM)
        hm = app.host.heightmap
        if hm is not None:  # ladění: kde systém vidí ruce
            for hx, hy in hm.hand_points:
                px, py = cx + hx / W * cw, cy + hy / D * ch
                p.circle(px, py, 12 * s, (1.0, 0.3, 0.9, 1), width=2.5 * s)
                p.text("ruka", px + 14 * s, py, size=int(12 * s), anchor="ml", color=(1.0, 0.5, 0.95, 1), outline=2)
        for i, (bx, by) in enumerate([(cx, cy), (cx + cw, cy), (cx + cw, cy + ch), (cx, cy + ch)]):
            p.text(str(i + 1), bx, by, size=int(13 * s), anchor="mc", bold=True, color=CORNER_COLORS[i],
                   outline=2, background=None)

        # zápatí
        top = fh - footer
        p.rect(0, top, fw, footer, PANEL)
        y = top + m
        for i, line in enumerate(lines):
            _, h = p.text(line, m, y, size=int((16 if i == 0 else 14) * s), color=TEXT if i == 0 else DIM,
                          bold=(i == 0))
            y += h + 6 * s
        for msg in app.host.active_messages()[-2:]:
            _, h = p.text(msg, m, y, size=int(15 * s), color=ACCENT, bold=True)
            y += h + 6 * s

        help_items = list(app.game.keys) if app.game and not app.calibration.active else []
        help_items += GLOBAL_KEYS
        if app.sensor.supports_interaction:
            help_items += SIM_KEYS
        x0 = fw * 0.58
        y = top + m
        for k, desc in help_items[:13]:
            p.text(k, x0, y, size=int(13 * s), color=ACCENT, bold=True)
            _, h = p.text(desc, x0 + 120 * s, y, size=int(13 * s), color=DIM)
            y += h + 4 * s

        # seznam her
        games = " · ".join(f"{i + 1} {g.title}" for i, g in enumerate(app.game_classes[:9]))
        p.text(games, m, fh - m, size=int(12 * s), anchor="bl", color=DIM)
        if target is None:
            self.win.swap()

    def _depth_overlay(self) -> None:
        p, s, app = self.painter, self.scale, self.app
        if app.calibration.active:
            pts = [self.pixel_to_depth_panel(*c) for c in app.calibration.depth_corners]
            if len(pts) >= 2:
                p.polyline(pts, (1, 1, 1, 0.9), 2 * s, closed=len(pts) == 4)
            for i, (x, y) in enumerate(pts):
                p.circle(x, y, 7 * s, CORNER_COLORS[i], width=2.5 * s)
                p.text(str(i + 1), x + 10 * s, y - 10 * s, size=int(16 * s), bold=True, color=CORNER_COLORS[i],
                       outline=2)
            return
        cal = app.terrain.calibration
        if cal is not None:
            W, D = cal.size
            corners = cal.box_to_pixel(np.array([0, W, W, 0]), np.array([0, 0, D, D]))
            p.polyline([self.pixel_to_depth_panel(u, v) for u, v in corners], (1, 1, 1, 0.7), 1.5 * s, closed=True)

    # --- myš ---------------------------------------------------------------------------------
    def mouse(self, kind: str, x: float, y: float, button: int) -> None:
        if self.app.calibration.active:
            uv = self.depth_panel_to_pixel(x, y)
            self.app.calibration.depth_mouse(kind, *(uv if uv else (None, None)), button)

    def sculpt(self, dt: float) -> None:
        """Simulace: držením tlačítka myši se přidává / ubírá písek."""
        sensor = self.app.sensor
        buttons = self.win.buttons()
        if not sensor.supports_interaction or not buttons or self.app.calibration.active:
            return
        amount = 0.12 * dt * (1 if MOUSE_LEFT in buttons else -1 if MOUSE_RIGHT in buttons else 0)
        if amount == 0:
            return
        x, y = self.win.cursor()
        uv = self.depth_panel_to_pixel(x, y)
        if uv is None:
            box = self.canvas_panel_to_box(x, y)
            cal = self.app.terrain.calibration
            if box is None or cal is None:
                return
            hm = self.app.host.heightmap
            h = hm.sample(*box) if hm else 0.0
            u, v = cal.box_to_pixel(np.array(box[0]), np.array(box[1]), np.array(h))
            uv = (float(u), float(v))
        sensor.interact(uv[0], uv[1], amount, radius=0.06)
