"""Hlavní aplikace: propojí senzor, terén, hry a projektor.

    Kinect ──► TerrainService (vlákno) ──► HeightMap ──► Game ──► plátno ──► ProjectorCompositor ──► BenQ
                                                                    └──────► OperatorView (notebook)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .calibration_store import calibration_path, load_calibration, save_calibration
from .config import Settings
from .games import Game, discover_games
from .projector import ProjectorCalibration, ProjectorCompositor
from .render.host import RenderHost
from .render.window import (MOD_SHIFT, Window, find_projector_monitor, poll_events, terminate_glfw)
from .sensor import create_sensor
from .terrain import KinectCalibration, TerrainService
from .ui.calibration import CalibrationMode
from .ui.operator import OperatorView

log = logging.getLogger(__name__)


class SandboxApp:
    def __init__(self, settings: Settings, calibrate: bool = False) -> None:
        self.settings = settings
        self.sensor = create_sensor(settings.sensor)
        log.info("Senzor: %s", self.sensor.name)
        self.cal_path = calibration_path(self.sensor.name)
        kinect_cal, proj_cal = load_calibration(self.cal_path, self.sensor.intrinsics)
        if kinect_cal is None and hasattr(self.sensor, "ground_truth"):
            normal, offset, corners = self.sensor.ground_truth()  # simulace se kalibruje sama
            kinect_cal = KinectCalibration.from_plane(self.sensor.intrinsics, normal, offset, corners)
        self.terrain = TerrainService(self.sensor, settings.terrain, kinect_cal)

        # okna
        disp = settings.display
        monitor = None if disp.windowed else find_projector_monitor(disp.monitor)
        if monitor is None and not disp.windowed:
            log.warning("Projektor (druhý monitor) nenalezen – projekce poběží v okně")
        self.proj_win = Window("Pískoviště – projektor", (1280, 800), monitor=monitor, vsync=True,
                               position=(80, 60))
        self.op_win = None
        if disp.operator_window:
            self.op_win = Window("Pískoviště – obsluha", (1280, 780), share=self.proj_win, vsync=False,
                                 position=(40, 40))
        self.proj_win.set_cursor_visible(monitor is None)

        self.host = RenderHost(self.proj_win.ctx, settings)
        self.compositor = ProjectorCompositor(self.proj_win.ctx, disp.mesh_resolution)
        fb_size = self.proj_win.framebuffer_size
        self.proj_cal = (proj_cal or ProjectorCalibration.default(fb_size)).scaled_to(fb_size)
        if self.proj_cal.height <= 0 and kinect_cal is not None and proj_cal is None:
            cx, cy, ch = kinect_cal.camera_position_box()
            self.proj_cal.position, self.proj_cal.height = (cx, cy), ch
        self._configure_host()

        self.game_classes = discover_games()
        self.game: Game | None = None
        self.calibration = CalibrationMode(self)
        self.operator = OperatorView(self, self.op_win) if self.op_win else None
        self.show_help = False
        self.fps = 0.0
        self._quit_armed_until = 0.0
        self._running = True

        for w in filter(None, [self.proj_win, self.op_win]):
            w.key_handlers.append(self.on_key)
        self.proj_win.mouse_handlers.append(self._projector_mouse)
        if self.op_win and self.operator:
            self.op_win.mouse_handlers.append(self.operator.mouse)

        self.select_game(settings.start_game)
        if calibrate or kinect_cal is None:
            self.calibration.enter()

    # --- kalibrace ------------------------------------------------------------------------
    def _configure_host(self) -> None:
        proc = self.terrain.processor
        if proc is not None:
            self.host.configure(proc.size, (proc.gh, proc.gw))
        else:
            self.host.configure((1.0, 0.75), (8, 8))
        self.host.heightmap = None

    def set_kinect_calibration(self, cal: KinectCalibration | None) -> None:
        self.terrain.set_calibration(cal)
        self._configure_host()
        if self.game is not None:  # hra si znovu vytvoří textury pro nový rozměr
            self.game.stop()
            self.game.start()

    def save_calibration(self) -> None:
        save_calibration(self.cal_path, self.terrain.calibration, self.proj_cal)

    # --- hry ---------------------------------------------------------------------------------
    def select_game(self, game_id_or_index: str | int) -> None:
        if isinstance(game_id_or_index, int):
            if not 0 <= game_id_or_index < len(self.game_classes):
                return
            cls = self.game_classes[game_id_or_index]
        else:
            cls = next((c for c in self.game_classes if c.id == game_id_or_index), self.game_classes[0])
        if self.game is not None:
            if type(self.game) is cls:
                return
            self.game.stop()
        self.proj_win.make_current()
        self.game = cls(self.host)
        self.game.start()
        self.host.notify(cls.title, 2.5)
        log.info("Hra: %s", cls.title)

    def _cycle_game(self, step: int) -> None:
        ids = [c.id for c in self.game_classes]
        i = ids.index(self.game.id) if self.game else 0
        self.select_game((i + step) % len(ids))

    # --- vstup -------------------------------------------------------------------------------
    def on_key(self, name: str, mods: int) -> None:
        shift = bool(mods & MOD_SHIFT)
        self.proj_win.make_current()  # hry mohou při stisku vytvářet textury
        if self.calibration.active and self.calibration.key(name, shift):
            return
        if name == "escape":
            if time.monotonic() < self._quit_armed_until:
                self._running = False
            else:
                self._quit_armed_until = time.monotonic() + 2.0
                self.host.notify("Stiskni Esc ještě jednou pro ukončení", 2.0)
        elif name == "tab":
            self._cycle_game(-1 if shift else 1)
        elif name.isdigit() and name != "0":
            self.select_game(int(name) - 1)
        elif name == "c":
            self.calibration.enter()
        elif name == "h":
            self.show_help = not self.show_help
        elif name == "z":
            self.terrain.frozen = not self.terrain.frozen
            self.host.notify("Terén zmrazen" if self.terrain.frozen else "Terén se znovu snímá")
        elif name == "f" and hasattr(self.sensor, "flatten"):
            self.sensor.flatten()
        elif name == "n" and hasattr(self.sensor, "randomize"):
            self.sensor.randomize()
        elif self.game is not None:
            self.game.key(name, shift)

    def _projector_mouse(self, kind: str, x: float, y: float, button: int) -> None:
        if self.calibration.active:
            self.calibration.projector_mouse(kind, x, y, button)

    # --- smyčka ------------------------------------------------------------------------------
    def run(self, duration: float | None = None, snapshot_dir: Path | None = None) -> None:
        t_prev = time.monotonic()
        t_end = t_prev + duration if duration else None
        last_seq = -1
        try:
            while self._running and not self.proj_win.should_close() \
                    and not (self.op_win and self.op_win.should_close()):
                poll_events()
                now = time.monotonic()
                dt = max(1e-3, now - t_prev)
                t_prev = now
                self.fps = 0.95 * self.fps + 0.05 / dt

                self.proj_win.make_current()
                hm = self.terrain.latest_heightmap()
                if hm is not None and hm.seq != last_seq:
                    self.host.update_heightmap(hm)
                    last_seq = hm.seq
                if self.calibration.active:
                    self.calibration.update()
                elif self.host.heightmap is not None and self.game is not None:
                    self.game.update(self.host.heightmap, dt)
                if self.operator:
                    self.operator.sculpt(dt)

                self._render_projector()
                if self.operator:
                    self.operator.render()
                if t_end and now >= t_end:
                    break
            if snapshot_dir:
                self.snapshot(snapshot_dir)
        finally:
            self.shutdown()

    def _render_projector(self, target=None) -> None:
        """Vykreslí výstup projektoru do okna (nebo do offscreen framebufferu `target`)."""
        host = self.host
        self.proj_win.make_current()
        host.begin_canvas()
        assert host.canvas is not None
        host.canvas.clear(0, 0, 0, 1)
        if self.calibration.active:
            self.calibration.render_canvas()
        elif host.heightmap is not None and self.game is not None:
            self.game.render()
        else:
            cw, ch = host.canvas_size
            host.painter.text("Čekám na data z Kinectu…" if self.terrain.calibration else
                              "Pískoviště není zkalibrované – stiskni C", cw / 2, ch / 2, size=ch // 18,
                              anchor="mc", color=(1, 1, 1, 1))
        host.painter.begin(host.canvas_size, screen=False)
        self._canvas_overlays()

        if target is None:
            fb = self.proj_win.begin()
            fb_size = self.proj_win.framebuffer_size
        else:
            fb = target
            fb.use()
            fb_size = fb.size
            fb.viewport = (0, 0, *fb_size)
        fb.clear(0, 0, 0, 1)
        assert host.canvas_texture is not None
        self.compositor.draw(host.canvas_texture, host.height_texture if host.heightmap else None,
                             self.proj_cal, host.box_size, fb_size)
        if self.calibration.active:
            self.calibration.render_projector_overlay(fb_size)
        if target is None:
            self.proj_win.swap()

    def _canvas_overlays(self) -> None:
        """Nápověda a krátké zprávy přímo na písku."""
        p, host = self.host.painter, self.host
        cw, ch = host.canvas_size
        size = max(14, ch // 34)
        msgs = host.active_messages()
        if msgs and not self.calibration.active:
            p.text(msgs[-1], size, ch - size, size=size, anchor="bl", color=(1, 1, 1, 1), outline=3)
        if self.show_help and self.game is not None:
            lines = [f"{self.game.title}", self.game.description, ""]
            lines += [f"{k}: {d}" for k, d in self.game.keys]
            lines += ["", "Hry: " + ", ".join(f"{i + 1} {g.title}" for i, g in enumerate(self.game_classes[:9]))]
            p.text("\n".join(lines), cw / 2, ch / 2, size=size, anchor="mc", color=(1, 1, 1, 1),
                   background=(0, 0, 0, 0.7), padding=size)

    def snapshot(self, directory: Path) -> None:
        """Uloží aktuální obraz projektoru a operátorského okna jako PNG (ladění)."""
        import cv2

        def save(name: str, fbo, screen: bool = True) -> None:
            w, h = fbo.size
            img = np.frombuffer(fbo.read(components=3), np.uint8).reshape(h, w, 3)
            if screen:  # obrazovka má počátek dole, plátno pískoviště nahoře
                img = img[::-1]
            cv2.imwrite(str(directory / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        directory.mkdir(parents=True, exist_ok=True)
        self.proj_win.make_current()
        fbo = self.proj_win.ctx.simple_framebuffer(self.proj_win.framebuffer_size)
        self._render_projector(fbo)
        save("projektor", fbo)
        if self.host.canvas is not None:
            save("platno", self.host.canvas, screen=False)
        if self.op_win and self.operator:
            self.op_win.make_current()
            fbo = self.op_win.ctx.simple_framebuffer(self.op_win.framebuffer_size)
            self.operator.render(fbo)
            save("obsluha", fbo)

    def shutdown(self) -> None:
        if self.game is not None:
            self.proj_win.make_current()
            self.game.stop()
            self.game = None
        self.terrain.stop()
        self.sensor.stop()
        if self.op_win:
            self.op_win.close()
        self.proj_win.close()
        terminate_glfw()
