"""Pomocné nástroje: nahrávání z Kinectu a vykreslení her bez okna."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Settings

log = logging.getLogger(__name__)


def record(settings: Settings, seconds: float, out: Path) -> Path:
    """Nahraje hloubková data ze senzoru do .npz (pro přehrávání: --sensor replay --replay SOUBOR)."""
    from .sensor import create_sensor
    from .sensor.replay import save_recording

    sensor = create_sensor(settings.sensor)
    frames: list[np.ndarray] = []
    seq = 0
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            frame = sensor.wait_frame(seq, timeout=1.0)
            if frame is None:
                continue
            seq = frame.seq
            frames.append(frame.depth.copy())
            print(f"\rNahráno {len(frames)} snímků", end="", flush=True)
    finally:
        sensor.stop()
    print()
    fps = len(frames) / max(time.monotonic() - t0, 1e-3)
    save_recording(out, frames, sensor.intrinsics, fps)
    return out


def render_snapshots(settings: Settings, out_dir: Path, frames: int = 20, size: tuple[int, int] = (1280, 800)) -> list[Path]:
    """Vykreslí každou hru nad simulovaným pískem do PNG – bez okna a bez projektoru.

    Hodí se na rychlou kontrolu nové hry nebo úpravy shaderu.
    """
    import moderngl

    from .games import discover_games
    from .projector import ProjectorCalibration, ProjectorCompositor
    from .render.host import RenderHost
    from .sensor.simulated import SimulatedSensor
    from .terrain import KinectCalibration, TerrainProcessor

    sim = SimulatedSensor(seed=3)
    normal, offset, corners = sim.ground_truth()
    cal = KinectCalibration.from_plane(sim.intrinsics, normal, offset, corners)
    proc = TerrainProcessor(cal, settings.terrain)

    ctx = moderngl.create_standalone_context(require=330)
    host = RenderHost(ctx, settings)
    host.configure(proc.size, (proc.gh, proc.gw))
    compositor = ProjectorCompositor(ctx, settings.display.mesh_resolution)
    proj_cal = ProjectorCalibration.default(size, inset=0.04)
    cx, cy, ch = cal.camera_position_box()
    proj_cal.position, proj_cal.height = (cx, cy), ch
    screen_fbo = ctx.simple_framebuffer(size)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fbo, path: Path, screen: bool = True) -> None:
        w, h = fbo.size
        img = np.frombuffer(fbo.read(components=3), np.uint8).reshape(h, w, 3)
        if screen:  # obrazovka má počátek dole, plátno pískoviště nahoře
            img = img[::-1]
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        written.append(path)

    for cls in discover_games():
        hm = None
        for i in range(frames):
            hm = proc.process(sim.render_depth(), seq=i)
        assert hm is not None
        # trocha "rukou" pro hry, které na ně reagují
        hands = np.zeros(hm.shape, bool)
        hands[hm.shape[0] // 3: hm.shape[0] // 2, hm.shape[1] // 4: hm.shape[1] // 3] = True
        hm.hands = hands
        host.update_heightmap(hm)
        game = cls(host)
        game.start()
        for _ in range(60):
            game.update(hm, 1 / 30)
        host.begin_canvas()
        host.canvas.clear(0, 0, 0, 1)  # type: ignore[union-attr]
        game.render()
        save(host.canvas, out_dir / f"{cls.order:02d}_{cls.id}_platno.png", screen=False)
        screen_fbo.use()
        screen_fbo.clear(0, 0, 0, 1)
        compositor.draw(host.canvas_texture, host.height_texture, proj_cal, host.box_size, size)  # type: ignore[arg-type]
        save(screen_fbo, out_dir / f"{cls.order:02d}_{cls.id}_projektor.png")
        game.stop()
    return written
