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


def render_snapshots(settings: Settings, out_dir: Path, frames: int = 20, size: tuple[int, int] = (1280, 800),
                     only: str | None = None, steps: int = 60) -> list[Path]:
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
        if only and cls.id != only:
            continue
        hm = None
        sim.set_hands([])
        for i in range(frames):
            hm = proc.process(sim.render_depth(), seq=i, timestamp=i / 30)
        # virtuální ruka nad pískem pro hry, které na ruce reagují (skutečná detekce, ne podvrh)
        W, D = sim.BOX_SIZE
        sim.set_hands([(W * 0.3, D * 0.42, 0.10)])
        for i in range(frames, frames + 6):
            hm = proc.process(sim.render_depth(), seq=i, timestamp=i / 30)
        assert hm is not None
        host.update_heightmap(hm)
        game = cls(host)
        game.start()
        for _ in range(steps):
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


def sound_preview(only: str | None = None, out: Path | None = None, volume: float = 0.8) -> int:
    """Přehraje (nebo uloží do WAV) zvuky z banky – na poslech a ladění."""
    import time
    import wave

    from . import sounds
    from .audio import TARGET_RMS, Audio, _backend_cmd

    names = [n for n in sounds.BANK if only is None or n == only]
    if not names:
        print(f"Neznámý zvuk {only!r}. Zvuky: {', '.join(sounds.BANK)}")
        return 1
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for name in names:
            x = sounds.build(name)
            rms = float(np.sqrt(np.mean(x ** 2))) or 1.0
            x = x * float(np.clip(TARGET_RMS / rms, 0.25, 1.0)) * 2.0
            pcm = (np.clip(np.repeat(x[:, None], 2, 1), -1, 1) * 32767).astype("<i2")
            with wave.open(str(out / f"{name}.wav"), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(sounds.SR)
                w.writeframes(pcm.tobytes())
            print(out / f"{name}.wav")
        return 0
    if _backend_cmd() is None:
        print("Není přehrávač (pw-cat / pacat / aplay).")
        return 1
    audio = Audio(volume)
    time.sleep(1.5)  # banka se generuje na pozadí
    try:
        for name in names:
            loop = sounds.BANK[name][1]
            print(f"{name}{' (smyčka)' if loop else ''}", flush=True)
            if loop:
                t_end = time.monotonic() + 5.0
                while time.monotonic() < t_end:
                    audio.loop(name, 1.0)
                    time.sleep(0.05)
                time.sleep(0.6)
            else:
                audio.play(name)
                time.sleep(len(audio._sounds.get(name, [])) / sounds.SR + 0.4)
    except KeyboardInterrupt:
        pass
    finally:
        audio.close()
    return 0
