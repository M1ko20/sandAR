"""RenderHost – služby, které aplikace poskytuje hrám.

Obsahuje OpenGL kontext, plátno pískoviště (pohled shora na bednu), texturu výšek
a pomocníky pro shadery. Funguje i bez okna (``moderngl.create_standalone_context``),
takže hry jde vykreslovat a testovat bez projektoru.
"""

from __future__ import annotations

import time
from typing import Any

import moderngl
import numpy as np

from ..config import Settings
from ..terrain.heightmap import HeightMap
from .glsl import FULLSCREEN_VS, LIB
from .painter import Painter


def set_uniform(prog: moderngl.Program, name: str, value: Any) -> None:
    """Nastaví uniformu, pokud ji shader používá (nepoužité kompilátor vyhodí)."""
    try:
        member = prog[name]
    except KeyError:
        return
    member.value = value  # type: ignore[union-attr]


class RenderHost:
    def __init__(self, ctx: moderngl.Context, settings: Settings) -> None:
        self.ctx = ctx
        self.settings = settings
        self.painter = Painter(ctx)
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self._vaos: dict[int, moderngl.VertexArray] = {}
        self.t0 = time.monotonic()
        self.heightmap: HeightMap | None = None
        self.height_texture: moderngl.Texture | None = None
        self.canvas: moderngl.Framebuffer | None = None
        self.canvas_texture: moderngl.Texture | None = None
        self.box_size = (1.0, 0.75)
        self.messages: list[tuple[str, float]] = []

    # --- plátno a výšky -----------------------------------------------------------
    @property
    def canvas_size(self) -> tuple[int, int]:
        assert self.canvas_texture is not None
        return self.canvas_texture.size

    @property
    def time(self) -> float:
        return time.monotonic() - self.t0

    def configure(self, box_size: tuple[float, float], grid_shape: tuple[int, int]) -> None:
        """(Pře)vytvoří plátno a texturu výšek pro daný rozměr bedny a mřížku."""
        self.box_size = box_size
        cw = int(self.settings.display.canvas_width)
        ch = max(16, int(round(cw * box_size[1] / box_size[0])))
        if self.canvas_texture is None or self.canvas_texture.size != (cw, ch):
            if self.canvas is not None:
                self.canvas.release()
                self.canvas_texture.release()  # type: ignore[union-attr]
            self.canvas_texture = self.ctx.texture((cw, ch), 4)
            self.canvas_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.canvas_texture.repeat_x = self.canvas_texture.repeat_y = False
            self.canvas = self.ctx.framebuffer([self.canvas_texture])
        gh, gw = grid_shape
        if self.height_texture is None or self.height_texture.size != (gw, gh):
            if self.height_texture is not None:
                self.height_texture.release()
            self.height_texture = self.float_texture((gw, gh))
            self.height_texture.write(np.zeros((gh, gw), np.float32).tobytes())

    def float_texture(self, size: tuple[int, int], data: np.ndarray | None = None) -> moderngl.Texture:
        tex = self.ctx.texture(size, 1, None if data is None else np.ascontiguousarray(data, np.float32).tobytes(),
                               dtype="f4")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        return tex

    def update_heightmap(self, hm: HeightMap) -> None:
        if self.height_texture is None or self.height_texture.size != (hm.shape[1], hm.shape[0]) \
                or self.box_size != hm.size:
            self.configure(hm.size, hm.shape)
        assert self.height_texture is not None
        self.height_texture.write(np.ascontiguousarray(hm.heights, np.float32).tobytes())
        self.heightmap = hm

    # --- shadery ---------------------------------------------------------------------
    def program(self, fragment_body: str) -> moderngl.Program:
        """Zkompiluje celoplošný shader; před ``fragment_body`` vloží knihovnu (glsl.LIB)."""
        return self.ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=LIB + "\n" + fragment_body)

    def bind_common(self, prog: moderngl.Program) -> None:
        m = self.settings.map
        assert self.height_texture is not None
        self.height_texture.use(0)
        set_uniform(prog, "u_height", 0)
        set_uniform(prog, "u_box", tuple(float(v) for v in self.box_size))
        set_uniform(prog, "u_canvas", tuple(float(v) for v in self.canvas_size))
        set_uniform(prog, "u_time", float(self.time))
        set_uniform(prog, "u_elev_min", float(m.elevation_min))
        set_uniform(prog, "u_elev_max", float(m.elevation_max))
        set_uniform(prog, "u_water_level", float(m.water_level))
        set_uniform(prog, "u_contour_interval", float(m.contour_interval))
        set_uniform(prog, "u_major_every", int(m.major_every))

    def draw_fullscreen(self, prog: moderngl.Program, **uniforms: Any) -> None:
        """Vykreslí shader přes celé plátno. Další textury předej jako uniformy typu (texture, unit)."""
        self.bind_common(prog)
        for name, value in uniforms.items():
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], moderngl.Texture):
                value[0].use(value[1])
                set_uniform(prog, name, value[1])
            else:
                set_uniform(prog, name, value)
        vao = self._vaos.get(prog.glo)
        if vao is None:
            vao = self.ctx.vertex_array(prog, [(self._quad, "2f", "in_pos")])
            self._vaos[prog.glo] = vao
        self.ctx.disable(moderngl.BLEND)
        vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.BLEND)

    def begin_canvas(self) -> None:
        assert self.canvas is not None
        self.canvas.use()
        self.canvas.viewport = (0, 0, *self.canvas_size)
        self.painter.begin(self.canvas_size, screen=False)

    # --- převody souřadnic -------------------------------------------------------------
    def box_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        cw, ch = self.canvas_size
        return x / self.box_size[0] * cw, y / self.box_size[1] * ch

    def grid_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        assert self.heightmap is not None
        gh, gw = self.heightmap.shape
        cw, ch = self.canvas_size
        return (ix + 0.5) / gw * cw, (iy + 0.5) / gh * ch

    def altitude(self, h: float) -> float:
        """Výška písku [m] -> 'nadmořská výška' na mapě [m n. m.]."""
        m = self.settings.map
        return m.base_altitude + h * m.vertical_scale

    def notify(self, text: str, seconds: float = 3.0) -> None:
        self.messages.append((text, time.monotonic() + seconds))

    def active_messages(self) -> list[str]:
        now = time.monotonic()
        self.messages = [(t, until) for t, until in self.messages if until > now]
        return [t for t, _ in self.messages]
