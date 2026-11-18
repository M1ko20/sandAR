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
        self._glow: dict[str, Any] | None = None

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

    # --- záře (bloom) ------------------------------------------------------------------
    _GLOW_BRIGHT = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform float u_threshold;
void main() {
    vec3 c = vec3(0.0);
    for (int y = -1; y <= 1; y += 2)
        for (int x = -1; x <= 1; x += 2)
            c += texture(u_src, v_uv + vec2(x, y) * u_texel).rgb;
    c *= 0.25;
    float hi = max(c.r, max(c.g, c.b));
    float sat = hi - min(c.r, min(c.g, c.b));
    // svítí jen jasné a barevné (láva, oheň, záře) – bílé texty a pěna jen trochu
    float w = smoothstep(u_threshold, 1.0, hi) * (0.35 + 0.65 * smoothstep(0.15, 0.6, sat));
    f_color = vec4(c * w, 1.0);
}
"""
    _GLOW_BLUR = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_src;
uniform vec2 u_dir;
void main() {
    const float W[5] = float[](0.227, 0.194, 0.122, 0.054, 0.016);
    vec3 c = texture(u_src, v_uv).rgb * W[0];
    for (int i = 1; i < 5; ++i) {
        c += texture(u_src, v_uv + u_dir * float(i)).rgb * W[i];
        c += texture(u_src, v_uv - u_dir * float(i)).rgb * W[i];
    }
    f_color = vec4(c, 1.0);
}
"""
    _GLOW_ADD = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_a;
uniform sampler2D u_b;
uniform float u_strength;
void main() {
    vec3 g = texture(u_a, v_uv).rgb * 0.6 + texture(u_b, v_uv).rgb * 0.8;
    f_color = vec4(g * u_strength, 1.0);
}
"""

    def glow(self, strength: float = 0.9, threshold: float = 0.55) -> None:
        """Záře kolem jasných barevných míst plátna (láva, oheň, magie). Volá mód v ``render``
        po vykreslení scény a PŘED texty (ty pak zůstanou ostré)."""
        if strength <= 0 or self.canvas_texture is None or self.canvas is None:
            return
        cw, ch = self.canvas_size
        g = self._glow
        size = (max(8, cw // 4), max(8, ch // 4))
        if g is None or g["size"] != size:
            if g is not None:
                for k in ("ta", "tb", "tc", "td"):
                    g[k].release()
                for k in ("fa", "fb", "fc", "fd"):
                    g[k].release()
            else:
                g = {"bright": self.ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=self._GLOW_BRIGHT),
                     "blur": self.ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=self._GLOW_BLUR),
                     "add": self.ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=self._GLOW_ADD)}
            small = (max(4, size[0] // 4), max(4, size[1] // 4))
            for k, sz in (("ta", size), ("tb", size), ("tc", small), ("td", small)):
                t = self.ctx.texture(sz, 3, dtype="f2")
                t.filter = (moderngl.LINEAR, moderngl.LINEAR)
                t.repeat_x = t.repeat_y = False
                g[k] = t
            g["fa"], g["fb"] = self.ctx.framebuffer([g["ta"]]), self.ctx.framebuffer([g["tb"]])
            g["fc"], g["fd"] = self.ctx.framebuffer([g["tc"]]), self.ctx.framebuffer([g["td"]])
            g["size"] = size
            self._glow = g

        def vao(prog):
            v = self._vaos.get(prog.glo)
            if v is None:
                v = self.ctx.vertex_array(prog, [(self._quad, "2f", "in_pos")])
                self._vaos[prog.glo] = v
            return v

        self.ctx.disable(moderngl.BLEND)
        # jasná místa ve 1/4 rozlišení
        g["fa"].use()
        self.canvas_texture.use(0)
        set_uniform(g["bright"], "u_src", 0)
        set_uniform(g["bright"], "u_texel", (1.0 / cw, 1.0 / ch))
        set_uniform(g["bright"], "u_threshold", float(threshold))
        vao(g["bright"]).render(moderngl.TRIANGLE_STRIP)

        def blur(src, dst_fbo, w, h):
            set_uniform(g["blur"], "u_src", 0)
            mid = g["tb"] if src is g["ta"] else g["td"]
            mid_fbo = g["fb"] if src is g["ta"] else g["fd"]
            mid_fbo.use()
            src.use(0)
            set_uniform(g["blur"], "u_dir", (1.0 / w, 0.0))
            vao(g["blur"]).render(moderngl.TRIANGLE_STRIP)
            dst_fbo.use()
            mid.use(0)
            set_uniform(g["blur"], "u_dir", (0.0, 1.0 / h))
            vao(g["blur"]).render(moderngl.TRIANGLE_STRIP)

        blur(g["ta"], g["fa"], *size)  # blízká záře
        g["fc"].use()  # široká záře v 1/16
        g["ta"].use(0)
        set_uniform(g["blur"], "u_src", 0)
        set_uniform(g["blur"], "u_dir", (0.0, 0.0))
        vao(g["blur"]).render(moderngl.TRIANGLE_STRIP)
        sw, sh = g["tc"].size
        blur(g["tc"], g["fc"], sw, sh)
        blur(g["tc"], g["fc"], sw, sh)
        # přičíst do plátna
        self.canvas.use()
        self.canvas.viewport = (0, 0, cw, ch)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        g["ta"].use(1)
        g["tc"].use(2)
        set_uniform(g["add"], "u_a", 1)
        set_uniform(g["add"], "u_b", 2)
        set_uniform(g["add"], "u_strength", float(strength))
        vao(g["add"]).render(moderngl.TRIANGLE_STRIP)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
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
