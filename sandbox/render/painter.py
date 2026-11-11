"""Jednoduché 2D kreslení (obdélníky, čáry, kruhy, obrázky, text) v pixelech.

Souřadnice mají počátek vlevo nahoře. Při kreslení na plátno pískoviště
(``begin(..., screen=False)``) odpovídá řádek y=0 hraně u rohů 1–2, takže
text je na písku čitelný z pohledu, z jakého je otočený projektor.
"""

from __future__ import annotations

import math
from typing import Sequence

import moderngl
import numpy as np

from .text import TextCache

Color = Sequence[float]

_COLOR_VS = """
#version 330
uniform vec2 u_size;
uniform float u_flip;
in vec2 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    vec2 p = in_pos / u_size * 2.0 - 1.0;
    p.y *= u_flip;
    gl_Position = vec4(p, 0.0, 1.0);
    v_color = in_color;
}
"""
_COLOR_FS = """
#version 330
in vec4 v_color;
out vec4 f_color;
void main() { f_color = v_color; }
"""
_TEX_VS = """
#version 330
uniform vec2 u_size;
uniform float u_flip;
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    vec2 p = in_pos / u_size * 2.0 - 1.0;
    p.y *= u_flip;
    gl_Position = vec4(p, 0.0, 1.0);
    v_uv = in_uv;
}
"""
_TEX_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec4 u_tint;
uniform int u_mode;   // 0 = RGBA, 1 = jeden kanál jako šedá
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 c = texture(u_tex, v_uv);
    if (u_mode == 1) c = vec4(c.rrr, 1.0);
    f_color = c * u_tint;
}
"""


class Painter:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._cprog = ctx.program(vertex_shader=_COLOR_VS, fragment_shader=_COLOR_FS)
        self._tprog = ctx.program(vertex_shader=_TEX_VS, fragment_shader=_TEX_FS)
        self._cvbo = ctx.buffer(reserve=6 * 4 * 4096, dynamic=True)
        self._tvbo = ctx.buffer(reserve=4 * 4 * 6, dynamic=True)
        self._cvao = ctx.vertex_array(self._cprog, [(self._cvbo, "2f 4f", "in_pos", "in_color")])
        self._tvao = ctx.vertex_array(self._tprog, [(self._tvbo, "2f 2f", "in_pos", "in_uv")])
        self.text_cache = TextCache(ctx)
        self.size = (1, 1)

    def begin(self, size: tuple[int, int], screen: bool = True) -> None:
        """screen=True: obrazovka (y dolů). screen=False: plátno pískoviště (y=0 = textura v=0)."""
        self.size = size
        flip = -1.0 if screen else 1.0
        for prog in (self._cprog, self._tprog):
            prog["u_size"].value = (float(size[0]), float(size[1]))
            prog["u_flip"].value = flip
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # --- základní tvary ---------------------------------------------------------
    def triangles(self, verts: np.ndarray, colors: np.ndarray | Color) -> None:
        verts = np.asarray(verts, np.float32).reshape(-1, 2)
        if len(verts) == 0:
            return
        colors = np.asarray(colors, np.float32)
        if colors.ndim == 1:
            colors = np.broadcast_to(colors, (len(verts), 4))
        data = np.hstack([verts, colors]).astype(np.float32)
        nbytes = data.nbytes
        if nbytes > self._cvbo.size:
            self._cvbo.orphan(nbytes * 2)
        self._cvbo.write(data.tobytes())
        self._cvao.render(moderngl.TRIANGLES, vertices=len(verts))

    def rect(self, x: float, y: float, w: float, h: float, color: Color) -> None:
        self.triangles(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]]), color)

    def polyline(self, pts, color: Color, width: float = 2.0, closed: bool = False) -> None:
        pts = np.asarray(pts, np.float32).reshape(-1, 2)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        if len(pts) < 2:
            return
        a, b = pts[:-1], pts[1:]
        d = b - a
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
        n *= width / 2
        quads = np.stack([a + n, b + n, b - n, a + n, b - n, a - n], axis=1).reshape(-1, 2)
        self.triangles(quads, color)

    def rect_outline(self, x, y, w, h, color: Color, width: float = 2.0) -> None:
        self.polyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], color, width, closed=True)

    def circle(self, cx: float, cy: float, r: float, color: Color, width: float | None = None, segments: int = 40) -> None:
        ang = np.linspace(0, 2 * math.pi, segments + 1)
        ring = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
        if width is not None:
            self.polyline(ring, color, width)
            return
        center = np.array([[cx, cy]])
        tris = np.concatenate([np.repeat(center, segments, 0)[:, None], ring[:-1, None], ring[1:, None]], axis=1)
        self.triangles(tris.reshape(-1, 2), color)

    # --- obrázky a text ------------------------------------------------------------
    def image(self, tex: moderngl.Texture, x: float, y: float, w: float, h: float,
              tint: Color = (1, 1, 1, 1), uv: tuple[float, float, float, float] = (0, 0, 1, 1),
              gray: bool = False) -> None:
        u0, v0, u1, v1 = uv
        data = np.array([x, y, u0, v0, x + w, y, u1, v0, x + w, y + h, u1, v1,
                         x, y, u0, v0, x + w, y + h, u1, v1, x, y + h, u0, v1], np.float32)
        self._tvbo.write(data.tobytes())
        tex.use(0)
        self._tprog["u_tex"].value = 0
        self._tprog["u_tint"].value = tuple(float(c) for c in tint)
        self._tprog["u_mode"].value = 1 if gray else 0
        self._tvao.render(moderngl.TRIANGLES, vertices=6)

    def text(self, s: str, x: float, y: float, size: int = 24, color: Color = (1, 1, 1, 1),
             anchor: str = "tl", bold: bool = False, outline: int = 0, outline_color: Color = (0, 0, 0, 1),
             background: Color | None = None, padding: float = 6) -> tuple[float, float]:
        """Vykreslí text. anchor = 2 znaky: t/m/b (svisle) + l/c/r (vodorovně). Vrací (šířka, výška)."""
        if not s:
            return 0.0, 0.0
        c255 = tuple(int(round(v * 255)) for v in color)
        o255 = tuple(int(round(v * 255)) for v in outline_color)
        tex = self.text_cache.get(s, int(size), c255, bold, outline, o255)
        w, h = tex.size
        if anchor[0] == "m":
            y -= h / 2
        elif anchor[0] == "b":
            y -= h
        if anchor[1] == "c":
            x -= w / 2
        elif anchor[1] == "r":
            x -= w
        if background is not None:
            self.rect(x - padding, y - padding, w + 2 * padding, h + 2 * padding, background)
        self.image(tex, x, y, w, h)
        return float(w), float(h)
