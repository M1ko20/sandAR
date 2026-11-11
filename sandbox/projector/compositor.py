"""Promítnutí plátna pískoviště na výstup projektoru.

Hry kreslí na plátno = pohled shora na bednu. Kompozitor ho natáhne na síť
trojúhelníků; každý vrchol se posune podle výšky písku (paralaxa) a homografie
projektoru. Hry tedy o geometrii projektoru nic nevědí.
"""

from __future__ import annotations

import moderngl
import numpy as np

from .calibration import ProjectorCalibration

_VS = """
#version 330
uniform sampler2D u_height;
uniform mat3 u_H;
uniform vec2 u_box;
uniform vec2 u_proj_pos;
uniform float u_proj_h;
uniform vec2 u_res;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    float h = texture(u_height, in_uv).r;
    vec2 P = in_uv * u_box;
    vec2 G = P;
    if (u_proj_h > 0.0) {
        G = P + (P - u_proj_pos) * h / max(u_proj_h - h, 1e-3);
    }
    vec3 q = u_H * vec3(G, 1.0);
    vec2 px = q.xy / q.z;
    gl_Position = vec4(px.x / u_res.x * 2.0 - 1.0, 1.0 - px.y / u_res.y * 2.0, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_FS = """
#version 330
uniform sampler2D u_canvas;
uniform float u_brightness;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = vec4(texture(u_canvas, v_uv).rgb * u_brightness, 1.0);
}
"""


class ProjectorCompositor:
    def __init__(self, ctx: moderngl.Context, mesh_resolution: int = 96) -> None:
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self.mesh_resolution = mesh_resolution
        self._mesh_key: tuple | None = None
        self._vao: moderngl.VertexArray | None = None
        self._zero = ctx.texture((1, 1), 1, np.zeros(1, np.float32).tobytes(), dtype="f4")
        self.brightness = 1.0

    def _ensure_mesh(self, box_size: tuple[float, float]) -> None:
        nx = self.mesh_resolution
        ny = max(4, int(round(nx * box_size[1] / box_size[0])))
        if self._mesh_key == (nx, ny):
            return
        u, v = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1))
        verts = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32)
        idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
        tris = np.stack([a, b, c, a, c, d], axis=-1).astype(np.uint32).ravel()
        if self._vao is not None:
            self._vao.release()
        vbo = self.ctx.buffer(verts.tobytes())
        ibo = self.ctx.buffer(tris.tobytes())
        self._vao = self.ctx.vertex_array(self.prog, [(vbo, "2f", "in_uv")], index_buffer=ibo, index_element_size=4)
        self._mesh_key = (nx, ny)

    def draw(self, canvas: moderngl.Texture, height: moderngl.Texture | None, calib: ProjectorCalibration,
             box_size: tuple[float, float], fb_size: tuple[int, int]) -> None:
        """Vykreslí plátno do aktuálního framebufferu (velikost fb_size)."""
        self._ensure_mesh(box_size)
        calib = calib.scaled_to(fb_size)
        H = calib.homography(box_size)
        pos = calib.position or (box_size[0] / 2, box_size[1] / 2)
        canvas.use(0)
        (height or self._zero).use(1)
        self.prog["u_canvas"].value = 0
        self.prog["u_height"].value = 1
        self.prog["u_H"].write(np.ascontiguousarray(H.T, np.float32).tobytes())
        self.prog["u_box"].value = tuple(float(v) for v in box_size)
        self.prog["u_proj_pos"].value = tuple(float(v) for v in pos)
        self.prog["u_proj_h"].value = float(calib.height)
        self.prog["u_res"].value = tuple(float(v) for v in fb_size)
        self.prog["u_brightness"].value = float(self.brightness)
        self.ctx.disable(moderngl.BLEND)
        assert self._vao is not None
        self._vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.BLEND)
