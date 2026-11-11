"""Kalibrační režim (klávesa C).

Postup:
1. Uhlaď písek.
2. V okně projektoru přetáhni myší úchyty 1–4 do rohů bedny (na úrovni písku).
3. V operátorském okně klikni do hloubkového obrazu na stejné rohy 1–4.
4. Enter – Kinect změří rovinu písku, rozměry bedny a kalibrace se uloží.

Projektor během kalibrace ukazuje mřížku po 10 cm a živé vrstevnice, takže je
hned vidět, jestli obraz sedí (polož na písek hrnek nebo ruku a zkontroluj obrys).
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np

from ..render.window import MOUSE_LEFT, MOUSE_RIGHT
from ..terrain.calibration import CalibrationError, KinectCalibration, detect_box_corners, unmirror_corners

if TYPE_CHECKING:
    from ..app import SandboxApp

log = logging.getLogger(__name__)

PATTERN = """
uniform int u_has_height;
void main() {
    vec2 uv = v_uv;
    vec2 p = uv * u_box;
    vec3 c = vec3(0.0);
    if (u_has_height == 1) {
        float h = height_at(uv);
        c = elevation_color(h) * 0.45;
        vec2 cl = contours(h);
        c = mix(c, vec3(1.0), clamp(cl.x * 0.7 + cl.y, 0.0, 1.0));
    }
    vec2 g = abs(fract(p / 0.1 + 0.5) - 0.5) * 0.1;
    vec2 fw = fwidth(p);
    float grid = max(1.0 - smoothstep(0.6 * fw.x, 1.6 * fw.x, g.x), 1.0 - smoothstep(0.6 * fw.y, 1.6 * fw.y, g.y));
    c = mix(c, vec3(0.45, 0.45, 0.5), grid * 0.8);
    float edge = min(min(p.x, u_box.x - p.x), min(p.y, u_box.y - p.y));
    c = mix(c, vec3(1.0), 1.0 - smoothstep(0.003, 0.007, edge));
    f_color = vec4(c, 1.0);
}
"""

HANDLE_COLORS = [(1.0, 0.3, 0.3, 1), (0.3, 1.0, 0.3, 1), (0.35, 0.6, 1.0, 1), (1.0, 0.9, 0.2, 1)]


class CalibrationMode:
    def __init__(self, app: "SandboxApp") -> None:
        self.app = app
        self.active = False
        self.prog = app.host.program(PATTERN)
        self.depth_corners: list[np.ndarray] = []
        self.corners_dirty = False
        self.selected = 0
        self._drag_handle: int | None = None
        self._drag_corner: int | None = None
        self.capturing = False
        self._purpose = "plane"  # co se děje se zachycenými snímky: "plane" | "detect"

    # --- vstup a výstup ---------------------------------------------------------------
    def enter(self) -> None:
        app = self.app
        self.active = True
        self._backup = (copy.deepcopy(app.proj_cal), app.terrain.calibration)
        kc = app.terrain.calibration
        self.depth_corners = [np.array(c, float) for c in kc.corners_px] if kc else []
        self.corners_dirty = kc is None
        self.capturing = False
        app.proj_win.set_cursor_visible(True)
        if not self.depth_corners:
            self.auto_detect()

    def auto_detect(self) -> None:
        """Najde rohy bedny v hloubkovém obraze automaticky (průměr 15 snímků)."""
        self.app.terrain.start_capture(15)
        self.capturing = True
        self._purpose = "detect"

    def leave(self, keep: bool) -> None:
        app = self.app
        if not keep:
            proj, kinect = self._backup
            app.proj_cal = proj
            if kinect is not app.terrain.calibration:
                app.set_kinect_calibration(kinect)
        self.active = False
        self.capturing = False
        app.proj_win.set_cursor_visible(False)

    # --- logika -----------------------------------------------------------------------
    def update(self) -> None:
        if not self.capturing:
            return
        depth = self.app.terrain.capture_result()
        if depth is None:
            return
        self.capturing = False
        if self._purpose == "detect":
            corners = detect_box_corners(self.app.sensor.intrinsics, depth)
            if corners is None:
                self.app.host.notify("Rohy bedny se nepodařilo najít – klikni je ručně", 6)
            else:
                try:  # obraz Kinectu je zrcadlený – rohy seřadit tak, aby plátno nebylo zrcadlově
                    probe = KinectCalibration.from_depth(self.app.sensor.intrinsics, depth, corners)
                    corners = unmirror_corners(self.app.sensor.intrinsics, corners, probe.normal)
                except CalibrationError:
                    pass
                self.depth_corners = [c for c in corners]
                self.corners_dirty = True
                self.app.host.notify("Rohy bedny nalezeny – zkontroluj je a stiskni Enter", 6)
            return
        try:
            cal = KinectCalibration.from_depth(self.app.sensor.intrinsics, depth, np.array(self.depth_corners))
        except CalibrationError as exc:
            self.app.host.notify(f"Kalibrace selhala: {exc}", 6)
            return
        self.app.set_kinect_calibration(cal)
        pc = self.app.proj_cal
        cx, cy, ch = cal.camera_position_box()
        if pc.height <= 0:  # odhad: projektor je u Kinectu
            pc.position, pc.height = (cx, cy), ch
        self.corners_dirty = False
        self.app.save_calibration()
        W, D = cal.size
        self.app.host.notify(f"Uloženo: pískoviště {W * 100:.0f} × {D * 100:.0f} cm, "
                             f"Kinect {ch * 100:.0f} cm nad pískem", 8)

    def confirm(self) -> None:
        if self.corners_dirty or self.app.terrain.calibration is None:
            if len(self.depth_corners) != 4:
                self.app.host.notify("Nejdřív klikni v operátorském okně na 4 rohy bedny", 5)
                return
            self.app.terrain.start_capture(30)
            self.capturing = True
            self._purpose = "plane"
            self.app.host.notify("Měřím rovinu písku… nehýbej s pískem", 3)
        else:
            self.app.save_calibration()
            self.app.host.notify("Kalibrace projektoru uložena", 4)

    def key(self, name: str, shift: bool) -> bool:
        pc = self.app.proj_cal
        step = 10.0 if shift else 1.0
        moves = {"left": (-step, 0), "right": (step, 0), "up": (0, -step), "down": (0, step)}
        if name == "escape":
            self.leave(keep=False)
            self.app.host.notify("Kalibrace zrušena")
        elif name == "enter":
            self.confirm()
        elif name in ("1", "2", "3", "4"):
            self.selected = int(name) - 1
        elif name in moves:
            pc.corners[self.selected] += moves[name]
        elif name == "r":
            from ..projector.calibration import ProjectorCalibration

            fresh = ProjectorCalibration.default(pc.resolution)
            pc.corners = fresh.corners
        elif name == "a":
            self.auto_detect()
        elif name == "o" and len(self.depth_corners) == 4:  # pootočit pořadí rohů 1→2→3→4
            self.depth_corners = self.depth_corners[1:] + self.depth_corners[:1]
            self.corners_dirty = True
        elif name == "x" and len(self.depth_corners) == 4:  # zrcadlit pořadí (prohodí 2 a 4)
            c = self.depth_corners
            self.depth_corners = [c[0], c[3], c[2], c[1]]
            self.corners_dirty = True
        elif name == "backspace" and self.depth_corners:
            self.depth_corners.pop()
            self.corners_dirty = True
        elif name in ("pageup", "pagedown"):
            if pc.height <= 0 and self.app.terrain.calibration:
                cx, cy, ch = self.app.terrain.calibration.camera_position_box()
                pc.position, pc.height = (cx, cy), ch
            pc.height = max(0.0, pc.height + (0.01 if name == "pageup" else -0.01))
            self.app.host.notify(f"Výška projektoru {pc.height * 100:.0f} cm (korekce paralaxy)")
        elif name == "c":
            self.leave(keep=True)
        else:
            return name not in ("tab", "h")  # ostatní klávesy hry během kalibrace nedostanou
        return True

    # --- myš ---------------------------------------------------------------------------
    def projector_mouse(self, kind: str, x: float, y: float, button: int) -> None:
        corners = self.app.proj_cal.corners
        if kind == "press" and button == MOUSE_LEFT:
            d = np.linalg.norm(corners - [x, y], axis=1)
            i = int(np.argmin(d))
            if d[i] < 80:
                self._drag_handle = i
                self.selected = i
        elif kind == "release":
            self._drag_handle = None
        elif kind == "move" and self._drag_handle is not None:
            corners[self._drag_handle] = (x, y)

    def depth_mouse(self, kind: str, u: float | None, v: float | None, button: int) -> None:
        if kind == "release":
            self._drag_corner = None
            return
        if u is None or v is None:
            return
        p = np.array([u, v])
        if kind == "press" and button == MOUSE_RIGHT:
            self.depth_corners = []
            self.corners_dirty = True
        elif kind == "press" and button == MOUSE_LEFT:
            if self.depth_corners:
                d = [np.linalg.norm(c - p) for c in self.depth_corners]
                i = int(np.argmin(d))
                if d[i] < 12:
                    self._drag_corner = i
                    return
            if len(self.depth_corners) < 4:
                self.depth_corners.append(p)
                self.corners_dirty = True
        elif kind == "move" and self._drag_corner is not None:
            self.depth_corners[self._drag_corner] = p
            self.corners_dirty = True

    # --- kreslení ----------------------------------------------------------------------
    def render_canvas(self) -> None:
        has = self.app.host.heightmap is not None and not self.corners_dirty
        self.app.host.draw_fullscreen(self.prog, u_has_height=int(has))

    def render_projector_overlay(self, fb_size: tuple[int, int]) -> None:
        p = self.app.host.painter
        p.begin(fb_size, screen=True)
        for i, (x, y) in enumerate(self.app.proj_cal.corners):
            col = HANDLE_COLORS[i]
            r = 26 if i == self.selected else 18
            p.circle(x, y, r, col, width=3)
            p.polyline([(x - r - 10, y), (x + r + 10, y)], col, 2)
            p.polyline([(x, y - r - 10), (x, y + r + 10)], col, 2)
            dx = 34 if x < fb_size[0] / 2 else -34
            dy = 34 if y < fb_size[1] / 2 else -34
            p.text(str(i + 1), x + dx, y + dy, size=40, anchor="mc", bold=True, color=col, outline=3)
        w, h = fb_size
        lines = "KALIBRACE · přetáhni úchyty 1–4 do rohů bedny · Enter = uložit · Esc = zrušit"
        if self.capturing:
            what = "Hledám bednu" if self._purpose == "detect" else "Měřím rovinu písku"
            lines = f"{what}… {self.app.terrain.capture_progress() * 100:.0f} %"
        p.text(lines, w / 2, h / 2, size=26, anchor="mc", color=(1, 1, 1, 1), outline=3,
               background=(0, 0, 0, 0.55), padding=10)

    def instructions(self) -> list[str]:
        n = len(self.depth_corners)
        return [
            "KALIBRACE",
            "1. Uhlaď písek.",
            "2. V okně projektoru přetáhni úchyty 1–4 do rohů bedny (na úrovni písku).",
            "   Šipky = jemný posun vybraného úchytu (Shift = ×10), klávesy 1–4 = výběr, R = reset.",
            f"3. Rohy 1–4 v hloubkovém obraze vlevo ({n}/4): A = najít automaticky, nebo klikni ručně.",
            "   Musí to být STEJNÉ fyzické rohy jako 1–4 na projektoru: O = pootočit pořadí, X = zrcadlit.",
            "   Tažením roh posuneš, pravé tlačítko = smazat rohy, Backspace = poslední roh.",
            "4. Enter = změřit a uložit.  PageUp/PageDown = výška projektoru (paralaxa).",
            "Esc = zrušit beze změn, C = odejít a ponechat.",
        ]
