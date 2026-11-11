"""Okna GLFW s OpenGL kontextem (moderngl).

Projektorové okno běží přes celou obrazovku na monitoru projektoru, operátorské
okno je na notebooku. Kontexty jsou sdílené, takže operátor vidí stejné textury.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from typing import Callable

import glfw
import moderngl

log = logging.getLogger(__name__)

_initialized = False


def _addr(monitor) -> int:
    """Adresa monitoru – pyGLFW vrací pro stejný monitor pokaždé nový ukazatel, nelze je porovnávat přímo."""
    return ctypes.cast(monitor, ctypes.c_void_p).value or 0


def init_glfw() -> None:
    global _initialized
    if not _initialized:
        if not glfw.init():
            raise RuntimeError("GLFW se nepodařilo inicializovat")
        _initialized = True


def terminate_glfw() -> None:
    global _initialized
    if _initialized:
        glfw.terminate()
        _initialized = False


def list_monitors() -> list[tuple[int, str, tuple[int, int], bool]]:
    init_glfw()
    primary = glfw.get_primary_monitor()
    out = []
    for i, m in enumerate(glfw.get_monitors()):
        mode = glfw.get_video_mode(m)
        name = glfw.get_monitor_name(m)
        name = name.decode() if isinstance(name, bytes) else str(name)
        out.append((i, name, (mode.size.width, mode.size.height), _addr(m) == _addr(primary)))
    return out


def find_projector_monitor(spec: str):
    """Monitor podle nastavení: 'auto' = první, který není hlavní; číslo; nebo část názvu."""
    init_glfw()
    monitors = glfw.get_monitors()
    if not monitors:
        return None
    if spec.isdigit():
        i = int(spec)
        return monitors[i] if i < len(monitors) else None
    if spec and spec != "auto":
        for m in monitors:
            name = glfw.get_monitor_name(m)
            name = name.decode() if isinstance(name, bytes) else str(name)
            if spec.lower() in name.lower():
                return m
        log.warning("Monitor '%s' nenalezen", spec)
        return None
    primary = glfw.get_primary_monitor()
    others = [m for m in monitors if _addr(m) != _addr(primary)]
    return others[0] if others else None


_KEY_NAMES = {
    glfw.KEY_ENTER: "enter", glfw.KEY_KP_ENTER: "enter", glfw.KEY_ESCAPE: "escape",
    glfw.KEY_TAB: "tab", glfw.KEY_SPACE: "space", glfw.KEY_BACKSPACE: "backspace",
    glfw.KEY_DELETE: "delete", glfw.KEY_UP: "up", glfw.KEY_DOWN: "down",
    glfw.KEY_LEFT: "left", glfw.KEY_RIGHT: "right", glfw.KEY_PAGE_UP: "pageup",
    glfw.KEY_PAGE_DOWN: "pagedown", glfw.KEY_HOME: "home", glfw.KEY_END: "end",
    glfw.KEY_KP_ADD: "plus", glfw.KEY_KP_SUBTRACT: "minus", glfw.KEY_EQUAL: "plus",
    glfw.KEY_MINUS: "minus",
}
_KEY_NAMES.update({glfw.KEY_0 + i: str(i) for i in range(10)})
_KEY_NAMES.update({glfw.KEY_KP_0 + i: str(i) for i in range(10)})
_KEY_NAMES.update({glfw.KEY_F1 + i: f"f{i + 1}" for i in range(12)})


def key_name(key: int, scancode: int) -> str | None:
    """Název klávesy nezávislý na rozložení (číslice podle pozice, písmena podle rozložení)."""
    if key in _KEY_NAMES:
        return _KEY_NAMES[key]
    if glfw.KEY_A <= key <= glfw.KEY_Z:
        name = glfw.get_key_name(key, scancode)
        if isinstance(name, bytes):
            name = name.decode()
        if name and len(name) == 1 and name.isalpha():
            return name.lower()
        return chr(key).lower()
    return None


class Window:
    def __init__(self, title: str, size: tuple[int, int] = (1280, 800), monitor=None,
                 share: "Window | None" = None, vsync: bool = True, position: tuple[int, int] | None = None,
                 visible: bool = True) -> None:
        init_glfw()
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.AUTO_ICONIFY, False)  # klik do operátorského okna nesmí shodit projektor
        glfw.window_hint(glfw.VISIBLE, visible)
        if monitor is not None:
            mode = glfw.get_video_mode(monitor)
            size = (mode.size.width, mode.size.height)
            glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)
        self.handle = glfw.create_window(size[0], size[1], title, monitor,
                                         share.handle if share else None)
        if not self.handle:
            raise RuntimeError("Nepodařilo se vytvořit okno (OpenGL 3.3)")
        if position and monitor is None:
            glfw.set_window_pos(self.handle, *position)
        self.fullscreen = monitor is not None
        glfw.make_context_current(self.handle)
        glfw.swap_interval(1 if vsync else 0)
        self.ctx = moderngl.create_context()
        self.share = share

        self.key_handlers: list[Callable[[str, int], None]] = []
        self.mouse_handlers: list[Callable[[str, float, float, int], None]] = []
        self._buttons: set[int] = set()
        glfw.set_key_callback(self.handle, self._on_key)
        glfw.set_mouse_button_callback(self.handle, self._on_button)
        glfw.set_cursor_pos_callback(self.handle, self._on_cursor)

    # --- vlastnosti --------------------------------------------------------------
    @property
    def framebuffer_size(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self.handle)

    @property
    def size(self) -> tuple[int, int]:
        return glfw.get_window_size(self.handle)

    def cursor(self) -> tuple[float, float]:
        """Poloha kurzoru v pixelech framebufferu (Retina = 2x)."""
        x, y = glfw.get_cursor_pos(self.handle)
        fw, fh = self.framebuffer_size
        w, h = self.size
        return x * fw / max(w, 1), y * fh / max(h, 1)

    def buttons(self) -> set[int]:
        return set(self._buttons)

    def set_cursor_visible(self, visible: bool) -> None:
        glfw.set_input_mode(self.handle, glfw.CURSOR, glfw.CURSOR_NORMAL if visible else glfw.CURSOR_HIDDEN)

    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.handle))

    def make_current(self) -> None:
        glfw.make_context_current(self.handle)

    def begin(self) -> moderngl.Framebuffer:
        self.make_current()
        fb = self.ctx.detect_framebuffer()
        fb.use()
        fb.viewport = (0, 0, *self.framebuffer_size)
        return fb

    def swap(self) -> None:
        glfw.swap_buffers(self.handle)

    def close(self) -> None:
        glfw.destroy_window(self.handle)

    # --- události ---------------------------------------------------------------
    def _on_key(self, _win, key, scancode, action, mods) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        name = key_name(key, scancode)
        if name:
            for h in self.key_handlers:
                h(name, mods)

    def _on_button(self, _win, button, action, mods) -> None:
        x, y = self.cursor()
        if action == glfw.PRESS:
            self._buttons.add(button)
            kind = "press"
        else:
            self._buttons.discard(button)
            kind = "release"
        for h in self.mouse_handlers:
            h(kind, x, y, button)

    def _on_cursor(self, _win, _x, _y) -> None:
        x, y = self.cursor()
        for h in self.mouse_handlers:
            h("move", x, y, -1)


def poll_events() -> None:
    glfw.poll_events()


MOD_SHIFT = glfw.MOD_SHIFT
MOD_CONTROL = glfw.MOD_CONTROL
MOUSE_LEFT = glfw.MOUSE_BUTTON_LEFT
MOUSE_RIGHT = glfw.MOUSE_BUTTON_RIGHT
