"""Okna GLFW s OpenGL kontextem (moderngl).

Projektorové okno běží přes celou obrazovku na monitoru projektoru, operátorské
okno je na notebooku. Kontexty jsou sdílené, takže operátor vidí stejné textury.
"""

from __future__ import annotations

import ctypes
import logging
import os
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
        # Výchozí je nativní Wayland (správné HiDPI: projektor 1:1, notebook 2×).
        # SANDBOX_GLFW_PLATFORM=x11 vynutí XWayland (pozor: KDE tam všechny výstupy škáluje).
        if sys.platform.startswith("linux") and os.environ.get("SANDBOX_GLFW_PLATFORM") == "x11":
            glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
        if not glfw.init():
            raise RuntimeError("GLFW se nepodařilo inicializovat")
        _initialized = True


def _is_wayland() -> bool:
    return hasattr(glfw, "get_platform") and glfw.get_platform() == glfw.PLATFORM_WAYLAND


class _GlfwLoader:
    """Načítá OpenGL funkce přes GLFW – funguje s jakýmkoli kontextem, který GLFW vytvoří
    (EGL na Waylandu, GLX na X11, CGL na macOS). Výchozí cesta moderngl u druhého,
    sdíleného okna na Linuxu natvrdo hledá GLX, a na Waylandu proto padala."""

    def load_opengl_function(self, name: str) -> int:
        ptr = glfw.get_proc_address(name)
        return ctypes.cast(ptr, ctypes.c_void_p).value or 0 if ptr else 0

    def release(self) -> None:
        pass


def _create_context() -> moderngl.Context:
    if sys.platform == "darwin":
        return moderngl.create_context()
    previous = moderngl._store.default_context
    moderngl._store.default_context = None
    try:
        moderngl.init_context(_GlfwLoader())
        ctx = moderngl._store.default_context
    finally:
        if previous is not None:
            moderngl._store.default_context = previous
    if ctx.version_code < 330:
        raise RuntimeError(f"Potřeba OpenGL 3.3, k dispozici {ctx.version_code}")
    return ctx


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


def operator_window_size(default: tuple[int, int] = (1280, 780)) -> tuple[int, int]:
    """Velikost operátorského okna podle hlavního monitoru (~90 % plochy), v jednotkách okna.

    Na Waylandu jsou to logické body (notebook 2880×1800 při škálování 2× = 1440×900),
    na X11 fyzické pixely, na macOS body – video mode je tam už v bodech."""
    init_glfw()
    mon = glfw.get_primary_monitor()
    if not mon:
        return default
    mode = glfw.get_video_mode(mon)
    w, h = mode.size.width, mode.size.height
    if _is_wayland():
        sx, sy = glfw.get_monitor_content_scale(mon)
        w, h = w / max(sx, 1e-3), h / max(sy, 1e-3)
    return max(800, int(w * 0.9)), max(560, int(h * 0.85))


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
        if position and monitor is None and not _is_wayland():  # Wayland pozici okna nepovolí
            glfw.set_window_pos(self.handle, *position)
        self.fullscreen = monitor is not None
        glfw.make_context_current(self.handle)
        # Na Waylandu swap s vsync blokuje, dokud kompozitor okno nezobrazí – zakryté okno
        # (projekce pod oknem obsluhy) by zastavilo celou smyčku. Tempo tam hlídá app (frame_limit).
        glfw.swap_interval(1 if vsync and not _is_wayland() else 0)
        self.vsync = vsync
        self.ctx = _create_context()
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
        fb = self.ctx.detect_framebuffer(0)  # výchozí framebuffer okna – ne ten, co je zrovna navázaný (plátno)
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


def needs_frame_limit() -> bool:
    """True, když swap neblokuje (Wayland) a smyčku je třeba brzdit ručně."""
    return _is_wayland()


def poll_events() -> None:
    glfw.poll_events()


MOD_SHIFT = glfw.MOD_SHIFT
MOD_CONTROL = glfw.MOD_CONTROL
MOUSE_LEFT = glfw.MOUSE_BUTTON_LEFT
MOUSE_RIGHT = glfw.MOUSE_BUTTON_RIGHT
