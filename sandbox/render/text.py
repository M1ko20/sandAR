"""Vykreslování textu (včetně české diakritiky) přes Pillow do textur."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
}


@lru_cache(maxsize=64)
def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES["bold" if bold else "regular"] + _FONT_CANDIDATES["regular"]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def render_text(text: str, size: int, color=(255, 255, 255, 255), bold: bool = False,
                outline: int = 0, outline_color=(0, 0, 0, 255)) -> np.ndarray:
    """Vrátí RGBA obrázek (H, W, 4) s textem (víc řádků odděleno \\n)."""
    font = get_font(size, bold)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.multiline_textbbox((0, 0), text, font=font, stroke_width=outline, spacing=size // 4)
    w, h = max(1, right - left + 2), max(1, bottom - top + 2)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).multiline_text((1 - left, 1 - top), text, font=font, fill=tuple(color),
                                       stroke_width=outline, stroke_fill=tuple(outline_color), spacing=size // 4)
    return np.asarray(img)


class TextCache:
    """Cache textur s texty, ať se stejný text nevykresluje každý snímek znovu."""

    def __init__(self, ctx, capacity: int = 256) -> None:
        self.ctx = ctx
        self.capacity = capacity
        self._cache: OrderedDict[tuple, object] = OrderedDict()

    def get(self, text: str, size: int, color=(255, 255, 255, 255), bold: bool = False, outline: int = 0,
            outline_color=(0, 0, 0, 255)):
        key = (text, size, tuple(color), bold, outline, tuple(outline_color))
        tex = self._cache.get(key)
        if tex is not None:
            self._cache.move_to_end(key)
            return tex
        img = render_text(text, size, color, bold, outline, outline_color)
        tex = self.ctx.texture((img.shape[1], img.shape[0]), 4, np.ascontiguousarray(img).tobytes())
        tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self._cache[key] = tex
        if len(self._cache) > self.capacity:
            _, old = self._cache.popitem(last=False)
            old.release()
        return tex
