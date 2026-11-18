"""Jednotný vzhled textů na písku: čitelné ze 2 m, kontrastní v tmavé místnosti.

* ``size(canvas, k)`` – velikost písma (základ = výška plátna / 22, nikdy pod / 30),
* ``label`` – text s tmavým podkladem (čte se na světlé i tmavé mapě),
* ``legend`` – řádek barevných čtverečků s popisky (co znamenají barvy),
* ``bar`` – ukazatel postupu.
"""

from __future__ import annotations

from typing import Sequence

Color = Sequence[float]

WHITE = (1.0, 1.0, 1.0, 1.0)
YELLOW = (1.0, 0.92, 0.35, 1.0)
GREEN = (0.35, 1.0, 0.45, 1.0)
PANEL = (0.0, 0.03, 0.08, 0.62)


def size(canvas: tuple[int, int], k: float = 1.0) -> int:
    """Velikost písma čitelná ze 2 m: základ výška plátna / 22, nejmenší výška / 30."""
    ch = canvas[1]
    return int(max(ch / 30, ch / 22 * k, 16))


def fit(text: str, sz: float, max_w: float, bold: bool = True) -> int:
    """Největší velikost písma <= sz, při které je text široký nejvýš max_w px."""
    from .text import get_font

    sz = int(sz)
    while sz > 12 and get_font(sz, bold).getlength(text) > max_w:
        sz = int(sz * 0.92)
    return sz


def label(p, text: str, x: float, y: float, sz: int, color: Color = WHITE, anchor: str = "mc",
          bold: bool = True, bg: Color | None = PANEL, rotation: float = 0.0, alpha: float = 1.0) -> tuple[float, float]:
    """Text s obrysem a (volitelně) tmavým podkladem."""
    return p.text(text, x, y, size=int(sz), anchor=anchor, bold=bold, color=color, outline=max(2, int(sz) // 11),
                  background=bg, padding=sz * 0.28, rotation=rotation, alpha=alpha)


def legend(p, items: Sequence[tuple[Color, str]], x: float, y: float, sz: int, anchor: str = "bc",
           gap: float = 1.2) -> None:
    """Řádek „■ popis   ■ popis“ se středem / okrajem v (x, y)."""
    sz = int(sz)
    widths = []
    for _, text in items:
        tex = p.text_cache.get(text, sz, (255, 255, 255, 255), True, max(2, sz // 11), (0, 0, 0, 255))
        widths.append(tex.size[0])
    sw = sz * 0.85  # barevný čtvereček
    total = sum(sw + sz * 0.35 + w for w in widths) + gap * sz * (len(items) - 1)
    pad = sz * 0.35
    left = x - total / 2 if anchor[1] == "c" else x - total if anchor[1] == "r" else x
    top = y - sz * 1.2 if anchor[0] == "b" else y if anchor[0] == "t" else y - sz * 0.6
    p.rect(left - pad, top - pad * 0.6, total + 2 * pad, sz * 1.2 + pad * 1.2, PANEL)
    cx = left
    cy = top + sz * 0.6
    for (col, text), w in zip(items, widths):
        p.rect(cx, cy - sw / 2, sw, sw, (0, 0, 0, 1))
        p.rect(cx + 2, cy - sw / 2 + 2, sw - 4, sw - 4, col)
        cx += sw + sz * 0.35
        p.text(text, cx, cy, size=sz, anchor="ml", bold=True, color=WHITE, outline=max(2, sz // 11))
        cx += w + gap * sz


def bar(p, x: float, y: float, w: float, h: float, frac: float, color: Color, back: Color = (0, 0, 0, 0.65)) -> None:
    """Vodorovný ukazatel (x, y = levý horní roh)."""
    frac = max(0.0, min(1.0, frac))
    p.rect(x - 3, y - 3, w + 6, h + 6, back)
    p.rect(x, y, w * frac, h, color)
