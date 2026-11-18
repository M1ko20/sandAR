"""Nápověda promítnutá na písek (klávesa H).

Karta uprostřed plátna: název módu, jak hrát (zalomené na šířku karty), ovládání módu
a přehled všech módů s klávesami. Velikost písma se volí co největší, aby se vše vešlo
do plátna (nikdy pod výšku plátna / 30 – čitelné ze 2 m); co se nevejde, vynechá se
(nejdřív přehled módů, ten má obsluha i na notebooku).
"""

from __future__ import annotations

from typing import Sequence

from ..render import hud
from ..render.text import get_font

TITLE = (1.0, 0.92, 0.45, 1)
KEY = (0.45, 0.85, 1.0, 1)
TEXT = (1, 1, 1, 1)
DIM = (0.80, 0.85, 0.92, 1)


def wrap(text: str, size: int, width: float, bold: bool = False) -> list[str]:
    """Zalomí text na řádky široké nejvýš ``width`` px (podle skutečné šířky písma)."""
    font = get_font(int(size), bold)
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = f"{line} {word}".strip()
            if line and font.getlength(test) > width:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines


def _layout(game, modes: Sequence[tuple[str, str]], f: int, card_w: float, with_modes: bool):
    """Řádky karty pro velikost písma ``f`` -> (seznam (text, velikost, barva, tučně, sloupec), výška)."""
    pad = f * 0.9
    inner = card_w - 2 * pad
    rows: list[tuple] = []
    rows.append((game.title, int(f * 1.5), TITLE, True))
    how = game.intro or game.description
    for line in wrap(how, f, inner):
        rows.append((line, f, TEXT, False))
    if game.keys:
        rows.append(("", f // 2, TEXT, False))
        rows.append(("Ovládání (obsluha)", int(f * 0.85), KEY, True))
        for k, d in game.keys:
            rows.append(((k, d), int(f * 0.85), DIM, False))
    if with_modes:
        rows.append(("", f // 2, TEXT, False))
        rows.append(("Módy", int(f * 0.85), KEY, True))
        rows.append((("__modes__",), int(f * 0.8), DIM, False))
    height = 2 * pad
    for row in rows:
        if row[0] == ("__modes__",):
            per_col = (len(modes) + 2) // 3
            height += per_col * row[1] * 1.3
        else:
            height += row[1] * 1.3
    return rows, height


def draw_help(host, game, game_classes, mode_key) -> None:
    p = host.painter
    cw, ch = host.canvas_size
    card_w = cw * 0.86
    modes = [(mode_key(i), g.title) for i, g in enumerate(game_classes)]
    f_min = int(max(ch / 30, 14))
    chosen = None
    for with_modes in (True, False):
        for f in range(int(ch / 18), f_min - 1, -1):
            rows, height = _layout(game, modes, f, card_w, with_modes)
            if height <= ch * 0.92:
                chosen = (rows, height, f)
                break
        if chosen:
            break
    if chosen is None:
        rows, height = _layout(game, modes, f_min, card_w, False)
        chosen = (rows, height, f_min)
    rows, height, f = chosen
    pad = f * 0.9
    x0, y0 = (cw - card_w) / 2, max(0.0, (ch - height) / 2)
    p.rect(0, 0, cw, ch, (0, 0, 0, 0.45))  # ztlumit hru pod kartou
    p.rect(x0, y0, card_w, height, (0.03, 0.05, 0.10, 0.92))
    p.rect(x0, y0, card_w, f * 0.18, KEY)
    y = y0 + pad
    kf = get_font(int(f * 0.85), True)
    key_w = max([kf.getlength(k) for k, _ in game.keys] or [0]) + f
    for text, size, color, bold in rows:
        if text == ("__modes__",):
            per_col = (len(modes) + 2) // 3
            cols = [modes[i:i + per_col] for i in range(0, len(modes), per_col)]
            fm = size
            while True:  # písmo přehledu módů zmenšit, dokud se tři sloupce nevejdou vedle sebe
                bf, rf = get_font(fm, True), get_font(fm, False)
                widths = [(max(bf.getlength(k) for k, _ in c) + fm * 0.6, max(rf.getlength(t) for _, t in c))
                          for c in cols]
                total = sum(kw + tw for kw, tw in widths) + fm * 1.5 * (len(cols) - 1)
                if total <= card_w - 2 * pad or fm <= 12:
                    break
                fm -= 1
            cx = x0 + pad
            for c, (kw, tw) in zip(cols, widths):
                for r, (k, title) in enumerate(c):
                    cy = y + r * size * 1.3
                    cur = title == game.title
                    p.text(k, cx, cy, size=fm, bold=True, color=TITLE if cur else KEY, outline=2)
                    p.text(title, cx + kw, cy, size=fm, bold=cur, color=TEXT if cur else DIM, outline=2)
                cx += kw + tw + fm * 1.5
            y += per_col * size * 1.3
            continue
        if isinstance(text, tuple):
            k, d = text
            p.text(k, x0 + pad, y, size=size, bold=True, color=KEY, outline=2)
            p.text(d, x0 + pad + key_w, y, size=size, color=TEXT, outline=2)
        elif text:
            p.text(text, x0 + pad, y, size=size, bold=bold, color=color, outline=2)
        y += size * 1.3
    hud.label(p, "H = zavřít nápovědu", cw - f * 0.4, ch - f * 0.3, int(f * 0.7), anchor="br", bold=False)
