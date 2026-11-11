"""Hry a výukové režimy. Každý modul v tomto balíčku se načte automaticky."""

from __future__ import annotations

import importlib
import logging
import pkgutil

from .base import Game

log = logging.getLogger(__name__)


def discover_games() -> list[type[Game]]:
    """Najde všechny potomky Game v balíčku sandbox.games (seřazené podle order)."""
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name == "base":
            continue
        try:
            importlib.import_module(f"{__name__}.{info.name}")
        except Exception:
            log.exception("Hru %s se nepodařilo načíst", info.name)
    found: dict[str, type[Game]] = {}
    stack = list(Game.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.id:
            found[cls.id] = cls
    return sorted(found.values(), key=lambda c: (c.order, c.id))


def mode_key(index: int) -> str:
    """Klávesa pro mód s pořadím `index` (0 = první): 1–9, 0, pak Shift+1–9."""
    if index < 9:
        return str(index + 1)
    if index == 9:
        return "0"
    if index < 19:
        return f"Shift+{index - 9}"
    return "Tab"


def mode_index(name: str, shift: bool) -> int | None:
    """Opak mode_key: číslice (+Shift) -> pořadí módu, jinak None."""
    if not (len(name) == 1 and name.isdigit()):
        return None
    d = int(name)
    if shift:
        return None if d == 0 else 9 + d
    return 9 if d == 0 else d - 1


__all__ = ["Game", "discover_games", "mode_index", "mode_key"]
