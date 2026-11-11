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


__all__ = ["Game", "discover_games"]
