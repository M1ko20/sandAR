"""Každý mód se musí dát spustit a vykreslit (shadery se zkompilují, text se vykreslí).

Běží bez okna (moderngl standalone kontext); bez OpenGL se test přeskočí.
"""

import pytest

from sandbox.config import Settings


def test_every_game_renders_headless(tmp_path):
    moderngl = pytest.importorskip("moderngl")
    try:
        moderngl.create_standalone_context(require=330).release()
    except Exception as exc:  # stroj bez GPU/EGL
        pytest.skip(f"OpenGL není k dispozici: {exc}")
    from sandbox.games import discover_games
    from sandbox.tools import render_snapshots

    written = render_snapshots(Settings(), tmp_path, frames=6, steps=8)
    ids = {p.name.split("_", 1)[1].rsplit("_", 1)[0] for p in written}
    assert ids == {g.id for g in discover_games()}
