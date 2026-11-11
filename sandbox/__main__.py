"""Spuštění:  python -m sandbox [příkaz] [volby]

Příkazy:
  run        spustit pískoviště (výchozí)
  calibrate  spustit rovnou v kalibračním režimu
  games      vypsat dostupné hry
  monitors   vypsat monitory (kvůli výběru projektoru)
  record     nahrát data z Kinectu do souboru
  snapshot   vykreslit všechny hry nad simulací do PNG (bez okna)
  autocalibrate  automatická kalibrace Kinectu i projektoru (barevná kamera Kinectu)
  checkcal   kontrola kalibrace: přesnost v cm + fotky mřížky a vrstevnic (--out složka)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import RECORDINGS_DIR, ROOT, load_settings


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sandbox", description="Interaktivní AR pískoviště",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("command", nargs="?", default="run",
                   choices=["run", "calibrate", "games", "monitors", "record", "snapshot", "autocalibrate", "checkcal"])
    p.add_argument("--sensor", choices=["auto", "kinect2", "sim", "replay"], help="zdroj hloubky")
    p.add_argument("--replay", metavar="SOUBOR", help="nahrávka pro --sensor replay")
    p.add_argument("--pipeline", choices=["gl", "cpu"], help="dekódování Kinectu v2")
    p.add_argument("--game", help="hra po spuštění (id, viz příkaz games)")
    p.add_argument("--monitor", help="monitor projektoru: index nebo část názvu (např. BenQ)")
    p.add_argument("--windowed", action="store_true", help="projekce v okně místo celé obrazovky")
    p.add_argument("--no-operator", action="store_true", help="bez operátorského okna")
    p.add_argument("--settings", type=Path, help="jiný soubor s nastavením")
    p.add_argument("--seconds", type=float, default=10.0, help="délka nahrávky (record)")
    p.add_argument("--out", type=Path, help="výstupní soubor/složka (record, snapshot)")
    p.add_argument("--duration", type=float, help="ukončit po N sekundách (testování)")
    p.add_argument("--snapshot-dir", type=Path, help="při ukončení uložit obraz oken do složky")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    settings = load_settings(args.settings)
    if settings.log.file and args.command in ("run", "calibrate"):
        from logging.handlers import RotatingFileHandler

        log_path = ROOT / settings.log.file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=settings.log.max_bytes, backupCount=settings.log.backups,
                                 encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)
    if args.sensor:
        settings.sensor.kind = args.sensor
    if args.replay:
        settings.sensor.replay_file = args.replay
        settings.sensor.kind = settings.sensor.kind if args.sensor else "replay"
    if args.pipeline:
        settings.sensor.pipeline = args.pipeline
    if args.game:
        settings.start_game = args.game
    if args.monitor:
        settings.display.monitor = args.monitor
    if args.windowed:
        settings.display.windowed = True
    if args.no_operator:
        settings.display.operator_window = False

    if args.command == "games":
        from .games import discover_games

        for i, g in enumerate(discover_games()):
            print(f"{i + 1}. {g.id:10s} {g.title} – {g.description}")
        return 0

    if args.command == "monitors":
        from .render.window import list_monitors, terminate_glfw

        for i, name, (w, h), primary in list_monitors():
            print(f"{i}: {name} {w}x{h}{'  (hlavní)' if primary else ''}")
        terminate_glfw()
        return 0

    if args.command == "record":
        from .tools import record

        out = args.out or RECORDINGS_DIR / "nahravka.npz"
        print(f"Uloženo: {record(settings, args.seconds, out)}")
        return 0

    if args.command == "snapshot":
        from .tools import render_snapshots

        for path in render_snapshots(settings, args.out or ROOT / "snapshots", only=args.game,
                                     steps=int(args.duration * 30) if args.duration else 60):
            print(path)
        return 0

    if args.command in ("autocalibrate", "checkcal"):
        import json

        from . import autocal

        if args.command == "autocalibrate":
            result = autocal.autocalibrate(settings)
        else:
            result = autocal.check_calibration(settings, args.out or ROOT / "snapshots" / "kontrola")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    from .app import SandboxApp
    from .sensor import SensorError

    try:
        app = SandboxApp(settings, calibrate=args.command == "calibrate")
    except SensorError as exc:
        logging.error("%s", exc)
        return 1
    app.run(duration=args.duration, snapshot_dir=args.snapshot_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
