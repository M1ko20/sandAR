"""Zvuk pískoviště: mixer v samostatném vlákně, výstup rourou do PipeWire / PulseAudio / ALSA.

Stejný přístup jako u Kinectu (most přes rouru): žádná nová závislost, a když přehrávač
spadne (odpojený reproduktor), mixer ho za chvíli spustí znovu. Bez zvukového výstupu
(testy, snapshot, server) se použije ``NullAudio`` – volání nic nedělají.

Módy používají ``self.sfx`` (viz ``games/base.py``):

* ``sfx.play("coin", gain=1, x=0.3)`` – jednorázový zvuk; ``x`` = poloha v bedně 0..1 zleva
  doprava (stereo panorama – zvuk zní z místa, kde se něco stalo), ``pitch`` = mírná změna výšky.
* ``sfx.loop("rain", level)`` – smyčka, volej každý snímek s požadovanou hlasitostí 0..1.
  Hlasitost se mění plynule; když ji mód přestane volat, smyčka sama dozní (~0,4 s).

Při přepnutí módu aplikace zavolá ``stop_all()``. V attract mode (nikdo nehraje) je hlasitost
ztlumená. Klávesy obsluhy: F9 ztlumit, F10/F11 hlasitost.
"""

from __future__ import annotations

import fcntl
import logging
import shutil
import subprocess
import threading
import time

import numpy as np

from . import sounds

log = logging.getLogger(__name__)

SR = sounds.SR
BLOCK = 1024  # [vzorky] ~21 ms
LOOP_TIMEOUT = 0.25  # [s] smyčka, kterou mód přestal volat, dozní
LOOP_RAMP = 0.35  # [s] plynulá změna hlasitosti smyčky
TARGET_RMS = 0.13  # vyrovnání hlasitosti zvuků banky (tónové zvuky by jinak řvaly víc než šum)


def _backend_cmd() -> list[str] | None:
    if shutil.which("pw-cat"):
        return ["pw-cat", "--playback", "--raw", "--format", "s16", "--rate", str(SR), "--channels", "2",
                "--latency", "40ms", "--media-role", "Game", "-"]
    if shutil.which("pacat"):
        return ["pacat", "--playback", "--raw", "--format=s16le", f"--rate={SR}", "--channels=2",
                "--latency-msec=40"]
    if shutil.which("aplay"):
        return ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SR), "-c", "2", "-B", "60000"]
    return None


class _Voice:
    __slots__ = ("data", "pos", "gain", "target", "pan", "loop", "last", "name")

    def __init__(self, data: np.ndarray, gain: float, pan: float, loop: bool, name: str = "") -> None:
        self.data = data
        self.pos = 0
        self.gain = gain if not loop else 0.0
        self.target = gain
        self.pan = pan
        self.loop = loop
        self.last = time.monotonic()
        self.name = name


class Mixer:
    """Míchá hlasy do stereo bloků (float32). Bezpečné volat z hlavního vlákna i z vlákna výstupu."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._voices: list[_Voice] = []
        self._loops: dict[str, _Voice] = {}
        self.master = 0.8
        self.duck = 1.0  # ztlumení (attract mode)
        self.muted = False
        self._g = 0.0  # vyhlazená celková hlasitost

    def play(self, data: np.ndarray, gain: float, pan: float) -> None:
        with self._lock:
            if len(self._voices) > 48:  # pojistka: nejstarší krátké zvuky pryč
                self._voices = self._voices[-40:]
            self._voices.append(_Voice(data, gain, pan, False))

    def loop(self, name: str, data: np.ndarray, gain: float, pan: float) -> None:
        with self._lock:
            v = self._loops.get(name)
            if v is None:
                v = _Voice(data, gain, pan, True, name)
                self._loops[name] = v
            v.target = gain
            v.pan = 0.85 * v.pan + 0.15 * pan
            v.last = time.monotonic()

    def stop_all(self) -> None:
        with self._lock:
            self._voices.clear()
            for v in self._loops.values():
                v.target = 0.0
                v.last = 0.0

    def render(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), np.float32)
        now = time.monotonic()
        step = n / SR / LOOP_RAMP
        with self._lock:
            for v in self._voices:
                chunk = v.data[v.pos:v.pos + n]
                v.pos += n
                self._add(out, chunk, v.gain, v.gain, v.pan)
            self._voices = [v for v in self._voices if v.pos < len(v.data)]
            dead = []
            for name, v in self._loops.items():
                target = v.target if now - v.last < LOOP_TIMEOUT else 0.0
                g0 = v.gain
                g1 = g0 + float(np.clip(target - g0, -step, step))
                v.gain = g1
                if g0 <= 1e-4 and g1 <= 1e-4:
                    if target <= 0:
                        dead.append(name)
                    continue
                idx = (np.arange(n) + v.pos) % len(v.data)
                v.pos = (v.pos + n) % len(v.data)
                self._add(out, v.data[idx], g0, g1, v.pan)
            for name in dead:
                del self._loops[name]
            want = 0.0 if self.muted else self.master * self.duck
        g0, self._g = self._g, self._g + float(np.clip(want - self._g, -0.05, 0.05))
        out *= np.linspace(g0, self._g, n, dtype=np.float32)[:, None]
        # měkký omezovač: víc zvuků najednou nepřebudí (tanh nad ~0,7)
        return np.where(np.abs(out) < 0.7, out, np.sign(out) * (0.7 + 0.3 * np.tanh((np.abs(out) - 0.7) / 0.3)))

    @staticmethod
    def _add(out: np.ndarray, chunk: np.ndarray, g0: float, g1: float, pan: float) -> None:
        m = len(chunk)
        if m == 0:
            return
        g = np.linspace(g0, g1, m, dtype=np.float32) if g0 != g1 else g0
        # konstantní výkon: střed = obě strany 0,707
        a = (pan + 1) * np.pi / 4
        out[:m, 0] += chunk * g * np.cos(a)
        out[:m, 1] += chunk * g * np.sin(a)

    @property
    def active(self) -> int:
        with self._lock:
            return len(self._voices) + len(self._loops)


class NullAudio:
    """Bez zvuku (testy, snapshot, chybějící výstup)."""

    enabled = False
    muted = True
    volume = 0.0

    def play(self, name: str, gain: float = 1.0, x: float | None = None, pitch: float = 1.0) -> None: ...
    def loop(self, name: str, level: float, x: float | None = None) -> None: ...
    def stop_all(self) -> None: ...
    def set_duck(self, duck: float) -> None: ...
    def toggle_mute(self) -> bool: return True
    def change_volume(self, step: float) -> float: return 0.0
    def close(self) -> None: ...
    def status(self) -> str: return "zvuk vypnutý"


class Audio(NullAudio):
    """Zvukový výstup s mixérem. Zvuky banky se vygenerují na pozadí při startu (~1 s)."""

    enabled = True

    def __init__(self, volume: float = 0.8, cmd: list[str] | None = None) -> None:
        self.mixer = Mixer()
        self.mixer.master = float(np.clip(volume, 0, 1))
        self._cmd = cmd or _backend_cmd()
        self._sounds: dict[str, np.ndarray] = {}
        self._pitched: dict[tuple[str, int], np.ndarray] = {}
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._error = ""
        threading.Thread(target=self._prepare, name="zvuky", daemon=True).start()
        self._thread = threading.Thread(target=self._run, name="zvuk", daemon=True)
        self._thread.start()

    # --- banka ---------------------------------------------------------------------------
    def _prepare(self) -> None:
        for name in sounds.BANK:
            if self._stop.is_set():
                return
            try:
                x = sounds.build(name)
                rms = float(np.sqrt(np.mean(x ** 2))) or 1.0
                self._sounds[name] = (x * float(np.clip(TARGET_RMS / rms, 0.25, 1.0))).astype(np.float32)
            except Exception:  # vadný zvuk nesmí shodit aplikaci
                log.exception("Zvuk %s se nepodařilo vytvořit", name)

    def _get(self, name: str, pitch: float = 1.0) -> np.ndarray | None:
        x = self._sounds.get(name)
        if x is None or abs(pitch - 1.0) < 0.01:
            return x
        key = (name, int(round(pitch * 100)))
        y = self._pitched.get(key)
        if y is None:
            idx = np.arange(0, len(x) - 1, pitch, dtype=np.float32)
            y = np.interp(idx, np.arange(len(x), dtype=np.float32), x).astype(np.float32)
            if len(self._pitched) > 200:
                self._pitched.clear()
            self._pitched[key] = y
        return y

    # --- API pro módy ----------------------------------------------------------------------
    def play(self, name: str, gain: float = 1.0, x: float | None = None, pitch: float = 1.0) -> None:
        data = self._get(name, pitch)
        if data is not None and gain > 0:
            self.mixer.play(data, float(gain), self._pan(x))

    def loop(self, name: str, level: float, x: float | None = None) -> None:
        data = self._sounds.get(name)
        if data is not None:
            self.mixer.loop(name, data, float(np.clip(level, 0, 1.5)), self._pan(x))

    @staticmethod
    def _pan(x: float | None) -> float:
        return 0.0 if x is None else float(np.clip((x - 0.5) * 1.4, -0.7, 0.7))

    def stop_all(self) -> None:
        self.mixer.stop_all()

    def set_duck(self, duck: float) -> None:
        self.mixer.duck = float(np.clip(duck, 0, 1))

    @property
    def muted(self) -> bool:  # type: ignore[override]
        return self.mixer.muted

    @property
    def volume(self) -> float:  # type: ignore[override]
        return self.mixer.master

    def toggle_mute(self) -> bool:
        self.mixer.muted = not self.mixer.muted
        return self.mixer.muted

    def change_volume(self, step: float) -> float:
        self.mixer.master = float(np.clip(round(self.mixer.master + step, 2), 0.0, 1.0))
        return self.mixer.master

    def status(self) -> str:
        if self._error:
            return f"zvuk: {self._error}"
        state = "ztlumený" if self.mixer.muted else f"hlasitost {self.mixer.master * 100:.0f} %"
        return f"zvuk: {state}, hraje {self.mixer.active}"

    # --- výstup --------------------------------------------------------------------------
    def _start_proc(self) -> bool:
        if not self._cmd:
            self._error = "není přehrávač (pw-cat / pacat / aplay)"
            return False
        try:
            self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
        except OSError as exc:
            self._error = str(exc)
            return False
        try:  # malá roura = malé zpoždění (jinak by se v ní nahromadilo ~0,3 s zvuku)
            fcntl.fcntl(self._proc.stdin.fileno(), 1031, BLOCK * 4 * 2)  # F_SETPIPE_SZ
        except OSError:
            pass
        self._error = ""
        log.info("Zvuk: %s", self._cmd[0])
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if (self._proc is None or self._proc.poll() is not None) and not self._start_proc():
                self._stop.wait(5.0)
                continue
            block = self.mixer.render(BLOCK)
            pcm = (np.clip(block, -1, 1) * 32767).astype("<i2").tobytes()
            try:
                self._proc.stdin.write(pcm)  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
            except (BrokenPipeError, OSError, ValueError):
                log.warning("Zvukový výstup spadl, zkusím ho znovu spustit")
                self._kill()
                self._stop.wait(2.0)

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._kill()


def create_audio(enabled: bool, volume: float) -> NullAudio:
    if not enabled:
        return NullAudio()
    if _backend_cmd() is None:
        log.warning("Zvuk: není přehrávač (pw-cat / pacat / aplay) – pískoviště poběží potichu")
        return NullAudio()
    return Audio(volume)
