"""Zvuky pískoviště – syntetizované v numpy (žádné soubory, žádné licence).

Každý zvuk je funkce, která vrátí mono float32 pole při ``SR`` Hz s vrcholem ~0,9.
Smyčky (déšť, vítr, moře…) se tvarují filtrem ve frekvenční oblasti – FFT filtr je kruhový,
takže smyčka navazuje bez švu. Krátké zvuky (zvonky, fanfáry) se skládají z tónů se
skutečnými poměry alikvót zvonu a s exponenciálním doznějím.

``BANK`` = {název: (funkce, smyčka?)}; ``build(name)`` zvuk vytvoří (výsledek se kešuje v audio.py).
"""

from __future__ import annotations

import numpy as np

SR = 48000


# --- stavební kameny ---------------------------------------------------------------------
def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR), dtype=np.float32) / SR


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _shape(x: np.ndarray, gain_fn) -> np.ndarray:
    """Filtr ve frekvenční oblasti: gain_fn(f [Hz]) -> zesílení. Kruhový (smyčky bez švu)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(spec * gain_fn(f), len(x)).astype(np.float32)


def _band(lo: float, hi: float, slope: float = 2.0):
    """Měkká pásmová propust (strmost ``slope`` řádů)."""
    def g(f):
        f = np.maximum(f, 1.0)
        return 1.0 / (1.0 + (lo / f) ** (2 * slope)) / (1.0 + (f / hi) ** (2 * slope))
    return g


def _noise(seconds: float, seed: int, color: float = 0.0, lo: float = 20, hi: float = 16000) -> np.ndarray:
    """Barevný šum: color 0 bílý, 1 růžový, 2 hnědý; ořezaný do pásma lo..hi."""
    x = _rng(seed).standard_normal(int(seconds * SR)).astype(np.float32)
    band = _band(lo, hi)
    return _shape(x, lambda f: band(f) / np.maximum(f, 20.0) ** (color / 2))


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return (x * (peak / m)).astype(np.float32)


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    """Obálka: lineární náběh ``attack`` s, pak exponenciální dozvuk s časovou konstantou ``decay``."""
    t = np.arange(n, dtype=np.float32) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / decay)


def _fade(x: np.ndarray, fin: float = 0.005, fout: float = 0.03) -> np.ndarray:
    n = len(x)
    i, o = int(fin * SR), int(fout * SR)
    if i:
        x[:i] *= np.linspace(0, 1, i, dtype=np.float32)
    if o:
        x[-o:] *= np.linspace(1, 0, o, dtype=np.float32)
    return x


def _bell(freq: float, seconds: float, decay: float = 0.9, bright: float = 1.0) -> np.ndarray:
    """Zvonek: alikvóty skutečného zvonu (hum, prima, tercie, kvinta, nominál…), vyšší dozní dřív."""
    t = _t(seconds)
    out = np.zeros_like(t)
    for ratio, amp, dk in ((0.5, 0.25, 1.4), (1.0, 1.0, 1.0), (1.19, 0.35, 0.7), (1.5, 0.3, 0.6),
                           (2.0, 0.55 * bright, 0.5), (2.74, 0.25 * bright, 0.35), (4.07, 0.15 * bright, 0.2)):
        out += amp * np.sin(2 * np.pi * freq * ratio * t) * np.exp(-t / (decay * dk))
    return _fade(out * np.clip(t / 0.002, 0, 1), 0.0, min(0.25, seconds * 0.3))


def _tone(freq: float, seconds: float, harmonics=(1.0, 0.5, 0.3, 0.15), vibrato: float = 0.0) -> np.ndarray:
    t = _t(seconds)
    ph = 2 * np.pi * freq * t + vibrato * np.sin(2 * np.pi * 5.5 * t)
    return sum(a * np.sin((k + 1) * ph) for k, a in enumerate(harmonics)).astype(np.float32)


def _mix(length: float, parts: list[tuple[float, np.ndarray]]) -> np.ndarray:
    out = np.zeros(int(length * SR), np.float32)
    for start, x in parts:
        i = int(start * SR)
        n = min(len(x), len(out) - i)
        if n > 0:
            out[i:i + n] += x[:n]
    return out


def _sprinkle(length: float, seed: int, rate: float, make, loop: bool = True) -> np.ndarray:
    """Náhodně rozházené krátké zvuky (kapky, praskání, cvrlikání). U smyčky přetékají na začátek."""
    rng = _rng(seed)
    n = int(length * SR)
    out = np.zeros(n, np.float32)
    for _ in range(rng.poisson(rate * length)):
        x = make(rng)
        i = int(rng.integers(0, n))
        idx = (np.arange(len(x)) + i)
        if loop:
            np.add.at(out, idx % n, x)
        else:
            ok = idx < n
            out[idx[ok]] += x[ok]
    return out


def _drop(rng) -> np.ndarray:
    """Kapka: krátký zvonivý „tik“ s klesající výškou (bublinka u hladiny)."""
    f0 = rng.uniform(1800, 4200)
    d = rng.uniform(0.012, 0.03)
    t = _t(d * 3)
    f = f0 * (1 + 0.6 * np.exp(-t / d))
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / d) * rng.uniform(0.2, 1.0)).astype(np.float32)


def _crack(rng) -> np.ndarray:
    """Prasknutí (oheň, láva): krátký šumový impulz."""
    n = int(rng.uniform(0.002, 0.012) * SR)
    x = rng.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / (n / 4)).astype(np.float32)
    return x * rng.uniform(0.2, 1.0)


def _chirp(rng) -> np.ndarray:
    """Ptačí cvrlikání: 2–5 rychlých frekvenčních obloučků."""
    out = []
    base = rng.uniform(2600, 4800)
    for _ in range(int(rng.integers(2, 6))):
        d = rng.uniform(0.04, 0.09)
        t = _t(d)
        k = t / d
        f = base * (1 + rng.uniform(-0.25, 0.35) * np.sin(np.pi * k))
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * k) ** 2
        out.append(s.astype(np.float32))
        out.append(np.zeros(int(rng.uniform(0.02, 0.06) * SR), np.float32))
    return np.concatenate(out) * rng.uniform(0.15, 0.5)


def _swell(n: int, period: float, depth: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    return (1 - depth) + depth * (0.5 + 0.5 * np.sin(2 * np.pi * t / period + phase)) ** 2


# --- smyčky (prostředí) -------------------------------------------------------------------
def rain() -> np.ndarray:
    hiss = _noise(6, 1, color=1.0, lo=500, hi=9000)
    hiss *= _swell(len(hiss), 3.0, 0.15)
    return _norm(0.5 * _norm(hiss) + _norm(_sprinkle(6, 2, 160, _drop)) * 0.6)


def wind() -> np.ndarray:
    x = _noise(8, 3, color=1.6, lo=60, hi=1400)
    x *= _swell(len(x), 8.0, 0.65) * _swell(len(x), 2.0, 0.2, 1.0)
    return _norm(x, 0.8)


def stream() -> np.ndarray:
    base = _noise(6, 4, color=0.8, lo=250, hi=3500)
    bub = _sprinkle(6, 5, 70, lambda r: _drop(r) * 0.6)
    return _norm(_norm(base) * 0.55 + _norm(bub) * 0.5, 0.8)


def sea() -> np.ndarray:
    x = _noise(8, 6, color=1.2, lo=80, hi=5000)
    # dvě vlny za 8 s: příboj nabíhá a opadá, výšky se při nárazu zjasní
    sw = _swell(len(x), 4.0, 0.85)
    bright = _noise(8, 7, color=0.5, lo=1500, hi=9000) * _swell(len(x), 4.0, 1.0, 0.6) ** 2
    return _norm(_norm(x * sw) + 0.35 * _norm(bright), 0.85)


def forest() -> np.ndarray:
    air = _noise(10, 8, color=1.6, lo=80, hi=1800) * _swell(int(10 * SR), 10.0, 0.5)
    birds = _sprinkle(10, 9, 0.9, _chirp)
    return _norm(_norm(air) * 0.45 + _norm(birds) * 0.55, 0.8)


def lava() -> np.ndarray:
    rum = _noise(6, 10, color=2.0, lo=25, hi=220)
    bubble = _sprinkle(6, 11, 6, lambda r: _bell(r.uniform(70, 140), 0.25, 0.06, 0.2) * r.uniform(0.3, 0.8))
    crack = _sprinkle(6, 12, 45, _crack)
    return _norm(_norm(rum) * 0.7 + _norm(bubble) * 0.3 + _norm(_shape(crack, _band(800, 8000))) * 0.25, 0.85)


def rumble() -> np.ndarray:
    """Hluboké dunění země (sopka se probouzí)."""
    x = _noise(6, 13, color=2.2, lo=18, hi=110)
    return _norm(x * _swell(len(x), 1.5, 0.5), 0.9)


def roar() -> np.ndarray:
    """Hukot velké vody (povodeň, valící se vlna)."""
    x = _noise(6, 14, color=1.0, lo=120, hi=4000)
    return _norm(x * _swell(len(x), 2.0, 0.25), 0.85)


def roll() -> np.ndarray:
    """Kutálení míčku po zrnité drti."""
    base = _noise(3, 15, color=1.4, lo=90, hi=900)
    grit = _shape(_sprinkle(3, 16, 220, _crack), _band(1500, 6000)) * 0.25
    return _norm(_norm(base) + grit, 0.8)


def fire() -> np.ndarray:
    roarx = _noise(5, 17, color=1.5, lo=60, hi=700) * _swell(int(5 * SR), 1.3, 0.4)
    crack = _shape(_sprinkle(5, 18, 30, _crack), _band(900, 7000))
    return _norm(_norm(roarx) * 0.5 + _norm(crack) * 0.7, 0.8)


def tension() -> np.ndarray:
    """Napětí před koncem: tichý pulzující tón (tep), smyčka 2 s = 2 údery."""
    t = _t(2.0)
    beat = np.zeros_like(t)
    for s in (0.0, 1.0):
        k = np.clip(t - s, 0, None)
        beat += np.sin(2 * np.pi * 55 * k) * np.exp(-k / 0.12) * (t >= s)
        k2 = np.clip(t - s - 0.18, 0, None)
        beat += 0.6 * np.sin(2 * np.pi * 50 * k2) * np.exp(-k2 / 0.12) * (t >= s + 0.18)
    return _norm(beat.astype(np.float32), 0.9)


# --- jednorázové zvuky -------------------------------------------------------------------
def thunder() -> np.ndarray:
    n = int(4.5 * SR)
    crack = _noise(4.5, 20, color=0.6, lo=300, hi=9000) * _env(n, 0.004, 0.12)
    roll_ = _noise(4.5, 21, color=2.0, lo=25, hi=400)
    env = _env(n, 0.08, 1.4) * (0.7 + 0.3 * _swell(n, 0.7, 1.0))
    return _fade(_norm(_norm(crack) * 0.6 + _norm(roll_ * env) * 0.9))


def boom() -> np.ndarray:
    """Výbuch sopky: úder, hluboký propad a dunění."""
    n = int(3.5 * SR)
    t = _t(3.5)
    sub = np.sin(2 * np.pi * np.cumsum(30 + 60 * np.exp(-t / 0.25)) / SR) * _env(n, 0.005, 0.9)
    blast = _noise(3.5, 22, color=1.5, lo=30, hi=3000) * _env(n, 0.003, 0.5)
    return _fade(_norm(sub * 0.8 + _norm(blast) * 0.7))


def thud() -> np.ndarray:
    """Dopad lávové bomby / kamene."""
    t = _t(0.5)
    x = np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-t / 0.04)) / SR) * np.exp(-t / 0.12)
    x += 0.3 * _noise(0.5, 23, color=1.0, lo=100, hi=2500) * np.exp(-t / 0.04)
    return _fade(_norm(x.astype(np.float32)))


def crash() -> np.ndarray:
    """Obří vlna: nabíhající hukot, náraz a dlouhé šumění."""
    n = int(5.0 * SR)
    t = _t(5.0)
    body = _noise(5.0, 24, color=1.1, lo=60, hi=7000)
    env = np.clip(t / 0.9, 0, 1) ** 2 * np.exp(-np.maximum(t - 0.9, 0) / 1.6)
    k = np.clip(t - 0.85, 0, None)
    hit = _noise(5.0, 25, color=0.4, lo=800, hi=10000) * np.clip(k / 0.03, 0, 1) * np.exp(-k / 0.35) * (t > 0.85)
    return _fade(_norm(_norm(body * env) + 0.5 * _norm(hit)))


def splash() -> np.ndarray:
    n = int(0.7 * SR)
    x = _noise(0.7, 26, color=0.6, lo=300, hi=8000) * _env(n, 0.006, 0.12)
    drops = _sprinkle(0.7, 27, 40, _drop, loop=False) * _env(n, 0.05, 0.25)
    return _fade(_norm(_norm(x) + 0.5 * _norm(drops)))


def plop() -> np.ndarray:
    """Míček zapadl do jamky: dutý „plop“."""
    t = _t(0.35)
    f = 260 + 500 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.07)
    return _fade(_norm(x.astype(np.float32)))


def tick() -> np.ndarray:
    """Odpal / ťuknutí do míčku."""
    t = _t(0.15)
    x = np.sin(2 * np.pi * 1300 * t) * np.exp(-t / 0.012) + 0.5 * np.sin(2 * np.pi * 2900 * t) * np.exp(-t / 0.006)
    return _fade(_norm(x.astype(np.float32)), 0.001, 0.01)


def coin() -> np.ndarray:
    """Nalezený poklad: dva jasné tóny nahoru (mince)."""
    a = _bell(1318.5, 0.9, 0.35, 0.6)
    b = _bell(1975.5, 1.2, 0.45, 0.6)
    return _fade(_norm(_mix(1.4, [(0.0, a * 0.7), (0.09, b)])))


def chime() -> np.ndarray:
    """Malý úspěch: rozložený durový akord zvonků."""
    notes = (784.0, 987.8, 1174.7, 1568.0)
    return _fade(_norm(_mix(2.2, [(i * 0.085, _bell(f, 1.8, 0.6, 0.8)) for i, f in enumerate(notes)])))


def fanfare() -> np.ndarray:
    """Vítězství / rekord: žesťová fanfára (G–C–E–G, pak akord) se zvonky."""
    def brass(f, d):
        x = _tone(f, d, (1.0, 0.7, 0.5, 0.35, 0.2, 0.1), vibrato=0.04)
        n = len(x)
        env = np.minimum(np.arange(n) / (0.03 * SR), 1) * np.exp(-np.arange(n) / (SR * d * 1.5))
        return _fade((x * env).astype(np.float32), 0.0, min(0.08, d * 0.4))

    seq = [(0.0, 392.0, 0.16), (0.16, 523.25, 0.16), (0.32, 659.25, 0.16), (0.48, 783.99, 0.9)]
    parts = [(s, brass(f, d)) for s, f, d in seq]
    parts += [(0.48, brass(523.25, 0.9) * 0.6), (0.48, brass(659.25, 0.9) * 0.5)]
    parts += [(0.48, _bell(1568.0, 1.6, 0.7) * 0.25), (0.6, _bell(2093.0, 1.4, 0.6) * 0.2)]
    return _fade(_norm(_mix(2.0, parts)))


def fail() -> np.ndarray:
    """Neúspěch: dva klesající měkké tóny (ne posměšně, jen „škoda“)."""
    def soft(f, d):
        x = _tone(f, d, (1.0, 0.3, 0.1))
        n = len(x)
        return _fade((x * np.minimum(np.arange(n) / (0.02 * SR), 1)
                      * np.exp(-np.arange(n) / (SR * 0.35))).astype(np.float32), 0.0, 0.08)
    return _fade(_norm(_mix(1.3, [(0.0, soft(392.0, 0.6)), (0.32, soft(311.1, 0.9))])))


def beep() -> np.ndarray:
    """Odpočet: krátký čistý tón."""
    t = _t(0.18)
    x = np.sin(2 * np.pi * 880 * t) * np.minimum(t / 0.004, 1) * np.exp(-t / 0.08)
    return _fade(_norm(x.astype(np.float32), 0.7), 0.001, 0.02)


def go() -> np.ndarray:
    """Start / konec odpočtu: delší vyšší tón."""
    t = _t(0.6)
    x = (np.sin(2 * np.pi * 1318.5 * t) + 0.3 * np.sin(2 * np.pi * 2637 * t)) * np.minimum(t / 0.004, 1)
    return _fade(_norm((x * np.exp(-t / 0.25)).astype(np.float32), 0.8))


def alarm() -> np.ndarray:
    """Varování tsunami: siréna (pomalu stoupá a klesá), 3 s."""
    t = _t(3.0)
    f = 520 + 260 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 1.5))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    return _fade(_norm(x.astype(np.float32), 0.75), 0.05, 0.2)


def whoosh() -> np.ndarray:
    """Úvodní slajd: měkký přelet (šum, jehož pásmo stoupne a klesne)."""
    n = int(1.1 * SR)
    t = _t(1.1)
    x = _rng(30).standard_normal(n).astype(np.float32)
    # postupně posouvané pásmo: rozdělit na kousky a filtrovat každý jinak
    out = np.zeros(n, np.float32)
    seg = 2048
    win = np.hanning(seg * 2).astype(np.float32)
    for i in range(0, n - seg, seg):
        k = i / n
        fc = 300 + 2600 * np.sin(np.pi * k) ** 1.5
        chunk = x[i:i + seg * 2]
        if len(chunk) < seg * 2:
            break
        out[i:i + seg * 2] += _shape(chunk * win, _band(fc * 0.5, fc * 2.0))
    env = np.sin(np.pi * np.clip(t / 1.1, 0, 1)) ** 2
    return _fade(_norm(out * env, 0.7))


def pop() -> np.ndarray:
    """Výbuch ohňostroje: suchá rána a praskající jiskry."""
    n = int(1.6 * SR)
    bang = _noise(1.6, 31, color=1.2, lo=80, hi=5000) * _env(n, 0.002, 0.06)
    sparks = _shape(_sprinkle(1.6, 32, 90, _crack, loop=False), _band(2000, 10000)) * _env(n, 0.15, 0.5)
    return _fade(_norm(_norm(bang) + 0.45 * _norm(sparks)))


def hiss() -> np.ndarray:
    """Pára (chladnoucí láva, voda na žhavém kameni)."""
    n = int(1.8 * SR)
    x = _noise(1.8, 33, color=0.3, lo=2500, hi=12000) * _env(n, 0.08, 0.6)
    return _fade(_norm(x, 0.7))


def gong() -> np.ndarray:
    """Konec kola."""
    return _fade(_norm(_mix(3.5, [(0.0, _bell(196.0, 3.5, 1.4, 0.7)), (0.0, 0.5 * _bell(293.7, 3.0, 1.0, 0.5))])))


def whistle() -> np.ndarray:
    """Start kola: dvojí hvizd."""
    def w(d):
        t = _t(d)
        f = 2300 + 120 * np.sin(2 * np.pi * 28 * t)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(t / 0.01, 1)
        x += 0.15 * _noise(d, 34, lo=2000, hi=6000)[:len(t)]
        return (x * np.minimum((d - t) / 0.02, 1)).astype(np.float32)
    return _fade(_norm(_mix(0.75, [(0.0, w(0.2)), (0.28, w(0.42))]), 0.7))


def ding() -> np.ndarray:
    """Změna vedení (nový král): jeden jasný zvonek."""
    return _fade(_norm(_bell(1046.5, 1.5, 0.6, 0.9)))


def bubble() -> np.ndarray:
    """Potopený ostrov: probublávání."""
    return _fade(_norm(_sprinkle(1.4, 35, 25, lambda r: _bell(r.uniform(180, 420), 0.2, 0.05, 0.1), loop=False)
                       * _env(int(1.4 * SR), 0.05, 0.6)))


def burn() -> np.ndarray:
    """Dům chytil: zafučení plamene."""
    n = int(1.5 * SR)
    x = _noise(1.5, 36, color=1.3, lo=100, hi=3000) * _env(n, 0.15, 0.5)
    c = _shape(_sprinkle(1.5, 37, 40, _crack, loop=False), _band(1000, 8000)) * _env(n, 0.1, 0.6)
    return _fade(_norm(_norm(x) + 0.5 * _norm(c)))


def sparkle() -> np.ndarray:
    """Třpyt (odhalení, kouzlo): rychlé vysoké zvonky."""
    rng = _rng(38)
    parts = [(i * 0.05, _bell(float(rng.choice([2093.0, 2349.3, 2637.0, 3136.0, 3520.0])), 0.8, 0.25, 0.4)
              * rng.uniform(0.4, 1.0)) for i in range(9)]
    return _fade(_norm(_mix(1.3, parts)))


def tada() -> np.ndarray:
    """Velký úspěch v klidném módu (krajina, předloha): akord zvonků + třpyt."""
    chord = sum(_bell(f, 2.6, 0.9, 0.7) for f in (523.25, 659.25, 783.99, 1046.5))
    return _fade(_norm(_mix(2.8, [(0.0, chord), (0.1, sparkle() * 0.4)])))


BANK: dict[str, tuple] = {
    # smyčky
    "rain": (rain, True), "wind": (wind, True), "stream": (stream, True), "sea": (sea, True),
    "forest": (forest, True), "lava": (lava, True), "rumble": (rumble, True), "roar": (roar, True),
    "roll": (roll, True), "fire": (fire, True), "tension": (tension, True),
    # jednorázové
    "thunder": (thunder, False), "boom": (boom, False), "thud": (thud, False), "crash": (crash, False),
    "splash": (splash, False), "plop": (plop, False), "tick": (tick, False), "coin": (coin, False),
    "chime": (chime, False), "fanfare": (fanfare, False), "fail": (fail, False), "beep": (beep, False),
    "go": (go, False), "alarm": (alarm, False), "whoosh": (whoosh, False), "pop": (pop, False),
    "hiss": (hiss, False), "gong": (gong, False), "whistle": (whistle, False), "ding": (ding, False),
    "bubble": (bubble, False), "burn": (burn, False), "sparkle": (sparkle, False), "tada": (tada, False),
}


def build(name: str) -> np.ndarray:
    fn, _ = BANK[name]
    x = np.asarray(fn(), np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"zvuk {name}: neplatné hodnoty")
    return x
