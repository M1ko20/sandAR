"""Viskózní láva – buněčný automat s teplotou, sdílená simulace pro módy.

Láva teče jako velmi hustá kapalina s mezí kluzu (Binghamova tekutina):
mezi sousedními buňkami přeteče jen tehdy, když rozdíl hladin (terén + láva)
překročí mez kluzu. Pohyblivost i mez kluzu závisí na teplotě – horká láva
teče, chladnoucí houstne a pod teplotou tuhnutí se zastaví (černá krusta).

Stav:
* ``thickness`` – tloušťka lávy [m] (zůstává i ztuhlá, leží na terénu),
* ``temp``      – teplota 0..1 (1 = čerstvá láva z kráteru, < ``solid`` = ztuhlá).

Teplo se přenáší s tokem lávy (směšování), chladnutí je rychlejší u tenké vrstvy
(okraje a čela proudů tuhnou dřív – jako u skutečné lávy).
"""

from __future__ import annotations

import numpy as np


def downsample(a: np.ndarray, f: int) -> np.ndarray:
    """Průměr bloků f×f (okraje, které se nevejdou celé, se oříznou)."""
    if f <= 1:
        return np.asarray(a, np.float32)
    gh, gw = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:gh, :gw].reshape(gh // f, f, gw // f, f).mean(axis=(1, 3)).astype(np.float32)


def blur3(a: np.ndarray) -> np.ndarray:
    """Rychlé rozmazání 3×3 (na hranách opakuje krajní hodnoty)."""
    p = np.pad(a, 1, mode="edge")
    s = p[:-2] + 2 * p[1:-1] + p[2:]
    return ((s[:, :-2] + 2 * s[:, 1:-1] + s[:, 2:]) / 16).astype(np.float32)


class LavaSim:
    def __init__(self, shape: tuple[int, int], cell: float, *, mobility: float = 25.0, yield_hot: float = 0.0004,
                 yield_cold: float = 0.006, full_flow: float = 0.006, solid: float = 0.18, cooling: float = 0.04,
                 thin_cooling: float = 0.00025) -> None:
        self.cell = cell
        self.thickness = np.zeros(shape, np.float32)
        self.temp = np.zeros(shape, np.float32)
        self.mobility = mobility  # [1/s] jak rychle se vyrovnávají hladiny horké lávy
        # mez kluzu τ/(ρg) [m]: láva teče, jen když je tloušťka > τ / sklon hladiny
        self.yield_hot = yield_hot
        self.yield_cold = yield_cold  # těsně nad teplotou tuhnutí
        self.full_flow = full_flow  # [m] nad kritickou tloušťkou o tolik už teče naplno
        self.solid = solid  # teplota tuhnutí (0..1)
        self.cooling = cooling  # [1/s] základní chladnutí
        self.thin_cooling = thin_cooling  # [m/s] navíc u tenké vrstvy (∝ 1/tloušťka)

    @property
    def shape(self) -> tuple[int, int]:
        return self.thickness.shape  # type: ignore[return-value]

    @property
    def volume(self) -> float:
        """Objem lávy [m³]."""
        return float(self.thickness.sum()) * self.cell * self.cell

    @property
    def molten(self) -> np.ndarray:
        """Maska buněk s tekutou lávou."""
        return (self.thickness > 1e-4) & (self.temp > self.solid)

    def clear(self) -> None:
        self.thickness[:] = 0
        self.temp[:] = 0

    def erupt(self, iy: float, ix: float, radius: float, rate: float, dt: float, temp: float = 1.0) -> None:
        """Výron lávy z kráteru: rate [m³/s] rozložený do kruhu o poloměru radius [buňky]."""
        gh, gw = self.shape
        y, x = np.ogrid[:gh, :gw]
        w = np.clip(1.0 - np.sqrt((y - iy) ** 2 + (x - ix) ** 2) / max(radius, 0.5), 0, None)
        s = float(w.sum())
        if s <= 0:
            return
        add = (rate * dt / (self.cell * self.cell)) * (w / s)
        h = self.thickness
        new = h + add
        mask = add > 0
        self.temp[mask] = (self.temp[mask] * h[mask] + temp * add[mask]) / new[mask]
        self.thickness[:] = new

    def advance(self, terrain: np.ndarray, frame_dt: float) -> None:
        """Posune simulaci o frame_dt (stabilní podkroky)."""
        n = int(min(10, max(1, np.ceil(frame_dt * self.mobility / 0.15))))
        rows = np.flatnonzero(self.thickness.any(axis=1))
        if len(rows) == 0:
            return
        cols = np.flatnonzero(self.thickness.any(axis=0))
        # počítá se jen v okolí lávy (za podkrok se posune nejvýš o buňku)
        m = n + 2
        gh, gw = self.shape
        win = (slice(max(0, rows[0] - m), min(gh, rows[-1] + m + 1)),
               slice(max(0, cols[0] - m), min(gw, cols[-1] + m + 1)))
        h, T, ter = self.thickness[win].copy(), self.temp[win].copy(), terrain[win]
        for _ in range(n):
            h, T = self._step(h, T, ter, frame_dt / n)
        self.thickness[win] = h
        self.temp[win] = T

    def step(self, terrain: np.ndarray, dt: float) -> None:
        """Jeden krok simulace na celé mřížce."""
        self.thickness[:], self.temp[:] = self._step(self.thickness, self.temp, terrain, dt)

    def _step(self, h: np.ndarray, T: np.ndarray, terrain: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        hot = np.clip((T - self.solid) / (1.0 - self.solid), 0.0, 1.0)
        # pohyblivost roste s teplotou strmě (viskozita chladnoucí lávy prudce stoupá)
        mob = self.mobility * dt * hot**1.5
        tau = (self.yield_cold + (self.yield_hot - self.yield_cold) * hot) * self.cell
        H = terrain + h
        inv_full = 1.0 / self.full_flow

        fluxes = []
        out = np.zeros_like(h)
        for (dy, dx), w in _DIRS:
            src, dst = _slices(dy, dx)
            dh = np.maximum(H[src] - H[dst], 0.0)
            dist = 1.4142 if dy and dx else 1.0
            # Binghamova tekutina: kritická tloušťka h_cr = τ / sklon hladiny; tenčí vrstva stojí
            h_cr = tau[src] * dist / np.maximum(dh, 1e-9)
            k = np.clip((h[src] - h_cr) * inv_full, 0.0, 1.0)
            f = np.zeros_like(h)
            f[src] = (w * mob[src]) * dh * k
            out += f
            fluxes.append((f, src, dst))
        # odteče nejvýš polovina toho, co v buňce je (hladiny se nepřehoupnou)
        scale = np.minimum(1.0, 0.5 * h / np.maximum(out, 1e-12))
        out *= scale
        inflow = np.zeros_like(h)
        ein = np.zeros_like(h)
        for f, src, dst in fluxes:
            fs = f[src] * scale[src]
            inflow[dst] += fs
            ein[dst] += fs * T[src]
        new_h = np.maximum(h - out + inflow, 0.0)
        new_e = np.maximum(h * T - out * T + ein, 0.0)  # „teplo“ přenášené s lávou
        T_new = np.where(new_h > 1e-7, new_e / np.maximum(new_h, 1e-12), 0.0)
        # chladnutí: tenká vrstva chladne rychleji
        T_new -= dt * (self.cooling + self.thin_cooling / (new_h + 0.002)) * T_new
        np.clip(T_new, 0.0, 1.0, out=T_new)
        T_new[new_h < 1e-6] = 0.0
        return new_h.astype(np.float32), T_new.astype(np.float32)


def _slices(dy: int, dx: int) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Řezy (zdroj, cíl) pro tok o (dy, dx) buněk."""
    def ax(d: int) -> tuple[slice, slice]:
        if d > 0:
            return slice(None, -d), slice(d, None)
        if d < 0:
            return slice(-d, None), slice(None, d)
        return slice(None), slice(None)
    sy, ty = ax(dy)
    sx, tx = ax(dx)
    return (sy, sx), (ty, tx)


# 8 směrů; diagonály mají menší váhu, aby se láva šířila zhruba kruhově, ne do kříže
_DIRS = [((0, 1), 1.0), ((0, -1), 1.0), ((1, 0), 1.0), ((-1, 0), 1.0),
         ((1, 1), 0.5), ((1, -1), 0.5), ((-1, 1), 0.5), ((-1, -1), 0.5)]
