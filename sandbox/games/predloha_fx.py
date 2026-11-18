"""Vizuální efekty předloh (Postav podle předlohy, Modeluj podle předlohy).

* barevná nápověda: červená = přidej, modrá = uber, zelená = sedí – s plynulým přechodem
  podle velikosti chyby a „dýcháním“ tam, kde je chyba největší,
* velké symboly + / − v místech s největší chybou (čitelné i bez rozlišení barev),
* holografický obrys hlavní části (záře, běžící světlo, skenovací pás),
* oslava při úspěchu (lesk, rázové vlny, konfety).

Logiku hodnocení nemění – jen kreslí.
"""

from __future__ import annotations

import cv2
import numpy as np

# Funkce s prefixem ``pred_``; vkládá se do shaderu módu před ``main``.
HINT_GLSL = """
const vec3 PRED_HOLO = vec3(0.30, 0.92, 1.0);

// barevná nápověda: d = písek − předloha [m], tol = tolerance [m]
vec3 pred_hint(vec3 base, float d, float tol) {
    float a = abs(d);
    float k = clamp((a - 0.5 * tol) / (2.0 * tol), 0.0, 1.0);
    float big = clamp((a - 1.5 * tol) / (2.5 * tol), 0.0, 1.0);
    vec3 add = mix(vec3(1.0, 0.28, 0.16), vec3(1.0, 0.05, 0.03), k);   // přidej: červená
    vec3 rem = mix(vec3(0.30, 0.58, 1.0), vec3(0.04, 0.22, 1.0), k);   // uber: modrá
    vec3 tint = d < 0.0 ? add : rem;
    // dýchání v místech s největší chybou (pomalé, ne blikání)
    float breath = 0.5 + 0.5 * sin(u_time * 3.6);
    tint *= 1.0 + 0.30 * big * breath;
    vec3 c = mix(base, tint, k * 0.92);
    float good = 1.0 - smoothstep(0.5 * tol, tol, a);
    vec3 green = vec3(0.12, 0.92, 0.36) * (0.92 + 0.08 * sin(u_time * 2.0));
    return mix(c, green, good * 0.62);
}

// střed buňky mřížky symbolů (uv)
vec2 pred_cell(vec2 uv, float cell) {
    vec2 pm = uv * u_box;
    return (floor(pm / cell) + 0.5) * cell / u_box;
}

float pred_box(vec2 p, vec2 b) {
    vec2 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}

// symbol + (přidej) / − (uber) uprostřed buňky, kde je chyba velká; dc = chyba ve středu buňky
vec3 pred_symbol(vec3 c, vec2 uv, vec2 cu, float dc, float mc, float tol) {
    float a = abs(dc);
    float show = smoothstep(1.5 * tol, 1.8 * tol, a) * smoothstep(0.6, 0.8, mc);
    if (show <= 0.0) return c;
    float big = clamp((a - 1.5 * tol) / (3.0 * tol), 0.0, 1.0);
    vec2 q = (uv - cu) * u_box;
    float ph = dot(cu, vec2(37.0, 21.0));
    float s = 1.0 + 0.12 * sin(u_time * 3.6 + ph);
    float L = (0.016 + 0.007 * big) * s, w = 0.0042 * s;
    float sd = pred_box(q, vec2(L, w));
    if (dc < 0.0) sd = min(sd, pred_box(q, vec2(w, L)));
    float px = 0.0014;  // ~1 px
    float fill = 1.0 - smoothstep(-px, px, sd);
    float rim = 1.0 - smoothstep(-px, px, sd - 0.0035);
    c = mix(c, vec3(0.0), rim * show * 0.75);
    return mix(c, vec3(1.0), fill * show);
}

// holografický obrys hlavní části: m = měkká maska, halo = široce rozmazaná maska
vec3 pred_holo(vec3 c, vec2 uv, float m, float halo) {
    vec2 pm = uv * u_box;
    float outside = 1.0 - smoothstep(0.35, 0.65, m);
    // záře kolem obrysu do tmavého okolí
    c += PRED_HOLO * halo * outside * (0.40 + 0.08 * sin(u_time * 2.0));
    // skenovací pás, který pomalu přejíždí přes bednu
    float sx = (fract(u_time * 0.09) * 1.5 - 0.25) * u_box.x;
    float scan = exp(-pow((pm.x - sx) / 0.04, 2.0));
    c += PRED_HOLO * scan * (0.30 * outside + 0.10 * (1.0 - outside));
    // obrys: světlé jádro s běžícími světly
    float edge = 1.0 - smoothstep(0.0, 0.17, abs(m - 0.5));
    float run = 0.5 + 0.5 * sin((pm.x + pm.y) * 22.0 - u_time * 3.0);
    vec3 ec = mix(PRED_HOLO, vec3(1.0), 0.25 + 0.6 * run);
    return mix(c, ec, edge * (0.80 + 0.20 * run));
}

// oslava úspěchu: lesk přes hlavní část a rázové vlny od jejího středu
vec3 pred_celebrate(vec3 c, vec2 uv, float m, float t, vec2 center) {
    if (t < 0.0) return c;
    float fade = smoothstep(0.0, 0.4, t);
    vec2 pm = uv * u_box;
    float s = pm.x + pm.y * 0.6;
    float band = fract(s / 1.4 - t * 0.55);
    float shine = exp(-pow((band - 0.5) * 1.4 / 0.05, 2.0));
    c = mix(c, c * vec3(0.9, 1.1, 0.9) + vec3(0.03, 0.08, 0.02), 0.35 * m * fade);
    c += vec3(1.0, 0.96, 0.75) * shine * 0.40 * m * fade;
    float r = length((uv - center) * u_box);
    float w = fract((r - t * 0.40) / 0.45);
    float ring = exp(-pow((w - 0.5) * 0.45 / 0.022, 2.0)) * exp(-r * 1.5);
    c += vec3(0.6, 1.0, 0.5) * ring * 0.9 * fade;
    return c;
}
"""

# barvy konfet při úspěchu
CONFETTI = ((0.25, 1.0, 0.4, 1.0), (1.0, 0.85, 0.2, 1.0), (1.0, 1.0, 1.0, 1.0), (0.3, 0.9, 1.0, 1.0))


def halo_mask(mask: np.ndarray, cell: float, radius: float = 0.035) -> np.ndarray:
    """Široce rozmazaná maska (záře kolem holografického obrysu) – 0..1, jen vně masky."""
    sigma = max(1.0, radius / max(cell, 1e-4))
    blur = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return np.clip(blur * 2.0, 0.0, 1.0).astype(np.float32)


def mask_center(mask: np.ndarray) -> tuple[float, float]:
    """Těžiště masky v uv (0..1); prázdná maska = střed."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.5, 0.5
    return float((xs.mean() + 0.5) / mask.shape[1]), float((ys.mean() + 0.5) / mask.shape[0])


class Smooth:
    """Plynulé přiblížení hodnoty (posun hladiny, zobrazené procento) – žádné skoky po 0,25 s."""

    def __init__(self, rate: float = 6.0) -> None:
        self.rate = rate
        self.value: float | None = None
        self._t: float | None = None

    def __call__(self, target: float, now: float) -> float:
        if self.value is None or self._t is None:
            self.value, self._t = target, now
            return target
        dt = max(0.0, min(0.5, now - self._t))
        self._t = now
        self.value += (target - self.value) * (1.0 - np.exp(-self.rate * dt))
        return self.value
