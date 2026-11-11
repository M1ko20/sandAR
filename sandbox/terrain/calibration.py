"""Kalibrace Kinectu vůči pískovišti.

Z uhlazeného písku se proloží základní rovina (výška 0). Uživatel klikne do
hloubkového obrazu na 4 rohy bedny; jejich paprsky protneme s rovinou a
získáme skutečné rozměry pískoviště v metrech. Souřadnicový systém bedny:

* počátek = roh 1, osa x směrem k rohu 2, osa y směrem k rohu 4,
* výška h = kolmá vzdálenost od základní roviny (kladně ke Kinectu).

Rohy nemusí tvořit dokonalý obdélník (nepřesné kliknutí, perspektiva) – proto
mapujeme rovinné souřadnice na obdélník homografií. Stejné rohy 1–4 se pak
používají i při kalibraci projektoru, takže obě strany sedí na sebe.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..sensor.base import Intrinsics


class CalibrationError(ValueError):
    pass


def fit_plane(points: np.ndarray, iterations: int = 4) -> tuple[np.ndarray, float]:
    """Robustní proložení roviny body (N,3). Vrací (normála, offset), h = n·p + offset.

    Normála míří ke kameře (kamera je v počátku → offset > 0).
    """
    if len(points) < 100:
        raise CalibrationError("Příliš málo platných bodů pro proložení roviny")
    mask = np.ones(len(points), bool)
    normal, offset = np.array([0.0, 0.0, -1.0]), 0.0
    for _ in range(iterations):
        p = points[mask]
        centroid = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - centroid, full_matrices=False)
        normal = vt[2]
        offset = -float(normal @ centroid)
        if offset < 0:
            normal, offset = -normal, -offset
        resid = points @ normal + offset
        med = np.median(resid[mask])
        mad = np.median(np.abs(resid[mask] - med)) + 1e-6
        new_mask = np.abs(resid - med) < max(0.004, 4 * 1.4826 * mad)
        if new_mask.sum() < 100 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return normal, offset


def _quad_mask(shape: tuple[int, int], quad: np.ndarray, shrink: float) -> np.ndarray:
    center = quad.mean(axis=0)
    q = center + (quad - center) * (1.0 - shrink)
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(q).astype(np.int32), 1)
    return mask.astype(bool)


def detect_box_corners(intrinsics: Intrinsics, depth: np.ndarray, sand_tolerance: float = 0.02) -> np.ndarray | None:
    """Rohy bedny: nejdřív podle stěn (funguje i s neuhlazeným pískem), pak podle roviny písku."""
    corners = detect_box_corners_walls(intrinsics, depth)
    if corners is None:
        corners = detect_box_corners_plane(intrinsics, depth, sand_tolerance)
    return corners


def detect_box_corners_plane(intrinsics: Intrinsics, depth: np.ndarray, sand_tolerance: float = 0.02) -> np.ndarray | None:
    """Najde vnitřní rohy bedny v hloubkovém obraze (uhlazený písek, Kinect zhruba nad středem).

    Proloží rovinu středem obrazu, vezme souvislou oblast „téměř v rovině“ kolem středu
    (písek ohraničený vyššími stěnami) a její obrys zjednoduší na čtyřúhelník.
    Vrací rohy (4,2) v pořadí vlevo nahoře, vpravo nahoře, vpravo dole, vlevo dole
    (v obraze Kinectu), nebo None.
    """
    H, W = depth.shape
    center = np.zeros(depth.shape, bool)
    center[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = True
    center &= depth > 0
    if center.sum() < 1000:
        return None
    rays = intrinsics.rays()
    fit_mask = center
    region = None
    # Rovina se nejdřív proloží středem obrazu, pak opakovaně celou nalezenou oblastí písku –
    # kopec uprostřed neuhlazeného písku jinak rovinu nakloní a oblast se useknou.
    for _ in range(4):
        normal, offset = fit_plane((rays[fit_mask] * depth[fit_mask][:, None]).astype(float))
        h = (rays @ normal) * depth + offset
        sand = ((np.abs(h) < sand_tolerance) & (depth > 0)).astype(np.uint8)
        sand = cv2.morphologyEx(sand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels = cv2.connectedComponents(sand)
        # Největší rovinná oblast zasahující do středu obrazu (ne nutně přímo středový pixel).
        counts = np.bincount(labels[center], minlength=n)
        counts[0] = 0
        label = int(np.argmax(counts))
        if label == 0:
            return None
        new_region = labels == label
        if region is not None and (new_region != region).mean() < 0.002:
            region = new_region
            break
        region = new_region
        fit_mask = region & (depth > 0)
    region = region.astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(float)
            break
    if quad is None or cv2.contourArea(quad.astype(np.float32)) < 0.1 * H * W:
        return None
    # pořadí: vlevo nahoře, vpravo nahoře, vpravo dole, vlevo dole
    s, d = quad.sum(axis=1), quad[:, 0] - quad[:, 1]
    return np.array([quad[np.argmin(s)], quad[np.argmax(d)], quad[np.argmax(s)], quad[np.argmin(d)]])


def _walk_to_wall(line: np.ndarray, start: int, direction: int, step_up: float, drop: float,
                  look: int) -> int | None:
    """Od `start` jde po profilu výšky, dokud nenarazí na stěnu (prudký schod nahoru)
    nebo sráz dolů (okraj bedny). Vrací poslední index uvnitř, None = došel na kraj obrazu."""
    n = len(line)
    base = np.nanmedian(line[max(0, start - 20):start + 20])
    i = start
    while 0 <= i + direction * look < n:
        j = i + direction * look
        if not np.isfinite(line[j]) or line[j] - line[i] > step_up or line[j] < base - drop:
            return i
        i += direction
    return None


def _fit_edge(pts: np.ndarray, vertical: bool, tol: float = 2.0) -> tuple[float, float] | None:
    """Robustní přímka hrany stěny (RANSAC): x = a·y + b pro svislou, y = a·x + b pro vodorovnou.
    Body, kde skenování zastavil svah kopce místo stěny, vypadnou jako odlehlé."""
    if len(pts) < 20:
        return None
    x, y = (pts[:, 1], pts[:, 0]) if vertical else (pts[:, 0], pts[:, 1])
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        i, j = rng.choice(len(x), 2, replace=False)
        if x[i] == x[j]:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i])
        if abs(a) > 0.5:  # hrana bedny je v obraze zhruba svislá / vodorovná
            continue
        inl = np.abs(y - (a * x + (y[i] - a * x[i]))) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < max(20, 0.3 * len(x)):
        return None
    a, b = np.polyfit(x[best], y[best], 1)
    return float(a), float(b)


def detect_box_corners_walls(intrinsics: Intrinsics, depth: np.ndarray, step_up: float = 0.04,
                             drop: float = 0.12) -> np.ndarray | None:
    """Najde rohy bedny podle vnitřních hran stěn – písek nemusí být uhlazený.

    Z prostředka obrazu jde po řádcích a sloupcích ven, dokud nenarazí na stěnu (výška
    skočí o víc než `step_up` na pár pixelech; kopce v písku jsou pozvolnější) nebo na
    sráz. Body hran proloží přímkami (RANSAC) a rohy jsou jejich průsečíky – roh smí
    ležet i kousek mimo záběr Kinectu. Pořadí rohů jako u `detect_box_corners_plane`.
    """
    H, W = depth.shape
    valid = depth > 0
    center = np.zeros(depth.shape, bool)
    center[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = True
    center &= valid
    if center.sum() < 1000:
        return None
    rays = intrinsics.rays()
    pts = rays * depth[..., None]
    normal, offset = fit_plane(pts[center].astype(float))
    h = cv2.GaussianBlur((pts @ normal + offset).astype(np.float32), (3, 3), 0)
    h = np.where(valid, h, np.nan)
    look = 3
    cu, cv_ = W // 2, H // 2
    sides: dict[str, list[tuple[float, float]]] = {"left": [], "right": [], "top": [], "bottom": []}
    for v in range(H // 6, H - H // 6, 2):
        for side, d in (("left", -1), ("right", 1)):
            u = _walk_to_wall(h[v], cu, d, step_up, drop, look)
            if u is not None:
                sides[side].append((u, v))
    for u in range(W // 6, W - W // 6, 2):
        for side, d in (("top", -1), ("bottom", 1)):
            v = _walk_to_wall(h[:, u], cv_, d, step_up, drop, look)
            if v is not None:
                sides[side].append((u, v))
    lines = {}
    for side in sides:
        lines[side] = _fit_edge(np.array(sides[side], float).reshape(-1, 2), side in ("left", "right"))
        if lines[side] is None:
            return None

    def cross(vert, horiz):  # x = a1·y + b1  ∩  y = a2·x + b2
        (a1, b1), (a2, b2) = vert, horiz
        x = (a1 * b2 + b1) / (1 - a1 * a2)
        return np.array([x, a2 * x + b2])

    quad = np.array([cross(lines["left"], lines["top"]), cross(lines["right"], lines["top"]),
                     cross(lines["right"], lines["bottom"]), cross(lines["left"], lines["bottom"])])
    # rozumnost: velká plocha, rohy nejvýš kousek za okrajem záběru
    if cv2.contourArea(quad.astype(np.float32)) < 0.25 * H * W:
        return None
    if np.any(quad < -0.1 * np.array([W, H])) or np.any(quad > 1.1 * np.array([W, H])):
        return None
    return quad


def unmirror_corners(intrinsics: Intrinsics, corners_px: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Pořadí rohů tak, aby plátno nebylo zrcadlově (obraz z libfreenect2 je zrcadlený).

    Osy bedny x (roh 1→2), y (roh 1→4) a normála ke Kinectu musí tvořit pravotočivou
    soustavu – jinak se prohodí rohy 1↔2 a 4↔3. Na kterou stranu bedny je plátno
    otočené, se tím nemění (to jde případně otočit přečíslováním o 180°)."""
    c = np.asarray(corners_px, float).reshape(4, 2)
    rays = np.column_stack([(c[:, 0] - intrinsics.cx) / intrinsics.fx, (c[:, 1] - intrinsics.cy) / intrinsics.fy,
                            np.ones(4)])
    ex, ey = rays[1] - rays[0], rays[3] - rays[0]
    n = np.asarray(normal, float)
    if n[2] > 0:  # normála má mířit ke Kinectu (z < 0 v souřadnicích kamery)
        n = -n
    return c[[1, 0, 3, 2]] if float(np.cross(ex, ey) @ n) < 0 else c


@dataclass
class KinectCalibration:
    intrinsics: Intrinsics
    normal: np.ndarray  # (3,) v souřadnicích kamery
    offset: float  # výška kamery nad základní rovinou [m]
    corners_px: np.ndarray  # (4,2) kliknuté rohy v hloubkovém obraze
    origin: np.ndarray  # (3,) bod roviny pod rohem 1
    ex: np.ndarray  # (3,) jednotkový vektor osy x v rovině
    ey: np.ndarray  # (3,) jednotkový vektor osy y v rovině
    size: tuple[float, float]  # (šířka, hloubka) [m]
    plane_to_box: np.ndarray  # 3x3 homografie (a,b) -> (x,y)

    # --- vytvoření -------------------------------------------------------------
    @classmethod
    def from_plane(cls, intrinsics: Intrinsics, normal: np.ndarray, offset: float,
                   corners_px: np.ndarray) -> "KinectCalibration":
        normal = np.asarray(normal, float)
        normal = normal / np.linalg.norm(normal)
        corners_px = np.asarray(corners_px, float).reshape(4, 2)
        rays = np.column_stack([(corners_px[:, 0] - intrinsics.cx) / intrinsics.fx,
                                (corners_px[:, 1] - intrinsics.cy) / intrinsics.fy,
                                np.ones(4)])
        denom = rays @ normal
        if np.any(np.abs(denom) < 1e-6):
            raise CalibrationError("Roh leží mimo základní rovinu")
        t = -offset / denom  # průsečík paprsku s rovinou h = 0
        if np.any(t <= 0):
            raise CalibrationError("Rovina je za kamerou – špatná kalibrace")
        P = rays * t[:, None]

        origin = P[0]
        ex = P[1] - P[0]
        ex -= normal * (ex @ normal)
        ex /= np.linalg.norm(ex)
        ey = P[3] - P[0]
        ey -= normal * (ey @ normal) + ex * (ey @ ex)
        if np.linalg.norm(ey) < 1e-6:
            raise CalibrationError("Rohy leží na přímce")
        ey /= np.linalg.norm(ey)

        ab = np.column_stack([(P - origin) @ ex, (P - origin) @ ey])
        width = 0.5 * (np.linalg.norm(ab[1] - ab[0]) + np.linalg.norm(ab[2] - ab[3]))
        depth = 0.5 * (np.linalg.norm(ab[3] - ab[0]) + np.linalg.norm(ab[2] - ab[1]))
        if width < 0.1 or depth < 0.1:
            raise CalibrationError(f"Pískoviště vychází podezřele malé ({width:.2f} x {depth:.2f} m)")
        rect = np.array([[0, 0], [width, 0], [width, depth], [0, depth]], np.float32)
        H = cv2.getPerspectiveTransform(ab.astype(np.float32), rect).astype(float)
        return cls(intrinsics, normal, float(offset), corners_px, origin, ex, ey, (float(width), float(depth)), H)

    @classmethod
    def from_depth(cls, intrinsics: Intrinsics, depth: np.ndarray, corners_px: np.ndarray,
                   shrink: float = 0.12) -> "KinectCalibration":
        """Kalibrace z (průměrovaného) snímku uhlazeného písku a 4 kliknutých rohů."""
        corners_px = np.asarray(corners_px, float).reshape(4, 2)
        mask = _quad_mask(depth.shape, corners_px, shrink) & (depth > 0)
        pts = intrinsics.rays()[mask] * depth[mask][:, None]
        normal, offset = fit_plane(pts.astype(float))
        return cls.from_plane(intrinsics, normal, offset, corners_px)

    # --- transformace ----------------------------------------------------------
    def camera_to_box(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3D body kamery (N,3) -> (x, y, h) v souřadnicích bedny."""
        rel = points - self.origin
        h = points @ self.normal + self.offset
        a = rel @ self.ex
        b = rel @ self.ey
        H = self.plane_to_box
        w = H[2, 0] * a + H[2, 1] * b + H[2, 2]
        x = (H[0, 0] * a + H[0, 1] * b + H[0, 2]) / w
        y = (H[1, 0] * a + H[1, 1] * b + H[1, 2]) / w
        return x, y, h

    def box_to_camera(self, x: np.ndarray, y: np.ndarray, h: np.ndarray | float = 0.0) -> np.ndarray:
        Hinv = np.linalg.inv(self.plane_to_box)
        x, y = np.asarray(x, float), np.asarray(y, float)
        w = Hinv[2, 0] * x + Hinv[2, 1] * y + Hinv[2, 2]
        a = (Hinv[0, 0] * x + Hinv[0, 1] * y + Hinv[0, 2]) / w
        b = (Hinv[1, 0] * x + Hinv[1, 1] * y + Hinv[1, 2]) / w
        h = np.asarray(h, float)
        return (self.origin + a[..., None] * self.ex + b[..., None] * self.ey
                + h[..., None] * self.normal)

    def box_to_pixel(self, x, y, h=0.0) -> np.ndarray:
        return self.intrinsics.project(self.box_to_camera(x, y, h))

    def camera_position_box(self) -> tuple[float, float, float]:
        """Poloha Kinectu v souřadnicích bedny (x, y, výška)."""
        foot = -self.offset * self.normal  # průmět kamery (počátku) do roviny
        x, y, _ = self.camera_to_box(foot[None, :])
        return float(x[0]), float(y[0]), self.offset

    # --- serializace --------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "intrinsics": self.intrinsics.to_dict(),
            "normal": self.normal.tolist(),
            "offset": self.offset,
            "corners_px": self.corners_px.tolist(),
            "size_m": list(self.size),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KinectCalibration":
        return cls.from_plane(Intrinsics.from_dict(d["intrinsics"]), np.array(d["normal"]),
                              float(d["offset"]), np.array(d["corners_px"]))
