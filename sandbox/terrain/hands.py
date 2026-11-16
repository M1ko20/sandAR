"""Detekce a sledování rukou nad pískem z hloubkového obrazu Kinectu.

Proč ne neuronová síť (MediaPipe apod.): v instalaci je tma, na rukou je promítnutý obraz
a Kinect se dívá přesně shora – na to modely trénované na barevných fotkách nefungují.
Hloubka je naopak v tmě spolehlivá a pro pískoviště má silnou geometrickou vlastnost:

    **ruka je vždycky konec paže, která do bedny vede přes její okraj.**

Hromádka písku se okraje nedotýká a nad pískem „nevisí“. Postup pro každý snímek:

1. Model pozadí po pixelech (hloubka písku, stěn, okolí). Vzdálenější změny (vyhrabaný
   důlek, odchod ruky) se přijmou rychle, bližší změny jen když vydrží a nejsou součástí
   ruky – nový kopec se tak stane pozadím, ruka nikdy.
2. Popředí = pixely aspoň ``fg_threshold`` blíž ke Kinectu než pozadí.
3. Rozříznutí podle hloubkových hran (skok > ``edge_jump`` mezi sousedy): ruka držená nad
   hromádkou se s ní nespojí a odfiltrují se i „létající pixely“ na obrysech.
4. Souvislé oblasti; ruka = oblast, která sahá za stěnu bedny (nebo k okraji obrazu)
   a má v bedně dost pixelů.
5. Dlaň = nejvzdálenější část paže od místa, kudy paže vstupuje do bedny (průměr
   posledních ``palm_depth`` cm), špička = úplný konec.
6. Sledování: každá ruka má stálé číslo, poloha se vyhlazuje filtrem One Euro (klidná
   v klidu, bez zpoždění v pohybu), krátké výpadky se přemostí.

Výstupem je seznam rukou v souřadnicích hrací plochy a maska pixelů, které terén nesmí
použít (ruka se tak nikdy nepropíše do výškové mapy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import HandSettings
from .calibration import KinectCalibration


@dataclass
class Hand:
    """Jedna sledovaná ruka (souřadnice hrací plochy v metrech)."""

    id: int
    x: float  # střed dlaně
    y: float
    tip_x: float  # konec prstů
    tip_y: float
    height: float  # [m] výška dlaně nad pískem
    vx: float = 0.0  # [m/s] rychlost dlaně
    vy: float = 0.0
    age: float = 0.0  # [s] jak dlouho ji sledujeme
    area: int = 0  # počet pixelů paže v bedně

    @property
    def pos(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def tip(self) -> tuple[float, float]:
        return self.tip_x, self.tip_y

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def touching(self) -> bool:
        """Dlaň je na písku nebo v něm (ne nad ním)."""
        return self.height < 0.04


@dataclass
class _Detection:
    x: float
    y: float
    tip_x: float
    tip_y: float
    height: float
    area: int
    primary: bool = True  # nejvzdálenější konec paže (ne vedlejší větev)


class _OneEuro:
    """Filtr One Euro (Casiez et al. 2012) pro 2D bod."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: np.ndarray | None = None
        self.dx = np.zeros(2)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, p: np.ndarray, dt: float) -> np.ndarray:
        if self.x is None:
            self.x = p.astype(float)
            return self.x
        dx = (p - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._alpha(cutoff, dt) * (p - self.x)
        return self.x


class _Track:
    def __init__(self, tid: int, det: _Detection, s: HandSettings) -> None:
        self.id = tid
        self.f_palm = _OneEuro(s.filter_min_cutoff, s.filter_beta)
        self.f_tip = _OneEuro(s.filter_min_cutoff, s.filter_beta)
        self.hits = 0
        self.missed = 0
        self.age = 0.0
        self.det = det
        self.palm = np.array([det.x, det.y])
        self.tip = np.array([det.tip_x, det.tip_y])
        self.height = det.height

    def update(self, det: _Detection, dt: float) -> None:
        self.det = det
        self.palm = self.f_palm(np.array([det.x, det.y]), dt)
        self.tip = self.f_tip(np.array([det.tip_x, det.tip_y]), dt)
        self.height = 0.6 * self.height + 0.4 * det.height if self.hits else det.height
        self.hits += 1
        self.missed = 0

    def hand(self) -> Hand:
        v = self.f_palm.dx
        return Hand(self.id, float(self.palm[0]), float(self.palm[1]), float(self.tip[0]), float(self.tip[1]),
                    float(self.height), float(v[0]), float(v[1]), self.age, self.det.area)


class HandTracker:
    """Přiřazuje detekce ke stálým rukám a vyhlazuje jejich polohu."""

    def __init__(self, settings: HandSettings) -> None:
        self.s = settings
        self.tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks = []

    def update(self, detections: list[_Detection], dt: float) -> list[Hand]:
        s = self.s
        dt = float(np.clip(dt, 1e-3, 0.2))
        # hladové párování podle vzdálenosti dlaní (rukou je málo)
        pairs = sorted(((math.hypot(t.palm[0] - d.x, t.palm[1] - d.y), ti, di)
                        for ti, t in enumerate(self.tracks) for di, d in enumerate(detections)))
        used_t: set[int] = set()
        used_d: set[int] = set()
        for dist, ti, di in pairs:
            if dist > s.match_distance or ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self.tracks[ti].update(detections[di], dt)
        for ti, t in enumerate(self.tracks):
            t.age += dt
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(detections):
            if di not in used_d:
                t = _Track(self._next_id, d, s)
                self._next_id += 1
                t.update(d, dt)
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.missed <= s.keep_frames]
        live = [t for t in self.tracks if t.hits >= s.confirm_frames]
        live.sort(key=lambda t: -t.det.area)
        return [t.hand() for t in live]


class HandDetector:
    """Pixelová detekce rukou + sledování. Volá se pro každý hloubkový snímek."""

    def __init__(self, cal: KinectCalibration, settings: HandSettings,
                 outer: tuple[float, float, float, float] | None = None) -> None:
        self.cal = cal
        self.s = settings
        W, D = cal.size
        # obdélník celé bedny (stěny) v souřadnicích hrací plochy – za ním už písek být nemůže
        self.outer = outer if outer is not None else (0.0, 0.0, W, D)
        intr = cal.intrinsics
        # Ruce jsou velké (dlaň ~ 1000 px), stačí poloviční rozlišení – 4× méně práce.
        self.ds = ds = max(1, int(settings.downsample))
        self.full_shape = (intr.height, intr.width)
        rays = intr.rays().astype(np.float32)[::ds, ::ds]
        self.shape = rays.shape[:2]
        self._rn = rays @ cal.normal.astype(np.float32)
        self._ra = rays @ cal.ex.astype(np.float32)
        self._rb = rays @ cal.ey.astype(np.float32)
        self._oa = float(cal.origin @ cal.ex)
        self._ob = float(cal.origin @ cal.ey)
        self._H = cal.plane_to_box.astype(np.float32)
        border = np.zeros(self.shape, bool)
        k = max(1, 4 // ds)
        border[:k, :] = border[-k:, :] = True
        border[:, :k] = border[:, -k:] = True
        self._border = border
        self.tracker = HandTracker(settings)
        self.reset()
        # poslední výsledky (pro operátora / ladění)
        self.hands: list[Hand] = []
        self.pixel_mask = np.zeros(self.full_shape, bool)
        self.fg_mask = np.zeros(self.shape, bool)

    def reset(self) -> None:
        self.bg = np.zeros(self.shape, np.float32)
        self._ref = np.zeros(self.shape, np.float32)
        self._stable = np.zeros(self.shape, np.int16)
        self._far = np.zeros(self.shape, np.int16)
        self._prev_hand = np.zeros(self.shape, bool)
        self.tracker.reset()

    # ------------------------------------------------------------------------------------
    def box_coords(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hloubka (H, W) [m] -> souřadnice hrací plochy x, y a výška h pro každý pixel."""
        h = z * self._rn + self.cal.offset
        a = z * self._ra - self._oa
        b = z * self._rb - self._ob
        H = self._H
        w = H[2, 0] * a + H[2, 1] * b + H[2, 2]
        x = (H[0, 0] * a + H[0, 1] * b + H[0, 2]) / w
        y = (H[1, 0] * a + H[1, 1] * b + H[1, 2]) / w
        return x, y, h

    def process(self, z: np.ndarray, dt: float, terrain: np.ndarray | None = None,
                ) -> tuple[list[Hand], np.ndarray]:
        """Jeden snímek (hloubka v metrech, 0 = neplatné).

        ``terrain`` = aktuální výšková mapa hrací plochy (na výšku dlaně nad pískem).
        Vrací (ruce, maska pixelů ruky v plném rozlišení, které terén nemá použít)."""
        s = self.s
        ds = self.ds
        if ds > 1:
            z_full = z
            z = z_full[::ds, ::ds]
            alt = z_full[ds // 2::ds, ds // 2::ds]
            z = np.where(z > 0, z, alt[:z.shape[0], :z.shape[1]])
        min_entry = max(1, s.min_entry_px // (ds * ds))
        min_area = max(4, s.min_area_px // (ds * ds))
        valid = z > 0
        bg = self.bg
        fresh = valid & (bg <= 0)
        bg[fresh] = z[fresh]
        d = np.where(valid & (bg > 0), bg - z, 0).astype(np.float32)

        x, y, h = self.box_coords(z)
        x0, y0, x1, y1 = self.outer
        band, em = s.band, s.entry_margin
        near_box = valid & (x > x0 - band) & (x < x1 + band) & (y > y0 - band) & (y < y1 + band)
        outside = (x < x0 - em) | (x > x1 + em) | (y < y0 - em) | (y > y1 + em)
        fg = near_box & (d > s.fg_threshold)

        # hloubkové hrany: skok mezi sousedy (neplatné sousedy ignorujeme)
        jump = np.zeros(self.shape, bool)
        J = s.edge_jump
        dv = np.abs(z[1:, :] - z[:-1, :]) > J
        dv &= valid[1:, :] & valid[:-1, :]
        jump[1:, :] |= dv
        jump[:-1, :] |= dv
        dh = np.abs(z[:, 1:] - z[:, :-1]) > J
        dh &= valid[:, 1:] & valid[:, :-1]
        jump[:, 1:] |= dh
        jump[:, :-1] |= dh
        # Vysoko nad pískem je hlava/tělo nakloněné nad bednou: vyřízne se a paže, které z něj
        # vycházejí, se tím oddělí (každá je pak samostatná ruka; navazují na tělo = vstup).
        tall = fg & (h > s.max_height)
        core = fg & ~jump & ~tall

        dets: list[_Detection] = []
        hand_px = np.zeros(self.shape, bool)
        if core.any():
            n, labels, stats, _ = cv2.connectedComponentsWithStats(core.view(np.uint8), connectivity=4)
            tall_edge = cv2.dilate(tall.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool) if tall.any() else tall
            entry = core & (outside | self._border | tall_edge)
            touch = np.bincount(labels[entry], minlength=n)
            inside = np.bincount(labels[core & ~outside], minlength=n)
            cand = np.flatnonzero((touch >= min_entry) & (inside >= min_area))
            cand = cand[cand > 0]
            for lab in cand:
                x0_, y0_, w_, h_ = stats[lab, :4]
                sl = (slice(y0_, y0_ + h_), slice(x0_, x0_ + w_))
                comp = labels[sl] == lab
                found = self._arm_ends(comp, entry[sl], x[sl], y[sl], h[sl], terrain, min_area)
                if found:
                    dets.extend(found)
                    hand_px[sl] |= comp
        self.fg_mask = fg

        hands = self.tracker.update(dets, dt)

        # maska pro terén: ruka + obrys (hrany, létající pixely) + kousek okolí
        if hand_px.any():
            grow = cv2.dilate(hand_px.view(np.uint8), np.ones((3, 3), np.uint8), iterations=2).view(bool)
            hand_px = grow & (fg | jump) | hand_px
            hand_px = cv2.dilate(hand_px.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool)
        # Pozadí nesmí pohltit paži ani dlaň. Nízké a nehybné části oblasti ruky daleko od dlaně
        # jsou ale hromádky písku, které ruka nahrnula – ty pozadí přijme, a tím se od ruky oddělí.
        guard = hand_px & (d >= s.arm_thickness)
        if dets and hand_px.any():
            near = np.zeros(self.shape, bool)
            for det in dets:
                if det.primary:
                    near |= (x - det.x) ** 2 + (y - det.y) ** 2 < s.palm_guard ** 2
                    near |= (x - det.tip_x) ** 2 + (y - det.tip_y) ** 2 < s.palm_guard ** 2
            guard |= hand_px & near
        self._update_background(z, valid, d, guard | self._prev_hand)
        self._prev_hand = guard
        self.hands = hands
        full = hand_px
        if ds > 1:
            H, W = self.full_shape
            full = np.repeat(np.repeat(hand_px, ds, 0), ds, 1)[:H, :W]
            if full.shape != (H, W):
                full = np.pad(full, ((0, H - full.shape[0]), (0, W - full.shape[1])), mode="edge")
        self.pixel_mask = full
        return hands, full

    def _arm_ends(self, comp: np.ndarray, entry: np.ndarray, x: np.ndarray, y: np.ndarray, h: np.ndarray,
                  terrain: np.ndarray | None, min_area: int) -> list[_Detection]:
        """Konce paží v jedné souvislé oblasti (může jich být víc – dvě ruce, ruka a hromádka).

        Pixely se promítnou do metrické mřížky (pohled shora), od místa vstupu přes stěnu se
        spočítá geodetická vzdálenost (cesta uvnitř oblasti) a konce = místní maxima té
        vzdálenosti. Dlaň = posledních ``palm_depth`` cm cesty u konce."""
        s = self.s
        cell = s.cell
        xs, ys, hs, ent = x[comp], y[comp], h[comp], entry[comp]
        if not ent.any():
            return []
        xmin, ymin = float(xs.min()), float(ys.min())
        gx = ((xs - xmin) / cell).astype(np.int32) + 1
        gy = ((ys - ymin) / cell).astype(np.int32) + 1
        gw, gh = int(gx.max()) + 2, int(gy.max()) + 2
        if gw * gh > 400_000:  # nesmysl (např. celý člověk nad bednou) – raději nic
            return []
        occ = np.zeros((gh, gw), np.uint8)
        occ[gy, gx] = 1
        occ |= cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # Analyzuje se jen vnitřek bedny: paže začíná tam, kde se dotýká stěny (nebo těla
        # vyříznutého nad bednou / okraje obrazu). Tělo za stěnou tak netvoří falešné větve.
        x0, y0, x1, y1 = self.outer
        cxs = xmin + (np.arange(gw) - 0.5) * cell
        cys = ymin + (np.arange(gh) - 0.5) * cell
        inside_cell = ((cxs[None, :] > x0) & (cxs[None, :] < x1)) & ((cys[:, None] > y0) & (cys[:, None] < y1))
        seed = np.zeros((gh, gw), np.uint8)
        seed[gy[ent], gx[ent]] = 1
        seed |= (occ.view(bool) & ~inside_cell).view(np.uint8)
        occ = occ & inside_cell.view(np.uint8)
        seed = cv2.dilate(seed, np.ones((3, 3), np.uint8)) & occ
        if not seed.any():
            return []
        k = max(1, int(round(s.entry_merge / cell)))
        merged = cv2.dilate(seed, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        n_arms, arm_lab = cv2.connectedComponents(merged, connectivity=8)
        seed_lab = np.where(seed > 0, arm_lab, 0).astype(np.int32)
        G, L = _geodesic(occ.view(bool), seed_lab)
        W, D = self.cal.size
        inset = s.min_inside
        palm_cells = s.palm_depth / cell
        yy_all, xx_all = np.nonzero(G >= 0)
        g_all, l_all = G[yy_all, xx_all], L[yy_all, xx_all]
        cx_all = xmin + (xx_all - 0.5) * cell
        cy_all = ymin + (yy_all - 0.5) * cell
        ok_all = (cx_all > x0 + inset) & (cx_all < x1 - inset) & (cy_all > y0 + inset) & (cy_all < y1 - inset)
        out: list[_Detection] = []
        taken: list[tuple[float, float]] = []
        occ_b = occ.view(bool)
        for arm in range(1, n_arms):
            for bi, (ex, ey, g_end, region) in enumerate(self._branch_ends(occ_b, G, L, arm, ok_all, yy_all,
                                                                          xx_all)):
                tipc = region & (G >= g_end - 2)
                tipc[:max(0, ey - 4)] = False
                tipc[ey + 5:] = False
                tipc[:, :max(0, ex - 4)] = False
                tipc[:, ex + 5:] = False
                ty_i, tx_i = np.nonzero(tipc)
                tx = xmin + (float(tx_i.mean()) - 0.5) * cell
                ty = ymin + (float(ty_i.mean()) - 0.5) * cell
                if any(math.hypot(tx - ax, ty - ay) < s.min_hand_gap for ax, ay in taken):
                    continue
                near = region[yy_all, xx_all] & (g_all >= g_end - palm_cells) & (
                    (xx_all - ex) ** 2 + (yy_all - ey) ** 2 <= ((s.palm_depth + 0.02) / cell) ** 2)
                if near.sum() < 4:
                    continue
                px = xmin + (float(xx_all[near].mean()) - 0.5) * cell
                py = ymin + (float(yy_all[near].mean()) - 0.5) * cell
                palm_mask = np.zeros((gh, gw), bool)
                palm_mask[yy_all[near], xx_all[near]] = True
                in_palm = palm_mask[gy, gx]
                ground = 0.0
                if terrain is not None and in_palm.any():
                    th, tw = terrain.shape
                    ix = np.clip((xs[in_palm] / W * tw).astype(np.int32), 0, tw - 1)
                    iy = np.clip((ys[in_palm] / D * th).astype(np.int32), 0, th - 1)
                    ground = terrain[iy, ix]
                height = float(np.median(hs[in_palm] - ground)) if in_palm.any() else 0.0
                if height > s.max_height or (bi > 0 and height > s.branch_max_height):
                    continue  # hlava, rameno nebo tělo nakloněné nad bednou
                taken.append((tx, ty))
                out.append(_Detection(float(np.clip(px, 0, W)), float(np.clip(py, 0, D)), float(np.clip(tx, 0, W)),
                                      float(np.clip(ty, 0, D)), max(0.0, height), int(in_palm.sum()),
                                      primary=bi == 0))
        return out

    def _branch_ends(self, occ: np.ndarray, G: np.ndarray, L: np.ndarray, arm: int, ok_all: np.ndarray,
                     yy_all: np.ndarray, xx_all: np.ndarray):
        """Konce jedné paže: nejvzdálenější bod + další větve delší než ``min_branch``.

        Větev bodu p vůči už nalezené cestě vstup→konec k: (G(p) + G_k(p) − G(k)) / 2
        = jak daleko p trčí z té cesty. Dvě ruce spojené tělem nebo hromádkou mají dlouhé
        větve, hrbolek písku u dlaně krátkou."""
        s = self.s
        cell = s.cell
        region = (L == arm) & occ
        okm = np.zeros(region.shape, bool)
        okm[yy_all[ok_all], xx_all[ok_all]] = True
        cand = region & okm
        if not cand.any():
            return []
        Gr = np.where(region, G, -1)
        k = int(np.argmax(np.where(cand, Gr, -1)))
        ey, ex = divmod(k, region.shape[1])
        if Gr[ey, ex] * cell < s.min_reach:
            return []
        ends = [(ex, ey, int(Gr[ey, ex]), region)]
        branch = np.full(region.shape, np.inf, np.float32)
        min_branch = s.min_branch / cell
        while len(ends) < s.max_hands_per_arm:
            sx, sy, sg, _ = ends[-1]
            seed = np.zeros(region.shape, np.int32)
            seed[sy, sx] = 1
            Gk, _ = _geodesic(region, seed)
            b = np.where(Gk >= 0, (Gr + Gk - sg) / 2.0, 0).astype(np.float32)
            branch = np.minimum(branch, b)
            score = np.where(cand, branch, -1)
            k = int(np.argmax(score))
            by, bx = divmod(k, region.shape[1])
            if score[by, bx] < min_branch or Gr[by, bx] * cell < s.min_reach:
                break
            ends.append((bx, by, int(Gr[by, bx]), region))
        return ends

    def _update_background(self, z: np.ndarray, valid: np.ndarray, d: np.ndarray, hand: np.ndarray) -> None:
        """Pozadí: dál = rychle přijmout, blíž = jen když vydrží a není to ruka."""
        s = self.s
        bg, ref, stable, far = self.bg, self._ref, self._stable, self._far
        T = s.bg_tolerance
        near = valid & (np.abs(d) < T)
        bg[near] += s.bg_rate * (z[near] - bg[near])

        farther = valid & (d <= -T)
        far[farther] += 1
        far[~farther] = 0
        go = far >= s.far_frames
        bg[go] = z[go]
        far[go] = 0

        closer = valid & (d >= T)
        same = closer & (np.abs(z - ref) < T)
        stable[same] += 1
        new = closer & ~same
        stable[new] = 1
        ref[new] = z[new]
        stable[~closer] = 0
        # písek: rychle; ruka: jen když se dlouho vůbec nehne (zapomenutý předmět na okraji)
        need = np.where(hand, s.still_frames, s.absorb_frames)
        go = stable >= need
        bg[go] = z[go]
        stable[go] = 0


def _geodesic(mask: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geodetická vzdálenost (v buňkách) od semínek uvnitř ``mask`` a příslušnost k semínku.

    ``seeds`` = štítky > 0. Vrací (G, L): G = -1 nedosažitelné, L = štítek nejbližšího semínka.
    Šíření vlny střídavě křížem a čtvercem (osmiúhelníková metrika, chyba < 8 %)."""
    m8 = mask.astype(np.uint8)
    L = np.where(mask, seeds, 0).astype(np.int32)
    G = np.where(L > 0, 0, -1).astype(np.int32)
    front = L > 0
    seen = front.copy()
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    square = np.ones((3, 3), np.uint8)
    step = 0
    while front.any():
        step += 1
        ker = cross if step % 2 else square
        lab = cv2.dilate(np.where(front, L, 0).astype(np.float32), ker)
        new = (lab > 0) & (m8 > 0) & ~seen
        if not new.any():
            break
        G[new] = step
        L[new] = lab[new].astype(np.int32)
        seen |= new
        front = new
    return G, L


def outer_rect(full: KinectCalibration | None, play: KinectCalibration) -> tuple[float, float, float, float] | None:
    """Obdélník celé bedny (stěny) v souřadnicích oříznuté hrací plochy."""
    if full is None or full is play:
        return None
    W, D = full.size
    pts = full.box_to_camera(np.array([0.0, W, W, 0.0]), np.array([0.0, 0.0, D, D]))
    x, y, _ = play.camera_to_box(pts)
    return float(min(x[0], x[3])), float(min(y[0], y[1])), float(max(x[1], x[2])), float(max(y[2], y[3]))
