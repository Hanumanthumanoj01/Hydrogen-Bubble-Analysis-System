"""
bubble_analysis.py
==================
Library module, not run directly: imported by main.py (Image Analyzer tab)
and run_all.py. To use it, start the backend:  uvicorn main:app --port 8000

Core image analysis engine.

Implements the two-branch pipeline from the project exposé:
  Branch 1 — Shannon (information) entropy, reproducing Wagner et al. (2025)
  Branch 2 — Geometric bubble detection: "Every bubble" (rim-coverage test,
             detect_bubbles) or "Main bubbles" (detect_primary.py)

All derived metrics follow the paper's calibration:
  - 0.0064 mm/pixel (telephoto/microscope lens)
  - Cathode area 2.54 cm²  (Experiment 1)
  - Faraday:  H2O + 2e⁻ → ½H₂ + OH⁻  (z = 2)
"""

import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Tuple, Optional


# ── Physical / calibration constants ─────────────────────────────────────────
# Imported from constants.py — the single source of truth shared with
# video_analysis.py. Do NOT redefine these here: if the image engine and the
# video engine hold separate copies, changing the calibration in one place
# silently leaves the other measuring in different units.
from constants import (
    MM_PER_PIXEL, FARADAY, Z_ELECTRONS, M_H2, MOLAR_VOLUME_H2,
    SIZE_BINS_MM, SIZE_BIN_LABELS,
)


class BubbleAnalyzer:

    # ── Public entry point ────────────────────────────────────────────────────
    def full_analysis(
        self,
        img_bgr: np.ndarray,
        current_density: float,
        electrode_type: str,
        cathode_area: float,
        sensitivity: str = "medium",
        contrast_mode: str = "dark",       # kept for old callers; both polarities are tested
        roi: Optional[Tuple[float, float, float]] = None,
        detector: str = "standard",
        mm_min: float = 0.20,
        mm_max: float = 0.90,
        min_pct: float = 3.0,
        max_pct: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Run the complete two-branch pipeline and return all metrics + images.

        detector selects which geometric branch runs.

          "standard"  the original multi-cue detector. It reports every bubble
                      it can find, and it is the one the published counts of
                      123 / 123 / 110 were validated against, so it stays the
                      default and those numbers stay reproducible.

          "primary"   the targeted detector of detect_primary.py, which finds
                      only the large, well-formed bubbles between mm_min and
                      mm_max and verifies each one against its radial
                      brightness profile. Use it when the question is whether
                      the big bubbles were found accurately rather than how
                      many bubbles there are in total.
        """

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ── Branch 1: Shannon entropy ─────────────────────────────────────────
        # Unaffected by the detector choice: it never looks at the detections.
        H, H_norm = self.shannon_entropy(gray)

        # ── Branch 2: Geometric bubble detection ──────────────────────────────
        if detector in ("primary", "ring"):
            bubbles = self._detect_primary(
                gray, mm_min, mm_max, min_pct, max_pct, bgr=img_bgr,
                mode=("ring" if detector == "ring" else "core"))
        else:
            bubbles = self.detect_bubbles(img_bgr, sensitivity, roi)

        # ── Derived metrics ───────────────────────────────────────────────────
        metrics = self.derive_metrics(bubbles, gray.shape, current_density, cathode_area)

        # ── Overlay images (base64 PNG) ───────────────────────────────────────
        overlay_b64 = self._make_overlay(img_bgr.copy(), bubbles, self.last_scores,
                                         bold=(detector in ("primary", "ring")))
        heatmap_b64 = self._make_heatmap(gray.shape, bubbles)
        histogram_b64 = self._make_histogram(gray)

        return {
            # Entropy branch
            "shannon_entropy": round(H, 4),
            "shannon_entropy_norm": round(H_norm * 100, 2),
            # Detection
            "bubble_count": len(bubbles),
            "bubbles": [{"x": int(b[0]), "y": int(b[1]),
                          "r_px": int(b[2]),
                          "r_mm": round(b[2] * MM_PER_PIXEL, 4),
                          "diameter_mm": round(b[2] * 2 * MM_PER_PIXEL, 4),
                          "curvature": round(1.0 / (b[2] * MM_PER_PIXEL), 4) if b[2] > 0 else 0,
                          "p_bubble": round(self.last_scores.get(b, 0.5), 3)}
                         for b in bubbles],
            # Derived
            **metrics,
            # Images
            "overlay_image": overlay_b64,
            "heatmap_image": heatmap_b64,
            "histogram_image": histogram_b64,
            # Meta
            "electrode_type": electrode_type,
            "current_density": current_density,
            "cathode_area": cathode_area,
            "mm_per_pixel": MM_PER_PIXEL,
        }

    # ── Branch 1: Shannon entropy ─────────────────────────────────────────────
    @staticmethod
    def shannon_entropy(gray: np.ndarray) -> Tuple[float, float]:
        """
        Compute Shannon (information) entropy H and normalised H_norm from
        the greyscale pixel histogram.

        H = -Σ p_i · ln(p_i)          (Eq. 7, Wagner et al. 2025)
        H_norm = H / ln(256)           (Eq. 8, Wagner et al. 2025)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()                    # normalise to probabilities
        nonzero = hist[hist > 0]
        H = float(-np.sum(nonzero * np.log(nonzero)))
        H_norm = H / np.log(256)
        return H, H_norm

    # ── Branch 2: Bubble detection ────────────────────────────────────────────
    def _detect_primary(self, gray, mm_min, mm_max, min_pct=3.0, max_pct=60.0, bgr=None,
                        mode="core"):
        """
        Bridge to detect_primary.py.

        min_pct and max_pct are the bubble diameter as a percentage of the
        image height. Sizing this way rather than in millimetres is what makes
        the same setting work on a 640 x 480 frame and on a zoomed screen
        capture of the same recording: a bubble that fills a tenth of the
        picture fills a tenth of it at any resolution.

        Returns the same (x, y, r) tuples the rest of this class expects, and
        fills last_scores with the detector's own confidence so the overlay
        colours mean the same thing in both modes. Imported here rather than at
        module level so that a missing scikit-image only disables this mode
        instead of breaking the whole backend.
        """
        try:
            from detect_primary import detect_primary
        except ImportError as exc:
            raise RuntimeError(
                "The large-bubble detector needs scikit-image. "
                "Install it with:  pip3 install scikit-image"
            ) from exc

        hits = detect_primary(gray, (mm_min, mm_max), MM_PER_PIXEL,
                              size_frac=(min_pct / 100.0, max_pct / 100.0), bgr=bgr,
                              mode=mode)
        out, scores = [], {}
        # a circle touching the camera's overlay text, the crosshair, a black
        # border or the player bar of a screen capture is not a bubble
        valid = picture_mask(bgr) if bgr is not None else np.ones(gray.shape, bool)
        H, W = valid.shape
        for h in hits:
            t = (int(h["x"]), int(h["y"]), int(round(h["r_px"])))
            x0, x1 = max(0, t[0] - t[2]), min(W, t[0] + t[2] + 1)
            y0, y1 = max(0, t[1] - t[2]), min(H, t[1] + t[2] + 1)
            if x1 - x0 < 2 * t[2] or y1 - y0 < 2 * t[2]:
                continue                       # cut off by the image edge
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disc = np.hypot(xx - t[0], yy - t[1]) <= t[2]
            if valid[y0:y1, x0:x1][disc].mean() < 0.97:
                continue
            out.append(t)
            # contrast is in standard deviations; 2 sd is already emphatic, so
            # that is taken as the top of the confidence scale.
            scores[t] = float(min(1.0, (h["contrast"] / 2.0) * h["agreement"]))
        self.last_scores = scores
        return out

    def detect_bubbles(
        self,
        bgr: np.ndarray,
        sensitivity: str = "medium",
        roi: Optional[Tuple[float, float, float]] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        "Every bubble, froth included": the rim-coverage detector.

        WHY IT WAS REWRITTEN (November 2026)
        ------------------------------------
        The earlier version proposed circles with the Hough transform and kept
        a circle when a weighted score passed a threshold. The largest weight
        went to "size close to 0.5 mm", so a ring drawn on empty background of
        the right size passed. Its counts on the three published images (123,
        123, 110) fell inside the published range, but many circles did not
        sit on bubbles, and on screen captures it circled the timestamp, the
        crosshair, the black border and the player bar.

        HOW IT WORKS
        ------------
        1. picture_mask() finds the camera picture: overlays, borders and a
           media-player bar are excluded, and so is any circle touching them.
        2. The picture is scaled to a height of 420 px, so the same settings
           work on a 640 x 480 frame and on a 2000 px screen capture.
        3. Candidates come from two matched filters at radii from 1.2 % to
           20 % of the picture height: a bright ring around a darker inside
           (how a bubble looks on the GDE images, lit from the camera side)
           and a dark disc in a brighter surround (the dark-core bubbles on
           the ANS plate).
        4. Each candidate is unrolled into polar coordinates (48 angles) and
           tested. A circle is kept only when
             - the rim stands out from the inside of the bubble at no fewer
               than 80-90 % of the angles (a straight mesh streak crosses a
               circle at two angles only),
             - the rim lies at the same radius all round (spread <= 12 % R),
             - it is round, not a square mesh opening (four-fold component of
               the rim radius <= 5 % R),
             - its contrast exceeds 4 to 6 times the texture noise around it.
        5. A large circle that contains two or more smaller accepted circles
           is dropped: a bubble does not contain bubbles.
        6. Circles more than three times the median size must be confirmed by
           the large-bubble rim detector (detect_primary.py), which is reliable
           for large bubbles; its confirmed circles replace them.

        Sensitivity sets the evidence required: low (fewer false circles),
        medium, high (more bubbles found, more false ones).

        Returns (x, y, r) in pixels of the input image; self.last_scores holds
        a confidence in [0, 1] per circle.
        """
        valid = picture_mask(bgr)
        if roi is not None:
            h, w = valid.shape
            cx_frac, cy_frac, r_frac = roi
            yy, xx = np.ogrid[:h, :w]
            valid &= (xx - w * cx_frac) ** 2 + (yy - h * cy_frac) ** 2 <= (min(w, h) / 2 * r_frac) ** 2
        hits = rim_coverage_detect(bgr, valid, sensitivity)
        out = [(x, y, r, sc) for x, y, r, sc in hits]
        r_med = float(np.median([t[2] for t in out])) if len(out) >= 10 else None
        if r_med is not None:
            out = [t for t in out if t[2] <= 3.0 * r_med]
        big = []
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for x, y, r in self._detect_primary(gray, 0.2, 0.9, 3.0, 60.0, bgr=bgr):
                if r_med is not None and r < 2.0 * r_med:
                    continue
                big.append((float(x), float(y), float(r), 1.0))
        except RuntimeError:
            pass                      # scikit-image missing: rim-coverage circles only
        out = [t for t in out if all(np.hypot(t[0] - B[0], t[1] - B[1]) > 0.9 * B[2] for B in big)]
        res = big + out
        self.last_scores = {(int(x), int(y), int(round(r))): float(sc) for x, y, r, sc in res}
        return [(int(x), int(y), int(round(r))) for x, y, r, _ in res]

    # ── Derived metrics ───────────────────────────────────────────────────────
    @staticmethod
    def derive_metrics(
        bubbles: List[Tuple[float, float, float]],
        img_shape: Tuple[int, int],
        current_density: float,
        cathode_area: float
    ) -> Dict[str, Any]:
        h, w = img_shape
        pixel_area = h * w

        if not bubbles:
            sizes_mm, curvatures = [], []
            mean_diameter_mm = 0.0
            mean_curvature = 0.0
            coverage = 0.0
        else:
            radii_px = np.array([b[2] for b in bubbles])
            sizes_mm = (radii_px * 2 * MM_PER_PIXEL).tolist()
            curvatures = (1.0 / (radii_px * MM_PER_PIXEL)).tolist()
            mean_diameter_mm = float(np.mean(sizes_mm))
            mean_curvature = float(np.mean(curvatures))

            # Coverage: binary mask to handle overlaps correctly
            mask = np.zeros((h, w), dtype=np.uint8)
            for x, y, r in bubbles:
                cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
            coverage = float(mask.sum() / pixel_area * 100)

        # Size distribution bins matching Wagner et al. (2025), Fig. 4
        bins = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.1, 9999]
        bin_labels = ["<0.1", "0.1–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.1", ">1.1"]
        counts = [0] * (len(bins) - 1)
        for d in sizes_mm:
            for i in range(len(bins) - 1):
                if bins[i] <= d < bins[i + 1]:
                    counts[i] += 1
                    break

        # Nucleation sites: count occupied cells of a coarse 8×8 grid
        if bubbles:
            cell_w, cell_h = w / 8, h / 8
            occupied = set()
            for x, y, _ in bubbles:
                occupied.add((int(x / cell_w), int(y / cell_h)))
            nucleation_sites = len(occupied)
        else:
            nucleation_sites = 0

        # Bubble density (per cm²)
        img_area_cm2 = pixel_area * (MM_PER_PIXEL / 10) ** 2
        bubble_density = len(bubbles) / img_area_cm2 if img_area_cm2 > 0 else 0

        # H₂ production rate — Faraday's law
        # n_dot = (J · A) / (z · F)   [mol/s]
        # V_dot = n_dot · Vm            [mL/s] → ×60 → mL/min
        J_A_per_cm2 = current_density / 1000.0   # A/cm²
        n_dot = (J_A_per_cm2 * cathode_area) / (Z_ELECTRONS * FARADAY)
        H2_mL_per_min = n_dot * MOLAR_VOLUME_H2 * 60.0

        return {
            "mean_diameter_mm": round(mean_diameter_mm, 4),
            "mean_curvature_1_per_mm": round(mean_curvature, 4),
            "surface_coverage_pct": round(coverage, 2),
            "nucleation_sites": nucleation_sites,
            "bubble_density_per_cm2": round(bubble_density, 1),
            "size_distribution": {"labels": bin_labels, "counts": counts},
            "H2_production_mL_per_min": round(H2_mL_per_min, 3),
        }

    # ── Overlay image ─────────────────────────────────────────────────────────
    @staticmethod
    def _make_overlay(img: np.ndarray, bubbles: List[Tuple],
                      scores: Optional[Dict] = None, bold: bool = False) -> str:
        """
        Draw the detections.

        Line thickness is set from the size of the image rather than fixed at
        one pixel. A one-pixel circle is legible on a 640 x 480 frame and all
        but invisible on a 2000 px screen capture shrunk to fit a panel, which
        is how the detections came to look faint and broken on screen.

        In bold mode every circle is drawn in one bright green, with a dark
        outline underneath so it stays visible over both the bright plate and
        the dark bubble. Confidence is not colour-coded there: in that mode
        everything shown has already passed verification, and a figure that is
        going into a report reads better when a circle is simply a circle.
        """
        h, w = img.shape[:2]
        t = max(2, int(round(min(h, w) / 260.0)))
        for x, y, r in bubbles:
            x, y, r = int(x), int(y), int(r)
            if bold:
                cv2.circle(img, (x, y), r, (0, 0, 0), t + 2, cv2.LINE_AA)
                cv2.circle(img, (x, y), r, (60, 255, 60), t, cv2.LINE_AA)
                cv2.drawMarker(img, (x, y), (60, 255, 60), cv2.MARKER_CROSS,
                               max(8, t * 5), t)
                continue
            if scores is not None:
                p = scores.get((x, y, r), 0.5)
                # green (confident) -> yellow -> red (marginal), BGR order
                color = (0, int(255 * min(1, p * 2)), int(255 * min(1, (1 - p) * 2)))
            else:
                color = (0, 0, 255)
            cv2.circle(img, (x, y), r, color, max(1, t - 1), cv2.LINE_AA)
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode()

    # ── Density heatmap ───────────────────────────────────────────────────────
    @staticmethod
    def _make_heatmap(shape: Tuple[int, int], bubbles: List[Tuple]) -> str:
        h, w = shape
        heat = np.zeros((h, w), dtype=np.float32)
        for x, y, r in bubbles:
            cv2.circle(heat, (int(x), int(y)), max(int(r * 2), 10), 1.0, -1)
        heat = cv2.GaussianBlur(heat, (51, 51), 0)
        if heat.max() > 0:
            heat = heat / heat.max()
        heat_u8 = (heat * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        _, buf = cv2.imencode(".png", colored)
        return base64.b64encode(buf).decode()

    # ── Pixel brightness histogram ────────────────────────────────────────────
    @staticmethod
    def _make_histogram(gray: np.ndarray) -> str:
        h_img = np.zeros((200, 512, 3), dtype=np.uint8)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = (hist / hist.max() * 180).astype(int)
        for i, val in enumerate(hist_norm):
            x = i * 2
            cv2.line(h_img, (x, 200), (x, 200 - val), (100, 180, 255), 2)
        _, buf = cv2.imencode(".png", h_img)
        return base64.b64encode(buf).decode()


# ─────────────────────────────────────────────────────────────────────────────
#  Camera picture inside a screen capture
# ─────────────────────────────────────────────────────────────────────────────

def picture_mask(bgr: np.ndarray) -> np.ndarray:
    """
    True where the pixels belong to the camera picture.

    A screen capture of the video player contains more than the picture: a
    black letterbox border, the player bar with its coloured progress line,
    and the camera's own red overlay (frame rate, timestamp, crosshair). None
    of it is a bubble. Excluded are
      - coloured pixels (the camera image is grey), widened by a few pixels,
      - columns and rows without texture at the sides (the letterbox),
      - everything below a coloured line that crosses most of the width in
        the bottom fifth of the image (the player's progress bar).
    On a plain camera frame the whole image is returned.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    colour = (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32), cv2.CV_32F))
    tex = cv2.blur(lap, (31, 31))

    def longest_run(v):
        ok = v > 0.35 * np.median(v)
        best, best_len, start = (0, len(v)), 0, None
        for i, o in enumerate(np.r_[ok, False]):
            if o and start is None:
                start = i
            if not o and start is not None:
                if i - start > best_len:
                    best_len, best = i - start, (start, i)
                start = None
        return best

    x0, x1 = longest_run(np.median(tex, axis=0))
    y0, y1 = longest_run(np.median(tex, axis=1))
    frac = colour[:, x0:x1].mean(axis=1)
    bar = [y for y in range(int(0.8 * H), H) if frac[y] > 0.5]
    if bar:
        y1 = min(y1, bar[0] - 4)
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    # Coloured overlay TEXT excludes its area. Thin coloured LINES (the
    # crosshair) do not: a crosshair drawn across a bubble does not make the
    # bubble less real, and rejecting every circle it touches loses bubbles.
    # A crosshair line is a band of at most a few pixels along a whole row or
    # column (broken into pieces where the plate is bright); text is a band
    # tens of pixels tall.
    text = colour.copy()

    def thin_bands(frac):
        on = np.r_[frac > 0.15, False]
        out, start = [], None
        for i, o in enumerate(on):
            if o and start is None:
                start = i
            if not o and start is not None:
                if i - start <= max(6, int(0.006 * len(frac))):
                    out.append((start, i))
                start = None
        return out
    for a0, a1 in thin_bands(colour[:, x0:x1].mean(axis=1)):
        text[max(0, a0 - 3):a1 + 3, :] = False
    for a0, a1 in thin_bands(colour[y0:y1, :].mean(axis=0)):
        text[:, max(0, a0 - 3):a1 + 3] = False
    text = text.astype(np.uint8)
    m &= ~(cv2.dilate(text, np.ones((7, 7), np.uint8)) > 0)
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  Rim-coverage detector (used by BubbleAnalyzer.detect_bubbles)
# ─────────────────────────────────────────────────────────────────────────────

RIM_NORM_H = 420.0          # working height of the picture, px
RIM_REL = (0.012, 0.20)     # radius range, fraction of the picture height
RIM_SENS = {                # per-angle k (x noise), coverage, contrast (x noise)
    "low":    (4.0, 0.90, 6.0),
    "medium": (3.5, 0.85, 5.0),
    "high":   (3.0, 0.80, 4.0),
}
RIM_SPREAD_MAX = 0.12       # std of the rim radius over the angles, x R
RIM_H4_MAX = 0.05           # four-fold (square) component of the rim radius, x R


def _rim_kernel(r, kind):
    s = int(np.ceil(1.6 * r)) + 2
    yy, xx = np.mgrid[-s:s + 1, -s:s + 1]
    d = np.hypot(xx, yy)
    if kind == "ring":
        pos, neg = np.abs(d - r) <= max(1.2, 0.15 * r), d <= 0.6 * r
    else:
        pos, neg = (d >= 1.1 * r) & (d <= 1.45 * r), d <= 0.7 * r
    return (pos / pos.sum() - neg / max(neg.sum(), 1)).astype(np.float32)


def _rim_polar(g, x, y, R, n=48, k=29):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)[:, None]
    rr = np.linspace(0.2, 1.6, k)
    mx = (x + R * rr[None, :] * np.cos(ang)).astype(np.float32)
    my = (y + R * rr[None, :] * np.sin(ang)).astype(np.float32)
    return cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), rr


def _rim_test(g, x, y, R, kind, noise, kfac):
    """coverage, spread, contrast, rim radius factor, four-fold component"""
    P, rr = _rim_polar(g, x, y, R)
    inner = np.median(P[:, (rr >= 0.45) & (rr <= 0.7)], axis=1)   # clear of the highlight dot
    if kind == "ring":
        band = (rr >= 0.8) & (rr <= 1.2)
        rim = P[:, band].max(axis=1)
        pos = rr[band][P[:, band].argmax(axis=1)]
        diff = rim - inner
    else:
        outer = np.median(P[:, (rr >= 1.15) & (rr <= 1.5)], axis=1)
        dP = np.diff(P, axis=1)
        rmid = 0.5 * (rr[1:] + rr[:-1])
        b2 = (rmid >= 0.75) & (rmid <= 1.25)
        pos = rmid[b2][dP[:, b2].argmax(axis=1)]
        diff = outer - inner
    hit = diff > kfac * noise
    a = np.linspace(0, 2 * np.pi, len(pos), endpoint=False)
    h4 = 2 * abs(np.mean((pos - pos.mean()) * np.exp(4j * a)))
    spread = float(np.std(pos[hit])) if hit.sum() > 3 else 1.0
    rpos = float(np.median(pos[hit])) if hit.any() else 1.0
    return float(hit.mean()), spread, float(np.median(diff) / noise), rpos, float(h4)


def rim_coverage_detect(bgr, valid, sensitivity="medium"):
    """Returns [(x, y, r, score)] in pixels of bgr. See detect_bubbles."""
    ys, xs = np.nonzero(valid)
    if len(ys) == 0:
        return []
    by0, by1 = ys.min(), ys.max() + 1
    s = RIM_NORM_H / max(1, by1 - by0)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    v = cv2.resize(valid.astype(np.uint8), (g.shape[1], g.shape[0]),
                   interpolation=cv2.INTER_NEAREST) > 0
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    g2 = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    noise_global = 1.4826 * float(np.median(np.abs(hp - np.median(hp)))) + 1e-3
    sd = {w: np.sqrt(cv2.blur(hp * hp, (w, w))) + 1e-3 for w in (11, 21, 41, 81, 161)}
    kfac, cmin, conmin = RIM_SENS.get(sensitivity, RIM_SENS["medium"])

    # candidates: local maxima of the two matched filters over position and radius
    radii, r = [], max(3.0, RIM_REL[0] * RIM_NORM_H)
    while r <= RIM_REL[1] * RIM_NORM_H:
        radii.append(r)
        r *= 1.12
    cands = []
    for kind in ("ring", "disk"):
        stack = np.stack([cv2.filter2D(g, -1, _rim_kernel(r, kind), borderType=cv2.BORDER_REFLECT)
                          for r in radii])
        for i, r in enumerate(radii):
            R = stack[i]
            mx = stack[max(0, i - 1):i + 2].max(axis=0)
            ks = max(3, int(r) | 1)
            pk = (R >= cv2.dilate(R, np.ones((ks, ks), np.uint8))) & (R >= mx) \
                & (R > 1.2 * noise_global) & v
            for y, x in zip(*np.nonzero(pk)):
                cands.append((float(R[y, x]), float(x), float(y), r, kind))
    # strongest first; the same spot at a neighbouring radius is one candidate
    cands.sort(reverse=True)
    pruned = []
    grid = {}
    for c in cands:
        resp, x, y, r, kind = c
        key = (int(x // 8), int(y // 8))
        near = [p for dx in (-1, 0, 1) for dy in (-1, 0, 1) for p in grid.get((key[0] + dx, key[1] + dy), [])]
        if any(p[4] == kind and np.hypot(p[1] - x, p[2] - y) < 0.35 * min(p[3], r)
               and 0.75 < p[3] / r < 1.33 for p in near):
            continue
        pruned.append(c)
        grid.setdefault(key, []).append(c)

    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    acc = []
    for resp, x, y, r, kind in pruned:
        w = min(sd, key=lambda k: abs(k - 4 * r))
        noise = max(noise_global, 0.75 * float(sd[w][int(y), int(x)]))
        best = None
        for f in (1.0, 0.85, 1.18):
            R = r * f
            if R < 8:     # small: test on the 2x image so the rim spans enough pixels
                res = _rim_test(g2, 2 * x + 0.5, 2 * y + 0.5, 2 * R, kind, noise, kfac)
            else:
                res = _rim_test(g, x, y, R, kind, noise, kfac)
            sc = res[0] + 0.1 * res[2]
            if best is None or sc > best[0]:
                best = (sc, R * res[3]) + res
            if f == 1.0 and res[0] < cmin - 0.2:
                break     # far from passing: no need to try other radii
        sc, Rf, cov, spread, con, _, h4 = best
        if not (cov >= cmin and spread <= RIM_SPREAD_MAX and h4 <= RIM_H4_MAX
                and con >= conmin + (1.0 if kind == "disk" else 0.0)):
            continue
        # the circle and its surroundings must lie in the camera picture
        px = (x + 1.5 * Rf * np.cos(t)).astype(int)
        py = (y + 1.5 * Rf * np.sin(t)).astype(int)
        ins = (px >= 0) & (py >= 0) & (px < g.shape[1]) & (py < g.shape[0])
        ok = np.zeros(len(t), bool)
        ok[ins] = v[py[ins], px[ins]]
        x0, x1 = int(max(0, x - 1.2 * Rf)), int(min(g.shape[1], x + 1.2 * Rf + 1))
        y0, y1 = int(max(0, y - 1.2 * Rf)), int(min(g.shape[0], y + 1.2 * Rf + 1))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disc = np.hypot(xx - x, yy - y) <= 1.2 * Rf
        if ok.mean() >= 0.95 and v[y0:y1, x0:x1][disc].all():
            acc.append((sc, x, y, Rf))
    acc.sort(reverse=True)
    acc = [a for a in acc if sum(1 for b in acc if b[3] < 0.6 * a[3]
                                 and np.hypot(b[1] - a[1], b[2] - a[2]) < 0.85 * a[3]) < 2]
    kept = []
    for a in acc:
        if all(np.hypot(a[1] - k[1], a[2] - k[2]) > 0.7 * max(a[3], k[3]) for k in kept):
            kept.append(a)
    top = max([k[0] for k in kept], default=1.0)
    return [(x / s, y / s, R / s, min(1.0, sc / top)) for sc, x, y, R in kept]
