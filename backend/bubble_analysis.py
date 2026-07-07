"""
bubble_analysis.py
==================
Core image analysis engine.

Implements the two-branch pipeline from the project exposé:
  Branch 1 — Shannon (information) entropy, reproducing Wagner et al. (2025)
  Branch 2 — Geometric bubble detection (Hough Circle Transform + top-hat filter)

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
MM_PER_PIXEL: float = 0.0064      # mm per pixel  (Wagner et al. 2025)
FARADAY: float = 96485.0          # C/mol
Z_ELECTRONS: int = 2              # electrons per H₂ molecule
M_H2: float = 2.016               # g/mol
MOLAR_VOLUME_H2: float = 22400.0  # mL/mol  (STP)


class BubbleAnalyzer:

    # ── Public entry point ────────────────────────────────────────────────────
    def full_analysis(
        self,
        img_bgr: np.ndarray,
        current_density: float,
        electrode_type: str,
        cathode_area: float,
        sensitivity: str = "medium",
        contrast_mode: str = "dark",
        roi: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """Run the complete two-branch pipeline and return all metrics + images."""

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ── Branch 1: Shannon entropy ─────────────────────────────────────────
        H, H_norm = self.shannon_entropy(gray)

        # ── Branch 2: Geometric bubble detection ──────────────────────────────
        bubbles = self.detect_bubbles(gray, sensitivity, contrast_mode, roi)

        # ── Derived metrics ───────────────────────────────────────────────────
        metrics = self.derive_metrics(bubbles, gray.shape, current_density, cathode_area)

        # ── Overlay images (base64 PNG) ───────────────────────────────────────
        overlay_b64 = self._make_overlay(img_bgr.copy(), bubbles, self.last_scores)
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
    def detect_bubbles(
        self,
        gray: np.ndarray,
        sensitivity: str = "medium",
        contrast_mode: str = "dark",
        roi: Optional[Tuple[float, float, float]] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Detect bubbles using:
          1. CLAHE                         — local contrast boost for faint/small bubbles
          2. Morphological black/top-hat filter — removes cross-hatched grid background
          3. Median blur                   — suppresses noise
          4. Hough Circle Transform        — finds circular features
          5. Non-max suppression           — collapses duplicate overlapping detections

        contrast_mode:
          'dark'   — real hydrogen bubbles (they block the backlight). Default.
          'bright' — for footage of specular/backlit round objects that show as
                     bright highlights rather than dark blobs (e.g. water
                     droplets on a lit reflective surface).
          'both'   — runs dark and bright as two independent passes and merges
                     the results. NOTE: this must stay two independent passes,
                     not one combined threshold — combining polarities in a
                     single binary mask lets nearby opposite-polarity features
                     get bridged together by the morphological closing step,
                     which then FAILS the circularity filter and rejects both.
                     (Verified: naive combination collapsed detections well
                     below either single polarity alone.)

        roi: optional (center_x_frac, center_y_frac, radius_frac) — restricts
        detection to a circular region of the frame (e.g. to exclude
        equipment/rim visible around a real viewing window). Fractions of
        min(width, height)/2 and image dimensions, matching the frontend's
        click-to-place ROI control.

        Returns list of (x, y, r) in pixels.
        """
        if contrast_mode == "both":
            dark = self._detect_bubbles_single_pass(gray, sensitivity, "dark", roi)
            dark_scores = dict(self.last_scores)
            bright = self._detect_bubbles_single_pass(gray, sensitivity, "bright", roi)
            dark_scores.update(self.last_scores)  # bright pass's scores take precedence on overlap
            self.last_scores = dark_scores
            return self._merge_bubble_lists(dark, bright)
        return self._detect_bubbles_single_pass(gray, sensitivity, contrast_mode, roi)

    @staticmethod
    def _merge_bubble_lists(a, b):
        """Two detections landing on the same spot (one from each polarity
        pass) are almost always the same physical object caught twice."""
        merged = list(a)
        for cb in b:
            dup = False
            for ca in merged:
                d = ((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2) ** 0.5
                if d < 0.6 * (ca[2] + cb[2]):
                    dup = True
                    break
            if not dup:
                merged.append(cb)
        return merged

    def _detect_bubbles_single_pass(
        self,
        gray: np.ndarray,
        sensitivity: str,
        contrast_mode: str,
        roi: Optional[Tuple[float, float, float]],
    ) -> List[Tuple[float, float, float]]:
        """
        Note on method: bilateral filtering and watershed segmentation were
        tested as alternatives/additions here and empirically REJECTED — on
        the real GDE sample images, watershed's jagged per-region boundaries
        made the (correct, round) bubbles fail circularity checks, collapsing
        counts from ~100 to ~25 (verified against these exact files). CLAHE
        was the one change that consistently helped (+12-14% recall across
        all three current-density samples, verified individually), so it's
        the only preprocessing addition kept.
        """
        # ── CLAHE — local contrast enhancement ────────────────────────────────
        # Boosts faint/small bubbles that a global threshold would miss, without
        # the aggressive smoothing of a bilateral filter (which, tested here,
        # slightly *reduced* recall by softening the edges Hough relies on).
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # ── Background / grid suppression (black-hat or top-hat) ─────────────
        # Bubbles are usually DARK quasi-circular blobs on a lighter electrode/
        # electrolyte background (they block the backlight): MORPH_BLACKHAT is
        # the correct polarity for that. 'bright' mode (MORPH_TOPHAT) is for
        # footage where round objects show as bright highlights instead —
        # arbitrary uploaded video isn't guaranteed to follow the lab-bubble
        # convention the way the calibrated GDE sample images do.
        kernel_size = max(31, int(gray.shape[0] * 0.06) | 1)  # odd, ~6% of height
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        op = cv2.MORPH_TOPHAT if contrast_mode == "bright" else cv2.MORPH_BLACKHAT
        filtered = cv2.morphologyEx(enhanced, op, kernel)

        # ── Denoise ──────────────────────────────────────────────────────────
        blurred = cv2.medianBlur(filtered, 5)

        # ── Optional circular ROI mask — exclude equipment/rim outside it ────
        if roi is not None:
            h, w = gray.shape
            cx_frac, cy_frac, r_frac = roi
            cx, cy = w * cx_frac, h * cy_frac
            max_r = min(w, h) / 2 * r_frac
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 > max_r ** 2
            blurred = blurred.copy()
            blurred[mask] = 0

        # ── Sensitivity → Hough parameters ───────────────────────────────────
        sens_map = {
            "low":    {"dp": 1.5, "minDist": 30, "param1": 60, "param2": 35, "minR": 5,  "maxR": 80},
            "medium": {"dp": 1.2, "minDist": 18, "param1": 50, "param2": 28, "minR": 4,  "maxR": 80},
            "high":   {"dp": 1.0, "minDist": 12, "param1": 40, "param2": 20, "minR": 3,  "maxR": 80},
        }
        p = sens_map.get(sensitivity, sens_map["medium"])

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=p["dp"],
            minDist=p["minDist"],
            param1=p["param1"],
            param2=p["param2"],
            minRadius=p["minR"],
            maxRadius=p["maxR"],
        )

        if circles is None:
            return []

        circles = np.round(circles[0]).astype(int)
        # Filter: keep circles fully inside the image
        h, w = gray.shape
        valid = [
            (float(x), float(y), float(r))
            for x, y, r in circles
            if 0 <= x - r and x + r <= w and 0 <= y - r and y + r <= h
        ]

        # ── Non-max suppression — collapse duplicate/overlapping detections ────
        # Hough Circle Transform frequently fits SEVERAL overlapping circles to
        # the same physical bubble (especially at larger radii, where a bigger
        # circle's longer perimeter accumulates more votes even when the fit is
        # weaker per-pixel than a smaller, tighter circle nearby). Left
        # unfiltered, this is exactly what inflates bubble_count and
        # surface_coverage_pct far past reality — the sum of all raw circle
        # areas can be 10x+ the image area even though only a fraction of the
        # pixels are actually bubble. We de-duplicate by processing smallest
        # circles first (a tight, well-localized fit is stronger evidence of a
        # real single bubble than a large loose one) and rejecting any larger
        # candidate whose center falls within the combined-radius neighborhood
        # of a circle already kept.
        valid.sort(key=lambda c: c[2])  # smallest radius first
        kept: List[Tuple[float, float, float]] = []
        for x, y, r in valid:
            is_duplicate = False
            for kx, ky, kr in kept:
                dist = ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5
                if dist < 0.5 * (r + kr):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append((x, y, r))
        hough_bubbles = kept

        # ── Second, independent candidate generator (contour/blob-based) ─────
        # Hough votes for circular edges; this instead finds connected regions
        # in the same thresholded map and fits an enclosing circle to each.
        # It's a genuinely different detection mechanism (not the same method
        # renamed), so it catches some real bubbles Hough's voting scheme
        # misses — merged in below, not used alone.
        contour_bubbles = self._contour_candidates(blurred, sensitivity)
        merged = self._merge_bubble_lists(hough_bubbles, contour_bubbles)

        # ── Multi-cue confidence score, replacing a chain of hard AND'd ───────
        # thresholds with one weighted composite. Cues: shape (circularity of
        # a locally re-derived contour), size plausibility (vs. the paper's
        # 0.1-1.1mm range), local contrast MAGNITUDE (checked empirically:
        # direction — dark-center vs bright-center — was NOT reliable, split
        # roughly 9/15 vs 6/15 on real detections, so only magnitude is used),
        # and edge strength (mean gradient sampled around the boundary).
        # Weights are hand-set and documented as a placeholder for what should
        # eventually be fit with logistic regression against human-corrected
        # labels — i.e. exactly the Sprint-7 ML classifier already planned,
        # just moved to candidate-level features instead of whole-image ones.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        score_thresh = {"low": 0.55, "medium": 0.45, "high": 0.35}.get(sensitivity, 0.45)
        scored_bubbles = []
        self.last_scores = {}
        for x, y, r in merged:
            p_bubble, components = self._score_candidate(gray, grad_mag, x, y, r)
            if p_bubble >= score_thresh:
                scored_bubbles.append((x, y, r))
                self.last_scores[(x, y, r)] = p_bubble

        return scored_bubbles

    @staticmethod
    def _contour_candidates(binary_or_gray, sensitivity):
        """Second candidate generator: connected-component contours on the
        same preprocessed map Hough uses, each fit to a minimum enclosing
        circle. Independent failure mode from Hough's circular voting."""
        _, binary = cv2.threshold(binary_or_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minR = {"low": 9, "medium": 6, "high": 5}.get(sensitivity, 6)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < np.pi * minR * minR * 0.5:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < minR or r > 110:
                continue
            out.append((float(x), float(y), float(r)))
        return out

    @staticmethod
    def _score_candidate(gray, grad_mag, x, y, r, mm_per_px: float = MM_PER_PIXEL):
        """Weighted multi-cue plausibility score in [0, 1]. See the note in
        _detect_bubbles_single_pass on where the weights come from and how
        they should eventually be replaced with fitted ones."""
        H, W = gray.shape
        xi, yi, ri = int(round(x)), int(round(y)), max(1, int(round(r)))

        # shape: circularity of a locally re-derived contour (robust to
        # whole-image segmentation artifacts since it's re-thresholded in a
        # small patch around just this candidate)
        pad = int(ri * 1.6)
        x0, x1 = max(0, xi - pad), min(W, xi + pad)
        y0, y1 = max(0, yi - pad), min(H, yi + pad)
        patch = gray[y0:y1, x0:x1]
        shape_score = 0.0
        if patch.size > 0:
            _, local_bin = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(local_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best = min(contours, key=lambda c: abs(cv2.contourArea(c) - np.pi * ri * ri))
                area = cv2.contourArea(best)
                perim = cv2.arcLength(best, True)
                if perim > 0 and area > 0:
                    shape_score = float(np.clip(4 * np.pi * area / (perim * perim), 0, 1))

        # size: bell-curve plausibility vs the paper's 0.1-1.1mm range
        diam_mm = 2 * ri * mm_per_px
        size_score = float(np.exp(-((diam_mm - 0.5) ** 2) / (2 * 0.45 ** 2)))

        # contrast: |inner - outer| MAGNITUDE only (direction tested unreliable)
        mask_in = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask_in, (xi, yi), max(1, int(ri * 0.5)), 255, -1)
        mask_out = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask_out, (xi, yi), int(ri * 1.4), 255, -1)
        cv2.circle(mask_out, (xi, yi), ri, 0, -1)
        inner = gray[mask_in == 255].mean() if (mask_in == 255).any() else 128.0
        outer = gray[mask_out == 255].mean() if (mask_out == 255).any() else 128.0
        contrast_score = float(np.clip(abs(inner - outer) / 40.0, 0, 1))

        # edge: mean gradient magnitude sampled around the boundary ring
        n = 24
        vals = []
        for i in range(n):
            th = 2 * np.pi * i / n
            px, py = int(xi + ri * np.cos(th)), int(yi + ri * np.sin(th))
            if 0 <= px < W and 0 <= py < H:
                vals.append(grad_mag[py, px])
        edge_score = float(np.clip((np.mean(vals) if vals else 0.0) / 80.0, 0, 1))

        weights = {"shape": 0.35, "size": 0.25, "contrast": 0.15, "edge": 0.25}
        components = {"shape": shape_score, "size": size_score,
                      "contrast": contrast_score, "edge": edge_score}
        p_bubble = sum(weights[k] * components[k] for k in weights)
        return float(np.clip(p_bubble, 0, 1)), components

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
    def _make_overlay(img: np.ndarray, bubbles: List[Tuple], scores: Optional[Dict] = None) -> str:
        for x, y, r in bubbles:
            if scores is not None:
                p = scores.get((x, y, r), 0.5)
                # green (confident) -> yellow -> red (marginal), BGR order
                color = (0, int(255 * min(1, p * 2)), int(255 * min(1, (1 - p) * 2)))
            else:
                color = (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), int(r), color, 1)
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
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
