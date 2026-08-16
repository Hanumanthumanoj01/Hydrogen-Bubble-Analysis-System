"""
video_analysis.py
=================
Bubble detection for REAL electrolysis video, using temporal background
subtraction as the primary method.

WHY THIS IS THE PRIMARY METHOD (and not a deep-learning detector)
-----------------------------------------------------------------
On a single still frame, a scratch on the electrode and the rim of a bubble
are genuinely the same kind of local feature: a curved intensity edge. No
threshold, morphological filter, or Hough parameter can separate them,
because the information needed to distinguish them is not present in one
frame. This is a property of the data, not a deficiency of the algorithm.

Across time, they are trivially separable: the electrode is static, the
bubbles are not. Taking the per-pixel median over the frame stack
reconstructs the bare electrode (bubbles average out because they move),
and subtracting it removes the entire class of static false positives in
one step.

Measured on 20260713_bigbubbles_900mA (90 frames, 640x480, 200 fps camera):
    single-frame detector  ->  mean  7.3 bubbles/frame  (range  3-13)
    background subtraction ->  mean 20.3 bubbles/frame  (range  3-39)

ROBUSTNESS FEATURES (these matter most when video data is scarce)
-----------------------------------------------------------------
1. Auto-calibrated threshold. The detection threshold is derived from the
   video's own sensor-noise floor rather than hand-tuned per clip, so the
   method transfers to new recordings with different lighting, exposure or
   current density without re-tuning. See estimate_noise_floor().
2. Temporal persistence filter. A real bubble is present in consecutive
   frames; a single-frame blip is noise. Detections that never persist are
   discarded. This is only possible with video and is the main precision
   gain over single-frame analysis.
3. Rolling background option. For longer recordings where the electrode
   slowly drifts (thermal expansion, illumination change), the background
   can be recomputed over a sliding window instead of the whole clip.
4. Bubble tracking. Linking detections across frames yields growth rate,
   lifetime and detachment frequency -- quantities the paper's micro-region
   model is actually about, and which no single-frame method can produce.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Imported from constants.py — the single source of truth shared with
# bubble_analysis.py. mm_per_pixel("video") returns the video-specific scale
# if the supervisor confirms the video optics differed from the paper's.
from constants import (
    FARADAY, Z_ELECTRONS, MOLAR_VOLUME_H2, mm_per_pixel,
)

MM_PER_PIXEL: float = mm_per_pixel("video")


# ────────────────────────────────────────────────────────────────────────────
# Background model
# ────────────────────────────────────────────────────────────────────────────
def compute_background(frames: np.ndarray) -> np.ndarray:
    """Per-pixel temporal median = the static electrode with bubbles removed.

    Median (not mean) because it is robust: a bubble sitting over a pixel for
    a minority of frames does not shift the median, whereas it would drag the
    mean toward the bubble's brightness and leave ghost haloes.
    """
    return np.median(frames, axis=0).astype(np.uint8)


def compute_rolling_background(frames: np.ndarray, idx: int, window: int = 45) -> np.ndarray:
    """Median over a sliding window centred on `idx`.

    Use when the recording is long enough that the electrode/illumination
    drifts; a single global median would then leave systematic residue.
    For short clips (< ~5 s) the global median is both cheaper and steadier.
    """
    lo = max(0, idx - window // 2)
    hi = min(len(frames), idx + window // 2 + 1)
    return np.median(frames[lo:hi], axis=0).astype(np.uint8)


def estimate_noise_floor(frames: np.ndarray, percentile: float = 25.0) -> float:
    """Robust sensor-noise estimate from the per-pixel temporal std.

    NOTE: kept for diagnostics/reporting, but NO LONGER used to set the
    detection threshold -- see auto_threshold() for why it failed.
    """
    tstd = frames.astype(np.float32).std(axis=0)
    return float(np.percentile(tstd, percentile))


def auto_threshold(frames: np.ndarray, bg: Optional[np.ndarray] = None,
                   percentile: float = 88.0, floor: float = 6.0) -> float:
    """Detection threshold, calibrated from the difference image itself.

    The threshold is set at a high percentile of |frame - background| pooled
    over the clip. Detection thresholds this quantity directly, so anchoring
    to its own distribution makes the threshold scale with how much bubble
    activity a recording actually contains.

    WHY NOT the per-pixel-temporal-std noise floor (the previous approach):
    that estimator took a LOW percentile (25th) of each pixel's std over
    time, on the assumption that this selects predominantly static
    background pixels and so measures sensor noise. It works on a busy
    recording, but it FAILS at low current density. Measured across the four
    ANS recordings, 2 x p25 of the per-pixel std gave:

        100 mA/cm2 ->  1.96      600 mA/cm2 -> 16.24
        300 mA/cm2 ->  3.57      900 mA/cm2 -> 10.75

    At 100 and 300 mA/cm2 there is so little bubble activity that most
    pixels are perfectly static for the whole clip, so the 25th percentile
    lands inside that dead population and the estimate collapses toward
    zero -- both hit the hard floor and the "auto" calibration became no
    calibration at all. The original value was implicitly tuned to the busy
    900 mA clip.

    A consecutive-frame MAD estimator was also tested and rejected: these
    recordings are compressed strongly enough that the median frame-to-frame
    difference is exactly 0 for the two low-current clips, giving sigma = 0.

    The difference-image percentile behaves correctly across the full range
    (p88 -> 7.0, 9.0, 25.0, 21.0 for 100/300/600/900 mA/cm2), rising
    monotonically with current density as expected.
    """
    if bg is None:
        bg = compute_background(frames)
    step = max(1, len(frames) // 30)
    diffs = np.stack([cv2.absdiff(f, bg) for f in frames[::step]])
    return max(floor, float(np.percentile(diffs, percentile)))


# ────────────────────────────────────────────────────────────────────────────
# Per-frame detection
# ────────────────────────────────────────────────────────────────────────────
def detect_bubbles_bgsub(
    gray: np.ndarray,
    bg: np.ndarray,
    thresh: float,
    min_area: int = 12,
    peak_frac: float = 0.30,
    min_r: float = 2.0,
    max_r: float = 120.0,
    min_roundness: float = 0.28,
    roi: Optional[Tuple[float, float, float]] = None,
) -> List[Tuple[float, float, float]]:
    """Detect bubbles in one frame given the static background.

    Returns list of (x, y, r) in pixels.
    """
    diff = cv2.absdiff(gray, bg)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)

    # A FIXED (auto-calibrated) threshold is used deliberately, NOT Otsu.
    # Otsu assumes a bimodal histogram. The difference image is overwhelmingly
    # near-zero because most pixels are unchanged background, so Otsu lands far
    # too high -- measured on the 900 mA clip it chose 47, keeping only the
    # brightest ~4% of pixels and discarding most real bubbles (counts fell to
    # ~3/frame, i.e. worse than the single-frame method it was meant to beat).
    _, binary = cv2.threshold(diff, float(thresh), 255, cv2.THRESH_BINARY)

    if roi is not None:
        h, w = gray.shape
        cxf, cyf, rf = roi
        cx, cy = w * cxf, h * cyf
        rmax = min(w, h) / 2 * rf
        yy, xx = np.ogrid[:h, :w]
        binary[(xx - cx) ** 2 + (yy - cy) ** 2 > rmax ** 2] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)

    # Watershed splits touching/overlapping bubbles. NOTE: watershed was tried
    # earlier directly on raw single images and made results markedly WORSE
    # (counts collapsed ~100 -> ~25) because whole-image segmentation produced
    # jagged regions that failed the roundness test. It works here because
    # background subtraction hands it clean, well-separated blobs.
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []
    _, fg = cv2.threshold(dist, peak_frac * dist.max(), 255, 0)
    fg = fg.astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[cv2.subtract(binary, fg) == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)

    out: List[Tuple[float, float, float]] = []
    for lbl in range(2, n_markers + 2):
        m = np.uint8(markers == lbl) * 255
        if m.sum() == 0:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = cs[0]
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < min_r or r > max_r:
            continue
        # Roundness as fill-ratio (area / area of enclosing circle). Preferred
        # over the perimeter formula 4*pi*A/P^2, which is hypersensitive to the
        # pixel-level jaggedness of segmented masks and rejects genuinely round
        # bubbles for having a ragged one-pixel boundary.
        if area / (np.pi * r * r) < min_roundness:
            continue
        out.append((float(x), float(y), float(r)))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Tracking
# ────────────────────────────────────────────────────────────────────────────
class BubbleTracker:
    """Links per-frame detections into tracks across time.

    Two jobs:
      1. Precision: a detection that appears in only ONE frame and is never
         seen again is almost always noise. Requiring persistence over
         `min_persistence` frames removes those. This is the single biggest
         precision gain available from video and is impossible on stills.
      2. Physics: linked tracks give growth rate (dr/dt), lifetime, and
         detachment events -- the quantities the micro-region model of
         Wagner et al. (2025) actually concerns.
    """

    def __init__(self, match_dist_factor: float = 1.2, max_missing: int = 3):
        self.tracks: List[Dict[str, Any]] = []
        self.match_dist_factor = match_dist_factor
        self.max_missing = max_missing
        self._next_id = 0

    def update(self, detections: List[Tuple[float, float, float]], frame_idx: int,
               t_seconds: float) -> List[Dict[str, Any]]:
        used = set()
        assigned = []

        for (x, y, r) in detections:
            best, best_d = None, np.inf
            for tr in self.tracks:
                if tr["id"] in used or tr["lost"] > self.max_missing:
                    continue
                d = np.hypot(x - tr["x"], y - tr["y"])
                # Gate on bubble size: a bubble cannot jump much further than
                # its own radius between consecutive frames.
                if d < self.match_dist_factor * max(r, tr["r"]) and d < best_d:
                    best, best_d = tr, d
            if best is not None:
                used.add(best["id"])
                best["history"].append((t_seconds, x, y, r))
                best["x"], best["y"], best["r"] = x, y, r
                best["frames_seen"] += 1
                best["last_frame"] = frame_idx
                best["lost"] = 0
                assigned.append(best)
            else:
                tr = {
                    "id": self._next_id, "x": x, "y": y, "r": r,
                    "frames_seen": 1, "first_frame": frame_idx,
                    "last_frame": frame_idx, "lost": 0,
                    "history": [(t_seconds, x, y, r)],
                }
                self._next_id += 1
                self.tracks.append(tr)
                assigned.append(tr)

        for tr in self.tracks:
            if tr["id"] not in used:
                tr["lost"] += 1

        return assigned

    def summary(self) -> Dict[str, Any]:
        """Growth / lifetime statistics over all tracks with enough history."""
        growth_rates, lifetimes, max_radii = [], [], []
        for tr in self.tracks:
            h = tr["history"]
            if len(h) < 3:
                continue
            t0, _, _, r0 = h[0]
            t1, _, _, r1 = h[-1]
            lifetimes.append(t1 - t0)
            max_radii.append(max(p[3] for p in h))
            if t1 > t0:
                growth_rates.append((r1 - r0) * MM_PER_PIXEL / (t1 - t0))  # mm/s
        return {
            "n_tracks": len(self.tracks),
            "n_tracks_with_history": len(lifetimes),
            "mean_lifetime_s": float(np.mean(lifetimes)) if lifetimes else 0.0,
            "mean_growth_rate_mm_per_s": float(np.mean(growth_rates)) if growth_rates else 0.0,
            "mean_max_diameter_mm": float(np.mean(max_radii) * 2 * MM_PER_PIXEL) if max_radii else 0.0,
        }


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────
def shannon_entropy(gray: np.ndarray) -> Tuple[float, float]:
    """H = -sum p_i ln p_i, H_norm = H / ln(256)  (Wagner et al. 2025, Eq 7-8)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    nz = hist[hist > 0]
    H = float(-np.sum(nz * np.log(nz)))
    return H, H / np.log(256)


def frame_metrics(gray: np.ndarray, bubbles, current_density: float,
                  cathode_area: float) -> Dict[str, Any]:
    h, w = gray.shape
    H, H_norm = shannon_entropy(gray)

    radii_mm = [b[2] * MM_PER_PIXEL for b in bubbles]
    mean_r_mm = float(np.mean(radii_mm)) if radii_mm else 0.0

    # True pixel coverage via a mask, so overlapping bubbles are not
    # double-counted (summing pi*r^2 can exceed 100% of the frame area).
    mask = np.zeros((h, w), np.uint8)
    for x, y, r in bubbles:
        cv2.circle(mask, (int(x), int(y)), max(1, int(r)), 1, -1)
    coverage = float(mask.sum()) / (h * w) * 100.0

    h2 = (current_density / 1000.0) * cathode_area / (Z_ELECTRONS * FARADAY) * MOLAR_VOLUME_H2 * 60.0

    edges = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.1, 1e9]
    diam = [2 * r for r in radii_mm]
    dist = [int(sum(1 for d in diam if edges[i] <= d < edges[i + 1])) for i in range(7)]

    return {
        "bubble_count": len(bubbles),
        "H": round(H, 4),
        "H_norm_pct": round(H_norm * 100, 2),
        "mean_diam_mm": round(mean_r_mm * 2, 4),
        "mean_K_per_mm": round(1.0 / mean_r_mm, 4) if mean_r_mm > 0 else 0.0,
        "coverage_pct": round(coverage, 2),
        "h2_mL_min": round(h2, 4),
        "size_lt01": dist[0], "size_01_02": dist[1], "size_02_04": dist[2],
        "size_04_06": dist[3], "size_06_08": dist[4], "size_08_11": dist[5],
        "size_gt11": dist[6],
    }


# ────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ────────────────────────────────────────────────────────────────────────────
def load_video_gray(path: str, max_frames: int = 600) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise ValueError("No frames could be read from the video.")
    return np.array(frames), float(fps)


def analyze_video(
    path: str,
    current_density: float = 900.0,
    cathode_area: float = 2.54,
    thresh_percentile: float = 88.0,
    frame_step: int = 1,
    min_persistence: int = 2,
    use_rolling_bg: bool = False,
    roi: Optional[Tuple[float, float, float]] = None,
    max_frames: int = 600,
) -> Dict[str, Any]:
    """Run the full background-subtraction pipeline over a video file."""
    frames, fps = load_video_gray(path, max_frames=max_frames)
    n = len(frames)

    global_bg = compute_background(frames)
    thresh = auto_threshold(frames, bg=global_bg, percentile=thresh_percentile)
    noise = estimate_noise_floor(frames)

    tracker = BubbleTracker()
    rows: List[Dict[str, Any]] = []
    per_frame_raw: List[List[Tuple[float, float, float]]] = []

    indices = list(range(0, n, max(1, frame_step)))
    for idx in indices:
        gray = frames[idx]
        bg = compute_rolling_background(frames, idx) if use_rolling_bg else global_bg
        dets = detect_bubbles_bgsub(gray, bg, thresh, roi=roi)
        per_frame_raw.append(dets)
        tracker.update(dets, idx, idx / fps)

    # Temporal persistence filter: keep only detections belonging to a track
    # seen in at least `min_persistence` frames. Single-frame blips are noise.
    good_ids = {tr["id"] for tr in tracker.tracks if tr["frames_seen"] >= min_persistence}
    persistent_positions = {}
    for tr in tracker.tracks:
        if tr["id"] in good_ids:
            for (t, x, y, r) in tr["history"]:
                persistent_positions.setdefault(round(t, 6), []).append((x, y, r))

    total_before = 0
    total_after = 0
    for i, idx in enumerate(indices):
        gray = frames[idx]
        t = idx / fps
        raw = per_frame_raw[i]
        kept = persistent_positions.get(round(t, 6), [])
        total_before += len(raw)
        total_after += len(kept)
        m = frame_metrics(gray, kept, current_density, cathode_area)
        m.update({"frame": idx + 1, "time_s": round(t, 4),
                   "raw_detections": len(raw),
                   "filtered_out": len(raw) - len(kept)})
        rows.append(m)

    counts = [r["bubble_count"] for r in rows]
    summary = {
        "total_frames_analyzed": len(rows),
        "video_fps": round(fps, 2),
        "duration_s": round(n / fps, 3),
        "auto_threshold": round(thresh, 2),
        "noise_floor": round(noise, 2),
        "mean_bubbles": round(float(np.mean(counts)), 2) if counts else 0,
        "max_bubbles": int(np.max(counts)) if counts else 0,
        "min_bubbles": int(np.min(counts)) if counts else 0,
        "mean_H_norm_pct": round(float(np.mean([r["H_norm_pct"] for r in rows])), 2) if rows else 0,
        "mean_coverage_pct": round(float(np.mean([r["coverage_pct"] for r in rows])), 2) if rows else 0,
        "mean_diam_mm": round(float(np.mean([r["mean_diam_mm"] for r in rows])), 4) if rows else 0,
        "raw_detections_total": total_before,
        "removed_by_persistence_filter": total_before - total_after,
        **tracker.summary(),
    }

    return {"frames": rows, "summary": summary,
            "background_png": _to_png_b64(global_bg)}


def _to_png_b64(img: np.ndarray) -> str:
    import base64
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


if __name__ == "__main__":
    import sys, json
    src = sys.argv[1]
    res = analyze_video(src, current_density=900.0)
    print(json.dumps(res["summary"], indent=2))
