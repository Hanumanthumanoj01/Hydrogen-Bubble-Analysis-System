"""
fisher_image.py
===============
Library module, not run directly: imported by main.py (Histogram / Fisher
tab). Quick test:  python3 -c "import fisher_image; print('ok')"

Everything that can be measured from ONE single picture:

    * the brightness histogram
    * Shannon entropy of that histogram
    * Fisher information of that histogram  (the 1-D, distribution version)
    * the spatial Fisher maps F_x and F_y   (the 2-D, image version)
    * a table of the strongest peaks, with a note on what each one sits on

This is the single-image companion to fisher_report.py, which does the same
for a recording and adds the time derivative.

THE TWO FISHER INFORMATIONS ARE NOT THE SAME THING
---------------------------------------------------
This is the point most easily muddled, so it is worth being explicit.

1.  Fisher information OF THE HISTOGRAM.  Treat the brightness histogram as a
    probability density p(I) over brightness I.  Then

        J = integral of (dp/dI)^2 / p(I)  dI

    This is one number for the whole picture, and it is the natural partner of
    the Shannon entropy of the same histogram.  Entropy asks how SPREAD OUT the
    histogram is; this asks how SHARPLY PEAKED it is — how fast the density
    changes as you move along the brightness axis.  A picture with a few clean
    brightness populations (dark bubbles, bright plate) has sharp histogram
    edges and a high J.  A washed-out picture has a smooth histogram and a low
    J.  The two numbers answer different questions about the same histogram,
    which is why both are reported.

2.  Fisher information OF THE IMAGE, spatially.  Here the derivative is taken
    against POSITION rather than against brightness:

        F_x(x,y) = (dI/dx)^2 / I        F_y(x,y) = (dI/dy)^2 / I

    These are maps, not numbers.  They say WHERE in the field the brightness is
    changing quickly, which on this material means where the boundaries are.

WHY X AND Y ARE KEPT SEPARATE
-----------------------------
They are not redundant.  A bubble is round, so its rim contributes to both: the
left and right of the rim appear in F_x, the top and bottom in F_y.  A machining
scratch on the plate runs in one direction only, so it appears strongly in one
and weakly in the other.  The ratio F_x / F_y is therefore a way of telling a
round object from a straight streak without detecting anything.  A frame full of
bubbles tends toward a ratio near 1; a frame dominated by machining marks does
not.

WHAT A PEAK MEANS, AND WHEN IT MEANS NOTHING
--------------------------------------------
A peak in F is a pixel whose brightness changes sharply with position.  On this
material that is a bubble rim, the edge of a bore, or a scratch.

The division by I matters and is the reason a raw gradient map is not the same
thing.  Dividing by the brightness makes the quantity RELATIVE: a step of ten
grey levels on a dark background counts for far more than the same step on a
bright one.  That is what is wanted physically, because the camera's noise also
grows with brightness.

The trap is that dividing by a very small I inflates noise in the darkest
pixels.  Every peak is therefore reported with two sanity checks: its height
relative to the frame's median F, and whether it lies on a genuine brightness
edge.  A tall peak that is NOT on an edge is the division by a small number,
not a physical boundary, and should be ignored.
"""

import base64
import io
import json

import numpy as np
import cv2


EPS = 1e-6


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------

def to_grey(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def burnt_in_text_mask(gray, thresh=235):
    """
    The cameras in this laboratory burn a bright timestamp into the corner of
    every frame. Its edges are the sharpest thing in the picture, so without
    excluding them they dominate every Fisher map and every peak table.

    On a single still there is no way to find them by their flicker, as the
    video version does, so they are found by what they are: a small, very
    bright, high-contrast region in the top portion of the frame. The mask is
    dilated so the character edges themselves are covered, not only their
    centres.
    """
    h, w = gray.shape
    band = np.zeros_like(gray, np.uint8)
    band[: int(0.18 * h), :] = 255
    bright = ((gray >= thresh).astype(np.uint8) * 255) & band
    if bright.sum() == 0:
        return np.zeros_like(gray, bool)
    m = cv2.dilate(bright, np.ones((9, 9), np.uint8), iterations=2)
    # Only keep it if it is a small fraction of the frame; a genuinely
    # bright image should not have its top fifth masked away.
    if (m > 0).mean() > 0.08:
        return np.zeros_like(gray, bool)
    return m > 0


# ---------------------------------------------------------------------------
# 1. the histogram, and the two numbers that come from it
# ---------------------------------------------------------------------------

def histogram(gray, mask=None, bins=256):
    v = gray[~mask] if mask is not None else gray.ravel()
    counts, _ = np.histogram(v, bins=bins, range=(0, 256))
    p = counts.astype(np.float64)
    p /= max(p.sum(), 1.0)
    return counts, p


def shannon_entropy(p):
    """H = -sum p ln p, in nats, with H_norm against a perfectly flat histogram."""
    nz = p[p > 0]
    H = float(-np.sum(nz * np.log(nz)))
    return H, H / np.log(len(p))


def fisher_of_histogram(p, smooth=2.0):
    """
    Fisher information of the brightness distribution itself.

        J = sum over bins of (dp/dI)^2 / p

    The histogram is smoothed first, because the derivative of a raw 256-bin
    histogram is dominated by bin-to-bin counting noise rather than by the
    shape of the distribution. The smoothing width is reported alongside the
    value so the number is reproducible; it is not a free parameter to tune
    until the answer looks good.

    Bins where p is essentially zero are skipped. Without that, empty bins beside
    occupied ones send the ratio to infinity and the whole sum becomes
    meaningless.
    """
    ps = cv2.GaussianBlur(p.astype(np.float32).reshape(-1, 1), (0, 0), smooth).ravel()
    ps = ps / max(ps.sum(), EPS)
    d = np.gradient(ps)
    ok = ps > 1e-8
    J = float(np.sum(d[ok] ** 2 / ps[ok]))
    return J, float(smooth)


# ---------------------------------------------------------------------------
# 2. the spatial maps
# ---------------------------------------------------------------------------

def fisher_spatial(gray, mask=None, floor_pct=5.0):
    """
    F_x and F_y over the picture.

    The brightness floor matters. F divides by I, so a pixel at I = 1 multiplies
    its gradient by a thousand. Clamping I at a low percentile of the frame
    stops the darkest corner of the image from producing the largest "Fisher
    information" in it, which is a numerical artefact and not a measurement.
    """
    I = gray.astype(np.float64)
    floor = max(float(np.percentile(I, floor_pct)), 1.0)
    Ic = np.maximum(I, floor)

    gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3) / 8.0

    Fx = gx ** 2 / Ic
    Fy = gy ** 2 / Ic

    if mask is not None:
        Fx[mask] = 0.0
        Fy[mask] = 0.0
    return Fx, Fy, floor


def anisotropy(Fx, Fy, mask=None):
    """
    How much more the brightness runs across the picture than down it.

    Reported as the ratio of the two totals. Near 1 means the structures are
    round or evenly oriented, which is what a field of bubbles gives. Far from 1
    means one direction dominates, which is what machining marks give.
    """
    if mask is not None:
        Fx, Fy = Fx[~mask], Fy[~mask]
    sx, sy = float(Fx.sum()), float(Fy.sum())
    return sx, sy, (sx / sy if sy > EPS else float("nan"))


def edge_flat_ratio(F, gray, mask=None):
    """
    How much more Fisher information sits on brightness edges than on flat areas.

    This is the single number that says whether a map means anything. Edges are
    found with Canny, entirely independently of F. A ratio near 1 says the map is
    spread evenly over the picture, which is what noise looks like. Well above 1
    says the map has found the boundaries.
    """
    g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(g8, (0, 0), 1.2), 50, 130) > 0
    if mask is not None:
        edges &= ~mask
        flat = (~edges) & (~mask)
    else:
        flat = ~edges
    if edges.sum() < 50 or flat.sum() < 50:
        return float("nan"), int(edges.sum())
    return float(F[edges].mean() / max(F[flat].mean(), EPS)), int(edges.sum())


# ---------------------------------------------------------------------------
# 3. peaks
# ---------------------------------------------------------------------------

def find_peaks(F, gray, n=20, min_sep=14, mask=None):
    """
    The strongest places in the map, with enough context to judge them.

    Peaks are taken greedily, each one suppressing a neighbourhood, so the list
    is spread over the picture instead of being twenty pixels of one rim.

    Each peak carries:
      height_vs_median  how many times the frame's median F it is. This is the
                        significance: a peak at 3x the median is unremarkable,
                        one at 300x is a genuine boundary.
      on_edge           whether an independently computed Canny edge passes
                        within a few pixels. A tall peak that is not on an edge
                        is almost always the division by a small brightness.
    """
    Fw = F.copy()
    if mask is not None:
        Fw[mask] = 0.0

    med = float(np.median(Fw[Fw > 0])) if np.any(Fw > 0) else EPS
    g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(g8, (0, 0), 1.2), 50, 130) > 0
    edge_near = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    out = []
    work = Fw.copy()
    h, w = work.shape
    for _ in range(n):
        idx = int(np.argmax(work))
        y, x = divmod(idx, w)
        v = float(work[y, x])
        if v <= 0:
            break
        out.append({
            "x": int(x),
            "y": int(y),
            "F": v,
            "height_vs_median": round(v / max(med, EPS), 1),
            "brightness": int(gray[y, x]),
            "on_edge": bool(edge_near[y, x]),
        })
        y0, y1 = max(0, y - min_sep), min(h, y + min_sep + 1)
        x0, x1 = max(0, x - min_sep), min(w, x + min_sep + 1)
        work[y0:y1, x0:x1] = 0.0
    return out, med


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


def colour_map(F, pct=99.3):
    """Fisher maps are extremely long-tailed; clipping at a high percentile is
    what makes the structure visible instead of one white dot on black."""
    hi = float(np.percentile(F, pct))
    v = np.clip(F / max(hi, EPS), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def histogram_png(counts, H, Hn, J, width=760, height=320):
    """The histogram drawn with OpenCV, so the backend needs no plotting stack."""
    img = np.full((height, width), 22, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pad_l, pad_b, pad_t = 56, 38, 34
    pw, ph = width - pad_l - 14, height - pad_b - pad_t

    mx = max(int(counts.max()), 1)
    for i, c in enumerate(counts):
        x = pad_l + int(i * pw / len(counts))
        bw = max(1, int(pw / len(counts)))
        bh = int(c / mx * ph)
        cv2.rectangle(img, (x, height - pad_b), (x + bw, height - pad_b - bh),
                      (235, 170, 95), -1)

    cv2.line(img, (pad_l, height - pad_b), (pad_l + pw, height - pad_b), (150, 150, 150), 1)
    cv2.line(img, (pad_l, pad_t), (pad_l, height - pad_b), (150, 150, 150), 1)
    for frac, lab in ((0.0, "0"), (0.25, "64"), (0.5, "128"), (0.75, "192"), (1.0, "255")):
        x = pad_l + int(frac * pw)
        cv2.line(img, (x, height - pad_b), (x, height - pad_b + 4), (150, 150, 150), 1)
        cv2.putText(img, lab, (x - 10, height - pad_b + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(img, "brightness", (pad_l + pw // 2 - 34, height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(img, "pixels", (6, pad_t - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(img, f"H = {H:.4f} nats   ({Hn*100:.2f} %)   Fisher J = {J:.3e}",
                (pad_l, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 235, 140), 1, cv2.LINE_AA)
    return img


def draw_peaks(gray, peaks, top=12):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = max(1, min(gray.shape) // 320)
    for i, p in enumerate(peaks[:top], 1):
        col = (90, 255, 90) if p["on_edge"] else (90, 160, 255)
        cv2.circle(vis, (p["x"], p["y"]), max(7, t * 7), col, max(1, t), cv2.LINE_AA)
        cv2.putText(vis, str(i), (p["x"] + 9, p["y"] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * max(1, t * 0.9), col,
                    max(1, t), cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# the one call the API makes
# ---------------------------------------------------------------------------

def analyse_image(img_bgr, n_peaks=20, hist_smooth=2.0, mask_timestamp=True):
    gray = to_grey(img_bgr)
    mask = burnt_in_text_mask(gray) if mask_timestamp else np.zeros(gray.shape, bool)

    counts, p = histogram(gray, mask)
    H, Hn = shannon_entropy(p)
    J, sm = fisher_of_histogram(p, hist_smooth)

    Fx, Fy, floor = fisher_spatial(gray, mask)
    F = Fx + Fy
    sx, sy, ratio = anisotropy(Fx, Fy, mask)
    ef, n_edge = edge_flat_ratio(F, gray, mask)
    peaks, med = find_peaks(F, gray, n=n_peaks, mask=mask)

    return {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "masked_pixels": int(mask.sum()),

        "shannon_entropy": round(H, 4),
        "shannon_entropy_norm": round(Hn * 100, 2),
        "fisher_histogram_J": float(f"{J:.6g}"),
        "histogram_smoothing": sm,

        "fisher_x_total": float(f"{sx:.6g}"),
        "fisher_y_total": float(f"{sy:.6g}"),
        "fisher_xy_ratio": round(ratio, 3) if np.isfinite(ratio) else None,
        "fisher_mean": float(f"{float(F.mean()):.6g}"),
        "fisher_median": float(f"{med:.6g}"),
        "edge_flat_ratio": round(ef, 2) if np.isfinite(ef) else None,
        "edge_pixels": n_edge,
        "brightness_floor": round(floor, 1),

        "histogram_counts": [int(c) for c in counts],
        "peaks": peaks,

        "histogram_image": _png_b64(histogram_png(counts, H, Hn, J)),
        "fisher_combined_image": _png_b64(colour_map(F)),
        "fisher_x_image": _png_b64(colour_map(Fx)),
        "fisher_y_image": _png_b64(colour_map(Fy)),
        "peaks_image": _png_b64(draw_peaks(gray, peaks)),
    }


if __name__ == "__main__":
    import sys
    im = cv2.imread(sys.argv[1])
    if im is None:
        raise SystemExit("cannot read " + sys.argv[1])
    r = analyse_image(im)
    for k, v in r.items():
        if k.endswith("_image") or k == "histogram_counts":
            continue
        if k == "peaks":
            print("peaks (top 8):")
            for i, pk in enumerate(v[:8], 1):
                print(f"   {i:>2}. ({pk['x']:>4},{pk['y']:>4})  "
                      f"{pk['height_vs_median']:>8} x median  "
                      f"I={pk['brightness']:>3}  "
                      f"{'on edge' if pk['on_edge'] else 'NOT on edge'}")
            continue
        print(f"{k:>24}: {v}")
