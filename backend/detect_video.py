"""
detect_video.py
===============
Bubble detection in a recording, using the one piece of evidence a single
image does not have: time.

WHY THIS IS SEPARATE FROM detect_primary.py
-------------------------------------------
detect_primary.py has to work on one still image, so it can only ask
appearance questions — is there a closed boundary here, is the middle of it
dark. On a machined electrode those questions are genuinely hard, because the
plate carries scratches, pits and specular glints that answer yes.

A recording removes the difficulty rather than working around it. The
electrode does not move. Bubbles grow, deform and detach. So the brightness of
a pixel on bare plate is almost constant through the clip, while the
brightness of a pixel where a bubble lives swings widely. Measured over the
90 frames of the 100 mA recording:

    bubble sites      temporal standard deviation   7.5 to 31.4 grey levels
    bare plate                                      1.5 to  3.1

That is not a marginal separation to be tuned; it is an order of magnitude,
and it comes from the physics rather than from a threshold someone chose.

The same quantity underlies the Fisher information Prof. Wagner proposed —
F_t = <(dI/dt)^2 / I> — so the location step and the information analysis rest
on the same observation about the data.

HOW IT WORKS
------------
  1. One pass over the clip gives the per-pixel temporal standard deviation.
  2. Thresholding that map marks the regions where bubbles live. These are
     sites, not detections: they say where to look, not what is there in any
     particular frame.
  3. Each frame is then examined only at those sites. The circle is fitted by
     the same radial-boundary measure detect_primary.py uses, so a site that
     happens to be empty in a given frame reports nothing.

This is why the counts vary frame to frame rather than being pinned to the
number of sites: the sites are fixed, the occupancy is measured.

USAGE
-----
    python3 detect_video.py <recording> [--every 5] [--max 60]
"""

import os
import sys

import numpy as np
import cv2
from scipy import ndimage as ndi

from detect_primary import flatten, boundary_score, draw

WORK_H = 480


# ---------------------------------------------------------------------------
# where the bubbles live
# ---------------------------------------------------------------------------

def activity_map(frames, blur=3.0):
    """
    Per-pixel temporal standard deviation, lightly smoothed.

    Smoothing matters: a bubble rim moving by one pixel produces a very thin
    line of high variation, and without blurring the components break into
    fragments that are each too small to survive the area filter.
    """
    # Running sums in blocks of 16 frames: the same standard deviation as
    # frames.std(axis=0), without the full-size temporary arrays that
    # frames.std() builds (which used several hundred MB on a 600-frame clip
    # and exceeded the 512 MB of a small cloud server).
    n = len(frames)
    s = np.zeros(frames.shape[1:], np.float64)
    s2 = np.zeros_like(s)
    for i in range(0, n, 16):
        c = np.asarray(frames[i:i + 16], dtype=np.float64)
        s += c.sum(axis=0)
        s2 += (c * c).sum(axis=0)
    m = s / n
    sd = np.sqrt(np.maximum(s2 / n - m * m, 0.0)).astype(np.float32)
    return cv2.GaussianBlur(sd, (0, 0), blur)


def find_sites(act, shape, q=70.0, sigma_frac=0.0375, sep_frac=0.115,
               border=0.015):
    """
    One site per bubble, as local maxima of the activity map.

    Thresholding the map and taking connected components was the obvious thing
    to do and it does not work: where bubbles are close together, or a stream
    of them rises through the field, the active regions touch and merge into
    one component. On the 100 mA recording that produced a single 162 pixel
    "site" spanning the two middle bubbles and the stream between them, and
    both bubbles were lost.

    Local maxima do not merge. Smoothing the map at roughly the scale of a
    bubble and requiring each peak to be the largest value within a bubble
    diameter gives one peak per bubble. Measured against the five bubbles
    marked by hand in the 100 mA recording, this finds 5 of 5.

    The percentile threshold only removes peaks on quiet plate; it is not
    doing the separating.
    """
    H, W = shape
    s = max(3.0, sigma_frac * min(H, W))
    sep = max(9, int(sep_frac * min(H, W)) | 1)

    a = cv2.GaussianBlur(act, (0, 0), s)
    mx = ndi.maximum_filter(a, size=sep)
    thr = float(np.percentile(a, q))

    bx, by = int(border * W), int(border * H)
    sites = []
    for y, x in np.argwhere((a == mx) & (a > thr)):
        if x < bx or y < by or x >= W - bx or y >= H - by:
            continue
        sites.append((int(x), int(y), float(0.5 * sep), 0, 0))
    sites.sort(key=lambda s_: -float(a[s_[1], s_[0]]))
    return sites


# ---------------------------------------------------------------------------
# what is at a site in one frame
# ---------------------------------------------------------------------------

def fit_at_site(gray, flat, sd, x, y, r_hint, rmin, rmax,
                min_strength=0.015, min_coverage=0.62):
    """
    Best circle at one site in one frame, or None if the site is empty.

    The radius ladder exists because the site tells us roughly where a bubble
    is but not how big it is in this frame — that is the quantity being
    measured. Each trial radius is scored by the radial-boundary test and the
    best scoring one is kept.
    """
    best = None
    for k in (0.55, 0.75, 1.0, 1.3, 1.7, 2.2):
        r0 = r_hint * k
        if not (rmin * 0.5 <= r0 <= rmax * 1.5):
            continue
        strength, coverage, rb, pol = boundary_score(flat, x, y, r0, sd)
        if not (rmin <= rb <= rmax):
            continue
        if strength < min_strength or coverage < min_coverage:
            continue
        score = strength * coverage
        if best is None or rb > best["r_px"] + 1e-6:
            best = dict(x=int(x), y=int(y), r_px=float(rb), polarity=pol,
                        contrast=strength, agreement=coverage, score=score)
    return best


def refine(flat, sd, h, rmin, rmax, steps=4):
    """
    Snap the circle onto the bubble.

    The site marks where the picture changes most over the recording, which for
    a bubble that grows and detaches is not the same point as its centre, and
    may be off by a third of a radius. Searching a small grid of centres and
    radii around the first fit, and keeping whichever circle has the highest
    boundary coverage, moves the drawn circle onto the rim the eye sees. This
    costs a few dozen evaluations per bubble and is what makes the overlay
    worth looking at.
    """
    best = dict(h)
    best["key"] = (h["agreement"], h["r_px"])
    span = 0.35 * h["r_px"]
    for dx in np.linspace(-span, span, steps):
        for dy in np.linspace(-span, span, steps):
            x, y = int(round(h["x"] + dx)), int(round(h["y"] + dy))
            for k in (0.8, 1.0, 1.25, 1.55):
                strength, coverage, rb, pol = boundary_score(
                    flat, x, y, h["r_px"] * k, sd)
                if not (rmin <= rb <= rmax):
                    continue
                key = (coverage, rb)
                if key > best["key"]:
                    best = dict(x=x, y=y, r_px=float(rb), polarity=pol,
                                contrast=strength, agreement=coverage,
                                score=strength * coverage, key=key)
    best.pop("key", None)
    return best


def has_highlight(gray, p98, x, y, r, floor=0.90):
    """
    Is a bubble actually sitting at this site in this frame?

    A site is a place where bubbles appear, not a promise that one is there
    now: they detach, and the site is then bare plate. Without this test the
    fitter will happily place a circle on the empty site, which is what put a
    circle on bare plate in frame 45 of the 100 mA recording.

    The test is optical. A bubble is a curved gas-liquid interface and mirrors
    the lamp, so it carries a specular highlight as bright as anything in the
    frame. Bare plate at the same spot does not. Comparing the brightest few
    percent inside the circle with the brightest few percent of the whole frame
    separated the two completely on that recording:

        occupied sites     ratio 1.00  (every one of 23 checked)
        empty sites        ratio 0.37, 0.50, 0.75

    so the cut at 0.90 is sitting in an empty gap rather than splitting a
    distribution.
    """
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((xx - x) ** 2 + (yy - y) ** 2) < (0.75 * r) ** 2
    if m.sum() < 8:
        return False
    return float(np.percentile(gray[m], 98)) >= floor * p98


def detect_in_frame(gray, sites, rmin, rmax):
    flat = flatten(gray)
    sd = float(flat.std()) + 1e-6
    p98 = float(np.percentile(gray, 98))
    hits = []
    for (x, y, r, bw, bh) in sites:
        h = fit_at_site(gray, flat, sd, x, y, r, rmin, rmax)
        if h is None:
            continue
        h = refine(flat, sd, h, rmin, rmax)
        if not has_highlight(gray, p98, h["x"], h["y"], h["r_px"]):
            continue
        if all(np.hypot(h["x"] - k["x"], h["y"] - k["y"])
               > 0.6 * (h["r_px"] + k["r_px"]) for k in hits):
            hits.append(h)
    hits.sort(key=lambda c: -c["r_px"])
    return hits


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def read_clip(path, max_frames=240):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open " + path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out = []
    while len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not out:
        raise IOError("no frames decoded from " + path)
    return np.stack(out), fps, total


def analyse(frames, size_frac=(0.03, 0.60), every=5, max_frames=60,
            site_q=70.0):
    """
    Returns (sites, [(frame_index, hits), ...]).

    Sites are computed once from the whole clip; detection then runs on every
    `every`-th frame up to `max_frames`.
    """
    F = frames.astype(np.float32)
    H, W = F.shape[1:]
    rmin = max(0.5 * size_frac[0] * min(H, W), 6.0)
    rmax = 0.5 * size_frac[1] * min(H, W)

    act = activity_map(F)
    sites = find_sites(act, (H, W), q=site_q)

    results = []
    for i in range(0, len(frames), every):
        if len(results) >= max_frames:
            break
        results.append((i, detect_in_frame(frames[i], sites, rmin, rmax)))
    return sites, results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 5
    mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 60

    frames, fps, total = read_clip(path)
    sites, res = analyse(frames, every=every, max_frames=mx)

    print(f"\n{os.path.basename(path)}")
    print(f"  frames in file   {total}   fps {fps:.1f}   decoded {len(frames)}")
    print(f"  bubble sites     {len(sites)}  (from the temporal variation of "
          f"the whole clip)")
    for x, y, r, bw, bh in sites:
        print(f"     site at ({x:>3},{y:>3})  about {2*r:.0f} px across")
    print(f"\n  {'frame':>6} {'bubbles':>8}   diameters (px)")
    outdir = "video_output"
    os.makedirs(outdir, exist_ok=True)
    for i, hits in res:
        ds = sorted(round(2 * h["r_px"]) for h in hits)
        print(f"  {i:>6} {len(hits):>8}   {ds}")
    stem = os.path.splitext(os.path.basename(path))[0][:48]
    for i, hits in (res[0], res[len(res)//2], res[-1]):
        cv2.imwrite(os.path.join(outdir, f"{stem}_f{i}.png"),
                    draw(frames[i], [dict(h, d_mm=0) for h in hits], 1.0,
                         show_mm=False))
    print(f"\n  overlays written to {outdir}/")


if __name__ == "__main__":
    main()
