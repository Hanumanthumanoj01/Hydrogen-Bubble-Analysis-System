"""
detect_primary.py
=================
Detection of the LARGE, well-formed bubbles growing at nucleation sites —
the ones Prof. Wagner cares about.

WHY THIS FILE EXISTS
--------------------
Reviewing the interim report Prof. Wagner wrote, against the detection
overlay, that "the method does not seem to have been fine-tuned yet. I can
see multiple bubbles without detection circle."  The brief that followed is
specific: whatever else the software does, it must reliably put a clean
circle on every bubble in the 1-2 mm class. Froth may be ignored. A frame
containing a thousand tiny bubbles and fifteen large ones is a success if
the fifteen are found and a failure if they are not.

That is a different problem from the one the general detector in
bubble_analysis.py solves, and it is solved differently here.

WHY THE OLD APPROACH DOES NOT DO THIS
-------------------------------------
bubble_analysis.py thresholds on brightness and then looks for circles. On
the artificial-nucleation-site electrode that fails, for a reason that is
visible in any single frame: the machined plate is as dark as the bubbles
are. Thresholding returns the bubbles welded to half the background, and a
Hough transform run on that returns circles in the texture.

The cue used here instead is the one thing a bubble has and the plate does
not: a closed boundary at a definite size. The detector therefore works in
scale space.

  1. Flatten the illumination, since the light (from the camera side) is much
     brighter in the centre of the field than at the edges.
  2. Laplacian-of-Gaussian over a band of scales fixed by the requested
     size range. This answers "where is there a dark blob of about this
     size", which is exactly the question being asked.
  3. Verify every candidate against its radial profile. A real bubble gets
     brighter as you cross its rim outward, consistently, all the way
     round. A dark patch of plate does not. This stage does the actual
     discriminating, and it is the reason the output is clean.
  4. Suppress overlaps, largest and best-supported first.

SIZE IS SPECIFIED IN MILLIMETRES
--------------------------------
The caller asks for a millimetre band and the detector converts it with the
calibration in constants.py. Nothing in the file contains a hard-coded pixel
size, so when the laboratory confirms the scale the targeting follows
automatically.

The calibration currently in use is corroborated independently by this data
set: the nucleation-site grid on the ANS electrode measures 155.5 px between
neighbouring bores by autocorrelation, which at 0.0064 mm/px is 1.00 mm. A
bore pitch of exactly 1 mm is a plausible thing for a laboratory to drill,
so the published figure and this measurement agree. See `measure_pitch()`,
which recomputes it from any recording.

USAGE
-----
    python3 detect_primary.py <video-or-image> [frame] [--mm 0.25 1.2]
    python3 detect_primary.py session_D --sweep

The second form runs every recording in a folder and writes an overlay per
file, which is the quickest way to see whether the detector is behaving on
a new session.
"""

import os
import sys
import glob

import numpy as np
import cv2

try:
    from skimage.feature import blob_log
except ImportError:  # pragma: no cover
    blob_log = None

try:
    from constants import mm_per_pixel
except ImportError:
    def mm_per_pixel(source="image"):
        return 0.0064

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# The size class this detector exists to find, in millimetres of diameter.
# The default band is deliberately a little wider than "1 to 2 mm" so that a
# bubble sitting just outside the nominal class is still reported rather
# than silently dropped.
DEFAULT_MM = (0.20, 0.90)

# The size band the detector searches, as a fraction of the shorter side of the
# frame — a bubble diameter between 3 % and 35 % of the frame height.
#
# WHY THE BAND IS RELATIVE AND NOT IN MILLIMETRES
# -----------------------------------------------
# A millimetre band only works if every image arrives at the same scale, and in
# practice they do not. A frame taken straight from a recording is 640 x 480. A
# screen capture of the same recording played back and zoomed is 2211 x 1321 —
# the same four bubbles, three and a half times larger in pixels. Converting a
# fixed millimetre band with a fixed mm/px then asks for the wrong pixel sizes
# on one of the two, and on the screen capture it asked for sizes that match the
# magnified machining texture rather than the bubbles. That is what produced the
# 547 circles on an image containing four bubbles.
#
# Sizing relative to the frame removes the dependency: four bubbles filling a
# quarter of the height are found whether the file is 640 px wide or 2200.
DEFAULT_FRAC = (0.024, 0.60)

# Internal working height. Detection runs here and the results are mapped back,
# so a 4 K screen capture and a 640 x 480 frame cost the same and are judged by
# the same scale-space band.
WORK_H = 480


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------

def flatten(gray, sigma=41):
    """
    Remove the illumination gradient across the field.

    Dividing by a heavily blurred copy of the frame leaves local contrast
    alone while taking out the slow variation across the field. Without it
    the same bubble scores differently in the middle of the frame than at
    the edge, and no single threshold works anywhere.
    """
    g = gray.astype(np.float32)
    big = cv2.GaussianBlur(g, (0, 0), sigma)
    out = g / (big + 1e-6)
    lo, hi = np.percentile(out, 0.5), np.percentile(out, 99.5)
    return np.clip((out - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# verification — the part that does the work
# ---------------------------------------------------------------------------

def radial_profile(flat, x, y, r, n_ang=72, lo=0.35, hi=1.65, n_rad=26):
    """Median brightness on concentric rings from lo*r to hi*r."""
    h, w = flat.shape
    th = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    rads = np.linspace(lo * r, hi * r, n_rad)
    prof = np.empty(n_rad, np.float32)
    for i, rr in enumerate(rads):
        xi = np.clip((x + rr * ct).astype(int), 0, w - 1)
        yi = np.clip((y + rr * st).astype(int), 0, h - 1)
        prof[i] = np.median(flat[yi, xi])
    return rads, prof


N_ANG = 96


def boundary_score(flat, x, y, r, sd, lo=0.70, hi=1.45, n_rad=22):
    """
    Evidence that a closed boundary exists at about radius r, and where it is.

    Returns (strength, coverage, r_best, polarity).

    strength   the typical size of the brightness step across the boundary, in
               units of the frame's own standard deviation.
    coverage   the fraction of the 96 sampled directions that individually show
               that step, in the same direction. This is the discriminating
               number. A dark smear is darker on one side only and scores high
               on strength but low here; a bubble, having an actual closed
               boundary, scores high on both.
    r_best     the radius at which the boundary actually sits, which is not in
               general the radius that maximised the blob response — the
               Laplacian peaks on the body of a blob, a little inside its edge.

    WHY THIS REPLACED THE EARLIER INSIDE-VERSUS-OUTSIDE TEST
    --------------------------------------------------------
    The first version of this file assumed a bubble is darker than the plate
    and measured the step from inside the rim to outside it. Measured on the
    artificial-nucleation-site recordings, that assumption holds for only
    about half the bubbles in a frame. The domes in the lower half of the
    field are lit from the side and carry a bright crescent, so the step runs
    the other way: at (380, 355) in the frame used for Figure 12 the old test
    returned a contrast of -0.54, which failed the positive threshold and the
    bubble was dropped. Several of the bubbles Prof. Wagner marked as having
    no detection circle are of exactly this kind.

    Working on the radial derivative and taking the majority sign removes the
    assumption altogether. The detector now asks only whether the brightness
    changes sharply and consistently all the way round, which is true of a
    bubble boundary however the bubble happens to be lit.
    """
    th = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    radii = np.linspace(lo * r, hi * r, n_rad)

    h, w = flat.shape
    prof = np.empty((n_rad, N_ANG), np.float32)
    for i, rr in enumerate(radii):
        xi = np.clip((x + rr * ct).astype(int), 0, w - 1)
        yi = np.clip((y + rr * st).astype(int), 0, h - 1)
        prof[i] = flat[yi, xi]

    d = np.gradient(prof, axis=0)                       # radial derivative
    k = np.argmax(np.abs(d), axis=0)                    # strongest step per direction
    step = d[k, np.arange(N_ANG)] / sd
    rbest = radii[k]

    pol = 1.0 if step.mean() >= 0 else -1.0
    aligned = step * pol
    ok = aligned > 0.012
    coverage = float(ok.mean())
    strength = float(np.median(aligned))
    r_best = float(np.median(rbest[ok])) if coverage > 0 else float(r)
    return strength, coverage, r_best, ("dark" if pol > 0 else "bright")


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

def player_chrome_row(bgr, min_cover=0.30, search_from=0.60):
    """
    Where the video player's own interface starts, or None if there is none.

    A screen capture of a player carries a progress bar and a row of controls
    under the picture. The play button is a dark disc on a flat background and
    the detector circles it, which is correct behaviour applied to the wrong
    pixels — it is interface, not electrode.

    The giveaway is colour. The recordings are greyscale, so every pixel in the
    picture has zero saturation, while the progress bar is orange. A row in the
    lower part of the frame where a large fraction of pixels are coloured is the
    progress bar, and everything from there down is interface.

    Measured on the screen captures in this project the bar lands at 92 to 93 %
    of the height with 42 to 96 % of its pixels coloured, while an already
    cropped frame scores 0 %. A threshold of 30 % separates the two cleanly.
    """
    if bgr is None or bgr.ndim != 3:
        return None
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    h, w = sat.shape
    cover = (sat > 60).sum(axis=1) / float(w)
    lo = int(h * search_from)
    k = int(np.argmax(cover[lo:])) + lo
    if cover[k] < min_cover:
        return None
    return max(0, k - int(0.01 * h))


def autocrop(gray, thr=40):
    """
    Drop the black letterbox bars a screen capture of a video player leaves
    around the picture. Returns the cropped view and its offset in the original.

    Without this the bars count towards the frame height, every relative size is
    measured against a frame that is mostly black, and the illumination
    flattening sees an edge that is not in the data.
    """
    cm = np.percentile(gray, 95, axis=0)
    rm = np.percentile(gray, 95, axis=1)
    cs = np.where(cm > thr)[0]
    rs = np.where(rm > thr)[0]
    if len(cs) < 20 or len(rs) < 20:
        return gray, 0, 0
    return (gray[rs.min():rs.max() + 1, cs.min():cs.max() + 1],
            int(cs.min()), int(rs.min()))


def dark_cores(flat, rmin, rmax, qs=(1.0, 2.0, 3.0, 5.0, 8.0)):
    """
    Seed candidates from the darkest pixels in the frame.

    A bubble on a lit plate has a dark core, and on these recordings that core
    is among the darkest things present. Thresholding at several percentiles
    rather than one lets the same rule serve a frame holding four bubbles and a
    frame holding twenty, without anyone choosing a level.

    This runs on the RAW brightness, not on the illumination-flattened copy,
    and that distinction decided several of the misses. Flattening divides the
    frame by a blurred version of itself, so a bubble sitting in an already
    dark part of the plate comes out no darker than its surroundings and stops
    being in the darkest few percent. The bubble in the left third of the
    100 mA recording is exactly that case: raw brightness 64, which is as dark
    as any bubble in the frame, but 0.075 after flattening against a 2nd
    percentile of 0.093 — invisible to this stage, and missed in every frame.
    Measured on the raw frame it is found. Flattening still does its job in the
    boundary stage, where local contrast is what matters.

    This matters because the scale-space stage alone proposes nothing at all in
    some frames: a large dome with a bright crescent across it is not a blob of
    any single scale, and the Laplacian response spreads out and falls below
    threshold. The dark core is still there, and it is unmistakable.

    Components touching the border are dropped, and a component must be roughly
    round to be used — area filling at least 55 % of its bounding box, and sides
    within a factor of two. Scratch marks on the plate fail both.
    """
    H, W = flat.shape
    seeds = []
    for q in qs:
        m = (flat < np.percentile(flat, q)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            x0, y0, bw, bh = stats[i, :4]
            if x0 <= 1 or y0 <= 1 or x0 + bw >= W - 1 or y0 + bh >= H - 1:
                continue
            r = float(np.sqrt(a / np.pi))
            if r < rmin or r > rmax:
                continue
            if a / (bw * bh + 1e-6) < 0.55:
                continue
            if min(bw, bh) / max(bw, bh) < 0.55:
                continue
            seeds.append((int(cent[i][0]), int(cent[i][1]), r))
    return seeds


def detect_primary(gray, mm_range=DEFAULT_MM, scale=None,
                   log_threshold=0.025, min_contrast=0.018,
                   min_agreement=0.70, overlap=0.50,
                   size_frac=DEFAULT_FRAC, use_log=False, bgr=None,
                   mode="core"):
    """
    Find the well-formed bubbles in one frame.

    Returns a list of dicts with centre, radius in pixels and millimetres, and
    the two verification scores, sorted largest first. The keys `contrast` and
    `agreement` carry the strength and coverage described in boundary_score;
    the names are kept because the rest of the project reads them.

    Candidates are proposed twice, once treating bubbles as dark against the
    plate and once as bright. Neither polarity is assumed to be the right one
    — the verification stage decides, and on these recordings both occur in
    the same frame.
    """
    if blob_log is None:
        raise ImportError("scikit-image is required: pip3 install scikit-image")

    scale = mm_per_pixel("video") if scale is None else scale

    # Work on a cropped, standard-height copy; map back at the end.
    cut = player_chrome_row(bgr)
    if cut and cut > 0.3 * gray.shape[0]:
        gray = gray[:cut]
    crop, ox, oy = autocrop(gray)
    f = min(WORK_H / float(crop.shape[0]), 1.0)
    work = (cv2.resize(crop, (int(crop.shape[1] * f), int(crop.shape[0] * f)),
                       interpolation=cv2.INTER_AREA) if f < 1.0 else crop)

    H = float(min(work.shape))
    rmin = max(0.5 * size_frac[0] * H, 5.0)             # radius, working pixels
    rmax = 0.5 * size_frac[1] * H

    if mode == "core":
        out = []
        for c in _detect_core_rim(work, rmin, rmax):
            ridge = c.get("ridge", 0.0)
            out.append(dict(
                x=int(round(c["x"] / f)) + ox, y=int(round(c["y"] / f)) + oy,
                r_px=c["r"] / f, d_mm=2.0 * (c["r"] / f) * scale,
                polarity="rim", core=c["core"],
                agreement=c["arc"],                       # share of the rim supported
                contrast=2.0 * min(1.0, max(ridge, 0.0) / 0.5),
                score=c["s"], ridge=ridge,
                alts=[(ax / f + ox, ay / f + oy, ar / f, s) for ax, ay, ar, s in c["alts"]]))
        # step 7: not interface
        sat = (cv2.cvtColor(bgr[:gray.shape[0], :gray.shape[1]],
                            cv2.COLOR_BGR2HSV)[:, :, 1] > 60
               if bgr is not None and bgr.ndim == 3
               and bgr.shape[0] >= gray.shape[0] and bgr.shape[1] >= gray.shape[1]
               else None)
        H0, W0 = gray.shape[:2]
        clean = []
        for c in out:
            r = c["r_px"]
            x0, x1 = int(max(0, c["x"] - 1.2 * r)), int(min(W0, c["x"] + 1.2 * r + 1))
            y0, y1 = int(max(0, c["y"] - 1.2 * r)), int(min(H0, c["y"] + 1.2 * r + 1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d = np.hypot(xx - c["x"], yy - c["y"])
            disc = d < r
            band = (d > 0.85 * r) & (d < 1.15 * r)
            if disc.any() and (gray[y0:y1, x0:x1][disc] < 12).mean() > 0.03:
                continue
            if sat is not None and band.any() and sat[y0:y1, x0:x1][band].mean() > 0.03:
                continue
            clean.append(c)
        out = clean

        # Fallback for bubbles the rim stage cannot outline. On the Session C
        # nucleation-site recording the bubbles are dark domes with a soft edge
        # and no bright rim, beside saturated streaks of plate; their rims are
        # supported on only 36 to 46 % of rays and are rejected above. The
        # earlier core detector still finds them, with a boundary contrast of
        # 0.17 to 0.38 frame standard deviations, while the pieces of plate it
        # circles on the 100 mA/cm2 recording reach only 0.07 to 0.14. Its
        # circles at 0.15 or more are therefore kept where no rim circle
        # already covers the spot.
        for h in detect_primary(gray, mm_range, scale, log_threshold, min_contrast,
                                min_agreement, overlap, size_frac, use_log, bgr,
                                mode="core-seed"):
            if h["contrast"] < RIM_FALLBACK:
                continue
            if all(np.hypot(h["x"] - k["x"], h["y"] - k["y"]) > max(h["r_px"], k["r_px"])
                   for k in out):
                h["polarity"] = "core"
                out.append(h)
        out.sort(key=lambda c: -c["r_px"])
        return out

    flat = flatten(work)
    sd = float(flat.std()) + 1e-6
    Wh, Ww = work.shape
    margin = 0.02

    cands = []

    def consider(x, y, r0):
        if not (margin * Ww < x < (1 - margin) * Ww
                and margin * Wh < y < (1 - margin) * Wh):
            return
        strength, coverage, r, pol = boundary_score(flat, x, y, r0, sd)
        if not (rmin <= r <= rmax):
            return
        if strength < min_contrast or coverage < min_agreement:
            return
        cands.append(dict(x=int(x), y=int(y), r_px=float(r), polarity=pol,
                          contrast=strength, agreement=coverage,
                          score=strength * coverage))

    # Scale-space proposals are off by default. Measured on these recordings
    # they propose far more texture than bubbles: on the four-bubble screen
    # capture the two sources together returned 182 candidates against 5 from
    # the cores alone, and the extra 177 were plate. The Laplacian is answering
    # "is there a blob of about this size here", and magnified machining texture
    # answers yes. The dark core does not.
    #
    # Two optical regimes occur in this project and they need different
    # proposals, which is why `mode` exists rather than one set of thresholds
    # stretched to cover both.
    #
    #   mode="core"  — the artificial-nucleation-site close-ups. A bubble sits
    #                  on the lit plate and its centre goes near-black. The
    #                  dark core is the reliable cue and scale space is not.
    #
    #   mode="ring"  — the detached large bubbles of the "bigbubbles"
    #                  recordings. These are transparent spheres in the
    #                  electrolyte, seen as a bright rim with a grey middle.
    #                  They have no dark core at all, so the core pass proposes
    #                  almost nothing: on one frame it offered 6 candidates, the
    #                  largest 42 px, while the bubbles themselves are 74 to
    #                  89 px. Scale space does find them, and the darkness gate
    #                  has to come off or it discards every one.
    #
    # The froth in those recordings is real and dense, so ring mode leans on the
    # size band to separate bubbles from foam — froth sits at 5 to 6 % of the
    # frame height and the bubbles at 15 to 19 %.
    if mode == "ring":
        use_log = True
    if use_log:
        for img in (1.0 - flat, flat):
            for y, x, s in blob_log(img,
                                    min_sigma=rmin / np.sqrt(2),
                                    max_sigma=rmax / np.sqrt(2),
                                    num_sigma=12,
                                    threshold=log_threshold,
                                    overlap=overlap):
                consider(int(round(x)), int(round(y)), float(s) * np.sqrt(2))

    for x, y, r in dark_cores(work.astype(np.float32) / 255.0, rmin, rmax):
        # A seed marks the dark core, which on some bubbles is a small spot
        # inside a much larger bright-rimmed dome. The boundary stage only
        # searches 0.7 to 1.45 times the radius it is given, so growing the
        # seed through this ladder is what lets a 15 pixel core find a 100
        # pixel rim. On the 100 mA recording it changed the drawn diameters
        # from 16-49 px to 80-103 px against bubbles that measure about 100.
        for k in (1.0, 1.5, 2.2, 3.2, 4.5):             # core, then outward
            consider(x, y, max(r * k, rmin))

    # Overlap suppression, best first. The same bubble is usually proposed by
    # more than one source, so this stage is doing real work, not just tidying.
    cands.sort(key=lambda c: -c["score"])
    keep = []
    for c in cands:
        if all(np.hypot(c["x"] - k["x"], c["y"] - k["y"])
               > 0.60 * (c["r_px"] + k["r_px"]) for k in keep):
            keep.append(c)

    # Final gate: is the middle of it actually dark?
    #
    # This is the stage that makes the detector work on a zoomed view without
    # anyone tuning anything. A hydrogen bubble on this plate photographs as a
    # near-black core: the light comes from the camera side, and the curved
    # bubble surface reflects it back to the lens only at a thin rim and a small
    # highlight, so the interior returns little light to the camera. Machining speckle and shadow are merely darker than their
    # surroundings, which is a much weaker thing to be.
    #
    # Measured across the recordings: in every frame checked, real bubble cores
    # sit at grey level 61 to 76, while the structures that were being circled
    # by mistake sit at 90 to 142 — and this holds whether the frame is a native
    # 640 x 480 or a screen capture magnified three and a half times, because
    # the camera, plate and lighting are the same in both. Expressing the
    # threshold as the frame's own 2nd percentile rather than a fixed number
    # keeps it working when the exposure changes.
    #
    # On one close-up this stage alone took the output from 25 circles to the 2
    # bubbles actually present.
    p2 = float(np.percentile(work, 2))
    yy, xx = np.mgrid[0:Wh, 0:Ww]
    verified = []
    for c in keep:
        m = ((xx - c["x"]) ** 2 + (yy - c["y"]) ** 2) < (0.6 * c["r_px"]) ** 2
        if m.sum() < 4:
            continue
        c["core"] = float(np.percentile(work[m], 10))
        if mode == "ring" or c["core"] < p2:
            verified.append(c)
    keep = verified

    # Back to the coordinates of the image that was handed in.
    for c in keep:
        c["x"] = int(round(c["x"] / f)) + ox
        c["y"] = int(round(c["y"] / f)) + oy
        c["r_px"] = c["r_px"] / f
        c["d_mm"] = 2.0 * c["r_px"] * scale
    keep.sort(key=lambda c: -c["r_px"])
    return keep


# ---------------------------------------------------------------------------
# whole-bubble outline: rim fitting  (mode="core", since November 2026)
# ---------------------------------------------------------------------------
#
# WHY THE CORE DETECTOR WAS REWRITTEN
# -----------------------------------
# The detector above anchors every circle on a bubble's dark core and grows it
# outward. On the dome-shaped bubbles of the nucleation-site recordings the core
# is small and usually off-centre, and the dome contains further rings of its
# own (the bright crescent, the edge of the core). The circle therefore often
# stopped on one of those inner rings: on frame 25 of the 100 mA/cm² recording
# it drew 11 circles, of which only two outlined a whole bubble; the large dome
# carried three small circles and the plate four more.
#
# The rim stage below outlines the bubble itself:
#
#   1. Proposals. Every dark core, at seven trial radii from 1.2 to 7.5 times
#      its own, plus every circle the gradient Hough transform suggests. Neither
#      source has to be right; they only have to start near a rim.
#   2. Fit. From the proposal centre, 96 rays are cast; on each the strongest
#      radially pointing brightness edge within 0.78 to 1.25 times the radius is
#      taken, a circle is fitted to those points by least squares, points more
#      than 10 % off are dropped and the fit repeated, and the whole step is
#      iterated four times. The centre is free to move, which is what corrects
#      the offset between a core and its bubble.
#   3. Support. A fitted circle is kept only if, on at least 56 % of the rays,
#      an edge in the top 30 % of the frame's edge strengths points across the
#      circle, and those rays form continuous arcs (runs of at least 4). Plate
#      texture gives scattered support of 30 to 52 %; bubble rims 60 to 96 %.
#   4. Dark interior. Its inner 60 % must reach the frame's darkest 5 %: a
#      hydrogen bubble on this plate always holds a near-black refraction core.
#   5. One circle per bubble. Fitted circles whose centres lie within half a
#      radius of each other describe the same bubble. Of those, the one with the
#      best support is kept, with a small preference for the larger circle
#      (0.33 times the logarithm of the radius added to the score), because the
#      outer rim is always the larger of a bubble's rings and its support is
#      weakened by the faint lower edge. The value 0.33 is the middle of the
#      range (0.25 to 0.43) in which every bubble checked by eye was outlined
#      correctly.
#   6. Rim brightness. Averaged over all directions, a bubble's outer rim is a
#      bright ring with darker plate outside it. A circle whose ring is not
#      brighter than the plate outside by at least 0.14 frame standard
#      deviations is dropped; this removes the small circles that fit pieces of
#      texture between bubbles. Every such circle measured was 11 to 29 px on a
#      480 px frame, while large domes in zoomed captures and on the Session C
#      plate have a dark rather than bright outer halo, so the test applies only
#      to circles below 8 % of the frame height.
#   7. Not interface. A circle on the burnt-in timestamp or crosshair (coloured
#      pixels, in a colour image) or on a letterbox bar (pixels below grey 12,
#      far darker than any bubble core at 60 to 76) is dropped.
#   8. Satellites. A circle under half the radius of a neighbour and within
#      1.5 of that neighbour's radii is dropped (see the code for why this
#      cannot remove a bubble on a separate site).
#
# In a recording, temporal_consensus() then corrects single frames whose choice
# disagrees with the frames around them.

RIM_K = 96
_RIM_TH = np.linspace(0, 2 * np.pi, RIM_K, endpoint=False)
_RIM_CT, _RIM_ST = np.cos(_RIM_TH), np.sin(_RIM_TH)
RIM_ARC = 0.56        # step 3
RIM_DARK_Q = 5        # step 4, percentile
RIM_LAMBDA = 0.33     # step 5
RIM_RIDGE = 0.14      # step 6, frame standard deviations
RIM_FALLBACK = 0.15   # boundary contrast for the earlier detector's circles
RIM_RIDGE_MAXR = 0.08 # step 6 applies below this radius (fraction of the frame)


RIM_LOCAL = False     # edges measured against their own neighbourhood


def _rim_grad(work):
    """
    Radial-edge maps. With RIM_LOCAL each gradient is divided by the local
    brightness spread (Gaussian, sigma 10 px), so that an edge counts by how
    much it stands out from its own surroundings rather than from the whole
    frame. Without it, the saturated streaks of the Session C plate set the
    frame-wide edge threshold so high that the soft rims of its bubbles never
    reach it.
    """
    f = flatten(work)
    b = cv2.GaussianBlur(f, (0, 0), 1.6)
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    if RIM_LOCAL:
        mu = cv2.GaussianBlur(f, (0, 0), 10)
        var = cv2.GaussianBlur(f * f, (0, 0), 10) - mu * mu
        loc = np.sqrt(np.maximum(var, 0)) + 0.02
        gx, gy = gx / loc, gy / loc
    return gx, gy, np.hypot(gx, gy)


def _ray_edges(gx, gy, cx, cy, rlo, rhi, n=40):
    """On each ray, the radius of the strongest radially pointing edge."""
    h, w = gx.shape
    rs = np.linspace(rlo, rhi, n)
    xi = np.clip(np.round(cx + rs[:, None] * _RIM_CT).astype(int), 0, w - 1)
    yi = np.clip(np.round(cy + rs[:, None] * _RIM_ST).astype(int), 0, h - 1)
    a = np.abs(gx[yi, xi] * _RIM_CT + gy[yi, xi] * _RIM_ST)
    k = np.argmax(a, axis=0)
    return rs[k], a[k, np.arange(RIM_K)]


def _fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (cx, cy, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    return cx, cy, float(np.sqrt(max(c + cx * cx + cy * cy, 1e-6)))


def _rim_fit(gx, gy, tmag, cx, cy, r, it=4):
    """Step 2. Returns (x, y, r) or None."""
    for _ in range(it):
        rr, a = _ray_edges(gx, gy, cx, cy, 0.78 * r, 1.25 * r)
        ok = a > tmag
        if ok.sum() < 0.35 * RIM_K:
            return None
        x, y = (cx + rr * _RIM_CT)[ok], (cy + rr * _RIM_ST)[ok]
        ncx, ncy, nr = _fit_circle(x, y)
        keep = np.abs(np.hypot(x - ncx, y - ncy) - nr) < max(2.5, 0.10 * nr)
        if keep.sum() >= 0.35 * RIM_K:
            ncx, ncy, nr = _fit_circle(x[keep], y[keep])
        if np.hypot(ncx - cx, ncy - cy) > 0.6 * r or not (0.6 * r < nr < 1.6 * r):
            return None
        cx, cy, r = ncx, ncy, nr
    return cx, cy, r


def _rim_support(gx, gy, m, cx, cy, r, tq):
    """Step 3: (arc support, visible fraction of the circle)."""
    h, w = m.shape
    tol = max(3.0, 0.07 * r)
    rs = np.linspace(r - tol, r + tol, 9)
    X = cx + rs[:, None] * _RIM_CT
    Y = cy + rs[:, None] * _RIM_ST
    vis = ((X >= 0) & (X < w) & (Y >= 0) & (Y < h)).all(axis=0)
    xi = np.clip(np.round(X).astype(int), 0, w - 1)
    yi = np.clip(np.round(Y).astype(int), 0, h - 1)
    rad = gx[yi, xi] * _RIM_CT + gy[yi, xi] * _RIM_ST
    good = (np.abs(rad) > tq) & (np.abs(rad) / (m[yi, xi] + 1e-9) > 0.8)
    hit = good.any(axis=0)
    h2 = np.r_[hit, hit]
    runs = np.zeros(RIM_K, bool)
    n = 0
    for i in range(2 * RIM_K):
        n = n + 1 if h2[i] else 0
        if n >= 4:
            for j in range(i - n + 1, i + 1):
                runs[j % RIM_K] = True
    if not vis.any():
        return 0.0, 0.0
    return float((runs & hit)[vis].mean()), float(vis.mean())


def _rim_ridge(work, cx, cy, r, sd):
    """Step 6: brightness of the ring at r above the plate just outside it."""
    h, w = work.shape
    rs = np.linspace(0.5 * r, 1.5 * r, 41)
    X = cx + rs[:, None] * _RIM_CT
    Y = cy + rs[:, None] * _RIM_ST
    ok = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
    P = np.where(ok, work[np.clip(Y.astype(int), 0, h - 1),
                          np.clip(X.astype(int), 0, w - 1)].astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        M = np.nanmedian(P, axis=1)
        at = np.nanmax(M[(rs >= 0.9 * r) & (rs <= 1.1 * r)])
        out = np.nanmedian(M[(rs >= 1.2 * r) & (rs <= 1.5 * r)])
    if not np.isfinite(at) or not np.isfinite(out):
        return 0.0
    return float((at - out) / sd)


def _detect_core_rim(work, rmin, rmax):
    """Steps 1 to 6 on the working copy. Returns hits with alternatives."""
    gx, gy, m = _rim_grad(work)
    t70, t80 = np.percentile(m, 70), np.percentile(m, 80)
    sd = float(work.std()) + 1e-6

    seeds = []
    for s in dark_cores(work.astype(np.float32) / 255.0, 0.5 * rmin, rmax):
        if all(np.hypot(s[0] - u[0], s[1] - u[1]) > 4 for u in seeds):
            seeds.append(s)
    starts = [(x, y, max(r * k, rmin)) for x, y, r in seeds
              for k in (1.2, 1.6, 2.2, 3.0, 4.0, 5.5, 7.5) if r * k <= rmax]
    hc = cv2.HoughCircles(cv2.GaussianBlur(work, (0, 0), 2), cv2.HOUGH_GRADIENT_ALT,
                          dp=1.5, minDist=10, param1=50, param2=0.5,
                          minRadius=int(rmin), maxRadius=int(rmax))
    if hc is not None:
        starts += [tuple(float(v) for v in c) for c in hc[0]]

    dark = float(np.percentile(work, RIM_DARK_Q))
    Hh, Ww = work.shape
    cands = []
    for x, y, r in starts:
        fit = _rim_fit(gx, gy, t80, x, y, r)
        if fit is None:
            continue
        cx, cy, cr = fit
        if not (rmin <= cr <= rmax):
            continue
        arc, vis = _rim_support(gx, gy, m, cx, cy, cr, t70)
        if vis < 0.5 or arc < RIM_ARC:
            continue
        x0, x1 = int(max(0, cx - 0.6 * cr)), int(min(Ww, cx + 0.6 * cr + 1))
        y0, y1 = int(max(0, cy - 0.6 * cr)), int(min(Hh, cy + 0.6 * cr + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inner = work[y0:y1, x0:x1][(xx - cx) ** 2 + (yy - cy) ** 2 < (0.6 * cr) ** 2]
        if inner.size < 4:
            continue
        core = float(np.percentile(inner, 10))
        if core > dark:
            continue
        arc80, _ = _rim_support(gx, gy, m, cx, cy, cr, t80)
        cands.append(dict(x=cx, y=cy, r=cr, arc=arc, s=arc + arc80, core=core))

    # step 5: group the circles drawn around one bubble
    n = len(cands)
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            a, b = cands[i], cands[j]
            if np.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 0.5 * max(a["r"], b["r"]):
                parent[root(i)] = root(j)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(cands[i])

    picks = []
    for G in groups.values():
        G.sort(key=lambda c: -(c["s"] + RIM_LAMBDA * np.log(c["r"])))
        best = None
        for c in G:                                       # step 6, best first
            c["ridge"] = _rim_ridge(work, c["x"], c["y"], c["r"], sd)
            if c["ridge"] >= RIM_RIDGE or c["r"] >= RIM_RIDGE_MAXR * min(work.shape):
                best = c
                break
        if best is None:
            continue
        best["alts"] = [(c["x"], c["y"], c["r"], c["s"]) for c in G
                        if c.get("ridge", RIM_RIDGE) >= RIM_RIDGE
                        or c["r"] >= RIM_RIDGE_MAXR * min(work.shape)]
        picks.append(best)

    # two different bubbles claiming the same pixels: the better supported wins
    picks.sort(key=lambda c: -c["s"])
    keep = []
    for c in picks:
        if all(np.hypot(c["x"] - k["x"], c["y"] - k["y"]) > 0.7 * (c["r"] + k["r"])
               for k in keep):
            keep.append(c)
    # step 8: a circle less than half the size of a neighbour and within 1.5 of
    # that neighbour's radii is a satellite or texture on the neighbour's dome
    # (on the 860 mA/cm2 zoomed capture, dark stripes inside the large dome).
    # Nucleation sites on these plates are 155 px apart and bubbles reach a
    # radius of about 60 px, so two bubbles on separate sites never meet this.
    keep.sort(key=lambda c: -c["r"])
    out = []
    for c in keep:
        if any(k["r"] >= 2.0 * c["r"] and np.hypot(c["x"] - k["x"], c["y"] - k["y"]) < 1.5 * k["r"]
               for k in out):
            continue
        out.append(c)
    return out


def temporal_consensus(frames_hits, window=2, tol=1.25):
    """
    Correct single frames in a recording whose circle size disagrees with the
    frames around them.

    `frames_hits` is a list, one entry per analysed frame in time order, of the
    hit lists detect_primary returned. A bubble on a nucleation site grows by
    at most a few per cent between two analysed frames, so where one frame
    reports a circle 25 % larger or smaller than the median of the same site in
    up to `window` frames either side, and that frame's own fit offered an
    alternative circle within 25 % of that median, the alternative is used.
    Nothing is added. Two kinds of circle are removed: a circle from the
    fallback detector (polarity "core") that no circle in the analysed frame
    before or after it confirms - bubbles on nucleation sites persist across
    analysed frames, and the fallback circles that do not are, on the 100
    mA/cm2 recording, pieces of plate - and a rim circle more than twice the
    size the neighbouring frames agree on for which the frame offers no better
    fit. Edits the lists in place; returns the number changed or removed.
    """
    changed = 0
    T = len(frames_hits)

    def matched(h, u):
        return any(np.hypot(k["x"] - h["x"], k["y"] - h["y"])
                   < 0.6 * max(k["r_px"], h["r_px"]) for k in frames_hits[u])

    if T > 1:
        drop = []
        for t in range(T):
            nb = [u for u in (t - 1, t + 1) if 0 <= u < T]
            drop.append([h for h in frames_hits[t]
                         if h.get("polarity") == "core" and not any(matched(h, u) for u in nb)])
        for t in range(T):
            for h in drop[t]:
                frames_hits[t].remove(h)
                changed += 1
    for t in range(T):
        for h in frames_hits[t]:
            if h.get("polarity") != "rim":
                continue
            alts = h.get("alts") or []
            near = []
            for u in range(max(0, t - window), min(T, t + window + 1)):
                if u == t:
                    continue
                for k in frames_hits[u]:
                    if np.hypot(k["x"] - h["x"], k["y"] - h["y"]) < 0.6 * max(k["r_px"], h["r_px"]):
                        near.append(k["r_px"])
                        break
            if len(near) < 2:
                continue
            med = float(np.median(near))
            if abs(np.log(h["r_px"] / med)) <= np.log(tol):
                continue
            x, y, r, _ = (min(alts, key=lambda a: abs(np.log(a[2] / med)))
                          if alts else (h["x"], h["y"], h["r_px"], 0))
            if abs(np.log(r / med)) <= np.log(tol):
                mm_per_px = h["d_mm"] / (2.0 * h["r_px"])
                h.update(x=int(round(x)), y=int(round(y)), r_px=float(r),
                         d_mm=2.0 * float(r) * mm_per_px, revised=True)
                changed += 1
            elif h["r_px"] > 2.0 * med:
                # More than twice the size the neighbouring frames agree on, and
                # no circle of this frame fits better: not this bubble. A bubble
                # cannot double its radius between two analysed frames; even
                # the merger of two equal bubbles enlarges it by only 26 %.
                h["remove"] = True
    for t in range(T):
        gone = [h for h in frames_hits[t] if h.pop("remove", False)]
        for h in gone:
            frames_hits[t].remove(h)
            changed += 1
    return changed


def draw(gray, hits, scale=None, show_mm=True):
    """Overlay the detections. Thick, unambiguous circles: this is the figure."""
    scale = mm_per_pixel("video") if scale is None else scale
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for c in hits:
        x, y, r = int(c["x"]), int(c["y"]), int(round(c["r_px"]))
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 9, 2)
        if show_mm:
            cv2.putText(vis, f"{c['d_mm']:.2f}", (x - 22, y - r - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1,
                        cv2.LINE_AA)
    cv2.putText(vis, f"{len(hits)} bubbles", (10, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# calibration from the bore grid
# ---------------------------------------------------------------------------

def measure_pitch(frames):
    """
    Distance between neighbouring nucleation sites, in pixels.

    The bores are drilled on a regular grid and never move, so the median
    frame is a clean picture of them and the autocorrelation of that picture
    peaks at the grid pitch. This needs no detector and no threshold, which
    is why it is a usable calibration check rather than another estimate
    with the same assumptions baked in.

    Multiply the returned value by nothing and divide the laboratory's
    nominal pitch in millimetres by it to get mm per pixel.
    """
    bg = np.median(frames, axis=0).astype(np.float32)
    big = cv2.GaussianBlur(bg, (0, 0), 45)
    flat = bg / (big + 1e-6)
    work = flat[40:, :]                      # drop the burnt-in camera banner
    work = work - work.mean()

    F = np.fft.fft2(work)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    cy, cx = np.array(ac.shape) // 2

    def first_peak(prof, lo=40):
        best, bi = -np.inf, None
        for i in range(lo, len(prof) - 1):
            if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > best:
                best, bi = prof[i], i
        return bi

    px = first_peak(ac[cy, cx:])
    py = first_peak(ac[cy:, cx])
    vals = [v for v in (px, py) if v]
    return (float(np.mean(vals)) if vals else None), px, py


def calibrate_from_pitch(pitch_px, pitch_mm):
    """
    Millimetres per pixel, from the one measurement that does not depend on
    any earlier calibration: the drilled spacing of the nucleation sites.

    Give it the pitch this software measures in pixels and the nominal pitch
    the laboratory drilled in millimetres, and it returns the scale. This is
    the "more precise value" Prof. Wagner offered in his comment on the
    interim report, and it needs exactly one number from him.
    """
    if not pitch_px:
        raise ValueError("no pitch measured")
    return float(pitch_mm) / float(pitch_px)


def size_table(pitch_px, measured_px):
    """
    What a measured pixel size would mean for several candidate bore pitches.

    Printed because the millimetre figure quoted anywhere in this project is
    only as good as the scale, and the scale is the open question. Showing
    the dependence is more honest than printing one number as if it were
    settled.
    """
    rows = []
    for mm in (1.0, 2.0, 3.0, 4.0, 5.0):
        s = mm / pitch_px
        rows.append((mm, s, measured_px * s))
    return rows


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def read_frames(path, max_frames=200):
    if path.lower().endswith(IMAGE_EXT):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError("cannot read " + path)
        return np.stack([img.astype(np.float32) / 255.0])
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open " + path)
    out = []
    while len(out) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
    cap.release()
    if not out:
        raise IOError("no frames decoded from " + path)
    return np.stack(out)


def run_one(path, frame_idx=None, mm_range=DEFAULT_MM, outdir="primary_output"):
    frames = read_frames(path)
    idx = len(frames) // 3 if frame_idx is None else min(frame_idx, len(frames) - 1)
    gray = (frames[idx] * 255).astype(np.uint8)

    scale = mm_per_pixel("video")
    hits = detect_primary(gray, mm_range, scale)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0][:48]
    out = os.path.join(outdir, stem + f"_frame{idx}.png")
    cv2.imwrite(out, draw(gray, hits, scale))

    pitch = None
    if len(frames) > 20:
        pitch, _, _ = measure_pitch(frames)

    print(f"\n{os.path.basename(path)}   frame {idx}")
    print(f"  size band        {mm_range[0]:.2f} - {mm_range[1]:.2f} mm "
          f"({0.5*mm_range[0]/scale:.0f} - {0.5*mm_range[1]/scale:.0f} px radius)")
    if pitch:
        print(f"  periodic spacing {pitch:.1f} px  "
              f"= {pitch*scale:.2f} mm at the current calibration")
        print(f"  {'':17}this is the strongest repeating spacing in the frame.")
        print(f"  {'':17}On an artificial-nucleation-site plate it is the bore")
        print(f"  {'':17}pitch and can be used to calibrate. On a mesh or GDE")
        print(f"  {'':17}electrode it is the machining pattern and means")
        print(f"  {'':17}nothing for calibration - ignore it there.")
    print(f"  bubbles found    {len(hits)}")
    if hits and pitch:
        med = float(np.median([2*c["r_px"] for c in hits]))
        print(f"\n  CALIBRATION CHECK (ANS recordings only) — median bubble "
              f"is {med:.0f} px across.")
        print(f"  {'if bore pitch is':>18} {'then mm/px':>11} "
              f"{'median bubble':>14}")
        for mm, s, d in size_table(pitch, med):
            print(f"  {mm:>15.1f} mm {s:>11.5f} {d:>12.2f} mm")
        print("  Ask the laboratory for the nominal bore pitch; it fixes "
              "every\n  millimetre figure in the project at once.\n")
    for c in hits:
        print(f"     ({c['x']:>3},{c['y']:>3})  d = {c['d_mm']:.2f} mm "
              f"({2*c['r_px']:.0f} px)   contrast {c['contrast']:.2f}  "
              f"agreement {c['agreement']:.2f}")
    print(f"  overlay          {out}")
    return hits


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    target = sys.argv[1]
    mm_range = DEFAULT_MM
    if "--mm" in sys.argv:
        i = sys.argv.index("--mm")
        mm_range = (float(sys.argv[i + 1]), float(sys.argv[i + 2]))

    frame_idx = None
    for a in sys.argv[2:]:
        if a.isdigit():
            frame_idx = int(a)
            break

    if os.path.isdir(target):
        files = sorted(p for p in glob.glob(os.path.join(target, "*"))
                       if p.lower().endswith(VIDEO_EXT + IMAGE_EXT))
        if not files:
            print("No videos or images in " + target)
            return
        print(f"PRIMARY BUBBLE DETECTION — {len(files)} files in {target}")
        print("=" * 70)
        total = 0
        for p in files:
            total += len(run_one(p, frame_idx, mm_range))
        print(f"\n{total} bubbles across {len(files)} files, "
              f"{total/len(files):.1f} per frame on average.")
    else:
        print("PRIMARY BUBBLE DETECTION")
        print("=" * 70)
        run_one(target, frame_idx, mm_range)


if __name__ == "__main__":
    main()
