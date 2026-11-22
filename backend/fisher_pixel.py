"""
fisher_pixel.py
===============
Pixel-level Fisher information from the brightness field.

This implements the method Prof. Wagner proposed in his comments on the
interim report of 21 September 2026:

    "The change between two frames is nothing else than the derivation of
     the pixel brightness. This x-y matrix should become the Fisher
     information I(x,y,t) ... especially in the ring-shaped contact area."

    "... compute the Fisher information the other way round: the local
     change of the intensity in square divided by the intensity
     F(x,y,t) = dI(x,y)^2 / dI(x,y), and this for different current
     densities."

Reading his words rather than his shorthand, the quantity is

                    ( dI(x,y) )^2
    F(x,y)  =       -------------
                      I(x,y)

which is the standard Fisher-information form  (dp/dtheta)^2 / p  with the
normalised brightness field standing in for p.  Two versions are computed:

  TEMPORAL      F_t(x,y) = < (dI/dt)^2 / I >  averaged over the recording.
                Answers "where does this electrode carry information?"
                High where the brightness keeps changing, i.e. on the
                growing and collapsing rims.  One map per recording.

  ACROSS j      F_j(x,y) = (dI/dj)^2 / I  between neighbouring current
                densities, using the time-averaged brightness at each j.
                Answers "where does the electrode respond to the current?"

Neither version needs a detector, a threshold, a radius or a filter.  The
only inputs are the raw pixels, so there is nothing to tune and nothing to
tune it against.

USAGE
-----
    python3 fisher_pixel.py <folder>            all videos in the folder
    python3 fisher_pixel.py <folder> --max 400  cap frames read per video
    python3 fisher_pixel.py <folder> --lag-sweep edge/flat ratio for longer time steps

The folder must contain videos whose filenames carry the current density
as "NNNmA", which is how the laboratory names them, e.g.

    20260713_bigbubbles_300mA_1971mV_39deg.mp4
    bubble_video_2283mV_688mA_41deg.avi

Everything is written to  fisher_pixel_output/.
"""

import os
import re
import sys
import glob

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "fisher_pixel_output"

# Brightness floor. Pixel values are scaled to 0..1, so this stops the
# division blowing up in the black regions where I -> 0. It is the only
# numerical constant in the file and it is not a tuning knob: the maps are
# reported normalised, so a different floor rescales everything equally.
EPS = 1e-3

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")


# ---------------------------------------------------------------------------
# the camera's burnt-in timestamp
# ---------------------------------------------------------------------------

FLICKER_CUT = 0.60      # fraction of frame transitions a pixel must flip on
BANNER_ROWS = 0.13      # search only the top 13 % of rows
BANNER_COLS = 0.55      # and the left 55 % of columns
BANNER_PAD = 4          # pixels of margin around the found digits


def overlay_mask(frames):
    """
    Locate the camera's burnt-in timestamp so it can be excluded.

    The laboratory camera prints "640 x 480 / 200 fps / 20 us" and a
    date-time into the top-left of every frame. The static part of that
    text never changes, so it contributes nothing to dI and can be left
    alone. The millisecond digits are different: they change in every
    single frame, swinging the full way between glyph and background. That
    is the largest and most regular brightness change anywhere in the
    recording, and it has nothing whatever to do with hydrogen. Left in, it
    puts a bright rectangle of pure artefact into the Fisher map.

    The digits are found by their signature rather than by a hard-coded
    rectangle: inside the top-left corner where the banner is printed, a
    pixel that crosses its own mean in more than 60 % of frame transitions
    is rendered text. Nothing in the fluid flips that regularly -- the
    median pixel in these recordings sits between 0.06 and 0.19, and even
    the frothiest region at 900 mA/cm2 reaches only 0.32 at the 99th
    percentile.

    The test is deliberately run on the whole corner rather than on a fixed
    box, so that it reports where the banner actually is. On all four
    Session B recordings it independently converges on the same region,
    y 29-55 and x 224-241, which is what gives confidence that it is
    finding the clock and not the bubbles.

    Returns a boolean mask, True where the pixel should be excluded.
    """
    h, w = frames.shape[1:]
    y1 = int(h * BANNER_ROWS)
    x1 = int(w * BANNER_COLS)

    mean = frames.mean(axis=0)
    sign = np.sign(frames - mean)
    flicker = (np.abs(np.diff(sign, axis=0)) > 0).mean(axis=0)

    hits = flicker[:y1, :x1] > FLICKER_CUT
    mask = np.zeros((h, w), dtype=bool)
    if not hits.any():
        return mask

    # Keep only the densest cluster of hits. The digits sit together in one
    # small block, whereas the occasional bright froth pixel that also
    # passes the flicker test is isolated. Without this step a single stray
    # pixel at the far edge of the corner stretches the bounding box across
    # a third of the frame and deletes real data with it.
    merged = cv2.dilate(hits.astype(np.uint8),
                        np.ones((9, 9), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = stats[biggest, :4]

    # Mask the bounding box rather than the individual glyph pixels: the
    # frame difference at a glyph edge spills into the background
    # immediately around it, and the box is small enough that covering it
    # costs nothing.
    mask[max(0, y - BANNER_PAD): y + bh + BANNER_PAD,
         max(0, x - BANNER_PAD): x + bw + BANNER_PAD] = True
    return mask


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def current_density(path):
    """Pull NNN out of 'NNNmA' in the filename. None if it is not there."""
    m = re.search(r"(?<!\d)(\d{2,5})\s*mA", os.path.basename(path), re.I)
    return int(m.group(1)) if m else None


def read_grey(path, max_frames=400, stride=1):
    """
    Read up to max_frames grey-scale frames as float32 in 0..1.

    Frames are returned in capture order. No CLAHE, no blur, no background
    subtraction: the whole point of this method is that it acts on the raw
    brightness field.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open " + path)

    frames = []
    i = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(g.astype(np.float32) / 255.0)
        i += 1
    cap.release()

    if not frames:
        raise IOError("no frames decoded from " + path)
    return np.stack(frames)


# ---------------------------------------------------------------------------
# the two Fisher maps
# ---------------------------------------------------------------------------

def fisher_temporal(frames, mask=None):
    """
    F_t(x,y) = mean over t of  (dI/dt)^2 / I

    dI/dt is the forward difference between consecutive frames. I is taken
    as the mean of the two frames involved, so the denominator is symmetric
    in time and the map does not depend on which frame is called "first".

    Pixels flagged in mask (the camera's timestamp digits) are set to zero
    rather than deleted, so the map keeps its shape and can still be shown
    as an image.

    Returns (F, dI_rms, I_mean).
    """
    dI = np.diff(frames, axis=0)                      # (n-1, h, w)
    I_mid = 0.5 * (frames[:-1] + frames[1:])          # (n-1, h, w)
    F = ((dI ** 2) / (I_mid + EPS)).mean(axis=0)
    dI_rms = np.sqrt((dI ** 2).mean(axis=0))

    if mask is not None and mask.any():
        F = F.copy(); dI_rms = dI_rms.copy()
        F[mask] = 0.0
        dI_rms[mask] = 0.0

    return F, dI_rms, frames.mean(axis=0)


def fisher_across_j(I_a, I_b, j_a, j_b):
    """
    F_j(x,y) = (dI/dj)^2 / I  between two current densities.

    I_a and I_b are the time-averaged brightness fields. Dividing by the
    spacing makes the map a derivative rather than a difference, which is
    what lets two pairs with different spacing be compared at all -- the
    same 1/dj^2 scaling that had to be corrected in the bubble-table Fisher
    analysis.
    """
    dj = float(j_b - j_a)
    if dj == 0:
        raise ValueError("two recordings at the same current density")
    dIdj = (I_b - I_a) / dj
    I_mid = 0.5 * (I_a + I_b)
    return (dIdj ** 2) / (I_mid + EPS)


def ring_score(F, I_mean, mask=None):
    """
    How much of the information sits on edges rather than on flat areas.

    The bubble rim is where the brightness gradient is large, so a map that
    genuinely picks out the ring-shaped contact area should put its mass
    where |grad I| is high. This returns the ratio of mean F on the top
    decile of gradient magnitude to mean F elsewhere. Above 1 means the
    information concentrates on edges; near 1 means it is spread evenly and
    the map is not telling us anything about structure.
    """
    gx = cv2.Sobel(I_mean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I_mean, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    valid = np.ones(F.shape, dtype=bool)
    if mask is not None:
        valid &= ~mask

    cut = np.percentile(mag[valid], 90)
    edge = (mag >= cut) & valid
    flat = (mag < cut) & valid
    if not flat.any() or F[flat].mean() <= 0:
        return float("nan")
    return float(F[edge].mean() / F[flat].mean())


# ---------------------------------------------------------------------------
# stable paths: does the information return to the same pixels?
# ---------------------------------------------------------------------------
#
# Prof. Wagner read the F_t maps (report Figure 20) as qualitative proof
# that "bubbles follow stable paths in the probability state". If that holds,
# the pixels that carry the information in one part of a recording must also
# carry it in a later, independent part. path_share() turns this into one
# number that can be checked against chance:
#
#   1. Only SIGNIFICANT brightness changes are kept: |dI| above four times the
#      pixel's own noise (robust median estimate, never below three grey
#      levels, so 8-bit rounding cannot count as an event). Their Fisher
#      contribution (dI)^2 / I forms the event map.
#   2. The TOP 5 % pixels of the event map of the FIRST half are marked.
#   3. share = fraction of all event information of the SECOND half that
#      falls on those marked pixels.
#
#   Random placement gives 0.05. To check that the marked region is not
#   favoured merely by where it sits in the picture, the marked region is
#   also shifted to 20 random positions (null): share must lie above the
#   95th percentile of the shifted values to count as a stable path.
#
#   The noise level is checked as well: a motionless copy of the scene with
#   the measured noise gives the event rate that noise alone produces
#   (noise_rate). It is of order 1e-4, against 1e-2 to 1e-1 in the
#   recordings, so the events are bubbles, not camera noise.
#
# Why not simply overlap the F_t maps of two windows? Because (dI)^2 / I is
# largest on dark pixels even when nothing moves, and dark pixels do not move:
# a motionless noise copy gives a HIGHER overlap than the real recordings.
# The event map removes that bias. (Report, Section 4.13.)

Q_LEVEL = 1.0 / 255.0


def event_fisher(frames, mask=None, k_sigma=4.0):
    """Per-frame Fisher contributions of significant changes, and the event rate."""
    d = np.diff(frames, axis=0)
    I = 0.5 * (frames[:-1] + frames[1:])
    sig = np.maximum(1.4826 * np.median(np.abs(d), axis=0), Q_LEVEL)
    ev = np.abs(d) > np.maximum(k_sigma * sig, 3 * Q_LEVEL)[None]
    if mask is not None and mask.any():
        ev[:, mask] = False
    return np.where(ev, d * d / (I + EPS), 0.0).astype(np.float32), float(ev.mean())


def _top_mask(A, valid, top):
    x = A[valid] + np.random.default_rng(0).uniform(0, 1e-12, int(valid.sum()))
    T = np.zeros(A.shape, bool)
    T[valid] = x >= np.quantile(x, 1.0 - top)
    return T


def path_share(frames, mask=None, top=0.05, smooth=2.0, n_null=20):
    """
    Cross-validated concentration of the event information.

    Returns dict(share, null, null95, stable, events, noise_rate) and the
    first-half map, the second-half map and the marked region, for plotting.
    """
    Fs, rate = event_fisher(frames, mask)
    h = len(Fs) // 2
    if h < 5:
        raise ValueError("recording too short")
    valid = np.ones(frames.shape[1:], bool) if mask is None else ~mask
    A = cv2.GaussianBlur(Fs[:h].mean(0), (0, 0), smooth)
    B = Fs[h:].sum(0)
    T = _top_mask(A, valid, top)
    tot = max(float(B[valid].sum()), 1e-12)
    share = float(B[T].sum() / tot)
    rng = np.random.default_rng(1)
    hh, ww = A.shape
    nul = []
    for _ in range(n_null):
        Tr = np.roll(np.roll(T, rng.integers(hh // 5, 4 * hh // 5), 0),
                     rng.integers(ww // 5, 4 * ww // 5), 1) & valid
        nul.append(float(B[Tr].sum() / tot))
    # motionless copy with the measured noise, rounded to 8 bit like the camera
    sig = np.maximum(1.4826 * np.median(np.abs(np.diff(frames, axis=0)), axis=0), Q_LEVEL)
    fake = frames.mean(0)[None] + rng.normal(0, 1, frames.shape).astype(np.float32) \
        * (sig / np.sqrt(2.0))[None]
    fake = np.round(np.clip(fake, 0, 1) * 255) / 255.0
    _, noise_rate = event_fisher(fake.astype(np.float32), mask)
    null95 = float(np.percentile(nul, 95))
    return (dict(share=share, null=float(np.mean(nul)), null95=null95,
                 stable=bool(share > null95), events=rate, noise_rate=noise_rate),
            A, cv2.GaussianBlur(Fs[h:].mean(0), (0, 0), smooth), T)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _show(ax, img, title, cmap="inferno", pct=99.0):
    """Imshow with a robust upper clip, so one hot pixel cannot wash out a map."""
    hi = np.percentile(img, pct)
    hi = hi if hi > 0 else (img.max() if img.max() > 0 else 1.0)
    im = ax.imshow(img, cmap=cmap, vmin=0, vmax=hi)
    ax.set_title(title, fontsize=14.9)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def plot_temporal(results):
    """One column per recording: mean frame, |dI/dt|, and F_t."""
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 12), squeeze=False)

    for k, r in enumerate(results):
        _show(axes[0][k], r["I_mean"], f"{r['j']} mA/cm$^2$\nmean brightness",
              cmap="gray", pct=100)
        _show(axes[1][k], r["dI_rms"], "RMS frame-to-frame change\n$|dI/dt|$")
        _show(axes[2][k], r["F_t"], "Fisher information\n$F_t=(dI/dt)^2/I$")

    axes[0][0].set_ylabel("raw", fontsize=16.2)
    fig.suptitle("Pixel-level Fisher information from the brightness field\n"
                 "$F_t(x,y)=\\langle (dI/dt)^2 / I \\rangle$  —  no detector, "
                 "no threshold, no tuned parameter",
                 fontsize=17.6, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    p = os.path.join(OUT, "fisher_pixel_temporal.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote " + p)


def plot_across_j(pairs):
    """One panel per neighbouring pair of current densities."""
    if not pairs:
        return
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.6), squeeze=False)
    for k, p in enumerate(pairs):
        _show(axes[0][k], p["F"],
              f"{p['j_a']} $\\to$ {p['j_b']} mA/cm$^2$\n"
              f"$\\Delta j$ = {p['dj']}")
    fig.suptitle("Fisher information across current density\n"
                 "$F_j(x,y)=(dI/dj)^2 / I$ on the time-averaged brightness",
                 fontsize=17.6)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    path = os.path.join(OUT, "fisher_pixel_across_j.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote " + path)


TOP_EXCLUDE = 0.35      # fraction of rows ignored when choosing the close-up


def busiest_window(F, half=70, mask=None):
    """
    Centre of the most information-rich region, for the close-up panel.

    F is smoothed with a box filter the size of the crop and the maximum is
    taken. Two regions are excluded first. The top third of the frame is
    where the detached gas collects and is always the brightest thing in
    the map without being the thing of interest, and the banner box is
    excluded because a crop centred on the camera's clock would be
    embarrassing rather than informative.
    """
    h, w = F.shape
    work = F.copy()
    if mask is not None:
        work[mask] = 0.0

    smooth = cv2.blur(work, (2 * half + 1, 2 * half + 1))
    smooth[:int(h * TOP_EXCLUDE)] = 0.0
    if mask is not None:
        # Push the crop away from the banner as well as off it.
        grown = cv2.dilate(mask.astype(np.uint8),
                           np.ones((2 * half + 1, 2 * half + 1), np.uint8))
        smooth[grown.astype(bool)] = 0.0

    if smooth.max() <= 0:
        return h // 2, w // 2

    cy, cx = np.unravel_index(int(np.argmax(smooth)), F.shape)
    cy = int(np.clip(cy, half, h - half - 1))
    cx = int(np.clip(cx, half, w - half - 1))
    return cy, cx


def plot_ring_detail(r, tag=""):
    """
    Close-up showing whether the information really does sit on the rims.

    Left: the mean frame. Centre: the same frame with the top 5 % of the
    Fisher map drawn over it in red. Right: a close-up of the busiest
    region. If the method works as Prof. Wagner expects, the red traces the
    bubble outlines rather than filling their interiors.
    """
    I = r["I_mean"]; F = r["F_t"]
    rgb = cv2.cvtColor((np.clip(I, 0, 1) * 255).astype(np.uint8),
                       cv2.COLOR_GRAY2RGB).astype(np.float32)
    hot = F >= np.percentile(F, 95)
    overlay = rgb.copy()
    overlay[hot] = 0.35 * overlay[hot] + 0.65 * np.array([255, 40, 40])
    overlay = overlay.astype(np.uint8)

    half = 70
    cy, cx = busiest_window(F, half, r.get("mask"))
    crop = overlay[cy - half:cy + half, cx - half:cx + half]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    ax[0].imshow(I, cmap="gray")
    ax[0].set_title(f"Mean brightness, {r['j']} mA/cm$^2$", fontsize=14.9)

    ax[1].imshow(overlay)
    ax[1].set_title("Top 5 % of $F_t$ in red\n"
                    f"edge-to-flat ratio = {r['ring']:.2f}", fontsize=14.9)
    ax[1].add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half,
                                  fill=False, edgecolor="#22d3ee", linewidth=2))

    ax[2].imshow(crop)
    ax[2].set_title("Close-up of the boxed region", fontsize=14.9)
    for s in ax[2].spines.values():
        s.set_edgecolor("#22d3ee"); s.set_linewidth(2.5)

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    path = os.path.join(OUT, f"fisher_pixel_ring_detail{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote " + path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    global OUT

    folder = sys.argv[1]
    max_frames = 400
    if "--max" in sys.argv:
        max_frames = int(sys.argv[sys.argv.index("--max") + 1])

    # Name the output after the input, so analysing a second folder does
    # not silently overwrite the first one's maps.
    stem = os.path.basename(os.path.normpath(folder))
    OUT = OUT + "_" + stem if stem else OUT

    vids = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                  if p.lower().endswith(VIDEO_EXT))
    if not vids:
        print("No videos found in: " + folder)
        print("Expected one of " + ", ".join(VIDEO_EXT))
        return

    os.makedirs(OUT, exist_ok=True)

    print("PIXEL-LEVEL FISHER INFORMATION")
    print("=" * 70)
    print("Method requested by Prof. Wagner, comments of 22 September 2026.")
    print("F = (change in brightness)^2 / brightness.  No detector involved.\n")

    if "--lag-sweep" in sys.argv:
        # Edge-to-flat ratio when dI/dt is taken over a longer interval
        # (Section 4.9): frames k apart instead of consecutive frames.
        for path in vids:
            fr = read_grey(path, max_frames=max_frames)
            mk = overlay_mask(fr)
            print("LAG SWEEP  " + os.path.basename(path))
            for k in (1, 2, 5, 10, 20, 40, 80):
                if k >= len(fr):
                    break
                dI = fr[k:] - fr[:-k]
                Im = 0.5 * (fr[k:] + fr[:-k])
                F = ((dI ** 2) / (Im + EPS)).mean(axis=0)
                F[mk] = 0.0
                print(f"  lag {k:>3} frames   edge/flat = {ring_score(F, fr.mean(axis=0), mk):.2f}")
        return

    results = []
    for path in vids:
        j = current_density(path)
        if j is None:
            print("  skipping (no NNNmA in filename): " + os.path.basename(path))
            continue

        frames = read_grey(path, max_frames=max_frames)
        mask = overlay_mask(frames)
        F_t, dI_rms, I_mean = fisher_temporal(frames, mask)
        ring = ring_score(F_t, I_mean, mask)

        results.append(dict(path=path, j=j, F_t=F_t, dI_rms=dI_rms,
                            I_mean=I_mean, ring=ring, n=len(frames),
                            mask=mask, masked=int(mask.sum())))

        np.savetxt(os.path.join(OUT, f"fisher_pixel_temporal_{j}mA.csv"),
                   F_t, delimiter=",", fmt="%.6e")

        print(f"  {os.path.basename(path)[:48]:<50} "
              f"j={j:>4}  frames={len(frames):>4}  "
              f"F_mean={F_t.mean():.3e}  edge/flat={ring:.2f}  "
              f"masked={int(mask.sum()):>4} px")

    if not results:
        print("\nNothing analysed. Check that the filenames contain 'NNNmA'.")
        return

    results.sort(key=lambda r: r["j"])

    print("\nMaps:")
    plot_temporal(results)
    # A close-up for every recording, because which one shows the ring best
    # is exactly the question being asked and should not be pre-judged.
    for r in results:
        plot_ring_detail(r, tag=f"_{r['j']}mA")

    # ---- across current density -------------------------------------------
    pairs = []
    shapes = {r["I_mean"].shape for r in results}
    if len(shapes) > 1:
        print("\n  Recordings have different frame sizes " + str(shapes) + ",")
        print("  so the across-j map is skipped. Use one session at a time.")
    else:
        for a, b in zip(results[:-1], results[1:]):
            F = fisher_across_j(a["I_mean"], b["I_mean"], a["j"], b["j"])
            pairs.append(dict(j_a=a["j"], j_b=b["j"], dj=b["j"] - a["j"], F=F))
            np.savetxt(os.path.join(OUT,
                       f"fisher_pixel_j_{a['j']}to{b['j']}.csv"),
                       F, delimiter=",", fmt="%.6e")
        plot_across_j(pairs)

    # ---- summary -----------------------------------------------------------
    print("\nSUMMARY")
    print("-" * 70)
    print(f"  {'j (mA/cm2)':>11} {'frames':>7} {'mean F_t':>12} "
          f"{'edge/flat':>10} {'masked px':>10}")
    for r in results:
        print(f"  {r['j']:>11} {r['n']:>7} {r['F_t'].mean():>12.4e} "
              f"{r['ring']:>10.2f} {r['masked']:>10}")
    print("\n  'masked px' is the camera's flickering timestamp digits, "
          "excluded from\n  every figure quoted above.")

    rings = [r["ring"] for r in results if np.isfinite(r["ring"])]
    if rings:
        print(f"\n  Edge-to-flat ratio averages {np.mean(rings):.2f}.")
        if np.mean(rings) > 1.5:
            print("  The information concentrates on the brightness edges, "
                  "which is\n  where the bubble rims are. This is the "
                  "ring-shaped contact area\n  Prof. Wagner expected the "
                  "method to reveal.")
        else:
            print("  The information does NOT concentrate on edges. Either "
                  "the recording\n  is too noisy or the bubbles move too far "
                  "between frames. Report this\n  rather than tuning it away.")

    if pairs:
        print(f"\n  {'pair':>14} {'dj':>5} {'mean F_j':>12} {'max F_j':>12}")
        for p in pairs:
            print(f"  {p['j_a']:>5}->{p['j_b']:<7} {p['dj']:>5} "
                  f"{p['F'].mean():>12.4e} {p['F'].max():>12.4e}")

    print("\nAll output in ./" + OUT + "/")


if __name__ == "__main__":
    main()
