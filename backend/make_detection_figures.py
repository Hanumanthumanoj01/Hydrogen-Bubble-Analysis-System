"""
make_detection_figures.py
=========================
Builds the figures for the detection sections of the report:

    detector_polarity.png     the dark-bubble assumption, before and after
    detector_ans_still.png    old and new detector on the same frames (Figure 23)
    detector_zoomed.png       the same detector on a magnified close-up
    video_activity_map.png    temporal variation: where the bubbles are
    video_frames.png          three frames of a recording, detections drawn

USAGE
-----
Run from the backend folder:

    python3 make_detection_figures.py \
        --ans   session_C/bubble_video_2283mV_688mA_41deg.avi \
        --big   videos/20260713_bigbubbles_100mA_1760mV_39deg.mp4 \
        --zoom  my_screenshots/two_bubble_closeup.png \
        --out   report_figures

--ans   the 688 mA/cm2 nucleation-site recording. Figures 12, 23 and 26
        come from it; the still frame is taken from it at --frame.
--big   the 100 mA/cm2 large-bubble recording. Figure 13 comes from it.
--zoom  OPTIONAL. Your own screen capture of a player zoomed into the
        recording. Figure 26 comes from it. Leave it out and that one figure
        is skipped; everything else is still made.

Every figure is made from the files you pass in, on your own machine. Nothing
is bundled with the script.
"""
import os
import sys

import cv2
import numpy as np

import detect_primary as dp
import detect_video as dv

import argparse

_ap = argparse.ArgumentParser(description="Detection figures for the report.")
_ap.add_argument("--ans", default="session_C/bubble_video_2283mV_688mA_41deg.avi",
                 help="688 mA/cm2 nucleation-site recording")
_ap.add_argument("--big", default="videos/20260713_bigbubbles_100mA_1760mV_39deg.mp4",
                 help="100 mA/cm2 large-bubble recording")
_ap.add_argument("--big-avi", default="videos/20260713_bigbubbles_100mA_1760mV_39deg.avi",
                 help="the AVI copy of the 100 mA/cm2 recording (Figure 23)")
_ap.add_argument("--zoom", default=None,
                 help="optional: your own zoomed screen capture (Figure 26)")
_ap.add_argument("--frame", type=int, default=60,
                 help="which frame of --ans to use as the still (default 60)")
_ap.add_argument("--out", default="report_figures")
ARGS = _ap.parse_args() if __name__ == "__main__" else _ap.parse_args([])

OUT = ARGS.out
ANS_VID = ARGS.ans
BIG_VID = ARGS.big
ZOOM = ARGS.zoom


def _need(path, what):
    if not path or not os.path.exists(path):
        sys.exit(f"\n  Cannot find {what}:\n      {path}\n"
                 f"  Pass the right path, for example\n"
                 f"      python3 make_detection_figures.py --ans <path-to-688mA-video>\n")


def ans_still():
    """The nucleation-site still, taken from the recording rather than from a
    loose image file, so the figure is traceable to a named frame."""
    _need(ANS_VID, "the 688 mA/cm2 recording (--ans)")
    cap = cv2.VideoCapture(ANS_VID)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ARGS.frame)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"  Could not read frame {ARGS.frame} of {ANS_VID}")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

SCALE = 0.0064
BAND = (0.03, 0.60)


def label(img, text, h=30):
    """A caption bar above the panel, so each panel says what it is."""
    bar = np.zeros((h, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def detect(gray):
    return dp.detect_primary(gray, (0.2, 0.9), SCALE, size_frac=BAND,
                             bgr=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


def video_hits(path, wanted, every=5):
    """
    Detections for the frames in `wanted`, exactly as the Video tab produces
    them: every `every`-th frame from the start is detected, then
    temporal_consensus() reconciles each frame with its neighbours.
    """
    frames, fps, total = dv.read_clip(path, max_frames=max(wanted) + 2 * every + 1)
    idx = list(range(0, len(frames), every))
    per = [detect(frames[i]) for i in idx]
    dp.temporal_consensus(per)
    got = dict(zip(idx, per))
    return frames, {i: got[i] for i in wanted}


def old_hits(gray, sites=None, rmin=None, rmax=None):
    """The detector as it stood until November 2026: the dark-core detector
    alone, plus the bubbles the active-site stage added in a recording."""
    hits = dp.detect_primary(gray, (0.2, 0.9), SCALE, size_frac=BAND,
                             bgr=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), mode="core-seed")
    for h in (dv.detect_in_frame(gray, sites, rmin, rmax) if sites else []):
        if h["agreement"] >= 0.70 and all(
                np.hypot(h["x"] - k["x"], h["y"] - k["y"]) > max(h["r_px"], k["r_px"])
                for k in hits):
            hits.append(h)
    return hits


def _sites(frames):
    F = frames.astype(np.float32)
    H, W = F.shape[1:]
    act = dv.activity_map(F)
    strong = float(np.percentile(act, 97))
    return ([s for s in dv.find_sites(act, (H, W)) if act[s[1], s[0]] >= strong],
            max(0.5 * BAND[0] * min(H, W), 6.0), 0.5 * BAND[1] * min(H, W))


def fig_polarity():
    """
    The failure Prof. Wagner pointed at, and the fix, on one frame.

    The left panel is the detector as it stood when the interim report was
    reviewed: it assumed a bubble is darker than the plate. The right panel
    is the same frame after that assumption was removed.
    """
    g = ans_still()

    # Reproduce the old behaviour: single polarity, inside-vs-outside test.
    flat = dp.flatten(g)
    sd = float(flat.std()) + 1e-6
    from skimage.feature import blob_log
    old = []
    for y, x, s in blob_log(1.0 - flat, min_sigma=16 / np.sqrt(2),
                            max_sigma=109 / np.sqrt(2), num_sigma=14,
                            threshold=0.055, overlap=0.30):
        x, y = int(round(x)), int(round(y))
        r0 = float(s) * np.sqrt(2)
        # the original also walked the radius out to the steepest brightness
        # rise before scoring; reproduced here so the comparison is fair
        rads = np.linspace(0.55 * r0, 1.6 * r0, 30)
        prof = np.empty(30, np.float32)
        thh = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        for ii, rr in enumerate(rads):
            xi = np.clip((x + rr * np.cos(thh)).astype(int), 0, g.shape[1] - 1)
            yi = np.clip((y + rr * np.sin(thh)).astype(int), 0, g.shape[0] - 1)
            prof[ii] = np.median(flat[yi, xi])
        r0 = float(np.clip(rads[int(np.argmax(np.gradient(prof)))],
                           0.6 * r0, 1.5 * r0))
        if not (16 <= r0 <= 109):
            continue
        th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        ct, st = np.cos(th), np.sin(th)

        def ring(rr):
            xi = np.clip((x + rr * ct).astype(int), 0, g.shape[1] - 1)
            yi = np.clip((y + rr * st).astype(int), 0, g.shape[0] - 1)
            return flat[yi, xi]
        inner = np.mean([ring(f * r0) for f in (0.45, 0.60, 0.75)], axis=0)
        outer = np.mean([ring(f * r0) for f in (1.20, 1.35, 1.50)], axis=0)
        contrast = float((outer - inner).mean() / sd)
        agree = float((outer - inner > 0.15 * sd).mean())
        if contrast < 0.55 or agree < 0.80:
            continue
        old.append(dict(x=x, y=y, r_px=r0, d_mm=0.0))
    keep = []
    for c in sorted(old, key=lambda c: -c["r_px"]):
        if all(np.hypot(c["x"] - k["x"], c["y"] - k["y"])
               > 0.55 * (c["r_px"] + k["r_px"]) for k in keep):
            keep.append(c)

    new = video_hits(ANS_VID, [ARGS.frame])[1][ARGS.frame]
    a = label(dp.draw(g, keep, SCALE, show_mm=False),
              f"before: bubbles assumed darker than the plate  ({len(keep)} found)")
    b = label(dp.draw(g, new, SCALE, show_mm=False),
              f"now: whole rim fitted, as in the Video tab  ({len(new)} found)")
    cv2.imwrite(os.path.join(OUT, "detector_polarity.png"), np.hstack([a, b]))
    return len(keep), len(new)


def _jlabel(path):
    import re
    m = re.search(r"(?<!\d)(\d{2,5})\s*mA", os.path.basename(path), re.I)
    ext = os.path.splitext(path)[1].lstrip(".").upper()
    return (f"{int(m.group(1))} mA/cm2 recording" if m else "recording") + f", {ext}"


def _draw_col(g, hits, colour):
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for h in hits:
        cv2.circle(vis, (int(h["x"]), int(h["y"])), int(round(h["r_px"])), colour, 2,
                   cv2.LINE_AA)
    return vis


def fig_ans_still():
    """
    Figure 23: the old and the new detector on the same two frames, as the
    Video tab drew them. Top: the 688 mA/cm2 nucleation-site recording, frame
    --frame. Bottom: the 100 mA/cm2 recording, frame 25 (the frame of the
    screenshot that started the rewrite).
    """
    rows, counts = [], []
    for path, i in ((ANS_VID, ARGS.frame), (ARGS.big_avi if os.path.exists(ARGS.big_avi)
                                            else BIG_VID, 25)):
        if not os.path.exists(path):
            continue
        frames, new = video_hits(path, [i])
        sites, rmin, rmax = _sites(frames)
        old = old_hits(frames[i], sites, rmin, rmax)
        a = label(_draw_col(frames[i], old, (0, 0, 255)),
                  f"before: {len(old)} circles   ({_jlabel(path)}, frame {i})")
        b = label(_draw_col(frames[i], new[i], (0, 220, 0)),
                  f"now: {len(new[i])} circles, whole outlines")
        rows.append(np.hstack([a, b]))
        counts.append((len(old), len(new[i])))
    cv2.imwrite(os.path.join(OUT, "detector_ans_still.png"), np.vstack(rows))
    return counts[0][1] if counts else None, counts


def fig_zoomed():
    if not ZOOM:
        return None
    _need(ZOOM, "the zoomed screen capture (--zoom)")
    col = cv2.imread(ZOOM, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    # the colour image is passed so that a video player's coloured progress
    # bar can be recognised and cropped (Section 3.12)
    h = dp.detect_primary(g, (0.2, 0.9), SCALE, size_frac=BAND, bgr=col)
    vis = dp.draw(g, h, SCALE, show_mm=False)
    vis = cv2.resize(vis, (1000, int(1000 * vis.shape[0] / vis.shape[1])))
    cv2.imwrite(os.path.join(OUT, "detector_zoomed.png"), vis)
    return len(h), g.shape


def fig_activity():
    """
    The quantity the video path adds: how much each pixel changes with time.
    """
    _need(BIG_VID, "the 100 mA/cm2 recording (--big)")
    frames, fps, total = dv.read_clip(BIG_VID, max_frames=90)
    F = frames.astype(np.float32)
    act = F.std(axis=0)
    hot = np.clip(act / max(1.0, float(np.percentile(act, 99.5))) * 255, 0, 255)
    hot = cv2.applyColorMap(hot.astype(np.uint8), cv2.COLORMAP_INFERNO)
    a = label(cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR), "one frame")
    b = label(hot, "standard deviation of each pixel over the whole recording")
    cv2.imwrite(os.path.join(OUT, "video_activity_map.png"), np.hstack([a, b]))

    # The five bubble sites and four bare-plate points were marked by eye on
    # this particular recording, the 100 mA/cm2 large-bubble clip. They give
    # the two ranges quoted in Section 3.13. On a different recording these
    # coordinates mean nothing and the printed ranges should be ignored.
    yy, xx = np.mgrid[0:F.shape[1], 0:F.shape[2]]
    bub, plate = [], []
    for (x, y) in [(135, 205), (320, 195), (500, 180), (320, 330), (490, 345)]:
        bub.append(float(act[((xx - x) ** 2 + (yy - y) ** 2) < 35 ** 2].mean()))
    for (x, y) in [(80, 400), (250, 430), (560, 440), (420, 120)]:
        plate.append(float(act[((xx - x) ** 2 + (yy - y) ** 2) < 35 ** 2].mean()))
    return (min(bub), max(bub)), (min(plate), max(plate))


def fig_video_frames():
    _need(ANS_VID, "the 688 mA/cm2 recording (--ans)")
    frames, got = video_hits(ANS_VID, [0, 50, 100])
    panels, counts = [], []
    for i in (0, 50, 100):
        g = frames[i]
        hits = got[i]
        counts.append(len(hits))
        panels.append(label(dp.draw(g, hits, SCALE, show_mm=False),
                            f"frame {i}   {len(hits)} bubbles"))
    cv2.imwrite(os.path.join(OUT, "video_frames.png"), np.hstack(panels))
    return counts


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("detector_polarity.png    before/after:", fig_polarity())
    n, pairs = fig_ans_still()
    print("detector_ans_still.png   bubbles:", n)
    print("detector_ans_still.png   old/new per row:", pairs)
    print("detector_zoomed.png      bubbles, size:", fig_zoomed())
    print("video_activity_map.png   bubble sd range, plate sd range:", fig_activity())
    print("video_frames.png         per frame:", fig_video_frames())
