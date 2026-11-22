"""
measure_detection.py
====================
Prints the two detection tables of Section 4.10 from your own files, so every
number in that section can be re-measured rather than taken on trust.

    python3 measure_detection.py

Defaults (relative to backend/, change with the options below):

    --ans    session_C/bubble_video_2283mV_688mA_41deg.avi
    --big    videos/20260713_bigbubbles_100mA_1760mV_39deg.mp4
    --big-avi videos/20260713_bigbubbles_100mA_1760mV_39deg.avi   (optional)
    --cap860 screenshots/Screenshot_2026-08-25_211925_bubbles_flattend_860mA_43deg.png
    --cap721 screenshots/Screenshot_2026-08-25_212403_bubbles_flattend_721mA_40deg.png

The "bubbles present" column is a count made by eye, written into the table
below. Everything else is measured when the script runs.
"""
import argparse
import re
import os
import time

import cv2
import numpy as np

import detect_primary as dp
import detect_video as dv
from constants import MM_PER_PIXEL

ap = argparse.ArgumentParser()
ap.add_argument("--ans", default="session_C/bubble_video_2283mV_688mA_41deg.avi")
ap.add_argument("--big", default="videos/20260713_bigbubbles_100mA_1760mV_39deg.mp4")
ap.add_argument("--big-avi", default="videos/20260713_bigbubbles_100mA_1760mV_39deg.avi")
ap.add_argument("--cap860", default="screenshots/Screenshot_2026-08-25_211925_bubbles_flattend_860mA_43deg.png")
ap.add_argument("--cap721", default="screenshots/Screenshot_2026-08-25_212403_bubbles_flattend_721mA_40deg.png")
ap.add_argument("--save", default="detection_check",
                help="folder for the annotated frames, so each count can be checked by eye")
a = ap.parse_args()
os.makedirs(a.save, exist_ok=True)


def _key(name):
    # 'Screenshot 2026-08-25 212403_...' (as macOS saves it) and
    # 'Screenshot_2026-08-25_212403_...' are the same file for this script
    return re.sub(r"[\s_]+", "_", os.path.basename(name)).lower()


def resolve(path):
    """The given path, or the same file name with spaces instead of
    underscores, in its own folder, screenshots/, videos/, backend/ or the
    project folder."""
    if os.path.exists(path):
        return path
    for d in (os.path.dirname(path) or ".", "screenshots", "videos", ".", ".."):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if _key(f) == _key(path):
                    return os.path.join(d, f)
    return path


for k in ("ans", "big", "big_avi", "cap860", "cap721"):
    setattr(a, k, resolve(getattr(a, k)))

# bubbles counted by eye in each frame (whole bubbles; bubbles cut by the
# frame edge are not counted)
PRESENT = {"cap860": 2, "cap721": 4, "ans60": 12, "big0": 5, "big45": 4, "big85": 4}


# The application's default: smallest bubble 3 %, largest 60 % of the shorter
# image side, the same band as Tabs 1 and 3 and make_detection_figures.py.
BAND = (0.03, 0.60)


def detect(gray, bgr=None):
    return dp.detect_primary(gray, dp.DEFAULT_MM, MM_PER_PIXEL, size_frac=BAND,
                             bgr=bgr)


def save(name, bgr, hits):
    v = bgr.copy()
    t = max(2, v.shape[1] // 320)
    for h in hits:
        cv2.circle(v, (int(h["x"]), int(h["y"])), int(round(h["r_px"])), (0, 220, 0), t)
    cv2.putText(v, f"{name}: {len(hits)}", (10, v.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t / 2, (0, 255, 255), t)
    cv2.imwrite(os.path.join(a.save, name + ".png"), v)


rows = []
for key, path, label in (("cap860", a.cap860, "860 mA/cm2 screen capture"),
                         ("cap721", a.cap721, "721 mA/cm2 screen capture")):
    if os.path.exists(path):
        im = cv2.imread(path)
        h = detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), im)
        save(key, im, h)
        rows.append((label, f"{im.shape[1]} x {im.shape[0]}", PRESENT[key], len(h)))
    else:
        rows.append((label, "file not found", PRESENT[key], None))

if os.path.exists(a.ans):
    fr, fps, _ = dv.read_clip(a.ans, max_frames=61)
    h = detect(fr[60])
    save("ans60", cv2.cvtColor(fr[60], cv2.COLOR_GRAY2BGR), h)
    rows.append(("688 mA/cm2 nucleation sites, frame 60", "640 x 480", PRESENT["ans60"], len(h)))

for src, tag in ((a.big, "MP4"), (a.big_avi, "AVI")):
    if not os.path.exists(src):
        continue
    fr, fps, _ = dv.read_clip(src, max_frames=90)
    for i in (0, 45, 85):
        h = detect(fr[i])
        save(f"big{i}_{tag}", cv2.cvtColor(fr[i], cv2.COLOR_GRAY2BGR), h)
        rows.append((f"100 mA/cm2 {tag}, frame {i}", "640 x 480", PRESENT[f"big{i}"], len(h)))

# ring mode on the same three frames of the 100 mA/cm2 MP4 (Section 3.12)
ring = []
if os.path.exists(a.big):
    fr, fps, _ = dv.read_clip(a.big, max_frames=90)
    for i in (0, 45, 85):
        g = fr[i]
        h = dp.detect_primary(g, dp.DEFAULT_MM, MM_PER_PIXEL, size_frac=(0.13, 0.60),
                              mode="ring", bgr=cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        save(f"ring{i}_MP4", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), h)
        ring.append(len(h))

print("\nSINGLE FRAMES  (detector 'Bubbles on the plate - dark centre')")
print("-" * 78)
print(f"  {'frame':<40} {'size':<12} {'present':>8} {'circles':>8}")
for r in rows:
    print(f"  {r[0]:<40} {r[1]:<12} {r[2]:>8} {'-' if r[3] is None else r[3]:>8}")
if ring:
    print(f"  ring mode, 100 mA/cm2 MP4, frames 0/45/85: {ring} circles")
print(f"\n  annotated frames written to {a.save}/  - check each count by eye")

# --- recordings, through the same function the Video tab calls -------------
import main as app

recs = []
print("\nRECORDINGS  (Video tab: every 10th frame, same detector)")
print("-" * 78)
for path, j, n in ((a.ans, 688, 12), (a.big, 100, 9), (a.big_avi, 100, 9)):
    if not os.path.exists(path):
        continue
    t = time.time()
    res, _, fps, total, sites = app._analyse_recording(
        path, j, "ans", 2.54, "primary", 3.0, 60.0, 10, n)
    c = [r["bubbles"] for r in res]
    recs.append((os.path.basename(path), c))
    print(f"  {os.path.basename(path):<48} {c}  mean {np.mean(c):.2f}  "
          f"{time.time() - t:.1f} s")
print()
