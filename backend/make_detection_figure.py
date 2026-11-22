"""
make_detection_figure.py
========================
Produces the figure that answers Prof. Wagner's central criticism:

    "the method does not seem to have been fine-tuned yet. I can see
     multiple bubbles without detection circle."

    python3 make_detection_figure.py

It runs detect_primary.py on both electrode types and lays the results out
side by side, so that whether the circles sit on real bubbles can be judged
by looking rather than by reading a number.

Expects, in the working folder:

    session_C/bubble_video_2283mV_688mA_41deg.avi    (or any ANS recording)
    bubble_density_600mA_02.jpg                      (the GDE still)

Missing inputs are skipped with a message rather than crashing, so the
figure can still be produced from whichever data is to hand.
"""

import os
import glob

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detect_primary import (read_frames, detect_primary, flatten,
                            measure_pitch, DEFAULT_MM)

try:
    from constants import mm_per_pixel
except ImportError:
    def mm_per_pixel(source="image"):
        return 0.0064

OUT = "report_figures"


def find(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def panel(ax, gray, hits, title, scale):
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for c in hits:
        ax.add_patch(plt.Circle((c["x"], c["y"]), c["r_px"], fill=False,
                                edgecolor="#22c55e", linewidth=1.6))
        ax.plot(c["x"], c["y"], "+", color="#ef4444", ms=6, mew=1.4)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def size_hist(ax, hits, scale, title):
    if not hits:
        ax.text(.5, .5, "no detections", ha="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
        return
    d = np.array([2*c["r_px"] for c in hits])
    ax.hist(d, bins=12, color="#2563eb", edgecolor="white")
    ax.set_xlabel("bubble diameter (pixels)", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=9)
    sec = ax.secondary_xaxis("top", functions=(lambda v: v*scale,
                                               lambda v: v/scale))
    sec.set_xlabel("diameter (mm) at the current calibration", fontsize=9)
    sec.tick_params(labelsize=8)


def main():
    os.makedirs(OUT, exist_ok=True)
    scale = mm_per_pixel("video")

    ans = find("session_C/*.avi", "session_C/*.mp4",
               "sessionC/*.avi", "*2283mV*.avi")
    gde = find("bubble_density_600mA*.jpg", "bubble_density*.jpg", "*.jpg")

    items = []

    if ans:
        frames = read_frames(ans, max_frames=200)
        idx = min(60, len(frames)-1)
        g = (frames[idx]*255).astype(np.uint8)
        hits = detect_primary(g, DEFAULT_MM, scale)
        pitch, _, _ = measure_pitch(frames)
        items.append(("Artificial nucleation sites, 688 mA/cm$^2$", g, hits, pitch))
        print(f"  ANS  : {len(hits)} bubbles, bore pitch {pitch:.1f} px")
    else:
        print("  ANS recording not found - skipping that panel")

    if gde:
        g2 = cv2.imread(gde, cv2.IMREAD_GRAYSCALE)
        hits2 = detect_primary(g2, DEFAULT_MM, scale)
        items.append(("Gas diffusion electrode, 600 mA/cm$^2$", g2, hits2, None))
        print(f"  GDE  : {len(hits2)} bubbles of the large class "
              f"(the frame holds of order a hundred in total)")
    else:
        print("  GDE still not found - skipping that panel")

    if not items:
        print("\nNothing to draw. Put the ANS recording and the GDE still "
              "beside this script.")
        return

    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(6.2*n, 10), squeeze=False)
    for k, (title, g, hits, pitch) in enumerate(items):
        sub = title
        if pitch:
            sub += f"\nbore pitch measured at {pitch:.0f} px"
        panel(axes[0][k], g, hits, f"{sub}\n{len(hits)} bubbles detected", scale)
        size_hist(axes[1][k], hits, scale, "Size distribution of the detections")

    fig.suptitle("Detection of the large bubbles at nucleation sites\n"
                 "Every circle is verified against its own radial brightness "
                 "profile, not just voted for",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT, "primary_detection.png")
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("\nWrote " + path)


if __name__ == "__main__":
    main()
