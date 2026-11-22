"""
make_filter_figures.py
======================
Produces the two figures that show the tiny-bubble filter working.

    python3 make_filter_figures.py <video.avi>

Example:
    python3 make_filter_figures.py session_C/bubble_video_2283mV_688mA_41deg.avi

Writes into ./report_figures/ :
    filter_before_after.png    original | unfiltered N(x,y) | filtered | diameter
    anchored_sites_final.png   the surviving nucleation sites on the frame
    background_reconstruction.png   the computed background and a subtracted frame
    computed_background.png    the median of all frames on its own      (Figure 7)
    background_subtracted.png  one frame minus that background, x4      (Figure 8)

These correspond to the figures in Section 4.3 and 4.4 of the report.
"""
import os
import re
import sys
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from video_analysis import (
    load_video_gray, compute_background, auto_threshold,
    classify_anchored_bubbles,
)
from fisher_analysis import (
    extract_bubble_table, bubble_frequency_map, mean_diameter_map,
)

OUT = "report_figures"
CELL = 20


def current_density_from_name(path):
    m = re.search(r"(?<!\d)(\d{2,5})\s*mA", os.path.basename(path), re.I)
    return int(m.group(1)) if m else 0


def main(path):
    if not os.path.exists(path):
        print("ERROR: file not found: " + path)
        sys.exit(1)

    os.makedirs(OUT, exist_ok=True)
    j = current_density_from_name(path)
    name = os.path.basename(path)
    print("Processing " + name + "  (j = " + str(j) + " mA/cm2)")

    frames, fps = load_video_gray(path)
    bg = compute_background(frames)
    thr = auto_threshold(frames, bg=bg)
    mid = frames[len(frames) // 2]
    print("  frames=%d  fps=%.1f  threshold=%.1f" % (len(frames), fps, thr))

    rows_raw, shape, _ = extract_bubble_table(path, j, filter_tiny=False)
    rows_filt, _, n_sites = extract_bubble_table(path, j, filter_tiny=True)
    removed = 100.0 * (1 - len(rows_filt) / max(1, len(rows_raw)))
    print("  raw=%d  filtered=%d  sites=%s  removed=%.1f%%"
          % (len(rows_raw), len(rows_filt), n_sites, removed))

    # ── Figure: before / after filtering ──────────────────────────────────
    N_raw = bubble_frequency_map(rows_raw, shape, CELL)
    N_filt = bubble_frequency_map(rows_filt, shape, CELL)
    dm_filt = mean_diameter_map(rows_filt, shape, CELL)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9.6))
    ax = ax.ravel()
    ax[0].imshow(mid, cmap="gray")
    ax[0].set_title("Original frame\n(%d mA/cm$^2$)" % j, fontsize=14)

    im = ax[1].imshow(N_raw, cmap="viridis")
    ax[1].set_title("N(x,y), unfiltered\n%d raw detections" % len(rows_raw), fontsize=14)
    plt.colorbar(im, ax=ax[1], fraction=0.046)

    im = ax[2].imshow(N_filt, cmap="viridis")
    ax[2].set_title("N(x,y), small drifting bubbles removed\n%d records, %s anchored sites"
                    % (len(rows_filt), n_sites), fontsize=14)
    plt.colorbar(im, ax=ax[2], fraction=0.046)

    im = ax[3].imshow(dm_filt, cmap="plasma")
    ax[3].set_title("Mean diameter, filtered [mm]", fontsize=14)
    plt.colorbar(im, ax=ax[3], fraction=0.046)

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle("Anchored-site filter: " + name, fontsize=16.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "filter_before_after.png"), dpi=145,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  wrote " + OUT + "/filter_before_after.png")

    # ── Figure: the surviving nucleation sites ────────────────────────────
    sites = classify_anchored_bubbles(frames, bg, thr, fps)
    ov = cv2.cvtColor(mid, cv2.COLOR_GRAY2BGR)
    for s in sites:
        cv2.circle(ov, (int(s["x"]), int(s["y"])), int(s["r"]), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT, "anchored_sites_final.png"),
                cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
    print("  wrote " + OUT + "/anchored_sites_final.png  (%d sites)" % len(sites))

    # ── Figure: background reconstruction ─────────────────────────────────
    diff = cv2.absdiff(mid, bg)
    amp = np.clip(diff.astype(np.float32) * 4, 0, 255).astype(np.uint8)

    # The two panels the report shows on their own (Figures 7 and 8).
    # They are the computed arrays written straight to disk, nothing drawn.
    cv2.imwrite(os.path.join(OUT, "computed_background.png"), bg)
    cv2.imwrite(os.path.join(OUT, "background_subtracted.png"), amp)
    print("  wrote " + OUT + "/computed_background.png")
    print("  wrote " + OUT + "/background_subtracted.png  (contrast x4)")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    ax[0].imshow(mid, cmap="gray")
    ax[0].set_title("Original frame", fontsize=14)
    ax[1].imshow(bg, cmap="gray")
    ax[1].set_title("Reconstructed electrode\n(median of all frames)", fontsize=14)
    ax[2].imshow(amp, cmap="gray")
    ax[2].set_title("After background subtraction\n(contrast x4 for visibility)", fontsize=14)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle("Temporal background subtraction - " + name, fontsize=16.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "background_reconstruction.png"), dpi=145,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  wrote " + OUT + "/background_reconstruction.png")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:  python3 make_filter_figures.py <video file>")
        sys.exit(1)
    main(sys.argv[1])
