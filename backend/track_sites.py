"""
track_sites.py
==============
Replaces the old nucleation-site figure (Figure 10 of the interim report).

WHY
---
The old figure was produced by the anchored-bubble filter, which worked on
blobs from a brightness threshold. On the artificial-nucleation-site electrode
most of its circles landed on machining streaks rather than on bubbles, and
several obvious bubbles carried no circle at all. That is the figure
Prof. Wagner was looking at when he wrote that bubbles were not captured well,
and it is not defensible.

This script rebuilds the same figure from detect_primary.py, which verifies
every circle against its own radial brightness profile, and adds the thing the
old figure could not show: how often each site is actually occupied.

WHAT IT DOES
------------
    1. Run the verified detector on every Nth frame.
    2. Cluster the detections by position. A nucleation site is a position
       that keeps producing detections; a one-off is noise.
    3. Report, per site, how many frames it was occupied in and the median
       diameter of the bubble sitting there.
    4. Draw the sites on the frame, with the circle at the measured size.

Occupancy is the honest quantity here. A bore whose bubble has just detached
has nothing to detect, so a site missing from one frame is physics rather than
a detector failure. Reporting occupancy says that out loud instead of hiding
it behind a single-frame count.

USAGE
-----
    python3 track_sites.py <video or folder> [more ...] [--every 5] [--max 300]
                           [--min-occ 0.25]

With several recordings (Figure 10 of the final report), every recording of
the bored plate - recognised by a bore grid within 10 % of 155.5 px - gets a small
site map, and one summary panel compares them across current density: each
site's median bubble diameter and the mean occupancy of the sites.
"""

import glob
import os
import re
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detect_primary import (read_frames, detect_primary, measure_pitch,
                            temporal_consensus, DEFAULT_MM, VIDEO_EXT)

try:
    from constants import mm_per_pixel
except ImportError:
    def mm_per_pixel(source="image"):
        return 0.0064

OUT = "report_figures"


def cluster_sites(dets, merge_px=40.0):
    """
    Group detections across frames into physical sites.

    A detection joins the nearest existing site if it is within merge_px of
    it, otherwise it starts a new one. merge_px is deliberately well below
    the bore pitch, so two neighbouring bores cannot merge into one site.
    """
    sites = []
    for d in dets:
        best, bd = None, 1e9
        for s in sites:
            dist = np.hypot(d["x"] - s["x"], d["y"] - s["y"])
            if dist < bd:
                best, bd = s, dist
        if best is not None and bd <= merge_px:
            best["hits"].append(d)
            n = len(best["hits"])
            best["x"] += (d["x"] - best["x"]) / n      # running mean position
            best["y"] += (d["y"] - best["y"]) / n
        else:
            sites.append(dict(x=float(d["x"]), y=float(d["y"]), hits=[d]))
    return sites


PITCH_OK = (140.0, 171.0)   # within 10 % of the 155.5 px bore pitch of the ANS plate


def track(path, every=5, maxf=300, min_occ=0.25, log=print):
    """
    Nucleation sites of one recording. Returns a dict with the sites (position,
    occupancy, median diameter in px and mm), the bore pitch, the middle frame
    and the current density read from the file name, or None if the file
    cannot be read.

    Detection is the Video tab's: detect_primary on every `every`-th frame,
    then temporal_consensus() across the sampled frames.
    """
    scale = mm_per_pixel("video")
    try:
        frames = read_frames(path, max_frames=maxf)
    except IOError:
        return None
    pitch, _, _ = measure_pitch(frames) if len(frames) > 20 else (None, None, None)
    used = list(range(0, len(frames), every))
    per = []
    for i in used:
        g = (frames[i] * 255).astype(np.uint8)
        per.append(detect_primary(g, DEFAULT_MM, scale))
    temporal_consensus(per)
    allde = []
    for i, hs in zip(used, per):
        for d in hs:
            d["frame"] = i
            allde.append(d)

    merge = 0.25 * pitch if pitch else 40.0
    sites = cluster_sites(allde, merge_px=merge)
    for s_ in sites:
        # Count DISTINCT frames, not detections. Two detections of the same
        # bubble in one frame are one occupied frame, and counting hits
        # instead produced occupancies above 100 %, which is nonsense.
        s_["occ"] = len({h["frame"] for h in s_["hits"]}) / len(used)
        s_["d_px"] = float(np.median([2 * h["r_px"] for h in s_["hits"]]))
        s_["d_mm"] = s_["d_px"] * scale
    sites = [s_ for s_ in sites if s_["occ"] >= min_occ]

    # A bubble wider than the spacing between bores would touch the bubble on
    # the neighbouring bore, so a detection of that width spans two sites.
    # (The limit was 0.55 of the pitch while circles marked only the dark
    # core; with whole outlines the large domes reach 0.75 of it.)
    toobig = []
    if pitch:
        toobig = [s_ for s_ in sites if s_["d_px"] > pitch]
        sites = [s_ for s_ in sites if s_["d_px"] <= pitch]
    sites.sort(key=lambda s_: -s_["occ"])

    m = re.search(r"(?<!\d)(\d{2,5})\s*mA", os.path.basename(path), re.I)
    log(f"  {os.path.basename(path)}: {len(frames)} frames, every {every}; "
        + (f"bore pitch {pitch:.1f} px; " if pitch else "no bore grid; ")
        + f"{len(sites)} sites occupied in at least {min_occ:.0%} of frames"
        + (f", {len(toobig)} rejected as wider than the pitch" if toobig else ""))
    return dict(path=path, name=os.path.basename(path), j=int(m.group(1)) if m else None,
                pitch=pitch, sites=sites, used=len(used), scale=scale,
                mid=(frames[len(frames) // 2] * 255).astype(np.uint8))


def site_map(ax, r, title_size=12, label=True):
    ax.imshow(r["mid"], cmap="gray", vmin=0, vmax=255)
    for s_ in r["sites"]:
        ax.add_patch(plt.Circle((s_["x"], s_["y"]), s_["d_px"] / 2, fill=False,
                                edgecolor="#22c55e", linewidth=1.8))
        if label:
            ax.text(s_["x"], s_["y"] - s_["d_px"] / 2 - 7, f"{s_['occ']:.0%}",
                    color="#fde047", fontsize=11, ha="center")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    def arg(flag, default, cast=int):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    every = arg("--every", 5)
    maxf = arg("--max", 300)
    min_occ = arg("--min-occ", 0.25, float)
    skip = {"--every", "--max", "--min-occ"}
    targets = [a for k, a in enumerate(sys.argv[1:], 1)
               if not a.startswith("--") and sys.argv[k - 1] not in skip]
    paths = []
    for t in targets:
        if os.path.isdir(t):
            paths += sorted(p for p in glob.glob(os.path.join(t, "*"))
                            if p.lower().endswith(VIDEO_EXT))
        elif os.path.exists(t):
            paths.append(t)

    print("NUCLEATION SITE TRACKING")
    print("=" * 70)
    res = [r for r in (track(p, every, maxf, min_occ) for p in paths) if r]
    # only recordings of the bored (ANS) plate: a measurable bore grid
    ans = [r for r in res if r["pitch"] and PITCH_OK[0] <= r["pitch"] <= PITCH_OK[1]
           and r["j"] is not None]
    ans.sort(key=lambda r: (r["j"], r["name"]))
    if not ans:
        print("  no recording with a bore grid of 140 to 171 px")
        return

    print(f"\n  {len(ans)} recording(s) of the bored plate")
    print(f"  {'recording':<44} {'j':>6} {'sites':>6} {'mean occ.':>10} "
          f"{'median d':>14}")
    for r in ans:
        d = [s_["d_px"] for s_ in r["sites"]]
        occ = [s_["occ"] for s_ in r["sites"]]
        r["d_med_mm"] = float(np.median(d)) * r["scale"] if d else float("nan")
        r["occ_mean"] = float(np.mean(occ)) if occ else float("nan")
        print(f"  {r['name'][:44]:<44} {r['j']:>6} {len(r['sites']):>6} "
              f"{r['occ_mean']:>9.0%} {np.median(d) if d else float('nan'):>7.0f} px "
              f"{r['d_med_mm']:>5.2f} mm")
    ref = next((r for r in ans if "688" in r["name"]), ans[0])
    print(f"\n  reference recording {ref['name']}: {len(ref['sites'])} sites occupied "
          f"in at least {min_occ:.0%} of frames")
    for k, s_ in enumerate(ref["sites"], 1):
        print(f"  {k:>5} {s_['x']:>5.0f} {s_['y']:>5.0f} {s_['occ']:>9.0%} "
              f"{s_['d_px']:>8.0f} px {s_['d_mm']:>6.2f} mm")

    # ---- one picture: a site map per recording and the summary --------------
    n = len(ans)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3.6 * cols + 0.4, 2.9 * rows + 5.6))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[2.9] * rows + [5.0])
    for k, r in enumerate(ans):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        site_map(ax, r, label=n <= 6)
        ax.set_title(f"{r['j']} mA/cm$^2$: {len(r['sites'])} sites, "
                     f"d = {r['d_med_mm']:.2f} mm", fontsize=12.5)
    ax = fig.add_subplot(gs[rows, :])
    for r in ans:
        d = np.array([s_["d_px"] for s_ in r["sites"]]) * r["scale"]
        ax.plot(np.full(len(d), r["j"]), d, "o", color="#22c55e", alpha=0.45, ms=6)
    ax.plot([r["j"] for r in ans], [r["d_med_mm"] for r in ans], "s", color="#15803d",
            ms=10, label="median bubble diameter over the sites (left axis)")
    ax.set_xlabel("Current density (mA/cm$^2$)", fontsize=15)
    ax.set_ylabel("bubble diameter at a site (mm)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot([r["j"] for r in ans], [100 * r["occ_mean"] for r in ans], "D",
             color="#f59e0b", ms=9, label="mean occupancy of the sites (right axis)")
    ax2.set_ylabel("mean occupancy (%)", fontsize=15)
    ax2.set_ylim(0, 105)
    ax2.tick_params(labelsize=13)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=12.5, loc="lower left")
    ax.set_title("Each dot is one site (its median diameter over the recording); "
                 "no line connects measured points", fontsize=13.5)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "nucleation_sites.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(OUT, "nucleation_sites_values.txt"), "w") as fh:
        for r in ans:
            fh.write(f"{r['name']}\t{r['j']}\t{len(r['sites'])}\t"
                     f"{r['occ_mean']:.3f}\t{r['d_med_mm']:.3f}\n")
    print("\nWrote " + out)


if __name__ == "__main__":
    main()
