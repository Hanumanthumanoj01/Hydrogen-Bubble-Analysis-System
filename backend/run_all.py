"""
run_all.py
==========
ONE COMMAND to process an entire laboratory data folder.

    python3 run_all.py <folder>

Processes every video (.avi/.mp4/.mov) and image (.png/.jpg) in the folder,
reads the operating conditions from each filename, and writes a complete
result set to ./results/.

Addresses Prof. Wagner's request of 25 August:
    "if you plot data tables or computational results, it will be helpful if
     you add a screenshot of the original bubble image for visual correlation"

Every figure produced here shows the ORIGINAL frame beside the computed
result, so each number can be checked against the picture it came from.

Filename convention (any order, case-insensitive):
    ..._<current>mA_..._<voltage>mV_..._<temperature>deg.<ext>
    e.g.  bubble_video_2283mV_688mA_41deg.avi
          Screenshot_..._bubbles_flattend_860mA_43deg.png
Decimal temperatures such as 44.5deg are handled.
Only the current density (NNNmA) is strictly required.

Outputs written to results/:
    per_file/<name>_panel.png     original + overlay + maps, side by side
    summary.csv                   one row per file, all metrics
    entropy_vs_current.png        with sample images inset
    gibbs_correlation.png         information entropy vs thermodynamic entropy
    fisher/                       Fisher-information maps (videos only)
    bubbles_<name>.csv            the (x, y, d, t, j) tables
"""
from __future__ import annotations

import os
import re
import sys
import glob
import csv
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import FARADAY, Z_ELECTRONS, MOLAR_VOLUME_H2, V_THERMONEUTRAL, mm_per_pixel
from bubble_analysis import BubbleAnalyzer
from video_analysis import (
    load_video_gray, compute_background, auto_threshold,
    detect_bubbles_bgsub, shannon_entropy,
)
from fisher_analysis import (
    bubble_frequency_map, mean_diameter_map,
    build_state_distribution, fisher_information, export_map_csv,
)

VIDEO_EXT = ("*.avi", "*.AVI", "*.mp4", "*.MP4", "*.mov", "*.MOV")
IMAGE_EXT = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
CELL = 20


# ────────────────────────────────────────────────────────────────────────────
def parse_conditions(path: str):
    """Extract (current_density, voltage_mV, temperature_C) from a filename."""
    n = os.path.basename(path)
    j = re.search(r"(?<!\d)(\d{2,5})\s*mA", n, re.I)
    v = re.search(r"(\d{3,5})\s*mV", n, re.I)
    t = re.search(r"(\d{1,3}(?:[.,]\d+)?)\s*deg", n, re.I)   # allows 44.5deg
    return (int(j.group(1)) if j else None,
            int(v.group(1)) if v else None,
            float(t.group(1).replace(",", ".")) if t else None)


def gibbs_entropy(voltage_mV, temp_C):
    """Irreversible entropy production  dS = z F eta / T  [J/(mol K)]."""
    if voltage_mV is None or temp_C is None:
        return None
    eta = voltage_mV / 1000.0 - V_THERMONEUTRAL
    return Z_ELECTRONS * FARADAY * eta / (temp_C + 273.15)


def h2_rate(j_mA_cm2, area_cm2=2.54):
    return (j_mA_cm2 / 1000.0) * area_cm2 / (Z_ELECTRONS * FARADAY) * MOLAR_VOLUME_H2 * 60.0


# ────────────────────────────────────────────────────────────────────────────
def process_video(path, j, out_dir, camera_fps=0.0):
    """Full video pipeline. Returns (metrics dict, representative frame, bubbles, rows)."""
    frames, fps = load_video_gray(path)
    if camera_fps and camera_fps > 0:
        # the files store 24 fps; the camera rate is printed in the frame
        fps = float(camera_fps)
    bg = compute_background(frames)
    thr = auto_threshold(frames, bg=bg)
    scale = mm_per_pixel("video")

    rows, per_frame = [], []
    for idx in range(len(frames)):
        t = idx / fps
        dets = detect_bubbles_bgsub(frames[idx], bg, thr)
        per_frame.append(len(dets))
        for (x, y, r) in dets:
            rows.append({"x": float(x), "y": float(y),
                          "d_px": float(2 * r), "d_mm": float(2 * r * scale),
                          "t": float(t), "j": float(j)})

    mid = len(frames) // 2
    mid_dets = detect_bubbles_bgsub(frames[mid], bg, thr)

    Hs = [shannon_entropy(f) for f in frames]
    H = float(np.mean([h[0] for h in Hs]))
    Hn = float(np.mean([h[1] for h in Hs])) * 100
    d = np.array([r["d_mm"] for r in rows]) if rows else np.array([0.0])

    return ({
        "kind": "video", "frames": len(frames), "fps": round(fps, 2),
        "threshold": round(thr, 2),
        "mean_bubbles": round(float(np.mean(per_frame)), 2),
        "max_bubbles": int(np.max(per_frame)), "min_bubbles": int(np.min(per_frame)),
        "H": round(H, 4), "H_norm_pct": round(Hn, 2),
        "mean_diam_mm": round(float(d.mean()), 4),
        "total_records": len(rows),
    }, frames[mid], mid_dets, rows, bg, frames[0].shape)


def process_image(path, j, out_dir):
    img = cv2.imread(path)
    if img is None:
        return None
    a = BubbleAnalyzer()
    r = a.full_analysis(img, float(j or 0), "ans", 2.54, "medium")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bubbles = [(b["x"], b["y"], b["r_px"]) for b in r["bubbles"]]
    return ({
        "kind": "image", "frames": 1, "fps": "",
        "threshold": "",
        "mean_bubbles": r["bubble_count"],
        "max_bubbles": r["bubble_count"], "min_bubbles": r["bubble_count"],
        "H": r["shannon_entropy"], "H_norm_pct": r["shannon_entropy_norm"],
        "mean_diam_mm": r["mean_diameter_mm"],
        "total_records": r["bubble_count"],
    }, gray, bubbles, [], None, gray.shape)


# ────────────────────────────────────────────────────────────────────────────
def make_panel(name, frame, bubbles, rows, shape, meta, out_path):
    """Original image + detection overlay + N map + diameter map, side by side.

    This is the 'visual correlation' Prof. Wagner asked for: no number is
    shown without the picture it was computed from.
    """
    has_maps = len(rows) > 0
    ncol = 4 if has_maps else 2
    # Two rows when there are four panels, so that the lettering stays
    # legible at the printed width of the report.
    if ncol == 4:
        fig, ax = plt.subplots(2, 2, figsize=(11, 9.4))
        ax = ax.ravel()
    else:
        fig, ax = plt.subplots(1, ncol, figsize=(5.5 * ncol, 5))

    ax[0].imshow(frame, cmap="gray")
    ax[0].set_title("Original (mid-frame)", fontsize=14)

    ov = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for (x, y, r) in bubbles:
        cv2.circle(ov, (int(x), int(y)), max(2, int(r)), (255, 40, 40), 2)
    ax[1].imshow(ov)
    ax[1].set_title(f"Detected: {len(bubbles)} bubbles", fontsize=14)

    if has_maps:
        N = bubble_frequency_map(rows, shape, CELL)
        dm = mean_diameter_map(rows, shape, CELL)
        im = ax[2].imshow(N, cmap="viridis")
        ax[2].set_title("Step 1: N(x,y)  bubble frequency", fontsize=14)
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        im = ax[3].imshow(dm, cmap="plasma")
        ax[3].set_title("Step 2: mean diameter [mm]", fontsize=14)
        plt.colorbar(im, ax=ax[3], fraction=0.046)

    for a in ax:
        a.set_xticks([]); a.set_yticks([])

    sub = (f"{meta.get('j','?')} mA/cm²"
           + (f" · {meta['V']} mV" if meta.get("V") else "")
           + (f" · {meta['T']} °C" if meta.get("T") else "")
           + f"   |   H = {meta['H']}, H_norm = {meta['H_norm_pct']} %"
           + f", ⌀ = {meta['mean_diam_mm']} mm")
    plt.suptitle(f"{name}\n{sub}", fontsize=15.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def make_overview(results, out_dir):
    """Entropy vs current density, and the Gibbs correlation — both with
    thumbnails of the original images so the trend can be seen visually."""
    pts = [r for r in results if r["j"] is not None]
    if len(pts) < 2:
        return
    pts.sort(key=lambda r: r["j"])

    # --- entropy vs current density, with image thumbnails -----------------
    fig, ax = plt.subplots(figsize=(11, 6))
    js = [p["j"] for p in pts]
    hs = [p["H_norm_pct"] for p in pts]
    ax.plot(js, hs, "o-", color="#1f77b4", lw=2, ms=7)
    ax.set_xlabel("Current density (mA/cm²)")
    ax.set_ylabel("H_norm (%)")
    ax.set_title("Information entropy vs current density")
    ax.grid(alpha=.3)

    # inset thumbnails of the actual frames, evenly spaced along the curve
    step = max(1, len(pts) // 5)
    for p in pts[::step]:
        if p.get("thumb") is None:
            continue
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        th = cv2.resize(p["thumb"], (70, 70))
        ab = AnnotationBbox(OffsetImage(th, cmap="gray", zoom=1.0),
                            (p["j"], p["H_norm_pct"]),
                            xybox=(0, 55), xycoords="data",
                            boxcoords="offset points", frameon=True, pad=0.2)
        ax.add_artist(ab)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_vs_current.png"), dpi=140)
    plt.close()

    # --- Gibbs correlation --------------------------------------------------
    gp = [p for p in pts if p.get("dS") is not None]
    if len(gp) >= 3:
        dS = np.array([p["dS"] for p in gp])
        Hn = np.array([p["H_norm_pct"] for p in gp])
        r = float(np.corrcoef(dS, Hn)[0, 1])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dS, Hn, "o", color="#d62728", ms=9)
        z = np.polyfit(dS, Hn, 1)
        xs = np.linspace(dS.min(), dS.max(), 50)
        ax.plot(xs, np.polyval(z, xs), "--", color="gray")
        for p in gp:
            ax.annotate(f"{p['j']} mA", (p["dS"], p["H_norm_pct"]),
                        textcoords="offset points", xytext=(7, -11), fontsize=11.2)
        ax.set_xlabel("Gibbs entropy production  zFη/T  (J mol⁻¹ K⁻¹)")
        ax.set_ylabel("Information entropy H_norm (%)")
        ax.set_title(f"Information vs thermodynamic entropy\nr = {r:.3f},  R² = {r**2:.3f},  n = {len(gp)}")
        ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "gibbs_correlation.png"), dpi=140)
        plt.close()
        print(f"\n   Gibbs correlation: r = {r:.4f}, R² = {r**2:.4f}  (n = {len(gp)})")


# ────────────────────────────────────────────────────────────────────────────
def main(folder, out_dir="results", camera_fps=0.0):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "per_file"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "fisher"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    paths = []
    for pat in VIDEO_EXT + IMAGE_EXT:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    paths = sorted(set(paths))
    if not paths:
        print(f"No videos or images found in: {folder}")
        return

    print(f"Found {len(paths)} files in {folder}\n")
    results, video_rows_by_j = [], {}
    failed = []

    for i, p in enumerate(paths, 1):
        name = os.path.splitext(os.path.basename(p))[0]
        j, V, T = parse_conditions(p)
        is_video = p.lower().endswith((".avi", ".mp4", ".mov"))
        tag = "video" if is_video else "image"
        print(f"[{i:>2}/{len(paths)}] {tag:<5} {os.path.basename(p)[:58]:<58}", end=" ")
        if j is None:
            print("SKIP (no NNNmA in filename)")
            failed.append((p, "no current density in filename"))
            continue
        try:
            if is_video:
                m, frame, bubbles, rows, bg, shape = process_video(p, j, out_dir, camera_fps)
                video_rows_by_j.setdefault(j, []).extend(rows)
                if rows:
                    with open(os.path.join(out_dir, "tables", f"bubbles_{name}.csv"),
                              "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["x", "y", "d_px", "d_mm", "t", "j"])
                        w.writeheader(); w.writerows(rows)
            else:
                got = process_image(p, j, out_dir)
                if got is None:
                    print("FAILED (unreadable)")
                    failed.append((p, "cv2 could not read"))
                    continue
                m, frame, bubbles, rows, bg, shape = got
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")
            failed.append((p, f"{type(e).__name__}: {e}"))
            continue

        m.update({"file": os.path.basename(p), "j": j, "V": V, "T": T,
                   "dS": gibbs_entropy(V, T),
                   "h2_mL_min": round(h2_rate(j), 4)})
        make_panel(name, frame, bubbles, rows, shape, m,
                   os.path.join(out_dir, "per_file", f"{name}_panel.png"))
        m["thumb"] = frame
        results.append(m)
        print(f"j={j:>4} bubbles={m['mean_bubbles']:>7} H_norm={m['H_norm_pct']:>6}%")

    # ---- summary table -----------------------------------------------------
    cols = ["file", "kind", "j", "V", "T", "dS", "frames", "fps", "threshold",
            "mean_bubbles", "min_bubbles", "max_bubbles",
            "H", "H_norm_pct", "mean_diam_mm", "h2_mL_min", "total_records"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda r: (r["j"] or 0)):
            w.writerow(r)

    make_overview(results, out_dir)

    # ---- Fisher information across current densities (videos only) --------
    if len(video_rows_by_j) >= 2:
        print("\n   Fisher information across current densities:")
        shape_guess = (480, 640)
        dists = {}
        for j in sorted(video_rows_by_j):
            p, _ = build_state_distribution(video_rows_by_j[j], shape_guess, CELL)
            dists[j] = p
        js = sorted(dists)
        for a, b in zip(js[:-1], js[1:]):
            I, empty = fisher_information(dists[a], dists[b], b - a)
            export_map_csv(I, os.path.join(out_dir, "fisher", f"fisher_I_{a}to{b}.csv"))
            flag = "  <-- sparse" if empty > 0.80 else ""
            print(f"     j={a:>4} -> {b:<4}  I_max={I.max():.3e}  empty={empty*100:5.1f}%{flag}")

    print(f"\nProcessed {len(results)} files, {len(failed)} failed.")
    if failed:
        print("Failures:")
        for p, why in failed:
            print(f"   {os.path.basename(p)}: {why}")
    print(f"\nAll outputs in: {os.path.abspath(out_dir)}")
    print("  per_file/   original + overlay + maps for every file")
    print("  summary.csv every metric, one row per file")
    print("  entropy_vs_current.png, gibbs_correlation.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:  python3 run_all.py <folder> [--fps 200]   (camera frame rate printed in the frame)")
        sys.exit(1)
    fps = float(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 0.0
    main(sys.argv[1], camera_fps=fps)
