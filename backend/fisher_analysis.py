"""
fisher_analysis.py
==================
Implementation of the workflow specified in

    "Procedure for data evaluation with AI based bubble analysis"
    Prof. Dr. Enno Wagner, Frankfurt UAS, 14 August 2026

Input data, per detected bubble:

    (x, y, d, t, j)

    x, y = bubble position (pixels)
    d    = bubble diameter
    t    = time (s)
    j    = current density (mA/cm2)

Steps implemented (numbering follows the handout):

  1. Spatial bubble-frequency map        N(x, y | j)
  2. Mean local bubble diameter          d_m(x, y | j)
  3. Comparison across current densities j1, j2, j3, ...
  4. Bubble-state distribution           p(x, y, d | j)
  5. Fisher Information map              I_j(x, y)          [asterisk task]

Per the professor's e-mail of 14 August, the Fisher Information is applied to
a restricted section (e.g. 2x2 or 3x3 artificial nucleation sites) to keep the
computation clear and the result interpretable.

NOTE ON STEP 5 AND ZERO PROBABILITIES
-------------------------------------
The handout's estimator

    d ln p / dj  ~=  [ln p(x,y,d | j+dj) - ln p(x,y,d | j)] / dj

is undefined wherever p = 0, which is common in sparse (x, y, d) cells. This
implementation applies additive (Laplace) smoothing with a small alpha before
taking logarithms, and reports the fraction of cells that were empty so the
reliability of each map is visible. This is a numerical necessity, not a
modification of the method.
"""
from __future__ import annotations

import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from video_analysis import (
    load_video_gray, compute_background, auto_threshold, detect_bubbles_bgsub,
)
from constants import mm_per_pixel


# ────────────────────────────────────────────────────────────────────────────
# Input data: build the (x, y, d, t, j) table
# ────────────────────────────────────────────────────────────────────────────
def extract_bubble_table(video_path: str, current_density: float,
                         frame_step: int = 1,
                         max_frames: int = 600) -> Tuple[List[Dict[str, float]], Tuple[int, int]]:
    """Run detection over a video and return the (x, y, d, t, j) table.

    This is the "Excel-table with x, y, dB" the handout describes as the
    output of the bubble analysis and the input to everything below.
    """
    frames, fps = load_video_gray(video_path, max_frames=max_frames)
    bg = compute_background(frames)
    thr = auto_threshold(frames, bg=bg)
    scale = mm_per_pixel("video")

    rows: List[Dict[str, float]] = []
    for idx in range(0, len(frames), max(1, frame_step)):
        t = idx / fps
        for (x, y, r) in detect_bubbles_bgsub(frames[idx], bg, thr):
            rows.append({
                "x": float(x), "y": float(y),
                "d_px": float(2 * r),
                "d_mm": float(2 * r * scale),
                "t": float(t),
                "j": float(current_density),
            })
    h, w = frames[0].shape
    return rows, (h, w)


# ────────────────────────────────────────────────────────────────────────────
# Step 1 — Spatial bubble-frequency map  N(x, y | j)
# ────────────────────────────────────────────────────────────────────────────
def bubble_frequency_map(rows, shape: Tuple[int, int], cell: int = 10) -> np.ndarray:
    """Count detected bubbles per (cell x cell) pixel cell.

    Result: 2D map of local bubble activity / preferred nucleation sites.
    """
    h, w = shape
    ny, nx = int(np.ceil(h / cell)), int(np.ceil(w / cell))
    N = np.zeros((ny, nx), dtype=float)
    for r in rows:
        cx, cy = int(r["x"] // cell), int(r["y"] // cell)
        if 0 <= cy < ny and 0 <= cx < nx:
            N[cy, cx] += 1
    return N


# ────────────────────────────────────────────────────────────────────────────
# Step 2 — Mean local bubble diameter  d_m(x, y | j)
# ────────────────────────────────────────────────────────────────────────────
def mean_diameter_map(rows, shape: Tuple[int, int], cell: int = 10,
                      use_mm: bool = True) -> np.ndarray:
    """Mean bubble diameter per cell, over all time points.

        d_m(x, y | j) = (1/N) * sum_i d_i

    Cells with no detections are returned as NaN (not 0), so that "no data"
    is distinguishable from "bubbles of zero size".
    """
    h, w = shape
    ny, nx = int(np.ceil(h / cell)), int(np.ceil(w / cell))
    total = np.zeros((ny, nx), dtype=float)
    count = np.zeros((ny, nx), dtype=float)
    key = "d_mm" if use_mm else "d_px"
    for r in rows:
        cx, cy = int(r["x"] // cell), int(r["y"] // cell)
        if 0 <= cy < ny and 0 <= cx < nx:
            total[cy, cx] += r[key]
            count[cy, cx] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return dm


# ────────────────────────────────────────────────────────────────────────────
# Step 4 — Bubble-state distribution  p(x, y, d | j)
# ────────────────────────────────────────────────────────────────────────────
def build_state_distribution(rows, shape: Tuple[int, int], cell: int = 10,
                             d_edges: Optional[np.ndarray] = None,
                             use_mm: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the discrete bubble micro-state distribution p(x, y, d | j).

    Uses the INDIVIDUAL bubble data (not the mean diameter), as the handout
    specifies. Returns (p, d_edges) where p has shape (ny, nx, n_d) and sums
    to 1 over all three axes.
    """
    h, w = shape
    ny, nx = int(np.ceil(h / cell)), int(np.ceil(w / cell))
    key = "d_mm" if use_mm else "d_px"

    if d_edges is None:
        # Default bins follow Wagner et al. (2025) Fig. 4 when working in mm
        d_edges = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.1, np.inf]) if use_mm \
            else np.array([0, 5, 10, 20, 30, 40, 60, np.inf], dtype=float)

    n_d = len(d_edges) - 1
    p = np.zeros((ny, nx, n_d), dtype=float)
    for r in rows:
        cx, cy = int(r["x"] // cell), int(r["y"] // cell)
        if not (0 <= cy < ny and 0 <= cx < nx):
            continue
        d = r[key]
        b = int(np.searchsorted(d_edges, d, side="right") - 1)
        b = max(0, min(n_d - 1, b))
        p[cy, cx, b] += 1

    total = p.sum()
    if total > 0:
        p = p / total
    return p, d_edges


# ────────────────────────────────────────────────────────────────────────────
# Step 5 — Fisher Information  I_j(x, y)              [asterisk task]
# ────────────────────────────────────────────────────────────────────────────
def fisher_information(p_j: np.ndarray, p_j_next: np.ndarray, delta_j: float,
                       alpha: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Fisher Information map with respect to current density j.

        I_j(x, y) = sum_d  p(x,y,d|j) * ( d ln p / dj )^2

    with the derivative approximated numerically between two neighbouring
    current densities, as specified in the handout:

        d ln p / dj ~= [ln p(...|j+dj) - ln p(...|j)] / dj

    `alpha` is additive smoothing applied before the logarithm — required
    because p = 0 in many sparse cells and ln(0) is undefined. Returns
    (I_map, empty_fraction) where empty_fraction is the share of (x,y,d)
    states that were empty in BOTH distributions, i.e. carried no data.
    """
    if p_j.shape != p_j_next.shape:
        raise ValueError("The two distributions must share binning "
                         f"({p_j.shape} vs {p_j_next.shape})")
    if delta_j == 0:
        raise ValueError("delta_j must be non-zero")

    empty = np.logical_and(p_j == 0, p_j_next == 0)
    empty_fraction = float(empty.mean())

    ps = p_j + alpha
    pn = p_j_next + alpha
    ps = ps / ps.sum()
    pn = pn / pn.sum()

    dlnp_dj = (np.log(pn) - np.log(ps)) / delta_j
    I_map = np.sum(ps * dlnp_dj ** 2, axis=2)
    return I_map, empty_fraction


# ────────────────────────────────────────────────────────────────────────────
# Region of interest — restrict to a section of nucleation sites
# ────────────────────────────────────────────────────────────────────────────
def crop_region(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                cell: int) -> np.ndarray:
    """Crop a cell-resolution map to a pixel-coordinate rectangle.

    Per the professor's e-mail: apply the Fisher Information to a section of
    e.g. 2x2 or 3x3 artificial nucleation sites rather than the whole frame.
    """
    cy0, cy1 = int(y0 // cell), int(np.ceil(y1 / cell))
    cx0, cx1 = int(x0 // cell), int(np.ceil(x1 / cell))
    return arr[cy0:cy1, cx0:cx1]


def filter_rows_to_region(rows, x0: int, y0: int, x1: int, y1: int):
    """Keep only bubbles inside a pixel rectangle."""
    return [r for r in rows if x0 <= r["x"] < x1 and y0 <= r["y"] < y1]


# ────────────────────────────────────────────────────────────────────────────
# Export
# ────────────────────────────────────────────────────────────────────────────
def export_bubble_table_csv(rows, path: str) -> None:
    """Write the (x, y, d, t, j) table — the handout's 'Excel-table'."""
    import csv
    with open(path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["x", "y", "d_px", "d_mm", "t", "j"])
        wtr.writeheader()
        wtr.writerows(rows)


def export_map_csv(arr: np.ndarray, path: str) -> None:
    np.savetxt(path, arr, delimiter=",", fmt="%.6g")


# ────────────────────────────────────────────────────────────────────────────
# Command-line runner — does the whole workflow in one go
# ────────────────────────────────────────────────────────────────────────────
def _parse_j_from_name(fname: str) -> Optional[int]:
    import re
    m = re.search(r"(\d{2,4})\s*mA", os.path.basename(fname), re.I)
    return int(m.group(1)) if m else None


def run_workflow(video_dir: str, out_dir: str = "fisher_output",
                 cell: int = 20, region: Optional[Tuple[int, int, int, int]] = None,
                 make_plot: bool = True) -> None:
    """Run Steps 1-5 over every video in `video_dir` and write all outputs.

    Videos are discovered automatically; the current density is read from the
    filename (e.g. ..._900mA_...mp4). At least two different current densities
    are required, because Step 5 needs a derivative between neighbouring j.
    """
    import glob

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for pat in ("*.mp4", "*.mov", "*.avi", "*.MP4", "*.MOV"):
        paths.extend(glob.glob(os.path.join(video_dir, pat)))

    videos: Dict[int, str] = {}
    for p in sorted(paths):
        j = _parse_j_from_name(p)
        if j is None:
            print(f"  ignored (no 'NNNmA' in name): {os.path.basename(p)}")
        elif j in videos:
            print(f"  ignored (duplicate {j} mA/cm2): {os.path.basename(p)}")
        else:
            videos[j] = p

    if len(videos) < 2:
        print(f"\nERROR: found {len(videos)} usable video(s); need at least 2.")
        print("Step 5 requires two neighbouring current densities to take a derivative.")
        return

    print(f"Found {len(videos)} videos: {sorted(videos)} mA/cm2\n")

    # ---- Step 0: bubble tables -------------------------------------------
    print("STEP 0  building (x, y, d, t, j) tables")
    tables, shape = {}, None
    for j in sorted(videos):
        rows, shape = extract_bubble_table(videos[j], j)
        tables[j] = rows
        export_bubble_table_csv(rows, os.path.join(out_dir, f"bubbles_{j}mA.csv"))
        print(f"   j={j:>4}: {len(rows):>6} bubble records  ->  bubbles_{j}mA.csv")

    # ---- Steps 1 & 2 ------------------------------------------------------
    print(f"\nSTEP 1  bubble-frequency map N(x,y|j)     [cell = {cell} px]")
    print("STEP 2  mean local diameter d_m(x,y|j)")
    for j in sorted(tables):
        N = bubble_frequency_map(tables[j], shape, cell)
        dm = mean_diameter_map(tables[j], shape, cell)
        export_map_csv(N, os.path.join(out_dir, f"N_map_{j}mA.csv"))
        export_map_csv(dm, os.path.join(out_dir, f"dmean_map_{j}mA.csv"))
        v = dm[~np.isnan(dm)]
        act = int((N > 0).sum())
        print(f"   j={j:>4}: active cells {act:>4}/{N.size}   "
              f"mean diameter {v.mean() if len(v) else 0:.4f} mm")

    # ---- Steps 4 & 5 ------------------------------------------------------
    work = tables
    sub_shape = shape
    if region is not None:
        x0, y0, x1, y1 = region
        print(f"\n   restricting Steps 4-5 to region x[{x0}:{x1}] y[{y0}:{y1}]")
        work = {}
        for j in tables:
            rr = [dict(r) for r in filter_rows_to_region(tables[j], x0, y0, x1, y1)]
            for r in rr:
                r["x"] -= x0
                r["y"] -= y0
            work[j] = rr
        sub_shape = (y1 - y0, x1 - x0)

    print("\nSTEP 4  bubble-state distribution p(x,y,d|j)")
    dists = {}
    for j in sorted(work):
        p, _ = build_state_distribution(work[j], sub_shape, cell)
        dists[j] = p
        print(f"   j={j:>4}: {int((p > 0).sum()):>5} non-empty states of {p.size}")

    print("\nSTEP 5  Fisher Information I_j(x,y)        [asterisk task]")
    js = sorted(dists)
    fisher_maps = {}
    for a, b in zip(js[:-1], js[1:]):
        I, empty = fisher_information(dists[a], dists[b], b - a)
        fisher_maps[(a, b)] = I
        export_map_csv(I, os.path.join(out_dir, f"fisher_I_{a}to{b}.csv"))
        flag = "  <-- SPARSE, treat with caution" if empty > 0.80 else ""
        print(f"   j={a:>4} -> {b:<4}  I_max={I.max():.3e}   "
              f"empty states {empty*100:5.1f}%{flag}")

    if make_plot:
        try:
            _plot_maps(tables, shape, fisher_maps, cell, out_dir)
            print(f"\n   wrote {out_dir}/workflow_maps.png")
        except Exception as e:
            print(f"\n   (plot skipped: {e})")

    print(f"\nDone. All outputs in: {os.path.abspath(out_dir)}")


def _plot_maps(tables, shape, fisher_maps, cell, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    js = sorted(tables)
    ncol = max(len(js), len(fisher_maps), 1)
    fig, ax = plt.subplots(3, ncol, figsize=(4.2 * ncol, 11), squeeze=False)
    for i, j in enumerate(js):
        N = bubble_frequency_map(tables[j], shape, cell)
        dm = mean_diameter_map(tables[j], shape, cell)
        im = ax[0][i].imshow(N, cmap="viridis")
        ax[0][i].set_title(f"Step 1: N(x,y | {j} mA/cm²)", fontsize=9)
        plt.colorbar(im, ax=ax[0][i], fraction=0.046)
        im = ax[1][i].imshow(dm, cmap="plasma")
        ax[1][i].set_title(f"Step 2: d_m(x,y | {j}) [mm]", fontsize=9)
        plt.colorbar(im, ax=ax[1][i], fraction=0.046)
    for i, (k, I) in enumerate(sorted(fisher_maps.items())):
        im = ax[2][i].imshow(I, cmap="inferno")
        ax[2][i].set_title(f"Step 5: Fisher I  j={k[0]}→{k[1]}", fontsize=9)
        plt.colorbar(im, ax=ax[2][i], fraction=0.046)
    for row in ax:
        for a in row:
            a.set_xticks([]); a.set_yticks([])
    plt.suptitle("Bubble-state analysis and Fisher Information (Wagner workflow)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "workflow_maps.png"), dpi=140)
    plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("    python3 fisher_analysis.py <video_folder> [cell_px] [x0 y0 x1 y1]")
        print("\nExamples:")
        print("    python3 fisher_analysis.py videos")
        print("    python3 fisher_analysis.py videos 20")
        print("    python3 fisher_analysis.py videos 20 0 0 560 400   # 3x3 site region")
        sys.exit(1)
    folder = sys.argv[1]
    cell_px = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    reg = tuple(int(v) for v in sys.argv[3:7]) if len(sys.argv) >= 7 else None
    run_workflow(folder, cell=cell_px, region=reg)
