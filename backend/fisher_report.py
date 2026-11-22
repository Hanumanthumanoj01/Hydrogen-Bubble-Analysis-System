"""
fisher_report.py
================
Everything Prof. Wagner asked for, computed on the bubble recordings:

  * the brightness histogram of each recording, and of each sampled frame
  * Shannon entropy frame by frame
  * Fisher information frame by frame, with the time axis in seconds
  * the SPATIAL Fisher information, split into x and y
  * a table of the strongest peaks in the Fisher map, and what they sit on
  * all of it written to one Excel workbook, one sheet per quantity

This extends fisher_pixel.py rather than replacing it. fisher_pixel.py makes
the figures for the report; this makes the numbers, in the form that can be
opened in Excel and sorted.

WHY THE SPATIAL VERSION IS SPLIT INTO X AND Y
---------------------------------------------
The temporal map answers "where does the picture change over time". The
spatial maps answer "which way does the brightness run". They are

    F_x(x,y) = (dI/dx)^2 / I        F_y(x,y) = (dI/dy)^2 / I

and they are not redundant with each other. A bubble is round, so its rim
contributes to both: the left and right of the rim show up in F_x, the top
and bottom in F_y. A machining scratch on the plate runs in one direction
only, so it shows up in one and not the other. The ratio of the two is
therefore a way of telling a round thing from a streak without detecting
anything.

WHAT A PEAK MEANS
-----------------
A peak in F is a pixel whose brightness changes sharply with respect to
whatever the derivative was taken against. For F_t that is time, so a peak
is a place where gas is arriving or leaving — a bubble rim that is moving,
or a site where one has just detached. For F_x and F_y it is position, so a
peak is a boundary in the picture. Peaks are reported with their position,
their height relative to the frame median, and whether they lie on a
brightness edge, because a peak that is NOT on an edge is usually the
camera's own noise rather than anything physical.

USAGE
-----
    python3 fisher_report.py <folder-or-video> [--every 5] [--max 200] [--fps 200]
        --fps  camera frame rate printed in the frame; the files store 24 fps

Writes  fisher_report_output/<name>.xlsx  and the matching figures.
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
import pandas as pd

VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv", ".m4v")
EPS = 1e-3          # brightness floor; see fisher_pixel.py for why
OUT = "fisher_report_output"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def current_density(path):
    """Pull the NNNmA out of the laboratory's file naming."""
    m = re.search(r"(?<!\d)(\d{2,5})\s*mA", os.path.basename(path), re.I)
    return int(m.group(1)) if m else None


def read_grey(path, max_frames=200, every=1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open " + path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames, idx = [], []
    i = 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            idx.append(i)
        i += 1
    cap.release()
    if not frames:
        raise IOError("no frames decoded from " + path)
    return np.stack(frames), np.array(idx), fps, total


def timestamp_mask(frames, persist=0.75, cap_frac=0.01):
    """
    Find the camera's burnt-in timestamp so it can be excluded.

    The millisecond digits are a counter: they change in essentially EVERY
    frame. Bubbles change often but not in literally every frame. That
    difference is what separates them, and the threshold has to be set high
    enough to use it.

    This was got wrong first and the measurement is worth keeping. With the
    threshold at 0.6 the mask covered 105 px on the 100 mA recording but
    1290 px on the 600 mA one, because at the higher current a stream of
    bubbles runs along the top of the field, right where the timestamp sits,
    and the mask ate them. The Fisher map of that recording then scored 0.89
    on the edge-to-flat ratio — below 1, meaning it had stopped finding
    structure at all. Raising the threshold to 0.75 gives comparable small regions on
    both recordings: the same small text region in both, which is what a
    timestamp should look like.

    cap_frac is a guard rather than a tuning knob. If the result still covers
    more than 1 % of the frame it is not a timestamp, so nothing is masked
    and the caller is told, which is safer than silently deleting physics.
    """
    F = frames.astype(np.float32)
    d = np.abs(np.diff(F, axis=0))
    always = (d > 8).mean(axis=0) > persist

    strip = np.zeros(always.shape, bool)
    strip[: int(0.14 * always.shape[0]), :] = True
    m = always & strip
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    if m.sum() > cap_frac * m.size:
        return np.zeros_like(m), False
    return m, True


# ---------------------------------------------------------------------------
# the quantities
# ---------------------------------------------------------------------------

def overlay_text_mask(frames, ts_mask):
    """
    The whole burnt-in caption, not just the digits that change.

    timestamp_mask finds the milliseconds, which is all the TEMPORAL map
    needs — the rest of the caption is static, so it contributes nothing to
    dI/dt. The SPATIAL maps are a different matter. "640 x 480 / 200 fps"
    is high-contrast lettering sitting on the picture, so every letter edge
    is a large dI/dx and dI/dy, and the caption shows up in F_x and F_y as
    brightly as any bubble rim. It is an artefact of the camera's overlay,
    not of the electrode.

    So for the spatial maps the caption is masked as well. It is found as
    pixels in the top strip that are bright, have a strong gradient, and
    barely change through the recording — which is what printed text on a
    moving scene looks like.
    """
    F = frames.astype(np.float32)
    I = F.mean(axis=0)
    h = I.shape[0]

    changing = np.abs(np.diff(F, axis=0)).mean(axis=0)
    gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    strip = np.zeros(I.shape, bool)
    strip[: int(0.14 * h), :] = True

    static = changing < np.percentile(changing, 40)
    sharp = mag > np.percentile(mag, 97)
    m = strip & static & sharp
    m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    m |= ts_mask
    if m.sum() > 0.03 * m.size:          # same guard as above
        return ts_mask
    return m


def shannon(gray):
    """H in nats, and H normalised by ln 256, from the pixel histogram."""
    h = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    h = h / max(h.sum(), 1.0)
    nz = h[h > 0]
    H = float(-np.sum(nz * np.log(nz)))
    return H, H / np.log(256), h


def fisher_temporal_per_frame(frames, mask=None):
    """
    F_t for each consecutive pair, returned frame by frame rather than
    averaged. This is what lets the Excel sheet show how the information
    content moves during the recording instead of collapsing it to one
    number per video.
    """
    F = frames.astype(np.float32) / 255.0
    out = []
    for k in range(len(F) - 1):
        dI = F[k + 1] - F[k]
        I = 0.5 * (F[k + 1] + F[k])
        m = (dI * dI) / (I + EPS)
        if mask is not None:
            m = m.copy()
            m[mask] = 0.0
        out.append(m)
    return np.stack(out)


def fisher_spatial(I, mask=None):
    """
    F_x and F_y on one brightness field, normalised to 0..1 beforehand.

    Sobel is used for the derivative rather than a plain difference because
    it averages over three rows or columns, which keeps single-pixel sensor
    noise from dominating a quantity that squares its input.
    """
    If = I.astype(np.float32) / 255.0 if I.dtype == np.uint8 else I.astype(np.float32)
    gx = cv2.Sobel(If, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(If, cv2.CV_32F, 0, 1, ksize=3)
    Fx = (gx * gx) / (If + EPS)
    Fy = (gy * gy) / (If + EPS)
    if mask is not None:
        Fx = Fx.copy(); Fy = Fy.copy()
        Fx[mask] = 0.0; Fy[mask] = 0.0
    return Fx, Fy


def edge_flat_ratio(F, I, mask=None):
    """
    How much of the information sits on brightness edges rather than on flat
    plate. Above about 1.5 means the map has found the rims; near 1.0 means
    it is spread evenly and is telling us nothing about structure.
    """
    If = I.astype(np.float32)
    gx = cv2.Sobel(If, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(If, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    valid = np.ones(F.shape, bool)
    if mask is not None:
        valid &= ~mask
    cut = np.percentile(mag[valid], 90)
    edge = (mag >= cut) & valid
    flat = (mag < cut) & valid
    if not flat.any() or F[flat].mean() <= 0:
        return float("nan")
    return float(F[edge].mean() / F[flat].mean())


def find_peaks(F, I, n=25, min_sep=12, mask=None, edge_ref=None):
    """
    The strongest, well-separated maxima of a Fisher map.

    Each is reported with its height as a multiple of the map's median, and
    with whether it lies on a brightness edge. A tall peak that is NOT on an
    edge is the signature of sensor noise or a hot pixel, not of a bubble,
    and reporting that column is what makes the peak list checkable rather
    than a list of numbers to be trusted.
    """
    Fw = F.copy()
    if mask is not None:
        Fw[mask] = 0.0
    If = I.astype(np.float32)
    # The edge reference matters and defaults to the wrong thing for the
    # temporal map. Averaging a recording blurs every moving bubble away, so
    # the time-averaged image is SMOOTH exactly where the temporal
    # information is highest, and testing against it marked all 25 peaks as
    # "not on an edge". A single frame is the right reference there.
    Ref = If if edge_ref is None else edge_ref.astype(np.float32)
    gx = cv2.Sobel(Ref, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ref, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_cut = np.percentile(mag, 90)

    med = float(np.median(Fw[Fw > 0])) if (Fw > 0).any() else 1.0
    work = Fw.copy()
    rows = []
    for _ in range(n):
        iy, ix = np.unravel_index(int(np.argmax(work)), work.shape)
        v = float(work[iy, ix])
        if v <= 0:
            break
        rows.append({
            "x_px": int(ix),
            "y_px": int(iy),
            "F": v,
            "F_over_median": v / max(med, 1e-9),
            "on_edge": bool(mag[iy, ix] >= edge_cut),
            "brightness": float(If[iy, ix]),
        })
        y0, y1 = max(0, iy - min_sep), min(work.shape[0], iy + min_sep + 1)
        x0, x1 = max(0, ix - min_sep), min(work.shape[1], ix + min_sep + 1)
        work[y0:y1, x0:x1] = 0.0
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# one recording
# ---------------------------------------------------------------------------

def analyse(path, every=1, max_frames=200, camera_fps=0.0):
    frames, idx, fps, total = read_grey(path, max_frames=max_frames, every=every)
    # The laboratory files store 24 fps in the container; the camera rate is
    # printed in the frame. Pass --fps to use it for every time value.
    if camera_fps and camera_fps > 0:
        fps = float(camera_fps)
    mask, mask_ok = timestamp_mask(frames)
    smask = overlay_text_mask(frames, mask)
    j = current_density(path)

    # --- per frame: entropy, Fisher, histogram
    Ft = fisher_temporal_per_frame(frames, mask)
    per_frame, hists = [], []
    for k in range(len(frames)):
        H, Hn, hist = shannon(frames[k])
        hists.append(hist)
        row = {
            "frame": int(idx[k]),
            "time_s": round(float(idx[k]) / fps, 4) if fps else None,
            "H_nats": round(H, 4),
            "H_norm_pct": round(Hn * 100, 2),
            "mean_brightness": round(float(frames[k].mean()), 2),
        }
        if k < len(Ft):
            row["F_t_mean"] = float(Ft[k][~mask].mean())
            row["F_t_max"] = float(Ft[k][~mask].max())
            row["F_t_edge_flat_ratio"] = round(
                edge_flat_ratio(Ft[k], frames[k], mask), 3)
        per_frame.append(row)
    per_frame = pd.DataFrame(per_frame)

    # --- maps over the whole recording
    I_mean = frames.mean(axis=0)
    F_t_mean = Ft.mean(axis=0)
    F_x, F_y = fisher_spatial(I_mean, smask)

    summary = {
        "file": os.path.basename(path),
        "current_density_mA_cm2": j,
        "fps": round(fps, 2) if fps else None,
        "frames_in_file": total,
        "frames_analysed": len(frames),
        "duration_analysed_s": round(float(idx[-1]) / fps, 3) if fps else None,
        "timestamp_pixels_masked": int(mask.sum()),
        "timestamp_mask_applied": bool(mask_ok),
        "H_mean_nats": round(float(per_frame["H_nats"].mean()), 4),
        "H_norm_mean_pct": round(float(per_frame["H_norm_pct"].mean()), 2),
        "F_t_mean": float(F_t_mean[~mask].mean()),
        "F_t_edge_flat_ratio": round(edge_flat_ratio(F_t_mean, I_mean, mask), 3),
        "F_x_mean": float(F_x[~smask].mean()),
        "F_y_mean": float(F_y[~smask].mean()),
        "Fx_over_Fy": round(float(F_x[~smask].mean() / max(F_y[~smask].mean(), 1e-12)), 3),
        "F_xy_edge_flat_ratio": round(
            edge_flat_ratio(F_x + F_y, I_mean, smask), 3),
        "overlay_pixels_masked_spatial": int(smask.sum()),
    }

    # The peak table uses the wider caption mask, not the narrow timestamp
    # one. The map itself keeps the narrow mask so that bubbles drifting
    # across the top of the field are not thrown away; but a PEAK list is a
    # diagnostic that gets read line by line, and one leftover digit of the
    # burnt-in clock outranked every real bubble in it — 3355 times the map
    # median against 1761 for the strongest genuine site.
    peaks_t = find_peaks(F_t_mean, I_mean, mask=smask,
                         edge_ref=frames[len(frames) // 2])
    peaks_xy = find_peaks(F_x + F_y, I_mean, mask=smask)

    hist_df = pd.DataFrame(np.stack(hists).T,
                           columns=[f"frame_{int(i)}" for i in idx])
    hist_df.insert(0, "grey_level", np.arange(256))
    hist_df.insert(1, "whole_recording",
                   np.stack(hists).mean(axis=0))

    return dict(path=path, frames=frames, idx=idx, fps=fps, mask=mask,
                per_frame=per_frame, hist=hist_df, summary=summary,
                I_mean=I_mean, F_t=F_t_mean, F_x=F_x, F_y=F_y,
                peaks_t=peaks_t, peaks_xy=peaks_xy)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def write_excel(res, outdir):
    stem = os.path.splitext(os.path.basename(res["path"]))[0][:40]
    xls = os.path.join(outdir, stem + ".xlsx")
    with pd.ExcelWriter(xls, engine="openpyxl") as w:
        pd.DataFrame([res["summary"]]).T.rename(columns={0: "value"}) \
            .to_excel(w, sheet_name="summary")
        res["per_frame"].to_excel(w, sheet_name="per_frame", index=False)
        res["hist"].to_excel(w, sheet_name="histogram", index=False)
        res["peaks_t"].to_excel(w, sheet_name="peaks_temporal", index=False)
        res["peaks_xy"].to_excel(w, sheet_name="peaks_spatial_xy", index=False)
    return xls


def _panel(ax, img, title, pct=99.0):
    v = np.percentile(img, pct) if np.isfinite(img).any() else 1.0
    ax.imshow(img, cmap="inferno", vmin=0, vmax=max(v, 1e-9))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def write_figures(res, outdir):
    stem = os.path.splitext(os.path.basename(res["path"]))[0][:40]

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    _panel(ax[0], res["I_mean"], "mean brightness")
    _panel(ax[1], res["F_t"], "$F_t=(dI/dt)^2/I$   (time)")
    _panel(ax[2], res["F_x"], "$F_x=(dI/dx)^2/I$")
    _panel(ax[3], res["F_y"], "$F_y=(dI/dy)^2/I$")
    fig.suptitle(os.path.basename(res["path"]), fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(outdir, stem + "_maps.png")
    fig.savefig(p1, dpi=150); plt.close(fig)

    pf = res["per_frame"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    t = pf["time_s"] if pf["time_s"].notna().all() else pf["frame"]
    ax[0].plot(t, pf["H_norm_pct"], lw=1.6, color="#2f7d32")
    ax[0].set_ylabel("$H_{norm}$ (%)")
    ax[0].grid(alpha=.3)
    if "F_t_mean" in pf:
        ax[1].plot(t[:-1], pf["F_t_mean"][:-1], lw=1.6, color="#c0392b")
    ax[1].set_ylabel("mean $F_t$"); ax[1].set_xlabel("time (s)")
    ax[1].grid(alpha=.3)
    fig.suptitle("Entropy and Fisher information through the recording",
                 fontsize=10)
    fig.tight_layout()
    p2 = os.path.join(outdir, stem + "_timeseries.png")
    fig.savefig(p2, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(res["hist"]["grey_level"], res["hist"]["whole_recording"],
            lw=1.4, color="#1f4e79")
    ax.set_xlabel("grey level"); ax.set_ylabel("fraction of pixels")
    ax.set_title("Brightness histogram — the distribution entropy is computed from",
                 fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    p3 = os.path.join(outdir, stem + "_histogram.png")
    fig.savefig(p3, dpi=150); plt.close(fig)
    return p1, p2, p3


# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    target = sys.argv[1]
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
    mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 200
    cam = float(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 0.0

    if os.path.isdir(target):
        files = sorted(p for p in glob.glob(os.path.join(target, "*"))
                       if p.lower().endswith(VIDEO_EXT))
    else:
        files = [target]
    if not files:
        print("No recordings found in " + target); return

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p in files:
        print(f"\n{os.path.basename(p)}")
        res = analyse(p, every=every, max_frames=mx, camera_fps=cam)
        xls = write_excel(res, OUT)
        write_figures(res, OUT)
        s = res["summary"]
        rows.append(s)
        print(f"  j = {s['current_density_mA_cm2']} mA/cm2, "
              f"{s['frames_analysed']} frames, {s['duration_analysed_s']} s")
        print(f"  H_norm mean      {s['H_norm_mean_pct']} %")
        print(f"  F_t mean         {s['F_t_mean']:.4e}   "
              f"edge/flat {s['F_t_edge_flat_ratio']}")
        print(f"  F_x / F_y        {s['Fx_over_Fy']}   "
              f"(1.0 = no preferred direction)")
        print(f"  timestamp pixels masked out: {s['timestamp_pixels_masked']}"
              + ("" if s["timestamp_mask_applied"] else "   (NOT applied - region too large, see docstring)"))
        print(f"  -> {xls}")

    if len(rows) > 1:
        allx = os.path.join(OUT, "all_recordings.xlsx")
        pd.DataFrame(rows).to_excel(allx, index=False)
        print(f"\nCombined summary -> {allx}")


if __name__ == "__main__":
    main()
