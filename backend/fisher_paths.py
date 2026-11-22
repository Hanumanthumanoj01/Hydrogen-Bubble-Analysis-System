"""
fisher_paths.py
===============
Do the bubbles follow stable paths?  A quantitative test of the pixel-level
Fisher-information maps (report Section 4.13, Figure 24).

HOW TO RUN (from the backend folder)
------------------------------------
    python3 fisher_paths.py                      default: the recordings staged by
                                                 make_all_figures.py in _pixel_fisher_B/
                                                 (Session B) and _pixel_fisher_C/ (Session C)
    python3 fisher_paths.py FOLDER [FOLDER ...]  any folders with recordings whose file
                                                 names carry "NNNmA"
    python3 fisher_paths.py --max 400            frames read per recording (default 400)
    python3 fisher_paths.py --out DIR            output folder (default figures_for_overleaf)

Output in figures_for_overleaf/:
    fisher_paths.png    Figure 24
    fisher_paths.tex    the numbers of Section 4.13 as the LaTeX macro \\FPtext
    fisher_paths.csv    one row per recording

WHAT IT MEASURES (details in fisher_pixel.path_share)
-----------------------------------------------------
Each recording is cut into a first and a second half. In the first half the
5 % of pixels with the most information from significant brightness changes
are marked. The script then asks: what share of the second half's information
falls on those same 5 % of pixels?

    share = 0.05       the information goes anywhere; no fixed paths
    share >> 0.05      the information keeps returning to the same pixels:
                       the bubbles grow, rise and detach along stable paths

The value is compared with the same region shifted to 20 random places
(null band); only a share above the 95th percentile of the shifted values
counts as a stable path.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import matplotlib                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import fisher_pixel as fp                    # noqa: E402

OUT = "figures_for_overleaf"
EXT = (".avi", ".mp4", ".mov", ".mkv")
ELECTRODE = {"B": "ANS", "C": "ANS"}
SESSION_OF = {"_pixel_fisher_B": "B", "_pixel_fisher_C": "C",
              "session_B": "B", "session_C": "C"}


def recordings(folder):
    """One recording per current value (the .mp4 is preferred to a duplicate .avi)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if not p.lower().endswith(EXT) or "__1_" in p:
            continue
        j = fp.current_density(p)
        if j is None:
            continue
        if j not in out or p.lower().endswith(".mp4"):
            out[j] = p
    return [out[j] for j in sorted(out)]


def analyse(paths, max_frames, folder_session):
    rows = []
    for p in paths:
        fr = fp.read_grey(p, max_frames=max_frames)
        mk = fp.overlay_mask(fr)
        res, A, B, T = fp.path_share(fr, mk)
        res.update(file=os.path.basename(p), j=fp.current_density(p),
                   session=folder_session[p], frames=len(fr), A=A, B=B, T=T,
                   I=fr.mean(0))
        rows.append(res)
        print(f"  {res['file'][:48]:48s} share {res['share']:.3f}  null "
              f"{res['null']:.3f} (95 % {res['null95']:.3f})  "
              f"{'STABLE' if res['stable'] else 'not stable'}  events "
              f"{res['events']:.4f}  noise {res['noise_rate']:.5f}")
    return rows


def figure(rows, path):
    n = len(rows)
    fig = plt.figure(figsize=(3.3 * n, 10.8))
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 1.15], hspace=0.22, wspace=0.06)
    for k, r in enumerate(rows):
        for row, (M, ttl) in enumerate(((r["A"], "first half"), (r["B"], "second half"))):
            ax = fig.add_subplot(gs[row, k])
            hi = np.percentile(M, 99.5) or 1.0
            ax.imshow(M, cmap="inferno", vmin=0, vmax=hi)
            ax.contour(r["T"].astype(float), levels=[0.5], colors="#00e5ff", linewidths=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            head = (f"Session {r['session']}, {ELECTRODE.get(r['session'], '')}\n"
                    f"{r['j']} mA/cm$^2$\n" if row == 0 else "")
            ax.set_title(head + ttl, fontsize=13.5)
    ax = fig.add_subplot(gs[2, :])
    x = np.arange(n)
    ax.bar(x, [r["share"] for r in rows], 0.55,
           color=["#2ca02c" if r["stable"] else "#9e9e9e" for r in rows],
           label="share on the region marked in the first half (green: stable path)")
    ax.hlines([r["null95"] for r in rows], x - 0.38, x + 0.38, colors="#d62728", lw=2.6,
              label="same region shifted at random: 95th percentile")
    ax.axhline(0.05, color="k", ls=":", lw=1.3, label="random placement (5 %)")
    for xi, r in zip(x, rows):
        ax.text(xi, max(r["share"], r["null95"]) + 0.006, f"{100 * r['share']:.0f} %", ha="center",
                va="bottom", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['session']}: {r['j']} mA/cm$^2$" for r in rows], fontsize=13)
    ax.set_ylabel("share of second-half\ninformation", fontsize=14.5)
    ax.set_ylim(0, max(0.12, 1.55 * max(r["share"] for r in rows)))
    ax.tick_params(labelsize=13); ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=12, loc="upper center", ncol=2, frameon=False)
    ax.set_title("Does the information return to the same 5 % of pixels?", fontsize=15)
    fig.suptitle("Stable bubble paths in the pixel-level Fisher information "
                 "(significant changes only; cyan: top 5 % of the first half)",
                 fontsize=15.5, y=0.96)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tex(rows):
    st = [r for r in rows if r["stable"]]
    ns = [r for r in rows if not r["stable"]]
    ev = [r["events"] for r in rows]
    nz = [r["noise_rate"] for r in rows]
    lst = lambda L: ", ".join(f"{100 * r['share']:.0f} \\% at {r['j']} mA/cm$^2$ "
                              f"(Session {r['session']})" for r in L)
    t = (f"Of the {len(rows)} recordings, {len(st)} show stable paths: " + (lst(st) or "none")
         + ". " + ("The share stays inside the band of the shifted region at "
                   + ", ".join(f"{r['j']} mA/cm$^2$ ({100 * r['share']:.0f} \\%)" for r in ns)
                   + ". " if ns else "")
         + f"Significant changes occur in {100 * min(ev):.1f} to {100 * max(ev):.1f} \\% of the "
           f"pixel-frame steps, while a motionless copy of each scene with the measured "
           f"camera noise gives {100 * min(nz):.3f} to {100 * max(nz):.2f} \\%; noise therefore contributes "
           f"less than {100 * max(z / e for z, e in zip(nz, ev)):.1f} \\% of the events.")
    tB = sorted((r for r in rows if r["session"] == "B"), key=lambda r: r["j"])
    if len(tB) >= 2:
        t += (f" In Session B the share falls from {100 * tB[0]['share']:.0f} \\% at "
              f"{tB[0]['j']} to {100 * tB[-1]['share']:.0f} \\% at {tB[-1]['j']} mA/cm$^2$.")
    return "% written by fisher_paths.py\n\\renewcommand{\\FPtext}{" + t + "}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*", default=["_pixel_fisher_B", "_pixel_fisher_C"])
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out = a.out
    paths, sess = [], {}
    for f in a.folders:
        if not os.path.isdir(f):
            print(f"  {f}: no such folder, skipped")
            continue
        for p in recordings(f):
            paths.append(p)
            sess[p] = SESSION_OF.get(os.path.basename(os.path.normpath(f)), "?")
    if not paths:
        sys.exit("no recordings found; run make_all_figures.py first or pass folders")
    rows = analyse(paths, a.max, sess)
    os.makedirs(out, exist_ok=True)
    figure(rows, os.path.join(out, "fisher_paths.png"))
    open(os.path.join(out, "fisher_paths.tex"), "w").write(tex(rows))
    with open(os.path.join(out, "fisher_paths.csv"), "w") as f:
        f.write("session,file,j,frames,share,null_mean,null95,stable,event_rate,noise_rate\n")
        for r in rows:
            f.write(f"{r['session']},{r['file']},{r['j']},{r['frames']},{r['share']:.4f},"
                    f"{r['null']:.4f},{r['null95']:.4f},{int(r['stable'])},"
                    f"{r['events']:.5f},{r['noise_rate']:.6f}\n")
    print(f"wrote {out}/fisher_paths.png, fisher_paths.tex, fisher_paths.csv")


if __name__ == "__main__":
    main()
