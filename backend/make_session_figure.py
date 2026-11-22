"""
make_session_figure.py
======================
Regenerates the session-comparison figure (Figure 17 of the report).

    python3 make_session_figure.py

It expects the batch results to already exist beside it:

    results_A/summary.csv        Session A, 20-500 mA/cm2, 12 recordings
    results_B/summary.csv        Session B, 100-950 mA/cm2, 16 recordings
    results_C/summary.csv        Session C, 688-838 mA/cm2, 5 recordings
    results_screenshots/summary.csv   the three single images (screen captures)

Produce those first with:

    python3 run_all.py session_A   &&  mv results results_A
    python3 run_all.py session_B   &&  mv results results_B
    ...

WHAT CHANGED AND WHY
--------------------
Prof. Wagner's comments on the interim report of 21 September 2026 asked
for two things on this figure:

    "Lettering enlargen (at the sides)"
    "Right picture: very good correlation! Better use errorbars and no
     lines between the points"

Both are applied here. There are no connecting lines anywhere, because a
line between two current densities implies that the intermediate values
were measured, and they were not. The axis lettering is set from one place
(FONT below) so it can be enlarged again in a single edit.

The error bars deserve an explanation, because there is only one current
density in Session A that was recorded more than once. Three repeats were
made at 250 mA/cm2, and their spread is the only direct measurement of how
much this method scatters when nothing about the experiment changes. That
spread is therefore used as the uncertainty on every Session A point. It is
an assumption -- that repeatability does not vary with current density --
and it is stated rather than hidden. The three repeats are also drawn as
their own mean with a true +/- 1 sigma bar, so the assumed bar and the
measured one can be compared directly.
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "report_figures"

# Every text size in the figure. Prof. Wagner asked for larger lettering at
# the sides, so these are set here rather than scattered through the code.
FONT = dict(tick=15, label=17, title=16, legend=13.5, note=13.5)

SESSIONS = [
    ("results_A",           "Session A, ANS, current series with repeats",  "#1f77b4", "o"),
    ("results_B",           "Session B, ANS, large bubbles", "#d62728", "s"),
    ("results_C",           "Session C, ANS, nucleation sites", "#2ca02c", "^"),
    ("results_screenshots", "Single images, ANS, screen captures", "#9467bd", "D"),
]


def load(folder):
    """Read a summary.csv, wherever run_all.py happened to leave it."""
    for pattern in (os.path.join(folder, "summary.csv"),
                    os.path.join(folder, "*", "summary.csv")):
        hits = glob.glob(pattern)
        if hits:
            return pd.read_csv(hits[0])
    return None


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=FONT["label"])
    ax.set_ylabel(ylabel, fontsize=FONT["label"])
    ax.set_title(title, fontsize=FONT["title"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    ax.grid(alpha=0.3)


def main():
    data = {}
    for folder, label, colour, marker in SESSIONS:
        d = load(folder)
        if d is not None and len(d):
            data[folder] = d
        else:
            print("  missing: " + folder + "/summary.csv  (panel will skip it)")

    if "results_A" not in data:
        print("\nresults_A/summary.csv is required and was not found.")
        print("Run:  python3 run_all.py session_A   then   mv results results_A")
        return

    os.makedirs(OUT, exist_ok=True)
    A = data["results_A"].sort_values("j")

    # ---- repeatability, from the only repeated operating point -----------
    counts = A.groupby("j").size()
    rep_j = counts[counts > 1]
    if len(rep_j):
        j_rep = int(rep_j.index[0])
        rep = A[A.j == j_rep]
        sigma = float(rep.H_norm_pct.std(ddof=1))
    else:
        j_rep, rep, sigma = None, None, float("nan")

    # Panels stacked rather than side by side: at the printed width of the
    # report the side-by-side version shrank the axis lettering to about
    # 4 pt, which is what the comment "Lettering enlargen (at the sides)"
    # referred to. Stacked, the lettering prints at roughly 10 pt.
    fig, ax = plt.subplots(3, 1, figsize=(10, 17.5))

    # ---- (a) every session, points only ----------------------------------
    for folder, label, colour, marker in SESSIONS:
        if folder not in data:
            continue
        d = data[folder]
        ax[0].plot(d.j, d.H_norm_pct, marker, ms=9, color=colour,
                   linestyle="none", label=f"{label} (n = {len(d)})")
    style(ax[0], "Current density  (mA/cm$^2$)", "$H_{norm}$  (%)",
          "(a) $H_{norm}$ by recording session")
    ax[0].legend(fontsize=FONT["legend"], loc="lower right")

    # ---- (b) pooled against Gibbs entropy, where it collapses ------------
    # Only the recordings whose filename carries a cell voltage have a
    # Gibbs entropy at all, so this panel is necessarily a subset.
    pooled = pd.concat(data.values(), ignore_index=True)
    pooled = pooled.dropna(subset=["dS", "H_norm_pct"])
    r_pool = float(np.corrcoef(pooled.dS, pooled.H_norm_pct)[0, 1]) \
        if len(pooled) >= 3 else float("nan")

    for folder, label, colour, marker in SESSIONS:
        if folder not in data:
            continue
        d = data[folder].dropna(subset=["dS"])
        if len(d):
            ax[1].plot(d.dS, d.H_norm_pct, marker, ms=9, color=colour,
                       linestyle="none", alpha=0.85)
    style(ax[1], "Gibbs entropy production  $zF\\eta/T$  (J mol$^{-1}$K$^{-1}$)",
          "$H_{norm}$  (%)",
          f"(b) All {len(pooled)} files with voltage, pooled\n"
          f"r = {r_pool:.3f}")

    # ---- (c) Session A alone, error bars, no lines -----------------------
    singles = A[A.j != j_rep] if j_rep is not None else A
    ax[2].errorbar(singles.j, singles.H_norm_pct,
                   yerr=sigma if np.isfinite(sigma) else None,
                   fmt="o", ms=9, color="#1f77b4",
                   capsize=5, elinewidth=1.4, linestyle="none",
                   label="single recording")

    if rep is not None and len(rep) > 1:
        ax[2].errorbar([j_rep], [rep.H_norm_pct.mean()],
                       yerr=[rep.H_norm_pct.std(ddof=1)],
                       fmt="s", ms=11, color="#d62728",
                       capsize=7, elinewidth=2.0, linestyle="none",
                       label=f"{j_rep} mA/cm$^2$, {len(rep)} repeats "
                             f"($\\pm 1\\sigma$)")

    # Logarithmic fit over every recording. Collapsing the three repeats to
    # their mean first would raise r from 0.911 to 0.954, which looks better
    # and is not what was measured, so the fit keeps all twelve points and
    # the repeated operating point keeps its full weight.
    z = np.polyfit(np.log(A.j), A.H_norm_pct, 1)
    r_log = float(np.corrcoef(np.log(A.j), A.H_norm_pct)[0, 1])
    xs = np.linspace(A.j.min(), A.j.max(), 300)
    ax[2].plot(xs, np.polyval(z, np.log(xs)), "--", color="gray",
               linewidth=1.6,
               label=f"logarithmic fit, r = {r_log:.3f}")

    style(ax[2], "Current density  (mA/cm$^2$)", "$H_{norm}$  (%)",
          f"(c) Session A alone, n = {len(A)}\n"
          f"logarithmic trend, r = {r_log:.3f}")
    ax[2].legend(fontsize=FONT["legend"], loc="lower right")

    if np.isfinite(sigma):
        ax[2].text(0.03, 0.96,
                   f"error bars = $\\pm${sigma:.2f} pp,\n"
                   f"measured from the {len(rep)} repeats",
                   transform=ax[2].transAxes, va="top",
                   fontsize=FONT["note"],
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9",
                             edgecolor="#94a3b8"))

    fig.tight_layout()
    path = os.path.join(OUT, "session_analysis.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)

    print("\nSESSION COMPARISON")
    print("=" * 62)
    for folder, label, colour, marker in SESSIONS:
        if folder in data:
            d = data[folder]
            print(f"  {label:<36} n={len(d):>3}  "
                  f"mean H_norm = {d.H_norm_pct.mean():.2f} %")
    if len(data) >= 2:
        means = {f: data[f].H_norm_pct.mean() for f in data}
        hi, lo = max(means.values()), min(means.values())
        print(f"\n  Largest baseline offset between sessions: {hi - lo:.2f} pp")
    print(f"  Pooled Gibbs correlation                : r = {r_pool:.3f}")
    print(f"  Session A alone, logarithmic fit        : r = {r_log:.3f}")
    if np.isfinite(sigma):
        print(f"  Repeatability at {j_rep} mA/cm2 ({len(rep)} repeats)    : "
              f"+/- {sigma:.2f} pp")
    print("\nWrote " + path)


if __name__ == "__main__":
    main()
