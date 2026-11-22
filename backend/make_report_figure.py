"""
make_report_figure.py
=====================
Builds the 3-panel summary figure for the supervisor report, using the
summary.csv files produced by run_all.py.

    python3 make_report_figure.py

Expects these folders to exist next to it (created by run_all.py + mv):
    results_A/summary.csv     Session A  (020mA_01.avi ... 500mA_01.avi)
    results_B/summary.csv     Session B  (20260713_bigbubbles_*)

Writes:
    report_figures/final_results.png
    report_figures/session_stats.txt      the numbers, for pasting into the email

Panels:
  (a) Session A: entropy vs current density, log fit, with the 250 mA repeat
      spread drawn as an error bar
  (b) Session A vs Session B: shows the baseline offset between sessions
  (c) Session B: Gibbs correlation (only Session B filenames carry mV)
"""
import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "report_figures"


def load(folder):
    p = os.path.join(folder, "summary.csv")
    if not os.path.exists(p):
        print(f"MISSING: {p}")
        print("  Run:  python3 run_all.py session_X   then   mv results results_X")
        return None
    d = pd.read_csv(p)
    for c in ("j", "H_norm_pct", "mean_bubbles", "dS", "mean_diam_mm"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    A = load("results_A")
    B = load("results_B")
    if A is None or B is None:
        sys.exit(1)

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---------- numbers ----------
    rl = np.corrcoef(np.log(A.j), A.H_norm_pct)[0, 1]
    rlin = np.corrcoef(A.j, A.H_norm_pct)[0, 1]
    rep = A[A.j == 250]
    sd = rep.H_norm_pct.std(ddof=1) if len(rep) > 1 else float("nan")
    signal = A.H_norm_pct.max() - A.H_norm_pct.min()

    say("=== SESSION A (main result) ===")
    say(f"  files                : {len(A)}")
    say(f"  current range        : {A.j.min():.0f} - {A.j.max():.0f} mA/cm2")
    say(f"  entropy range        : {A.H_norm_pct.min():.2f} - {A.H_norm_pct.max():.2f} %")
    say(f"  H_norm vs j (linear) : r = {rlin:.4f}   R2 = {rlin**2:.4f}")
    say(f"  H_norm vs ln(j)      : r = {rl:.4f}   R2 = {rl**2:.4f}   (logarithmic form)")
    if len(rep) > 1:
        say(f"  repeatability at 250 : +/- {sd:.2f} pp (1 sigma, n={len(rep)})")
        say(f"  total signal         : {signal:.2f} pp")
        say(f"  signal-to-noise      : {signal/sd:.1f} : 1")
    say()

    say("=== SESSION OFFSET ===")
    say(f"  Session A mean H_norm : {A.H_norm_pct.mean():.2f} %  (n={len(A)})")
    say(f"  Session B mean H_norm : {B.H_norm_pct.mean():.2f} %  (n={len(B)})")
    say(f"  offset                : {B.H_norm_pct.mean()-A.H_norm_pct.mean():.2f} pp")
    say(f"  offset is comparable to the {signal:.1f} pp Session A trend")
    say()

    Bg = B.dropna(subset=["dS"])
    rg = np.nan
    if len(Bg) >= 3:
        rg = np.corrcoef(Bg.dS, Bg.H_norm_pct)[0, 1]
        say("=== GIBBS CORRELATION (Session B only) ===")
        say(f"  n = {len(Bg)}   r = {rg:.4f}   R2 = {rg**2:.4f}")
        say()

    # ---------- figure ----------
    # Stacked so that the lettering stays legible at the printed width.
    fig, ax = plt.subplots(3, 1, figsize=(10, 17))
    plt.rcParams.update({"font.size": 15})

    g = A.sort_values("j")
    ax[0].plot(g.j, g.H_norm_pct, "o", ms=9, color="#1f77b4")
    z = np.polyfit(np.log(g.j), g.H_norm_pct, 1)
    xs = np.linspace(g.j.min(), g.j.max(), 200)
    ax[0].plot(xs, np.polyval(z, np.log(xs)), "--", color="gray",
               label=f"log fit  R² = {rl**2:.3f}")
    if len(rep) > 1:
        ax[0].errorbar([250], [rep.H_norm_pct.mean()], yerr=[sd], fmt="s",
                       color="#d62728", ms=9, capsize=6,
                       label=f"250 mA ×{len(rep)} repeats (±1σ)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Current density (mA/cm²)  [log scale]")
    ax[0].set_ylabel("H_norm (%)")
    ax[0].set_title(f"(a) Session A, n={len(A)}: logarithmic trend, r = {rl:.3f}", fontsize=16)
    ax[0].legend(fontsize=13)
    ax[0].grid(alpha=.3, which="both")

    ax[1].plot(A.j, A.H_norm_pct, "o", ms=8, color="#1f77b4",
               label=f"Session A, ANS (n = {len(A)} recordings), mean {A.H_norm_pct.mean():.1f} %")
    ax[1].plot(B.j, B.H_norm_pct, "s", ms=8, color="#d62728",
               label=f"Session B, ANS (n = {len(B)} recordings), mean {B.H_norm_pct.mean():.1f} %")
    ax[1].axhline(A.H_norm_pct.mean(), color="#1f77b4", ls=":", alpha=.6)
    ax[1].axhline(B.H_norm_pct.mean(), color="#d62728", ls=":", alpha=.6)
    xm = (A.j.max() + B.j.max()) / 2
    ax[1].annotate("", xy=(xm, A.H_norm_pct.mean()), xytext=(xm, B.H_norm_pct.mean()),
                   arrowprops=dict(arrowstyle="<->", color="k"))
    ax[1].text(xm * 1.03, (A.H_norm_pct.mean() + B.H_norm_pct.mean()) / 2,
               f"{B.H_norm_pct.mean()-A.H_norm_pct.mean():.1f} pp\noffset", fontsize=14)
    ax[1].set_xlabel("Current density (mA/cm²)")
    ax[1].set_ylabel("H_norm (%)")
    ax[1].set_title("(b) Baselines of Session A and Session B", fontsize=16)
    ax[1].legend(fontsize=13)
    ax[1].grid(alpha=.3)

    if len(Bg) >= 3:
        ax[2].plot(Bg.dS, Bg.H_norm_pct, "s", ms=9, color="#d62728")
        z2 = np.polyfit(Bg.dS, Bg.H_norm_pct, 1)
        xs2 = np.linspace(Bg.dS.min(), Bg.dS.max(), 50)
        ax[2].plot(xs2, np.polyval(z2, xs2), "--", color="gray")
        ax[2].set_title(f"(c) Gibbs correlation, Session B only, n={len(Bg)}, r = {rg:.3f}", fontsize=16)
    else:
        ax[2].text(.5, .5, "no voltage data", ha="center", transform=ax[2].transAxes)
        ax[2].set_title("(c) Gibbs correlation — unavailable")
    ax[2].set_xlabel("Gibbs entropy production  zFη/T  (J mol⁻¹K⁻¹)")
    ax[2].set_ylabel("H_norm (%)")
    ax[2].grid(alpha=.3)

    for a in ax:
        a.xaxis.label.set_size(17)
        a.yaxis.label.set_size(17)
        a.tick_params(axis="both", labelsize=15)
    plt.tight_layout()
    out = os.path.join(OUT, "final_results.png")
    plt.savefig(out, dpi=145)
    plt.close()

    with open(os.path.join(OUT, "session_stats.txt"), "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {out}")
    print(f"Wrote {os.path.join(OUT, 'session_stats.txt')}")


if __name__ == "__main__":
    main()
