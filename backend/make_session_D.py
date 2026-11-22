"""
make_session_D.py
=================
Analyses Session D and writes Section 4.14 of the report from the results.

    python3 make_session_D.py                 (folder Session_D/ next to this file)
    python3 make_session_D.py path/to/Session_D

Session D is ten recordings at 50, 100, ..., 500 mA/cm2: one session, a
constant step of 50 mA/cm2, and the cell voltage in every file name. It is the
series that Sections 4.4, 4.5 and 4.12.4 of the report ask for, so the report
contains a section for it that is filled from your data by this script.

It writes into figures_for_overleaf/ :

    session_D.tex       Section 4.14 with every number computed here, plus the
                        sentences for the abstract and the conclusion
    session_D.png       Figure 26
    session_D_values.txt  the same numbers as plain text

Upload session_D.tex and session_D.png into figures/ on Overleaf together with
the other figures. Without them the report shows a framed box at Section 4.14.

Nothing in the text is written by hand: every value, and every word that
depends on a value (for example "weak", "moderate" or "strong" for a
correlation), is derived from your recordings by the rules stated below.
"""
import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
OUT = "figures_for_overleaf"
EXT = (".avi", ".mp4", ".mov", ".mkv")

folder = next((a for a in sys.argv[1:] if not a.startswith("-")), "Session_D")
if not os.path.isdir(folder):
    sys.exit(f"Folder not found: {folder}")
vids = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(EXT))
if len(vids) < 3:
    sys.exit(f"Need at least three recordings in {folder}, found {len(vids)}.")
os.makedirs(OUT, exist_ok=True)

import pandas as pd
import run_all as R
import fisher_pixel as fp


def strength(r):
    """Fixed wording rule for a correlation coefficient."""
    a = abs(r)
    if not np.isfinite(r):
        return "undetermined"
    word = "weak" if a < 0.3 else ("moderate" if a < 0.7 else "strong")
    return word if a < 0.3 else word + (" positive" if r > 0 else " negative")


def num(x, d=2):
    return f"{x:.{d}f}"


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m} $\\times$ 10$^{{{int(e)}}}$"


# ---------------------------------------------------------------- 1. metrics
res_dir = "results_D"
summ = os.path.join(res_dir, "summary.csv")
if not os.path.exists(summ) or "--force" in sys.argv:
    print("1. run_all.py on", folder, "(a few minutes)")
    if os.path.isdir("results"):
        sys.exit("A folder named results/ exists; move or rename it first.")
    r = subprocess.run([sys.executable, "run_all.py", folder])
    if r.returncode != 0 or not os.path.isdir("results"):
        sys.exit("run_all.py failed")
    if os.path.isdir(res_dir):
        shutil.rmtree(res_dir)
    shutil.move("results", res_dir)
else:
    print(f"1. {summ} exists, not recomputed (use --force to redo)")

D = pd.read_csv(summ).sort_values("j").reset_index(drop=True)
D = D[D.kind == "video"]
n = len(D)
js = D.j.astype(int).tolist()
uj = sorted(set(js))
steps = sorted(set(b - a for a, b in zip(uj[:-1], uj[1:])))
repeats = n - len(uj)
r_log = float(np.corrcoef(np.log(D.j), D.H_norm_pct)[0, 1])
r_lin = float(np.corrcoef(D.j, D.H_norm_pct)[0, 1])
Dg = D.dropna(subset=["dS"])
r_gibbs = float(np.corrcoef(Dg.dS, Dg.H_norm_pct)[0, 1]) if len(Dg) >= 3 else float("nan")
span = float(D.H_norm_pct.max() - D.H_norm_pct.min())
t_min, t_max = float(D["T"].min()), float(D["T"].max())
dates = sorted({m.group(0) for m in (re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(p)) for p in vids) if m})
months = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
def _d(s):
    y, m, d = s.split("-")
    return f"{int(d)} {months[int(m) - 1]} {y}"
when = ("recorded on " + " and ".join(_d(x) for x in dates)) if dates else "recorded"

# comparison with Session A at the current densities both sessions contain
A = pd.read_csv("results_A/summary.csv") if os.path.exists("results_A/summary.csv") else None
offset_txt, common = "", []
if A is not None:
    Am = A.groupby("j").H_norm_pct.mean()
    common = [j for j in js if j in Am.index]
    if common:
        d = [float(D[D.j == j].H_norm_pct.iloc[0] - Am[j]) for j in common]
        offset_mean, offset_sd = float(np.mean(d)), float(np.std(d, ddof=1)) if len(d) > 1 else 0.0

# ------------------------------------------------------- 2. pixel Fisher
print("2. pixel-level Fisher information for every recording")
pix = []
for p in vids:
    j = fp.current_density(p)
    if j is None:
        continue
    fr = fp.read_grey(p, max_frames=400)
    mk = fp.overlay_mask(fr)
    F_t, _, I_mean = fp.fisher_temporal(fr, mk)
    pix.append(dict(j=j, ring=fp.ring_score(F_t, I_mean, mk), F=float(F_t.mean()),
                    I=I_mean, n=len(fr)))
pix.sort(key=lambda r: r["j"])
pairs = []
if len({r["I"].shape for r in pix}) == 1:
    for a, b in zip(pix[:-1], pix[1:]):
        F = fp.fisher_across_j(a["I"], b["I"], a["j"], b["j"])
        pairs.append(dict(a=a["j"], b=b["j"], mean=float(F.mean())))
ring = {r["j"]: r["ring"] for r in pix}

# ------------------------------------------------------------- 3. figure
print("3. figure")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 1, figsize=(10, 15.5))
lab, tick = 17, 15
ax[0].plot(D.j, D.H_norm_pct, "o", ms=9, color="#d62728", label=f"Session D (n={n})")
if A is not None:
    ax[0].plot(A.j, A.H_norm_pct, "s", ms=8, color="#1f77b4", alpha=0.8,
               label=f"Session A (n={len(A)})")
z = np.polyfit(np.log(D.j), D.H_norm_pct, 1)
xs = np.linspace(D.j.min(), D.j.max(), 200)
ax[0].plot(xs, np.polyval(z, np.log(xs)), "--", color="#d62728", lw=1.5,
           label=f"Session D, logarithmic fit, r = {r_log:.3f}")
ax[0].set_xlabel("Current density (mA/cm$^2$)", fontsize=lab)
ax[0].set_ylabel("$H_{norm}$ (%)", fontsize=lab)
ax[0].set_title("(a) Entropy against current density", fontsize=16)
ax[0].legend(fontsize=13); ax[0].grid(alpha=.3)

if len(Dg) >= 3:
    ax[1].plot(Dg.dS, Dg.H_norm_pct, "o", ms=9, color="#d62728")
    z2 = np.polyfit(Dg.dS, Dg.H_norm_pct, 1)
    xs2 = np.linspace(Dg.dS.min(), Dg.dS.max(), 50)
    ax[1].plot(xs2, np.polyval(z2, xs2), "--", color="gray", lw=1.5)
    for _, rr in Dg.iterrows():
        ax[1].annotate(f"{int(rr.j)}", (rr.dS, rr.H_norm_pct), textcoords="offset points",
                       xytext=(6, -12), fontsize=12)
ax[1].set_xlabel("Gibbs entropy production $zF\\eta/T$ (J mol$^{-1}$K$^{-1}$)", fontsize=lab)
ax[1].set_ylabel("$H_{norm}$ (%)", fontsize=lab)
ax[1].set_title(f"(b) Entropy against Gibbs entropy production, r = {r_gibbs:.3f}"
                "  (labels: mA/cm$^2$)", fontsize=16)
ax[1].grid(alpha=.3)

ax[2].plot([r["j"] for r in pix], [r["ring"] for r in pix], "o", ms=9, color="#9467bd")
ax[2].axhline(1.0, color="gray", ls=":", lw=1.2)
ax[2].set_xlabel("Current density (mA/cm$^2$)", fontsize=lab)
ax[2].set_ylabel("edge-to-flat ratio of $F_t$", fontsize=lab)
ax[2].set_title("(c) Pixel-level Fisher information: concentration on edges", fontsize=16)
ax[2].grid(alpha=.3)
for a_ in ax:
    a_.tick_params(labelsize=tick)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "session_D.png"), dpi=150, facecolor="white")
plt.close(fig)

# --------------------------------------------------------------- 4. text
print("4. Section 4.14")
better = "logarithmic" if abs(r_log) >= abs(r_lin) else "linear"
rows = []
for _, rr in D.iterrows():
    j = int(rr.j)
    rows.append(f"{j} & {int(rr.V) if not pd.isna(rr.V) else '--'} & "
                f"{num(rr['T'], 1)} & {num(rr.dS, 0) if not pd.isna(rr.dS) else '--'} & "
                f"{num(rr.H_norm_pct)} & {num(rr.mean_bubbles, 1)} & "
                f"{num(ring[j]) if j in ring else '--'} \\\\")
pair_txt = ""
if pairs:
    first, last = pairs[0], pairs[-1]
    if len(steps) == 1:
        pair_txt = (f"Between neighbouring current densities all {len(pairs)} pairs are "
                    f"{steps[0]} mA/cm$^2$ apart, so the values of $F_j$ are directly "
                    f"comparable without the correction of Section 4.12.4. The mean of $F_j$ is "
                    f"{sci(first['mean'])} between {first['a']} and {first['b']} mA/cm$^2$ and "
                    f"{sci(last['mean'])} between {last['a']} and {last['b']} mA/cm$^2$, a "
                    f"ratio of {first['mean'] / last['mean']:.1f}.")
    else:
        # F_j scales as 1/dj^2 (Section 4.12.4): compare F_j * dj^2
        fa = first['mean'] * (first['b'] - first['a']) ** 2
        fb = last['mean'] * (last['b'] - last['a']) ** 2
        pair_txt = (f"The {len(pairs)} pairs of neighbouring current densities are not "
                    f"evenly spaced, and $F_j$ scales as $1/\\Delta j^2$ (Section 4.12.4), so "
                    f"$F_j \\cdot \\Delta j^2$ is compared: {sci(fa)} between {first['a']} "
                    f"and {first['b']} mA/cm$^2$ and {sci(fb)} between {last['a']} and "
                    f"{last['b']} mA/cm$^2$, a ratio of {fa / fb:.1f}.")
offset_par = ""
if common:
    offset_par = (f"At the {len(common)} current densities contained in both Session A and "
                  f"Session D, $H_{{norm}}$ of Session D differs from that of Session A by "
                  f"{offset_mean:+.2f} $\\pm$ {offset_sd:.2f} percentage points (mean $\\pm$ "
                  f"standard deviation over the {len(common)} pairs), which is a further "
                  f"measurement of the session-to-session offset of Section 4.4.")
even = (f"a constant step of {steps[0]} mA/cm$^2$" if len(steps) == 1
        else "steps of " + ", ".join(str(s) for s in steps) + " mA/cm$^2$")
rings = [ring[j] for j in js if j in ring]
title = ("evenly spaced operating points with recorded voltage" if len(steps) == 1
         else "one session with recorded voltage")
has_all = ("Session D has all three." if len(steps) == 1 and not repeats else
           "Session D has a single session and a recorded voltage"
           + (", but its steps are not constant." if len(steps) > 1 else
              f", and {repeats} repeated operating point(s)."))

tex = f"""% Written by make_session_D.py from the recordings in {folder}/. Do not edit
% by hand: re-run the script instead.
\\renewcommand{{\\SDabstract}}{{ Session D, {when} with {n} operating points at
{even} and a recorded cell voltage, gives r = {r_gibbs:.3f} against the Gibbs entropy
production and r = {r_log:.3f} against the logarithm of the current density
(Section 4.14).}}
\\renewcommand{{\\SDconclusion}}{{ In Session D, {n} operating points at {even} within one
session with recorded voltage, the correlation of $H_{{norm}}$ with the Gibbs entropy
production is {strength(r_gibbs)} (r = {r_gibbs:.3f}), and with the logarithm of the current
density r = {r_log:.3f} (Section 4.14).}}
\\renewcommand{{\\SDsection}}{{%
\\subsection{{4.14 Session D: {title}}}

Sections 4.4, 4.5 and 4.12.4 identify three properties a data set needs for a direct
test of the relationship between image entropy and thermodynamic entropy production:
one recording session, a recorded cell voltage, and evenly spaced current densities.
{has_all} It was {when} and consists of {n} recordings at {', '.join(str(j) for j in js)}
mA/cm$^2$, {even}, with cell voltages in the file names and cell temperatures from
{num(t_min, 1)} to {num(t_max, 1)} \\textdegree C. The electrode is not stated in the file
names. All values below are computed by \\texttt{{make\\_session\\_D.py}} with the same
code as Sessions A to C (\\texttt{{run\\_all.py}}, \\texttt{{fisher\\_pixel.py}}).

\\begin{{center}}\\small
\\begin{{tabular}}{{rrrrrrr}}
\\hline
$j$ (mA/cm$^2$) & $V$ (mV) & $T$ (\\textdegree C) & $\\Delta S_{{irr}}$ & $H_{{norm}}$ (\\%) & bubbles/frame & edge/flat \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\end{{center}}

$\\Delta S_{{irr}}$ in J\\,mol$^{{-1}}$K$^{{-1}}$; bubbles per frame from background
subtraction; edge/flat is the ratio of Section 3.10 for $F_t$.

Against the current density, $H_{{norm}}$ gives r = {r_lin:.3f} for a linear and
r = {r_log:.3f} for a logarithmic dependence; the {better} form fits better. The total
change of $H_{{norm}}$ across the series is {span:.2f} percentage points. Against the
Gibbs entropy production the correlation is r = {r_gibbs:.3f} ({len(Dg)} recordings), a
{strength(r_gibbs)} correlation by the rule used in this section ($|r| < 0.3$ weak,
$0.3 \\le |r| < 0.7$ moderate, $|r| \\ge 0.7$ strong). For comparison, Session B
gives r = 0.453 with sixteen unevenly spaced recordings (Section 4.5).

{offset_par}

The edge-to-flat ratio of the pixel-level Fisher information ranges from
{min(rings):.2f} to {max(rings):.2f} across the series (Figure 26 (c)). {pair_txt}

\\figslot{{session_D.png}}{{Session D.}}{{\\textbf{{Figure 26 --- Session D.}} (a) $H_{{norm}}$
against current density for Session D, with Session A for comparison and the logarithmic
fit for Session D. (b) $H_{{norm}}$ against the Gibbs entropy production, each point
labelled with its current density. (c) Edge-to-flat ratio of $F_t$ for each recording; the
dotted line marks a uniform map. No lines connect measured points.}}
}}
"""
open(os.path.join(OUT, "session_D.tex"), "w").write(tex)

with open(os.path.join(OUT, "session_D_values.txt"), "w") as f:
    f.write(f"recordings {n}\ncurrent densities {js}\nsteps {steps}\n"
            f"temperature {t_min}-{t_max}\nr linear {r_lin:.4f}\nr log {r_log:.4f}\n"
            f"r Gibbs {r_gibbs:.4f}\nspan {span:.2f}\n"
            f"edge/flat {[(r['j'], round(r['ring'], 2)) for r in pix]}\n"
            f"F_j {[(p['a'], p['b'], p['mean']) for p in pairs]}\n")
    if common:
        f.write(f"offset vs Session A {offset_mean:+.2f} +/- {offset_sd:.2f} over {common}\n")
print(open(os.path.join(OUT, "session_D_values.txt")).read())
print(f"wrote {OUT}/session_D.tex and {OUT}/session_D.png")
