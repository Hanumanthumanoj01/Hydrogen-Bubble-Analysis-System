"""
make_sessions.py
================
Analyses Sessions D, E and F and writes Sections 4.14 to 4.17 of the report
from the results.

    python3 make_sessions.py              (folders Session_D/, Session_E/, Session_F/
                                           next to this file)
    python3 make_sessions.py --force      recompute everything

For every session folder it runs the same code as for Sessions A to C
(run_all.py for entropy, bubbles and Gibbs entropy production; fisher_pixel.py
for the pixel-level Fisher information) and writes into figures_for_overleaf/:

    sessions_DEF.tex      the entropy overview at the start of Chapter 4,
                          Sections 4.14 (Session D), 4.15 (Session E),
                          4.16 (Session F) and 4.17 (all six sessions), plus the
                          sentences for the abstract and the conclusion
    entropy_overview.png  Figure 16
    session_D.png  session_E.png  session_F.png  sessions_all.png
                          Figures 28 to 31
    sessions_values.txt   every number of those sections as plain text

Upload all of them into figures/ on Overleaf together with the other figures.
Without sessions_DEF.tex the report shows a framed box at Section 4.14.

Nothing in the text is written by hand. Every value, and every word that
depends on a value ("weak", "moderate", "strong", "fits better", ...), is
derived from the recordings by the rules stated in the section.

Rules used
----------
* correlation strength: |r| < 0.3 weak, 0.3 <= |r| < 0.7 moderate, else strong
* recordings are grouped by the words left in the file name once numbers,
  units and dates are removed (for example "3D" and "mesh" in Session E);
  each group with at least three recordings gets its own correlations
* a recording that is the same file (same name and size) as one in an
  earlier session is analysed with its session but counted once in the
  pooled values of Section 4.17
* repeatability: standard deviation of H_norm over recordings at the same
  current value within one session and group
"""
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
OUT = "figures_for_overleaf"
EXT = (".avi", ".mp4", ".mov", ".mkv")
FORCE = "--force" in sys.argv
os.makedirs(OUT, exist_ok=True)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_all as R
import fisher_pixel as fp

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728", "E": "#9467bd",
          "F": "#8c564b", "screenshots": "#7f7f7f"}
SEC = {"D": "4.14", "E": "4.15", "F": "4.16"}
FIG = {"D": 28, "E": 29, "F": 30}
FIG_ENG = 32
SEC_ENG = "4.18"

# Electrode of each session, as confirmed by the laboratory (Prof. Wagner):
# A, B, C, D and F were recorded on the ANS electrode; Session E compares a
# 3D structure with the standard mesh electrode, a gas diffusion electrode
# (GDE) of Raney nickel rolled onto nickel mesh, at the same geometric area.
ELECTRODE = {"A": "ANS", "B": "ANS", "C": "ANS", "D": "ANS", "E": "3D structure / GDE",
             "F": "ANS"}
EXPERIMENT = {"A": "current-density series with repeats",
              "B": "current-density series, large bubbles",
              "C": "nucleation-site recordings",
              "D": "evenly spaced current densities, voltage recorded",
              "E": "3D structure against mesh GDE, same area",
              "F": "repeat of Session D"}


def electrode(tag, group=None):
    """Electrode label for a session, or for one group of file names in it."""
    g = (group or "").lower()
    if "mesh" in g:
        return "GDE"
    if "3d" in g:
        return "3D structure"
    return ELECTRODE.get(tag, "")


def legend(tag, n, group=None):
    """Self-explaining legend entry, e.g. 'Session A, ANS (n = 12 recordings)'."""
    g = "" if group in (None, "all") else f", {group}"
    e = electrode(tag, group)
    return (f"Session {tag}{g}" + (f", {e}" if e and e.lower() not in g.lower() else "")
            + f" (n = {n} recordings)")


import constants as C                                  # noqa: E402
V_TH = C.V_THERMONEUTRAL                               # V
KWH_PER_KG_V = 2 * 96485.33 / 2.01588e-3 / 3.6e6       # kWh per kg H2 per volt of cell voltage


# ------------------------------------------------------------------ helpers
def strength(r):
    if r is None or not np.isfinite(r):
        return "undetermined"
    a = abs(r)
    word = "weak" if a < 0.3 else ("moderate" if a < 0.7 else "strong")
    return word if a < 0.3 else word + (" positive" if r > 0 else " negative")


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def rtxt(r):
    return "--" if r is None else f"{r:.3f}"


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m} $\\times$ 10$^{{{int(e)}}}$"


def esc(s):
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def date_of(name):
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", name)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if 1 <= mo <= 12 and 1 <= d <= 31:
        return f"{d} {MONTHS[mo - 1]} {y}"
    return None


FILLER = {"bubble", "bubbles", "video", "videos", "recording", "rec", "bigbubbles", "session",
          "copy", "final", "repeat", "repeated", "rep", "again", "wdh", "wiederholung", "take",
          "second", "new", "run", "test", "measurement", "messung", "flattend", "flattened"}
UNIT = re.compile(r"^\d+(?:[.,]\d+)?(ma|mv|deg|degc|c|fps|v|a|cm2|s|min)?$")


def group_of(name):
    """Words left in the file name once numbers, units, dates and filler words
    such as 'repeat' are removed; '3D' and 'mesh' are kept."""
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"20\d{2}[-_.]?\d{2}[-_.]?\d{2}", " ", s)
    words = [w for w in re.split(r"[\s_\-]+", s) if w]
    keep = [w for w in words if not UNIT.match(w.lower()) and w.lower() not in FILLER
            and not re.fullmatch(r"\d+(?:[.,]\d+)?", w)]
    return " ".join(keep) or "all"


def find_folder(tag):
    for d in (f"Session_{tag}", f"session_{tag}", f"Session {tag}", f"session {tag}",
              f"../Session_{tag}", f"../Session {tag}"):
        if os.path.isdir(d):
            return d
    return None


def videos(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(EXT))


def date_text(paths):
    ds = []
    for p in paths:
        d = date_of(os.path.basename(p))
        if d and d not in ds:
            ds.append(d)
    if not ds:
        return None
    return ds[0] if len(ds) == 1 else ", ".join(ds[:-1]) + " and " + ds[-1]


def unit_of(js):
    """Values above 2000 in the file names are total currents (for example
    19 000 mA on a large electrode), not current densities."""
    return "mA" if max(js) > 2000 else "mA/cm$^2$"


def steps_of(js):
    u = sorted(set(js))
    return sorted(set(b - a for a, b in zip(u[:-1], u[1:])))


def step_text(js, unit="mA/cm$^2$"):
    st = steps_of(js)
    if not st:
        return "a single current value"
    if len(st) == 1:
        return f"a constant step of {st[0]} {unit}"
    return "uneven steps (" + ", ".join(str(s) for s in st) + f" {unit})"


def ratio(x):
    return f"{x:.2g}" if x < 1 else f"{x:.1f}"


# ----------------------------------------------------------- Sessions A-C
prev = {}
for tag, d in (("A", "results_A"), ("B", "results_B"), ("C", "results_C"),
               ("screenshots", "results_screenshots")):
    p = os.path.join(d, "summary.csv")
    if os.path.exists(p):
        prev[tag] = pd.read_csv(p)
A = prev.get("A")
seen = set()                      # (name, size) of files already counted
for tag, folder in (("A", "session_A"), ("B", "session_B"), ("C", "session_C")):
    if os.path.isdir(folder):
        for p in videos(folder):
            seen.add((os.path.basename(p).lower(), os.path.getsize(p)))


# ------------------------------------------------------ one session's work
def analyse(tag):
    folder = find_folder(tag)
    if folder is None or len(videos(folder)) == 0:
        print(f"Session {tag}: no folder Session_{tag}/ with recordings, skipped")
        return None
    vids = videos(folder)
    res = f"results_{tag}"
    summ = os.path.join(res, "summary.csv")
    if FORCE or not os.path.exists(summ):
        print(f"\n=== Session {tag}: run_all.py on {folder} ({len(vids)} recordings)")
        R.main(folder, out_dir=res)
    else:
        print(f"\n=== Session {tag}: {summ} exists, not recomputed (--force to redo)")
    D = pd.read_csv(summ)
    D = D[D.kind == "video"].copy()
    size = {os.path.basename(p): os.path.getsize(p) for p in vids}
    labels = [group_of(f) for f in D.file]
    canon = {}
    for l in labels:
        canon.setdefault(l.lower(), l)
    D["group"] = [canon[l.lower()] for l in labels]
    if D.group.nunique() > 4:                     # names too varied to group on
        D["group"] = "all"
    D["dup"] = [(f.lower(), size.get(f, -1)) in seen for f in D.file]
    for f in D.file:
        seen.add((f.lower(), size.get(f, -1)))
    D = D.sort_values(["group", "j"]).reset_index(drop=True)

    # pixel-level Fisher information, one recording at a time
    print(f"    pixel-level Fisher information, {len(vids)} recordings")
    pix = {}
    for p in vids:
        j = fp.current_density(p)
        if j is None:
            continue
        try:
            import cv2
            cap = cv2.VideoCapture(p)
            w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            nmax = int(max(50, min(400, 4.5e8 / max(1.0, w * h * 4))))
            fr = fp.read_grey(p, max_frames=nmax)
            mk = fp.overlay_mask(fr)
            F_t, _, I_mean = fp.fisher_temporal(fr, mk)
            try:
                ps = fp.path_share(fr, mk)[0]
            except ValueError:
                ps = dict(share=np.nan, stable=False)
            pix[os.path.basename(p)] = dict(j=j, ring=fp.ring_score(F_t, I_mean, mk), I=I_mean,
                                            share=ps["share"], stable=ps["stable"])
            del fr
        except Exception as e:
            print(f"      {os.path.basename(p)}: {type(e).__name__}: {e}")
    D["ring"] = [pix[f]["ring"] if f in pix else np.nan for f in D.file]
    D["paths"] = [100 * pix[f]["share"] if f in pix else np.nan for f in D.file]
    D["stable"] = [bool(pix[f]["stable"]) if f in pix else False for f in D.file]

    groups = []
    for g, G in D.groupby("group", sort=False):
        js = G.j.astype(int).tolist()
        Gv = G.dropna(subset=["dS"])
        rep = [float(np.std(x.H_norm_pct, ddof=1)) for _, x in G.groupby("j") if len(x) > 1]
        # F_j between neighbouring current values; repeats at one value are averaged
        acc = {}
        for f, j in zip(G.file, G.j):
            if f in pix:
                acc.setdefault(int(j), []).append(pix[f]["I"])
        first = {j: (np.mean(v, axis=0) if len({x.shape for x in v}) == 1 else v[0])
                 for j, v in acc.items()}
        pairs = []
        u = sorted(first)
        if len({first[j].shape for j in u}) == 1:
            for a, b in zip(u[:-1], u[1:]):
                Fm = float(fp.fisher_across_j(first[a], first[b], a, b).mean())
                pairs.append(dict(a=a, b=b, F=Fm, Fn=Fm * (b - a) ** 2))
        groups.append(dict(
            name=g, n=len(G), js=js, vmin=min(js), vmax=max(js),
            r_log=corr(np.log(G.j), G.H_norm_pct), r_lin=corr(G.j, G.H_norm_pct),
            r_gibbs=corr(Gv.dS, Gv.H_norm_pct), n_v=len(Gv),
            span=float(G.H_norm_pct.max() - G.H_norm_pct.min()),
            rep=float(np.mean(rep)) if rep else None, n_rep=len(rep),
            rings=[x for x in G.ring if np.isfinite(x)], pairs=pairs,
            paths=[x for x in G.paths if np.isfinite(x)], n_stable=int(G.stable.sum())))

    offset = None
    if A is not None and max(D.j) <= 2000:
        Am = A.groupby("j").H_norm_pct.mean()
        common = sorted(set(int(j) for j in D.j) & set(int(j) for j in Am.index))
        if len(common) >= 2:
            dd = [float(D[D.j == j].H_norm_pct.mean() - Am[j]) for j in common]
            offset = dict(n=len(common), mean=float(np.mean(dd)), sd=float(np.std(dd, ddof=1)))
    return dict(tag=tag, folder=folder, D=D, groups=groups, offset=offset,
                when=date_text(vids), T=(float(D["T"].min()), float(D["T"].max())))


S = {}
for tag in ("D", "E", "F"):
    s = analyse(tag)
    if s:
        S[tag] = s

# ------------------------------------------------------------------ figures
MARK = ["o", "s", "^", "D"]
GCOL = ["", "#17becf", "#bcbd22", "#e377c2"]


def session_figure(s):
    D, tag = s["D"], s["tag"]
    fig, ax = plt.subplots(4, 1, figsize=(10, 20.5))
    lab, tick = 17, 15
    n_v = int(D.dS.notna().sum())
    for k, (g, G) in enumerate(D.groupby("group", sort=False)):
        lbl = legend(tag, len(G), g)
        ax[0].plot(G.j, G.H_norm_pct, MARK[k % 4], ms=9, color=(COLORS[tag] if k == 0 else GCOL[k % 4]),
                   label=lbl)
        Gv = G.dropna(subset=["dS"])
        if len(Gv):
            ax[1].plot(Gv.dS, Gv.H_norm_pct, MARK[k % 4], ms=9,
                       color=(COLORS[tag] if k == 0 else GCOL[k % 4]), label=lbl)
            for _, rr in (Gv.iterrows() if n_v <= 15 else []):
                ax[1].annotate(f"{int(rr.j)}", (rr.dS, rr.H_norm_pct), textcoords="offset points",
                               xytext=(6, -12), fontsize=11)
        Gr = G[np.isfinite(G.ring)]
        ax[2].plot(Gr.j, Gr.ring, MARK[k % 4], ms=9, color=(COLORS[tag] if k == 0 else GCOL[k % 4]),
                   label=lbl)
        Gp = G[np.isfinite(G.paths)]
        col = COLORS[tag] if k == 0 else GCOL[k % 4]
        for st, fill in ((True, col), (False, "white")):
            X = Gp[Gp.stable == st]
            if len(X):
                ax[3].plot(X.j, X.paths, MARK[k % 4], ms=9, color=col, mfc=fill,
                           label=lbl + ("" if st else ", not above shifted region"))
    if A is not None and D.j.max() <= 1000:
        ax[0].plot(A.j, A.H_norm_pct, "s", ms=7, color=COLORS["A"], alpha=0.6,
                   label=legend("A", len(A)))
    if D.j.max() / max(1, D.j.min()) > 20:
        for a_ in (ax[0], ax[2], ax[3]):
            a_.set_xscale("log")
    u = unit_of(D.j.astype(int).tolist())
    ax[0].set_xlabel(f"Current value in the file name ({u})", fontsize=lab)
    ax[0].set_ylabel("$H_{norm}$ (%)", fontsize=lab)
    ax[0].set_title("(a) Entropy against current", fontsize=16)
    ax[1].set_xlabel("Gibbs entropy production $zF\\eta/T$ (J mol$^{-1}$K$^{-1}$)", fontsize=lab)
    ax[1].set_ylabel("$H_{norm}$ (%)", fontsize=lab)
    ax[1].set_title("(b) Entropy against Gibbs entropy production"
                    + (" (labels: current)" if n_v <= 15 else ""), fontsize=16)
    ax[2].axhline(1.0, color="gray", ls=":", lw=1.2)
    ax[2].set_xlabel(f"Current value in the file name ({u})", fontsize=lab)
    ax[2].set_ylabel("edge-to-flat ratio of $F_t$", fontsize=lab)
    ax[2].set_title("(c) Pixel-level Fisher information: concentration on edges", fontsize=16)
    ax[3].axhline(5.0, color="gray", ls=":", lw=1.2)
    ax[3].set_xlabel(f"Current value in the file name ({u})", fontsize=lab)
    ax[3].set_ylabel("path share (%)", fontsize=lab)
    ax[3].set_title("(d) Stable bubble paths (Section 4.9.5); dotted: random, 5 %", fontsize=16)
    for a_ in ax:
        a_.tick_params(labelsize=tick); a_.grid(alpha=.3)
        if a_.get_legend_handles_labels()[0]:
            a_.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"session_{tag}.png"), dpi=150, facecolor="white")
    plt.close(fig)


for s in S.values():
    session_figure(s)

# pooled over all sessions, each file once
pool = []
for tag, P in prev.items():
    for _, r in P.iterrows():
        pool.append(dict(tag=tag, j=r.j, dS=r.dS, H=r.H_norm_pct))
for tag, s in S.items():
    for _, r in s["D"].iterrows():
        if not r.dup:
            pool.append(dict(tag=tag, j=r.j, dS=r.dS, H=r.H_norm_pct))
P = pd.DataFrame(pool)
Pv = P.dropna(subset=["dS"]) if len(P) else P
r_pool = corr(Pv.dS, Pv.H) if len(Pv) else None
r_pool_abc = None
if prev:
    Q = Pv[Pv.tag.isin(["A", "B", "C", "screenshots"])]
    r_pool_abc = corr(Q.dS, Q.H)
within = {}
for tag in ("A", "B", "C"):
    Q = Pv[Pv.tag == tag] if len(Pv) else Pv
    within[tag] = (corr(Q.dS, Q.H), len(Q)) if len(Q) else (None, 0)
for tag in ("D", "E", "F"):          # every recording of the session, as in its section
    if tag in S:
        Q = S[tag]["D"].dropna(subset=["dS"])
        within[tag] = (corr(Q.dS, Q.H_norm_pct), len(Q))
    else:
        within[tag] = (None, 0)

if len(Pv):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for tag in ("A", "B", "C", "screenshots", "D", "E", "F"):
        Q = Pv[Pv.tag == tag]
        if len(Q):
            name = (f"single images (n = {len(Q)} pictures)" if tag == "screenshots"
                    else legend(tag, len(Q)))
            ax.plot(Q.dS, Q.H, "o", ms=8, color=COLORS[tag], label=name)
    ax.set_xlabel("Gibbs entropy production $zF\\eta/T$ (J mol$^{-1}$K$^{-1}$)", fontsize=17)
    ax.set_ylabel("$H_{norm}$ (%)", fontsize=17)
    ax.set_title(f"All sessions with a recorded voltage, each file once: r = {rtxt(r_pool)}\n"
                 "ANS: Sessions A-D, F; Session E: 3D structure and GDE", fontsize=15)
    ax.tick_params(labelsize=15); ax.grid(alpha=.3); ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sessions_all.png"), dpi=150, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------- text
def group_par(s, g):
    G = s["D"][s["D"].group == g["name"]]
    who = (f"In Session {s['tag']} " if g["name"] == "all"
           else f"For the recordings named \\emph{{{esc(g['name'])}}} ")
    t = []
    if g["n"] < 3:
        return (f"{who}there are only {g['n']} recording(s), too few for a "
                f"correlation; $H_{{norm}}$ is "
                + ", ".join(f"{h:.2f} \\%" for h in G.H_norm_pct) + ".")
    # "fits better" only where there is a relationship to fit (|r| >= 0.3) and the
    # two forms differ by at least 0.02 in |r|; otherwise say so plainly
    better = None
    if g["r_log"] is not None and g["r_lin"] is not None:
        a_log, a_lin = abs(g["r_log"]), abs(g["r_lin"])
        if max(a_log, a_lin) < 0.3:
            better = "none"
        elif abs(a_log - a_lin) < 0.02:
            better = "equal"
        else:
            better = "logarithmic" if a_log > a_lin else "linear"
    u = unit_of(s["D"].j.astype(int).tolist())
    t.append(f"{who}({g['n']} recordings, {g['vmin']} to {g['vmax']} {u}, "
             f"{step_text(g['js'], u)}), $H_{{norm}}$ gives r = {rtxt(g['r_lin'])} against the "
             f"current value and r = {rtxt(g['r_log'])} against its logarithm"
             + ({None: "", "none": "; neither form shows a relationship (|r| < 0.3)",
                 "equal": "; both forms fit equally well"}.get(better,
                                                              f"; the {better} form fits better"))
             + f". The total change of $H_{{norm}}$ is {g['span']:.2f} percentage points.")
    if g["r_gibbs"] is not None:
        t.append(f"Against the Gibbs entropy production the correlation is "
                 f"r = {g['r_gibbs']:.3f} ({g['n_v']} recordings with a voltage), "
                 f"{strength(g['r_gibbs'])}.")
    else:
        t.append("A correlation with the Gibbs entropy production is not possible, because "
                 "fewer than three of these recordings carry a voltage and a temperature in "
                 "the file name.")
    if g["rep"] is not None:
        t.append(f"At the {g['n_rep']} current value(s) recorded more than once, the standard "
                 f"deviation of $H_{{norm}}$ is {g['rep']:.2f} percentage points on average, "
                 f"against 1.09 points for the three repeats of Session A (Section 4.3).")
    if g["rings"]:
        t.append(f"The edge-to-flat ratio of $F_t$ ranges from {min(g['rings']):.2f} to "
                 f"{max(g['rings']):.2f}.")
    if g["paths"]:
        t.append(f"The path share lies between {min(g['paths']):.0f} and {max(g['paths']):.0f} "
                 f"\\% (random: 5 \\%); {g['n_stable']} of {len(g['paths'])} recordings show "
                 f"stable paths.")
    if g["pairs"]:
        f0, f1 = g["pairs"][0], g["pairs"][-1]
        if len(steps_of(g["js"])) == 1:
            t.append(f"All {len(g['pairs'])} pairs of neighbouring current values are equally "
                     f"spaced, so $F_j$ is compared directly: {sci(f0['F'])} between {f0['a']} "
                     f"and {f0['b']} and {sci(f1['F'])} between {f1['a']} and {f1['b']} "
                     f"{u}, a ratio of {ratio(f0['F'] / f1['F'])}.")
        elif len(g["pairs"]) >= 2:
            t.append(f"The pairs of neighbouring current values are not equally spaced, and "
                     f"$F_j$ scales as $1/\\Delta j^2$ (Section 4.12.4), so "
                     f"$F_j \\cdot \\Delta j^2$ is compared: {sci(f0['Fn'])} between {f0['a']} "
                     f"and {f0['b']} and {sci(f1['Fn'])} between {f1['a']} and {f1['b']} "
                     f"{u}, a ratio of {ratio(f0['Fn'] / f1['Fn'])}.")
    return " ".join(t)


def session_tex(s):
    tag, D = s["tag"], s["D"]
    gs = s["groups"]
    js = D.j.astype(int).tolist()
    even = len(steps_of(js)) == 1 and len(set(js)) == len(js)
    has_v = int(D.dS.notna().sum())
    if tag == "D":
        title = ("evenly spaced operating points with recorded voltage" if even and has_v == len(D)
                 else "one session with recorded voltage")
    else:
        title = ("two electrodes" if len(gs) == 2 else f"{len(D)} recordings")
    u = unit_of(js)
    intro = (f"{'It' if tag == 'D' else 'Session ' + tag} "
             f"{'was recorded on ' + s['when'] + ' and consists' if s['when'] else 'consists'} "
             f"of {len(D)} recordings, with current values from {min(js)} to {max(js)} {u} in "
             f"the file names"
             + (" (total currents, not current densities)" if u == "mA" else "") + "; "
             f"{has_v} of them also carry the cell voltage and temperature, from "
             f"{s['T'][0]:.1f} to {s['T'][1]:.1f} \\textdegree C.")
    if len(gs) > 1:
        intro += (" The file names distinguish " + str(len(gs)) + " groups ("
                  + ", ".join(f"\\emph{{{esc(g['name'])}}}: {g['n']}" for g in gs)
                  + "), which are analysed separately below.")
    if tag == "D":
        intro = ("Sections 4.4, 4.5 and 4.12.4 identify three properties a data set needs for "
                 "a direct test of the relationship between image entropy and thermodynamic "
                 "entropy production: one recording session, a recorded cell voltage, and "
                 "evenly spaced current densities. "
                 + ("Session D has all three. " if even and has_v == len(D) else
                    "Session D has a single session and a recorded voltage, but not all three "
                    "properties. ") + intro)
    ndup = int(D.dup.sum())
    if ndup:
        intro += (f" {ndup} of these recordings are the same files (same name and size) as "
                  f"recordings of an earlier session; they are analysed here with Session {tag} "
                  f"but counted only once in Section 4.17.")
    intro += (" All values are computed by \\texttt{make\\_sessions.py} with the same code as "
              "for Sessions A to C (\\texttt{run\\_all.py}, \\texttt{fisher\\_pixel.py}).")
    rows = []
    for _, r in D.iterrows():
        rows.append(" & ".join([
            str(int(r.j)),
            "--" if pd.isna(r.V) else str(int(r.V)),
            "--" if pd.isna(r["T"]) else f"{r['T']:.1f}",
            "--" if pd.isna(r.dS) else f"{r.dS:.0f}",
            f"{r.H_norm_pct:.2f}", f"{r.mean_bubbles:.1f}",
            "--" if not np.isfinite(r.ring) else f"{r.ring:.2f}",
            "--" if not np.isfinite(r.paths) else f"{r.paths:.0f}" + ("" if r.stable else "*")]
            + ([esc(r.group) + (f" ({electrode(tag, r.group)})"
                                if electrode(tag, r.group).lower() not in
                                ("", ELECTRODE.get(tag, "").lower(), r.group.lower()) else "")]
               if len(gs) > 1 else [])) + " \\\\")
    gcol = len(gs) > 1
    head = ("$j$ & $V$ & $T$ & $\\Delta S_{irr}$ & $H_{norm}$ & bubbles & edge/flat & paths"
            + (" & group (electrode)" if gcol else "") + " \\\\\n"
            f"({u}) & (mV) & (\\textdegree C) & & (\\%) & per frame & & (\\%)"
            + (" &" if gcol else "") + " \\\\")
    table = ("{\\setlength{\\tabcolsep}{3.5pt}\\footnotesize\\begin{longtable}{rrrrrrrr"
             + ("p{3.2cm}" if gcol else "") + "}\n\\hline\n" + head +
             "\n\\hline\n\\endhead\n" + "\n".join(rows) + "\n\\hline\n\\end{longtable}}\n")
    pars = "\n\n".join(group_par(s, g) for g in gs)
    off = ""
    if s["offset"]:
        o = s["offset"]
        off = (f"\n\nAt the {o['n']} current densities contained in both Session A and Session "
               f"{tag}, $H_{{norm}}$ of Session {tag} differs from Session A by "
               f"{o['mean']:+.2f} $\\pm$ {o['sd']:.2f} percentage points (mean $\\pm$ standard "
               f"deviation over the pairs), a further measurement of the session-to-session "
               f"offset of Section 4.4.")
    n = FIG[tag]
    return (f"\\subsection{{{SEC[tag]} Session {tag}: {title}}}\n\n{intro}\n\n"
            f"{table}\n$\\Delta S_{{irr}}$ in J\\,mol$^{{-1}}$K$^{{-1}}$; bubbles per frame from "
            f"background subtraction; edge/flat is the ratio of Section 3.10 for $F_t$; paths is the "
            f"path share of Section 4.9.5, marked * where it is not above the shifted region. "
            f"Electrode: {ELECTRODE[tag]}.\n\n"
            f"{pars}{off}\n\n"
            f"\\figslot{{session_{tag}.png}}{{Session {tag}.}}{{\\textbf{{Figure {n} --- Session "
            f"{tag}, {ELECTRODE[tag]} ({EXPERIMENT[tag]}).}} (a) $H_{{norm}}$ against the current value in the file name"
            + (", with Session A for comparison" if A is not None and D.j.max() <= 1000 else "")
            + ". (b) $H_{norm}$ against the Gibbs entropy production, each point labelled with "
              "its current value" + (" (labels left out when there are more than 15 points)"
                                       if D.dS.notna().sum() > 15 else "")
            + ". (c) Edge-to-flat ratio of $F_t$ for each recording; the "
              "dotted line marks a uniform map. (d) Path share: the part of the Fisher "
              "information of the second half of a recording that falls on the 5 \\% of "
              "pixels marked in the first half; filled where it lies above the shifted "
              "region. n is the number of recordings. No lines connect measured points.}\n")


def missing_tex(tag):
    return (f"\\subsection{{{SEC[tag]} Session {tag}}}\n\n\\fbox{{\\begin{{minipage}}{{0.9\\linewidth}}"
            f"\\centering\\textcolor{{red}}{{\\bfseries No recordings found for Session {tag}}}"
            f"\\\\[0.3em]\\small Put them in \\texttt{{backend/Session\\_{tag}/}} and run "
            f"\\texttt{{python3 make\\_sessions.py}}.\\end{{minipage}}}}\n")


def all_tex():
    rows = []
    names = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F"}
    counts = {}
    for tag in ("A", "B", "C", "D", "E", "F"):
        if tag in prev:
            X = prev[tag]; nuniq = len(X)
        elif tag in S:
            X = S[tag]["D"]; nuniq = int((~X.dup).sum())
        else:
            continue
        counts[tag] = nuniq
        r, nv = within[tag]
        Am = prev["A"].H_norm_pct.mean() if "A" in prev else None
        rows.append(" & ".join([
            names[tag], {"E": "3D, GDE"}.get(tag, ELECTRODE[tag]), EXPERIMENT[tag], str(len(X)), str(nuniq), f"{int(X.j.min())}--{int(X.j.max())}",
            f"{X.H_norm_pct.mean():.2f}",
            "--" if Am is None else f"{X.H_norm_pct.mean() - Am:+.2f}",
            str(int(X.dS.notna().sum())), rtxt(r)]) + " \\\\")
    table = ("{\\setlength{\\tabcolsep}{2.5pt}\\footnotesize\n"
             "\\begin{longtable}{llp{2.9cm}rrrrrrr}\n\\hline\nSession & electrode & experiment & "
             "files & counted & $j$ range & mean $H_{norm}$ & vs.\\ A & with $V$ & r \\\\\n"
             " & & & & & & (\\%) & (pp) & & (Gibbs) \\\\\n\\hline\n"
             "\\endhead\n" + "\n".join(rows) + "\n\\hline\n\\end{longtable}}\n")
    t = ("\\subsection{4.17 All six sessions}\n\nThe table collects the six sessions. "
         "\\emph{Counted} excludes recordings that are the same file as in an earlier session; "
         "\\emph{vs.\\ A} is the difference of the session mean of $H_{norm}$ from the mean of "
         "Session A, in percentage points (pp) of $H_{norm}$; \\emph{r (Gibbs)} is the correlation of $H_{norm}$ with the Gibbs entropy "
         "production within the session, where at least three recordings carry a voltage.\n\n"
         + table)
    within_r = [(k, within[k][0]) for k in ("B", "C", "D", "E", "F") if within[k][0] is not None]
    if within_r:
        t += ("Within single sessions, the correlation with the Gibbs entropy production is "
              + "; ".join(f"{strength(r)} in Session {k} (r = {r:.3f})" for k, r in within_r)
              + ". ")
    if r_pool is not None:
        t += (f"Pooled over all {len(Pv)} files with a recorded voltage, each file counted once, "
              f"r = {r_pool:.3f} ({strength(r_pool)}), against r = 0.283 for the 21 files of "
              f"Sessions A to C (Section 4.4). ")
    t += ("Pooling mixes sessions whose entropy baselines differ (column \\emph{vs.\\ A}), so the "
          "within-session values are the direct test (Figure 31).\n\n"
          "\\figslot{sessions_all.png}{All sessions.}{\\textbf{Figure 31 --- All sessions.} "
          "$H_{norm}$ against the Gibbs entropy production for every file with a recorded "
          "voltage, coloured by session. The legend gives the electrode of each session (ANS "
          "for Sessions A to D and F; Session E compares a 3D structure with the mesh GDE) and "
          "n, the number of recordings with a voltage; a file present in two sessions is "
          "shown once.}\n")
    return t


# ------------------------------------------------ entropy overview (Note 3)
def overview_rows():
    """One row per session: the entropy results that the report is about."""
    out = []
    for tag in ("A", "B", "C", "D", "E", "F"):
        if tag in prev:
            X = prev[tag]
        elif tag in S:
            X = S[tag]["D"]
        else:
            continue
        X = X[np.isfinite(X.H_norm_pct) & (X.j > 0)]
        if not len(X):
            continue
        lj = np.log(X.j.astype(float))
        r = corr(lj, X.H_norm_pct)
        b = a = None
        if len(X) >= 3 and np.std(lj) > 0:
            b, a = np.polyfit(lj, X.H_norm_pct, 1)
        Xv = X.dropna(subset=["dS"])
        out.append(dict(tag=tag, n=len(X), jmin=int(X.j.min()), jmax=int(X.j.max()),
                        unit=unit_of(X.j.astype(int).tolist()), mean=float(X.H_norm_pct.mean()),
                        span=float(X.H_norm_pct.max() - X.H_norm_pct.min()), r=r, a=a, b=b,
                        rg=corr(Xv.dS, Xv.H_norm_pct), nv=len(Xv), X=X))
    return out


OV = overview_rows()
if OV:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for k, o in enumerate(OV):
        X = o["X"]
        hollow = o["unit"] == "mA"
        ax.plot(X.j, X.H_norm_pct, "o", ms=8, color=COLORS[o["tag"]],
                mfc="white" if hollow else COLORS[o["tag"]],
                label=legend(o["tag"], o["n"]) + (", total current in mA" if hollow else ""))
        if o["b"] is not None:
            jj = np.geomspace(o["jmin"], o["jmax"], 50)
            ax.plot(jj, o["a"] + o["b"] * np.log(jj), "--", color=COLORS[o["tag"]], lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Current density in the file name (mA/cm$^2$; Session E: mA)", fontsize=16)
    ax.set_ylabel("$H_{norm}$ (%)", fontsize=17)
    ax.set_title("Entropy of every recording; dashed: fit $H_{norm} = a + b\\,\\ln j$ "
                 "within each session", fontsize=15)
    ax.tick_params(labelsize=14); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "entropy_overview.png"), dpi=150, facecolor="white")
    plt.close(fig)


def overview_tex():
    if not OV:
        return ""
    rows = []
    for o in OV:
        per2 = "--" if o["b"] is None else f"{o['b'] * np.log(2):+.2f}"
        rows.append(" & ".join([
            o["tag"], {"E": "3D, GDE"}.get(o["tag"], ELECTRODE[o["tag"]]), str(o["n"]), f"{o['jmin']}--{o['jmax']}" + (" mA" if o["unit"] == "mA" else ""),
            f"{o['mean']:.2f}", f"{o['span']:.2f}", rtxt(o["r"]), per2,
            rtxt(o["rg"]) if o["nv"] >= 3 else "--"]) + " \\\\")
    table = ("{\\setlength{\\tabcolsep}{2.5pt}\\footnotesize\n"
             "\\begin{longtable}{llrrrrrrr}\n\\hline\n"
             "Session & electrode & files & $j$ & mean $H_{norm}$ & change & r & per doubling & r \\\\\n"
             " & & & (mA/cm$^2$) & (\\%) & (pp) & ($\\ln j$) & of $j$ (pp) & (Gibbs) \\\\\n"
             "\\hline\n\\endhead\n" + "\n".join(rows) + "\n\\hline\n\\end{longtable}}\n")
    rising = [o for o in OV if o["r"] is not None and o["r"] >= 0.3]
    falling = [o for o in OV if o["r"] is not None and o["r"] <= -0.3]
    flat = [o for o in OV if o["r"] is not None and abs(o["r"]) < 0.3]
    names = lambda L: ", ".join(o["tag"] for o in L) or "none"
    slopes = [o["b"] * np.log(2) for o in rising if o["b"] is not None]
    means = [o["mean"] for o in OV]
    t = ("\\subsection{Overview: entropy in all six sessions}\n\n"
         "This overview collects the entropy results of every session before they are "
         "discussed one by one; it is written by \\texttt{make\\_sessions.py} from the "
         "batch results of Sessions A to F. For each session, $H_{norm}$ is fitted as "
         "$H_{norm} = a + b \\ln j$, the form that Section 4.2 found to fit Session A best; "
         "\\emph{per doubling} is $b \\ln 2$, the change of $H_{norm}$ when the current "
         "density doubles, and \\emph{change} is the difference between the largest and "
         "the smallest value of the session.\n\n" + table + "\n")
    t += (f"$H_{{norm}}$ rises with the logarithm of the current density "
          f"(r $\\geq$ 0.3) in Sessions {names(rising)}"
          + (f", by {min(slopes):.2f} to {max(slopes):.2f} percentage points per doubling"
             if slopes else "")
          + f"; the relationship is weak (|r| < 0.3) in Sessions {names(flat)}"
          + (f" and falling in Sessions {names(falling)}" if falling else "") + ". "
          f"The session means lie between {min(means):.2f} and {max(means):.2f} \\%, a spread "
          f"of {max(means) - min(means):.2f} percentage points, which is why the sessions "
          f"are compared by their trends and not pooled (Section 4.4). "
          "The correlation with the Gibbs entropy production is given where at least three "
          "recordings of a session carry a voltage; Sections 4.14 to 4.17 discuss it.\n\n"
          "\\figslot{entropy_overview.png}{Entropy in all six sessions.}{\\textbf{Figure 16 "
          "--- Entropy in all six sessions.} $H_{norm}$ of every recording against the current "
          "value in its file name, on a logarithmic axis, coloured by session; the dashed "
          "curves are the fits $H_{norm} = a + b \\ln j$ (slope in the table), drawn over the range of each session. Session E "
          "(hollow) is labelled in total current (mA), not current density.}\n")
    return t


# ------------------------------------- engineering comparison (Section 4.18)
#
# Prof. Wagner: "try to present the engineering point of view and try to show
# how the different methods can be used to show which structure works more
# efficiently." Session E is the one session in which two electrode
# structures of the same geometric area were recorded at the same currents.
#
# Rules used
#   * reference = the group whose name contains "mesh" (the standard GDE);
#     candidate = the largest other group with at least three recordings
#   * values are averaged per current value within a group; only current
#     values present in both groups, with a voltage, are compared
#   * voltage efficiency eps = V_therm / V (higher heating value basis)
#   * electrical energy per kg of hydrogen  W = 2 F V / M_H2 = 26.59 kWh/kg per volt
#   * a method "separates" the two structures when the difference has the
#     same sign at no fewer than 80 % of the common current values
#   * "trend" is the correlation of the quantity with ln(current) over all
#     recordings of the group; words follow the strength rule above

METHODS = [("V", "cell voltage $V$ (mV)", "{:.0f}"),
           ("eps", "voltage efficiency $V_{therm}/V$ (\\%)", "{:.1f}"),
           ("dS", "$\\Delta S_{irr}$ (J\\,mol$^{-1}$K$^{-1}$)", "{:.0f}"),
           ("H", "$H_{norm}$ (\\%)", "{:.2f}"),
           ("b", "bubbles per frame", "{:.1f}"),
           ("ring", "edge/flat of $F_t$", "{:.2f}"),
           ("paths", "path share (\\%)", "{:.0f}")]


def engineering(s):
    if s is None:
        return None
    D = s["D"].copy()
    D["eps"] = 100 * V_TH * 1000 / D.V
    ref = next((g["name"] for g in s["groups"] if "mesh" in g["name"].lower()), None)
    cands = [g for g in s["groups"] if g["name"] != ref and g["n"] >= 3]
    if ref is None or not cands:
        return None
    cand = max(cands, key=lambda g: g["n"])["name"]

    def per_j(g):
        G = D[D.group == g]
        return G.groupby(G.j.astype(int))[["V", "T", "dS", "eps", "H_norm_pct", "mean_bubbles",
                                           "ring", "paths"]].mean().rename(
            columns={"H_norm_pct": "H", "mean_bubbles": "b"})
    a, b = per_j(cand), per_j(ref)
    common = sorted(set(a.dropna(subset=["V"]).index) & set(b.dropna(subset=["V"]).index))
    if len(common) < 2:
        return None
    a, b = a.loc[common], b.loc[common]
    meth = []
    for key, name, fmt in METHODS:
        d = (a[key] - b[key]).dropna()
        if len(d) < 2:
            continue
        lower = int((d < 0).sum())
        k = max(lower, int((d > 0).sum()))
        tr = []
        for g in (cand, ref):
            G = D[D.group == g].dropna(subset=[key if key in D else "V"])
            col = {"H": "H_norm_pct", "b": "mean_bubbles"}.get(key, key)
            G = G.dropna(subset=[col])
            tr.append(corr(np.log(G.j.astype(float)), G[col]) if len(G) >= 3 else None)
        meth.append(dict(key=key, name=name, fmt=fmt, a=float(a[key].median()),
                         b=float(b[key].median()), lower=lower, higher=int((d > 0).sum()), n=len(d),
                         sep=k >= np.ceil(0.8 * len(d)), cand_lower=lower > len(d) / 2,
                         tr_a=tr[0], tr_b=tr[1]))
    return dict(cand=cand, ref=ref, a=a, b=b, common=common, meth=meth, D=D,
                others=[g["name"] for g in s["groups"] if g["name"] not in (cand, ref)])


ENG = engineering(S.get("E"))


def eng_name(g):
    e = electrode("E", g)
    return f"{e} (\\emph{{{esc(g)}}})" if e.lower() not in g.lower() else e


def engineering_figure(e):
    D = e["D"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 11.5))
    ax = ax.ravel()
    panels = [("V", "cell voltage $V$ (mV)", "(a) Cell voltage"),
              ("dS", "$\\Delta S_{irr} = zF\\eta/T$ (J mol$^{-1}$K$^{-1}$)",
               "(b) Thermodynamic entropy production"),
              ("H_norm_pct", "$H_{norm}$ (%)", "(c) Image entropy"),
              ("paths", "path share (%)", "(d) Stable bubble paths (Fisher)")]
    if not D.paths.notna().any():
        panels[3] = ("ring", "edge/flat of $F_t$", "(d) Fisher information on edges")
    for k, (g, col, mk) in enumerate(((e["cand"], "#9467bd", "o"), (e["ref"], "#e377c2", "s"))):
        G = D[D.group == g]
        for i, (key, yl, ttl) in enumerate(panels):
            X = G.dropna(subset=[key])
            ax[i].plot(X.j, X[key], mk, ms=8, color=col, alpha=0.85,
                       label=f"{electrode('E', g)}, {g} (n = {len(X)} recordings)"
                       if electrode("E", g).lower() not in g.lower()
                       else f"{g} (n = {len(X)} recordings)")
    ax[0].axhline(V_TH * 1000, color="gray", ls=":", lw=1.3)
    ax[0].text(ax[0].get_xlim()[0], V_TH * 1000 + 8, "  $V_{therm}$ = 1481 mV", fontsize=12,
               color="gray", va="bottom")
    if panels[3][0] == "paths":
        ax[3].axhline(5, color="gray", ls=":", lw=1.3)
    for i, (key, yl, ttl) in enumerate(panels):
        if not D[key].notna().any():
            ax[i].text(0.5, 0.5, "not available for these recordings", ha="center",
                       va="center", fontsize=14, color="gray", transform=ax[i].transAxes)
    for i, (key, yl, ttl) in enumerate(panels):
        ax[i].set_xlabel("total current (mA)", fontsize=15)
        ax[i].set_ylabel(yl, fontsize=15)
        ax[i].set_title(ttl, fontsize=15.5)
        ax[i].tick_params(labelsize=13); ax[i].grid(alpha=.3)
        ax[i].legend(fontsize=11.5)
    fig.suptitle("Session E: two electrode structures of the same geometric area, "
                 "recorded in one session", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(os.path.join(OUT, "engineering_E.png"), dpi=150, facecolor="white")
    plt.close(fig)


def engineering_tex(e):
    if e is None:
        return (f"\\subsection{{{SEC_ENG} Which electrode structure works more efficiently?}}\n\n"
                "\\fbox{\\begin{minipage}{0.9\\linewidth}\\centering\\textcolor{red}{\\bfseries "
                "Needs Session E with a mesh group and a second group at common currents}"
                "\\end{minipage}}\n")
    a, b, cn, rn = e["a"], e["b"], eng_name(e["cand"]), eng_name(e["ref"])
    c0, c1 = e["common"][0], e["common"][-1]
    dV = (a.V - b.V)
    rel = 100 * dV / b.V
    lower = int((dV < 0).sum())
    who_low, who_high = (cn, rn) if lower > len(dV) / 2 else (rn, cn)
    consistent = max(lower, len(dV) - lower) >= np.ceil(0.8 * len(dV))
    rows = []
    for j in e["common"]:
        rows.append(" & ".join([str(j), f"{a.V[j]:.0f}", f"{b.V[j]:.0f}", f"{dV[j]:+.0f}",
                                f"{a.eps[j]:.1f}", f"{b.eps[j]:.1f}", f"{a.dS[j]:.0f}",
                                f"{b.dS[j]:.0f}", f"{a['T'][j]:.1f}", f"{b['T'][j]:.1f}",
                                f"{(dV[j] / 1000) * KWH_PER_KG_V:+.2f}"]) + " \\\\")
    t1 = ("{\\setlength{\\tabcolsep}{3pt}\\footnotesize\n\\begin{longtable}{rrrrrrrrrrr}\n"
          "\\hline\n$I$ & \\multicolumn{2}{c}{$V$ (mV)} & $\\Delta V$ & "
          "\\multicolumn{2}{c}{$V_{therm}/V$ (\\%)} & \\multicolumn{2}{c}{$\\Delta S_{irr}$} & "
          "\\multicolumn{2}{c}{$T$ (\\textdegree C)} & $\\Delta W$ \\\\\n"
          "(mA) & 3D & GDE & (mV) & 3D & GDE & 3D & GDE & 3D & GDE & (kWh/kg) \\\\\n"
          "\\hline\n\\endhead\n" + "\n".join(rows) + "\n\\hline\n\\end{longtable}}\n")
    mrows = []
    for m in e["meth"]:
        mrows.append(" & ".join([
            m["name"], m["fmt"].format(m["a"]), m["fmt"].format(m["b"]),
            f"{m['lower']} of {m['n']}", rtxt(m["tr_a"]), rtxt(m["tr_b"]),
            "yes" if m["sep"] else "no"]) + " \\\\")
    t2 = ("{\\setlength{\\tabcolsep}{3.5pt}\\footnotesize\n\\begin{longtable}{lrrrrrl}\n"
          "\\hline\nmethod & 3D & GDE & 3D lower & trend 3D & trend GDE & separates \\\\\n"
          " & (median) & (median) & at & r ($\\ln I$) & r ($\\ln I$) & \\\\\n\\hline\n"
          "\\endhead\n" + "\n".join(mrows) + "\n\\hline\n\\end{longtable}}\n")
    M = {m["key"]: m for m in e["meth"]}
    Ta, Tb = float(a["T"].mean()), float(b["T"].mean())
    jmax = e["common"][-1]
    t = (f"\\subsection{{{SEC_ENG} Which electrode structure works more efficiently?}}\n\n"
         f"Session E is the only session in which two electrode structures were recorded in one "
         f"session at the same currents and, as stated by the laboratory, with the same "
         f"geometric area: the {cn} and the standard mesh electrode (\\emph{{{esc(e['ref'])}}}), a "
         f"gas diffusion electrode (GDE) of Raney nickel rolled onto nickel mesh. This makes it possible to "
         f"compare the structures from an engineering point of view and, at the same time, to "
         f"test what each method of this work contributes to such a comparison. "
         + ("The groups " + ", ".join(f"\\emph{{{esc(g)}}}" for g in e["others"])
            + " were recorded under other conditions and are not part of the comparison. "
            if e["others"] else "")
         + f"Values are averaged per current; {len(e['common'])} current values from {c0} to "
           f"{c1} mA were recorded for both structures with a voltage. All numbers below are "
           f"written by \\texttt{{make\\_sessions.py}}.\n\n"
         "\\paragraph{Efficiency.} The electrical measures come first, because they decide "
         "efficiency: at a given current, the electrode that needs the lower cell voltage "
         "produces the same amount of hydrogen (Faraday's law) with less electrical energy. "
         "The voltage efficiency is $\\varepsilon = V_{therm}/V$, and the electrical energy per "
         "kilogram of hydrogen is $W = 2FV/M_{H_2} = 26.6$~kWh\\,kg$^{-1}$ per volt.\n\n"
         + t1 +
         "$\\Delta V$ and $\\Delta W$: 3D minus GDE; $\\Delta S_{irr}$ in "
         "J\\,mol$^{-1}$K$^{-1}$.\n\n"
         + (f"The {who_low} needs the lower voltage at {max(lower, len(dV) - lower)} of the "
            f"{len(dV)} common currents. " if consistent else
            f"Neither structure needs the lower voltage consistently (3D lower at {lower} of "
            f"{len(dV)} common currents). ")
         + f"The difference (3D minus GDE) is {dV.iloc[0]:+.0f} mV at {c0} mA and "
         f"{dV.iloc[-1]:+.0f} mV at {c1} mA ({rel.iloc[-1]:+.1f} \\%); the voltage efficiency "
         f"is {a.eps.min():.1f} to {a.eps.max():.1f} \\% for the 3D structure and "
         f"{b.eps.min():.1f} to {b.eps.max():.1f} \\% for the GDE. At {jmax} mA the difference "
         f"corresponds to {abs(dV.iloc[-1]) / 1000 * KWH_PER_KG_V:.1f} kWh less electrical energy "
         f"per kilogram of hydrogen for the {who_low}. ")
    if (lower > len(dV) / 2 and Ta < Tb) or (lower <= len(dV) / 2 and Tb < Ta):
        t += (f"The {who_low} ran colder on average ({min(Ta, Tb):.1f} against "
              f"{max(Ta, Tb):.1f} \\textdegree C). A warmer cell normally needs less voltage, so "
              f"the temperature difference works against the result and does not explain it. ")
    else:
        t += (f"The {who_low} also ran warmer ({max(Ta, Tb):.1f} against {min(Ta, Tb):.1f} "
              f"\\textdegree C), which lowers the voltage by itself, so part of the difference "
              f"may come from temperature. ")
    if "dS" in M:
        t += (f"$\\Delta S_{{irr}}$ is computed from the same voltage (Equation~\\ref{{eq:gibbs}}), "
              f"so it repeats this ranking in thermodynamic units: the {who_low} produces less "
              f"entropy at {M['dS']['lower'] if who_low == cn else M['dS']['n'] - M['dS']['lower']}"
              f" of {M['dS']['n']} currents.\n\n")
    t += ("\\paragraph{What the image methods add.} The camera methods do not measure "
          "efficiency; they show how the two structures release their gas. The table compares "
          "every method on the common currents: the median of each structure, at how many "
          "currents the 3D structure has the lower value, and the trend of each structure with "
          "the current.\n\n" + t2 +
          "\\emph{Separates}: the difference has the same sign at no fewer than 80 \\% of the "
          "common currents.\n\n")
    img = [m for m in e["meth"] if m["key"] in ("H", "b", "ring", "paths")]
    sep = [m for m in img if m["sep"]]
    nos = [m for m in img if not m["sep"]]
    clean = lambda m: re.sub(r"\s*\(.*?\)$", "", m["name"])
    if sep:
        t += ("Of the image methods, " + ", ".join(clean(m) for m in sep)
              + (" separate" if len(sep) > 1 else " separates")
              + " the two structures consistently"
              + ("; " + ", ".join(clean(m) for m in nos) + (" do" if len(nos) > 1 else " does")
                 + " not" if nos else "") + ". ")
    else:
        t += "None of the image methods separates the two structures consistently. "
    if "H" in M and M["H"]["tr_a"] is not None and M["H"]["tr_b"] is not None:
        ha, hb = M["H"]["tr_a"], M["H"]["tr_b"]
        word = lambda r: "rises" if r >= 0.3 else ("falls" if r <= -0.3 else "barely changes")
        t += (f"The entropy reacts differently to the current: $H_{{norm}}$ {word(ha)} with the "
              f"current on the 3D structure (r = {ha:.3f}) and {word(hb)} on the GDE "
              f"(r = {hb:.3f}). ")
        if (ha >= 0.3 and hb <= -0.3) or (ha <= -0.3 and hb >= 0.3):
            t += ("Because the two structures respond in opposite directions, the image entropy "
                  "is not one universal function of the operating point: it carries the "
                  "signature of the electrode structure. ")
    if "paths" in M:
        p = M["paths"]
        hi, cnt = (("3D structure", p["higher"]) if p["higher"] >= p["lower"]
                   else ("GDE", p["lower"]))
        t += (f"The path share is {p['a']:.0f} \\% for the 3D structure and {p['b']:.0f} \\% for "
              f"the GDE (medians); "
              + (f"the {hi} releases its gas along more stable paths. " if p["sep"]
                 else f"the {hi} is higher at only {cnt} of {p['n']} currents, which is not "
                      f"consistent enough to rank the two structures by this measure. "))
    t += ("\n\n\\paragraph{Engineering reading.} The methods therefore play different roles. "
          "The cell voltage and $\\Delta S_{irr}$ rank the structures by efficiency; they need "
          "electrical contact to the cell. Entropy and Fisher information need only the camera "
          "and show \\emph{how} the structures differ: how much of the picture is changing "
          "(entropy), whether the bubbles return to the same places (path share), and how "
          "strongly the information sits on bubble rims (edge/flat). In a development programme "
          "the camera methods are therefore a diagnosis tool next to the polarisation curve: a "
          "new structure is first ranked by its voltage, and the images then show whether its "
          "gas release behaves like the better or the worse reference. Three limits apply: the "
          "comparison rests on one session; the equal geometric area is taken from the "
          "laboratory's statement; and the correlation between bubble behaviour and voltage "
          "does not by itself prove that the bubbles cause the voltage difference, because the "
          "structures also differ in active surface and in ohmic and charge-transfer "
          "resistance.\n\n"
          f"\\figslot{{engineering_E.png}}{{Electrode structures compared.}}{{\\textbf{{Figure "
          f"{FIG_ENG} --- Electrode structures compared, Session E.}} 3D structure (circles) "
          f"and mesh GDE, Raney nickel on nickel mesh (squares), same geometric area, one "
          f"session. (a) Cell voltage; dotted: thermoneutral voltage. (b) Gibbs entropy "
          f"production. (c) Normalised image entropy. (d) Path share of the pixel-level Fisher "
          f"information; dotted: random placement. n is the number of recordings; x axis: "
          f"total current in mA.}}\n")
    return t


if ENG:
    engineering_figure(ENG)

body = []
for tag in ("D", "E", "F"):
    body.append(session_tex(S[tag]) if tag in S else missing_tex(tag))
body.append(all_tex())
body.append(engineering_tex(ENG))

abs_parts, con_parts = [], []
if "D" in S:
    g = S["D"]["groups"][0]
    abs_parts.append(f"Session D, {len(S['D']['D'])} recordings at {step_text(g['js'])} with a "
                     f"recorded cell voltage, gives r = {rtxt(g['r_gibbs'])} against the Gibbs "
                     f"entropy production and r = {rtxt(g['r_log'])} against the logarithm of "
                     f"the current density (Section 4.14).")
    con_parts.append(f"In Session D the correlation of $H_{{norm}}$ with the Gibbs entropy "
                     f"production is {strength(g['r_gibbs'])} (r = {rtxt(g['r_gibbs'])}), and "
                     f"with the logarithm of the current density r = {rtxt(g['r_log'])} "
                     f"(Section 4.14).")
ef = [t for t in ("E", "F") if t in S]
if ef:
    def gtxt(t):
        return ", ".join(f"r = {rtxt(g['r_gibbs'])}"
                         + (f" for the recordings named \\emph{{{esc(g['name'])}}}"
                            if g["name"] != "all" else "")
                         for g in S[t]["groups"] if g["r_gibbs"] is not None) \
            or "too few recordings with a voltage for a correlation"
    con_parts.append("Against the Gibbs entropy production, " + "; ".join(
        f"Session {t} gives {gtxt(t)} (Section {SEC[t]})" for t in ef) + ".")
if ENG:
    _dV = ENG["a"].V - ENG["b"].V
    _low = "3D structure" if (_dV < 0).sum() > len(_dV) / 2 else "mesh GDE"
    abs_parts.append(f"Comparing two electrode structures of the same area in Session E, the "
                     f"{_low} needs up to {abs(_dV).max():.0f} mV less cell voltage at equal "
                     f"current, and the image methods show a different gas release for the two "
                     f"structures (Section {SEC_ENG}).")
    con_parts.append(f"From the engineering point of view (Section {SEC_ENG}), the cell voltage "
                     f"and $\\Delta S_{{irr}}$ rank the electrode structures of Session E: the "
                     f"{_low} needs up to {abs(_dV).max():.0f} mV less cell voltage at equal current "
                     f"(between {ENG['common'][0]} and {ENG['common'][-1]} mA), about "
                     f"{abs(_dV.iloc[-1]) / 1000 * KWH_PER_KG_V:.1f} kWh less electrical energy per "
                     f"kilogram of hydrogen at the highest common current. Entropy and Fisher "
                     f"information need only the camera and show how the gas release of the two "
                     f"structures differs, which makes them a diagnosis tool next to the "
                     f"polarisation curve.")
if r_pool is not None:
    abs_parts.append(f"Pooled over all six sessions ({len(Pv)} files with a voltage, each counted "
                     f"once) the correlation is r = {r_pool:.3f} (Section 4.17).")
    con_parts.append(f"Pooled over all six sessions, r = {r_pool:.3f} ({strength(r_pool)}; "
                     f"Section 4.17).")

tex = ("% Written by make_sessions.py from the recordings of Sessions D, E and F.\n"
       "% Do not edit by hand: re-run the script instead.\n"
       "\\renewcommand{\\SDabstract}{" + ("\\par\\textbf{Sessions D to F.} " + " ".join(abs_parts)
                                         if abs_parts else "") + "}\n"
       "\\renewcommand{\\SDconclusion}{" + ("\\par\\textbf{Sessions D to F.} " + " ".join(con_parts)
                                           if con_parts else "") + "}\n"
       "\\renewcommand{\\SDsection}{%\n" + "\n".join(body) + "}\n"
       "\\renewcommand{\\SDoverview}{%\n" + overview_tex() + "}\n")
open(os.path.join(OUT, "sessions_DEF.tex"), "w").write(tex)

with open(os.path.join(OUT, "sessions_values.txt"), "w") as f:
    for tag, s in S.items():
        f.write(f"Session {tag}  folder {s['folder']}  recordings {len(s['D'])}  "
                f"duplicates {int(s['D'].dup.sum())}\n")
        for g in s["groups"]:
            f.write(f"  group {g['name']}: n {g['n']}  j {sorted(set(g['js']))}  "
                    f"steps {steps_of(g['js'])}\n"
                    f"    r linear {rtxt(g['r_lin'])}  r log {rtxt(g['r_log'])}  "
                    f"r Gibbs {rtxt(g['r_gibbs'])} (n {g['n_v']})  span {g['span']:.2f}  "
                    f"repeat sd {g['rep']}\n"
                    f"    edge/flat {[round(x, 2) for x in g['rings']]}\n"
                    f"    path share % {[round(x) for x in g['paths']]}  stable {g['n_stable']}\n"
                    f"    F_j {[(p['a'], p['b'], float('%.4e' % p['F'])) for p in g['pairs']]}\n")
        if s["offset"]:
            f.write(f"  offset vs Session A {s['offset']['mean']:+.2f} +/- "
                    f"{s['offset']['sd']:.2f} over {s['offset']['n']} values\n")
    for o in OV:
        if o["b"] is not None:
            f.write(f"overview fit Session {o['tag']}: H_norm = {o['a']:.2f} + {o['b']:.3f} ln j\n")
    f.write(f"pooled r Gibbs, all sessions {rtxt(r_pool)} over {len(Pv)} files\n")
    f.write(f"pooled r Gibbs, A-C {rtxt(r_pool_abc)}\n")
    for k, (r, n) in within.items():
        f.write(f"within {k}: r {rtxt(r)} (n {n})\n")
print("\n" + open(os.path.join(OUT, "sessions_values.txt")).read())
print(f"wrote {OUT}/sessions_DEF.tex and the figures session_D/E/F.png, sessions_all.png")
