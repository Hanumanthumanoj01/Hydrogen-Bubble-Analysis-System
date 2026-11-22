"""
check_results.py
================
Acceptance test for the report. Compares every number the report quotes from
your data with what your own files and scripts produce, and prints PASS or
FAIL for each, with the value found and the value the report states.

Run it in backend/ AFTER make_all_figures.py:

    python3 check_results.py           about 1 minute
    python3 check_results.py --full    also the MP4/AVI comparison and the
                                       time-step test, about 3 minutes

It changes nothing. Send the whole output back if any line says FAIL.
"""
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
FULL = "--full" in sys.argv
OUT = "figures_for_overleaf"

results = []


def check(name, got, want, tol=0.0, section=""):
    """want may be a number, a string, or a callable returning True/False."""
    if got is None:
        ok, shown = None, "not found"
    elif callable(want):
        ok, shown = bool(want(got)), got
    elif isinstance(want, (int, float)) and isinstance(got, (int, float, np.floating)):
        ok, shown = abs(float(got) - float(want)) <= tol + 1e-12, got
    else:
        ok, shown = str(got) == str(want), got
    mark = {True: "PASS", False: "FAIL", None: "skip"}[ok]
    exp = "" if callable(want) else f"   report: {want}"
    results.append((mark, name))
    print(f"  [{mark}] {section:<7} {name:<52} {shown}{exp}")


def section(title):
    print("\n" + title)
    print("-" * 78)


def read_csv(path):
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------- figures
section("1. Figure files in figures_for_overleaf/")
try:
    import make_all_figures as maf
    names = [f for f, _, _ in maf.FIGURES]
except Exception:
    names = []
have = [n for n in names if os.path.exists(os.path.join(OUT, n))]
check("figure files present (32 figures + sessions_DEF.tex + fisher_paths.tex)", len(have),
      len(names) or 34,
      section="App. A")
for n in names:
    if n not in have:
        print(f"           missing: {n}")

# ---------------------------------------------------------- batch results
section("2. Batch results (results_A, results_B, results_C, results_screenshots)")
A = read_csv("results_A/summary.csv")
B = read_csv("results_B/summary.csv")
C = read_csv("results_C/summary.csv")
S = read_csv("results_screenshots/summary.csv")
if A is not None:
    A = A.sort_values("j")
    check("Session A recordings", len(A), 12, section="4.2")
    rl = np.corrcoef(np.log(A.j), A.H_norm_pct)[0, 1]
    rlin = np.corrcoef(A.j, A.H_norm_pct)[0, 1]
    check("Session A, r against ln j", round(rl, 3), 0.911, 0.0005, "4.2")
    check("Session A, R2 against ln j", round(rl ** 2, 3), 0.830, 0.0005, "4.2")
    check("Session A, r against j (linear)", round(rlin, 3), 0.876, 0.0005, "4.2")
    rep = A[A.j == 250].H_norm_pct.tolist()
    check("250 mA/cm2 repeats", sorted(rep), sorted([75.49, 74.46, 76.63]), section="4.3")
    sd = float(np.std(rep, ddof=1)) if len(rep) > 1 else None
    check("repeatability (1 sigma, pp)", None if sd is None else round(sd, 2), 1.09, 0.005, "4.3")
    sig = A.H_norm_pct.max() - A.H_norm_pct.min()
    check("total change, Session A (pp)", round(sig, 2), 5.74, 0.005, "4.3")
    check("Session A mean H_norm (%)", round(A.H_norm_pct.mean(), 2), 75.28, 0.005, "4.4")
    check("Session A mean bubbles per frame", round(A.mean_bubbles.mean(), 1), 8.3, 0.05, "4.4")
if B is not None:
    check("Session B recordings", len(B), 16, section="4.4")
    check("Session B mean H_norm (%)", round(B.H_norm_pct.mean(), 2), 79.93, 0.005, "4.4")
    check("Session B mean bubbles per frame", round(B.mean_bubbles.mean(), 1), 19.8, 0.05, "4.4")
    Bg = B.dropna(subset=["dS"])
    check("Session B, r against Gibbs entropy", round(np.corrcoef(Bg.dS, Bg.H_norm_pct)[0, 1], 3),
          0.453, 0.0005, "4.5")
if A is not None and B is not None:
    check("offset Session B - Session A (pp)",
          round(B.H_norm_pct.mean() - A.H_norm_pct.mean(), 2), 4.65, 0.005, "4.4")
if C is not None:
    check("Session C recordings", len(C), 5, section="4.4")
    check("Session C mean H_norm (%)", round(C.H_norm_pct.mean(), 2), 80.23, 0.005, "4.4")
    row = C[C.j == 688]
    if len(row):
        check("688 mA/cm2: background-subtraction records", int(row.total_records.iloc[0]), 5541,
              section="3.7")
        check("688 mA/cm2: detections per frame", float(row.mean_bubbles.iloc[0]), 27.7, 0.05, "3.7")
if B is not None and C is not None:
    import pandas as pd
    pooled = pd.concat([d for d in (A, B, C, S) if d is not None], ignore_index=True)
    pooled = pooled.dropna(subset=["dS", "H_norm_pct"])
    check("files with a cell voltage", len(pooled), 21, section="4.4")
    check("pooled r against Gibbs entropy", round(np.corrcoef(pooled.dS, pooled.H_norm_pct)[0, 1], 3),
          0.283, 0.0005, "4.4")
if S is not None:
    check("single images: mean H_norm (%)", round(S.H_norm_pct.mean(), 2), 82.11, 0.005, "4.4")
    check("single images: mean bubbles", round(S.mean_bubbles.mean()), 95, 0.5, "4.4")
    s721 = S[S.j == 721].H_norm_pct.tolist()
    if len(s721) == 2:
        check("two 721 mA/cm2 captures differ by (pp)", round(abs(s721[0] - s721[1]), 2), 0.07,
              0.005, "4.3")

# ------------------------------------------------ Session C Fisher spacing
section("3. Session C bubble-table Fisher maps (results_C/tables)")
try:
    import csv
    from fisher_analysis import build_state_distribution, fisher_information
    tabs = {}
    for f in sorted(glob.glob("results_C/tables/bubbles_*.csv")):
        m = re.search(r"_(\d+)mA", f)
        if m:
            tabs[int(m.group(1))] = [{k: float(v) for k, v in r.items()}
                                     for r in csv.DictReader(open(f))]
    if len(tabs) >= 2:
        d = {j: build_state_distribution(tabs[j], (480, 640), 20)[0] for j in sorted(tabs)}
        js = sorted(d)
        raw, norm, empty = [], [], []
        for a, b in zip(js[:-1], js[1:]):
            I, e = fisher_information(d[a], d[b], b - a)
            raw.append(I.max()); norm.append(I.max() * (b - a) ** 2); empty.append(100 * e)
            print(f"           {a:>4} -> {b:<4} dj={b - a:>4}  I_max={I.max():.3e}  "
                  f"I_max*dj^2={I.max() * (b - a) ** 2:.3e}  empty={100 * e:.1f}%")
        check("peak Fisher spread across pairs", round(max(raw) / min(raw)), 233, 1, "4.12.4")
        check("spread after multiplying by dj^2", round(max(norm) / min(norm), 1), 9.0, 0.05, "4.12.4")
        check("empty cells, lowest (%)", round(min(empty), 1), 75.3, 0.05, "4.12.4")
        check("empty cells, highest (%)", round(max(empty), 1), 83.4, 0.05, "4.12.4")
    else:
        check("Session C bubble tables", None, "5 files")
except Exception as e:
    check("Session C Fisher", f"ERROR {e}", "ok")

# ------------------------------------------------------ machine learning
section("3b. Machine learning (training_report.json, written by train_from_videos.py)")
try:
    import json
    tr = json.load(open("training_report.json"))
    check("leave-one-recording-out MAE (mA/cm2)", round(tr["mae_leave_one_video_out"], 1), 238.2, 0.05, "4.7")
    check("frames", tr["n_frames"], 370, section="4.7")
    fi = tr["feature_importance"]
    check("importance shannon_H", round(fi["shannon_H"], 3), 0.328, 0.0005, "4.7")
    check("importance H_norm", round(fi["H_norm"], 3), 0.325, 0.0005, "4.7")
    if "random_split_r2" in tr:
        check("random 5-fold split R2", round(tr["random_split_r2"], 3), 0.769, 0.0005, "4.7")
        check("random 5-fold split MAE", round(tr["random_split_mae"], 1), 84.9, 0.05, "4.7")
    else:
        check("random 5-fold split (re-run train_from_videos.py)", None, 0.769)
except FileNotFoundError:
    check("training_report.json", None, "present")

# ------------------------------------------------------- Sessions D, E, F
section("3c. Sessions D, E, F (results_D/E/F against sessions_values.txt)")
vp = os.path.join(OUT, "sessions_values.txt")
vals = open(vp).read() if os.path.exists(vp) else ""
check("sessions_values.txt (run make_sessions.py)", "present" if vals else None, "present",
      section="4.14")
for t in "DEF":
    Dd = read_csv(f"results_{t}/summary.csv")
    if Dd is None:
        check(f"Session {t} results (run make_sessions.py)", None, f"results_{t}/summary.csv")
        continue
    Dd = Dd[Dd.kind == "video"].sort_values("j")
    if t == "D":
        jd = sorted(set(Dd.j.astype(int)))
        check("Session D recordings", len(Dd), 10, section="3.2")
        check("Session D current-density steps (mA/cm2)",
              sorted(set(b - a for a, b in zip(jd[:-1], jd[1:]))), [50], section="3.2")
        check("Session D recordings with a voltage", int(Dd.dS.notna().sum()), len(Dd),
              section="3.2")
    m = re.search(rf"Session {t}  folder \S+  recordings (\d+)", vals)
    check(f"Session {t}: recordings in the section = in results_{t}",
          int(m.group(1)) if m else None, len(Dd), section="4.1" + "456"["DEF".index(t)])
    Dv = Dd.dropna(subset=["dS"])
    want = (round(float(np.corrcoef(Dv.dS, Dv.H_norm_pct)[0, 1]), 3)
            if len(Dv) >= 3 else "--")
    m = re.search(rf"within {t}: r (\S+)", vals)
    got = None if not m else (m.group(1) if m.group(1) == "--" else float(m.group(1)))
    check(f"Session {t}: r against Gibbs, section = recomputed", got, want,
          0.0005 if want != "--" else 0.0, "4.17")
for f in ("sessions_DEF.tex", "session_D.png", "session_E.png", "session_F.png",
          "sessions_all.png", "engineering_E.png"):
    check(f"{f} present", os.path.exists(os.path.join(OUT, f)) or None, True, section="4.14")
m = re.search(r"pooled r Gibbs, A-C (\S+)", vals)
check("pooled r, Sessions A-C, as in Section 4.4", float(m.group(1)) if m else None, 0.283,
      0.0005, "4.17")

# ------------------------------------------------- stable bubble paths
section("3d. Stable bubble paths (fisher_paths.csv, Section 4.9.5 and Key results)")
fpc = os.path.join(OUT, "fisher_paths.csv")
if os.path.exists(fpc):
    FP = read_csv(fpc)
    for sess, j, want in (("B", 100, 23), ("B", 300, 16), ("B", 600, 10), ("B", 900, 5),
                          ("C", 688, 22)):
        row = FP[(FP.session == sess) & (FP.j == j)]
        check(f"path share, Session {sess}, {j} mA/cm2 (%)",
              round(100 * float(row.share.iloc[0])) if len(row) else None, want, 0.5, "4.9.5")
    for sess, j, want in (("B", 100, 1), ("B", 300, 1), ("B", 600, 0), ("B", 900, 0),
                          ("C", 688, 1)):
        row = FP[(FP.session == sess) & (FP.j == j)]
        check(f"stable path, Session {sess}, {j} mA/cm2",
              int(row.stable.iloc[0]) if len(row) else None, want, section="4.9.5")
else:
    check("fisher_paths.csv", None, "present (run fisher_paths.py)")

# ----------------------------------------------------------- figure log
section("4. Values printed while the figures were made (make_all_figures_log.txt)")
log = ""
lp = os.path.join(OUT, "make_all_figures_log.txt")
if os.path.exists(lp):
    log = open(lp).read()


def grab(pattern, cast=str):
    m = re.search(pattern, log)
    return cast(m.group(1)) if m else None


check("filter: raw / filtered / sites",
      grab(r"raw=(\d+\s+filtered=\d+\s+sites=\d+)"),
      lambda g: re.sub(r"\s+", " ", g) == "5541 filtered=349 sites=10", section="3.8")
check("nucleation sites, 688 mA/cm2 (>= 25 % occupancy)",
      grab(r"reference recording \S+: (\d+) sites occupied", int), 12, section="3.8")
check("bore pitch (px)", grab(r"bore pitch ([\d.]+) px", float), 155.5, 0.05, "4.12.3")
check("Session A log fit (figure script)", grab(r"logarithmic fit\s*:\s*r = ([\d.]+)", float), 0.911, 0.0005, "4.2")
for j, ef in ((100, 1.32), (300, 2.08), (600, 1.00), (900, 1.52), (688, 5.01)):
    v = grab(rf"\n\s+{j}\s+\d+\s+[\d.e+-]+\s+([\d.]+)\s+\d+", float)
    check(f"pixel Fisher edge/flat, {j} mA/cm2", v, ef, 0.005, "4.9")
for pair, mean in (("100->300", "8.8120e-07"), ("300->600", "2.8306e-07"), ("600->900", "2.5189e-07")):
    v = grab(rf"{pair}\s+\d+\s+([\d.e+-]+)")
    check(f"F_j mean {pair}", v, mean, section="4.9")
check("polarity figure, before / after", grab(r"before/after: \((\d+, \d+)\)"), "8, 13", section="3.12")
check("Figure 23, 688 mA/cm2 frame 60, new detector", grab(r"detector_ans_still.png\s+bubbles: (\d+)", int), 13, section="4.10")
check("Figure 23, old / new circles per row", grab(r"old/new per row: (\[.*?\])"), "[(14, 13), (12, 6)]", section="4.10")
check("zoomed capture circles", grab(r"detector_zoomed.png\s+bubbles, size: \((\d+)", int), 3, section="4.10")
check("three recording frames", grab(r"per frame: (\[[\d, ]+\])"), "[12, 11, 13]", section="4.10")
act = grab(r"plate sd range: (\(\(.*?\)\))")
check("activity map (bubble sd, plate sd) MP4", act,
      lambda a: [round(float(x), 1) for x in re.findall(r"[\d.]+", a)] == [7.1, 31.1, 0.7, 2.6],
      section="3.13")

# ------------------------------------------------------- live re-analysis
section("5. Re-analysis with the final code")
import cv2
img = next(iter(sum((glob.glob(os.path.join(d, "bubble_density_600mA_02.jpg"))
                    for d in (".", "..", "screenshots", "../screenshots", "images",
                              os.path.expanduser("~/Downloads"))), [])), None)
if img:
    from bubble_analysis import BubbleAnalyzer
    r = BubbleAnalyzer().full_analysis(cv2.imread(img), 600.0, "gde", 2.54, "medium", detector="standard")
    check("600 mA/cm2 image: bubbles (medium)", r["bubble_count"], 85, section="4.1")
    # the other sensitivities and the other two published images (Section 4.1)
    A_ = BubbleAnalyzer()
    for sens, want in (("low", 33), ("high", 172)):
        check(f"600 mA/cm2 image: bubbles ({sens})",
              A_.full_analysis(cv2.imread(img), 600.0, "gde", 2.54, sens,
                               detector="standard")["bubble_count"], want, section="4.1")
    for name, want in (("bubble_density_100mA_01.jpg", (14, 48, 109)),
                       ("bubble_density_200mA_01.jpg", (17, 45, 100))):
        pth = os.path.join(os.path.dirname(img), name)
        if os.path.exists(pth):
            got = tuple(A_.full_analysis(cv2.imread(pth), 100.0, "gde", 2.54, sv,
                                         detector="standard")["bubble_count"]
                        for sv in ("low", "medium", "high"))
            check(f"{name}: bubbles low/medium/high", got, want, section="4.1")
    # a screen capture: no circle on the overlay text, border or player bar
    cap = next(iter(sum((glob.glob(os.path.join(d, "*212403*721mA*.png"))
                         for d in (".", "..", "screenshots", "../screenshots",
                                   os.path.expanduser("~/Downloads"))), [])), None)
    if cap:
        rc = A_.full_analysis(cv2.imread(cap), 721.0, "ans", 2.54, "low", detector="standard")
        check("screen capture 721 mA/cm2: every bubble (low)", rc["bubble_count"], 20, section="3.5")
        rp = A_.full_analysis(cv2.imread(cap), 721.0, "ans", 2.54, "medium", detector="primary")
        check("screen capture 721 mA/cm2: main bubbles", rp["bubble_count"], 4, section="3.12")
    check("600 mA/cm2 image: H", r["shannon_entropy"], 4.951, 0.00005, "4.1")
    check("600 mA/cm2 image: H_norm (%)", r["shannon_entropy_norm"], 89.29, 0.005, "4.1")
    check("600 mA/cm2 image: H2 (mL/min)", r["H2_production_mL_per_min"], 10.614, 0.0005, "4.1")
    import fisher_image
    f = fisher_image.analyse_image(cv2.imread(img))
    check("Tab 4: Fisher J", round(f["fisher_histogram_J"], 6), 0.001668, 0.0000005, "3.14")
    check("Tab 4: x / y ratio", f["fisher_xy_ratio"], 1.025, 0.0005, "3.14")
    check("Tab 4: edge / flat", f["edge_flat_ratio"], 4.22, 0.005, "3.14")
    check("Tab 4: peaks on an edge", sum(p["on_edge"] for p in f["peaks"]), 20, section="3.14")
else:
    check("bubble_density_600mA_02.jpg", None, "found")

try:
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import measure_detection as md   # runs and prints its own tables
    sys.argv = _argv
    want = {"860 mA/cm2 screen capture": 3, "721 mA/cm2 screen capture": 4,
            "688 mA/cm2 nucleation sites, frame 60": 13,
            "100 mA/cm2 MP4, frame 0": 6, "100 mA/cm2 MP4, frame 45": 4,
            "100 mA/cm2 MP4, frame 85": 5, "100 mA/cm2 AVI, frame 0": 6,
            "100 mA/cm2 AVI, frame 45": 4, "100 mA/cm2 AVI, frame 85": 6}
    print()
    for label, size, present, circles in md.rows:
        if label in want:
            check("circles: " + label, circles, want[label], section="4.10")
    check("ring mode circles, frames 0/45/85", md.ring, [2, 3, 5], section="3.12")
    rwant = {"bubble_video_2283mV_688mA_41deg.avi": [12, 12, 12, 12, 11, 11, 13, 12, 12, 12, 13, 12],
             "20260713_bigbubbles_100mA_1760mV_39deg.mp4": [6, 6, 6, 6, 4, 4, 4, 6, 5],
             "20260713_bigbubbles_100mA_1760mV_39deg.avi": [6, 6, 6, 5, 5, 4, 5, 6, 4]}
    for name, c in md.recs:
        if name in rwant:
            check("recording: " + name[-26:], c, rwant[name], section="4.10")
except SystemExit:
    pass
except Exception as e:
    check("measure_detection.py", f"ERROR {e}", "ok")

if FULL:
    section("6. Full checks")
    import run_all as R
    pairs = (("100mA_1760mV", 80.06, 79.39), ("300mA_1971mV", 80.47, 79.83),
             ("600mA_2133mV", 81.39, 81.08), ("900mA_2264mV", 81.86, 81.22))
    pool = glob.glob("videos/*") + glob.glob("session_B/*")
    for key, mp4, avi in pairs:
        for ext, want in ((".mp4", mp4), (".avi", avi)):
            hit = [p for p in pool if key in p and p.lower().endswith(ext)]
            if hit:
                j, V, T = R.parse_conditions(hit[0])
                m, *_ = R.process_video(hit[0], j, "/tmp")
                check(f"H_norm {key} {ext}", m["H_norm_pct"], want, 0.005, "4.6")
    import fisher_pixel as fp
    ans = "session_C/bubble_video_2283mV_688mA_41deg.avi"
    if os.path.exists(ans):
        fr = fp.read_grey(ans, max_frames=400)
        mk = fp.overlay_mask(fr)
        got = []
        for k in (1, 2, 5, 10, 20, 40, 80):
            dI = fr[k:] - fr[:-k]
            F = ((dI ** 2) / (0.5 * (fr[k:] + fr[:-k]) + fp.EPS)).mean(axis=0)
            F[mk] = 0.0
            got.append(round(fp.ring_score(F, fr.mean(axis=0), mk), 2))
        check("time-step test 10..800 ms", got, [5.01, 5.17, 4.89, 4.35, 3.69, 3.21, 3.02], section="4.9")

# ------------------------------------------------------------------ summary
n_pass = sum(m == "PASS" for m, _ in results)
n_fail = sum(m == "FAIL" for m, _ in results)
n_skip = sum(m == "skip" for m, _ in results)
print("\n" + "=" * 78)
print(f"  {n_pass} passed   {n_fail} failed   {n_skip} skipped")
if n_fail == 0 and n_skip == 0:
    print("  Every number checked agrees with the report.")
print("=" * 78)
