"""
make_all_figures.py
===================
Produces every computed figure of the report from your own recordings, on
your own machine, and copies each one into ONE folder under the exact file
name the LaTeX project expects. Nothing is downloaded and nothing is bundled:
every picture is made from the files you pass in.

Run from the backend folder:

    python3 make_all_figures.py

With the default folder layout (all relative to backend/):

    session_A/            the 12 Session A recordings        (Figure 18)
    session_B/            the 16 Session B recordings
    session_C/            the 5 Session C recordings          (Figure 11)
    results_A/ results_B/ results_C/ results_screenshots/
                          batch results from run_all.py       (Figures 16, 17)
    screenshots/          YOUR screenshots of the app, see SCREENSHOTS below
    Session_D/ Session_E/ Session_F/
                          the Session D, E, F recordings     (Sections 4.14-4.17,
                                                              Figures 16, 28-31)

Output:

    figures_for_overleaf/    upload the whole folder into figures/ on Overleaf

At the end it prints a table: every figure of the report, whether it was
made, and for the ones still missing exactly what to do.

Options (all optional):

    --session-a DIR  --session-b DIR  --session-c DIR
    --ans FILE       the 688 mA/cm2 nucleation-site recording
    --big FILE       the 100 mA/cm2 large-bubble recording
    --zoom FILE      your screen capture of a zoomed player (Figure 26);
                     default screenshots/Screenshot_2026-08-25_211925_..._860mA_43deg.png
    --panel FILE     the recording used for Figure 18 (a Session A file)
    --rerun-batch    run run_all.py again even if results_X/summary.csv exists
    --out DIR        default figures_for_overleaf
"""

import argparse
import glob
import re
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ---------------------------------------------------------------------------
# The four Session B recordings used for the pixel-level Fisher figures
# (Section 4.9). The 300 mA/cm2 point was recorded twice; the 1971 mV
# recording is the one analysed in the report.
PIXEL_FISHER_B = ("100mA_1760mV", "300mA_1971mV", "600mA_2133mV", "900mA_2264mV")

# Screenshots that only you can take. Save them in screenshots/ under these
# names; the settings for each are in the report, Appendix A.
SCREENSHOTS = {
    "screenshot_image_analyzer.png":
        "Tab 1, bubble_density_600mA_02.jpg, 600 mA/cm2, 'Every bubble', Medium, GDE",
    "screenshot_detection_overlay.png":
        "Tab 1, same analysis, the detection-overlay image only",
    "screenshot_video_analyzer.png":
        "Tab 3, 20260713_bigbubbles_100mA_1760mV_39deg.mp4, interval strip visible",
    "screenshot_fisher_map.png":
        "Tab 4, bubble_density_600mA_02.jpg, view 'F = Fx + Fy', numbers visible",
    "screenshot_fisher_peaks.png":
        "Tab 4, same picture, view 'Peaks'",
}

ZOOM_DEFAULT = "Screenshot_2026-08-25_211925_bubbles_flattend_860mA_43deg.png"

# Every figure file the LaTeX project includes, in report order, with the
# step that produces it.
FIGURES = [
    ("entropy_bridge.png",            "1",  "diagrams"),
    ("system_flow.png",               "2",  "diagrams"),
    ("screenshot_image_analyzer.png", "3",  "screenshot"),
    ("detection_stages.png",          "4",  "diagrams"),
    ("screenshot_detection_overlay.png", "5", "screenshot"),
    ("screenshot_video_analyzer.png", "6",  "screenshot"),
    ("computed_background.png",       "7",  "filter"),
    ("background_subtracted.png",     "8",  "filter"),
    ("filter_before_after.png",       "9",  "filter"),
    ("nucleation_sites.png",          "10", "sites"),
    ("fisher_workflow_maps.png",      "11", "fisher_table (needs Session C)"),
    ("detector_polarity.png",         "12", "detection"),
    ("video_activity_map.png",        "13", "detection"),
    ("screenshot_fisher_map.png",     "14", "screenshot"),
    ("screenshot_fisher_peaks.png",   "15", "screenshot"),
    ("entropy_overview.png",          "16", "make_sessions.py (entropy overview)"),
    ("final_results.png",             "17", "results (needs results_A, results_B)"),
    ("session_analysis.png",          "18", "results (needs results_A)"),
    ("per_file_panel.png",            "19", "panel (needs a Session A recording)"),
    ("fisher_pixel_temporal.png",     "20", "fisher_pixel"),
    ("fisher_pixel_ring_300mA.png",   "21", "fisher_pixel"),
    ("fisher_pixel_ring_688mA.png",   "22", "fisher_pixel"),
    ("fisher_pixel_across_j.png",     "23", "fisher_pixel"),
    ("fisher_paths.png",              "24", "fisher_paths.py (stable bubble paths)"),
    ("detector_ans_still.png",        "25", "detection"),
    ("detector_zoomed.png",           "26", "detection (needs the zoomed capture in screenshots/)"),
    ("video_frames.png",              "27", "detection"),
    ("session_D.png",                 "28", "make_sessions.py (needs Session_D/)"),
    ("session_E.png",                 "29", "make_sessions.py (needs Session_E/)"),
    ("session_F.png",                 "30", "make_sessions.py (needs Session_F/)"),
    ("sessions_all.png",              "31", "make_sessions.py"),
    ("engineering_E.png",             "32", "make_sessions.py (electrode structures, Session E)"),
    ("sessions_DEF.tex",              "4.0", "make_sessions.py (overview, Sections 4.14-4.18)"),
    ("fisher_paths.tex",              "4.9", "fisher_paths.py (numbers of Section 4.9.5)"),
]

log_lines = []


def say(s=""):
    print(s)
    log_lines.append(s)


def run(cmd, what):
    """Run one generator. A failure is reported and the rest carry on."""
    say(f"\n--- {what}\n    $ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    lines = (p.stdout + p.stderr).strip().splitlines()
    for line in lines:                      # full output into the log file
        log_lines.append("    " + line)
    for line in lines[-12:]:                # the last lines on the screen
        print("    " + line)
    if p.returncode != 0:
        say(f"    !! failed with exit code {p.returncode}")
    return p.returncode == 0


def copy(src, name, out):
    src = os.path.join(HERE, src)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(out, name))
        return True
    say(f"    (not produced: {os.path.relpath(src, HERE)})")
    return False


def _key(name):
    """File-name key that ignores spaces versus underscores and case, so
    'Screenshot 2026-08-25 211925_...png' (as macOS saves it) matches
    'Screenshot_2026-08-25_211925_...png'."""
    return re.sub(r"[\s_]+", "_", os.path.basename(name)).lower()


def find_file(name, folders=("screenshots", ".", "..")):
    """Path of `name` in one of `folders` (relative to backend/), accepting the
    spaces-for-underscores variant. None if it is nowhere."""
    want = _key(name)
    for d in folders:
        d = os.path.join(HERE, d)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if _key(f) == want:
                return os.path.relpath(os.path.join(d, f), HERE)
    return None


def videos_in(folder):
    exts = (".mp4", ".avi", ".mov")
    return sorted(p for p in glob.glob(os.path.join(HERE, folder, "*"))
                  if p.lower().endswith(exts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-a", default="session_A")
    ap.add_argument("--session-b", default="session_B")
    ap.add_argument("--session-c", default="session_C")
    ap.add_argument("--ans", default="session_C/bubble_video_2283mV_688mA_41deg.avi")
    ap.add_argument("--big", default="videos/20260713_bigbubbles_100mA_1760mV_39deg.mp4")
    ap.add_argument("--zoom", default=None)
    ap.add_argument("--panel", default=None,
                    help="recording for Figure 18 (default: the first 250 mA "
                         "file in --session-a)")
    ap.add_argument("--rerun-batch", action="store_true")
    ap.add_argument("--out", default="figures_for_overleaf")
    a = ap.parse_args()
    if not a.zoom:
        # Figure 26: your own screen capture of the 860 mA/cm2 recording,
        # played back zoomed in a video player
        a.zoom = find_file(ZOOM_DEFAULT)

    out = os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)
    work = os.path.join(HERE, "report_figures")
    os.makedirs(work, exist_ok=True)
    made = {}

    say("MAKE ALL FIGURES")
    say("=" * 74)
    say(f"  backend folder : {HERE}")
    say(f"  output folder  : {out}")

    # 1. diagrams ------------------------------------------------------------
    if run([PY, "make_flow_diagram.py"], "Figures 1, 2, 4: diagrams"):
        for f in ("entropy_bridge.png", "system_flow.png", "detection_stages.png"):
            made[f] = copy(os.path.join("report_figures", f), f, out)

    # 2. background and filter ----------------------------------------------
    if os.path.exists(os.path.join(HERE, a.ans)):
        if run([PY, "make_filter_figures.py", a.ans], "Figures 7, 8, 9: background and filter"):
            for f in ("computed_background.png", "background_subtracted.png",
                      "filter_before_after.png"):
                made[f] = copy(os.path.join("report_figures", f), f, out)
        # 3. nucleation sites across current densities ------------------------
        # every recording of Session C and of Sessions D, E, F is offered;
        # track_sites.py keeps those with the bore grid of the ANS plate
        src = [a.session_c if os.path.isdir(os.path.join(HERE, a.session_c)) else a.ans]
        src += [d for d in ("Session_D", "Session_E", "Session_F")
                if os.path.isdir(os.path.join(HERE, d))]
        if run([PY, "track_sites.py"] + src, "Figure 10: nucleation sites across current density"):
            made["nucleation_sites.png"] = copy("report_figures/nucleation_sites.png",
                                                "nucleation_sites.png", out)
            copy("report_figures/nucleation_sites_values.txt",
                 "nucleation_sites_values.txt", out)
    else:
        say(f"\n--- Figures 7-10 skipped: {a.ans} not found (use --ans)")

    # 4. batch results --------------------------------------------------------
    for tag, folder in (("A", a.session_a), ("B", a.session_b), ("C", a.session_c)):
        res = os.path.join(HERE, f"results_{tag}")
        if os.path.exists(os.path.join(res, "summary.csv")) and not a.rerun_batch:
            say(f"\n--- results_{tag}/summary.csv exists, batch run skipped "
                f"(--rerun-batch to redo it)")
            continue
        if not videos_in(folder):
            say(f"\n--- results_{tag}: no recordings in {folder}/, skipped")
            continue
        tmp = os.path.join(HERE, "results")
        if os.path.exists(tmp):
            say("\n!! a folder named results/ already exists; move it away first")
            break
        if run([PY, "run_all.py", folder], f"batch run, Session {tag}"):
            if os.path.exists(res):
                shutil.rmtree(res)
            shutil.move(tmp, res)

    # 4b. single images: bubble counts with the current detector -------------
    # results_screenshots/ was made once by run_all.py. Its bubble counts come
    # from the "Every bubble" detector, which was replaced in November 2026
    # (report Section 3.15.7), so they are recomputed here for exactly the
    # files listed in it. Entropy does not depend on the detector.
    ss = os.path.join(HERE, "results_screenshots", "summary.csv")
    if os.path.exists(ss):
        import pandas as pd
        import cv2
        from bubble_analysis import BubbleAnalyzer
        S = pd.read_csv(ss)
        az = BubbleAnalyzer()
        say("\n--- single images: bubble counts with the current detector")
        for i, rw in S.iterrows():
            name = os.path.basename(str(rw["file"]))
            hits = [os.path.join(dp, name) for dp, _, fs in os.walk(HERE) if name in fs
                    and "results" not in dp]
            if not hits:
                say(f"    {name}: not found under backend/, count left as it was")
                continue
            r = az.full_analysis(cv2.imread(hits[0]), float(rw.get("j") or 0), "ans", 2.54, "medium")
            for col in ("mean_bubbles", "max_bubbles", "min_bubbles", "total_records"):
                if col in S.columns:
                    S.at[i, col] = r["bubble_count"]
            if "mean_diam_mm" in S.columns:
                S.at[i, "mean_diam_mm"] = r["mean_diameter_mm"]
            say(f"    {name}: {r['bubble_count']} bubbles, H_norm {r['shannon_entropy_norm']:.2f} %")
        S.to_csv(ss, index=False)

    # 5. main results and session comparison ---------------------------------
    if os.path.exists(os.path.join(HERE, "results_A", "summary.csv")):
        if run([PY, "make_report_figure.py"], "Figure 17: main results"):
            made["final_results.png"] = copy("report_figures/final_results.png",
                                             "final_results.png", out)
        if run([PY, "make_session_figure.py"], "Figure 18: session comparison"):
            made["session_analysis.png"] = copy("report_figures/session_analysis.png",
                                                "session_analysis.png", out)
        for f in ("session_stats.txt",):
            copy(os.path.join("report_figures", f), f, out)
    else:
        say("\n--- Figures 17, 18 skipped: results_A/summary.csv missing")

    # 6. Figure 18, one per-file panel, made fresh ----------------------------
    panel = a.panel
    if panel is None:
        cands = [p for p in videos_in(a.session_a) if "250mA" in os.path.basename(p)]
        panel = cands[0] if cands else None
    if panel and os.path.exists(os.path.join(HERE, panel) if not os.path.isabs(panel) else panel):
        say(f"\n--- Figure 18: per-file panel from {os.path.basename(panel)}")
        try:
            sys.path.insert(0, HERE)
            import run_all as R
            j, V, T = R.parse_conditions(panel)
            m, frame, bubbles, rows, bg, shape = R.process_video(panel, j, work)
            m.update({"j": j, "V": V, "T": T, "dS": R.gibbs_entropy(V, T)})
            name = os.path.splitext(os.path.basename(panel))[0]
            R.make_panel(name, frame, bubbles, rows, shape, m,
                         os.path.join(out, "per_file_panel.png"))
            made["per_file_panel.png"] = True
            say(f"    H_norm = {m['H_norm_pct']} %, mean bubbles = {m['mean_bubbles']}")
        except Exception as e:
            say(f"    !! {type(e).__name__}: {e}")
    else:
        say("\n--- Figure 19 skipped: no Session A recording found (use --panel)")

    # 7. Fisher from the bubble table, Session C -------------------------------
    if len(videos_in(a.session_c)) >= 2:
        if run([PY, "fisher_analysis.py", a.session_c, "--filter-tiny"],
               "Figure 11: Fisher workflow maps, Session C"):
            made["fisher_workflow_maps.png"] = copy("fisher_output/workflow_maps.png",
                                                    "fisher_workflow_maps.png", out)
    else:
        say(f"\n--- Figure 11 skipped: needs at least two recordings in {a.session_c}/")

    # 8. Fisher from the brightness field ---------------------------------------
    pool = videos_in(a.session_b) + videos_in("videos")
    four = []
    for key in PIXEL_FISHER_B:
        hit = [p for p in pool if key in os.path.basename(p)]
        if hit:
            # prefer the .mp4 when both containers exist
            hit.sort(key=lambda p: not p.lower().endswith(".mp4"))
            four.append(hit[0])
    if len(four) == len(PIXEL_FISHER_B):
        stage = os.path.join(HERE, "_pixel_fisher_B")
        os.makedirs(stage, exist_ok=True)
        for f in os.listdir(stage):
            os.remove(os.path.join(stage, f))
        for p in four:
            dst = os.path.join(stage, os.path.basename(p))
            try:
                os.symlink(os.path.abspath(p), dst)
            except OSError:
                shutil.copyfile(p, dst)
        if run([PY, "fisher_pixel.py", "_pixel_fisher_B"],
               "Figures 20, 21, 23: pixel Fisher, four Session B recordings"):
            o = "fisher_pixel_output__pixel_fisher_B"
            made["fisher_pixel_temporal.png"] = copy(f"{o}/fisher_pixel_temporal.png",
                                                     "fisher_pixel_temporal.png", out)
            made["fisher_pixel_ring_300mA.png"] = copy(f"{o}/fisher_pixel_ring_detail_300mA.png",
                                                       "fisher_pixel_ring_300mA.png", out)
            made["fisher_pixel_across_j.png"] = copy(f"{o}/fisher_pixel_across_j.png",
                                                     "fisher_pixel_across_j.png", out)
    else:
        say("\n--- Figures 20, 21, 23 skipped: need the four recordings "
            + ", ".join(PIXEL_FISHER_B) + f" in {a.session_b}/ or videos/")

    if os.path.exists(os.path.join(HERE, a.ans)):
        stage = os.path.join(HERE, "_pixel_fisher_C")
        os.makedirs(stage, exist_ok=True)
        for f in os.listdir(stage):
            os.remove(os.path.join(stage, f))
        dst = os.path.join(stage, os.path.basename(a.ans))
        try:
            os.symlink(os.path.abspath(os.path.join(HERE, a.ans)), dst)
        except OSError:
            shutil.copyfile(os.path.join(HERE, a.ans), dst)
        if run([PY, "fisher_pixel.py", "_pixel_fisher_C"], "Figure 22: pixel Fisher, 688 mA/cm2"):
            made["fisher_pixel_ring_688mA.png"] = copy(
                "fisher_pixel_output__pixel_fisher_C/fisher_pixel_ring_detail_688mA.png",
                "fisher_pixel_ring_688mA.png", out)

    # 8b. stable bubble paths (Figure 24) ----------------------------------------
    stages = [d for d in ("_pixel_fisher_B", "_pixel_fisher_C")
              if os.path.isdir(os.path.join(HERE, d)) and os.listdir(os.path.join(HERE, d))]
    if stages:
        if run([PY, "fisher_paths.py"] + stages + ["--out", out],
               "Figure 24: stable bubble paths in the pixel-level Fisher information"):
            made["fisher_paths.png"] = os.path.exists(os.path.join(out, "fisher_paths.png"))
    else:
        say("\n--- Figure 24 skipped: needs the Session B or C recordings of Figures 20 to 23")

    # 9. detection figures ---------------------------------------------------
    cmd = [PY, "make_detection_figures.py", "--ans", a.ans, "--big", a.big,
           "--out", "report_figures"]
    if a.zoom:
        cmd += ["--zoom", a.zoom]
    if run(cmd, "Figures 12, 13, 25, 26, 27: detection"):
        for f in ("detector_polarity.png", "detector_ans_still.png",
                  "detector_zoomed.png", "video_activity_map.png", "video_frames.png"):
            if f == "detector_zoomed.png" and not a.zoom:
                continue
            made[f] = copy(os.path.join("report_figures", f), f, out)

    # 10. your screenshots ----------------------------------------------------
    shots = os.path.join(HERE, "screenshots")
    for f in SCREENSHOTS:
        src = find_file(f, ("screenshots",))
        if src:
            shutil.copyfile(os.path.join(HERE, src), os.path.join(out, f))
            made[f] = True

    # 11. Sessions D, E, F: overview, Sections 4.14-4.17, Figures 16, 28-31 ---
    have = [t for t in "DEF" if any(
        os.path.isdir(os.path.join(HERE, d)) and videos_in(d)
        for d in (f"Session_{t}", f"session_{t}", f"Session {t}", f"../Session_{t}",
                  f"../Session {t}"))]
    if have:
        run([PY, "make_sessions.py"], "Overview, Sections 4.14-4.17, Figures 16 and 28-31: Sessions "
            + ", ".join(have))
    else:
        say("\n--- Sections 4.14-4.18 skipped: no folders Session_D/, Session_E/, Session_F/")

    # 12. the checklist ---------------------------------------------------------
    say("\n" + "=" * 74)
    say("  FIGURE CHECKLIST")
    say("=" * 74)
    missing = 0
    for f, fig, how in FIGURES:
        ok = os.path.exists(os.path.join(out, f))
        missing += not ok
        say(f"  {'made' if ok else '----'}   Fig {fig:<4} {f:<34} {'' if ok else how}")
    say("")
    for f, what in SCREENSHOTS.items():
        if not os.path.exists(os.path.join(out, f)):
            say(f"  screenshot to take -> screenshots/{f}\n      {what}")
    if not a.zoom:
        say("  Figure 26 needs your zoomed capture: put " + ZOOM_DEFAULT)
        say("      into screenshots/ (or pass --zoom FILE) and run again")
    say("")
    say(f"  {len(FIGURES) - missing} of {len(FIGURES)} files (32 figures, "
        f"sessions_DEF.tex and fisher_paths.tex) are in {a.out}/")
    say("  Logo: download the official Frankfurt UAS logo and save it in the")
    say("  Overleaf project as figures/logo.png (it is not produced here).")
    with open(os.path.join(out, "make_all_figures_log.txt"), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
