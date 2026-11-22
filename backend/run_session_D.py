"""
run_session_D.py
================
Everything Session D needs, in one command.

    python3 run_session_D.py Session_D

WHY SESSION D MATTERS MORE THAN THE OTHER SESSIONS
--------------------------------------------------
Its ten recordings run 50, 100, 150, 200, 250, 300, 350, 400, 450 and
500 mA/cm2 — a constant 50 mA/cm2 step. Every earlier session was unevenly
spaced, and because Fisher information scales as 1/dj^2, unevenly spaced
operating points make the Fisher maps largely incomparable. That was the
single largest open problem in the interim report, recorded there as item 2 of
the open questions. Session D removes it: with a constant step, every pair in
the sequence is on the same footing and the maps can be read against each
other directly.

The filenames also carry the cell voltage, so the Gibbs entropy can be
computed for all ten, and the temperature span is narrow (35.4 to 37.1 deg C)
which weakens the temperature-current confounding described in the
limitations.

WHAT THIS RUNS
--------------
    1. run_all.py            per-file metrics and the summary table
    2. detect_primary.py     the large-bubble detector, one overlay per file
    3. track_sites.py        nucleation sites and their occupancy
    4. fisher_pixel.py       pixel-level Fisher information across j
    5. fisher_analysis.py    the bubble-table Fisher workflow

Anything already present is not recomputed unless --force is given.
"""

import os
import re
import sys
import glob
import subprocess

VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv")


def run(cmd, label):
    print("\n" + "=" * 70)
    print("  " + label)
    print("=" * 70)
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        print(f"  !! {label} exited with code {r.returncode} — continuing")
    return r.returncode == 0


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "Session_D"
    force = "--force" in sys.argv

    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        print("Put the ten Session D recordings in a folder of that name next")
        print("to this script, or pass the path:")
        print("    python3 run_session_D.py ~/Downloads/Session_D")
        return

    vids = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                  if p.lower().endswith(VIDEO_EXT))
    if not vids:
        print(f"No videos in {folder}")
        return

    js = []
    for v in vids:
        m = re.search(r"(\d{2,4})\s*mA", os.path.basename(v), re.I)
        if m:
            js.append(int(m.group(1)))

    print("SESSION D")
    print("=" * 70)
    print(f"  folder          {folder}")
    print(f"  recordings      {len(vids)}")
    if js:
        js = sorted(js)
        steps = [b - a for a, b in zip(js[:-1], js[1:])]
        print(f"  current density {min(js)} to {max(js)} mA/cm2")
        print(f"  steps           {sorted(set(steps))} mA/cm2")
        if len(set(steps)) == 1:
            print(f"  -> evenly spaced at {steps[0]} mA/cm2. The Fisher maps "
                  f"across j are\n     directly comparable, which was not true "
                  f"of any earlier session.")
        else:
            print("  -> NOT evenly spaced. The 1/dj^2 scaling described in the "
                  "report\n     applies; read the dj-normalised column, not the "
                  "raw one.")

    results = folder.rstrip("/") + "_results"

    if force or not os.path.isdir(results):
        if run(["run_all.py", folder], "Per-file metrics and summary table"):
            if os.path.isdir("results"):
                os.replace("results", results)
    else:
        print(f"\n  {results}/ already exists — skipping run_all "
              f"(use --force to redo)")

    run(["detect_primary.py", folder], "Large-bubble detection, one overlay per file")

    mid = vids[len(vids)//2]
    run(["track_sites.py", mid, "--every", "10", "--max", "150"],
        f"Nucleation sites and occupancy ({os.path.basename(mid)})")

    run(["fisher_pixel.py", folder], "Pixel-level Fisher information across j")

    run(["fisher_analysis.py", folder, "--filter-tiny"],
        "Bubble-table Fisher workflow")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"  {results}/                     per-file metrics, summary.csv")
    print(f"  primary_output/               one detection overlay per recording")
    print(f"  report_figures/               nucleation_sites.png")
    print(f"  fisher_pixel_output_*/        F_t and F_j maps")
    print(f"  fisher_output/                bubble-table Fisher maps")
    print()
    print("  Check first: open primary_output/ and look at a few overlays.")
    print("  If the circles are not on bubbles, nothing downstream is worth")
    print("  reading, and the size band is the first thing to adjust:")
    print("      python3 detect_primary.py <one video> --mm 0.15 0.8")


if __name__ == "__main__":
    main()
