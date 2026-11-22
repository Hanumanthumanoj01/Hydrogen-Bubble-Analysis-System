"""
verify_install.py
=================
One command that checks the whole installation and prints a report you can
paste back.

    cd ~/Downloads/hydrogen-bubble-analysis
    python3 verify_install.py

Run it from the PROJECT folder — the one that contains backend/ and frontend/.

It checks, in order:

  1. every file from the bundle is present and is the right version
  2. your own files that the bundle relies on are still there
  3. every Python package the backend needs is installed
  4. the published validation numbers still come out exactly
  5. the new histogram / Fisher analysis gives the expected numbers
  6. a recording can be decoded and analysed      (if one is found)
  7. the running server answers                   (if it is running)

Nothing is changed on your machine. It only reads and computes.
"""

import hashlib
import importlib
import json
import os
import sys
import time
import urllib.request

# Filled in when the bundle is built. Path relative to the project folder.
MANIFEST = {
    "backend/bubble_analysis.py": "f92246065ea8c92b536c0cd691d88996",
    "backend/check_results.py": "65e0050d8c745d7ab865623abe8606f9",
    "backend/constants.py": "d47df5712319411c69eabd7b8ed846d7",
    "backend/detect_primary.py": "ddc1392214b0c0c65d7c9842562f4623",
    "backend/detect_video.py": "af869cfbbe28729449d18a937e3a9190",
    "backend/fisher_analysis.py": "fa8e6c8f4732909c84c30d084cd13a15",
    "backend/fisher_image.py": "f674a26070e89663dfa72faa93e89c89",
    "backend/fisher_paths.py": "4fc5c9347fe3c238ee78f87938166aa2",
    "backend/fisher_pixel.py": "b640a88bc84705c91be8d2d4d630ebae",
    "backend/fisher_report.py": "b7d306405ca84bc5b503f01213677e06",
    "backend/main.py": "d10141422b641cb67433f2535b818074",
    "backend/make_all_figures.py": "597ffa6e90619efdef7b224b9a7527e9",
    "backend/make_detection_figures.py": "80314f3b338adf5bfc49022c44694a72",
    "backend/make_filter_figures.py": "e6fb97031a8a61bb1684e90ff5fd3fd7",
    "backend/make_flow_diagram.py": "872100609ee770c37dca72907d7ef26a",
    "backend/make_report_figure.py": "bf3ff9456a9b2333947e2eb51c5a8e4b",
    "backend/make_session_figure.py": "de39be1747a45434ee50ea91a4066c70",
    "backend/make_sessions.py": "39907ca179203edcd403919edd3bbfa9",
    "backend/measure_detection.py": "3465bae7677ce1e9c34a299ff3d1f5c4",
    "backend/requirements.txt": "ca9252e7a3772ae1e25c8887d4a15da8",
    "backend/run_all.py": "4f8bda7371a96fc826f4ea75f92f5438",
    "backend/track_sites.py": "fe07151dd6d41763b1330d4d9696b6bd",
    "backend/train_from_videos.py": "2cf30d1bdb08bc6329ad5713d514fd52",
    "backend/video_analysis.py": "ed76690392fa188ac8bf97e23ade98ab",
    "frontend/src/App.jsx": "3998984a3422aab4b8741c67c341d22f",
    "frontend/src/pages/FisherAnalyzer.jsx": "747411569b52fc7616961b51f6b1f40f",
    "frontend/src/pages/ImageAnalyzer.jsx": "ddb9b20e7e4ce0ed2a46e1235cf5907a",
    "frontend/src/pages/VideoAnalyzer.jsx": "791c48f96366552146c7b8a99428637f",
}

# Files that are yours, not part of the bundle, that the bundle needs.
YOURS = [
    "backend/simulator.py",
    "frontend/package.json",
    "frontend/src/pages/CameraAnalyzer.jsx",
    "frontend/src/utils/export.js",
    "frontend/src/index.css",
]

PACKAGES = [
    ("cv2", "opencv-python"), ("numpy", "numpy"), ("scipy", "scipy"),
    ("skimage", "scikit-image"), ("pandas", "pandas"),
    ("matplotlib", "matplotlib"), ("openpyxl", "openpyxl"),
    ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
    ("multipart", "python-multipart"),
    ("sklearn", "scikit-learn"), ("joblib", "joblib"),
]

EXPECT_IMAGE = dict(bubbles=85, H=4.951, Hn=89.29, H2=10.614)
EXPECT_FISHER = dict(H=4.951, Hn=89.29, J=1.668e-3, xy=1.025, ef=4.22)

results = []


def row(ok, what, detail=""):
    mark = {True: "PASS", False: "FAIL", None: "skip"}[ok]
    results.append((mark, what, detail))
    print(f"  [{mark}]  {what:<52} {detail}")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def find(name, roots):
    for r in roots:
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if d not in ("node_modules", ".git", "venv", "__pycache__")]
            if name in fn:
                return os.path.join(dp, name)
    return None


def main():
    root = os.path.abspath(os.getcwd())
    if not os.path.isdir(os.path.join(root, "backend")) and \
            os.path.basename(root) == "backend":
        root = os.path.dirname(root)
    if not os.path.isdir(os.path.join(root, "backend")):
        print("\n  Run this from the project folder, the one containing backend/ "
              "and frontend/.\n")
        sys.exit(1)

    print("\nVERIFY INSTALLATION")
    print("=" * 78)
    print(f"  project folder: {root}\n")

    # 1 ---------------------------------------------------------------- files
    print("1. Bundle files")
    stale = 0
    for rel, want in sorted(MANIFEST.items()):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            row(False, rel, "MISSING")
            stale += 1
        elif md5(p) != want:
            row(False, rel, "OLD VERSION — copy it again from the bundle")
            stale += 1
        else:
            row(True, rel)

    # 2 ---------------------------------------------------------- your files
    print("\n2. Your own files the bundle relies on")
    for rel in YOURS:
        p = os.path.join(root, rel)
        row(os.path.exists(p), rel, "" if os.path.exists(p) else "MISSING")

    # 3 ------------------------------------------------------------ packages
    print("\n3. Python packages")
    missing = []
    for mod, pipname in PACKAGES:
        try:
            importlib.import_module(mod)
            row(True, pipname)
        except Exception:
            row(False, pipname, f"pip3 install {pipname}")
            missing.append(pipname)
    if missing:
        print(f"\n  Install the missing ones with:\n      pip3 install {' '.join(missing)}")

    # the computational checks need the backend on the path
    sys.path.insert(0, os.path.join(root, "backend"))
    os.chdir(os.path.join(root, "backend"))

    img = find("bubble_density_600mA_02.jpg", [root, os.path.expanduser("~/Downloads")])

    # 4 ---------------------------------------------------------- regression
    print("\n4. Published validation numbers (Section 4.1)")
    if not img:
        row(None, "bubble_density_600mA_02.jpg", "not found — put it in the project folder")
    else:
        try:
            import cv2
            from bubble_analysis import BubbleAnalyzer
            r = BubbleAnalyzer().full_analysis(cv2.imread(img), 600.0, "gde", 2.54,
                                               "medium", detector="standard")
            got = dict(bubbles=r["bubble_count"], H=r["shannon_entropy"],
                       Hn=r["shannon_entropy_norm"], H2=r["H2_production_mL_per_min"])
            ok = (got["bubbles"] == 85 and abs(got["H"] - 4.951) < 1e-3
                  and abs(got["Hn"] - 89.29) < 0.01 and abs(got["H2"] - 10.614) < 1e-3)
            row(ok, "600 mA/cm² image, every-bubble mode",
                f"{got['bubbles']} / {got['H']:.4f} / {got['Hn']:.2f} % / {got['H2']:.3f}"
                f"   expect 85 / 4.9510 / 89.29 % / 10.614")
        except Exception as e:
            row(False, "600 mA/cm² image, every-bubble mode", f"ERROR: {e}")

    # 5 -------------------------------------------------------------- fisher
    print("\n5. Histogram / Fisher analysis")
    if not img:
        row(None, "fisher_image on the same picture", "image not found")
    else:
        try:
            import cv2, fisher_image
            f = fisher_image.analyse_image(cv2.imread(img))
            ok_h = abs(f["shannon_entropy"] - 4.951) < 1e-3 and \
                abs(f["shannon_entropy_norm"] - 89.29) < 0.01
            row(ok_h, "entropy agrees with step 4 (independent code path)",
                f"H {f['shannon_entropy']:.4f}, {f['shannon_entropy_norm']:.2f} %")
            ok_f = (abs(f["fisher_histogram_J"] - 1.668e-3) < 2e-5
                    and abs((f["fisher_xy_ratio"] or 0) - 1.025) < 0.005
                    and abs((f["edge_flat_ratio"] or 0) - 4.22) < 0.02)
            row(ok_f, "Fisher J, x/y ratio, edge/flat",
                f"{f['fisher_histogram_J']:.3e} / {f['fisher_xy_ratio']} / {f['edge_flat_ratio']}"
                f"   expect 1.668e-03 / 1.025 / 4.22")
            on_edge = sum(p["on_edge"] for p in f["peaks"])
            row(on_edge == len(f["peaks"]), "every reported peak lies on an edge",
                f"{on_edge} of {len(f['peaks'])}")
        except Exception as e:
            row(False, "fisher_image", f"ERROR: {e}")

    # 6 --------------------------------------------------------- a recording
    print("\n6. Recording")
    vid = None
    for name in ("bubble_video_2283mV_688mA_41deg.avi",
                 "20260713_bigbubbles_100mA_1760mV_39deg.mp4",
                 "20260713_bigbubbles_100mA_1760mV_39deg.avi"):
        vid = find(name, [root, os.path.expanduser("~/Downloads")])
        if vid:
            break
    if not vid:
        row(None, "recording", "none of the known recordings found — skipped")
    else:
        try:
            import detect_video as dv
            t = time.time()
            frames, fps, total = dv.read_clip(vid, max_frames=40)
            row(True, f"decode {os.path.basename(vid)}",
                f"{len(frames)} frames read, {fps:.0f} fps, {time.time()-t:.1f} s")
            import main as app_main
            t = time.time()
            rows, overlays, fps, total, n = app_main._analyse_recording(
                vid, 688.0, "ans", 2.54, "primary", 3.0, 60.0, 10, 4)
            row(len(rows) == 4 and len(overlays) == 4, "analyse 4 intervals",
                f"bubbles per interval {[r['bubbles'] for r in rows]}, {time.time()-t:.1f} s")
        except Exception as e:
            row(False, "recording", f"ERROR: {e}")

    # 7 ------------------------------------------------------- running server
    print("\n7. Running server  (start it first with:  uvicorn main:app --port 8000)")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as r:
            row(r.status == 200, "backend answers on port 8000")
        with urllib.request.urlopen("http://127.0.0.1:8000/openapi.json", timeout=5) as r:
            paths = set(json.loads(r.read())["paths"])
        for ep in ("/analyze", "/analyze-video", "/analyze-video-frames",
                   "/analyze-fisher", "/analyze-fisher-xlsx"):
            row(ep in paths, f"endpoint {ep}", "" if ep in paths else
                "MISSING — main.py is an old version, or uvicorn needs restarting")
    except Exception:
        row(None, "backend on port 8000", "not running — skipped")

    # summary -----------------------------------------------------------------
    n_fail = sum(1 for m, _, _ in results if m == "FAIL")
    n_pass = sum(1 for m, _, _ in results if m == "PASS")
    n_skip = sum(1 for m, _, _ in results if m == "skip")
    print("\n" + "=" * 78)
    print(f"  {n_pass} passed   {n_fail} failed   {n_skip} skipped")
    if n_fail == 0:
        print("  Everything checked is in place.")
    else:
        print("  Fix the FAIL lines from the top down; later failures are often")
        print("  caused by an earlier one. Then run this again.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
