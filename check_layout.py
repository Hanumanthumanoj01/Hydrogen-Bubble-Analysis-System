"""
check_layout.py
===============
Checks that every recording, image and result the report needs is in the
folder where the scripts look for it, and can put it there for you.

Run it in the project folder (the one that holds backend/ and frontend/):

    python3 check_layout.py          only looks, changes nothing
    python3 check_layout.py --fix    also puts things in place

What --fix does, and nothing else:
  * a session folder with another spelling ("Session A", "session-a", ...)
    gets a link under the expected name (session_A); your folder stays as it is
  * a required file found somewhere else in the project, in ~/Downloads or on
    the Desktop is COPIED to where it is needed; the original stays where it is
  * an MP4 copy lying in session_B/ is MOVED to videos/, because session_B/
    must hold only the 16 AVI files that the report analyses
Every action is printed and written to layout_report.txt.

Sessions D, E and F are analysed in Sections 4.14 to 4.17 of the report;
their recordings are listed in full so that their contents are on record.

Nothing is deleted.
"""
import os
import re
import shutil
import sys

FIX = "--fix" in sys.argv
VID = (".avi", ".mp4", ".mov", ".mkv")
SKIP_DIRS = {"node_modules", ".git", "venv", ".venv", "__pycache__", "figures_for_overleaf",
             "report_figures", "detection_check", "primary_output", "dist"}

# ------------------------------------------------------------------ folders
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(HERE, "backend")):
    ROOT = HERE
elif os.path.exists(os.path.join(HERE, "main.py")):
    ROOT = os.path.dirname(HERE)
else:
    ROOT = os.getcwd()
BACK = os.path.join(ROOT, "backend")
if not os.path.isdir(BACK):
    sys.exit("No backend/ folder here. Run this in the project folder.")

# ----------------------------------------------- what the report is built on
SESSION_A = ["020mA_01.avi", "050mA_01.avi", "100mA_01.avi", "150mA_01.avi", "200mA_02.avi",
             "250mA_01.avi", "250mA_02.avi", "250mA_03.avi", "300mA_01.avi", "350mA_1.avi",
             "450mA_01.avi", "500mA_01.avi"]
SESSION_B = ["20260713_bigbubbles_%s.avi" % s for s in (
    "100mA_1760mV_39deg", "200mA_1863mV_40deg", "300mA_1971mV_39deg", "300mA_1983mV_40deg",
    "400mA_2024mV_39deg", "450mA_2054mV_40deg", "500mA_2077mV_40deg", "550mA_2108mV_40deg",
    "600mA_2133mV_41deg", "650mA_2157mV_40deg", "700mA_2186mV_40deg", "750mA_2205mV_40deg",
    "800mA_2232mV_40deg", "850mA_2252mV_40deg", "900mA_2264mV_42deg", "950mA_2285mV_41deg")]
SESSION_C = ["bubble_video_2283mV_688mA_41deg.avi", "bubble_video_2283mV_694mA_39deg.avi",
             "bubble_video_2283mV_715mA_39deg.avi", "bubble_video_2283mV_721mA_40deg.avi",
             "bubble_video_2335mV_838mA_41deg.avi"]
VIDEOS = ["20260713_bigbubbles_100mA_1760mV_39deg.mp4", "20260713_bigbubbles_300mA_1971mV_39deg.mp4",
          "20260713_bigbubbles_600mA_2133mV_41deg.mp4", "20260713_bigbubbles_900mA_2264mV_42deg.mp4",
          "20260713_bigbubbles_100mA_1760mV_39deg.avi"]
CAPTURES = ["Screenshot_2026-08-25_211925_bubbles_flattend_860mA_43deg.png",
            "Screenshot_2026-08-25_212403_bubbles_flattend_721mA_40deg.png"]
APP_SHOTS = {
    "screenshot_image_analyzer.png": "Step 4a, Tab 1, the whole window",
    "screenshot_detection_overlay.png": "Step 4a, Tab 1, only the overlay image",
    "screenshot_fisher_map.png": "Step 4b, Tab 4, view 'F = Fx + Fy'",
    "screenshot_fisher_peaks.png": "Step 4b, Tab 4, view 'Peaks'",
    "screenshot_video_analyzer.png": "Step 4c, Tab 3",
}
IMAGE = "bubble_density_600mA_02.jpg"

out_lines, actions, problems = [], [], []


def say(s=""):
    print(s)
    out_lines.append(s)


def key(name):
    """'Screenshot 2026-08-25 212403_x.png' == 'Screenshot_2026-08-25_212403_x.png';
    'Session A' == 'session_A' == 'session-a'."""
    return re.sub(r"[\s_\-]+", "_", os.path.basename(name.rstrip("/"))).lower()


def rel(p):
    try:
        r = os.path.relpath(p, ROOT)
        return p if r.startswith("../..") else r
    except ValueError:
        return p


# ---------------------------------------------------- index of every file
index = {}


def walk(top, depth):
    top = os.path.abspath(top)
    base = top.count(os.sep)
    for dp, dn, fn in os.walk(top, followlinks=False):
        dn[:] = [d for d in dn if d not in SKIP_DIRS and not d.startswith(".")
                 and not d.startswith("results") and "fisher" not in d.lower()
                 and dp.count(os.sep) - base < depth]
        for f in fn:
            index.setdefault(key(f), []).append(os.path.join(dp, f))


walk(ROOT, 6)
for extra in ("~/Downloads", "~/Desktop"):
    d = os.path.expanduser(extra)
    if os.path.isdir(d):
        walk(d, 2)


def find_elsewhere(name, not_in):
    hits = [p for p in index.get(key(name), [])
            if os.path.dirname(os.path.abspath(p)) != os.path.abspath(not_in)]
    hits.sort(key=lambda p: (not p.startswith(ROOT), len(p)))
    return hits[0] if hits else None


def in_folder(folder, name):
    """Path of `name` in `folder`, accepting either spelling, else None."""
    if not os.path.isdir(folder):
        return None
    for f in os.listdir(folder):
        if key(f) == key(name):
            return os.path.join(folder, f)
    return None


def act(text, fn):
    actions.append(text)
    if FIX:
        try:
            fn()
            say(f"      done: {text}")
        except Exception as e:
            say(f"      !! could not {text}: {e}")
            problems.append(text)
    else:
        say(f"      --fix would: {text}")


# --------------------------------------------------- 1. session folders
def folder(canon, required=True, link=True):
    """Return the path of backend/<canon>, linking a differently spelt folder."""
    p = os.path.join(BACK, canon)
    if os.path.isdir(p):
        return p
    cands = []
    for top in (BACK, ROOT):
        for d in os.listdir(top):
            full = os.path.join(top, d)
            if os.path.isdir(full) and key(d) == key(canon):
                cands.append(full)
        for d in os.listdir(top):           # one level down, e.g. data/Session A
            sub = os.path.join(top, d)
            if os.path.isdir(sub) and d not in SKIP_DIRS and not d.startswith("."):
                try:
                    for e in os.listdir(sub):
                        if key(e) == key(canon) and os.path.isdir(os.path.join(sub, e)):
                            cands.append(os.path.join(sub, e))
                except OSError:
                    pass
    if cands:
        src = cands[0]
        if not link:
            say(f"  found at {rel(src)}/")
            return src
        say(f"  [move]  {canon}/ is at {rel(src)}/")
        act(f"link backend/{canon} -> {rel(src)}",
            lambda: os.symlink(os.path.relpath(src, BACK), p))
        return p if FIX else src
    if required:
        say(f"  [MISS]  backend/{canon}/ not found")
        problems.append(f"backend/{canon}/")
    return None


planned = set()      # files a move above will bring into videos/


def check_files(fold, canon, names, label):
    ok, missing = 0, []
    for n in names:
        if fold and in_folder(fold, n):
            ok += 1
            continue
        if (canon, key(n)) in planned:
            say(f"  [move]  {n}  comes from session_B/ (see above)")
            continue
        src = find_elsewhere(n, fold or "")
        if src:
            say(f"  [move]  {n}  is at  {rel(src)}")
            dst = os.path.join(BACK, canon)
            act(f"copy {rel(src)} -> backend/{canon}/",
                lambda s=src, d=dst: (os.makedirs(d, exist_ok=True), shutil.copy2(s, d)))
        else:
            missing.append(n)
            problems.append(f"{canon}/{n}")
    if missing:
        say(f"  [MISS]  {len(missing)} file(s) not found in the project, ~/Downloads or on the "
            f"Desktop:")
        for n in missing:
            say(f"            {n}")
    say(f"  {label}: {ok} of {len(names)} in place")
    return ok


def videos(fold):
    if not fold or not os.path.isdir(fold):
        return []
    def j_of(f):
        m = re.search(r"(\d{2,5})\s*mA", f, re.I)
        return (int(m.group(1)) if m else 10 ** 9, f)
    return sorted((f for f in os.listdir(fold) if f.lower().endswith(VID)), key=j_of)


def describe(fold, title):
    v = videos(fold)
    js = []
    for f in v:
        m = re.search(r"(\d{2,5})\s*mA", f, re.I)
        js.append(int(m.group(1)) if m else None)
    volt = sum(bool(re.search(r"\d{3,5}\s*mV", f, re.I)) for f in v)
    say(f"  {title}: {len(v)} recordings, {volt} with a voltage in the name")
    good = sorted(j for j in js if j is not None)
    if good:
        say(f"      current values in the names: {good}")
    for f in v:
        say(f"      {f}")
    return v, good


say("CHECK LAYOUT" + ("   (--fix: putting things in place)" if FIX else "   (looking only)"))
say("=" * 78)
say(f"  project folder: {ROOT}")

for canon, names, label in (("session_A", SESSION_A, "Session A"),
                            ("session_B", SESSION_B, "Session B"),
                            ("session_C", SESSION_C, "Session C")):
    say(f"\n{label}  ->  backend/{canon}/")
    say("-" * 78)
    f = folder(canon)
    check_files(f, canon, names, label)
    extra = [x for x in videos(f) if key(x) not in {key(n) for n in names}]
    for x in extra:
        src = os.path.join(f, x)
        if canon == "session_B" and x.lower().endswith(".mp4") and key(x) in {key(n) for n in VIDEOS}:
            say(f"  [move]  {x} is an MP4 copy and belongs in videos/, not session_B/")
            dst = os.path.join(BACK, "videos")
            if in_folder(dst, x):
                say("      (videos/ already has it; this copy in session_B/ can be deleted by you)")
                problems.append(f"duplicate {canon}/{x}")
            else:
                planned.add(("videos", key(x)))
                act(f"move backend/{canon}/{x} -> backend/videos/",
                    lambda s=src, d=dst: (os.makedirs(d, exist_ok=True), shutil.move(s, d)))
        else:
            say(f"  [note]  {x}: not one of the files the report uses. It is ignored as long as "
                f"results_{canon[-1]}/summary.csv exists; remove it before any --rerun-batch.")

say("\nMP4 and AVI copies  ->  backend/videos/")
say("-" * 78)
check_files(folder("videos", required=False), "videos", VIDEOS, "videos")

say("\nScreen captures and the validation image  ->  backend/screenshots/")
say("-" * 78)
shots = folder("screenshots", required=False)
check_files(shots, "screenshots", CAPTURES, "screen captures (either spelling is fine)")
img_ok = any(in_folder(d, IMAGE) for d in (BACK, ROOT, os.path.join(BACK, "screenshots")))
if img_ok:
    say(f"  {IMAGE}: in place")
else:
    src = find_elsewhere(IMAGE, "")
    if src:
        say(f"  [move]  {IMAGE} is at {rel(src)}")
        act(f"copy {rel(src)} -> backend/", lambda: shutil.copy2(src, BACK))
    else:
        say(f"  [MISS]  {IMAGE}")
        problems.append(IMAGE)

say("\nScreenshots of the application (you take these, Step 4)  ->  backend/screenshots/")
say("-" * 78)
for n, how in APP_SHOTS.items():
    if in_folder(os.path.join(BACK, "screenshots"), n):
        say(f"  [ok]    {n}")
        continue
    src = find_elsewhere(n, os.path.join(BACK, "screenshots"))
    if src:
        say(f"  [move]  {n} is at {rel(src)}")
        act(f"copy {rel(src)} -> backend/screenshots/",
            lambda s=src: (os.makedirs(os.path.join(BACK, "screenshots"), exist_ok=True),
                           shutil.copy2(s, os.path.join(BACK, "screenshots"))))
    else:
        say(f"  [TAKE]  {n}   ({how}; save it under exactly this name)")
        problems.append(n)

say("\nBatch results already computed")
say("-" * 78)
for r in ("results_A", "results_B", "results_C", "results_screenshots"):
    p = os.path.join(BACK, r, "summary.csv")
    if os.path.exists(p):
        say(f"  [ok]    {r}/summary.csv")
    else:
        say(f"  [MISS]  {r}/summary.csv  (make_all_figures.py recomputes it from the recordings)")
tr = os.path.join(BACK, "training_report.json")
if os.path.exists(tr) and "random_split_r2" in open(tr).read():
    say("  [ok]    training_report.json (current version)")
else:
    say("  [RUN]   training_report.json is missing or old: python3 train_from_videos.py videos")
    problems.append("training_report.json")

say("\nSession D  ->  backend/Session_D/   (Section 4.14)")
say("-" * 78)
fd = folder("Session_D")
v, js = describe(fd, "Session D")
uj = sorted(set(js))
steps = sorted(set(b - a for a, b in zip(uj[:-1], uj[1:])))
if v:
    say(f"  steps between current values: {steps}")
    if len(v) != 10 or steps != [50]:
        say("  [note]  The report describes Session D as ten recordings at 50 to 500 "
            "mA/cm2 in steps of 50.")
        say("          make_session_D.py writes Section 4.14 from whatever is in the folder, "
            "but Table 3.2")
        say("          states ten recordings: tell me if this folder is meant to be different.")

for s, sec in (("Session_E", "4.15"), ("Session_F", "4.16")):
    say(f"\nSession {s[-1]}  ->  backend/{s}/   (Section {sec})")
    say("-" * 78)
    fs = folder(s)
    v, _ = describe(fs, "Session " + s[-1])
    if fs and not v:
        say(f"  [MISS]  no recordings in {s}/")
        problems.append(f"{s}/ recordings")

say("\n" + "=" * 78)
if not actions and not problems:
    say("  Everything is in place. Continue with Step 5 of the test plan.")
else:
    if actions and not FIX:
        say(f"  {len(actions)} thing(s) can be put in place automatically:  python3 check_layout.py --fix")
    if actions and FIX:
        say(f"  {len(actions)} action(s) carried out. Run  python3 check_layout.py  again to confirm.")
    left = [p for p in problems if p not in APP_SHOTS and p != "training_report.json"]
    if left:
        say(f"  {len(left)} item(s) not found anywhere (listed above with [MISS]).")
    todo = [p for p in problems if p in APP_SHOTS]
    if todo:
        say(f"  {len(todo)} screenshot(s) still to take (Step 4).")
say("=" * 78)
with open(os.path.join(ROOT, "layout_report.txt"), "w") as fh:
    fh.write("\n".join(out_lines) + "\n")
print("  written to layout_report.txt")
