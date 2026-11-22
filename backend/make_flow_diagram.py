"""
make_flow_diagram.py
====================
Draws the three diagrams of the report. Reads no data, so it always works.

WHERE IT LIVES   backend/make_flow_diagram.py
HOW TO RUN       cd backend
                 python3 make_flow_diagram.py
OUTPUT           report_figures/entropy_bridge.png    Figure 1, the two entropies
                 report_figures/system_flow.png       Figure 2, processing flow
                 report_figures/detection_stages.png  Figure 4, image-pipeline stages
SAFE TO RE-RUN   yes; it only overwrites these three files.
make_all_figures.py calls it and copies the files to figures_for_overleaf/.

How the boxes are drawn (November 2026)
---------------------------------------
Every box is sized from the measured size of its own text: the text is drawn
first, its extent is read back from the renderer, and the box is drawn around
it with a fixed margin. A label therefore cannot run over its border, whatever
the figure size or font size. (Earlier versions used a fixed line height that
was right for one figure size only, and the text of Figure 1 overflowed its
boxes.) The figures are drawn at the size they are printed in the report, so a
font size here is close to the printed size.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "report_figures"

C_INPUT = "#dbeafe"    # blue    input
C_IMG = "#fef3c7"      # amber   image / entropy
C_DET = "#e0f2fe"      # sky     detection
C_FIS = "#ede9fe"      # violet  Fisher information
C_THERMO = "#dcfce7"   # green   thermodynamics
C_OUT = "#fee2e2"      # red     comparison / results
C_EDGE = "#334155"
INK = "#0f172a"


class Canvas:
    """A figure whose axes fill it, with coordinates 0..100 in both directions."""

    def __init__(self, w_in, h_in):
        self.fig = plt.figure(figsize=(w_in, h_in))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()

    def _extent(self, t):
        bb = t.get_window_extent(self.r).transformed(self.ax.transData.inverted())
        return bb.width, bb.height

    def box(self, cx, top, text, color, fs=11, bold=False, minw=0, pad=1.2):
        """Box centred at cx with its top edge at `top`. Returns (x0, y0, x1, y1)."""
        t = self.ax.text(cx, 0, text, ha="center", va="center", fontsize=fs,
                         weight="bold" if bold else "normal", color=INK,
                         linespacing=1.35, zorder=3)
        w, h = self._extent(t)
        w = max(w + 2 * pad, minw)
        h = h + 1.6 * pad
        y0 = top - h
        t.set_y(y0 + h / 2)
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, y0), w, h,
                          boxstyle="round,pad=0,rounding_size=0.8",
                          facecolor=color, edgecolor=C_EDGE, linewidth=1.2, zorder=2))
        return (cx - w / 2, y0, cx + w / 2, top)

    def arrow(self, x1, y1, x2, y2, ls="-"):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                          mutation_scale=16, color=C_EDGE, linewidth=1.4,
                          linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))

    def text(self, x, y, s, fs=11, **kw):
        kw.setdefault("color", INK)
        self.ax.text(x, y, s, ha="center", va="center", fontsize=fs, **kw)

    def save(self, name):
        os.makedirs(OUT, exist_ok=True)
        self.fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight",
                         facecolor="white", pad_inches=0.15)
        plt.close(self.fig)
        # trim the empty margin below the last box
        import numpy as np
        from PIL import Image
        im = Image.open(os.path.join(OUT, name)).convert("RGB")
        a = np.asarray(im)
        rows = np.where((a < 245).any(axis=(1, 2)))[0]
        if len(rows):
            im.crop((0, 0, im.width, min(im.height, rows.max() + 30))).save(
                os.path.join(OUT, name))
        print("wrote " + OUT + "/" + name)


def chain(c, cx, top, items, color, fs, minw, gap=3.0):
    """Vertical chain of boxes joined by arrows; returns the last box."""
    y, last = top, None
    for t in items:
        bold = t.startswith("*")
        b = c.box(cx, y, t.lstrip("*"), color, fs, bold, minw)
        if last:
            c.arrow(cx, last[1], cx, b[3])
        last = b
        y = b[1] - gap
    return last


# ---------------------------------------------------------------- Figure 1
def entropy_bridge():
    c = Canvas(10.5, 8.2)
    c.text(50, 97, "Two entropies obtained by separate routes", fs=16, weight="bold")
    top = c.box(50, 92, "Electrolyser\nelectrical energy in, hydrogen and heat out",
                C_INPUT, 12, True, minw=46)
    left = chain(c, 24, 78, [
        "Camera\nimage of the electrode surface",
        "grey-scale histogram\n256 brightness levels, $p_i$",
        "*$H = -\\sum_i p_i \\ln p_i$\n$H_{norm} = H / \\ln 256$",
        "*Shannon entropy\n(information theory)"], C_IMG, 12, 38)
    right = chain(c, 76, 78, [
        "Voltmeter and thermometer\ncell voltage $V_{cell}$, temperature $T$",
        "overvoltage\n$\\eta = V_{cell} - V_{therm}$",
        "*$\\Delta S_{irr} = z F \\eta / T$",
        "*Thermodynamic entropy\nproduction"], C_THERMO, 12, 38)
    c.arrow(42, top[1], 24, 78)
    c.arrow(58, top[1], 76, 78)
    c.ax.text(50, 50, "no shared\ninput data", ha="center", va="center", fontsize=12,
              weight="bold", color="#b91c1c", zorder=3,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#fee2e2",
                        edgecolor="#b91c1c", linewidth=1.4))
    q = c.box(50, 26, "Question tested\nDoes the image entropy follow the thermodynamic "
              "state of the cell?\nIf it does, a camera is a non-contact measure of that state.",
              C_OUT, 12, True, minw=80)
    c.arrow(24, left[1], 34, q[3])
    c.arrow(76, right[1], 66, q[3])
    c.save("entropy_bridge.png")


# ---------------------------------------------------------------- Figure 2
def system_flow():
    c = Canvas(11.0, 12.5)
    FS, W = 13.5, 22
    c.text(50, 98, "Processing flow of the analysis", fs=17, weight="bold")
    inp = c.box(50, 95, "Laboratory data: images (.png, .jpg) and recordings (.avi, .mp4)\n"
                "operating point read from the file name, e.g. ..._688mA_2283mV_41deg",
                C_INPUT, FS, False, minw=80)
    cols = {"ent": 13, "det": 37.5, "fis": 62, "thr": 87}
    heads = {"ent": "Entropy", "det": "Bubble detection", "fis": "Fisher information",
             "thr": "Thermodynamics"}
    y0 = inp[1] - 6
    for k, x in cols.items():
        c.arrow(50 + (x - 50) * 0.45, inp[1], x, y0 + 0.2)
        c.text(x, y0 - 1.6, heads[k], fs=14.5, weight="bold")
    top = y0 - 4.5
    a = chain(c, cols["ent"], top, [
        "grey-scale\nhistogram",
        "*Shannon\nentropy $H$",
        "*$H_{norm}$\n(no adjustable\nparameter)"], C_IMG, FS, W)
    b = chain(c, cols["det"], top, [
        "single image:\nrim test at\n48 angles",
        "recording:\nbackground\nsubtraction",
        "rim fit with\ncheck against\nneighbouring frames",
        "size, coverage,\nsites, tracks"], C_DET, FS, W)
    f = chain(c, cols["fis"], top, [
        "one picture:\n$J$, $F_x$, $F_y$",
        "brightness field:\n$F_t = \\langle (\\partial_t I)^2 / I \\rangle$",
        "stable bubble\npaths",
        "across current:\n$F_j$, $I_j$"], C_FIS, FS, W)
    t = chain(c, cols["thr"], top, [
        "cell voltage $V$,\ntemperature $T$",
        "overvoltage\n$\\eta = V - V_{therm}$",
        "*$\\Delta S_{irr}$\n$= zF\\eta/T$"], C_THERMO, FS, W)
    low = min(a[1], b[1], f[1], t[1]) - 6
    cmp_ = c.box(50, low, "Comparison and engineering evaluation\n"
                 "image entropy against thermodynamic entropy production;\n"
                 "electrode structures compared by voltage, $\\Delta S_{irr}$, "
                 "entropy and Fisher information", C_OUT, FS + 0.5, True, minw=88)
    for box_, x in ((a, cols["ent"]), (b, cols["det"]), (f, cols["fis"]), (t, cols["thr"])):
        c.arrow(x, box_[1], 50 + (x - 50) * 0.7, cmp_[3])
    c.save("system_flow.png")


# ---------------------------------------------------------------- Figure 4
def detection_stages():
    """Each stage of the "Every bubble" detector (rim-coverage test, November 2026),
    what it does and its measured effect."""
    rows = [
        ("picture mask", "keeps only the camera picture: drops\noverlay text, black borders, player bar",
         "Screen capture, sensitivity low:\n215 circles before, 20 after."),
        ("working scale", "picture scaled to 420 px height;\nCLAHE local contrast",
         "Same settings for a 640 x 480 frame\nand a 2000 px screen capture."),
        ("ring and disc\nfilters", "candidates at 1.2-20 % of the height:\nbright ring (GDE), dark disc (ANS core)",
         "Candidates only; nothing is\naccepted at this stage."),
        ("rim test\n(48 angles)", "rim visible all round, at one radius,\nround (not a mesh square), above texture",
         "Rejects mesh streaks and openings.\nReplaced a size-weighted score."),
        ("size and\nnesting check", "no circle around smaller ones; large\ncircles confirmed by the rim detector",
         "Removes giant circles drawn\nacross several mesh cells."),
        ("duplicates", "strongest circle first;\none circle per bubble",
         "600 mA/cm$^2$, medium: 85 circles,\nnearly all on a visible rim."),
    ]
    c = Canvas(11.0, 11.5)
    c.text(50, 98, 'Detection stages, "Every bubble"', fs=16, weight="bold")
    c.text(41, 94.5, "what it does", fs=12, weight="bold", color="#475569")
    c.text(79, 94.5, "measured effect", fs=12, weight="bold", color="#475569")
    y, prev = 92, None
    for name, does, eff in rows:
        a = c.box(11, y, name, C_IMG, 11.5, True, minw=19)
        b = c.box(41, y, does, "#f8fafc", 10.5, False, minw=35)
        d = c.box(79, y, eff, "#f1f5f9", 10.5, False, minw=37)
        assert b[2] < d[0] - 0.5 and a[2] < b[0] - 0.5, f"Figure 4 boxes overlap: {name}"
        bottom = min(a[1], b[1], d[1])
        if prev:
            c.arrow(11, prev, 11, y)
        prev = a[1]
        y = bottom - 3.5
    c.save("detection_stages.png")


if __name__ == "__main__":
    entropy_bridge()
    system_flow()
    detection_stages()
    print("\nAll diagrams written to ./" + OUT + "/")
