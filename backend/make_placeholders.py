"""
make_placeholders.py
====================
Creates clearly-marked placeholder images for the figures that must be
produced by hand (screenshots and a chosen per-file panel).

    python3 make_placeholders.py

Each placeholder prints its own production steps onto the image, so the
report stays self-documenting until the real pictures are swapped in.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "report_figures"


def placeholder(filename, title, steps, expect=None, size=(11, 6.6)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.add_patch(FancyBboxPatch((2, 2), 96, 96,
                 boxstyle="round,pad=0.6,rounding_size=1",
                 facecolor="#fffbeb", edgecolor="#d97706",
                 linewidth=3, linestyle="--"))

    ax.text(50, 90, "REPLACE THIS IMAGE", ha="center", fontsize=16,
            weight="bold", color="#b45309")
    ax.text(50, 83, title, ha="center", fontsize=12,
            weight="bold", color="#0f172a")

    ax.text(50, 74.5, "How to produce it:", ha="center", fontsize=10.5,
            weight="bold", color="#334155")

    y = 67
    for i, s in enumerate(steps, 1):
        ax.text(8, y, str(i) + ".", fontsize=10.5, weight="bold", color="#b45309")
        ax.text(12.5, y, s, fontsize=10, color="#0f172a", va="center")
        y -= 7.5

    if expect:
        y -= 2
        ax.text(50, y, "You should see:  " + expect, ha="center", fontsize=9.5,
                weight="bold", color="#166534",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#dcfce7",
                          edgecolor="#16a34a", linewidth=1.2))

    ax.text(50, 7, "Delete this placeholder once the real image is in place.",
            ha="center", fontsize=8.5, style="italic", color="#78716c")

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, filename), dpi=140,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote " + OUT + "/" + filename)


# ── Figure 3 — Image Analyzer ──────────────────────────────────────────────
placeholder(
    "PLACEHOLDER_image_analyzer.png",
    "Figure 3 — Image Analyzer (Tab 2) with a completed analysis",
    [
        "Terminal 1:  cd backend    uvicorn main:app --reload --port 8000",
        "Terminal 2:  cd frontend   npm run dev",
        "Open  localhost:5173  and go to Tab 2",
        "Drag in  bubble_density_600mA_02.jpg",
        "Set: 600 mA/cm2, area 2.54, sensitivity MEDIUM, electrode GDE",
        "Press  Analyze image  and screenshot the whole window",
    ],
    expect="110 bubbles,  H = 4.9510,  H_norm = 89.29 %,  H2 = 10.614 mL/min",
)

# ── Figure 5 — detection overlay ───────────────────────────────────────────
placeholder(
    "PLACEHOLDER_detection_overlay.png",
    "Figure 5 — Detection overlay on a real sample image",
    [
        "Run the same analysis as Figure 3 (600 mA, medium, GDE)",
        "Click the  Detection overlay  button above the image",
        "Screenshot only the overlay panel, not the whole window",
        "Each circle is an accepted detection; colour = confidence score",
    ],
    expect="110 circles, mostly green (high confidence)",
)

# ── Figure 6 — Video Analyzer ──────────────────────────────────────────────
placeholder(
    "PLACEHOLDER_video_analyzer.png",
    "Figure 6 — Video Analyzer (Tab 4) with a completed analysis",
    [
        "Both servers running, as for Figure 3",
        "Open  localhost:5173  and go to Tab 4",
        "Load  bubble_video_2283mV_688mA_41deg.avi  (AVI is supported)",
        "Analysis method:  Background subtraction",
        "Set current density 688, then press  Analyze video",
        "Screenshot the results panel once it finishes",
    ],
    expect="mean bubbles/frame, lifetime, growth rate, and the background image",
)

# ── Figure 14 — per-file panel ─────────────────────────────────────────────
placeholder(
    "PLACEHOLDER_per_file_panel.png",
    "Figure 14 — One per-file panel: original, detections, N(x,y), diameter",
    [
        "cd backend",
        "python3 run_all.py session_A      then    mv results results_A",
        "Open the folder  results_A/per_file/",
        "Pick a panel where the circles sit on real bubbles",
        "Good candidates: 250mA_01, 450mA_01, 500mA_01",
    ],
)

print("\nDone. Four placeholders written to ./" + OUT + "/")
