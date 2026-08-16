"""
constants.py
============
SINGLE SOURCE OF TRUTH for the physical and calibration constants used by
BOTH analysis engines (bubble_analysis.py for images, video_analysis.py for
video).

WHY THIS FILE EXISTS
--------------------
These values were previously written out separately in each engine. That is
dangerous: changing the calibration in one file and forgetting the other
produces no error — the image pipeline and the video pipeline simply start
measuring in different units, and every millimetre result silently disagrees
between them.

Change a value HERE and both engines pick it up. Never redefine these inside
the engines.

OPEN QUESTIONS FOR THE SUPERVISOR (see notes on each value)
-----------------------------------------------------------
Two of these are assumptions inherited from the paper and are NOT yet
confirmed for this project's own data. Both are flagged below.
"""

# ─────────────────────────────────────────────────────────────────────────────
# ⚠ UNCONFIRMED — image scale
# ─────────────────────────────────────────────────────────────────────────────
# Taken from Wagner et al. (2025), for the telephoto/microscope lens used in
# that work.
#
# STATUS: confirmed for neither dataset in this project.
#   - GDE still images : ~455x415 px, from the paper's setup — probably valid
#   - ANS videos       : 640x480 px, recorded 13/07/2026 — UNKNOWN whether the
#                        magnification or working distance matched the paper
#
# A rough independent check using the known 0.5 mm ANS bores suggested a value
# nearer 0.0082 mm/px for the video setup (~28% larger), but that rested on a
# single feature near the frame edge and is not reliable.
#
# EVERYTHING IN MILLIMETRES DEPENDS ON THIS: bubble diameter, curvature,
# growth rate, the size-distribution bins, bubble density per cm².
# NOT affected: Shannon entropy, raw bubble counts (both are pixel-only).
#
# If the video setup differs, set MM_PER_PIXEL_VIDEO below rather than
# changing MM_PER_PIXEL, so the two datasets stay independently correct.
MM_PER_PIXEL: float = 0.0064          # mm per pixel — images

# Set this to a different value ONLY if the supervisor confirms the video
# optics differed. While it is None, video uses MM_PER_PIXEL.
MM_PER_PIXEL_VIDEO: float | None = None


def mm_per_pixel(source: str = "image") -> float:
    """Return the correct scale for 'image' or 'video' data."""
    if source == "video" and MM_PER_PIXEL_VIDEO is not None:
        return MM_PER_PIXEL_VIDEO
    return MM_PER_PIXEL


# ─────────────────────────────────────────────────────────────────────────────
# Electrochemistry — these are exact, no ambiguity
# ─────────────────────────────────────────────────────────────────────────────
FARADAY: float = 96485.0              # C/mol
Z_ELECTRONS: int = 2                  # electrons per H₂ molecule
M_H2: float = 2.016                   # g/mol

# ─────────────────────────────────────────────────────────────────────────────
# ⚠ UNCONFIRMED — molar volume (affects the H₂ production estimate only)
# ─────────────────────────────────────────────────────────────────────────────
# The exposé's Table 2 reports 1.90 / 3.80 / 11.39 mL/min for 100/200/600
# mA cm⁻². This code with 22400 mL/mol gives 1.769 / 3.538 / 10.614 — a
# constant ratio of 1.074, i.e. the table was produced with a molar volume of
# roughly 24055 mL/mol (~20 °C) rather than 22400 (0 °C, STP).
#
# The document and the code therefore currently DISAGREE. Ask the supervisor
# at which temperature/pressure the laboratory reports H₂ volume, then set
# this once and update the exposé table to match.
#
#   22400  → 0 °C, 1 atm (STP)          → 100 mA gives 1.769 mL/min
#   24055  → ~20 °C, 1 atm              → 100 mA gives 1.900 mL/min  ← exposé
#   24466  → 25 °C, 1 atm               → 100 mA gives 1.932 mL/min
MOLAR_VOLUME_H2: float = 22400.0      # mL/mol

# ─────────────────────────────────────────────────────────────────────────────
# Cell geometry
# ─────────────────────────────────────────────────────────────────────────────
CATHODE_AREA_CM2: float = 2.54        # Experiment 1, Wagner et al. (2025)

# ─────────────────────────────────────────────────────────────────────────────
# Thermodynamics — used for the Gibbs entropy comparison
# ─────────────────────────────────────────────────────────────────────────────
# Thermoneutral voltage at 25 °C. The ANS recordings ran at 39–42 °C, so this
# may warrant a temperature correction — another question for the supervisor.
V_THERMONEUTRAL: float = 1.481        # V

# Size-distribution bin edges in mm, matching Wagner et al. (2025), Fig. 4
SIZE_BINS_MM = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.1, 9999]
SIZE_BIN_LABELS = ["<0.1", "0.1–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.1", ">1.1"]
