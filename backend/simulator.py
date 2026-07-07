"""
simulator.py
============
Physics-grounded electrolyzer simulator.

Provides analytically-computed metrics (entropy, bubble count, H₂ production,
cell voltage, nucleation sites, coverage) as functions of:
  - Current density J  [mA/cm²]
  - KOH concentration  [mol/L]
  - Temperature        [°C]
  - Electrode type     (gde | mesh | ans)
  - Cathode area       [cm²]

Relationships are calibrated to match the trends reported in:
  Wagner, Tennert, Probsthain & Mishra (2025) Heat and Mass Transfer 61, 108.
"""

import math
from typing import Dict, Any


FARADAY = 96485.0
Z = 2
MOLAR_VOLUME_H2 = 22400.0   # mL/mol at STP


class ElectrolyzerSimulator:

    def compute(
        self,
        current_density: float,
        koh_concentration: float,
        temperature: float,
        electrode_type: str,
        cathode_area: float,
    ) -> Dict[str, Any]:
        """Return all simulated metrics for the given operating point."""

        J = current_density
        C = koh_concentration
        T = temperature

        # ── Electrode type factors (fitted to paper trends) ──────────────────
        el_factors = {
            "gde":  {"count_mult": 1.0,  "size_mult": 1.0,  "entropy_boost": 0.0},
            "mesh": {"count_mult": 3.5,  "size_mult": 0.3,  "entropy_boost": -0.03},
            "ans":  {"count_mult": 0.65, "size_mult": 1.3,  "entropy_boost": 0.01},
        }
        ef = el_factors.get(electrode_type, el_factors["gde"])

        # ── Bubble count ─────────────────────────────────────────────────────
        base_count = 10 + J * 0.20 * (1 + C * 0.05) * (1 + (T - 25) * 0.004)
        bubble_count = max(1, round(base_count * ef["count_mult"]))

        # ── Mean bubble diameter (mm) ─────────────────────────────────────────
        # Smaller bubbles at higher J; increases with KOH and temperature
        mean_diam_mm = max(0.08, (0.95 - J * 0.00055) * ef["size_mult"]
                          * (1 + (T - 25) * 0.001) * (1 + C * 0.01))

        # ── Shannon information entropy (normalised) ──────────────────────────
        # H_norm increases logarithmically with J — matches Fig. 5 of the paper
        H_norm = min(0.98,
                     0.60 + 0.33 * math.log1p(J / 100) / math.log1p(10)
                     * (1 + C * 0.015) * (1 + (T - 25) * 0.002)
                     + ef["entropy_boost"])
        H = H_norm * math.log(256)

        # ── Cell voltage (simplified Butler–Volmer + ohmic drop) ─────────────
        V_rev = 1.23
        V_thermo = 1.48
        eta_ohm = (J / 1000) * 0.5 / max(C, 0.1)  # ohmic: drops with KOH
        eta_act = 0.15 * math.log1p(J / 20)
        eta_temp = -(T - 25) * 0.001
        V_cell = V_rev + eta_ohm + eta_act + eta_temp
        V_cell = max(V_rev, round(V_cell, 3))

        # ── H₂ production rate — Faraday's law ───────────────────────────────
        J_A = J / 1000.0          # convert to A/cm²
        n_dot = (J_A * cathode_area) / (Z * FARADAY)   # mol/s
        H2_mL_min = round(n_dot * MOLAR_VOLUME_H2 * 60.0, 3)

        # ── Surface coverage (%) ──────────────────────────────────────────────
        r_mm = mean_diam_mm / 2.0
        electrode_area_mm2 = cathode_area * 100.0   # cm² → mm²
        bubble_area = bubble_count * math.pi * r_mm ** 2
        coverage = min(90.0, round(bubble_area / electrode_area_mm2 * 100.0, 1))

        # ── Nucleation sites ──────────────────────────────────────────────────
        ns = round(bubble_count * 0.3 + 4)

        # ── Gibbs entropy (thermodynamic) for comparison ──────────────────────
        # ΔS_prod = η·I / T  (Gibbs–Helmholtz)
        eta_total = V_cell - V_thermo
        I_total = (J / 1000.0) * cathode_area          # A
        T_K = temperature + 273.15
        gibbs_entropy_rate = round(eta_total * I_total / T_K * 1000, 4)   # mW/K

        return {
            "current_density": J,
            "koh_concentration": C,
            "temperature": T,
            "electrode_type": electrode_type,
            "cathode_area": cathode_area,
            "bubble_count": bubble_count,
            "mean_diameter_mm": round(mean_diam_mm, 4),
            "shannon_entropy": round(H, 4),
            "shannon_entropy_norm_pct": round(H_norm * 100, 2),
            "cell_voltage_V": V_cell,
            "H2_production_mL_per_min": H2_mL_min,
            "surface_coverage_pct": coverage,
            "nucleation_sites": ns,
            "gibbs_entropy_rate_mW_per_K": gibbs_entropy_rate,
        }
