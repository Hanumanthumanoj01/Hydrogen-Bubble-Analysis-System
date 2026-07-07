# Hydrogen Bubble Analysis System
**AI-based bubble detection · Shannon entropy · Alkaline water electrolysis**

M.Sc. Information Technology · Frankfurt University of Applied Sciences · Fuel Cell Laboratory  
Author: Manoj Hanumanthu | Supervisor: Prof. Dr. Enno Wagner

---

## Project structure

```
hydrogen-bubble-analysis/
├── backend/
│   ├── main.py              ← FastAPI app — /analyze, /simulate endpoints
│   ├── bubble_analysis.py   ← Shannon entropy + Hough bubble detection + all metrics
│   ├── simulator.py         ← Physics-grounded electrolyzer simulator
│   ├── ml_model.py          ← Random Forest / XGBoost classifier
│   ├── ml_routes.py         ← ML API endpoints (/ml/train, /ml/predict, …)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                     ← 3-tab root
│   │   ├── pages/
│   │   │   ├── LiveSimulator.jsx       ← Tab 1
│   │   │   ├── ImageAnalyzer.jsx       ← Tab 2
│   │   │   └── CameraAnalyzer.jsx      ← Tab 3
│   │   └── utils/export.js             ← Excel / CSV export (SheetJS)
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── calibrate.py         ← Interactive Hough parameter tuner (OpenCV GUI)
├── data/
│   └── samples/             ← Place bubble_density_*.jpg here
├── ml/                      ← (Sprint 7) ML experiments / notebooks
├── start_windows.bat
├── start_mac_linux.sh
└── README.md
```

---

## Quick start

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Install & run backend
```bash
cd backend
pip install -r requirements.txt
python main.py
# → API running at http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

### 2. Install & run frontend
```bash
cd frontend
npm install
npm run dev
# → App running at http://localhost:5173
```

### Or use the one-click start scripts:
- **Windows:** double-click `start_windows.bat`
- **Mac / Linux:** `bash start_mac_linux.sh`

---

## Tab 1 — Live Electrolyzer Simulator
- Sliders: current density (50–1000 mA/cm²), KOH concentration, temperature, electrode type, cathode area
- Animated bubble canvas: bubble count and size respond physically to parameters
- Real-time metrics: Shannon H, H_norm, bubble count, mean diameter, H₂ (Faraday), cell voltage, nucleation sites, coverage, Gibbs entropy rate
- Entropy vs. current density line chart
- Frame log table
- **Export to Excel (.xlsx)**

## Tab 2 — Image Analyzer
- Drag-and-drop or click to upload microscope image
- Full two-branch pipeline:
  - **Branch 1:** Shannon entropy from pixel histogram (reproduces Wagner et al. 2025)
  - **Branch 2:** Top-hat background suppression → Hough Circle Transform → bubble list
- Derived metrics: diameter, curvature K=1/r, coverage (binary mask), nucleation sites (grid method), size distribution (Wagner et al. bins), H₂ Faraday estimate
- Three views: detection overlay (red circles) | density heatmap | pixel brightness histogram
- Entropy interpretation label
- **Export to Excel** — per-bubble coordinates + summary sheet

## Tab 3 — Camera / Video
- Enumerates all cameras including USB lab cameras
- Live video feed with detection overlay at 2 fps
- Frame log table: frame, timestamp, bubble count, H, H_norm%, mean Ø, coverage
- Capture frame on demand
- **Export frame log to Excel**

---

## Calibration (Sprint 1–2)
```bash
python scripts/calibrate.py --image data/samples/bubble_density_100mA.jpg
```
OpenCV window with live trackbars. Press **s** to save `data/calibration_params.json`.

---

## Machine Learning (Sprint 7+)
```
POST /ml/add-sample   — label an image and add to dataset
POST /ml/train        — train Random Forest (or XGBoost with ?use_xgboost=true)
POST /ml/predict      — predict electrode type + current density band from image
GET  /ml/feature-importance  — show feature weights
GET  /ml/dataset-info        — sample count, model status
```

---

## Key equations (Wagner et al. 2025)

| Symbol | Formula | Description |
|--------|---------|-------------|
| H | −Σ pᵢ ln pᵢ | Shannon information entropy (Eq. 7) |
| H_norm | H / ln(256) | Normalised entropy, 0–1 (Eq. 8) |
| K | 1 / r | Bubble curvature (1/mm) — micro-region model |
| ṅ | (J·A) / (z·F) | H₂ molar flow rate (Faraday's law) |
| V̇_H₂ | ṅ × 22400 × 60 | H₂ volume flow (mL/min, STP) |

Calibration: **0.0064 mm/pixel** · Cathode area: **2.54 cm²** · z = 2 · F = 96485 C/mol

---

## References
- Wagner, E., Tennert, R., Probsthain, L. & Mishra, R. (2025). *Heat and Mass Transfer*, 61, 108.
- Shannon, C. E. (1948). *Bell System Technical Journal*, 27, 379–423.
