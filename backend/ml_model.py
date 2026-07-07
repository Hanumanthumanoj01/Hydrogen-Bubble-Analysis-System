"""
ml_model.py
===========
Machine-learning pipeline for the hydrogen bubble analysis system.

Sprint 7+ deliverable — baseline Random Forest / XGBoost classifier.

Task: predict operating regime (electrode type + current density band)
      from the bubble-image feature vector produced by bubble_analysis1.py.

Feature vector (per image / frame):
  - shannon_entropy
  - shannon_entropy_norm
  - bubble_count
  - mean_diameter_mm
  - mean_curvature_1_per_mm
  - surface_coverage_pct
  - nucleation_sites
  - bubble_density_per_cm2
  - size_distribution counts [7 bins]
  → 15 features total

Labels:
  - current_density_band:  low (<200), medium (200–500), high (>500)  [mA/cm²]
  - electrode_type:        gde | mesh | ans
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Tuple

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data/labelled_dataset.json")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "../data/bubble_rf_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../data/label_encoder.pkl")


def extract_features(analysis_result: Dict[str, Any]) -> np.ndarray:
    """Extract 15-dimensional feature vector from an analysis result dict."""
    size_dist = analysis_result.get("size_distribution", {})
    dist_counts = size_dist.get("counts", [0] * 7)
    total = max(sum(dist_counts), 1)
    dist_norm = [c / total for c in dist_counts]   # relative frequencies

    features = [
        analysis_result.get("shannon_entropy", 0),
        analysis_result.get("shannon_entropy_norm", 0),
        analysis_result.get("bubble_count", 0),
        analysis_result.get("mean_diameter_mm", 0),
        analysis_result.get("mean_curvature_1_per_mm", 0),
        analysis_result.get("surface_coverage_pct", 0),
        analysis_result.get("nucleation_sites", 0),
        analysis_result.get("bubble_density_per_cm2", 0),
    ] + dist_norm
    return np.array(features, dtype=np.float32)


def current_density_band(cd: float) -> str:
    if cd < 200:
        return "low"
    elif cd <= 500:
        return "medium"
    else:
        return "high"


class BubbleMLModel:

    def __init__(self):
        self.model = None
        self.le_electrode = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.le_band = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.dataset: List[Dict] = []
        self._load_dataset()
        self._try_load_model()

    # ── Dataset management ────────────────────────────────────────────────────
    def _load_dataset(self):
        if os.path.exists(DATASET_PATH):
            with open(DATASET_PATH) as f:
                self.dataset = json.load(f)

    def _save_dataset(self):
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, "w") as f:
            json.dump(self.dataset, f, indent=2)

    def add_sample(
        self,
        analysis_result: Dict[str, Any],
        current_density: float,
        electrode_type: str,
    ):
        """Add a labelled sample to the dataset."""
        record = {
            "features": extract_features(analysis_result).tolist(),
            "current_density": current_density,
            "current_density_band": current_density_band(current_density),
            "electrode_type": electrode_type,
        }
        self.dataset.append(record)
        self._save_dataset()
        return {"dataset_size": len(self.dataset)}

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, use_xgboost: bool = False) -> Dict[str, Any]:
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not installed"}
        if len(self.dataset) < 10:
            return {"error": f"Need ≥10 samples; have {len(self.dataset)}"}

        X = np.array([s["features"] for s in self.dataset])
        y_electrode = np.array([s["electrode_type"] for s in self.dataset])
        y_band      = np.array([s["current_density_band"] for s in self.dataset])

        y_electrode_enc = self.le_electrode.fit_transform(y_electrode)
        y_band_enc      = self.le_band.fit_transform(y_band)

        # Combine into single multi-output target
        y_combined = y_electrode_enc * 10 + y_band_enc   # simple encoding trick

        X_tr, X_te, y_tr, y_te = train_test_split(X, y_combined, test_size=0.2, random_state=42)

        if use_xgboost and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                                            use_label_encoder=False, eval_metric="mlogloss")
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        cv_scores = cross_val_score(self.model, X, y_combined, cv=min(5, len(self.dataset)))

        # Save
        if SKLEARN_AVAILABLE:
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump((self.le_electrode, self.le_band), ENCODER_PATH)

        return {
            "accuracy": round(acc, 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std":  round(float(cv_scores.std()), 4),
            "dataset_size": len(self.dataset),
            "model_type": "XGBoost" if (use_xgboost and XGBOOST_AVAILABLE) else "RandomForest",
        }

    def _try_load_model(self):
        if not SKLEARN_AVAILABLE:
            return
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.le_electrode, self.le_band = joblib.load(ENCODER_PATH)

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Model not trained yet. Add samples and call /ml/train first."}
        X = extract_features(analysis_result).reshape(1, -1)
        combined = self.model.predict(X)[0]
        el_enc  = combined // 10
        band_enc = combined % 10
        electrode = self.le_electrode.inverse_transform([el_enc])[0]
        band      = self.le_band.inverse_transform([band_enc])[0]

        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0].max()

        return {
            "predicted_electrode_type": electrode,
            "predicted_current_density_band": band,
            "confidence": round(float(proba), 3) if proba is not None else None,
        }

    def feature_importance(self) -> Dict[str, Any]:
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {"error": "Model not available"}
        names = [
            "shannon_entropy", "shannon_entropy_norm", "bubble_count",
            "mean_diameter_mm", "mean_curvature", "surface_coverage",
            "nucleation_sites", "bubble_density",
            "dist_<0.1", "dist_0.1-0.2", "dist_0.2-0.4",
            "dist_0.4-0.6", "dist_0.6-0.8", "dist_0.8-1.1", "dist_>1.1"
        ]
        fi = self.model.feature_importances_.tolist()
        return {"feature_importances": dict(zip(names, [round(v, 4) for v in fi]))}
