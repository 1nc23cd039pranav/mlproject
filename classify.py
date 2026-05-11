"""
classify.py
-----------
Loads the pre-trained RandomForestClassifier (severity_model.pkl) and
provides a single function to predict pothole severity from features.

Severity classes:  Low | Medium | High

Fixes applied:
  - Force str() cast on le.inverse_transform() output to prevent np.str_ type
    leaking into downstream comparisons (e.g. severity == "High" failing silently
    in Python 3.14 due to numpy scalar types).
  - Cache key now includes model_path so different model files don't collide.
"""

import os
import pickle
import numpy as np


# ──────────────────────────────────────────────
# Model loader  (singleton pattern, keyed by path)
# ──────────────────────────────────────────────
_CACHE: dict = {}   # { model_path: (model, label_encoder) }


def _load_model(model_path: str = "severity_model.pkl") -> tuple:
    """Return (model, label_encoder), loading from disk only once per path."""
    # Resolve to absolute path so relative/absolute variants share same cache
    abs_path = os.path.abspath(model_path)

    if abs_path not in _CACHE:
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(
                f"[CLASSIFY] Model not found: {abs_path}\n"
                "Run  python train_ml.py  first to generate severity_model.pkl"
            )
        with open(abs_path, "rb") as f:
            payload = pickle.load(f)
        _CACHE[abs_path] = (payload["model"], payload["label_encoder"])
        print(f"[CLASSIFY] Model loaded from {abs_path}")

    return _CACHE[abs_path]


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def predict_severity(
    width: float,
    height: float,
    area: float,
    confidence: float,
    model_path: str = "severity_model.pkl",
) -> dict:
    """
    Predict the severity of a single pothole.

    Parameters
    ----------
    width, height, area : Bounding-box dimensions in pixels
    confidence          : YOLOv8 detection confidence (0-1)
    model_path          : Path to severity_model.pkl

    Returns
    -------
    dict:
        {
          "severity"   : "Low" | "Medium" | "High",
          "confidence" : detection confidence (echo),
          "probabilities": {"Low": p, "Medium": p, "High": p}
        }
    """
    model, le = _load_model(model_path)

    features  = np.array([[width, height, area, confidence]], dtype=float)
    class_idx = model.predict(features)[0]

    # FIX: Force plain Python str – le.inverse_transform returns np.str_ in
    # numpy >= 1.24 / Python 3.14 which breaks == comparisons with str literals.
    severity  = str(le.inverse_transform([class_idx])[0])

    # Class probabilities  (order matches le.classes_)
    proba     = model.predict_proba(features)[0]
    prob_dict = {str(cls): round(float(p), 4) for cls, p in zip(le.classes_, proba)}

    return {
        "severity":      severity,
        "confidence":    round(confidence, 4),
        "probabilities": prob_dict,
    }


def predict_batch(detections: list, model_path: str = "severity_model.pkl") -> list:
    """
    Predict severity for a list of detections (output of detect.detect_potholes).

    Returns a list of plain-str severity strings aligned with the input list.
    """
    if not detections:
        return []

    model, le = _load_model(model_path)

    features = np.array(
        [[d["width"], d["height"], d["area"], d["confidence"]] for d in detections],
        dtype=float,
    )
    class_indices = model.predict(features)

    # FIX: Cast every element to plain str to avoid np.str_ comparison issues
    severities = [str(s) for s in le.inverse_transform(class_indices)]
    return severities


# ──────────────────────────────────────────────
# CLI quick test:  python classify.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {"width": 40,  "height": 30,  "area": 1200,  "confidence": 0.72},
        {"width": 90,  "height": 70,  "area": 6300,  "confidence": 0.87},
        {"width": 185, "height": 138, "area": 25530, "confidence": 0.94},
    ]

    print("Severity Prediction Test")
    print("-" * 45)
    for tc in test_cases:
        result = predict_severity(**tc)
        sev = result["severity"]
        # Verify it's a plain str (critical for downstream == comparisons)
        assert isinstance(sev, str), f"Expected str, got {type(sev)}"
        print(
            f"  w={tc['width']:>4}  h={tc['height']:>4}  area={tc['area']:>6}  "
            f"conf={tc['confidence']:.2f}  ->  {sev}  [type: {type(sev).__name__}]"
        )
    print("[PASS] All predictions are plain str type")
