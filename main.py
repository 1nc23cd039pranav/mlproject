"""
main.py
-------
End-to-end pipeline orchestrator for the Smart Pothole Detection System.

Usage:
    python main.py <image_path> [--no-email] [--model best.pt] [--out results/]

This module ties together:
    1. YOLOv8 pothole detection    (detect.py)
    2. RandomForest severity       (classify.py)
    3. Email alert automation      (email_alert.py)
    4. SQLite detection logging    (database/db.py)
    5. Annotated image saving
"""

# ── Force UTF-8 on Windows so print() doesn't crash with UnicodeEncodeError ──
import sys, io as _io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import sys
import argparse
import json
from datetime import datetime

import cv2

# Local modules
from detect import load_yolo_model, detect_potholes, draw_detections
from classify import predict_batch, predict_severity
from email_alert import send_alert

# Optional: SQLite logging (non-fatal if database module unavailable)
try:
    from database.db import log_detection, init_db
    init_db()
    _DB_AVAILABLE = True
except Exception as _db_err:
    print(f"[WARNING] Database module not available: {_db_err}")
    _DB_AVAILABLE = False


# ──────────────────────────────────────────────
# Lazy model handles (loaded once per process)
# ──────────────────────────────────────────────
_yolo_model = None


def get_yolo_model(model_path: str = "best.pt"):
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = load_yolo_model(model_path)
    return _yolo_model


# ──────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────
def run_pipeline(
    image_path: str,
    yolo_model_path: str = "best.pt",
    ml_model_path: str   = "severity_model.pkl",
    output_dir: str      = "static/results",
    send_email: bool     = True,
    location: str        = "Unknown Location",
) -> dict:
    """
    Full detection + classification + alert pipeline.

    Parameters
    ----------
    image_path      : Path to the input road image
    yolo_model_path : Path to best.pt (or yolov8n.pt fallback)
    ml_model_path   : Path to severity_model.pkl
    output_dir      : Directory to save the annotated output image
    send_email      : Whether to dispatch email alerts for HIGH severity
    location        : Human-readable location string (optional)

    Returns
    -------
    dict with keys:
        pothole_count, detections, severities, highest_severity,
        alert_sent, output_image_path, timestamp
    """

    # ── Validate input ──────────────────────────────────────────
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*55}")
    print(f"  POTHOLE DETECTION PIPELINE  –  {timestamp}")
    print(f"{'='*55}")
    print(f"  Input image : {image_path}")

    # ── Step 1 : Load image ─────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    h_img, w_img = image.shape[:2]
    print(f"  Image size  : {w_img}×{h_img} px")

    # ── Step 2 : Detect potholes with YOLOv8 ───────────────────
    print("\n[STEP 1] Running YOLOv8 pothole detection …")
    # Increase sensitivity during early-stage training
    model = get_yolo_model(yolo_model_path)
    detections = detect_potholes(model, image, conf_threshold=0.10)

    pothole_count = len(detections)
    print(f"  -> {pothole_count} pothole(s) detected")

    if pothole_count == 0:
        print("  [INFO] No potholes found. Pipeline complete.")
        result = {
            "pothole_count": 0,
            "detections": [],
            "severities": [],
            "highest_severity": "None",
            "avg_confidence": 0,
            "alert_sent": False,
            "output_image_path": None,
            "timestamp": timestamp,
        }
        # Log even zero-pothole results
        if _DB_AVAILABLE:
            log_detection(
                image_path=image_path, location=location,
                pothole_count=0, highest_severity="None",
                avg_confidence=0, alert_sent=False,
                output_image=None, timestamp=timestamp,
            )
        return result

    # Print individual detections
    for i, d in enumerate(detections, 1):
        print(f"  Pothole {i}: "
              f"w={d['width']}px  h={d['height']}px  "
              f"area={d['area']}px²  conf={d['confidence']:.2%}")

    # ── Step 3 : Classify severity ──────────────────────────────
    print("\n[STEP 2] Classifying severity with RandomForest …")
    severities = predict_batch(detections, ml_model_path)

    for i, (det, sev) in enumerate(zip(detections, severities), 1):
        print(f"  Pothole {i}: {sev}")

    # Overall highest severity  (High > Medium > Low)
    sev_rank = {"High": 3, "Medium": 2, "Low": 1}
    highest_severity = max(severities, key=lambda s: sev_rank.get(s, 0))
    avg_confidence   = sum(d["confidence"] for d in detections) / len(detections)

    print(f"\n  Highest severity : {highest_severity}")
    print(f"  Avg confidence   : {avg_confidence:.2%}")

    # ── Step 4 : Draw & save annotated image ───────────────────
    # Save image FIRST so we can attach it to the email
    print("\n[STEP 3] Saving annotated output image …")
    os.makedirs(output_dir, exist_ok=True)

    annotated = draw_detections(image, detections, severities)
    fname = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, annotated)
    print(f"  -> Saved: {out_path}")

    # ── Step 5 : Send alert if HIGH ─────────────────────────────
    alert_sent = False
    if send_email and highest_severity == "High":
        print("\n[STEP 4] HIGH severity detected – sending email alert …")
        alert_sent = send_alert(
            severity=highest_severity,
            confidence=avg_confidence,
            pothole_count=pothole_count,
            image_path=out_path,     # attach the annotated image
            location=location,
        )
    else:
        print(f"\n[STEP 4] No alert required (severity = {highest_severity})")

    # ── Step 6 : Log to SQLite ──────────────────────────────────
    if _DB_AVAILABLE:
        db_id = log_detection(
            image_path=image_path,
            location=location,
            pothole_count=pothole_count,
            highest_severity=highest_severity,
            avg_confidence=avg_confidence,
            alert_sent=alert_sent,
            output_image=out_path,
            detections=detections,
            severities=severities,
            timestamp=timestamp,
        )
        print(f"\n[STEP 5] Detection logged to SQLite (ID: {db_id})")

    # ── Summary ─────────────────────────────────────────────────
    result = {
        "pothole_count":    pothole_count,
        "detections":       detections,
        "severities":       severities,
        "highest_severity": highest_severity,
        "avg_confidence":   round(avg_confidence, 4),
        "alert_sent":       alert_sent,
        "output_image_path": out_path,
        "timestamp":        timestamp,
    }

    print(f"\n{'='*55}")
    print("  PIPELINE COMPLETE")
    print(f"  Potholes      : {pothole_count}")
    print(f"  Highest Sev.  : {highest_severity}")
    print(f"  Alert sent    : {alert_sent}")
    print(f"  Output image  : {out_path}")
    print(f"{'='*55}\n")

    return result


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Smart Pothole Detection – End-to-End Pipeline"
    )
    parser.add_argument("image", help="Path to input road image")
    parser.add_argument("--model",    default="best.pt",           help="YOLOv8 model path")
    parser.add_argument("--ml-model", default="severity_model.pkl", help="ML model path")
    parser.add_argument("--out",      default="static/results",    help="Output directory")
    parser.add_argument("--location", default="Unknown Location",  help="Location description")
    parser.add_argument("--no-email", action="store_true",         help="Disable email alerts")

    args = parser.parse_args()

    try:
        result = run_pipeline(
            image_path      = args.image,
            yolo_model_path = args.model,
            ml_model_path   = args.ml_model,
            output_dir      = args.out,
            send_email      = not args.no_email,
            location        = args.location,
        )
        # Dump JSON summary (excluding raw detection list for brevity)
        print(json.dumps({
            k: v for k, v in result.items() if k != "detections"
        }, indent=2))

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
