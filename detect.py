"""
detect.py
---------
Standalone pothole detection module using YOLOv8.

Functions
---------
load_yolo_model(model_path)   – Load (or lazily download) the YOLO model
detect_potholes(model, image) – Run inference and return structured results
draw_detections(image, dets)  – Draw bounding boxes on the image
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────
def load_yolo_model(model_path: str = "best.pt") -> YOLO:
    """
    Load a YOLOv8 model.

    If *model_path* does not exist (e.g., training hasn't been run yet),
    fall back to the pretrained 'yolov8n.pt' so the app still works.
    """
    if os.path.isfile(model_path):
        print(f"[YOLO] Loading custom model: {model_path}")
        return YOLO(model_path)
    else:
        print(f"[YOLO] '{model_path}' not found – falling back to yolov8n.pt")
        return YOLO("yolov8n.pt")


# ──────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────
def detect_potholes(model: YOLO, image: np.ndarray, conf_threshold: float = 0.25):
    """
    Run YOLOv8 inference on a BGR image array.

    Returns
    -------
    list[dict] – One dict per detected pothole:
        {
          "x1": int, "y1": int, "x2": int, "y2": int,
          "confidence": float,
          "width": int, "height": int, "area": int,
          "class_id": int, "class_name": str
        }
    """
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf     = float(box.conf[0])
            class_id = int(box.cls[0])

            # Class name – use model names dict when available
            if result.names:
                class_name = result.names.get(class_id, "pothole")
            else:
                class_name = "pothole"

            w    = x2 - x1
            h    = y2 - y1
            area = w * h

            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": round(conf, 4),
                "width": w, "height": h, "area": area,
                "class_id": class_id, "class_name": class_name,
            })

    return detections


# ──────────────────────────────────────────────
# Annotation
# ──────────────────────────────────────────────
# Colour palette per severity
SEVERITY_COLORS = {
    "High":   (0, 0, 220),    # Red (BGR)
    "Medium": (0, 165, 255),  # Orange
    "Low":    (0, 200, 0),    # Green
    "Unknown":(200, 200, 0),  # Cyan-ish
}


def draw_detections(image: np.ndarray, detections: list, severities: list | None = None) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels on *image*.

    Parameters
    ----------
    image      : BGR numpy array (will be modified in-place on a copy)
    detections : list of dicts from detect_potholes()
    severities : optional list of severity strings aligned with detections

    Returns
    -------
    Annotated BGR numpy array
    """
    annotated = image.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf  = det["confidence"]
        sev   = (severities[i] if severities and i < len(severities) else "Unknown")
        color = SEVERITY_COLORS.get(sev, SEVERITY_COLORS["Unknown"])

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"Pothole | {sev} | {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Count overlay in top-left corner
    count_text = f"Potholes detected: {len(detections)}"
    cv2.putText(
        annotated, count_text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        annotated, count_text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 1, cv2.LINE_AA,
    )

    return annotated


# ──────────────────────────────────────────────
# CLI quick test:  python detect.py <image_path>
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not img_path or not os.path.isfile(img_path):
        print("Usage: python detect.py <path_to_image>")
        sys.exit(1)

    mdl = load_yolo_model("best.pt")
    img = cv2.imread(img_path)
    dets = detect_potholes(mdl, img)

    print(f"Detected {len(dets)} pothole(s):")
    for d in dets:
        print(f"  [{d['x1']},{d['y1']}->{d['x2']},{d['y2']}] "
              f"conf={d['confidence']:.2f}  w={d['width']}  h={d['height']}  area={d['area']}")

    annotated = draw_detections(img, dets)
    out = "detected_output.jpg"
    cv2.imwrite(out, annotated)
    print(f"Saved annotated image -> {out}")
