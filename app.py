"""
app.py
------
Minimal Flask backend for the Smart Pothole Detection System.

Fixes applied:
  - Forces UTF-8 stdout/stderr at startup to prevent UnicodeEncodeError
    when running on Windows without 'python -X utf8'.
  - Warns at startup if email credentials are not configured.
"""

# ── Force UTF-8 output on Windows (fix UnicodeEncodeError) ────
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import os
import sys
import uuid
from datetime import datetime

from flask import (
    Flask, request, render_template, jsonify,
    send_from_directory, url_for
)
from werkzeug.utils import secure_filename

# Import local pipeline
from main import run_pipeline
from email_alert import is_configured as _email_configured

# Optional: SQLite history/stats endpoints
try:
    from database.db import get_recent_detections, get_stats, init_db
    init_db()
    _DB_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Database not initialized: {e}")
    _DB_AVAILABLE = False

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

UPLOAD_DIR  = os.path.join("static", "uploads")
RESULTS_DIR = os.path.join("static", "results")
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}

os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Warn at startup if email is unconfigured
if not _email_configured():
    print("[WARNING] Email alerts are disabled – POTHOLE_SENDER_EMAIL /")
    print("          POTHOLE_SENDER_PASSWORD env vars are not set.")
    print("          See README.md -> Email Alert Setup for instructions.")


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the upload form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accept a road image, run the full AI pipeline, return JSON.
    """
    # ── Validate upload ──────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file in request (key='image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXT}"}), 400

    # ── Save upload ──────────────────────────────────────────────
    ext      = file.filename.rsplit(".", 1)[1].lower()
    fname    = f"{uuid.uuid4().hex}.{ext}"
    img_path = os.path.join(UPLOAD_DIR, secure_filename(fname))
    file.save(img_path)

    location = request.form.get("location", "Road Segment (uploaded)")

    # ── Run pipeline ─────────────────────────────────────────────
    try:
        result = run_pipeline(
            image_path      = img_path,
            yolo_model_path = "best.pt",
            ml_model_path   = "severity_model.pkl",
            output_dir      = RESULTS_DIR,
            send_email      = True,
            location        = location,
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500

    # ── Build response ───────────────────────────────────────────
    output_image_url = None
    if result.get("output_image_path") and os.path.isfile(result["output_image_path"]):
        img_filename     = os.path.basename(result["output_image_path"])
        output_image_url = url_for("serve_result", filename=img_filename, _external=False)

    response = {
        "success":          True,
        "pothole_count":    result["pothole_count"],
        "highest_severity": result["highest_severity"],
        "avg_confidence":   result.get("avg_confidence", 0),
        "alert_sent":       result["alert_sent"],
        "timestamp":        result["timestamp"],
        "output_image_url": output_image_url,
        "detections": [
            {
                "bbox":       [d["x1"], d["y1"], d["x2"], d["y2"]],
                "width":      d["width"],
                "height":     d["height"],
                "area":       d["area"],
                "confidence": d["confidence"],
                "severity":   result["severities"][i] if i < len(result["severities"]) else "Unknown",
            }
            for i, d in enumerate(result["detections"])
        ],
    }

    return jsonify(response), 200


@app.route("/history")
def history():
    """GET /history – return recent detection runs from SQLite."""
    if not _DB_AVAILABLE:
        return jsonify({"error": "Database not available"}), 503
    limit = min(int(request.args.get("limit", 20)), 100)
    records = get_recent_detections(limit)
    return jsonify({"records": records, "count": len(records)}), 200


@app.route("/stats")
def stats():
    """GET /stats – return aggregate detection statistics."""
    if not _DB_AVAILABLE:
        return jsonify({"error": "Database not available"}), 503
    return jsonify(get_stats()), 200


@app.route("/results/<path:filename>")
def serve_result(filename):
    """Serve annotated result images."""
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serve original uploaded images."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/health")
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "db_available": _DB_AVAILABLE,
    }), 200


# ──────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Smart Pothole Detection System – Flask Server")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
