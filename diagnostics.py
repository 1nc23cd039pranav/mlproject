"""
diagnostics.py
--------------
Full system health check for the Smart Pothole Detection System.
Run with: python -X utf8 diagnostics.py
"""
import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "database"))

PASS = "[OK]  "
FAIL = "[FAIL]"
WARN = "[WARN]"

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

all_ok = True

# ─────────────────────────────────────────────
# 1. Python version
# ─────────────────────────────────────────────
section("1. Python Version")
ver = sys.version_info
print(f"  Python {ver.major}.{ver.minor}.{ver.micro}")
if ver.major < 3 or (ver.major == 3 and ver.minor < 8):
    print(f"  {FAIL} Python 3.8+ required!")
    all_ok = False
else:
    print(f"  {PASS} Version OK")

# ─────────────────────────────────────────────
# 2. Module imports
# ─────────────────────────────────────────────
section("2. Module Imports")

modules = {
    "flask":          "Flask",
    "cv2":            "opencv-python",
    "ultralytics":    "ultralytics (YOLOv8)",
    "sklearn":        "scikit-learn",
    "numpy":          "numpy",
    "pandas":         "pandas",
    "matplotlib":     "matplotlib",
    "PIL":            "Pillow",
    "werkzeug":       "werkzeug",
}

for mod, label in modules.items():
    try:
        __import__(mod)
        print(f"  {PASS} {label}")
    except ImportError as e:
        print(f"  {FAIL} {label} -- {e}")
        all_ok = False

# ─────────────────────────────────────────────
# 3. Local module imports
# ─────────────────────────────────────────────
section("3. Local Module Imports")

local_modules = ["detect", "classify", "email_alert", "db"]
for mod in local_modules:
    try:
        __import__(mod)
        print(f"  {PASS} {mod}.py")
    except Exception as e:
        print(f"  {FAIL} {mod}.py -- {e}")
        all_ok = False

# Import main separately to avoid double YOLO load
try:
    from main import run_pipeline
    print(f"  {PASS} main.py (run_pipeline)")
except Exception as e:
    print(f"  {FAIL} main.py -- {e}")
    all_ok = False

try:
    from app import app as flask_app
    print(f"  {PASS} app.py (Flask app)")
except Exception as e:
    print(f"  {FAIL} app.py -- {e}")
    all_ok = False

# ─────────────────────────────────────────────
# 4. Required files
# ─────────────────────────────────────────────
section("4. Required Files")

required_files = [
    ("severity_model.pkl",   "Trained ML model"),
    ("pothole_dataset.csv",  "ML training dataset"),
    ("data.yaml",            "YOLO dataset config"),
    ("templates/index.html", "Flask HTML template"),
]

optional_files = [
    ("best.pt",    "Custom YOLO model (optional, falls back to yolov8n.pt)"),
    ("yolov8n.pt", "Pretrained YOLO fallback"),
]

for path, label in required_files:
    if os.path.isfile(path):
        size = os.path.getsize(path)
        print(f"  {PASS} {path:<30} ({size:,} bytes) -- {label}")
    else:
        print(f"  {FAIL} {path:<30} MISSING -- {label}")
        all_ok = False

for path, label in optional_files:
    if os.path.isfile(path):
        size = os.path.getsize(path)
        print(f"  {PASS} {path:<30} ({size:,} bytes) -- {label}")
    else:
        print(f"  {WARN} {path:<30} not found -- {label}")

# ─────────────────────────────────────────────
# 5. Required directories
# ─────────────────────────────────────────────
section("5. Directory Structure")

dirs = [
    "static/uploads",
    "static/results",
    "database",
    "templates",
    "dataset/train/images",
    "dataset/train/labels",
    "dataset/valid/images",
    "dataset/valid/labels",
    "dataset/test/images",
    "dataset/test/labels",
]

for d in dirs:
    if os.path.isdir(d):
        count = len(os.listdir(d))
        print(f"  {PASS} {d:<35} ({count} files)")
    else:
        print(f"  {FAIL} {d:<35} MISSING")
        os.makedirs(d, exist_ok=True)
        print(f"         Created: {d}")

# ─────────────────────────────────────────────
# 6. ML model integrity
# ─────────────────────────────────────────────
section("6. ML Model Integrity (RandomForest)")

try:
    with open("severity_model.pkl", "rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    le    = payload["label_encoder"]

    test_cases = [
        ([35,  25,  875,   0.72], "Low"),
        ([90,  68,  6120,  0.85], "Medium"),
        ([185, 138, 25530, 0.94], "High"),
    ]

    for feat, expected in test_cases:
        pred_idx = model.predict([feat])[0]
        pred = le.inverse_transform([pred_idx])[0]
        ok = pred == expected
        if not ok:
            all_ok = False
        tag = PASS if ok else FAIL
        print(f"  {tag} w={feat[0]:>4} h={feat[1]:>4} area={feat[2]:>6} conf={feat[3]:.2f}"
              f"  -> {pred} (expected {expected})")

    print(f"  {PASS} Classes: {list(le.classes_)}")
    print(f"  {PASS} Estimators: {model.n_estimators}")

except FileNotFoundError:
    print(f"  {FAIL} severity_model.pkl not found – run: python train_ml.py")
    all_ok = False
except Exception as e:
    print(f"  {FAIL} {e}")
    all_ok = False

# ─────────────────────────────────────────────
# 7. Flask routes
# ─────────────────────────────────────────────
section("7. Flask API Routes")

try:
    from app import app as flask_app
    rules = sorted([r.rule for r in flask_app.url_map.iter_rules()])
    for r in rules:
        print(f"  {PASS} {r}")
except Exception as e:
    print(f"  {FAIL} Could not inspect Flask routes: {e}")
    all_ok = False

# ─────────────────────────────────────────────
# 8. Database
# ─────────────────────────────────────────────
section("8. SQLite Database")

try:
    from database.db import init_db, get_stats
    init_db()
    stats = get_stats()
    print(f"  {PASS} database/db.py initialized")
    print(f"  {PASS} Total runs logged: {stats.get('total_runs', 0)}")
    print(f"  {PASS} Total potholes:    {stats.get('total_potholes', 0)}")
    print(f"  {PASS} Total alerts sent: {stats.get('total_alerts', 0)}")
except Exception as e:
    print(f"  {FAIL} Database error: {e}")
    all_ok = False

# ─────────────────────────────────────────────
# 9. Email config check
# ─────────────────────────────────────────────
section("9. Email Configuration")

sender   = os.environ.get("POTHOLE_SENDER_EMAIL",   "")
password = os.environ.get("POTHOLE_SENDER_PASSWORD", "")
receiver = os.environ.get("POTHOLE_RECEIVER_EMAIL",  "")

if sender and sender != "your_email@gmail.com":
    print(f"  {PASS} POTHOLE_SENDER_EMAIL    = {sender}")
else:
    print(f"  {WARN} POTHOLE_SENDER_EMAIL    not set (email alerts will fail)")

if password and password != "your_app_password":
    print(f"  {PASS} POTHOLE_SENDER_PASSWORD = (set)")
else:
    print(f"  {WARN} POTHOLE_SENDER_PASSWORD not set (email alerts will fail)")

if receiver and receiver != "maintenance@city.gov":
    print(f"  {PASS} POTHOLE_RECEIVER_EMAIL  = {receiver}")
else:
    print(f"  {WARN} POTHOLE_RECEIVER_EMAIL  using default: maintenance@city.gov")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
section("SUMMARY")
if all_ok:
    print("  ALL CHECKS PASSED -- System is ready!")
    print("  Run: python -X utf8 app.py")
    print("  Then open: http://127.0.0.1:5000")
else:
    print("  SOME CHECKS FAILED -- Fix issues above before running app.py")
print("=" * 55)
