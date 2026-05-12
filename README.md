# Smart Pothole Detection and Maintenance Alert System

A complete end-to-end AI system that detects potholes from road images using Deep Learning (YOLOv8), classifies their severity using Machine Learning (RandomForestClassifier), automatically sends email alerts for high-severity cases, and stores all results in a local SQLite database.

---
## Project Structure
```
mlproject/
│
├── app.py                  # Flask backend (API server)
├── main.py                 # Pipeline orchestrator
├── detect.py               # YOLOv8 detection module
├── classify.py             # RandomForest severity classifier
├── email_alert.py          # SMTP email alert system
├── train_yolo.py           # YOLOv8 training script
├── train_ml.py             # ML model training script
├── generate_sample_data.py # Synthetic dataset generator
├── test_pipeline.py        # End-to-end smoke test
│
├── data.yaml               # YOLO dataset configuration
├── pothole_dataset.csv     # ML training data (120 samples)
├── severity_model.pkl      # Trained RandomForest model (generated)
├── best.pt                 # Trained YOLO model (generated after training)
│
├── requirements.txt        # Python dependencies
├── setup.bat               # One-click Windows setup script
├── run.bat                 # One-click Flask server launcher
│
├── dataset/                # YOLO dataset folder
│   ├── train/images/       # Training images
│   ├── train/labels/       # Training YOLO labels
│   ├── valid/images/
│   ├── valid/labels/
│   ├── test/images/
│   └── test/labels/
│
├── templates/
│   └── index.html          # Web UI
│
├── static/
│   ├── uploads/            # Uploaded road images
│   └── results/            # Annotated output images + ML plots
│
└── database/
    ├── db.py               # SQLite logging module
    └── pothole_log.db      # SQLite database (auto-created)
```

---

## Quick Start (Windows)

### Option A – One-Click Setup

```bat
setup.bat    # Creates venv, installs packages, trains ML model
run.bat      # Starts Flask server at http://127.0.0.1:5000
```

### Option B – Manual Setup

```powershell
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic dataset (for testing without real images)
python generate_sample_data.py

# 4. Train the ML severity classifier
python train_ml.py

# 5. (Optional) Train YOLOv8 on real pothole dataset
python train_yolo.py

# 6. Run end-to-end smoke test
python -X utf8 test_pipeline.py

# 7. Start Flask server
python -X utf8 app.py
```

---

## System Workflow

```
User uploads road image
        |
        v
[YOLOv8] Detects potholes
  - draws bounding boxes
  - outputs coordinates + confidence scores
        |
        v
[Feature Extraction]
  - width, height, area, confidence
        |
        v
[RandomForestClassifier] Classifies severity
  - Low   (small potholes, low confidence)
  - Medium (mid-size potholes)
  - High  (large potholes, high confidence)
        |
        v
IF severity == HIGH
  --> Send email alert via SMTP
        |
        v
Save annotated output image (static/results/)
        |
        v
Log to SQLite (database/pothole_log.db)
        |
        v
Return JSON response to frontend
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI (upload form) |
| POST | `/predict` | Run AI pipeline on uploaded image |
| GET | `/history` | Recent detection history from SQLite |
| GET | `/stats` | Aggregate detection statistics |
| GET | `/health` | Health check |
| GET | `/results/<filename>` | Serve annotated result images |

### Example `/predict` Response

```json
{
  "success": true,
  "pothole_count": 2,
  "highest_severity": "High",
  "avg_confidence": 0.87,
  "alert_sent": true,
  "timestamp": "2026-05-11 20:30:00",
  "output_image_url": "/results/result_20260511_203000.jpg",
  "detections": [
    {
      "bbox": [120, 80, 350, 220],
      "width": 230,
      "height": 140,
      "area": 32200,
      "confidence": 0.94,
      "severity": "High"
    }
  ]
}
```

---

## Email Alert Setup

Edit `email_alert.py` or set environment variables:

```powershell
$env:POTHOLE_SENDER_EMAIL    = "your_gmail@gmail.com"
$env:POTHOLE_SENDER_PASSWORD = "your_app_password_here"
$env:POTHOLE_RECEIVER_EMAIL  = "maintenance@city.gov"
```

> **Important:** Use a Gmail **App Password**, not your regular password.
> Generate one at: https://myaccount.google.com/apppasswords
> (Requires 2-Factor Authentication to be enabled)

---

## YOLOv8 Training (with Real Dataset)

### Dataset Sources

| Dataset | URL |
|---------|-----|
| Pothole Detection (Kaggle) | https://www.kaggle.com/datasets/sovitrath/pothole-detection |
| RDD2022 Road Damage | https://github.com/sekilab/RoadDamageDetector |
| Pothole-600 | https://sites.google.com/view/pothole-600 |

### Dataset Structure Required

```
dataset/
├── train/
│   ├── images/   (*.jpg)
│   └── labels/   (*.txt in YOLO format)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### YOLO Label Format (one line per pothole)

```
0 <cx_norm> <cy_norm> <width_norm> <height_norm>
```

### Run Training

```bash
python train_yolo.py
# OR using yolo CLI:
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Training output: `runs/detect/pothole_detector/weights/best.pt`  
The script auto-copies it to `best.pt` in the project root.

---

## ML Model Details

| Property | Value |
|----------|-------|
| Algorithm | RandomForestClassifier |
| Features | width, height, area, confidence |
| Classes | Low, Medium, High |
| Training samples | 120 (40 per class) |
| Test accuracy | 100% |
| Cross-val (5-fold) | 100% |
| Saved as | `severity_model.pkl` |

### Severity Thresholds (approx.)

| Class | Width | Height | Area |
|-------|-------|--------|------|
| Low | < 65px | < 45px | < 3000px² |
| Medium | 65–130px | 45–85px | 3000–9000px² |
| High | > 130px | > 85px | > 9000px² |

---

## CLI Usage

```bash
# Detect potholes in a single image (no email)
python -X utf8 main.py path/to/road.jpg --no-email

# With custom output folder
python -X utf8 main.py road.jpg --out my_results/ --location "MG Road, Block 5"

# With email alert enabled
python -X utf8 main.py road.jpg --location "Highway NH-48, km 132"

# Quick classification test
python -X utf8 classify.py

# Quick YOLO detection test
python -X utf8 detect.py path/to/image.jpg
```

---

## Testing

```bash
python -X utf8 test_pipeline.py
```

This runs 4 automated tests:
1. ML model training (or load if already exists)
2. Severity classification correctness
3. YOLOv8 detection on synthetic image
4. Full end-to-end pipeline (no email)

---

## Dependencies

```
flask>=2.3.0
werkzeug>=2.3.0
ultralytics>=8.0.0       # YOLOv8
opencv-python>=4.8.0
scikit-learn>=1.3.0      # RandomForest
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pillow>=10.0.0
```

---
## Notes
- Run all Python commands with `python -X utf8` on Windows to avoid Unicode encoding issues in the terminal.
- The system works **without** a custom-trained `best.pt` — it falls back to `yolov8n.pt` (COCO-pretrained) which still detects common objects including road damage.
- For best pothole detection accuracy, train YOLOv8 on a real annotated pothole dataset.
- SQLite database is auto-created at `database/pothole_log.db` on first run.
