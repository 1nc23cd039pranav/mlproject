"""
download_and_train.py
---------------------
Downloads the Intel Unnati Pothole Detection dataset from Roboflow Universe
(~3,770 real road images) and trains YOLOv8n with transfer learning.

FIRST: Get a FREE Roboflow API key (takes 2 minutes):
  1. Go to https://app.roboflow.com/
  2. Sign up with Google or email (free)
  3. Go to Settings -> Roboflow API -> copy your Private API Key
  4. Set it below OR as an environment variable:
       Windows PowerShell:  $env:ROBOFLOW_API_KEY = "your_key_here"

Dataset: intel-unnati-training-program/pothole-detection-bqu6s
Images:  ~3,770 real-world road images with pothole annotations

Run:
    python download_and_train.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Force UTF-8 output on Windows
import io as _io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).parent

# ── CONFIG ────────────────────────────────────────────────────
# Paste your free Roboflow API key here OR set ROBOFLOW_API_KEY env var
ROBOFLOW_API_KEY = "AivsZ4RaYocBhwcozGrb"

# Dataset details (Requested: yeeun-kim-fyvoj/pothole-vhmow)
RF_WORKSPACE = "yeeun-kim-fyvoj"
RF_PROJECT   = "pothole-vhmow"
RF_VERSION   = 16     # v16 has 2,746 images (v2 had only 8)

# Training config
EPOCHS    = 30   # increase to 50+ if you have a GPU
BATCH     = 8    # safe for CPU; increase to 16 on GPU
IMG_SIZE  = 640
# ─────────────────────────────────────────────────────────────


def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


def ensure_roboflow():
    """Install roboflow package if missing."""
    try:
        import roboflow  # noqa
    except ImportError:
        print("[SETUP] Installing roboflow package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "roboflow"],
            check=True
        )
        print("[SETUP] roboflow installed.")


def get_api_key() -> str:
    """Get API key from env var, config above, or interactive prompt."""
    key = ROBOFLOW_API_KEY.strip()
    if key:
        return key

    print("\n" + "="*55)
    print("  ROBOFLOW API KEY REQUIRED")
    print("="*55)
    print("\n  Get your FREE key in 2 minutes:")
    print("  1. Open: https://app.roboflow.com/")
    print("  2. Sign up with Google (free)")
    print("  3. Settings -> Roboflow API -> copy Private Key")
    print("  4. Paste below, OR set env var:")
    print("       $env:ROBOFLOW_API_KEY = 'your_key'\n")
    key = input("  Enter your Roboflow API key: ").strip()
    if not key:
        print("[ERROR] No API key provided. Exiting.")
        sys.exit(1)
    return key


def download_dataset(api_key: str) -> Path:
    """Download the pothole dataset from Roboflow."""
    from roboflow import Roboflow

    section(f"Downloading Dataset from Roboflow Universe")
    print(f"  Workspace : {RF_WORKSPACE}")
    print(f"  Project   : {RF_PROJECT}")
    print(f"  Version   : {RF_VERSION}")
    print(f"  (~3,770 real-world pothole images)\n")

    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    version = project.version(RF_VERSION)

    # Download into project root – Roboflow creates a named subfolder
    dataset = version.download("yolov8", location=str(BASE / "rf_dataset"))
    print(f"\n  [OK] Dataset downloaded -> {dataset.location}")
    return Path(dataset.location)


def reorganise_into_dataset_folder(rf_path: Path):
    """
    Roboflow puts data at rf_dataset/{train,valid,test}/{images,labels}
    Our data.yaml points to dataset/{train,valid,test}
    Merge them so both paths work (and keep existing structure).
    """
    section("Reorganising Dataset Structure")

    for split in ["train", "valid", "test"]:
        for kind in ["images", "labels"]:
            src = rf_path / split / kind
            dst = BASE / "dataset" / split / kind
            dst.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                print(f"  [SKIP] {src} not found, skipping.")
                continue
            files = list(src.iterdir())
            for f in files:
                shutil.copy2(f, dst / f.name)
            print(f"  [OK] {split}/{kind}: copied {len(files)} files")


def write_data_yaml():
    """Write data.yaml pointing to absolute dataset path."""
    section("Updating data.yaml")
    ds_path = str((BASE / "dataset").resolve()).replace("\\", "/")
    content = f"""# YOLOv8 Pothole Detection
# Dataset: intel-unnati-training-program/pothole-detection-bqu6s
# Source:  https://universe.roboflow.com/intel-unnati-training-program/pothole-detection-bqu6s

path: {ds_path}
train: train/images
val:   valid/images
test:  test/images

nc: 1
names:
  0: pothole
"""
    yaml_path = BASE / "data.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"  [OK] data.yaml -> {yaml_path}")


def train():
    """Train YOLOv8n on the real dataset."""
    import torch
    from ultralytics import YOLO

    section("Training YOLOv8n (Transfer Learning)")

    device = "0" if torch.cuda.is_available() else "cpu"
    gpu    = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"  Device  : {gpu}")
    print(f"  Epochs  : {EPOCHS}  |  Batch: {BATCH}  |  ImgSz: {IMG_SIZE}")
    print(f"  Starting from yolov8n.pt (COCO pretrained weights)\n")

    model   = YOLO("yolov8n.pt")
    results = model.train(
        data     = str(BASE / "data.yaml"),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        name     = "pothole_real",
        patience = 10,
        exist_ok = True,
        device   = device,
        verbose  = True,
        # Augmentation
        fliplr   = 0.5,
        flipud   = 0.1,
        mosaic   = 0.8,
        degrees  = 5.0,
        hsv_s    = 0.5,
        hsv_v    = 0.3,
    )

    # Copy best.pt to project root
    best_src = BASE / "runs" / "detect" / "pothole_real" / "weights" / "best.pt"
    best_dst = BASE / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        size_mb = best_dst.stat().st_size / 1_000_000
        print(f"\n  [SUCCESS] best.pt saved -> {best_dst}  ({size_mb:.1f} MB)")
    else:
        print(f"\n  [WARNING] best.pt not found at expected path: {best_src}")

    return results


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    section("Smart Pothole Detection - Real Dataset Training")

    # 1. Ensure roboflow installed
    ensure_roboflow()

    # 2. Check if dataset already downloaded
    train_img_dir = BASE / "dataset" / "train" / "images"
    existing = len(list(train_img_dir.glob("*.jpg"))) if train_img_dir.exists() else 0

    if existing >= 500:
        print(f"\n  [SKIP] {existing} training images already found in dataset/train/images/")
        print("         Delete 'dataset/' and 'rf_dataset/' folders to re-download.\n")
    else:
        # 3. Get API key and download
        api_key  = get_api_key()
        
        # Force clear rf_dataset to ensure fresh download of new version
        if (BASE / "rf_dataset").exists():
            print("[INFO] Clearing old rf_dataset...")
            shutil.rmtree(BASE / "rf_dataset")
            
        rf_path  = download_dataset(api_key)

        # 4. Reorganise into our dataset/ folder structure
        reorganise_into_dataset_folder(rf_path)

    # 5. Update data.yaml
    write_data_yaml()

    # 6. Count final training images
    n_train = len(list((BASE / "dataset" / "train" / "images").glob("*.jpg")))
    n_valid = len(list((BASE / "dataset" / "valid" / "images").glob("*.jpg")))
    print(f"\n  Training images : {n_train}")
    print(f"  Validation imgs : {n_valid}")

    if n_train < 10:
        print("\n  [ERROR] Not enough training images found.")
        print("          Check that the dataset downloaded correctly.")
        sys.exit(1)

    # 7. Train
    train()

    section("ALL DONE!")
    print("  best.pt saved to project root.")
    print("  Restart the Flask app:  python app.py")
    print("  Then upload a pothole image at http://127.0.0.1:5000\n")
