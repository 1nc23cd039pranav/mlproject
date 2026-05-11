"""
train_ml.py
-----------
Trains a RandomForestClassifier to classify pothole severity
based on extracted bounding-box features (width, height, area, confidence).

Output: severity_model.pkl  (saved in project root)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "pothole_dataset.csv")
df = pd.read_csv(CSV_PATH)

print("=" * 50)
print("POTHOLE SEVERITY CLASSIFICATION TRAINING")
print("=" * 50)
print(f"\nDataset loaded: {len(df)} samples")
print(df.head())
print("\nClass distribution:")
print(df["severity"].value_counts())

# ──────────────────────────────────────────────
# 2. Feature / Label split
# ──────────────────────────────────────────────
FEATURES = ["width", "height", "area", "confidence"]
X = df[FEATURES].values
y = df["severity"].values

# Encode labels  Low->0  Medium->1  High->2
le = LabelEncoder()
le.fit(["Low", "Medium", "High"])   # fixed order
y_enc = le.transform(y)

# ──────────────────────────────────────────────
# 3. Train / Test split
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ──────────────────────────────────────────────
# 4. Train RandomForestClassifier
# ──────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)
print("\n[INFO] RandomForestClassifier trained successfully.")

# ──────────────────────────────────────────────
# 5. Evaluate
# ──────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y_enc, cv=5)

print(f"\nTest Accuracy    : {acc * 100:.2f}%")
print(f"Cross-Val Scores : {cv_scores}")
print(f"Mean CV Accuracy : {cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ──────────────────────────────────────────────
# 6. Feature importances
# ──────────────────────────────────────────────
importances = model.feature_importances_
fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values(
    "Importance", ascending=False
)
print("\nFeature Importances:")
print(fi_df.to_string(index=False))

# ──────────────────────────────────────────────
# 7. Plot confusion matrix & feature importance
# ──────────────────────────────────────────────
os.makedirs("static/results", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

# Feature importance bar chart
axes[1].barh(fi_df["Feature"], fi_df["Importance"], color="steelblue")
axes[1].set_xlabel("Importance Score")
axes[1].set_title("Feature Importances - RandomForest")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("static/results/ml_evaluation.png", dpi=150)
plt.close()
print("\n[INFO] Evaluation plots saved -> static/results/ml_evaluation.png")

# ──────────────────────────────────────────────
# 8. Save model + encoder
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "severity_model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "label_encoder": le}, f)

print(f"\n[SUCCESS] Model saved -> {MODEL_PATH}")
print("=" * 50)
