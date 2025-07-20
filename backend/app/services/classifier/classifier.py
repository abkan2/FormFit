# app/services/classifier/classifier.py

import joblib
import os
from pathlib import Path

# 1) Locate the models directory
MODEL_DIR = Path(__file__).parents[2] / "models"

# 2) Find all timestamped classifier files
cands = list(MODEL_DIR.glob("movement_classifier_*.pkl"))
if not cands:
    raise FileNotFoundError(f"No model files found in {MODEL_DIR!r}")

# 3) Pick the one with the most recent modification time
latest_model = max(cands, key=lambda p: p.stat().st_mtime)

# 4) Load it
clf = joblib.load(latest_model)
print(f"🔄 Loaded classifier: {latest_model.name}")

def predict_from_landmarks(landmarks):
    vec = __import__('numpy').array(landmarks).flatten().reshape(1, -1)
    return clf.predict(vec)[0]

def predict_proba(landmarks):
    vec = __import__('numpy').array(landmarks).flatten().reshape(1, -1)
    return clf.predict_proba(vec)[0]
