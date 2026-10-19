"""Train the thread-pitch classifier from iso_thread_dataset/pitch_<value>/ folders.

Run:  python code/train_model.py
Output: models/thread_pitch_classifier.pkl  (labels are pitch in mm, stored as strings)
"""
import os
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from thread_features import extract_thread_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "iso_thread_dataset")
MODEL_PATH = os.path.join(ROOT, "models", "thread_pitch_classifier.pkl")

X, y, skipped = [], [], []
for folder in sorted(os.listdir(DATA_DIR)):
    if not folder.startswith("pitch_"):
        continue
    pitch = float(folder.split("_", 1)[1])
    fdir = os.path.join(DATA_DIR, folder)
    for fn in sorted(os.listdir(fdir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        res = extract_thread_features(cv2.imread(os.path.join(fdir, fn)))
        if res is None:
            skipped.append(f"{folder}/{fn}")
            continue
        X.append(res["features"])
        y.append(str(pitch))

X, y = np.array(X), np.array(y)
print(f"Samples used: {len(y)} | skipped (detection failed): {len(skipped)}")
for s in skipped:
    print("  skipped:", s)
print("Class counts:", {str(c): int((y == c).sum()) for c in np.unique(y)})

clf = RandomForestClassifier(n_estimators=200, random_state=42)
n_splits = min(5, min(int((y == c).sum()) for c in np.unique(y)))
if n_splits >= 2:
    scores = cross_val_score(clf, X, y, cv=n_splits)
    print(f"{n_splits}-fold CV accuracy: {scores.mean():.2f} +/- {scores.std():.2f}")
else:
    print("Too few samples per class for cross-validation.")

clf.fit(X, y)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print("Saved model to", MODEL_PATH)
