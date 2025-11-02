import sys, os
import joblib
import numpy as np
from preprocess import load_audio
from features import extract_features

MODEL_DIR = os.path.join("..","models")

def predict_file(path, model="svm"):
    y, sr = load_audio(path, sr=22050)
    feat = extract_features(y, sr).reshape(1, -1)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_small.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder_small.joblib"))
    feat_s = scaler.transform(feat)
    if model == "svm":
        clf = joblib.load(os.path.join(MODEL_DIR, "svm_small.joblib"))
    else:
        clf = joblib.load(os.path.join(MODEL_DIR, "rf_small.joblib"))
    pred = clf.predict(feat_s)
    label = le.inverse_transform(pred)[0]
    proba = clf.predict_proba(feat_s)[0].max() if hasattr(clf, "predict_proba") else None
    return label, proba

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_demo.py path/to/audio.wav")
        sys.exit(1)
    path = sys.argv[1]
    lab, prob = predict_file(path, model="svm")
    print("Prediction:", lab, "  Confidence:", prob)
