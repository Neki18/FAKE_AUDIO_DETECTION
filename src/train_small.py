# src/train_small.py
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

from preprocess import load_audio
from features import extract_features

# CONFIG - adjust paths if needed
DATA_DIR = os.path.join("..", "data")   # run from src/ with: python train_small.py
OUT_CSV = os.path.join("..", "outputs", "features_small.csv")
MODEL_DIR = os.path.join("..", "models")

os.makedirs(os.path.join("..","outputs"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def build_feature_table():
    rows = []
    labels = []
    for label in ["human", "ai"]:
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".wav", ".mp3")):
                continue
            path = os.path.join(folder, fname)
            try:
                y, sr = load_audio(path, sr=22050)
                feat = extract_features(y, sr)
                rows.append(feat)
                labels.append(label)
                print(f"Processed: {label}/{fname}")
            except Exception as e:
                print("Failed:", path, e)
    if not rows:
        print("No audio files found. Please add samples in data/human and data/ai.")
        sys.exit(1)
    X = np.vstack(rows)
    n_mfcc = 13
    cols = [f"mfcc{i+1}" for i in range(n_mfcc)] + ["centroid","rolloff","zcr","rms","echo"]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = labels
    df.to_csv(OUT_CSV, index=False)
    print("Saved features to", OUT_CSV)
    return df

def train_models(df):
    X = df.drop("label", axis=1).values.astype(float)
    y = df["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)  # human->0, ai->1
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SVM
    svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    svm.fit(X_train_s, y_train)
    y_pred = svm.predict(X_test_s)
    print("=== SVM Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)
    print("=== Random Forest Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

    # save scaler, models, encoder
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_small.joblib"))
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm_small.joblib"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_small.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder_small.joblib"))
    print("Saved models and scaler to", MODEL_DIR)

if __name__ == "__main__":
    df = build_feature_table()
    train_models(df)
