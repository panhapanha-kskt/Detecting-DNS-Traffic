#!/usr/bin/env python3
"""
model.py — Path A: RF + Transformer Ensemble (8-feature classification)
DNS Tunnel Detector — Week 5 ML Layer

Fixes from original:
  - Uses shared dns_parser.py (no duplicated parsers)
  - Transformer is a TRUE BINARY CLASSIFIER (sigmoid + binary_crossentropy)
  - Stratified train/test split to ensure attack samples in both sets
  - Threshold calibration via precision-recall curve
  - Scaler saved/loaded alongside models
  - Inference returns week3-compatible enriched JSON for ml_bridge.py

Usage:
  python model.py --train week3_features_all.json
  python model.py --infer week3_features_all.json
  python model.py --train dns_labeled.csv
  python model.py --infer capture.pcap --rf-weight 0.4 --tf-weight 0.6
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_recall_curve, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Import shared parser
from dns_parser import load_as_dataframe, FEATURES_8

SEQUENCE_LENGTH = 10
MODEL_RF   = "dns_tunnel_rf.pkl"
MODEL_TF   = "dns_tunnel_transformer.keras"
MODEL_SCALER = "dns_scaler.pkl"
MODEL_THRESHOLD = "dns_threshold.json"


# ─────────────────────────────────────────────
# TRANSFORMER ARCHITECTURE (binary classifier)
# ─────────────────────────────────────────────

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_transformer(input_shape, head_size=32, num_heads=2, ff_dim=64,
                      num_transformer_blocks=2, mlp_units=None,
                      dropout=0.1, mlp_dropout=0.1):
    """
    Binary classification transformer.
    Output: sigmoid → probability of being tunneling/attack traffic.
    """
    if mlp_units is None:
        mlp_units = [64]

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    # sigmoid for binary classification (NOT regression)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs), np.array(ys)


def build_sequences_unlabeled(X: np.ndarray, seq_len: int) -> np.ndarray:
    return np.array([X[i: i + seq_len] for i in range(len(X) - seq_len + 1)])


# ─────────────────────────────────────────────
# THRESHOLD CALIBRATION
# ─────────────────────────────────────────────

def calibrate_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Find optimal classification threshold using F1 from precision-recall curve.
    Returns the threshold that maximises F1 on the validation set.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # F1 = 2 * P * R / (P + R), avoid divide-by-zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0
    )
    best_idx = np.argmax(f1_scores[:-1])  # thresholds is 1 shorter
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    print(f"   Calibrated threshold: {best_threshold:.4f}  (F1={best_f1:.4f})")
    return best_threshold


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n── Random Forest ──────────────────────────────────────────────")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",   # handles imbalanced attack/normal ratio
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(classification_report(y_test, rf.predict(X_test),
                                 target_names=["benign", "attack"]))
    joblib.dump(rf, MODEL_RF)
    print(f"Saved → {MODEL_RF}")
    return rf


def train_transformer(X_seq_train, y_seq_train, X_seq_test, y_seq_test):
    print("\n── Transformer (Binary Classifier) ───────────────────────────")
    model = build_transformer(
        input_shape=(SEQUENCE_LENGTH, X_seq_train.shape[2]),
        head_size=32, num_heads=2, ff_dim=64,
        num_transformer_blocks=2, mlp_units=[64],
        dropout=0.1, mlp_dropout=0.1,
    )
    # Binary cross-entropy — correct loss for classification
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    model.summary()

    model.fit(
        X_seq_train, y_seq_train,
        validation_data=(X_seq_test, y_seq_test),
        epochs=30,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor="val_auc", mode="max"
            )
        ],
        verbose=1
    )
    loss, acc, auc = model.evaluate(X_seq_test, y_seq_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}  |  Test AUC: {auc:.4f}")
    model.save(MODEL_TF)
    print(f"Saved → {MODEL_TF}")
    return model


# ─────────────────────────────────────────────
# ENSEMBLE INFERENCE
# ─────────────────────────────────────────────

def ensemble_predict(rf_model, tf_model, X_flat, X_seq,
                     threshold=0.5, rf_weight=0.5, tf_weight=0.5):
    """
    Weighted ensemble of RF + Transformer probabilities.
    Both models output P(attack). Combined score >= threshold → attack.
    """
    rf_probs = rf_model.predict_proba(X_flat)[:, 1]           # shape: (N,)
    tf_probs = tf_model.predict(X_seq, verbose=0).flatten()   # shape: (N - seq_len + 1,)

    # Align: RF has N scores, TF has N-pad scores (sequence windowing lag)
    pad = len(rf_probs) - len(tf_probs)
    # For the first `pad` samples: use only RF score
    combined = np.concatenate([
        rf_probs[:pad],
        rf_weight * rf_probs[pad:] + tf_weight * tf_probs
    ])

    predictions = (combined >= threshold).astype(int)
    return predictions, combined


def load_threshold() -> float:
    """Load calibrated threshold from disk, default 0.5 if not found."""
    if os.path.exists(MODEL_THRESHOLD):
        with open(MODEL_THRESHOLD) as f:
            data = json.load(f)
        return float(data.get("threshold", 0.5))
    return 0.5


# ─────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ─────────────────────────────────────────────

def run_training(input_path: str, rf_weight=0.5, tf_weight=0.5):
    print(f"\n[*] Loading training data: {input_path}")
    df = load_as_dataframe(input_path)

    # Fill any missing feature columns
    for col in FEATURES_8:
        if col not in df.columns:
            df[col] = 0

    if "label" not in df.columns:
        raise ValueError(
            "Training data must have a 'label' column (0=benign, 1=attack). "
            "For Week 3 data, run ml_bridge.py first to generate labels from attack_count."
        )

    print(f"[*] Dataset: {len(df)} records  |  "
          f"Benign: {(df['label']==0).sum()}  |  "
          f"Attack: {(df['label']==1).sum()}")

    X = df[FEATURES_8].values
    y = df["label"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODEL_SCALER)
    print(f"Saved → {MODEL_SCALER}")

    # Stratified split — ensures both classes appear in train AND test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Build sequences for transformer
    X_seq_all, y_seq_all = build_sequences(X_scaled, y, SEQUENCE_LENGTH)
    X_seq_tr, X_seq_te, y_seq_tr, y_seq_te = train_test_split(
        X_seq_all, y_seq_all, test_size=0.3, random_state=42, stratify=y_seq_all
    )

    # Train both models
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    tf_model = train_transformer(X_seq_tr, y_seq_tr, X_seq_te, y_seq_te)

    # Calibrate ensemble threshold on test set
    print("\n── Calibrating Ensemble Threshold ────────────────────────────")
    X_seq_eval, y_seq_eval = build_sequences(X_test, y_test, SEQUENCE_LENGTH)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    tf_probs = tf_model.predict(X_seq_eval, verbose=0).flatten()
    pad = len(rf_probs) - len(tf_probs)
    combined = np.concatenate([
        rf_probs[:pad],
        rf_weight * rf_probs[pad:] + tf_weight * tf_probs
    ])
    threshold = calibrate_threshold(y_test, combined)

    with open(MODEL_THRESHOLD, "w") as f:
        json.dump({"threshold": threshold, "rf_weight": rf_weight, "tf_weight": tf_weight}, f)
    print(f"Saved → {MODEL_THRESHOLD}")

    # Final ensemble evaluation
    print("\n── Final Ensemble Evaluation ─────────────────────────────────")
    preds = (combined >= threshold).astype(int)
    min_len = min(len(y_test), len(preds))
    print(classification_report(
        y_test[:min_len], preds[:min_len],
        target_names=["benign", "attack"]
    ))
    try:
        auc = roc_auc_score(y_test[:min_len], combined[:min_len])
        print(f"Ensemble ROC-AUC: {auc:.4f}")
    except Exception:
        pass

    print("\n✅ Training complete. Artifacts saved:")
    print(f"   {MODEL_RF}  |  {MODEL_TF}  |  {MODEL_SCALER}  |  {MODEL_THRESHOLD}")


# ─────────────────────────────────────────────
# INFERENCE PIPELINE
# ─────────────────────────────────────────────

def run_inference(input_path: str, rf_weight=0.5, tf_weight=0.5,
                  output_json: str = None):
    """
    Load any supported file, run ensemble inference.
    Saves results to CSV and optionally enriched JSON (for ml_bridge.py).
    """
    print(f"[*] Loading models...")
    rf_model = joblib.load(MODEL_RF)
    tf_model  = keras.models.load_model(MODEL_TF)
    scaler    = joblib.load(MODEL_SCALER)
    threshold = load_threshold()
    print(f"[*] Using ensemble threshold: {threshold:.4f}")

    df = load_as_dataframe(input_path)
    for col in FEATURES_8:
        if col not in df.columns:
            df[col] = 0

    X_scaled = scaler.transform(df[FEATURES_8].values)
    X_seq    = build_sequences_unlabeled(X_scaled, SEQUENCE_LENGTH)

    preds, scores = ensemble_predict(
        rf_model, tf_model, X_scaled, X_seq, threshold, rf_weight, tf_weight
    )

    pad       = len(X_scaled) - len(scores)
    result_df = df.iloc[pad:].copy().reset_index(drop=True)
    result_df["ml_score_A"]   = scores[pad:]
    result_df["ml_label_A"]   = preds[pad:]
    result_df["ml_verdict_A"] = result_df["ml_label_A"].map({1: "ATTACK", 0: "benign"})

    # Save CSV
    stem     = Path(input_path).stem
    out_csv  = f"{stem}_model_results.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\n[+] CSV saved → {out_csv}")
    print(f"    Total analyzed : {len(preds)}")
    print(f"    Attack detected: {preds.sum()}")
    print(f"    Benign         : {(preds == 0).sum()}")

    # Save enriched JSON for ml_bridge.py consumption
    if output_json:
        records = result_df.to_dict(orient="records")
        with open(output_json, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"[+] Enriched JSON saved → {output_json}")

    return result_df


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DNS Tunnel Detector — Path A: RF + Transformer Ensemble (8 features)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Train on Week 3 labeled JSON (after running ml_bridge.py --label-only)
  python model.py --train week3_features_labeled.json

  # Train on classic labeled CSV
  python model.py --train dns_labeled.csv

  # Infer on Week 3 features
  python model.py --infer week3_features_all.json --output-json week3_with_ml_A.json

  # Infer on PCAP
  python model.py --infer capture.pcap

  # Infer with custom ensemble weights
  python model.py --infer traffic.json --rf-weight 0.4 --tf-weight 0.6
        """
    )
    parser.add_argument("--train",      metavar="FILE",  help="Labeled file to train both models")
    parser.add_argument("--infer",      metavar="FILE",  help="File to run inference on")
    parser.add_argument("--output-json",metavar="FILE",  help="Save inference results as JSON (for ml_bridge.py)")
    parser.add_argument("--rf-weight",  type=float, default=0.5)
    parser.add_argument("--tf-weight",  type=float, default=0.5)
    args = parser.parse_args()

    if args.train:
        run_training(args.train, args.rf_weight, args.tf_weight)

    if args.infer:
        run_inference(args.infer, args.rf_weight, args.tf_weight, args.output_json)

    if not args.train and not args.infer:
        parser.print_help()


if __name__ == "__main__":
    main()
