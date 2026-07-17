#!/usr/bin/env python3
"""
ml_bridge.py — The Glue: connects Week 3 → ML (Path A + B) → Week 4
DNS Traffic Anomaly Detection System — Week 5

What this file does:
  1. Loads Week 3 features JSON  (week3_features_all.json)
  2. Runs Path A inference       (Model.py RF+Transformer → ml_score_A)
  3. Runs Path B inference       (Predict.py Trafficformer → ml_score_B)
  4. Merges both scores into each Week 3 record
  5. Adds combined ml_score and ml_label
  6. Saves week3_features_with_ml.json  ← input for Week 4 detection_engine.py
  7. Can also generate labels from Week 3 attack_count (for training)

Usage:
  # Full pipeline (inference mode — requires trained models)
  python Ml_bridge.py \\
    --week3  data/week3_features_all.json \\
    --output data/week3_features_with_ml.json \\
    --path-b-model dns_trafficformer

  # Generate labels only (for training mode — no models needed)
  python Ml_bridge.py \\
    --week3 data/week3_features_all.json \\
    --label-only \\
    --output data/week3_features_labeled.json

  # Skip Path B if not trained yet
  python Ml_bridge.py \\
    --week3 data/week3_features_all.json \\
    --output data/week3_features_with_ml.json \\
    --no-path-b
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ── FIX: add scripts directory to path so Python finds Model.py / Predict.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tensorflow import keras

# Import shared parser (dns_parser.py — lowercase, symlink exists)
from dns_parser import (
    load_as_dataframe, load_as_array, FEATURES_8,
    _load_json_records, week3_record_to_length
)

# ── FIX: import from Model (capital M, matches your actual filename)
# All of these functions/constants exist in Model.py — verified from source
import importlib
_model_module = importlib.import_module("Model")

build_sequences_unlabeled = _model_module.build_sequences_unlabeled
ensemble_predict          = _model_module.ensemble_predict
load_threshold            = _model_module.load_threshold
SEQUENCE_LENGTH           = _model_module.SEQUENCE_LENGTH
MODEL_RF                  = _model_module.MODEL_RF
MODEL_TF                  = _model_module.MODEL_TF
MODEL_SCALER              = _model_module.MODEL_SCALER

# ── FIX: import from Predict (capital P, matches your actual filename)
_predict_module = importlib.import_module("Predict")

load_model_artifacts         = _predict_module.load_model_artifacts
build_windows                = _predict_module.build_windows
compute_reconstruction_errors = _predict_module.compute_reconstruction_errors
normalize_scores             = _predict_module.normalize_scores


# ─────────────────────────────────────────────
# LABEL GENERATION FROM WEEK 3 ATTACK FLAGS
# ─────────────────────────────────────────────

def generate_labels_from_week3(records: list) -> list:
    """
    Create binary labels from Week 3 attack_count field.
    Week 3 already detected 6 attack patterns — we use that ground truth.

    Rules:
      attack_count >= 1  → label = 1 (attack)
      attack_count == 0  → label = 0 (benign)
    """
    ATTACK_TYPE_MAP = {
        "NORMAL":           0,
        "NXDOMAIN_FLOOD":   1,
        "RANDOM_SUBDOMAIN": 2,
        "AMPLIFICATION":    3,
        "CACHE_POISONING":  4,
        "HIGH_ENTROPY":     5,
        "ENCODING":         6,
    }

    labeled = []
    label_counts = {0: 0, 1: 0}

    for rec in records:
        r = dict(rec)

        attack_count = int(r.get("attack_count", 0))
        r["label"] = 1 if attack_count >= 1 else 0
        label_counts[r["label"]] += 1

        # Primary attack type (first in signature)
        sig = str(r.get("attack_signature", "NORMAL"))
        primary = sig.split(",")[0].strip() if sig else "NORMAL"
        r["attack_type_id"] = ATTACK_TYPE_MAP.get(primary, 0)

        labeled.append(r)

    total = len(labeled)
    print(f"   Labels generated: {label_counts[0]} benign  |  {label_counts[1]} attack")
    print(f"   Attack rate: {label_counts[1]/max(total,1)*100:.1f}%")

    return labeled


# ─────────────────────────────────────────────
# PATH A INFERENCE
# ─────────────────────────────────────────────

def run_path_a(week3_json_path: str) -> pd.DataFrame:
    """
    Run Path A (RF + Transformer) on Week 3 features JSON.
    Returns DataFrame with ml_score_A and ml_label_A columns.
    """
    print("\n── Path A: RF + Transformer (8-feature ensemble) ─────────────")

    for m in [MODEL_RF, MODEL_TF, MODEL_SCALER]:
        if not os.path.exists(m):
            raise FileNotFoundError(
                f"Model not found: {m}\n"
                f"Run first: python Model.py --train <labeled_data>"
            )

    rf_model  = joblib.load(MODEL_RF)
    tf_model  = keras.models.load_model(MODEL_TF)
    scaler    = joblib.load(MODEL_SCALER)
    threshold = load_threshold()

    print(f"   Using ensemble threshold: {threshold:.4f}")

    df = load_as_dataframe(week3_json_path)
    for col in FEATURES_8:
        if col not in df.columns:
            df[col] = 0

    X_scaled = scaler.transform(df[FEATURES_8].values)
    X_seq    = build_sequences_unlabeled(X_scaled, SEQUENCE_LENGTH)

    preds, scores = ensemble_predict(
        rf_model, tf_model, X_scaled, X_seq, threshold
    )

    pad = len(X_scaled) - len(scores)
    result_df = df.copy()
    result_df["ml_score_A"] = np.nan
    result_df["ml_label_A"] = np.nan
    result_df.iloc[pad:, result_df.columns.get_loc("ml_score_A")] = scores
    result_df.iloc[pad:, result_df.columns.get_loc("ml_label_A")] = preds

    rf_probs = rf_model.predict_proba(X_scaled)[:, 1]
    result_df.iloc[:pad, result_df.columns.get_loc("ml_score_A")] = rf_probs[:pad]
    result_df.iloc[:pad, result_df.columns.get_loc("ml_label_A")] = (rf_probs[:pad] >= threshold).astype(int)

    result_df["ml_verdict_A"] = result_df["ml_label_A"].map({1.0: "ATTACK", 0.0: "benign"})

    a_attacks = int((result_df["ml_label_A"] == 1).sum())
    print(f"   Path A results: {a_attacks} attack  |  {len(result_df)-a_attacks} benign")

    return result_df


# ─────────────────────────────────────────────
# PATH B INFERENCE
# ─────────────────────────────────────────────

def run_path_b(week3_json_path: str, save_name: str,
               model_dir: str = "model"):
    """
    Run Path B (Trafficformer) on Week 3 features JSON.
    Returns (Series ml_score_B, Series ml_label_B).
    """
    print(f"\n── Path B: Trafficformer (sequence reconstruction) ───────────")

    model, threshold, window_size, error_mean, error_std, data_min, data_range = \
        load_model_artifacts(save_name, model_dir)

    records  = _load_json_records(week3_json_path)
    lengths  = np.array([week3_record_to_length(r) for r in records], dtype=np.float32)
    n_total  = len(lengths)

    lengths_norm = (lengths - data_min) / max(data_range, 1e-9)

    X_windows = build_windows(lengths_norm, window_size)
    errors    = compute_reconstruction_errors(model, X_windows)
    scores    = normalize_scores(errors, error_mean, error_std)
    anomalies = (errors > threshold).astype(int)

    pad               = n_total - len(scores)
    padded_scores     = np.concatenate([np.zeros(pad), scores])
    padded_anomalies  = np.concatenate([np.zeros(pad, dtype=int), anomalies])

    b_attacks = int(padded_anomalies.sum())
    print(f"   Path B results: {b_attacks} anomalies  |  {n_total-b_attacks} normal")
    print(f"   Score range: {scores.min():.4f} – {scores.max():.4f}")

    return (pd.Series(padded_scores, name="ml_score_B"),
            pd.Series(padded_anomalies, name="ml_label_B"))


# ─────────────────────────────────────────────
# MERGE AND COMPUTE COMBINED SCORE
# ─────────────────────────────────────────────

def merge_ml_scores(df_path_a: pd.DataFrame,
                    score_b: pd.Series,
                    label_b: pd.Series,
                    weight_a: float = 0.6,
                    weight_b: float = 0.4) -> pd.DataFrame:
    df = df_path_a.copy()
    df["ml_score_B"] = score_b.values
    df["ml_label_B"] = label_b.values

    df["ml_combined_score"] = (
        weight_a * df["ml_score_A"].fillna(0.0) +
        weight_b * df["ml_score_B"].fillna(0.0)
    )

    df["ml_combined_label"] = (
        (df["ml_label_A"] == 1) |
        (df["ml_label_B"] == 1) |
        (df["ml_combined_score"] >= 0.5)
    ).astype(int)

    df["ml_combined_verdict"] = df["ml_combined_label"].map({
        1: "ML_ATTACK", 0: "ml_benign"
    })

    def confidence_tier(score):
        if score >= 0.8:   return "HIGH"
        elif score >= 0.5: return "MEDIUM"
        elif score >= 0.3: return "LOW"
        else:              return "CLEAN"

    df["ml_confidence"] = df["ml_combined_score"].apply(confidence_tier)

    return df


# ─────────────────────────────────────────────
# WEEK 4 COMPATIBILITY
# ─────────────────────────────────────────────

def ensure_week4_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns Week 4 detection_engine.py expects are present.
    Does NOT modify existing Week 3 columns.
    """
    week4_expected = {
        "entropy":         "query_entropy",
        "subdomain_depth": "subdomain_count",
    }
    for w4_col, ml_col in week4_expected.items():
        if w4_col not in df.columns and ml_col in df.columns:
            df[w4_col] = df[ml_col]

    if "attack_count" in df.columns:
        normalised_rule_score = df["attack_count"].clip(0, 6) / 6.0
        df["ml_threat_score"] = (
            0.4 * normalised_rule_score +
            0.35 * df.get("ml_score_A", pd.Series(0.0, index=df.index)).fillna(0) +
            0.25 * df.get("ml_score_B", pd.Series(0.0, index=df.index)).fillna(0)
        )
    else:
        df["ml_threat_score"] = df.get(
            "ml_combined_score", pd.Series(0.0, index=df.index)
        ).fillna(0)

    return df


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_bridge(week3_path: str, output_path: str,
               path_b_model: str = None,
               model_dir: str = "model",
               weight_a: float = 0.6,
               weight_b: float = 0.4,
               label_only: bool = False,
               no_path_b: bool = False):

    print("\n" + "="*70)
    print("ML BRIDGE — Connecting Week 3 → ML Models → Week 4")
    print("="*70)
    print(f"Input  : {week3_path}")
    print(f"Output : {output_path}")

    records = _load_json_records(week3_path)
    print(f"\n[*] Loaded {len(records)} Week 3 records")

    # ── LABEL GENERATION MODE
    if label_only:
        print("\n[*] LABEL ONLY MODE — generating labels from attack_count")
        labeled = generate_labels_from_week3(records)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(labeled, f, indent=2, default=str)
        print(f"\n✅ Labeled data saved → {output_path}")
        print(f"   Use for training: python Model.py --train {output_path}")
        return

    # ── INFERENCE MODE

    # Path A
    try:
        df_a = run_path_a(week3_path)
    except FileNotFoundError as e:
        print(f"\n⚠ Path A skipped: {e}")
        df_a = load_as_dataframe(week3_path)
        for col in FEATURES_8:
            if col not in df_a.columns:
                df_a[col] = 0
        df_a["ml_score_A"]   = 0.0
        df_a["ml_label_A"]   = 0
        df_a["ml_verdict_A"] = "SKIP"

    # Path B
    if not no_path_b and path_b_model:
        try:
            score_b, label_b = run_path_b(week3_path, path_b_model, model_dir)
        except FileNotFoundError as e:
            print(f"\n⚠ Path B skipped: {e}")
            score_b = pd.Series(np.zeros(len(df_a)), name="ml_score_B")
            label_b = pd.Series(np.zeros(len(df_a), dtype=int), name="ml_label_B")
    else:
        if no_path_b:
            print("\n[*] Path B skipped (--no-path-b flag)")
        else:
            print("\n[*] Path B skipped (--path-b-model not specified)")
        score_b = pd.Series(np.zeros(len(df_a)), name="ml_score_B")
        label_b = pd.Series(np.zeros(len(df_a), dtype=int), name="ml_label_B")

    # Merge
    print("\n── Merging ML scores ──────────────────────────────────────────")
    df_merged = merge_ml_scores(df_a, score_b, label_b, weight_a, weight_b)
    df_merged = ensure_week4_compatible(df_merged)

    # Preserve ALL original Week 3 columns not already in df_merged
    week3_df = pd.DataFrame(records)
    for col in week3_df.columns:
        if col not in df_merged.columns:
            if len(week3_df) == len(df_merged):
                df_merged[col] = week3_df[col].values

    # Summary
    ml_attacks = int(df_merged["ml_combined_label"].sum())
    total      = len(df_merged)
    print(f"\n── Bridge Summary ─────────────────────────────────────────────")
    print(f"   Total records      : {total}")
    print(f"   ML attacks flagged : {ml_attacks} ({ml_attacks/max(total,1)*100:.1f}%)")
    print(f"   HIGH confidence    : {(df_merged['ml_confidence']=='HIGH').sum()}")
    print(f"   MEDIUM confidence  : {(df_merged['ml_confidence']=='MEDIUM').sum()}")
    print(f"   LOW confidence     : {(df_merged['ml_confidence']=='LOW').sum()}")
    print(f"   CLEAN              : {(df_merged['ml_confidence']=='CLEAN').sum()}")

    new_cols = [
        "ml_score_A", "ml_label_A", "ml_verdict_A",
        "ml_score_B", "ml_label_B",
        "ml_combined_score", "ml_combined_label", "ml_combined_verdict",
        "ml_confidence", "ml_threat_score"
    ]
    print(f"\n   New columns added to Week 3 features:")
    for col in new_cols:
        if col in df_merged.columns:
            print(f"     + {col}")

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True
    )
    records_out = df_merged.to_dict(orient="records")
    with open(output_path, "w") as f:
        json.dump(records_out, f, indent=2, default=str)

    print(f"\n✅ Bridge output saved → {output_path}")
    print(f"   Next step: python Week4_detection_engine.py --input {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ml_bridge.py — Connects Week 3 features → ML scores → Week 4 detection",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Workflow:

  STEP 1: Generate labels from Week 3 (for training)
    python Ml_bridge.py --week3 data/week3_features_all.json \\
                        --label-only \\
                        --output data/week3_features_labeled.json

  STEP 2: Train Path A on labeled data
    python Model.py --train data/week3_features_labeled.json

  STEP 3: Train Path B
    python Train.py --filename week3_features_all.json \\
                    --n_days 20 --save_name dns_trafficformer

  STEP 4: Run bridge (inference mode) → produces input for Week 4
    python Ml_bridge.py --week3 data/week3_features_all.json \\
                        --output data/week3_features_with_ml.json \\
                        --path-b-model dns_trafficformer

  STEP 5: Run Week 4 detection engine
    python Week4_detection_engine.py \\
        --input data/week3_features_with_ml.json \\
        --output data/week4_unified_alerts.json
        """
    )
    parser.add_argument("--week3",        type=str, required=True)
    parser.add_argument("--output",       type=str, required=True)
    parser.add_argument("--path-b-model", type=str, default=None)
    parser.add_argument("--model-dir",    type=str, default="model")
    parser.add_argument("--weight-a",     type=float, default=0.6)
    parser.add_argument("--weight-b",     type=float, default=0.4)
    parser.add_argument("--label-only",   action="store_true")
    parser.add_argument("--no-path-b",    action="store_true")
    args = parser.parse_args()

    run_bridge(
        week3_path    = args.week3,
        output_path   = args.output,
        path_b_model  = args.path_b_model,
        model_dir     = args.model_dir,
        weight_a      = args.weight_a,
        weight_b      = args.weight_b,
        label_only    = args.label_only,
        no_path_b     = args.no_path_b,
    )


if __name__ == "__main__":
    main()
