#!/usr/bin/env python3
"""
predict.py — Path B: Trafficformer Inference (query-length time-series)
DNS Anomaly Detector — Week 5 ML Layer

KEY FIX from original:
  - Original: int(pred) where pred is a regression score → wrong
    (int(0.00031) = 0 and int(0.8) = 0 are both "benign" — that's useless)
  - This version: reconstruction_error vs. calibrated threshold
    High reconstruction error → traffic pattern doesn't match learned normal → ANOMALY
  - Output now includes: anomaly_score, is_anomaly, ml_score_B (0.0–1.0 normalized)
  - ml_score_B is normalized so ml_bridge.py can combine it with ml_score_A

Usage:
  python predict.py --filename week3_features_all.json --save_name dns_trafficformer
  python predict.py --filename predict.traff --save_name my_model
  python predict.py --filename capture.pcap  --save_name pcap_model
"""

import os
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Import shared parser
from dns_parser import load_as_array


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_model_artifacts(save_name: str, model_dir: str = "model"):
    """Load model + threshold + normalization params."""
    model_path     = os.path.join(model_dir, f"{save_name}.keras")
    threshold_path = os.path.join(model_dir, f"{save_name}_threshold.json")
    norm_path      = os.path.join(model_dir, f"{save_name}_norm.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[*] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load calibrated threshold
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold_data = json.load(f)
        threshold   = float(threshold_data["threshold"])
        window_size = int(threshold_data.get("window_size", 20))
        error_mean  = float(threshold_data.get("error_mean", 0.0))
        error_std   = float(threshold_data.get("error_std", 1.0))
        print(f"[*] Loaded anomaly threshold: {threshold:.6f}")
    else:
        print("[!] No threshold file found — using fallback threshold=0.01")
        threshold   = 0.01
        window_size = 20
        error_mean  = 0.0
        error_std   = 1.0

    # Load normalization params
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm_data = json.load(f)
        data_min   = float(norm_data["min"])
        data_range = float(norm_data["range"])
    else:
        print("[!] No normalization file found — using raw values (may reduce accuracy)")
        data_min   = 0.0
        data_range = 1.0

    return model, threshold, window_size, error_mean, error_std, data_min, data_range


def build_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """Build overlapping windows from 1-D array → shape (N, window_size, 1)."""
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i: i + window_size])
    return np.array(windows, dtype=np.float32).reshape(-1, window_size, 1)


def compute_reconstruction_errors(model, X_windows: np.ndarray) -> np.ndarray:
    """
    Run model reconstruction and compute per-window MSE error.
    Shape: (N,) — one error score per window.
    """
    reconstructed = model.predict(X_windows, verbose=0)
    errors = np.mean(np.square(X_windows - reconstructed), axis=(1, 2))
    return errors


def normalize_scores(errors: np.ndarray, error_mean: float,
                     error_std: float) -> np.ndarray:
    """
    Normalize reconstruction errors to [0, 1] range using Z-score sigmoid.
    This gives a probability-like ml_score_B for ml_bridge.py to use.
    """
    if error_std == 0:
        return np.zeros_like(errors)
    z = (errors - error_mean) / error_std
    # Sigmoid maps z-score to (0, 1) — higher error = higher score
    return 1.0 / (1.0 + np.exp(-z))


# ─────────────────────────────────────────────
# REUSABLE PREDICTION FUNCTION
# ─────────────────────────────────────────────

def get_predict(DATA_PATH: str, SAVE_NAME: str,
                data_dir: str = "data", model_dir: str = "model") -> dict:
    """
    Reusable prediction function for use by ml_bridge.py.
    Returns a dict with summary stats and per-window results.
    """
    full_path = (
        Path(data_dir) / DATA_PATH
        if not Path(DATA_PATH).is_absolute()
        else Path(DATA_PATH)
    )

    # Load model artifacts ONCE (caller should cache if calling in a loop)
    model, threshold, window_size, error_mean, error_std, data_min, data_range = \
        load_model_artifacts(SAVE_NAME, model_dir)

    # Load and normalize data
    data = load_as_array(str(full_path))
    data_norm = (data.astype(np.float32) - data_min) / max(data_range, 1e-9)

    # Build windows and compute errors
    X_windows = build_windows(data_norm, window_size)
    errors    = compute_reconstruction_errors(model, X_windows)
    scores    = normalize_scores(errors, error_mean, error_std)
    anomalies = (errors > threshold).astype(int)

    return {
        "num_windows":     len(errors),
        "num_anomalies":   int(anomalies.sum()),
        "anomaly_rate":    float(anomalies.mean()),
        "mean_error":      float(errors.mean()),
        "max_error":       float(errors.max()),
        "threshold":       threshold,
        "errors":          errors.tolist(),
        "scores":          scores.tolist(),
        "anomalies":       anomalies.tolist(),
    }


# ─────────────────────────────────────────────
# CLI MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trafficformer Predictor (Path B — reconstruction anomaly detection)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python predict.py --filename week3_features_all.json --save_name dns_trafficformer
  python predict.py --filename predict.traff --save_name my_model
  python predict.py --filename capture.pcap  --save_name pcap_model
  python predict.py --filename dns.log       --save_name zeek_model
  python predict.py --filename traffic.json  --save_name json_model
        """
    )
    parser.add_argument("--filename",  type=str, default="predict.traff",
                        help="Input file (.traff / .pcap / .json / .log)")
    parser.add_argument("--save_name", type=str, required=True,
                        help="Model name (loaded from model/<save_name>.h5)")
    parser.add_argument("--data_dir",  type=str, default="data",
                        help="Data directory (default: data/)")
    parser.add_argument("--model_dir", type=str, default="model",
                        help="Model directory (default: model/)")
    parser.add_argument("--output",    type=str, default=None,
                        help="Save per-window results to this CSV file")
    args = parser.parse_args()

    # Load model artifacts
    model, threshold, window_size, error_mean, error_std, data_min, data_range = \
        load_model_artifacts(args.save_name, args.model_dir)

    # Load data
    full_path = Path(args.data_dir) / args.filename
    data      = load_as_array(str(full_path))
    data_norm = (data.astype(np.float32) - data_min) / max(data_range, 1e-9)

    print(f"[*] Loaded {len(data)} data points")
    print(f"[*] Building windows of size {window_size}...")

    X_windows = build_windows(data_norm, window_size)
    print(f"[*] Input shape: {X_windows.shape}")

    # Compute reconstruction errors
    errors  = compute_reconstruction_errors(model, X_windows)
    scores  = normalize_scores(errors, error_mean, error_std)
    anomaly_flags = (errors > threshold).astype(int)

    # Results
    print("\n=======MODEL OUTPUT (Path B — Trafficformer)=================")
    print(f"File analyzed     : {args.filename}")
    print(f"Total windows     : {len(errors)}")
    print(f"Anomaly threshold : {threshold:.6f}")
    print(f"Anomalies detected: {anomaly_flags.sum()}  ({anomaly_flags.mean()*100:.1f}%)")
    print(f"Mean recon. error : {errors.mean():.6f}")
    print(f"Max  recon. error : {errors.max():.6f}")
    print(f"ml_score_B range  : {scores.min():.4f} – {scores.max():.4f}")
    print("=============================================================\n")

    # Save per-window results if requested
    if args.output:
        pad = len(data) - len(errors)
        result_df = pd.DataFrame({
            "window_start":   range(pad, len(data)),
            "query_length":   data[pad:],
            "recon_error":    errors,
            "ml_score_B":     scores,
            "is_anomaly":     anomaly_flags,
            "verdict_B":      ["ANOMALY" if a else "normal" for a in anomaly_flags],
        })
        result_df.to_csv(args.output, index=False)
        print(f"[+] Per-window results saved → {args.output}")
