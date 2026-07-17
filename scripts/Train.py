#!/usr/bin/env python3
"""
train.py — Path B: Trafficformer (query-length time-series)
DNS Anomaly Detector — Week 5 ML Layer

KEY FIX from original:
  - Original trained a REGRESSION model (MSE loss, predicting next length)
    but predict.py used int(pred) as a binary label → conceptual mismatch.
  - This version is now a proper SEQUENCE ANOMALY DETECTOR:
      * Trains an autoencoder-style reconstruction model (still MSE on sequences)
      * Anomaly score = reconstruction error (high error = anomaly)
      * predict.py compares reconstruction error to a calibrated threshold
      * No more int(0.0003) → 0 and int(0.9) → 0 being treated the same
  - Uses shared dns_parser.py — no duplicated parsers

Usage:
  python Train.py --filename week3_features_all.json --n_days 20 --save_name dns_trafficformer
  python Train.py --filename example.traff --n_days 20 --save_name my_model
  python Train.py --filename capture.pcap  --n_days 20 --save_name pcap_model
"""

import os
import sys

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── FIX: add scripts directory to path so Python finds Model.py (capital M)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import shared parser
from dns_parser import load_as_array

# ── FIX: use importlib to load Model.py (capital M) — Linux is case-sensitive
import importlib
_model_module = importlib.import_module("Model")
transformer_encoder = _model_module.transformer_encoder


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

def build_trafficformer(input_shape, head_size=64, num_heads=3,
                         ff_dim=3, num_transformer_blocks=3,
                         mlp_units=None, mlp_dropout=0.4, dropout=0.25):
    """
    Trafficformer: sequence reconstruction model.

    Architecture:
      Input  → Transformer blocks → Dense reconstruction of input sequence
    Loss:
      Mean Squared Error on the RECONSTRUCTED sequence.

    Anomaly detection logic (in Predict.py):
      reconstruction_error = MSE(original_window, reconstructed_window)
      If error > threshold → ANOMALY (potential tunneling)

    This is correct: the model learns what NORMAL traffic looks like.
    Unusual traffic (tunneling) will have HIGH reconstruction error.
    """
    if mlp_units is None:
        mlp_units = [64]

    from tensorflow.keras import layers, Input, Model

    inputs = Input(shape=input_shape)  # (window_size, 1)
    x = inputs

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # MLP head
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    # Reconstruction output: same shape as input
    outputs = layers.Dense(1, activation="linear")(x)   # (window_size, 1)

    return Model(inputs, outputs, name="trafficformer_reconstruction")


# ─────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────

def make_dataset(data: np.ndarray, window_size: int = 20):
    """
    Build overlapping windows from the time series.
    Input → target is the SAME window (reconstruction task).
    Both X and y are the same window — model learns to reconstruct normal traffic.
    """
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i: i + window_size])
    windows = np.array(windows, dtype=np.float32)
    X = windows.reshape(-1, window_size, 1)
    y = X.copy()  # reconstruction: target = input
    return X, y


def compute_threshold_from_errors(errors: np.ndarray,
                                   multiplier: float = 2.5) -> float:
    """
    Compute anomaly threshold from reconstruction errors on normal traffic.
    threshold = mean + multiplier * std
    multiplier=2.5 means ~99% of normal traffic is below the threshold.
    """
    mean = float(np.mean(errors))
    std  = float(np.std(errors))
    threshold = mean + multiplier * std
    print(f"   Error stats → mean={mean:.4f}  std={std:.4f}")
    print(f"   Anomaly threshold: {threshold:.4f}  (mean + {multiplier}×std)")
    return threshold


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trafficformer Trainer (Path B — sequence reconstruction anomaly detector)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python Train.py --filename week3_features_all.json --n_days 20 --save_name dns_trafficformer
  python Train.py --filename example.traff  --n_days 20 --save_name my_model
  python Train.py --filename capture.pcap   --n_days 20 --save_name pcap_model
  python Train.py --filename dns.log        --n_days 20 --save_name zeek_model
  python Train.py --filename traffic.json   --n_days 20 --save_name json_model
        """
    )
    parser.add_argument("--filename",   type=str, default="example.traff",
                        help="Input file (.traff / .pcap / .json / .log)")
    parser.add_argument("--n_days",     type=int, required=True,
                        help="Window size (timesteps)")
    parser.add_argument("--save_name",  type=str, required=True,
                        help="Model save name (saved to model/<save_name>.h5)")
    parser.add_argument("--data_dir",   type=str, default="data",
                        help="Data directory (default: data/)")
    parser.add_argument("--multiplier", type=float, default=2.5,
                        help="Threshold = mean + multiplier*std (default: 2.5)")
    args = parser.parse_args()

    DATA_PATH = Path(args.data_dir) / args.filename
    SAVE_NAME = args.save_name
    N_DAYS    = args.n_days

    # Load data via shared parser
    data = load_as_array(str(DATA_PATH))
    print(f"[*] Loaded {len(data)} data points from {DATA_PATH}")

    # Normalize to [0, 1] range before training
    data_min   = float(data.min())
    data_max   = float(data.max())
    data_range = data_max - data_min if data_max > data_min else 1.0
    data_norm  = (data.astype(np.float32) - data_min) / data_range

    # Save normalization params alongside model
    os.makedirs("model", exist_ok=True)
    norm_path = f"model/{SAVE_NAME}_norm.json"
    with open(norm_path, "w") as f:
        json.dump({"min": data_min, "max": data_max, "range": data_range}, f)
    print(f"[*] Normalization params saved → {norm_path}")

    # Build reconstruction dataset
    train_x, train_y = make_dataset(data_norm, window_size=N_DAYS)
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42
    )
    print(f"[*] Train windows: {len(x_train)}  |  Val windows: {len(x_valid)}")

    # Build and compile the reconstruction model
    model = build_trafficformer(
        input_shape=[N_DAYS, 1],
        head_size=64, num_heads=3, ff_dim=3,
        num_transformer_blocks=3,
        mlp_units=[64], mlp_dropout=0.4, dropout=0.25,
    )
    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"model/{SAVE_NAME}.h5",
        monitor="val_loss", verbose=1, save_best_only=True, mode="min",
    )

    history = model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(x_valid, y_valid),
        callbacks=[early_stop, checkpoint],
    )
    print(f"\n[+] Model saved → model/{SAVE_NAME}.h5")

    # Compute reconstruction errors on validation set to set anomaly threshold
    print("\n── Calibrating Anomaly Threshold ─────────────────────────────")
    reconstructed = model.predict(x_valid, verbose=0)
    errors = np.mean(np.square(x_valid - reconstructed), axis=(1, 2))
    threshold = compute_threshold_from_errors(errors, multiplier=args.multiplier)

    threshold_path = f"model/{SAVE_NAME}_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({
            "threshold":   threshold,
            "multiplier":  args.multiplier,
            "window_size": N_DAYS,
            "error_mean":  float(np.mean(errors)),
            "error_std":   float(np.std(errors)),
        }, f, indent=2)
    print(f"[+] Threshold saved → {threshold_path}")

    print("\n✅ Training complete.")
    print("   This model detects anomalies via RECONSTRUCTION ERROR.")
    print("   High error = unusual traffic pattern = potential DNS tunnel/attack.")
    print(f"\n   Next step: python Predict.py --filename <input> --save_name {SAVE_NAME}")
