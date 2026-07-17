#!/usr/bin/env python3
"""
dns_parser.py — Shared DNS input parser
Used by: model.py, train.py, predict.py, ml_bridge.py

Supports: .pcap / .pcapng / .cap / .json / .jsonl / .ndjson / .log / .csv / .traff
Single source of truth — no duplication across files.
"""

import json
import math
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# Optional scapy
try:
    from scapy.all import rdpcap, DNS, DNSQR
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ─────────────────────────────────────────────
# FEATURE CONSTANTS (shared across all modules)
# ─────────────────────────────────────────────

# 8-feature vector used by model.py (Path A: RF + Transformer)
FEATURES_8 = [
    "query_length",
    "query_entropy",
    "has_digits",
    "answer_count",
    "subdomain_count",
    "longest_label",
    "digit_ratio",
    "unique_char_ratio",
]

# Week 3 feature names that map to FEATURES_8 (for bridge alignment)
WEEK3_TO_MODEL_MAP = {
    "query_length":    "query_length",      # direct match
    "entropy":         "query_entropy",     # week3 calls it 'entropy'
    "numeric_ratio":   "digit_ratio",       # week3 calls it 'numeric_ratio'
    "subdomain_depth": "subdomain_count",   # week3 calls it 'subdomain_depth'
    "answer_size":     "answer_count",      # week3 calls it 'answer_size'
    "first_label_length": "longest_label",  # approximation
}


# ─────────────────────────────────────────────
# FEATURE HELPERS
# ─────────────────────────────────────────────

def compute_entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_features_from_record(record: dict) -> dict:
    """
    Extract 8-feature vector from a DNS record dict.
    Flexible key handling — works for PCAP-derived, JSON, and Week 3 records.
    """
    # Resolve domain name from various key names
    query = str(record.get(
        "query", record.get(
        "qname", record.get(
        "query_name", record.get(
        "domain", "")))))
    query = query.rstrip(".")

    # Resolve answer count from various formats
    answers = record.get("answers", record.get("answer", record.get("answer_count", record.get("answer_size", 0))))
    if isinstance(answers, list):
        answer_count = len(answers)
    elif isinstance(answers, (int, float)):
        answer_count = int(answers)
    else:
        answer_count = 0

    labels = [p for p in query.split(".") if p]

    features = {
        "query_length":      len(query),
        "query_entropy":     round(compute_entropy(query), 6),
        "has_digits":        int(any(c.isdigit() for c in query)),
        "answer_count":      answer_count,
        "subdomain_count":   query.count("."),
        "longest_label":     max((len(p) for p in labels), default=0),
        "digit_ratio":       round(sum(c.isdigit() for c in query) / max(len(query), 1), 6),
        "unique_char_ratio": round(len(set(query)) / max(len(query), 1), 6),
    }

    if "label" in record:
        features["label"] = int(record["label"])

    return features


def dns_record_to_length(record: dict) -> int:
    """
    Convert a DNS record to a single integer (query length).
    Used by Path B (train.py / predict.py) for time-series input.
    """
    query = str(record.get(
        "query", record.get(
        "qname", record.get(
        "query_name", record.get(
        "domain", "")))))
    return len(query.rstrip("."))


def week3_record_to_length(record: dict) -> int:
    """
    Convert a Week 3 feature record to an integer length signal.
    Week 3 already computed query_length — reuse it directly.
    """
    return int(record.get("query_length", record.get("query_length", 0)))


# ─────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────

def parse_pcap_to_df(path: str) -> pd.DataFrame:
    """Parse PCAP → DataFrame of 8-feature vectors."""
    if not SCAPY_AVAILABLE:
        raise RuntimeError("scapy required for PCAP. Run: pip install scapy")
    print(f"[*] Reading PCAP: {path}")
    packets = rdpcap(path)
    records = []
    for pkt in packets:
        if pkt.haslayer(DNS) and pkt.haslayer(DNSQR):
            qname = pkt[DNSQR].qname
            qname = qname.decode(errors="ignore") if isinstance(qname, bytes) else str(qname)
            records.append({"query": qname, "answer_count": getattr(pkt[DNS], "ancount", 0)})
    print(f"[*] Extracted {len(records)} DNS queries from PCAP.")
    return pd.DataFrame([extract_features_from_record(r) for r in records])


def parse_pcap_to_array(path: str) -> np.ndarray:
    """Parse PCAP → 1-D integer array of query lengths (Path B signal)."""
    if not SCAPY_AVAILABLE:
        raise RuntimeError("scapy required for PCAP. Run: pip install scapy")
    print(f"[*] Reading PCAP: {path}")
    packets = rdpcap(path)
    data = []
    for pkt in packets:
        if pkt.haslayer(DNS) and pkt.haslayer(DNSQR):
            qname = pkt[DNSQR].qname
            qname = qname.decode(errors="ignore") if isinstance(qname, bytes) else str(qname)
            data.append(len(qname.rstrip(".")))
    print(f"[*] Extracted {len(data)} DNS queries from PCAP.")
    return np.array(data)


def _load_json_records(path: str) -> list:
    """Internal helper: parse JSON / JSON Lines → list of record dicts."""
    with open(path, "r") as f:
        content = f.read().strip()
    records = []
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            records = parsed.get("records", parsed.get("dns", parsed.get("data", [parsed])))
    except json.JSONDecodeError:
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def parse_json_to_df(path: str) -> pd.DataFrame:
    """Parse JSON/JSONL → DataFrame of 8-feature vectors."""
    print(f"[*] Reading JSON: {path}")
    records = _load_json_records(path)
    print(f"[*] Parsed {len(records)} DNS records from JSON.")
    return pd.DataFrame([extract_features_from_record(r) for r in records])


def parse_json_to_array(path: str) -> np.ndarray:
    """Parse JSON/JSONL → 1-D integer array of query lengths (Path B)."""
    print(f"[*] Reading JSON: {path}")
    records = _load_json_records(path)
    print(f"[*] Parsed {len(records)} DNS records from JSON.")
    return np.array([dns_record_to_length(r) for r in records])


def parse_week3_json_to_df(path: str) -> pd.DataFrame:
    """
    Parse Week 3 features JSON → DataFrame aligned to FEATURES_8.
    Week 3 already extracted features — this remaps column names so
    model.py can consume them directly without re-extracting from raw logs.
    """
    print(f"[*] Reading Week 3 features JSON: {path}")
    records = _load_json_records(path)
    print(f"[*] Loaded {len(records)} Week 3 feature records.")

    df = pd.DataFrame(records)

    # Remap Week 3 column names → model feature names
    rename_map = {
        "entropy":         "query_entropy",
        "numeric_ratio":   "digit_ratio",
        "subdomain_depth": "subdomain_count",
        "answer_size":     "answer_count",
        "first_label_length": "longest_label",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure has_digits exists (derive from domain if missing)
    if "has_digits" not in df.columns and "domain" in df.columns:
        df["has_digits"] = df["domain"].apply(lambda d: int(any(c.isdigit() for c in str(d))))
    elif "has_digits" not in df.columns:
        df["has_digits"] = 0

    # Ensure unique_char_ratio exists
    if "unique_char_ratio" not in df.columns and "domain" in df.columns:
        df["unique_char_ratio"] = df["domain"].apply(
            lambda d: round(len(set(str(d))) / max(len(str(d)), 1), 6)
        )
    elif "unique_char_ratio" not in df.columns:
        df["unique_char_ratio"] = 0.0

    # Fill missing FEATURES_8 columns with 0
    for col in FEATURES_8:
        if col not in df.columns:
            df[col] = 0

    return df


def parse_week3_json_to_array(path: str) -> np.ndarray:
    """
    Parse Week 3 features JSON → 1-D integer array of query lengths.
    Used by train.py / predict.py (Path B) when input is a Week 3 JSON.
    """
    records = _load_json_records(path)
    return np.array([week3_record_to_length(r) for r in records])


def parse_traff(path: str) -> np.ndarray:
    """Original .traff format: one integer per line."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(int(line))
                except ValueError:
                    continue
    return np.array(data)


def parse_csv(path: str) -> pd.DataFrame:
    """Load CSV → DataFrame (backward compat)."""
    print(f"[*] Reading CSV: {path}")
    return pd.read_csv(path)


# ─────────────────────────────────────────────
# UNIFIED LOADERS
# ─────────────────────────────────────────────

def load_as_dataframe(path: str) -> pd.DataFrame:
    """
    Load any supported file → DataFrame with FEATURES_8 columns.
    Auto-detects Week 3 JSON (has 'entropy' or 'subdomain_depth' key).
    """
    ext = Path(path).suffix.lower()

    if ext in (".pcap", ".pcapng", ".cap"):
        return parse_pcap_to_df(path)

    elif ext in (".json", ".jsonl", ".ndjson"):
        # Peek at first record to detect Week 3 format
        records = _load_json_records(path)
        if records and ("entropy" in records[0] or "subdomain_depth" in records[0] or "first_label" in records[0]):
            return parse_week3_json_to_df(path)
        return parse_json_to_df(path)

    elif ext == ".log":
        # Could be Zeek or Week 3 — try Week 3 first, fall back to generic JSON
        try:
            return parse_week3_json_to_df(path)
        except Exception:
            return parse_json_to_df(path)

    elif ext == ".csv":
        df = parse_csv(path)
        # Remap Week 3 columns if present
        rename_map = {
            "entropy":         "query_entropy",
            "numeric_ratio":   "digit_ratio",
            "subdomain_depth": "subdomain_count",
            "answer_size":     "answer_count",
            "first_label_length": "longest_label",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        for col in FEATURES_8:
            if col not in df.columns:
                df[col] = 0
        return df

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_as_array(path: str) -> np.ndarray:
    """
    Load any supported file → 1-D integer array of query lengths.
    Used by Path B (train.py / predict.py).
    """
    ext = Path(path).suffix.lower()

    if ext in (".pcap", ".pcapng", ".cap"):
        return parse_pcap_to_array(path)

    elif ext in (".json", ".jsonl", ".ndjson"):
        records = _load_json_records(path)
        if records and ("query_length" in records[0] or "entropy" in records[0]):
            # Week 3 format
            return np.array([week3_record_to_length(r) for r in records])
        return np.array([dns_record_to_length(r) for r in records])

    elif ext == ".log":
        try:
            records = _load_json_records(path)
            return np.array([week3_record_to_length(r) for r in records])
        except Exception:
            return parse_traff(path)

    elif ext == ".csv":
        df = pd.read_csv(path)
        if "query_length" in df.columns:
            return df["query_length"].values.astype(int)
        elif "domain" in df.columns:
            return df["domain"].apply(lambda d: len(str(d).rstrip("."))).values.astype(int)
        else:
            raise ValueError("CSV has no 'query_length' or 'domain' column for Path B")

    else:
        # Default: .traff or any plain integer-per-line file
        return parse_traff(path)
