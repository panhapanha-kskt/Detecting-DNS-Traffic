#!/bin/bash
CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'

_MEM_DIR="$(dirname "${BASH_SOURCE[0]}")"
_META="$_MEM_DIR/training_metadata.json"
_PROV="$_MEM_DIR/training_provenance.json"
_REG="$_MEM_DIR/trained_files_registry.json"

pipeline_print_training_context() {
  echo ""
  echo -e "${BOLD}${CYAN}  ╔══════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${CYAN}  ║   PIPELINE TRAINING CONTEXT                              ║${NC}"
  echo -e "${BOLD}${CYAN}  ╚══════════════════════════════════════════════════════════╝${NC}"

  if [[ ! -f "$_META" ]]; then
    echo -e "${RED}  ❌ No training metadata found at $_META${NC}"
    echo -e "${YELLOW}  ⚠  Run ./Train_Models.sh first before using the pipeline.${NC}"
    echo ""
    return 1
  fi

  python3 - <<PYEOF
import json, os
from datetime import datetime, timezone

meta  = json.load(open("$_META"))
try:   prov = json.load(open("$_PROV"))
except: prov = {}
try:   reg  = json.load(open("$_REG"))
except: reg  = {"trained_files": []}

now = datetime.now(timezone.utc)

def age(ts):
    try:
        t = datetime.fromisoformat(ts.replace("Z","+00:00"))
        d = (now - t).days
        return f"{d}d ago" if d > 0 else "today"
    except: return ts

print(f"\n  {'Last trained':<22}: {meta.get('last_trained','?')}  ({age(meta.get('last_trained',''))})")
print(f"  {'Retrain mode':<22}: {meta.get('retrain_mode', 'initial')}")
print(f"  {'Path A trained':<22}: {meta.get('path_a_trained','?')}")
print(f"  {'Path B trained':<22}: {meta.get('path_b_trained','?')}")
print()
print(f"  {'Total samples':<22}: {meta.get('total_samples','?')}")
print(f"  {'Attack samples':<22}: {meta.get('attack_samples','?')}")
print(f"  {'Benign samples':<22}: {meta.get('benign_samples','?')}")
print(f"  {'Attack types seen':<22}: {', '.join(meta.get('attack_types_seen', ['?']))}")
print()

total_trained = len(reg.get("trained_files", []))
print(f"  {'Files in registry':<22}: {total_trained} (all files ever trained on)")
print()

hist = meta.get("training_history", [])
if len(hist) > 1:
    print(f"  Training history (last {min(len(hist),5)} sessions):")
    for h in hist[-5:]:
        pa   = "A" if h.get("path_a") == "true" else "-"
        pb   = "B" if h.get("path_b") == "true" else "-"
        mode = h.get("retrain_mode", "?")
        new  = h.get("new_files", "?")
        skip = h.get("skipped_files", "?")
        print(f"    {h['date']}  samples={h['total_samples']:<7} new={new:<4} skipped={skip:<4} mode={mode}  [{pa}{pb}]")
    print()
PYEOF
  echo ""
}

pipeline_check_drift() {
  local live_file="${1:-}"
  [[ -f "$live_file" ]] || return 0
  [[ -f "$_META"     ]] || return 0

  python3 - <<PYEOF
import json, statistics, sys

meta  = json.load(open("$_META"))
snap  = meta.get("feature_snapshot", {})
if not snap:
    sys.exit(0)

try:
    with open("$live_file") as f: live = json.load(f)
except:
    sys.exit(0)

WARN  = '\033[1;33m'
GREEN = '\033[0;32m'
NC    = '\033[0m'

warnings = []
for feat, ref in snap.items():
    vals = [r[feat] for r in live if isinstance(r.get(feat), (int, float))]
    if not vals: continue
    live_mean = statistics.mean(vals)
    ref_mean  = ref["mean"]
    ref_std   = ref["stdev"] if ref["stdev"] > 0 else 1
    z = abs(live_mean - ref_mean) / ref_std
    if z > 3:
        warnings.append((feat, ref_mean, live_mean, z))

if warnings:
    print(f"\n{WARN}  ⚠  DATA DRIFT DETECTED — live traffic differs from training data:{NC}")
    for feat, ref_m, live_m, z in warnings:
        print(f"{WARN}     {feat:<30} train_mean={ref_m:.4f}  live_mean={live_m:.4f}  z={z:.1f}{NC}")
    print(f"{WARN}  Run: ./Train_Models.sh --retrain  (will only train on new files){NC}\n")
else:
    print(f"\n{GREEN}  ✅ Live traffic features match training distribution (no drift){NC}\n")
PYEOF
}

pipeline_print_training_context
