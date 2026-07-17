#!/bin/bash
# ============================================================
#  DNS Traffic Anomaly Detection — Detection Pipeline
#  Runs full detection using SAVED trained models.
#  Does NOT retrain — training is done separately by Train_Models.sh
#
#  First time setup:
#    1. ./Train_Models.sh        ← train models once (saves memory)
#    2. ./Run_Pipeline.sh        ← run detection anytime after
#
#  Usage:
#    ./Run_Pipeline.sh                  # full detection pipeline
#    ./Run_Pipeline.sh --week3-only     # only extract features, stop
#    ./Run_Pipeline.sh --detect-only    # skip Week 3, only run detection
#    ./Run_Pipeline.sh --help
# ============================================================

set -euo pipefail

# ── COLOURS ────────────────────────────────────────────────
RED='\033[0;31m';  GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m';     NC='\033[0m'

ok()   { echo -e "${GREEN}  ✅ $*${NC}"; }
info() { echo -e "${CYAN}  ℹ  $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠  $*${NC}"; }
fail() { echo -e "${RED}  ❌ $*${NC}"; exit 1; }
step() { echo -e "\n${BOLD}${CYAN}══ $* ${NC}"; }
# After — force variable expansion with printf
banner() {
  printf "\n${BOLD}${CYAN}"
  printf "  ╔══════════════════════════════════════════════════════╗\n"
  printf "  ║   DNS Traffic Anomaly Detection — Pipeline Runner   ║\n"
  printf "  ╚══════════════════════════════════════════════════════╝\n"
  printf "${NC}"
}

# ── HARDCODED PATHS ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/home/paifern/DNS-_raffic-_Anomaly_Detection"
DATA_DIR="$BASE_DIR/data"
LOG_DIR="$BASE_DIR/logs"
MODEL_MEMORY_DIR="$BASE_DIR/model_memory"
DNS_LOG_DIR="$BASE_DIR/technitium-dns/data/dns_logs"

# ── DEFAULTS ───────────────────────────────────────────────
WEEK3_ONLY=false
DETECT_ONLY=false
PATH_B_MODEL="dns_trafficformer"
ZSCORE_THRESHOLD=2.0

# ── ARG PARSING ────────────────────────────────────────────
show_help() {
  echo ""
  echo -e "${BOLD}Usage:${NC}  ./Run_Pipeline.sh [options]"
  echo ""
  echo "Options:"
  echo "  --week3-only       Run Week 3 feature extraction only, then stop"
  echo "  --detect-only      Skip Week 3 extraction, run detection only"
  echo "  --zscore N         Z-score threshold for Week 4 (default: 2.0)"
  echo "  --help             Show this help"
  echo ""
  echo -e "${BOLD}First time setup:${NC}"
  echo "  ./Train_Models.sh    ← run ONCE to train and save models"
  echo "  ./Run_Pipeline.sh    ← run ANYTIME to detect threats"
  echo ""
  echo -e "${BOLD}How it works:${NC}"
  echo "  Step 1: Extract features from new DNS logs (Week 3)"
  echo "  Step 2: Load trained models from model_memory/"
  echo "  Step 3: Run ML bridge — apply trained models to new traffic"
  echo "  Step 4: Run detection engine — generate alerts (Week 4)"
  echo ""
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --week3-only)   WEEK3_ONLY=true;         shift ;;
    --detect-only)  DETECT_ONLY=true;        shift ;;
    --zscore)       ZSCORE_THRESHOLD="$2";   shift 2 ;;
    --help|-h)      show_help ;;
    *) warn "Unknown option: $1 (ignored)"; shift ;;
  esac
done

# ── FILE PATHS ─────────────────────────────────────────────
WEEK3_FEATURES="$DATA_DIR/week3_features_all.json"
WEEK3_WITH_ML="$DATA_DIR/week3_features_with_ml.json"
WEEK4_ALERTS="$DATA_DIR/week4_unified_alerts.json"
PIPELINE_LOG="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Model memory paths
MEM_RF="$MODEL_MEMORY_DIR/dns_tunnel_rf.pkl"
MEM_TF="$MODEL_MEMORY_DIR/dns_tunnel_transformer.keras"
MEM_SCALER="$MODEL_MEMORY_DIR/dns_scaler.pkl"
MEM_THRESHOLD="$MODEL_MEMORY_DIR/dns_threshold.json"
MEM_PATH_B="$MODEL_MEMORY_DIR/${PATH_B_MODEL}.h5"
MEM_PATH_B_THRESHOLD="$MODEL_MEMORY_DIR/${PATH_B_MODEL}_threshold.json"
MEM_PATH_B_NORM="$MODEL_MEMORY_DIR/${PATH_B_MODEL}_norm.json"
MEM_METADATA="$MODEL_MEMORY_DIR/training_metadata.json"

# ── SETUP ──────────────────────────────────────────────────
banner
mkdir -p "$DATA_DIR" "$LOG_DIR" "$SCRIPT_DIR/model"
cd "$SCRIPT_DIR"

echo "" | tee -a "$PIPELINE_LOG"
info "Script dir : $SCRIPT_DIR"         | tee -a "$PIPELINE_LOG"
info "Data dir   : $DATA_DIR"           | tee -a "$PIPELINE_LOG"
info "Model mem  : $MODEL_MEMORY_DIR"   | tee -a "$PIPELINE_LOG"
info "Log file   : $PIPELINE_LOG"       | tee -a "$PIPELINE_LOG"

START_TIME=$(date +%s)

# ── PREFLIGHT: CHECK TRAINED MODELS EXIST ─────────────────
step "Checking trained model memory"

PATH_A_READY=false
PATH_B_READY=false

if [[ -f "$MEM_RF" && -f "$MEM_TF" && -f "$MEM_SCALER" && -f "$MEM_THRESHOLD" ]]; then
  PATH_A_READY=true
  TRAINED_A=$(python3 -c "
import json
try:
    d = json.load(open('$MEM_METADATA'))
    print(d.get('path_a_trained', 'unknown'))
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
  ok "Path A models ready (trained: $TRAINED_A)"
else
  warn "Path A models NOT found in model_memory/"
fi

if [[ -f "$MEM_PATH_B" && -f "$MEM_PATH_B_THRESHOLD" && -f "$MEM_PATH_B_NORM" ]]; then
  PATH_B_READY=true
  TRAINED_B=$(python3 -c "
import json
try:
    d = json.load(open('$MEM_METADATA'))
    print(d.get('path_b_trained', 'unknown'))
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
  ok "Path B model ready (trained: $TRAINED_B)"
else
  warn "Path B model NOT found in model_memory/"
fi

# If no models at all — tell user to train first
if [[ "$PATH_A_READY" == false && "$PATH_B_READY" == false && "$WEEK3_ONLY" == false ]]; then
  echo ""
  echo -e "${BOLD}${RED}══════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}${RED}  ❌  No trained models found.${NC}"
  echo -e "${BOLD}${RED}══════════════════════════════════════════════════════${NC}"
  echo ""
  echo -e "  You need to train the models first:"
  echo -e "  ${BOLD}${CYAN}./Train_Models.sh${NC}"
  echo ""
  echo -e "  This only needs to be done ONCE."
  echo -e "  After training, run this script again."
  echo ""
  exit 1
fi

# Restore models from memory into scripts/model/ so Python can find them
if [[ "$PATH_A_READY" == true ]]; then
  cp "$MEM_RF"        "$SCRIPT_DIR/dns_tunnel_rf.pkl"
  cp "$MEM_TF"        "$SCRIPT_DIR/dns_tunnel_transformer.keras"
  cp "$MEM_SCALER"    "$SCRIPT_DIR/dns_scaler.pkl"
  cp "$MEM_THRESHOLD" "$SCRIPT_DIR/dns_threshold.json"
  info "Path A models loaded from memory" | tee -a "$PIPELINE_LOG"
fi

if [[ "$PATH_B_READY" == true ]]; then
  mkdir -p "$SCRIPT_DIR/model"
  cp "$MEM_PATH_B"           "$SCRIPT_DIR/model/${PATH_B_MODEL}.h5"
  cp "$MEM_PATH_B_THRESHOLD" "$SCRIPT_DIR/model/${PATH_B_MODEL}_threshold.json"
  cp "$MEM_PATH_B_NORM"      "$SCRIPT_DIR/model/${PATH_B_MODEL}_norm.json"
  info "Path B model loaded from memory"  | tee -a "$PIPELINE_LOG"
fi

# Show training metadata
if [[ -f "$MEM_METADATA" ]]; then
  python3 - <<PYEOF
import json
try:
    d = json.load(open('$MEM_METADATA'))
    print(f"\n  📋 Model memory info:")
    print(f"     Last trained   : {d.get('last_trained', 'unknown')}")
    print(f"     Trained on     : {d.get('total_samples', '?')} samples ({d.get('attack_samples','?')} attacks + {d.get('benign_samples','?')} benign)")
    history = d.get('training_history', [])
    if len(history) > 1:
        print(f"     Training runs  : {len(history)} total")
except Exception as e:
    pass
PYEOF
fi

# ── PREFLIGHT: SCRIPTS AND PACKAGES ───────────────────────
step "Preflight checks"

python3 --version &>/dev/null || fail "python3 not found"
ok "python3: $(python3 --version)"

REQUIRED_SCRIPTS=(
  "DNS_feature_extractor.py"
  "dns_parser.py"
  "Model.py"
  "Train.py"
  "Predict.py"
  "Ml_bridge.py"
  "Week4_detection_engine.py"
)
for f in "${REQUIRED_SCRIPTS[@]}"; do
  [[ -f "$SCRIPT_DIR/$f" ]] || fail "Missing script: $SCRIPT_DIR/$f"
done
ok "All script files present"

# Check DNS logs
SKIP_WEEK3=false
if [[ ! -d "$DNS_LOG_DIR" ]]; then
  warn "DNS log dir not found: $DNS_LOG_DIR"
  [[ -f "$WEEK3_FEATURES" ]] || fail "No DNS logs and no cached Week 3 features."
  SKIP_WEEK3=true
  info "Using cached Week 3 features"
else
  LOG_COUNT=$(find "$DNS_LOG_DIR" -name "*.log" 2>/dev/null | wc -l)
  if [[ "$LOG_COUNT" -eq 0 ]]; then
    [[ -f "$WEEK3_FEATURES" ]] || fail "No .log files and no cached features."
    SKIP_WEEK3=true
    info "No new log files — using cached features"
  else
    ok "Found $LOG_COUNT DNS log file(s)"
  fi
fi

# Python packages
MISSING_PKGS=()
python3 -c "import pandas"     2>/dev/null || MISSING_PKGS+=("pandas")
python3 -c "import numpy"      2>/dev/null || MISSING_PKGS+=("numpy")
python3 -c "import sklearn"    2>/dev/null || MISSING_PKGS+=("scikit-learn")
python3 -c "import tensorflow" 2>/dev/null || MISSING_PKGS+=("tensorflow")
python3 -c "import joblib"     2>/dev/null || MISSING_PKGS+=("joblib")
python3 -c "import tqdm"       2>/dev/null || MISSING_PKGS+=("tqdm")
python3 -c "import dateutil"   2>/dev/null || MISSING_PKGS+=("python-dateutil")

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  warn "Installing: ${MISSING_PKGS[*]}"
  pip install "${MISSING_PKGS[@]}" --break-system-packages -q || \
    fail "pip install failed"
fi
ok "Python packages ready"

# ── STEP 1: WEEK 3 FEATURE EXTRACTION ─────────────────────
if [[ "$DETECT_ONLY" == false && "$SKIP_WEEK3" == false ]]; then
  step "STEP 1 — Week 3: Feature Extraction (new traffic)"
  info "Parsing new DNS logs from: $DNS_LOG_DIR" | tee -a "$PIPELINE_LOG"

  python3 DNS_feature_extractor.py 2>&1 | tee -a "$PIPELINE_LOG"
  [[ $? -eq 0 && -f "$WEEK3_FEATURES" ]] || fail "DNS_feature_extractor.py failed"

  RECORD_COUNT=$(python3 -c "import json; d=json.load(open('$WEEK3_FEATURES')); print(len(d))" 2>/dev/null || echo "?")
  ok "Week 3 complete — $RECORD_COUNT records extracted"
else
  [[ "$DETECT_ONLY" == true ]] && info "STEP 1 skipped (--detect-only)" || info "STEP 1 skipped (using cached features)"
  [[ -f "$WEEK3_FEATURES" ]] || fail "week3_features_all.json not found"
fi

# Stop here if --week3-only
if [[ "$WEEK3_ONLY" == true ]]; then
  ok "Week 3 complete. Stopping (--week3-only)."
  info "Output: $WEEK3_FEATURES"
  exit 0
fi

# ── STEP 2: ML BRIDGE (apply trained models to new traffic) ─
step "STEP 2 — ML Bridge: Applying trained model memory to new traffic"
info "Using models trained on: $(python3 -c "
import json
try:
    d = json.load(open('$MEM_METADATA'))
    print(d.get('last_trained', 'unknown'))
except:
    print('unknown')
" 2>/dev/null || echo "unknown")"

BRIDGE_ARGS=(
  --week3  "$WEEK3_FEATURES"
  --output "$WEEK3_WITH_ML"
  --weight-a "0.6"
  --weight-b "0.4"
)

if [[ "$PATH_B_READY" == true ]]; then
  BRIDGE_ARGS+=(--path-b-model "$PATH_B_MODEL")
  info "Path B: active (Trafficformer reconstruction anomaly detection)"
else
  BRIDGE_ARGS+=(--no-path-b)
  warn "Path B: inactive (model not trained — run ./Train_Models.sh)"
fi

if [[ "$PATH_A_READY" == false ]]; then
  warn "Path A: inactive (models not trained — run ./Train_Models.sh)"
fi

python3 Ml_bridge.py "${BRIDGE_ARGS[@]}" 2>&1 | tee -a "$PIPELINE_LOG"
[[ $? -eq 0 && -f "$WEEK3_WITH_ML" ]] || fail "Ml_bridge.py failed"

ok "ML bridge complete — new traffic scored against trained models"

# ── STEP 3: WEEK 4 DETECTION ENGINE ───────────────────────
step "STEP 3 — Week 4: Detection Engine"
info "Running 6 detection rules (4 Sentinel + Z-score + 2 ML rules)" | tee -a "$PIPELINE_LOG"

python3 Week4_detection_engine.py \
  --input  "$WEEK3_WITH_ML" \
  --output "$WEEK4_ALERTS" \
  --zscore "$ZSCORE_THRESHOLD" \
  --csv 2>&1 | tee -a "$PIPELINE_LOG"

[[ $? -eq 0 && -f "$WEEK4_ALERTS" ]] || fail "Week4_detection_engine.py failed"

ALERT_COUNT=$(python3 -c "
import json
d = json.load(open('$WEEK4_ALERTS'))
print(d.get('total_alerts', len(d.get('alerts', []))))
" 2>/dev/null || echo "?")

HIGH_COUNT=$(python3 -c "
import json
d = json.load(open('$WEEK4_ALERTS'))
alerts = d.get('alerts', [])
print(sum(1 for a in alerts if a.get('severity','') == 'High'))
" 2>/dev/null || echo "?")

MEDIUM_COUNT=$(python3 -c "
import json
d = json.load(open('$WEEK4_ALERTS'))
alerts = d.get('alerts', [])
print(sum(1 for a in alerts if a.get('severity','') == 'Medium'))
" 2>/dev/null || echo "?")

ok "Week 4 complete — $ALERT_COUNT total alerts ($HIGH_COUNT High, $MEDIUM_COUNT Medium)"

# ── FINAL SUMMARY ──────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  ✅  DETECTION COMPLETE  (${MINS}m ${SECS}s)${NC}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Results:${NC}"
echo "  🚨 Total alerts      : $ALERT_COUNT"
echo "  🔴 High severity     : $HIGH_COUNT"
echo "  🟡 Medium severity   : $MEDIUM_COUNT"
echo ""
echo -e "${BOLD}Output files:${NC}"
[[ -f "$WEEK3_FEATURES" ]] && echo "  📄 Week 3 features    : $WEEK3_FEATURES"
[[ -f "$WEEK3_WITH_ML"  ]] && echo "  🤖 ML-enriched data   : $WEEK3_WITH_ML"
[[ -f "$WEEK4_ALERTS"   ]] && echo "  🚨 Alerts (JSON)      : $WEEK4_ALERTS"
CSV_PATH="${WEEK4_ALERTS%.json}.csv"
[[ -f "$CSV_PATH"       ]] && echo "  📊 Alerts (CSV)       : $CSV_PATH"
echo "  📋 Pipeline log      : $PIPELINE_LOG"
echo ""

# Show which ML rules fired
python3 - <<PYEOF
import json
try:
    d = json.load(open('$WEEK4_ALERTS'))
    by_rule = d.get('alerts_by_rule', {})
    if by_rule:
        print("  Detection breakdown by rule:")
        for rule, count in sorted(by_rule.items(), key=lambda x: x[1], reverse=True):
            icon = "🤖" if "ML" in rule else "🔍"
            print(f"    {icon} {rule}: {count}")
        print()
except:
    pass
PYEOF

echo -e "${BOLD}Model memory used:${NC}"
echo -e "  Stored in: ${CYAN}$MODEL_MEMORY_DIR${NC}"
if [[ -f "$MEM_METADATA" ]]; then
  python3 -c "
import json
d = json.load(open('$MEM_METADATA'))
print(f\"  Trained on : {d.get('last_trained','?')}\")
print(f\"  Samples    : {d.get('total_samples','?')} ({d.get('attack_samples','?')} attack + {d.get('benign_samples','?')} benign)\")
" 2>/dev/null || true
fi
echo ""
echo -e "${CYAN}  To retrain on new data:${NC}"
echo -e "  ${BOLD}./Train_Models.sh --retrain${NC}"
echo ""
