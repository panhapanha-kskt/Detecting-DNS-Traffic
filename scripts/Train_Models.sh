#!/bin/bash
# ============================================================
#  DNS Traffic Anomaly Detection — ML Model Trainer v2
#  Run this ONCE (or whenever you want to retrain on new data).
#  Saves trained models as persistent "memory" files.
#
#  NEW in v2:
#    - Train from specific CSV / JSON / PCAP / LOG files or folders
#    - Auto-ingests raw Technitium .log files before training
#    - Rich training memory (file provenance, checksums, attack types)
#    - Run_Pipeline.sh reads memory and warns on data mismatch
#
#  NEW in v3 (smart retrain):
#    - --retrain no longer forces full retrain of existing data
#    - Instead, it finds NEW files not seen in previous training runs
#    - Already-trained data is skipped with a clear banner
#    - Only genuinely new data triggers retraining
#
#  Usage:
#    ./Train_Models.sh                              # default: scan DNS_LOG_DIR + data/
#    ./Train_Models.sh --data-dir /path/to/folder   # scan a specific folder
#    ./Train_Models.sh --files a.csv b.pcap c.log   # specific files
#    ./Train_Models.sh --data-dir /folder --files x.json   # both
#    ./Train_Models.sh --retrain                    # find NEW data and retrain on it
#    ./Train_Models.sh --force-retrain              # old behavior: retrain everything
#    ./Train_Models.sh --path-a-only
#    ./Train_Models.sh --path-b-only
#    ./Train_Models.sh --n-days 30
#    ./Train_Models.sh --help
# ============================================================

set -euo pipefail

# ── COLOURS ────────────────────────────────────────────────
RED='\033[0;31m';   GREEN='\033[0;32m';  YELLOW='\033[1;33m'
CYAN='\033[0;36m';  MAGENTA='\033[0;35m'; BOLD='\033[1m';  NC='\033[0m'

ok()     { echo -e "${GREEN}  ✅ $*${NC}"; }
info()   { echo -e "${CYAN}  ℹ  $*${NC}"; }
warn()   { echo -e "${YELLOW}  ⚠  $*${NC}"; }
fail()   { echo -e "${RED}  ❌ $*${NC}"; exit 1; }
step()   { echo -e "\n${BOLD}${CYAN}══ $* ${NC}"; }
detail() { echo -e "${MAGENTA}     ↳ $*${NC}"; }

banner() {
  printf "\n${BOLD}${CYAN}"
  printf "  ╔══════════════════════════════════════════════════════════════╗\n"
  printf "  ║   DNS Anomaly Detection — ML Model Trainer v3               ║\n"
  printf "  ║   Sources: CSV · JSON · PCAP · LOG                          ║\n"
  printf "  ║   Memory:  full provenance saved for pipeline awareness      ║\n"
  printf "  ╚══════════════════════════════════════════════════════════════╝\n"
  printf "${NC}"
}

already_trained_banner() {
  local file="$1"
  local records="$2"
  local trained_date="$3"
  printf "\n${BOLD}${YELLOW}"
  printf "  ╔══════════════════════════════════════════════════════════════╗\n"
  printf "  ║   ⚠  ALREADY TRAINED — SKIPPING                            ║\n"
  printf "  ╚══════════════════════════════════════════════════════════════╝\n"
  printf "${NC}"
  echo -e "${YELLOW}  File     : $file${NC}"
  echo -e "${YELLOW}  Records  : $records${NC}"
  echo -e "${YELLOW}  Trained  : $trained_date${NC}"
  echo -e "${YELLOW}  Action   : Ignored — looking for new data instead${NC}"
  printf "\n"
}

no_new_data_banner() {
  printf "\n${BOLD}${CYAN}"
  printf "  ╔══════════════════════════════════════════════════════════════╗\n"
  printf "  ║   ℹ  NO NEW DATA FOUND                                      ║\n"
  printf "  ╚══════════════════════════════════════════════════════════════╝\n"
  printf "${NC}"
  echo -e "${CYAN}  All discovered files have already been used for training.${NC}"
  echo -e "${CYAN}  Models are up to date — nothing new to learn.${NC}"
  echo -e "${CYAN}  Add new log/pcap/json/csv files and run again to retrain.${NC}"
  echo -e "${CYAN}  Use --force-retrain to retrain on ALL data regardless.${NC}"
  printf "\n"
}

new_data_found_banner() {
  local count="$1"
  printf "\n${BOLD}${GREEN}"
  printf "  ╔══════════════════════════════════════════════════════════════╗\n"
  printf "  ║   ✅  NEW DATA FOUND — RETRAINING                           ║\n"
  printf "  ╚══════════════════════════════════════════════════════════════╝\n"
  printf "${NC}"
  echo -e "${GREEN}  New files found : $count${NC}"
  echo -e "${GREEN}  Action          : Training on new data only${NC}"
  printf "\n"
}

# ── HARDCODED PATHS ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/home/paifern/DNS-_raffic-_Anomaly_Detection"
DATA_DIR="$BASE_DIR/data"
LOG_DIR="$BASE_DIR/logs"
MODEL_MEMORY_DIR="$BASE_DIR/model_memory"
DNS_LOG_DIR="$BASE_DIR/technitium-dns/data/dns_logs"

# ── DEFAULTS ───────────────────────────────────────────────
RETRAIN=false
FORCE_RETRAIN=false
PATH_A_ONLY=false
PATH_B_ONLY=false
PATH_B_MODEL="dns_trafficformer"
N_DAYS=20
RF_WEIGHT=0.6
TF_WEIGHT=0.4
CUSTOM_DATA_DIR=""
declare -a EXTRA_FILES=()

# ── ARG PARSING ────────────────────────────────────────────
show_help() {
  echo ""
  echo -e "${BOLD}Usage:${NC}  ./Train_Models.sh [options]"
  echo ""
  echo "Source options:"
  echo "  --data-dir PATH     Scan a folder for CSV/JSON/PCAP/LOG files"
  echo "  --files F1 F2 ...   Specific files to train on (any mix of types)"
  echo "                      Both flags can be combined."
  echo ""
  echo "Training options:"
  echo "  --retrain           Find NEW files not seen before and retrain on them"
  echo "                      Already-trained files are skipped automatically"
  echo "  --force-retrain     Retrain on ALL data regardless of history"
  echo "  --path-a-only       Train only Path A (RF + Transformer, 8 features)"
  echo "  --path-b-only       Train only Path B (Trafficformer, sequence)"
  echo "  --n-days N          Sequence window size for Path B (default: 20)"
  echo "  --model-name NAME   Path B model save name (default: dns_trafficformer)"
  echo "  --rf-weight N       RF weight in ensemble (default: 0.6)"
  echo "  --tf-weight N       Transformer weight (default: 0.4)"
  echo "  --help              Show this help"
  echo ""
  echo -e "${BOLD}Supported file types:${NC}"
  echo "  .csv    Pre-extracted feature tables"
  echo "  .json   week3_features-format feature files"
  echo "  .pcap   Raw packet captures (parsed via tshark)"
  echo "  .log    Raw Technitium DNS log files"
  echo ""
  echo -e "${BOLD}Smart retrain behavior:${NC}"
  echo "  --retrain        Skips already-trained files, only trains on NEW ones"
  echo "  --force-retrain  Trains on everything regardless of history"
  echo ""
  echo -e "${BOLD}Examples:${NC}"
  echo "  ./Train_Models.sh --data-dir /home/kali/captures/"
  echo "  ./Train_Models.sh --files /tmp/attack.pcap /tmp/normal.log"
  echo "  ./Train_Models.sh --retrain                    # smart: new data only"
  echo "  ./Train_Models.sh --force-retrain              # full retrain everything"
  echo "  ./Train_Models.sh --data-dir /captures --files extra.json --retrain"
  echo ""
  echo -e "${BOLD}After running this:${NC}"
  echo "  ./Run_Pipeline.sh    (uses saved models; prints training provenance)"
  echo ""
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --retrain)        RETRAIN=true;           shift ;;
    --force-retrain)  FORCE_RETRAIN=true;     shift ;;
    --path-a-only)    PATH_A_ONLY=true;       shift ;;
    --path-b-only)    PATH_B_ONLY=true;       shift ;;
    --n-days)         N_DAYS="$2";            shift 2 ;;
    --model-name)     PATH_B_MODEL="$2";      shift 2 ;;
    --rf-weight)      RF_WEIGHT="$2";         shift 2 ;;
    --tf-weight)      TF_WEIGHT="$2";         shift 2 ;;
    --data-dir)       CUSTOM_DATA_DIR="$2";   shift 2 ;;
    --files)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        EXTRA_FILES+=("$1"); shift
      done ;;
    --help|-h)      show_help ;;
    *) warn "Unknown option: $1 (ignored)"; shift ;;
  esac
done

# ── FILE PATHS ─────────────────────────────────────────────
MERGED_FEATURES="$DATA_DIR/merged_features_all.json"
MERGED_LABELED="$DATA_DIR/merged_features_labeled.json"
TRAINED_FILES_REGISTRY="$MODEL_MEMORY_DIR/trained_files_registry.json"

MEM_RF="$MODEL_MEMORY_DIR/dns_tunnel_rf.pkl"
MEM_TF="$MODEL_MEMORY_DIR/dns_tunnel_transformer.keras"
MEM_SCALER="$MODEL_MEMORY_DIR/dns_scaler.pkl"
MEM_THRESHOLD="$MODEL_MEMORY_DIR/dns_threshold.json"
MEM_PATH_B="$MODEL_MEMORY_DIR/${PATH_B_MODEL}.h5"
MEM_PATH_B_THRESHOLD="$MODEL_MEMORY_DIR/${PATH_B_MODEL}_threshold.json"
MEM_PATH_B_NORM="$MODEL_MEMORY_DIR/${PATH_B_MODEL}_norm.json"
MEM_METADATA="$MODEL_MEMORY_DIR/training_metadata.json"
MEM_PROVENANCE="$MODEL_MEMORY_DIR/training_provenance.json"

TRAIN_LOG="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

# ── SETUP ──────────────────────────────────────────────────
banner
mkdir -p "$DATA_DIR" "$LOG_DIR" "$MODEL_MEMORY_DIR" "$SCRIPT_DIR/model"
cd "$SCRIPT_DIR"

START_TIME=$(date +%s)
echo "" | tee -a "$TRAIN_LOG"
info "Script dir    : $SCRIPT_DIR"        | tee -a "$TRAIN_LOG"
info "Model memory  : $MODEL_MEMORY_DIR"  | tee -a "$TRAIN_LOG"
info "Data dir      : $DATA_DIR"          | tee -a "$TRAIN_LOG"
info "Log file      : $TRAIN_LOG"         | tee -a "$TRAIN_LOG"

# ── INITIALISE TRAINED FILES REGISTRY ─────────────────────
# Registry stores sha256 checksums of every file already trained on.
# --retrain uses this to skip known files and find new ones.
# --force-retrain ignores this registry entirely.
if [[ ! -f "$TRAINED_FILES_REGISTRY" ]]; then
  echo '{"trained_files": []}' > "$TRAINED_FILES_REGISTRY"
fi

# ── HELPER: check if a file has been trained on before ─────
file_already_trained() {
  local filepath="$1"
  [[ -f "$TRAINED_FILES_REGISTRY" ]] || return 1
  python3 -c "
import json, hashlib, sys
registry = json.load(open('$TRAINED_FILES_REGISTRY'))
trained  = {entry['checksum'] for entry in registry.get('trained_files', [])}

h = hashlib.sha256()
with open('$filepath', 'rb') as f:
    for chunk in iter(lambda: f.read(65536), b''): h.update(chunk)
checksum = h.hexdigest()

sys.exit(0 if checksum in trained else 1)
" 2>/dev/null
}

# ── HELPER: get trained date for a file ────────────────────
file_trained_date() {
  local filepath="$1"
  python3 -c "
import json, hashlib
registry = json.load(open('$TRAINED_FILES_REGISTRY'))

h = hashlib.sha256()
with open('$filepath', 'rb') as f:
    for chunk in iter(lambda: f.read(65536), b''): h.update(chunk)
checksum = h.hexdigest()

for entry in registry.get('trained_files', []):
    if entry['checksum'] == checksum:
        print(entry.get('trained_at', 'unknown date'))
        break
" 2>/dev/null || echo "unknown date"
}

# ── HELPER: get record count label for display ─────────────
file_record_count() {
  local filepath="$1"
  python3 -c "
import json, hashlib
registry = json.load(open('$TRAINED_FILES_REGISTRY'))

h = hashlib.sha256()
with open('$filepath', 'rb') as f:
    for chunk in iter(lambda: f.read(65536), b''): h.update(chunk)
checksum = h.hexdigest()

for entry in registry.get('trained_files', []):
    if entry['checksum'] == checksum:
        print(entry.get('records', '?'))
        break
" 2>/dev/null || echo "?"
}

# ── HELPER: register a file as trained ─────────────────────
register_trained_file() {
  local filepath="$1"
  local records="$2"
  local filetype="$3"
  python3 - <<PYEOF
import json, hashlib
from datetime import datetime, timezone

registry_path = "$TRAINED_FILES_REGISTRY"
try:
    registry = json.load(open(registry_path))
except:
    registry = {"trained_files": []}

h = hashlib.sha256()
with open("$filepath", "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
checksum = h.hexdigest()

# Don't add duplicates
existing = {e["checksum"] for e in registry["trained_files"]}
if checksum not in existing:
    registry["trained_files"].append({
        "path":       "$filepath",
        "checksum":   checksum,
        "type":       "$filetype",
        "records":    "$records",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
PYEOF
}

# ── CHECK IF ALREADY TRAINED ───────────────────────────────
step "Checking existing trained models"

PATH_A_EXISTS=false
PATH_B_EXISTS=false

if [[ -f "$MEM_RF" && -f "$MEM_TF" && -f "$MEM_SCALER" && -f "$MEM_THRESHOLD" ]]; then
  PATH_A_EXISTS=true
  TRAINED_DATE=$(python3 -c "
import json
try:
    d = json.load(open('$MEM_METADATA'))
    print(d.get('path_a_trained', 'unknown date'))
except: print('unknown date')
" 2>/dev/null || echo "unknown date")
  ok "Path A models exist (trained: $TRAINED_DATE)"
  [[ "$FORCE_RETRAIN" == false && "$RETRAIN" == false && "$PATH_B_ONLY" == false ]] && \
    warn "Path A will be SKIPPED — use --retrain (new data only) or --force-retrain (all data)"
fi

if [[ -f "$MEM_PATH_B" && -f "$MEM_PATH_B_THRESHOLD" && -f "$MEM_PATH_B_NORM" ]]; then
  PATH_B_EXISTS=true
  TRAINED_DATE_B=$(python3 -c "
import json
try:
    d = json.load(open('$MEM_METADATA'))
    print(d.get('path_b_trained', 'unknown date'))
except: print('unknown date')
" 2>/dev/null || echo "unknown date")
  ok "Path B model exists (trained: $TRAINED_DATE_B)"
  [[ "$FORCE_RETRAIN" == false && "$RETRAIN" == false && "$PATH_A_ONLY" == false ]] && \
    warn "Path B will be SKIPPED — use --retrain (new data only) or --force-retrain (all data)"
fi

TRAIN_PATH_A=false
TRAIN_PATH_B=false
[[ "$PATH_B_ONLY" == false ]] && { [[ "$PATH_A_EXISTS" == false || "$RETRAIN" == true || "$FORCE_RETRAIN" == true ]] && TRAIN_PATH_A=true; } || true
[[ "$PATH_A_ONLY" == false ]] && { [[ "$PATH_B_EXISTS" == false || "$RETRAIN" == true || "$FORCE_RETRAIN" == true ]] && TRAIN_PATH_B=true; } || true

if [[ "$TRAIN_PATH_A" == false && "$TRAIN_PATH_B" == false ]]; then
  echo ""
  echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}${GREEN}  ✅  All models already trained. Nothing to do.${NC}"
  echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${NC}"
  echo ""
  echo -e "  Models stored in : ${BOLD}$MODEL_MEMORY_DIR${NC}"
  echo -e "  Trained on       : $(python3 -c "import json; d=json.load(open('$MEM_PROVENANCE')); print(str(len(d.get('source_files',[]))) + ' source files')" 2>/dev/null || echo "see $MEM_METADATA")"
  echo -e "  Smart retrain    : ${BOLD}./Train_Models.sh --retrain${NC}  (new files only)"
  echo -e "  Force retrain    : ${BOLD}./Train_Models.sh --force-retrain${NC}  (all data)"
  echo -e "  Run detection    : ${BOLD}./Run_Pipeline.sh${NC}"
  echo ""
  exit 0
fi

info "Training mode — RETRAIN: $RETRAIN  |  FORCE_RETRAIN: $FORCE_RETRAIN" | tee -a "$TRAIN_LOG"
info "Training needed — Path A: $TRAIN_PATH_A  |  Path B: $TRAIN_PATH_B"   | tee -a "$TRAIN_LOG"

# ── PREFLIGHT CHECKS ───────────────────────────────────────
step "Preflight checks"

python3 --version &>/dev/null || fail "python3 not found"
ok "python3: $(python3 --version)"

TSHARK_AVAILABLE=false
if command -v tshark &>/dev/null; then
  TSHARK_AVAILABLE=true
  ok "tshark: $(tshark --version 2>&1 | head -1)"
else
  warn "tshark not found — PCAP files will be skipped (install: sudo apt install tshark)"
fi

REQUIRED_SCRIPTS=("DNS_feature_extractor.py" "dns_parser.py" "Model.py" "Train.py" "Ml_bridge.py")
for f in "${REQUIRED_SCRIPTS[@]}"; do
  [[ -f "$SCRIPT_DIR/$f" ]] || fail "Missing script: $SCRIPT_DIR/$f"
done
ok "All required scripts present"

MISSING_PKGS=()
python3 -c "import pandas"     2>/dev/null || MISSING_PKGS+=("pandas")
python3 -c "import numpy"      2>/dev/null || MISSING_PKGS+=("numpy")
python3 -c "import sklearn"    2>/dev/null || MISSING_PKGS+=("scikit-learn")
python3 -c "import tensorflow" 2>/dev/null || MISSING_PKGS+=("tensorflow")
python3 -c "import joblib"     2>/dev/null || MISSING_PKGS+=("joblib")
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  warn "Installing missing packages: ${MISSING_PKGS[*]}"
  pip install "${MISSING_PKGS[@]}" --break-system-packages -q || \
    fail "pip install failed"
fi
ok "Python packages ready"

# ══════════════════════════════════════════════════════════════
#  SOURCE DISCOVERY — collect all input files
# ══════════════════════════════════════════════════════════════
step "SOURCE DISCOVERY — scanning for training data"

declare -a SRC_CSV=()
declare -a SRC_JSON=()
declare -a SRC_PCAP=()
declare -a SRC_LOG=()

# Counters for summary
SKIPPED_ALREADY_TRAINED=0
NEW_FILES_FOUND=0

classify_file() {
  local f="$1"
  [[ -f "$f" ]] || { warn "File not found, skipping: $f"; return; }

  # ── Smart retrain: check registry before bucketing ────────
  if [[ "$FORCE_RETRAIN" == false && "$RETRAIN" == true ]]; then
    if file_already_trained "$f"; then
      local trained_date
      trained_date=$(file_trained_date "$f")
      local rec_count
      rec_count=$(file_record_count "$f")
      already_trained_banner "$f" "$rec_count" "$trained_date"
      SKIPPED_ALREADY_TRAINED=$(( SKIPPED_ALREADY_TRAINED + 1 ))
      return
    fi
  fi

  case "${f,,}" in
    *.csv)           SRC_CSV+=("$f");  detail "CSV  : $f"; NEW_FILES_FOUND=$(( NEW_FILES_FOUND + 1 )) ;;
    *.json)          SRC_JSON+=("$f"); detail "JSON : $f"; NEW_FILES_FOUND=$(( NEW_FILES_FOUND + 1 )) ;;
    *.pcap|*.pcapng) SRC_PCAP+=("$f"); detail "PCAP : $f"; NEW_FILES_FOUND=$(( NEW_FILES_FOUND + 1 )) ;;
    *.log)           SRC_LOG+=("$f");  detail "LOG  : $f"; NEW_FILES_FOUND=$(( NEW_FILES_FOUND + 1 )) ;;
    *) warn "Unknown extension, skipping: $f" ;;
  esac
}

# ── Scan custom --data-dir if given ───────────────────────
if [[ -n "$CUSTOM_DATA_DIR" ]]; then
  [[ -d "$CUSTOM_DATA_DIR" ]] || fail "--data-dir not found: $CUSTOM_DATA_DIR"
  info "Scanning folder: $CUSTOM_DATA_DIR"
  while IFS= read -r -d '' f; do
    classify_file "$f"
  done < <(find "$CUSTOM_DATA_DIR" -maxdepth 3 \
    \( -iname "*.csv" -o -iname "*.json" -o -iname "*.pcap" \
       -o -iname "*.pcapng" -o -iname "*.log" \) \
    -print0 2>/dev/null)
fi

# ── Individual --files entries ─────────────────────────────
if [[ ${#EXTRA_FILES[@]} -gt 0 ]]; then
  info "Individual files provided: ${#EXTRA_FILES[@]}"
  for f in "${EXTRA_FILES[@]}"; do
    classify_file "$f"
  done
fi

# ── Fallback: default DNS_LOG_DIR ─────────────────────────
TOTAL_SRC=$(( ${#SRC_CSV[@]} + ${#SRC_JSON[@]} + ${#SRC_PCAP[@]} + ${#SRC_LOG[@]} ))
if [[ $TOTAL_SRC -eq 0 && $SKIPPED_ALREADY_TRAINED -eq 0 ]]; then
  warn "No files specified — falling back to default DNS log directory"
  info "Scanning: $DNS_LOG_DIR"
  while IFS= read -r -d '' f; do
    classify_file "$f"
  done < <(find "$DNS_LOG_DIR" -maxdepth 3 \
    \( -iname "*.log" -o -iname "*.csv" -o -iname "*.json" \) \
    -print0 2>/dev/null)
fi

TOTAL_SRC=$(( ${#SRC_CSV[@]} + ${#SRC_JSON[@]} + ${#SRC_PCAP[@]} + ${#SRC_LOG[@]} ))

echo ""
ok "Source discovery complete:"
ok "  New CSV  : ${#SRC_CSV[@]} file(s)"
ok "  New JSON : ${#SRC_JSON[@]} file(s)"
ok "  New PCAP : ${#SRC_PCAP[@]} file(s)"
ok "  New LOG  : ${#SRC_LOG[@]} file(s)"
[[ $SKIPPED_ALREADY_TRAINED -gt 0 ]] && \
  warn "  Skipped  : $SKIPPED_ALREADY_TRAINED file(s) already trained — ignored"

# ── If --retrain and all files were already trained → nothing new ──
if [[ "$RETRAIN" == true && "$FORCE_RETRAIN" == false && $TOTAL_SRC -eq 0 ]]; then
  no_new_data_banner
  exit 0
fi

# ── If new files found under --retrain, show banner ────────
if [[ "$RETRAIN" == true && "$FORCE_RETRAIN" == false && $NEW_FILES_FOUND -gt 0 ]]; then
  new_data_found_banner "$NEW_FILES_FOUND"
fi

[[ $TOTAL_SRC -gt 0 ]] || fail "No training data found anywhere. Use --data-dir or --files."

# ══════════════════════════════════════════════════════════════
#  STEP 0-A: PRE-PROCESS RAW LOGS → JSON features
# ══════════════════════════════════════════════════════════════
step "STEP 0-A — Pre-processing raw LOG files"

declare -a JSON_FROM_LOGS=()

if [[ ${#SRC_LOG[@]} -gt 0 ]]; then
  info "Found ${#SRC_LOG[@]} new log file(s) — extracting features via DNS_feature_extractor.py"
  for log_file in "${SRC_LOG[@]}"; do
    base=$(basename "$log_file" .log)
    out_json="$DATA_DIR/extracted_${base}_$(date +%s).json"
    detail "Extracting: $log_file → $out_json"
    python3 DNS_feature_extractor.py \
      --input  "$log_file" \
      --output "$out_json" 2>&1 | tee -a "$TRAIN_LOG" || {
        warn "Extraction failed for $log_file — skipping"
        continue
      }
    if [[ -f "$out_json" ]]; then
      JSON_FROM_LOGS+=("$out_json")
      SRC_JSON+=("$out_json")
      # Register the original log file as trained (not the extracted JSON)
      RECORD_COUNT_LOG=$(python3 -c "import json; print(len(json.load(open('$out_json'))))" 2>/dev/null || echo "?")
      register_trained_file "$log_file" "$RECORD_COUNT_LOG" "log"
    fi
  done
  ok "Log pre-processing done — ${#JSON_FROM_LOGS[@]} JSON file(s) produced"
else
  info "No new LOG files — skipping log pre-processing"
fi

# ══════════════════════════════════════════════════════════════
#  STEP 0-B: PRE-PROCESS PCAP → JSON features (via tshark)
# ══════════════════════════════════════════════════════════════
step "STEP 0-B — Pre-processing PCAP files"

declare -a JSON_FROM_PCAP=()

if [[ ${#SRC_PCAP[@]} -gt 0 && "$TSHARK_AVAILABLE" == true ]]; then
  info "Found ${#SRC_PCAP[@]} new PCAP file(s) — extracting DNS records via tshark"
  for pcap_file in "${SRC_PCAP[@]}"; do
    base=$(basename "$pcap_file" .pcap)
    base="${base%.pcapng}"
    raw_csv="$DATA_DIR/pcap_raw_${base}_$(date +%s).csv"
    out_json="$DATA_DIR/pcap_feat_${base}_$(date +%s).json"

    detail "tshark parsing: $pcap_file"
    tshark -r "$pcap_file" \
      -Y "dns" \
      -T fields \
      -e frame.time_epoch \
      -e ip.src \
      -e dns.qry.name \
      -e dns.qry.type \
      -e dns.flags.response \
      -e dns.resp.len \
      -E header=y \
      -E separator=, \
      -E quote=d \
      -E occurrence=f \
      > "$raw_csv" 2>>"$TRAIN_LOG" || {
        warn "tshark failed for $pcap_file — skipping"
        continue
      }

    python3 dns_parser.py \
      --input  "$raw_csv" \
      --output "$out_json" \
      --source pcap 2>&1 | tee -a "$TRAIN_LOG" || {
        warn "dns_parser.py failed for $raw_csv — skipping"
        continue
      }

    if [[ -f "$out_json" ]]; then
      JSON_FROM_PCAP+=("$out_json")
      SRC_JSON+=("$out_json")
      RECORD_COUNT_PCAP=$(python3 -c "import json; print(len(json.load(open('$out_json'))))" 2>/dev/null || echo "?")
      register_trained_file "$pcap_file" "$RECORD_COUNT_PCAP" "pcap"
    fi
    rm -f "$raw_csv"
  done
  ok "PCAP pre-processing done — ${#JSON_FROM_PCAP[@]} JSON file(s) produced"
elif [[ ${#SRC_PCAP[@]} -gt 0 && "$TSHARK_AVAILABLE" == false ]]; then
  warn "PCAP files found but tshark is not installed — skipping all PCAP files"
  warn "Install with: sudo apt install tshark"
else
  info "No new PCAP files — skipping PCAP pre-processing"
fi

# ══════════════════════════════════════════════════════════════
#  STEP 0-C: MERGE ALL SOURCES → single features file
# ══════════════════════════════════════════════════════════════
step "STEP 0-C — Merging all sources into one feature dataset"

MERGE_ARGS=""
for f in "${SRC_JSON[@]}"; do MERGE_ARGS="$MERGE_ARGS --json $f"; done
for f in "${SRC_CSV[@]}";  do MERGE_ARGS="$MERGE_ARGS --csv  $f"; done

[[ -z "$MERGE_ARGS" ]] && fail "No usable feature files after pre-processing. Cannot train."

python3 - <<PYEOF 2>&1 | tee -a "$TRAIN_LOG"
import json, csv, os, sys, hashlib
from datetime import datetime, timezone

json_files = [$(printf '"%s",' "${SRC_JSON[@]}" 2>/dev/null | sed 's/,$//')]
csv_files  = [$(printf '"%s",' "${SRC_CSV[@]}"  2>/dev/null | sed 's/,$//')]

merged   = []
file_log = []

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()[:16]

for path in json_files:
    if not path or not os.path.isfile(path): continue
    try:
        with open(path) as f: data = json.load(f)
        if not isinstance(data, list): data = [data]
        merged.extend(data)
        file_log.append({
            "path": path, "type": "json",
            "records": len(data), "checksum": sha256(path)
        })
        print(f"  [json] {len(data):>6} records  ← {path}")
    except Exception as e:
        print(f"  [warn] Could not load {path}: {e}", file=sys.stderr)

for path in csv_files:
    if not path or not os.path.isfile(path): continue
    try:
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: (float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v)
                              for k, v in row.items()})
        merged.extend(rows)
        file_log.append({
            "path": path, "type": "csv",
            "records": len(rows), "checksum": sha256(path)
        })
        print(f"  [csv]  {len(rows):>6} records  ← {path}")
    except Exception as e:
        print(f"  [warn] Could not load {path}: {e}", file=sys.stderr)

if not merged:
    print("ERROR: merged dataset is empty", file=sys.stderr)
    sys.exit(1)

with open("$MERGED_FEATURES", "w") as f:
    json.dump(merged, f)

with open("$MEM_PROVENANCE", "w") as f:
    json.dump({
        "merged_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_records": len(merged),
        "source_files": file_log,
        "source_counts": {
            "json_files":  len(json_files),
            "csv_files":   len(csv_files),
            "pcap_converted": len([x for x in file_log if "pcap" in x["path"]]),
            "log_converted":  len([x for x in file_log if "extracted_" in x["path"]]),
        }
    }, f, indent=2)

print(f"\n  Merged {len(merged)} total records from {len(file_log)} source file(s)")
print(f"  Provenance saved → $MEM_PROVENANCE")

# Register JSON and CSV source files in the trained registry
registry_path = "$TRAINED_FILES_REGISTRY"
try:
    registry = json.load(open(registry_path))
except:
    registry = {"trained_files": []}

existing_checksums = {e["checksum"] for e in registry["trained_files"]}
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

for entry in file_log:
    full_checksum = sha256(entry["path"])
    if full_checksum not in existing_checksums:
        registry["trained_files"].append({
            "path":       entry["path"],
            "checksum":   full_checksum,
            "type":       entry["type"],
            "records":    entry["records"],
            "trained_at": now_str,
        })

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)
print(f"  Registry updated → {registry_path}  ({len(registry['trained_files'])} total trained files)")
PYEOF

[[ -f "$MERGED_FEATURES" ]] || fail "Merge step failed — $MERGED_FEATURES not created"
RECORD_COUNT=$(python3 -c "import json; print(len(json.load(open('$MERGED_FEATURES'))))" 2>/dev/null || echo "?")
ok "Merged dataset: $RECORD_COUNT records → $MERGED_FEATURES"

# ══════════════════════════════════════════════════════════════
#  STEP 1: GENERATE LABELS
# ══════════════════════════════════════════════════════════════
step "STEP 1 — Generate labels from merged dataset"

python3 Ml_bridge.py \
  --week3  "$MERGED_FEATURES" \
  --output "$MERGED_LABELED" \
  --label-only 2>&1 | tee -a "$TRAIN_LOG"

[[ $? -eq 0 && -f "$MERGED_LABELED" ]] || fail "Label generation failed"

ATTACK_COUNT=$(python3 -c "
import json; d=json.load(open('$MERGED_LABELED'))
print(sum(1 for r in d if r.get('label',0)==1))" 2>/dev/null || echo "?")
BENIGN_COUNT=$(python3 -c "
import json; d=json.load(open('$MERGED_LABELED'))
print(sum(1 for r in d if r.get('label',0)==0))" 2>/dev/null || echo "?")
TOTAL_COUNT=$(python3 -c "
import json; print(len(json.load(open('$MERGED_LABELED'))))" 2>/dev/null || echo "?")

ATTACK_TYPES=$(python3 -c "
import json
d = json.load(open('$MERGED_LABELED'))
types = sorted(set(r.get('attack_type','unknown') for r in d if r.get('label',0)==1))
print(','.join(types) if types else 'none')" 2>/dev/null || echo "unknown")

ok "Labels: attacks=$ATTACK_COUNT  benign=$BENIGN_COUNT  total=$TOTAL_COUNT"
ok "Attack types found: $ATTACK_TYPES"

# ══════════════════════════════════════════════════════════════
#  STEP 2: TRAIN PATH A
# ══════════════════════════════════════════════════════════════
if [[ "$TRAIN_PATH_A" == true ]]; then
  step "STEP 2 — Training Path A: RF + Transformer (8 features)"
  info "Learns WHICH patterns are attacks vs benign (supervised)"
  info "Dataset : $MERGED_LABELED ($TOTAL_COUNT records)"

  python3 Model.py \
    --train     "$MERGED_LABELED" \
    --rf-weight "$RF_WEIGHT" \
    --tf-weight "$TF_WEIGHT" 2>&1 | tee -a "$TRAIN_LOG"

  [[ $? -eq 0 ]] || fail "Path A training failed"

  [[ -f "$SCRIPT_DIR/dns_tunnel_rf.pkl" ]]            || fail "dns_tunnel_rf.pkl not saved"
  [[ -f "$SCRIPT_DIR/dns_tunnel_transformer.keras" ]] || fail "dns_tunnel_transformer.keras not saved"
  [[ -f "$SCRIPT_DIR/dns_scaler.pkl" ]]               || fail "dns_scaler.pkl not saved"
  [[ -f "$SCRIPT_DIR/dns_threshold.json" ]]           || fail "dns_threshold.json not saved"

  cp "$SCRIPT_DIR/dns_tunnel_rf.pkl"            "$MEM_RF"
  cp "$SCRIPT_DIR/dns_tunnel_transformer.keras" "$MEM_TF"
  cp "$SCRIPT_DIR/dns_scaler.pkl"               "$MEM_SCALER"
  cp "$SCRIPT_DIR/dns_threshold.json"           "$MEM_THRESHOLD"

  ok "Path A trained and saved to model_memory/"
else
  step "STEP 2 — Path A SKIPPED (restoring saved models)"
  cp "$MEM_RF"        "$SCRIPT_DIR/dns_tunnel_rf.pkl"
  cp "$MEM_TF"        "$SCRIPT_DIR/dns_tunnel_transformer.keras"
  cp "$MEM_SCALER"    "$SCRIPT_DIR/dns_scaler.pkl"
  cp "$MEM_THRESHOLD" "$SCRIPT_DIR/dns_threshold.json"
  ok "Path A models restored from memory"
fi

# ══════════════════════════════════════════════════════════════
#  STEP 3: TRAIN PATH B
# ══════════════════════════════════════════════════════════════
if [[ "$TRAIN_PATH_B" == true ]]; then
  step "STEP 3 — Training Path B: Trafficformer (sequence reconstruction)"
  info "Learns WHAT normal traffic looks like (unsupervised)"
  info "Window size: $N_DAYS timesteps"

  mkdir -p "$SCRIPT_DIR/model"
  python3 Train.py \
    --filename  "$MERGED_FEATURES" \
    --n_days    "$N_DAYS" \
    --save_name "$PATH_B_MODEL" \
    --data_dir  "/" 2>&1 | tee -a "$TRAIN_LOG"

  [[ $? -eq 0 ]] || fail "Path B training failed"
  [[ -f "$SCRIPT_DIR/model/${PATH_B_MODEL}.h5" ]] || fail "Path B model not saved"

  cp "$SCRIPT_DIR/model/${PATH_B_MODEL}.h5"             "$MEM_PATH_B"
  cp "$SCRIPT_DIR/model/${PATH_B_MODEL}_threshold.json" "$MEM_PATH_B_THRESHOLD"
  cp "$SCRIPT_DIR/model/${PATH_B_MODEL}_norm.json"      "$MEM_PATH_B_NORM"

  ok "Path B trained and saved to model_memory/"
else
  step "STEP 3 — Path B SKIPPED (restoring saved model)"
  mkdir -p "$SCRIPT_DIR/model"
  cp "$MEM_PATH_B"           "$SCRIPT_DIR/model/${PATH_B_MODEL}.h5"
  cp "$MEM_PATH_B_THRESHOLD" "$SCRIPT_DIR/model/${PATH_B_MODEL}_threshold.json"
  cp "$MEM_PATH_B_NORM"      "$SCRIPT_DIR/model/${PATH_B_MODEL}_norm.json"
  ok "Path B model restored from memory"
fi

# ══════════════════════════════════════════════════════════════
#  WRITE RICH TRAINING METADATA
# ══════════════════════════════════════════════════════════════
step "Saving rich training memory"

NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

python3 - <<PYEOF
import json, os

meta_path = "$MEM_METADATA"
try:
    with open(meta_path) as f: meta = json.load(f)
except:
    meta = {}

try:
    with open("$MEM_PROVENANCE") as f: prov = json.load(f)
except:
    prov = {}

update = {
    "last_trained":         "$NOW",
    "training_data":        "$MERGED_FEATURES",
    "total_samples":        "$TOTAL_COUNT",
    "attack_samples":       "$ATTACK_COUNT",
    "benign_samples":       "$BENIGN_COUNT",
    "attack_types_seen":    "$ATTACK_TYPES".split(","),
    "path_b_model_name":    "$PATH_B_MODEL",
    "path_b_window_size":   $N_DAYS,
    "rf_weight":            $RF_WEIGHT,
    "tf_weight":            $TF_WEIGHT,
    "retrain_mode":         "force" if "$FORCE_RETRAIN" == "true" else ("smart" if "$RETRAIN" == "true" else "initial"),
    "new_files_this_run":   $NEW_FILES_FOUND,
    "skipped_files_this_run": $SKIPPED_ALREADY_TRAINED,
    "source_summary": {
        "total_source_files": len(prov.get("source_files", [])),
        "csv_files":          prov.get("source_counts", {}).get("csv_files", 0),
        "json_files":         prov.get("source_counts", {}).get("json_files", 0),
        "pcap_converted":     prov.get("source_counts", {}).get("pcap_converted", 0),
        "log_converted":      prov.get("source_counts", {}).get("log_converted", 0),
    },
    "feature_snapshot": {},
    "model_files": {
        "rf":           "$MEM_RF",
        "transformer":  "$MEM_TF",
        "scaler":       "$MEM_SCALER",
        "threshold":    "$MEM_THRESHOLD",
        "trafficformer":"$MEM_PATH_B",
        "tf_threshold": "$MEM_PATH_B_THRESHOLD",
        "tf_norm":      "$MEM_PATH_B_NORM",
        "provenance":   "$MEM_PROVENANCE",
    }
}

try:
    import statistics
    with open("$MERGED_LABELED") as f: data = json.load(f)
    numeric_keys = [k for k in (data[0].keys() if data else [])
                    if isinstance(data[0].get(k), (int, float))
                    and k not in ("label",)]
    snap = {}
    for k in numeric_keys[:12]:
        vals = [r[k] for r in data if isinstance(r.get(k), (int, float))]
        if vals:
            snap[k] = {
                "mean":   round(statistics.mean(vals), 4),
                "stdev":  round(statistics.stdev(vals) if len(vals) > 1 else 0, 4),
                "min":    round(min(vals), 4),
                "max":    round(max(vals), 4),
            }
    update["feature_snapshot"] = snap
except Exception as e:
    print(f"  [warn] Feature snapshot skipped: {e}")

if "$TRAIN_PATH_A" == "true":
    update["path_a_trained"] = "$NOW"
    update["path_a_prev"]    = meta.get("path_a_trained", "first time")
else:
    update["path_a_trained"] = meta.get("path_a_trained", "$NOW")

if "$TRAIN_PATH_B" == "true":
    update["path_b_trained"] = "$NOW"
    update["path_b_prev"]    = meta.get("path_b_trained", "first time")
else:
    update["path_b_trained"] = meta.get("path_b_trained", "$NOW")

history = meta.get("training_history", [])
history.append({
    "date":              "$NOW",
    "total_samples":     "$TOTAL_COUNT",
    "attack_samples":    "$ATTACK_COUNT",
    "attack_types":      "$ATTACK_TYPES".split(","),
    "source_files":      len(prov.get("source_files", [])),
    "new_files":         $NEW_FILES_FOUND,
    "skipped_files":     $SKIPPED_ALREADY_TRAINED,
    "retrain_mode":      "force" if "$FORCE_RETRAIN" == "true" else ("smart" if "$RETRAIN" == "true" else "initial"),
    "path_a":            "$TRAIN_PATH_A",
    "path_b":            "$TRAIN_PATH_B",
})
update["training_history"] = history[-10:]

meta.update(update)
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata   → {meta_path}")
print(f"  Provenance → $MEM_PROVENANCE")
PYEOF

ok "Training memory saved"

# ══════════════════════════════════════════════════════════════
#  GENERATE: pipeline_awareness_check.sh
# ══════════════════════════════════════════════════════════════
step "Generating pipeline awareness check script"

cat > "$MODEL_MEMORY_DIR/pipeline_awareness_check.sh" << 'AWARENESS_EOF'
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
AWARENESS_EOF

chmod +x "$MODEL_MEMORY_DIR/pipeline_awareness_check.sh"
ok "Pipeline awareness check → $MODEL_MEMORY_DIR/pipeline_awareness_check.sh"

RUN_PIPELINE="$SCRIPT_DIR/../Run_Pipeline.sh"
if [[ -f "$RUN_PIPELINE" ]]; then
  if ! grep -q "pipeline_awareness_check.sh" "$RUN_PIPELINE"; then
    sed -i '2s|^|# ── Training context (auto-inserted by Train_Models.sh) ──\nsource "'"$MODEL_MEMORY_DIR"'/pipeline_awareness_check.sh" || true\n\n|' "$RUN_PIPELINE"
    ok "Patched Run_Pipeline.sh to load training context on startup"
  else
    ok "Run_Pipeline.sh already has awareness check — no patch needed"
  fi
else
  warn "Run_Pipeline.sh not found at $RUN_PIPELINE — patch skipped"
fi

# ══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
printf "${BOLD}${GREEN}══════════════════════════════════════════════════════════════\n${NC}"
printf "${BOLD}${GREEN}  ✅  ML MODEL TRAINING COMPLETE  (${MINS}m ${SECS}s)\n${NC}"
printf "${BOLD}${GREEN}══════════════════════════════════════════════════════════════\n${NC}"
echo ""
echo -e "${BOLD}This run:${NC}"
echo "  New files trained    : $NEW_FILES_FOUND"
echo "  Files skipped        : $SKIPPED_ALREADY_TRAINED (already in registry)"
echo "  Retrain mode         : $([ "$FORCE_RETRAIN" == "true" ] && echo "force (all data)" || ([ "$RETRAIN" == "true" ] && echo "smart (new data only)" || echo "initial"))"
echo ""
echo -e "${BOLD}Trained on:${NC}"
echo "  Total samples   : $TOTAL_COUNT"
echo "  Attack samples  : $ATTACK_COUNT"
echo "  Benign samples  : $BENIGN_COUNT"
echo "  Attack types    : $ATTACK_TYPES"
echo ""
echo -e "${BOLD}Model memory saved to:${NC}  $MODEL_MEMORY_DIR"
[[ -f "$MEM_RF" ]]               && echo "  Random Forest          : dns_tunnel_rf.pkl"
[[ -f "$MEM_TF" ]]               && echo "  Transformer            : dns_tunnel_transformer.keras"
[[ -f "$MEM_SCALER" ]]           && echo "  Scaler                 : dns_scaler.pkl"
[[ -f "$MEM_THRESHOLD" ]]        && echo "  Threshold              : dns_threshold.json"
[[ -f "$MEM_PATH_B" ]]           && echo "  Trafficformer          : ${PATH_B_MODEL}.h5"
[[ -f "$MEM_PATH_B_THRESHOLD" ]] && echo "  Trafficformer thresh   : ${PATH_B_MODEL}_threshold.json"
[[ -f "$MEM_PROVENANCE" ]]       && echo "  Provenance             : training_provenance.json"
[[ -f "$MEM_METADATA" ]]         && echo "  Metadata               : training_metadata.json"
[[ -f "$TRAINED_FILES_REGISTRY" ]] && echo "  File registry          : trained_files_registry.json"
echo "  Training log           : $TRAIN_LOG"
echo ""
echo -e "${BOLD}${CYAN}  ▶  Run detection:${NC}"
echo -e "  ${BOLD}./Run_Pipeline.sh${NC}"
echo ""
echo -e "${CYAN}  Smart retrain (new files only):${NC}"
echo -e "  ${BOLD}./Train_Models.sh --retrain${NC}"
echo ""
echo -e "${CYAN}  Force retrain everything:${NC}"
echo -e "  ${BOLD}./Train_Models.sh --force-retrain${NC}"
echo ""
