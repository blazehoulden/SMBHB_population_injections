#!/usr/bin/env bash
# =============================================================================
# submit_stage2_recovery.sh  —  Resubmit stage2 for resumable simulations
#
# After running scan_stage2_timeouts.py, this script:
#   1. Reads the classification CSV
#   2. Filters for resumable sims (status = incomplete_late)
#   3. Generates sbatch commands for stage2-only re-submission
#   4. Uses extended walltime to finish the remaining phases
#
# With resumability patches applied to stage2_inject.py:
#   - Baseline phase skips if phase_handoff.json exists (already done)
#   - Scenario phases skip if CGW SNR field is populated (already done)
#   - Each phase is idempotent — retrying is safe
#
# Usage:
#   bash submit_stage2_recovery.sh \
#     --classification results.csv \
#     --output-dir /path/to/runs/2026-06-16_optimistic \
#     --time 06:00:00 \
#     --dry-run
#
# Options:
#   --classification    CSV from scan_stage2_timeouts.py
#   --output-dir        Root output directory (from original submit_main_HPC.sh)
#   --time              Wall time for recovery jobs (default: 06:00:00)
#   --dry-run           Print sbatch commands without submitting
# =============================================================================

set -eo pipefail

# =============================================================================
# DEFAULTS
# =============================================================================
CLASSIFICATION_CSV=""
OUTPUT_DIR=""
S2_TIME="06:00:00"      # Longer than default (03:00:00)
S2_CPUS="1"
S2_MEM="30G"
DRY_RUN=0
REPO_DIR="/fred/oz005/users/bhoulden/SMBHB_population_injections"
ENV_SETUP="module purge; unset PYTHONPATH; module load mamba; mamba activate smbhb312;"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --classification)   CLASSIFICATION_CSV="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";         shift 2 ;;
        --time)             S2_TIME="$2";            shift 2 ;;
        --dry-run)          DRY_RUN=1;               shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Validate
if [[ -z "$CLASSIFICATION_CSV" ]]; then
    echo "ERROR: --classification required" >&2
    exit 1
fi
if [[ ! -f "$CLASSIFICATION_CSV" ]]; then
    echo "ERROR: Classification CSV not found: $CLASSIFICATION_CSV" >&2
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --output-dir required" >&2
    exit 1
fi
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: Output dir not found: $OUTPUT_DIR" >&2
    exit 1
fi

# =============================================================================
# PARSE CLASSIFICATION CSV AND EXTRACT RESUMABLE SIMS
# =============================================================================
echo "Reading classification from: $CLASSIFICATION_CSV"

RESUMABLE_SIMS=()
while IFS=',' read -r sim_id status phase_reached resumable reason log_file; do
    # Skip header and non-resumable rows
    [[ "$sim_id" == "sim_id" ]] && continue
    [[ "$resumable" != "True" ]] && continue
    
    RESUMABLE_SIMS+=("$sim_id")
done < "$CLASSIFICATION_CSV"

if [[ ${#RESUMABLE_SIMS[@]} -eq 0 ]]; then
    echo "No resumable simulations found in classification CSV"
    exit 0
fi

echo "Found ${#RESUMABLE_SIMS[@]} resumable simulations:"
for sim_id in "${RESUMABLE_SIMS[@]}"; do
    echo "  sim$(printf '%03d' "$sim_id")"
done

# =============================================================================
# GENERATE SBATCH COMMANDS
# =============================================================================
log() { echo "[recovery] $*"; }

run_sbatch() {
    local desc="$1"; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "DRY-RUN [$desc]: sbatch $*" >&2
        echo "FAKE_${desc^^}"
    else
        sbatch --parsable "$@"
    fi
}

log "======================================================================"
log "Stage 2 Recovery — submitting resumable simulations"
log "======================================================================"
log "  Classification  : $CLASSIFICATION_CSV"
log "  Output dir      : $OUTPUT_DIR"
log "  S2 time         : $S2_TIME"
log "  S2 mem          : $S2_MEM"
log "  Dry run         : $DRY_RUN"
log ""

# Read full CSV to get config details (target SNR, etc.) from first sim
CONFIG=$(grep -v "^sim_id" "$CLASSIFICATION_CSV" | head -1 | cut -d',' -f1)
if [[ -z "$CONFIG" ]]; then
    # Try to infer from output dir name
    CONFIG=$(basename "$OUTPUT_DIR" | sed 's/^.*_//')
fi

log "Detected config: $CONFIG"

# For each resumable sim, read metadata to extract original stage2 parameters
echo ""
JOBS_SUBMITTED=()

for sim_id in "${RESUMABLE_SIMS[@]}"; do
    sim_dir=$(printf "%s/sim%03d" "$OUTPUT_DIR" "$sim_id")
    meta_dir="$sim_dir/metadata"
    config_file="$meta_dir/config.json"
    
    if [[ ! -f "$config_file" ]]; then
        log "  Skipping sim$(printf '%03d' "$sim_id"): no config.json"
        continue
    fi
    
    # Extract parameters from config.json
    TARGET_SNR=$(jq -r '.target_snr // 3.75' "$config_file" 2>/dev/null || echo "3.75")
    SNR_LOW=$(jq -r '.snr_range[0] // 3.5' "$config_file" 2>/dev/null || echo "3.5")
    SNR_HIGH=$(jq -r '.snr_range[1] // 4.0' "$config_file" 2>/dev/null || echo "4.0")
    N_CHUNKS=$(jq -r '.n_chunks // 10' "$config_file" 2>/dev/null || echo "10")
    N_SUB_CHUNKS=$(jq -r '.n_sub_chunks // 1' "$config_file" 2>/dev/null || echo "1")
    N_SIMS=$(jq -r '.n_sims // 400' "$config_file" 2>/dev/null || echo "400")
    NOISE_SEED_BASE=$(jq -r '.noise_seed_base // 26072001' "$config_file" 2>/dev/null || echo "26072001")
    CGW_FLAG=$(jq -r '.cgw // true' "$config_file" 2>/dev/null && echo "--cgw" || echo "")
    SYNTHETIC_PTAS=$(jq -r '.synthetic_ptas // false' "$config_file" 2>/dev/null && echo "--synthetic-ptas" || echo "")
    
    S2_JOB=$(run_sbatch "s2_recovery_sim$(printf '%03d' "$sim_id")" \
        --job-name="s2_recovery_$(printf '%03d' "$sim_id")" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="$S2_CPUS" \
        --mem="$S2_MEM" \
        --time="$S2_TIME" \
        --output="${OUTPUT_DIR}/logs/stage2_recovery_sim$(printf '%03d' "$sim_id")_%j.out" \
        --error="${OUTPUT_DIR}/logs/stage2_recovery_sim$(printf '%03d' "$sim_id")_%j.err" \
        --wrap="${ENV_SETUP} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python -u ${REPO_DIR}/stage2_inject.py \
            --output-dir ${OUTPUT_DIR} \
            --config ${CONFIG} \
            --target-snr ${TARGET_SNR} \
            --snr-range ${SNR_LOW} ${SNR_HIGH} \
            --sim-id ${sim_id} \
            --n-chunks ${N_CHUNKS} \
            --n-sub-chunks ${N_SUB_CHUNKS} \
            --noise-seed-base ${NOISE_SEED_BASE} \
            --n-sims ${N_SIMS} \
            --cgw ${CGW_FLAG} \
            ${SYNTHETIC_PTAS}"
    )
    S2_JOB=$(echo "$S2_JOB" | tr -d '[:space:]')
    JOBS_SUBMITTED+=("$S2_JOB")
    
    log "  sim$(printf '%03d' "$sim_id"): job_id=${S2_JOB}  (SNR=[${SNR_LOW},${SNR_HIGH}])"
done

echo ""
if [[ ${#JOBS_SUBMITTED[@]} -eq 0 ]]; then
    log "No jobs submitted"
else
    log "======================================================================"
    log "Submitted ${#JOBS_SUBMITTED[@]} recovery jobs"
    log ""
    log "Job IDs: ${JOBS_SUBMITTED[*]}"
    log ""
    log "Monitor:"
    log "  squeue -u \$USER"
    log "  tail -f ${OUTPUT_DIR}/logs/stage2_recovery_sim*.out"
    log ""
    log "Each resumable sim will:"
    log "  1. Detect phase_handoff.json (baseline already done) → skip baseline"
    log "  2. Detect populated cgw_snr_* fields (scenario done) → skip that scenario"
    log "  3. Complete any remaining scenarios"
    log "  4. Write stage2_complete.json marker"
    log "======================================================================"
fi

exit 0