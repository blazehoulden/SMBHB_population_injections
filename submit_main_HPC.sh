#!/usr/bin/env bash
# =============================================================================
# submit_pipeline.sh  —  SMBHB population analysis Slurm orchestrator
#
# Pipeline:
#   stage1 array  (N_SIMS jobs, one per population)
#     └─afterok──> stage2 array  (N_SIMS jobs, one per population)
#
# Stage 1 (stage1_setup.py):
#   Each array task generates one SMBHB population, computes the per-pulsar
#   TOA changes, and saves both the population shard and the delta-stoa arrays.
#   Array index == sim_id (0 .. N_SIMS-1).
#
# Stage 2 (stage2_inject.py):
#   Each array task loads the corresponding population shard and all its
#   delta-stoa files, simulates noise, adds the GW signal, computes the
#   optimal-statistic SNR, scales to target, updates h0/D_comov/z in the
#   store, and finally computes CGW SNRs for the top binaries.
#   Array index == sim_id (0 .. N_SIMS-1).
#
# Usage
# ─────
#   bash submit_pipeline.sh [OPTIONS]
#
# Options
#   --config          population config name               [optimistic]
#   --target-snr      target OS SNR                        [4.0]
#   --snr-range       SNR acceptance window (low high)     [3.5 4.25]
#   --simulations     number of populations                [10]
#   --chunk-size      binaries per population shard        [1000000]
#   --output-dir      root directory for outputs           [auto]
#   --project         Slurm account/project                [$SLURM_ACCOUNT]
#   --partition       Slurm partition                      [compute]
#   --python          python interpreter to use            [python]
#   --cgw             enable CGW analysis in stage 2       [off]
#   --dry-run         print sbatch commands only           [off]
#
# Stage resource overrides
#   --s1-time / --s2-time   wall-clock limits
#   --s1-mem  / --s2-mem    memory per job
#   --s1-cpus / --s2-cpus   CPUs per job
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# CONFIG DEFAULTS
# =============================================================================
CONFIG="${SMBHB_CONFIG:-optimistic}"
TARGET_SNR="${SMBHB_TARGET_SNR:-4.0}"
SNR_LOW="${SMBHB_SNR_LOW:-3.5}"
SNR_HIGH="${SMBHB_SNR_HIGH:-4.25}"
N_SIMS="${SMBHB_SIMULATIONS:-10}"
CHUNK_SIZE="${SMBHB_CHUNK_SIZE:-1000000}"
PROJECT="${SMBHB_PROJECT:-${SLURM_ACCOUNT:-default}}"
PARTITION="${SMBHB_PARTITION:-compute}"
PYTHON="${SMBHB_PYTHON:-python}"
CGW_FLAG=""
DRY_RUN=0

# Stage 1: population synthesis + TOA delta computation (CPU-heavy, NUFFT)
S1_CPUS=32;  S1_MEM="64G";  S1_TIME="08:00:00"

# Stage 2: noise sim + SNR scaling + CGW analysis (memory-heavy, Enterprise)
S2_CPUS=16;  S2_MEM="64G";  S2_TIME="04:00:00"

# Max simultaneous jobs per array (avoid flooding the scheduler)
S1_MAX_CONCURRENT=50
S2_MAX_CONCURRENT=50

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2";                   shift 2 ;;
        --target-snr)    TARGET_SNR="$2";               shift 2 ;;
        --snr-range)     SNR_LOW="$2"; SNR_HIGH="$3";   shift 3 ;;
        --simulations)   N_SIMS="$2";                   shift 2 ;;
        --chunk-size)    CHUNK_SIZE="$2";               shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";               shift 2 ;;
        --project)       PROJECT="$2";                  shift 2 ;;
        --partition)     PARTITION="$2";                shift 2 ;;
        --python)        PYTHON="$2";                   shift 2 ;;
        --cgw)           CGW_FLAG="--cgw";              shift   ;;
        --dry-run)       DRY_RUN=1;                     shift   ;;
        --s1-time)       S1_TIME="$2";                  shift 2 ;;
        --s2-time)       S2_TIME="$2";                  shift 2 ;;
        --s1-mem)        S1_MEM="$2";                   shift 2 ;;
        --s2-mem)        S2_MEM="$2";                   shift 2 ;;
        --s1-cpus)       S1_CPUS="$2";                  shift 2 ;;
        --s2-cpus)       S2_CPUS="$2";                  shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Auto-generate output dir if not set
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    DATE=$(date +%Y-%m-%d)
    OUTPUT_DIR="${REPO_DIR}/runs/${DATE}_${CONFIG}"
fi

mkdir -p "${OUTPUT_DIR}/logs"

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[submit_pipeline] $*"; }

# Submit via sbatch, or echo in dry-run mode.
# Always uses --parsable so we get a bare job ID back.
# Returns the job ID string.
run_sbatch() {
    local desc="$1"; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "DRY-RUN [$desc]: sbatch $*" >&2
        echo "FAKE_${desc^^}_JOB"
    else
        sbatch --parsable "$@"
    fi
}

# =============================================================================
# SUMMARY
# =============================================================================
log "============================================================"
log "SMBHB Slurm pipeline"
log "  Repo dir    : ${REPO_DIR}"
log "  Output dir  : ${OUTPUT_DIR}"
log "  Config      : ${CONFIG}"
log "  Target SNR  : ${TARGET_SNR}  range=[${SNR_LOW}, ${SNR_HIGH}]"
log "  Populations : ${N_SIMS}"
log "  Chunk size  : ${CHUNK_SIZE}"
log "  Partition   : ${PARTITION}"
log "  Project     : ${PROJECT}"
log "  CGW         : ${CGW_FLAG:-off}"
log "  Dry run     : ${DRY_RUN}"
log "============================================================"

ARRAY_RANGE="0-$((N_SIMS - 1))"

# =============================================================================
# STAGE 1 — population synthesis array
#
# One job per simulation (sim_id = SLURM_ARRAY_TASK_ID).
# Each job:
#   - loads pulsars
#   - generates one SMBHB population shard  (populations/subpop_NNN.pkl.gz)
#   - computes per-pulsar TOA deltas         (stoas/simNNNN/{psr}_delta.npy)
#   - writes metadata/config.json            (first task wins, idempotent content)
# =============================================================================
log "Submitting stage 1 array (${N_SIMS} jobs)..."

S1_JOB=$(run_sbatch "stage1" \
    --job-name="smbhb_s1_${CONFIG}" \
    --account="${PROJECT}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${S1_CPUS}" \
    --mem="${S1_MEM}" \
    --time="${S1_TIME}" \
    --array="${ARRAY_RANGE}%${S1_MAX_CONCURRENT}" \
    --output="${OUTPUT_DIR}/logs/stage1_%A_%a.out" \
    --error="${OUTPUT_DIR}/logs/stage1_%A_%a.err" \
    --wrap="${PYTHON} ${REPO_DIR}/stage1_setup.py \
        --config ${CONFIG} \
        --target-snr ${TARGET_SNR} \
        --snr-range ${SNR_LOW} ${SNR_HIGH} \
        --simulations ${N_SIMS} \
        --chunk-size ${CHUNK_SIZE} \
        --output-dir ${OUTPUT_DIR} \
        --task-id \$SLURM_ARRAY_TASK_ID"
)
S1_JOB=$(echo "$S1_JOB" | tr -d '[:space:]')
log "  Stage 1 array job ID: ${S1_JOB}"

# =============================================================================
# STAGE 2 — noise simulation + SNR scaling + CGW analysis array
#
# One job per simulation (sim_id = SLURM_ARRAY_TASK_ID), runs after ALL
# stage-1 jobs complete successfully (afterok on the full array).
# Each job:
#   - loads pulsars + the population shard for its sim_id
#   - sums the saved per-pulsar TOA deltas for that sim
#   - simulates noise to produce a noise-only set of stoas
#   - adds the GW TOA deltas to produce noise+GW stoas
#   - computes optimal-statistic SNR
#   - iterates distance/redshift scaling until SNR lands in target range
#   - updates h0, D_comov, z in the population shard
#   - computes CGW SNR for a pre-filtered subset of binaries, saves to shard
# =============================================================================
log "Submitting stage 2 array (${N_SIMS} jobs, depends on stage 1)..."

S2_JOB=$(run_sbatch "stage2" \
    --job-name="smbhb_s2_${CONFIG}" \
    --account="${PROJECT}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${S2_CPUS}" \
    --mem="${S2_MEM}" \
    --time="${S2_TIME}" \
    --array="${ARRAY_RANGE}%${S2_MAX_CONCURRENT}" \
    --dependency="afterok:${S1_JOB}" \
    --output="${OUTPUT_DIR}/logs/stage2_%A_%a.out" \
    --error="${OUTPUT_DIR}/logs/stage2_%A_%a.err" \
    --wrap="${PYTHON} ${REPO_DIR}/stage2_inject.py \
        --output-dir ${OUTPUT_DIR} \
        --config ${CONFIG} \
        --target-snr ${TARGET_SNR} \
        --snr-range ${SNR_LOW} ${SNR_HIGH} \
        ${CGW_FLAG} \
        --task-id \$SLURM_ARRAY_TASK_ID"
)
S2_JOB=$(echo "$S2_JOB" | tr -d '[:space:]')
log "  Stage 2 array job ID: ${S2_JOB}"

# =============================================================================
# DONE
# =============================================================================
log ""
log "Pipeline submitted successfully."
log ""
log "Dependency chain:"
log "  stage1 array (${S1_JOB})  [${N_SIMS} jobs, sim_id 0...$((N_SIMS-1))]"
log "    └─ stage2 array (${S2_JOB})  [${N_SIMS} jobs, sim_id 0...$((N_SIMS-1))]"
log ""
log "Monitor with:"
log "  squeue -j ${S1_JOB},${S2_JOB}"
log "  tail -f ${OUTPUT_DIR}/logs/stage1_${S1_JOB}_0.out"
log ""
log "Output directory:"
log "  ${OUTPUT_DIR}/"
log "    populations/    — SMBHB population shards (updated by stage 2)"
log "    stoas/          — per-sim per-pulsar TOA delta files (from stage 1)"
log "    metadata/       — config.json"