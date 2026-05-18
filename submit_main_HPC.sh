#!/usr/bin/env bash
# =============================================================================
# submit_main_HPC.sh  —  SMBHB population analysis Slurm orchestrator
#
# Pipeline (repeated N_SIMS times independently):
#
#   stage1 flat array  (N_SIMS × N_CHUNKS jobs)
#     task_id = sim_id * N_CHUNKS + chunk_id
#     each job generates one chunk of one simulation
#     └─afterok──> stage2 single job
#                  waits for ALL N_S1_TASKS stage-1 tasks to complete,
#                  then combines all chunks for all sims, scales SNR,
#                  updates shards, optionally runs CGW
#
# Output layout:
#   <output_dir>/
#     sim000/
#       populations/   subpop_000.pkl.gz ... subpop_{N_CHUNKS-1:03d}.pkl.gz
#       stoas/         sim0000/ ... sim{N_CHUNKS-1:04d}/
#       metadata/      config.json
#     sim001/
#       ...
#
# Usage
# ─────
#   bash submit_main_HPC.sh [OPTIONS]
#
# Options
#   --config          population config name               [optimistic]
#   --target-snr      target OS SNR                        [4.0]
#   --snr-range       SNR acceptance window (low high)     [3.5 4.25]
#   --simulations     number of independent populations    [400]
#   --n-chunks        number of chunks per simulation      [10]
#   --chunk-size      binaries per chunk                   [1000000]
#   --output-dir      root directory for all outputs       [auto]
#   --cgw             enable CGW analysis in stage 2       [on]
#   --no-cgw          disable CGW analysis in stage 2      [off]
#   --dry-run         print sbatch commands only           [off]
#
# Stage resource overrides
#   --s1-time / --s2-time   wall-clock limits
#   --s1-mem  / --s2-mem    memory per job
#   --s1-cpus / --s2-cpus   CPUs per job
# =============================================================================

set -eo pipefail
PS1="${PS1:-}"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Job Name: ${SLURM_JOB_NAME:-n/a}"
echo "Node: ${SLURM_NODELIST:-n/a}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

ENV_SETUP="module purge; unset PYTHONPATH; module load mamba; mamba activate smbhb312;"

REPO_DIR="/fred/oz005/users/bhoulden/SMBHB_population_injections"

# =============================================================================
# CONFIG DEFAULTS
# =============================================================================
CONFIG="${SMBHB_CONFIG:-optimistic}"
TARGET_SNR="${SMBHB_TARGET_SNR:-4.0}"
SNR_LOW="${SMBHB_SNR_LOW:-3.5}"
SNR_HIGH="${SMBHB_SNR_HIGH:-4.25}"
N_SIMS="${SMBHB_SIMULATIONS:-400}"
N_CHUNKS="${SMBHB_N_CHUNKS:-10}"
CHUNK_SIZE="${SMBHB_CHUNK_SIZE:-1000000}"
CGW_FLAG="${SMBHB_CGW_FLAG:---cgw}"
DRY_RUN=0

# Stage 1: one chunk per task — pop synthesis + NUFFT
S1_CPUS=1;  S1_MEM="14G";  S1_TIME="00:15:00"

# Stage 2: single job — loads all chunks for all sims, Enterprise SNR loop
S2_CPUS=1;  S2_MEM="44G";  S2_TIME="01:30:00"

# Max simultaneous jobs per array (stage 1 only)
S1_MAX_CONCURRENT=100

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2";                   shift 2 ;;
        --target-snr)    TARGET_SNR="$2";               shift 2 ;;
        --snr-range)     SNR_LOW="$2"; SNR_HIGH="$3";   shift 3 ;;
        --simulations)   N_SIMS="$2";                   shift 2 ;;
        --n-chunks)      N_CHUNKS="$2";                 shift 2 ;;
        --chunk-size)    CHUNK_SIZE="$2";               shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";               shift 2 ;;
        --cgw)           CGW_FLAG="--cgw";              shift   ;;
        --no-cgw)        CGW_FLAG="";                   shift   ;;
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

# Derived counts
N_S1_TASKS=$(( N_SIMS * N_CHUNKS ))
S1_LAST=$(( N_S1_TASKS - 1 ))

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[submit_pipeline] $*"; }

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
log "  Repo dir      : ${REPO_DIR}"
log "  Output dir    : ${OUTPUT_DIR}"
log "  Config        : ${CONFIG}"
log "  Target SNR    : ${TARGET_SNR}  range=[${SNR_LOW}, ${SNR_HIGH}]"
log "  Simulations   : ${N_SIMS}"
log "  Chunks / sim  : ${N_CHUNKS}  x  ${CHUNK_SIZE} binaries"
log "  Total pop/sim : $(( N_CHUNKS * CHUNK_SIZE )) binaries"
log "  S1 tasks      : ${N_S1_TASKS}  (${N_SIMS} sims x ${N_CHUNKS} chunks)"
log "  S2 tasks      : 1  (single job, combines all sims after stage 1)"
log "  CGW           : $([[ -n \"${CGW_FLAG}\" ]] && echo on || echo off)"
log "  Dry run       : ${DRY_RUN}"
log "============================================================"

# =============================================================================
# STAGE 1 — flat array: task_id = sim_id * N_CHUNKS + chunk_id
#
# Each task decodes:
#   sim_id   = task_id // N_CHUNKS
#   chunk_id = task_id  % N_CHUNKS
# and writes to <output_dir>/sim{sim_id:03d}/
# =============================================================================
log "Submitting stage 1 flat array (${N_S1_TASKS} tasks)..."

S1_JOB=$(run_sbatch "stage1" \
    --job-name="smbhb_s1_${CONFIG}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${S1_CPUS}" \
    --mem="${S1_MEM}" \
    --time="${S1_TIME}" \
    --array="0-${S1_LAST}%${S1_MAX_CONCURRENT}" \
    --output="${OUTPUT_DIR}/logs/stage1_%A_%a.out" \
    --error="${OUTPUT_DIR}/logs/stage1_%A_%a.err" \
    --wrap="${ENV_SETUP} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -u ${REPO_DIR}/stage1_setup.py \
        --config ${CONFIG} \
        --target-snr ${TARGET_SNR} \
        --snr-range ${SNR_LOW} ${SNR_HIGH} \
        --chunk-size ${CHUNK_SIZE} \
        --n-chunks ${N_CHUNKS} \
        --output-dir ${OUTPUT_DIR} \
        --task-id \$SLURM_ARRAY_TASK_ID"
)
S1_JOB=$(echo "$S1_JOB" | tr -d '[:space:]')
log "  Stage 1 array job ID: ${S1_JOB}"

# =============================================================================
# STAGE 2 — single job that combines ALL stage-1 outputs
#
# --dependency=afterok:<array_job_id> blocks until EVERY task in the
# stage-1 array has completed successfully before this job starts.
# stage2_inject.py receives --n-sims and loops over all sim_ids itself.
# =============================================================================
log "Submitting stage 2 single job (depends on ALL ${N_S1_TASKS} stage-1 tasks)..."

S2_JOB=$(run_sbatch "stage2" \
    --job-name="smbhb_s2_${CONFIG}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${S2_CPUS}" \
    --mem="${S2_MEM}" \
    --time="${S2_TIME}" \
    --dependency="afterok:${S1_JOB}_*" \
    --output="${OUTPUT_DIR}/logs/stage2_%j.out" \
    --error="${OUTPUT_DIR}/logs/stage2_%j.err" \
    --wrap="${ENV_SETUP} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -u ${REPO_DIR}/stage2_inject.py \
        --output-dir ${OUTPUT_DIR} \
        --config ${CONFIG} \
        --target-snr ${TARGET_SNR} \
        --snr-range ${SNR_LOW} ${SNR_HIGH} \
        --n-sims ${N_SIMS} \
        --n-chunks ${N_CHUNKS} \
        ${CGW_FLAG}"
)
S2_JOB=$(echo "$S2_JOB" | tr -d '[:space:]')
log "  Stage 2 job ID: ${S2_JOB}"

# =============================================================================
# DONE
# =============================================================================
log ""
log "Pipeline submitted successfully."
log ""
log "Dependency chain:"
log "  stage1 flat array (${S1_JOB})  [${N_S1_TASKS} tasks — all must succeed]"
log "    └─ stage2 single job (${S2_JOB})  [combines all ${N_SIMS} sim(s)]"
log ""
log "Monitor with:"
log "  squeue -j ${S1_JOB},${S2_JOB}"
log "  tail -f ${OUTPUT_DIR}/logs/stage2_${S2_JOB}.out"
log ""
log "Output directory:"
log "  ${OUTPUT_DIR}/"
log "    sim000/ ... sim$(printf '%03d' $(( N_SIMS - 1 )))/"
log "      populations/  — shards updated by stage 2"
log "      stoas/        — per-chunk TOA deltas from stage 1"
log "      metadata/     — config.json"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE