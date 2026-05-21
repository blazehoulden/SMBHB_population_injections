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
#   If stage2 fails → resubmit stage1 array + stage2 (up to MAX_RETRIES times)
#   Each retry cleans sim directories so stage1 regenerates cleanly.
#   All logs (including retries) saved to <output_dir>/logs/
#
# Output layout:
#   <output_dir>/
#     sim000/
#       populations/   subpop_000.pkl.gz ... subpop_{N_CHUNKS-1:03d}.pkl.gz
#       stoas/         sim0000/ ... sim{N_CHUNKS-1:04d}/
#       metadata/      config.json
#     sim001/
#       ...
#     logs/
#       stage1_<attempt>_<array_id>_<task_id>.out/err
#       stage2_<attempt>_<job_id>.out/err
#       orchestrator_<job_id>.out   ← this script's own log
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
#   --max-retries     max full pipeline retries            [2]
#   --dry-run         print sbatch commands only           [off]
#   --n-test          number of binaries to proxy test     [1000]
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
TARGET_SNR="${SMBHB_TARGET_SNR:-3.75}"
SNR_LOW="${SMBHB_SNR_LOW:-3.5}"
SNR_HIGH="${SMBHB_SNR_HIGH:-4.00}"
N_SIMS="${SMBHB_SIMULATIONS:-400}"
N_CHUNKS="${SMBHB_N_CHUNKS:-10}"
CHUNK_SIZE="${SMBHB_CHUNK_SIZE:-1000000}"
CGW_FLAG="${SMBHB_CGW_FLAG:---cgw}"
MAX_RETRIES=2
DRY_RUN=0
PROXY_ONLY_FLAG=""
N_TEST="${SMBHB_N_TEST:-1000}"

# Stage 1: one chunk per task — pop synthesis + NUFFT
S1_CPUS=1;  S1_MEM="14G";  S1_TIME="00:15:00"

# Stage 2: single job — loads all chunks for all sims, Enterprise SNR loop
S2_CPUS=1;  S2_MEM="40G";  S2_TIME="01:30:00"

# Max simultaneous stage-1 array tasks
S1_MAX_CONCURRENT=100

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2";                      shift 2 ;;
        --target-snr)    TARGET_SNR="$2";                  shift 2 ;;
        --snr-range)     SNR_LOW="$2"; SNR_HIGH="$3";      shift 3 ;;
        --simulations)   N_SIMS="$2";                      shift 2 ;;
        --n-chunks)      N_CHUNKS="$2";                    shift 2 ;;
        --chunk-size)    CHUNK_SIZE="$2";                  shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";                  shift 2 ;;
        --cgw)           CGW_FLAG="--cgw";                 shift   ;;
        --no-cgw)        CGW_FLAG="";                      shift   ;;
        --max-retries)   MAX_RETRIES="$2";                 shift 2 ;;
        --dry-run)       DRY_RUN=1;                        shift   ;;
        --s1-time)       S1_TIME="$2";                     shift 2 ;;
        --s2-time)       S2_TIME="$2";                     shift 2 ;;
        --s1-mem)        S1_MEM="$2";                      shift 2 ;;
        --s2-mem)        S2_MEM="$2";                      shift 2 ;;
        --s1-cpus)       S1_CPUS="$2";                     shift 2 ;;
        --s2-cpus)       S2_CPUS="$2";                     shift 2 ;;
        --proxy-only)    PROXY_ONLY_FLAG="--proxy-only";   shift   ;;
        --n-test)        N_TEST="$2";                      shift 2 ;;
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

# Submit a stage-1 array + stage-2 pair for a given attempt number.
# $1 = attempt number (1-based)
# $2 = dependency string for stage-1 (empty = no dependency, i.e. run immediately)
# Prints the stage-2 job ID to stdout.
submit_attempt() {
    local attempt="$1"
    local s1_dep="$2"

    log "──────────────────────────────────────────────"
    log "Submitting attempt ${attempt} / $(( MAX_RETRIES + 1 ))"
    log "──────────────────────────────────────────────"

    # Clean job on retry (same as before)
    local clean_dep_flag=""
    if [[ $attempt -gt 1 && -n "$s1_dep" ]]; then
        log "  Submitting pre-clean job for attempt ${attempt}..."
        CLEAN_JOB=$(run_sbatch "clean_attempt${attempt}" \
            --job-name="smbhb_clean_${CONFIG}_try${attempt}" \
            --nodes=1 --ntasks=1 --cpus-per-task=1 \
            --mem="2G" --time="00:05:00" \
            --dependency="${s1_dep}" \
            --output="${OUTPUT_DIR}/logs/clean_attempt${attempt}_%j.out" \
            --error="${OUTPUT_DIR}/logs/clean_attempt${attempt}_%j.err" \
            --wrap="${ENV_SETUP} python -u ${REPO_DIR}/stage2_inject.py \
                --output-dir ${OUTPUT_DIR} \
                --n-chunks ${N_CHUNKS} \
                --config ${CONFIG} \
                --target-snr ${TARGET_SNR} \
                --snr-range ${SNR_LOW} ${SNR_HIGH} \
                --clean-failed"
        )
        CLEAN_JOB=$(echo "$CLEAN_JOB" | tr -d '[:space:]')
        log "  Clean job ID: ${CLEAN_JOB}"
        clean_dep_flag="afterok:${CLEAN_JOB}"
    fi

    # ── Submit one stage-1 array + one stage-2 job PER SIM ───────────────────
    # Each stage-2 job depends only on its own sim's stage-1 tasks,
    # so sims can pipeline independently rather than all waiting for each other.
    ALL_S2_JOBS=()

    for sim_id in $(seq 0 $(( N_SIMS - 1 ))); do
        sim_id_padded=$(printf '%03d' "$sim_id")

        # Stage-1 array for this sim only: chunk_id = 0 .. N_CHUNKS-1
        local s1_dep_flag=""
        if [[ -n "$clean_dep_flag" ]]; then
            s1_dep_flag="--dependency=${clean_dep_flag}"
        fi

        S1_JOB=$(run_sbatch "s1_sim${sim_id_padded}_attempt${attempt}" \
            --job-name="s1_${sim_id_padded}_try${attempt}" \
            --nodes=1 --ntasks=1 \
            --cpus-per-task="${S1_CPUS}" \
            --mem="${S1_MEM}" \
            --time="${S1_TIME}" \
            --array="0-$(( N_CHUNKS - 1 ))%${S1_MAX_CONCURRENT}" \
            --requeue \
            ${s1_dep_flag} \
            --output="${OUTPUT_DIR}/logs/stage1_sim${sim_id_padded}_attempt${attempt}_%A_%a.out" \
            --error="${OUTPUT_DIR}/logs/stage1_sim${sim_id_padded}_attempt${attempt}_%A_%a.err" \
            --wrap="${ENV_SETUP} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                python -u ${REPO_DIR}/stage1_setup.py \
                --config ${CONFIG} \
                --target-snr ${TARGET_SNR} \
                --snr-range ${SNR_LOW} ${SNR_HIGH} \
                --chunk-size ${CHUNK_SIZE} \
                --n-chunks ${N_CHUNKS} \
                --output-dir ${OUTPUT_DIR} \
                --sim-id ${sim_id} \
                --task-id \$SLURM_ARRAY_TASK_ID"
        )
        S1_JOB=$(echo "$S1_JOB" | tr -d '[:space:]')

        # Stage-2 for this sim — depends only on its own stage-1 array
        S2_JOB=$(run_sbatch "s2_sim${sim_id_padded}_attempt${attempt}" \
            --job-name="s2_${sim_id_padded}_try${attempt}" \
            --nodes=1 --ntasks=1 \
            --cpus-per-task="${S2_CPUS}" \
            --mem="${S2_MEM}" \
            --time="${S2_TIME}" \
            --dependency="afterok:${S1_JOB}_*" \
            --output="${OUTPUT_DIR}/logs/stage2_sim${sim_id_padded}_attempt${attempt}_%j.out" \
            --error="${OUTPUT_DIR}/logs/stage2_sim${sim_id_padded}_attempt${attempt}_%j.err" \
            --wrap="${ENV_SETUP} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                python -u ${REPO_DIR}/stage2_inject.py \
                --output-dir ${OUTPUT_DIR} \
                --config ${CONFIG} \
                --target-snr ${TARGET_SNR} \
                --snr-range ${SNR_LOW} ${SNR_HIGH} \
                --sim-id ${sim_id} \
                --n-chunks ${N_CHUNKS} \
                --n-test ${N_TEST} \
                ${CGW_FLAG} \
                ${PROXY_ONLY_FLAG}"
        )
        S2_JOB=$(echo "$S2_JOB" | tr -d '[:space:]')
        ALL_S2_JOBS+=("$S2_JOB")

        log "  sim${sim_id_padded}: s1=${S1_JOB}  s2=${S2_JOB}"
    done

    # Return colon-separated list of all s2 job IDs for retry dependency
    local joined
    joined=$(IFS=:; echo "${ALL_S2_JOBS[*]}")
    echo "$joined"
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
log "  Max retries   : ${MAX_RETRIES}"
log "  CGW           : $([[ -n "${CGW_FLAG}" ]] && echo on || echo off)"
log "  Proxy only    : $([[ -n "${PROXY_ONLY_FLAG}" ]] && echo on || echo off)"
log "  N test        : ${N_TEST}"
log "  Dry run       : ${DRY_RUN}"
log "============================================================"

# =============================================================================
# SUBMIT ATTEMPT 1 — no dependency, runs immediately after stage 1
# =============================================================================
PREV_S2_JOB=$(submit_attempt 1 "")

# =============================================================================
# SUBMIT RETRIES — each depends on previous stage-2 failing
# Each retry: clean stale data → rerun stage1 → rerun stage2
# =============================================================================
for attempt in $(seq 2 $(( MAX_RETRIES + 1 ))); do
    PREV_S2_JOB=$(submit_attempt "$attempt" "afternotok:${PREV_S2_JOB}")
done

# =============================================================================
# DONE
# =============================================================================
log ""
log "Pipeline submitted successfully."
log ""
log "Dependency chain:"
log "  attempt 1: stage1 → stage2  (runs immediately)"
for attempt in $(seq 2 $(( MAX_RETRIES + 1 ))); do
    log "  attempt ${attempt}: clean → stage1 → stage2  (only if attempt $(( attempt - 1 )) fails)"
done
log ""
log "Logs (all attempts):"
log "  ${OUTPUT_DIR}/logs/"
log "    stage1_attempt1_<array_id>_<task>.out"
log "    stage2_attempt1_<job_id>.out"
log "    stage1_attempt2_<array_id>_<task>.out   ← only written if attempt 1 fails"
log "    stage2_attempt2_<job_id>.out"
log "    ..."
log ""
log "Monitor with:"
log "  squeue -u \$USER"
log "  tail -f ${OUTPUT_DIR}/logs/stage2_attempt1_*.out"
log ""
log "Output directory:"
log "  ${OUTPUT_DIR}/"
log "    sim000/ ... sim$(printf '%03d' $(( N_SIMS - 1 )))/"
log "      populations/  — shards updated by stage 2"
log "      stoas/        — per-chunk TOA deltas from stage 1"
log "      metadata/     — config.json  +  stage2_complete.json (on success)"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE