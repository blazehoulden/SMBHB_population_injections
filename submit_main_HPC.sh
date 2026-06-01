#!/usr/bin/env bash
# =============================================================================
# submit_main_HPC.sh  —  SMBHB population analysis Slurm orchestrator
#
# Pipeline (repeated N_SIMS times independently):
#
#   stage1 flat array  (N_CHUNKS tasks per sim)
#     each job generates one chunk + TOA deltas for all enabled scenarios
#     └─afterok──> stage2 single job per sim
#                  waits for that sim's stage-1 tasks, then combines chunks,
#                  scales SNR, computes CGW SNRs for all scenarios, writes
#                  per-scenario fields into shards
#
#   Sims pipeline independently — sim001's stage-2 starts as soon as sim001's
#   chunks finish, without waiting for any other sim.
#
#   If any stage-2 fails → retry chain (up to MAX_RETRIES times):
#     clean stale sim dirs → rerun stage-1 array → rerun stage-2
#   All logs (including retries) saved to <output_dir>/logs/
#
# Synthetic PTA scenarios
# ───────────────────────
#   --synthetic-ptas       enable the three default scenarios:
#                            5x_cadence | 4x_precision | 5x_cad_4x_prec
#   --synthetic-pta-config JSON string overriding default scenario definitions
#                          e.g. '{"my_scenario": {"cadence_factor": 3,
#                                                  "toaerr_factor": 0.5,
#                                                  "best_only": true}}'
#   Both flags are forwarded identically to stage1 and stage2.
#
# Output layout:
#   <output_dir>/
#     sim000/
#       populations/   subpop_000.pkl.gz ... subpop_{N_CHUNKS-1:03d}.pkl.gz
#       stoas/         chunk_0000.npz  chunk_0000_5x_cadence.npz  ...
#       residuals/     noise/ population/ combined/  (baseline)
#       residuals_5x_cadence/  ...                   (synthetic scenarios)
#       metadata/      config.json  stage2_complete.json
#     sim001/
#       ...
#     logs/
#       stage1_sim000_try1_<array_id>_<task>.out/err
#       stage2_sim000_try1_<job_id>.out/err
#       clean_attempt2_<job_id>.out/err              ← only on retry
#       ...
#
# Usage
# ─────
#   bash submit_main_HPC.sh [OPTIONS]
#
# Options
#   --config              population config name               [optimistic]
#   --target-snr          target OS SNR                        [3.75]
#   --snr-range           SNR acceptance window (low high)     [3.5 4.0]
#   --simulations         number of independent populations    [400]
#   --n-chunks            chunks per simulation                [10]
#   --chunk-size          binaries per chunk                   [1000000]
#   --output-dir          root directory for all outputs       [auto]
#   --cgw / --no-cgw      enable/disable CGW analysis          [on]
#   --synthetic-ptas      enable synthetic PTA scenarios       [off]
#   --synthetic-pta-config  JSON scenario override             [none]
#   --noise-seed-base     base for per-sim noise seeds         [0]
#                         sim noise seed = base + sim_id * 1000
#   --max-retries         max full pipeline retries            [2]
#   --dry-run             print sbatch commands only           [off]
#   --proxy-only          validate proxy only (no scaling)     [off]
#   --n-test              binaries for proxy validation        [1000]
#
# Stage resource overrides
#   --s1-time / --s2-time   wall-clock limits
#   --s1-mem  / --s2-mem    memory per job
#   --s1-cpus / --s2-cpus   CPUs per job
# =============================================================================

set -eo pipefail
PS1="${PS1:-}"

echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID:-n/a}"
echo "Job Name: ${SLURM_JOB_NAME:-n/a}"
echo "Node:     ${SLURM_NODELIST:-n/a}"
echo "Start:    $(date)"
echo "=========================================="
echo ""

ENV_SETUP="module purge; unset PYTHONPATH; module load mamba; mamba activate smbhb312;"
REPO_DIR="/fred/oz005/users/bhoulden/SMBHB_population_injections"

# =============================================================================
# DEFAULTS
# =============================================================================
CONFIG="${SMBHB_CONFIG:-optimistic}"
TARGET_SNR="${SMBHB_TARGET_SNR:-3.75}"
SNR_LOW="${SMBHB_SNR_LOW:-3.5}"
SNR_HIGH="${SMBHB_SNR_HIGH:-4.00}"
N_SIMS="${SMBHB_SIMULATIONS:-400}"
N_CHUNKS="${SMBHB_N_CHUNKS:-10}"
CHUNK_SIZE="${SMBHB_CHUNK_SIZE:-1000000}"
CGW_FLAG="${SMBHB_CGW_FLAG:---cgw}"
SYNTHETIC_PTAS_FLAG=""          # empty = disabled
SYNTHETIC_PTA_CONFIG=""         # empty = use defaults when --synthetic-ptas set
NOISE_SEED_BASE=0
MAX_RETRIES=2
DRY_RUN=0
PROXY_ONLY_FLAG=""
N_TEST="${SMBHB_N_TEST:-1000}"

# Stage 1: pop synthesis + NUFFT (all scenarios done inside one task).
# Time is higher than the old script to account for synthetic scenario NUFFTs.
S1_CPUS=1;  S1_MEM="14G";  S1_TIME="00:30:00"

# Stage 2: noise sim + SNR scaling + CGW for all scenarios.
# Memory is higher than the old script because each synthetic scenario loads
# a full Enterprise PTA object (only one at a time, but they're large).
S2_CPUS=1;  S2_MEM="30G";  S2_TIME="03:00:00"

# Max simultaneous stage-1 array tasks (across all sims)
S1_MAX_CONCURRENT=100

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)                CONFIG="$2";                      shift 2 ;;
        --target-snr)            TARGET_SNR="$2";                  shift 2 ;;
        --snr-range)             SNR_LOW="$2"; SNR_HIGH="$3";      shift 3 ;;
        --simulations)           N_SIMS="$2";                      shift 2 ;;
        --n-chunks)              N_CHUNKS="$2";                    shift 2 ;;
        --chunk-size)            CHUNK_SIZE="$2";                  shift 2 ;;
        --output-dir)            OUTPUT_DIR="$2";                  shift 2 ;;
        --cgw)                   CGW_FLAG="--cgw";                 shift   ;;
        --no-cgw)                CGW_FLAG="";                      shift   ;;
        --synthetic-ptas)        SYNTHETIC_PTAS_FLAG="--synthetic-ptas"; shift ;;
        --synthetic-pta-config)  SYNTHETIC_PTA_CONFIG="$2";        shift 2 ;;
        --noise-seed-base)       NOISE_SEED_BASE="$2";             shift 2 ;;
        --max-retries)           MAX_RETRIES="$2";                 shift 2 ;;
        --dry-run)               DRY_RUN=1;                        shift   ;;
        --s1-time)               S1_TIME="$2";                     shift 2 ;;
        --s2-time)               S2_TIME="$2";                     shift 2 ;;
        --s1-mem)                S1_MEM="$2";                      shift 2 ;;
        --s2-mem)                S2_MEM="$2";                      shift 2 ;;
        --s1-cpus)               S1_CPUS="$2";                     shift 2 ;;
        --s2-cpus)               S2_CPUS="$2";                     shift 2 ;;
        --proxy-only)            PROXY_ONLY_FLAG="--proxy-only";   shift   ;;
        --n-test)                N_TEST="$2";                      shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Auto-generate output dir if not set
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    DATE=$(date +%Y-%m-%d)
    OUTPUT_DIR="${REPO_DIR}/runs/${DATE}_${CONFIG}"
fi
mkdir -p "${OUTPUT_DIR}/logs"

# Build the optional --synthetic-pta-config passthrough.
# Single-quote the JSON so it survives bash expansion inside --wrap strings.
SYN_CONFIG_ARG=""
if [[ -n "$SYNTHETIC_PTA_CONFIG" ]]; then
    SYN_CONFIG_ARG="--synthetic-pta-config '${SYNTHETIC_PTA_CONFIG}'"
fi

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[orchestrator] $*"; }

run_sbatch() {
    local desc="$1"; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "DRY-RUN [$desc]: sbatch $*" >&2
        echo "FAKE_${desc^^}"
    else
        sbatch --parsable "$@"
    fi
}

# Submit one stage-1 array + one stage-2 job PER SIM for a given attempt.
#
# $1 = attempt number (1-based)
# $2 = dependency string for the optional pre-clean job (empty on attempt 1)
#
# Prints a colon-separated list of all stage-2 job IDs to stdout so the
# caller can build the retry dependency string.
submit_attempt() {
    local attempt="$1"
    local s1_dep="$2"
    local ALL_S2_JOBS=()

    log "──────────────────────────────────────────────"
    log "Submitting attempt ${attempt} / $(( MAX_RETRIES + 1 ))"
    log "──────────────────────────────────────────────"

    # On retries, run a pre-clean job that removes incomplete sim dirs so
    # stage1 regenerates them cleanly.
    local clean_dep_flag=""
    if [[ $attempt -gt 1 && -n "$s1_dep" ]]; then
        log "  Submitting pre-clean job for attempt ${attempt}..."
        CLEAN_JOB=$(run_sbatch "clean_attempt${attempt}" \
            --job-name="smbhb_clean_try${attempt}" \
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

    # ── One stage-1 array + one stage-2 job per sim ───────────────────────────
    # Each stage-2 depends only on its own sim's stage-1 array, so sims
    # pipeline independently (sim001's stage-2 doesn't wait for sim099's
    # stage-1 to finish).
    for sim_id in $(seq 0 $(( N_SIMS - 1 ))); do
        sim_id_padded=$(printf '%03d' "$sim_id")
        noise_seed=$(( NOISE_SEED_BASE + sim_id * 1000 ))

        local s1_dep_flag=""
        [[ -n "$clean_dep_flag" ]] && s1_dep_flag="--dependency=${clean_dep_flag}"

        S1_JOB=$(run_sbatch "s1_sim${sim_id_padded}_try${attempt}" \
            --job-name="s1_${sim_id_padded}_try${attempt}" \
            --nodes=1 --ntasks=1 \
            --cpus-per-task="${S1_CPUS}" \
            --mem="${S1_MEM}" \
            --time="${S1_TIME}" \
            --array="0-$(( N_CHUNKS - 1 ))%${S1_MAX_CONCURRENT}" \
            --requeue \
            ${s1_dep_flag} \
            --output="${OUTPUT_DIR}/logs/stage1_sim${sim_id_padded}_try${attempt}_%A_%a.out" \
            --error="${OUTPUT_DIR}/logs/stage1_sim${sim_id_padded}_try${attempt}_%A_%a.err" \
            --wrap="${ENV_SETUP} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                python -u ${REPO_DIR}/stage1_setup.py \
                --config ${CONFIG} \
                --target-snr ${TARGET_SNR} \
                --snr-range ${SNR_LOW} ${SNR_HIGH} \
                --chunk-size ${CHUNK_SIZE} \
                --n-chunks ${N_CHUNKS} \
                --output-dir ${OUTPUT_DIR} \
                --sim-id ${sim_id} \
                --task-id \$SLURM_ARRAY_TASK_ID \
                ${SYNTHETIC_PTAS_FLAG} \
                ${SYN_CONFIG_ARG}"
        )
        S1_JOB=$(echo "$S1_JOB" | tr -d '[:space:]')

        # Stage-2 for this sim — depends only on its own stage-1 array finishing
        S2_JOB=$(run_sbatch "s2_sim${sim_id_padded}_try${attempt}" \
            --job-name="s2_${sim_id_padded}_try${attempt}" \
            --nodes=1 --ntasks=1 \
            --cpus-per-task="${S2_CPUS}" \
            --mem="${S2_MEM}" \
            --time="${S2_TIME}" \
            --dependency="afterok:${S1_JOB}_*" \
            --output="${OUTPUT_DIR}/logs/stage2_sim${sim_id_padded}_try${attempt}_%j.out" \
            --error="${OUTPUT_DIR}/logs/stage2_sim${sim_id_padded}_try${attempt}_%j.err" \
            --wrap="${ENV_SETUP} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                python -u ${REPO_DIR}/stage2_inject.py \
                --output-dir ${OUTPUT_DIR} \
                --config ${CONFIG} \
                --target-snr ${TARGET_SNR} \
                --snr-range ${SNR_LOW} ${SNR_HIGH} \
                --sim-id ${sim_id} \
                --n-chunks ${N_CHUNKS} \
                --n-test ${N_TEST} \
                --noise-seed ${noise_seed} \
                ${CGW_FLAG} \
                ${PROXY_ONLY_FLAG} \
                ${SYNTHETIC_PTAS_FLAG} \
                ${SYN_CONFIG_ARG}"
        )
        S2_JOB=$(echo "$S2_JOB" | tr -d '[:space:]')
        ALL_S2_JOBS+=("$S2_JOB")

        log "  sim${sim_id_padded}: s1=${S1_JOB}  s2=${S2_JOB}  noise_seed=${noise_seed}"
    done

    # Return colon-separated list of all s2 job IDs for the retry dependency
    local joined
    joined=$(IFS=:; echo "${ALL_S2_JOBS[*]}")
    echo "$joined"
}

# =============================================================================
# SUMMARY
# =============================================================================
log "============================================================"
log "SMBHB Slurm pipeline"
log "  Repo          : ${REPO_DIR}"
log "  Output        : ${OUTPUT_DIR}"
log "  Config        : ${CONFIG}"
log "  Target SNR    : ${TARGET_SNR}  range=[${SNR_LOW}, ${SNR_HIGH}]"
log "  Simulations   : ${N_SIMS}"
log "  Chunks/sim    : ${N_CHUNKS}  ×  ${CHUNK_SIZE} binaries"
log "  Total pop/sim : $(( N_CHUNKS * CHUNK_SIZE )) binaries"
log "  S1 tasks/sim  : ${N_CHUNKS}  (one array per sim)"
log "  S2 jobs       : ${N_SIMS}  (one per sim, independent)"
log "  CGW           : $([[ -n "${CGW_FLAG}" ]] && echo on || echo off)"
log "  Synthetic PTAs: $([[ -n "${SYNTHETIC_PTAS_FLAG}" ]] && echo on || echo off)"
[[ -n "$SYNTHETIC_PTA_CONFIG" ]] && log "  Syn config    : ${SYNTHETIC_PTA_CONFIG}"
log "  Noise seed    : base=${NOISE_SEED_BASE}  (per sim: base + sim_id × 1000)"
log "  Proxy only    : $([[ -n "${PROXY_ONLY_FLAG}" ]] && echo on || echo off)"
log "  N test        : ${N_TEST}"
log "  Max retries   : ${MAX_RETRIES}"
log "  Dry run       : ${DRY_RUN}"
log "  S1 resources  : cpus=${S1_CPUS}  mem=${S1_MEM}  time=${S1_TIME}"
log "  S2 resources  : cpus=${S2_CPUS}  mem=${S2_MEM}  time=${S2_TIME}"
log "============================================================"

# =============================================================================
# SUBMIT
# =============================================================================
PREV_S2_JOB=$(submit_attempt 1 "")

for attempt in $(seq 2 $(( MAX_RETRIES + 1 ))); do
    PREV_S2_JOB=$(submit_attempt "$attempt" "afternotok:${PREV_S2_JOB}")
done

# =============================================================================
# DONE
# =============================================================================
log ""
log "Pipeline submitted."
log ""
log "Dependency chain:"
log "  attempt 1: stage1[sim] → stage2[sim]  (runs immediately, per-sim)"
for attempt in $(seq 2 $(( MAX_RETRIES + 1 ))); do
    log "  attempt ${attempt}: clean → stage1[sim] → stage2[sim]  (only if attempt $(( attempt - 1 )) has any failures)"
done
log ""
log "Usage examples:"
log "  # Baseline only:"
log "  bash submit_main_HPC.sh --cgw"
log ""
log "  # Default 3 synthetic scenarios (5x cadence, 4x precision, combined):"
log "  bash submit_main_HPC.sh --cgw --synthetic-ptas"
log ""
log "  # Custom scenario via JSON:"
log "  bash submit_main_HPC.sh --cgw --synthetic-ptas \\"
log "    --synthetic-pta-config '{\"3x_cadence\": {\"cadence_factor\": 3, \"toaerr_factor\": 1.0, \"best_only\": true}}'"
log ""
log "Monitor:"
log "  squeue -u \$USER"
log "  tail -f ${OUTPUT_DIR}/logs/stage2_sim000_try1_*.out"
log ""
log "Shard SNR fields written per sim:"
log "  cgw_snr                  (baseline, always)"
[[ -n "${SYNTHETIC_PTAS_FLAG}" ]] && {
    if [[ -n "$SYNTHETIC_PTA_CONFIG" ]]; then
        log "  cgw_snr_<your_scenario>  (from --synthetic-pta-config)"
    else
        log "  cgw_snr_5x_cadence"
        log "  cgw_snr_4x_precision"
        log "  cgw_snr_5x_cad_4x_prec"
    fi
}

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "Finished: $(date)  exit=${EXIT_CODE}"
echo "=========================================="
exit $EXIT_CODE