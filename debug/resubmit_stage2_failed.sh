#!/usr/bin/env bash
# =============================================================================
# resubmit_stage2_failed.sh
#
# One-off recovery script for the "Seed must be between 0 and 2**32 - 1"
# stage-2 failure (fixed in stage2_inject.py — noise_seed now comes from the
# already-bounded SeedSequence spawn instead of the overflow-prone
# `base + sim_id**2 * 10000` formula).
#
# Stage 1 for these sims already completed successfully — only stage 2 needs
# to be rerun, and only for the sims that actually failed. This script:
#
#   1. Scans <output_dir>/sim###/ for the given sim-id range.
#   2. Skips any sim whose stage 1 output isn't fully there (populations/
#      doesn't have n_chunks * n_sub_chunks shard files) — resubmitting
#      stage2 for those wouldn't help; rerun stage1 for them separately.
#   3. Skips any sim that already has metadata/stage2_complete.json
#      (already succeeded — don't waste a job on it).
#   4. Submits a stage-2-only sbatch job (via the FIXED stage2_inject.py)
#      for every remaining sim.
#
# IMPORTANT: point --stage2-script at the patched stage2_inject.py (the one
# with the noise_seed fix) — NOT the original REPO_DIR copy — unless you've
# already overwritten it there.
#
# Usage (matches the flags from your original submit_main_HPC.sh call):
#
#   bash resubmit_stage2_failed.sh \
#     --output-dir   /fred/oz005/users/bhoulden/SMBHB_population_injections/runs/2026-07-19_pessimistic \
#     --stage2-script /path/to/fixed/stage2_inject.py \
#     --config pessimistic \
#     --snr-range 3.0 3.4 \
#     --n-chunks 8 --n-sub-chunks 6 \
#     --cgw --synthetic-ptas \
#     --s2-time "05:00:00" --s2-mem "20G" \
#     --sim-start 500 --sim-end 775 \
#     --noise-seed-base 26072001
#
#   Add --dry-run first to see exactly what would be submitted.
# =============================================================================

set -eo pipefail

REPO_DIR="/fred/oz005/users/bhoulden/SMBHB_population_injections"
ENV_SETUP="module purge; unset PYTHONPATH; module load mamba; mamba activate smbhb312;"

# ── defaults (mirrors submit_main_HPC.sh) ────────────────────────────────────
OUTPUT_DIR=""
STAGE2_SCRIPT="${REPO_DIR}/stage2_inject.py"   # override with --stage2-script
CONFIG="pessimistic"
TARGET_SNR="3.2"
SNR_LOW="3.0"
SNR_HIGH="3.4"
N_CHUNKS=""
N_SUB_CHUNKS=1
CGW_FLAG=""
SYNTHETIC_PTAS_FLAG=""
SYNTHETIC_PTA_CONFIG=""
NOISE_SEED_BASE=26072001
N_TEST=1000
S2_CPUS=1; S2_MEM="20G"; S2_TIME="05:00:00"
SIM_START=""
SIM_END=""
DRY_RUN=0

# ── arg parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)            OUTPUT_DIR="$2";              shift 2 ;;
        --stage2-script)         STAGE2_SCRIPT="$2";           shift 2 ;;
        --config)                CONFIG="$2";                  shift 2 ;;
        --target-snr)            TARGET_SNR="$2";               shift 2 ;;
        --snr-range)              SNR_LOW="$2"; SNR_HIGH="$3";  shift 3 ;;
        --n-chunks)               N_CHUNKS="$2";                 shift 2 ;;
        --n-sub-chunks)           N_SUB_CHUNKS="$2";             shift 2 ;;
        --cgw)                    CGW_FLAG="--cgw";              shift   ;;
        --synthetic-ptas)         SYNTHETIC_PTAS_FLAG="--synthetic-ptas"; shift ;;
        --synthetic-pta-config)   SYNTHETIC_PTA_CONFIG="$2";     shift 2 ;;
        --noise-seed-base)        NOISE_SEED_BASE="$2";          shift 2 ;;
        --n-test)                 N_TEST="$2";                   shift 2 ;;
        --s2-time)                S2_TIME="$2";                  shift 2 ;;
        --s2-mem)                 S2_MEM="$2";                   shift 2 ;;
        --s2-cpus)                S2_CPUS="$2";                  shift 2 ;;
        --sim-start)              SIM_START="$2";                shift 2 ;;
        --sim-end)                SIM_END="$2";                  shift 2 ;;
        --dry-run)                DRY_RUN=1;                     shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" || -z "$N_CHUNKS" || -z "$SIM_START" || -z "$SIM_END" ]]; then
    echo "ERROR: --output-dir, --n-chunks, --sim-start, and --sim-end are required." >&2
    exit 1
fi
if [[ ! -f "$STAGE2_SCRIPT" ]]; then
    echo "ERROR: stage2 script not found: $STAGE2_SCRIPT" >&2
    echo "       Pass --stage2-script /path/to/fixed/stage2_inject.py" >&2
    exit 1
fi

SYN_CONFIG_ARG=""
if [[ -n "$SYNTHETIC_PTA_CONFIG" ]]; then
    SYN_CONFIG_ARG="--synthetic-pta-config '${SYNTHETIC_PTA_CONFIG}'"
fi

EXPECTED_SHARDS=$(( N_CHUNKS * N_SUB_CHUNKS ))
mkdir -p "${OUTPUT_DIR}/logs"

log() { echo "[resubmit] $*"; }

log "============================================================"
log "Scanning sim${SIM_START}..sim${SIM_END} under ${OUTPUT_DIR}"
log "Expecting ${EXPECTED_SHARDS} population shards per sim (n_chunks=${N_CHUNKS} × n_sub_chunks=${N_SUB_CHUNKS})"
log "Fixed stage2 script: ${STAGE2_SCRIPT}"
log "Dry run: ${DRY_RUN}"
log "============================================================"

N_SUBMITTED=0
N_SKIPPED_DONE=0
N_SKIPPED_NO_STAGE1=0
N_SKIPPED_IN_QUEUE=0

for sim_id in $(seq "$SIM_START" "$SIM_END"); do
    sim_id_padded=$(printf '%03d' "$sim_id")
    sim_dir="${OUTPUT_DIR}/sim${sim_id_padded}"
    pop_dir="${sim_dir}/populations"
    sentinel="${sim_dir}/metadata/stage2_complete.json"

    if [[ -f "$sentinel" ]]; then
        N_SKIPPED_DONE=$(( N_SKIPPED_DONE + 1 ))
        continue
    fi

    # Skip if a stage2 job for this sim is already pending/running in Slurm
    # (from the ORIGINAL submit_main_HPC.sh call). Those jobs will pick up
    # the fix automatically once you've overwritten stage2_inject.py on
    # disk with the patched version — no need to resubmit, and doing so
    # would double-run this sim.
    if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q "^s2_${sim_id_padded}_try"; then
        log "  sim${sim_id_padded}: SKIP — stage2 job already pending/running in Slurm queue"
        N_SKIPPED_IN_QUEUE=$(( N_SKIPPED_IN_QUEUE + 1 ))
        continue
    fi

    if [[ ! -d "$pop_dir" ]]; then
        log "  sim${sim_id_padded}: SKIP — no populations/ dir (stage1 didn't run)"
        N_SKIPPED_NO_STAGE1=$(( N_SKIPPED_NO_STAGE1 + 1 ))
        continue
    fi

    n_shards=$(find "$pop_dir" -maxdepth 1 -name 'subpop_*.pkl.gz' | wc -l)
    if [[ "$n_shards" -lt "$EXPECTED_SHARDS" ]]; then
        log "  sim${sim_id_padded}: SKIP — only ${n_shards}/${EXPECTED_SHARDS} stage1 shards present (stage1 incomplete)"
        N_SKIPPED_NO_STAGE1=$(( N_SKIPPED_NO_STAGE1 + 1 ))
        continue
    fi

    log "  sim${sim_id_padded}: stage1 OK (${n_shards}/${EXPECTED_SHARDS} shards), stage2 missing → resubmitting"

    CMD=(sbatch --parsable
        --job-name="s2fix_${sim_id_padded}"
        --nodes=1 --ntasks=1
        --cpus-per-task="${S2_CPUS}"
        --mem="${S2_MEM}"
        --time="${S2_TIME}"
        --output="${OUTPUT_DIR}/logs/stage2_sim${sim_id_padded}_seedfix_%j.out"
        --error="${OUTPUT_DIR}/logs/stage2_sim${sim_id_padded}_seedfix_%j.err"
        --wrap="${ENV_SETUP} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python -u ${STAGE2_SCRIPT} \
            --output-dir ${OUTPUT_DIR} \
            --config ${CONFIG} \
            --target-snr ${TARGET_SNR} \
            --snr-range ${SNR_LOW} ${SNR_HIGH} \
            --sim-id ${sim_id} \
            --n-chunks ${N_CHUNKS} \
            --n-sub-chunks ${N_SUB_CHUNKS} \
            --n-test ${N_TEST} \
            --noise-seed-base ${NOISE_SEED_BASE} \
            ${CGW_FLAG} \
            ${SYNTHETIC_PTAS_FLAG} \
            ${SYN_CONFIG_ARG}"
    )

    if [[ $DRY_RUN -eq 1 ]]; then
        log "    DRY-RUN: ${CMD[*]}"
    else
        JOB_ID=$("${CMD[@]}")
        log "    submitted job ${JOB_ID}"
    fi
    N_SUBMITTED=$(( N_SUBMITTED + 1 ))
done

log "============================================================"
log "Done. Submitted=${N_SUBMITTED}  already-complete(skipped)=${N_SKIPPED_DONE}  in-queue(skipped)=${N_SKIPPED_IN_QUEUE}  no-stage1(skipped)=${N_SKIPPED_NO_STAGE1}"
log "============================================================"


# sbatch resubmit_stage2_failed.sh \
#     --output-dir   /fred/oz005/users/bhoulden/SMBHB_population_injections/runs/2026-07-19_realistic \
#     --stage2-script ./stage2_inject.py \
#     --config realistic \
#     --snr-range 3.0 3.4 \
#     --n-chunks 8 --n-sub-chunks 6 \
#     --cgw --synthetic-ptas \
#     --s2-time "06:00:00" --s2-mem "20G" \
#     --sim-start 0 --sim-end 775 \
#     --noise-seed-base 26072001