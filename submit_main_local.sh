#!/usr/bin/env bash
# submit_main_local.sh — local (non-Slurm) driver for stage1+stage2.
# Mirrors submit_main_HPC.sh's flags/defaults 1:1 so the two are interchangeable.

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${SMBHB_CONFIG:-optimistic}"
TARGET_SNR="${SMBHB_TARGET_SNR:-3.2}"
SNR_LOW="${SMBHB_SNR_LOW:-3.0}"
SNR_HIGH="${SMBHB_SNR_HIGH:-3.40}"
N_SIMS="${SMBHB_SIMULATIONS:-1}"
N_CHUNKS="${SMBHB_N_CHUNKS:-2}"
CHUNK_SIZE="${SMBHB_CHUNK_SIZE:-50000}"
N_SUB_CHUNKS=1
CGW_FLAG="--cgw"
SYNTHETIC_PTAS_FLAG=""
SYNTHETIC_PTA_CONFIG=""
NOISE_SEED_BASE=26072001
PROXY_ONLY_FLAG=""
N_TEST=1000
SIM_START=0
SIM_END=""
N_JOBS_PARALLEL=1   # how many stage1 chunk subprocesses to run at once

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)                CONFIG="$2";                      shift 2 ;;
        --target-snr)            TARGET_SNR="$2";                  shift 2 ;;
        --snr-range)             SNR_LOW="$2"; SNR_HIGH="$3";      shift 3 ;;
        --simulations)           N_SIMS="$2";                      shift 2 ;;
        --n-chunks)              N_CHUNKS="$2";                    shift 2 ;;
        --chunk-size)            CHUNK_SIZE="$2";                  shift 2 ;;
        --n-sub-chunks)          N_SUB_CHUNKS="$2";                shift 2 ;;
        --output-dir)            OUTPUT_DIR="$2";                  shift 2 ;;
        --cgw)                   CGW_FLAG="--cgw";                 shift   ;;
        --no-cgw)                CGW_FLAG="";                      shift   ;;
        --synthetic-ptas)        SYNTHETIC_PTAS_FLAG="--synthetic-ptas"; shift ;;
        --synthetic-pta-config)  SYNTHETIC_PTA_CONFIG="$2";        shift 2 ;;
        --noise-seed-base)       NOISE_SEED_BASE="$2";             shift 2 ;;
        --proxy-only)            PROXY_ONLY_FLAG="--proxy-only";   shift   ;;
        --n-test)                N_TEST="$2";                      shift 2 ;;
        --sim-start)             SIM_START="$2";                   shift 2 ;;
        --sim-end)               SIM_END="$2";                     shift 2 ;;
        --jobs)                  N_JOBS_PARALLEL="$2";             shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${OUTPUT_DIR:-}" ]]; then
    DATE=$(date +%Y-%m-%d)
    OUTPUT_DIR="${REPO_DIR}/runs/${DATE}_${CONFIG}_local"
fi
mkdir -p "${OUTPUT_DIR}/logs"

CHUNK_SIZE_NUM="${CHUNK_SIZE//_/}"
if (( CHUNK_SIZE_NUM % N_SUB_CHUNKS != 0 )); then
    echo "ERROR: --chunk-size (${CHUNK_SIZE}) must be divisible by --n-sub-chunks (${N_SUB_CHUNKS})" >&2
    exit 1
fi

SYN_CONFIG_ARG=()
[[ -n "$SYNTHETIC_PTA_CONFIG" ]] && SYN_CONFIG_ARG=(--synthetic-pta-config "$SYNTHETIC_PTA_CONFIG")

[[ -z "${SIM_END:-}" ]] && SIM_END=$(( SIM_START + N_SIMS - 1 ))

log() { echo "[local-orchestrator] $*"; }

log "============================================================"
log "Output    : ${OUTPUT_DIR}"
log "Config    : ${CONFIG}   target_snr=${TARGET_SNR} range=[${SNR_LOW},${SNR_HIGH}]"
log "Sims      : ${SIM_START}..${SIM_END}   chunks/sim=${N_CHUNKS}  sub-chunks/task=${N_SUB_CHUNKS}"
log "Chunk size: ${CHUNK_SIZE_NUM}   total/sim=$(( N_CHUNKS * CHUNK_SIZE_NUM ))"
log "CGW       : $([[ -n "$CGW_FLAG" ]] && echo on || echo off)"
log "Parallel  : ${N_JOBS_PARALLEL} stage1 chunk(s) at once"
log "============================================================"

for sim_id in $(seq "$SIM_START" "$SIM_END"); do
    sim_padded=$(printf '%03d' "$sim_id")
    log "── sim${sim_padded}: stage1 (${N_CHUNKS} chunks) ──"

    run_chunk() {
        local chunk_id="$1"
        python -u "${REPO_DIR}/stage1_setup.py" \
            --config "${CONFIG}" \
            --target-snr "${TARGET_SNR}" \
            --snr-range "${SNR_LOW}" "${SNR_HIGH}" \
            --chunk-size "${CHUNK_SIZE_NUM}" \
            --n-chunks "${N_CHUNKS}" \
            --n-sub-chunks "${N_SUB_CHUNKS}" \
            --output-dir "${OUTPUT_DIR}" \
            --sim-id "${sim_id}" \
            --task-id "${chunk_id}" \
            --noise-seed-base "${NOISE_SEED_BASE}" \
            --n-sims "${N_SIMS}" \
            ${SYNTHETIC_PTAS_FLAG} \
            "${SYN_CONFIG_ARG[@]}" \
            > "${OUTPUT_DIR}/logs/stage1_sim${sim_padded}_chunk$(printf '%04d' "$chunk_id").out" \
            2> "${OUTPUT_DIR}/logs/stage1_sim${sim_padded}_chunk$(printf '%04d' "$chunk_id").err"
    }
    export -f run_chunk
    export REPO_DIR CONFIG TARGET_SNR SNR_LOW SNR_HIGH CHUNK_SIZE_NUM N_CHUNKS \
           N_SUB_CHUNKS OUTPUT_DIR sim_id NOISE_SEED_BASE N_SIMS \
           SYNTHETIC_PTAS_FLAG sim_padded
    SYN_CONFIG_ARG_STR="${SYN_CONFIG_ARG[*]:-}"
    export SYN_CONFIG_ARG_STR

    seq 0 $(( N_CHUNKS - 1 )) | xargs -P "${N_JOBS_PARALLEL}" -I{} bash -c 'run_chunk "$@"' _ {}

    # fail fast if any chunk log shows a nonzero exit — xargs swallows individual
    # exit codes, so check for the stage1 completion print in each log instead.
    for chunk_id in $(seq 0 $(( N_CHUNKS - 1 ))); do
        chunk_padded=$(printf '%04d' "$chunk_id")
        logfile="${OUTPUT_DIR}/logs/stage1_sim${sim_padded}_chunk${chunk_padded}.out"
        if ! grep -q "✅ Stage 1 chunk=" "$logfile" 2>/dev/null; then
            echo "ERROR: stage1 sim=${sim_id} chunk=${chunk_id} did not complete — see ${logfile%.out}.err" >&2
            exit 1
        fi
    done
    log "  ✓ all ${N_CHUNKS} chunks complete for sim${sim_padded}"

    log "── sim${sim_padded}: stage2 ──"
    python -u "${REPO_DIR}/stage2_inject.py" \
        --output-dir "${OUTPUT_DIR}" \
        --config "${CONFIG}" \
        --target-snr "${TARGET_SNR}" \
        --snr-range "${SNR_LOW}" "${SNR_HIGH}" \
        --sim-id "${sim_id}" \
        --n-chunks "${N_CHUNKS}" \
        --n-sub-chunks "${N_SUB_CHUNKS}" \
        --n-test "${N_TEST}" \
        --noise-seed-base "${NOISE_SEED_BASE}" \
        --n-sims "${N_SIMS}" \
        ${CGW_FLAG} \
        ${PROXY_ONLY_FLAG} \
        ${SYNTHETIC_PTAS_FLAG} \
        "${SYN_CONFIG_ARG[@]}" \
        2>&1 | tee "${OUTPUT_DIR}/logs/stage2_sim${sim_padded}.out"

    log "  ✓ sim${sim_padded} complete → ${OUTPUT_DIR}/sim${sim_padded}/summary.pkl.gz"
done

log ""
log "Done. Summaries:"
for sim_id in $(seq "$SIM_START" "$SIM_END"); do
    printf '  %s/sim%03d/summary.pkl.gz\n' "$OUTPUT_DIR" "$sim_id"
done