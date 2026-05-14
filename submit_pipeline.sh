#!/usr/bin/env bash
# =============================================================================
# submit_pipeline.sh  —  SMBHB population analysis Slurm orchestrator
#
# Submits stages 1–4 with proper dependency chaining:
#   stage1 (1 job)  ──afterok──>  build_task_map (1 job)
#                               ──afterok──>  stage2 array (N tasks)
#                                           ──afterok──>  stage3 array (1/pop)
#                                                       ──afterok──>  stage4 (1 job)
#
# Usage
# ─────
#   bash submit_pipeline.sh [OPTIONS]
#
# Options (all have defaults — edit the CONFIG section below or pass as env vars)
#   --config          population config name     [optimistic]
#   --target-snr      target OS SNR              [4.0]
#   --snr-range       SNR acceptance window      [3.5 4.25]
#   --simulations     number of populations      [10]
#   --chunk-size      binaries per injection job [1000000]
#   --output-dir      root directory for outputs [auto: runs/<date>_<config>]
#   --project         Slurm account/project      [$SLURM_ACCOUNT or "default"]
#   --partition       Slurm partition            [compute]
#   --cgw             add CGW analysis in stage3 [off]
#   --dry-run         print sbatch commands only [off]
#
# Environment variable overrides (same names, upper-case):
#   SMBHB_CONFIG, SMBHB_TARGET_SNR, SMBHB_SNR_RANGE, SMBHB_SIMULATIONS,
#   SMBHB_CHUNK_SIZE, SMBHB_OUTPUT_DIR, SMBHB_PROJECT, SMBHB_PARTITION
# =============================================================================

set -euo pipefail

# ── locate repository root (directory containing this script) ─────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# CONFIG DEFAULTS  (edit these to match your cluster / science needs)
# =============================================================================
CONFIG="${SMBHB_CONFIG:-optimistic}"
TARGET_SNR="${SMBHB_TARGET_SNR:-4.0}"
SNR_LOW="${SMBHB_SNR_LOW:-3.5}"
SNR_HIGH="${SMBHB_SNR_HIGH:-4.25}"
N_SIMS="${SMBHB_SIMULATIONS:-10}"
CHUNK_SIZE="${SMBHB_CHUNK_SIZE:-1000000}"
PROJECT="${SMBHB_PROJECT:-${SLURM_ACCOUNT:-default}}"
PARTITION="${SMBHB_PARTITION:-compute}"
PYTHON="${SMBHB_PYTHON:-python}"   # e.g. /opt/conda/envs/pta/bin/python
CGW_FLAG=""
DRY_RUN=0

# ── stage resource requests ───────────────────────────────────────────────────
# Tune these for your cluster.  Stage 2 is CPU-heavy (NUFFT); stage 3 is
# memory-heavy (Enterprise snapshot + OS computation).

S1_NODES=1;   S1_NTASKS=1;  S1_CPUS=32;  S1_MEM="64G";  S1_TIME="08:00:00"
S2_NODES=1;   S2_NTASKS=1;  S2_CPUS=4;   S2_MEM="16G";  S2_TIME="02:00:00"
S3_NODES=1;   S3_NTASKS=1;  S3_CPUS=16;  S3_MEM="64G";  S3_TIME="04:00:00"
S4_NODES=1;   S4_NTASKS=1;  S4_CPUS=1;   S4_MEM="4G";   S4_TIME="00:30:00"

# ── simultaneous array limits (avoid flooding the scheduler) ─────────────────
S2_MAX_CONCURRENT=200
S3_MAX_CONCURRENT=50

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2";      shift 2 ;;
        --target-snr)    TARGET_SNR="$2";  shift 2 ;;
        --snr-range)     SNR_LOW="$2"; SNR_HIGH="$3"; shift 3 ;;
        --simulations)   N_SIMS="$2";      shift 2 ;;
        --chunk-size)    CHUNK_SIZE="$2";  shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";  shift 2 ;;
        --project)       PROJECT="$2";     shift 2 ;;
        --partition)     PARTITION="$2";   shift 2 ;;
        --python)        PYTHON="$2";      shift 2 ;;
        --cgw)           CGW_FLAG="--cgw"; shift   ;;
        --dry-run)       DRY_RUN=1;        shift   ;;
        --s1-time)       S1_TIME="$2";     shift 2 ;;
        --s2-time)       S2_TIME="$2";     shift 2 ;;
        --s3-time)       S3_TIME="$2";     shift 2 ;;
        --s2-mem)        S2_MEM="$2";      shift 2 ;;
        --s3-mem)        S3_MEM="$2";      shift 2 ;;
        --s2-cpus)       S2_CPUS="$2";     shift 2 ;;
        --s3-cpus)       S3_CPUS="$2";     shift 2 ;;
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

sbatch_or_dry() {
    # Usage: sbatch_or_dry <description> <sbatch args...>
    local desc="$1"; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "DRY-RUN [$desc]: sbatch $*"
        echo "FAKE_JOB_${desc^^}"   # fake job ID
    else
        sbatch "$@"
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

# =============================================================================
# STAGE 1 — population synthesis
# =============================================================================
log "Submitting stage 1..."

S1_JOB=$(sbatch_or_dry "stage1" \
    --job-name="smbhb_s1_${CONFIG}" \
    --account="${PROJECT}" \
    --partition="${PARTITION}" \
    --nodes="${S1_NODES}" \
    --ntasks="${S1_NTASKS}" \
    --cpus-per-task="${S1_CPUS}" \
    --mem="${S1_MEM}" \
    --time="${S1_TIME}" \
    --output="${OUTPUT_DIR}/logs/stage1_%j.out" \
    --error="${OUTPUT_DIR}/logs/stage1_%j.err" \
    --parsable \
    --wrap="${PYTHON} ${REPO_DIR}/stage1_setup.py \
        --config ${CONFIG} \
        --target-snr ${TARGET_SNR} \
        --snr-range ${SNR_LOW} ${SNR_HIGH} \
        --simulations ${N_SIMS} \
        --chunk-size ${CHUNK_SIZE} \
        --output-dir ${OUTPUT_DIR}"
)
S1_JOB=$(echo "$S1_JOB" | tr -d '[:space:]')
log "  Stage 1 job ID: ${S1_JOB}"

# =============================================================================
# TASK MAP BUILD — runs after stage 1, writes task_map.json
# =============================================================================
log "Submitting task-map builder (depends on stage 1)..."

MAP_JOB=$(sbatch_or_dry "task_map" \
    --job-name="smbhb_map_${CONFIG}" \
    --account="${PROJECT}" \
    --partition="${PARTITION}" \
    --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem="2G" \
    --time="00:10:00" \
    --dependency="afterok:${S1_JOB}" \
    --output="${OUTPUT_DIR}/logs/task_map_%j.out" \
    --error="${OUTPUT_DIR}/logs/task_map_%j.err" \
    --parsable \
    --wrap="${PYTHON} ${REPO_DIR}/build_task_map.py \
        --output-dir ${OUTPUT_DIR} \
        > ${OUTPUT_DIR}/logs/task_map_array_arg.txt 2>&1"
)
MAP_JOB=$(echo "$MAP_JOB" | tr -d '[:space:]')
log "  Task-map job ID: ${MAP_JOB}"

# =============================================================================
# STAGE 2 — injection array
#
# We cannot know N_TASKS at submit time (it depends on stage 1 output).
# Work-around: submit a "launcher" job that reads task_map.json and
# submits the real stage-2 array with the correct --array range.
# The launcher runs after MAP_JOB completes.
# =============================================================================
log "Submitting stage-2 launcher (depends on task-map job)..."

# Write the launcher script to disk so it can be submitted cleanly
S2_LAUNCHER="${OUTPUT_DIR}/logs/launch_stage2.sh"
cat > "${S2_LAUNCHER}" << LAUNCHER
#!/usr/bin/env bash
set -euo pipefail
ARRAY_ARG=\$(grep "sbatch --array" "${OUTPUT_DIR}/logs/task_map_array_arg.txt" \
             | awk '{print \$NF}')
if [[ -z "\${ARRAY_ARG}" ]]; then
    # Fallback: parse directly from task_map.json
    N_TASKS=\$(python3 -c "
import json
tm = json.load(open('${OUTPUT_DIR}/metadata/task_map.json'))
print(len(tm))
")
    ARRAY_ARG="0-\$((N_TASKS-1))"
fi
echo "Launching stage 2 with --array=\${ARRAY_ARG}%${S2_MAX_CONCURRENT}"

S2_JOB=\$(sbatch \\
    --job-name="smbhb_s2_${CONFIG}" \\
    --account="${PROJECT}" \\
    --partition="${PARTITION}" \\
    --nodes="${S2_NODES}" \\
    --ntasks="${S2_NTASKS}" \\
    --cpus-per-task="${S2_CPUS}" \\
    --mem="${S2_MEM}" \\
    --time="${S2_TIME}" \\
    --array="\${ARRAY_ARG}%${S2_MAX_CONCURRENT}" \\
    --output="${OUTPUT_DIR}/logs/stage2_%A_%a.out" \\
    --error="${OUTPUT_DIR}/logs/stage2_%A_%a.err" \\
    --parsable \\
    --wrap="${PYTHON} ${REPO_DIR}/stage2_inject.py --output-dir ${OUTPUT_DIR}"
)
S2_JOB=\$(echo "\$S2_JOB" | tr -d '[:space:]')
echo "Stage 2 array job ID: \${S2_JOB}"

# Now submit stage 3 (one task per population) depending on all of stage 2
S3_JOB=\$(sbatch \\
    --job-name="smbhb_s3_${CONFIG}" \\
    --account="${PROJECT}" \\
    --partition="${PARTITION}" \\
    --nodes="${S3_NODES}" \\
    --ntasks="${S3_NTASKS}" \\
    --cpus-per-task="${S3_CPUS}" \\
    --mem="${S3_MEM}" \\
    --time="${S3_TIME}" \\
    --array="0-\$((${N_SIMS}-1))%${S3_MAX_CONCURRENT}" \\
    --dependency="afterok:\${S2_JOB}" \\
    --output="${OUTPUT_DIR}/logs/stage3_%A_%a.out" \\
    --error="${OUTPUT_DIR}/logs/stage3_%A_%a.err" \\
    --parsable \\
    --wrap="${PYTHON} ${REPO_DIR}/stage3_reduce.py \\
        --output-dir ${OUTPUT_DIR} \\
        ${CGW_FLAG}"
)
S3_JOB=\$(echo "\$S3_JOB" | tr -d '[:space:]')
echo "Stage 3 array job ID: \${S3_JOB}"

# Stage 4 — aggregate
sbatch \\
    --job-name="smbhb_s4_${CONFIG}" \\
    --account="${PROJECT}" \\
    --partition="${PARTITION}" \\
    --nodes="${S4_NODES}" \\
    --ntasks="${S4_NTASKS}" \\
    --cpus-per-task="${S4_CPUS}" \\
    --mem="${S4_MEM}" \\
    --time="${S4_TIME}" \\
    --dependency="afterok:\${S3_JOB}" \\
    --output="${OUTPUT_DIR}/logs/stage4_%j.out" \\
    --error="${OUTPUT_DIR}/logs/stage4_%j.err" \\
    --wrap="${PYTHON} ${REPO_DIR}/stage4_aggregate.py --output-dir ${OUTPUT_DIR} --verbose"

echo "Pipeline fully submitted."
LAUNCHER

chmod +x "${S2_LAUNCHER}"

LAUNCHER_JOB=$(sbatch_or_dry "s2_launcher" \
    --job-name="smbhb_launch2_${CONFIG}" \
    --account="${PROJECT}" \
    --partition="${PARTITION}" \
    --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem="4G" \
    --time="00:15:00" \
    --dependency="afterok:${MAP_JOB}" \
    --output="${OUTPUT_DIR}/logs/launch_stage2_%j.out" \
    --error="${OUTPUT_DIR}/logs/launch_stage2_%j.err" \
    --parsable \
    "${S2_LAUNCHER}"
)
LAUNCHER_JOB=$(echo "$LAUNCHER_JOB" | tr -d '[:space:]')
log "  Stage-2 launcher job ID: ${LAUNCHER_JOB}"

# =============================================================================
# DONE
# =============================================================================
log ""
log "Pipeline submitted successfully."
log ""
log "Dependency chain:"
log "  stage1 (${S1_JOB})"
log "    └─ task_map (${MAP_JOB})"
log "       └─ s2_launcher (${LAUNCHER_JOB})"
log "          └─ stage2 array  [submitted by launcher after stage1 output known]"
log "             └─ stage3 array  (one task per population)"
log "                └─ stage4 aggregate"
log ""
log "Monitor with:"
log "  squeue -u \$USER"
log "  tail -f ${OUTPUT_DIR}/logs/stage1_*.out"
log ""
log "Final results will appear in:"
log "  ${OUTPUT_DIR}/results/summary.json"
log "  ${OUTPUT_DIR}/results/summary_table.txt"
