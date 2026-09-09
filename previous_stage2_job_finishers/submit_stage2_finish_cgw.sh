#!/usr/bin/env bash
# =============================================================================
# submit_stage2_finish_cgw.sh  —  Finish stage 2 from saved residuals
#
# This wrapper mirrors the retry submitter, but it only runs the CGW resume
# path. Use it when the heavy stage-2 work has already finished and only the
# candidate ranking / summary needs to be completed from residuals/combined.
#
# Examples
# ────────
#   sbatch submit_stage2_finish_cgw.sh --output-dir runs/2026-05-23_pessimistic --sim-id 123
#   bash   submit_stage2_finish_cgw.sh --output-dir runs/2026-05-23_pessimistic --sim-id 123 --validate-proxy
# =============================================================================

#SBATCH --job-name=s2_finish_cgw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=02:40:00
#SBATCH --chdir=/fred/oz005/users/bhoulden/SMBHB_population_injections
#SBATCH --output=/fred/oz005/users/bhoulden/SMBHB_population_injections/logs/s2_finish_cgw_%j.out
#SBATCH --error=/fred/oz005/users/bhoulden/SMBHB_population_injections/logs/s2_finish_cgw_%j.err

set -euo pipefail
PS1="${PS1:-}"

REPO_DIR="/fred/oz005/users/bhoulden/SMBHB_population_injections"
LOG_DIR="${REPO_DIR}/logs"
PYTHON_SCRIPT="${REPO_DIR}/stage2_finish_cgw.py"
OUTPUT_DIR=""

ENV_SETUP="module purge; unset PYTHONPATH; module load mamba; mamba activate smbhb312;"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: could not find ${PYTHON_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

cmd=(python "${PYTHON_SCRIPT}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sim-id)
            cmd+=(--sim-id "$2")
            shift 2
            ;;
        --config)
            cmd+=(--config "$2")
            shift 2
            ;;
        --validate-proxy)
            cmd+=(--validate-proxy)
            shift
            ;;
        --n-test)
            cmd+=(--n-test "$2")
            shift 2
            ;;
        --dry-run)
            cmd+=(--dry-run)
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: --output-dir is required (the specific run directory, e.g. runs/2026-05-23_pessimistic)" >&2
    exit 1
fi

cmd+=(--output-dir "${OUTPUT_DIR}")

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Job Name: ${SLURM_JOB_NAME:-n/a}"
echo "Node: ${SLURM_NODELIST:-n/a}"
echo "Start Time: $(date)"
echo "=========================================="
echo

echo "Activating environment and running CGW resume..."
echo "Command: ${cmd[*]}"
echo

eval "${ENV_SETUP}"
exec "${cmd[@]}"