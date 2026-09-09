#!/usr/bin/env bash
# =============================================================================
# submit_retry_stage2_timeouts.sh  —  Scan stage2 logs and resubmit timeouts
#
# This wrapper follows the same Slurm-style conventions as the main pipeline.
# Submit it with sbatch, or run it directly on the login node.
#
# Default behaviour:
#   - scans yesterday's run directories under ./runs
#   - prints a timeout / failure report
#
# With --submit:
#   - resubmits stage2 jobs whose logs ended with TIMEOUT
#   - uses a longer wallclock via --s2-time
#
# Examples
# ────────
#   sbatch submit_retry_stage2_timeouts.sh --run-dir runs/2026-05-23_pessimistic
#   sbatch submit_retry_stage2_timeouts.sh --date 2026-05-23 --submit --s2-time 04:00:00
#   bash   submit_retry_stage2_timeouts.sh --run-dir runs/2026-05-23_pessimistic --submit
#
# Notes
# ─────
#   - The actual retry logic lives in retry_stage2_timeouts.py.
#   - This script is just the Slurm-friendly submission wrapper.
# =============================================================================

#SBATCH --job-name=s2_retry_scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --chdir=/fred/oz005/users/bhoulden/SMBHB_population_injections
#SBATCH --output=/fred/oz005/users/bhoulden/SMBHB_population_injections/logs/s2_retry_scan_%j.out
#SBATCH --error=/fred/oz005/users/bhoulden/SMBHB_population_injections/logs/s2_retry_scan_%j.err

set -euo pipefail
PS1="${PS1:-}"

REPO_DIR="/fred/oz005/users/bhoulden/SMBHB_population_injections"
LOG_DIR="${REPO_DIR}/logs"
PYTHON_SCRIPT="${REPO_DIR}/retry_stage2_timeouts.py"

ENV_SETUP="module purge; unset PYTHONPATH; module load mamba; mamba activate smbhb312;"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: could not find ${PYTHON_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

cmd=(python "${PYTHON_SCRIPT}" --output-root "${REPO_DIR}/runs" --repo-dir "${REPO_DIR}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)
            cmd+=(--submit)
            shift
            ;;
        --dry-run)
            cmd+=(--dry-run)
            shift
            ;;
        --run-dir)
            cmd+=(--run-dir "$2")
            shift 2
            ;;
        --date)
            cmd+=(--date "$2")
            shift 2
            ;;
        --s2-time)
            cmd+=(--s2-time "$2")
            shift 2
            ;;
        --s2-mem)
            cmd+=(--s2-mem "$2")
            shift 2
            ;;
        --s2-cpus)
            cmd+=(--s2-cpus "$2")
            shift 2
            ;;
        --target-snr)
            cmd+=(--target-snr "$2")
            shift 2
            ;;
        --snr-range)
            cmd+=(--snr-range "$2" "$3")
            shift 3
            ;;
        --n-test)
            cmd+=(--n-test "$2")
            shift 2
            ;;
        --report-json)
            cmd+=(--report-json "$2")
            shift 2
            ;;
        --job-name-prefix)
            cmd+=(--job-name-prefix "$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Job Name: ${SLURM_JOB_NAME:-n/a}"
echo "Node: ${SLURM_NODELIST:-n/a}"
echo "Start Time: $(date)"
echo "=========================================="
echo

echo "Activating environment and running retry scan..."
echo "Command: ${cmd[*]}"
echo

eval "${ENV_SETUP}"
exec "${cmd[@]}"