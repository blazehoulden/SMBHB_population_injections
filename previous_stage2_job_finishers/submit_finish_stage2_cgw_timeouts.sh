#!/usr/bin/env bash
# =============================================================================
# submit_finish_stage2_cgw_timeouts.sh  —  Scan timeout stage2 jobs and submit
# CGW-only finish jobs for simulations that already have saved residuals.
# =============================================================================

#SBATCH --job-name=s2_finish_scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --chdir=/fred/oz005/users/bhoulden/SMBHB_population_injections
#SBATCH --output=/fred/oz005/users/bhoulden/SMBHB_population_injections/logs/s2_finish_scan_%j.out
#SBATCH --error=/fred/oz005/users/bhoulden/SMBHB_population_injections/logs/s2_finish_scan_%j.err

set -euo pipefail
PS1="${PS1:-}"

REPO_DIR="/fred/oz005/users/bhoulden/SMBHB_population_injections"
LOG_DIR="${REPO_DIR}/logs"
PYTHON_SCRIPT="${REPO_DIR}/finish_stage2_cgw_timeouts.py"

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
        --validate-proxy)
            cmd+=(--validate-proxy)
            shift
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

echo "Activating environment and running finish scan..."
echo "Command: ${cmd[*]}"
echo

eval "${ENV_SETUP}"
exec "${cmd[@]}"