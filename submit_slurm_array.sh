#!/bin/bash
# SBATCH template for launching chunked_inject_driver.py as an array job
# Usage example:
#   sbatch --array=0-99 submit_slurm_array.sh --population-zarr data/population.zarr --n-chunks 100

#SBATCH --job-name=chunked-inject
#SBATCH --output=slurm-chunked-inject-%%A_%%a.out
#SBATCH --error=slurm-chunked-inject-%%A_%%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Load modules or activate conda env if needed
# module load anaconda
# source activate enter312

# Parse remaining args and forward to Python driver
PY=python
DRIVER=chunked_inject_driver.py

echo "Running chunk index: ${SLURM_ARRAY_TASK_ID}"
${PY} ${DRIVER} --population-zarr "$@" --chunk-index ${SLURM_ARRAY_TASK_ID}
