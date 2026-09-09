#!/bin/bash

#SBATCH --job-name=smbhb_pessimistic
#SBATCH --output=logs/smbhb_pessimistic_%j.out
#SBATCH --error=logs/smbhb_pessimistic_%j.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40GB
#SBATCH --array=0-30

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NESTED=FALSE
export OMP_MAX_ACTIVE_LEVELS=1

# Job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load modules
module purge
unset PYTHONPATH

# module load gcc/13.2.0
# module load scipy-bundle/2023.11
# module load astropy/7.0.0
# module load openmpi/4.1.6
# module load hdf5
# module load gsl 2>/dev/null || echo "GSL not loaded"

# Activate virtual environment
# module unload python 
module load mamba
mamba activate smbhb312

# export TEMPO2=$HOME/.local/share/tempo2
# export TEMPO2=$CONDA_PREFIX/share/tempo2

echo $TEMPO2

# Print environment info
# echo "Python: $(which python)"
# echo "Python version: $(python --version)"
# echo ""

# CRITICAL CHECK: Must be Python 3.11, not 3.12
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PYTHON_VERSION" != "3.11" ]; then
    echo "ERROR: Python version is $PYTHON_VERSION, but ephem requires 3.11!"
    echo "Please recreate environment with: conda create -n SMBHB312 python=3.11"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to repository directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Print loaded modules
echo "Loaded modules:"
module list
echo ""

# Run the analysis with test configuration
echo "Starting SMBHB analysis..."
echo "=========================================="

# Test run with minimal settings
python -u main.py \
    --config pessimistic \
    --simulations 1 \
    --target-snr 4.0 \
    --save-name "run_${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
