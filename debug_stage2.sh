#!/usr/bin/env bash
#SBATCH --job-name=s2_debug_sim000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/fred/oz005/users/bhoulden/SMBHB_population_injections/runs/2026-06-02_optimistic/logs/s2_debug_sim000_%j.out
#SBATCH --error=/fred/oz005/users/bhoulden/SMBHB_population_injections/runs/2026-06-02_optimistic/logs/s2_debug_sim000_%j.err

echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURM_NODELIST}"
echo "Start:    $(date)"
echo "=========================================="

cd /fred/oz005/users/bhoulden/SMBHB_population_injections

module purge
unset PYTHONPATH
module load mamba
mamba activate smbhb312

OMP_NUM_THREADS=1
MKL_NUM_THREADS=1

python -u stage2_inject.py \
    --output-dir  /fred/oz005/users/bhoulden/SMBHB_population_injections/runs/2026-06-02_optimistic \
    --config      optimistic \
    --target-snr  3.75 \
    --snr-range   3.5 4.0 \
    --sim-id      0 \
    --n-chunks    1 \
    --noise-seed  26072001 \
    --n-test      1000 \
    --cgw \
    --synthetic-ptas \

echo "=========================================="
echo "Finished: $(date)  exit=$?"
echo "=========================================="