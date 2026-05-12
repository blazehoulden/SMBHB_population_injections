#!/bin/bash

################################################################################
# QUICK START EXAMPLE: Submit SMBHB Slurm Pipeline
#
# This script demonstrates how to submit a complete SMBHB population analysis
# pipeline with chunked injection and CGW analysis.
#
# Usage:
#   bash quickstart_slurm_submission.sh
#
# Or submit directly:
#   sbatch submit_slurm_pipeline.sh --config pessimistic --n-sims 3 --n-chunks 50
#
################################################################################

set -e

echo "=========================================="
echo "SMBHB Slurm Pipeline Quick Start"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found. Run from project root directory."
    exit 1
fi

# Configuration
CONFIG="pessimistic"           # 10M binaries per population
N_SIMS=3                        # Number of independent populations
N_CHUNKS=50                     # Chunks per population (adjust for memory)
OUTPUT_DIR="data/slurm_test"   # Results location

echo ""
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Simulations: $N_SIMS"
echo "  Chunks per population: $N_CHUNKS"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Create logs directory
mkdir -p logs

echo "Submitting pipeline..."
echo ""

# Submit the orchestrator job
JOB_ID=$(sbatch \
    --job-name="smbhb_quickstart" \
    --output="logs/quickstart_%j.out" \
    --error="logs/quickstart_%j.err" \
    submit_slurm_pipeline.sh \
    --config "$CONFIG" \
    --n-sims "$N_SIMS" \
    --n-chunks "$N_CHUNKS" \
    --output-dir "$OUTPUT_DIR" \
    | awk '{print $NF}')

echo "✓ Pipeline submitted with job ID: $JOB_ID"
echo ""
echo "Next steps:"
echo "  1. Monitor progress:"
echo "     squeue -u \$USER | grep smbhb"
echo ""
echo "  2. Watch generation log:"
echo "     tail -f logs/generation_*.out"
echo ""
echo "  3. When complete, load notebook:"
echo "     jupyter notebook population_cgw_analysis_pipeline.ipynb"
echo "     → Update RESULTS_DIR = '$OUTPUT_DIR'"
echo ""
echo "  4. Check status:"
echo "     sbatch job info: scontrol show job $JOB_ID"
echo "     All slurm logs: ls logs/"
echo ""
echo "Estimated time:"
echo "  - Generation: ~5-10 min"
echo "  - Chunks ($N_CHUNKS per pop): ~30-60 min"
echo "  - Reductions & CGW: ~10-20 min"
echo "  - Aggregation: ~5 min"
echo "  Total: ~1-2 hours"
echo ""

exit 0
