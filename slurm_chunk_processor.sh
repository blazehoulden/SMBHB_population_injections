#!/bin/bash

################################################################################
# Chunk Processor - Per-Array-Task Worker
# 
# Processes a single chunk of a population zarr file.
# Runs as array task within a Slurm array job.
#
# Usage (called by slurm, not directly):
#   sbatch --array=0-99 slurm_chunk_processor.sh \
#       --population-zarr /path/to/population.zarr \
#       --n-chunks 100 \
#       --output-dir /path/to/output \
#       --pop-idx 0
################################################################################

#SBATCH --job-name=smbhb_chunk
# Other SBATCH directives passed from parent script

module purge
unset PYTHONPATH
module load mamba
mamba activate smbhb312

set -e

# Parse arguments
POPULATION_ZARR=""
N_CHUNKS=""
OUTPUT_DIR=""
POP_IDX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --population-zarr) POPULATION_ZARR="$2"; shift 2 ;;
        --n-chunks) N_CHUNKS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --pop-idx) POP_IDX="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validation
if [ -z "$POPULATION_ZARR" ] || [ -z "$N_CHUNKS" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$POP_IDX" ]; then
    echo "ERROR: Missing required arguments"
    exit 1
fi

if [ ! -d "$POPULATION_ZARR" ]; then
    echo "ERROR: Population zarr not found: $POPULATION_ZARR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CHUNK_INDEX=$SLURM_ARRAY_TASK_ID
CHUNK_OUTPUT="$OUTPUT_DIR/chunk_pop${POP_IDX}_idx${CHUNK_INDEX}.npz"

echo "=========================================="
echo "Processing Chunk $CHUNK_INDEX / $N_CHUNKS"
echo "Population: $POP_IDX"
echo "Population Zarr: $POPULATION_ZARR"
echo "Start time: $(date)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

# Run chunked injection driver
python chunked_inject_driver.py \
    --population-zarr "$POPULATION_ZARR" \
    --chunk-index "$CHUNK_INDEX" \
    --n-chunks "$N_CHUNKS" \
    --output-dir "$OUTPUT_DIR" \
    --accumulate \
    --psr-ra 1.57 \
    --psr-dec 0.52 \
    --accum-grid-size 1000 \
    --top-k 1000

echo "=========================================="
echo "Chunk $CHUNK_INDEX completed at: $(date)"
echo "Output: $CHUNK_OUTPUT"
echo "=========================================="

exit 0
