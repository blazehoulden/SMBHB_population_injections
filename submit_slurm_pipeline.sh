#!/bin/bash

################################################################################
# SMBHB Slurm Pipeline Orchestrator
# 
# Usage:
#   sbatch submit_slurm_pipeline.sh --config pessimistic --n-sims 10 --n-chunks 100
#
# This script orchestrates the full pipeline:
#   1. Population generation + zarr write (main.py)
#   2. Per-population chunk array jobs (parallel)
#   3. Per-population reduction + CGW analysis (depends on chunk job)
#   4. Final aggregation (depends on all CGW jobs)
################################################################################

#SBATCH --job-name=smbhb_pipeline
#SBATCH --output=logs/smbhb_pipeline_%j.out
#SBATCH --error=logs/smbhb_pipeline_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10GB

# Parse arguments
CONFIG="pessimistic"
N_SIMS=1
N_CHUNKS=100
TARGET_SNR=4.0
SNR_MIN=3.5
SNR_MAX=4.25
CHUNKED_OUTPUT_DIR=""
CHUNKED_TEST=false
MINIMAL_POP_STORAGE=true
CLEANUP_INTERMEDIATES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --n-sims) N_SIMS="$2"; shift 2 ;;
        --n-chunks) N_CHUNKS="$2"; shift 2 ;;
        --target-snr) TARGET_SNR="$2"; shift 2 ;;
        --snr-min) SNR_MIN="$2"; shift 2 ;;
        --snr-max) SNR_MAX="$2"; shift 2 ;;
        --output-dir) CHUNKED_OUTPUT_DIR="$2"; shift 2 ;;
        --chunked-test) CHUNKED_TEST=true; shift ;;
        --no-minimal-pop-storage) MINIMAL_POP_STORAGE=false; shift ;;
        --no-cleanup-intermediates) CLEANUP_INTERMEDIATES=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

set -e

module purge
unset PYTHONPATH
module load mamba
mamba activate smbhb312

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "=========================================="
echo "SMBHB Slurm Pipeline Orchestrator"
echo "=========================================="
echo "Config: $CONFIG"
echo "N_SIMS: $N_SIMS"
echo "N_CHUNKS: $N_CHUNKS"
echo "Target SNR: $TARGET_SNR"
echo "SNR range: [$SNR_MIN, $SNR_MAX]"
echo "Minimal pop storage: $MINIMAL_POP_STORAGE"
echo "Cleanup intermediates: $CLEANUP_INTERMEDIATES"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================================
# STEP 1: Population Generation + Zarr Writing
# ============================================================================
echo ""
echo "Step 1: Generating populations and writing zarr files..."
echo ""

# Default output directory
if [ -z "$CHUNKED_OUTPUT_DIR" ]; then
    DATE=$(date +%Y-%m-%d)
    CHUNKED_OUTPUT_DIR="data/$DATE/${CONFIG}_pipeline/chunks"
fi

# Run population generation
GENERATION_LOG="logs/generation_${SLURM_JOB_ID}.log"
MAIN_EXTRA_ARGS=()
if [ "$MINIMAL_POP_STORAGE" = true ]; then
    MAIN_EXTRA_ARGS+=("--minimal-pop-storage")
fi
python -u main.py \
    --config "$CONFIG" \
    --simulations "$N_SIMS" \
    --target-snr "$TARGET_SNR" \
    --snr-range "$SNR_MIN" "$SNR_MAX" \
    --save-name "pipeline_${SLURM_JOB_ID}" \
    --chunked-injection \
    --defer-cgw-to-slurm \
    --n-chunks "$N_CHUNKS" \
    --chunked-output-dir "$CHUNKED_OUTPUT_DIR" \
    $([ "$CHUNKED_TEST" = true ] && echo "--chunked-test") \
    "${MAIN_EXTRA_ARGS[@]}" \
    2>&1 | tee "$GENERATION_LOG"

# Check if populations were written
if [ ! -d "$CHUNKED_OUTPUT_DIR" ]; then
    echo "ERROR: Chunked output directory not created!"
    exit 1
fi

# Find all population zarr files
POPULATION_ZARRS=($(find "$CHUNKED_OUTPUT_DIR" -maxdepth 1 -name "population_*.zarr" -type d | sort))
N_POPULATIONS=${#POPULATION_ZARRS[@]}

if [ "$N_POPULATIONS" -eq 0 ]; then
    echo "ERROR: No population zarr files found in $CHUNKED_OUTPUT_DIR"
    exit 1
fi

CONSISTENCY_SUMMARY="$CHUNKED_OUTPUT_DIR/sgwb_consistency_summary.json"
if [ ! -f "$CONSISTENCY_SUMMARY" ]; then
    echo "ERROR: SGWB consistency summary missing: $CONSISTENCY_SUMMARY"
    exit 1
fi

python - <<PY
import json
import math
path = "$CONSISTENCY_SUMMARY"
with open(path, "r") as f:
    s = json.load(f)
lo, hi = s.get("snr_range", [None, None])
bad = []
for p in s.get("populations", []):
    v = float(p.get("snr_final", float("nan")))
    if not (math.isfinite(v) and lo <= v <= hi):
        bad.append((p.get("pop_idx"), v))
if bad:
    raise SystemExit(f"Found out-of-range SGWB SNR populations: {bad[:10]}")
print(f"SGWB consistency verified for {len(s.get('populations', []))} populations in range [{lo}, {hi}].")
PY

echo ""
echo "✓ Generated $N_POPULATIONS populations"
echo "  Output directory: $CHUNKED_OUTPUT_DIR"
echo ""

# ============================================================================
# STEP 2: Submit Chunk Array Jobs (one per population, parallel)
# ============================================================================
echo "Step 2: Submitting chunk processing array jobs..."
echo ""

CHUNK_JOB_IDS=()
REDUCTION_JOB_IDS=()

for pop_idx in "${!POPULATION_ZARRS[@]}"; do
    ZARR_PATH="${POPULATION_ZARRS[$pop_idx]}"
    POP_NAME=$(basename "$ZARR_PATH")
    
    echo "  Submitting chunk job for population $pop_idx ($POP_NAME)..."
    
    # Submit chunk array job
    CHUNK_JOB_ID=$(sbatch \
        --job-name="smbhb_chunks_pop${pop_idx}" \
        --output="logs/chunks_pop${pop_idx}_%a.out" \
        --error="logs/chunks_pop${pop_idx}_%a.err" \
        --array="0-$((N_CHUNKS-1))" \
        --time="00:30:00" \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem-per-cpu=15GB \
        slurm_chunk_processor.sh \
        --population-zarr "$ZARR_PATH" \
        --n-chunks "$N_CHUNKS" \
        --output-dir "$CHUNKED_OUTPUT_DIR" \
        --pop-idx "$pop_idx" \
        | awk '{print $NF}')
    
    CHUNK_JOB_IDS+=("$CHUNK_JOB_ID")
    echo "    → Chunk array job ID: $CHUNK_JOB_ID"
    
    # Submit reduction + CGW job (depends on chunk job completion)
    REDUCTION_JOB_ID=$(sbatch \
        --job-name="smbhb_reduce_pop${pop_idx}" \
        --output="logs/reduce_pop${pop_idx}.out" \
        --error="logs/reduce_pop${pop_idx}.err" \
        --time="00:20:00" \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem-per-cpu=8GB \
        --dependency="afterok:$CHUNK_JOB_ID" \
        slurm_reduce_and_cgw.sh \
        --population-zarr "$ZARR_PATH" \
        --n-chunks "$N_CHUNKS" \
        --input-dir "$CHUNKED_OUTPUT_DIR" \
        --pop-idx "$pop_idx" \
        $([ "$CLEANUP_INTERMEDIATES" = true ] && echo "--cleanup-intermediates") \
        | awk '{print $NF}')
    
    REDUCTION_JOB_IDS+=("$REDUCTION_JOB_ID")
    echo "    → Reduction + CGW job ID: $REDUCTION_JOB_ID (depends on $CHUNK_JOB_ID)"
    
done

echo ""
echo "✓ Submitted $N_POPULATIONS chunk array jobs and $N_POPULATIONS reduction jobs"
echo ""

# ============================================================================
# STEP 3: Submit Aggregation Job
# ============================================================================
echo "Step 3: Submitting final aggregation job..."
echo ""

# Create dependency string from all reduction jobs
DEPENDENCY_STR=$(printf ":%s" "${REDUCTION_JOB_IDS[@]}")
DEPENDENCY_STR="afterok${DEPENDENCY_STR}"

AGGREGATION_JOB_ID=$(sbatch \
    --job-name="smbhb_aggregate" \
    --output="logs/aggregation.out" \
    --error="logs/aggregation.err" \
    --time="00:10:00" \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem-per-cpu=4GB \
    --dependency="$DEPENDENCY_STR" \
    slurm_aggregate_results.sh \
    --input-dir "$CHUNKED_OUTPUT_DIR" \
    --n-populations "$N_POPULATIONS" \
    --config "$CONFIG" \
    | awk '{print $NF}')

echo "✓ Aggregation job ID: $AGGREGATION_JOB_ID"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=========================================="
echo "PIPELINE SUBMISSION COMPLETE"
echo "=========================================="
echo ""
echo "Chunk array jobs:"
for i in "${!CHUNK_JOB_IDS[@]}"; do
    echo "  Pop $i: ${CHUNK_JOB_IDS[$i]}"
done
echo ""
echo "Reduction + CGW jobs:"
for i in "${!REDUCTION_JOB_IDS[@]}"; do
    echo "  Pop $i: ${REDUCTION_JOB_IDS[$i]}"
done
echo ""
echo "Final aggregation: $AGGREGATION_JOB_ID"
echo ""
echo "Monitor progress:"
echo "  squeue -u $USER | grep smbhb"
echo ""
echo "Results will be written to:"
echo "  $CHUNKED_OUTPUT_DIR/final_aggregated_results.json"
echo ""
echo "Load in notebook:"
echo "  import json"
echo "  with open('$CHUNKED_OUTPUT_DIR/final_aggregated_results.json') as f:"
echo "      results = json.load(f)"
echo ""
echo "=========================================="

exit 0
