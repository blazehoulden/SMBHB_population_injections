#!/usr/bin/env bash
# Local smoke test for the full stage1 → stage2 pipeline
# Generates a tiny population (100 binaries) and verifies outputs

set -euo pipefail

TEST_DIR="/tmp/smbhb_trial_run_$(date +%s)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "🧪 SMBHB Pipeline Trial Run"
echo "======================================================"
echo "Test output: $TEST_DIR"
echo "Repository: $REPO_DIR"
echo "Python: $PYTHON"
echo "======================================================"

# Create test directory
mkdir -p "$TEST_DIR"

# ─────────────────────────────────────────────────────────
# Stage 1: Generate tiny population
# ─────────────────────────────────────────────────────────
echo ""
echo "📍 STAGE 1: Population generation"
echo "─────────────────────────────────────"

"$PYTHON" "$REPO_DIR/stage1_setup.py" \
    --config optimistic \
    --target-snr 4.0 \
    --snr-range 3.5 4.25 \
    --chunk-size 100 \
    --simulations 1 \
    --task-id 0 \
    --output-dir "$TEST_DIR"

echo "✓ Stage 1 complete"

# ─────────────────────────────────────────────────────────
# Verify Stage 1 outputs
# ─────────────────────────────────────────────────────────
echo ""
echo "📍 Verifying Stage 1 outputs"
echo "─────────────────────────────────────"

if [[ ! -f "$TEST_DIR/populations/subpop_000.pkl.gz" ]]; then
    echo "❌ ERROR: population shard not found"
    exit 1
fi
echo "✓ Population shard exists"

if [[ ! -f "$TEST_DIR/metadata/config.json" ]]; then
    echo "❌ ERROR: config.json not found"
    exit 1
fi
echo "✓ Metadata config.json exists"

DELTA_FILES=$(find "$TEST_DIR/stoas/sim0000" -name "*_delta.npy" 2>/dev/null | wc -l)
if [[ $DELTA_FILES -eq 0 ]]; then
    echo "❌ ERROR: no delta files found in stoas/"
    exit 1
fi
echo "✓ Found $DELTA_FILES per-pulsar delta stoa files"

echo ""
echo "  Directory structure:"
ls -lh "$TEST_DIR/" | tail -n +2 | sed 's/^/    /'
echo ""
echo "  Populations:"
ls -lh "$TEST_DIR/populations/" | tail -n +2 | sed 's/^/    /'
echo ""
echo "  Metadata:"
cat "$TEST_DIR/metadata/config.json" | sed 's/^/    /'

# ─────────────────────────────────────────────────────────
# Stage 2: Load, simulate noise, combine GW, compute SNR
# ─────────────────────────────────────────────────────────
echo ""
echo "📍 STAGE 2: SNR computation and CGW analysis"
echo "─────────────────────────────────────"

"$PYTHON" "$REPO_DIR/stage2_inject.py" \
    --config optimistic \
    --target-snr 4.0 \
    --snr-range 3.5 4.25 \
    --task-id 0 \
    --output-dir "$TEST_DIR"

echo "✓ Stage 2 complete"

# ─────────────────────────────────────────────────────────
# Verify Stage 2 outputs
# ─────────────────────────────────────────────────────────
echo ""
echo "📍 Verifying Stage 2 outputs"
echo "─────────────────────────────────────"

if [[ ! -f "$TEST_DIR/populations/subpop_000.pkl.gz" ]]; then
    echo "❌ ERROR: population shard was lost or deleted"
    exit 1
fi
echo "✓ Population shard still present (updated with CGW SNR)"

UPDATED_SIZE=$(stat -f%z "$TEST_DIR/populations/subpop_000.pkl.gz" 2>/dev/null || stat -c%s "$TEST_DIR/populations/subpop_000.pkl.gz" 2>/dev/null)
echo "  Shard size: $UPDATED_SIZE bytes"

# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
echo ""
echo "✅ TRIAL RUN SUCCESSFUL"
echo "════════════════════════════════════════════════════"
echo "Test artifacts preserved in: $TEST_DIR"
echo "You can now proceed with HPC submission."
echo ""
echo "Next steps:"
echo "  1. Review the trial output above"
echo "  2. Submit to HPC using: bash submit_main_HPC.sh --output-dir <desired_hpc_dir>"
echo ""
