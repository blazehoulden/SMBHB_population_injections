#!/usr/bin/env bash
# =============================================================================
# local_test.sh  —  Run the full pipeline locally (no Slurm) for smoke testing
#
# Uses a tiny configuration (1 simulation, small chunk size, 1 iteration)
# to verify all stages execute without errors.
#
# Usage
# ─────
#   bash local_test.sh [--output-dir /tmp/smbhb_test] [--config optimistic]
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${TMPDIR:-/tmp}/smbhb_pipeline_test_$(date +%s)"
CONFIG="optimistic"
N_SIMS=1
CHUNK_SIZE=100000    # small for local test
TARGET_SNR=4.0
SNR_LOW=3.5
SNR_HIGH=4.25
PYTHON="${SMBHB_PYTHON:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --config)     CONFIG="$2";     shift 2 ;;
        --python)     PYTHON="$2";     shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  SMBHB Pipeline — local smoke test"
echo "  Output: ${OUTPUT_DIR}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

# ── Stage 1 ───────────────────────────────────────────────────────────────────
echo ""
echo "▶ Stage 1: population synthesis..."
"${PYTHON}" "${REPO_DIR}/stage1_setup.py" \
    --config "${CONFIG}" \
    --target-snr "${TARGET_SNR}" \
    --snr-range "${SNR_LOW}" "${SNR_HIGH}" \
    --simulations "${N_SIMS}" \
    --chunk-size "${CHUNK_SIZE}" \
    --output-dir "${OUTPUT_DIR}"
echo "  ✓ Stage 1 complete"

# ── Task map ──────────────────────────────────────────────────────────────────
echo ""
echo "▶ Building task map..."
"${PYTHON}" "${REPO_DIR}/build_task_map.py" \
    --output-dir "${OUTPUT_DIR}"
echo "  ✓ Task map built"

# ── Stage 2 (all tasks, sequential) ──────────────────────────────────────────
echo ""
echo "▶ Stage 2: chunked injection..."
N_TASKS=$("${PYTHON}" -c "
import json
tm = json.load(open('${OUTPUT_DIR}/metadata/task_map.json'))
print(len(tm))
")
echo "  Tasks: ${N_TASKS}"

for task_id in $(seq 0 $((N_TASKS - 1))); do
    echo "  task ${task_id}/${N_TASKS}..."
    SLURM_ARRAY_TASK_ID="${task_id}" "${PYTHON}" "${REPO_DIR}/stage2_inject.py" \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${task_id}"
done
echo "  ✓ Stage 2 complete"

# ── Stage 3 (all populations, sequential) ────────────────────────────────────
echo ""
echo "▶ Stage 3: reduction + OS..."
N_POPS=$("${PYTHON}" -c "
import json
cfg = json.load(open('${OUTPUT_DIR}/metadata/config.json'))
print(len(cfg['populations']))
")

for pop_idx in $(seq 0 $((N_POPS - 1))); do
    echo "  pop ${pop_idx}..."
    "${PYTHON}" "${REPO_DIR}/stage3_reduce.py" \
        --output-dir "${OUTPUT_DIR}" \
        --pop-idx "${pop_idx}"
done
echo "  ✓ Stage 3 complete"

# ── Stage 4 ───────────────────────────────────────────────────────────────────
echo ""
echo "▶ Stage 4: aggregate..."
"${PYTHON}" "${REPO_DIR}/stage4_aggregate.py" \
    --output-dir "${OUTPUT_DIR}" \
    --verbose
echo "  ✓ Stage 4 complete"

echo ""
echo "========================================"
echo "  ALL STAGES PASSED"
echo "  Results: ${OUTPUT_DIR}/results/"
echo "========================================"
