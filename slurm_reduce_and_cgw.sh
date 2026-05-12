#!/bin/bash

################################################################################
# Reduce & CGW Analysis - Merges chunks and runs CGW SNR analysis
#
# After all chunks complete, this job:
#   1. Merges frequency-domain accumulators from all chunks
#   2. Runs inverse FFT to get final residuals
#   3. Runs CGW SNR analysis on the population
#   4. Extracts loudest (highest SNR) CGW candidate
#   5. Saves metadata for notebook aggregation
#
# Usage (called by orchestrator):
#   sbatch slurm_reduce_and_cgw.sh \
#       --population-zarr /path/to/population.zarr \
#       --n-chunks 100 \
#       --input-dir /path/to/chunks \
#       --pop-idx 0
################################################################################

#SBATCH --job-name=smbhb_reduce
# Other SBATCH directives passed from parent script

module purge
unset PYTHONPATH
module load mamba
mamba activate smbhb312

set -e

# Parse arguments
POPULATION_ZARR=""
N_CHUNKS=""
INPUT_DIR=""
POP_IDX=""
CLEANUP_INTERMEDIATES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --population-zarr) POPULATION_ZARR="$2"; shift 2 ;;
        --n-chunks) N_CHUNKS="$2"; shift 2 ;;
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --pop-idx) POP_IDX="$2"; shift 2 ;;
        --cleanup-intermediates) CLEANUP_INTERMEDIATES=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$POPULATION_ZARR" ] || [ -z "$N_CHUNKS" ] || [ -z "$INPUT_DIR" ] || [ -z "$POP_IDX" ]; then
    echo "ERROR: Missing required arguments"
    exit 1
fi

echo "=========================================="
echo "Reduction & CGW Analysis - Population $POP_IDX"
echo "Population Zarr: $POPULATION_ZARR"
echo "Start time: $(date)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

# ============================================================================
# Step 1: Merge chunk frequency accumulators (IFFT reduction)
# ============================================================================
echo ""
echo "Step 1: Merging chunk outputs with IFFT..."

python chunked_inject_driver.py \
    --merge-accum \
    --input-dir "$INPUT_DIR" \
    --output-dir "$INPUT_DIR" \
    --ifft-out \
    --n-chunks "$N_CHUNKS" \
    --pop-idx "$POP_IDX"

FINAL_RESIDUALS="$INPUT_DIR/final_residuals_pop${POP_IDX}.npz"
if [ ! -f "$FINAL_RESIDUALS" ]; then
    echo "ERROR: Final residuals not created: $FINAL_RESIDUALS"
    exit 1
fi

echo "✓ Residuals merged and saved to: $FINAL_RESIDUALS"

# ============================================================================
# Step 2: Run CGW Analysis (Python subprocess)
# ============================================================================
echo ""
echo "Step 2: Running CGW SNR analysis..."
echo ""

cat > /tmp/cgw_analysis_pop${POP_IDX}.py << 'EOFPYTHON'
import json
import sys
import numpy as np
import heapq
from types import SimpleNamespace
from io_backends import population_slice, get_population_length

# Args passed by parent script
pop_zarr = sys.argv[1]
pop_idx = int(sys.argv[2])
output_dir = sys.argv[3]

# Import necessary functions
from debug.test_CGW_sky_loc import sky_sensitivity_weight
from consistent_pop_synth import suppress_enterprise_warnings
from CGW_SNR import compute_cgw_snr_optimal_population
from data_loader import load_pulsars, filter_pulsars_15yr, parse_pulsar_parameters
from pta_builder import build_pta_and_params
import config

# Load pulsars and build PTA
print(f"Loading pulsars for population {pop_idx}...")
with suppress_enterprise_warnings():
    psrs_unfiltered = load_pulsars(verbose=False)
    psrs_clean, raw_noise_params, Tspan_seconds = filter_pulsars_15yr(psrs_unfiltered, verbose=False)

parsed_noise_params = parse_pulsar_parameters(config.NOISEFILE)

# Get population size
n_binaries = get_population_length(pop_zarr)
print(f"Population size: {n_binaries} binaries")

# Build PTA
print("Building PTA...")
pta, model, params_complete = build_pta_and_params(
    psrs=psrs_clean, noise_params_15yr=raw_noise_params, Tspan=Tspan_seconds
)

# ========================================================================
# CGW Analysis: Streaming pre-filter and rank
# ========================================================================
N_PRE_FILTER = min(2000, max(200, n_binaries // 5000))
N_TOP_SOURCES = 50
SCAN_CHUNK = 2_000_000

print(f"\nRunning CGW analysis...")
print(f"  Streaming pre-filter top {N_PRE_FILTER} by proxy...")

# Keep a min-heap of best proxy candidates: (proxy, global_idx)
heap = []

for start in range(0, n_binaries, SCAN_CHUNK):
    end = min(n_binaries, start + SCAN_CHUNK)
    pop = population_slice(pop_zarr, start, end)

    f = pop['f']
    h0 = pop['h0']
    ra = pop['ra']
    dec = pop['dec']

    try:
        sky_w = sky_sensitivity_weight(ra, dec)
    except Exception:
        sky_w = np.array([sky_sensitivity_weight(r, d) for r, d in zip(ra, dec)])

    proxy = (h0 / (2.0 * np.pi * f)) * sky_w

    k_local = min(N_PRE_FILTER, len(proxy))
    if k_local == 0:
        continue
    if k_local < len(proxy):
        part = np.argpartition(proxy, -k_local)[-k_local:]
        local_idx = part[np.argsort(proxy[part])[::-1]]
    else:
        local_idx = np.argsort(proxy)[::-1]

    for li in local_idx:
        gi = start + int(li)
        p = float(proxy[li])
        if len(heap) < N_PRE_FILTER:
            heapq.heappush(heap, (p, gi))
        elif p > heap[0][0]:
            heapq.heapreplace(heap, (p, gi))

if not heap:
    raise RuntimeError("No CGW proxy candidates found during pre-filter")

pre_filtered_idx = [idx for _, idx in sorted(heap, key=lambda x: x[0], reverse=True)]

# Materialize only the top pre-filtered binaries as lightweight objects.
pre_filtered = []
for gi in pre_filtered_idx:
    one = population_slice(pop_zarr, gi, gi + 1)
    b = SimpleNamespace(
        f=float(one['f'][0]),
        Mc=float(one['Mc'][0]) if 'Mc' in one else float('nan'),
        Mtot=float(one['Mtot'][0]) if 'Mtot' in one else float('nan'),
        D_comov=float(one['D_comov'][0]) if 'D_comov' in one else float('nan'),
        z=float(one['z'][0]) if 'z' in one else float('nan'),
        h0=float(one['h0'][0]),
        ra=float(one['ra'][0]),
        dec=float(one['dec'][0]),
        psi=float(one['psi'][0]),
        iota=float(one['iota'][0]),
        phi0=float(one['phi0'][0]),
    )
    sw = float(sky_sensitivity_weight(b.ra, b.dec))
    pre_filtered.append(
        {
            "global_index": int(gi),
            "binary": b,
            "sky_weight": sw,
            "proxy": float((b.h0 / (2.0 * np.pi * b.f)) * sw),
        }
    )

# Compute CGW SNRs
print(f"  Computing CGW SNR for {N_PRE_FILTER} candidates...")
pre_filter_snrs = compute_cgw_snr_optimal_population(
    psrs=psrs_clean,
    pta=pta,
    population=[item["binary"] for item in pre_filtered],
    raw_noise_params=raw_noise_params,
    parsed_noise_params=parsed_noise_params,
    Tspan=Tspan_seconds,
    profile=False,
)

# Rank by SNR
ranked_sources = sorted(
    (
        {
            "proxy_rank": proxy_rank,
            "global_index": item["global_index"],
            "proxy_value": item["proxy"],
            "sky_weight": item["sky_weight"],
            "binary": item["binary"],
            "snr": snr,
        }
        for proxy_rank, (item, snr) in enumerate(zip(pre_filtered, pre_filter_snrs), start=1)
    ),
    key=lambda x: x["snr"],
    reverse=True,
)

top_sources = ranked_sources[:N_TOP_SOURCES]
loudest = top_sources[0]

print(f"\n✓ Top CGW candidate (loudest):")
print(f"  SNR Rank: 1")
print(f"  Proxy Rank: {loudest['proxy_rank']}")
print(f"  CGW SNR: {loudest['snr']:.4f}")
b = loudest['binary']
print(f"  Frequency: {b.f:.2e} Hz")
print(f"  Chirp Mass: {b.Mc:.2e} kg")
print(f"  h0: {b.h0:.2e}")
print(f"  Sky Position: RA={b.ra:.4f}, Dec={b.dec:.4f}")

# Save loudest metadata
loudest_metadata = {
    "pop_idx": pop_idx,
    "n_binaries": n_binaries,
    "loudest_cgw": {
        "snr_rank": 1,
        "proxy_rank": int(loudest["proxy_rank"]),
        "global_index": int(loudest["global_index"]),
        "cgw_snr": float(loudest["snr"]),
        "f": float(b.f),
        "Mc": float(b.Mc),
        "h0": float(b.h0),
        "D_comov": float(b.D_comov),
        "z": float(b.z),
        "ra": float(b.ra),
        "dec": float(b.dec),
        "psi": float(b.psi),
        "iota": float(b.iota),
        "phi0": float(b.phi0),
    },
    "top_10_cgw_snrs": [float(s["snr"]) for s in top_sources[:10]],
}

# Save to JSON
loudest_file = f"{output_dir}/loudest_cgw_pop{pop_idx}.json"
with open(loudest_file, 'w') as f:
    json.dump(loudest_metadata, f, indent=2)

print(f"\n✓ Loudest candidate metadata saved to: {loudest_file}")

sys.exit(0)
EOFPYTHON

python /tmp/cgw_analysis_pop${POP_IDX}.py \
    "$POPULATION_ZARR" \
    "$POP_IDX" \
    "$INPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: CGW analysis failed for population $POP_IDX"
    exit 1
fi

echo ""
echo "=========================================="
echo "Reduction & CGW complete at: $(date)"
echo "=========================================="
echo ""

if [ "$CLEANUP_INTERMEDIATES" = true ]; then
    echo "Cleaning up intermediates for population $POP_IDX..."
    rm -rf "$POPULATION_ZARR" || true
    rm -f "$INPUT_DIR/chunk_pop${POP_IDX}_idx"*.npz || true
    rm -f "$INPUT_DIR/final_residuals_pop${POP_IDX}.npz" || true
    echo "Cleanup complete for population $POP_IDX"
fi

exit 0
