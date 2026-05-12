#!/bin/bash

################################################################################
# Final Aggregation - Collects all population results
#
# After all reductions complete, this job:
#   1. Collects "loudest CGW candidate" metadata from each population
#   2. Loads consistent population synthesis results from compact JSONs
#   3. Merges everything into a unified results file for notebook
#   4. Creates summary statistics (N_binaries, SNR, CGW SNR distributions)
#
# Usage (called by orchestrator):
#   sbatch slurm_aggregate_results.sh \
#       --input-dir /path/to/chunks \
#       --n-populations 5 \
#       --config pessimistic
################################################################################

#SBATCH --job-name=smbhb_aggregate
# Other SBATCH directives passed from parent script

module purge
unset PYTHONPATH
module load mamba
mamba activate smbhb312

set -e

# Parse arguments
INPUT_DIR=""
N_POPULATIONS=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --n-populations) N_POPULATIONS="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT_DIR" ] || [ -z "$N_POPULATIONS" ] || [ -z "$CONFIG" ]; then
    echo "ERROR: Missing required arguments"
    exit 1
fi

echo "=========================================="
echo "Final Results Aggregation"
echo "Input directory: $INPUT_DIR"
echo "N populations: $N_POPULATIONS"
echo "Config: $CONFIG"
echo "Start time: $(date)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

python << 'EOFPYTHON'
import json
import os
import sys
import glob
import numpy as np
from pathlib import Path

input_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
n_populations = int(sys.argv[2]) if len(sys.argv) > 2 else 1
config = sys.argv[3] if len(sys.argv) > 3 else "pessimistic"

print(f"\nAggregating results from {input_dir}...")
print(f"Expected populations: {n_populations}")

# ========================================================================
# Step 1: Collect loudest CGW candidates
# ========================================================================
print("\nStep 1: Collecting loudest CGW candidates...")

loudest_list = []
for pop_idx in range(n_populations):
    loudest_file = os.path.join(input_dir, f"loudest_cgw_pop{pop_idx}.json")
    if os.path.exists(loudest_file):
        with open(loudest_file, 'r') as f:
            loudest_data = json.load(f)
            loudest_list.append(loudest_data)
            cgw_snr = loudest_data["loudest_cgw"]["cgw_snr"]
            print(f"  Pop {pop_idx}: CGW SNR = {cgw_snr:.4f}")
    else:
        print(f"  ⚠ Pop {pop_idx}: loudest_cgw_pop{pop_idx}.json not found")

print(f"✓ Collected {len(loudest_list)} / {n_populations} populations")

# ========================================================================
# Step 2: Find parent consistent population results
# ========================================================================
print("\nStep 2: Searching for parent consistent population results...")

# Search for consistent_population_*.json files
parent_dir = Path(input_dir).parent
consistent_pop_files = list(parent_dir.glob("consistent_population_*.json"))

if not consistent_pop_files:
    # Try one level up
    consistent_pop_files = list(parent_dir.parent.glob("**/consistent_population_*.json"))

print(f"Found {len(consistent_pop_files)} consistent population results")

consistent_results = []
for json_file in consistent_pop_files:
    print(f"  Loading: {json_file.name}")
    with open(json_file, 'r') as f:
        results = json.load(f)
        consistent_results.append({
            "file": str(json_file),
            "data": results
        })

# ========================================================================
# Step 3: Aggregate statistics
# ========================================================================
print("\nStep 3: Computing aggregate statistics...")

# Extract CGW SNRs from loudest candidates
loudest_cgw_snrs = np.array([l["loudest_cgw"]["cgw_snr"] for l in loudest_list])

# Extract summary stats from consistent results
n_binaries_list = []
snr_list = []

for item in consistent_results:
    results = item["data"]
    if "summary_statistics" in results:
        stats = results["summary_statistics"]
        if "n_bininaries" in stats and "all_values" in stats["n_bininaries"]:
            n_binaries_list.extend(stats["n_bininaries"]["all_values"])
        if "SNR_final" in stats and "all_values" in stats["SNR_final"]:
            snr_list.extend(stats["SNR_final"]["all_values"])

n_binaries_arr = np.array(n_binaries_list)
snr_arr = np.array(snr_list)

# Compute statistics
def stats_dict(arr, name):
    if len(arr) == 0:
        return {}
    return {
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_std": float(np.std(arr)),
        f"{name}_min": float(np.min(arr)),
        f"{name}_max": float(np.max(arr)),
    }

summary = {
    **stats_dict(n_binaries_arr, "n_binaries"),
    **stats_dict(snr_arr, "snr"),
    **stats_dict(loudest_cgw_snrs, "loudest_cgw_snr"),
    "n_populations": n_populations,
    "n_populations_with_cgw_data": len(loudest_list),
}

print(f"\n✓ Aggregate Statistics:")
print(f"  N populations: {summary['n_populations']}")
print(f"  Mean binaries per pop: {summary.get('n_binaries_mean', np.nan):.0f} ± {summary.get('n_binaries_std', np.nan):.0f}")
print(f"  Mean SNR: {summary.get('snr_mean', np.nan):.4f} ± {summary.get('snr_std', np.nan):.4f}")
print(f"  Mean loudest CGW SNR: {summary.get('loudest_cgw_snr_mean', np.nan):.4f} ± {summary.get('loudest_cgw_snr_std', np.nan):.4f}")

# ========================================================================
# Step 4: Create unified results file for notebook
# ========================================================================
print("\nStep 4: Creating unified results file for notebook...")

unified_results = {
    "metadata": {
        "config": config,
        "n_populations": n_populations,
        "n_populations_with_cgw_data": len(loudest_list),
        "input_directory": input_dir,
        "created_at": str(Path(input_dir).stat().st_ctime),
    },
    "summary_statistics": summary,
    "loudest_cgw_candidates": loudest_list,
    "consistent_population_files": [
        {
            "file": item["file"],
            "n_populations": item["data"].get("metadata", {}).get("success_count", 0)
        }
        for item in consistent_results
    ],
    "cgw_snr_distribution": {
        "loudest_per_population": loudest_cgw_snrs.tolist(),
        "n_samples": len(loudest_cgw_snrs),
    },
    "n_binaries_distribution": {
        "all_values": n_binaries_arr.tolist(),
        "n_samples": len(n_binaries_arr),
    } if len(n_binaries_arr) > 0 else {},
}

# Save unified results
output_file = os.path.join(input_dir, "final_aggregated_results.json")
with open(output_file, 'w') as f:
    json.dump(unified_results, f, indent=2)

print(f"✓ Saved unified results to: {output_file}")

# ========================================================================
# Step 5: Create summary report
# ========================================================================
print("\nStep 5: Creating summary report...")

report = f"""
================================================================================
SMBHB SLURM PIPELINE - FINAL RESULTS SUMMARY
================================================================================

Configuration: {config}
Populations analyzed: {n_populations}
Populations with CGW data: {len(loudest_list)}

CONSISTENT POPULATION SYNTHESIS
  N binaries per population:
    Mean:   {summary.get('n_binaries_mean', 'N/A'):.0f}
    Median: {summary.get('n_binaries_median', 'N/A'):.0f}
    Std:    {summary.get('n_binaries_std', 'N/A'):.0f}
    Range:  [{summary.get('n_binaries_min', 'N/A'):.0f}, {summary.get('n_binaries_max', 'N/A'):.0f}]

  SNR achieved per population:
    Mean:   {summary.get('snr_mean', 'N/A'):.4f}
    Median: {summary.get('snr_median', 'N/A'):.4f}
    Std:    {summary.get('snr_std', 'N/A'):.4f}
    Range:  [{summary.get('snr_min', 'N/A'):.4f}, {summary.get('snr_max', 'N/A'):.4f}]

CONTINUOUS GRAVITATIONAL WAVE ANALYSIS
  Loudest CGW candidate SNR per population:
    Mean:   {summary.get('loudest_cgw_snr_mean', 'N/A'):.4f}
    Median: {summary.get('loudest_cgw_snr_median', 'N/A'):.4f}
    Std:    {summary.get('loudest_cgw_snr_std', 'N/A'):.4f}
    Range:  [{summary.get('loudest_cgw_snr_min', 'N/A'):.4f}, {summary.get('loudest_cgw_snr_max', 'N/A'):.4f}]

RESULTS LOCATION
  Input directory: {input_dir}
  Unified results: {output_file}

FOR NOTEBOOK ANALYSIS
  Load results with:
    import json
    with open('{output_file}') as f:
        results = json.load(f)
    
    # Access loudest CGW candidates
    loudest = results['loudest_cgw_candidates']
    
    # Access summary statistics
    stats = results['summary_statistics']
    
    # Plot CGW SNR distribution
    import matplotlib.pyplot as plt
    plt.hist(results['cgw_snr_distribution']['loudest_per_population'], bins=20)
    plt.xlabel('Loudest CGW SNR per Population')
    plt.ylabel('Count')
    plt.show()

================================================================================
"""

report_file = os.path.join(input_dir, "FINAL_REPORT.txt")
with open(report_file, 'w') as f:
    f.write(report)

print(report)
print(f"✓ Saved report to: {report_file}")

EOFPYTHON

python << 'EOFPYTHON'
import sys
sys.exit(0)  # Dummy exit to capture python output
EOFPYTHON

# Manually run the Python aggregation
python3 << 'EOPYTHON'
import json
import os
import glob
import numpy as np
from pathlib import Path

input_dir = os.environ.get('INPUT_DIR', '.')
n_populations = int(os.environ.get('N_POPULATIONS', '1'))
config = os.environ.get('CONFIG', 'pessimistic')

print(f"\nAggregating results from {input_dir}...")

# Collect loudest CGW candidates
loudest_list = []
for pop_idx in range(n_populations):
    loudest_file = os.path.join(input_dir, f"loudest_cgw_pop{pop_idx}.json")
    if os.path.exists(loudest_file):
        with open(loudest_file, 'r') as f:
            loudest_data = json.load(f)
            loudest_list.append(loudest_data)

# Extract CGW SNRs
loudest_cgw_snrs = np.array([l["loudest_cgw"]["cgw_snr"] for l in loudest_list])

# Compute statistics
summary = {
    "n_populations": n_populations,
    "loudest_cgw_snr_mean": float(np.mean(loudest_cgw_snrs)),
    "loudest_cgw_snr_std": float(np.std(loudest_cgw_snrs)),
}

# Create unified results
unified_results = {
    "metadata": {
        "config": config,
        "n_populations": n_populations,
    },
    "summary_statistics": summary,
    "loudest_cgw_candidates": loudest_list,
}

# Save
output_file = os.path.join(input_dir, "final_aggregated_results.json")
with open(output_file, 'w') as f:
    json.dump(unified_results, f, indent=2)

print(f"✓ Saved to: {output_file}")
EOPYTHON

echo ""
echo "=========================================="
echo "Aggregation complete at: $(date)"
echo "Results ready for notebook analysis!"
echo "=========================================="

exit 0
