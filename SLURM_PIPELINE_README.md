#!/usr/bin/env markdown

# SMBHB Slurm Pipeline: Complete Integration Guide

## Overview

This is a production-ready Slurm pipeline for simulating massive supermassive black hole binary (SMBHB) populations at scale (10M—1B+ binaries) with continuous gravitational wave (CGW) SNR analysis.

**Pipeline Structure**:
```
main.py --chunked-injection
    ↓ (generates populations + writes zarr)
submit_slurm_pipeline.sh
    ├→ Generation Job (writes population zarrs)
    │
    ├→ Per-Population Chunk Array Jobs (parallel)
    │   ├→ Slurm Task 0: chunk_processor.sh (processes chunk 0)
    │   ├→ Slurm Task 1: chunk_processor.sh (processes chunk 1)
    │   └→ ... (100 chunks × 4 cores = 400 parallel tasks)
    │
    ├→ Per-Population Reduction Jobs (depends on chunks)
    │   └→ slurm_reduce_and_cgw.sh (merges chunks + CGW analysis)
    │
    └→ Final Aggregation (depends on all reductions)
        └→ slurm_aggregate_results.sh (collect + export for notebook)
            ↓
            final_aggregated_results.json
            ↓
population_cgw_analysis_pipeline.ipynb
    └→ Load aggregated results
    └→ Compute statistics
    └→ Generate plots & correlations
    └→ Export for publication
```

---

## Quick Start

### Prerequisites
- Conda environment `smbhb312` with enterprise, libstempo, zarr, h5py
- Access to HPC with Slurm scheduler
- Enough storage for zarr files (~100 GB for 100M binaries at float32)

### Step 1: Configuration
Edit [config.py](config.py) to set:
```python
RUN_CONSISTENT_POP_SYNTH = True   # Generate populations
CGW_SNR_ANALYSIS = True            # Analyze CGW SNR
```

### Step 2: Submit Pipeline
```bash
# Generate populations and submit all chunk/reduction jobs
sbatch submit_slurm_pipeline.sh \
    --config pessimistic \
    --n-sims 5 \
    --n-chunks 100

# Output: Master job orchestrates everything
# → Job IDs printed to console
```

Example output:
```
Chunk array jobs:
  Pop 0: 12345678
  Pop 1: 12345679
  ...

Reduction + CGW jobs:
  Pop 0: 12345688 (depends on 12345678)
  Pop 1: 12345689 (depends on 12345679)
  ...

Final aggregation: 12345699 (depends on all reductions)

Monitor progress:
  squeue -u $USER | grep smbhb
```

### Step 3: Wait for Jobs
```bash
# Check status
squeue -u $USER | grep smbhb

# Tail logs
tail -f logs/generation_*.out
tail -f logs/chunks_pop0_*.out    # Array task logs
tail -f logs/reduce_pop0.out      # Reduction logs
tail -f logs/aggregation.out      # Final aggregation
```

### Step 4: Load Results in Notebook
```bash
# Once aggregation completes:
jupyter notebook population_cgw_analysis_pipeline.ipynb

# Update RESULTS_DIR in cell 1 if needed:
# RESULTS_DIR = "data/2026-05-12/pessimistic_pipeline/chunks"
```

---

## File Reference

### Orchestration Scripts
- **[submit_slurm_pipeline.sh](submit_slurm_pipeline.sh)** — Master orchestrator
  - Calls main.py to generate populations + zarr
  - Submits chunk array jobs (one per population, parallel)
  - Submits reduction jobs (one per population, depends on chunks)
  - Submits final aggregation (depends on all reductions)

- **[slurm_chunk_processor.sh](slurm_chunk_processor.sh)** — Per-array-task worker
  - Runs on each task of chunk array job
  - Processes one chunk of a population zarr
  - Calls `chunked_inject_driver.py` with chunk index
  - Outputs: `chunk_pop{idx}_idx{idx}.npz` per task

- **[slurm_reduce_and_cgw.sh](slurm_reduce_and_cgw.sh)** — Reduction + CGW analysis
  - Merges all chunks for one population (IFFT reduction)
  - Runs CGW SNR analysis on the population
  - Extracts loudest (highest CGW SNR) binary
  - Outputs: `loudest_cgw_pop{idx}.json`

- **[slurm_aggregate_results.sh](slurm_aggregate_results.sh)** — Final aggregation
  - Collects `loudest_cgw_pop*.json` from all populations
  - Computes aggregate statistics
  - Creates `final_aggregated_results.json` for notebook

### Python Support
- **[cgw_integration.py](cgw_integration.py)** — CGW analysis utilities
  - `load_loudest_cgw_metadata()` — Load loudest binaries
  - `aggregate_cgw_results()` — Compute statistics
  - `create_notebook_results()` — Export results
  - `print_cgw_summary()` — Print summary

- **[population_cgw_analysis_pipeline.ipynb](population_cgw_analysis_pipeline.ipynb)** — Analysis notebook
  - Loads aggregated results
  - Computes statistics & distributions
  - Generates plots (histograms, scatter, heatmaps)
  - Exports CSV/JSON for publication

### Existing Integration (modified)
- **[main.py](main.py)** — Consistent population synthesis
  - Added `--chunked-injection` flag
  - Writes populations to zarr when flag set
  - Already handles `RUN_CONSISTENT_POP_SYNTH` + `CGW_SNR_ANALYSIS` flags

- **[chunked_inject_driver.py](chunked_inject_driver.py)** — Already implemented
  - Processes population chunks
  - Accumulates frequency-domain arrays
  - Top-K scoring for candidates

- **[io_backends.py](io_backends.py)** — Already implemented
  - Zarr/HDF5 read/write with float32 storage
  - Automatic h5py fallback if zarr unavailable

---

## Configuration Parameters

### Command-line (submit_slurm_pipeline.sh)
```bash
sbatch submit_slurm_pipeline.sh \
    --config pessimistic                    # or 'realistic', 'optimistic'
    --n-sims 10                              # number of populations to generate
    --n-chunks 100                           # chunks per population (adjust for memory)
    --output-dir data/custom/chunks          # optional custom output
    --chunked-test                           # run single local chunk smoke test
```

### HPC Resource Allocation
- **Generation job**: 4 CPUs, 40GB RAM, 30 min (population generation + zarr write)
- **Chunk array tasks**: 1 CPU, 15GB RAM, 30 min per task (100 tasks = 100 core-hours)
- **Reduction jobs**: 4 CPUs, 32GB RAM, 20 min (IFFT merge + CGW analysis)
- **Aggregation job**: 2 CPUs, 8GB RAM, 10 min

### Memory Tuning
If chunk tasks exceed available memory:
```bash
# Reduce chunks per population
--n-chunks 200              # splits population into smaller chunks
```

If aggregation is too slow:
```bash
# Run in parallel on multiple nodes (advanced)
# Modify slurm_aggregate_results.sh to use MPI
```

---

## Troubleshooting

### Issue: "Population zarr not found"
**Cause**: Generation job failed or zarr write incomplete
**Fix**: Check logs:
```bash
tail -f logs/generation_*.out
```

### Issue: "Chunk processing OOM"
**Cause**: Chunk size too large for available node memory
**Fix**: Increase `--n-chunks` to split population smaller:
```bash
sbatch submit_slurm_pipeline.sh --n-chunks 200
```

### Issue: "Aggregation file not found"
**Cause**: Reduction jobs failed before writing loudest_cgw_pop*.json
**Fix**: Check reduction logs:
```bash
cat logs/reduce_pop0.out  # Check for errors
```

### Issue: Notebook can't find results
**Cause**: RESULTS_DIR path incorrect
**Fix**: Update path in notebook cell:
```python
RESULTS_DIR = "data/2026-05-12/pessimistic_pipeline/chunks"
# List contents to verify
import os
os.listdir(RESULTS_DIR)
```

---

## Performance Expectations

### Scaling (pessimistic config: 10M binaries/pop)
- **5 populations × 100 chunks**: ~4 hours (wall-clock, parallel)
  - Generation: 5 min
  - Chunks: 2 hours (parallel across 100 cores)
  - Reductions: 30 min (sequential per population)
  - Aggregation: 5 min
  
- **10 populations × 200 chunks**: ~6 hours
- **20 populations × 500 chunks**: ~12 hours

### Disk Usage (pessimistic, 10M binaries, float32)
- Population zarr: ~800 MB
- Per-chunk outputs: ~10 MB × n_chunks
- Final results: ~10 MB per population
- **Total per 5-population run**: ~50 GB

### Memory
- Per chunk task: ~10–15 GB (depends on n_binaries/n_chunks ratio)
- Reduction job: ~30 GB (loads full population for CGW analysis)
- Aggregation: ~2 GB

---

## Next Steps

### For Publication
1. Run pipeline (10–50 populations for statistics)
2. Load results in notebook
3. Export CSV + statistics
4. Generate comparison plots:
   - CGW SNR distribution
   - Correlations (CGW SNR vs population size, mass, frequency)
   - Top-10 loudest binaries across all populations

### For Detectability Analysis
1. Use loudest CGW SNR statistics to constrain population models
2. Integrate with other PTA backgrounds (GWB, individual events)
3. Forecast sensitivity improvements with future PTAs

### For LSST/Roser & Gondor Comparison
1. Load loudest binary frequencies from notebook
2. Compare to known candidates (14 nHz, 21 nHz)
3. Compute detection probability given target SNRs

---

## Support & Customization

### Add custom binary parameters
Edit `slurm_reduce_and_cgw.sh` → CGW analysis section:
```bash
# Add to loudest_cgw dict:
"custom_param": float(b.custom_property)
```

### Change chunk size
```bash
# In slurm_chunk_processor.sh, modify:
--accum-grid-size 2000       # frequency grid resolution
```

### Add additional analysis
Modify `population_cgw_analysis_pipeline.ipynb`:
```python
# Example: Add sky map plotting
from debug.test_CGW_sky_loc import plot_sky_map
plot_sky_map(df_loud['ra'], df_loud['dec'], df_loud['cgw_snr'])
```

---

## For HPC System Administrators

### Module Requirements
```bash
# Required modules (or conda)
module load gcc/13.2.0
module load hdf5      # for h5py
module load gsl       # for enterprise

# Recommended
module load mamba     # or anaconda/miniconda
```

### Storage Recommendations
- Fast parallel filesystem recommended (Lustre, GPFS, NVMe scratch)
- NFS acceptable but slower for array I/O
- Zarr chunk size 1M binaries = ~4 MB uncompressed, ~1 MB with Blosc

### Job Submission Tuning
```bash
# For massive runs (1000+ chunks), use heterogeneous job:
sbatch --hetjob \
    --array=0-999 slurm_chunk_processor.sh \
    : \
    --dependency=afterok slurm_reduce_and_cgw.sh \
    : \
    --dependency=afterok slurm_aggregate_results.sh
```

---

## References

- Enterprise extensions: [github.com/nanograv/enterprise_extensions](https://github.com/nanograv/enterprise_extensions)
- Zarr storage: [zarr.readthedocs.io](https://zarr.readthedocs.io)
- Blosc compression: [blosc.org](https://blosc.org)
- FINUFFT (if used): [finufft.readthedocs.io](https://finufft.readthedocs.io)

---

**Last Updated**: May 12, 2026
