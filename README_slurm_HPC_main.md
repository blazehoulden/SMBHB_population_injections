# SMBHB Slurm Pipeline

Replaces the monolithic `main.py` with a four-stage Slurm pipeline that
parallelises signal injection across binaries, enabling populations of
100 million+ sources to be processed in wall-clock hours rather than days.

## Architecture

```
Stage 1 (1 job)       Population synthesis + noise simulation
    │                 Outputs: pulsar stoas snapshots, population zarr files
    │
build_task_map        Maps Slurm array task IDs → (pop_idx, chunk_idx)
    │
Stage 2 (N×M tasks)   Chunked injection — one task per (population, chunk)
    │                 Each task injects ~1M binaries and saves ΔTOA residuals
    │
Stage 3 (N tasks)     Reduction — sums all chunks, runs Enterprise + OS/SNR
    │                 One task per population
    │
Stage 4 (1 job)       Aggregation — summary statistics across all populations
```

All physics is identical to `main.py`:
- Same population synthesis (`generate_snr_consistent_populations_distance_scaling`)
- Same noise simulation (`simulate_psr`)
- Same NUFFT injection (`inject_population_nufft`)
- Same Enterprise PTA build (`build_pta_and_params`)
- Same Optimal Statistic (`opt_stat.OptimalStatistic`, HD correlation)

The key insight: **signal injection is linear** — the total TOA residual from
N binaries equals the sum of residuals from each binary individually. So we
can inject chunk-by-chunk, save the delta, and sum at the end. This is
mathematically exact (verified in `local_test.sh`).

## Output Layout

```
<output_dir>/
  metadata/
    config.json                  run parameters
    consistency_summary.json     SNR + n_binaries per population
    psr_names.json               list of pulsar names
    task_map.json                task_id → [pop_idx, chunk_idx]
    Tspan.txt
  pulsars/
    {psr}_toas.npy               raw TOA values (days)
    {psr}_toaerrs.npy            TOA errors (days)
    {psr}_stoas_sim{NNNN}.npy    post-noise stoas for each population
    noise_params.pkl
    parsed_noise_params.pkl
  populations/
    pop_{NNNN}.zarr              binary parameters (zarr store)
  injections/
    pop_{NNNN}/
      chunk_{NNNNNN}.npz         per-pulsar ΔTOA for each chunk (days, float64)
  results/
    pop_{NNNN}_result.json       OS, OS_sig, SNR per population
    pop_{NNNN}_os_xi_rho.npz     cross-correlation arrays for plotting
    summary.json                 aggregated statistics
    summary_table.txt            human-readable table
  logs/
    stage1_*.{out,err}
    stage2_*_*.{out,err}
    stage3_*_*.{out,err}
    stage4_*.{out,err}
```

## Quick Start

### 1. Submit to Slurm (recommended)

```bash
# Minimal — uses all defaults
bash submit_pipeline.sh --output-dir /scratch/$USER/smbhb_run1

# Full options
bash submit_pipeline.sh \
    --config optimistic \
    --target-snr 4.0 \
    --snr-range 3.5 4.25 \
    --simulations 10 \
    --chunk-size 1000000 \
    --output-dir /scratch/$USER/smbhb_run1 \
    --project my_hpc_project \
    --partition compute \
    --python /opt/conda/envs/pta/bin/python

# Preview without submitting
bash submit_pipeline.sh --dry-run --output-dir /scratch/$USER/test
```

### 2. Local smoke test (no Slurm)

```bash
bash local_test.sh --output-dir /tmp/smbhb_test --config optimistic
```

This runs all stages sequentially with 1 population and a small chunk size.

### 3. Plotting / downstream analysis

The results directory contains everything needed:

```python
import json, numpy as np

with open("runs/.../results/summary.json") as f:
    summary = json.load(f)

snrs = [r["SNR"] for r in summary["population_results"]]
print(f"Mean SNR: {np.mean(snrs):.4f} ± {np.std(snrs):.4f}")

# Per-population cross-correlation arrays
data = np.load("runs/.../results/pop_0000_os_xi_rho.npz")
xi, rho, sig = data["xi"], data["rho"], data["sig"]
```

## Tuning Chunk Size

The `--chunk-size` parameter (binaries per injection task) controls the
stage-2 job count and memory use.

| Population size | Chunk size | Stage-2 tasks | Memory/task |
|----------------|------------|---------------|-------------|
| 10M binaries   | 1,000,000  | 10 per pop    | ~8 GB       |
| 100M binaries  | 1,000,000  | 100 per pop   | ~8 GB       |
| 100M binaries  | 5,000,000  | 20 per pop    | ~32 GB      |

Rule of thumb: keep stage-2 tasks under 500 total per submission.

## Resource Guidelines

These are starting points — profile on your cluster.

| Stage | CPUs | Memory | Wall time (100M binaries, 67 pulsars) |
|-------|------|--------|---------------------------------------|
| 1     | 32   | 64 GB  | 4–8 hours                             |
| 2     | 4    | 16 GB  | 15–45 min per chunk                   |
| 3     | 16   | 64 GB  | 1–3 hours per population              |
| 4     | 1    | 4 GB   | < 5 min                               |

Override via `--s2-mem`, `--s3-mem`, `--s2-cpus`, `--s3-cpus`, `--s2-time`,
`--s3-time` flags to `submit_pipeline.sh`.

## Re-running Failed Jobs

Stage 2 is idempotent: if a chunk file already exists it skips injection.
Re-submit failed array tasks with their original task IDs:

```bash
sbatch --array=42,57,103 stage2_inject.py ...
```

Stage 3 is also idempotent in the same way.

## CGW Analysis

Pass `--cgw` to `submit_pipeline.sh` to enable the per-source CGW SNR
analysis inside stage 3 (runs after the OS computation, same job).

```bash
bash submit_pipeline.sh --cgw --output-dir /scratch/$USER/run_cgw
```

Results appear in each `pop_{NNNN}_result.json` under the `"cgw"` key.

### Optimistic settings
sbatch submit_main_HPC.sh --config optimistic --simulations 1 --chunk-size 10_000_000 --n-chunks 1 --cgw --s1-time "00:15:00" --s1-mem "15G" --s2-time "01:30:00" --s2-mem "48G"

### Pessimistic settings
sbatch submit_main_HPC.sh --config pessimistic --simulations 1 --chunk-size 10_000_000 --n-chunks 10  --cgw --s1-time "00:15:00" --s1-mem "15G" --s2-time "01:30:00" --s2-mem "48G"

### Realistic settings
sbatch submit_main_HPC.sh --config realistic --simulations 1 --chunk-size 10_000_000 --n-chunks 10 --cgw --s1-time "00:15:00" --s1-mem "15G" --s2-time "01:30:00" --s2-mem "48G"

## Files

| File                  | Description                              |
|-----------------------|------------------------------------------|
| `stage1_setup.py`     | Population synthesis + zarr export       |
| `stage2_inject.py`    | Array task: chunk injection → Δstoas     |
| `submit_HPC_main.sh`  | Slurm orchestrator (submits all stages)  |
| `local_test.sh`       | Local smoke test (no Slurm needed)       |
