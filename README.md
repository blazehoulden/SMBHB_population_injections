# SMBHB population injections

This repository contains code for generating synthetic supermassive black-hole
binary (SMBHB) populations and injecting their gravitational-wave signals into
pulsar timing data. It includes population synthesis, signal injection,
optimal-statistic/SNR calculations, scaling analyses, visualisation, and
validation scripts.

The main workflow is in [`main.py`](main.py). The code is intended for both
interactive local experiments and longer runs on a high-performance computing
(HPC) cluster.

> **Status:** This is research software. The default workflow can be
> computationally and memory intensive, so start with a one-simulation smoke
> test before launching a production job.

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Command-line entry point for the analysis pipeline |
| `config.py` | Population presets, data paths, and analysis switches |
| `SMBHB_pop_synth.py` | SMBHB population generation and strain calculations |
| `data_loader.py` | Loading and filtering pulsars and noise parameters |
| `signal_injection.py` | Noise and SMBHB signal injection |
| `consistent_pop_synth.py` | SNR-consistent population synthesis |
| `optimal_SNR_calc.py` | Optimal SNR and noise calculations |
| `visualisation.py` | Plotting and diagnostic output |
| `debug/` | Debugging and validation scripts |
| `old_tempo2_methods/` | Older Tempo2-based implementations; not the main workflow |
| `population_analysis.ipynb` | Interactive analysis notebook |
| `simulating_pta_with_libstempo.ipynb` | Libstempo/ PTA experiments |

## Requirements

The supported environment is currently specified in
[`environment.yml`](environment.yml), which uses Python 3.11. The project
depends on compiled scientific and pulsar-timing packages, including
`libstempo`, `enterprise-pulsar`, `enterprise_extensions`, `numba`, `healpy`,
and `finufft`. Installation can therefore be platform-dependent.

The repository also contains [`requirements.txt`](requirements.txt) as a pip
requirements file. It is less reliable for installing the compiled
dependencies from a completely empty environment; use the conda environment
when possible.

## Required input data

The default configuration expects:

```text
psars_narrowband/par/   # pulsar .par files
psars_narrowband/tim/   # matching pulsar .tim files
15yr_noise.json         # NANOGrav 15-year noise parameters
```

For each pulsar, the loader looks for a matching timing file. For example:

```text
psars_narrowband/par/J1713+0747.par
psars_narrowband/tim/J1713+0747.tim
```

The exact filenames supplied by the NANOGrav data release may differ. By
default, the project resolves these paths relative to the repository root,
even when the command is launched from another directory. You can override
them without editing source code:

```bash
export SMBHB_PAR_DIR=/path/to/par
export SMBHB_TIM_DIR=/path/to/tim
export SMBHB_NOISE_FILE=/path/to/15yr_noise.json
export SMBHB_PULSAR_CACHE=/path/to/nanograv_pulsars_cache.pkl
```

Relative override paths are resolved relative to the repository root. The
corresponding Python settings are `PAR_DIR`, `TIM_DIR`, and `NOISEFILE` in
`config.py`.

`15yr_noise_params.json` is retained for auxiliary analyses; the main
pipeline currently reads `15yr_noise.json` through `config.NOISEFILE`.

The loader creates `nanograv_pulsars_cache.pkl` after a successful load when
`USE_PULSAR_CACHE = True`. This cache is machine-specific and should not be
copied between incompatible environments.

## Local installation

### Option A: conda (recommended)

From the repository root:

```bash
conda env create -f environment.yml
conda activate SMBHB312
```

If the environment already exists and `environment.yml` has changed:

```bash
conda env update -f environment.yml --prune
conda activate SMBHB312
```

Check that the key imports work:

```bash
python - <<'PY'
import astropy
import libstempo
import enterprise
import enterprise_extensions
import finufft
print("Core imports succeeded")
PY
```

### Option B: pip

Use a supported Python installation and create an isolated virtual
environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installation fails while building `libstempo` or another compiled package,
use the conda environment or follow the cluster-specific installation
instructions below.

## First run: smoke test

Run commands from any directory. The default input paths are resolved relative
to the repository root, or can be overridden with the environment variables
above.

Before a full simulation, use one simulation and a dedicated output directory:

```bash
python main.py \
  --config optimistic \
  --simulations 1 \
  --save-name smoke-test \
  --save-dir outputs/smoke-test
```

The command should load the available pulsars, filter them using the
15-year noise file, generate the selected population, and write results under
`outputs/smoke-test/`.

See all available command-line options with:

```bash
python main.py --help
```

## Running the main workflow

The population preset is selected with `--config`:

| Preset | Approximate population size | Intended use |
| --- | ---: | --- |
| `optimistic` | 1,000 binaries | Fast exploratory runs |
| `realistic` | 20,000 binaries | Main analysis case |
| `pessimistic` | 10,000,000 binaries | Large, resource-intensive studies |

Example local run:

```bash
python main.py \
  --config realistic \
  --simulations 10 \
  --target-snr 4.0 \
  --snr-range 3.5 4.0 \
  --save-name realistic-run-001 \
  --save-dir outputs/realistic-run-001
```

Important options:

```text
--config, -c       optimistic, realistic, or pessimistic
--simulations, -s  Number of population realisations
--target-snr       Target SNR for analyses that use a target
--snr-range        Lower and upper SNR range
--initial-guess    Initial number of binaries, or "auto"
--save-name        Label used in output names
--save-dir         Explicit output directory
--save-nearest     Number of nearest binaries retained in compact results
--save-loudest     Number of loudest binaries retained in compact results
--seed             Reproducibility seed; overrides --noise-seed-base
--dry-run          Validate configuration and data paths without running
--list-configs     List available population presets
```

The analysis stages are controlled by switches in `config.py`, for example:

```python
RUN_CONSISTENT_POP_SYNTH = True
RUN_SCALING_ANALYSIS = False
RUN_INDIVIDUAL_BINARY_ANALYSIS = False
MEMORY_PROFILE_ENABLED = True
```

These switches currently require editing `config.py`. Keep a copy of any
configuration used for a production run, and record the Git commit alongside
the output.

Every normal run now writes `run_metadata.json` and `run.log` into its output
directory. The metadata records the command, Git commit, Python version,
selected options, and resolved input paths. Use `--seed` for repeatable
stage1/stage2 noise and population seeding:

```bash
python main.py --config test --simulations 1 --seed 12345 \
  --save-dir outputs/test-seed-12345
```

Before a long run, inspect the selected configuration and paths without
starting the pipeline:

```bash
python main.py --config test --dry-run
python main.py --list-configs
```

## Running on an HPC cluster

Install the environment once in a persistent location, ideally on a shared
software or project filesystem:

```bash
conda env create -f environment.yml
conda activate SMBHB312
```

If compute nodes cannot access the internet, create the environment on a
login or build node first. Do not install packages into the system Python.

Copy or link the pulsar data and noise file into the project location visible
to compute nodes. Confirm the paths in `config.py` and test interactively on a
short allocation before submitting a production job.

### Example SLURM job

Save the following as `run_smbhb.slurm` and adapt the account, partition,
module names, memory, and time limit to your cluster:

```bash
#!/bin/bash
#SBATCH --job-name=smbhb
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

# Use the cluster's conda initialisation method if it differs.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SMBHB312

python --version
python main.py \
  --config realistic \
  --simulations 10 \
  --target-snr 4.0 \
  --save-name "slurm-${SLURM_JOB_ID}" \
  --save-dir "outputs/slurm-${SLURM_JOB_ID}"
```

Submit and monitor it with:

```bash
sbatch run_smbhb.slurm
squeue --me
tail -f logs/smbhb-<JOBID>.out
```

For a job array, use the array index to keep outputs separate:

```bash
#SBATCH --array=0-9
```

and include `${SLURM_ARRAY_TASK_ID}` in `--save-name` and `--save-dir`.
Use separate output directories for separate jobs; otherwise results can
overwrite each other.

### HPC resource guidance

- Start with `optimistic` and `--simulations 1`.
- Request more memory for `realistic`, and especially `pessimistic`,
  populations.
- Avoid running several large simulations in one process unless the memory
  requirement has been measured.
- Keep the pulsar cache on fast, persistent storage when possible.
- Write results to a job-specific directory rather than a shared filename.
- Check the scheduler's memory and wall-time reports before scaling up.

The `pessimistic` preset contains ten million binaries and should not be used
as a first test. It may require changes to the job resources and storage
strategy.

## Outputs

By default, `main.py` creates a date-based directory under `data/`. Supplying
`--save-dir` is recommended so that generated files are clearly separated
from input data and version-controlled source:

```text
outputs/
└── realistic-run-001/
    └── consistent_population_realistic_targetSNR4.0_sims10.json
```

Plots created by older analysis functions may still be written to
`figures/`. Treat files in `data/`, `figures/`, and `outputs/` as generated
artifacts unless they are deliberately retained as example results.

For reproducible studies:

1. Use `--seed` and keep the generated `run_metadata.json`.
2. Record the environment with `conda env export --no-builds`.
3. Preserve the input noise-file and pulsar-data provenance.
4. Keep separate output directories for each configuration and seed.

## Dependency environment maintenance

`environment.yml` is a hand-maintained, Python 3.11 conda environment rather
than a guaranteed export from one machine. To determine whether it is the
correct environment for a cluster, create it on a clean test allocation and
run:

```bash
conda env create -f environment.yml
conda activate SMBHB312
python -m py_compile config.py main.py stage1_setup.py stage2_inject.py
python -m unittest discover -s tests -p "test_config.py"
python - <<'PY'
import astropy, enterprise, enterprise_extensions, finufft, libstempo, numba
print("scientific and pulsar-timing imports succeeded")
PY
```

After a successful installation, capture the tested environment:

```bash
conda env export --no-builds > environment-tested.yml
```

Do not replace `environment.yml` with that export until it has been tested on
the target HPC system. Compare the files and keep only packages actually
required by the supported workflow. In particular, check the cluster's
available compiler, MPI/OpenMP configuration, and whether `libstempo`,
`enterprise_extensions`, and `finufft` are available from the chosen channels.
The repository's quick CI checks intentionally avoid these compiled
dependencies; full scientific validation must be done in the target conda
environment.

The existing `tests/test_chunked_pipeline.py` is an integration test for the
chunked pipeline and requires the full scientific environment plus the
chunked I/O dependencies. It is not part of the dependency-free CI check.

## Notebooks and validation scripts

Install and register the environment as a Jupyter kernel if needed:

```bash
conda activate SMBHB312
python -m ipykernel install --user --name SMBHB312 --display-name "Python (SMBHB312)"
jupyter lab
```

When using a notebook, start Jupyter from the repository root or set the
`SMBHB_PAR_DIR`, `SMBHB_TIM_DIR`, and `SMBHB_NOISE_FILE` variables before
launching it.

Useful validation scripts include:

```bash
python validate_injection.py
python debug/validate_population.py
```

These scripts can create plots in the current directory. Run them from a
temporary output directory if you do not want to modify the repository root.

## Troubleshooting

### `FileNotFoundError` for pulsar or noise files

Check the configured paths and input files:

```bash
python - <<'PY'
from config import PAR_DIR, TIM_DIR, NOISEFILE
print("PAR_DIR:", PAR_DIR)
print("TIM_DIR:", TIM_DIR)
print("NOISEFILE:", NOISEFILE)
PY
```

### No pulsars are loaded

Check that each `.par` file has a matching `.tim` file, that the `.tim` file
contains valid TOAs, and that the files are readable from the compute node.
Delete `nanograv_pulsars_cache.pkl` if it was created from an incomplete or
incompatible data set, then rerun the loader.

### Import or binary-library errors

Confirm that the intended environment is active:

```bash
which python
python --version
conda info --envs
```

Compiled dependencies are the most common reason to prefer the conda
installation over a plain pip installation.

### A run is too slow or exceeds memory

Reduce `--simulations`, use the `optimistic` preset, disable optional analysis
switches in `config.py`, and request more memory or wall time on the HPC
cluster. Do not begin with the `pessimistic` preset.

## Citation and provenance

Before publishing results generated with this repository, record the Git
revision, input data release, environment, population configuration, random
seed (when configured), and command line. Add project-specific citations for
the NANOGrav data release, Enterprise, libstempo, and any astrophysical model
used in the analysis.
