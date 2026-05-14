#!/usr/bin/env python3
"""
Stage 1 — Population synthesis + TOA delta computation.

Called as a Slurm array task; one job per simulation.
  sim_id = SLURM_ARRAY_TASK_ID  (passed via --task-id)

What this job does
──────────────────
1.  Load + filter pulsars (same every job, but cheap).
2.  Generate one SMBHB population shard of --chunk-size binaries using the
    chosen population config.
3.  Compute the per-pulsar TOA change (Δstoas, in days) produced by the
    entire population via the NUFFT injection code.
4.  Save the Δstoas per pulsar to
        <output_dir>/stoas/sim{SIM_ID:04d}/{psr_name}_delta.npy
5.  Save the population shard (downcast to float32/float16) to
        <output_dir>/populations/subpop_{SIM_ID:03d}.pkl.gz
6.  Write (or overwrite — content is idempotent) metadata/config.json.

Stage 2 reads both outputs:
  - The population shard to know binary parameters.
  - The Δstoa files to construct the noise+GW stoas without re-running NUFFT.

Output layout
─────────────
<output_dir>/
  populations/
    subpop_{sim_id:03d}.pkl.gz   — SMBHB binary parameters (PopulationArrays)
  stoas/
    sim{sim_id:04d}/
      {psr_name}_delta.npy       — per-pulsar TOA delta (days, float64)
  metadata/
    config.json                  — run configuration (written by sim_id == 0)
"""

import argparse
import gc
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# ── locate repository root ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from consistent_pop_synth import suppress_enterprise_warnings
from data_loader import load_pulsars, filter_pulsars_15yr
from signal_injection import change_in_TOAs_days_population_nufft
from SMBHB_pop_synth import PopulationArrays


# =============================================================================
# ShardedPickleStore — one gzipped pickle per sub-population
# =============================================================================

# Storage dtypes — angles in float16, everything else float32
FIELD_DTYPES: Dict[str, type] = {
    "f":        np.float32,
    "Mc":       np.float32,
    "Mtot":     np.float32,
    "D_comov":  np.float32,
    "z":        np.float32,
    "h0":       np.float32,
    "ra":       np.float16,
    "dec":      np.float16,
    "psi":      np.float16,
    "iota":     np.float16,
    "phi0":     np.float16,
    "cgw_snr":  np.float16,
}
SCALAR_FIELDS = list(FIELD_DTYPES.keys())


class ShardedPickleStore:
    """
    One pkl.gz per sub-population stored in a single directory.

    Directory layout::

        <root>/
            subpop_000.pkl.gz
            subpop_001.pkl.gz
            ...

    Scalar arrays are downcasted on write (see FIELD_DTYPES).
    Amplitude dicts (amp_A, amp_B) are stored as float32.
    """

    def __init__(self, directory: str, compress_level: int = 6):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.compress_level = compress_level

    # ------------------------------------------------------------------ #
    # Read / write                                                         #
    # ------------------------------------------------------------------ #

    def write(self, idx: int, pop: PopulationArrays) -> None:
        """Downcast and write a sub-population to disk."""
        compact = self._downcast(pop)
        self._dump(self._path(idx), compact)

    def read(self, idx: int) -> PopulationArrays:
        """Load a sub-population from disk."""
        with gzip.open(self._path(idx), "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------ #
    # Partial rewrite                                                      #
    # ------------------------------------------------------------------ #

    def update(
        self,
        idx: int,
        h0:      Optional[np.ndarray] = None,
        D_comov: Optional[np.ndarray] = None,
        z:       Optional[np.ndarray] = None,
        cgw_snr: Optional[np.ndarray] = None,
        amp_A:   Optional[Dict[str, np.ndarray]] = None,
        amp_B:   Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Replace any combination of fields in one load/save cycle.
        Only fields you pass are modified.  Amplitude dicts are merged —
        pulsar entries not in the supplied dict are preserved.
        """
        pop = self.read(idx)

        if h0      is not None: pop.h0      = h0.astype(np.float32)
        if D_comov is not None: pop.D_comov = D_comov.astype(np.float32)
        if z       is not None: pop.z       = z.astype(np.float32)
        if cgw_snr is not None: pop.cgw_snr = cgw_snr.astype(np.float32)

        if amp_A is not None:
            for psr, arr in amp_A.items():
                pop.amp_A[psr] = arr.astype(np.float32)
        if amp_B is not None:
            for psr, arr in amp_B.items():
                pop.amp_B[psr] = arr.astype(np.float32)

        self._dump(self._path(idx), pop)

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def available(self) -> list:
        """Sorted list of shard indices present on disk."""
        return sorted(
            int(p.stem.split("_")[1])
            for p in self.dir.glob("subpop_*.pkl.gz")
        )

    def summary(self) -> None:
        indices = self.available()
        total_mb = sum(
            self._path(i).stat().st_size for i in indices
        ) / 1024 ** 2
        print(f"ShardedPickleStore @ {self.dir}")
        print(f"  shards : {len(indices)}")
        print(f"  total  : {total_mb:.1f} MB")
        for i in indices:
            pop  = self.read(i)
            size = self._path(i).stat().st_size / 1024 ** 2
            print(
                f"  subpop_{i:03d}  N={len(pop):>12,}  "
                f"{size:.2f} MB  "
                f"pulsars={len(pop.amp_A)}"
            )

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _path(self, idx: int) -> Path:
        return self.dir / f"subpop_{idx:03d}.pkl.gz"

    def _dump(self, path: Path, obj) -> None:
        with gzip.open(path, "wb", compresslevel=self.compress_level) as f:
            pickle.dump(obj, f, protocol=5)

    @staticmethod
    def _downcast(pop: PopulationArrays) -> PopulationArrays:
        # Mutate in place rather than constructing a new PopulationArrays.
        # Re-instantiating causes a pickle identity error when sys.path
        # causes SMBHB_pop_synth to be imported under two different module
        # paths — the class used to construct the new object differs from
        # the class pickle sees when it looks up SMBHB_pop_synth.PopulationArrays.
        for name, dtype in FIELD_DTYPES.items():
            setattr(pop, name, getattr(pop, name).astype(dtype))
        for psr in list(pop.amp_A):
            pop.amp_A[psr] = pop.amp_A[psr].astype(np.float32)
        for psr in list(pop.amp_B):
            pop.amp_B[psr] = pop.amp_B[psr].astype(np.float32)
        return pop


# =============================================================================
# Helpers
# =============================================================================

def _save_toa_deltas(
    delta_stoas: list,
    out_dir: str,
    sim_id: int,
) -> None:
    """
    Save per-pulsar Δstoa arrays (days, float64) for one simulation.

    delta_stoas is the raw return value of change_in_TOAs_days_population_nufft:
        [[psr_name, delta_array], ...]

    Path: <out_dir>/stoas/sim{sim_id:04d}/{psr_name}_delta.npy
    """
    sim_dir = os.path.join(out_dir, "stoas", f"sim{sim_id:04d}")
    os.makedirs(sim_dir, exist_ok=True)
    for psr_name, delta in delta_stoas:
        np.save(
            os.path.join(sim_dir, f"{psr_name}_delta.npy"),
            delta.astype(np.float64),
        )
    print(f"  Saved Δstoas for {len(delta_stoas)} pulsars → {sim_dir}")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: population synthesis")
    p.add_argument("--config", "-c", default="optimistic",
                   choices=list(config.POPULATION_CONFIGS.keys()))
    p.add_argument("--target-snr", type=float, default=4.0)
    p.add_argument("--snr-range", nargs=2, type=float, default=[3.5, 4.25])
    p.add_argument("--output-dir", type=str, required=True,
                   help="Root output directory (shared across all stages)")
    p.add_argument("--chunk-size", type=int, default=1_000_000,
                   help="Number of binaries to generate for this population shard")
    p.add_argument("--simulations", type=int, default=10,
                   help="Total number of simulations in the pipeline run (used for metadata only)")
    p.add_argument("--task-id", type=int, default=None,
                   help="Simulation index for this job (overrides $SLURM_ARRAY_TASK_ID)")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t0   = time.time()

    # ── resolve sim_id ────────────────────────────────────────────────────────
    sim_id = args.task_id
    if sim_id is None:
        env_val = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_val is None:
            sys.exit("ERROR: --task-id not set and $SLURM_ARRAY_TASK_ID is not defined.")
        sim_id = int(env_val)

    out_dir  = args.output_dir
    pop_dir  = os.path.join(out_dir, "populations")
    meta_dir = os.path.join(out_dir, "metadata")

    os.makedirs(pop_dir,  exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Stage 1 — sim_id={sim_id}  config={args.config}")
    print(f"{'='*60}")

    # ── 1. Load + filter pulsars ──────────────────────────────────────────────
    print("\n📡 Loading pulsars...")
    psrs_unfiltered = load_pulsars(verbose=True)

    print("\n🔍 Filtering pulsars (15-year array)...")
    with suppress_enterprise_warnings():
        psrs_clean, _, Tspan_seconds = filter_pulsars_15yr(
            psrs_unfiltered, verbose=True
        )

    print(f"✓ {len(psrs_clean)} pulsars, Tspan = {Tspan_seconds / (365.25 * 86400):.1f} yr")

    # ── 2. Generate population ────────────────────────────────────────────────
    print(f"\n🌌 Generating SMBHB population ({args.chunk_size:,} binaries)...")
    selected_config = config.POPULATION_CONFIGS[args.config]
    smbhb_module    = config.load_smbhb_module()

    population_batch = config.generate_population(
        config=selected_config,
        smbhb_module=smbhb_module,
        n_binaries=args.chunk_size,
    )
    print(f"✓ Generated {len(population_batch):,} binaries")

    # ── 3. Compute per-pulsar TOA deltas (NUFFT injection) ────────────────────
    print("\n⚡ Computing TOA changes via NUFFT injection...")
    delta_stoas: Dict[str, np.ndarray] = change_in_TOAs_days_population_nufft(
        psrs_clean, population_batch, verbose=True
    )
    # delta_stoas: {psr_name: ndarray of shape (n_toas,), units=days}
    print(f"✓ Computed Δstoas for {len(delta_stoas)} pulsars")

    # ── 4. Save TOA deltas ────────────────────────────────────────────────────
    _save_toa_deltas(delta_stoas, out_dir, sim_id)

    # ── 5. Save population shard ──────────────────────────────────────────────
    print(f"\n💾 Saving population shard → subpop_{sim_id:03d}.pkl.gz ...")
    store = ShardedPickleStore(pop_dir)
    store.write(sim_id, population_batch)
    print(f"✓ Shard written")

    del population_batch, delta_stoas
    gc.collect()

    # ── 6. Write metadata/config.json (sim_id 0 wins; content is idempotent) ──
    config_path = os.path.join(meta_dir, "config.json")
    if sim_id == 0 or not os.path.exists(config_path):
        config_json = {
            "config":       args.config,
            "target_snr":   args.target_snr,
            "snr_range":    args.snr_range,
            "simulations":  args.simulations,
            "chunk_size":   args.chunk_size,
            "Tspan_seconds": Tspan_seconds,
        }
        with open(config_path, "w") as fh:
            json.dump(config_json, fh, indent=2)
        print(f"✓ Wrote metadata/config.json")

    elapsed = time.time() - t0
    print(f"\n✅ Stage 1 sim_id={sim_id} complete in {elapsed / 60:.1f} min")
    print(f"   Output: {out_dir}")


if __name__ == "__main__":
    main()