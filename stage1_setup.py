#!/usr/bin/env python3
"""
Stage 1 — Population synthesis + TOA delta computation.

Called as a Slurm flat array task.
  task_id  = SLURM_ARRAY_TASK_ID
  sim_id   = task_id // n_chunks
  chunk_id = task_id  % n_chunks

What this job does
──────────────────
1.  Decode sim_id and chunk_id from the flat task_id.
2.  Load + filter pulsars.
3.  Generate one chunk of --chunk-size binaries for sim_id.
4.  Compute per-pulsar TOA deltas via NUFFT injection.
5.  Save deltas to   <output_dir>/sim{sim_id:03d}/stoas/sim{chunk_id:04d}/{psr}_delta.npy
6.  Save population  <output_dir>/sim{sim_id:03d}/populations/subpop_{chunk_id:03d}.pkl.gz
7.  Write metadata   <output_dir>/sim{sim_id:03d}/metadata/config.json  (chunk_id==0 wins)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from consistent_pop_synth import suppress_enterprise_warnings
from data_loader import load_pulsars, filter_pulsars_15yr
from signal_injection import change_in_TOAs_days_population_nufft
from SMBHB_pop_synth import PopulationArrays
from debug.test_CGW_sky_loc import sky_sensitivity_weight

# =============================================================================
# ShardedPickleStore
# =============================================================================

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
N_PRE_FILTER_PER_CHUNK  = 2_500


class ShardedPickleStore:
    """One pkl.gz per chunk stored in a single directory."""

    def __init__(self, directory: str, compress_level: int = 6):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.compress_level = compress_level

    def write(self, idx: int, pop: PopulationArrays) -> None:
        compact = self._downcast(pop)
        self._dump(self._path(idx), compact)

    def read(self, idx: int) -> PopulationArrays:
        with gzip.open(self._path(idx), "rb") as f:
            return pickle.load(f)

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

    def available(self) -> list:
        return sorted(
            int(p.name.split("_")[1].split(".")[0])
            for p in self.dir.glob("subpop_*.pkl.gz")
        )

    def _path(self, idx: int) -> Path:
        return self.dir / f"subpop_{idx:03d}.pkl.gz"

    def _dump(self, path: Path, obj) -> None:
        with gzip.open(path, "wb", compresslevel=self.compress_level) as f:
            pickle.dump(obj, f, protocol=5)

    @staticmethod
    def _downcast(pop: PopulationArrays) -> PopulationArrays:
        for name, dtype in FIELD_DTYPES.items():
            setattr(pop, name, getattr(pop, name).astype(dtype))
        for psr in list(pop.amp_A):
            pop.amp_A[psr] = pop.amp_A[psr].astype(np.float32)
        for psr in list(pop.amp_B):
            pop.amp_B[psr] = pop.amp_B[psr].astype(np.float32)
        return pop

def _filter_population_extremes(
    pop: PopulationArrays,
    n_keep: int = 100,
    n_total: int = None,
) -> PopulationArrays:
    """
    Keep extreme binaries across all parameters PLUS the top binaries by
    CGW proxy SNR (h0 / (2π f) * sky_sensitivity_weight).

    The CGW-proxy slice fills up to `n_total` slots (defaulting to
    N_PRE_FILTERED if defined, else 30 * n_keep), with parameter-extreme
    binaries unioned in on top.
    """
    n = len(pop)

    # How many total to keep (CGW-proxy dominated)
    if n_total is None:
        n_total = globals().get("N_PRE_FILTERED", 30 * n_keep)
    n_total = min(n_total, n)
    n_keep  = min(n_keep, n // 2)   # safety cap on extremes slice

    indices = set()

    # ── 1. CGW proxy SNR (primary filter) ───────────────────────────────────
    proxies = (pop.h0 / (2.0 * np.pi * pop.f)) * np.array([
        sky_sensitivity_weight(pop.ra[i], pop.dec[i]) for i in range(n)
    ])
    n_cgw = min(n_total, n)
    if n_cgw == n:
        cgw_top = np.argsort(proxies)[::-1]
    else:
        cgw_top = np.argpartition(proxies, -n_cgw)[-n_cgw:]
        cgw_top = cgw_top[np.argsort(proxies[cgw_top])[::-1]]
    indices.update(cgw_top.tolist())

    # ── 2. Parameter extremes (union'd in so nothing rare is missed) ─────────
    for arr in (pop.f, pop.D_comov, pop.h0, pop.Mc, pop.Mtot):
        order = np.argsort(arr)
        indices.update(order[:n_keep].tolist())
        indices.update(order[-n_keep:].tolist())
    # h0 / loudest: top only (no "quietest" extreme needed)
    h0_order = np.argsort(pop.h0)
    indices.update(h0_order[-n_keep:].tolist())

    idx = np.array(sorted(indices))
    print(
        f"  Population filtered: {n:,} → {len(idx):,} kept "
        f"(top-{n_cgw} CGW-proxy + {len(idx) - n_cgw} parameter extremes)"
    )
    return pop[idx]

# =============================================================================
# Helpers
# =============================================================================

def _save_toa_deltas(delta_stoas, out_dir: str, chunk_id: int) -> None:
    """Save per-pulsar Δstoa arrays for one chunk."""
    sim_dir = os.path.join(out_dir, "stoas", f"sim{chunk_id:04d}")
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
    p.add_argument("--target-snr",  type=float, default=4.0)
    p.add_argument("--snr-range",   nargs=2, type=float, default=[3.5, 4.25])
    p.add_argument("--output-dir",  type=str, required=True)
    p.add_argument("--chunk-size",  type=int, default=1_000_000)
    p.add_argument("--n-chunks",    type=int, default=10,
                   help="Number of chunks per simulation (used to decode task_id)")
    p.add_argument("--task-id",     type=int, default=None,
                   help="Flat array task ID (overrides $SLURM_ARRAY_TASK_ID)")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t0   = time.time()

    # ── decode flat task_id → sim_id, chunk_id ────────────────────────────────
    task_id = args.task_id
    if task_id is None:
        env_val = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_val is None:
            sys.exit("ERROR: --task-id not set and $SLURM_ARRAY_TASK_ID is not defined.")
        task_id = int(env_val)

    sim_id   = task_id // args.n_chunks
    chunk_id = task_id  % args.n_chunks

    # Per-simulation output directory
    sim_out_dir = os.path.join(args.output_dir, f"sim{sim_id:03d}")
    pop_dir     = os.path.join(sim_out_dir, "populations")
    meta_dir    = os.path.join(sim_out_dir, "metadata")

    os.makedirs(pop_dir,  exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Stage 1 — task_id={task_id}  sim_id={sim_id}  chunk_id={chunk_id}")
    print(f"  config={args.config}  chunk_size={args.chunk_size:,}")
    print(f"  output → {sim_out_dir}")
    print(f"{'='*60}")

    # ── 1. Load + filter pulsars ──────────────────────────────────────────────
    print("\n📡 Loading pulsars...")
    psrs_unfiltered = load_pulsars(verbose=True)

    print("\n🔍 Filtering pulsars (15-year array)...")
    with suppress_enterprise_warnings():
        psrs_clean, _, Tspan_seconds = filter_pulsars_15yr(psrs_unfiltered, verbose=True)
    print(f"✓ {len(psrs_clean)} pulsars, Tspan = {Tspan_seconds / (365.25 * 86400):.1f} yr")

    # ── 2. Generate population chunk ──────────────────────────────────────────
    print(f"\n🌌 Generating population chunk ({args.chunk_size:,} binaries)...")
    selected_config  = config.POPULATION_CONFIGS[args.config]
    smbhb_module     = config.load_smbhb_module()
    population_batch = config.generate_population(
        config=selected_config,
        smbhb_module=smbhb_module,
        n_binaries=args.chunk_size,
    )
    print(f"✓ Generated {len(population_batch):,} binaries")

    # ── 3. Compute TOA deltas ─────────────────────────────────────────────────
    print("\n⚡ Computing TOA changes via NUFFT injection...")
    delta_stoas = change_in_TOAs_days_population_nufft(
        psrs_clean, population_batch, verbose=True
    )
    print(f"✓ Computed Δstoas for {len(delta_stoas)} pulsars")

    # ── 4. Save TOA deltas ────────────────────────────────────────────────────
    _save_toa_deltas(delta_stoas, sim_out_dir, chunk_id)

    # ── 5. Save population shard ──────────────────────────────────────────────
    print(f"\n💾 Filtering to extreme binaries...")
    population_batch = _filter_population_extremes(population_batch, n_keep=100, )

    print(f"\n💾 Saving population shard → subpop_{chunk_id:03d}.pkl.gz ...")
    store = ShardedPickleStore(pop_dir)
    store.write(chunk_id, population_batch)
    print(f"✓ Shard written")

    del population_batch, delta_stoas
    gc.collect()

    # ── 6. Write metadata (chunk_id 0 wins; content is idempotent) ────────────
    config_path = os.path.join(meta_dir, "config.json")
    if chunk_id == 0 or not os.path.exists(config_path):
        config_json = {
            "config":         args.config,
            "target_snr":     args.target_snr,
            "snr_range":      args.snr_range,
            "n_chunks":       args.n_chunks,
            "chunk_size":     args.chunk_size,
            "Tspan_seconds":  Tspan_seconds,
        }
        with open(config_path, "w") as fh:
            json.dump(config_json, fh, indent=2)
        print(f"✓ Wrote metadata/config.json")

    elapsed = time.time() - t0
    print(f"\n✅ Stage 1 task_id={task_id} (sim={sim_id}, chunk={chunk_id}) "
          f"complete in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()