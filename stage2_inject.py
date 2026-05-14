#!/usr/bin/env python3
"""
Stage 2 — Noise simulation, SNR scaling, and CGW analysis.

Called as a Slurm array task; one job per simulation.
  sim_id = SLURM_ARRAY_TASK_ID  (passed via --task-id)

What this job does
──────────────────
1.  Load + filter pulsars (same list every job).
2.  Load the population shard for this sim_id from ShardedPickleStore.
3.  Sum the per-pulsar Δstoa files written by stage 1 to get the total GW
    TOA contribution for this population.
4.  Simulate noise for each pulsar to produce noise-only stoas.
5.  Add the GW Δstoas to the noise-only stoas → noise+GW stoas.
6.  Compute the optimal-statistic SNR using Enterprise + the OS module.
7.  If SNR is outside [snr_low, snr_high], scale the GW signal by adjusting
    distances and redshifts, then iterate until convergence (or raise after
    max_iterations).
8.  Apply the final distance/redshift/h0 scaling to every binary in the
    population shard (in place via ShardedPickleStore.update).
9.  Pre-filter the population by a sky-weighted h0/f proxy, compute CGW SNRs
    for the top N_PRE_FILTER binaries, and store the results back into the
    shard via ShardedPickleStore.update(cgw_snr=...).

Inputs (from stage 1)
─────────────────────
  <output_dir>/populations/subpop_{sim_id:03d}.pkl.gz
  <output_dir>/stoas/sim{sim_id:04d}/{psr_name}_delta.npy   (one per pulsar)

Outputs (written / updated by this job)
───────────────────────────────────────
  <output_dir>/populations/subpop_{sim_id:03d}.pkl.gz  — updated h0/D_comov/z
                                                          + cgw_snr field
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from consistent_pop_synth import compute_population_snr, suppress_enterprise_warnings
from data_loader import load_pulsars, filter_pulsars_15yr, parse_pulsar_parameters
from signal_injection import simulate_psr
from stage1_setup import ShardedPickleStore
from CGW_SNR import compute_cgw_snr_optimal_population
from debug.test_CGW_sky_loc import sky_sensitivity_weight

# Number of binaries to evaluate CGW SNR for (pre-filtered by sky-weighted proxy)
N_PRE_FILTER  = 2_000
N_TOP_SOURCES = 50

# Maximum SNR-scaling iterations before giving up
MAX_SCALE_ITER = 20


# =============================================================================
# Helpers
# =============================================================================

def _load_toa_deltas(
    out_dir: str,
    sim_id: int,
    psr_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Load the per-pulsar Δstoa arrays saved by stage 1 for sim_id.

    Returns {psr_name: delta_array (days, float64)}.
    Exits with an error if any pulsar file is missing.
    """
    sim_dir = os.path.join(out_dir, "stoas", f"sim{sim_id:04d}")
    if not os.path.isdir(sim_dir):
        sys.exit(
            f"ERROR: stage-1 stoa directory not found: {sim_dir}\n"
            f"Ensure stage1_setup.py completed for sim_id={sim_id}."
        )

    delta_stoas: Dict[str, np.ndarray] = {}
    for name in psr_names:
        fpath = os.path.join(sim_dir, f"{name}_delta.npy")
        if not os.path.isfile(fpath):
            sys.exit(
                f"ERROR: missing delta file for pulsar '{name}': {fpath}"
            )
        delta_stoas[name] = np.load(fpath).astype(np.float64)

    print(f"  Loaded Δstoas for {len(delta_stoas)} pulsars from {sim_dir}")
    return delta_stoas


def _comov_redshift_from_scaling(
    D_comov: np.ndarray,
    z: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a distance scaling factor, find the nearest (D_comov, z) grid point
    such that h0 ∝ (1+z)^(2/3) / D_comov scales by `scale`.

    Vectorised over (N,) input arrays.
    Returns new_D_comov, new_z (both shape (N,), float32).
    """
    from SMBHB_pop_synth import _Z_GRID, _CHI_GRID  # precomputed cosmology grids

    # For each binary the target value of (1+z)^(2/3) / D_comov after scaling
    targets = (1.0 + z) ** (2.0 / 3.0) / D_comov / scale  # shape (N,)

    # Grid of (1+z)^(2/3) / D_comov values — computed once
    frac_grid = (1.0 + _Z_GRID) ** (2.0 / 3.0) / _CHI_GRID  # shape (G,)

    # Nearest grid point for each binary — shape (N,)
    idx = np.argmin(np.abs(frac_grid[None, :] - targets[:, None]), axis=1)

    return _CHI_GRID[idx].astype(np.float32), _Z_GRID[idx].astype(np.float32)


def _scale_and_iterate(
    psrs_clean,
    delta_stoas: Dict[str, np.ndarray],
    noise_stoas: Dict[str, np.ndarray],
    target_snr: float,
    snr_low: float,
    snr_high: float,
    Tspan_seconds: float,
    raw_noise_params: Dict[str, Dict],
    max_iterations: int = MAX_SCALE_ITER,
) -> Tuple[float, object, list]:
    noise_only_snr = compute_population_snr(
        psrs_clean=psrs_clean,
        population=None,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan_seconds,
        current_stoas=noise_stoas,
        return_psrs_pta=False,
    )
    print(f"  Noise-only OS SNR: {noise_only_snr:.4f}")

    def _empirical_scale(history, cumulative_scale):
        xs   = np.array([h['cumulative_scale'] for h in history])
        snrs = np.array([h['snr'] for h in history])

        above = [(x, s) for x, s in zip(xs, snrs) if s >= target_snr]
        below = [(x, s) for x, s in zip(xs, snrs) if s <  target_snr]

        if above and below:
            best_above = max(above, key=lambda p: p[0])
            best_below = min(below, key=lambda p: p[0])
            if abs(best_above[0] - best_below[0]) < 1e-10:
                return None
            log_mid    = 0.5 * (np.log(best_above[0]) + np.log(best_below[0]))
            cum_target = np.exp(log_mid)
            if abs(cum_target - cumulative_scale) / cumulative_scale < 1e-4:
                cum_target = best_above[0]
            incremental = cum_target / cumulative_scale
            print(f"  [bisection] bracket: "
                  f"below=({best_below[0]:.4f}×, SNR={best_below[1]:.4f})  "
                  f"above=({best_above[0]:.4f}×, SNR={best_above[1]:.4f})  "
                  f"→ mid={cum_target:.4f}×  incremental={incremental:.4f}")
            return incremental

        # No bracket yet — fall back to analytic
        return None

    def _analytic_scale(snr_current, snr_target, snr_noise_baseline):
        snr_signal_current = snr_current - snr_noise_baseline
        snr_signal_target  = snr_target  - snr_noise_baseline
        if snr_signal_current <= 0:
            raise ValueError(
                f"Signal-only SNR is non-positive ({snr_signal_current:.4f}). "
                "The GW signal may be too weak relative to noise."
            )
        if snr_signal_target <= 0:
            raise ValueError(
                f"Target signal SNR is non-positive ({snr_signal_target:.4f}). "
                "Check target SNR and noise level."
            )
        return np.sqrt(snr_signal_current / snr_signal_target)

    cumulative_scale = 1.0
    snr_history = []
    pta = None
    enterprise_psrs = None

    for iteration in range(max_iterations):
        signal_stoas = {
            name: noise_stoas[name] + delta_stoas[name]
            for name in noise_stoas
        }
        snr, pta, enterprise_psrs = compute_population_snr(
            psrs_clean=psrs_clean,
            population=None,
            raw_noise_params=raw_noise_params,
            Tspan=Tspan_seconds,
            current_stoas=signal_stoas,
        )
        print(f"  Iteration {iteration + 1:2d}: OS SNR = {snr:.4f}  "
              f"(target [{snr_low}, {snr_high}])  cumulative_scale={cumulative_scale:.4f}×")

        if snr_low <= snr <= snr_high:
            print(f"  ✓ Converged at iteration {iteration + 1}")
            return cumulative_scale, pta, enterprise_psrs

        snr_history.append({'cumulative_scale': cumulative_scale, 'snr': snr})

        if len(snr_history) == 1:
            factor = _analytic_scale(snr, target_snr, noise_only_snr)
            print(f"  [analytic] factor={factor:.4f}")
        else:
            factor = _empirical_scale(snr_history, cumulative_scale)
            if factor is None:
                factor = _analytic_scale(snr, target_snr, noise_only_snr)
                print(f"  [analytic fallback] factor={factor:.4f}")

        factor = float(np.clip(factor, 0.1, 10.0))
        for name in delta_stoas:
            delta_stoas[name] /= factor
        cumulative_scale /= factor

    raise RuntimeError(
        f"SNR scaling failed to converge within {max_iterations} iterations. "
        f"Last SNR={snr:.4f}, target=[{snr_low}, {snr_high}]."
    )


def _update_population_scaling(
    store: ShardedPickleStore,
    sim_id: int,
    cumulative_scale: float,
) -> None:
    """
    Apply the amplitude scaling to the stored population shard.

    h0  → h0 / cumulative_scale   (larger scale = sources moved further away)
    D_comov, z updated to nearest cosmological grid point consistent with
    the new h0 ∝ (1+z)^(2/3) / D_comov.
    """
    pop = store.read(sim_id)

    new_h0 = pop.h0 / cumulative_scale
    new_D_comov, new_z = _comov_redshift_from_scaling(
        pop.D_comov, pop.z, 1.0 / cumulative_scale
    )

    store.update(sim_id, h0=new_h0, D_comov=new_D_comov, z=new_z)
    print(
        f"  Updated shard: h0 scaled by 1/{cumulative_scale:.4f}, "
        f"D_comov and z remapped on cosmological grid"
    )


def _compute_cgw_snrs(
    sim_id: int,
    store: ShardedPickleStore,
    pta,
    enterprise_psrs,
    raw_noise_params,
    parsed_noise_params,
    Tspan_seconds: float,
) -> None:
    pop = store.read(sim_id)
    n_binaries = len(pop)
    n_filter = min(N_PRE_FILTER, n_binaries)

    print(f"\n  CGW analysis: pre-filtering {n_binaries:,} binaries → top {n_filter}...")
    if n_filter < N_PRE_FILTER:
        print(f"  (population smaller than N_PRE_FILTER={N_PRE_FILTER}, using all {n_binaries} binaries)")

    proxies = (pop.h0 / (2.0 * np.pi * pop.f)) * np.array([
        sky_sensitivity_weight(pop.ra[i], pop.dec[i]) for i in range(n_binaries)
    ])

    if n_filter == n_binaries:
        top_indices = np.argsort(proxies)[::-1]
    else:
        top_indices = np.argpartition(proxies, -n_filter)[-n_filter:]
        top_indices = top_indices[np.argsort(proxies[top_indices])[::-1]]

    top_binaries = [pop[i] for i in top_indices]

    print(f"  Computing CGW SNR for {len(top_binaries)} candidates...")
    top_snrs = compute_cgw_snr_optimal_population(
        psrs=enterprise_psrs,
        pta=pta,
        population=top_binaries,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan=Tspan_seconds,
        profile=True,
    )

    cgw_snr_full = np.zeros(n_binaries, dtype=np.float32)
    for local_rank, global_idx in enumerate(top_indices):
        cgw_snr_full[global_idx] = top_snrs[local_rank]
    store.update(sim_id, cgw_snr=cgw_snr_full)

    ranked = sorted(zip(top_indices, top_snrs), key=lambda x: x[1], reverse=True)
    n_show = min(N_TOP_SOURCES, len(ranked))
    print(f"\n  Top {n_show} CGW candidates:")
    for rank, (global_idx, snr) in enumerate(ranked[:n_show], start=1):
        print(
            f"    {rank:2d}.  global_idx={global_idx:7d}  "
            f"f={pop.f[global_idx]:.2e} Hz  "
            f"Mc={pop.Mc[global_idx]:.2e} Msun  "
            f"h0={pop.h0[global_idx]:.2e}  "
            f"ra={np.degrees(pop.ra[global_idx]):.1f}°  "
            f"dec={np.degrees(pop.dec[global_idx]):.1f}°  "
            f"CGW_SNR={snr:.4f}"
        )
    print(f"\n  CGW SNR stored for {n_filter} binaries (zeroed elsewhere)")

# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: noise sim + SNR scaling + CGW analysis")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--config", "-c", default="optimistic",
                   choices=list(config.POPULATION_CONFIGS.keys()))
    p.add_argument("--target-snr", type=float, default=4.0)
    p.add_argument("--snr-range", nargs=2, type=float, default=[3.5, 4.25])
    p.add_argument("--task-id", type=int, default=None,
                   help="Override $SLURM_ARRAY_TASK_ID (for local testing)")
    p.add_argument("--cgw", action="store_true",
                   help="Enable CGW SNR computation (default: off)")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args    = parse_args()
    t0      = time.time()
    out_dir = args.output_dir

    # ── resolve sim_id ────────────────────────────────────────────────────────
    sim_id = args.task_id
    if sim_id is None:
        env_val = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_val is None:
            sys.exit("ERROR: --task-id not set and $SLURM_ARRAY_TASK_ID is not defined.")
        sim_id = int(env_val)

    print(f"\n{'='*60}")
    print(f"Stage 2 — sim_id={sim_id}  config={args.config}")
    print(f"  Target SNR: {args.target_snr}  range=[{args.snr_range[0]}, {args.snr_range[1]}]")
    print(f"  CGW analysis: {'enabled' if args.cgw else 'disabled'}")
    print(f"{'='*60}")

    # ── load metadata ─────────────────────────────────────────────────────────
    meta_dir      = os.path.join(out_dir, "metadata")
    config_path   = os.path.join(meta_dir, "config.json")
    if not os.path.isfile(config_path):
        sys.exit(f"ERROR: metadata/config.json not found in {meta_dir}. Run stage 1 first.")
    with open(config_path) as fh:
        run_config = json.load(fh)
    Tspan_seconds = run_config["Tspan_seconds"]

    # ── 1. Load + filter pulsars ──────────────────────────────────────────────
    print("\n📡 Loading pulsars...")
    psrs_unfiltered = load_pulsars(verbose=True)

    print("🔍 Filtering pulsars (15-year array)...")
    with suppress_enterprise_warnings():
        psrs_clean, raw_noise_params, _ = filter_pulsars_15yr(
            psrs_unfiltered, verbose=True
        )
    psr_names = [psr.name for psr in psrs_clean]
    print(f"✓ {len(psrs_clean)} pulsars loaded")

    parsed_noise_params = parse_pulsar_parameters(config.NOISEFILE)

    # ── 2. Load the population shard ─────────────────────────────────────────
    pop_dir = os.path.join(out_dir, "populations")
    store   = ShardedPickleStore(pop_dir)

    print(f"\n💾 Loading population shard subpop_{sim_id:03d}.pkl.gz ...")
    pop = store.read(sim_id)
    print(f"✓ {len(pop):,} binaries loaded")

    # ── 3. Load stage-1 TOA deltas ────────────────────────────────────────────
    print(f"\n📂 Loading stage-1 Δstoa arrays for sim_id={sim_id}...")
    delta_stoas = _load_toa_deltas(out_dir, sim_id, psr_names)

    # ── 4. Simulate noise → noise-only stoas ─────────────────────────────────
    print("\n🔊 Simulating pulsar noise...")
    for psr in psrs_clean:
        simulate_psr(psr, raw_noise_params, add_WN=True, add_RN=True)
    noise_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}
    print(f"✓ Noise stoas generated for {len(noise_stoas)} pulsars")

    # ── 5. SNR scaling loop ───────────────────────────────────────────────────
    print("\n📐 Calculating noise SNR, then scaling GW signal to target OS SNR...")
    snr_low, snr_high = args.snr_range

    cumulative_scale, pta, enterprise_psrs = _scale_and_iterate(
        psrs_clean=psrs_clean,
        delta_stoas=delta_stoas,
        noise_stoas=noise_stoas,
        target_snr=args.target_snr,
        snr_low=snr_low,
        snr_high=snr_high,
        Tspan_seconds=Tspan_seconds,
        raw_noise_params=raw_noise_params,
    )
    print(f"✓ Final cumulative amplitude scale factor: {cumulative_scale:.6f}")

    # ── 6. Update population shard: h0, D_comov, z ───────────────────────────
    print("\n💾 Updating population shard with scaled distances...")
    _update_population_scaling(store, sim_id, cumulative_scale)
    print("✓ Shard updated")

    del pop
    gc.collect()

    # ── 7. CGW SNR analysis (optional) ───────────────────────────────────────
    if args.cgw:
        print("\n🔭 Computing CGW SNRs...")
        _compute_cgw_snrs(
            sim_id=sim_id,
            store=store,
            pta=pta,
            enterprise_psrs=enterprise_psrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds,
        )
        print("✓ CGW SNR computation complete")
    else:
        print("\n(CGW analysis skipped — use --cgw to enable)")

    elapsed = time.time() - t0
    print(f"\n✅ Stage 2 sim_id={sim_id} complete in {elapsed / 60:.1f} min")
    print(f"   Output: {out_dir}/populations/subpop_{sim_id:03d}.pkl.gz")


if __name__ == "__main__":
    main()