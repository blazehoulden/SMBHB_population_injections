#!/usr/bin/env python3
"""
Stage 2 — Noise simulation, SNR scaling, and CGW analysis.

Called as a single Slurm job per sim_id after all stage-1 chunks complete.

What this job does (per simulation)
────────────────────────────────────
1.  Locate <output_dir>/sim{sim_id:03d}/.
2.  Load + filter pulsars.
3.  Sum the per-pulsar Δstoa files across ALL chunks for this sim to get
    the total GW TOA contribution from the full population.
4.  Simulate noise for each pulsar.
5.  Add combined Δstoas → noise+GW stoas.
6.  Compute optimal-statistic SNR (Enterprise).
7.  Scale GW signal iteratively until SNR converges in [snr_low, snr_high].
8.  Apply scale factor to h0/D_comov/z in every chunk shard for this sim.
9.  Optionally compute CGW SNRs across all chunks combined.

Inputs  (from stage 1, all under <output_dir>/sim{sim_id:03d}/)
────────────────────────────────────────────────────────────────
  populations/subpop_{chunk_id:03d}.pkl.gz
  stoas/chunk_{chunk_id:04d}.npz

Outputs (updated in place)
──────────────────────────
  populations/subpop_{chunk_id:03d}.pkl.gz   — updated h0/D_comov/z + cgw_snr
  metadata/stage2_complete.json              — written on success
  sim{sim_id:03d}/summary.pkl.gz             — per-sim summary object
"""

import argparse
import gc
import glob
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from consistent_pop_synth import compute_population_snr, suppress_enterprise_warnings
from data_loader import load_pulsars, filter_pulsars_15yr, parse_pulsar_parameters
from signal_injection import simulate_psr
from stage1_setup import ShardedPickleStore, _compute_analytic_proxy
from CGW_SNR import compute_cgw_snr_optimal_population_fast
from debug.test_cgw_proxy import validate_cgw_proxy, validate_proxy_filtering_ratio

N_PRE_FILTER_PER_CHUNK  = 12_500
N_GLOBAL_CGW_CANDIDATES = 12_500
N_RESCUE                = 500    # per frequency regime
N_TOP_SOURCES           = 50
MAX_SCALE_ITER          = 20


# =============================================================================
# Helpers
# =============================================================================

def _load_and_sum_toa_deltas(
    sim_out_dir: str,
    chunk_ids: List[int],
    psr_names: List[str],
) -> Dict[str, np.ndarray]:

    combined = {name: None for name in psr_names}

    for chunk_id in chunk_ids:
        fpath = os.path.join(sim_out_dir, "stoas", f"chunk_{chunk_id:04d}.npz")
        if not os.path.isfile(fpath):
            sys.exit(f"ERROR: missing delta file: {fpath}")

        with np.load(fpath) as data:
            for name in psr_names:
                delta = data[name]
                if combined[name] is None:
                    combined[name] = delta.astype(np.float64)
                else:
                    combined[name] += delta

    print(f"  Summed Δstoas across {len(chunk_ids)} chunks for {len(combined)} pulsars")
    return combined


def _cleanup_chunk_stoas(sim_out_dir: str, chunk_ids: List[int]) -> None:
    """
    Delete per-chunk Δstoa .npz files after the combined residuals have been
    saved. The combined signal is preserved under residuals/; the raw chunks
    are no longer needed.
    """
    stoa_dir = os.path.join(sim_out_dir, "stoas")
    removed, skipped, freed_bytes = 0, 0, 0

    for chunk_id in chunk_ids:
        fpath = os.path.join(stoa_dir, f"chunk_{chunk_id:04d}.npz")
        if os.path.isfile(fpath):
            freed_bytes += os.path.getsize(fpath)
            os.remove(fpath)
            removed += 1
        else:
            skipped += 1

    try:
        os.rmdir(stoa_dir)
        dir_removed = True
    except OSError:
        dir_removed = False

    freed_mb = freed_bytes / 1e6
    print(
        f"\n🗑️  Cleaned up chunk Δstoa files: "
        f"{removed} removed, {skipped} missing  "
        f"({freed_mb:.1f} MB freed)"
        + (f"  — stoas/ dir removed" if dir_removed else "")
    )


def _comov_redshift_from_scaling(
    D_comov: np.ndarray,
    z: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from SMBHB_pop_synth import _Z_GRID, _CHI_GRID

    valid      = _CHI_GRID > 0
    z_grid     = _Z_GRID[valid]
    chi_grid   = _CHI_GRID[valid]
    frac_grid  = (1.0 + z_grid) ** (2.0 / 3.0) / chi_grid

    targets = (1.0 + z) ** (2.0 / 3.0) / D_comov / scale

    if frac_grid[0] > frac_grid[-1]:
        frac_asc = frac_grid[::-1]
        chi_asc  = chi_grid[::-1]
        z_asc    = z_grid[::-1]
    else:
        frac_asc = frac_grid
        chi_asc  = chi_grid
        z_asc    = z_grid

    raw = np.searchsorted(frac_asc, targets, side="left")
    raw = np.clip(raw, 1, len(frac_asc) - 1)
    lo  = raw - 1

    pick_raw = np.abs(frac_asc[raw] - targets) <= np.abs(frac_asc[lo] - targets)
    idx      = np.where(pick_raw, raw, lo)

    return chi_asc[idx].astype(np.float32), z_asc[idx].astype(np.float32)


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
    curn_components: int = 14,
    rn_components: int = 30,
) -> Tuple[float, object, list]:

    noise_only_snr = compute_population_snr(
        psrs_clean=psrs_clean,
        population=None,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan_seconds,
        current_stoas=noise_stoas,
        return_psrs_pta=False,
        curn_components=curn_components,
        rn_components=rn_components,
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
            print(f"  [bisection] below=({best_below[0]:.4f}×, SNR={best_below[1]:.4f})  "
                  f"above=({best_above[0]:.4f}×, SNR={best_above[1]:.4f})  "
                  f"→ mid={cum_target:.4f}×  incr={incremental:.4f}")
            return incremental
        return None

    def _analytic_scale(snr_current, snr_target, snr_noise_baseline):
        sig_cur = snr_current - snr_noise_baseline
        sig_tgt = snr_target  - snr_noise_baseline
        if sig_cur <= 0:
            raise ValueError(f"Signal-only SNR non-positive ({sig_cur:.4f}).")
        if sig_tgt <= 0:
            raise ValueError(f"Target signal SNR non-positive ({sig_tgt:.4f}).")
        return np.sqrt(sig_cur / sig_tgt)

    cumulative_scale = 1.0
    snr_history      = []
    pta              = None
    enterprise_psrs  = None

    original_delta_stoas = {n: delta_stoas[n].copy() for n in delta_stoas}

    for iteration in range(max_iterations):
        scaled_delta = {n: original_delta_stoas[n] / cumulative_scale for n in original_delta_stoas}
        signal_stoas = {n: noise_stoas[n] + scaled_delta[n] for n in noise_stoas}

        snr, pta, enterprise_psrs = compute_population_snr(
            psrs_clean=psrs_clean,
            population=None,
            raw_noise_params=raw_noise_params,
            Tspan=Tspan_seconds,
            current_stoas=signal_stoas,
            curn_components=curn_components,
            rn_components=rn_components,
        )
        print(f"  Iter {iteration+1:2d}: OS SNR={snr:.4f}  "
              f"target=[{snr_low},{snr_high}]  scale={cumulative_scale:.4f}×")

        if snr_low <= snr <= snr_high:
            print(f"  ✓ Converged at iteration {iteration+1}")
            for n in delta_stoas:
                delta_stoas[n] = scaled_delta[n]
            return cumulative_scale, pta, enterprise_psrs

        snr_history.append({'cumulative_scale': cumulative_scale, 'snr': snr})

        if len(snr_history) == 1:
            factor = _analytic_scale(snr, target_snr, noise_only_snr)
            print(f"  [analytic] factor={factor:.4f}")
            cumulative_scale *= factor
        else:
            factor = _empirical_scale(snr_history, cumulative_scale)
            if factor is None:
                factor = _analytic_scale(snr, target_snr, noise_only_snr)
                print(f"  [analytic fallback] factor={factor:.4f}")
            factor = float(np.clip(factor, 0.1, 10.0))
            cumulative_scale *= factor

    raise RuntimeError(
        f"SNR scaling failed in {max_iterations} iterations. "
        f"Last SNR={snr:.4f}, target=[{snr_low},{snr_high}]."
    )


def _apply_scaling_to_all_chunks(
    store: ShardedPickleStore,
    chunk_ids: List[int],
    cumulative_scale: float,
) -> None:
    print(f"\n  Applying scale 1/{cumulative_scale:.6f} to {len(chunk_ids)} chunks...")
    for chunk_id in chunk_ids:
        pop = store.read(chunk_id)

        new_h0 = pop.h0 / cumulative_scale
        new_D_comov, new_z = _comov_redshift_from_scaling(
            pop.D_comov, pop.z, 1.0 / cumulative_scale
        )

        store.update(chunk_id, h0=new_h0, D_comov=new_D_comov, z=new_z)

        del pop, new_h0, new_D_comov, new_z
        gc.collect()

        print(f"    chunk {chunk_id:03d}: updated", flush=True)


def _compute_cgw_snrs(
    store: ShardedPickleStore,
    chunk_ids: List[int],
    pta,
    enterprise_psrs,
    raw_noise_params,
    parsed_noise_params,
    Tspan_seconds: float,
) -> None:
    """
    Two-pass CGW SNR computation across all chunks.
    Pass 1: pre-filter each chunk using analytic s^T N^{-1} s proxy.
            Rescued sources from proxy-failure frequency regimes are added
            unconditionally.
    Pass 2: evaluate full CGW SNR on top N_GLOBAL_CGW_CANDIDATES globally.
    """
    print(f"\n  CGW: pre-filtering {len(chunk_ids)} chunks "
          f"(top {N_PRE_FILTER_PER_CHUNK} each by analytic proxy)...")

    candidate_list = []  # (chunk_id, local_idx, proxy)

    # ── Pass 1: analytic proxy across all chunks ──────────────────────────────
    for chunk_id in chunk_ids:
        pop    = store.read(chunk_id)
        n      = len(pop)
        n_keep = min(N_PRE_FILTER_PER_CHUNK, n)

        # Use Stage 1 proxy scores if available, else compute fresh
        if hasattr(pop, 'cgw_proxy') and pop.cgw_proxy is not None:
            proxies = pop.cgw_proxy.astype(np.float64)
        else:
            proxies = _compute_analytic_proxy(pop, enterprise_psrs,
                                              Tspan_seconds=Tspan_seconds)

        if n_keep == n:
            top_local = np.argsort(proxies)[::-1]
        else:
            top_local = np.argpartition(proxies, -n_keep)[-n_keep:]
            top_local = top_local[np.argsort(proxies[top_local])[::-1]]

        for local_idx in top_local:
            candidate_list.append((chunk_id, int(local_idx), float(proxies[local_idx])))

        del pop
        gc.collect()

    candidate_list.sort(key=lambda x: x[2], reverse=True)
    global_candidates = candidate_list[:N_GLOBAL_CGW_CANDIDATES]
    print(f"  {len(candidate_list)} candidates merged → top {len(global_candidates)} for full CGW SNR")

    # ── Rescue proxy failure regimes ──────────────────────────────────────────
    # Force top-h0 sources from low-f and mid-f regimes into the evaluation
    # set regardless of proxy rank. These regimes are where the proxy fails:
    #   - f*T < 10:  timing model absorbs signal, proxy overestimates
    #   - 10 < f*T < 50: red noise Sigma correction large, proxy underestimates
    existing = {(c, i) for c, i, _ in global_candidates}

    for chunk_id in chunk_ids:
        pop = store.read(chunk_id)
        f_T = np.asarray(pop.f, dtype=np.float64) * Tspan_seconds

        for mask in [
            f_T < 10.0,                            # low-f regime
            (f_T >= 10.0) & (f_T < 50.0),         # mid-f regime
        ]:
            if mask.sum() == 0:
                continue
            regime_idx   = np.where(mask)[0]
            top_h0_local = regime_idx[np.argsort(pop.h0[regime_idx])[-N_RESCUE:]]
            for local_idx in top_h0_local:
                key = (chunk_id, int(local_idx))
                if key not in existing:
                    global_candidates.append(
                        (chunk_id, int(local_idx), float(pop.h0[local_idx]))
                    )
                    existing.add(key)

        del pop
        gc.collect()

    n_rescued = len(global_candidates) - N_GLOBAL_CGW_CANDIDATES
    print(f"  After regime rescue: {len(global_candidates)} candidates total "
          f"(+{n_rescued} rescued from low/mid-f regimes)")

    # ── Assemble top binaries (grouped by chunk to minimise re-loads) ─────────
    by_chunk: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for global_rank, (chunk_id, local_idx, _) in enumerate(global_candidates):
        by_chunk[chunk_id].append((global_rank, local_idx))

    top_binaries = [None] * len(global_candidates)
    for chunk_id, entries in by_chunk.items():
        pop = store.read(chunk_id)
        for global_rank, local_idx in entries:
            top_binaries[global_rank] = pop[local_idx]
        del pop
        gc.collect()

    # ── Pass 2: full SNR evaluation ───────────────────────────────────────────
    print(f"  Computing full CGW SNR for {len(top_binaries)} candidates...")
    top_snrs, top_breakdowns = compute_cgw_snr_optimal_population_fast(
        psrs=enterprise_psrs,
        pta=pta,
        population=top_binaries,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan=Tspan_seconds,
        profile=True,
        return_breakdown=True,
    )

    # ── Write cgw_snr back to store ───────────────────────────────────────────
    chunk_snr_maps: Dict[int, Dict[int, float]] = defaultdict(dict)
    for global_rank, (chunk_id, local_idx, _) in enumerate(global_candidates):
        chunk_snr_maps[chunk_id][local_idx] = float(top_snrs[global_rank])

    for chunk_id in chunk_ids:
        pop         = store.read(chunk_id)
        cgw_snr_arr = np.zeros(len(pop), dtype=np.float32)
        for local_idx, snr_val in chunk_snr_maps.get(chunk_id, {}).items():
            cgw_snr_arr[local_idx] = snr_val
        store.update(chunk_id, cgw_snr=cgw_snr_arr)
        del pop
        gc.collect()

    # ── Persist top-5 breakdowns for summary builder ──────────────────────────
    proxy_rank_map = {
        (chunk_id, local_idx): proxy_rank
        for proxy_rank, (chunk_id, local_idx, _) in enumerate(global_candidates, start=1)
    }

    ranked  = sorted(zip(global_candidates, top_snrs), key=lambda x: x[1], reverse=True)
    n_show  = min(N_TOP_SOURCES, len(ranked))

    candidate_breakdown_map = {
        (chunk_id, local_idx): top_breakdowns[i]
        for i, (chunk_id, local_idx, _) in enumerate(global_candidates)
    }

    top_breakdown_records = []
    for rank, ((chunk_id, local_idx, proxy), snr_val) in enumerate(ranked[:5], start=1):
        proxy_rank = proxy_rank_map[(chunk_id, local_idx)]
        per_pulsar = candidate_breakdown_map[(chunk_id, local_idx)]
        top_breakdown_records.append({
            "rank":              rank,
            "proxy_rank":        proxy_rank,
            "chunk_id":          int(chunk_id),
            "local_idx":         int(local_idx),
            "proxy":             float(proxy),
            "cgw_snr":           float(snr_val),
            "per_pulsar_rho_sq": {k: float(v) for k, v in per_pulsar.items()},
        })

    metadata_dir = os.path.join(os.path.dirname(str(store.dir)), "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, "top_cgw_breakdowns.json"), "w") as fh:
        json.dump(top_breakdown_records, fh, indent=2)

    # ── Print top sources ─────────────────────────────────────────────────────
    chunk_cache = {}
    for (chunk_id, local_idx, _), _ in ranked[:n_show]:
        if chunk_id not in chunk_cache:
            chunk_cache[chunk_id] = store.read(chunk_id)

    print(f"\n  Top {n_show} CGW candidates:")
    for rank, ((chunk_id, local_idx, proxy), snr_val) in enumerate(ranked[:n_show], start=1):
        proxy_rank = proxy_rank_map[(chunk_id, local_idx)]
        pop = chunk_cache[chunk_id]
        print(
            f"    {rank:2d}.  chunk={chunk_id:03d}  local_idx={local_idx:7d}  "
            f"f={pop.f[local_idx]:.2e} Hz  "
            f"Mc={pop.Mc[local_idx]:.2e} Msun  "
            f"h0={pop.h0[local_idx]:.2e}  "
            f"proxy={proxy:.4e} (rank #{proxy_rank})  "
            f"CGW_SNR={snr_val:.4f}"
        )
    del chunk_cache
    gc.collect()


# =============================================================================
# Per-simulation processing
# =============================================================================

def _save_toa_residuals(
    sim_out_dir: str,
    psrs_clean,
    noise_stoas: Dict[str, np.ndarray],
    combined_delta_stoas: Dict[str, np.ndarray],
    cumulative_scale: float,
) -> None:
    base = os.path.join(sim_out_dir, "residuals")
    dirs = {
        "noise":      os.path.join(base, "noise"),
        "population": os.path.join(base, "population"),
        "combined":   os.path.join(base, "combined"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"\n💾 Saving TOA residuals to {base}/")
    for psr in psrs_clean:
        name = psr.name
        noise_arr = noise_stoas[name]
        pop_arr   = combined_delta_stoas[name] / cumulative_scale
        comb_arr  = noise_arr + pop_arr

        np.save(os.path.join(dirs["noise"],      f"{name}.npy"), noise_arr.astype(np.float64))
        np.save(os.path.join(dirs["population"], f"{name}.npy"), pop_arr.astype(np.float64))
        np.save(os.path.join(dirs["combined"],   f"{name}.npy"), comb_arr.astype(np.float64))

    print(f"  ✓ Saved residuals for {len(psrs_clean)} pulsars "
          f"(noise / population / combined)")

    manifest = {
        "cumulative_scale": cumulative_scale,
        "n_pulsars":        len(psrs_clean),
        "psr_names":        [psr.name for psr in psrs_clean],
        "description": {
            "noise":      "Simulated noise residuals only (no GW signal)",
            "population": "GW TOA contribution from full population, scaled to target SNR",
            "combined":   "noise + population (what an observer would measure)",
        },
    }
    with open(os.path.join(base, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


def process_sim(sim_id: int, args, psrs_clean, raw_noise_params, parsed_noise_params) -> bool:
    """
    Run the full stage-2 pipeline for a single sim_id.
    Returns True on success, False on failure.
    """
    t_sim = time.time()
    sim_out_dir = os.path.join(args.output_dir, f"sim{sim_id:03d}")

    print(f"\n{'='*60}")
    print(f"Stage 2 — sim_id={sim_id}")
    print(f"  Dir        : {sim_out_dir}")
    print(f"  Target SNR : {args.target_snr}  range={args.snr_range}")
    print(f"  CGW        : {'enabled' if args.cgw else 'disabled'}")
    print(f"{'='*60}")

    # ── load metadata ─────────────────────────────────────────────────────────
    config_path = os.path.join(sim_out_dir, "metadata", "config.json")
    if not os.path.isfile(config_path):
        print(f"ERROR: config.json not found at {config_path}", file=sys.stderr)
        return False
    with open(config_path) as fh:
        run_config = json.load(fh)
    Tspan_seconds = run_config["Tspan_seconds"]

    # ── discover chunks ───────────────────────────────────────────────────────
    pop_dir   = os.path.join(sim_out_dir, "populations")
    store     = ShardedPickleStore(pop_dir)
    chunk_ids = store.available()

    if not chunk_ids:
        print(f"ERROR: no population shards found in {pop_dir}", file=sys.stderr)
        return False
    print(f"\n📦 Found {len(chunk_ids)} chunks: {chunk_ids}")

    # ── proxy-only mode ───────────────────────────────────────────────────────
    if args.proxy_only:
        print(f"\n{'='*60}")
        print(f"Proxy-only mode — sim_id={sim_id}")
        print(f"  Skipping noise simulation and SNR scaling.")
        print(f"{'='*60}")

        print("\n🔊 Simulating noise (for PTA structure only)...")
        for i, psr in enumerate(psrs_clean):
            print(f"  [{i+1}/{len(psrs_clean)}] {psr.name}...", end=" ", flush=True)
            simulate_psr(psr, raw_noise_params, add_WN=True, add_RN=True)
            print("done", flush=True)
        noise_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}

        _, pta, enterprise_psrs = compute_population_snr(
            psrs_clean=psrs_clean,
            population=None,
            raw_noise_params=raw_noise_params,
            Tspan=Tspan_seconds,
            current_stoas=noise_stoas,
            return_psrs_pta=True,
        )

        validate_cgw_proxy(
            store=store,
            chunk_ids=chunk_ids,
            pta=pta,
            enterprise_psrs=enterprise_psrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds,
            n_test=args.n_test,
        )

        validate_proxy_filtering_ratio(
            store=store,
            chunk_ids=chunk_ids,
            pta=pta,
            enterprise_psrs=enterprise_psrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds,
            n_keep=N_PRE_FILTER_PER_CHUNK,
        )

        sentinel = os.path.join(sim_out_dir, "metadata", "stage2_complete.json")
        with open(sentinel, "w") as fh:
            json.dump({
                "sim_id":       sim_id,
                "proxy_only":   True,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, fh, indent=2)
        return True

    # ── 1. Sum Δstoas across all chunks ───────────────────────────────────────
    psr_names = [psr.name for psr in psrs_clean]
    print(f"\n📂 Summing Δstoas across {len(chunk_ids)} chunks...")
    combined_delta_stoas = _load_and_sum_toa_deltas(sim_out_dir, chunk_ids, psr_names)

    # ── 2. Simulate noise ─────────────────────────────────────────────────────
    print("\n🔊 Simulating pulsar noise...")
    for i, psr in enumerate(psrs_clean):
        print(f"  [{i+1}/{len(psrs_clean)}] simulating {psr.name}...", end=" ", flush=True)
        simulate_psr(psr, raw_noise_params, add_WN=True, add_RN=True)
        print("done", flush=True)
    noise_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}
    print(f"✓ Noise stoas for {len(noise_stoas)} pulsars")

    # ── 3. SNR scaling loop ───────────────────────────────────────────────────
    print("\n📐 Scaling combined GW signal to target OS SNR...")
    snr_low, snr_high = args.snr_range

    try:
        cumulative_scale, pta, enterprise_psrs = _scale_and_iterate(
            psrs_clean=psrs_clean,
            delta_stoas=combined_delta_stoas,
            noise_stoas=noise_stoas,
            target_snr=args.target_snr,
            snr_low=snr_low,
            snr_high=snr_high,
            Tspan_seconds=Tspan_seconds,
            raw_noise_params=raw_noise_params,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False

    print(f"✓ Converged — scale factor: {cumulative_scale:.6f}")

    # ── 4. Save TOA residuals ─────────────────────────────────────────────────
    _save_toa_residuals(
        sim_out_dir=sim_out_dir,
        psrs_clean=psrs_clean,
        noise_stoas=noise_stoas,
        combined_delta_stoas=combined_delta_stoas,
        cumulative_scale=cumulative_scale,
    )

    # ── 5. Clean up per-chunk stoa files ─────────────────────────────────────
    _cleanup_chunk_stoas(sim_out_dir, chunk_ids)

    del combined_delta_stoas, noise_stoas
    gc.collect()

    # ── 6. Apply scale factor to all chunks ───────────────────────────────────
    print("\n💾 Updating all shards...")
    _apply_scaling_to_all_chunks(store, chunk_ids, cumulative_scale)

    # ── 7. CGW SNR analysis ───────────────────────────────────────────────────
    if args.cgw:
        print("\n🔭 Computing CGW SNRs...")

        if args.validate_proxy:
            validate_cgw_proxy(
                store=store,
                chunk_ids=chunk_ids,
                pta=pta,
                enterprise_psrs=enterprise_psrs,
                raw_noise_params=raw_noise_params,
                parsed_noise_params=parsed_noise_params,
                Tspan_seconds=Tspan_seconds,
                n_test=args.n_test,
            )

        _compute_cgw_snrs(
            store=store,
            chunk_ids=chunk_ids,
            pta=pta,
            enterprise_psrs=enterprise_psrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds,
        )
        print("✓ CGW complete")
    else:
        print("\n(CGW skipped — use --cgw to enable)")

    # ── 8. Write completion sentinel ──────────────────────────────────────────
    sentinel = os.path.join(sim_out_dir, "metadata", "stage2_complete.json")
    with open(sentinel, "w") as fh:
        json.dump({
            "sim_id":           sim_id,
            "cumulative_scale": float(cumulative_scale),
            "completed_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, fh, indent=2)

    elapsed = time.time() - t_sim
    print(f"\n✅ Stage 2 sim_id={sim_id} complete in {elapsed / 60:.1f} min")
    print(f"   Output: {sim_out_dir}/populations/")
    return True


# =============================================================================
# Per-simulation summary object
# =============================================================================

def _build_summary_object(
    output_dir: str,
    sim_id: int,
    n_keep_per_category: int = 200,
) -> None:
    import gzip
    import pickle
    import heapq

    sim_out_dir = os.path.join(output_dir, f"sim{sim_id:03d}")
    out_path    = os.path.join(sim_out_dir, "summary.pkl.gz")

    pop_dir   = os.path.join(sim_out_dir, "populations")
    store     = ShardedPickleStore(pop_dir)
    chunk_ids = store.available()

    print(f"\n{'='*60}")
    print(f"Building summary object for sim{sim_id:03d} → {out_path}")
    print(f"  n_keep_per_category = {n_keep_per_category}")
    print(f"{'='*60}")

    categories = {
        "cgw_snr_high": ([], True,  "cgw_snr"),
        "D_comov_near": ([], False, "D_comov"),
        "D_comov_far":  ([], True,  "D_comov"),
        "f_low":        ([], False, "f"),
        "f_high":       ([], True,  "f"),
        "Mc_low":       ([], False, "Mc"),
        "Mc_high":      ([], True,  "Mc"),
        "Mtot_low":     ([], False, "Mtot"),
        "Mtot_high":    ([], True,  "Mtot"),
        "h0_high":      ([], True,  "h0"),
        "z_low":        ([], False, "z"),
        "z_high":       ([], True,  "z"),
    }

    field_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
    field_names:   List[str] = []
    total_scanned = 0

    def _heap_push(heap, is_max, key, gidx, cap):
        entry = (-key if is_max else key, gidx)
        if len(heap) < cap:
            heapq.heappush(heap, entry)
        elif entry < heap[0]:
            heapq.heapreplace(heap, entry)

    for chunk_id in chunk_ids:
        pop     = store.read(chunk_id)
        n       = len(pop)
        has_cgw = hasattr(pop, "cgw_snr") and pop.cgw_snr is not None

        if not field_names:
            field_names = [
                name for name, value in vars(pop).items()
                if isinstance(value, np.ndarray)
            ]

        for field_name in field_names:
            field_buffers[field_name].append(np.asarray(getattr(pop, field_name)))

        for local_i in range(n):
            gidx = total_scanned + local_i
            vals = {
                "cgw_snr": float(pop.cgw_snr[local_i]) if has_cgw else -1.0,
                "D_comov": float(pop.D_comov[local_i]),
                "f":       float(pop.f[local_i]),
                "Mc":      float(pop.Mc[local_i]),
                "Mtot":    float(pop.Mtot[local_i]),
                "h0":      float(pop.h0[local_i]),
                "z":       float(pop.z[local_i]),
            }

            qualifies = False
            for cat_name, (heap, is_max, field) in categories.items():
                key   = vals[field]
                entry = (-key if is_max else key, gidx)
                if len(heap) < n_keep_per_category or entry < heap[0]:
                    qualifies = True
                    break

            if qualifies:
                for cat_name, (heap, is_max, field) in categories.items():
                    _heap_push(heap, is_max, vals[field], gidx, n_keep_per_category)

        total_scanned += n
        del pop
        gc.collect()

    category_indices: Dict[str, List[int]] = {}
    for cat_name, (heap, is_max, field) in categories.items():
        category_indices[cat_name] = sorted({gidx for (_, gidx) in heap})

    print(f"  Total binaries scanned: {total_scanned:,}")
    for cat_name, idxs in category_indices.items():
        print(f"    {cat_name:<20s}: {len(idxs):,}")

    arrays: Dict[str, np.ndarray] = {
        field_name: np.concatenate(buffers)
        for field_name, buffers in field_buffers.items()
        if buffers
    }
    arrays["global_idx"] = np.arange(total_scanned, dtype=np.int64)

    top_cgw_breakdowns = []
    breakdown_path = os.path.join(sim_out_dir, "metadata", "top_cgw_breakdowns.json")
    if os.path.isfile(breakdown_path):
        with open(breakdown_path) as fh:
            top_cgw_breakdowns = json.load(fh)

    summary_meta = {
        "category_indices":   {cat: idxs for cat, idxs in category_indices.items()},
        "sim_id":             sim_id,
        "n_keep_per_category": n_keep_per_category,
        "total_scanned":      total_scanned,
        "summary_version":    2,
        "full_population":    True,
        "top_cgw_breakdowns": top_cgw_breakdowns,
    }

    payload = {"arrays": arrays, "meta": summary_meta}
    with gzip.open(out_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"✅ Summary written: {out_path}  ({size_mb:.1f} MB)")
    print(f"   Load with:")
    print(f"     import gzip, pickle")
    print(f"     with gzip.open('{out_path}', 'rb') as f:")
    print(f"         d = pickle.load(f)")
    print(f"     arrays = d['arrays']   # dict of np.ndarray, all same length")
    print(f"     meta   = d['meta']     # category_indices, top_cgw_breakdowns, etc.")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: SNR scaling + CGW (one job per sim)")
    p.add_argument("--output-dir",     type=str, required=True)
    p.add_argument("--config", "-c",   default="optimistic",
                   choices=list(config.POPULATION_CONFIGS.keys()))
    p.add_argument("--target-snr",     type=float, default=4.0)
    p.add_argument("--snr-range",      nargs=2, type=float, default=[3.5, 4.25])
    p.add_argument("--n-chunks",       type=int, required=True,
                   help="Number of chunks per simulation (used for logging only)")
    p.add_argument("--sim-id",         type=int, default=None,
                   help="Simulation ID to process (required)")
    p.add_argument("--cgw",            action="store_true")
    p.add_argument("--validate-proxy", action="store_true",
                   help="Validate CGW proxy vs true SNR before full computation")
    p.add_argument("--proxy-only",     action="store_true",
                   help="Skip noise/scaling — just validate the CGW proxy")
    p.add_argument("--n-test",         type=int, default=1_000,
                   help="Number of binaries to sample for proxy validation (default: 1000)")
    p.add_argument("--clean-failed",   action="store_true",
                   help="Remove incomplete sim directories so stage1 can regenerate cleanly")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t0   = time.time()

    # ── clean-failed mode ─────────────────────────────────────────────────────
    if args.clean_failed:
        import shutil
        print("\n🧹 Cleaning incomplete sim directories...")
        for sim_dir in sorted(glob.glob(os.path.join(args.output_dir, "sim[0-9][0-9][0-9]"))):
            sentinel = os.path.join(sim_dir, "metadata", "stage2_complete.json")
            if not os.path.isfile(sentinel):
                print(f"  Removing incomplete {os.path.basename(sim_dir)}...")
                shutil.rmtree(sim_dir)
            else:
                print(f"  Preserving complete {os.path.basename(sim_dir)}")
        print("✓ Clean done")
        sys.exit(0)

    # ── Guard sim_id ──────────────────────────────────────────────────────────
    if args.sim_id is None:
        print("ERROR: --sim-id is required for stage 2", file=sys.stderr)
        sys.exit(1)
    sim_id = args.sim_id

    print(f"\n{'='*60}")
    print(f"Stage 2 — sim_id={sim_id}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Config     : {args.config}")
    print(f"  Target SNR : {args.target_snr}  range={args.snr_range}")
    print(f"  Chunks/sim : {args.n_chunks}")
    print(f"  CGW        : {'enabled' if args.cgw else 'disabled'}")
    print(f"{'='*60}\n")

    # ── Load pulsars ──────────────────────────────────────────────────────────
    print("📡 Loading pulsars...")
    psrs_unfiltered = load_pulsars(verbose=True)
    with suppress_enterprise_warnings():
        psrs_clean, raw_noise_params, _ = filter_pulsars_15yr(psrs_unfiltered, verbose=True)
    print(f"✓ {len(psrs_clean)} pulsars loaded\n")

    parsed_noise_params = parse_pulsar_parameters(config.NOISEFILE)

    # ── Process sim ───────────────────────────────────────────────────────────
    success = process_sim(sim_id, args, psrs_clean, raw_noise_params, parsed_noise_params)

    if not success:
        print(f"ERROR: sim_id={sim_id} failed", file=sys.stderr)
        sys.exit(1)

    # ── Build per-sim summary ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Stage 2 complete — sim_id={sim_id} in {elapsed/60:.1f} min")

    _build_summary_object(args.output_dir, sim_id=sim_id, n_keep_per_category=200)

    print(f"{'='*60}")


if __name__ == "__main__":
    main()