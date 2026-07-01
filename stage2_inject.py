#!/usr/bin/env python3
"""
Stage 2 — Noise simulation, SNR scaling (via chunk selection), and CGW analysis.

Key change from previous version
─────────────────────────────────
The old design rescaled GW signal amplitudes (h0, D_comov, z) to hit the
target OS SNR.  This is physically incorrect — it changes the source population
rather than modelling a real observational outcome.

The new design exploits the fact that the OS SNR scales as √N_binaries:

  SNR(k sub-chunks) ≈ SNR(N_total sub-chunks) × √(k / N_total)

We find the integer k ∈ [1, N_total_sub_chunks] such that including exactly k
sub-chunks gives an OS SNR inside [snr_low, snr_high].  All source parameters
(h0, D_comov, z) remain at their physical values — nothing is rescaled.

The same k is used for all synthetic PTA scenarios so that every scenario sees
the same injected GW population (physically consistent).

Subprocess architecture
───────────────────────
Each phase (baseline + one per synthetic scenario) runs in its own subprocess
so that libstempo/tempo2 C-level global state is fully cleared between phases.

Phase handoff (metadata/phase_handoff.json)
───────────────────────────────────────────
  n_active_sub_chunks   int    — number of sub-chunks included (the "k")
  active_shard_ids      list   — list of [chunk_id, sub_id] pairs actually used
  Tspan_seconds         float
  noise_seed            int
  baseline_candidates   list   — [[chunk_id, sub_id, local_idx, proxy], ...]
  baseline_snrs         list   — floats

Output layout (per sim)
───────────────────────
  populations/     subpop_{chunk:03d}_{sub:03d}.pkl.gz  (all shards, unchanged)
  stoas/           chunk_{chunk:04d}_{sub:04d}[_<scenario>].npz  (deleted after use)
  residuals/       noise/ population/ combined/    (baseline, active sub-chunks only)
  residuals_<scen>/                                (synthetic scenarios)
  metadata/        config.json  phase_handoff.json  stage2_complete.json
"""

import argparse
import gc
import glob
import gzip
import heapq
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from consistent_pop_synth import compute_population_snr, suppress_enterprise_warnings
from data_loader import (
    load_pulsars, filter_pulsars_15yr, parse_pulsar_parameters,
    SCENARIOS as DEFAULT_SCENARIOS,
)
from signal_injection import simulate_psr
from CGW_SNR import compute_cgw_snr_optimal_population_fast
from debug.test_cgw_proxy import validate_cgw_proxy, validate_proxy_filtering_ratio

try:
    from stage1_setup import (
        ShardedPickleStore,
        _compute_analytic_proxy,
        _filter_population_extremes,
        FIELD_DTYPES,
        SCALAR_FIELDS,
        N_PRE_FILTER_PER_CHUNK,
    )
    _STAGE1_SETUP_IMPORTED = True
except ImportError:
    _STAGE1_SETUP_IMPORTED = False

from enterprise.pulsar import Pulsar as EnterprisePulsar
from pta_builder import build_pta_and_params

N_PRE_FILTER_PER_CHUNK    = 12_500
N_GLOBAL_CGW_CANDIDATES   = 12_500
N_SCENARIO_CGW_CANDIDATES = 1_000
N_RESCUE                  = 500
N_TOP_SOURCES             = 50
MAX_SCALE_ITER            = 20       # max bisection steps for chunk-count search

HANDOFF_FILENAME = 'phase_handoff.json'


# =============================================================================
# ShardedPickleStore  (two-index: chunk × sub-chunk)
# self-contained fallback if stage1_setup not importable
# =============================================================================

if not _STAGE1_SETUP_IMPORTED:
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
        One pkl.gz per (chunk_id, sub_id) pair.

        File pattern:  subpop_{chunk_id:03d}_{sub_id:03d}.pkl.gz
        available() → sorted list of (chunk_id, sub_id) int tuples.
        """

        def __init__(self, directory: str, compress_level: int = 6):
            self.dir = Path(directory)
            self.dir.mkdir(parents=True, exist_ok=True)
            self.compress_level = compress_level

        def write(self, chunk_id: int, sub_id: int, pop) -> None:
            self._dump(self._path(chunk_id, sub_id), self._downcast(pop))

        def read(self, chunk_id: int, sub_id: int):
            with gzip.open(self._path(chunk_id, sub_id), "rb") as f:
                return pickle.load(f)

        def update(self, chunk_id: int, sub_id: int,
                   h0=None, D_comov=None, z=None,
                   cgw_snr=None, cgw_proxy=None,
                   amp_A=None, amp_B=None) -> None:
            pop = self.read(chunk_id, sub_id)
            if h0        is not None: pop.h0        = h0.astype(np.float32)
            if D_comov   is not None: pop.D_comov   = D_comov.astype(np.float32)
            if z         is not None: pop.z         = z.astype(np.float32)
            if cgw_snr   is not None: pop.cgw_snr   = cgw_snr.astype(np.float32)
            if cgw_proxy is not None: pop.cgw_proxy = cgw_proxy.astype(np.float32)
            if amp_A is not None:
                for psr, arr in amp_A.items():
                    pop.amp_A[psr] = arr.astype(np.float32)
            if amp_B is not None:
                for psr, arr in amp_B.items():
                    pop.amp_B[psr] = arr.astype(np.float32)
            self._dump(self._path(chunk_id, sub_id), pop)

        def available(self):
            results = []
            pattern = re.compile(r"subpop_(\d{3})_(\d{3})\.pkl\.gz$")

            for p in self.dir.glob("subpop_*.pkl.gz"):
                m = pattern.match(p.name)
                if m:
                    results.append((int(m.group(1)), int(m.group(2))))

            return sorted(results)

        def _path(self, chunk_id: int, sub_id: int) -> Path:
            return self.dir / f"subpop_{chunk_id:03d}_{sub_id:03d}.pkl.gz"

        def _dump(self, path, obj) -> None:
            with gzip.open(path, "wb", compresslevel=self.compress_level) as f:
                pickle.dump(obj, f, protocol=5)

        @staticmethod
        def _downcast(pop):
            for name, dtype in FIELD_DTYPES.items():
                if hasattr(pop, name):
                    setattr(pop, name, getattr(pop, name).astype(dtype))
            for psr in list(pop.amp_A):
                pop.amp_A[psr] = pop.amp_A[psr].astype(np.float32)
            for psr in list(pop.amp_B):
                pop.amp_B[psr] = pop.amp_B[psr].astype(np.float32)
            return pop


# =============================================================================
# TOA delta loading  (two-index naming: chunk × sub)
# =============================================================================

def _delta_filename(chunk_id: int, sub_id: int,
                    scenario: Optional[str] = None) -> str:
    """chunk_{chunk:04d}_{sub:04d}[_<scenario>].npz"""
    suffix = f'_{scenario}' if scenario else ''
    return f'chunk_{chunk_id:04d}_{sub_id:04d}{suffix}.npz'


def _load_and_sum_toa_deltas(
    sim_out_dir:  str,
    shard_ids:    List[Tuple[int, int]],   # (chunk_id, sub_id) pairs to include
    psr_names:    List[str],
    scenario:     Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Sum per-shard Δstoa .npz files for a given scenario across all active shards.
    shard_ids is the ACTIVE subset (the k chosen by chunk-count selection).
    """
    combined: Dict[str, Optional[np.ndarray]] = {n: None for n in psr_names}

    for chunk_id, sub_id in shard_ids:
        fname = _delta_filename(chunk_id, sub_id, scenario)
        fpath = os.path.join(sim_out_dir, 'stoas', fname)
        if not os.path.isfile(fpath):
            sys.exit(f'ERROR: missing delta file: {fpath}')
        with np.load(fpath) as data:
            for name in psr_names:
                if name not in data:
                    sys.exit(f'ERROR: pulsar {name} missing from {fpath}')
                arr = data[name].astype(np.float64)
                combined[name] = arr if combined[name] is None else combined[name] + arr

    print(f'  Summed Δstoas across {len(shard_ids)} shards '
          f'[scenario={scenario or "baseline"}] for {len(combined)} pulsars')
    return combined  # type: ignore[return-value]


def _cleanup_shard_stoas(
    sim_out_dir: str,
    shard_ids:   List[Tuple[int, int]],
    scenario:    Optional[str] = None,
) -> None:
    """Delete .npz files for the given shard_ids and scenario."""
    stoa_dir = os.path.join(sim_out_dir, 'stoas')
    removed, freed_bytes = 0, 0
    for chunk_id, sub_id in shard_ids:
        fpath = os.path.join(stoa_dir, _delta_filename(chunk_id, sub_id, scenario))
        if os.path.isfile(fpath):
            freed_bytes += os.path.getsize(fpath)
            os.remove(fpath)
            removed += 1
    try:
        os.rmdir(stoa_dir)
        dir_note = '  — stoas/ dir removed'
    except OSError:
        dir_note = ''
    print(f'  🗑️  Cleaned {removed} shard files '
          f'[scenario={scenario or "baseline"}]  '
          f'({freed_bytes / 1e6:.1f} MB freed){dir_note}')

def _cleanup_inactive_population_shards(
    store:         ShardedPickleStore,
    all_shards:    List[Tuple[int, int]],
    active_shards: List[Tuple[int, int]],
) -> None:
    """Delete population shard pkl.gz files that are not in active_shards."""
    active_set = set(active_shards)
    inactive   = [s for s in all_shards if s not in active_set]
    removed, freed_bytes = 0, 0
    for chunk_id, sub_id in inactive:
        path = store._path(chunk_id, sub_id)
        if path.is_file():
            freed_bytes += path.stat().st_size
            path.unlink()
            removed += 1
    print(f'  🗑️  Removed {removed} inactive population shards '
          f'({freed_bytes / 1e6:.1f} MB freed)  '
          f'({len(active_set)} active shards retained)')

# =============================================================================
# Noise simulation (seeded)
# =============================================================================

def _simulate_noise(
    psrs_clean,
    raw_noise_params,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Simulate white + red noise for a list of pulsars with a fixed numpy seed.
    Returns dict: psr_name -> stoas_after_noise (days).
    Using the same seed across scenarios ensures noise realisations are
    comparable — only PTA sensitivity differs.
    """
    for i, psr in enumerate(psrs_clean):
        psr_seed = seed + i * 13579 if seed is not None else None
        simulate_psr(psr, raw_noise_params, add_WN=True, add_RN=True, seed=psr_seed)
    return {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}


# =============================================================================
# Enterprise PTA builder
# =============================================================================

def _build_pta(psrs_libstempo, raw_noise_params, Tspan):
    enterprise_psrs = [
        EnterprisePulsar(psr, ephem='DE440', backend='tempo2')
        for psr in psrs_libstempo
    ]
    pta, _, _ = build_pta_and_params(
        psrs=enterprise_psrs,
        noise_params=raw_noise_params,
        Tspan=Tspan,
    )
    gc.collect()
    return pta, enterprise_psrs


# =============================================================================
# Chunk-count SNR selection
# =============================================================================
#
# Physical motivation
# ───────────────────
# The GW background from N_total independent binaries produces an OS signal
# that grows as √N (central limit theorem in the quadratic OS estimator).
# Therefore:
#
#   SNR(k) ≈ SNR(N_total) × √(k / N_total)
#
# We want SNR(k) ∈ [snr_low, snr_high].  The analytic estimate gives a first
# guess; we then verify with actual OS evaluations and bisect if needed.
#
# Source amplitudes h0, D_comov, z are NEVER modified — the population stays
# at its physical values.  Only the number of included sub-chunks changes.
#
# The sub-chunks to include are selected in a fixed order (shard_ids[:k]) so
# the selection is deterministic and reproducible.
def _snr_for_shard_subset(
    shard_ids:       List[Tuple[int, int]],
    sim_out_dir:     str,
    psr_names:       List[str],
    noise_stoas:     Dict[str, np.ndarray],
    psrs_clean,
    raw_noise_params,
    Tspan_seconds:   float,
    curn_components: int = 120,
    rn_components:   int  = 120,
) -> float:
    """Evaluate OS SNR for an arbitrary subset of shards (helper)."""
    combined_delta = _load_and_sum_toa_deltas(
        sim_out_dir, shard_ids, psr_names, scenario=None)
    signal_stoas = {n: noise_stoas[n] + combined_delta[n] for n in noise_stoas}
    snr, _, _ = compute_population_snr(
        psrs_clean=psrs_clean,
        population=None,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan_seconds,
        current_stoas=signal_stoas,
        curn_components=curn_components,
        rn_components=rn_components,
    )
    return float(snr)


def _select_active_shards(
    all_shard_ids:   List[Tuple[int, int]],
    psrs_clean,
    raw_noise_params,
    sim_out_dir:     str,
    psr_names:       List[str],
    noise_stoas:     Dict[str, np.ndarray],
    target_snr:      float,
    snr_low:         float,
    snr_high:        float,
    Tspan_seconds:   float,
    max_iterations:  int = MAX_SCALE_ITER,
    curn_components: int = 120,
    rn_components:   int = 120,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Find a subset of all_shard_ids giving OS SNR ∈ [snr_low, snr_high].

    Primary strategy: bisect on the prefix all_shard_ids[:k].
    Fallback strategy: when the prefix-bisection bracket collapses without
    a valid k (happens when individual chunks contain unusually loud sources
    that break the √k scaling assumption), we:

      1. Find the largest valid prefix k* where SNR < snr_low  (the "floor").
      2. Scan every remaining chunk individually and evaluate SNR(floor_shards
         + [candidate_chunk]).
      3. Accept the first candidate that lands in [snr_low, snr_high].
         If multiple would overshoot, take the one that overshoots least
         (closest to target from above) rather than giving up.

    This way we never fail just because the population happens to be lumpy.
    """
    N_total = len(all_shard_ids)
    if N_total == 0:
        raise ValueError('No shard IDs provided to _select_active_shards.')

    def _snr_for_k(k: int) -> float:
        k = max(1, min(k, N_total))
        return _snr_for_shard_subset(
            all_shard_ids[:k], sim_out_dir, psr_names,
            noise_stoas, psrs_clean, raw_noise_params, Tspan_seconds,
            curn_components, rn_components,
        )

    # ── noise floor ───────────────────────────────────────────────────────────
    snr_noise, _, _ = compute_population_snr(
        psrs_clean=psrs_clean, population=None,
        raw_noise_params=raw_noise_params, Tspan=Tspan_seconds,
        current_stoas={n: noise_stoas[n] for n in noise_stoas},
        curn_components=curn_components, rn_components=rn_components,
    )
    print(f'  Noise-only OS SNR: {snr_noise:.4f}')
    if snr_high <= snr_noise:
        raise ValueError(
            f'SNR band ceiling ({snr_high:.4f}) ≤ noise-only SNR '
            f'({snr_noise:.4f}). Widen target band or use a different seed.')

    # ── full-population SNR ───────────────────────────────────────────────────
    print(f'  Evaluating SNR with all {N_total} sub-chunks...')
    snr_total = _snr_for_k(N_total)
    print(f'  SNR (all {N_total} sub-chunks): {snr_total:.4f}  '
          f'target=[{snr_low}, {snr_high}]')

    if snr_low <= snr_total <= snr_high:
        print(f'  ✓ Full population in target range — using all {N_total} sub-chunks')
        return all_shard_ids[:N_total], snr_total

    if snr_total < snr_low:
        print("  Full population below target. Testing fractional populations...")

        trial_fracs = [
            1/2,
            1/4,
            3/4,
            1/8,
            3/8,
            5/8,
            7/8,
            1/16,
            3/16,
            5/16,
            7/16,
            9/16,
            11/16,
            13/16,
            15/16,
        ]

        best_k = N_total
        best_snr = snr_total
        best_dist = abs(snr_total - target_snr)

        for frac in trial_fracs:

            k = max(1, int(round(frac * N_total)))

            print(f"    Testing k={k} ({frac:.1%})")

            snr_k = _snr_for_k(k)

            print(f"      SNR={snr_k:.4f}")

            dist = abs(snr_k - target_snr)

            if dist < best_dist:
                best_dist = dist
                best_k = k
                best_snr = snr_k

            if snr_low <= snr_k <= snr_high:
                print(
                    f"  ✓ Fractional population works: "
                    f"k={k}, SNR={snr_k:.4f}"
                )
                return all_shard_ids[:k], snr_k

        print(
            f"  ⚠ No fractional population reached target. "
            f"Using closest: k={best_k}, SNR={best_snr:.4f}"
        )

        return all_shard_ids[:best_k], best_snr

    # ── analytic first guess ──────────────────────────────────────────────────
    sig_total  = snr_total - snr_noise
    sig_target = target_snr - snr_noise
    if sig_total <= 0 or sig_target <= 0:
        raise ValueError(
            f'Signal-only SNR non-positive '
            f'(sig_total={sig_total:.4f}, sig_target={sig_target:.4f}).')

    k_guess = int(round(N_total * (sig_target / sig_total) ** 2))
    k_guess = max(1, min(k_guess, N_total))
    print(f'  Analytic first guess: k={k_guess}')

    # ── bisection on prefix all_shard_ids[:k] ────────────────────────────────
    k_lo, k_hi = 1, N_total
    history    = []   # (k, snr)

    for iteration in range(max_iterations):
        snr_k = _snr_for_k(k_guess)
        print(f'  Iter {iteration+1:2d}: k={k_guess:4d}/{N_total}  '
              f'OS SNR={snr_k:.4f}  target=[{snr_low},{snr_high}]')
        history.append((k_guess, snr_k))

        if snr_low <= snr_k <= snr_high:
            print(f'  ✓ Converged at k={k_guess} in {iteration+1} iterations')
            return all_shard_ids[:k_guess], snr_k

        if snr_k < snr_low:
            k_lo = k_guess
        else:
            k_hi = k_guess

        if k_hi - k_lo <= 1:
            # Bracket collapsed — check history first
            valid = [(k, s) for k, s in history if snr_low <= s <= snr_high]
            if valid:
                best_k, best_snr = max(valid, key=lambda x: x[1])
                print(f'  ✓ Bracket collapsed — using k={best_k} '
                      f'(SNR={best_snr:.4f}) from history')
                return all_shard_ids[:best_k], best_snr

            # ── fallback: chunk-addition scan ─────────────────────────────────
            # Find the largest prefix that is safely below snr_low, then try
            # adding each remaining chunk one at a time.
            print(f'\n  ⚠️  Bracket collapsed without valid k. '
                  f'Switching to chunk-addition scan...')

            # Identify floor: largest k in history with SNR < snr_low
            below = [(k, s) for k, s in history if s < snr_low]
            if below:
                k_floor, snr_floor = max(below, key=lambda x: x[0])
            else:
                # No prefix gave SNR < snr_low — use k=0 (noise only) as floor
                k_floor, snr_floor = 0, snr_noise

            floor_shards = all_shard_ids[:k_floor]
            print(f'  Floor: k={k_floor} shards, SNR={snr_floor:.4f}')

            # Candidate chunks: everything not already in the floor prefix
            candidate_chunks = all_shard_ids[k_floor:]
            print(f'  Scanning {len(candidate_chunks)} candidate chunks...')

            # We want floor_shards + [one extra chunk] to land in [snr_low, snr_high]
            hits   = []   # (snr, chunk_id, sub_id) — landed in band
            near   = []   # (snr, chunk_id, sub_id) — overshot but closest

            for cid, sid in candidate_chunks:
                trial_shards = floor_shards + [(cid, sid)]
                snr_trial = _snr_for_shard_subset(
                    trial_shards, sim_out_dir, psr_names,
                    noise_stoas, psrs_clean, raw_noise_params, Tspan_seconds,
                    curn_components, rn_components,
                )
                print(f'    chunk=({cid},{sid})  SNR={snr_trial:.4f}')

                if snr_low <= snr_trial <= snr_high:
                    hits.append((snr_trial, cid, sid))
                elif snr_trial > snr_high:
                    near.append((snr_trial, cid, sid))

                # Take the first hit we find — no need to scan all chunks
                if hits:
                    best_snr, best_c, best_s = hits[0]
                    active = floor_shards + [(best_c, best_s)]
                    print(f'  ✓ Chunk-addition found: floor({k_floor}) + '
                          f'chunk({best_c},{best_s}) → SNR={best_snr:.4f}')
                    return active, best_snr

            # All candidates either undershot or overshot — take closest overshoot
            if near:
                near.sort(key=lambda x: x[0])   # ascending SNR — smallest overshoot first
                best_snr, best_c, best_s = near[0]
                active = floor_shards + [(best_c, best_s)]
                print(f'  ⚠️  No chunk landed in band. '
                      f'Taking closest overshoot: chunk({best_c},{best_s}) '
                      f'SNR={best_snr:.4f} (target=[{snr_low},{snr_high}])')
                return active, best_snr

            # Every candidate undershot even when added to the floor — give up
            raise RuntimeError(
                f'Chunk-addition scan exhausted: no single chunk raises '
                f'SNR from {snr_floor:.4f} (floor k={k_floor}) into or above '
                f'[{snr_low},{snr_high}]. '
                f'Consider generating more sub-chunks or widening the target band.')

        # Continue bisection
        k_guess = int(round(np.exp(0.5 * (np.log(k_lo + 1) + np.log(k_hi)))))
        k_guess = max(k_lo + 1, min(k_guess, k_hi - 1))

    # Ran out of iterations — check history
    valid = [(k, s) for k, s in history if snr_low <= s <= snr_high]
    if valid:
        best_k, best_snr = max(valid, key=lambda x: x[1])
        return all_shard_ids[:best_k], best_snr

    last_k, last_snr = history[-1]
    raise RuntimeError(
        f'Chunk-count SNR selection failed in {max_iterations} iterations. '
        f'Last: k={last_k}, SNR={last_snr:.4f}, target=[{snr_low},{snr_high}].')

# =============================================================================
# CGW SNR infrastructure
# =============================================================================

def _build_candidate_list(
    store:         ShardedPickleStore,
    active_shards: List[Tuple[int, int]],
    enterprise_psrs,
    Tspan_seconds: float,
) -> List[Tuple[int, int, int, float]]:
    """
    Pass 1 (baseline only): scan active shards using analytic proxy to build
    a ranked global candidate list, then rescue low/mid-f regimes.

    Returns list of (chunk_id, sub_id, local_idx, proxy_score), sorted desc.
    """
    print(f'  Pre-filtering {len(active_shards)} active shards via analytic proxy...')
    candidate_list = []

    for chunk_id, sub_id in active_shards:
        pop    = store.read(chunk_id, sub_id)
        n      = len(pop)
        n_keep = min(N_PRE_FILTER_PER_CHUNK, n)

        if hasattr(pop, 'cgw_proxy') and pop.cgw_proxy is not None:
            proxies = pop.cgw_proxy.astype(np.float64)
        else:
            proxies = _compute_analytic_proxy(pop, enterprise_psrs,
                                              Tspan_seconds=Tspan_seconds)

        top_local = (np.argsort(proxies)[::-1] if n_keep == n
                     else np.argpartition(proxies, -n_keep)[-n_keep:])
        for local_idx in top_local:
            candidate_list.append(
                (chunk_id, sub_id, int(local_idx), float(proxies[local_idx])))
        del pop; gc.collect()

    candidate_list.sort(key=lambda x: x[3], reverse=True)
    global_candidates = candidate_list[:N_GLOBAL_CGW_CANDIDATES]

    # rescue low/mid-f proxy failure regimes
    existing = {(c, s, i) for c, s, i, _ in global_candidates}
    for chunk_id, sub_id in active_shards:
        pop = store.read(chunk_id, sub_id)
        f_T = np.asarray(pop.f, dtype=np.float64) * Tspan_seconds
        for mask in [f_T < 10.0, (f_T >= 10.0) & (f_T < 50.0)]:
            if mask.sum() == 0:
                continue
            regime_idx   = np.where(mask)[0]
            top_h0_local = regime_idx[np.argsort(pop.h0[regime_idx])[-N_RESCUE:]]
            for local_idx in top_h0_local:
                key = (chunk_id, sub_id, int(local_idx))
                if key not in existing:
                    global_candidates.append(
                        (chunk_id, sub_id, int(local_idx), float(pop.h0[local_idx])))
                    existing.add(key)
        del pop; gc.collect()

    n_rescued = len(global_candidates) - N_GLOBAL_CGW_CANDIDATES
    print(f'  {len(global_candidates)} total candidates '
          f'(top-{N_GLOBAL_CGW_CANDIDATES} proxy + {n_rescued} rescued)')
    return global_candidates


def _assemble_binaries(
    store:      ShardedPickleStore,
    candidates: List[Tuple[int, int, int, float]],
) -> List:
    """Load binary objects for a candidate list, grouped by shard to minimise reads."""
    # group by (chunk_id, sub_id)
    by_shard: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for global_rank, (chunk_id, sub_id, local_idx, _) in enumerate(candidates):
        by_shard[(chunk_id, sub_id)].append((global_rank, local_idx))

    binaries = [None] * len(candidates)
    for (chunk_id, sub_id), entries in by_shard.items():
        pop = store.read(chunk_id, sub_id)
        for global_rank, local_idx in entries:
            binaries[global_rank] = pop[local_idx]
        del pop; gc.collect()
    return binaries


def _write_snrs_to_shards(
    store:      ShardedPickleStore,
    candidates: List[Tuple[int, int, int, float]],
    snrs:       np.ndarray,
    active_shards: List[Tuple[int, int]],
    snr_field:  str,
) -> None:
    """Write per-binary SNR into each active shard under snr_field."""
    # Map (chunk_id, sub_id, local_idx) → snr
    shard_snr_maps: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
    for global_rank, (chunk_id, sub_id, local_idx, _) in enumerate(candidates):
        shard_snr_maps[(chunk_id, sub_id)][local_idx] = float(snrs[global_rank])

    for chunk_id, sub_id in active_shards:
        pop         = store.read(chunk_id, sub_id)
        cgw_snr_arr = np.zeros(len(pop), dtype=np.float32)
        for local_idx, snr_val in shard_snr_maps.get((chunk_id, sub_id), {}).items():
            cgw_snr_arr[local_idx] = snr_val
        setattr(pop, snr_field, cgw_snr_arr)
        store._dump(store._path(chunk_id, sub_id), pop)
        del pop; gc.collect()


def _print_top_sources(
    store:      ShardedPickleStore,
    candidates: List[Tuple[int, int, int, float]],
    snrs:       np.ndarray,
    snr_field:  str,
    n_show:     int = N_TOP_SOURCES,
) -> None:
    ranked = sorted(zip(candidates, snrs), key=lambda x: x[1], reverse=True)
    n_show = min(n_show, len(ranked))
    by_shard: Dict[Tuple[int, int], object] = {}
    for (chunk_id, sub_id, _, _), _ in ranked[:n_show]:
        key = (chunk_id, sub_id)
        if key not in by_shard:
            by_shard[key] = store.read(chunk_id, sub_id)
    proxy_rank_map = {
        (c, s, i): rank
        for rank, (c, s, i, _) in enumerate(candidates, start=1)
    }
    print(f'\n  Top {n_show} [{snr_field}]:')
    for rank, ((chunk_id, sub_id, local_idx, proxy), snr_val) in enumerate(
            ranked[:n_show], start=1):
        pop        = by_shard[(chunk_id, sub_id)]
        proxy_rank = proxy_rank_map.get((chunk_id, sub_id, local_idx), -1)
        print(f'    {rank:2d}.  chunk={chunk_id:03d}  sub={sub_id:03d}  '
              f'local_idx={local_idx:7d}  '
              f'f={pop.f[local_idx]:.2e} Hz  '
              f'Mc={pop.Mc[local_idx]:.2e} Msun  '
              f'h0={pop.h0[local_idx]:.2e}  '
              f'proxy={proxy:.4e} (rank #{proxy_rank})  '
              f'{snr_field}={snr_val:.4f}')
    del by_shard; gc.collect()


def _compute_cgw_snrs_baseline(
    store:               ShardedPickleStore,
    active_shards:       List[Tuple[int, int]],
    pta,
    enterprise_psrs,
    raw_noise_params:    dict,
    parsed_noise_params: dict,
    Tspan_seconds:       float,
    meta_dir:            str,
) -> Tuple[List[Tuple[int, int, int, float]], np.ndarray]:
    """
    Two-pass CGW SNR for the baseline PTA over active shards.
    Returns (global_candidates, top_snrs).
    """
    print('\n  [baseline] Building candidate list...')
    global_candidates = _build_candidate_list(
        store, active_shards, enterprise_psrs, Tspan_seconds)

    top_binaries = _assemble_binaries(store, global_candidates)

    print(f'  [baseline] Computing CGW SNR for {len(top_binaries)} candidates...')
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
    del top_binaries; gc.collect()

    _write_snrs_to_shards(store, global_candidates, top_snrs, active_shards, 'cgw_snr')
    _print_top_sources(store, global_candidates, np.asarray(top_snrs), 'cgw_snr')

    # Persist top-5 breakdowns
    ranked = sorted(zip(global_candidates, top_snrs), key=lambda x: x[1], reverse=True)
    proxy_rank_map = {
        (c, s, i): rank
        for rank, (c, s, i, _) in enumerate(global_candidates, start=1)
    }
    cand_breakdown = {
        (c, s, i): top_breakdowns[idx]
        for idx, (c, s, i, _) in enumerate(global_candidates)
    }
    top_breakdown_records = []
    for rank, ((chunk_id, sub_id, local_idx, proxy), snr_val) in enumerate(
            ranked[:5], start=1):
        proxy_rank = proxy_rank_map[(chunk_id, sub_id, local_idx)]
        per_pulsar = cand_breakdown[(chunk_id, sub_id, local_idx)]
        top_breakdown_records.append({
            'rank':              rank,
            'proxy_rank':        proxy_rank,
            'chunk_id':          int(chunk_id),
            'sub_id':            int(sub_id),
            'local_idx':         int(local_idx),
            'proxy':             float(proxy),
            'cgw_snr':           float(snr_val),
            'per_pulsar_rho_sq': {k: float(v) for k, v in per_pulsar.items()},
        })
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, 'top_cgw_breakdowns.json'), 'w') as fh:
        json.dump(top_breakdown_records, fh, indent=2)

    return global_candidates, np.asarray(top_snrs)


def _compute_cgw_snrs_scenario(
    store:                ShardedPickleStore,
    active_shards:        List[Tuple[int, int]],
    pta,
    enterprise_psrs,
    raw_noise_params:     dict,
    parsed_noise_params:  dict,
    Tspan_seconds:        float,
    snr_field:            str,
    baseline_candidates:  List[Tuple[int, int, int, float]],
    baseline_snrs:        np.ndarray,
    n_candidates:         int = N_SCENARIO_CGW_CANDIDATES,
) -> None:
    """
    Lightweight CGW pass for a synthetic scenario — uses top-N by baseline SNR.
    No proxy re-scan needed.
    """
    ranked_by_snr = sorted(
        zip(baseline_candidates, baseline_snrs),
        key=lambda x: x[1], reverse=True,
    )
    candidates_for_scenario = [c for c, _ in ranked_by_snr[:n_candidates]]

    print(f'\n  [{snr_field}] Using top-{len(candidates_for_scenario)} '
          f'by baseline CGW SNR')

    top_binaries = _assemble_binaries(store, candidates_for_scenario)

    print(f'  [{snr_field}] Computing CGW SNR for {len(top_binaries)} candidates...')
    top_snrs = compute_cgw_snr_optimal_population_fast(
        psrs=enterprise_psrs,
        pta=pta,
        population=top_binaries,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan=Tspan_seconds,
        profile=True,
        return_breakdown=False,
    )
    del top_binaries; gc.collect()

    _write_snrs_to_shards(store, candidates_for_scenario, top_snrs, active_shards, snr_field)
    _print_top_sources(store, candidates_for_scenario, np.asarray(top_snrs), snr_field)


# =============================================================================
# Residual saving
# =============================================================================

def _save_toa_residuals(
    sim_out_dir:    str,
    psrs_clean,
    noise_stoas:    Dict[str, np.ndarray],
    combined_delta: Dict[str, np.ndarray],
    n_active:       int,
    n_total:        int,
    scenario:       Optional[str] = None,
) -> None:
    """
    Save noise / population / combined residual arrays.
    combined_delta is already the sum over active shards (no further scaling).
    """
    suffix = f'_{scenario}' if scenario else ''
    base   = os.path.join(sim_out_dir, f'residuals{suffix}')
    dirs   = {k: os.path.join(base, k)
              for k in ('noise', 'population', 'combined')}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f'\n💾 Saving TOA residuals [{scenario or "baseline"}] → {base}/')
    for psr in psrs_clean:
        name    = psr.name
        noise_a = noise_stoas[name]
        pop_a   = combined_delta[name]      # physical — no rescaling
        comb_a  = noise_a + pop_a
        np.save(os.path.join(dirs['noise'],      f'{name}.npy'), noise_a.astype(np.float64))
        np.save(os.path.join(dirs['population'], f'{name}.npy'), pop_a.astype(np.float64))
        np.save(os.path.join(dirs['combined'],   f'{name}.npy'), comb_a.astype(np.float64))

    manifest = {
        'n_active_sub_chunks': n_active,
        'n_total_sub_chunks':  n_total,
        'fraction_used':       n_active / max(n_total, 1),
        'n_pulsars':           len(psrs_clean),
        'psr_names':           [p.name for p in psrs_clean],
        'scenario':            scenario or 'baseline',
        'description': {
            'noise':      'Simulated noise residuals only (no GW signal)',
            'population': 'GW TOA contribution from active sub-chunks (physical, unscaled)',
            'combined':   'noise + population (what an observer would measure)',
        },
    }
    with open(os.path.join(base, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'  ✓ Saved residuals for {len(psrs_clean)} pulsars '
          f'({n_active}/{n_total} sub-chunks, no rescaling)')


# =============================================================================
# Subprocess phase runner
# =============================================================================

def _run_phase(extra_argv: list, label: str) -> None:
    cmd = [sys.executable, os.path.abspath(__file__)] + extra_argv
    print(f'\n  ▶ Subprocess: {label}', flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f'Subprocess phase "{label}" exited with code {result.returncode}')
    print(f'  ✓ Subprocess complete: {label}', flush=True)


# =============================================================================
# Phase entry points
# =============================================================================

def _phase_baseline(args, syn_scenarios, combined_scenarios, noise_seed):
    """
    Baseline phase (subprocess):
      1. Load baseline pulsars.
      2. Simulate noise; retry with shifted seed if noise-only SNR too high.
      3. Find k = number of sub-chunks giving OS SNR ∈ [snr_low, snr_high].
         Population parameters are NOT modified.
      4. Save residuals (active sub-chunks only, unscaled).
      5. Compute baseline CGW SNRs.
      6. Write phase_handoff.json.
    """
    sim_out_dir = os.path.join(args.output_dir, f'sim{args.sim_id:03d}')
    meta_dir    = os.path.join(sim_out_dir, 'metadata')

    with open(os.path.join(meta_dir, 'config.json')) as fh:
        run_config = json.load(fh)
    Tspan_seconds = run_config['Tspan_seconds']

    pop_dir      = os.path.join(sim_out_dir, 'populations')
    store        = ShardedPickleStore(pop_dir)
    all_shards   = store.available()      # list of (chunk_id, sub_id)
    N_total      = len(all_shards)
    if N_total == 0:
        sys.exit(f'ERROR: no shards in {pop_dir}')

    snr_low, snr_high = args.snr_range

    print('\n📡 [baseline] Loading pulsars...')
    psrs_unfiltered = load_pulsars(verbose=True, scenario='baseline',
                                   scenarios=combined_scenarios)
    with suppress_enterprise_warnings():
        psrs_clean, raw_noise_params, _ = filter_pulsars_15yr(
            psrs_unfiltered, verbose=True)
    del psrs_unfiltered; gc.collect()
    psr_names = [p.name for p in psrs_clean]

    parsed_noise_params = parse_pulsar_parameters(config.NOISEFILE)

    # ── proxy-only mode ───────────────────────────────────────────────────────
    if args.proxy_only:
        print(f'\n{"="*60}\nProxy-only mode\n{"="*60}')
        noise_stoas = _simulate_noise(psrs_clean, raw_noise_params, seed=noise_seed)
        _, pta, enterprise_psrs = compute_population_snr(
            psrs_clean=psrs_clean, population=None,
            raw_noise_params=raw_noise_params, Tspan=Tspan_seconds,
            current_stoas=noise_stoas, return_psrs_pta=True,
        )
        validate_cgw_proxy(
            store=store, chunk_ids=all_shards, pta=pta,
            enterprise_psrs=enterprise_psrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds, n_test=args.n_test,
        )
        validate_proxy_filtering_ratio(
            store=store, chunk_ids=all_shards, pta=pta,
            enterprise_psrs=enterprise_psrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds, n_keep=N_PRE_FILTER_PER_CHUNK,
        )
        sentinel = os.path.join(meta_dir, 'stage2_complete.json')
        with open(sentinel, 'w') as fh:
            json.dump({'sim_id': args.sim_id, 'proxy_only': True,
                       'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S')}, fh, indent=2)
        with open(os.path.join(meta_dir, HANDOFF_FILENAME), 'w') as fh:
            json.dump({'n_active_sub_chunks': N_total,
                       'active_shard_ids': [[c, s] for c, s in all_shards],
                       'Tspan_seconds': Tspan_seconds,
                       'noise_seed': noise_seed,
                       'baseline_candidates': [], 'baseline_snrs': []}, fh)
        return

    # ── Noise simulation with retry on high noise floor or NaN SNR ───────────────
    max_noise_retries = 20
    active_shards     = None
    snr_achieved      = None
    winning_seed      = noise_seed

    for noise_attempt in range(max_noise_retries):
        # Each attempt gets a seed offset by attempt * large_prime to avoid
        # overlap with other sim_ids (which are spaced by 10000).
        # Using a prime (97) ensures no two (sim_id, attempt) pairs collide:
        # seed = base + sim_id^2 * 10000 + attempt * 97
        # For any two sims i != j and attempts a, b:
        # i^2 * 10000 + a*97 != j^2 * 10000 + b*97 (for i,j < 100, a,b < 20)
        attempt_seed = noise_seed + noise_attempt * 97

        if noise_attempt == 0:
            print(f'\n🔊 [baseline] Simulating noise (seed={attempt_seed})...')
        else:
            print(f'\n🔁 [baseline] Retrying noise '
                f'(attempt {noise_attempt + 1}/{max_noise_retries}, '
                f'seed={attempt_seed})...')

        try:
            noise_stoas = _simulate_noise(psrs_clean, raw_noise_params,
                                        seed=attempt_seed)
        except Exception as e:
            print(f'  ⚠️  Noise simulation failed: {e}, retrying...')
            continue

        # Quick NaN check on noise-only SNR before attempting full shard selection
        print('\n📐 [baseline] Checking noise floor...')
        try:
            snr_noise, _, _ = compute_population_snr(
                psrs_clean=psrs_clean,
                population=None,
                raw_noise_params=raw_noise_params,
                Tspan=Tspan_seconds,
                current_stoas=noise_stoas,
                return_psrs_pta=True,
                verbose=False,
            )
        except Exception as e:
            print(f'  ⚠️  Noise-only SNR computation failed: {e}, retrying...')
            continue

        if not np.isfinite(snr_noise):
            print(f'  ⚠️  Noise-only SNR is NaN (seed={attempt_seed}), retrying...')
            if noise_attempt == max_noise_retries - 1:
                sys.exit(f'ERROR: noise-only SNR NaN after '
                        f'{max_noise_retries} attempts — giving up')
            continue

        if abs(snr_noise) > snr_high:
            print(f'  ⚠️  Noise-only SNR={snr_noise:.3f} exceeds ceiling '
                f'{snr_high}, retrying...')
            if noise_attempt == max_noise_retries - 1:
                sys.exit(f'ERROR: noise floor too high after '
                        f'{max_noise_retries} attempts — giving up')
            continue

        print(f'  ✓ Noise-only SNR={snr_noise:.4f} (seed={attempt_seed})')

        print('\n📐 [baseline] Selecting active sub-chunks (SNR ∝ √N)...')
        try:
            active_shards, snr_achieved = _select_active_shards(
                all_shard_ids=all_shards,
                psrs_clean=psrs_clean,
                raw_noise_params=raw_noise_params,
                sim_out_dir=sim_out_dir,
                psr_names=psr_names,
                noise_stoas=noise_stoas,
                target_snr=args.target_snr,
                snr_low=snr_low,
                snr_high=snr_high,
                Tspan_seconds=Tspan_seconds,
            )
            winning_seed = attempt_seed
            print(f'✓ Converged — k={len(active_shards)}/{N_total} sub-chunks  '
                f'SNR={snr_achieved:.4f}  noise_seed={winning_seed}')
            break

        except ValueError as e:
            if 'noise-only SNR' in str(e) or 'SNR band ceiling' in str(e):
                print(f'  ⚠️  {e}')
                if noise_attempt == max_noise_retries - 1:
                    sys.exit(f'ERROR: noise floor too high after '
                            f'{max_noise_retries} attempts — giving up')
                continue
            sys.exit(f'ERROR: {e}')
        except RuntimeError as e:
            sys.exit(f'ERROR: {e}')

    n_active = len(active_shards)
    print(f'\n  Active sub-chunks: {n_active}/{N_total} '
          f'({100 * n_active / N_total:.1f}%)')

    # ── Sum deltas for active shards and save residuals ───────────────────────
    print('\n📂 [baseline] Summing Δstoas for active shards...')
    combined_delta = _load_and_sum_toa_deltas(
        sim_out_dir, active_shards, psr_names, scenario=None)

    _save_toa_residuals(sim_out_dir, psrs_clean, noise_stoas,
                        combined_delta, n_active, N_total, scenario=None)
    _cleanup_shard_stoas(sim_out_dir, all_shards, scenario=None)
    del combined_delta; gc.collect()

    # delete unused shards
    _cleanup_inactive_population_shards(store, all_shards, active_shards)

    # Reconstruct signal_stoas from saved residuals
    resid_base = os.path.join(sim_out_dir, 'residuals', 'combined')
    signal_stoas = {}
    for name in psr_names:
        arr = np.load(os.path.join(resid_base, f'{name}.npy'))
        signal_stoas[name] = arr

    # Rebuild PTA from combined (noise + signal) stoas
    _, pta, enterprise_psrs = compute_population_snr(
        psrs_clean=psrs_clean,
        population=None,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan_seconds,
        current_stoas=signal_stoas,
    )

    baseline_candidates: Optional[List] = None
    baseline_snrs:       Optional[np.ndarray] = None

    if args.cgw:
        if args.validate_proxy:
            print('\n🔍 Validating CGW proxy...')
            validate_cgw_proxy(
                store=store, chunk_ids=active_shards, pta=pta,
                enterprise_psrs=enterprise_psrs,
                raw_noise_params=raw_noise_params,
                parsed_noise_params=parsed_noise_params,
                Tspan_seconds=Tspan_seconds, n_test=args.n_test,
            )

        print('\n🔭 [baseline] Computing CGW SNRs...')
        baseline_candidates, baseline_snrs = _compute_cgw_snrs_baseline(
            store=store, active_shards=active_shards,
            pta=pta, enterprise_psrs=enterprise_psrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds,
            meta_dir=meta_dir,
        )
        print('✓ Baseline CGW complete')
    else:
        print('\n(CGW skipped — use --cgw to enable)')

    del pta, enterprise_psrs, psrs_clean; gc.collect()

    # Serialise candidates as [chunk_id, sub_id, local_idx, proxy]
    handoff = {
        'n_active_sub_chunks': n_active,
        'n_total_sub_chunks':  N_total,
        'active_shard_ids':    [[c, s] for c, s in active_shards],
        'Tspan_seconds':       Tspan_seconds,
        'noise_seed':          winning_seed,
        'baseline_candidates': [[c, s, i, float(p)]
                                for c, s, i, p in (baseline_candidates or [])],
        'baseline_snrs':       (baseline_snrs.tolist()
                                if baseline_snrs is not None else []),
    }
    with open(os.path.join(meta_dir, HANDOFF_FILENAME), 'w') as fh:
        json.dump(handoff, fh)
    print(f'✓ [baseline] Wrote {HANDOFF_FILENAME}  '
          f'(k={n_active}/{N_total}, noise_seed={winning_seed})')


def _phase_scenario(args, scenario_label, combined_scenarios):
    """
    Scenario phase (subprocess):
      - Read handoff: active_shard_ids, noise_seed, baseline candidate ranking.
      - Load scenario pulsars; simulate noise with SAME seed as baseline.
      - Sum Δstoas for active shards ONLY (same k as baseline).
      - Save residuals; clean scenario npz files.
      - Build Enterprise PTA; compute CGW SNRs; write to shards.
    """
    sim_out_dir = os.path.join(args.output_dir, f'sim{args.sim_id:03d}')
    meta_dir    = os.path.join(sim_out_dir, 'metadata')

    handoff_path = os.path.join(meta_dir, HANDOFF_FILENAME)
    if not os.path.isfile(handoff_path):
        sys.exit(f'ERROR: handoff file not found: {handoff_path}')
    with open(handoff_path) as fh:
        handoff = json.load(fh)

    active_shard_ids    = [tuple(x) for x in handoff['active_shard_ids']]
    n_active            = handoff['n_active_sub_chunks']
    n_total             = handoff.get('n_total_sub_chunks', n_active)
    Tspan_seconds       = handoff['Tspan_seconds']
    noise_seed          = handoff['noise_seed']
    baseline_candidates = [(c, s, i, p)
                           for c, s, i, p in handoff['baseline_candidates']]
    baseline_snrs       = np.array(handoff['baseline_snrs'], dtype=np.float64)

    pop_dir   = os.path.join(sim_out_dir, 'populations')
    store     = ShardedPickleStore(pop_dir)
    all_shards = store.available()

    parsed_noise_params = parse_pulsar_parameters(config.NOISEFILE)

    print(f'\n📡 [{scenario_label}] Loading pulsars...')
    psrs_unfiltered = load_pulsars(verbose=True, scenario=scenario_label,
                                   scenarios=combined_scenarios)
    with suppress_enterprise_warnings():
        psrs_clean, raw_noise_params, _ = filter_pulsars_15yr(
            psrs_unfiltered, verbose=False)
    del psrs_unfiltered; gc.collect()
    psr_names = [p.name for p in psrs_clean]

    # Use the SAME active shard subset as baseline (same k)
    print(f'\n📂 [{scenario_label}] Summing Δstoas for '
          f'{n_active}/{n_total} active shards...')
    combined_delta = _load_and_sum_toa_deltas(
        sim_out_dir, active_shard_ids, psr_names, scenario=scenario_label)

    print(f'\n🔊 [{scenario_label}] Simulating noise (seed={noise_seed})...')
    noise_stoas = _simulate_noise(psrs_clean, raw_noise_params, seed=noise_seed)

    _save_toa_residuals(sim_out_dir, psrs_clean, noise_stoas,
                        combined_delta, n_active, n_total, scenario=scenario_label)
    _cleanup_shard_stoas(sim_out_dir, all_shards, scenario=scenario_label)
    del noise_stoas, combined_delta; gc.collect()

    print(f'\n🔧 [{scenario_label}] Building Enterprise PTA...')
    # Reconstruct signal stoas from saved residuals
    resid_base   = os.path.join(sim_out_dir, f'residuals_{scenario_label}', 'combined')
    signal_stoas = {}
    for name in psr_names:
        arr = np.load(os.path.join(resid_base, f'{name}.npy'))
        signal_stoas[name] = arr

    for psr in psrs_clean:
        psr.stoas[:] = signal_stoas[psr.name]

    pta, epsrs = _build_pta(psrs_clean, raw_noise_params, Tspan_seconds)
    del psrs_clean; gc.collect()

    if args.cgw and len(baseline_candidates) > 0:
        _compute_cgw_snrs_scenario(
            store=store, active_shards=active_shard_ids,
            pta=pta, enterprise_psrs=epsrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds,
            snr_field=f'cgw_snr_{scenario_label}',
            baseline_candidates=baseline_candidates,
            baseline_snrs=baseline_snrs,
            n_candidates=N_SCENARIO_CGW_CANDIDATES,
        )
    elif args.cgw:
        print(f'  WARNING: no baseline candidates, skipping CGW for {scenario_label}')

    del pta, epsrs; gc.collect()
    print(f'✓ [{scenario_label}] Phase complete')


# =============================================================================
# Per-sim orchestrator
# =============================================================================

def process_sim(
    sim_id:             int,
    args,
    syn_scenarios:      dict,
    combined_scenarios: dict,
    noise_seed:         int,
) -> bool:
    t_sim       = time.time()
    sim_out_dir = os.path.join(args.output_dir, f'sim{sim_id:03d}')

    print(f'\n{"="*60}')
    print(f'Stage 2 orchestrator — sim_id={sim_id}')
    print(f'  Scenarios : baseline + {list(syn_scenarios.keys())}')
    print(f'  Each phase runs in its own subprocess (clean C heap)')
    print(f'{"="*60}')

    config_path = os.path.join(sim_out_dir, 'metadata', 'config.json')
    if not os.path.isfile(config_path):
        print(f'ERROR: config.json not found at {config_path}', file=sys.stderr)
        return False

    def _common_argv():
        argv = [
            '--output-dir',      args.output_dir,
            '--config',          args.config,
            '--target-snr',      str(args.target_snr),
            '--snr-range',       str(args.snr_range[0]), str(args.snr_range[1]),
            '--sim-id',          str(sim_id),
            '--n-chunks',        str(args.n_chunks),
            '--n-sub-chunks',    str(args.n_sub_chunks),
            '--noise-seed-base', str(args.noise_seed_base),
            '--n-test',          str(args.n_test),
        ]
        if args.cgw:              argv.append('--cgw')
        if args.validate_proxy:   argv.append('--validate-proxy')
        if args.proxy_only:       argv.append('--proxy-only')
        if args.synthetic_ptas:   argv.append('--synthetic-ptas')
        if args.synthetic_pta_config:
            argv += ['--synthetic-pta-config', args.synthetic_pta_config]
        return argv

    try:
        _run_phase(_common_argv() + ['--phase', 'baseline'],
                   label=f'sim{sim_id:03d}/baseline')
    except RuntimeError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return False

    for scenario_label in syn_scenarios:
        try:
            _run_phase(
                _common_argv() + ['--phase', 'scenario',
                                  '--phase-scenario', scenario_label],
                label=f'sim{sim_id:03d}/{scenario_label}',
            )
        except RuntimeError as e:
            print(f'ERROR: {e}', file=sys.stderr)
            return False

    # Read handoff for summary metadata
    handoff_path = os.path.join(sim_out_dir, 'metadata', HANDOFF_FILENAME)
    n_active, n_total = 0, 0
    if os.path.isfile(handoff_path):
        with open(handoff_path) as fh:
            h = json.load(fh)
            n_active = h.get('n_active_sub_chunks', 0)
            n_total  = h.get('n_total_sub_chunks', 0)

    sentinel = os.path.join(sim_out_dir, 'metadata', 'stage2_complete.json')
    with open(sentinel, 'w') as fh:
        json.dump({
            'sim_id':              sim_id,
            'n_active_sub_chunks': n_active,
            'n_total_sub_chunks':  n_total,
            'fraction_used':       n_active / max(n_total, 1),
            'scenarios':           ['baseline'] + list(syn_scenarios.keys()),
            'n_cgw_baseline':      N_GLOBAL_CGW_CANDIDATES,
            'n_cgw_scenario':      N_SCENARIO_CGW_CANDIDATES,
            'completed_at':        time.strftime('%Y-%m-%dT%H:%M:%S'),
        }, fh, indent=2)

    elapsed = time.time() - t_sim
    print(f'\n✅ Stage 2 sim_id={sim_id} complete in {elapsed/60:.1f} min')
    return True


# =============================================================================
# Summary builder
# =============================================================================

def _build_summary_object(
    output_dir:          str,
    sim_id:              int,
    n_keep_per_category: int = 200,
) -> None:
    sim_out_dir = os.path.join(output_dir, f'sim{sim_id:03d}')
    out_path    = os.path.join(sim_out_dir, 'summary.pkl.gz')
    pop_dir     = os.path.join(sim_out_dir, 'populations')
    store       = ShardedPickleStore(pop_dir)
    all_shards  = store.available()    # list of (chunk_id, sub_id)

    print(f'\n{"="*60}')
    print(f'Building summary — sim{sim_id:03d} → {out_path}')
    print(f'  n_keep_per_category = {n_keep_per_category}')
    print(f'{"="*60}')

    first_pop  = store.read(*all_shards[0])
    snr_fields = [attr for attr in vars(first_pop) if attr.startswith('cgw_snr')]
    all_fields = [attr for attr in vars(first_pop)
                  if isinstance(getattr(first_pop, attr), np.ndarray)]
    print(f'  SNR fields  : {snr_fields}')
    print(f'  All fields  : {all_fields}')
    del first_pop

    _param_categories: Dict[str, Tuple[list, bool, str]] = {
        'D_comov_near': ([], False, 'D_comov'),
        'D_comov_far':  ([], True,  'D_comov'),
        'f_low':        ([], False, 'f'),
        'f_high':       ([], True,  'f'),
        'Mc_low':       ([], False, 'Mc'),
        'Mc_high':      ([], True,  'Mc'),
        'Mtot_low':     ([], False, 'Mtot'),
        'Mtot_high':    ([], True,  'Mtot'),
        'h0_high':      ([], True,  'h0'),
        'z_low':        ([], False, 'z'),
        'z_high':       ([], True,  'z'),
    }
    for sf in snr_fields:
        _param_categories[f'{sf}_high'] = ([], True, sf)

    def _heap_push(heap, is_max, key, gidx, cap):
        entry = (-key if is_max else key, gidx)
        if len(heap) < cap:
            heapq.heappush(heap, entry)
        elif entry < heap[0]:
            heapq.heapreplace(heap, entry)

    field_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
    total_scanned = 0

    for chunk_id, sub_id in all_shards:
        pop = store.read(chunk_id, sub_id)
        n   = len(pop)

        for field in all_fields:
            if hasattr(pop, field):
                arr = getattr(pop, field)
                if isinstance(arr, np.ndarray):
                    field_buffers[field].append(arr)

        for local_i in range(n):
            gidx = total_scanned + local_i
            vals = {
                field: float(getattr(pop, field)[local_i])
                for field in all_fields
                if hasattr(pop, field) and
                   isinstance(getattr(pop, field), np.ndarray)
            }
            for cat_name, (heap, is_max, field) in _param_categories.items():
                if field in vals:
                    _heap_push(heap, is_max, vals[field], gidx,
                               n_keep_per_category)

        total_scanned += n
        del pop; gc.collect()

    category_indices: Dict[str, List[int]] = {
        cat_name: sorted({gidx for (_, gidx) in heap})
        for cat_name, (heap, _, _) in _param_categories.items()
    }

    print(f'  Total binaries scanned: {total_scanned:,}')
    for cat_name, idxs in category_indices.items():
        print(f'    {cat_name:<30s}: {len(idxs):,}')

    arrays: Dict[str, np.ndarray] = {
        field: np.concatenate(bufs)
        for field, bufs in field_buffers.items()
        if bufs
    }
    arrays['global_idx'] = np.arange(total_scanned, dtype=np.int64)

    top_cgw_breakdowns = []
    breakdown_path = os.path.join(sim_out_dir, 'metadata', 'top_cgw_breakdowns.json')
    if os.path.isfile(breakdown_path):
        with open(breakdown_path) as fh:
            top_cgw_breakdowns = json.load(fh)

    # Read handoff metadata for summary
    handoff_path = os.path.join(sim_out_dir, 'metadata', HANDOFF_FILENAME)
    handoff_meta = {}
    if os.path.isfile(handoff_path):
        with open(handoff_path) as fh:
            handoff_meta = json.load(fh)

    payload = {
        'arrays': arrays,
        'meta': {
            'sim_id':              sim_id,
            'total_scanned':       total_scanned,
            'n_keep_per_category': n_keep_per_category,
            'snr_fields':          snr_fields,
            'category_indices':    category_indices,
            'top_cgw_breakdowns':  top_cgw_breakdowns,
            'n_active_sub_chunks': handoff_meta.get('n_active_sub_chunks'),
            'n_total_sub_chunks':  handoff_meta.get('n_total_sub_chunks'),
            'fraction_used':       handoff_meta.get('n_active_sub_chunks', 0)
                                   / max(handoff_meta.get('n_total_sub_chunks', 1), 1),
            'summary_version':     4,
            'full_population':     True,
        },
    }
    with gzip.open(out_path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=4)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'✅ Summary written: {out_path}  ({size_mb:.1f} MB)')


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Stage 2: chunk-count SNR selection + CGW')
    p.add_argument('--output-dir',           type=str, required=True)
    p.add_argument('--config', '-c',         default='optimistic',
                   choices=list(config.POPULATION_CONFIGS.keys()))
    p.add_argument('--target-snr',           type=float, default=4.0)
    p.add_argument('--snr-range',            nargs=2, type=float, default=[3.5, 4.25])
    p.add_argument('--n-chunks',             type=int, required=True)
    p.add_argument('--n-sub-chunks',         type=int, default=1,
                   help='Number of sub-chunks per Slurm task (must match stage 1).')
    p.add_argument('--sim-id',               type=int, required=True)
    p.add_argument('--cgw',                  action='store_true')
    p.add_argument('--validate-proxy',       action='store_true')
    p.add_argument('--proxy-only',           action='store_true')
    p.add_argument('--n-test',               type=int, default=1_000)
    p.add_argument('--clean-failed',         action='store_true')
    p.add_argument('--noise-seed-base',      type=int, default=None)
    p.add_argument('--n-sims',              type=int, default=1)
    p.add_argument('--synthetic-ptas',       action='store_true', default=False)
    p.add_argument('--synthetic-pta-config', type=str, default=None)
    p.add_argument('--phase',                type=str, default=None,
                   choices=['baseline', 'scenario'],
                   help=argparse.SUPPRESS)
    p.add_argument('--phase-scenario',       type=str, default=None,
                   help=argparse.SUPPRESS)
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t0   = time.time()

    if args.clean_failed:
        import shutil
        print('\n🧹 Cleaning incomplete sim directories...')
        for sim_dir in sorted(glob.glob(
                os.path.join(args.output_dir, 'sim[0-9][0-9][0-9]'))):
            sentinel = os.path.join(sim_dir, 'metadata', 'stage2_complete.json')
            if not os.path.isfile(sentinel):
                print(f'  Removing {os.path.basename(sim_dir)}...')
                shutil.rmtree(sim_dir)
            else:
                print(f'  Preserving {os.path.basename(sim_dir)}')
        print('✓ Done'); sys.exit(0)

    if args.sim_id is None:
        print('ERROR: --sim-id required', file=sys.stderr); sys.exit(1)

    root_sq  = np.random.SeedSequence(args.noise_seed_base)
    sim_seq  = root_sq.spawn(10000)[args.sim_id]
    rng      = np.random.default_rng(sim_seq)
    seed_int = rng.integers(2**31).item()

    if args.synthetic_ptas:
        if args.synthetic_pta_config:
            syn_scenarios = json.loads(args.synthetic_pta_config)
        else:
            syn_scenarios = {k: v for k, v in DEFAULT_SCENARIOS.items()
                             if k != 'baseline'}
    else:
        syn_scenarios = {}

    combined_scenarios = dict(DEFAULT_SCENARIOS)
    combined_scenarios.update(syn_scenarios)
    noise_seed = (args.noise_seed_base + args.sim_id**2 * 10000
                if args.noise_seed_base is not None
                else args.sim_id * 10000)

    # ── Subprocess phase dispatch ─────────────────────────────────────────────
    if args.phase == 'baseline':
        _phase_baseline(args, syn_scenarios, combined_scenarios, noise_seed)
        sys.exit(0)

    if args.phase == 'scenario':
        if not args.phase_scenario:
            sys.exit('ERROR: --phase scenario requires --phase-scenario <label>')
        _phase_scenario(args, args.phase_scenario, combined_scenarios)
        sys.exit(0)

    # ── Top-level orchestrator ────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Stage 2 — sim_id={args.sim_id}')
    print(f'  Output dir    : {args.output_dir}')
    print(f'  Config        : {args.config}')
    print(f'  Target SNR    : {args.target_snr}  range={args.snr_range}')
    print(f'  Chunks/sim    : {args.n_chunks}  ×  {args.n_sub_chunks} sub-chunks')
    print(f'  Total shards  : {args.n_chunks * args.n_sub_chunks}')
    print(f'  CGW           : {"enabled" if args.cgw else "disabled"}')
    print(f'  Noise seed    : {noise_seed}')
    print(f'  Syn scenarios : {list(syn_scenarios.keys()) or "none"}')
    print(f'{"="*60}\n')

    success = process_sim(
        sim_id=args.sim_id,
        args=args,
        syn_scenarios=syn_scenarios,
        combined_scenarios=combined_scenarios,
        noise_seed=noise_seed,
    )

    if not success:
        print(f'ERROR: sim_id={args.sim_id} failed', file=sys.stderr)
        sys.exit(1)

    _build_summary_object(args.output_dir, sim_id=args.sim_id,
                          n_keep_per_category=200)

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'Stage 2 complete — sim_id={args.sim_id} in {elapsed/60:.1f} min')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()