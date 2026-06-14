#!/usr/bin/env python3
"""
Stage 2 — Noise simulation, SNR scaling, and CGW analysis.

For each simulation:
  1. Load baseline pulsars; sum Δstoa chunks; simulate noise; scale GW
     signal to target OS SNR.
  2. For each synthetic PTA scenario: load scenario pulsars; sum scenario
     Δstoa chunks; simulate noise with the SAME seed; reuse the baseline
     scale factor; compute CGW SNRs; write results into the same shard under
     a scenario-specific field (e.g. cgw_snr_5x_cadence).
  3. Write per-sim summary object.

The baseline scale factor is applied to all scenarios so that the injected
GW population is drawn at the same physical distance / strain in every case —
only the PTA sensitivity changes between scenarios.

Memory strategy
───────────────
Each phase (baseline + one per synthetic scenario) runs in its own subprocess
so that libstempo/tempo2's C-level global state is fully cleared between
phases.  free()-level heap corruption (the OOM-kill cause) cannot occur
because each subprocess exits completely before the next starts.

Within each subprocess only one Enterprise PTA is alive at a time.  The
baseline CGW SNR ranking is computed once and reused by synthetic scenarios
(top-N by actual baseline inner-product SNR) — no repeated proxy scans.
Per-chunk Δstoa files are deleted after residuals are saved.

Subprocess handoff
──────────────────
The baseline subprocess writes metadata/phase_handoff.json containing:
  cumulative_scale    float
  Tspan_seconds       float
  baseline_candidates list of [chunk_id, local_idx, proxy]
  baseline_snrs       list of float
Scenario subprocesses read this file; the parent orchestrator reads only
cumulative_scale from it for the completion sentinel.
"""

import argparse
import gc
import glob
import gzip
import heapq
import json
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
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
N_GLOBAL_CGW_CANDIDATES   = 12_500   # baseline: proxy pre-filter budget
N_SCENARIO_CGW_CANDIDATES = 1_000    # synthetic PTAs: top-N by BASELINE cgw_snr
N_RESCUE                  = 500      # per frequency regime
N_TOP_SOURCES             = 50
MAX_SCALE_ITER            = 20

HANDOFF_FILENAME          = 'phase_handoff.json'


# =============================================================================
# ShardedPickleStore (self-contained fallback if stage1_setup not importable)
# =============================================================================

if not _STAGE1_SETUP_IMPORTED:
    FIELD_DTYPES: Dict[str, type] = {
        "f":        np.float32,
        "Mc":       np.float32,
        "Mtot":     np.float32,
        "D_comov":  np.float64,
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
        """One pkl.gz per chunk stored in a single directory."""

        def __init__(self, directory: str, compress_level: int = 6):
            from pathlib import Path
            self.dir = Path(directory)
            self.dir.mkdir(parents=True, exist_ok=True)
            self.compress_level = compress_level

        def write(self, idx: int, pop) -> None:
            self._dump(self._path(idx), self._downcast(pop))

        def read(self, idx: int):
            with gzip.open(self._path(idx), "rb") as f:
                return pickle.load(f)

        def update(self, idx: int, h0=None, D_comov=None, z=None,
                   cgw_snr=None, cgw_proxy=None, amp_A=None, amp_B=None) -> None:
            pop = self.read(idx)
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
            self._dump(self._path(idx), pop)

        def available(self) -> list:
            return sorted(
                int(p.name.split("_")[1].split(".")[0])
                for p in self.dir.glob("subpop_*.pkl.gz")
            )

        def _path(self, idx: int):
            return self.dir / f"subpop_{idx:03d}.pkl.gz"

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
# TOA delta loading
# =============================================================================

def _delta_filename(chunk_id: int, scenario: Optional[str] = None) -> str:
    suffix = f'_{scenario}' if scenario else ''
    return f'chunk_{chunk_id:04d}{suffix}.npz'


def _load_and_sum_toa_deltas(
    sim_out_dir: str,
    chunk_ids:   List[int],
    psr_names:   List[str],
    scenario:    Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Sum per-chunk Δstoa .npz files for a given scenario across all chunks."""
    combined: Dict[str, Optional[np.ndarray]] = {n: None for n in psr_names}

    for chunk_id in chunk_ids:
        fname = _delta_filename(chunk_id, scenario)
        fpath = os.path.join(sim_out_dir, 'stoas', fname)
        if not os.path.isfile(fpath):
            sys.exit(f'ERROR: missing delta file: {fpath}')
        with np.load(fpath) as data:
            for name in psr_names:
                if name not in data:
                    sys.exit(f'ERROR: pulsar {name} missing from {fpath}')
                arr = data[name].astype(np.float64)
                combined[name] = arr if combined[name] is None else combined[name] + arr

    print(f'  Summed Δstoas across {len(chunk_ids)} chunks '
          f'[scenario={scenario or "baseline"}] for {len(combined)} pulsars')
    return combined  # type: ignore[return-value]


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
    for psr in psrs_clean:
        simulate_psr(psr, raw_noise_params, add_WN=True, add_RN=True, seed=seed)
    return {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}


# =============================================================================
# Enterprise PTA builder
# =============================================================================

def _build_pta(psrs_libstempo, raw_noise_params, Tspan):
    """
    Build an Enterprise PTA from libstempo pulsar objects.
    Returns (pta, enterprise_psrs).  Caller is responsible for deleting both.
    """
    enterprise_psrs = [
        EnterprisePulsar(psr, ephem='DE440', backend='tempo2')
        for psr in psrs_libstempo
    ]
    pta, _, _ = build_pta_and_params(
        psrs=enterprise_psrs,
        noise_params_15yr=raw_noise_params,
        Tspan=Tspan,
    )
    gc.collect()
    return pta, enterprise_psrs


# =============================================================================
# SNR scaling
# =============================================================================

def _scale_and_iterate(
    psrs_clean,
    delta_stoas:     Dict[str, np.ndarray],
    noise_stoas:     Dict[str, np.ndarray],
    target_snr:      float,
    snr_low:         float,
    snr_high:        float,
    Tspan_seconds:   float,
    raw_noise_params,
    max_iterations:  int = MAX_SCALE_ITER,
    curn_components: int = 14,
    rn_components:   int = 30,
) -> Tuple[float, object, list]:
    """
    Iteratively scale the combined GW TOA signal until the OS SNR falls in
    [snr_low, snr_high].  Uses analytic first step then log-midpoint bisection.

    Returns (cumulative_scale, pta, enterprise_psrs).
    delta_stoas is updated in-place to contain the final scaled signal.
    """
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
    print(f'  Noise-only OS SNR: {noise_only_snr:.4f}')
    # ── guard: target must be reachable above the noise floor ────────────────
    if target_snr <= noise_only_snr:
        raise ValueError(
            f'Target SNR ({target_snr:.4f}) is at or below the noise-only '
            f'SNR ({noise_only_snr:.4f}). No GW scaling can reach this target. '
            f'Increase target_snr or check your noise model.')
    if snr_high <= noise_only_snr:
        raise ValueError(
            f'SNR band ceiling ({snr_high:.4f}) is at or below the noise-only '
            f'SNR ({noise_only_snr:.4f}). Widen the target band upward.')
    
    def _analytic_factor(snr_cur, snr_tgt, snr_noise):
        sig_cur = snr_cur - snr_noise
        sig_tgt = snr_tgt - snr_noise
        if sig_cur <= 0 or sig_tgt <= 0:
            raise ValueError(
                f'Signal-only SNR non-positive '
                f'(cur={sig_cur:.4f}, tgt={sig_tgt:.4f})')
        return np.sqrt(sig_cur / sig_tgt)

    def _bisect_factor(history, cumulative_scale):
        above = [(h['cs'], h['snr']) for h in history if h['snr'] >= target_snr]
        below = [(h['cs'], h['snr']) for h in history if h['snr'] <  target_snr]
        if not (above and below):
            return None
        best_above = max(above, key=lambda p: p[0])
        best_below = min(below, key=lambda p: p[0])
        if abs(best_above[0] - best_below[0]) < 1e-10:
            return None
        cum_target  = np.exp(0.5 * (np.log(best_above[0]) + np.log(best_below[0])))
        incremental = cum_target / cumulative_scale
        print(f'  [bisection] below=({best_below[0]:.4f}×,{best_below[1]:.4f}) '
              f'above=({best_above[0]:.4f}×,{best_above[1]:.4f}) '
              f'→ mid={cum_target:.4f}×  incr={incremental:.4f}')
        return incremental

    cumulative_scale = 1.0
    history          = []
    original_delta   = {n: delta_stoas[n].copy() for n in delta_stoas}

    for iteration in range(max_iterations):
        scaled_delta = {n: original_delta[n] / cumulative_scale for n in original_delta}
        signal_stoas = {n: noise_stoas[n] + scaled_delta[n]     for n in noise_stoas}

        snr, pta, enterprise_psrs = compute_population_snr(
            psrs_clean=psrs_clean,
            population=None,
            raw_noise_params=raw_noise_params,
            Tspan=Tspan_seconds,
            current_stoas=signal_stoas,
            curn_components=curn_components,
            rn_components=rn_components,
        )
        print(f'  Iter {iteration+1:2d}: OS SNR={snr:.4f}  '
              f'target=[{snr_low},{snr_high}]  scale={cumulative_scale:.4f}×')

        if snr_low <= snr <= snr_high:
            print(f'  ✓ Converged at iteration {iteration+1}')
            for n in delta_stoas:
                delta_stoas[n] = scaled_delta[n]
            return cumulative_scale, pta, enterprise_psrs

        history.append({'cs': cumulative_scale, 'snr': snr})
        if len(history) == 1:
            factor = _analytic_factor(snr, target_snr, noise_only_snr)
            print(f'  [analytic] factor={factor:.4f}')
        else:
            factor = _bisect_factor(history, cumulative_scale)
            if factor is None:
                factor = _analytic_factor(snr, target_snr, noise_only_snr)
                print(f'  [analytic fallback] factor={factor:.4f}')
        cumulative_scale *= float(np.clip(factor, 0.1, 10.0))

    raise RuntimeError(
        f'SNR scaling failed in {max_iterations} iterations. '
        f'Last SNR={snr:.4f}, target=[{snr_low},{snr_high}].')


# =============================================================================
# CGW SNR infrastructure
# =============================================================================

def _build_candidate_list(
    store:         ShardedPickleStore,
    chunk_ids:     List[int],
    enterprise_psrs,
    Tspan_seconds: float,
) -> List[Tuple[int, int, float]]:
    """
    Pass 1 (baseline only): scan all shards using the analytic proxy to
    build a ranked global candidate list, then rescue low/mid-f regimes.

    Returns list of (chunk_id, local_idx, proxy_score), sorted descending.
    """
    print(f'  Pre-filtering {len(chunk_ids)} chunks via analytic proxy...')
    candidate_list = []

    for chunk_id in chunk_ids:
        pop    = store.read(chunk_id)
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
                (chunk_id, int(local_idx), float(proxies[local_idx])))
        del pop; gc.collect()

    candidate_list.sort(key=lambda x: x[2], reverse=True)
    global_candidates = candidate_list[:N_GLOBAL_CGW_CANDIDATES]

    # rescue low/mid-f proxy failure regimes
    existing = {(c, i) for c, i, _ in global_candidates}
    for chunk_id in chunk_ids:
        pop = store.read(chunk_id)
        f_T = np.asarray(pop.f, dtype=np.float64) * Tspan_seconds
        for mask in [f_T < 10.0, (f_T >= 10.0) & (f_T < 50.0)]:
            if mask.sum() == 0:
                continue
            regime_idx   = np.where(mask)[0]
            top_h0_local = regime_idx[np.argsort(pop.h0[regime_idx])[-N_RESCUE:]]
            for local_idx in top_h0_local:
                key = (chunk_id, int(local_idx))
                if key not in existing:
                    global_candidates.append(
                        (chunk_id, int(local_idx), float(pop.h0[local_idx])))
                    existing.add(key)
        del pop; gc.collect()

    n_rescued = len(global_candidates) - N_GLOBAL_CGW_CANDIDATES
    print(f'  {len(global_candidates)} total candidates '
          f'(top-{N_GLOBAL_CGW_CANDIDATES} proxy + {n_rescued} rescued from '
          f'low/mid-f regimes)')
    return global_candidates


def _assemble_binaries(
    store:      ShardedPickleStore,
    candidates: List[Tuple[int, int, float]],
) -> List:
    """Load binary objects for a candidate list, grouped by chunk to minimise reads."""
    by_chunk: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for global_rank, (chunk_id, local_idx, _) in enumerate(candidates):
        by_chunk[chunk_id].append((global_rank, local_idx))

    binaries = [None] * len(candidates)
    for chunk_id, entries in by_chunk.items():
        pop = store.read(chunk_id)
        for global_rank, local_idx in entries:
            binaries[global_rank] = pop[local_idx]
        del pop; gc.collect()
    return binaries


def _write_snrs_to_shards(
    store:      ShardedPickleStore,
    candidates: List[Tuple[int, int, float]],
    snrs:       np.ndarray,
    chunk_ids:  List[int],
    snr_field:  str,
) -> None:
    """Write a per-binary SNR array into each shard under snr_field."""
    chunk_snr_maps: Dict[int, Dict[int, float]] = defaultdict(dict)
    for global_rank, (chunk_id, local_idx, _) in enumerate(candidates):
        chunk_snr_maps[chunk_id][local_idx] = float(snrs[global_rank])

    for chunk_id in chunk_ids:
        pop         = store.read(chunk_id)
        cgw_snr_arr = np.zeros(len(pop), dtype=np.float32)
        for local_idx, snr_val in chunk_snr_maps.get(chunk_id, {}).items():
            cgw_snr_arr[local_idx] = snr_val
        setattr(pop, snr_field, cgw_snr_arr)
        store._dump(store._path(chunk_id), pop)
        del pop; gc.collect()


def _print_top_sources(
    store:      ShardedPickleStore,
    candidates: List[Tuple[int, int, float]],
    snrs:       np.ndarray,
    snr_field:  str,
    n_show:     int = N_TOP_SOURCES,
) -> None:
    ranked = sorted(zip(candidates, snrs), key=lambda x: x[1], reverse=True)
    n_show = min(n_show, len(ranked))
    by_chunk: Dict[int, object] = {}
    for (chunk_id, _, _), _ in ranked[:n_show]:
        if chunk_id not in by_chunk:
            by_chunk[chunk_id] = store.read(chunk_id)
    proxy_rank_map = {
        (chunk_id, local_idx): rank
        for rank, (chunk_id, local_idx, _) in enumerate(candidates, start=1)
    }
    print(f'\n  Top {n_show} [{snr_field}]:')
    for rank, ((chunk_id, local_idx, proxy), snr_val) in enumerate(
            ranked[:n_show], start=1):
        pop        = by_chunk[chunk_id]
        proxy_rank = proxy_rank_map.get((chunk_id, local_idx), -1)
        print(f'    {rank:2d}.  chunk={chunk_id:03d}  local_idx={local_idx:7d}  '
              f'f={pop.f[local_idx]:.2e} Hz  '
              f'Mc={pop.Mc[local_idx]:.2e} Msun  '
              f'h0={pop.h0[local_idx]:.2e}  '
              f'proxy={proxy:.4e} (rank #{proxy_rank})  '
              f'{snr_field}={snr_val:.4f}')
    del by_chunk; gc.collect()


def _compute_cgw_snrs_baseline(
    store:               ShardedPickleStore,
    chunk_ids:           List[int],
    pta,
    enterprise_psrs,
    raw_noise_params:    dict,
    parsed_noise_params: dict,
    Tspan_seconds:       float,
    meta_dir:            str,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Full two-pass CGW SNR for the baseline PTA.

    Pass 1: analytic proxy pre-filter + regime rescue → global_candidates.
    Pass 2: full Enterprise inner-product SNR on all candidates.

    Writes top-5 breakdown records to metadata/top_cgw_breakdowns.json.
    Returns (global_candidates, top_snrs) for reuse by synthetic scenarios.
    """
    print('\n  [baseline] Building candidate list...')
    global_candidates = _build_candidate_list(
        store, chunk_ids, enterprise_psrs, Tspan_seconds)

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

    _write_snrs_to_shards(store, global_candidates, top_snrs, chunk_ids, 'cgw_snr')
    _print_top_sources(store, global_candidates, np.asarray(top_snrs), 'cgw_snr')

    # Persist top-5 per-pulsar breakdowns for the summary builder
    ranked = sorted(zip(global_candidates, top_snrs), key=lambda x: x[1], reverse=True)
    proxy_rank_map = {
        (chunk_id, local_idx): rank
        for rank, (chunk_id, local_idx, _) in enumerate(global_candidates, start=1)
    }
    candidate_breakdown_map = {
        (chunk_id, local_idx): top_breakdowns[i]
        for i, (chunk_id, local_idx, _) in enumerate(global_candidates)
    }
    top_breakdown_records = []
    for rank, ((chunk_id, local_idx, proxy), snr_val) in enumerate(ranked[:5], start=1):
        proxy_rank = proxy_rank_map[(chunk_id, local_idx)]
        per_pulsar = candidate_breakdown_map[(chunk_id, local_idx)]
        top_breakdown_records.append({
            'rank':              rank,
            'proxy_rank':        proxy_rank,
            'chunk_id':          int(chunk_id),
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
    chunk_ids:            List[int],
    pta,
    enterprise_psrs,
    raw_noise_params:     dict,
    parsed_noise_params:  dict,
    Tspan_seconds:        float,
    snr_field:            str,
    baseline_candidates:  List[Tuple[int, int, float]],
    baseline_snrs:        np.ndarray,
    n_candidates:         int = N_SCENARIO_CGW_CANDIDATES,
) -> None:
    """
    Lightweight CGW SNR pass for a synthetic PTA scenario.

    Picks the top `n_candidates` by BASELINE cgw_snr (actual inner-product,
    not the proxy) — no expensive proxy scan required.  If a source is loud
    in the baseline PTA it is a plausible candidate for an improved PTA.
    """
    ranked_by_snr = sorted(
        zip(baseline_candidates, baseline_snrs),
        key=lambda x: x[1], reverse=True,
    )
    candidates_for_scenario = [c for c, _ in ranked_by_snr[:n_candidates]]

    print(f'\n  [{snr_field}] Using top-{len(candidates_for_scenario)} '
          f'by baseline CGW SNR (no proxy re-scan needed)')

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

    _write_snrs_to_shards(store, candidates_for_scenario, top_snrs, chunk_ids, snr_field)
    _print_top_sources(store, candidates_for_scenario, np.asarray(top_snrs), snr_field)


# =============================================================================
# Scaling helpers
# =============================================================================

def _comov_redshift_from_scaling(
    D_comov: np.ndarray,
    z:       np.ndarray,
    scale:   float,
) -> Tuple[np.ndarray, np.ndarray]:
    from SMBHB_pop_synth import _Z_GRID, _CHI_GRID
    valid      = _CHI_GRID > 0
    z_grid     = _Z_GRID[valid];  chi_grid = _CHI_GRID[valid]
    frac_grid  = (1.0 + z_grid) ** (2.0 / 3.0) / chi_grid
    targets    = (1.0 + z) ** (2.0 / 3.0) / D_comov / scale
    if frac_grid[0] > frac_grid[-1]:
        frac_grid = frac_grid[::-1]
        chi_grid  = chi_grid[::-1]
        z_grid    = z_grid[::-1]
    raw = np.clip(np.searchsorted(frac_grid, targets, side='left'),
                  1, len(frac_grid) - 1)
    lo  = raw - 1
    idx = np.where(np.abs(frac_grid[raw] - targets) <=
                   np.abs(frac_grid[lo]  - targets), raw, lo)
    return chi_grid[idx].astype(np.float32), z_grid[idx].astype(np.float32)


def _apply_scaling_to_all_chunks(
    store:            ShardedPickleStore,
    chunk_ids:        List[int],
    cumulative_scale: float,
) -> None:
    print(f'\n  Applying scale 1/{cumulative_scale:.6f} to {len(chunk_ids)} chunks...')
    for chunk_id in chunk_ids:
        pop          = store.read(chunk_id)
        new_h0       = pop.h0 / cumulative_scale
        new_D, new_z = _comov_redshift_from_scaling(
            pop.D_comov, pop.z, 1.0 / cumulative_scale)
        store.update(chunk_id, h0=new_h0, D_comov=new_D, z=new_z)
        del pop, new_h0, new_D, new_z; gc.collect()
        print(f'    chunk {chunk_id:03d}: updated', flush=True)


def _cleanup_chunk_stoas(
    sim_out_dir: str,
    chunk_ids:   List[int],
    scenario:    Optional[str] = None,
) -> None:
    stoa_dir = os.path.join(sim_out_dir, 'stoas')
    removed, freed_bytes = 0, 0
    for chunk_id in chunk_ids:
        fpath = os.path.join(stoa_dir, _delta_filename(chunk_id, scenario))
        if os.path.isfile(fpath):
            freed_bytes += os.path.getsize(fpath)
            os.remove(fpath)
            removed += 1
    try:
        os.rmdir(stoa_dir)
        dir_note = '  — stoas/ dir removed'
    except OSError:
        dir_note = ''
    print(f'  🗑️  Cleaned {removed} chunk files '
          f'[scenario={scenario or "baseline"}]  '
          f'({freed_bytes / 1e6:.1f} MB freed){dir_note}')


def _save_toa_residuals(
    sim_out_dir:      str,
    psrs_clean,
    noise_stoas:      Dict[str, np.ndarray],
    combined_delta:   Dict[str, np.ndarray],
    cumulative_scale: float,
    scenario:         Optional[str] = None,
) -> None:
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
        pop_a   = combined_delta[name] / cumulative_scale
        comb_a  = noise_a + pop_a
        np.save(os.path.join(dirs['noise'],      f'{name}.npy'), noise_a.astype(np.float64))
        np.save(os.path.join(dirs['population'], f'{name}.npy'), pop_a.astype(np.float64))
        np.save(os.path.join(dirs['combined'],   f'{name}.npy'), comb_a.astype(np.float64))

    manifest = {
        'cumulative_scale': cumulative_scale,
        'n_pulsars':        len(psrs_clean),
        'psr_names':        [p.name for p in psrs_clean],
        'scenario':         scenario or 'baseline',
        'description': {
            'noise':      'Simulated noise residuals only (no GW signal)',
            'population': 'GW TOA contribution from full population, scaled to target SNR',
            'combined':   'noise + population (what an observer would measure)',
        },
    }
    with open(os.path.join(base, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'  ✓ Saved residuals for {len(psrs_clean)} pulsars '
          f'(noise / population / combined)')


# =============================================================================
# Subprocess phase runner
# =============================================================================

def _run_phase(extra_argv: list, label: str) -> None:
    """Launch this same script as a subprocess with extra_argv appended."""
    cmd = [sys.executable, os.path.abspath(__file__)] + extra_argv
    print(f'\n  ▶ Subprocess: {label}', flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f'Subprocess phase "{label}" exited with code {result.returncode}')
    print(f'  ✓ Subprocess complete: {label}', flush=True)


# =============================================================================
# Phase entry points — called inside subprocesses
# =============================================================================

def _phase_baseline(args, syn_scenarios, combined_scenarios, noise_seed):
    """
    Baseline phase (subprocess):
      - load + filter baseline pulsars
      - simulate noise; retry with shifted seed if noise-only SNR too high
      - scale GW signal to target OS SNR
      - save residuals; update shards with scale factor
      - optionally validate proxy
      - compute baseline CGW SNR (proxy-only mode: skip scaling, just validate)
      - write phase_handoff.json (including winning noise_seed) for scenario subprocesses
    """
    sim_out_dir = os.path.join(args.output_dir, f'sim{args.sim_id:03d}')
    meta_dir    = os.path.join(sim_out_dir, 'metadata')

    with open(os.path.join(meta_dir, 'config.json')) as fh:
        run_config = json.load(fh)
    Tspan_seconds = run_config['Tspan_seconds']

    pop_dir   = os.path.join(sim_out_dir, 'populations')
    store     = ShardedPickleStore(pop_dir)
    chunk_ids = store.available()
    if not chunk_ids:
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
        print(f'\n{"="*60}')
        print('Proxy-only mode — skipping noise simulation and SNR scaling.')
        print(f'{"="*60}')

        print('\n🔊 Simulating noise (for PTA structure only)...')
        noise_stoas = _simulate_noise(psrs_clean, raw_noise_params, seed=noise_seed)

        _, pta, enterprise_psrs = compute_population_snr(
            psrs_clean=psrs_clean,
            population=None,
            raw_noise_params=raw_noise_params,
            Tspan=Tspan_seconds,
            current_stoas=noise_stoas,
            return_psrs_pta=True,
        )
        validate_cgw_proxy(
            store=store, chunk_ids=chunk_ids, pta=pta,
            enterprise_psrs=enterprise_psrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds, n_test=args.n_test,
        )
        validate_proxy_filtering_ratio(
            store=store, chunk_ids=chunk_ids, pta=pta,
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
            json.dump({'cumulative_scale': 1.0, 'Tspan_seconds': Tspan_seconds,
                       'noise_seed': noise_seed,
                       'baseline_candidates': [], 'baseline_snrs': []}, fh)
        return

    # ── full pipeline ─────────────────────────────────────────────────────────
    print('\n📂 [baseline] Summing Δstoas...')
    combined_delta = _load_and_sum_toa_deltas(
        sim_out_dir, chunk_ids, psr_names, scenario=None)

    # ── noise simulation with retry on high noise-only SNR ───────────────────
    max_noise_retries = 10
    cumulative_scale  = None
    pta               = None
    epsrs             = None
    winning_seed      = noise_seed

    for noise_attempt in range(max_noise_retries):
        attempt_seed = noise_seed + noise_attempt
        if noise_attempt == 0:
            print(f'\n🔊 [baseline] Simulating noise (seed={attempt_seed})...')
        else:
            print(f'\n🔁 [baseline] Retrying noise simulation '
                  f'(attempt {noise_attempt + 1}/{max_noise_retries}, '
                  f'seed={attempt_seed})...')
        noise_stoas = _simulate_noise(psrs_clean, raw_noise_params, seed=attempt_seed)

        print('\n📐 [baseline] Scaling GW signal...')
        try:
            cumulative_scale, pta, epsrs = _scale_and_iterate(
                psrs_clean=psrs_clean,
                delta_stoas=combined_delta,
                noise_stoas=noise_stoas,
                target_snr=args.target_snr,
                snr_low=snr_low, snr_high=snr_high,
                Tspan_seconds=Tspan_seconds,
                raw_noise_params=raw_noise_params,
            )
            winning_seed = attempt_seed
            print(f'✓ Converged — scale={cumulative_scale:.6f}  '
                  f'noise_seed={winning_seed}')
            break
        except ValueError as e:
            if 'noise-only SNR' in str(e) or 'SNR band ceiling' in str(e):
                print(f'  ⚠️  {e}')
                if noise_attempt == max_noise_retries - 1:
                    sys.exit(f'ERROR: noise-only SNR too high after '
                             f'{max_noise_retries} attempts — giving up')
                continue
            sys.exit(f'ERROR: {e}')
        except RuntimeError as e:
            sys.exit(f'ERROR: {e}')

    _save_toa_residuals(sim_out_dir, psrs_clean, noise_stoas,
                        combined_delta, cumulative_scale, scenario=None)
    _cleanup_chunk_stoas(sim_out_dir, chunk_ids, scenario=None)
    del noise_stoas, combined_delta, psrs_clean; gc.collect()

    print('\n💾 [baseline] Updating shards...')
    _apply_scaling_to_all_chunks(store, chunk_ids, cumulative_scale)

    baseline_candidates: Optional[List] = None
    baseline_snrs:       Optional[np.ndarray] = None

    if args.cgw:
        if args.validate_proxy:
            print('\n🔍 Validating CGW proxy...')
            validate_cgw_proxy(
                store=store, chunk_ids=chunk_ids, pta=pta,
                enterprise_psrs=epsrs,
                raw_noise_params=raw_noise_params,
                parsed_noise_params=parsed_noise_params,
                Tspan_seconds=Tspan_seconds, n_test=args.n_test,
            )

        print('\n🔭 [baseline] Computing CGW SNRs...')
        baseline_candidates, baseline_snrs = _compute_cgw_snrs_baseline(
            store=store, chunk_ids=chunk_ids,
            pta=pta, enterprise_psrs=epsrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds,
            meta_dir=meta_dir,
        )
        print('✓ Baseline CGW complete')
    else:
        print('\n(CGW skipped — use --cgw to enable)')

    del pta, epsrs; gc.collect()

    handoff = {
        'cumulative_scale':    float(cumulative_scale),
        'Tspan_seconds':       Tspan_seconds,
        'noise_seed':          winning_seed,
        'baseline_candidates': [[c, i, float(p)]
                                for c, i, p in (baseline_candidates or [])],
        'baseline_snrs':       (baseline_snrs.tolist()
                                if baseline_snrs is not None else []),
    }
    with open(os.path.join(meta_dir, HANDOFF_FILENAME), 'w') as fh:
        json.dump(handoff, fh)
    print(f'✓ [baseline] Wrote {HANDOFF_FILENAME} (noise_seed={winning_seed})')


def _phase_scenario(args, scenario_label, combined_scenarios):
    """
    Scenario phase (subprocess, one per scenario):
      - read handoff (cumulative_scale, baseline candidate ranking, noise_seed)
      - load + filter scenario pulsars
      - simulate noise with same seed as baseline (from handoff)
      - save residuals; clean chunk files
      - build Enterprise PTA
      - compute CGW SNR for top-N by baseline SNR; write to shards
    """
    sim_out_dir = os.path.join(args.output_dir, f'sim{args.sim_id:03d}')
    meta_dir    = os.path.join(sim_out_dir, 'metadata')

    handoff_path = os.path.join(meta_dir, HANDOFF_FILENAME)
    if not os.path.isfile(handoff_path):
        sys.exit(f'ERROR: handoff file not found: {handoff_path}')
    with open(handoff_path) as fh:
        handoff = json.load(fh)

    cumulative_scale    = handoff['cumulative_scale']
    Tspan_seconds       = handoff['Tspan_seconds']
    noise_seed          = handoff.get('noise_seed', args.noise_seed_base + args.sim_id**2 * 100
                                  if args.noise_seed_base is not None
                                  else args.sim_id * 1000)
    baseline_candidates = [(c, i, p) for c, i, p in handoff['baseline_candidates']]
    baseline_snrs       = np.array(handoff['baseline_snrs'], dtype=np.float64)

    pop_dir   = os.path.join(sim_out_dir, 'populations')
    store     = ShardedPickleStore(pop_dir)
    chunk_ids = store.available()

    parsed_noise_params = parse_pulsar_parameters(config.NOISEFILE)

    print(f'\n📡 [{scenario_label}] Loading pulsars...')
    psrs_unfiltered = load_pulsars(verbose=True, scenario=scenario_label,
                                   scenarios=combined_scenarios)
    with suppress_enterprise_warnings():
        psrs_clean, raw_noise_params, _ = filter_pulsars_15yr(
            psrs_unfiltered, verbose=False)
    del psrs_unfiltered; gc.collect()
    psr_names = [p.name for p in psrs_clean]

    print(f'\n📂 [{scenario_label}] Summing Δstoas...')
    combined_delta = _load_and_sum_toa_deltas(
        sim_out_dir, chunk_ids, psr_names, scenario=scenario_label)

    print(f'\n🔊 [{scenario_label}] Simulating noise (seed={noise_seed})...')
    noise_stoas = _simulate_noise(psrs_clean, raw_noise_params, seed=noise_seed)

    _save_toa_residuals(sim_out_dir, psrs_clean, noise_stoas,
                        combined_delta, cumulative_scale, scenario=scenario_label)
    _cleanup_chunk_stoas(sim_out_dir, chunk_ids, scenario=scenario_label)
    del noise_stoas, combined_delta; gc.collect()

    print(f'\n🔧 [{scenario_label}] Building Enterprise PTA...')
    pta, epsrs = _build_pta(psrs_clean, raw_noise_params, Tspan_seconds)
    del psrs_clean; gc.collect()

    if args.cgw and len(baseline_candidates) > 0:
        _compute_cgw_snrs_scenario(
            store=store, chunk_ids=chunk_ids,
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
# Per-simulation orchestrator
# =============================================================================

def process_sim(
    sim_id:            int,
    args,
    syn_scenarios:     dict,
    combined_scenarios: dict,
    noise_seed:        int,
) -> bool:
    """
    Orchestrate the per-simulation pipeline by launching each phase in a
    clean subprocess.  The parent process holds only the argument namespace;
    all pulsars and PTA objects live exclusively inside subprocesses.

    Each subprocess exits completely before the next starts, so libstempo/
    tempo2's C-level global state is guaranteed to be cleared between phases.
    """
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
            '--noise-seed-base', str(args.noise_seed_base),  # ← was --noise-seed
            '--n-test',          str(args.n_test),
        ]
        if args.cgw:              argv.append('--cgw')
        if args.validate_proxy:   argv.append('--validate-proxy')
        if args.proxy_only:       argv.append('--proxy-only')
        if args.synthetic_ptas:   argv.append('--synthetic-ptas')
        if args.synthetic_pta_config:
            argv += ['--synthetic-pta-config', args.synthetic_pta_config]
        return argv

    # Phase 1: baseline
    try:
        _run_phase(_common_argv() + ['--phase', 'baseline'],
                   label=f'sim{sim_id:03d}/baseline')
    except RuntimeError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return False

    # Phase 2: one subprocess per synthetic scenario
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

    # Write completion sentinel (read cumulative_scale from handoff)
    handoff_path = os.path.join(sim_out_dir, 'metadata', HANDOFF_FILENAME)
    cumulative_scale = 1.0
    if os.path.isfile(handoff_path):
        with open(handoff_path) as fh:
            cumulative_scale = json.load(fh).get('cumulative_scale', 1.0)

    sentinel = os.path.join(sim_out_dir, 'metadata', 'stage2_complete.json')
    with open(sentinel, 'w') as fh:
        json.dump({
            'sim_id':           sim_id,
            'cumulative_scale': float(cumulative_scale),
            'scenarios':        ['baseline'] + list(syn_scenarios.keys()),
            'n_cgw_baseline':   N_GLOBAL_CGW_CANDIDATES,
            'n_cgw_scenario':   N_SCENARIO_CGW_CANDIDATES,
            'completed_at':     time.strftime('%Y-%m-%dT%H:%M:%S'),
        }, fh, indent=2)

    elapsed = time.time() - t_sim
    print(f'\n✅ Stage 2 sim_id={sim_id} complete in {elapsed/60:.1f} min')
    return True


# =============================================================================
# Per-simulation summary builder
# =============================================================================

def _build_summary_object(
    output_dir:          str,
    sim_id:              int,
    n_keep_per_category: int = 200,
) -> None:
    """
    Build a compact summary object covering all population shards for a sim.

    Discovers all SNR fields (any attribute starting with 'cgw_snr') from the
    first shard automatically.  Category indices are built with heapq so the
    full population array never needs to be held in memory at once.

    Writes summary.pkl.gz:
        d['arrays']  — dict of np.ndarray, all same length (one entry per binary)
        d['meta']    — category_indices, top_cgw_breakdowns, snr_fields, etc.
    """
    sim_out_dir = os.path.join(output_dir, f'sim{sim_id:03d}')
    out_path    = os.path.join(sim_out_dir, 'summary.pkl.gz')
    pop_dir     = os.path.join(sim_out_dir, 'populations')
    store       = ShardedPickleStore(pop_dir)
    chunk_ids   = store.available()

    print(f'\n{"="*60}')
    print(f'Building summary — sim{sim_id:03d} → {out_path}')
    print(f'  n_keep_per_category = {n_keep_per_category}')
    print(f'{"="*60}')

    # Discover field layout from first shard
    first_pop  = store.read(chunk_ids[0])
    snr_fields = [attr for attr in vars(first_pop) if attr.startswith('cgw_snr')]
    all_fields = [attr for attr in vars(first_pop)
                  if isinstance(getattr(first_pop, attr), np.ndarray)]
    print(f'  SNR fields  : {snr_fields}')
    print(f'  All fields  : {all_fields}')
    del first_pop

    # Heap-based category tracking (never loads whole population into RAM)
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

    for chunk_id in chunk_ids:
        pop = store.read(chunk_id)
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

    # Top-5 CGW breakdowns written by _compute_cgw_snrs_baseline
    top_cgw_breakdowns = []
    breakdown_path = os.path.join(sim_out_dir, 'metadata', 'top_cgw_breakdowns.json')
    if os.path.isfile(breakdown_path):
        with open(breakdown_path) as fh:
            top_cgw_breakdowns = json.load(fh)

    payload = {
        'arrays': arrays,
        'meta': {
            'sim_id':              sim_id,
            'total_scanned':       total_scanned,
            'n_keep_per_category': n_keep_per_category,
            'snr_fields':          snr_fields,
            'category_indices':    category_indices,
            'top_cgw_breakdowns':  top_cgw_breakdowns,
            'summary_version':     3,
            'full_population':     True,
        },
    }
    with gzip.open(out_path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=4)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'✅ Summary written: {out_path}  ({size_mb:.1f} MB)')
    print(f'   SNR fields: {snr_fields}')
    print(f'   Load with:')
    print(f'     import gzip, pickle')
    print(f"     with gzip.open('{out_path}', 'rb') as f:")
    print(f'         d = pickle.load(f)')
    print(f"     arrays = d['arrays']")
    print(f"     meta   = d['meta']")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Stage 2: SNR scaling + CGW')
    p.add_argument('--output-dir',           type=str, required=True)
    p.add_argument('--config', '-c',         default='optimistic',
                   choices=list(config.POPULATION_CONFIGS.keys()))
    p.add_argument('--target-snr',           type=float, default=4.0)
    p.add_argument('--snr-range',            nargs=2, type=float, default=[3.5, 4.25])
    p.add_argument('--n-chunks',             type=int, required=True)
    p.add_argument('--sim-id',               type=int, required=True)
    p.add_argument('--cgw',                  action='store_true')
    p.add_argument('--validate-proxy',       action='store_true')
    p.add_argument('--proxy-only',           action='store_true')
    p.add_argument('--n-test',               type=int, default=1_000)
    p.add_argument('--clean-failed',         action='store_true')
    p.add_argument('--noise-seed-base',           type=int, default=None)
    p.add_argument('--n-sims', type=int, default=1, help='Total number of simulations (for seeding purposes)')
    p.add_argument('--synthetic-ptas',       action='store_true', default=False)
    p.add_argument('--synthetic-pta-config', type=str, default=None)
    # Internal subprocess dispatch flags — not used by sbatch scripts directly
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

    root_sq = np.random.SeedSequence(args.noise_seed_base)
    sim_seq = root_sq.spawn(10000)[args.sim_id]

    rng = np.random.default_rng(sim_seq)
    seed_int = rng.integers(2**31).item()

    # Resolve synthetic scenarios
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

    noise_seed = (args.noise_seed_base + args.sim_id**2 * 100) if args.noise_seed_base is not None else args.sim_id * 1000

    # ── Subprocess phase dispatch ─────────────────────────────────────────────
    # These branches are only entered when this script is called by _run_phase.
    if args.phase == 'baseline':
        _phase_baseline(args, syn_scenarios, combined_scenarios, noise_seed)
        sys.exit(0)

    if args.phase == 'scenario':
        if not args.phase_scenario:
            sys.exit('ERROR: --phase scenario requires --phase-scenario <label>')
        _phase_scenario(args, args.phase_scenario, combined_scenarios)  # ← indented
        sys.exit(0)                                                      # ← indented

    # ── Top-level orchestrator ────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Stage 2 — sim_id={args.sim_id}')
    print(f'  Output dir    : {args.output_dir}')
    print(f'  Config        : {args.config}')
    print(f'  Target SNR    : {args.target_snr}  range={args.snr_range}')
    print(f'  Chunks/sim    : {args.n_chunks}')
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