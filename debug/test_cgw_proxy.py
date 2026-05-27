#!/usr/bin/env python3
"""
debug/test_cgw_proxy.py

Validation of the analytic CGW proxy against true SNRs.
Can be called directly from stage2_inject.py via --proxy-only or --validate-proxy,
or run standalone:
    python debug/test_cgw_proxy.py --output-dir /path/to/output --sim-id 0 --n-test 500
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import gc
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

import config
from data_loader import load_pulsars, filter_pulsars_15yr, parse_pulsar_parameters
from signal_injection import simulate_psr
from stage1_setup import ShardedPickleStore, _compute_analytic_proxy
from CGW_SNR import compute_cgw_snr_optimal_population_fast
from consistent_pop_synth import compute_population_snr, suppress_enterprise_warnings


def validate_cgw_proxy(
    store: ShardedPickleStore,
    chunk_ids: List[int],
    pta,
    enterprise_psrs,
    raw_noise_params,
    parsed_noise_params,
    Tspan_seconds: float,
    n_test: int = 500,
    seed: int = 42,
) -> None:
    """
    Validate the analytic CGW proxy against true SNRs on a random sample.

    Reports:
      - Spearman rank correlation
      - Recall @ top-K
      - Worst misses (high true SNR, low proxy rank)
      - Proxy overestimates (high proxy rank, low true SNR)
      - SNR and proxy score distribution summaries
    """
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"CGW Proxy Validation  (n_test={n_test})")
    print(f"{'='*60}")

    # ── Precompute noise matrices ─────────────────────────────────────────────
    print("  Precomputing PTA noise matrices...")
    phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
    TNTs    = pta.get_TNT(raw_noise_params)
    Ts      = pta.get_basis()
    Nvecs   = pta.get_ndiag(raw_noise_params)
    psr_map = {psr.name: psr for psr in enterprise_psrs}
    Sigmas  = [
        TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        for TNT, phiinv in zip(TNTs, phiinvs)
    ]

    # ── Count binaries per chunk ──────────────────────────────────────────────
    chunk_sizes = {}
    for chunk_id in chunk_ids:
        pop = store.read(chunk_id)
        chunk_sizes[chunk_id] = len(pop)
        del pop
        gc.collect()

    total  = sum(chunk_sizes.values())
    n_test = min(n_test, total)
    print(f"  Total binaries across {len(chunk_ids)} chunks: {total:,}")
    print(f"  Sampling {n_test} uniformly at random...")

    # ── Map flat random indices → (chunk_id, local_idx) ──────────────────────
    flat_indices = sorted(rng.choice(total, size=n_test, replace=False).tolist())
    sample_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
    cumulative = 0
    flat_ptr   = 0
    for chunk_id in chunk_ids:
        n         = chunk_sizes[chunk_id]
        chunk_end = cumulative + n
        while flat_ptr < len(flat_indices) and flat_indices[flat_ptr] < chunk_end:
            local_idx = flat_indices[flat_ptr] - cumulative
            sample_map[chunk_id].append((flat_ptr, local_idx))
            flat_ptr += 1
        cumulative = chunk_end
        if flat_ptr >= len(flat_indices):
            break

    # ── Collect sampled binaries and proxy scores ─────────────────────────────
    test_binaries  = [None] * n_test
    proxy_scores   = np.zeros(n_test)
    test_f         = np.zeros(n_test)
    test_h0        = np.zeros(n_test)
    test_chunk     = np.zeros(n_test, dtype=int)
    test_local_idx = np.zeros(n_test, dtype=int)

    for chunk_id, entries in sample_map.items():
        pop = store.read(chunk_id)

        if hasattr(pop, 'cgw_proxy') and pop.cgw_proxy is not None:
            chunk_proxies = pop.cgw_proxy.astype(np.float64)
        else:
            chunk_proxies = _compute_analytic_proxy(pop, enterprise_psrs,
                                                    Tspan_seconds=Tspan_seconds)

        for sample_pos, local_idx in entries:
            test_binaries[sample_pos]  = pop[local_idx]
            proxy_scores[sample_pos]   = chunk_proxies[local_idx]
            test_f[sample_pos]         = float(pop.f[local_idx])
            test_h0[sample_pos]        = float(pop.h0[local_idx])
            test_chunk[sample_pos]     = chunk_id
            test_local_idx[sample_pos] = local_idx

        del pop
        gc.collect()

    # ── Compute true SNRs ─────────────────────────────────────────────────────
    print(f"  Computing true SNR for {n_test} binaries...")
    t0 = time.time()
    true_snrs = np.array(compute_cgw_snr_optimal_population_fast(
        psrs=enterprise_psrs,
        pta=pta,
        population=test_binaries,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan=Tspan_seconds,
        profile=False,
    ))
    print(f"  True SNR computation took {time.time() - t0:.1f}s")

    # ── Rank correlation ──────────────────────────────────────────────────────
    rho, pval   = spearmanr(proxy_scores, true_snrs)
    proxy_ranks = n_test - np.argsort(np.argsort(proxy_scores))   # rank 1 = best
    true_ranks  = n_test - np.argsort(np.argsort(true_snrs))

    print(f"\n  {'─'*50}")
    print(f"  Spearman rank correlation : {rho:+.4f}  (p={pval:.2e})")
    print(f"  {'─'*50}")

    # ── Recall @ top-K ────────────────────────────────────────────────────────
    print(f"\n  Recall @ top-K  (proxy top-K ∩ true top-K) / K:")
    for k in [10, 25, 50, 100, 200]:
        if k > n_test:
            continue
        true_top_k  = set(np.argsort(true_snrs)[-k:])
        proxy_top_k = set(np.argsort(proxy_scores)[-k:])
        recall      = len(true_top_k & proxy_top_k) / k
        missed      = k - len(true_top_k & proxy_top_k)
        print(f"    top-{k:4d}:  recall={recall:.1%}  ({missed} missed)")

    # ── Worst misses: high true SNR, low proxy rank ───────────────────────────
    top10_true     = true_ranks  <= max(1, n_test // 10)
    bottom50_proxy = proxy_ranks >  n_test // 2
    miss_mask      = top10_true & bottom50_proxy
    n_misses       = miss_mask.sum()

    print(f"\n  Worst misses (true top-10%, proxy bottom-50%): {n_misses}")
    if n_misses > 0:
        miss_idx = np.where(miss_mask)[0]
        miss_idx = miss_idx[np.argsort(true_snrs[miss_idx])[::-1]]
        print(f"  {'pos':>4}  {'true_snr':>10}  {'proxy':>12}  {'true_rank':>10}  "
              f"{'proxy_rank':>11}  {'f_Hz':>10}  {'h0':>10}  chunk  local_idx")
        for i in miss_idx[:10]:
            print(f"  {i:4d}  {true_snrs[i]:10.4f}  {proxy_scores[i]:12.4e}  "
                  f"{true_ranks[i]:10d}  {proxy_ranks[i]:11d}  "
                  f"{test_f[i]:10.2e}  {test_h0[i]:10.2e}  "
                  f"{test_chunk[i]:5d}  {test_local_idx[i]:9d}")

    # ── Proxy overestimates: high proxy rank, low true SNR ────────────────────
    top10_proxy   = proxy_ranks <= max(1, n_test // 10)
    bottom50_true = true_ranks  >  n_test // 2
    over_mask     = top10_proxy & bottom50_true
    n_over        = over_mask.sum()

    print(f"\n  Proxy overestimates (proxy top-10%, true bottom-50%): {n_over}")
    if n_over > 0:
        over_idx = np.where(over_mask)[0]
        over_idx = over_idx[np.argsort(proxy_scores[over_idx])[::-1]]
        print(f"  {'pos':>4}  {'true_snr':>10}  {'proxy':>12}  {'true_rank':>10}  "
              f"{'proxy_rank':>11}  {'f_Hz':>10}  {'h0':>10}  chunk  local_idx")
        for i in over_idx[:10]:
            print(f"  {i:4d}  {true_snrs[i]:10.4f}  {proxy_scores[i]:12.4e}  "
                  f"{true_ranks[i]:10d}  {proxy_ranks[i]:11d}  "
                  f"{test_f[i]:10.2e}  {test_h0[i]:10.2e}  "
                  f"{test_chunk[i]:5d}  {test_local_idx[i]:9d}")

    # ── Frequency analysis of top-50 misses ───────────────────────────────────
    k = 50
    if n_test >= k:
        true_top_k  = set(np.argsort(true_snrs)[-k:])
        proxy_top_k = set(np.argsort(proxy_scores)[-k:])
        missed_idx  = np.array(list(true_top_k - proxy_top_k))
        caught_idx  = np.array(list(true_top_k & proxy_top_k))

        if len(missed_idx) > 0:
            print(f"\n  Frequency analysis of top-{k} misses:")
            print(f"  {'':>6}  {'true_snr':>10}  {'proxy':>12}  {'f_Hz':>10}  "
                  f"{'h0':>10}  {'proxy_rank':>11}")
            for i in missed_idx[np.argsort(true_snrs[missed_idx])[::-1]]:
                print(f"  {'MISS':>6}  {true_snrs[i]:10.4f}  {proxy_scores[i]:12.4e}  "
                      f"{test_f[i]:10.2e}  {test_h0[i]:10.2e}  {proxy_ranks[i]:11d}")
            for i in caught_idx[np.argsort(true_snrs[caught_idx])[::-1]][:5]:
                print(f"  {'CAUGHT':>6}  {true_snrs[i]:10.4f}  {proxy_scores[i]:12.4e}  "
                      f"{test_f[i]:10.2e}  {test_h0[i]:10.2e}  {proxy_ranks[i]:11d}")

            if len(missed_idx) > 0:
                print(f"\n  Missed  — f: median={np.median(test_f[missed_idx]):.2e}  "
                      f"min={test_f[missed_idx].min():.2e}  max={test_f[missed_idx].max():.2e}")
            if len(caught_idx) > 0:
                print(f"  Caught  — f: median={np.median(test_f[caught_idx]):.2e}  "
                      f"min={test_f[caught_idx].min():.2e}  max={test_f[caught_idx].max():.2e}")

    # ── Distribution summaries ────────────────────────────────────────────────
    print(f"\n  True SNR stats across {n_test} sampled binaries:")
    print(f"    min={true_snrs.min():.4f}  max={true_snrs.max():.4f}  "
          f"median={np.median(true_snrs):.4f}  "
          f"p95={np.percentile(true_snrs, 95):.4f}")
    print(f"  Proxy score stats:")
    print(f"    min={proxy_scores.min():.4e}  max={proxy_scores.max():.4e}  "
          f"median={np.median(proxy_scores):.4e}  "
          f"p95={np.percentile(proxy_scores, 95):.4e}")

    print(f"\n{'='*60}\n")


def validate_proxy_filtering_ratio(
    store: ShardedPickleStore,
    chunk_ids: List[int],
    pta,
    enterprise_psrs,
    raw_noise_params,
    parsed_noise_params,
    Tspan_seconds: float,
    n_keep: int = 12_500,
    seed: int = 42,
) -> None:
    """
    Production-relevant proxy validation: compute proxy for ALL binaries,
    keep top n_keep, evaluate true SNR on all kept + a sample of rejected.

    Reports whether any rejected binary would have beaten the worst kept
    binary — the exact failure mode that matters in production.
    """
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"CGW Proxy Filtering Ratio Test  (n_keep={n_keep:,})")
    print(f"{'='*60}")

    # ── Precompute noise matrices ─────────────────────────────────────────────
    print("  Precomputing PTA noise matrices...")
    phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
    TNTs    = pta.get_TNT(raw_noise_params)
    Ts      = pta.get_basis()
    Nvecs   = pta.get_ndiag(raw_noise_params)
    psr_map = {psr.name: psr for psr in enterprise_psrs}
    Sigmas  = [
        TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        for TNT, phiinv in zip(TNTs, phiinvs)
    ]

    # ── Get all proxies for first chunk ──────────────────────────────────────
    # Use first chunk only — test is per-chunk filtering behaviour
    chunk_id = chunk_ids[0]
    pop = store.read(chunk_id)
    n   = len(pop)

    if hasattr(pop, 'cgw_proxy') and pop.cgw_proxy is not None:
        proxies = pop.cgw_proxy.astype(np.float64)
    else:
        proxies = _compute_analytic_proxy(pop, enterprise_psrs,
                                          Tspan_seconds=Tspan_seconds)

    # ── Split into kept and rejected ──────────────────────────────────────────
    n_keep_actual = min(n_keep, n)
    top_idx       = np.argpartition(proxies, -n_keep_actual)[-n_keep_actual:]
    reject_mask   = np.ones(n, dtype=bool)
    reject_mask[top_idx] = False
    reject_idx    = np.where(reject_mask)[0]

    n_check_rejected = min(n_keep_actual, len(reject_idx))
    if n_check_rejected == 0:
        print(f"  Only {n:,} binaries — all kept (n_keep={n_keep_actual}). "
              f"Test not meaningful; use a larger chunk.")
        del pop
        gc.collect()
        return

    rejected_sample = rng.choice(reject_idx, size=n_check_rejected, replace=False)
    eval_idx        = np.concatenate([top_idx, rejected_sample])
    binaries        = [pop[i] for i in eval_idx]
    del pop
    gc.collect()

    # ── Evaluate true SNR ─────────────────────────────────────────────────────
    print(f"\n  Evaluating true SNR on {len(binaries)} binaries "
          f"({n_keep_actual:,} kept + {n_check_rejected:,} rejected sample)...")
    t0 = time.time()
    true_snrs = np.array(compute_cgw_snr_optimal_population_fast(
        psrs=enterprise_psrs,
        pta=pta,
        population=binaries,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan=Tspan_seconds,
        profile=False,
    ))
    print(f"  Took {time.time() - t0:.1f}s")

    kept_snrs     = true_snrs[:n_keep_actual]
    rejected_snrs = true_snrs[n_keep_actual:]

    print(f"\n  Filtering ratio: keeping {n_keep_actual:,} / {n:,} "
          f"({100*n_keep_actual/n:.3f}%)")
    print(f"\n  True SNR — KEPT by proxy (n={n_keep_actual:,}):")
    print(f"    min={kept_snrs.min():.6f}  median={np.median(kept_snrs):.6f}  "
          f"max={kept_snrs.max():.6f}  p95={np.percentile(kept_snrs, 95):.6f}")
    print(f"  True SNR — REJECTED by proxy sample (n={n_check_rejected:,}):")
    print(f"    min={rejected_snrs.min():.6f}  median={np.median(rejected_snrs):.6f}  "
          f"max={rejected_snrs.max():.6f}  p95={np.percentile(rejected_snrs, 95):.6f}")

    kept_min     = kept_snrs.min()
    worst_missed = rejected_snrs.max()
    n_missed     = (rejected_snrs > kept_min).sum()

    print(f"\n  Worst kept SNR    : {kept_min:.6f}")
    print(f"  Worst missed SNR  : {worst_missed:.6f}")
    print(f"  Rejected beats kept threshold: {n_missed} / {n_check_rejected} "
          f"({100*n_missed/n_check_rejected:.2f}%)")

    if n_keep_actual / n > 0.5:
        print(f"\n  ⚠️  WARNING: keeping {100*n_keep_actual/n:.1f}% of chunk — "
              f"filtering ratio too low to be meaningful.")
        print(f"     Use a chunk of at least {n_keep_actual * 20:,} binaries "
              f"for a production-representative test.")
    elif worst_missed > kept_min:
        print(f"\n  ⚠️  PROXY IS MISSING SOURCES above the kept threshold")
        print(f"     Recommend increasing N_PRE_FILTER_PER_CHUNK or relying on "
              f"the regime rescue logic in _compute_cgw_snrs")
    else:
        print(f"\n  ✓  No rejected binary exceeds worst kept binary — proxy is safe")

    print(f"\n{'='*60}\n")


def main():
    import json

    p = argparse.ArgumentParser(description="Validate CGW proxy vs true SNR")
    p.add_argument("--output-dir", required=True,
                   help="Stage 2 output directory containing sim subdirs")
    p.add_argument("--sim-id",     type=int, default=0,
                   help="Which sim to test against (default: 0)")
    p.add_argument("--n-test",     type=int, default=500,
                   help="Number of binaries to evaluate (default: 500)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--config", "-c", default="optimistic",
                   choices=list(config.POPULATION_CONFIGS.keys()))
    args = p.parse_args()

    sim_out_dir = os.path.join(args.output_dir, f"sim{args.sim_id:03d}")

    config_path = os.path.join(sim_out_dir, "metadata", "config.json")
    with open(config_path) as fh:
        run_config = json.load(fh)
    Tspan_seconds = run_config["Tspan_seconds"]
    print(f"  Tspan = {Tspan_seconds/3.156e7:.2f} yr")

    print("\n📡 Loading pulsars...")
    psrs_unfiltered = load_pulsars(verbose=True)
    with suppress_enterprise_warnings():
        psrs_clean, raw_noise_params, _ = filter_pulsars_15yr(psrs_unfiltered, verbose=True)
    print(f"✓ {len(psrs_clean)} pulsars loaded")

    parsed_noise_params = parse_pulsar_parameters(config.NOISEFILE)

    print("\n🔊 Simulating noise (for PTA structure only)...")
    for i, psr in enumerate(psrs_clean):
        print(f"  [{i+1}/{len(psrs_clean)}] {psr.name}...", end=" ", flush=True)
        simulate_psr(psr, raw_noise_params, add_WN=True, add_RN=True)
        print("done", flush=True)
    noise_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}

    print("\n🔧 Building PTA object...")
    _, pta, enterprise_psrs = compute_population_snr(
        psrs_clean=psrs_clean,
        population=None,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan_seconds,
        current_stoas=noise_stoas,
        return_psrs_pta=True,
    )
    print(f"  ✓ PTA built with {len(enterprise_psrs)} pulsars")

    pop_dir   = os.path.join(sim_out_dir, "populations")
    store     = ShardedPickleStore(pop_dir)
    chunk_ids = store.available()
    print(f"\n📦 Found {len(chunk_ids)} chunks in {pop_dir}")

    validate_cgw_proxy(
        store=store,
        chunk_ids=chunk_ids,
        pta=pta,
        enterprise_psrs=enterprise_psrs,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan_seconds=Tspan_seconds,
        n_test=args.n_test,
        seed=args.seed,
    )

    validate_proxy_filtering_ratio(
        store=store,
        chunk_ids=chunk_ids,
        pta=pta,
        enterprise_psrs=enterprise_psrs,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan_seconds=Tspan_seconds,
        n_keep=12_500,
    )


if __name__ == "__main__":
    main()