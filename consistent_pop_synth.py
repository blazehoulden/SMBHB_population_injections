import gc
import os
from tabnanny import verbose
import numpy as np
import time
import tracemalloc
import config
from config import generate_population
from signal_injection import inject_population_nufft, simulate_psr
from pta_builder import build_pta_and_params
from data_loader import restore_original_residuals
from memory_profile import log_memory
import scipy.linalg as sl
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from SMBHB_pop_synth import PopulationArrays, precompute_amplitudes
import enterprise_extensions.frequentist.optimal_statistic as opt_stat
from enterprise_extensions import models
from enterprise.signals import gp_priors, signal_base, utils
from enterprise.pulsar import Pulsar as EnterprisePulsar


# calculates the SNR for a given population
def compute_population_snr(
    population,
    psrs_clean,
    raw_noise_params,
    Tspan,
    current_stoas,         # post-noise stoas to reset to before GW injection
    verbose=False,
    timer=True,
    profile=True,
    inject_eps=1e-6,
    precompute_before_injection=False,
    precompute_parallel=False,
    precompute_chunk_size=10_000_000,
    precompute_workers=None,
):
    if timer:
        t_start = time.perf_counter()

    # 1. Reset to clean (post-noise, pre-GW) stoas
    if verbose:
        print("Resetting pulsars to clean stoas...")
    for psr in psrs_clean:
        psr.stoas[:] = current_stoas[psr.name]

    if timer:
        t_restore = time.perf_counter()
        print(f"Restore: {t_restore - t_start:.2f} s")

    # 2. Precompute amplitudes if requested
    if precompute_before_injection:
        missing = [psr for psr in psrs_clean if psr.name not in population.amp_A]
        if missing:
            if verbose:
                print(f"Precomputing amplitudes for {len(missing)} pulsars...")
            if precompute_parallel and len(missing) > 1:
                n_workers = precompute_workers or min(len(missing), os.cpu_count() or 1)
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = [
                        pool.submit(precompute_amplitudes, population, psr, precompute_chunk_size)
                        for psr in missing
                    ]
                    for fut in futures:
                        fut.result()
            else:
                for psr in missing:
                    precompute_amplitudes(population, psr, precompute_chunk_size)

    gc.collect()

    # 3. Inject GW signal into stoas
    if verbose:
        print("Injecting population...")
    inject_population_nufft(
        psrs_clean,
        population,
        verbose=verbose,
        eps=inject_eps,
        cache_precomputed_amplitudes=precompute_before_injection,
    )
    gc.collect()

    if timer:
        t_inject = time.perf_counter()
        print(f"Inject: {t_inject - t_restore:.2f} s")

    # 4. Snapshot into enterprise AFTER injection
    if verbose:
        print("Snapshotting into enterprise Pulsar objects...")
    with suppress_enterprise_warnings():
        # Parallel enterprise snapshot
        enterprise_psrs = [
        EnterprisePulsar(psr, ephem='DE440', backend='tempo2')
        for psr in psrs_clean
    ]

    gc.collect()

    # 5. Build PTA with enterprise objects
    if verbose:
        print("Building PTA...")
    pta, _, params_out = build_pta_and_params(
        psrs=enterprise_psrs,
        noise_params_15yr=raw_noise_params,
        Tspan=Tspan,
    )
    gc.collect()

    if timer:
        t_build = time.perf_counter()
        print(f"PTA build: {t_build - t_inject:.2f} s")

    # 6. Compute optimal statistic
    if verbose:
        print("Computing optimal statistic...")
    ostat = opt_stat.OptimalStatistic(enterprise_psrs, pta=pta, orf='hd')
    _, _, _, OS, OS_sig = ostat.compute_os(params=params_out)
    snr = OS / OS_sig

    if timer:
        t_end = time.perf_counter()
        print(f"OS: {t_end - t_build:.2f} s")
        print(f"Total: {t_end - t_start:.2f} s")

    gc.collect()
    return snr
 
 
# ============================================================================
# INTERNAL HELPERS
# ============================================================================
 
def _log_memory(label: str) -> None:
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        mb = proc.memory_info().rss / 1024**2
        print(f"[MEM]   {label}: {mb:.1f} MB")
    except ImportError:
        pass


def _get_rss_mb() -> float | None:
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024**2
    except ImportError:
        return None
 
 
def _clear_enterprise_cache(pta) -> None:
    """Clear enterprise's internal delay cache so updated residuals are seen."""
    for sc in pta._signalcollections:
        sc._cache_get_delay      = {}
        sc._cache_list_get_delay = []
 
 
def _restore_zero_residuals(psrs) -> None:
    for psr in psrs:
        psr._residuals = np.zeros(len(psr.toas))

def reset_pulsars(psrs, original_stoas):
    for psr in psrs:
        psr.stoas[:] = original_stoas[psr.name]
 
import warnings
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_enterprise_warnings():
    """Suppress tempo2/enterprise stderr noise during pulsar loading."""
    # Redirect stderr to devnull
    devnull = open(os.devnull, 'w')
    old_stderr = sys.stderr
    sys.stderr = devnull
    # Also suppress Python warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            sys.stderr = old_stderr
            devnull.close()
 
from concurrent.futures import ThreadPoolExecutor

def _make_enterprise_psr(psr):
    with suppress_enterprise_warnings():
        return EnterprisePulsar(psr, ephem='DE440', backend='tempo2')
    
# ============================================================================
# SINGLE-POPULATION SEARCH
# ============================================================================
 
def generate_snr_consistent_population(
    config_template,
    smbhb_module,
    psrs_clean,
    detailed_noise_params,
    pulsar_noise_params_classified,
    Tspan,
    SNR_range,
    N_initial_guess   = 2000,
    N_max_initial     = 10000,
    max_iterations    = 10,
    tolerance         = 0.05,
    verbose           = True,
    profile           = False,
    use_cache         = True,
    cache_threshold   = 7000,
    batch_size        = 10000,
    toggle_memory_profiling = False,
    convergence_threshold   = 0.05,
    detailed_output_SNR     = False,
    block_size        = 2000,
):
 
    SNR_min, SNR_max = SNR_range
    if SNR_min >= SNR_max:
        raise ValueError(
            f"SNR_range must be (min, max) with min < max, got {SNR_range}"
        )
 
    # =========================================================================
    # Phase 0: Generate initial population pool
    # =========================================================================
    N_current = N_max_initial
 
    if verbose:
        print(f"\nGenerating SNR-consistent population...")
        print(f"Target SNR range: [{SNR_min}, {SNR_max}]")
        print(f"Initial guess: N = {N_initial_guess}")
        print(f"Generating initial population pool: N = {N_current}")
 
    if N_current > batch_size:
        if verbose:
            print(f"Generating population in batches of {batch_size}...")
        population = None
        n_batches  = int(np.ceil(N_current / batch_size))
        n_done     = 0
        for batch_idx in range(n_batches):
            batch_n      = min(batch_size, N_current - n_done)
            config_batch = {**config_template, 'n_binaries': batch_n}
            batch_pop    = generate_population(
                config_batch, smbhb_module, T_obs_seconds=Tspan
            )
            if population is None:
                population = batch_pop
            else:
                population = _concat_populations(population, batch_pop)
            n_done += batch_n
    else:
        config    = {**config_template, 'n_binaries': N_current}
        population = generate_population(config, smbhb_module, T_obs_seconds=Tspan)
 
    # =========================================================================
    # Phase 1: Precompute amplitudes ONCE on the full pool
    # =========================================================================
    # population[:N] slicing carries amp_A/B via PopulationArrays.__getitem__,
    # so inject_population_nufft always has the amplitudes it needs.
    if verbose:
        print(f"Precomputing amplitudes for {len(population):,} binaries "
              f"× {len(psrs_clean)} pulsars...")
    t_pre = time.perf_counter()
    for psr in psrs_clean:
        psr._residuals = np.zeros(len(psr.toas))
        precompute_amplitudes(population, psr)
    if verbose:
        print(f"✓ Amplitudes ready ({time.perf_counter()-t_pre:.1f} s)\n")
 
    # =========================================================================
    # Caches and helpers
    # =========================================================================
    snr_cache        = {}
    os_details_cache = {}
    N_tested_list    = []
    SNR_tested_list  = []
    timing_list      = []
 
    # =========================================================================
    # compute_and_cache — the inner loop
    # =========================================================================
    def compute_and_cache(N: int) -> float:
        """
        Inject first N binaries, compute OS SNR, cache and return.
 
        PTA and OptimalStatistic are rebuilt on every call because enterprise
        snapshots psr.residuals at OptimalStatistic construction time rather
        than holding a live reference.  build_pta_and_params is therefore
        unavoidable per iteration.
 
        The speedup vs the old code comes from:
          - inject_population_nufft: O(N + N_freq log N_freq) not O(N * N_toa)
          - population[:N] slicing carries precomputed amp_A/B — no recompute
          - snr_cache: each N tested at most once
        """
        if N in snr_cache:
            return snr_cache[N]
 
        if N > len(population):
            raise ValueError(f"N={N} exceeds pool size {len(population)}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
 
        if toggle_memory_profiling:
            _log_memory(f"Before injection N={N}")
 
        t0 = time.perf_counter() if profile else None
 
        # -- inject signal from first N binaries --------------------------------
        # population[:N] carries precomputed amp_A/B via __getitem__,
        # so no recomputation happens inside inject_population_nufft.
        if verbose:
            print("restoring zero residuals...")
        _restore_zero_residuals(psrs_clean)
        if verbose:
            print("injecting population...")
        inject_population_nufft(
            psrs_clean,
            population[:N],
            pure_signal = True,
            verbose     = False,
        )
        if verbose:
            print("pta building...")
 
        # -- rebuild PTA and OS with updated residuals --------------------------
        # enterprise snapshots residuals at OptimalStatistic.__init__ time,
        # so both must be rebuilt after each injection.
        pta, _, params_out = build_pta_and_params(
            psrs              = psrs_clean,
            noise_params_15yr = detailed_noise_params,
            Tspan             = Tspan,
        )
        if verbose:
            print("computing optimal statistic...")
        ostat = opt_stat.OptimalStatistic(psrs_clean, pta=pta, orf='hd')
        _, _, _, OS, OS_sig = ostat.compute_os(params=params_out)
        snr = OS / OS_sig
 
        if toggle_memory_profiling:
            _log_memory(f"After compute_os N={N}")
 
        if profile:
            timing_list.append({'N': N, 'time': time.perf_counter() - t0})
 
        if detailed_output_SNR:
            os_details_cache[N] = {
                'xi'    : _.tolist() if hasattr(_, 'tolist') else [],
                'OS'    : float(OS),
                'OS_sig': float(OS_sig),
            }
 
        gc.collect()
 
        snr_cache[N] = snr
        N_tested_list.append(N)
        SNR_tested_list.append(snr)
 
        if verbose:
            elapsed = f" ({time.perf_counter()-t0:.1f}s)" if profile else ""
            print(f"    N = {N:>6,}  →  SNR = {snr:.4f}{elapsed}")
 
        return snr
 
    # =========================================================================
    # Phase 3: Test initial guess
    # =========================================================================
    if verbose:
        print("─" * 50)
        print("Phase 1: initial guess")
 
    N_test   = min(N_initial_guess, len(population))
    snr_test = compute_and_cache(N_test)
 
    if SNR_min <= snr_test <= SNR_max:
        search_direction = "verify"
        N_low,  SNR_low  = 1,      None
        N_high, SNR_high = N_test, snr_test
    elif snr_test < SNR_min:
        search_direction = "upward"
        N_low,  SNR_low  = N_test, snr_test
        N_high, SNR_high = None,   None
    else:
        search_direction = "downward"
        N_high, SNR_high = N_test, snr_test
        N_low,  SNR_low  = 1,      None
 
    if verbose:
        status = ("in range!" if search_direction == "verify"
                  else ("below" if search_direction == "upward" else "above")
                  + " target")
        print(f"  Initial SNR {snr_test:.4f} is {status} → searching {search_direction}")
 
    # =========================================================================
    # Phase 4: Find bracketing points
    # =========================================================================
    if verbose:
        print("\nPhase 2: bracket search")
 
    expansion_count = 0
    max_expansions  = 6
 
    if search_direction == "downward":
        snr_at_1 = compute_and_cache(1)
        if snr_at_1 > SNR_max:
            if verbose:
                print(f"  ✗ SNR at N=1 ({snr_at_1:.4f}) exceeds target max — "
                      f"check population and OS_sig")
            return _build_result(
                population, 1, snr_at_1, SNR_range,
                N_tested_list, SNR_tested_list,
                iterations=0, expansions=0,
                warning='SNR_at_N1_exceeds_target', broken=True,
                detailed_output_SNR=detailed_output_SNR,
                os_details_cache=os_details_cache,
            )
        elif SNR_min <= snr_at_1 <= SNR_max:
            if verbose:
                print(f"  ✓ N=1 already in target range")
            return _build_result(
                population, 1, snr_at_1, SNR_range,
                N_tested_list, SNR_tested_list,
                iterations=0, expansions=0,
                detailed_output_SNR=detailed_output_SNR,
                os_details_cache=os_details_cache,
            )
        else:
            N_low, SNR_low = 1, snr_at_1
 
    while expansion_count < max_expansions:
        if search_direction == "upward":
            N_high_target = int((N_high or N_low) * 1.5)
 
            while N_high_target > len(population):
                if expansion_count >= max_expansions:
                    N_high_target = len(population)
                    break
                expansion_count += 1
                N_to_add = int(len(population) * 0.5)
                if verbose:
                    print(f"  ⚠ Pool too small — expanding by {N_to_add:,}")
 
                config_add   = {**config_template, 'n_binaries': N_to_add}
                new_pop      = generate_population(
                    config_add, smbhb_module, T_obs_seconds=Tspan
                )
                # Precompute amplitudes for new binaries
                for psr in psrs_clean:
                    precompute_amplitudes(new_pop, psr)
                population = _concat_populations(population, new_pop)
 
            N_high   = N_high_target
            snr_high = compute_and_cache(N_high)
 
            if snr_high > SNR_max:
                SNR_high = snr_high
                if SNR_low is not None and SNR_low < SNR_min:
                    break
            elif snr_high >= SNR_min:
                SNR_high = snr_high
                if SNR_low is not None and SNR_low < SNR_min:
                    break
            else:
                N_low, SNR_low = N_high, snr_high
 
        elif search_direction == "downward":
            N_new = max(1, (N_low + N_high) // 2)
            if N_new == N_low or N_new == N_high:
                break
            snr_new = compute_and_cache(N_new)
            if snr_new < SNR_min:
                N_low, SNR_low = N_new, snr_new
                break
            N_high, SNR_high = N_new, snr_new
 
        else:  # verify
            break
 
    # =========================================================================
    # Phase 5: Bisection
    # =========================================================================
    if verbose:
        if SNR_low is not None and SNR_high is not None:
            print(f"\n  ✓ Bracketed: N ∈ [{N_low}, {N_high}], "
                  f"SNR ∈ [{SNR_low:.4f}, {SNR_high:.4f}]")
        else:
            print(f"\n  ⚠ Could not bracket after {expansion_count} expansions")
        print("\nPhase 3: bisection")
 
    found_in_range = False
    iteration      = 0
 
    for iteration in range(max_iterations):
        if N_high - N_low <= 1:
            if verbose:
                print(f"  ✓ Converged (N bracket width = 1)")
            break
        if found_in_range and (N_high - N_low) / N_high <= convergence_threshold:
            if verbose:
                print(f"  ✓ Converged (bracket < {convergence_threshold*100:.0f}% of N)")
            break
 
        frac  = 0.15 if found_in_range else 0.5
        N_mid = int(N_low + frac * (N_high - N_low))
        N_mid = max(N_low + 1, min(N_mid, N_high - 1))
 
        snr_mid = compute_and_cache(N_mid)
 
        if snr_mid < SNR_min:
            N_low,  SNR_low  = N_mid, snr_mid
            found_in_range   = False
        elif snr_mid > SNR_max:
            N_high, SNR_high = N_mid, snr_mid
            found_in_range   = False
        else:
            found_in_range   = True
            N_high, SNR_high = N_mid, snr_mid
 
    # =========================================================================
    # Phase 6: Select best result
    # =========================================================================
    valid   = [i for i, s in enumerate(SNR_tested_list) if SNR_min <= s <= SNR_max]
    if valid:
        best_idx = min(valid, key=lambda i: N_tested_list[i])
    else:
        above = [i for i, s in enumerate(SNR_tested_list) if s > SNR_max]
        if above:
            best_idx = min(above, key=lambda i: N_tested_list[i])
        else:
            mid      = 0.5 * (SNR_min + SNR_max)
            best_idx = min(range(len(SNR_tested_list)),
                           key=lambda i: abs(SNR_tested_list[i] - mid))
 
    N_final   = N_tested_list[best_idx]
    SNR_final = SNR_tested_list[best_idx]
 
    if verbose:
        print(f"\n✓ Done: N = {N_final:,}, SNR = {SNR_final:.4f} "
              f"(target [{SNR_min}, {SNR_max}])")
 
    return _build_result(
        population, N_final, SNR_final, SNR_range,
        N_tested_list, SNR_tested_list,
        iterations  = iteration + 1,
        expansions  = expansion_count,
        timing_list = timing_list if profile else None,
        detailed_output_SNR = detailed_output_SNR,
        os_details_cache    = os_details_cache,
    )
 
 
# ============================================================================
# ENSEMBLE WRAPPER
# ============================================================================
 
def generate_snr_consistent_populations(
    config_template,
    smbhb_module,
    psrs_clean,
    detailed_noise_params,
    pulsar_noise_params_classified,
    Tspan,
    SNR_range,
    N_sims            = 20,
    N_initial_guess   = 2000,
    N_max_initial     = 10000,
    verbose           = True,
    save_populations  = True,
    profile           = False,
    use_cache         = True,
    cache_threshold   = 7000,
    batch_size        = 10000,
    toggle_memory_profiling = False,
    detailed_output_SNR     = False,
    block_size        = 2000,
):
    """
    Generate N_sims SMBHB populations each consistent with SNR_range.
 
    Args and return format identical to the previous version.
    """
    start_time = time.time()
 
    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING SNR-CONSISTENT POPULATIONS")
        print(f"{'='*70}")
        print(f"Target SNR range:    [{SNR_range[0]}, {SNR_range[1]}]")
        print(f"Simulations:         {N_sims}")
        print(f"Initial guess:       N = {N_initial_guess}")
        print(f"Max initial pool:    N = {N_max_initial}")
        print(f"Batch size:          {batch_size}")
        print(f"{'='*70}\n")
 
    populations       = []
    n_bininaries_list = []
    SNR_achieved_list = []
    success_count     = 0
 
    for sim_idx in range(N_sims):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"SIMULATION {sim_idx + 1}/{N_sims}")
            print(f"{'─'*70}")
 
        t_sim = time.time()
        result = generate_snr_consistent_population(
            config_template             = config_template,
            smbhb_module                = smbhb_module,
            psrs_clean                  = psrs_clean,
            detailed_noise_params       = detailed_noise_params,
            pulsar_noise_params_classified = pulsar_noise_params_classified,
            Tspan                       = Tspan,
            SNR_range                   = SNR_range,
            N_initial_guess             = N_initial_guess,
            N_max_initial               = N_max_initial,
            verbose                     = verbose,
            profile                     = profile,
            use_cache                   = use_cache,
            cache_threshold             = cache_threshold,
            batch_size                  = batch_size,
            toggle_memory_profiling     = toggle_memory_profiling,
            detailed_output_SNR         = detailed_output_SNR,
            block_size                  = block_size,
        )
 
        if result is not None:
            success_count += 1
            n_bininaries_list.append(result['n_bininaries'])
            SNR_achieved_list.append(result['SNR_achieved'])
            result['sim_index'] = sim_idx
 
            if save_populations:
                populations.append(result)
            else:
                populations.append({
                    'sim_index'   : sim_idx,
                    'n_bininaries': result['n_bininaries'],
                    'SNR_achieved': result['SNR_achieved'],
                    'SNR_target'  : result['SNR_target'],
                })
 
            if verbose:
                print(f"✓ Simulation {sim_idx+1} done "
                      f"({time.time()-t_sim:.1f} s)")
        else:
            if verbose:
                print(f"✗ Simulation {sim_idx+1} FAILED")
 
    # =========================================================================
    # Compile summary statistics
    # =========================================================================
    N_arr   = np.array(n_bininaries_list) if n_bininaries_list else np.array([])
    SNR_arr = np.array(SNR_achieved_list) if SNR_achieved_list else np.array([])
 
    def _stats(arr):
        if len(arr) == 0:
            return dict(mean=None, median=None, std=None, min=None, max=None)
        return dict(
            mean   = float(np.mean(arr)),
            median = float(np.median(arr)),
            std    = float(np.std(arr)),
            min    = float(np.min(arr)),
            max    = float(np.max(arr)),
        )
 
    total_time = time.time() - start_time
 
    results = {
        'populations': populations,
        'summary_statistics': {
            'n_bininaries': {**_stats(N_arr),
                             'all_values': n_bininaries_list},
            'SNR_achieved': {**_stats(SNR_arr),
                             'all_values': SNR_achieved_list},
        },
        'config': {
            'SNR_range'      : SNR_range,
            'N_sims'         : N_sims,
            'N_initial_guess': N_initial_guess,
            'N_max_initial'  : N_max_initial,
            'use_cache'      : use_cache,
            'cache_threshold': cache_threshold,
            'batch_size'     : batch_size,
            'config_template': config_template,
        },
        'metadata': {
            'success_count': success_count,
            'success_rate' : success_count / N_sims,
            'total_time'   : total_time,
            'save_populations': save_populations,
        },
    }
 
    if verbose:
        print(f"\n{'='*70}")
        print(f"ENSEMBLE SUMMARY")
        print(f"{'='*70}")
        print(f"Success rate: {results['metadata']['success_rate']:.1%} "
              f"({success_count}/{N_sims})")
        if success_count > 0:
            s = results['summary_statistics']
            print(f"\nBinaries per population:")
            print(f"  Mean ± std : {s['n_bininaries']['mean']:.0f} "
                  f"± {s['n_bininaries']['std']:.0f}")
            print(f"  Range      : [{s['n_bininaries']['min']:.0f}, "
                  f"{s['n_bininaries']['max']:.0f}]")
            print(f"\nSNR achieved:")
            print(f"  Mean ± std : {s['SNR_achieved']['mean']:.4f} "
                  f"± {s['SNR_achieved']['std']:.4f}")
            print(f"  Range      : [{s['SNR_achieved']['min']:.4f}, "
                  f"{s['SNR_achieved']['max']:.4f}]")
        print(f"\nTotal time: {total_time:.1f} s ({total_time/60:.1f} min)")
        print(f"{'='*70}\n")
 
    return results
 
 
# ============================================================================
# PRIVATE HELPERS
# ============================================================================
 
def _concat_populations(a: PopulationArrays,
                         b: PopulationArrays) -> PopulationArrays:
    """Concatenate two PopulationArrays, merging amp_A/B dicts."""
    fields = ['f','Mc','Mtot','D_comov','z','h0','ra','dec','psi','iota','phi0']
    kwargs = {k: np.concatenate([getattr(a, k), getattr(b, k)]) for k in fields}
    new    = PopulationArrays(**kwargs)
    # Merge amplitude dicts — keys present in both get concatenated
    for psr_name in set(list(a.amp_A.keys()) + list(b.amp_A.keys())):
        A_parts = []
        B_parts = []
        if psr_name in a.amp_A:
            A_parts.append(a.amp_A[psr_name])
            B_parts.append(a.amp_B[psr_name])
        if psr_name in b.amp_A:
            A_parts.append(b.amp_A[psr_name])
            B_parts.append(b.amp_B[psr_name])
        new.amp_A[psr_name] = np.concatenate(A_parts)
        new.amp_B[psr_name] = np.concatenate(B_parts)
    return new
 
 
def _build_result(
    population, N_final, SNR_final, SNR_range,
    N_tested_list, SNR_tested_list,
    iterations=0, expansions=0,
    warning=None, broken=False,
    timing_list=None,
    detailed_output_SNR=False,
    os_details_cache=None,
):
    meta = {
        'N_tested'   : N_tested_list,
        'SNR_tested' : SNR_tested_list,
        'iterations' : iterations,
        'expansions' : expansions,
        'used_cache' : True,
        'warning'    : warning,
        'broken'     : broken,
    }
    if timing_list is not None:
        meta['timing'] = timing_list
 
    result = {
        'population'   : population[:N_final],
        'n_bininaries' : N_final,
        'SNR_achieved' : float(SNR_final),
        'SNR_target'   : SNR_range,
        'search_metadata': meta,
    }
    if detailed_output_SNR and os_details_cache:
        result['os_details']         = os_details_cache.get(N_final)
        result['os_details_history'] = os_details_cache
    return result


def _build_result_distance_scaling(
    population, SNR_final, 
    warning=None, broken=False,
    timing_list=None,
    timing_profile=None,
    memory_profile=None,
    detailed_output_SNR=False,
):
    meta = {
        'warning'    : warning,
        'broken'     : broken,
    }
    if timing_list is not None:
        meta['timing'] = timing_list
    if timing_profile is not None:
        meta['timing_profile'] = timing_profile
    if memory_profile is not None:
        meta['memory_profile'] = memory_profile
 
    result = {
        'population'   : population,
        'n_bininaries' : len(population),
        'SNR_final' : float(SNR_final),
        'search_metadata': meta,
    }
    return result


def generate_consistent_population_distance_scaling(
    config_template,
    smbhb_module,
    psrs_clean,
    raw_noise_params,
    Tspan,
    target_SNR,
    original_stoas,              # stoas BEFORE any noise or GW — raw loaded state
    timer=True,
    verbose=True,
    test=False,
    toggle_memory_profiling=False,
    keep_amplitudes_in_result=False,
    inject_eps=1e-6,
    precompute_parallel=False,
):
    """
    Given an existing population, compute its SNR and scale distances to achieve target_SNR. 
    Utilises the fact that SNR ∝ h0^2 ∝ 1/D^2, so scaling distances by a factor scales SNR inversely by the same factor.
    
    Arguments:
    - config_template: template configuration for generating the population
    - smbhb_module: the SMBHB module to use for population generation
    - psrs_clean: list of pulsar objects with clean residuals (no signal injected)
    - raw_noise_params: dict of raw noise parameters for the pulsars
    - Tspan: observation time span in seconds
    - target_SNR: desired SNR to achieve after scaling distances

    Returns:
    - result dict with the same high-level shape used by generate_snr_consistent_population
    """
 
    profile_clock = timer or toggle_memory_profiling
    t_start = time.perf_counter() if profile_clock else None

    timing_profile = {}
    memory_profile = {
        'rss_mb': {},
        'traced_peak_mb': None,
        'traced_current_mb': None,
    }

    if toggle_memory_profiling:
        tracemalloc.start()
        rss0 = _get_rss_mb()
        if rss0 is not None:
            memory_profile['rss_mb']['start'] = float(rss0)


    if verbose:
        print(f"\nGenerating SNR-consistent population...")
        print(f"Target SNR: {target_SNR}")
 
    if target_SNR <= 0:
        raise ValueError(f"target_SNR must be > 0, got {target_SNR}")

    t_pop0 = time.perf_counter() if profile_clock else None
    population = generate_population(config_template, smbhb_module, T_obs_seconds=Tspan)
    if profile_clock and t_pop0 is not None:
        timing_profile['generate_population_s'] = time.perf_counter() - t_pop0
    if toggle_memory_profiling:
        rss = _get_rss_mb()
        if rss is not None:
            memory_profile['rss_mb']['after_generate_population'] = float(rss)



    # Compute current SNR of the population
    t_trial0 = time.perf_counter() if profile_clock else None
    snr_trial = compute_population_snr(
        population,
        psrs_clean,
        raw_noise_params,
        Tspan,
        current_stoas=original_stoas,
        timer=timer,
        verbose=verbose,
        inject_eps=inject_eps,
        precompute_before_injection=False,
        precompute_parallel=precompute_parallel,
    )
    if profile_clock and t_trial0 is not None:
        timing_profile['initial_snr_compute_s'] = time.perf_counter() - t_trial0
    if toggle_memory_profiling:
        rss = _get_rss_mb()
        if rss is not None:
            memory_profile['rss_mb']['after_initial_snr'] = float(rss)
    if verbose:
        print(f"Initial SNR: {snr_trial:.4f}, Target SNR: {target_SNR:.4f}")

    if snr_trial <= 0:
        print(f"  ✗ Initial SNR {snr_trial:.4f} is non-positive, cannot scale distances to achieve target SNR.")
        return None

    
    # Calculate scaling factor for distances based on the ratio of trial SNR to target SNR
    snr_scaling_factor = snr_trial / target_SNR
    distance_scaling_factor = snr_scaling_factor ** (1./2.)
    if verbose:
        print(f"Scaling distances by factor {distance_scaling_factor:.4f} to achieve target SNR...")

    # Scale distances and amplitudes accordingly
    population.D_comov *= distance_scaling_factor
    population.h0 /= distance_scaling_factor  # h0 ∝ 1/D, so scale inversely    
    # Amplitude dictionaries are very large and usually not needed for post-run
    # analysis. Dropping them from stored results greatly reduces peak memory.
    
    if test:
        population.amp_A = {
        psr: amp / distance_scaling_factor for psr, amp in population.amp_A.items()
        }
        population.amp_B = {
            psr: amp / distance_scaling_factor for psr, amp in population.amp_B.items()
        }
        # Recompute SNR after scaling to verify it matches target
        t_test0 = time.perf_counter() if profile_clock else None
        snr_final = compute_population_snr(
            population,
            psrs_clean,
            raw_noise_params,
            Tspan,
            timer=timer,
            verbose=verbose,
            inject_eps=inject_eps,
            precompute_before_injection=False,
        )
        if profile_clock and t_test0 is not None:
            timing_profile['post_scale_snr_compute_s'] = time.perf_counter() - t_test0
        if toggle_memory_profiling:
            rss = _get_rss_mb()
            if rss is not None:
                memory_profile['rss_mb']['after_post_scale_snr'] = float(rss)
        if verbose:
            print(f"Final SNR after scaling: {snr_final:.4f}")
            
    if not keep_amplitudes_in_result:
        population.amp_A = {}
        population.amp_B = {}
        if toggle_memory_profiling:
            rss = _get_rss_mb()
            if rss is not None:
                memory_profile['rss_mb']['after_drop_amplitudes'] = float(rss)

    snr_final = float(target_SNR)

    if profile_clock and t_start is not None:
        timing_profile['total_s'] = time.perf_counter() - t_start

    if toggle_memory_profiling:
        current, peak = tracemalloc.get_traced_memory()
        memory_profile['traced_current_mb'] = float(current / 1024**2)
        memory_profile['traced_peak_mb'] = float(peak / 1024**2)
        rss = _get_rss_mb()
        if rss is not None:
            memory_profile['rss_mb']['end'] = float(rss)
        tracemalloc.stop()

        if verbose:
            print("Memory profile (RSS MB):")
            for label, mb in memory_profile['rss_mb'].items():
                print(f"  {label}: {mb:.1f} MB")
            print(f"  traced_peak_mb: {memory_profile['traced_peak_mb']:.1f} MB")

    

    n_binaries = len(population)
    return _build_result_distance_scaling(
        population=population,
        SNR_final=snr_final,
        timing_profile=timing_profile if profile_clock else None,
        memory_profile=memory_profile if toggle_memory_profiling else None,
    )


def generate_snr_consistent_populations_distance_scaling(
    config_template,
    smbhb_module,
    psrs_clean,
    raw_noise_params,
    Tspan,
    target_SNR,
    resimulate_noise=True,       # toggle: new noise draw per simulation
    original_stoas=None,          # needed if resimulate_noise=False to reset to clean state
    N_sims            = 20,
    verbose           = True,
    save_populations  = True,
    profile           = False,
    test              = False,
    toggle_memory_profiling = False,
    keep_amplitudes_in_result = False,
    inject_eps        = 1e-6,
    precompute_parallel = False,
):
    """
    Generate N_sims SMBHB populations each consistent with SNR_range.
 
    Args and return format identical to the previous version.
    """
    start_time = time.time()

    if target_SNR <= 0:
        raise ValueError(
            f"target_SNR must be positive, got {target_SNR}"
        )
 
    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING SNR-CONSISTENT POPULATIONS")
        print(f"{'='*70}")
        print(f"Target SNR:          {target_SNR}")
        print(f"Simulations:         {N_sims}")
        print(f"{'='*70}\n")
 
    populations       = []
    n_bininaries_list = []
    SNR_final_list = []
    success_count     = 0
 
    for sim_idx in range(N_sims):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"SIMULATION {sim_idx + 1}/{N_sims}")
            print(f"{'─'*70}")
        if resimulate_noise:
            # Reset to raw loaded stoas (no noise, no GW)
            for psr in psrs_clean:
                psr.stoas[:] = original_stoas[psr.name]
            # Fresh noise draw
            for psr in psrs_clean:
                simulate_psr(psr, raw_noise_params, add_WN=True, add_RN=True)
            # Save this noise realisation as the clean state for THIS snr evaluation
            current_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}
        else:
            # Use fixed noise — original_stoas should already be post-noise state
            current_stoas = original_stoas
 
        t_sim = time.time()
        result = None
        ii = 0
        while result is None: # iterate until we get a valid population (in case of non-positive initial SNR or other issues)
            if verbose and ii > 0:
                print(f"✗ Simulation {sim_idx+1}, trial {ii} FAILED, retrying...")
            ii += 1
            result = generate_consistent_population_distance_scaling(
            config_template             = config_template,
            smbhb_module                = smbhb_module,
            psrs_clean                  = psrs_clean,
            raw_noise_params            = raw_noise_params,
            Tspan                       = Tspan,
            target_SNR                  = target_SNR,
            original_stoas              = current_stoas,
            verbose                     = verbose,
            timer                       = profile,
            test                        = test,
            toggle_memory_profiling     = toggle_memory_profiling,
            keep_amplitudes_in_result   = keep_amplitudes_in_result,
            inject_eps                  = inject_eps,
            precompute_parallel         = precompute_parallel,
        )
 
        if result is not None:
            success_count += 1
            n_bininaries_list.append(result['n_bininaries'])
            SNR_final_list.append(result['SNR_final'])
            result['sim_index'] = sim_idx
 
            if save_populations:
                populations.append(result)
            else:
                populations.append({
                    'n_bininaries': result['n_bininaries'],
                    'SNR_final'   : result['SNR_final'],
                })
 
            if verbose:
                print(f"✓ Simulation {sim_idx+1} done "
                      f"({time.time()-t_sim:.1f} s)")
        else:

            if verbose:
                print(f"✗ Simulation {sim_idx+1} FAILED")
 
    # =========================================================================
    # Compile summary statistics
    # =========================================================================
    N_arr   = np.array(n_bininaries_list) if n_bininaries_list else np.array([])
    SNR_arr = np.array(SNR_final_list) if SNR_final_list else np.array([])
 
    def _stats(arr):
        if len(arr) == 0:
            return dict(mean=None, median=None, std=None, min=None, max=None)
        return dict(
            mean   = float(np.mean(arr)),
            median = float(np.median(arr)),
            std    = float(np.std(arr)),
            min    = float(np.min(arr)),
            max    = float(np.max(arr)),
        )
 
    total_time = time.time() - start_time
 
    results = {
        'populations': populations,
        'summary_statistics': {
            'n_bininaries': {**_stats(N_arr),
                             'all_values': n_bininaries_list},
            'SNR_final': {**_stats(SNR_arr),
                          'all_values': SNR_final_list},
        },
        'config': {
            'N_sims'         : N_sims,
            'config_template': config_template,
            'target_snr'     : float(target_SNR),
            'profile'        : bool(profile),
            'memory_profiling': bool(toggle_memory_profiling),
            'keep_amplitudes_in_result': bool(keep_amplitudes_in_result),
            'inject_eps'     : float(inject_eps),
            'precompute_parallel': bool(precompute_parallel),
        },
        'metadata': {
            'success_count': success_count,
            'success_rate' : success_count / N_sims,
            'total_time'   : total_time,
            'save_populations': save_populations,
        },
    }
 
    if verbose:
        print(f"\n{'='*70}")
        print(f"ENSEMBLE SUMMARY")
        print(f"{'='*70}")
        print(f"Success rate: {results['metadata']['success_rate']:.1%} "
              f"({success_count}/{N_sims})")
        if success_count > 0:
            s = results['summary_statistics']
            print(f"\nBinaries per population:")
            print(f"  Mean ± std : {s['n_bininaries']['mean']:.0f} "
                  f"± {s['n_bininaries']['std']:.0f}")
            print(f"  Range      : [{s['n_bininaries']['min']:.0f}, "
                  f"{s['n_bininaries']['max']:.0f}]")
            print(f"\nSNR achieved:")
            print(f"  Mean ± std : {s['SNR_final']['mean']:.4f} "
                  f"± {s['SNR_final']['std']:.4f}")
            print(f"  Range      : [{s['SNR_final']['min']:.4f}, "
                  f"{s['SNR_final']['max']:.4f}]")
        print(f"\nTotal time: {total_time:.1f} s ({total_time/60:.1f} min)")
        print(f"{'='*70}\n")
 
    return results