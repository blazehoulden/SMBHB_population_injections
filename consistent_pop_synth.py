import gc
import numpy as np
import time
import config
from signal_injection import precompute_binary_signals, inject_population_subset_cached, inject_population_into_psrs, _auto_chunk_size, r_k, _gw_residuals_chunked, _gw_residuals_vec
from pta_builder import build_pta_and_params
from data_loader import restore_original_residuals
from memory_profile import log_memory
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def compute_population_snr(population, psrs_clean, detailed_noise_params, pulsar_noise_params_classified, Tspan, verbose=False, timer=False, profile=False):
    """
    Compute SNR for a given population of binaries (accounting for interference).
    
    Binaries interfere with one another, so we must compute SNR for the entire
    population together, not as a sum of individual contributions.
    
    Parameters:
        profile: if True, return timing breakdown along with SNR as (snr, timing_dict)
                 if False, return just SNR
    
    Returns:
        snr (float) or (snr, timing_dict) if profile=True
    """
    try:
        # PROFILE: Injection step
        if profile:
            t0 = time.time()
        psrs_injected = inject_population_into_psrs(
            psrs_clean, population, pure_signal=True, verbose=False, pulsar_noise_params=pulsar_noise_params_classified
        )
        if profile:
            t_inject = time.time() - t0
        
        # PROFILE: PTA building step
        if profile:
            t0 = time.time()
        pta, _, params_out = build_pta_and_params(
            psrs=psrs_injected, noise_params_15yr=detailed_noise_params, 
            Tspan=Tspan
        )
        if profile:
            t_pta = time.time() - t0
        
        # PROFILE: OptimalStatistic instantiation
        if profile:
            t0 = time.time()
        ostat = opt_stat.OptimalStatistic(psrs_injected, pta=pta, orf='hd')
        if profile:
            t_ostat_init = time.time() - t0
        
        # PROFILE: compute_os step
        if profile or timer:
            t0 = time.time()
        xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
        if profile or timer:
            t_compute_os = time.time() - t0
        
        snr = OS / OS_sig
        
        if verbose:
            print(f"  N={len(population)}: SNR = {snr:.3f}")
        
        if timer and not profile:
            print(f"    Optimal Statistic computed in {t_compute_os:.2f} s")
        
        if profile:
            timing = {
                'inject': t_inject,
                'pta': t_pta,
                'ostat_init': t_ostat_init,
                'compute_os': t_compute_os,
                'total': t_inject + t_pta + t_ostat_init + t_compute_os
            }
            return snr, timing
        else:
            return snr
        
    except Exception as e:
        if verbose:
            print(f"  N={len(population)}: FAILED - {e}")
        if profile:
            return np.nan, None
        else:
            return np.nan

def generate_snr_consistent_population(
    config_template, smbhb_module, psrs_clean, detailed_noise_params, 
    pulsar_noise_params_classified, Tspan,
    SNR_range, N_initial_guess=2000, N_max_initial=10000,
    max_iterations=10, tolerance=0.05, verbose=True, profile=False,
    use_cache=True, cache_threshold=7000, batch_size=10000, 
    toggle_memory_profiling=False, convergence_threshold=0.05, 
    detailed_output_SNR=False, block_size=2000
):
    from config import generate_population

    SNR_min, SNR_max = SNR_range
    if SNR_min >= SNR_max:
        raise ValueError(f"SNR_range must be (min, max) with min < max, got {SNR_range}")

    # =====================================================================
    # Phase 0: Generate and filter initial population pool
    # =====================================================================
    N_current = N_max_initial

    if verbose:
        print(f"\nGenerating SNR-consistent population...")
        print(f"Target SNR range: [{SNR_min}, {SNR_max}]")
        print(f"Initial guess: N = {N_initial_guess}")
        print(f"Generating initial population pool: N = {N_current}")

    if N_current > batch_size:
        if verbose:
            print(f"Generating population in batches...")
        population = []
        n_batches = int(np.ceil(N_current / batch_size))
        for batch_idx in range(n_batches):
            batch_n = min(batch_size, N_current - len(population))
            config_batch = {**config_template, 'n_binaries': batch_n}
            population.extend(generate_population(
                config_batch, smbhb_module, 
                T_obs_years=Tspan/(365.25*86400)
            ))
    else:
        config = {**config_template, 'n_binaries': N_current}
        population = generate_population(
            config, smbhb_module, 
            T_obs_years=Tspan/(365.25*86400)
        )

    # =====================================================================
    # KEY OPTIMISATION 1: Pre-compute ALL per-binary GW signals once
    # =====================================================================
    # Each binary's residual for each pulsar is computed once and stored.
    # compute_and_cache then just sums the first N entries — O(N*n_psr)
    # instead of recomputing from scratch each time — O(N*n_psr*n_toa).
    #
    # Memory: n_psr * N_max * n_toa_avg * 8 bytes
    # For 67 pulsars, 3000 binaries, 10000 TOAs avg: ~16 GB — too large
    # So we store per-pulsar cumulative sums instead (see below).
    
    if verbose:
        print(f"Pre-computing per-binary GW signals...")
    
    if verbose:
        print(f"Initialising lazy signal cache (block_size={block_size})...")

    signal_cache = LazyCumsumCache(
        psrs=psrs_clean,
        population=population,
        block_size=block_size,           # tune based on n_toa and available memory
        n_workers=min(8, len(psrs_clean)),
        max_memory_mb=500,
        verbose=verbose,
    )

    if verbose:
        print(f"✓ Lazy cache ready — signals computed on demand\n")
    # =====================================================================
    # KEY OPTIMISATION 2: Build PTA and OptimalStatistic ONCE
    # =====================================================================
    # OptimalStatistic.__init__ only stores:
    #   - references to psrs_clean (not residual values)
    #   - the ORF matrix (HD coefficients, depends only on pulsar positions)
    #   - PTA signal collections
    # It does NOT read psr.residuals at construction time.
    #
    # compute_os() reads psr.residuals fresh on every call, so as long as
    # we update psr._residuals before each compute_os call, the OS sees
    # the correct injected signal. This means we can safely reuse the same
    # ostat object across all N values tested in the search.
    #
    # build_pta_and_params uses:
    #   - psrs_clean: pulsar objects (residuals don't matter here — 
    #                 PTA is built from noise params only)
    #   - detailed_noise_params: full noisefile params (EFAC/EQUAD/ECORR/RN)
    #   - Tspan: observation span for Fourier basis construction
    # pulsar_noise_params_classified is NOT used here — that's the parsed
    # version used for drawing noise realisations in inject_population_into_psrs,
    # which we no longer call inside compute_and_cache (we use cumsum instead).

    if verbose:
        print(f"Building PTA and OptimalStatistic (once)...")

    # Zero residuals for PTA construction — the noise model (EFAC/EQUAD/RN)
    # is built from detailed_noise_params, not from the residual values.
    # Zeroing here just ensures nothing unexpected is in psr.residuals
    # when enterprise initialises its internal structures.
    for psr in psrs_clean:
        psr._residuals = np.zeros(len(psr.toas))

    # detailed_noise_params: the raw noisefile dict with keys like
    # '{psr_name}_efac', '{psr_name}_red_noise_log10_A' etc.
    # This is what build_pta_and_params uses to set Constant() parameters.
    pta, _, params_out = build_pta_and_params(
        psrs=psrs_clean,
        noise_params_15yr=detailed_noise_params,
        Tspan=Tspan
    )

    # Build OS
    # orf='hd' uses Hellings-Downs coefficients (isotropic GWB assumption).
    # NOTE: for finite SMBHB populations this is an approximation — see
    # compute_antenna_pattern_orf_vectorised for the correct discrete-source ORF.
    ostat = opt_stat.OptimalStatistic(psrs_clean, pta=pta, orf='hd')

    # Pre-compute OS_sig (denominator) once.
    # OS_sig = 1/sqrt(sum_ab tr(C_a^{-1} S_ab C_b^{-1} S_ba))
    # This depends only on:
    #   - C_a: noise covariance (from detailed_noise_params via pta)
    #   - S_ab: GWB cross-correlation template (from ORF + PSD model)
    # Neither depends on the injected signal, so OS_sig is identical
    # across all calls regardless of what is in psr.residuals.
    _, _, _, _, OS_sig = ostat.compute_os(params=params_out)


    if verbose:
        print(f"✓ PTA built. OS_sig = {OS_sig:.3e} (constant for all N)\n")

    # =====================================================================
    # Caches and helpers
    # =====================================================================
    snr_cache        = {}
    os_details_cache = {}
    N_tested_list    = []
    SNR_tested_list  = []
    timing_list      = []


    def compute_and_cache(N):
        """
        Compute OS SNR for first N binaries from the population.

        Flow:
          1. inject_from_cumsum(N)  — set psr._residuals to GW signal
          2. ostat.compute_os()     — reads psr.residuals, computes OS numerator
          3. snr = OS / OS_sig      — OS_sig pre-computed, constant

        Variables from outer scope:
          psrs_clean               — pulsar objects (residuals updated in place)
          ostat                    — pre-built OptimalStatistic (reused)
          params_out               — noise params dict from build_pta_and_params
                                     (uses detailed_noise_params internally)
          OS_sig                   — pre-computed denominator (constant)
          binary_signals           — cumsum cache keyed by psr.name
          pulsar_noise_params_classified — NOT used here (no noise injection)
        """
        if N in snr_cache:
            return snr_cache[N]

        if N > len(population):
            raise ValueError(f"N={N} exceeds population size {len(population)}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")

        if toggle_memory_profiling:
            log_memory(f"  Before injection N={N}")

        t0_total = time.time() if profile else None

        # Restore clean baseline then inject GW signal from first N binaries
        restore_original_residuals(psrs_clean)
        
        subset_population = population[:N]
        psrs_injected = inject_population_into_psrs(
            psrs_clean, subset_population,
            pure_signal=True, verbose=False,
            pulsar_noise_params=pulsar_noise_params_classified
        )

        # Build PTA with injected residuals
        pta, _, params_out = build_pta_and_params(
            psrs=psrs_injected,
            noise_params_15yr=detailed_noise_params,
            Tspan=Tspan
        )

        # # Set psr._residuals = cumsum of GW signals from first N binaries.
        # # After this call, psrs_clean[i].residuals is the pure GW signal
        # # from binaries 0..N-1, ready for compute_os to read.
        # signal_cache.inject(N)

        # Clear enterprise's internal delay cache on every signal collection.
        # enterprise caches get_delay() keyed only on params — so when residuals
        # change but params don't, it returns the stale cached delay from the
        # first call. Clearing forces it to recompute from current psr.residuals.
        for sc in pta._signalcollections:
            sc._cache_get_delay = {}
            sc._cache_list_get_delay = []

        # compute_os reads psr.residuals (just set above) and computes:
        #   OS  = sum_ab r_a^T C_a^{-1} S_ab C_b^{-1} r_b   (numerator)
        #   OS_sig = 1/sqrt(sum_ab tr(C_a^{-1} S_ab C_b^{-1} S_ba)) (denominator)
        # We discard the returned OS_sig and use the pre-computed one.
        ostat = opt_stat.OptimalStatistic(psrs_injected, pta=pta, orf='hd')
        xi, rho, sig, OS, _ = ostat.compute_os(params=params_out)
        snr = OS / OS_sig

        if toggle_memory_profiling:
            log_memory(f"  After compute_os N={N}")

        if profile:
            timing_list.append({'N': N, 'total': time.time() - t0_total})

        if detailed_output_SNR:
            os_details_cache[N] = {
                'xi':     xi.tolist(),
                'rho':    rho.tolist(),
                'sig':    sig.tolist(),
                'OS':     float(OS),
                'OS_sig': float(OS_sig),
            }

        gc.collect()

        if toggle_memory_profiling:
            log_memory(f"  After cleanup N={N}")

        snr_cache[N]     = snr
        N_tested_list.append(N)
        SNR_tested_list.append(snr)
        return snr

    # =====================================================================
    # Phase 1: Test initial guess
    # =====================================================================
    N_test   = min(N_initial_guess, len(population))
    snr_test = compute_and_cache(N_test)

    if verbose:
        print(f"  N = {N_test}: SNR = {snr_test:.3f}")

    if SNR_min <= snr_test <= SNR_max:
        search_direction = "verify"
        N_low, SNR_low   = 1, None
        N_high, SNR_high = N_test, snr_test
    elif snr_test < SNR_min:
        search_direction = "upward"
        N_low, SNR_low   = N_test, snr_test
        N_high, SNR_high = None, None
    else:
        search_direction = "downward"
        N_high, SNR_high = N_test, snr_test
        N_low,  SNR_low  = 1, None

    if verbose:
        status = ("in range!" if search_direction == "verify" 
                  else ("below" if search_direction == "upward" else "above") + " target")
        print(f"  Initial guess is {status}")
        print(f"  Searching {search_direction}...\n")

    # =====================================================================
    # Phase 2: Find bracketing points
    # =====================================================================
    expansion_count = 0
    max_expansions  = 6

    # Handle downward search: test N=1 before entering loop
    if search_direction == "downward":
        snr_at_1 = compute_and_cache(1)
        if verbose:
            print(f"  N = 1: SNR = {snr_at_1:.3f}")

        if snr_at_1 > SNR_max:
            if verbose:
                print(f"\n  ✗ BROKEN: SNR at N=1 ({snr_at_1:.3f}) exceeds target "
                      f"max ({SNR_max:.3f}) — check filter_population and OS_sig")
            return {
                'population': population[:1],
                'n_bininaries': 1,
                'SNR_achieved': float(snr_at_1),
                'SNR_target': SNR_range,
                'search_metadata': {
                    'N_tested': N_tested_list,
                    'SNR_tested': SNR_tested_list,
                    'iterations': 0, 'expansions': 0,
                    'used_cache': True,
                    'warning': 'SNR_at_N1_exceeds_target',
                    'broken': True,
                }
            }
        elif SNR_min <= snr_at_1 <= SNR_max:
            if verbose:
                print(f"  ✓ N=1 already in target range, done.")
            return {
                'population': population[:1],
                'n_bininaries': 1,
                'SNR_achieved': float(snr_at_1),
                'SNR_target': SNR_range,
                'search_metadata': {
                    'N_tested': N_tested_list,
                    'SNR_tested': SNR_tested_list,
                    'iterations': 0, 'expansions': 0,
                    'used_cache': True,
                    'warning': None, 'broken': False,
                }
            }
        else:
            N_low, SNR_low = 1, snr_at_1
            if verbose:
                print(f"  N=1 below target, bracket [{N_low}, {N_high}]")

    while expansion_count < max_expansions:
        if search_direction == "upward":
            N_high_target = int((N_high or N_low) * 1.5)

            while N_high_target > len(population):
                if expansion_count >= max_expansions:
                    N_high_target = len(population)
                    break

                expansion_count += 1
                N_to_add   = int(len(population) * 0.5)
                N_current  = len(population) + N_to_add

                if verbose:
                    print(f"  ⚠ Expanding population: +{N_to_add} (total {N_current})")

                config_add   = {**config_template, 'n_binaries': N_to_add}
                new_binaries = generate_population(config_add, smbhb_module,
                                        T_obs_years=Tspan/(365.25*86400))
                    
                signal_cache.extend_population(new_binaries)


            N_high    = N_high_target
            snr_high  = compute_and_cache(N_high)

            if verbose:
                print(f"  N = {N_high}: SNR = {snr_high:.3f}")

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
            if verbose:
                print(f"  N = {N_new}: SNR = {snr_new:.3f}")

            if snr_new < SNR_min:
                N_low, SNR_low = N_new, snr_new
                break
            N_high, SNR_high = N_new, snr_new

        else:  # verify
            break

    # =====================================================================
    # Phase 3: Verify bracket
    # =====================================================================
    if SNR_low is None or SNR_high is None:
        if verbose:
            print(f"\n  ⚠ WARNING: Could not bracket after {expansion_count} expansions")
    elif verbose:
        print(f"\n  ✓ Bracketed: N ∈ [{N_low}, {N_high}], "
              f"SNR ∈ [{SNR_low:.3f}, {SNR_high:.3f}]")

    # =====================================================================
    # Phase 4: Bisection
    # =====================================================================
    found_in_range = False
    iteration      = 0

    for iteration in range(max_iterations):
        if N_high - N_low <= 1:
            if verbose:
                print(f"✓ Bracket converged (N difference ≤ 1)")
            break

        if found_in_range and (N_high - N_low) / N_high <= convergence_threshold:
            if verbose:
                print(f"✓ Bracket converged (within "
                      f"{convergence_threshold*100:.0f}% of N={N_high})")
            break

        frac  = 0.15 if found_in_range else 0.5
        N_mid = int(N_low + frac * (N_high - N_low))
        N_mid = max(N_low + 1, min(N_mid, N_high - 1))

        snr_mid = compute_and_cache(N_mid)
        if verbose:
            print(f"  Iter {iteration+1}: N = {N_mid}, SNR = {snr_mid:.3f}")

        if snr_mid < SNR_min:
            N_low, SNR_low = N_mid, snr_mid
            found_in_range = False
        elif snr_mid > SNR_max:
            N_high, SNR_high = N_mid, snr_mid
            found_in_range = False
        else:
            found_in_range   = True
            N_high, SNR_high = N_mid, snr_mid

    # =====================================================================
    # Phase 5: Select final population
    # =====================================================================
    valid_indices = [i for i, snr in enumerate(SNR_tested_list)
                     if SNR_min <= snr <= SNR_max]

    if valid_indices:
        best_idx = min(valid_indices, key=lambda i: N_tested_list[i])
    else:
        above_indices = [i for i, snr in enumerate(SNR_tested_list)
                         if snr > SNR_max]
        if above_indices:
            best_idx = min(above_indices, key=lambda i: N_tested_list[i])
        else:
            range_mid = 0.5 * (SNR_min + SNR_max)
            best_idx  = min(range(len(SNR_tested_list)),
                            key=lambda i: abs(SNR_tested_list[i] - range_mid))

    N_final   = N_tested_list[best_idx]
    SNR_final = SNR_tested_list[best_idx]

    if verbose:
        print(f"\n✓ Population generated: N = {N_final}, SNR = {SNR_final:.3f}")
        print(f"  (Target range: [{SNR_min}, {SNR_max}])\n")

    result = {
        'population':      population[:N_final],
        'n_bininaries':    N_final,
        'SNR_achieved':    float(SNR_final),
        'SNR_target':      SNR_range,
        'search_metadata': {
            'N_tested':    N_tested_list,
            'SNR_tested':  SNR_tested_list,
            'iterations':  iteration + 1,
            'expansions':  expansion_count,
            'used_cache':  True,
        }
    }

    if detailed_output_SNR:
        result['os_details']         = os_details_cache.get(N_final)
        result['os_details_history'] = os_details_cache

    return result


def generate_snr_consistent_populations(
    config_template, smbhb_module, psrs_clean, detailed_noise_params, pulsar_noise_params_classified, Tspan,
    SNR_range, N_sims=20, N_initial_guess=2000, N_max_initial=10000,
    verbose=True, save_populations=True, profile=False,
    use_cache=True, cache_threshold=7000, batch_size=10000, toggle_memory_profiling=config.MEMORY_PROFILE_ENABLED,
    detailed_output_SNR = False
):
    """
    Generate multiple SMBHB populations, each consistent with target SNR range.
    
    This is the main function for generating an ensemble of populations that
    all produce SNR values within the specified range. All binary properties
    are preserved for later analysis.
    
    Parameters:
        config_template: base config dict for population generation
        smbhb_module: module containing binary evolution functions
        psrs_clean: clean pulsar data
        detailed_noise_params: full noise parameters from noise file
        detailed_noise_params: parsed noise parameters from noise file
        Tspan: observation timespan
        SNR_range: tuple (SNR_min, SNR_max) for target SNR range
        N_sims: number of populations to generate
        N_initial_guess: starting N value for search (consistent across sims)
        N_max_initial: initial population pool size
        verbose: print progress
        save_populations: if True, store full population objects; if False, 
                         store only summary statistics to save memory
        use_cache: if True, pre-compute signal cache for populations below cache_threshold
        cache_threshold: max population size for caching (default 50000)
        batch_size: if N_max_initial > cache_threshold, generate in batches of this size
    
    Returns:
        dict containing:
            'populations': list of population dicts (if save_populations=True)
            'summary_statistics': compiled stats across all populations
            'config': configuration used
            'metadata': timing and success information
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING SNR-CONSISTENT POPULATIONS")
        print(f"{'='*70}")
        print(f"Target SNR range: [{SNR_range[0]}, {SNR_range[1]}]")
        print(f"Number of simulations: {N_sims}")
        print(f"Initial guess (fixed): N = {N_initial_guess}")
        print(f"Max initial pool: N = {N_max_initial}")
        if use_cache:
            print(f"Caching: enabled (threshold = {cache_threshold})")
        else:
            print(f"Caching: disabled")
        if N_max_initial > cache_threshold:
            print(f"Batch generation: enabled (batch size = {batch_size})")
        print(f"{'='*70}\n")
    
    populations = []
    n_bininaries_list = []
    SNR_achieved_list = []
    success_count = 0
    
    for sim_idx in range(N_sims):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"SIMULATION {sim_idx + 1}/{N_sims}")
            print(f"{'─'*70}")
        
        result = generate_snr_consistent_population(
            config_template=config_template,
            smbhb_module=smbhb_module,
            psrs_clean=psrs_clean,
            detailed_noise_params=detailed_noise_params,
            pulsar_noise_params_classified=pulsar_noise_params_classified,
            Tspan=Tspan,
            SNR_range=SNR_range,
            N_initial_guess=N_initial_guess,
            N_max_initial=N_max_initial,
            verbose=verbose,
            profile=profile,
            use_cache=use_cache,
            cache_threshold=cache_threshold,
            batch_size=batch_size,
            toggle_memory_profiling=toggle_memory_profiling,
            detailed_output_SNR = detailed_output_SNR
        )
        
        if result is not None:
            success_count += 1
            n_bininaries_list.append(result['n_bininaries'])
            SNR_achieved_list.append(result['SNR_achieved'])
            
            # Add simulation index to result
            result['sim_index'] = sim_idx
            
            if save_populations:
                populations.append(result)
            else:
                # Store only summary to save memory
                summary = {
                    'sim_index': sim_idx,
                    'n_bininaries': result['n_bininaries'],
                    'SNR_achieved': result['SNR_achieved'],
                    'SNR_target': result['SNR_target']
                }
                populations.append(summary)
            
            if verbose:
                print(f"✓ Simulation {sim_idx + 1} SUCCESS")
        else:
            if verbose:
                print(f"✗ Simulation {sim_idx + 1} FAILED")
    
    # =====================================================================
    # Compile summary statistics
    # =====================================================================
    N_array = np.array(n_bininaries_list) if n_bininaries_list else np.array([])
    SNR_array = np.array(SNR_achieved_list) if SNR_achieved_list else np.array([])
    
    total_time = time.time() - start_time
    
    results = {
        'populations': populations,
        'summary_statistics': {
            'n_bininaries': {
                'mean': float(np.mean(N_array)) if len(N_array) > 0 else None,
                'median': float(np.median(N_array)) if len(N_array) > 0 else None,
                'std': float(np.std(N_array)) if len(N_array) > 0 else None,
                'min': int(np.min(N_array)) if len(N_array) > 0 else None,
                'max': int(np.max(N_array)) if len(N_array) > 0 else None,
                'all_values': n_bininaries_list
            },
            'SNR_achieved': {
                'mean': float(np.mean(SNR_array)) if len(SNR_array) > 0 else None,
                'median': float(np.median(SNR_array)) if len(SNR_array) > 0 else None,
                'std': float(np.std(SNR_array)) if len(SNR_array) > 0 else None,
                'min': float(np.min(SNR_array)) if len(SNR_array) > 0 else None,
                'max': float(np.max(SNR_array)) if len(SNR_array) > 0 else None,
                'all_values': SNR_achieved_list
            }
        },
        'config': {
            'SNR_range': SNR_range,
            'N_sims': N_sims,
            'N_initial_guess': N_initial_guess,
            'N_max_initial': N_max_initial,
            'use_cache': use_cache,
            'cache_threshold': cache_threshold,
            'batch_size': batch_size,
            'config_template': config_template
        },
        'metadata': {
            'success_count': success_count,
            'success_rate': success_count / N_sims,
            'total_time': total_time,
            'save_populations': save_populations
        }
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ENSEMBLE SUMMARY")
        print(f"{'='*70}")
        print(f"Success rate: {results['metadata']['success_rate']:.1%} ({success_count}/{N_sims})")
        
        if success_count > 0:
            print(f"\nNumber of binaries per population:")
            print(f"  Mean:   {results['summary_statistics']['n_bininaries']['mean']:.0f}")
            print(f"  Median: {results['summary_statistics']['n_bininaries']['median']:.0f}")
            print(f"  Std:    {results['summary_statistics']['n_bininaries']['std']:.0f}")
            print(f"  Range:  [{results['summary_statistics']['n_bininaries']['min']}, "
                  f"{results['summary_statistics']['n_bininaries']['max']}]")
            
            print(f"\nSNR achieved:")
            print(f"  Mean:   {results['summary_statistics']['SNR_achieved']['mean']:.3f}")
            print(f"  Median: {results['summary_statistics']['SNR_achieved']['median']:.3f}")
            print(f"  Std:    {results['summary_statistics']['SNR_achieved']['std']:.3f}")
            print(f"  Range:  [{results['summary_statistics']['SNR_achieved']['min']:.3f}, "
                  f"{results['summary_statistics']['SNR_achieved']['max']:.3f}]")
        
        print(f"\nTotal time: {total_time:.1f} s ({total_time/60:.1f} min)")
        print(f"{'='*70}\n")
    
    return results


def extract_binary_properties(populations_result, property_names=None):
    """
    Extract and compile binary properties across all populations.
    
    Parameters:
        populations_result: output from generate_snr_consistent_populations()
        property_names: list of property names to extract (e.g., ['mass1', 'mass2', 'distance'])
                       If None, attempts to extract all available properties
    
    Returns:
        dict with compiled properties across all populations
    """
    if not populations_result['metadata']['save_populations']:
        raise ValueError("Cannot extract properties - populations were not saved. "
                        "Run with save_populations=True")
    
    all_properties = {}
    
    for pop_result in populations_result['populations']:
        if 'population' not in pop_result:
            continue
        
        sim_idx = pop_result['sim_index']
        population = pop_result['population']
        
        # Extract properties from each binary
        for binary_idx, binary in enumerate(population):
            # If property_names not specified, try to get all attributes
            if property_names is None:
                property_names = [attr for attr in dir(binary) 
                                if not attr.startswith('_') and not callable(getattr(binary, attr))]
            
            for prop in property_names:
                if hasattr(binary, prop):
                    key = f"{prop}_sim{sim_idx}_bin{binary_idx}"
                    
                    if prop not in all_properties:
                        all_properties[prop] = []
                    
                    all_properties[prop].append({
                        'sim_index': sim_idx,
                        'binary_index': binary_idx,
                        'value': getattr(binary, prop)
                    })
    
    return all_properties


class LazyCumsumCache:
    """
    Computes per-binary GW signals on demand, in blocks, parallelised over pulsars.

    Design
    ------
    Rather than precomputing GW signals for all N_pool binaries upfront
    (expensive for N_pool ~ 180k), signals are computed lazily in blocks
    of `block_size` binaries only when a new N value is requested.

    At each block boundary a checkpoint is saved:
        checkpoints[psr_name] = [(N_boundary, cumsum_at_boundary), ...]
    so that get_signal(N) can find the nearest checkpoint below N and only
    recompute the tail (at most block_size individual r_k calls).

    Parallelism
    -----------
    Each block is computed in parallel across pulsars using ThreadPoolExecutor.
    The GW computation (numpy-dominated) releases the GIL so thread parallelism
    is effective. For 67 pulsars on 8 cores this gives ~5-6x block speedup.

    Memory
    ------
    Only checkpoint arrays are stored long-term: n_checkpoints * n_psr * n_toa * 8 bytes.
    For 67 pulsars, 3000 TOAs avg, block_size=2000, N_pool=180k:
        n_checkpoints = 90, memory ~ 90 * 67 * 3000 * 8 ~ 145 MB  (manageable)
    """

    def __init__(self, psrs, population, block_size=2000, n_workers=4,
                 max_memory_mb=200, verbose=False):
        self.psrs       = psrs
        self.population = population
        self.block_size = block_size
        self.n_workers  = n_workers
        self.verbose    = verbose

        # Pre-extract TOA arrays and build pulsar lookup — avoids generator
        # searches inside threads which are slow and not thread-safe
        self.toas    = {psr.name: np.asarray(psr.toas, dtype=np.float64)
                        for psr in psrs}
        self.psr_map = {psr.name: psr for psr in psrs}

        # Auto chunk size per pulsar for vectorised GW computation
        self.cs = {psr.name: _auto_chunk_size(len(psr.toas), max_memory_mb)
                   for psr in psrs}

        # Checkpoints: list of (N_at_boundary, cumsum_array) per pulsar
        # Initialised with a zero checkpoint at N=0
        zeros = {psr.name: np.zeros(len(psr.toas), dtype=np.float64)
                 for psr in psrs}
        self.checkpoints = {psr.name: [(0, zeros[psr.name].copy())]
                            for psr in psrs}

        # Running cumulative sum — updated as blocks are computed
        self.running_sum = {psr.name: np.zeros(len(psr.toas), dtype=np.float64)
                            for psr in psrs}

        # How many binaries have been fully processed into checkpoints
        self.N_computed = 0

    def _block_signal_for_psr(self, psr_name, block_start, block_end):
        """
        Compute the SUM of GW residuals from population[block_start:block_end]
        for one pulsar. Returns (n_toa,) float64.

        Called inside ThreadPoolExecutor — uses only pre-extracted arrays
        (self.toas, self.psr_map) which are read-only and thread-safe.
        """
        t_sec = self.toas[psr_name]
        psr   = self.psr_map[psr_name]
        block = self.population[block_start:block_end]
        cs    = self.cs[psr_name]

        if len(block) <= cs:
            return _gw_residuals_vec(t_sec, psr, block).astype(np.float64)
        else:
            return _gw_residuals_chunked(t_sec, psr, block, cs).astype(np.float64)

    def ensure_computed_up_to(self, N):
        """
        Ensure checkpoint data exists for all binaries up to index N.
        Computes any missing blocks, saving a checkpoint at each block boundary.
        Is a no-op if N <= self.N_computed.
        """
        N = min(N, len(self.population))
        if N <= self.N_computed:
            return

        start = self.N_computed
        end   = N

        if self.verbose:
            n_new_blocks = int(np.ceil((end - start) / self.block_size))
            print(f"  [Cache] Computing blocks for binaries {start}→{end} "
                  f"({n_new_blocks} blocks × {len(self.psrs)} pulsars)...")

        psr_names = [psr.name for psr in self.psrs]

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for block_start in range(start, end, self.block_size):
                block_end = min(block_start + self.block_size, end)

                # Submit all pulsars for this block in parallel
                futures = {
                    name: executor.submit(
                        self._block_signal_for_psr, name, block_start, block_end
                    )
                    for name in psr_names
                }

                # Collect and accumulate — futures.result() blocks until done
                for name in psr_names:
                    self.running_sum[name] += futures[name].result()

                # Checkpoint at block boundary
                N_boundary = block_end
                for name in psr_names:
                    self.checkpoints[name].append(
                        (N_boundary, self.running_sum[name].copy())
                    )

        self.N_computed = end

        if self.verbose:
            print(f"  [Cache] ✓ Ready up to N={end}")

    def get_signal(self, N, psr_name):
        """
        Return total GW residual from first N binaries for psr_name.

        Finds the largest checkpoint <= N, then adds individual r_k calls
        for the tail [checkpoint_N, N). Tail length <= block_size.
        """
        self.ensure_computed_up_to(N)

        checkpoints = self.checkpoints[psr_name]

        # Binary search for largest checkpoint index <= N
        lo, hi = 0, len(checkpoints) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if checkpoints[mid][0] <= N:
                lo = mid
            else:
                hi = mid - 1

        N_cp, cumsum_cp = checkpoints[lo]

        if N_cp == N:
            return cumsum_cp.copy()

        # Add tail: individual binaries from N_cp to N
        # At most block_size calls — fast even without vectorisation
        t_sec  = self.toas[psr_name]
        psr    = self.psr_map[psr_name]
        extra  = np.zeros(len(t_sec), dtype=np.float64)
        for binary in self.population[N_cp:N]:
            extra += r_k(t_sec, psr, binary)

        return cumsum_cp + extra

    def inject(self, N):
        """
        Set psr._residuals for all pulsars to GW signal from first N binaries.
        This is what compute_and_cache calls before each ostat.compute_os().
        """
        self.ensure_computed_up_to(N)
        for psr in self.psrs:
            psr._residuals = self.get_signal(N, psr.name)

    def extend_population(self, new_binaries):
        """
        Append new binaries. Their signals are computed lazily on next inject().
        N_computed stays unchanged — new signals only computed when needed.
        """
        self.population.extend(new_binaries)
        if self.verbose:
            print(f"  [Cache] Population extended to {len(self.population)} binaries "
                  f"(computed up to {self.N_computed})")