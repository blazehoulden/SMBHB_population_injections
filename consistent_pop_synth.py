import numpy as np
import time
from signal_injection import precompute_binary_signals, inject_population_subset_cached, inject_population_into_psrs
from pta_builder import build_pta_and_params
from enterprise_extensions.frequentist import optimal_statistic as opt_stat

def compute_population_snr(population, psrs_clean, params, Tspan, verbose=False, timer=False, profile=False):
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
            psrs_clean, population, pure_signal=True, verbose=False
        )
        if profile:
            t_inject = time.time() - t0
        
        # PROFILE: PTA building step
        if profile:
            t0 = time.time()
        pta, _, params_out = build_pta_and_params(
            psrs=psrs_injected, noise_params_15yr=params, 
            Tspan=Tspan, use_efac_only=True
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
    config_template, smbhb_module, psrs_clean, params, Tspan,
    SNR_range, N_initial_guess=2000, N_max_initial=10000,
    max_iterations=20, tolerance=0.05, verbose=True, profile=False,
    use_cache=True, cache_threshold=7000, batch_size=10000
):
    """
    Generate a single SMBHB population consistent with target SNR range.
    
    This function searches for the minimum number of binaries needed to achieve
    an SNR within the specified range, then returns that population along with
    all binary properties.
    
    Parameters:
        config_template: base config dict for population generation
        smbhb_module: module containing binary evolution functions
        psrs_clean: clean pulsar data
        params: noise parameters
        Tspan: observation timespan
        SNR_range: tuple (SNR_min, SNR_max) for target SNR range
        N_initial_guess: starting N value for search
        N_max_initial: initial population pool size
        max_iterations: max bisection iterations
        tolerance: relative convergence criterion
        verbose: print progress
        use_cache: if True, pre-compute signal cache for populations below cache_threshold
        cache_threshold: max population size for caching (default 50000)
        batch_size: if N_max_initial > cache_threshold, generate in batches of this size
    
    Returns:
        dict containing:
            'population': list of binary objects (with all properties)
            'N_binaries': number of binaries in consistent population
            'SNR_achieved': actual SNR value achieved
            'SNR_target': target SNR range
            'search_metadata': dict with search history
    """
    from config import generate_population
    
    SNR_min, SNR_max = SNR_range
    if SNR_min >= SNR_max:
        raise ValueError(f"SNR_range must be (min, max) with min < max, got {SNR_range}")
    
    # =====================================================================
    # Phase 0: Generate initial population pool
    # =====================================================================
    N_current = N_max_initial
    
    if verbose:
        print(f"\nGenerating SNR-consistent population...")
        print(f"Target SNR range: [{SNR_min}, {SNR_max}]")
        print(f"Initial guess: N = {N_initial_guess}")
        print(f"Generating initial population pool: N = {N_current}")
        if use_cache and N_current <= cache_threshold:
            print(f"Caching enabled (threshold: {cache_threshold})")
        else:
            print(f"Caching disabled (population size {N_current} > threshold {cache_threshold})")
        if N_current > cache_threshold:
            print(f"Using batch generation (batch size: {batch_size})")
    
    # Generate population - use batching for large populations
    if N_current > batch_size:
        if verbose:
            print(f"Generating population in batches...")
        population = []
        n_batches = int(np.ceil(N_current / batch_size))
        for batch_idx in range(n_batches):
            batch_n = min(batch_size, N_current - len(population))
            if verbose:
                print(f"  Batch {batch_idx+1}/{n_batches}: generating {batch_n} binaries...")
            config_batch = {**config_template, 'N_binaries': batch_n}
            population.extend(generate_population(config_batch, smbhb_module))
    else:
        config = {**config_template, 'N_binaries': N_current}
        population = generate_population(config, smbhb_module)
    
    # =====================================================================
    # Optionally pre-compute signal cache
    # =====================================================================
    signal_cache = None
    use_cached_injection = use_cache and N_current <= cache_threshold
    
    if use_cached_injection:
        if verbose:
            print(f"Pre-computing binary signals for caching...")
        signal_cache = precompute_binary_signals(psrs_clean, population)
        if verbose:
            print(f"✓ Signal cache ready\n")
    else:
        if verbose:
            print(f"✓ Population ready (no caching)\n")
    
    # SNR cache: N -> SNR
    snr_cache = {}
    N_tested_list = []
    SNR_tested_list = []
    timing_list = []
    
    def compute_and_cache(N):
        """Compute SNR for N binaries, use cache if available."""
        if N in snr_cache:
            return snr_cache[N]
        
        if N > len(population):
            raise ValueError(f"N={N} exceeds population size {len(population)}")
        
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        
        if profile:
            t0 = time.time()
        
        # Choose injection method based on caching
        if use_cached_injection and signal_cache is not None:
            # Use pre-computed cached signals
            psrs_injected = inject_population_subset_cached(
                psrs_clean, population, N,
                psrs_injected_cache=signal_cache,
                pure_signal=True, verbose=False
            )
        else:
            # Compute signals on-the-fly
            subset_population = population[:N]
            psrs_injected = inject_population_into_psrs(
                psrs_clean, subset_population,
                pure_signal=True, verbose=False
            )
        if profile:
            t_inject = time.time() - t0

        if profile:
            t0 = time.time()
        # Build PTA and compute OS
        pta, _, params_out = build_pta_and_params(
            psrs=psrs_injected, noise_params_15yr=params, 
            Tspan=Tspan, use_efac_only=True
        )
        if profile:
            t_pta = time.time() - t0
        
        if profile:
            t0 = time.time()
        ostat = opt_stat.OptimalStatistic(psrs_injected, pta=pta, orf='hd')
        if profile:
            t_ostat_init = time.time() - t0

        if profile:
            t0 = time.time()
        xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
        if profile:
            t_compute_os = time.time() - t0
        snr = OS / OS_sig
        
        if profile:
            timing = {
                'inject': t_inject,
                'pta': t_pta,
                'ostat_init': t_ostat_init,
                'compute_os': t_compute_os,
                'total': t_inject + t_pta + t_ostat_init + t_compute_os
            }
            timing_list.append({'N': N, 'timing': timing})
        snr_cache[N] = snr
        N_tested_list.append(N)
        SNR_tested_list.append(snr)
        return snr
    
    # =====================================================================
    # Phase 1: Test initial guess and determine search direction
    # =====================================================================
    N_test = min(N_initial_guess, N_current)
    snr_test = compute_and_cache(N_test)
    
    if verbose:
        print(f"  N = {N_test}: SNR = {snr_test:.3f}")
    
    # Determine search direction
    if SNR_min <= snr_test <= SNR_max:
        search_direction = "verify"
        N_low = 1
        SNR_low = None
        N_high = N_test
        SNR_high = snr_test
    elif snr_test < SNR_min:
        search_direction = "upward"
        N_low = N_test
        SNR_low = snr_test
        N_high = None
        SNR_high = None
    else:  # snr_test > SNR_max
        search_direction = "downward"
        N_high = N_test
        SNR_high = snr_test
        N_low = 1
        SNR_low = None
    
    if verbose:
        status = "in range!" if search_direction == "verify" else ("below" if search_direction == "upward" else "above") + " target"
        print(f"  Initial guess is {status}")
        print(f"  Searching {search_direction}...\n")
    
    # =====================================================================
    # Phase 2: Find bracketing points
    # =====================================================================
    expansion_count = 0
    max_expansions = 6
    
    while expansion_count < max_expansions:
        if search_direction == "upward":
            if N_high is None:
                N_high = min(N_current, int(N_low * 1.5))
            else:
                N_high = min(N_current, int(N_high * 1.5))
            
            snr_high = compute_and_cache(N_high)
            if verbose:
                print(f"  N = {N_high}: SNR = {snr_high:.3f}")
            
            # Check if bracketed
            if SNR_low < SNR_min and snr_high > SNR_max:
                SNR_high = snr_high
                break
            
            # Expand if needed
            if N_high >= N_current:
                expansion_count += 1
                N_to_add = int(N_current * 0.5)
                N_current += N_to_add
                if verbose:
                    print(f"  ⚠ Expanding population: +{N_to_add} (total {N_current})")
                
                # Generate additional binaries
                config_add = {**config_template, 'N_binaries': N_to_add}
                new_binaries = generate_population(config_add, smbhb_module)
                population.extend(new_binaries)
                
                # If we were using cache and still below threshold, update cache
                if use_cached_injection and N_current <= cache_threshold and signal_cache is not None:
                    if verbose:
                        print(f"  Updating signal cache...")
                    # Need to recompute cache for new total population
                    signal_cache = precompute_binary_signals(psrs_clean, population)
                elif use_cached_injection and N_current > cache_threshold:
                    # Population grew beyond cache threshold, disable caching
                    if verbose:
                        print(f"  Population exceeded cache threshold, disabling cache")
                    signal_cache = None
                    use_cached_injection = False
        
        elif search_direction == "downward":
            if N_low == 1 and SNR_low is None:
                snr_low = compute_and_cache(1)
                if verbose:
                    print(f"  N = 1: SNR = {snr_low:.3f}")
                
                if snr_low < SNR_min and SNR_high > SNR_max:
                    SNR_low = snr_low
                    break
            else:
                N_new = max(1, (N_low + N_high) // 2)
                if N_new == N_low or N_new == N_high:
                    break
                
                snr_new = compute_and_cache(N_new)
                if verbose:
                    print(f"  N = {N_new}: SNR = {snr_new:.3f}")
                
                if snr_new < SNR_min:
                    N_low = N_new
                    SNR_low = snr_new
                    break
                
                N_high = N_new
                SNR_high = snr_new
    
    # =====================================================================
    # Phase 3: Verify bracket
    # =====================================================================
    if SNR_low is None or SNR_high is None:
        if verbose:
            print(f"✗ Could not bracket target range")
        return None
    
    if not (SNR_low < SNR_min and SNR_high > SNR_max):
        if verbose:
            print(f"✗ Invalid bracket for range [{SNR_min}, {SNR_max}]")
        return None
    
    if verbose:
        print(f"✓ Bracket found: N=[{N_low}, {N_high}], SNR=[{SNR_low:.3f}, {SNR_high:.3f}]\n")
    
    # =====================================================================
    # Phase 4: Bisection to find minimum N in range
    # =====================================================================
    found_in_range = False
    
    for iteration in range(max_iterations):
        if N_high - N_low <= 1:
            if verbose:
                print(f"✓ Bracket converged")
            break
        
        # Adaptive step size
        if found_in_range:
            frac = 0.15  # Small steps when in range
        else:
            frac = 0.5  # Normal bisection
        
        N_mid = int(N_low + frac * (N_high - N_low))
        N_mid = max(N_low + 1, min(N_mid, N_high - 1))
        
        snr_mid = compute_and_cache(N_mid)
        if verbose:
            print(f"  Iter {iteration+1}: N = {N_mid}, SNR = {snr_mid:.3f}")
        
        # Update bracket
        if snr_mid < SNR_min:
            N_low, SNR_low = N_mid, snr_mid
            found_in_range = False
        elif snr_mid > SNR_max:
            N_high, SNR_high = N_mid, snr_mid
            found_in_range = False
        else:
            # In range! Search downward for minimum
            found_in_range = True
            N_high, SNR_high = N_mid, snr_mid
    
    # =====================================================================
    # Phase 5: Select final population
    # =====================================================================
    # Find lowest N where SNR is in range
    valid_indices = [i for i, snr in enumerate(SNR_tested_list) 
                    if SNR_min <= snr <= SNR_max]
    
    if valid_indices:
        best_idx = min(valid_indices, key=lambda i: N_tested_list[i])
        N_final = N_tested_list[best_idx]
        SNR_final = SNR_tested_list[best_idx]
    else:
        # Use closest to range midpoint
        range_mid = (SNR_min + SNR_max) / 2
        closest_idx = min(range(len(SNR_tested_list)), 
                        key=lambda i: abs(SNR_tested_list[i] - range_mid))
        N_final = N_tested_list[closest_idx]
        SNR_final = SNR_tested_list[closest_idx]
    
    if verbose:
        print(f"\n✓ Population generated: N = {N_final}, SNR = {SNR_final:.3f}")
        print(f"  (Target range: [{SNR_min}, {SNR_max}])\n")
    
    # Return the first N_final binaries from the population
    final_population = population[:N_final]
    
    return {
        'population': final_population,
        'N_binaries': N_final,
        'SNR_achieved': float(SNR_final),
        'SNR_target': SNR_range,
        'search_metadata': {
            'N_tested': N_tested_list,
            'SNR_tested': SNR_tested_list,
            'iterations': iteration + 1,
            'expansions': expansion_count,
            'used_cache': signal_cache is not None
        }
    }


def generate_snr_consistent_populations(
    config_template, smbhb_module, psrs_clean, params, Tspan,
    SNR_range, N_sims=20, N_initial_guess=2000, N_max_initial=10000,
    verbose=True, save_populations=True, profile=False,
    use_cache=True, cache_threshold=7000, batch_size=10000
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
        params: noise parameters
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
    N_binaries_list = []
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
            params=params,
            Tspan=Tspan,
            SNR_range=SNR_range,
            N_initial_guess=N_initial_guess,
            N_max_initial=N_max_initial,
            verbose=verbose,
            profile=profile,
            use_cache=use_cache,
            cache_threshold=cache_threshold,
            batch_size=batch_size
        )
        
        if result is not None:
            success_count += 1
            N_binaries_list.append(result['N_binaries'])
            SNR_achieved_list.append(result['SNR_achieved'])
            
            # Add simulation index to result
            result['sim_index'] = sim_idx
            
            if save_populations:
                populations.append(result)
            else:
                # Store only summary to save memory
                summary = {
                    'sim_index': sim_idx,
                    'N_binaries': result['N_binaries'],
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
    N_array = np.array(N_binaries_list) if N_binaries_list else np.array([])
    SNR_array = np.array(SNR_achieved_list) if SNR_achieved_list else np.array([])
    
    total_time = time.time() - start_time
    
    results = {
        'populations': populations,
        'summary_statistics': {
            'N_binaries': {
                'mean': float(np.mean(N_array)) if len(N_array) > 0 else None,
                'median': float(np.median(N_array)) if len(N_array) > 0 else None,
                'std': float(np.std(N_array)) if len(N_array) > 0 else None,
                'min': int(np.min(N_array)) if len(N_array) > 0 else None,
                'max': int(np.max(N_array)) if len(N_array) > 0 else None,
                'all_values': N_binaries_list
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
            print(f"  Mean:   {results['summary_statistics']['N_binaries']['mean']:.0f}")
            print(f"  Median: {results['summary_statistics']['N_binaries']['median']:.0f}")
            print(f"  Std:    {results['summary_statistics']['N_binaries']['std']:.0f}")
            print(f"  Range:  [{results['summary_statistics']['N_binaries']['min']}, "
                  f"{results['summary_statistics']['N_binaries']['max']}]")
            
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