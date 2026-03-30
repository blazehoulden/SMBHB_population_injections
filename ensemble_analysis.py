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
            psrs_clean, population, pure_signal=True, verbose=False,
            pulsar_noise_params=params
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
        if verbose:
            print(f"  N={len(population)}: FAILED - {e}")
        return np.nan

def find_N_binaries_for_target_snr(
    config_template, smbhb_module, psrs_clean, params, Tspan,
    target_SNR=3.75, SNR_range=None, N_initial_guess=None, N_max_initial=None,
    max_iterations=20, tolerance=0.05, verbose=True, profile=False,
    use_cache=True, cache_threshold=7000, batch_size=10000
):
    """
    Find the MINIMUM N_binaries reaching target SNR via adaptive bisection.
    
    Key improvements:
    - Always searches for the MINIMUM N (not just any N close to target)
    - Consistent initial guesses across realisations
    - Clear bracketing: N_low (below target) and N_high (above target)
    - If initial guess overshoots, searches downward first
    - Efficient caching avoids redundant computations (when enabled)
    - Transparent logic throughout
    - Optional SNR_range for consistency checking
    - Memory-efficient options for large populations
    
    Parameters:
        config_template: base config dict
        target_SNR: target SNR value (positive or negative)
        SNR_range: tuple (SNR_min, SNR_max) for acceptable range, or None for point target
                   If provided, searches for N where SNR falls within this range
        N_initial_guess: starting N value (same for all realisations)
        N_max_initial: initial population pool size
        tolerance: relative convergence criterion (default 5%)
        max_iterations: max bisection iterations
        verbose: print progress
        profile: print detailed timing breakdown for each SNR computation
        use_cache: if True, pre-compute signal cache for populations below cache_threshold
        cache_threshold: max population size for caching (default 50000)
        batch_size: if N_max_initial > cache_threshold, generate in batches of this size
    
    Returns:
        N_needed: minimum N reaching target SNR (or None if impossible)
        metadata: dict with test history and convergence info
    """
    from config import generate_population
    
    if target_SNR == 0:
        raise ValueError("target_SNR must be non-zero")
    
    if SNR_range is None:
        SNR_min = SNR_max = target_SNR
        use_range = False
    else:
        SNR_min, SNR_max = SNR_range
        use_range = True
        if SNR_min >= SNR_max:
            raise ValueError(f"SNR_range must be (min, max) with min < max, got {SNR_range}")
    
    if N_initial_guess is None:
        N_initial_guess = 2000
    
    if N_max_initial is None:
        N_max_initial = config_template.get('N_binaries', 10000)
    
    # =====================================================================
    # Phase 0: Generate initial population pool
    # =====================================================================
    N_current = N_max_initial
    
    if verbose:
        if use_range:
            print(f"\nSearching for minimum N_binaries with SNR in range [{SNR_min}, {SNR_max}]")
        else:
            print(f"\nSearching for minimum N_binaries to reach SNR = {target_SNR}")
        print(f"Initial guess: N = {N_initial_guess}")
        print(f"Generating initial population pool: N = {N_current}")
        if use_cache and N_current <= cache_threshold:
            print(f"Caching enabled (threshold: {cache_threshold})")
        else:
            print(f"Caching disabled (population size {N_current} > threshold {cache_threshold})")
        if N_current > cache_threshold:
            print(f"Using batch generation (batch size: {batch_size})")
        if profile:
            print(f"[PROFILING ENABLED - detailed timing breakdown will be shown]\n")
    
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
            population.extend(generate_population(config_batch, smbhb_module, T_obs_seconds=Tspan))
    else:
        config = {**config_template, 'N_binaries': N_current}
        population = generate_population(config, smbhb_module, T_obs_seconds=Tspan)
    
    # =====================================================================
    # Optionally pre-compute signal cache
    # =====================================================================
    signal_cache = None
    use_cached_injection = use_cache and N_current <= cache_threshold
    
    if use_cached_injection:
        if verbose:
            print(f"Pre-computing individual binary signals for caching...")
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
    timing_list = []  # Track timing for each computation
    
    def compute_and_cache(N):
        """Compute SNR for N binaries, use cache if available."""
        if N in snr_cache:
            return snr_cache[N]
        
        if N > len(population):
            raise ValueError(f"N={N} exceeds population size {len(population)}")
        
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        
        # Choose injection method based on caching
        if use_cached_injection and signal_cache is not None:
            # Use pre-computed cached signals
            psrs_injected = inject_population_subset_cached(
                psrs_clean, population, N,
                psrs_injected_cache=signal_cache,
                pure_signal=True, verbose=False,
                pulsar_noise_params=params
            )
        else:
            # Compute signals on-the-fly
            subset_population = population[:N]
            psrs_injected = inject_population_into_psrs(
                psrs_clean, subset_population,
                pure_signal=True, verbose=False,
                pulsar_noise_params=params
            )
        
        # Now build PTA and compute OS
        if profile:
            t0 = time.time()
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
        
        if profile or True:  # Always time compute_os
            t0 = time.time()
        xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
        if profile or True:
            t_compute_os = time.time() - t0
        
        snr = OS / OS_sig
        
        if profile:
            # Track timing
            timing = {
                'pta': t_pta,
                'ostat_init': t_ostat_init,
                'compute_os': t_compute_os,
                'total': t_pta + t_ostat_init + t_compute_os
            }
            timing_list.append({'N': N, 'timing': timing})
        
        snr_cache[N] = snr
        N_tested_list.append(N)
        SNR_tested_list.append(snr)
        return snr
    
    # =====================================================================
    # Phase 1: Test initial guess and determine bracketing strategy
    # =====================================================================
    N_test = min(N_initial_guess, N_current)
    snr_test = compute_and_cache(N_test)
    
    if verbose:
        print(f"  N = {N_test}: SNR = {snr_test:.3f}")
        if profile and timing_list:
            t = timing_list[-1]['timing']
            print(f"    └─ pta: {t['pta']:.3f}s | ostat_init: {t['ostat_init']:.3f}s | compute_os: {t['compute_os']:.3f}s | total: {t['total']:.3f}s")
    
    # Determine if we need to search upward or downward
    target_positive = target_SNR > 0
    
    # Check if initial guess already brackets or overshoots
    if use_range:
        # For range: check if SNR is within, below, or above range
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
    else:
        # Original point-target logic
        if target_positive:
            if snr_test >= target_SNR:
                search_direction = "downward"
                N_high = N_test
                SNR_high = snr_test
                N_low = 1
                SNR_low = None
            else:
                search_direction = "upward"
                N_low = N_test
                SNR_low = snr_test
                N_high = None
                SNR_high = None
        else:
            if snr_test <= target_SNR:
                search_direction = "downward"
                N_high = N_test
                SNR_high = snr_test
                N_low = 1
                SNR_low = None
            else:
                search_direction = "upward"
                N_low = N_test
                SNR_low = snr_test
                N_high = None
                SNR_high = None
    
    if verbose:
        print(f"  Initial guess {'is in range!' if search_direction == 'verify' else 'is ' + ('OVERSHOOTS' if search_direction == 'downward' else 'below') + ' target'}")
        print(f"  Searching {search_direction}...\n")
    
    # =====================================================================
    # Phase 2: Expand population and find bracketing points
    # =====================================================================
    expansion_count = 0
    max_expansions = 6
    
    def is_valid_bracket(N_low, SNR_low, N_high, SNR_high, target_range_min, target_range_max):
        """Check if bracket properly straddles the target range."""
        if use_range:
            below = SNR_low < target_range_min
            above = SNR_high > target_range_max
            return below and above
        else:
            if target_range_max > 0:
                return SNR_low < target_range_max and SNR_high >= target_range_max
            else:
                return SNR_high <= target_range_max and SNR_low > target_range_max
    
    while expansion_count < max_expansions:
        if search_direction == "upward":
            if N_high is None:
                N_high = min(N_current, int(N_low * 1.5))
            else:
                N_high = min(N_current, int(N_high * 1.5))
            
            snr_high = compute_and_cache(N_high)
            if verbose:
                print(f"  N = {N_high}: SNR = {snr_high:.3f}")
                if profile and timing_list:
                    t = timing_list[-1]['timing']
                    print(f"    └─ pta: {t['pta']:.3f}s | ostat_init: {t['ostat_init']:.3f}s | compute_os: {t['compute_os']:.3f}s | total: {t['total']:.3f}s")
            
            if is_valid_bracket(N_low, SNR_low, N_high, snr_high, SNR_min, SNR_max):
                SNR_high = snr_high
                break
            
            if N_high >= N_current:
                expansion_count += 1
                N_to_add = int(N_current * 0.5)
                N_current += N_to_add
                if verbose:
                    print(f"  ⚠ Expanding population: +{N_to_add} (total {N_current})")
                
                config_add = {**config_template, 'N_binaries': N_to_add}
                new_binaries = generate_population(config_add, smbhb_module, T_obs_seconds=Tspan)
                population.extend(new_binaries)
                
                if use_cached_injection and N_current <= cache_threshold and signal_cache is not None:
                    if verbose:
                        print(f"  Updating signal cache...")
                    signal_cache = precompute_binary_signals(psrs_clean, population)
                elif use_cached_injection and N_current > cache_threshold:
                    if verbose:
                        print(f"  Population exceeded cache threshold, disabling cache")
                    signal_cache = None
                    use_cached_injection = False
        
        else:  # search_direction == "downward"
            if N_low == 1 and SNR_low is None:
                snr_low = compute_and_cache(1)
                if verbose:
                    print(f"  N = 1: SNR = {snr_low:.3f}")
                    if profile and timing_list:
                        t = timing_list[-1]['timing']
                        print(f"    └─ pta: {t['pta']:.3f}s | ostat_init: {t['ostat_init']:.3f}s | compute_os: {t['compute_os']:.3f}s | total: {t['total']:.3f}s")
                
                if target_positive:
                    if snr_low < target_SNR <= SNR_high:
                        SNR_low = snr_low
                        break
                else:
                    if SNR_high <= target_SNR < snr_low:
                        SNR_low = snr_low
                        break
            else:
                N_new = max(1, (N_low + N_high) // 2)
                if N_new == N_low or N_new == N_high:
                    break
                
                snr_new = compute_and_cache(N_new)
                if verbose:
                    print(f"  N = {N_new}: SNR = {snr_new:.3f}")
                    if profile and timing_list:
                        t = timing_list[-1]['timing']
                        print(f"    └─ pta: {t['pta']:.3f}s | ostat_init: {t['ostat_init']:.3f}s | compute_os: {t['compute_os']:.3f}s | total: {t['total']:.3f}s")
                
                if target_positive:
                    if snr_new < target_SNR:
                        N_low = N_new
                        SNR_low = snr_new
                        break
                else:
                    if snr_new > target_SNR:
                        N_low = N_new
                        SNR_low = snr_new
                        break
                
                N_high = N_new
                SNR_high = snr_new
    
    # =====================================================================
    # Phase 3: Verify we have a valid bracket
    # =====================================================================
    if SNR_low is None or SNR_high is None:
        if verbose:
            print(f"✗ Could not bracket target after expansions")
            print(f"  Current range tested: {min(N_tested_list)} to {max(N_tested_list)}")
        return None, {
            'N_tested': N_tested_list,
            'SNR_tested': SNR_tested_list,
            'bracket_found': False,
            'expansions': expansion_count,
            'used_cache': signal_cache is not None
        }
    
    # Verify bracket makes sense
    if use_range:
        if not (SNR_low < SNR_min and SNR_high >= SNR_max):
            if verbose:
                print(f"✗ Invalid bracket for range [{SNR_min}, {SNR_max}]")
                print(f"  Got: SNR_low={SNR_low:.3f}, SNR_high={SNR_high:.3f}")
            return None, {
                'N_tested': N_tested_list,
                'SNR_tested': SNR_tested_list,
                'bracket_found': False,
                'expansions': expansion_count,
                'used_cache': signal_cache is not None
            }
    else:
        if target_positive:
            if not (SNR_low < target_SNR <= SNR_high):
                if verbose:
                    print(f"✗ Invalid bracket: SNR_low={SNR_low:.3f}, target={target_SNR}, SNR_high={SNR_high:.3f}")
                return None, {
                    'N_tested': N_tested_list,
                    'SNR_tested': SNR_tested_list,
                    'bracket_found': False,
                    'expansions': expansion_count,
                    'used_cache': signal_cache is not None
                }
        else:
            if not (SNR_high <= target_SNR < SNR_low):
                if verbose:
                    print(f"✗ Invalid bracket: SNR_high={SNR_high:.3f}, target={target_SNR}, SNR_low={SNR_low:.3f}")
                return None, {
                    'N_tested': N_tested_list,
                    'SNR_tested': SNR_tested_list,
                    'bracket_found': False,
                    'expansions': expansion_count,
                    'used_cache': signal_cache is not None
                }
    
    if verbose:
        print(f"✓ Bracket found: N=[{N_low}, {N_high}], SNR=[{SNR_low:.3f}, {SNR_high:.3f}]\n")
    
    # =====================================================================
    # Phase 4: Bisection refinement
    # =====================================================================
    found_in_range = False
    
    def adaptive_step_size(snr_mid, snr_target_min, snr_target_max):
        """Compute adaptive step size based on distance from target range."""
        if snr_target_min <= snr_mid <= snr_target_max:
            distance = 0
        elif snr_mid < snr_target_min:
            distance = snr_target_min - snr_mid
        else:
            distance = snr_mid - snr_target_max
        
        range_width = abs(snr_target_max - snr_target_min)
        if range_width == 0:
            range_width = 1
        
        normalized_distance = distance / range_width
        
        if normalized_distance >= 3.0:
            frac = 0.7
        elif normalized_distance <= 0.0:
            frac = 0.1
        else:
            frac = 0.1 + (normalized_distance / 3.0) * 0.6
        
        return frac
    
    for iteration in range(max_iterations):
        if N_high - N_low <= 1:
            if verbose:
                print(f"✓ Bracket converged to adjacent values")
            break
        
        if use_range:
            if found_in_range:
                frac = 0.15
            else:
                if SNR_low < SNR_min:
                    ref_snr = SNR_low
                else:
                    ref_snr = SNR_high
                frac = adaptive_step_size(ref_snr, SNR_min, SNR_max)
        else:
            ref_snr = (SNR_low + SNR_high) / 2
            frac = adaptive_step_size(ref_snr, target_SNR - abs(target_SNR) * 0.1, 
                                     target_SNR + abs(target_SNR) * 0.1)
        
        N_mid = int(N_low + frac * (N_high - N_low))
        N_mid = max(N_low + 1, min(N_mid, N_high - 1))
        
        snr_mid = compute_and_cache(N_mid)
        if verbose:
            print(f"  Iter {iteration+1}: N = {N_mid}, SNR = {snr_mid:.3f} (step frac: {frac:.2f})")
            if profile and timing_list:
                t = timing_list[-1]['timing']
                print(f"    └─ pta: {t['pta']:.3f}s | ostat_init: {t['ostat_init']:.3f}s | compute_os: {t['compute_os']:.3f}s | total: {t['total']:.3f}s")
        
        if abs(snr_mid - target_SNR) / abs(target_SNR) < tolerance:
            if verbose:
                print(f"✓ SNR converged to within {tolerance*100:.1f}%")
            break
        
        if use_range:
            if snr_mid < SNR_min:
                N_low, SNR_low = N_mid, snr_mid
                found_in_range = False
            elif snr_mid > SNR_max:
                N_high, SNR_high = N_mid, snr_mid
                found_in_range = False
            else:
                found_in_range = True
                N_high, SNR_high = N_mid, snr_mid
        else:
            if target_positive:
                if snr_mid < target_SNR:
                    N_low, SNR_low = N_mid, snr_mid
                else:
                    N_high, SNR_high = N_mid, snr_mid
            else:
                if snr_mid > target_SNR:
                    N_low, SNR_low = N_mid, snr_mid
                else:
                    N_high, SNR_high = N_mid, snr_mid
    
    # =====================================================================
    # Phase 5: Return optimal N based on target or range  - finds the closest not the closest above the SNR range
    # =====================================================================
    # if use_range:
    #     valid_indices = [i for i, snr in enumerate(SNR_tested_list) 
    #                     if SNR_min <= snr <= SNR_max]
        
    #     if valid_indices:
    #         best_idx = min(valid_indices, key=lambda i: N_tested_list[i])
    #         N_needed = N_tested_list[best_idx]
    #         snr_needed = SNR_tested_list[best_idx]
            
    #         if verbose:
    #             print(f"\n✓ Optimal N found: N = {N_needed}, SNR = {snr_needed:.3f}")
    #             print(f"  (Within range [{SNR_min}, {SNR_max}])\n")
    #     else:
    #         range_mid = (SNR_min + SNR_max) / 2
    #         closest_idx = min(range(len(SNR_tested_list)), 
    #                         key=lambda i: abs(SNR_tested_list[i] - range_mid))
    #         N_needed = N_tested_list[closest_idx]
    #         snr_needed = SNR_tested_list[closest_idx]
            
    #         if verbose:
    #             print(f"\n⚠ No point strictly in range, using closest:")
    #             print(f"  N = {N_needed}, SNR = {snr_needed:.3f}\n")
    # else:
    #     if target_positive:
    #         valid_indices = [i for i, snr in enumerate(SNR_tested_list) if snr >= target_SNR]
    #     else:
    #         valid_indices = [i for i, snr in enumerate(SNR_tested_list) if snr <= target_SNR]
        
    #     if valid_indices:
    #         closest_idx = min(valid_indices, key=lambda i: abs(SNR_tested_list[i] - target_SNR))
    #         N_needed = N_tested_list[closest_idx]
    #         snr_needed = SNR_tested_list[closest_idx]
            
    #         if verbose:
    #             print(f"\n✓ Optimal N found: N = {N_needed}, SNR = {snr_needed:.3f}")
    #             print(f"  (Error from target: {abs(snr_needed - target_SNR):.3f})\n")
    #     else:
    #         N_needed = N_high
    #         snr_needed = SNR_high
            
    #         if verbose:
    #             print(f"\n⚠ Using bracket point: N = {N_needed}, SNR = {snr_needed:.3f}\n")
    # Phase 5: Return the minimum N that reaches the target (above for positive target)
    if target_positive:
        valid_indices = [i for i, snr in enumerate(SNR_tested_list) if snr >= target_SNR]
    else:
        valid_indices = [i for i, snr in enumerate(SNR_tested_list) if snr <= target_SNR]

    if valid_indices:
        # NEW BEHAVIOR:
        # choose the **minimum N** that meets threshold (not nearest)
        best_idx = min(valid_indices, key=lambda i: N_tested_list[i])
        N_needed = N_tested_list[best_idx]
        snr_needed = SNR_tested_list[best_idx]

        if verbose:
            print(f"\n✓ Minimum N above target: N = {N_needed}, SNR = {snr_needed:.3f}\n")
    else:
        # fallback if never reached (use bracket high)
        N_needed = N_high
        snr_needed = SNR_high

        if verbose:
            print(f"\n⚠ Target not reached; using upper bracket: N = {N_needed}, SNR = {snr_needed:.3f}\n")
    
    return N_needed, {
        'N_tested': N_tested_list,
        'SNR_tested': SNR_tested_list,
        'bracket_found': True,
        'expansions': expansion_count,
        'final_N': N_needed,
        'final_SNR': float(snr_needed),
        'iterations': iteration + 1,
        'timing_breakdown': timing_list if profile else None,
        'used_cache': signal_cache is not None
    }


def find_N_ensemble(config_template, smbhb_module, psrs_clean, params, Tspan,
                   target_SNR=3.75, SNR_range=None, n_realisations=20, N_initial_guess=None, 
                   N_max_initial=None, verbose=True, profile=True,
                   use_cache=True, cache_threshold=7000, batch_size=10000):
    """
    Find minimum N_binaries across multiple population realisations.
    
    Uses CONSISTENT initial guess across all realisations to enable
    proper statistical comparison. Includes memory-efficient options
    for large populations.
    
    Parameters:
        SNR_range: tuple (SNR_min, SNR_max) for acceptable range, or None for point target
        profile: if True, print detailed timing breakdown for each SNR computation
        use_cache: if True, pre-compute signal cache for populations below cache_threshold
        cache_threshold: max population size for caching (default 50000)
        batch_size: if N_max_initial > cache_threshold, generate in batches of this size
    
    Returns:
        results dict with N_needed distribution and statistics
    """
    start_time = time.time()
    
    if N_initial_guess is None:
        N_initial_guess = 2000
    
    if verbose:
        print(f"\n{'='*70}")
        if SNR_range:
            print(f"ENSEMBLE ANALYSIS - Finding Minimum N for SNR in range {SNR_range}")
        else:
            print(f"ENSEMBLE ANALYSIS - Finding Minimum N for SNR = {target_SNR}")
        print(f"{'='*70}")
        print(f"Realizations: {n_realisations}")
        print(f"Initial guess (fixed): N = {N_initial_guess}")
        if N_max_initial:
            print(f"Max initial pool: N = {N_max_initial}")
        if use_cache:
            print(f"Caching: enabled (threshold = {cache_threshold})")
        else:
            print(f"Caching: disabled")
        if N_max_initial and N_max_initial > cache_threshold:
            print(f"Batch generation: enabled (batch size = {batch_size})")
        print(f"{'='*70}\n")
    
    N_needed_list = []
    all_results = []
    
    for i in range(n_realisations):
        if verbose:
            print(f"\n[{i+1}/{n_realisations}]")
        
        N_needed, metadata = find_N_binaries_for_target_snr(
            config_template, smbhb_module, psrs_clean, params, Tspan,
            target_SNR=target_SNR,
            SNR_range=SNR_range,
            N_initial_guess=N_initial_guess,
            N_max_initial=N_max_initial,
            verbose=verbose,
            profile=profile,
            use_cache=use_cache,
            cache_threshold=cache_threshold,
            batch_size=batch_size
        )
        
        metadata['realisation'] = i
        metadata['success'] = N_needed is not None
        metadata['N_needed'] = N_needed
        all_results.append(metadata)
        
        if N_needed is not None:
            N_needed_list.append(N_needed)
            if verbose:
                print(f"✓ Success: N_needed = {N_needed}")
        else:
            if verbose:
                print(f"✗ Failed to reach target SNR")
    
    # =====================================================================
    # Compile statistics
    # =====================================================================
    N_array = np.array(N_needed_list) if N_needed_list else np.array([])
    
    ensemble_results = {
        'config': config_template,
        'target_SNR': target_SNR,
        'n_realisations': n_realisations,
        'N_initial_guess': N_initial_guess,
        'N_needed_list': N_needed_list,
        'all_results': all_results,
        'statistics': {
            'success_rate': len(N_needed_list) / n_realisations,
            'mean': float(np.mean(N_array)) if len(N_array) > 0 else None,
            'median': float(np.median(N_array)) if len(N_array) > 0 else None,
            'std': float(np.std(N_array)) if len(N_array) > 0 else None,
            'min': int(np.min(N_array)) if len(N_array) > 0 else None,
            'max': int(np.max(N_array)) if len(N_array) > 0 else None,
            'percentile_16': int(np.percentile(N_array, 16)) if len(N_array) > 0 else None,
            'percentile_84': int(np.percentile(N_array, 84)) if len(N_array) > 0 else None,
        },
        'total_time': time.time() - start_time,
        'memory_config': {
            'use_cache': use_cache,
            'cache_threshold': cache_threshold,
            'batch_size': batch_size
        }
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ENSEMBLE RESULTS")
        print(f"{'='*70}")
        print(f"Success rate: {ensemble_results['statistics']['success_rate']:.1%} ({len(N_needed_list)}/{n_realisations})")
        if len(N_needed_list) > 0:
            print(f"\nMinimum N_binaries (distribution):")
            print(f"  Mean:      {ensemble_results['statistics']['mean']:.0f}")
            print(f"  Median:    {ensemble_results['statistics']['median']:.0f}")
            print(f"  Std dev:   {ensemble_results['statistics']['std']:.0f}")
            print(f"  Min:       {ensemble_results['statistics']['min']}")
            print(f"  Max:       {ensemble_results['statistics']['max']}")
            print(f"  16-84%ile: [{ensemble_results['statistics']['percentile_16']}, "
                  f"{ensemble_results['statistics']['percentile_84']}]")
        print(f"\nTotal time: {ensemble_results['total_time']:.1f} s ({ensemble_results['total_time']/60:.1f} min)")
        print(f"{'='*70}\n")
    
    return ensemble_results


####### OLD CODE: KEEP FOR REFERENCE #######


# import numpy as np
# import time
# import json
# from copy import deepcopy
# from scipy.interpolate import interp1d
# from signal_injection import inject_population_into_psrs
# from pta_builder import build_pta_and_params
# from enterprise_extensions.frequentist import optimal_statistic as opt_stat


# def compute_os_for_N_fast(N_bin, population, psrs_clean, params, Tspan):
#     """Streamlined OS computation for ensemble."""
#     try:
#         pop_subset = population[:N_bin]
#         psrs_temp = inject_population_into_psrs(psrs_clean, pop_subset, pure_signal=True, verbose=False)
        
#         pta_temp, _, params_out = build_pta_and_params(
#             psrs=psrs_temp, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
#         )
        
#         ostat = opt_stat.OptimalStatistic(psrs_temp, pta=pta_temp, orf='hd')
#         xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
        
#         return OS / OS_sig
#     except Exception as e:
#         return np.nan




# def find_N_for_target_adaptive(population, psrs_clean, params, Tspan, target_SNR=4.0, 
#                                N_initial_guess=None, tolerance=0.3, max_iterations=15):
#     """Find N needed using adaptive bisection approach."""
#     N_max = len(population)
    
#     # Initial guess: if not provided, start with a rough estimate
#     if N_initial_guess is None:
#         N_initial_guess = max(100, N_max // 10)
    
#     # First, get SNR at initial guess
#     print(f"  Initial guess: N = {N_initial_guess}")
#     SNR_guess = compute_os_for_N_fast(N_initial_guess, population, psrs_clean, params, Tspan)
    
#     if not np.isfinite(SNR_guess):
#         return None, {'error': 'Initial guess failed', 'N_tested': [N_initial_guess]}
    
#     # Store all tested points
#     N_tested = [N_initial_guess]
#     SNR_tested = [SNR_guess]
    
#     # Determine if we need more or fewer binaries
#     if abs(SNR_guess - target_SNR) / target_SNR < tolerance:
#         print(f"  ✓ Initial guess was good! SNR = {SNR_guess:.2f}")
#         return N_initial_guess, {'N_tested': N_tested, 'SNR_tested': SNR_tested}
    
#     # Set search bounds
#     if SNR_guess < target_SNR:
#         N_low, SNR_low = N_initial_guess, SNR_guess
#         N_high, SNR_high = N_max, None
#     else:
#         N_low, SNR_low = 1, None
#         N_high, SNR_high = N_initial_guess, SNR_guess
    
#     # Adaptive bisection
#     for iteration in range(max_iterations):
#         # If we don't have both bounds, test the missing one
#         if SNR_high is None:
#             N_test = N_high
#             print(f"  Iteration {iteration+1}: Testing upper bound N = {N_test}")
#             SNR_test = compute_os_for_N_fast(N_test, population, psrs_clean, params, Tspan)
#             N_tested.append(N_test)
#             SNR_tested.append(SNR_test)
            
#             if not np.isfinite(SNR_test):
#                 return None, {'error': 'Upper bound test failed', 'N_tested': N_tested}
            
#             SNR_high = SNR_test
#             if SNR_high < target_SNR:
#                 return None, {'error': f'Even max N gives SNR={SNR_high:.2f} < target={target_SNR}',
#                             'N_tested': N_tested, 'SNR_tested': SNR_tested}
        
#         elif SNR_low is None:
#             N_test = N_low
#             print(f"  Iteration {iteration+1}: Testing lower bound N = {N_test}")
#             SNR_test = compute_os_for_N_fast(N_test, population, psrs_clean, params, Tspan)
#             N_tested.append(N_test)
#             SNR_tested.append(SNR_test)
            
#             if not np.isfinite(SNR_test):
#                 return None, {'error': 'Lower bound test failed', 'N_tested': N_tested}
            
#             SNR_low = SNR_test
#             if SNR_low > target_SNR:

#                 return None, {'error': f'Even min N gives SNR={SNR_low:.2f} > target={target_SNR}',
#                             'N_tested': N_tested, 'SNR_tested': SNR_tested}
        
#         # Now we have both bounds, do bisection
#         else:
#             # Use linear interpolation for next guess
#             N_test = int(N_low + (N_high - N_low) * (target_SNR - SNR_low) / (SNR_high - SNR_low))
#             N_test = max(N_low + 1, min(N_test, N_high - 1))  # Stay within bounds
            
#             print(f"  Iteration {iteration+1}: Testing N = {N_test} (bounds: [{N_low}, {N_high}])")
#             SNR_test = compute_os_for_N_fast(N_test, population, psrs_clean, params, Tspan)
#             N_tested.append(N_test)
#             SNR_tested.append(SNR_test)
            
#             if not np.isfinite(SNR_test):
#                 print(f"    Warning: N={N_test} gave invalid SNR, trying midpoint")
#                 N_test = (N_low + N_high) // 2
#                 SNR_test = compute_os_for_N_fast(N_test, population, psrs_clean, params, Tspan)
#                 N_tested.append(N_test)
#                 SNR_tested.append(SNR_test)
#                 if not np.isfinite(SNR_test):
#                     return None, {'error': 'Midpoint test failed', 'N_tested': N_tested}
            
#             # Check if we've converged
#             if abs(SNR_test - target_SNR) / target_SNR < tolerance:
#                 print(f"  ✓ Converged! N = {N_test}, SNR = {SNR_test:.2f}")
#                 return N_test, {'N_tested': N_tested, 'SNR_tested': SNR_tested, 
#                                'iterations': iteration + 1}
            
#             # Update bounds
#             if SNR_test < target_SNR:
#                 N_low, SNR_low = N_test, SNR_test
#             else:
#                 N_high, SNR_high = N_test, SNR_test
            
#             # Check if bounds are too close
#             if N_high - N_low <= 1:
#                 # Pick the one closer to target
#                 if abs(SNR_low - target_SNR) < abs(SNR_high - target_SNR):
#                     return N_low, {'N_tested': N_tested, 'SNR_tested': SNR_tested,
#                                   'iterations': iteration + 1, 'converged_to_bound': True}
#                 else:
#                     return N_high, {'N_tested': N_tested, 'SNR_tested': SNR_tested,
#                                    'iterations': iteration + 1, 'converged_to_bound': True}
    
#     # Max iterations reached - return best guess
#     idx_best = np.argmin([abs(s - target_SNR) for s in SNR_tested])
#     return N_tested[idx_best], {'N_tested': N_tested, 'SNR_tested': SNR_tested,
#                                 'iterations': max_iterations, 'max_iterations_reached': True}


# def run_ensemble_analysis(config_name, config, psrs_clean, params, Tspan, smbhb_module,
#                           n_realisations=20, target_SNR=4.0, N_init_guess=None, n_test_points=5):
#     """Run ensemble analysis with multiple population realisations."""
#     from config import generate_population
    
#     N_needed_list = []
#     all_results = []
#     start_time = time.time()
    
#     for i in range(n_realisations):
#         print(f"\n[{i+1}/{n_realisations}] Generating population...")
#         population = generate_population(config, smbhb_module)
        
#         N_needed, result = find_N_for_target_adaptive(
#             population=population, psrs_clean=psrs_clean, params=params, Tspan=Tspan, target_SNR=target_SNR, N_initial_guess=N_init_guess, max_iterations=n_test_points
#         )
        
#         if N_needed:
#             N_needed_list.append(N_needed)
#             print(f"    ✓ N_needed = {N_needed}")
#         else:
#             print(f"    ✗ Failed: {result.get('error', 'Unknown')}")
        
#         result['realisation'] = i
#         result['success'] = N_needed is not None
#         all_results.append(result)
    
#     ensemble_results = {
#         'config_name': config_name,
#         'config': config,
#         'target_SNR': target_SNR,
#         'n_realisations': n_realisations,
#         'N_needed_list': N_needed_list,
#         'all_results': all_results,
#         'total_time': time.time() - start_time
#     }
    
#     # Calculate statistics
#     if len(N_needed_list) > 0:
#         N_array = np.array(N_needed_list)
#         ensemble_results['statistics'] = {
#             'mean': float(np.mean(N_array)),
#             'median': float(np.median(N_array)),
#             'std': float(np.std(N_array)),
#             'min': int(np.min(N_array)),
#             'max': int(np.max(N_array))
#         }
    
#     return ensemble_results

# def compute_snr_in_batches(population, psrs_clean, params, Tspan, batch_size=50, 
#                            target_SNR=None, refine_batches=True, use_incremental=True, verbose=True):
#     """
#     Compute cumulative SNR by adding binaries in batches.
#     Stops and refines once target is exceeded.
    
#     Parameters:
#         use_incremental: If True, use add=True in inject_population_into_psrs to build incrementally
#                         This is MUCH faster as it avoids recomputing previous binaries
    
#     Returns:
#         N_values: array of number of binaries at each evaluation point
#         SNR_values: array of cumulative SNR at each point
#         batch_size_used: actual batch size used
#     """
#     N_total = len(population)
#     N_values = []
#     SNR_values = []
    
#     if verbose:
#         print(f"Computing SNR for up to {N_total} binaries in batches of {batch_size}...")
#         if use_incremental:
#             print(f"  Using incremental injection (add=True) for speed")
#         if target_SNR:
#             print(f"  Target SNR: {target_SNR:.2f} (will refine once exceeded)")
    
#     # Phase 1: Add batches until we exceed target
#     N_current = 0
#     current_batch = batch_size
#     exceeded_target = False
#     psrs_prev = None  # Store previous state for incremental building
    
#     while N_current < N_total:
#         N_test = min(N_current + current_batch, N_total)
        
#         try:
#             if use_incremental and psrs_prev is not None:
#                 # Incremental: only inject new binaries since last computation
#                 pop_new = population[N_current:N_test]
#                 psrs_temp = inject_population_into_psrs(
#                     psrs_prev, pop_new, pure_signal=True, add=True, verbose=False
#                 )
#             else:
#                 # From scratch: inject all binaries from beginning
#                 pop_subset = population[:N_test]
#                 psrs_temp = inject_population_into_psrs(
#                     psrs_clean, pop_subset, pure_signal=True, verbose=False
#                 )
            
#             # Build PTA and compute OS
#             pta_temp, _, params_out = build_pta_and_params(
#                 psrs=psrs_temp, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
#             )
            
#             ostat = opt_stat.OptimalStatistic(psrs_temp, pta=pta_temp, orf='hd')
#             xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
            
#             snr = OS / OS_sig
#             N_values.append(N_test)
#             SNR_values.append(snr)
            
#             # Store for next iteration
#             if use_incremental:
#                 psrs_prev = psrs_temp
            
#             if verbose:
#                 marker = " ← EXCEEDED TARGET" if target_SNR and snr >= target_SNR and not exceeded_target else ""
#                 print(f"  N = {N_test:5d}: SNR = {snr:.3f}{marker}")
            
#             # Check if we exceeded target
#             if target_SNR and snr >= target_SNR and not exceeded_target:
#                 exceeded_target = True
                
#                 if refine_batches and len(N_values) >= 2:
#                     # We have a bracket: previous point was below, this is above
#                     N_below = N_values[-2]
#                     SNR_below = SNR_values[-2]
#                     N_above = N_test
#                     SNR_above = snr
                    
#                     if verbose:
#                         print(f"  → Refining between N={N_below} (SNR={SNR_below:.3f}) and N={N_above} (SNR={SNR_above:.3f})")
                    
#                     # Refine with smaller batches (from scratch, for accuracy)
#                     refine_batch_size = max(10, current_batch // 10)
#                     N_refine = N_below + refine_batch_size
                    
#                     while N_refine < N_above:
#                         try:
#                             pop_subset = population[:N_refine]
#                             psrs_temp = inject_population_into_psrs(
#                                 psrs_clean, pop_subset, pure_signal=True, verbose=False
#                             )
                            
#                             pta_temp, _, params_out = build_pta_and_params(
#                                 psrs=psrs_temp, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
#                             )
                            
#                             ostat = opt_stat.OptimalStatistic(psrs_temp, pta=pta_temp, orf='hd')
#                             xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
#                             snr_refine = OS / OS_sig
                            
#                             N_values.append(N_refine)
#                             SNR_values.append(snr_refine)
                            
#                             if verbose:
#                                 print(f"    Refine N = {N_refine:5d}: SNR = {snr_refine:.3f}")
                            
#                             N_refine += refine_batch_size
                            
#                         except Exception as e:
#                             if verbose:
#                                 print(f"    Refine N = {N_refine}: FAILED")
#                             N_refine += refine_batch_size
                
#                 # Stop after refining
#                 break
            
#             N_current = N_test
                
#         except Exception as e:
#             if verbose:
#                 print(f"  N = {N_test:5d}: FAILED - {e}")
#             N_current = N_test
#             # Reset incremental state on failure
#             if use_incremental:
#                 psrs_prev = None
#             continue
    
#     # Sort by N (refinement points might be out of order)
#     if len(N_values) > 0:
#         sorted_idx = np.argsort(N_values)
#         N_values = np.array(N_values)[sorted_idx]
#         SNR_values = np.array(SNR_values)[sorted_idx]
#     else:
#         N_values = np.array(N_values)
#         SNR_values = np.array(SNR_values)
    
#     if verbose:
#         print(f"\n  ✓ Computed {len(N_values)} data points")
#         if len(SNR_values) > 0:
#             print(f"  Range: N ∈ [{N_values[0]}, {N_values[-1]}], SNR ∈ [{SNR_values.min():.3f}, {SNR_values.max():.3f}]")
    
#     return N_values, SNR_values, batch_size


# def find_N_from_batched_snr(N_values, SNR_values, target_SNR=4.0):
#     """
#     Find N needed to reach target SNR from batched measurements.
#     Uses linear interpolation between measured points.
#     """
#     if len(N_values) == 0:
#         return None, None
    
#     # Check if target is achievable
#     if SNR_values[-1] < target_SNR:
#         return None, SNR_values[-1]  # Return max achievable
    
#     if SNR_values[0] >= target_SNR:
#         return N_values[0], SNR_values[0]  # First batch already exceeds
    
#     # Find bracketing points
#     idx_above = np.argmax(SNR_values >= target_SNR)
#     idx_below = idx_above - 1
    
#     # Linear interpolation
#     N_below, SNR_below = N_values[idx_below], SNR_values[idx_below]
#     N_above, SNR_above = N_values[idx_above], SNR_values[idx_above]
    
#     # Interpolate
#     N_needed = N_below + (N_above - N_below) * (target_SNR - SNR_below) / (SNR_above - SNR_below)
#     N_needed = int(np.round(N_needed))
    
#     # Estimate achieved SNR at N_needed
#     achieved_SNR = SNR_below + (SNR_above - SNR_below) * (N_needed - N_below) / (N_above - N_below)
    
#     return N_needed, achieved_SNR


# def run_ensemble_batched(config_name, config, psrs_clean, params, Tspan, smbhb_module,
#                          n_realisations=20, target_SNR=4.0, N_binaries_pool=10000,
#                          batch_size=50, refine=True, use_incremental=True):
#     """
#     Run ensemble analysis using batched SNR computation.
#     Much faster than individual binary approach.
    
#     Parameters:
#         use_incremental: Use add=True in injection for speed (highly recommended!)
#     """
#     from config import generate_population
    
#     start_time = time.time()
    
#     print(f"\n{'='*70}")
#     print(f"ENSEMBLE ANALYSIS (BATCHED): {config_name}")
#     print(f"{'='*70}")
#     print(f"Target SNR: {target_SNR}")
#     print(f"Pool size: {N_binaries_pool} binaries")
#     print(f"Batch size: {batch_size} binaries")
#     print(f"Incremental injection: {'enabled' if use_incremental else 'disabled'}")
#     print(f"Refinement: {'enabled' if refine else 'disabled'}")
#     print(f"Realizations: {n_realisations}")
#     print(f"{'='*70}\n")
    
#     all_results = []
#     N_needed_list = []
    
#     for i in range(n_realisations):
#         print(f"\n[{i+1}/{n_realisations}] Generating population...")
#         realisation_start = time.time()
        
#         population = generate_population(
#             {**config, 'N_binaries': N_binaries_pool}, 
#             smbhb_module
#         )
        
#         # Compute SNR in batches with early stopping
#         N_values, SNR_values, batch_used = compute_snr_in_batches(
#             population, psrs_clean, params, Tspan, 
#             batch_size=batch_size, target_SNR=target_SNR, 
#             refine_batches=refine, use_incremental=use_incremental, verbose=True
#         )
        
#         # Find N needed
#         N_needed, achieved_SNR = find_N_from_batched_snr(N_values, SNR_values, target_SNR)
        
#         realisation_time = time.time() - realisation_start
        
#         result = {
#             'realisation': i,
#             'N_values': N_values.tolist(),
#             'SNR_values': SNR_values.tolist(),
#             'N_needed': N_needed,
#             'achieved_SNR': float(achieved_SNR) if achieved_SNR else None,
#             'success': N_needed is not None,
#             'time': realisation_time,
#             'batch_size': batch_used
#         }
        
#         if N_needed is not None:
#             N_needed_list.append(N_needed)
#             print(f"    ✓ N_needed = {N_needed} (SNR ≈ {achieved_SNR:.2f})")
#             print(f"    Time: {realisation_time:.1f}s")
#         else:
#             max_snr = achieved_SNR  # This is actually max_achievable
#             print(f"    ✗ Cannot reach target! Max SNR: {max_snr:.2f}")
#             print(f"    Time: {realisation_time:.1f}s")
        
#         all_results.append(result)
    
#     total_time = time.time() - start_time
    
#     # Compile ensemble results
#     ensemble_results = {
#         'config_name': config_name,
#         'config': config,
#         'target_SNR': target_SNR,
#         'N_binaries_pool': N_binaries_pool,
#         'batch_size': batch_size,
#         'n_realisations': n_realisations,
#         'N_needed_list': N_needed_list,
#         'all_results': all_results,
#         'total_time': total_time
#     }
    
#     # Calculate statistics
#     if len(N_needed_list) > 0:
#         N_array = np.array(N_needed_list)
#         ensemble_results['statistics'] = {
#             'mean': float(np.mean(N_array)),
#             'median': float(np.median(N_array)),
#             'std': float(np.std(N_array)),
#             'min': int(np.min(N_array)),
#             'max': int(np.max(N_array)),
#             'success_rate': len(N_needed_list) / n_realisations,
#             'avg_time_per_realisation': total_time / n_realisations
#         }
        
#         print(f"\n{'='*70}")
#         print(f"ENSEMBLE SUMMARY: {config_name}")
#         print(f"{'='*70}")
#         print(f"  Success rate: {ensemble_results['statistics']['success_rate']:.1%}")
#         print(f"  Mean N: {ensemble_results['statistics']['mean']:.0f}")
#         print(f"  Median N: {ensemble_results['statistics']['median']:.0f}")
#         print(f"  Std: {ensemble_results['statistics']['std']:.0f}")
#         print(f"  Range: [{ensemble_results['statistics']['min']}, {ensemble_results['statistics']['max']}]")
#         print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
#         print(f"  Avg time per realisation: {ensemble_results['statistics']['avg_time_per_realisation']:.1f}s")
#         print(f"{'='*70}\n")
#     else:
#         print(f"\n{'='*70}")
#         print(f"WARNING: No realisations succeeded!")
#         print(f"Consider increasing N_binaries_pool or decreasing target_SNR")
#         print(f"{'='*70}\n")
    
#     return ensemble_results


# def compute_snr_hybrid_search(population, psrs_clean, params, Tspan, target_SNR=4.0,
#                              initial_guess=None, batch_size=500, verbose=True):
#     """
#     Hybrid approach combining incremental building with adaptive refinement.
    
#     Strategy:
#     1. Start with initial guess
#     2. Use incremental batches to quickly find rough bracket
#     3. Once target is exceeded, use bisection for precise refinement
    
#     This gives you:
#     - Fast incremental building for coarse search
#     - Efficient bisection for final refinement
    
#     Returns:
#         N_needed: number of binaries needed
#         achieved_SNR: SNR at N_needed
#         N_values: all tested N values
#         SNR_values: all tested SNR values
#     """
#     N_total = len(population)
#     N_values = []
#     SNR_values = []
    
#     if initial_guess is None:
#         initial_guess = batch_size
    
#     if verbose:
#         print(f"Hybrid search for target SNR = {target_SNR}")
#         print(f"  Initial guess: N = {initial_guess}")
    
#     # ========================================================================
#     # PHASE 0: Test initial guess
#     # ========================================================================
#     try:
#         pop_subset = population[:initial_guess]
#         psrs_temp = inject_population_into_psrs(
#             psrs_clean, pop_subset, pure_signal=True, verbose=False
#         )
        
#         pta_temp, _, params_out = build_pta_and_params(
#             psrs=psrs_temp, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
#         )
        
#         ostat = opt_stat.OptimalStatistic(psrs_temp, pta=pta_temp, orf='hd')
#         xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
#         snr_guess = OS / OS_sig
        
#         N_values.append(initial_guess)
#         SNR_values.append(snr_guess)
        
#         if verbose:
#             print(f"  Initial: N = {initial_guess:5d}: SNR = {snr_guess:.3f}")
        
#         # Check if initial guess is already close enough
#         if abs(snr_guess - target_SNR) / target_SNR < 0.1:  # Within 10%
#             if verbose:
#                 print(f"  ✓ Initial guess is close enough!")
#             return initial_guess, snr_guess, np.array(N_values), np.array(SNR_values)
        
#     except Exception as e:
#         if verbose:
#             print(f"  Initial guess failed: {e}")
#         snr_guess = None
    
#     # ========================================================================
#     # PHASE 1: Incremental batch search to find rough bracket
#     # ========================================================================
#     N_below, SNR_below = None, None
#     N_above, SNR_above = None, None
    
#     if snr_guess is not None:
#         if snr_guess >= target_SNR:
#             # Initial guess is too high - need to search downward
#             if verbose:
#                 print(f"  Phase 1: Searching downward (initial guess too high)")
#             N_above, SNR_above = initial_guess, snr_guess
            
#             # Search downward in decreasing batches
#             N_test = max(batch_size, initial_guess - batch_size)
#             while N_test > 0:
#                 try:
#                     pop_subset = population[:N_test]
#                     psrs_temp = inject_population_into_psrs(
#                         psrs_clean, pop_subset, pure_signal=True, verbose=False
#                     )
                    
#                     pta_temp, _, params_out = build_pta_and_params(
#                         psrs=psrs_temp, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
#                     )
                    
#                     ostat = opt_stat.OptimalStatistic(psrs_temp, pta=pta_temp, orf='hd')
#                     xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
#                     snr = OS / OS_sig
                    
#                     N_values.append(N_test)
#                     SNR_values.append(snr)
                    
#                     if verbose:
#                         print(f"    N = {N_test:5d}: SNR = {snr:.3f}")
                    
#                     if snr < target_SNR:
#                         N_below, SNR_below = N_test, snr
#                         if verbose:
#                             print(f"    ✓ Bracket found: [{N_below}, {N_above}]")
#                         break
#                     else:
#                         N_above, SNR_above = N_test, snr
                    
#                     N_test -= batch_size
                    
#                 except Exception as e:
#                     if verbose:
#                         print(f"    N = {N_test}: FAILED - {e}")
#                     N_test -= batch_size
#                     continue
            
#             if N_below is None:
#                 N_below, SNR_below = batch_size, 0  # Approximate lower bound
        
#         else:
#             # Initial guess is too low - need to search upward
#             if verbose:
#                 print(f"  Phase 1: Searching upward from N = {initial_guess} (incremental batches of {batch_size})")
#             N_below, SNR_below = initial_guess, snr_guess
            
#             # Search upward incrementally
#             N_current = initial_guess
#             psrs_prev = psrs_temp  # Reuse from initial guess
            
#             while N_current < N_total:
#                 N_test = min(N_current + batch_size, N_total)
                
#                 try:
#                     # Incremental: add new binaries
#                     pop_new = population[N_current:N_test]
#                     psrs_temp = inject_population_into_psrs(
#                         psrs_prev, pop_new, pure_signal=True, add=True, verbose=False
#                     )
                    
#                     # Compute SNR
#                     pta_temp, _, params_out = build_pta_and_params(
#                         psrs=psrs_temp, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
#                     )
                    
#                     ostat = opt_stat.OptimalStatistic(psrs_temp, pta=pta_temp, orf='hd')
#                     xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
#                     snr = OS / OS_sig
                    
#                     N_values.append(N_test)
#                     SNR_values.append(snr)
#                     psrs_prev = psrs_temp
                    
#                     if verbose:
#                         print(f"    N = {N_test:5d}: SNR = {snr:.3f}")
                    
#                     # Check if we've bracketed the target
#                     if snr >= target_SNR:
#                         N_above, SNR_above = N_test, snr
#                         if verbose:
#                             print(f"    ✓ Bracket found: [{N_below}, {N_above}]")
#                         break
#                     else:
#                         N_below, SNR_below = N_test, snr
                    
#                     N_current = N_test
                    
#                 except Exception as e:
#                     if verbose:
#                         print(f"    N = {N_test}: FAILED - {e}")
#                     N_current = N_test
#                     psrs_prev = None
#                     continue
    
#     else:
#         # Initial guess failed, start from scratch
#         if verbose:
#             print(f"  Phase 1: Starting from scratch with batches of {batch_size}")
        
#         N_current = 0
#         psrs_prev = None
        
#         while N_current < N_total:
#             N_test = min(N_current + batch_size, N_total)
            
#             try:
#                 if psrs_prev is not None:
#                     pop_new = population[N_current:N_test]
#                     psrs_temp = inject_population_into_psrs(
#                         psrs_prev, pop_new, pure_signal=True, add=True, verbose=False
#                     )
#                 else:
#                     pop_subset = population[:N_test]
#                     psrs_temp = inject_population_into_psrs(
#                         psrs_clean, pop_subset, pure_signal=True, verbose=False
#                     )
                
#                 pta_temp, _, params_out = build_pta_and_params(
#                     psrs=psrs_temp, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
#                 )
                
#                 ostat = opt_stat.OptimalStatistic(psrs_temp, pta=pta_temp, orf='hd')
#                 xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
#                 snr = OS / OS_sig
                
#                 N_values.append(N_test)
#                 SNR_values.append(snr)
#                 psrs_prev = psrs_temp
                
#                 if verbose:
#                     print(f"    N = {N_test:5d}: SNR = {snr:.3f}")
                
#                 if snr >= target_SNR:
#                     N_above, SNR_above = N_test, snr
#                     if len(N_values) >= 2:
#                         N_below, SNR_below = N_values[-2], SNR_values[-2]
#                     if verbose:
#                         print(f"    ✓ Bracket found: [{N_below}, {N_above}]")
#                     break
                
#                 N_current = N_test
                
#             except Exception as e:
#                 if verbose:
#                     print(f"    N = {N_test}: FAILED - {e}")
#                 N_current = N_test
#                 psrs_prev = None
#                 continue
    
#     # Check if we found a bracket
#     if N_above is None:
#         if verbose:
#             print(f"  ✗ Could not reach target with {N_total} binaries")
#         return None, SNR_values[-1] if SNR_values else None, np.array(N_values), np.array(SNR_values)
    
#     if N_below is None:
#         if verbose:
#             print(f"  First test already exceeded target!")
#         return N_above, SNR_above, np.array(N_values), np.array(SNR_values)
    
#     # ========================================================================
#     # PHASE 2: Bisection refinement within bracket
#     # ========================================================================
#     if verbose:
#         print(f"  Phase 2: Bisection refinement")
#         print(f"    Bracket: N ∈ [{N_below}, {N_above}], SNR ∈ [{SNR_below:.3f}, {SNR_above:.3f}]")
    
#     refine_threshold = max(10, batch_size // 10)
#     max_bisect_iterations = 10
    
#     for iteration in range(max_bisect_iterations):
#         if N_above - N_below <= refine_threshold:
#             if verbose:
#                 print(f"    ✓ Converged: bracket width = {N_above - N_below} ≤ {refine_threshold}")
#             break
        
#         N_mid = (N_below + N_above) // 2
        
#         if N_mid in N_values:
#             if verbose:
#                 print(f"    Already tested N={N_mid}, stopping refinement")
#             break
        
#         try:
#             pop_subset = population[:N_mid]
#             psrs_temp = inject_population_into_psrs(
#                 psrs_clean, pop_subset, pure_signal=True, verbose=False
#             )
            
#             pta_temp, _, params_out = build_pta_and_params(
#                 psrs=psrs_temp, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
#             )
            
#             ostat = opt_stat.OptimalStatistic(psrs_temp, pta=pta_temp, orf='hd')
#             xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
#             snr_mid = OS / OS_sig
            
#             N_values.append(N_mid)
#             SNR_values.append(snr_mid)
            
#             if verbose:
#                 print(f"    Bisect N = {N_mid:5d}: SNR = {snr_mid:.3f}")
            
#             if snr_mid >= target_SNR:
#                 N_above, SNR_above = N_mid, snr_mid
#             else:
#                 N_below, SNR_below = N_mid, snr_mid
            
#         except Exception as e:
#             if verbose:
#                 print(f"    Bisect N = {N_mid}: FAILED - {e}")
#             break
    
#     # ========================================================================
#     # PHASE 3: Interpolate final answer
#     # ========================================================================
#     N_values = np.array(N_values)
#     SNR_values = np.array(SNR_values)
    
#     sorted_idx = np.argsort(N_values)
#     N_values = N_values[sorted_idx]
#     SNR_values = SNR_values[sorted_idx]
    
#     idx_above = np.argmax(SNR_values >= target_SNR)
#     if idx_above == 0:
#         N_needed = N_values[0]
#         achieved_SNR = SNR_values[0]
#     else:
#         idx_below = idx_above - 1
#         N_below, SNR_below = N_values[idx_below], SNR_values[idx_below]
#         N_above, SNR_above = N_values[idx_above], SNR_values[idx_above]
        
#         N_needed = N_below + (N_above - N_below) * (target_SNR - SNR_below) / (SNR_above - SNR_below)
#         N_needed = int(np.round(N_needed))
#         achieved_SNR = SNR_below + (SNR_above - SNR_below) * (N_needed - N_below) / (N_above - N_below)
    
#     if verbose:
#         print(f"\n  ✓ Final result: N = {N_needed}, SNR ≈ {achieved_SNR:.3f}")
#         print(f"  Total evaluations: {len(N_values)}")
    
#     return N_needed, achieved_SNR, N_values, SNR_values




# def compute_individual_snr_contributions(population, psrs_clean, params, Tspan, 
#                                          batch_size=50, verbose=True):
#     """
#     Compute SNR contribution from each binary individually.
#     Uses batching for efficiency while maintaining individual contributions.
    
#     Returns:
#         snr_contributions: array of SNR values, one per binary
#         failed_indices: list of indices where computation failed
#     """
#     N_total = len(population)
#     snr_contributions = np.zeros(N_total)
#     failed_indices = []
    
#     if verbose:
#         print(f"Computing individual SNR contributions for {N_total} binaries...")
#         print(f"Using batch size: {batch_size}")
    
#     # Process in batches for efficiency
#     n_batches = int(np.ceil(N_total / batch_size))
    
#     for batch_idx in range(n_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, N_total)
#         batch_indices = range(start_idx, end_idx)
        
#         if verbose:
#             print(f"  Batch {batch_idx+1}/{n_batches}: binaries {start_idx}-{end_idx-1}")
        
#         for idx in batch_indices:
#             try:
#                 # Inject single binary
#                 single_binary = [population[idx]]
#                 psrs_temp = inject_population_into_psrs(
#                     psrs_clean, single_binary, pure_signal=True, verbose=False
#                 )
                
#                 # Build PTA and compute OS
#                 pta_temp, _, params_out = build_pta_and_params(
#                     psrs=psrs_temp, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
#                 )
                
#                 ostat = opt_stat.OptimalStatistic(psrs_temp, pta=pta_temp, orf='hd')
#                 xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
                
#                 snr_contributions[idx] = OS / OS_sig
                
#             except Exception as e:
#                 if verbose:
#                     print(f"    Warning: Binary {idx} failed: {e}")
#                 snr_contributions[idx] = 0.0
#                 failed_indices.append(idx)
        
#         if verbose and (batch_idx + 1) % 5 == 0:
#             valid_snrs = snr_contributions[start_idx:end_idx]
#             valid_snrs = valid_snrs[valid_snrs > 0]
#             if len(valid_snrs) > 0:
#                 print(f"    Batch stats: mean SNR = {np.mean(valid_snrs):.3f}, "
#                       f"max SNR = {np.max(valid_snrs):.3f}")
    
#     # Filter out failed binaries
#     valid_mask = snr_contributions > 0
#     snr_contributions = snr_contributions[valid_mask]
    
#     if verbose:
#         print(f"\n  ✓ Successfully computed {len(snr_contributions)} SNR contributions")
#         print(f"  ✗ Failed: {len(failed_indices)} binaries")
#         if len(snr_contributions) > 0:
#             print(f"  SNR stats: min={np.min(snr_contributions):.3f}, "
#                   f"mean={np.mean(snr_contributions):.3f}, max={np.max(snr_contributions):.3f}")
    
#     return snr_contributions, failed_indices


# def find_N_from_snr_contributions(snr_contributions, target_SNR=4.0, 
#                                    combination_method='quadrature'):
#     """
#     Find how many binaries needed to reach target SNR.
    
#     Parameters:
#         snr_contributions: array of individual SNR values
#         target_SNR: target total SNR
#         combination_method: 'quadrature' (add in quadrature) or 'linear' (linear sum)
    
#     Returns:
#         N_needed: number of binaries needed
#         cumulative_snr: array of cumulative SNR vs N
#     """
#     # Sort by SNR (largest first) for fastest convergence
#     sorted_snrs = np.sort(snr_contributions)[::-1]
    
#     if combination_method == 'quadrature':
#         # Add in quadrature: SNR_total = sqrt(sum(SNR_i^2))
#         cumulative_snr = np.sqrt(np.cumsum(sorted_snrs**2))
#     elif combination_method == 'linear':
#         # Linear sum (probably not physical, but included for comparison)
#         cumulative_snr = np.cumsum(sorted_snrs)
#     else:
#         raise ValueError(f"Unknown combination method: {combination_method}")
    
#     # Find where we exceed target
#     exceeds_target = cumulative_snr >= target_SNR
    
#     if not np.any(exceeds_target):
#         # Can't reach target even with all binaries
#         return None, cumulative_snr
    
#     N_needed = np.argmax(exceeds_target) + 1  # +1 because of 0-indexing
    
#     return N_needed, cumulative_snr


# def run_ensemble_with_precomputed_snr(config_name, config, psrs_clean, params, Tspan, 
#                                       smbhb_module, n_realisations=20, target_SNR=4.0,
#                                       N_binaries_pool=10000, batch_size=50,
#                                       combination_method='quadrature'):
#     """
#     Run ensemble analysis by pre-computing individual SNR contributions.
    
#     This is much more efficient for multiple realisations because:
#     1. Each binary's SNR is computed only once
#     2. Finding N_needed is then just a simple cumulative sum operation
#     3. Can easily test different target SNRs without recomputing
#     """
#     from config import generate_population
    
#     start_time = time.time()
    
#     print(f"\n{'='*70}")
#     print(f"ENSEMBLE ANALYSIS: {config_name}")
#     print(f"{'='*70}")
#     print(f"Target SNR: {target_SNR}")
#     print(f"Pool size: {N_binaries_pool} binaries")
#     print(f"Realizations: {n_realisations}")
#     print(f"Combination method: {combination_method}")
#     print(f"{'='*70}\n")
    
#     all_results = []
#     N_needed_list = []
    
#     for i in range(n_realisations):
#         print(f"\n[{i+1}/{n_realisations}] Generating population of {N_binaries_pool} binaries...")
#         population = generate_population(
#             {**config, 'N_binaries': N_binaries_pool}, 
#             smbhb_module
#         )
        
#         # Compute individual SNR contributions
#         snr_contributions, failed = compute_individual_snr_contributions(
#             population, psrs_clean, params, Tspan, batch_size=batch_size, verbose=True
#         )
        
#         # Find N needed for target SNR
#         N_needed, cumulative_snr = find_N_from_snr_contributions(
#             snr_contributions, target_SNR, combination_method
#         )
        
#         result = {
#             'realisation': i,
#             'N_pool': len(snr_contributions),
#             'N_failed': len(failed),
#             'snr_contributions': snr_contributions.tolist(),
#             'cumulative_snr': cumulative_snr.tolist(),
#             'N_needed': N_needed,
#             'success': N_needed is not None
#         }
        
#         if N_needed is not None:
#             N_needed_list.append(N_needed)
#             actual_snr = cumulative_snr[N_needed - 1]
#             print(f"    ✓ N_needed = {N_needed} (SNR = {actual_snr:.2f})")
#             result['achieved_snr'] = float(actual_snr)
#         else:
#             max_snr = cumulative_snr[-1]
#             print(f"    ✗ Cannot reach target! Max SNR with {len(snr_contributions)} binaries: {max_snr:.2f}")
#             result['max_achievable_snr'] = float(max_snr)
        
#         all_results.append(result)
    
#     # Compile ensemble results
#     ensemble_results = {
#         'config_name': config_name,
#         'config': config,
#         'target_SNR': target_SNR,
#         'N_binaries_pool': N_binaries_pool,
#         'n_realisations': n_realisations,
#         'combination_method': combination_method,
#         'N_needed_list': N_needed_list,
#         'all_results': all_results,
#         'total_time': time.time() - start_time
#     }
    
#     # Calculate statistics
#     if len(N_needed_list) > 0:
#         N_array = np.array(N_needed_list)
#         ensemble_results['statistics'] = {
#             'mean': float(np.mean(N_array)),
#             'median': float(np.median(N_array)),
#             'std': float(np.std(N_array)),
#             'min': int(np.min(N_array)),
#             'max': int(np.max(N_array)),
#             'success_rate': len(N_needed_list) / n_realisations
#         }
        
#         print(f"\n{'='*70}")
#         print(f"ENSEMBLE SUMMARY: {config_name}")
#         print(f"{'='*70}")
#         print(f"  Success rate: {ensemble_results['statistics']['success_rate']:.1%}")
#         print(f"  Mean N: {ensemble_results['statistics']['mean']:.0f}")
#         print(f"  Median N: {ensemble_results['statistics']['median']:.0f}")
#         print(f"  Std: {ensemble_results['statistics']['std']:.0f}")
#         print(f"  Range: [{ensemble_results['statistics']['min']}, {ensemble_results['statistics']['max']}]")
#         print(f"  Total time: {ensemble_results['total_time']:.1f}s")
#         print(f"  Time per realisation: {ensemble_results['total_time']/n_realisations:.1f}s")
#         print(f"{'='*70}\n")
#     else:
#         print(f"\n{'='*70}")
#         print(f"WARNING: No realisations succeeded!")
#         print(f"Consider increasing N_binaries_pool (current: {N_binaries_pool})")
#         print(f"{'='*70}\n")
    
#     return ensemble_results


# def analyze_snr_scaling(snr_contributions, target_SNRs=[3.0, 4.0, 5.0], 
#                         combination_method='quadrature'):
#     """
#     Analyze how many binaries needed for different target SNRs.
#     Useful for understanding scaling without recomputing.
#     """
#     results = {}
#     cumulative_snr = None
    
#     for target in target_SNRs:
#         N_needed, cumulative_snr = find_N_from_snr_contributions(
#             snr_contributions, target, combination_method
#         )
#         results[target] = {
#             'N_needed': N_needed,
#             'achieved_snr': cumulative_snr[N_needed-1] if N_needed else None
#         }
    
#     return results, cumulative_snr