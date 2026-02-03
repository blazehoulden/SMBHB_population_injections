"""
Vectorized and optimized SNR calculation for PTA detection of SMBHB population.
Memory-efficient implementation with proper array handling.
NOW WITH EXTENSIVE DEBUGGING TO DIAGNOSE HIGH SNR VALUES
"""

import numpy as np
from SMBHB_pop_synth import H0, h_circ
from config import generate_population


# ============================================================================
# DEBUGGING CONFIGURATION
# ============================================================================

DEBUG_CONFIG = {
    'enabled': True,
    'verbose_level': 2,  # 0=off, 1=summary, 2=detailed, 3=extreme
    'save_intermediate': True,  # Save intermediate values to file
    'check_units': True,  # Verify units are reasonable
    'validate_ranges': True,  # Check if values are in expected ranges
    'print_first_N': 5,  # Print details for first N binaries
    'log_file': 'snr_debug.log'
}

def debug_print(message, level=1):
    """Print debug message if level is sufficient."""
    if DEBUG_CONFIG['enabled'] and DEBUG_CONFIG['verbose_level'] >= level:
        print(f"[DEBUG L{level}] {message}")

def debug_log(message, level=1):
    """Log debug message to file."""
    if DEBUG_CONFIG['enabled'] and DEBUG_CONFIG['save_intermediate']:
        with open(DEBUG_CONFIG['log_file'], 'a') as f:
            f.write(f"[L{level}] {message}\n")


# ============================================================================
# MAIN FUNCTIONS WITH DEBUGGING
# ============================================================================

def omega_GW(f, h_cont):
    """
    Gravitational wave energy density parameter.
    
    Parameters
    ----------
    f : array_like
        Frequency [Hz]
    h_cont : array_like
        Characteristic strain contribution
        
    Returns
    -------
    array_like
        Omega_GW at given frequencies
    """
    omega = 2 * np.pi**2 / (3 * H0**2) * f**3 * h_cont**2
    
    if DEBUG_CONFIG['enabled'] and DEBUG_CONFIG['verbose_level'] >= 3:
        debug_print(f"omega_GW: f={f:.3e} Hz, h_cont={h_cont:.3e}, omega={omega:.3e}", 3)
        debug_print(f"  H0 = {H0:.3e}", 3)
        debug_print(f"  f^3 = {f**3:.3e}", 3)
        debug_print(f"  h_cont^2 = {h_cont**2:.3e}", 3)
    
    return omega


def pulsar_psd(pulsar_noise_params, f):
    """
    Power spectral density for a single pulsar's red noise.
    
    Parameters
    ----------
    pulsar_noise_params : dict
        Dictionary of noise parameters for a single pulsar
        Expected structure:
        {
            'red_noise': {'log10_A': float, 'gamma': float},
            'white_noise': {...}
        }
    f : array_like
        Frequency [Hz]
        
    Returns
    -------
    array_like
        PSD at given frequencies
    """
    fyr = 1.0 / (365.25 * 24 * 3600)  # 1/year in Hz
    pulsar_A = pulsar_noise_params['red_noise']['log10_A']
    pulsar_gamma = pulsar_noise_params['red_noise']['gamma']
    
    psd = 10**(2 * pulsar_A) * (f / fyr)**(-pulsar_gamma)
    
    if DEBUG_CONFIG['enabled'] and DEBUG_CONFIG['verbose_level'] >= 3:
        debug_print(f"pulsar_psd: log10_A={pulsar_A:.3f}, gamma={pulsar_gamma:.3f}", 3)
        debug_print(f"  f={f:.3e} Hz, fyr={fyr:.3e} Hz", 3)
        debug_print(f"  10^(2*log10_A) = {10**(2*pulsar_A):.3e}", 3)
        debug_print(f"  (f/fyr)^(-gamma) = {(f/fyr)**(-pulsar_gamma):.3e}", 3)
        debug_print(f"  PSD = {psd:.3e}", 3)
    
    return psd


def gamma_naught(pulsar1, pulsar2):
    """
    Hellings-Downs correlation coefficient for a pulsar pair.
    
    Parameters
    ----------
    pulsar1, pulsar2 : objects
        Pulsar objects with pos attribute (unit vectors)
        
    Returns
    -------
    float
        Correlation coefficient
    """
    cos_xi = np.dot(pulsar1.pos, pulsar2.pos)
    # Hellings-Downs function
    gamma = 3 * (1.0/3.0 + (1.0 - cos_xi) / 2.0 * (np.log((1.0 - cos_xi) / 2.0) - 1.0/6.0))
    
    if DEBUG_CONFIG['enabled'] and DEBUG_CONFIG['verbose_level'] >= 3:
        debug_print(f"gamma_naught: cos_xi={cos_xi:.3f}, gamma={gamma:.3f}", 3)
    
    return gamma


def precompute_pulsar_pairs(pulsars):
    """
    Precompute all pulsar pair correlations and indices.
    
    Parameters
    ----------
    pulsars : list
        List of pulsar objects
        
    Returns
    -------
    dict
        Contains:
        - 'pairs': array of shape (N_pairs, 2) with pulsar indices
        - 'gamma': array of shape (N_pairs,) with correlation coefficients
        - 'N_pairs': number of unique pairs
    """
    N = len(pulsars)
    N_pairs = N * (N - 1) // 2
    
    pairs = np.zeros((N_pairs, 2), dtype=int)
    gamma = np.zeros(N_pairs)
    
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            pairs[idx] = [i, j]
            gamma[idx] = gamma_naught(pulsars[i], pulsars[j])
            idx += 1
    
    debug_print(f"Precomputed {N_pairs} pulsar pairs from {N} pulsars", 1)
    debug_print(f"  Gamma range: [{np.min(gamma):.3f}, {np.max(gamma):.3f}]", 2)
    debug_print(f"  Gamma mean: {np.mean(gamma):.3f}, std: {np.std(gamma):.3f}", 2)
    
    return {
        'pairs': pairs,
        'gamma': gamma,
        'N_pairs': N_pairs
    }


def compute_psd_matrix(pulsars, pulsar_noise_params, frequencies):
    """
    Compute PSD for all pulsars at all frequencies.
    
    Parameters
    ----------
    pulsars : list
        List of pulsar objects (must have .name attribute matching keys in pulsar_noise_params)
    pulsar_noise_params : dict
        Dictionary of noise parameters from the parsed JSON file
    frequencies : array_like
        Array of frequencies [Hz]
        
    Returns
    -------
    ndarray
        Array of shape (N_pulsars, N_frequencies) with PSD values
    """
    N_pulsars = len(pulsars)
    N_freq = len(frequencies)
    psd_matrix = np.zeros((N_pulsars, N_freq))
    
    debug_print(f"\n{'='*70}", 1)
    debug_print(f"Computing PSD matrix: {N_pulsars} pulsars x {N_freq} frequencies", 1)
    debug_print(f"Frequency range: [{np.min(frequencies):.3e}, {np.max(frequencies):.3e}] Hz", 1)
    
    missing_pulsars = []
    no_red_noise = []
    
    for i, pulsar in enumerate(pulsars):
        # Check if this pulsar has red noise parameters
        if pulsar.name in pulsar_noise_params:
            if pulsar_noise_params[pulsar.name]['red_noise']:
                psd_matrix[i, :] = pulsar_psd(pulsar_noise_params[pulsar.name], frequencies)
                
                if i < DEBUG_CONFIG['print_first_N']:
                    params = pulsar_noise_params[pulsar.name]['red_noise']
                    debug_print(f"  Pulsar {i} ({pulsar.name}): log10_A={params['log10_A']:.3f}, gamma={params['gamma']:.3f}", 2)
                    debug_print(f"    PSD range: [{np.min(psd_matrix[i, :]):.3e}, {np.max(psd_matrix[i, :]):.3e}]", 2)
            else:
                no_red_noise.append(pulsar.name)
                debug_print(f"Warning: No red noise parameters for {pulsar.name}", 1)
        else:
            missing_pulsars.append(pulsar.name)
            debug_print(f"Warning: Pulsar {pulsar.name} not found in noise parameters", 1)
    
    # Summary statistics
    debug_print(f"\nPSD Matrix Statistics:", 1)
    debug_print(f"  Overall PSD range: [{np.min(psd_matrix):.3e}, {np.max(psd_matrix):.3e}]", 1)
    debug_print(f"  Mean PSD: {np.mean(psd_matrix):.3e}", 1)
    debug_print(f"  Pulsars with red noise: {N_pulsars - len(no_red_noise) - len(missing_pulsars)}/{N_pulsars}", 1)
    if missing_pulsars:
        debug_print(f"  Missing from noise params: {missing_pulsars}", 1)
    if no_red_noise:
        debug_print(f"  No red noise: {no_red_noise}", 1)
    
    return psd_matrix


def optimal_SNR_sq_single_BH_vectorized(binary, strain_data, pulsar_pair_data, 
                                        psd_matrix, T_obs, binary_idx=None):
    """
    Vectorized computation of SNR² for a single binary across all pulsar pairs.
    
    Parameters
    ----------
    binary : dict
        Binary parameters including 'f' (frequency)
    strain_data : dict
        Strain data including 'h_c_individual', 'bin_edges'
    pulsar_pair_data : dict
        Precomputed pulsar pair data from precompute_pulsar_pairs()
    psd_matrix : ndarray
        PSD matrix from compute_psd_matrix(), shape (N_pulsars,) for single frequency
    T_obs : float
        Observation time [s]
    binary_idx : int, optional
        Index of binary for debugging
        
    Returns
    -------
    float
        Total SNR² for this binary summed over all pulsar pairs
    """
    freq = binary['f']
    
    # Find which binary this is in the population
    bin_idx = binary.get('freq_bin', binary_idx if binary_idx is not None else 0)
    
    # Get the individual contribution
    if isinstance(strain_data, dict):
        h_circ_contrib = strain_data['h_c_individual'][bin_idx]
    else:
        h_circ_contrib = binary.get('h_c_contrib', 0)
    
    # ============ ADD DEBUGGING HERE ============
    if binary_idx is not None and binary_idx < DEBUG_CONFIG['print_first_N']:
        debug_print(f"\n{'='*70}", 2)
        debug_print(f"Binary {binary_idx} SNR Calculation:", 2)
        debug_print(f"  Frequency: {freq:.6e} Hz ({freq * 365.25 * 86400:.6e} Hz * yr)", 2)
        
        # NEW DEBUGGING - Check freq_bin indexing
        debug_print(f"  freq_bin from binary dict: {binary.get('freq_bin', 'NOT SET')}", 2)
        debug_print(f"  bin_idx being used: {bin_idx}", 2)
        debug_print(f"  h_c from strain_data[bin_idx={bin_idx}]: {h_circ_contrib:.6e}", 2)
        debug_print(f"  h_c from binary.get('h_c_contrib'): {binary.get('h_c_contrib', 'NOT SET')}", 2)
        
        # Check if they match
        if 'h_c_contrib' in binary:
            h_c_from_dict = binary['h_c_contrib']
            if not np.isclose(h_circ_contrib, h_c_from_dict, rtol=1e-6):
                debug_print(f"  ⚠️⚠️⚠️ MISMATCH! Using {h_circ_contrib:.3e} but binary dict has {h_c_from_dict:.3e}", 1)
                debug_print(f"  ⚠️⚠️⚠️ Ratio: {h_circ_contrib/h_c_from_dict:.1f}x", 1)
        
        # Also check what's in the strain_data array around this index
        debug_print(f"  strain_data['h_c_individual'] length: {len(strain_data['h_c_individual'])}", 2)
        debug_print(f"  binary_idx: {binary_idx}, bin_idx: {bin_idx}", 2)
    
    # Omega_GW for this binary
    omega = omega_GW(freq, h_circ_contrib)
    
    if binary_idx is not None and binary_idx < DEBUG_CONFIG['print_first_N']:
        debug_print(f"  Omega_GW: {omega:.6e}", 2)
    
    # Get pulsar pairs
    pairs = pulsar_pair_data['pairs']
    gamma_vals = pulsar_pair_data['gamma']
    N_pairs = len(pairs)
    
    # Get PSDs for each pulsar at this frequency
    psd_i = psd_matrix[pairs[:, 0]]  # PSD for first pulsar in each pair
    psd_j = psd_matrix[pairs[:, 1]]  # PSD for second pulsar in each pair
    
    # DEBUG: Check PSD values
    if binary_idx is not None and binary_idx < DEBUG_CONFIG['print_first_N']:
        debug_print(f"  PSD statistics across {N_pairs} pairs:", 2)
        debug_print(f"    psd_i range: [{np.min(psd_i):.3e}, {np.max(psd_i):.3e}]", 2)
        debug_print(f"    psd_j range: [{np.min(psd_j):.3e}, {np.max(psd_j):.3e}]", 2)
        debug_print(f"    psd_i * psd_j range: [{np.min(psd_i*psd_j):.3e}, {np.max(psd_i*psd_j):.3e}]", 2)
    
    # Vectorized calculation across all pairs
    numerator = omega**2 * gamma_vals**2
    denominator = freq**6 * psd_i * psd_j
    
    # DEBUG: Check numerator and denominator
    if binary_idx is not None and binary_idx < DEBUG_CONFIG['print_first_N']:
        debug_print(f"  Numerator (omega^2 * gamma^2):", 2)
        debug_print(f"    omega^2: {omega**2:.3e}", 2)
        debug_print(f"    gamma^2 range: [{np.min(gamma_vals**2):.3e}, {np.max(gamma_vals**2):.3e}]", 2)
        debug_print(f"    numerator range: [{np.min(numerator):.3e}, {np.max(numerator):.3e}]", 2)
        debug_print(f"  Denominator (f^6 * psd_i * psd_j):", 2)
        debug_print(f"    f^6: {freq**6:.3e}", 2)
        debug_print(f"    denominator range: [{np.min(denominator):.3e}, {np.max(denominator):.3e}]", 2)
    
    # Integration over frequency bin
    bin_edges = strain_data.get('bin_edges', None)
    if bin_edges is not None and bin_idx < len(bin_edges) - 1:
        delta_f = bin_edges[bin_idx + 1] - bin_edges[bin_idx]
    else:
        delta_f = freq * 0.1  # Default 10% bandwidth
        debug_print(f"  WARNING: Using default delta_f = {delta_f:.3e} Hz (10% of freq)", 1)
    
    if binary_idx is not None and binary_idx < DEBUG_CONFIG['print_first_N']:
        debug_print(f"  Delta_f (frequency bin width): {delta_f:.3e} Hz", 2)
        debug_print(f"    Fractional bandwidth: {delta_f/freq:.3f}", 2)
    
    # Calculate integrand
    integrand = numerator / denominator * delta_f
    
    # DEBUG: Check integrand values
    if binary_idx is not None and binary_idx < DEBUG_CONFIG['print_first_N']:
        debug_print(f"  Integrand (before sum):", 2)
        debug_print(f"    range: [{np.min(integrand):.3e}, {np.max(integrand):.3e}]", 2)
        debug_print(f"    mean: {np.mean(integrand):.3e}", 2)
        debug_print(f"    sum: {np.sum(integrand):.3e}", 2)
        
        # Show top contributors
        top_indices = np.argsort(integrand)[-5:][::-1]
        debug_print(f"    Top 5 pair contributions:", 2)
        for rank, idx in enumerate(top_indices, 1):
            i, j = pairs[idx]
            debug_print(f"      {rank}. Pair ({i},{j}): {integrand[idx]:.3e} (gamma={gamma_vals[idx]:.3f})", 3)
    
    # Prefactor
    prefactor = (H0**2 / (4 * np.pi**2))**2 * T_obs
    
    if binary_idx is not None and binary_idx < DEBUG_CONFIG['print_first_N']:
        debug_print(f"  Prefactor:", 2)
        debug_print(f"    H0 = {H0:.3e}", 2)
        debug_print(f"    T_obs = {T_obs:.3e} s ({T_obs/(365.25*86400):.2f} yr)", 2)
        debug_print(f"    (H0^2/(4*pi^2))^2 = {(H0**2 / (4 * np.pi**2))**2:.3e}", 2)
        debug_print(f"    Full prefactor = {prefactor:.3e}", 2)
    
    # Sum over all pairs
    SNR_squared_total = np.sum(integrand) * prefactor
    
    # CRITICAL DEBUG: Final SNR check
    if binary_idx is not None and binary_idx < DEBUG_CONFIG['print_first_N']:
        debug_print(f"  FINAL SNR² = {SNR_squared_total:.6e}", 2)
        debug_print(f"  FINAL SNR = {np.sqrt(SNR_squared_total):.6e}", 2)
        
        # Check if this is suspiciously high
        if SNR_squared_total > 1.0:
            debug_print(f"  ⚠️  WARNING: SNR² > 1 for single binary!", 1)
            debug_print(f"  ⚠️  This suggests an error in units or formula!", 1)
    
    # Validate ranges
    if DEBUG_CONFIG['validate_ranges']:
        if np.isnan(SNR_squared_total) or np.isinf(SNR_squared_total):
            debug_print(f"  ❌ ERROR: Invalid SNR² (NaN or Inf) for binary {binary_idx}", 1)
        if SNR_squared_total < 0:
            debug_print(f"  ❌ ERROR: Negative SNR² = {SNR_squared_total}", 1)
    
    return SNR_squared_total


def optimal_SNR_total_population_vectorized(population, strain_data, pulsars, pulsar_noise_params, T_obs,
                                           batch_size=1000):
    """
    Compute total SNR² for entire population with memory-efficient batching.
    NOW WITH EXTENSIVE DEBUGGING
    
    Parameters
    ----------
    population : list of dict
        List of binary dictionaries
    strain_data : dict
        Strain data dictionary from generate_SMBHB_population
    pulsars : list
        List of pulsar objects
    pulsar_noise_params : dict
        Dictionary of noise parameters for all pulsars
    T_obs : float
        Observation time [s]
    batch_size : int, optional
        Number of binaries to process at once (for memory efficiency)
        
    Returns
    -------
    float
        Total SNR² summed over all binaries
    """
    debug_print(f"\n{'='*70}", 1)
    debug_print(f"STARTING POPULATION SNR CALCULATION", 1)
    debug_print(f"{'='*70}", 1)
    debug_print(f"Population size: {len(population)} binaries", 1)
    debug_print(f"Number of pulsars: {len(pulsars)}", 1)
    debug_print(f"Observation time: {T_obs:.3e} s ({T_obs/(365.25*86400):.2f} yr)", 1)
    
    # Precompute pulsar pair data once
    pulsar_pair_data = precompute_pulsar_pairs(pulsars)
    
    N_binaries = len(population)
    SNR_squared_total = 0.0
    
    # Get unique frequencies from population
    frequencies = np.array([binary['f'] for binary in population])
    
    debug_print(f"\nPopulation frequency statistics:", 1)
    debug_print(f"  Range: [{np.min(frequencies):.3e}, {np.max(frequencies):.3e}] Hz", 1)
    debug_print(f"  Mean: {np.mean(frequencies):.3e} Hz", 1)
    debug_print(f"  Median: {np.median(frequencies):.3e} Hz", 1)
    
    # Compute PSD matrix for all pulsars at all frequencies
    psd_matrix_full = compute_psd_matrix(pulsars, pulsar_noise_params, frequencies)
    
    # Check h_c values
    if 'h_c_individual' in strain_data:
        h_c_vals = strain_data['h_c_individual']
        debug_print(f"\nCharacteristic strain statistics:", 1)
        debug_print(f"  Range: [{np.min(h_c_vals):.3e}, {np.max(h_c_vals):.3e}]", 1)
        debug_print(f"  Mean: {np.mean(h_c_vals):.3e}", 1)
        debug_print(f"  Median: {np.median(h_c_vals):.3e}", 1)
    
    # Track SNR contributions
    snr_contributions = []
    
    # Process in batches to avoid memory issues
    debug_print(f"\nProcessing {N_binaries} binaries in batches of {batch_size}...", 1)
    
    for batch_start in range(0, N_binaries, batch_size):
        batch_end = min(batch_start + batch_size, N_binaries)
        
        if DEBUG_CONFIG['verbose_level'] >= 1:
            print(f"  Batch {batch_start//batch_size + 1}: binaries {batch_start}-{batch_end-1}")
        
        # Process this batch
        for i in range(batch_start, batch_end):
            binary = population[i]
            
            # Get PSD at this binary's frequency
            psd_at_freq = psd_matrix_full[:, i]
            
            # Compute SNR² for this binary
            SNR_sq = optimal_SNR_sq_single_BH_vectorized(
                binary, strain_data, pulsar_pair_data, psd_at_freq, T_obs, binary_idx=i
            )
            
            SNR_squared_total += SNR_sq
            snr_contributions.append(np.sqrt(SNR_sq))
            
            # Progress updates
            if i < 10 or (i + 1) % max(1, N_binaries // 20) == 0:
                cumulative_SNR = np.sqrt(SNR_squared_total)
                debug_print(f"    Binary {i+1}/{N_binaries}: SNR_this={np.sqrt(SNR_sq):.3f}, SNR_cumulative={cumulative_SNR:.3f}", 1)
    
    # Final statistics
    snr_contributions = np.array(snr_contributions)
    final_SNR = np.sqrt(SNR_squared_total)
    
    debug_print(f"\n{'='*70}", 1)
    debug_print(f"FINAL RESULTS", 1)
    debug_print(f"{'='*70}", 1)
    debug_print(f"Total SNR² = {SNR_squared_total:.6e}", 1)
    debug_print(f"Total SNR = {final_SNR:.6f}", 1)
    debug_print(f"\nPer-binary SNR statistics:", 1)
    debug_print(f"  Range: [{np.min(snr_contributions):.3e}, {np.max(snr_contributions):.3e}]", 1)
    debug_print(f"  Mean: {np.mean(snr_contributions):.3e}", 1)
    debug_print(f"  Median: {np.median(snr_contributions):.3e}", 1)
    debug_print(f"  Top 10 loudest binaries (by SNR):", 1)
    top_10_idx = np.argsort(snr_contributions)[-10:][::-1]
    for rank, idx in enumerate(top_10_idx, 1):
        debug_print(f"    {rank}. Binary {idx}: SNR = {snr_contributions[idx]:.3f}", 1)
    
    # CRITICAL CHECKS
    debug_print(f"\n{'='*70}", 1)
    debug_print(f"DIAGNOSTIC CHECKS", 1)
    debug_print(f"{'='*70}", 1)
    
    expected_SNR_per_binary = 4.0 / np.sqrt(4000)  # ~0.063
    debug_print(f"Expected SNR per binary (if SNR=4 for 4000): ~{expected_SNR_per_binary:.3f}", 1)
    debug_print(f"Actual mean SNR per binary: {np.mean(snr_contributions):.3f}", 1)
    debug_print(f"Ratio (actual/expected): {np.mean(snr_contributions)/expected_SNR_per_binary:.1f}x", 1)
    
    if final_SNR > 10:
        debug_print(f"\n⚠️  WARNING: SNR is very high ({final_SNR:.1f})!", 1)
        debug_print(f"⚠️  Possible issues to check:", 1)
        debug_print(f"    1. Are h_c values too large? (Check strain calculation)", 1)
        debug_print(f"    2. Are PSD values too small? (Check noise parameters)", 1)
        debug_print(f"    3. Is delta_f too large? (Check frequency binning)", 1)
        debug_print(f"    4. Is T_obs correct? (Should be in seconds)", 1)
        debug_print(f"    5. Are there unit conversion errors?", 1)
    
    return SNR_squared_total


def find_N_needed_for_target_SNR_optimized(population, strain_data, pulsars, pulsar_noise_params,
                                          target_SNR, T_obs):
    """
    Find minimum number of binaries needed to reach target SNR.
    WITH DEBUGGING
    """
    debug_print(f"\n{'='*70}", 1)
    debug_print(f"Finding N binaries needed for SNR = {target_SNR}", 1)
    debug_print(f"{'='*70}", 1)
    
    frequencies = strain_data['bin_centres']
    
    # Precompute pulsar pair data
    pulsar_pair_data = precompute_pulsar_pairs(pulsars)

    # Compute PSD matrix for all pulsars at all frequencies
    psd_matrix_full = compute_psd_matrix(pulsars, pulsar_noise_params, frequencies)
    
    # Cumulative SNR²
    SNR_squared_cumulative = 0.0
    
    for i, binary in enumerate(population):
        # Compute PSD at this binary's frequency
        freq = binary['f']
        psd_at_freq = psd_matrix_full[:, i]
        
        # Add this binary's contribution
        SNR_sq = optimal_SNR_sq_single_BH_vectorized(
            binary, strain_data, pulsar_pair_data, psd_at_freq, T_obs, binary_idx=i
        )
        
        SNR_squared_cumulative += SNR_sq
        SNR_current = np.sqrt(SNR_squared_cumulative)
        
        if (i + 1) % 100 == 0 or i < 10:
            debug_print(f"  N={i+1}: SNR={SNR_current:.3f}", 1)
        
        # Check if we've reached target
        if SNR_current >= target_SNR:
            N_needed = i + 1
            selected_population = population[:N_needed]
            debug_print(f"\n✓ Target SNR reached with N={N_needed} binaries", 1)
            debug_print(f"  Final SNR = {SNR_current:.3f}", 1)
            return selected_population, N_needed, SNR_current
    
    # If we get here, all binaries don't reach target
    SNR_current = np.sqrt(SNR_squared_cumulative)
    debug_print(f"\n⚠️  Target not reached with {len(population)} binaries", 1)
    debug_print(f"  Maximum SNR achieved = {SNR_current:.3f}", 1)
    return population, len(population), SNR_current


def compute_SNR_vs_N_binaries(population, strain_data, pulsars, pulsar_noise_params, T_obs, 
                              N_steps=100, sort_by='loudness'):
    """
    Compute SNR as a function of number of binaries included.
    WITH DEBUGGING
    """
    debug_print(f"\n{'='*70}", 1)
    debug_print(f"Computing SNR vs N binaries curve", 1)
    debug_print(f"  N_steps = {N_steps}, sort_by = {sort_by}", 1)
    debug_print(f"{'='*70}", 1)
    
    # Sort if requested
    if sort_by == 'loudness':
        h_c_contribs = np.array([binary.get('h_c_contrib', 0) for binary in population])
        sorted_indices = np.argsort(h_c_contribs)[::-1]
        population_sorted = [population[i] for i in sorted_indices]
        debug_print(f"Sorted by loudness: h_c range [{np.min(h_c_contribs):.3e}, {np.max(h_c_contribs):.3e}]", 1)
    else:
        population_sorted = population
    
    # Determine sampling points
    N_total = len(population_sorted)
    if N_steps > N_total:
        N_steps = N_total
    
    N_samples = np.logspace(0, np.log10(N_total), N_steps, dtype=int)
    N_samples = np.unique(N_samples)
    
    # Precompute pulsar data
    pulsar_pair_data = precompute_pulsar_pairs(pulsars)
    
    # Get all frequencies
    frequencies = np.array([binary['f'] for binary in population_sorted])
    psd_matrix_full = compute_psd_matrix(pulsars, pulsar_noise_params, frequencies)
    
    SNR_array = np.zeros(len(N_samples))
    SNR_squared_cumulative = 0.0
    
    binary_idx = 0
    
    for i, N_target in enumerate(N_samples):
        # Add binaries until we reach N_target
        while binary_idx < N_target:
            binary = population_sorted[binary_idx]
            psd_at_freq = psd_matrix_full[:, binary_idx]
            
            SNR_sq = optimal_SNR_sq_single_BH_vectorized(
                binary, strain_data, pulsar_pair_data, psd_at_freq, T_obs,
                binary_idx=binary_idx
            )
            
            SNR_squared_cumulative += SNR_sq
            binary_idx += 1
        
        SNR_array[i] = np.sqrt(SNR_squared_cumulative)
        
        if i % max(1, len(N_samples) // 10) == 0:
            debug_print(f"  N={N_target}: SNR={SNR_array[i]:.3f}", 1)
    
    debug_print(f"\nFinal: N={N_total}, SNR={SNR_array[-1]:.3f}", 1)
    
    return {
        'N_array': N_samples,
        'SNR_array': SNR_array
    }


# Memory-efficient version with generator
def SNR_contribution_generator(population, strain_data, pulsars, pulsar_noise_params, T_obs):
    """
    Generator that yields SNR² contribution for each binary sequentially.
    Memory-efficient for very large populations.
    WITH DEBUGGING
    """
    pulsar_pair_data = precompute_pulsar_pairs(pulsars)
    frequencies = np.array([binary['f'] for binary in population])
    psd_matrix_full = compute_psd_matrix(pulsars, pulsar_noise_params, frequencies)
    
    for i, binary in enumerate(population):
        psd_at_freq = psd_matrix_full[:, i]
        
        SNR_sq = optimal_SNR_sq_single_BH_vectorized(
            binary, strain_data, pulsar_pair_data, psd_at_freq, T_obs, binary_idx=i
        )
        
        yield SNR_sq


# ============================================================================
# UTILITY FUNCTIONS FOR DEBUGGING
# ============================================================================

def save_debug_snapshot(population, strain_data, pulsars, pulsar_noise_params, T_obs, filename='debug_snapshot.npz'):
    """Save all intermediate values for offline analysis."""
    frequencies = np.array([binary['f'] for binary in population])
    psd_matrix = compute_psd_matrix(pulsars, pulsar_noise_params, frequencies)
    
    pulsar_pair_data = precompute_pulsar_pairs(pulsars)
    
    h_c_vals = strain_data.get('h_c_individual', np.zeros(len(population)))
    
    np.savez(filename,
             frequencies=frequencies,
             h_c_individual=h_c_vals,
             psd_matrix=psd_matrix,
             gamma_values=pulsar_pair_data['gamma'],
             pairs=pulsar_pair_data['pairs'],
             T_obs=T_obs,
             N_pulsars=len(pulsars),
             N_binaries=len(population))
    
    debug_print(f"Debug snapshot saved to {filename}", 1)


def analyze_debug_snapshot(filename='debug_snapshot.npz'):
    """Analyze saved debug snapshot."""
    data = np.load(filename)
    
    print("\n" + "="*70)
    print("DEBUG SNAPSHOT ANALYSIS")
    print("="*70)
    print(f"File: {filename}")
    print(f"N_pulsars: {data['N_pulsars']}")
    print(f"N_binaries: {data['N_binaries']}")
    print(f"T_obs: {data['T_obs']:.3e} s ({data['T_obs']/(365.25*86400):.2f} yr)")
    
    print("\nFrequencies:")
    print(f"  Range: [{np.min(data['frequencies']):.3e}, {np.max(data['frequencies']):.3e}] Hz")
    print(f"  Mean: {np.mean(data['frequencies']):.3e} Hz")
    
    print("\nCharacteristic strain:")
    print(f"  Range: [{np.min(data['h_c_individual']):.3e}, {np.max(data['h_c_individual']):.3e}]")
    print(f"  Mean: {np.mean(data['h_c_individual']):.3e}")
    
    print("\nPSD matrix:")
    print(f"  Shape: {data['psd_matrix'].shape}")
    print(f"  Range: [{np.min(data['psd_matrix']):.3e}, {np.max(data['psd_matrix']):.3e}]")
    print(f"  Mean: {np.mean(data['psd_matrix']):.3e}")
    
    print("\nGamma (HD) coefficients:")
    print(f"  Range: [{np.min(data['gamma_values']):.3f}, {np.max(data['gamma_values']):.3f}]")
    print(f"  Mean: {np.mean(data['gamma_values']):.3f}")




# ============================================================================
# SKETCH OF CODE
# ============================================================================
'''
For the optimal SNR calculation, we need:
* overlap function (Hellings-Downs), easy
* pulsar noise PSD function - need to include white noise as well, and do sampling I think to get to this (get different populations of pulsar)
* SMBHB PSD - Riccardo's very useful code


1. Pulsar noise PSD function
    - Input: pulsar noise params dict, frequency array
    - Output: PSD array
    * To get this, I can use the noise parameters from the JSON file
    * For each pulsar:
        - compute the red noise PSD using the formula
        PSD(f) = 10^(2 x log10_A) * (f/fyr)^(-gamma)
        where we have all of the parameters so no worries there
        - compute the white noise
    
2.
3.
4.
5.
'''