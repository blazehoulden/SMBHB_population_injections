"""
Vectorized and optimized SNR calculation for PTA detection of SMBHB population.
Memory-efficient implementation with proper array handling.
NOW WITH EXTENSIVE DEBUGGING TO DIAGNOSE HIGH SNR VALUES
"""

import numpy as np
from SMBHB_pop_synth import H0_KMS_MPC, MEGAPARSEC_IN_METERS
from config import generate_population

H0_S = H0_KMS_MPC * (1e3) / MEGAPARSEC_IN_METERS  # Convert H0 to SI units (1/s)

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
    omega = 2 * np.pi**2 / (3 * H0_S**2) * f**2 * h_cont**2
    return omega

def signal_psd(f, h_cont):
    """
    Power spectral density of the GW signal from a binary.
    
    Parameters
    ----------
    f : array_like
        Frequency [Hz]
    h_cont : array_like
        Characteristic strain contribution
        
    Returns
    -------
    array_like
        Signal PSD at given frequencies [seconds^3]
    """
    psd = h_cont**2 / (12 * np.pi**2 * f**3)
    # psd = h_cont**2 / (f) # Alternative definition from Thrane and Romano (2013)
    print("BH psd:", psd, " for h_cont:", h_cont, " at f:", f)
    return psd # [s^3]


def pulsar_psd(pulsar_noise_params, f, sigma_ns = 100.0, delta_t_yr = 1.0/20.0):
    """
    Power spectral density for a single pulsar's noise.
    
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
    sigma_ns : float
        RMS timing residuals [ns]
    delta_t_yr : float
        Cadence [years]
        
    Returns
    -------
    array_like
        PSD at given frequencies
    """
    fyr = 1.0 / (365.25 * 24 * 3600)  # 1/year in Hz
    pulsar_A = pulsar_noise_params['red_noise']['log10_A']
    pulsar_gamma = pulsar_noise_params['red_noise']['gamma']
    
    psd_red_noise = 10**(2 * pulsar_A) / (12 * np.pi**2) * (f / fyr)**(-pulsar_gamma) # yr^3
    psd_red_noise = psd_red_noise * (365.25 * 24 * 3600)**3  # Convert to seconds^3

    sigma_s = sigma_ns * 1e-9  # Convert ns to seconds
    delta_t_s = delta_t_yr * 365.25 * 24 * 3600  # Convert years to seconds
    psd_white_noise = 2 * sigma_s**2 * delta_t_s # Footnote 2 https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.104028#fn2
    psd = psd_red_noise + psd_white_noise
    print("Pulsar psd:", psd, " red:", psd_red_noise, " white:", psd_white_noise)
    return psd # [s^3]


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
    
    missing_pulsars = []
    no_red_noise = []
    
    for i, pulsar in enumerate(pulsars):
        # Check if this pulsar has red noise parameters
        if pulsar.name in pulsar_noise_params:
            if pulsar_noise_params[pulsar.name]['red_noise']:
                psd_matrix[i, :] = pulsar_psd(pulsar_noise_params[pulsar.name], frequencies)
                
            else:
                no_red_noise.append(pulsar.name)
        else:
            missing_pulsars.append(pulsar.name)

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
    h_circ_contrib = binary.get('h_c_contrib', 0)
    
    # Omega_GW for this binary
    omega = omega_GW(freq, h_circ_contrib)
    
    # Get pulsar pairs
    pairs = pulsar_pair_data['pairs']
    gamma_vals = pulsar_pair_data['gamma']
    N_pairs = len(pairs)
    
    # Get PSDs for each pulsar at this frequency
    psd_i = psd_matrix[pairs[:, 0]]  # PSD for first pulsar in each pair
    psd_j = psd_matrix[pairs[:, 1]]  # PSD for second pulsar in each pair
    
    # # Vectorized calculation across all pairs
    # numerator = omega**2 * gamma_vals**2
    # denominator = freq**6 * psd_i * psd_j

    # using my PSD
    signal_psd_val = signal_psd(freq, h_circ_contrib)
    numerator = signal_psd_val**2 * gamma_vals**2
    denominator = (signal_psd_val + psd_i) * (signal_psd_val + psd_j)



    # Integration over frequency bin
    bin_edges = strain_data.get('bin_edges', None)
    if bin_edges is not None and bin_idx < len(bin_edges) - 1:
        delta_f = bin_edges[bin_idx + 1] - bin_edges[bin_idx]
    else:
        delta_f = freq * 0.1  # Default 10% bandwidth
        print(f"  WARNING: Using default delta_f = {delta_f:.3e} Hz (10% of freq)", 1)
    
    # Calculate integrand
    integrand = numerator / denominator * delta_f
    # Prefactor
    # prefactor = (H0_S**2 / (4 * np.pi**2))**2 * T_obs
    prefactor = 2 * T_obs  # From PTA SNR formula

    # Sum over all pairs
    SNR_squared_total = np.sum(integrand) * prefactor

    return SNR_squared_total


def find_N_needed_for_target_SNR_optimized(population, strain_data, pulsars, pulsar_noise_params,
                                          target_SNR, T_obs):
    """
    Find minimum number of binaries needed to reach target SNR.
    """
    
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
        freq_bin = binary['freq_bin']
        psd_at_freq = psd_matrix_full[:, freq_bin]
        
        # Add this binary's contribution
        SNR_sq = optimal_SNR_sq_single_BH_vectorized(
            binary, strain_data, pulsar_pair_data, psd_at_freq, T_obs, binary_idx=i
        )
        
        SNR_squared_cumulative += SNR_sq
        SNR_current = np.sqrt(SNR_squared_cumulative)

        print(f"  Added binary {i+1}/{len(population)}: Cumulative SNR = {SNR_current:.3e}", "Binary contrib:", np.sqrt(SNR_sq))

        # Check if we've reached target
        if SNR_current >= target_SNR:
            N_needed = i + 1
            selected_population = population[:N_needed]
            print(f"\n✓ Target SNR reached with N={N_needed} binaries")
            print(f"  Final SNR = {SNR_current:.3f}")
            return selected_population, N_needed, SNR_current

    # If we get here, all binaries don't reach target
    SNR_current = np.sqrt(SNR_squared_cumulative)
    print(f"\n⚠️  Target not reached with {len(population)} binaries")
    print(f"  Maximum SNR achieved = {SNR_current:.3f}")
    return population, len(population), SNR_current


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
    # Precompute pulsar pair data once
    pulsar_pair_data = precompute_pulsar_pairs(pulsars)
    
    N_binaries = len(population)
    SNR_squared_total = 0.0
    
    # Get unique frequencies from population
    frequencies = np.array([binary['f'] for binary in population])
    
    # Compute PSD matrix for all pulsars at all frequencies
    psd_matrix_full = compute_psd_matrix(pulsars, pulsar_noise_params, frequencies)
    
    # Track SNR contributions
    snr_contributions = []
        
    for batch_start in range(0, N_binaries, batch_size):
        batch_end = min(batch_start + batch_size, N_binaries)
        
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
    
    # Final statistics
    snr_contributions = np.array(snr_contributions)
    final_SNR = np.sqrt(SNR_squared_total)

    return SNR_squared_total


def compute_SNR_vs_N_binaries(population, strain_data, pulsars, pulsar_noise_params, T_obs, 
                              N_steps=100):
    """
    Compute SNR as a function of number of binaries included.
    """
    # Sort if requested
    
    # Determine sampling points
    N_total = len(population)
    if N_steps > N_total:
        N_steps = N_total
    
    N_samples = np.logspace(0, np.log10(N_total), N_steps, dtype=int)
    N_samples = np.unique(N_samples)
    
    # Precompute pulsar data
    pulsar_pair_data = precompute_pulsar_pairs(pulsars)
    
    # Get all frequencies
    frequencies = np.array([binary['f'] for binary in population])
    psd_matrix_full = compute_psd_matrix(pulsars, pulsar_noise_params, frequencies)
    
    SNR_array = np.zeros(len(N_samples))
    SNR_squared_cumulative = 0.0
    
    binary_idx = 0
    
    for i, N_target in enumerate(N_samples):
        # Add binaries until we reach N_target
        while binary_idx < N_target:
            binary = population[binary_idx]
            psd_at_freq = psd_matrix_full[:, binary_idx]
            
            SNR_sq = optimal_SNR_sq_single_BH_vectorized(
                binary, strain_data, pulsar_pair_data, psd_at_freq, T_obs,
                binary_idx=binary_idx
            )
            
            SNR_squared_cumulative += SNR_sq
            binary_idx += 1
        
        SNR_array[i] = np.sqrt(SNR_squared_cumulative)

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