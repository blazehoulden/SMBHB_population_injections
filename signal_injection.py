import numpy as np
from copy import deepcopy
from config import c, G, Msun, pc


def strain_amplitude(Mc, f, D):
    """Calculate characteristic strain amplitude for circular binary."""
    D_si = D * 1e6 * pc
    return (2 * (G * Mc)**(5/3) * (np.pi * f)**(2/3)) / (c**4 * D_si)


def antenna_response(psr_ra, psr_dec, src_ra, src_dec, psi):
    """Compute antenna pattern functions F+ and Fx."""
    k_hat = np.array([
        np.cos(src_dec) * np.cos(src_ra),
        np.cos(src_dec) * np.sin(src_ra),
        np.sin(src_dec),
    ])
    
    p_hat = np.array([
        np.cos(psr_dec) * np.cos(psr_ra),
        np.cos(psr_dec) * np.sin(psr_ra),
        np.sin(psr_dec),
    ])

    m_hat = np.array([np.sin(src_ra), -np.cos(src_ra), 0.0])
    n_hat = np.cross(k_hat, m_hat)

    m_rot = np.cos(psi) * m_hat + np.sin(psi) * n_hat
    n_rot = -np.sin(psi) * m_hat + np.cos(psi) * n_hat
    m_hat, n_hat = m_rot, n_rot

    denom = 1 + np.dot(k_hat, p_hat)
    Fp = 0.5 * ((np.dot(p_hat, m_hat)**2 - np.dot(p_hat, n_hat)**2) / denom)
    Fx = (np.dot(p_hat, m_hat) * np.dot(p_hat, n_hat)) / denom
    
    return Fp, Fx


def r_k(t, psr, binary):
    """Calculate timing residual from single circular SMBHB (Earth term only)."""
    f = binary['f']
    Mc = binary['Mc']
    D = binary['D']
    ra = binary['ra']
    dec = binary['dec']
    psi = binary.get('psi', 0.0)
    phi0 = binary.get('phi0', 0.0)

    h0 = strain_amplitude(Mc, f, D)
    ra_psr = psr._raj
    dec_psr = psr._decj
    Fp, Fx = antenna_response(ra_psr, dec_psr, ra, dec, psi)

    t_ref = t[0]
    t_rel = t - t_ref
    phase = 2 * np.pi * f * t_rel + phi0
    
    h_plus = h0 * np.sin(phase)
    h_cross = h0 * np.cos(phase)

    r = (Fp * h_plus + Fx * h_cross) / (2 * np.pi * f)
    return r


def population_residuals(t, psr, population):
    """Calculate total timing residual from entire SMBHB population."""
    total_r = np.zeros_like(t)
    for binary in population:
        total_r += r_k(t, psr, binary)
    return total_r


# =====================================================================
# VECTORIZED FUNCTIONS (10-50x faster for large populations)
# =====================================================================

def antenna_response_vectorized(psr_ra, psr_dec, src_ra_arr, src_dec_arr, psi_arr):
    """
    Vectorized antenna response for multiple sources.
    
    Args:
        psr_ra, psr_dec: scalar pulsar position
        src_ra_arr, src_dec_arr, psi_arr: arrays of source parameters (N_binaries,)
    
    Returns:
        Fp_arr, Fx_arr: arrays of antenna patterns (N_binaries,)
    """
    N = len(src_ra_arr)
    
    # Pulsar direction (same for all sources)
    p_hat = np.array([
        np.cos(psr_dec) * np.cos(psr_ra),
        np.cos(psr_dec) * np.sin(psr_ra),
        np.sin(psr_dec),
    ])
    
    # Source directions (vectorized)
    k_hat = np.array([
        np.cos(src_dec_arr) * np.cos(src_ra_arr),
        np.cos(src_dec_arr) * np.sin(src_ra_arr),
        np.sin(src_dec_arr),
    ])  # Shape: (3, N_binaries)
    
    # Base vectors (vectorized)
    m_hat = np.array([
        np.sin(src_ra_arr),
        -np.cos(src_ra_arr),
        np.zeros(N)
    ])  # Shape: (3, N_binaries)
    
    # Cross product k × m (vectorized)
    n_hat = np.cross(k_hat.T, m_hat.T).T  # Shape: (3, N_binaries)
    
    # Rotate by polarization angle psi (vectorized)
    cos_psi = np.cos(psi_arr)
    sin_psi = np.sin(psi_arr)
    
    m_rot = cos_psi[np.newaxis, :] * m_hat + sin_psi[np.newaxis, :] * n_hat
    n_rot = -sin_psi[np.newaxis, :] * m_hat + cos_psi[np.newaxis, :] * n_hat
    
    # Denominator (vectorized dot product)
    denom = 1 + np.einsum('i,ij->j', p_hat, k_hat)  # Shape: (N_binaries,)
    
    # Antenna patterns (vectorized)
    p_dot_m = np.einsum('i,ij->j', p_hat, m_rot)  # Shape: (N_binaries,)
    p_dot_n = np.einsum('i,ij->j', p_hat, n_rot)  # Shape: (N_binaries,)
    
    Fp_arr = 0.5 * ((p_dot_m**2 - p_dot_n**2) / denom)
    Fx_arr = (p_dot_m * p_dot_n) / denom
    
    return Fp_arr, Fx_arr


def population_residuals_vectorized(t, psr, population, use_vectorized=True):
    """
    Vectorized computation of timing residuals from SMBHB population.
    
    Speedup: 10-50x faster than loop-based version for large populations.
    
    Args:
        t: TOA times (N_toas,)
        psr: pulsar object
        population: list of binary dicts
        use_vectorized: if False, fall back to loop (for debugging)
    
    Returns:
        total_r: timing residuals (N_toas,)
    """
    if not use_vectorized or len(population) < 10:
        # Fall back to loop for small populations or debugging
        return population_residuals(t, psr, population)
    
    N_binaries = len(population)
    N_toas = len(t)
    
    # Extract all parameters into arrays (vectorized)
    f_arr = np.array([b['f'] for b in population])
    Mc_arr = np.array([b['Mc'] for b in population])
    D_arr = np.array([b['D'] for b in population])
    ra_arr = np.array([b['ra'] for b in population])
    dec_arr = np.array([b['dec'] for b in population])
    psi_arr = np.array([b.get('psi', 0.0) for b in population])
    phi0_arr = np.array([b.get('phi0', 0.0) for b in population])
    
    # Compute strain amplitudes (vectorized)
    D_si = D_arr * 1e6 * pc
    h0_arr = (2 * (G * Mc_arr)**(5/3) * (np.pi * f_arr)**(2/3)) / (c**4 * D_si)
    
    # Compute antenna responses (vectorized)
    ra_psr = psr._raj
    dec_psr = psr._decj
    Fp_arr, Fx_arr = antenna_response_vectorized(
        ra_psr, dec_psr, ra_arr, dec_arr, psi_arr
    )
    
    # Compute phases (vectorized over time and binaries)
    t_ref = t[0]
    t_rel = t - t_ref
    # Broadcasting: (N_toas, 1) * (1, N_binaries) + (1, N_binaries)
    phase = 2 * np.pi * t_rel[:, np.newaxis] * f_arr[np.newaxis, :] + phi0_arr[np.newaxis, :]
    # Shape: (N_toas, N_binaries)
    
    # Compute waveforms (vectorized)
    h_plus = h0_arr[np.newaxis, :] * np.sin(phase)  # (N_toas, N_binaries)
    h_cross = h0_arr[np.newaxis, :] * np.cos(phase)
    
    # Compute timing residuals (vectorized)
    r_matrix = (Fp_arr[np.newaxis, :] * h_plus + 
                Fx_arr[np.newaxis, :] * h_cross) / (2 * np.pi * f_arr[np.newaxis, :])
    # Shape: (N_toas, N_binaries)
    
    # Sum over all binaries
    total_r = np.sum(r_matrix, axis=1)  # Shape: (N_toas,)
    
    return total_r


# =====================================================================
# UPDATE YOUR INJECTION FUNCTIONS TO USE VECTORIZATION
# =====================================================================

def inject_population_into_psrs(psrs, population, pure_signal=True, add=False, 
                                verbose=False, use_vectorized=True):
    """
    Inject SMBHB population signals into pulsar residuals.
    
    Args:
        use_vectorized: if True, use fast vectorized computation (default)
    """
    for psr in psrs:
        t_sec = np.asarray(psr.toas, dtype=float)
        
        # Use vectorized computation by default
        r_pop = population_residuals_vectorized(t_sec, psr, population, use_vectorized)

        if verbose:
            print(f"{psr.name}: signal RMS = {np.sqrt(np.var(r_pop))*1e6:.3f} μs")

        if pure_signal:
            if add:
                psr._residuals = psr.residuals + r_pop
            else:
                psr._residuals = r_pop
        else:
            psr._residuals = psr.residuals + r_pop

    return psrs


def inject_population_subset_cached(psrs, population, N_binaries, 
                                    psrs_injected_cache=None, pure_signal=True, 
                                    verbose=False, use_vectorized=True):
    """
    Inject first N binaries using pre-computed cached signals.
    
    Args:
        use_vectorized: if True, use fast vectorized computation (default)
    """
    if psrs_injected_cache is None:
        # Fall back to regular computation
        return inject_population_into_psrs(psrs, population[:N_binaries], 
                                          pure_signal=pure_signal, verbose=verbose,
                                          use_vectorized=use_vectorized)
    
    # Modify pulsars in-place (no copying)
    for psr in psrs:
        psr_name = psr.name
        
        # Sum up individual binary signals up to N
        if psr_name in psrs_injected_cache:
            total_r = np.zeros(len(psr.toas), dtype=float)
            for bin_idx in range(N_binaries):
                if bin_idx in psrs_injected_cache[psr_name]:
                    total_r += psrs_injected_cache[psr_name][bin_idx]
            
            r_pop = total_r
        else:
            # Fallback: compute fresh with vectorization
            t_sec = np.asarray(psr.toas, dtype=float)
            r_pop = population_residuals_vectorized(
                t_sec, psr, population[:N_binaries], use_vectorized
            )
        
        if verbose:
            print(f"{psr.name}: signal RMS = {np.sqrt(np.var(r_pop))*1e6:.3f} μs")
        
        # Modify residuals in-place
        if pure_signal:
            psr._residuals = r_pop
        else:
            psr._residuals = psr.residuals + r_pop
    
    return psrs


# CACHING STRATEGY FOR ENSEMBLE SEARCHES
# Pre-compute and cache individual binary signals for reuse

def precompute_binary_signals(psrs, population, cache=None):
    """
    Pre-compute timing residual signals from EACH binary individually.
    
    Returns a cache dict: {psr_name: {binary_idx: residual_array}}
    
    This allows fast computation of subset signals:
    N-binary signal = sum of individual binary signals (no recomputation needed)
    """
    if cache is None:
        cache = {}
    
    for psr in psrs:
        psr_name = psr.name
        if psr_name not in cache:
            cache[psr_name] = {}
        
        t_sec = np.asarray(psr.toas, dtype=float)
        
        for bin_idx, binary in enumerate(population):
            if bin_idx not in cache[psr_name]:
                # Compute signal from this single binary
                cache[psr_name][bin_idx] = r_k(t_sec, psr, binary)
    
    return cache
