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


def inject_population_into_psrs(psrs, population, pure_signal=True, add=False, verbose=False):
    """Inject SMBHB population signals into pulsar residuals."""
    psrs_injected = []

    for psr in psrs:
        psr_inj = deepcopy(psr)
        t_sec = np.asarray(psr_inj.toas, dtype=float)
        r_pop = population_residuals(t_sec, psr_inj, population)

        if verbose:
            print(f"{psr.name}: signal RMS = {np.sqrt(np.var(r_pop))*1e6:.3f} μs")

        if pure_signal:
            if add:
                psr_inj._residuals = psr_inj.residuals + r_pop
            else:
                psr_inj._residuals = r_pop
        else:
            psr_inj._residuals = psr_inj.residuals + r_pop

        psrs_injected.append(psr_inj)

    return psrs_injected



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


def inject_population_subset_cached(psrs, population, N_binaries, 
                                    psrs_injected_cache=None, pure_signal=True, 
                                    verbose=False):
    """
    Inject first N binaries using pre-computed cached signals.
    
    Usage:
        # First call: compute cache
        cache = precompute_binary_signals(psrs, population)
        
        # Then for each N in your search:
        psrs_N = inject_population_subset_cached(psrs, population, N, 
                                                  psrs_injected_cache=cache)
    """
    if psrs_injected_cache is None:
        # Fall back to regular computation
        return inject_population_into_psrs(psrs, population[:N_binaries], 
                                          pure_signal=pure_signal, verbose=verbose)
    
    psrs_injected = []
    
    for psr in psrs:
        psr_name = psr.name
        psr_inj = deepcopy(psr)
        
        # Sum up individual binary signals up to N
        if psr_name in psrs_injected_cache:
            total_r = np.zeros_like(psr_inj.toas, dtype=float)
            for bin_idx in range(N_binaries):
                if bin_idx in psrs_injected_cache[psr_name]:
                    total_r += psrs_injected_cache[psr_name][bin_idx]
            
            r_pop = total_r
        else:
            # Fallback: compute fresh
            t_sec = np.asarray(psr_inj.toas, dtype=float)
            r_pop = population_residuals(t_sec, psr_inj, population[:N_binaries])
        
        if verbose:
            print(f"{psr.name}: signal RMS = {np.sqrt(np.var(r_pop))*1e6:.3f} μs")
        
        if pure_signal:
            psr_inj._residuals = r_pop
        else:
            psr_inj._residuals = psr_inj.residuals + r_pop
        
        psrs_injected.append(psr_inj)
    
    return psrs_injected