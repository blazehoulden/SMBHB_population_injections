"""
O(N log N) SMBHB population injection
==========================================================

Key ideas
---------
1.  Pre-compute per-binary *time-independent* amplitudes A_k, B_k and store
    them on the binary dict.  These absorb h0, Fp/Fx, iota, phi0 completely,
    so the TOA-dependent work per binary collapses to:

        r_k(t) = A_k * sin(2π f_k t)  +  B_k * cos(2π f_k t)

2.  Bin {A_k, B_k, f_k} onto a uniform frequency grid and evaluate the entire
    population sum with a single real IFFT — O(N_freq log N_freq) regardless
    of how many billions of binaries you have.

3.  For non-uniformly sampled TOAs (the realistic case) use NUFFT type-2
    (finufft) which costs O(N_freq log N_freq + N_toa) — still independent of
    the population size.

4.  A "direct batched" path (numpy matmul) is provided as a fallback and for
    small populations where FFT overhead is not worth paying.

Performance summary (rough)
---------------------------
Method              | Cost                     | 10^9 binaries?
--------------------|--------------------------|----------------
Old vectorised      | O(N_toa * N_binary)      | No — 800 GB RAM
New FFT             | O(N_binary + N_f log N_f)| Yes — ~8 GB RAM
New NUFFT           | O(N_binary + N_f log N_f + N_toa) | Yes

Pre-computation of A_k, B_k
----------------------------
Call `precompute_amplitudes(population, psr)` once per pulsar before
any injection loop.  The amplitudes are stored in-place on each binary dict
under keys 'A_<psr_name>' and 'B_<psr_name>'.  If you are simulating many
realisations with the same population, this pre-computation is done once and
reused for free.

Usage example
-------------
    from signal_injection import (
        precompute_amplitudes,
        inject_population_nufft,
    )

    # One-time setup: draw orbital params + precompute amplitudes
    population = draw_population(N=int(1e9), ...)   # your sampler
    for psr in psrs:
        precompute_amplitudes(population, psr)

    # Fast injection (pick one):
    inject_population_nufft(psrs, population, N_freq=2**20) # non-uniform TOAs

Dependencies: numpy (always), finufft.
"""

import json
from logging import config

from SMBHB_pop_synth import precompute_amplitudes
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from config import c, G, pc
import finufft 
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
from astropy.coordinates import SkyCoord
import astropy.units as u

# ──────────────────────────────────────────────────────────────────────────────
# 1.  STRAIN AMPLITUDE  
# ──────────────────────────────────────────────────────────────────────────────
SOLAR_MASS = 1.989e30  # kg

def strain_amplitude(Mc, fGW, d_comov, z):
    """
    Mc     : chirp mass in SOLAR MASSES
    fGW    : GW frequency in Hz
    d_comov: comoving distance in Mpc
    z      : redshift
    """
    Mc_kg      = Mc * SOLAR_MASS          # convert to kg
    f_rest_orb = 0.5 * fGW * (1 + z)
    d_comov_si = d_comov * 1e6 * pc       # Mpc -> m
    return (2 * (G * Mc_kg)**(5/3) * (2 * np.pi * f_rest_orb)**(2/3)) / (c**4 * d_comov_si)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  ANTENNA RESPONSE  (vectorised, unchanged logic)
# ──────────────────────────────────────────────────────────────────────────────

def antenna_response_vec(psr_ra, psr_dec, ra_arr, dec_arr, psi_arr):
    """Vectorised antenna response over N binaries. Returns Fp, Fx each (N,)."""
    N             = len(ra_arr)
    src_polar     = np.pi / 2 - dec_arr
    psr_polar     = np.pi / 2 - psr_dec

    omega_hat = np.array([
        -np.sin(src_polar) * np.cos(ra_arr),
        -np.sin(src_polar) * np.sin(ra_arr),
        -np.cos(src_polar),
    ])  # (3, N)

    p_hat = np.array([
        np.sin(psr_polar) * np.cos(psr_ra),
        np.sin(psr_polar) * np.sin(psr_ra),
        np.cos(psr_polar),
    ])  # (3,)

    m_hat = np.array([np.sin(ra_arr), -np.cos(ra_arr), np.zeros(N)])       # (3, N)
    n_hat = np.array([
        -np.cos(src_polar) * np.cos(ra_arr),
        -np.cos(src_polar) * np.sin(ra_arr),
         np.sin(src_polar),
    ])  # (3, N)

    cos_psi = np.cos(psi_arr); sin_psi = np.sin(psi_arr)
    m_rot   =  cos_psi * m_hat + sin_psi * n_hat
    n_rot   = -sin_psi * m_hat + cos_psi * n_hat

    # --- antenna patterns ---
    denom = 1.0 + np.dot(p_hat, omega_hat)        # (N,)
    p_m   = np.dot(p_hat, m_rot)                  # (N,)
    p_n   = np.dot(p_hat, n_rot)                  # (N,)

    Fp = 0.5 * (p_m**2 - p_n**2) / denom
    Fx =       (p_m   * p_n)     / denom
    return Fp, Fx


def _topk_indices_from_amplitudes(A_arr, B_arr, k, chunk_size=500_000):
    """Return top-k indices by A^2 + B^2 using chunked processing."""
    n = len(A_arr)
    if k <= 0 or n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    k_eff = min(int(k), n)
    top_idx = np.array([], dtype=np.int64)
    top_val = np.array([], dtype=np.float64)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        A_chunk = A_arr[start:stop]
        B_chunk = B_arr[start:stop]
        score_chunk = A_chunk * A_chunk + B_chunk * B_chunk

        if score_chunk.size <= k_eff:
            local_rel = np.arange(score_chunk.size, dtype=np.int64)
        else:
            local_rel = np.argpartition(score_chunk, score_chunk.size - k_eff)[-k_eff:]

        local_idx = local_rel + start
        local_val = score_chunk[local_rel]

        if top_idx.size == 0:
            top_idx = local_idx
            top_val = local_val
        else:
            top_idx = np.concatenate([top_idx, local_idx])
            top_val = np.concatenate([top_val, local_val])

        if top_idx.size > k_eff:
            keep = np.argpartition(top_val, top_val.size - k_eff)[-k_eff:]
            top_idx = top_idx[keep]
            top_val = top_val[keep]

    order = np.argsort(top_val)[::-1]
    return top_idx[order].astype(np.int64, copy=False), top_val[order].astype(np.float64, copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  DIRECT BATCHED PATH  (small populations, exact, no FFT)
# ──────────────────────────────────────────────────────────────────────────────

def inject_population_direct(
        psrs, population,
        pure_signal=True,
        chunk_size=50_000):
    """
    Direct batched injection using pre-computed A_k, B_k amplitudes.

    Cost:  O(N_toa * N_binary)  — same as before
    RAM:   O(N_toa * chunk_size) per chunk  — much lower than building full matrix

    This is exact and recommended for N_binary < ~10^5.  For larger populations
    use inject_population_fft or inject_population_nufft.

    Pre-requisite
    -------------
    Call precompute_amplitudes(population, psr) for each psr first.

    Key improvement over the old approach
    --------------------------------------
    The old _gw_residuals_vec built the full (N_toa × N_binary) phase matrix in
    one shot.  This path processes chunk_size binaries at a time, keeping RAM
    proportional to chunk_size rather than N_binary.  The phase calculation is
    still vectorised within each chunk.
    """

    phi0_arr = population.phi0
    f_arr    = population.f
    cos_phi0 = np.cos(phi0_arr)
    sin_phi0 = np.sin(phi0_arr)

    for psr in psrs:
        psr_name = psr.name
        if psr_name not in population.amp_A:
            precompute_amplitudes(population, psr)

        A_arr = population.amp_A[psr_name]
        B_arr = population.amp_B[psr_name]

        # S_k = coeff of sin(2π f t_rel), C_k = coeff of cos(2π f t_rel)
        S = A_arr * cos_phi0 - B_arr * sin_phi0
        C = A_arr * sin_phi0 + B_arr * cos_phi0

        t_sec = np.asarray(psr.toas, dtype=np.float64)
        t_rel = t_sec - t_sec[0]   # (N_toa,)

        # phase matrix (N_toa, N_binary)
        phase = 2 * np.pi * f_arr[np.newaxis, :] * t_rel[:, np.newaxis]

        r_new = (S * np.sin(phase) + C * np.cos(phase)).sum(axis=1)

        if pure_signal:
            psr._residuals = r_new
        else:
            psr._residuals = psr.residuals + r_new

    return psrs


# ──────────────────────────────────────────────────────────────────────────────
# 7.  POPULATION DRAWING WITH PRE-STORED ATTRIBUTES  (convenience)
# ──────────────────────────────────────────────────────────────────────────────

def draw_population_with_amplitudes(
        N, f_min, f_max,
        Mc_min=1e7, Mc_max=1e10,
        z_min=0.01, z_max=2.0,
        rng=None):
    """
    Draw a random population of N SMBHBs and pre-store all source-side
    attributes needed for injection.

    Draws uniformly in log-frequency, log-chirp-mass, and comoving volume.
    Orientation angles (ra, dec, psi, iota, phi0) are drawn isotropically.

    Storing h0, psi, phi0, iota in the dict means precompute_amplitudes
    reads them directly and skips recomputation.

    Parameters
    ----------
    N            : number of binaries
    f_min, f_max : GW frequency range (Hz)
    Mc_min/max   : chirp mass range (solar masses * G/c^3 in SI — pass in SI)
    z_min/max    : redshift range
    rng          : numpy random Generator (for reproducibility)

    Returns
    -------
    population : list of N binary dicts, each containing:
                 f, Mc, D_comov, z, h0, ra, dec, psi, iota, phi0
    """
    if rng is None:
        rng = np.random.default_rng()

    # Cosmology: comoving distance ≈ cz/H0 (good for z < 1; use astropy for z > 1)
    H0      = 70e3 / 3.086e22   # s^-1
    c_light = 3e8               # m/s
    Msun_kg = 1.989e30
    G_SI    = 6.674e-11

    # Log-uniform in frequency and chirp mass
    f_arr   = np.exp(rng.uniform(np.log(f_min),    np.log(f_max),    N))
    Mc_arr  = np.exp(rng.uniform(np.log(Mc_min),   np.log(Mc_max),   N)) * Msun_kg * G_SI / c_light**3
    # Comoving volume → uniform in z^3 approximation
    z_arr   = rng.uniform(z_min**(1/3), z_max**(1/3), N)**3
    D_comov = c_light * z_arr / H0 / 3.086e22   # Mpc

    # Isotropic sky and orientation
    ra_arr  = rng.uniform(0,        2*np.pi, N)
    dec_arr = np.arcsin(rng.uniform(-1, 1,   N))
    psi_arr = rng.uniform(0,        np.pi,   N)
    iota_arr= np.arccos(rng.uniform(-1, 1,   N))
    phi0_arr= rng.uniform(0,        2*np.pi, N)

    # Pre-compute h0 vectorised — store so precompute_binary_amplitudes can skip it
    f_rest  = 0.5 * (1 + z_arr) * f_arr
    D_SI    = D_comov * 1e6 * pc
    h0_arr  = (2 * (G_SI * Mc_arr)**(5/3) * (2*np.pi*f_rest)**(2/3)) / (c_light**4 * D_SI)

    population = [
        {
            'f':       float(f_arr[i]),
            'Mc':      float(Mc_arr[i]),
            'D_comov': float(D_comov[i]),
            'z':       float(z_arr[i]),
            'h0':      float(h0_arr[i]),
            'ra':      float(ra_arr[i]),
            'dec':     float(dec_arr[i]),
            'psi':     float(psi_arr[i]),
            'iota':    float(iota_arr[i]),
            'phi0':    float(phi0_arr[i]),
        }
        for i in range(N)
    ]
    return population


# ──────────────────────────────────────────────────────────────────────────────
# 8.  RECOMMENDED ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def inject_fast(psrs, population, method='auto', pure_signal=True, **kwargs):
    """
    Fast injection dispatcher.  Picks the best method automatically.

    method='auto'  (default):
        N < 1e5          → direct batched (exact, low overhead)
        1e5 ≤ N < 1e7   → FFT (excellent approximation, ~1000× faster)
        N ≥ 1e7          → NUFFT if finufft available, else FFT

    Precomputes amplitudes automatically if not already present.

    Parameters
    ----------
    psrs       : list of enterprise Pulsar objects
    population : list of binary dicts (drawn with draw_population_with_amplitudes
                 or manually with the required keys)
    method     : 'auto', 'direct', 'fft', 'nufft'
    pure_signal: replace residuals (True) or add (False)
    **kwargs   : passed through to the underlying injection function
    """
    N = len(population)

    # Check if amplitudes are already precomputed for first pulsar
    key = f'A_{psrs[0].name}'
    if key not in population[0]:
        print(f"Precomputing amplitudes for {len(psrs)} pulsars × {N:,} binaries...")
        for psr in psrs:
            precompute_amplitudes(population, psr)
        print("Precomputation done.")

    if method == 'auto':
        if N < 100_000:
            method = 'direct'
        elif N < 10_000_000:
            method = 'fft'
        else:
            try:
                import finufft
                method = 'nufft'
            except ImportError:
                method = 'fft'

    print(f"Injecting N={N:,} binaries via method='{method}'")

    if method == 'direct':
        return inject_population_direct(psrs, population, pure_signal=pure_signal, **kwargs)
    elif method == 'fft':
        return inject_population_fft(psrs, population, pure_signal=pure_signal, **kwargs)
    elif method == 'nufft':
        return inject_population_nufft(psrs, population, pure_signal=pure_signal, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: auto, direct, fft, nufft")
    



##### OLD CODE BELOW (for reference, to be deleted) #####

def strain_amplitude(Mc, fGW, d_comov, z):
    """Calculate strain amplitude for circular binary. See Eqn. 26 in https://arxiv.org/pdf/1003.0677"""
    f_rest_orb = 0.5 * fGW * (1 + z)
    d_comov_si = d_comov * 1e6 * pc
    return (2 * (G * Mc)**(5/3) * (2 * np.pi * f_rest_orb)**(2/3)) / (c**4 * d_comov_si)


def antenna_response(psr_ra, psr_dec, src_ra, src_dec, psi, norm = True):
    """
    Parameters
    ----------
    psr_ra, psr_dec : float
        Right ascension and declination of the pulsar [radians].
    src_ra, src_dec : float
        Right ascension and declination of the GW source [radians].
    psi : float
        Polarization angle of the GW source [radians].
        Returns
        -------
        Fp, Fx : float
            Antenna pattern functions for the plus and cross polarizations, https://arxiv.org/pdf/1003.0677 & Anholm et al. 2009.
    """
    # convert RA/Dec to polar/azimuthal angles for the pulsar and source
    psr_polar_angle = np.pi / 2 - psr_dec 
    psr_azimuthal_angle = psr_ra
    src_polar_angle = np.pi / 2 - src_dec
    src_azimuthal_angle = src_ra

    omega_hat = np.array([
        -np.sin(src_polar_angle) * np.cos(src_azimuthal_angle),
        -np.sin(src_polar_angle) * np.sin(src_azimuthal_angle),
        -np.cos(src_polar_angle),
    ])

    p_hat = np.array([
        np.sin(psr_polar_angle) * np.cos(psr_azimuthal_angle),
        np.sin(psr_polar_angle) * np.sin(psr_azimuthal_angle),
        np.cos(psr_polar_angle),
    ])

    m_hat = np.array(
        [np.sin(src_azimuthal_angle), 
         -np.cos(src_azimuthal_angle), 
         0.0])
    
    n_hat = np.array([
        -np.cos(src_polar_angle) * np.cos(src_azimuthal_angle),
        -np.cos(src_polar_angle) * np.sin(src_azimuthal_angle),
        np.sin(src_polar_angle)
    ])

    m_rot = np.cos(psi) * m_hat + np.sin(psi) * n_hat
    n_rot = -np.sin(psi) * m_hat + np.cos(psi) * n_hat
    m_hat, n_hat = m_rot, n_rot

    # changed sign. convention  to 1 - from 1 + for enterprise
    denom = 1 + np.dot(omega_hat, p_hat)
    Fp = 0.5  * ((np.dot(p_hat, m_hat)**2 - np.dot(p_hat, n_hat)**2) / denom)
    Fx =  (np.dot(p_hat, m_hat) * np.dot(p_hat, n_hat)) / denom

    return Fp, Fx



def r_k(t, psr, binary):
    """Calculate timing residual from single circular SMBHB (Earth term only) This assumption holds for most cases (see Appendix in https://arxiv.org/pdf/1003.0677)."""

    f = binary.f
    ra = binary.ra
    dec = binary.dec
    psi = binary.psi
    phi0 = binary.phi0
    iota = binary.iota
    h0 = binary.h0

    # pars = psr.pars()

    # if 'RAJ' not in pars or 'DECJ' not in pars:
    #     elong = psr['ELONG'].val  # radians
    #     elat  = psr['ELAT'].val   # radians

    #     coord = SkyCoord(lon=elong*u.rad, lat=elat*u.rad, frame='geocentricmeanecliptic')
    #     equatorial = coord.icrs

    #     psr_ra  = equatorial.ra.rad
    #     psr_dec = equatorial.dec.rad

    # else:
    psr_ra   = psr._raj
    psr_dec  = psr._decj

    Fp, Fx = antenna_response(psr_ra, psr_dec, ra, dec, psi)

    t_ref = t[0]
    t_rel = t - t_ref
    phase = 2 * np.pi * f * t_rel + phi0
    
    h_plus = h0 * (1 + np.cos(iota)**2) * np.sin(phase) # Eq. 40 http://arxiv.org/abs/2512.18822
    h_cross = h0 * (- 2 * np.cos(iota)) * np.cos(phase) # ""

    r = (Fp * h_plus + Fx * h_cross) / (2 * np.pi * f) # Eq. 4.21-23 nHz GW Astronomer
    return r


#### EFFECT OF NOISE REALISATIONS ON RESIDUALS AND SNR CALCULATION - UNSURE IF NECESSARY ####

def draw_red_noise_residuals(psr, log10_A, gamma, Tobs, nmodes=30):
    # Build F matrix
    F, Ffreqs = createfourierdesignmatrix_red(psr.toas, nmodes=nmodes)
    
    # Build the PSD diagonal (phi vector) — power-law red noise
    # P(f) = A^2 / (12 pi^2) * (f / f_yr)^-gamma * f_yr^-3
    f_yr = 1 / (365.25 * 24 * 3600)
    A = 10**log10_A
    kappa = (A**2 / (12 * np.pi**2)) * (Ffreqs / f_yr)**(-gamma) * f_yr**-3 * (1 / Tobs)
    
    # a are the Fourier coefficients, probability distribution is a normal dist with variance phi: a ~ N(0, phi)
    a = np.sqrt(kappa) * np.random.randn(2 * nmodes)

    # checked that kappa matches get_phi in enterprise using: 
    # 
    # for i, (psr, sc) in enumerate(zip(psrs_clean, pta_check._signalcollections)):
    
    # # Find just the red noise signal within this pulsar's signal collection
    # for signal in sc._signals:
    #     if 'red_noise' in signal.name and 'gw' not in signal.name:
    #         phi_rn = signal.get_phi(params_check)
    #         print(f"{psr.name} red noise phi: {phi_rn}")
    #         break
    
    # # Your kappa (note: needs repeat for sin+cos)
    # F, freqs = createfourierdesignmatrix_red(psr.toas, nmodes=30, Tspan=Tspan)
    # f_yr = 1 / (365.25 * 24 * 3600)
    # A     = 10**params_check[f'{psr.name}_red_noise_log10_A']
    # gamma = params_check[f'{psr.name}_red_noise_gamma']
    # kappa = (A**2 / (12*np.pi**2)) * (freqs/f_yr)**(-gamma) * f_yr**-3 * (1/Tspan)
    # kappa_full = np.repeat(kappa, 2)  # shape (60,) — this was missing before
    
    # print(np.allclose(phi_rn, kappa_full)) # returned all True so all good (did this in main.py)

    # Red noise residual = F @ a
    return F @ a


def white_noise_residual(pulsar, pulsar_noise_params):

    wn_params = pulsar_noise_params[pulsar.name]['white_noise']
    r_wn = np.zeros(len(pulsar.toas))
    sel = selections.Selection(selections.by_backend)

    # --- EFAC + EQUAD diagonal part ---
    # Build MeasurementNoise exactly as before — this works fine
    params_dict = {}
    for backend, bp in wn_params.items():
        params_dict[f'{pulsar.name}_{backend}_efac']          = bp['efac']
        params_dict[f'{pulsar.name}_{backend}_log10_t2equad'] = bp['log10_t2equad']
        params_dict[f'{pulsar.name}_{backend}_log10_ecorr']   = bp['log10_ecorr']

    efac_p  = parameter.Constant()
    equad_p = parameter.Constant()
    mn      = white_signals.MeasurementNoise(
                  efac=efac_p,
                  log10_t2equad=equad_p,
                  selection=sel
              )
    mn_signal = mn(pulsar)
    Nvec      = mn_signal.get_ndiag(params_dict)   # (EFAC*sigma)^2 + EQUAD^2
    r_wn     += np.sqrt(Nvec) * np.random.randn(len(pulsar.toas))

    # --- ECORR epoch-correlated part ---
    # create_quantization_matrix is what EcorrKernelNoise uses internally
    # it groups TOAs into epochs using the same logic enterprise uses
    for backend, bp in wn_params.items():
        mask  = pulsar.flags['f'] == backend
        if not np.any(mask):
            continue

        ecorr = 10**bp['log10_ecorr']  # seconds

        # This is the exact same call made inside EcorrKernelNoise.__init__
        # nmin=2 means an epoch needs at least 2 TOAs to get an ECORR block
        # (single TOAs are already handled by EFAC/EQUAD)
        U, _ = create_quantization_matrix(pulsar.toas[mask], nmin=2)
        # U_ij = 1 if TOA i belongs to epoch j, zero otherwise
        # U shape: (n_toas_in_backend, n_epochs)
        # each column is a binary vector — 1 for TOAs in that epoch, 0 otherwise

        n_epochs = U.shape[1]
        z        = np.random.randn(n_epochs)

        # Place the ECORR draw back into the full TOA array at the right indices
        r_wn[mask] += U @ (ecorr * z)

    return r_wn


def population_residuals(t, psr, population, Tspan,
                         pulsar_noise_params=None,
                         include_GW=True,
                         include_RN=False,  
                         include_WN=False): 
    """
    Scalar (loop-based) calculation of total timing residuals for one pulsar.
    Kept for debugging and for small populations where vectorisation overhead
    is not worth it.

    Parameters
    ----------
    t                   : TOA array (seconds)
    psr                 : enterprise Pulsar object
    population          : list of binary dicts
    Tspan               : observation span (seconds), needed for RN draw
    pulsar_noise_params : classified noise dict (required if include_RN or include_WN)
    include_GW          : include GW signal from population (default True)
    include_RN          : include red noise draw (default False)
                          NOTE: for OS-based SNR estimation this should be False.
                          The OS noise model marginalises over RN analytically via
                          F Phi F^T in the covariance. Including a RN realisation
                          in the residuals double-counts the red noise and produces
                          spurious cross-correlations. See Chamberlin et al. (2015).
    include_WN          : include white noise draw (default False)
                          Same reasoning as include_RN — WN is handled analytically
                          by the OS via the N_WN = diag[(EFAC*sigma)^2 + EQUAD^2]
                          + U ECORR^2 U^T term in the covariance.

    Returns
    -------
    total_r : np.ndarray, shape (N_toas,)
    """
    total_r = np.zeros_like(t, dtype=float)

    # --- GW signal: sum over all binaries in population ---
    # Each binary contributes an Earth-term-only residual r_k(t).
    # Earth-term approximation valid when pulsar distance >> GW wavelength,
    # which holds for most PTA sources. See Appendix of arxiv:1003.0677.
    if include_GW:
        for binary in population:
            total_r += r_k(t, psr, binary)

    # --- Red noise: single draw from GP prior ---
    # Draws Fourier coefficients a ~ N(0, phi) and returns F @ a.
    # phi is the power-law PSD: phi_k = A^2/(12pi^2) * (f_k/f_yr)^-gamma * f_yr^-2 / Tspan
    # This matches enterprise's internal get_phi() exactly (verified).
    if include_RN:
        if pulsar_noise_params is None:
            raise ValueError("pulsar_noise_params required when include_RN=True")
        rn = pulsar_noise_params[psr.name]['red_noise']
        total_r += draw_red_noise_residuals(
            psr, rn['log10_A'], rn['gamma'], Tspan
        )

    # --- White noise: EFAC + EQUAD + ECORR draw ---
    # EFAC: rescales TOA uncertainties per backend (accounts for calibration errors)
    # EQUAD: additive noise floor per backend (pulse jitter etc.)
    # ECORR: epoch-correlated noise — single shared offset per observing epoch
    #        per backend, implemented via the quantization matrix U.
    # See Lentati et al. (2014) for the full model.
    if include_WN:
        if pulsar_noise_params is None:
            raise ValueError("pulsar_noise_params required when include_WN=True")
        total_r += white_noise_residual(psr, pulsar_noise_params)

    return total_r


# =====================================================================
# VECTORISED CORE
# =====================================================================

def _antenna_response_vec(psr_ra, psr_dec, ra_arr, dec_arr, psi_arr):
    """Vectorised antenna response over N binaries.
    Returns Fp_arr, Fx_arr each of shape (N,)."""
    N             = len(ra_arr)
    src_polar     = np.pi/2 - dec_arr        # (N,)
    psr_polar     = np.pi/2 - psr_dec        # scalar

    omega_hat = np.array([                   # (3, N)
        -np.sin(src_polar) * np.cos(ra_arr),
        -np.sin(src_polar) * np.sin(ra_arr),
        -np.cos(src_polar),
    ])
    p_hat = np.array([                       # (3,)
        np.sin(psr_polar) * np.cos(psr_ra),
        np.sin(psr_polar) * np.sin(psr_ra),
        np.cos(psr_polar),
    ])
    m_hat = np.array([                       # (3, N)
        np.sin(ra_arr),
        -np.cos(ra_arr),
        np.zeros(N),
    ])
    n_hat = np.array([                       # (3, N)
        -np.cos(src_polar) * np.cos(ra_arr),
        -np.cos(src_polar) * np.sin(ra_arr),
         np.sin(src_polar),
    ])

    cos_psi = np.cos(psi_arr)               # (N,)
    sin_psi = np.sin(psi_arr)               # (N,)
    m_rot   = cos_psi * m_hat + sin_psi * n_hat    # (3, N)
    n_rot   = -sin_psi * m_hat + cos_psi * n_hat   # (3, N)

    denom   = 1 + p_hat @ omega_hat         # (N,)  dot of (3,) with (3,N)
    p_m     = p_hat @ m_rot                 # (N,)
    p_n     = p_hat @ n_rot                 # (N,)

    Fp = 0.5 * (p_m**2 - p_n**2) / denom
    Fx =       (p_m   * p_n)     / denom
    return Fp, Fx                            # each (N,)

def _get_psr_radec(psr):
    """
    Extract (ra, dec) in radians from either a libstempo or Enterprise pulsar object.
    Tries RAJ/DECJ first, falls back to ELONG/ELAT ecliptic coords, 
    then falls back to _raj/_decj (Enterprise).
    """
    # --- libstempo path ---
    if hasattr(psr, 'pars'):
        try:
            pars = psr.pars()
            if 'RAJ' in pars and 'DECJ' in pars:
                return psr['RAJ'].val, psr['DECJ'].val
            elif 'ELONG' in pars and 'ELAT' in pars:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                coord = SkyCoord(
                    lon=psr['ELONG'].val * u.rad,
                    lat=psr['ELAT'].val  * u.rad,
                    frame='geocentricmeanecliptic'
                )
                return coord.icrs.ra.rad, coord.icrs.dec.rad
        except Exception:
            pass  # fall through to _raj/_decj

    # --- Enterprise path (or libstempo fallback) ---
    if hasattr(psr, '_raj') and hasattr(psr, '_decj'):
        return psr._raj, psr._decj

    raise AttributeError(
        f"Cannot extract RA/Dec from pulsar object of type {type(psr)}. "
        f"Expected RAJ/DECJ or ELONG/ELAT params, or _raj/_decj attributes."
    )

def _gw_residuals_vec(t, psr, population):
    """
    Vectorised GW residual from a full population.
    Processes all binaries simultaneously.
    Returns r of shape (N_toas,).
    """
    N = len(population)

    f_arr    = np.array([b.f              for b in population])  # (N,)
    z_arr    = np.array([b.z              for b in population])
    ra_arr   = np.array([b.ra             for b in population])
    dec_arr  = np.array([b.dec            for b in population])
    psi_arr  = np.array([b.psi            for b in population])
    phi0_arr = np.array([b.phi0           for b in population])
    iota_arr = np.array([b.iota           for b in population])
    h0_arr   = np.array([b.h0             for b in population])  # optional pre-computed h0
    
    # pars = psr.pars()

    # if 'RAJ' not in pars or 'DECJ' not in pars:
    #     elong = psr['ELONG'].val  # radians
    #     elat  = psr['ELAT'].val   # radians

    #     coord = SkyCoord(lon=elong*u.rad, lat=elat*u.rad, frame='geocentricmeanecliptic')
    #     equatorial = coord.icrs

    #     psr_ra  = equatorial.ra.rad
    #     psr_dec = equatorial.dec.rad

    # else:
    psr_ra   = psr._raj
    psr_dec  = psr._decj

    # Antenna patterns (N,)
    Fp_arr, Fx_arr = _antenna_response_vec(
        psr_ra, psr_dec, ra_arr, dec_arr, psi_arr
    )

    # Phase matrix (N_toas, N)
    t_rel = (t - t[0])[:, np.newaxis]          # (N_toas, 1)
    phase = 2*np.pi * f_arr * t_rel + phi0_arr # (N_toas, N)

    # GW polarisations weighted by antenna patterns (N_toas, N)
    weighted = (
        Fp_arr * h0_arr * (1 + np.cos(iota_arr)**2) * np.sin(phase)
      + Fx_arr * h0_arr * (-2 * np.cos(iota_arr))   * np.cos(phase)
    )

    # Residual = integral of h / (2 pi f), sum over binaries
    r = np.sum(weighted / (2*np.pi*f_arr), axis=1)   # (N_toas,)
    return r


def _gw_residuals_chunked(t, psr, population, chunk_size):
    """Chunked wrapper around _gw_residuals_vec to cap memory use."""
    N       = len(population)
    total_r = np.zeros(len(t), dtype=float)
    for start in range(0, N, chunk_size):
        total_r += _gw_residuals_vec(t, psr, population[start:start+chunk_size])
    return total_r


def _auto_chunk_size(N_toas, max_memory_mb=200):
    """Each binary needs ~5 float64 arrays of length N_toas."""
    return max(1, int(max_memory_mb * 1024**2 / (N_toas * 8 * 5)))


def population_residuals_vectorised(
        t, psr, population, Tspan,
        pulsar_noise_params=None,
        include_GW=True,
        include_RN=False,
        include_WN=False,
        chunk_size=None,
        max_memory_mb=200):
    """
    Compute total timing residuals for one pulsar — vectorised over population.

    Automatically falls back to scalar loop for populations of size < 10
    (vectorisation overhead not worth it for tiny populations).

    Parameters
    ----------
    t                    : TOA array (seconds)
    psr                  : enterprise Pulsar object
    population           : list of binary dicts
    Tspan                : observation span (seconds)
    pulsar_noise_params  : classified noise dict (required if include_RN or include_WN)
    include_GW           : include GW signal (default True)
    include_RN           : include red noise draw (default False)
    include_WN           : include white noise draw (default False)
    chunk_size           : binaries per GW chunk (None = auto from max_memory_mb)
    max_memory_mb        : memory cap for auto chunk sizing (default 200 MB)

    Notes on include_RN / include_WN defaults
    ------------------------------------------
    For OS-based SNR estimation these should both be False (the default).
    The OS covariance C_a = N_WN + F Phi F^T already contains both noise
    contributions analytically — adding noise realisations to the residuals
    on top of this double-counts the noise and produces spurious SNR.
    Reference: Chamberlin et al. (2015).

    For realistic noise simulations (SNR distributions, false alarm studies),
    set both to True and average OS results over many realisations.
    """
    total_r = np.zeros(len(t), dtype=float)

    # --- GW signal ---
    if include_GW and len(population) > 0:
        if len(population) < 10:
            # Scalar fallback for tiny populations — avoids vectorisation overhead
            # and array allocation for populations that are too small to benefit
            for binary in population:
                total_r += r_k(t, psr, binary)
        else:
            cs = chunk_size or _auto_chunk_size(len(t), max_memory_mb)
            if len(population) <= cs:
                # Entire population fits in one vectorised call
                total_r += _gw_residuals_vec(t, psr, population)
            else:
                # Population too large for one call — process in memory-safe chunks
                # Each chunk is independently vectorised, results summed
                total_r += _gw_residuals_chunked(t, psr, population, cs)

    # --- Red noise draw ---
    # Only used for realistic noise simulations, not for OS SNR estimation.
    # Draws a ~ N(0, phi) and returns F @ a where phi is the power-law PSD.
    if include_RN:
        if pulsar_noise_params is None:
            raise ValueError("pulsar_noise_params required when include_RN=True")
        rn      = pulsar_noise_params[psr.name]['red_noise']
        total_r += draw_red_noise_residuals(
            psr, rn['log10_A'], rn['gamma'], Tspan
        )

    # --- White noise draw ---
    # Only used for realistic noise simulations, not for OS SNR estimation.
    # Includes EFAC+EQUAD (independent per TOA) and ECORR (epoch-correlated).
    if include_WN:
        if pulsar_noise_params is None:
            raise ValueError("pulsar_noise_params required when include_WN=True")
        total_r += white_noise_residual(psr, pulsar_noise_params)

    return total_r


# Method for CGW calculation
def gw_residuals_matrix(t, psr, population):
    """
    Returns GW residuals for ALL binaries simultaneously.
    
    Shape: (B, n_toa) — one row per binary, NOT summed.
    
    This is _gw_residuals_vec but returning before the np.sum,
    so the vectorised inner product loop can use it directly.
    
    Parameters
    ----------
    t          : (n_toa,) TOA array in seconds
    psr        : enterprise Pulsar object
    population : list of B binary objects
    
    Returns
    -------
    R : (B, n_toa) ndarray in seconds
    """
    B = len(population)

    f_arr    = np.array([b.f       for b in population])   # (B,)
    ra_arr   = np.array([b.ra      for b in population])
    dec_arr  = np.array([b.dec     for b in population])
    psi_arr  = np.array([b.psi     for b in population])
    phi0_arr = np.array([b.phi0    for b in population])
    iota_arr = np.array([b.iota    for b in population])
    h0_arr   = np.array([b.h0      for b in population])


    # Pulsar sky position
    psr_ra  = psr._raj
    psr_dec = psr._decj

    # Antenna patterns (B,)
    Fp_arr, Fx_arr = _antenna_response_vec(
        psr_ra, psr_dec, ra_arr, dec_arr, psi_arr
    )

    # Phase: (n_toa, B) then transpose to (B, n_toa)
    t_rel = (t - t[0])[:, None]                        # (n_toa, 1)
    phase = 2*np.pi * f_arr[None, :] * t_rel \
            + phi0_arr[None, :]                         # (n_toa, B)

    # Weighted polarisations (n_toa, B)
    weighted = (
        Fp_arr * h0_arr * (1 + np.cos(iota_arr)**2) * np.sin(phase)
      + Fx_arr * h0_arr * (-2 * np.cos(iota_arr))   * np.cos(phase)
    )                                                   # (n_toa, B)

    # Divide by 2pi*f per binary, transpose to (B, n_toa)
    R = (weighted / (2*np.pi*f_arr[None, :])).T         # (B, n_toa)

    return R

def make_ideal_nofit(psr):
    """Zero residuals without refitting - avoids singular matrix issues with GLS timing models."""
    res = psr.residuals(updatebats=True, formresiduals=True)  # seconds
    psr.stoas[:] -= res / 86400.0

def get_base_name(psrname):
    """Strip telescope suffix (ao, gbt, vla) from pulsar name."""
    for suffix in ['ao', 'gbt', 'vla', 'fast']:
        if psrname.endswith(suffix):
            return psrname[:-len(suffix)]
    return psrname

import matplotlib.pyplot as plt

            
def simulate_psr_old(psr, noise_dict, add_WN=True, add_RN=True, add_GWB=False, plot=False, seed=None):
    psrname = psr.name
    basename = get_base_name(psrname)

    # Step 1: zero residuals WITH jumps active (absorbs jump offsets into stoas)
    print(f"  [{psrname}] zeroing residuals...", flush=True)
    make_ideal_nofit(psr)


    # Step 2: add red noise, white noise as before
    psr_keys = {k: v for k, v in noise_dict.items() if k.startswith(basename)}
    
    if add_RN:
        rn_log10_A_key = f"{basename}_red_noise_log10_A"
        rn_gamma_key   = f"{basename}_red_noise_gamma"
        if rn_log10_A_key in psr_keys and rn_gamma_key in psr_keys:
            log10_A = psr_keys[rn_log10_A_key]
            gamma   = psr_keys[rn_gamma_key]
            print(f"  [{psrname}] adding red noise...", flush=True)
            LT.add_rednoise(psr, 10**log10_A, gamma, components=30)

    if add_WN:
        systems = set()
        for k in psr_keys:
            middle = k.replace(f"{basename}_", "")
            for suffix in ['_efac', '_log10_ecorr', '_log10_t2equad']:
                if middle.endswith(suffix):
                    systems.add(middle.replace(suffix, ""))

        for sys in systems:
            efac  = psr_keys.get(f"{basename}_{sys}_efac", 1.0)
            equad = 10**psr_keys.get(f"{basename}_{sys}_log10_t2equad", -100)
            ecorr = 10**psr_keys.get(f"{basename}_{sys}_log10_ecorr", -100)

            try:
                flag_vals = psr.flagvals('f')
                mask = np.array([sys in fv for fv in flag_vals])
            except Exception:
                mask = np.ones(psr.nobs, dtype=bool)

            if mask.sum() == 0:
                continue
            print(f"  [{psrname}] white noise for {sys} ({mask.sum()} TOAs)...", flush=True)
            LT.add_efac(psr, efac, flagid='f', flags=sys, seed=seed)
            LT.add_equad(psr, equad, flagid='f', flags=sys, seed=seed)
            LT.add_jitter(psr, ecorr, flagid='f', flags=sys, seed=seed)
    if plot:
        print(f"[{psrname}] plotting final residuals with all noise...", flush=True)
        # plot_residuals_raw(psr, title="Final Residuals with All Noise")
        LP.plotres(psr)
        plt.show()

    return psr

def inject_population_nufft(psrs, population,
                             verbose=False, eps=1e-6,
                             cache_precomputed_amplitudes=True,
                             track_contributors=False,
                             top_k_per_pulsar=20,
                             top_k_global=50,
                             contributor_chunk_size=500_000,
                             contributor_summary=None):
    """
    Inject SMBHB population via NUFFT type-3 (no frequency quantisation error).

    Uses finufft.nufft1d3:
        f(x_j) = sum_k c_k * exp(i * s_k * x_j)

    where:
        x_j = t_j - t[0]          (TOA times, non-uniform, in seconds)
        s_k = 2π f_k              (source frequencies, non-uniform, in rad/s)
        c_k = (C_k - i S_k) / 2  (complex amplitude, see derivation below)

    Derivation
    ----------
    r(t) = sum_k A_k sin(2π f_k t_rel + φ0_k) + B_k cos(2π f_k t_rel + φ0_k)

    Expanding with phi0:
        S_k = A_k cos(φ0) - B_k sin(φ0)   [coeff of sin(2π f_k t_rel)]
        C_k = A_k sin(φ0) + B_k cos(φ0)   [coeff of cos(2π f_k t_rel)]

    Writing in complex form using e^{isx} = cos(sx) + i sin(sx):
        r = Re[ sum_k (C_k - i S_k) * e^{i 2π f_k t_rel} ]

    For real output we also need the conjugate (negative frequency) term:
        r = Re[ sum_k (C_k - i S_k) * e^{+i 2π f_k t_rel}
                    + (C_k + i S_k) * e^{-i 2π f_k t_rel} ] / 2

    Using nufft1d3 with only positive frequencies and dividing by 2:
        c_k = (C_k - i S_k) / 2
        r = 2 * Re[ nufft1d3(x, s, c) ]

        No grid, no quantisation — exact to NUFFT tolerance (eps).

                Optional amplitude precompute controls:
                    - cache_precomputed_amplitudes: if False, amp_A/B generated during
                        this call are removed after each pulsar to reduce peak memory.
                    - track_contributors: track per-pulsar and global top contributors
                        by amplitude proxy A^2 + B^2.
                    - top_k_per_pulsar: number of binaries tracked per pulsar.
                    - top_k_global: number of globally ranked binaries returned.
                    - contributor_chunk_size: chunk size for top-k tracking, controls
                        memory use while scanning A/B arrays.
                    - contributor_summary: optional dict to populate with tracking output.
    """
    f_arr    = population.f
    phi0_arr = population.phi0

    # Source frequencies in rad/s — these are the NUFFT "s" points
    # No gridding, no rounding — exact frequencies
    s_arr = 2 * np.pi * f_arr   # (N,) rad/s

    # phi0 rotation: same for all pulsars (source property)
    cos_phi0 = np.cos(phi0_arr)
    sin_phi0 = np.sin(phi0_arr)

    per_pulsar_top = {}
    contributor_candidates = set()

    for psr in psrs:
        psr_name = psr.name
        computed_here = False
        if psr_name not in population.amp_A:
            precompute_amplitudes(population, psr)
            computed_here = True

        A_arr = population.amp_A[psr_name]
        B_arr = population.amp_B[psr_name]

        if track_contributors:
            idx_top, score_top = _topk_indices_from_amplitudes(
                A_arr,
                B_arr,
                k=top_k_per_pulsar,
                chunk_size=contributor_chunk_size,
            )
            contributor_candidates.update(idx_top.tolist())
            per_pulsar_top[psr_name] = {
                'indices': idx_top.tolist(),
                'amp2': score_top.tolist(),
            }

        # phi0 rotation
        S = A_arr * cos_phi0 - B_arr * sin_phi0   # sin(2π f t_rel) coeff
        C = A_arr * sin_phi0 + B_arr * cos_phi0   # cos(2π f t_rel) coeff

        # Complex amplitudes: c_k = (C_k - i S_k) / 2
        c = (C - 1j * S) / 2   # (N,)

        # TOA times relative to first TOA — the NUFFT "x" points
        t_sec = np.asarray(psr.stoas, dtype=np.float64) * 86400.0  # days -> seconds
        x     = t_sec - t_sec[0]   # relative times in seconds  

        # nufft1d3: f(x_j) = sum_k c_k * exp(i * s_k * x_j)
        # isign=+1 matches our e^{+i 2π f t} convention
        x       = np.ascontiguousarray(x,     dtype=np.float64)
        s_nufft = np.ascontiguousarray(s_arr, dtype=np.float64)
        c_nufft = np.ascontiguousarray(c,     dtype=np.complex128)

        f_out = finufft.nufft1d3(s_nufft, c_nufft, x, isign=+1, eps=eps)

        # Multiply by 2: we only passed positive frequencies,
        # negative frequencies contribute equal real part
        time_change = 2 * np.real(f_out)

        if verbose:
            print(f"  {psr_name}: RMS = {time_change.std()*1e9:.3f} ns")

        psr.stoas[:] += time_change / 86400.0  # seconds -> days


        if computed_here and not cache_precomputed_amplitudes:
            population.amp_A.pop(psr_name, None)
            population.amp_B.pop(psr_name, None)

    if track_contributors:
        global_top = {'indices': [], 'score': []}
        if contributor_candidates:
            candidate_idx = np.asarray(sorted(contributor_candidates), dtype=np.int64)
            global_score = np.zeros(candidate_idx.size, dtype=np.float64)

            for psr in psrs:
                psr_name = psr.name
                A_c = population.amp_A[psr_name][candidate_idx]
                B_c = population.amp_B[psr_name][candidate_idx]
                global_score += A_c * A_c + B_c * B_c

            k_glob = min(max(int(top_k_global), 0), candidate_idx.size)
            if k_glob > 0:
                keep = np.argpartition(global_score, global_score.size - k_glob)[-k_glob:]
                order = np.argsort(global_score[keep])[::-1]
                keep_ord = keep[order]
                global_top = {
                    'indices': candidate_idx[keep_ord].tolist(),
                    'score': global_score[keep_ord].tolist(),
                }

        summary = {
            'score_definition': 'A^2 + B^2 per pulsar; global score is sum over pulsars',
            'n_binaries_total': int(len(population.f)),
            'n_candidates_global': int(len(contributor_candidates)),
            'per_pulsar_top_k': int(top_k_per_pulsar),
            'global_top_k': int(top_k_global),
            'per_pulsar': per_pulsar_top,
            'global': global_top,
        }

        if contributor_summary is not None:
            contributor_summary.clear()
            contributor_summary.update(summary)
        else:
            try:
                population.contributor_summary = summary
            except Exception:
                pass

    return psrs


def change_in_TOAs_days_population_nufft(psrs, population,
                             verbose=False, eps=1e-6,
                             cache_precomputed_amplitudes=False):
    """
    Calculate injection of SMBHB population via NUFFT type-3 (no frequency quantisation error).

    Uses finufft.nufft1d3:
        f(x_j) = sum_k c_k * exp(i * s_k * x_j)

    where:
        x_j = t_j - t[0]          (TOA times, non-uniform, in seconds)
        s_k = 2π f_k              (source frequencies, non-uniform, in rad/s)
        c_k = (C_k - i S_k) / 2  (complex amplitude, see derivation below)

    Derivation
    ----------
    r(t) = sum_k A_k sin(2π f_k t_rel + φ0_k) + B_k cos(2π f_k t_rel + φ0_k)

    Expanding with phi0:
        S_k = A_k cos(φ0) - B_k sin(φ0)   [coeff of sin(2π f_k t_rel)]
        C_k = A_k sin(φ0) + B_k cos(φ0)   [coeff of cos(2π f_k t_rel)]

    Writing in complex form using e^{isx} = cos(sx) + i sin(sx):
        r = Re[ sum_k (C_k - i S_k) * e^{i 2π f_k t_rel} ]

    For real output we also need the conjugate (negative frequency) term:
        r = Re[ sum_k (C_k - i S_k) * e^{+i 2π f_k t_rel}
                    + (C_k + i S_k) * e^{-i 2π f_k t_rel} ] / 2

    Using nufft1d3 with only positive frequencies and dividing by 2:
        c_k = (C_k - i S_k) / 2
        r = 2 * Re[ nufft1d3(x, s, c) ]

        No grid, no quantisation — exact to NUFFT tolerance (eps).

                Optional amplitude precompute controls:
                    - cache_precomputed_amplitudes: if False, amp_A/B generated during
                        this call are removed after each pulsar to reduce peak memory.
    """
    f_arr    = population.f
    phi0_arr = population.phi0

    # Source frequencies in rad/s — these are the NUFFT "s" points
    # No gridding, no rounding — exact frequencies
    s_arr = 2 * np.pi * f_arr   # (N,) rad/s

    # phi0 rotation: same for all pulsars (source property)
    cos_phi0 = np.cos(phi0_arr)
    sin_phi0 = np.sin(phi0_arr)

    per_pulsar_top = {}
    contributor_candidates = set()

    pulsar_time_changes_arr = []

    for psr in psrs:
        print(f"Processing {psr.name}...", flush=True)
        psr_name = psr.name
        computed_here = False
        if psr_name not in population.amp_A:
            precompute_amplitudes(population, psr)
            computed_here = True

        A_arr = population.amp_A[psr_name]
        B_arr = population.amp_B[psr_name]

        # phi0 rotation
        S = A_arr * cos_phi0 - B_arr * sin_phi0   # sin(2π f t_rel) coeff
        C = A_arr * sin_phi0 + B_arr * cos_phi0   # cos(2π f t_rel) coeff

        # Complex amplitudes: c_k = (C_k - i S_k) / 2
        c = (C - 1j * S) / 2   # (N,)

        # TOA times relative to first TOA — the NUFFT "x" points
        t_sec = np.asarray(psr.stoas, dtype=np.float64) * 86400.0  # days -> seconds
        x     = t_sec - t_sec[0]   # relative times in seconds  

        # nufft1d3: f(x_j) = sum_k c_k * exp(i * s_k * x_j)
        # isign=+1 matches our e^{+i 2π f t} convention
        x       = np.ascontiguousarray(x,     dtype=np.float64)
        s_nufft = np.ascontiguousarray(s_arr, dtype=np.float64)
        c_nufft = np.ascontiguousarray(c,     dtype=np.complex128)

        f_out = finufft.nufft1d3(s_nufft, c_nufft, x, isign=+1, eps=eps)

        # Multiply by 2: we only passed positive frequencies,
        # negative frequencies contribute equal real part
        time_change = 2 * np.real(f_out)

        if verbose:
            print(f"  {psr_name}: RMS = {time_change.std()*1e9:.3f} ns")

        pulsar_time_changes_arr.append([psr_name, time_change/ 86400.0]) # store to add to stoas later


        if computed_here and not cache_precomputed_amplitudes:
            population.amp_A.pop(psr_name, None)
            population.amp_B.pop(psr_name, None)

    return pulsar_time_changes_arr

### Modifying pulsarss to have higher cadence and lower errors to make synthetic pulsar data

# ---------------------------------------------------------------------------
# Step 1: augment a libstempo pulsar's TOAs before noise simulation
# ---------------------------------------------------------------------------

def augment_psr_cadence(psr, cadence_factor=2, toaerr_factor=1.0):
    """
    Insert interleaved TOAs into a libstempo pulsar object to simulate
    increased cadence, and/or scale toaerrs to simulate increased precision.

    This must be called BEFORE simulate_psr / any noise injection,
    because the noise functions act on psr.stoas / psr.toaerrs in place.

    Parameters
    ----------
    psr : libstempo.tempopulsar
    cadence_factor : int
        Number of times denser to make the TOA grid (2 = twice as many obs).
        New TOAs are linearly interpolated between existing ones.
        The flag ('f') of each new TOA is copied from its left neighbour,
        so EFAC/EQUAD/ECORR assignments remain backend-consistent.
    toaerr_factor : float
        Multiply all TOA errors by this factor.
        0.5 = twice as precise (e.g. 2× integration time or better backend).

    Returns
    -------
    psr  (modified in place, also returned for chaining)
    """
    if cadence_factor > 1:
        toas   = psr.stoas.copy()          # MJD
        errs   = psr.toaerrs.copy()        # microseconds
        flags  = psr.flagvals('f').copy()  # backend strings

        extra_toas  = []
        extra_errs  = []
        extra_flags = []

        for i in range(len(toas) - 1):
            dt = toas[i+1] - toas[i]
            for k in range(1, cadence_factor):
                frac = k / cadence_factor
                extra_toas.append(toas[i] + frac * dt)
                # inherit error from nearest neighbour
                extra_errs.append(errs[i] if frac < 0.5 else errs[i+1])
                extra_flags.append(flags[i] if frac < 0.5 else flags[i+1])

        if extra_toas:
            all_toas  = np.concatenate([toas,  extra_toas])
            all_errs  = np.concatenate([errs,  extra_errs])
            all_flags = np.concatenate([flags, extra_flags])
            idx       = np.argsort(all_toas)
            all_toas  = all_toas[idx]
            all_errs  = all_errs[idx]
            all_flags = all_flags[idx]

            # Write back into the libstempo object
            # stoas is a writable array; we need to resize it by rebuilding
            # the underlying .tim file and reloading — OR use the array API
            # directly if your libstempo version supports it.
            # The cleanest approach is to write a new .tim and reload:
            _rewrite_tim_and_reload(psr, all_toas, all_errs, all_flags)

    if toaerr_factor != 1.0:
        psr.toaerrs[:] *= toaerr_factor

    return psr


def _rewrite_tim_and_reload(psr, new_stoas_mjd, new_errs_us, new_flags, tmpdir='/tmp'):
    """
    Write a new .tim file with augmented TOAs and reload into the same
    libstempo object in place.

    libstempo doesn't expose a direct 'resize stoas' API, so the cleanest
    path is to write a minimal tempo2-format .tim and call psr.readtim().

    new_stoas_mjd : array of MJD (float64, barycentric)
    new_errs_us   : array of TOA errors in microseconds
    new_flags     : array of backend flag strings (for -f flag)
    """
    import os, tempfile

    tim_path = os.path.join(tmpdir, f'{psr.name}_augmented.tim')

    with open(tim_path, 'w') as f:
        f.write('FORMAT 1\n')
        for toa, err, flag in zip(new_stoas_mjd, new_errs_us, new_flags):
            # FORMAT 1: name freq toa err telescope [-flags]
            # Use the original frequency from the first TOA as a placeholder;
            # frequency doesn't affect SNR calculations here
            freq = 1400.0   # MHz — placeholder, fine for noise simulations
            f.write(f'{psr.name}  {freq:.4f}  {toa:.15f}  {err:.4f}  @  -f {flag}\n')

    maxobs = max(60000, len(new_stoas_mjd) + 1000)
    new_psr = T.tempopulsar(
        parfile=psr.parfile,
        timfile=tim_path,
        maxobs=maxobs,
        dofit=False,
    )
    # re-zero residuals after reload since the timing model still applies
    make_ideal_nofit(new_psr)
    return new_psr



def simulate_psr_modified(
    psr,
    noise_dict,
    add_WN        = True,
    add_RN        = True,
    cadence_factor  = 1,
    toaerr_factor   = 1.0,
    plot          = False,
):
    """
    Like simulate_psr, but optionally augments the TOA grid and/or
    scales timing errors before injecting noise.

    The cadence/precision modifications happen BEFORE noise injection
    so that the noise is self-consistent with the new TOA set.
    """
    psrname  = psr.name
    basename = get_base_name(psrname)

    print(f"  [{psrname}] zeroing residuals...", flush=True)
    make_ideal_nofit(psr)

    # --- apply observing strategy modifications ---
    if cadence_factor > 1 or toaerr_factor != 1.0:
        print(f"  [{psrname}] augmenting: cadence×{cadence_factor}, "
              f"err×{toaerr_factor:.2f} ({psr.nobs} → ", end='', flush=True)
        psr = augment_psr_cadence(psr, cadence_factor=cadence_factor,
                                  toaerr_factor=toaerr_factor)
        print(f"{psr.nobs} TOAs)", flush=True)

    if plot:
        LP.plotres(psr); plt.show()

    # --- red noise (identical to your simulate_psr) ---
    if add_RN:
        psr_keys = {k: v for k, v in noise_dict.items() if k.startswith(basename)}
        rn_A_key   = f"{basename}_red_noise_log10_A"
        rn_gam_key = f"{basename}_red_noise_gamma"
        if rn_A_key in psr_keys and rn_gam_key in psr_keys:
            log10_A = psr_keys[rn_A_key]
            gamma   = psr_keys[rn_gam_key]
            print(f"  [{psrname}] adding red noise...", flush=True)
            LT.add_rednoise(psr, 10**log10_A, gamma, components=30)

    # --- white noise (identical to your simulate_psr) ---
    if add_WN:
        psr_keys = {k: v for k, v in noise_dict.items() if k.startswith(basename)}
        systems  = set()
        for k in psr_keys:
            middle = k.replace(f"{basename}_", "")
            for suffix in ['_efac', '_log10_ecorr', '_log10_t2equad']:
                if middle.endswith(suffix):
                    systems.add(middle.replace(suffix, ""))

        for sys in systems:
            efac  = psr_keys.get(f"{basename}_{sys}_efac", 1.0)
            equad = 10**psr_keys.get(f"{basename}_{sys}_log10_t2equad", -100)
            ecorr = 10**psr_keys.get(f"{basename}_{sys}_log10_ecorr", -100)

            try:
                flag_vals = psr.flagvals('f')
                mask = np.array([sys in fv for fv in flag_vals])
            except:
                mask = np.ones(psr.nobs, dtype=bool)

            if mask.sum() == 0:
                continue
            print(f"  [{psrname}] white noise for {sys} ({mask.sum()} TOAs)...", flush=True)
            LT.add_efac(psr, efac, flagid='f', flags=sys)
            LT.add_equad(psr, equad, flagid='f', flags=sys)
            LT.add_jitter(psr, ecorr, flagid='f', flags=sys)

    if plot:
        LP.plotres(psr); plt.show()

    return psr



import math
import numpy as np

day  = 86400.0
year = 3.15581498e7

def make_ideal_nofit(psr):
    res = psr.residuals(updatebats=True, formresiduals=True)
    psr.stoas[:] -= res / day

def get_base_name(psrname):
    for suffix in ['ao', 'gbt', 'vla', 'fast']:
        if psrname.endswith(suffix):
            return psrname[:-len(suffix)]
    return psrname

def _quantize_fast(times, flags=None, dt=1.0):
    """Exact copy of libstempo's quantize_fast."""
    isort = np.argsort(times)
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    avetoas = np.array([np.mean(times[ind]) for ind in bucket_ind], 'd')
    if flags is not None:
        aveflags = np.array([flags[ind[0]] for ind in bucket_ind])
    U = np.zeros((len(times), len(bucket_ind)), 'd')
    for i, l in enumerate(bucket_ind):
        U[l, i] = 1
    if flags is not None:
        return avetoas, aveflags, U
    else:
        return avetoas, U

def _add_efac(psr, efac, flagid, flags, seed=None):
    """Exact reimplementation of LT.add_efac."""
    if seed is not None:
        np.random.seed(seed)
    efacvec = np.ones(psr.nobs)
    ind = np.array([fv == flags for fv in psr.flagvals(flagid)]) 
    efacvec[ind] = efac
    psr.stoas[:] += efacvec * psr.toaerrs * (1e-6 / day) * np.random.randn(psr.nobs)

def _add_equad(psr, equad, flagid, flags, seed=None):
    """Exact reimplementation of LT.add_equad."""
    if seed is not None:
        np.random.seed(seed)
    equadvec = np.zeros(psr.nobs)
    ind = np.array([fv == flags for fv in psr.flagvals(flagid)])
    equadvec[ind] = equad
    psr.stoas[:] += (equadvec / day) * np.random.randn(psr.nobs)

def _add_jitter(psr, ecorr, flagid, flags, coarsegrain=0.1, seed=None):
    """Exact reimplementation of LT.add_jitter."""
    if seed is not None:
        np.random.seed(seed)
    t = psr.toas()
    f = np.array(psr.flagvals(flagid))
    _, aveflags, U = _quantize_fast(t, flags=f, dt=coarsegrain)
    ecorrvec = np.zeros(U.shape[1])
    ind = aveflags == flags
    ecorrvec[ind] = ecorr
    psr.stoas[:] += (1 / day) * np.dot(U * ecorrvec, np.random.randn(U.shape[1]))

def _add_rednoise(psr, A, gamma, components=10, tspan=None, seed=None):
    """Exact reimplementation of LT.add_rednoise."""
    if seed is not None:
        np.random.seed(seed)
    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (day / year) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (day / year) * tspan
    size = 2 * components
    F = np.zeros((psr.nobs, size), 'd')
    f = np.zeros(size, 'd')
    for i in range(components):
        F[:, 2*i]   = np.cos(2 * math.pi * (i+1) * x)
        F[:, 2*i+1] = np.sin(2 * math.pi * (i+1) * x)
        f[2*i] = f[2*i+1] = (i+1) / T
    norm  = A**2 * year**2 / (12 * math.pi**2 * T)
    prior = norm * f**(-gamma)
    y = np.sqrt(prior) * np.random.randn(size)
    psr.stoas[:] += (1.0 / day) * np.dot(F, y)


def simulate_psr(psr, noise_dict, add_WN=True, add_RN=True,
                 add_GWB=False, plot=False, seed=None):
    psrname  = psr.name
    basename = get_base_name(psrname)
    psr_keys = {k: v for k, v in noise_dict.items() if k.startswith(basename)}

    print(f"  [{psrname}] zeroing residuals...", flush=True)
    make_ideal_nofit(psr)

    if add_RN:
        rn_log10_A_key = f"{basename}_red_noise_log10_A"
        rn_gamma_key   = f"{basename}_red_noise_gamma"
        if rn_log10_A_key in psr_keys and rn_gamma_key in psr_keys:
            log10_A = psr_keys[rn_log10_A_key]
            gamma   = psr_keys[rn_gamma_key]
            print(f"  [{psrname}] adding red noise...", flush=True)
            _add_rednoise(psr, 10**log10_A, gamma, components=30, seed=seed)

    if add_WN:
        try:
            flag_vals = np.array(psr.flagvals('f'))
        except Exception:
            flag_vals = np.array([''] * psr.nobs)

        systems = set()
        for k in psr_keys:
            middle = k.replace(f"{basename}_", "")
            for suffix in ['_efac', '_log10_ecorr', '_log10_t2equad']:
                if middle.endswith(suffix):
                    systems.add(middle.replace(suffix, ""))

        for sys in systems:
            efac  = psr_keys.get(f"{basename}_{sys}_efac",          1.0)
            equad = 10**psr_keys.get(f"{basename}_{sys}_log10_t2equad", -100.0)
            ecorr = 10**psr_keys.get(f"{basename}_{sys}_log10_ecorr",   -100.0)

            mask = np.array([sys in fv for fv in flag_vals])
            if mask.sum() == 0:
                continue

            print(f"  [{psrname}] white noise for {sys} ({mask.sum()} TOAs)...", flush=True)
            _add_efac(  psr, efac,  flagid='f', flags=sys, seed=seed)
            _add_equad( psr, equad, flagid='f', flags=sys, seed=seed)
            _add_jitter(psr, ecorr, flagid='f', flags=sys, seed=seed)

    return psr

import numpy as np
import libstempo as LT

def test_simulator_consistency(psr_path, par_path, noise_dict, seed=42, tol_frac=0.05):
    """
    Compare old (libstempo) vs new (reimplemented) simulate_psr.

    Checks that per-component and total residual RMS agree within tol_frac
    (default 5%) for a single pulsar.

    Parameters
    ----------
    psr_path : str   path to .tim file
    par_path : str   path to .par file
    noise_dict : dict noise parameters
    seed : int       RNG seed (same for both)
    tol_frac : float fractional tolerance on RMS comparison
    """
    def load_fresh():
        return LT.tempopulsar(parfile=par_path, timfile=psr_path, maxobs=30000)

    results = {}
    for label, use_lt in [('libstempo', True), ('reimplemented', False)]:
        psr = load_fresh()
        if use_lt:
            simulate_psr_old(psr, noise_dict, add_WN=True, add_RN=True, seed=seed)
        else:
            simulate_psr(psr, noise_dict, add_WN=True, add_RN=True, seed=seed)
        res_s = psr.residuals() * 86400.0          # days → seconds
        results[label] = res_s
        print(f"[{label}] RMS = {np.std(res_s)*1e6:.3f} µs")

    rms_lt  = np.std(results['libstempo'])
    rms_new = np.std(results['reimplemented'])
    frac    = abs(rms_lt - rms_new) / rms_lt

    print(f"\nRMS fractional difference: {frac*100:.2f}%  (tolerance: {tol_frac*100:.0f}%)")

    # ── per-component checks ─────────────────────────────────────────────────
    def _rms_component(add_WN, add_RN, label_suffix):
        psr_lt  = load_fresh()
        psr_new = load_fresh()
        simulate_psr_old(psr_lt,  noise_dict, add_WN=add_WN, add_RN=add_RN, seed=seed)
        simulate_psr(psr_new, noise_dict, add_WN=add_WN, add_RN=add_RN, seed=seed)
        rms_lt  = np.std(psr_lt.residuals()  * 86400.0)
        rms_new = np.std(psr_new.residuals() * 86400.0)
        frac    = abs(rms_lt - rms_new) / (rms_lt + 1e-30)
        print(f"  {label_suffix:20s}  LT={rms_lt*1e6:.3f} µs  "
              f"new={rms_new*1e6:.3f} µs  diff={frac*100:.2f}%")
        return frac

    print("\nPer-component breakdown:")
    f_rn = _rms_component(add_WN=False, add_RN=True,  label_suffix="red noise only")
    f_wn = _rms_component(add_WN=True,  add_RN=False, label_suffix="white noise only")

    assert f_rn < tol_frac, f"Red noise RMS differs by {f_rn*100:.1f}% > {tol_frac*100:.0f}%"
    assert f_wn < tol_frac, f"White noise RMS differs by {f_wn*100:.1f}% > {tol_frac*100:.0f}%"
    assert frac  < tol_frac, f"Total RMS differs by {frac*100:.1f}% > {tol_frac*100:.0f}%"

    print("\n✓ All components within tolerance.")
    return results
#### METHOD FOR MEERKAT ETC.
import math
import numpy as np
 
day  = 86400.0
year = 3.15581498e7
 
# MPTA reference frequency
NU_REF_MHZ = 1400.0  # Reference frequency in MHz
 
 
def make_ideal_nofit(psr):
    """Zero out residuals by adjusting TOAs."""
    res = psr.residuals(updatebats=True, formresiduals=True)
    psr.stoas[:] -= res / day
 
 
def get_base_name(psrname):
    """Extract base pulsar name by removing system suffix."""
    for suffix in ['ao', 'gbt', 'vla', 'fast']:
        if psrname.endswith(suffix):
            return psrname[:-len(suffix)]
    return psrname
 
 
def _quantize_fast(times, flags=None, dt=1.0):
    """Exact copy of libstempo's quantize_fast for binning TOAs."""
    isort = np.argsort(times)
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    avetoas = np.array([np.mean(times[ind]) for ind in bucket_ind], 'd')
    if flags is not None:
        aveflags = np.array([flags[ind[0]] for ind in bucket_ind])
    U = np.zeros((len(times), len(bucket_ind)), 'd')
    for i, l in enumerate(bucket_ind):
        U[l, i] = 1
    if flags is not None:
        return avetoas, aveflags, U
    else:
        return avetoas, U
 
 
def _add_efac(psr, efac, flagid, flags, seed=None):
    """Add EFAC (quadrature jitter scaling) to TOAs."""
    if seed is not None:
        np.random.seed(seed)
    flag_vals = np.array(psr.flagvals(flagid))
    mask = np.array([fv == flags for fv in flag_vals])
    noise = efac * psr.toaerrs * (1e-6 / day) * np.random.randn(psr.nobs)
    psr.stoas[mask] += noise[mask]
 
 
def _add_equad(psr, equad, flagid, flags, seed=None):
    """Add EQUAD (timing system jitter) to TOAs."""
    if seed is not None:
        np.random.seed(seed)
    flag_vals = np.array(psr.flagvals(flagid))
    mask = np.array([fv == flags for fv in flag_vals])
    noise = (equad / day) * np.random.randn(psr.nobs)
    psr.stoas[mask] += noise[mask]
 
 
def _add_ecorr(psr, ecorr, flagid, flags, coarsegrain=0.1, seed=None):
    """Add ECORR (epoch-correlated jitter) to TOAs."""
    if seed is not None:
        np.random.seed(seed)
    t = psr.toas()
    f = np.array(psr.flagvals(flagid))
    avetoas, aveflags, U = _quantize_fast(t, flags=f, dt=coarsegrain)
    epoch_mask = np.array([fv == flags for fv in aveflags])
    ecorrvec = np.where(epoch_mask, ecorr, 0.0)
    psr.stoas[:] += (1 / day) * np.dot(U * ecorrvec, np.random.randn(U.shape[1]))
 
 
def _add_red_noise_achromatic(psr, log10_A, gamma, components=10, tspan=None, seed=None, verbose=False):
    """
    Add ACHROMATIC RED NOISE (frequency-independent).
    
    P_red(f) = (A²/12π²) × (f/f_c)^(-γ)
    
    No frequency dependence - same at all observing frequencies.
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (day / year) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (day / year) * tspan
    
    A = 10**log10_A
    norm = A**2 * year**2 / (12 * math.pi**2 * T)
    
    if verbose:
        print(f"        T={T:.6f} yr, nobs={psr.nobs}, A={A:.4e}, γ={gamma:.3f}, norm={norm:.4e}")
    
    # Build design matrix with sine/cosine components
    size = 2 * components
    F = np.zeros((psr.nobs, size), 'd')
    f = np.zeros(size, 'd')
    
    for i in range(components):
        F[:, 2*i]   = np.cos(2 * math.pi * (i+1) * x)
        F[:, 2*i+1] = np.sin(2 * math.pi * (i+1) * x)
        f[2*i] = f[2*i+1] = (i+1) / T
    
    prior = norm * f**(-gamma)
    y = np.sqrt(prior) * np.random.randn(size)
    psr.stoas[:] += (1.0 / day) * np.dot(F, y)
 
 
def _add_dm_noise(psr, log10_A_DM, gamma_DM, components=10, tspan=None, seed=None, verbose=False):
    """
    Add DM NOISE with proper frequency dependence (MPTA Eq. 7).
    
    P_DM(f; A_DM, γ_DM) = (A_DM²/12π²) × (f/f_c)^(-γ_DM) × (ν/ν_ref)^(-4)
    
    where:
      - (f/f_c)^(-γ_DM): Power law in gravitational wave frequency f
      - (ν/ν_ref)^(-4): Scaling with radio frequency ν
      - ν_ref = 1400 MHz (MPTA reference)
    
    The frequency scaling comes from the fact that DM delays scale as 1/ν².
    When computing power spectral density, this becomes (1/ν²)² = 1/ν⁴.
    
    Implementation:
    1. Generate red noise Fourier coefficients
    2. Scale amplitudes by (ν/ν_ref)^(-4) for each TOA
    3. Generate time series from scaled coefficients
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get observation frequencies (MHz)
    try:
        freqs_MHz = np.array(psr.freqs)
        if freqs_MHz is None or len(freqs_MHz) == 0:
            raise ValueError("No frequencies in pulsar object")
    except (AttributeError, ValueError, TypeError):
        # Fallback: assume 1400 MHz if not available
        if verbose:
            print(f"        WARNING: No frequency information in pulsar, assuming {NU_REF_MHZ:.0f} MHz")
        freqs_MHz = np.full(psr.nobs, NU_REF_MHZ)
    
    # Time span setup
    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (day / year) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (day / year) * tspan
    
    # Power law parameters
    A_DM = 10**log10_A_DM
    norm = A_DM**2 * year**2 / (12 * math.pi**2 * T)
    
    if verbose:
        print(f"        T={T:.6f} yr, nobs={psr.nobs}, A={A_DM:.4e}, γ={gamma_DM:.3f}")
        print(f"        Freq range: {freqs_MHz.min():.0f} - {freqs_MHz.max():.0f} MHz")
    
    # Build design matrix
    size = 2 * components
    F = np.zeros((psr.nobs, size), 'd')
    f = np.zeros(size, 'd')
    
    for i in range(components):
        F[:, 2*i]   = np.cos(2 * math.pi * (i+1) * x)
        F[:, 2*i+1] = np.sin(2 * math.pi * (i+1) * x)
        f[2*i] = f[2*i+1] = (i+1) / T
    
    # Power law spectrum (before frequency scaling)
    prior = norm * f**(-gamma_DM)
    
    # Draw Fourier coefficients
    y = np.sqrt(prior) * np.random.randn(size)
    
    # Generate base red noise
    red_noise = np.dot(F, y)  # Shape: (nobs,)
    
    # Apply FREQUENCY SCALING: (ν/ν_ref)^(-4)
    # This is what makes it DM noise instead of generic red noise
    freq_scaling = (freqs_MHz / NU_REF_MHZ)**(-4.0/2.0)  # Shape: (nobs,)
    dm_noise = freq_scaling * red_noise
    
    psr.stoas[:] += (1.0 / day) * dm_noise
 
 
def _add_chromatic_noise(psr, log10_A_chrom, gamma_chrom, beta, components=10, tspan=None, seed=None, verbose=False):
    """
    Add CHROMATIC NOISE (scattering/refractive index) with frequency dependence.
    
    P_chrom(f; A_ch, γ_ch) = (A_ch²/12π²) × (f/f_c)^(-γ_ch) × (ν/ν_ref)^(-β)
    
    where:
      - (f/f_c)^(-γ_ch): Power law in gravitational wave frequency
      - (ν/ν_ref)^(-β): Scaling with radio frequency
      - β is the chromatic index (typically 2-4)
    
    The β parameter varies depending on the physical process:
      - β = 2: Some absorption processes
      - β = 4: Refractive index variations (common in ISM)
      - β varies with frequency (more realistic models)
    
    For MPTA, β is extracted as the chrom_beta parameter.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get observation frequencies (MHz)
    try:
        freqs_MHz = np.array(psr.freqs)
        if freqs_MHz is None or len(freqs_MHz) == 0:
            raise ValueError("No frequencies in pulsar object")
    except (AttributeError, ValueError, TypeError):
        if verbose:
            print(f"        WARNING: No frequency information in pulsar, assuming {NU_REF_MHZ:.0f} MHz")
        freqs_MHz = np.full(psr.nobs, NU_REF_MHZ)
    
    # Time span setup
    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (day / year) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (day / year) * tspan
    
    # Power law parameters
    A_chrom = 10**log10_A_chrom
    norm = A_chrom**2 * year**2 / (12 * math.pi**2 * T)
    
    if verbose:
        print(f"        T={T:.6f} yr, nobs={psr.nobs}, A={A_chrom:.4e}, γ={gamma_chrom:.3f}, β={beta:.3f}")
        print(f"        Freq range: {freqs_MHz.min():.0f} - {freqs_MHz.max():.0f} MHz")
    
    # Build design matrix
    size = 2 * components
    F = np.zeros((psr.nobs, size), 'd')
    f = np.zeros(size, 'd')
    
    for i in range(components):
        F[:, 2*i]   = np.cos(2 * math.pi * (i+1) * x)
        F[:, 2*i+1] = np.sin(2 * math.pi * (i+1) * x)
        f[2*i] = f[2*i+1] = (i+1) / T
    
    # Power law spectrum
    prior = norm * f**(-gamma_chrom)
    
    # Draw Fourier coefficients
    y = np.sqrt(prior) * np.random.randn(size)
    
    # Generate base red noise
    red_noise = np.dot(F, y)  # Shape: (nobs,)
    
    # Apply FREQUENCY SCALING: (ν/ν_ref)^(-β)
    # β is the chromatic index from MPTA extraction
    freq_scaling = (freqs_MHz / NU_REF_MHZ)**(-beta)  # Shape: (nobs,)
    chrom_noise = freq_scaling * red_noise
    
    psr.stoas[:] += (1.0 / day) * chrom_noise
 
 
def _add_sw_noise(psr, log10_A_SW, gamma_SW, components=10, tspan=None, seed=None, verbose=False):
    """
    Add SOLAR WIND NOISE with frequency dependence.
    
    P_SW(f; A_SW, γ_SW) = (A_SW²/12π²) × (f/f_c)^(-γ_SW) × (ν/ν_ref)^(-4)
    
    Similar to DM noise (both scale as 1/ν²), so same (ν/ν_ref)^(-4) scaling.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get observation frequencies
    try:
        freqs_MHz = np.array(psr.freqs)
        if freqs_MHz is None or len(freqs_MHz) == 0:
            raise ValueError("No frequencies in pulsar object")
    except (AttributeError, ValueError, TypeError):
        if verbose:
            print(f"        WARNING: No frequency information in pulsar, assuming {NU_REF_MHZ:.0f} MHz")
        freqs_MHz = np.full(psr.nobs, NU_REF_MHZ)
    
    # Time span setup
    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (day / year) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (day / year) * tspan
    
    # Power law parameters
    A_SW = 10**log10_A_SW
    norm = A_SW**2 * year**2 / (12 * math.pi**2 * T)
    
    if verbose:
        print(f"        T={T:.6f} yr, nobs={psr.nobs}, A={A_SW:.4e}, γ={gamma_SW:.3f}")
    
    # Build design matrix
    size = 2 * components
    F = np.zeros((psr.nobs, size), 'd')
    f = np.zeros(size, 'd')
    
    for i in range(components):
        F[:, 2*i]   = np.cos(2 * math.pi * (i+1) * x)
        F[:, 2*i+1] = np.sin(2 * math.pi * (i+1) * x)
        f[2*i] = f[2*i+1] = (i+1) / T
    
    # Power law spectrum
    prior = norm * f**(-gamma_SW)
    
    # Draw Fourier coefficients
    y = np.sqrt(prior) * np.random.randn(size)
    
    # Generate base red noise
    red_noise = np.dot(F, y)
    
    # Apply FREQUENCY SCALING: (ν/ν_ref)^(-4) (same as DM)
    freq_scaling = (freqs_MHz / NU_REF_MHZ)**(-4.0/2.0)
    sw_noise = freq_scaling * red_noise
    
    psr.stoas[:] += (1.0 / day) * sw_noise
 
 
def _get_system_list(basename, psr_keys, patterns):
    """Extract unique system/backend names from noise_dict keys."""
    systems = set()
    for k in psr_keys:
        key_suffix = k.replace(f"{basename}_", "")
        for pattern in patterns:
            if key_suffix.endswith(pattern):
                system = key_suffix.replace(pattern, "")
                if system:
                    systems.add(system)
    return systems
 
 
def get_rn_components(par_path, default=30):
    """Determine number of red noise components from .par file."""
    if par_path is None:
        return default
    
    try:
        with open(par_path) as f:
            for line in f:
                key = line.strip().split()[0].upper() if line.strip() else ''
                if key in ('RNAMP', 'RNIDX'):
                    return 100
                if key == 'TNREDC':
                    try:
                        return int(float(line.strip().split()[1]))
                    except (IndexError, ValueError):
                        pass
    except (IOError, FileNotFoundError):
        pass
    
    return default
 
 
def simulate_psr(psr, noise_dict,
                 add_WN=True, add_RN=True, add_DM=True, add_chrom=True, add_SW=False,
                 seed=None, par_path=None, verbose=False):
    """
    Inject noise processes into a pulsar with CORRECT frequency dependence.
    
    Uses MPTA Equation 7 for DM noise:
    
    P_DM(f; A_DM, γ_DM) = (A_DM²/12π²) × (f/f_c)^(-γ_DM) × (ν/ν_ref)^(-4)
    
    where ν is the radio observation frequency from the .tim file, and Equation 8 for chromatic noise:
    
    P_chrom(f; A_chrom, γ_chrom, beta) = (A_chrom²/12π²) × (f/f_c)^(-γ_chrom) × (ν/ν_ref)^(-2 * beta)
    
    Parameters:
    -----------
    psr : libstempo.tempopulsar
        The pulsar to add noise to
    noise_dict : dict
        Parameters from MPTA extraction
    add_WN, add_RN, add_DM, add_chrom, add_SW : bool
        Which noise types to inject
    seed : int, optional
        Random seed
    par_path : str, optional
        Path to .par file for RN components
    verbose : bool
        Print diagnostic information
    
    Returns:
    --------
    psr : modified pulsar object
    """
    psrname  = psr.name
    basename = get_base_name(psrname)
    psr_keys = {k: v for k, v in noise_dict.items()
                if k.startswith(basename + '_')}
    rn_seed    = seed
    dm_seed    = seed + 1 if seed is not None else None
    chrom_seed = seed + 2 if seed is not None else None
    sw_seed    = seed + 3 if seed is not None else None
    wn_seed    = seed + 4 if seed is not None else None
    
    if not psr_keys:
        if verbose:
            print(f"  [{psrname}] no parameters found, returning unmodified")
        return psr
    
    if verbose:
        print(f"  [{psrname}] found {len(psr_keys)} parameters")
    
    # Zero residuals
    if verbose:
        print(f"  [{psrname}] zeroing residuals...", flush=True)
    make_ideal_nofit(psr)
    
    t = psr.toas()
    tspan = t.max() - t.min()
    if verbose:
        print(f"  [{psrname}] tspan = {tspan/365.25:.3f} yr, nobs = {psr.nobs}")
    
    # ========== RED NOISE (ACHROMATIC) ==========
    if add_RN:
        rn_log10_A_key = f"{basename}_red_log10_A"
        rn_gamma_key   = f"{basename}_red_gamma"
        
        if rn_log10_A_key in psr_keys and rn_gamma_key in psr_keys:
            log10_A = psr_keys[rn_log10_A_key]
            gamma   = psr_keys[rn_gamma_key]
            rn_components = get_rn_components(par_path, default=120)
            
            if verbose:
                print(f"  [{psrname}] adding red noise (components={rn_components})...", flush=True)
            
            _add_red_noise_achromatic(psr, log10_A, gamma, components=rn_components,
                                      tspan=tspan, seed=rn_seed, verbose=verbose)
        elif verbose:
            print(f"  [{psrname}] red noise not in parameters, skipping")
    
    # ========== DM NOISE ==========
    if add_DM:
        dm_log10_A_key = f"{basename}_dm_log10_A"
        dm_gamma_key   = f"{basename}_dm_gamma"
        
        if dm_log10_A_key in psr_keys and dm_gamma_key in psr_keys:
            log10_A = psr_keys[dm_log10_A_key]
            gamma   = psr_keys[dm_gamma_key]
            dm_components = get_rn_components(par_path, default=120)
            
            if verbose:
                print(f"  [{psrname}] adding DM noise (components={dm_components})...", flush=True)
            
            _add_dm_noise(psr, log10_A, gamma, components=dm_components,
                         tspan=tspan, seed=dm_seed, verbose=verbose)
        elif verbose:
            print(f"  [{psrname}] DM noise not in parameters, skipping")
    
    # ========== CHROMATIC NOISE ==========
    if add_chrom:
        chrom_log10_A_key = f"{basename}_chrom_log10_A"
        chrom_gamma_key   = f"{basename}_chrom_gamma"
        chrom_beta_key    = f"{basename}_chrom_beta"
        
        if (chrom_log10_A_key in psr_keys and chrom_gamma_key in psr_keys and 
            chrom_beta_key in psr_keys):
            log10_A = psr_keys[chrom_log10_A_key]
            gamma   = psr_keys[chrom_gamma_key]
            beta    = psr_keys[chrom_beta_key]
            chrom_components = get_rn_components(par_path, default=120)
            
            if verbose:
                print(f"  [{psrname}] adding chromatic noise (β={beta:.2f}, components={chrom_components})...",
                      flush=True)
            
            _add_chromatic_noise(psr, log10_A, gamma, beta, components=chrom_components,
                                tspan=tspan, seed=chrom_seed, verbose=verbose)
        elif verbose:
            print(f"  [{psrname}] chromatic noise not in parameters, skipping")
    
    # ========== SOLAR WIND NOISE ==========
    if add_SW:
        sw_log10_A_key = f"{basename}_sw_log10_A"
        sw_gamma_key   = f"{basename}_sw_gamma"
        
        if sw_log10_A_key in psr_keys and sw_gamma_key in psr_keys:
            log10_A = psr_keys[sw_log10_A_key]
            gamma   = psr_keys[sw_gamma_key]
            sw_components = get_rn_components(par_path, default=120)
            
            if verbose:
                print(f"  [{psrname}] adding solar wind noise (components={sw_components})...", flush=True)
            
            _add_sw_noise(psr, log10_A, gamma, components=sw_components,
                         tspan=tspan, seed=sw_seed, verbose=verbose)
        elif verbose:
            print(f"  [{psrname}] solar wind noise not in parameters, skipping")
    
    # ========== WHITE NOISE ==========
    if add_WN:
        try:
            flag_vals = np.array(psr.flagvals('f'))
        except Exception:
            flag_vals = np.array([''] * psr.nobs)
        
        wn_patterns = ['_efac', '_log10_equad', '_log10_ecorr']
        systems = _get_system_list(basename, psr_keys, wn_patterns)
        
        if verbose and systems:
            print(f"  [{psrname}] found {len(systems)} systems: {systems}")
        
        for sys in systems:
            mask = np.array([sys in str(fv) for fv in flag_vals])
            if mask.sum() == 0:
                if verbose:
                    print(f"  [{psrname}] no TOAs for system {sys}, skipping")
                continue
            
            efac = psr_keys.get(f"{basename}_{sys}_efac", 1.0)
            log10_equad = psr_keys.get(f"{basename}_{sys}_log10_equad",
                          psr_keys.get(f"{basename}_{sys}_log10_t2equad", -100.0))
            equad = 10**log10_equad
            
            log10_ecorr = psr_keys.get(f"{basename}_{sys}_log10_ecorr", -100.0)
            ecorr = 10**log10_ecorr
            
            if verbose:
                print(f"  [{psrname}] {sys}: EFAC={efac:.4f}, EQUAD={equad:.4e}, "
                      f"ECORR={ecorr:.4e} ({mask.sum()} TOAs)", flush=True)
            
            if efac > 0.1:
                _add_efac(psr, efac, flagid='f', flags=sys, seed=wn_seed)
            if equad > 1e-8:
                _add_equad(psr, equad, flagid='f', flags=sys, seed=int(wn_seed+1))
            if ecorr > 1e-8:
                _add_ecorr(psr, ecorr, flagid='f', flags=sys, seed=int(wn_seed+2))

    if verbose:
        print(f"  [{psrname}] done", flush=True)
    return psr