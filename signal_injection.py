import numpy as np
from copy import deepcopy
from config import c, G, Msun, pc
from enterprise.signals.gp_bases import createfourierdesignmatrix_red
from enterprise.signals import white_signals, selections
import enterprise.signals.parameter as parameter
from enterprise.signals.utils import create_quantization_matrix


def strain_amplitude(Mc, fGW, d_comov, z):
    """Calculate strain amplitude for circular binary. See Eqn. 26 in https://arxiv.org/pdf/1003.0677"""
    f_rest = 0.5 * fGW * (1 + z)
    d_comov_si = d_comov * 1e6 * pc
    return (2 * (G * Mc)**(5/3) * (2 * np.pi * f_rest)**(2/3)) / (c**4 * d_comov_si)


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

    # # Following enterprise: There is a factor of 3/2 difference between the Hellings & Downs
    # # integral, and the one presented in Jenet et al. (2005; also used by Gair
    # # et al. 2014). This factor 'normalises' the correlation matrix.    
    # if norm:
    #     # Add extra factor of 3/2
    #     c = np.sqrt(1.5) 
    # else:
    #     c = 1.0 

    m_rot = np.cos(psi) * m_hat + np.sin(psi) * n_hat
    n_rot = -np.sin(psi) * m_hat + np.cos(psi) * n_hat
    m_hat, n_hat = m_rot, n_rot

    # changed sign. convention  to 1 - from 1 + for enterprise
    denom = 1 + np.dot(omega_hat, p_hat)
    Fp = 0.5  * ((np.dot(p_hat, m_hat)**2 - np.dot(p_hat, n_hat)**2) / denom)
    Fx =  (np.dot(p_hat, m_hat) * np.dot(p_hat, n_hat)) / denom

    return Fp, Fx


# function from enterprise copied below
# def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True, norm=True):
#     """
#     Create the signal response matrix. All parameters are assumed to be of the
#     same dimensionality.

#     @param pphi:    Phi of the pulsars
#     @param ptheta:  Theta of the pulsars
#     @param gwphi:   Phi of GW propagation direction
#     @param gwtheta: Theta of GW propagation direction
#     @param plus:    Whether or not this is the plus-polarization
#     @param norm:    Normalise the correlations to equal Jenet et. al (2005)

#     @return:    Signal response matrix of Earth-term
#     """
#     # Create the unit-direction vectors. First dimension
#     # will be collapsed later. Sign convention of Gair et al. (2014)
#     Omega = np.array([-np.sin(gwtheta) * np.cos(gwphi), -np.sin(gwtheta) * np.sin(gwphi), -np.cos(gwtheta)])

#     mhat = np.array([-np.sin(gwphi), np.cos(gwphi), np.zeros(gwphi.shape)])
#     nhat = np.array([-np.cos(gwphi) * np.cos(gwtheta), -np.cos(gwtheta) * np.sin(gwphi), np.sin(gwtheta)])

#     p = np.array([np.cos(pphi) * np.sin(ptheta), np.sin(pphi) * np.sin(ptheta), np.cos(ptheta)])

#     # There is a factor of 3/2 difference between the Hellings & Downs
#     # integral, and the one presented in Jenet et al. (2005; also used by Gair
#     # et al. 2014). This factor 'normalises' the correlation matrix.
#     npixels = Omega.shape[2]
#     if norm:
#         # Add extra factor of 3/2
#         c = np.sqrt(1.5) / np.sqrt(npixels)
#     else:
#         c = 1.0 / np.sqrt(npixels)

#     # Calculate the Fplus or Fcross antenna pattern. Definitions as in Gair et
#     # al. (2014), with right-handed coordinate system
#     if plus:
#         # The sum over axis=0 represents an inner-product
#         Fsig = (
#             0.5 * c * (np.sum(nhat * p, axis=0) ** 2 - np.sum(mhat * p, axis=0) ** 2) / (1 - np.sum(Omega * p, axis=0))
#         )
#     else:
#         # The sum over axis=0 represents an inner-product
#         Fsig = c * np.sum(mhat * p, axis=0) * np.sum(nhat * p, axis=0) / (1 - np.sum(Omega * p, axis=0))

#     return Fsig


def r_k(t, psr, binary):
    """Calculate timing residual from single circular SMBHB (Earth term only) This assumption holds for most cases (see Appendix in https://arxiv.org/pdf/1003.0677)."""
    f = binary['f']
    Mc = binary['Mc']
    D_comov = binary['D_comov']
    z = binary['z']
    D_lum = D_comov * (1 + z) # in Mpc
    ra = binary['ra']
    dec = binary['dec']
    psi = binary.get('psi', 0.0)
    phi0 = binary.get('phi0', 0.0)
    iota = binary.get('iota', 0.0)

    h0 = strain_amplitude(Mc, f, D_comov, z)
    ra_psr = psr._raj
    dec_psr = psr._decj
    Fp, Fx = antenna_response(ra_psr, dec_psr, ra, dec, psi)

    t_ref = t[0]
    t_rel = t - t_ref
    phase = 2 * np.pi * f * t_rel + phi0
    
    h_plus = h0 * (1 + np.cos(iota)**2) * np.sin(phase) # Eq. 40 http://arxiv.org/abs/2512.18822
    h_cross = h0 * (- 2 * np.cos(iota)) * np.cos(phase) # ""

    r = (Fp * h_plus + Fx * h_cross) / (2 * np.pi * f) # Eq. 4.21-23 nHz GW Astronomer
    return r

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
                         include_RN=True,   # changed from True
                         include_WN=True):  # changed from True
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
                          spurious cross-correlations. See Chamberlin et al. (2015)
                          PRD 91, 044048, eq. 26-27.
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
    # See Lentati et al. (2014) MNRAS 437, 3004 for the full model.
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


def _gw_residuals_vec(t, psr, population):
    """
    Vectorised GW residual from a full population.
    Processes all binaries simultaneously.
    Returns r of shape (N_toas,).
    """
    N = len(population)

    f_arr    = np.array([b['f']              for b in population])  # (N,)
    Mc_arr   = np.array([b['Mc']             for b in population])
    Dcom_arr = np.array([b['D_comov']        for b in population])
    z_arr    = np.array([b['z']              for b in population])
    ra_arr   = np.array([b['ra']             for b in population])
    dec_arr  = np.array([b['dec']            for b in population])
    psi_arr  = np.array([b.get('psi',  0.0) for b in population])
    phi0_arr = np.array([b.get('phi0', 0.0) for b in population])
    iota_arr = np.array([b.get('iota', 0.0) for b in population])

    # Strain amplitudes (N,)
    f_rest   = 0.5 * (1 + z_arr) * f_arr
    Dcom_si  = Dcom_arr * 1e6 * pc
    h0_arr   = (2 * (G * Mc_arr)**(5/3) * (2*np.pi*f_rest)**(2/3)) / (c**4 * Dcom_si)

    # Antenna patterns (N,)
    Fp_arr, Fx_arr = _antenna_response_vec(
        psr._raj, psr._decj, ra_arr, dec_arr, psi_arr
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
    Reference: Chamberlin et al. (2015) PRD 91, 044048, eq. 26-27.

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



# =====================================================================
# INJECTION FUNCTIONS
# =====================================================================

def inject_population_into_psrs(
        psrs, population,
        pure_signal=True,
        add=False,
        verbose=False,
        include_RN=False,
        include_WN=False,
        pulsar_noise_params=None,
        chunk_size=None,
        max_memory_mb=200):
    """
    Inject SMBHB population (and optionally noise) into pulsar residuals.

    Parameters
    ----------
    psrs                 : list of enterprise Pulsar objects
    population           : list of binary dicts
    pure_signal          : if True, replace residuals (don't add to existing)
    add                  : if True AND pure_signal=True, add signal to existing residuals
    verbose              : print per-pulsar signal RMS
    include_RN           : also draw and inject red noise (default False)
    include_WN           : also draw and inject white noise (default False)
    pulsar_noise_params  : required if include_RN or include_WN
    chunk_size           : binaries per GW chunk (None = auto)
    max_memory_mb        : memory cap for auto chunk sizing

    Notes
    -----
    For OS SNR estimation: pure_signal=True, include_RN=False, include_WN=False.
    For realistic noise simulations: include_RN=True, include_WN=True.
    """
    tmin  = min(p.toas.min() for p in psrs)
    tmax  = max(p.toas.max() for p in psrs)
    Tspan = tmax - tmin

    for psr in psrs:
        t_sec = np.asarray(psr.toas, dtype=float)

        r_new = population_residuals_vectorised(
            t_sec, psr, population, Tspan=Tspan,
            pulsar_noise_params=pulsar_noise_params,
            include_GW=True,
            include_RN=include_RN,
            include_WN=include_WN,
            chunk_size=chunk_size,
            max_memory_mb=max_memory_mb,
        )

        if verbose:
            print(f"{psr.name}: injected RMS = {r_new.std()*1e6:.3f} μs")

        if pure_signal:
            psr._residuals = r_new if not add else psr.residuals + r_new
        else:
            psr._residuals = psr.residuals + r_new

    return psrs


def inject_population_subset_cached(
        psrs, population, N_binaries,
        psrs_injected_cache=None,
        pure_signal=True,
        verbose=False,
        include_RN=False,
        include_WN=False,
        pulsar_noise_params=None,
        chunk_size=None,
        max_memory_mb=200):
    """
    Inject first N_binaries using pre-computed cached signals.

    Falls back to live computation if cache is None or pulsar missing from cache.
    include_RN / include_WN are drawn fresh each call (not cached).
    """
    if psrs_injected_cache is None:
        return inject_population_into_psrs(
            psrs, population[:N_binaries],
            pure_signal=pure_signal, verbose=verbose,
            include_RN=include_RN, include_WN=include_WN,
            pulsar_noise_params=pulsar_noise_params,
            chunk_size=chunk_size, max_memory_mb=max_memory_mb,
        )

    tmin  = min(p.toas.min() for p in psrs)
    tmax  = max(p.toas.max() for p in psrs)
    Tspan = tmax - tmin

    for psr in psrs:
        # --- GW from cache ---
        if psr.name in psrs_injected_cache:
            r_gw = np.zeros(len(psr.toas), dtype=float)
            cache_psr = psrs_injected_cache[psr.name]
            for bin_idx in range(N_binaries):
                if bin_idx in cache_psr:
                    r_gw += cache_psr[bin_idx]
        else:
            # Fallback: compute GW fresh
            t_sec = np.asarray(psr.toas, dtype=float)
            cs    = chunk_size or _auto_chunk_size(len(t_sec), max_memory_mb)
            r_gw  = _gw_residuals_chunked(t_sec, psr, population[:N_binaries], cs)

        r_new = r_gw.copy()

        # --- Noise (always drawn fresh — not cached) ---
        if include_RN:
            if pulsar_noise_params is None:
                raise ValueError("pulsar_noise_params required when include_RN=True")
            rn     = pulsar_noise_params[psr.name]['red_noise']
            r_new += draw_red_noise_residuals(
                psr, rn['log10_A'], rn['gamma'], Tspan
            )
        if include_WN:
            if pulsar_noise_params is None:
                raise ValueError("pulsar_noise_params required when include_WN=True")
            r_new += white_noise_residual(psr, pulsar_noise_params)

        if verbose:
            print(f"{psr.name}: injected RMS = {r_new.std()*1e6:.3f} μs")

        if pure_signal:
            psr._residuals = r_new
        else:
            psr._residuals = psr.residuals + r_new

    return psrs

# =====================================================================
# CACHING STRATEGY FOR ENSEMBLE SEARCHES
# =====================================================================

def precompute_binary_signals(psrs, population, cache=None, verbose=False):
    """
    Pre-compute timing residual signals from EACH binary individually.
    
    Returns a cache dict: {psr_name: {binary_idx: residual_array}}
    
    This allows fast computation of subset signals:
    N-binary signal = sum of individual binary signals (no recomputation needed)
    
    Args:
        psrs: list of pulsar objects
        population: list of binary dictionaries
        cache: existing cache to update (if None, creates new cache)
        verbose: if True, print progress
    
    Returns:
        cache: dictionary of pre-computed signals
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
                
                if verbose and (bin_idx + 1) % 100 == 0:
                    print(f"  {psr_name}: cached {bin_idx + 1}/{len(population)} binaries")
    
    if verbose:
        print(f"Cache complete: {len(cache)} pulsars, {len(population)} binaries each")
    
    return cache