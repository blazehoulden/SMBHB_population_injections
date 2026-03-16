import numpy as np
from SMBHB_pop_synth import H0_KMS_MPC, MEGAPARSEC_IN_METERS
import sys
from config import generate_population
from signal_injection import draw_red_noise_residuals, strain_amplitude, white_noise_residual
from enterprise.signals import white_signals, selections
import enterprise.signals.parameter as parameter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def GWB_PSD(freq, h_contrib):
    """
    Power spectral density of the gravitational wave background (GWB).

    Eqn. 16 of https://iopscience.iop.org/article/10.1088/0264-9381/30/22/224015/meta

    Parameters
    ----------
    freq : float or np.ndarray
        GW frequency [Hz]. Can be shape (B,) for B binaries.
    h_contrib : float or np.ndarray
        Characteristic strain contribution (dimensionless). Same shape as freq.

    Returns
    -------
    float or np.ndarray
        One-sided PSD [s^3].
    """
    return h_contrib**2 / (12.0 * np.pi**2.0) * freq**(-3.0) # [s^3]

def omega_GW(f, h_cont):
    """
    Gravitational wave energy density parameter. Unnecessary now.
    
    Parameters
    ----------
    f : array_like
        Frequency [Hz]
    h_cont : array_like
        Characteristic strain contribution
        
    Returns
    -------
    array_like
        Omega_GW at given frequencies [dimensionless]
    """
    omega = 2 * np.pi**2 / (3 * H0_S**2) * f**2 * h_cont**2 # Thrane and Romano (2013) 
    return omega # [dimensionless]

def strain_PSD(f, h_cont):
    """
    Power spectral density of the GW signal in terms of strain. Also unnecessary now.
    
    Parameters
    ----------
    f : array_like
        Frequency [Hz]
    h_cont : array_like
        Characteristic strain contribution
        
    Returns
    -------
    array_like
        Signal PSD in terms of strain at given frequencies [strain^2 Hz^-1]
    """
    psd_strain = h_cont**2 * f**(-1) # Eq. 1.54 Di Marco thesis
    return psd_strain # [strain^2 Hz^-1] 

def pulsar_white_noise_psd(sigma_t=100.0*1e-9, delta_t=1.0/20.0):
    """
    White noise PSD for a single pulsar.

    Parameters
    ----------
    sigma_t : float
        TOA measurement uncertainty [s]. Default 100 ns.
    delta_t : float
        Cadence [yr]. Default 1/20 yr (≈ fortnightly).

    Returns
    -------
    float
        White noise PSD [s^3].
    """
    delta_t = delta_t * 365.25 * 86400  # convert years → seconds
    return 2 * sigma_t**2 * delta_t # [s^3]

def analytic_white_noise_psd(psr, noise_params):
    """
    Compute white noise PSD using EFAC/EQUAD-corrected, epoch-averaged Nvec,
    without needing to build a full ENTERPRISE PTA.
    """
    # Build per-TOA effective variance using EFAC/EQUAD from noise params
    Nvec = np.zeros(len(psr.toas))
    # Get backend flags for each TOA
    
    for itoa, (sigma, backend) in enumerate(zip(psr._toaerrs, psr._flags['f'])):
        efac  = noise_params.get(f"{psr.name}_{backend}_efac", 1.0)
        equad = 10**noise_params.get(f"{psr.name}_{backend}_log10_t2equad", -10.0)
        Nvec[itoa] = efac**2 * sigma**2 + equad**2

    # Epoch-average (harmonic mean within each epoch)
    U, _ = create_quantization_matrix(psr.toas, nmin=1)
    n_epochs = U.shape[1]
    psr_tspan = psr.toas.max() - psr.toas.min()
    cadence = psr_tspan / n_epochs

    epoch_variance = np.zeros(n_epochs)
    for j in range(n_epochs):
        mask = U[:, j].astype(bool)
        epoch_variance[j] = 1.0 / np.sum(1.0 / Nvec[mask])

    sigma_epoch_sq = np.median(epoch_variance)
    return 2.0 * sigma_epoch_sq * cadence

def effective_toaerrs(psr, noise_params):
    """
    Compute EFAC/EQUAD-corrected TOA uncertainties for a pulsar,
    averaged over backends, matching what ENTERPRISE puts in Nvec.

    Returns sigma_eff in seconds.
    """
    backends = set()
    for key in noise_params:
        if key.startswith(psr.name + '_') and key.endswith('_efac'):
            backend = key[len(psr.name)+1:-len('_efac')]
            backends.add(backend)

    if not backends:
        return np.median(psr.toaerrs)  # fallback

    # Compute effective sigma for each backend and take median
    sigma_effs = []
    for backend in backends:
        efac   = noise_params.get(f"{psr.name}_{backend}_efac",   1.0)
        equad  = 10**noise_params.get(f"{psr.name}_{backend}_log10_t2equad", -10.0)
        # ENTERPRISE Nvec = efac^2 * sigma_t^2 + equad^2
        sigma_eff = np.sqrt(efac**2 * np.median(psr.toaerrs)**2 + equad**2)
        sigma_effs.append(sigma_eff)

    return np.median(sigma_effs)

def actual_cadence_yr(psr):
    tspan = psr.toas.max() - psr.toas.min()
    U, _ = create_quantization_matrix(psr.toas, nmin=1)
    n_epochs = U.shape[1]
    return (tspan / n_epochs) / (365.25 * 86400)


def pulsar_red_noise_psd(freq, log10A_red, gamma_red, fyr=1.0/(365.25*86400)):
    """
    Power-law red noise PSD for a single pulsar.

    Parameters
    ----------
    freq : float or np.ndarray
        Frequency [Hz]. Can be shape (B,) for B binaries.
    log10A_red : float or np.ndarray
        log10 of the red noise amplitude. Can be shape (N,) for N pulsars.
    gamma_red : float or np.ndarray
        Red noise spectral index. Can be shape (N,) for N pulsars.
    fyr : float
        Reference frequency (1/yr) [Hz].

    Returns
    -------
    float or np.ndarray
        Red noise PSD [s^3]. Shape (B, N) if freq is (B,) and params are (N,).
    """
    A_red = 10**log10A_red
    return A_red**2 / (12.0 * np.pi**2) * (freq / fyr)**(-gamma_red) * fyr**(-3.0) # [s^3]

def plot_pulsar_psd(parsed_noise_params, frequencies, sigma_ns = 100.0, delta_t_yr = 1.0/20.0):
    """
    Plot the PSD for a single pulsar's noise across frequencies.
    
    Parameters
    ----------
    parsed_noise_params : dict
        Dictionary of noise parameters for a single pulsar
    frequencies : array_like
        Array of frequencies [Hz]
    """
    import matplotlib.pyplot as plt
    
    red_psd_values = pulsar_red_noise_psd(frequencies, parsed_noise_params['red_noise']['log10_A'], parsed_noise_params['red_noise']['gamma'])
    white_psd_values = pulsar_white_noise_psd()
    total_psd = red_psd_values + white_psd_values
    plt.figure(figsize=(8, 5))
    plt.loglog(frequencies, total_psd, label='Total Noise PSD')

        
    plt.loglog(frequencies, red_psd_values, label='Red Noise PSD', linestyle='--')
    plt.axhline(white_psd_values, color='r', label='White Noise PSD', linestyle='--')
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [s^3]')
    plt.title('Pulsar Noise PSD')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()


def chi_coeff(pulsar1, pulsar2):
    """
    Hellings-Downs correlation coefficient for a pulsar pair.

    Values lie in roughly (-0.1519, 0.5).

    Parameters
    ----------
    pulsar1, pulsar2 : Pulsar
        Pulsar objects with a `.pos` unit-vector attribute.

    Returns
    -------
    float
        HD coefficient Γ_{IJ}.
    """
    cos_zeta = np.dot(pulsar1.pos, pulsar2.pos)
    # δ_{IJ} = 1 only for an auto-correlation (same pulsar)
    if np.array_equal(pulsar1.pos, pulsar2.pos):
        delta_IJ = 1  # no two distinct pulsars share a position in PTAs, so should never get 1
    else:
        delta_IJ = 0
    x = (1 - cos_zeta) / 2.0
    return 3.0 / 2.0 * (1.0 / 3.0 + x * (np.log(x) - 1.0 / 6.0)) + 0.5 * delta_IJ


def chi_coeff_matrix(pulsars):
    """
    Upper-triangular matrix of Hellings-Downs coefficients for an array of pulsars.

    Parameters
    ----------
    pulsars : list[Pulsar]
        List of pulsar objects.

    Returns
    -------
    np.ndarray, shape (N, N)
        Upper-triangular HD coefficient matrix (diagonal and lower triangle are zero).
    """
    N_pulsars = len(pulsars)
    coeff_matrix = np.zeros((N_pulsars, N_pulsars))
    for i in range(N_pulsars):
        for j in range(i + 1, N_pulsars):
            coeff_matrix[i, j] = chi_coeff(pulsars[i], pulsars[j])
    return coeff_matrix


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

def antenna_response_vectorised(
    raj_arr:  np.ndarray,   # (N,) pulsar RA  [rad]
    decj_arr: np.ndarray,   # (N,) pulsar Dec [rad]
    src_ra:   np.ndarray,   # (B,) source RA  [rad]
    src_dec:  np.ndarray,   # (B,) source Dec [rad]
    psi:      np.ndarray,   # (B,) polarisation angle [rad]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised version of antenna_response() for B sources × N pulsars.
    Matches exactly the scalar convention in antenna_response():
        - omega_hat has leading minus signs
        - m_hat = [sin(az), -cos(az), 0]
        - n_hat uses polar/azimuthal form
        - denominator = 1 + omega_hat · p_hat  (pulsar term)
        - Fp = 0.5 * (pm² - pn²) / denom
        - Fx = (pm * pn) / denom

    Returns
    -------
    Fp, Fx : ndarray, shape (B, N)
    """
    # ------------------------------------------------------------------ pulsar vectors (N, 3)
    psr_polar = np.pi / 2.0 - decj_arr          # (N,)
    p_hat = np.stack([
        np.sin(psr_polar) * np.cos(raj_arr),
        np.sin(psr_polar) * np.sin(raj_arr),
        np.cos(psr_polar),
    ], axis=1)  # (N, 3)

    # ------------------------------------------------------------------ source vectors (B, 3)
    src_polar = np.pi / 2.0 - src_dec           # (B,)
    src_az    = src_ra                           # (B,)

    # Propagation direction — note the leading minus signs matching your convention
    omega_hat = np.stack([
        -np.sin(src_polar) * np.cos(src_az),
        -np.sin(src_polar) * np.sin(src_az),
        -np.cos(src_polar),
    ], axis=1)  # (B, 3)

    # Polarisation basis — matching your m_hat and n_hat exactly
    m_hat = np.stack([
         np.sin(src_az),
        -np.cos(src_az),
        np.zeros_like(src_az),
    ], axis=1)  # (B, 3)

    n_hat = np.stack([
        -np.cos(src_polar) * np.cos(src_az),
        -np.cos(src_polar) * np.sin(src_az),
         np.sin(src_polar),
    ], axis=1)  # (B, 3)

    # Rotate by polarisation angle — matching your m_rot, n_rot
    cos_psi = np.cos(psi)[:, None]   # (B, 1)
    sin_psi = np.sin(psi)[:, None]   # (B, 1)

    m_rot = cos_psi * m_hat + sin_psi * n_hat    # (B, 3)
    n_rot = -sin_psi * m_hat + cos_psi * n_hat   # (B, 3)

    # ------------------------------------------------------------------ projections
    # p_hat · m_rot for all (B, N) pairs: einsum 'ni,bi->bn'
    pm = np.einsum('ni,bi->bn', p_hat, m_rot)    # (B, N)
    pn = np.einsum('ni,bi->bn', p_hat, n_rot)    # (B, N)

    # Denominator: 1 + omega_hat · p_hat — shape (B, N)
    denom = 1.0 + np.einsum('ni,bi->bn', p_hat, omega_hat)   # (B, N)

    # Antenna patterns — matching your scalar formula exactly
    Fp = 0.5 * (pm**2 - pn**2) / denom   # (B, N)
    Fx = (pm * pn) / denom               # (B, N)

    return Fp, Fx


def overlap_reduction_function(pulsar1, pulsar2, binary):
    """
    Compute the overlap reduction function (ORF) Γ_{IJ} for a pair of pulsars and a given binary source.

    Parameters
    ----------
    pulsar1, pulsar2 : Pulsar
        Pulsar objects with a `.pos` unit-vector attribute.

    Returns
    -------
    float
        Overlap reduction function Γ_{IJ} - when the number of binaries approaches infinity, this becomes the HD correlation.
    """
    ant_rep_p1_plus, ant_rep_p1_cross = antenna_response(pulsar1._raj, pulsar1._dec, binary['ra'], binary['dec'], binary['psi'])
    ant_rep_p2_plus, ant_rep_p2_cross = antenna_response(pulsar2._raj, pulsar2._dec, binary['ra'], binary['dec'], binary['psi'])
    beta = 1  # normalization factor for the ORF
    return beta * (ant_rep_p1_plus * ant_rep_p2_plus + ant_rep_p1_cross * ant_rep_p2_cross)


def compute_orf_sq_chunk(binaries_chunk, pulsars, i_idx, j_idx):
    """
    Compute squared ORF values for a chunk of binaries and all unique pulsar pairs.

    When averaged over many isotropically distributed binaries the mean ORF²
    should converge to the squared Hellings-Downs coefficient, providing a
    consistency check between the two approaches.

    Parameters
    ----------
    binaries_chunk : list[dict]
        Subset of binary dicts for this chunk.  Each must have 'ra', 'dec',
        and optionally 'psi' (defaults to 0.0).
    pulsars : list[Pulsar]
        Full pulsar list (same ordering used to build i_idx / j_idx).
    i_idx : np.ndarray, shape (P,)
        Row indices of the upper-triangle pulsar pairs.
    j_idx : np.ndarray, shape (P,)
        Column indices of the upper-triangle pulsar pairs.

    Returns
    -------
    np.ndarray, shape (B, P)
        orf_sq[b, p] = Γ_{i_idx[p], j_idx[p]}(binary b)²
    """
    B = len(binaries_chunk)
    P = len(i_idx)
    orf_sq = np.empty((B, P))
    orf_vals = np.empty((B, P))  # for diagnostics
    
    pos = np.array([p.pos for p in pulsars])  # (N, 3)
    print("dir:", dir(pulsars[0]))
    xi_arr = np.arccos(np.clip(
        np.einsum('ij,ij->i', pos[i_idx], pos[j_idx]),
        -1.0, 1.0
    ))  # shape (P,) — same for all binaries

    for b, binary in enumerate(binaries_chunk):
        # Pre-compute antenna responses for every pulsar for this binary
        # shape: (N, 2) — [Fp, Fx] per pulsar
        ra  = binary['ra']
        dec = binary['dec']
        psi = binary['psi']
        beta = 1  # normalization factor for the ORF

        ant = np.array([
            antenna_response(p._raj, p._decj, ra, dec, psi)
            for p in pulsars
        ])  # (N, 2)

        Fp = ant[:, 0]  # (N,)
        Fx = ant[:, 1]  # (N,)

        # ORF for every unique pair using index arrays — no inner Python loop
        orf_vals[b] = beta * (Fp[i_idx] * Fp[j_idx] + Fx[i_idx] * Fx[j_idx])  # (P,)
        orf_sq[b] = orf_vals[b] ** 2


    return orf_sq, xi_arr, orf_vals  # (B, P), (P,), (B, P) 


def build_pulsar_cache(pulsars, parsed_noise_params):
    """
    Precompute all pulsar-dependent quantities that are frequency-independent
    and therefore constant across all binaries. Call this once before any
    binary loop.

    Quantities precomputed
    ----------------------
    - White noise PSD per pulsar                        shape (N,)
    - Red noise amplitude and spectral index per pulsar shape (N,)
    - Full HD coefficient matrix                        shape (N, N)
    - Upper-triangle pair indices (i_idx, j_idx)        shape (P,) each, P = N(N-1)/2
    - HD coefficients² for each unique pair             shape (P,)
    - Pulsar list (stored for ORF mode)                 list[Pulsar]   # CHANGED: added

    Parameters
    ----------
    pulsars : list[Pulsar]
        Pulsar objects (must have `.name`, `.pos`, `.toaerrs`).
    parsed_noise_params : dict
        Nested dict {pulsar_name: {'red_noise': {'log10_A': ..., 'gamma': ...}}}.

    Returns
    -------
    dict
        Cache containing all precomputed arrays, ready to pass into
        `SNR_sq_all_binaries` and `N_needed_for_population`.
    """
    N = len(pulsars)

    # White noise: one value per pulsar, depends only on TOA errors and cadence
    white_noise_arr = np.array([
        pulsar_white_noise_psd(sigma_t=np.median(p.toaerrs), delta_t=1.0 / 20.0)
        for p in pulsars
    ])  # shape (N,)

    # Red noise parameters: extract into arrays for broadcasting over frequencies
    log10A_arr = np.array([
        parsed_noise_params[p.name]['red_noise']['log10_A'] for p in pulsars
    ])  # shape (N,)
    gamma_arr = np.array([
        parsed_noise_params[p.name]['red_noise']['gamma'] for p in pulsars
    ])  # shape (N,)

    # HD coefficient matrix and upper-triangle extraction
    chi_matrix   = chi_coeff_matrix(pulsars)           # shape (N, N)
    i_idx, j_idx = np.triu_indices(N, k=1)             # shape (P,) each
    chi_sq_pairs = chi_matrix[i_idx, j_idx]**2         # shape (P,)

    return {
        'white_noise_arr': white_noise_arr,   # (N,)
        'log10A_arr':      log10A_arr,         # (N,)
        'gamma_arr':       gamma_arr,          # (N,)
        'chi_matrix':      chi_matrix,         # (N, N)
        'i_idx':           i_idx,              # (P,)
        'j_idx':           j_idx,              # (P,)
        'chi_sq_pairs':    chi_sq_pairs,       # (P,)
        'pulsars':         pulsars,            # list[Pulsar]  # CHANGED: stored for ORF mode
    }


def estimate_chunk_size(N_pulsars, target_memory_GB=1.0, dtype=np.float64):
    """
    Estimate a safe chunk size (number of binaries per batch) so that the
    dominant working array — total_noise of shape (B, N) used to form
    noise_pairs of shape (B, P) — stays within a target memory budget.

    Memory breakdown per chunk of B binaries
    -----------------------------------------
    - signal_psd      (B,)    —  B * 8 bytes
    - red_noise       (B, N)  —  B * N * 8 bytes
    - total_noise     (B, N)  —  B * N * 8 bytes
    - noise_pairs     (B, P)  —  B * P * 8 bytes   ← dominant term, P = N(N-1)/2
    - pair_integrands (B, P)  —  B * P * 8 bytes
    Total ≈ B * (2 + 2N + 4P) * bytes_per_element

    Parameters
    ----------
    N_pulsars : int
        Number of pulsars in the array.
    target_memory_GB : float
        Soft memory ceiling for one chunk [GB]. Default 1 GB.
    dtype : np.dtype
        Floating point dtype in use. Default float64 (8 bytes).

    Returns
    -------
    int
        Recommended chunk size (at least 1).
    """
    P              = N_pulsars * (N_pulsars - 1) // 2      # number of unique pairs
    bytes_per_elem = np.dtype(dtype).itemsize               # 8 for float64
    # Coefficients from the memory breakdown above
    scalars_per_binary = 2 + 2 * N_pulsars + 4 * P
    bytes_per_binary   = scalars_per_binary * bytes_per_elem
    target_bytes       = target_memory_GB * 1024**3
    chunk_size         = max(1, int(target_bytes / bytes_per_binary))
    print(
        f"Memory estimate: {bytes_per_binary / 1024**2:.2f} MB per binary "
        f"(N={N_pulsars}, P={P}) → chunk size = {chunk_size} "
        f"for {target_memory_GB} GB target"
    )
    return chunk_size


def SNR_sq_chunk(freqs, h_contribs, delta_fs, pulsar_cache, T_obs,
                 binaries_chunk=None, use_orf=True):
    """
    Core vectorised SNR² kernel for a single chunk of binaries.

    All B binaries in the chunk and all P = N(N-1)/2 pulsar pairs are
    handled simultaneously via numpy broadcasting — no Python loops.

    The formula for each binary b, summed over unique pairs (i < j), is:

        SNR²_b = 2 T_obs Δf_b  Σ_{i<j}  [Γ_{ij}² S_h,b²] / [P_{b,i} P_{b,j}]

    where:
        S_h,b   = GWB_PSD(f_b, h_b)                         signal PSD
        P_{b,k} = S_h,b + S_red,b,k + S_white,k             total noise PSD

    Γ_{ij} is either:
        - the Hellings-Downs coefficient (use_orf=False)
        - the binary-specific ORF from antenna_response (use_orf=True)

    Parameters
    ----------
    freqs : np.ndarray, shape (B,)
        GW frequencies for this chunk [Hz].
    h_contribs : np.ndarray, shape (B,)
        Characteristic strain contributions for this chunk.
    delta_fs : np.ndarray, shape (B,)
        Frequency bin widths for this chunk [Hz].
    pulsar_cache : dict
        Output of `build_pulsar_cache`.
    T_obs : float
        Observation time [s].
    binaries_chunk : list[dict] or None                      # CHANGED: new parameter
        The raw binary dicts for this chunk.  Required when use_orf=True so
        that sky positions (ra, dec, psi) can be read.  Ignored when
        use_orf=False.
    use_orf : bool                                           # CHANGED: new parameter
        If True, compute the binary-dependent ORF for each binary instead of
        using the precomputed Hellings-Downs chi_sq_pairs.  Default False
        preserves the original behaviour exactly.

    Returns
    -------
    np.ndarray, shape (B,)
        SNR² for each binary in the chunk.
    """
    white_noise_arr = pulsar_cache['white_noise_arr']    # (N,)
    log10A_arr      = pulsar_cache['log10A_arr']         # (N,)
    gamma_arr       = pulsar_cache['gamma_arr']          # (N,)
    i_idx           = pulsar_cache['i_idx']              # (P,)
    j_idx           = pulsar_cache['j_idx']              # (P,)

    # Signal PSD — shape (B,)
    signal_psd = GWB_PSD(freqs, h_contribs)

    # Red noise — shape (B, N)
    # freqs[:, None] is (B, 1); broadcasting against (N,) gives (B, N)
    fyr = 1.0 / (365.25 * 86400)
    A_red = 10**log10A_arr                                              # (N,)
    red_noise = (
        A_red**2 / (12.0 * np.pi**2)
        * (freqs[:, None] / fyr)**(-gamma_arr)                          # (B, N)
        * fyr**(-3.0)
    )  # (B, N)

    # Total noise per pulsar — shape (B, N)
    total_noise = signal_psd[:, None] + red_noise + white_noise_arr     # (B, N)

    # Pair noise products — shape (B, P)
    # Selecting columns i_idx and j_idx from (B, N) gives (B, P) each
    noise_pairs = total_noise[:, i_idx] * total_noise[:, j_idx]         # (B, P)

    if use_orf:
        # Binary-dependent ORF: shape (B, P) — one ORF² per binary per pair
        if binaries_chunk is None:
            raise ValueError("binaries_chunk must be supplied when use_orf=True")
        pulsars = pulsar_cache['pulsars']
        corr_sq_pairs, xi_arr, _ = compute_orf_sq_chunk(
            binaries_chunk, pulsars, i_idx, j_idx
        )  # (B, P)
    else:
        # Original behaviour: precomputed HD chi² — shape (P,), broadcasts to (B, P)
        corr_sq_pairs = pulsar_cache['chi_sq_pairs']  # (P,)

    # SNR² per binary — shape (B,)
    # corr_sq_pairs is (B, P) [ORF] or (P,) [HD], both broadcast correctly
    # with signal_psd[:, None]**2 shape (B, 1) → (B, P)
    pair_integrands  = corr_sq_pairs * signal_psd[:, None]**2 / noise_pairs  # (B, P)
    SNR_sq_binaries  = 2.0 * T_obs * delta_fs * pair_integrands.sum(axis=1)  # (B,)

    return SNR_sq_binaries, corr_sq_pairs  # return corr_sq_pairs for diagnostics


def SNR_sq_all_binaries(binaries, pulsar_cache, strain_data, T_obs,
                        chunk_size=None, target_memory_GB=1.0, use_orf=True):
    """
    Compute SNR² for every binary, processing in chunks to cap peak memory use.

    If `chunk_size` is None it is estimated automatically from `target_memory_GB`
    via `estimate_chunk_size`. Set `chunk_size` explicitly to override.

    Parameters
    ----------
    binaries : list[dict]
        Each dict must contain:
            'h_c_contrib' : float  — characteristic strain contribution
            'f'           : float  — GW frequency [Hz]
            'freq_bin'    : int    — index into strain_data['bin_edges']
            'Mc'          : float  — chirp mass [kg]
        When use_orf=True each dict must also contain 'ra', 'dec', and
        optionally 'psi'.                                  # CHANGED: noted
    pulsar_cache : dict
        Output of `build_pulsar_cache`.
    strain_data : dict
        Must contain 'bin_edges': array of frequency bin edges [Hz].
    T_obs : float
        Observation time [s].
    chunk_size : int or None
        Number of binaries per chunk. If None, estimated automatically.
    target_memory_GB : float
        Memory budget per chunk used by auto-estimation [GB]. Default 1 GB.
    use_orf : bool                                         # CHANGED: new parameter
        If True, use the binary-dependent antenna-pattern ORF instead of
        the precomputed Hellings-Downs coefficients.  Default False.

    Returns
    -------
    SNR_sq_binaries : np.ndarray, shape (B,)
        SNR² contribution of each binary.
    """
    bin_edges    = strain_data['bin_edges']
    freqs        = np.array([b['f']           for b in binaries])   # (B,)
    h_contribs   = np.array([b['h_c_contrib'] for b in binaries])   # (B,)
    chirp_masses = np.array([b['Mc']          for b in binaries])   # (B,)
    delta_fs     = np.array([
        bin_edges[b['freq_bin'] + 1] - bin_edges[b['freq_bin']] for b in binaries
    ])  # (B,)

    B = len(binaries)

    # ------------------------------------------------------------------ determine chunk size
    N_pulsars = len(pulsar_cache['white_noise_arr'])
    if chunk_size is None:
        chunk_size = estimate_chunk_size(N_pulsars, target_memory_GB=target_memory_GB)

    n_chunks = int(np.ceil(B / chunk_size))
    print(f"Processing {B} binaries in {n_chunks} chunk(s) of up to {chunk_size}")

    # ------------------------------------------------------------------ chunked computation
    SNR_sq_binaries = np.empty(B)
    chi_sq_pairs = []  # collect for diagnostics
    for chunk_idx, start in enumerate(range(0, B, chunk_size)):
        end   = min(start + chunk_size, B)
        chunk = slice(start, end)
        print(f"  Chunk {chunk_idx + 1}/{n_chunks}: binaries {start}–{end - 1}")

        SNR_sq_binaries[chunk], chi_sq_pairs_chunk = SNR_sq_chunk(
            freqs          = freqs[chunk],
            h_contribs     = h_contribs[chunk],
            delta_fs       = delta_fs[chunk],
            pulsar_cache   = pulsar_cache,
            T_obs          = T_obs,
            binaries_chunk = binaries[start:end] if use_orf else None,  
            use_orf        = use_orf,                                    
        )
        chi_sq_pairs.append(chi_sq_pairs_chunk)  # collect for diagnostics

    # ------------------------------------------------------------------ diagnostics 
    white_noise_arr = pulsar_cache['white_noise_arr']
    chi_sq_pairs = np.concatenate(chi_sq_pairs, axis=0)  # (B, P)
    fyr = 1.0 / (365.25 * 86400)
    A_red = 10**pulsar_cache['log10A_arr']
    red_noise_all = (
        A_red**2 / (12.0 * np.pi**2)
        * (freqs[:, None] / fyr)**(-pulsar_cache['gamma_arr'])
        * fyr**(-3.0)
    )  # (B, N) — recomputed cheaply for printing only
    signal_psd_all = GWB_PSD(freqs, h_contribs)                     # (B,)
    for b_idx in range(B):
        if SNR_sq_binaries[b_idx] > 0.5**2:
            print(
                f"Binary {b_idx}: f={freqs[b_idx]:.3e} Hz, SNR={np.sqrt(SNR_sq_binaries[b_idx]):.3e}, "
                f"signal PSD={signal_psd_all[b_idx]:.3e}, h_contrib={h_contribs[b_idx]:.3e}, "
                f"chirp mass={chirp_masses[b_idx] / (1.989e30):.3e} Msun, "
                f"pulsar white noise range=({white_noise_arr.min():.3e}, {white_noise_arr.max():.3e}), "
                f"pulsar red noise range=({red_noise_all[b_idx].min():.3e}, {red_noise_all[b_idx].max():.3e}), "
                f"chi coeff^2 range=({chi_sq_pairs.min():.3e}, {chi_sq_pairs.max():.3e})"
            )

    return SNR_sq_binaries, chi_sq_pairs


def N_needed_for_population(binaries, pulsars, parsed_noise_params, strain_data,
                            target_SNR, T_obs, chunk_size=None,
                            target_memory_GB=1.0, use_orf=True):
    """
    Find the minimum number of binaries (in input order) needed to reach a
    target cumulative SNR, using chunked vectorised SNR² computation.

    Parameters
    ----------
    binaries : list[dict]
        Binary parameter dicts (pre-sorted, typically by descending strain).
        When use_orf=True each dict must also have 'ra', 'dec', 'psi'.  # CHANGED: noted
    pulsars : list[Pulsar]
        Pulsar objects (must have `.name`, `.pos`, `.toaerrs`).
    parsed_noise_params : dict
        Noise parameters dict {pulsar_name: {'red_noise': {'log10_A', 'gamma'}}}.
    strain_data : dict
        Must contain 'bin_edges' array.
    target_SNR : float
        Cumulative SNR threshold to reach.
    T_obs : float
        Observation time [s].
    chunk_size : int or None
        Binaries per chunk passed to `SNR_sq_all_binaries`. None = auto.
    target_memory_GB : float
        Memory budget for auto chunk sizing [GB]. Default 1 GB.
    use_orf : bool                                          # CHANGED: new parameter
        If True, use the binary-dependent antenna-pattern ORF.
        If False (default), use the precomputed Hellings-Downs coefficients.
        In the limit of many isotropically distributed binaries the two
        approaches should converge.

    Returns
    -------
    selected_binaries : list[dict]
        Binaries used before the target was reached (or all if not reached).
    N_needed : int
        Number of binaries required (or total count if target not reached).
    SNR_current : float
        Cumulative SNR achieved.
    SNR_sq_binaries : np.ndarray, shape (B,)
        Individual SNR² for every binary in the input list.
    """
    # Build pulsar cache once — all frequency-independent quantities live here
    pulsar_cache = build_pulsar_cache(pulsars, parsed_noise_params)

    # Compute all SNR² values in chunked vectorised batches
    SNR_sq_binaries, _ = SNR_sq_all_binaries(
        binaries         = binaries,
        pulsar_cache     = pulsar_cache,
        strain_data      = strain_data,
        T_obs            = T_obs,
        chunk_size       = chunk_size,
        target_memory_GB = target_memory_GB,
        use_orf          = use_orf,         
    )

    # Cumulative SNR after including 1, 2, … B binaries
    cumulative_SNR = np.sqrt(np.cumsum(SNR_sq_binaries))           # (B,)
    cum_SNR_sq = np.cumsum(SNR_sq_binaries)                        # (B,)
    crossing_idx_sq = np.searchsorted(cum_SNR_sq, target_SNR**2)  # index where SNR² crosses target²

    # searchsorted finds the threshold crossing in O(log B)
    crossing_idx = np.searchsorted(cumulative_SNR, target_SNR)
    print("crossing_idx_sq:", crossing_idx_sq, "crossing_idx:", crossing_idx)
    if any(cumulative_SNR < 0 for cumulative_SNR in cumulative_SNR):
        print("Warning: negative SNR values found, check noise model and SNR² computation for bugs")

    if crossing_idx < len(binaries):
        N_needed          = int(crossing_idx) + 1
        SNR_current       = float(cumulative_SNR[crossing_idx])
        selected_binaries = binaries[:crossing_idx]
        snr_sq = 0
        for i in range(len(SNR_sq_binaries)):
            snr_sq += SNR_sq_binaries[i]
            if SNR_sq_binaries[i] < 0:
                print(f"Warning: negative SNR² contribution from binary {i}, check noise model and SNR² computation for bugs")
            if i == crossing_idx:
                print(f"Crossing point at binary {i}: cumulative SNR²={snr_sq:.3e}, cumulative SNR={np.sqrt(snr_sq):.3e}, (SNR_current={SNR_current:.3e})")
        print(f"Target SNR reached with {N_needed} binaries achieving SNR={SNR_current:.3f}")
    else:
        N_needed          = len(binaries)
        SNR_current       = float(cumulative_SNR[-1])
        selected_binaries = binaries
        print(f"Target not reached with {len(binaries)} binaries")
        print(f"Max SNR reached: {SNR_current}")

    return selected_binaries, N_needed, SNR_current, SNR_sq_binaries


def convergence_test(binaries, pulsars, parsed_noise_params, strain_data, T_obs,
                     n_samples_list=None, chunk_size=None, target_memory_GB=1.0,
                     rng_seed=42):
    """
    Verify that the ORF approach converges to the HD approach as the number
    of isotropically distributed binaries grows.

    For a population of binaries drawn uniformly over the sky (isotropic),
    the average ORF² over all binaries should equal the HD chi² for each
    pulsar pair.  This function checks that the two SNR² totals converge
    as N → ∞.

    Strategy
    --------
    1. Take the first `n` binaries from `binaries` (assumed to already have
       ra/dec/psi set, or random ones are injected if missing).
    2. Compute total SNR² with use_orf=False (HD) and use_orf=True (ORF).
    3. Report the fractional difference — it should shrink toward 0 as n grows.

    Parameters
    ----------
    binaries : list[dict]
        Binary dicts.  If 'ra'/'dec' are absent they are randomly assigned
        from an isotropic distribution so the test is self-contained.
    pulsars, parsed_noise_params, strain_data, T_obs : (see N_needed_for_population)
    n_samples_list : list[int] or None
        Number of binaries to test at each step.
        Default: [10, 100, 1000, len(binaries)] (clipped to available count).
    chunk_size, target_memory_GB : passed through to the SNR² computation.
    rng_seed : int
        Random seed for injecting sky positions if needed.

    Returns
    -------
    results : list[dict]
        One entry per n_samples with keys:
            'n', 'snr_sq_hd', 'snr_sq_orf', 'frac_diff'
    """
    import copy

    rng = np.random.default_rng(rng_seed)
    B   = len(binaries)

    if n_samples_list is None:
        n_samples_list = sorted(set([
            min(10,   B),
            min(100,  B),
            min(1000, B),
            B,
        ]))

    # Inject random isotropic sky positions if any binary is missing them
    needs_sky = any('ra' not in b or 'dec' not in b for b in binaries)
    if needs_sky:
        print("convergence_test: injecting random isotropic sky positions into binaries")
        binaries = copy.deepcopy(binaries)
        for b in binaries:
            if 'ra' not in b:
                b['ra']  = rng.uniform(0.0, 2 * np.pi)
            if 'dec' not in b:
                # isotropic: dec = arcsin(uniform(-1, 1))
                b['dec'] = np.arcsin(rng.uniform(-1.0, 1.0))
            if 'psi' not in b:
                b['psi'] = rng.uniform(0.0, np.pi)

    pulsar_cache = build_pulsar_cache(pulsars, parsed_noise_params)

    results = []
    for n in n_samples_list:
        subset = binaries[:n]

        snr_sq_hd, chi_coeff_hd = SNR_sq_all_binaries(
            binaries         = subset,
            pulsar_cache     = pulsar_cache,
            strain_data      = strain_data,
            T_obs            = T_obs,
            chunk_size       = chunk_size,
            target_memory_GB = target_memory_GB,
            use_orf          = False,
        )
        snr_sq_hd = snr_sq_hd.sum()  # total SNR² across all binaries for HD

        snr_sq_orf, chi_coeff_orf = SNR_sq_all_binaries(
            binaries         = subset,
            pulsar_cache     = pulsar_cache,
            strain_data      = strain_data,
            T_obs            = T_obs,
            chunk_size       = chunk_size,
            target_memory_GB = target_memory_GB,
            use_orf          = True,
        )
        snr_sq_orf = snr_sq_orf.sum()  # total SNR² across all binaries for ORF

        print("chi_coeff_hd:", chi_coeff_hd)
        print("chi_coeff_orf:", chi_coeff_orf)
        print(f"Binary 1: ra={subset[0]['ra']:.3f}, dec={subset[0]['dec']:.3f}, psi={subset[0].get('psi', 0.0):.3f}")
        print(f"Pulsar 1: pos={pulsars[0].pos}, ra={pulsars[0]._raj}, dec={pulsars[0]._decj}")
        print(f"Pulsar 2: pos={pulsars[1].pos}, ra={pulsars[1]._raj}, dec={pulsars[1]._decj}")
        print(f"Pulsar 1 & 2 dot product: {np.dot(pulsars[0].pos, pulsars[1].pos):.3f}")
        

        frac_diff = abs(snr_sq_orf - snr_sq_hd) / (abs(snr_sq_hd) + 1e-300)
        results.append(dict(n=n, snr_sq_hd=snr_sq_hd,
                            snr_sq_orf=snr_sq_orf, frac_diff=frac_diff))
        print(
            f"  n={n:>6d}  SNR²_HD={snr_sq_hd:.4e}  "
            f"SNR²_ORF={snr_sq_orf:.4e}  frac_diff={frac_diff:.3%}"
        )

    return results

def plot_overlap_reduction_function(pulsars, binaries, parsed_noise_params):
    """
    Plot the ORF values for a few binaries to visualize how they vary with sky position.

    Parameters
    ----------
    pulsars : list[Pulsar]
        List of pulsar objects.
    binaries : list[dict]
        List of binary dicts with 'ra', 'dec', and 'psi' keys.
    pulsar_cache : dict
        Output of `build_pulsar_cache` containing the pulsar list and indices.
    """
    import matplotlib.pyplot as plt
    pulsar_cache = build_pulsar_cache(pulsars, parsed_noise_params)  # build cache to get i_idx, j_idx, and pulsars
    i_idx = pulsar_cache['i_idx']
    j_idx = pulsar_cache['j_idx']
    pulsars = pulsar_cache['pulsars']

    plt.figure(figsize=(10, 6))
    for b in binaries[:1000]:  # plot for the first 50 binaries
        orf_sq, xi_arr, orf_vals = compute_orf_sq_chunk([b], pulsars, i_idx, j_idx)  # (P,)
        plt.scatter(xi_arr * 180.0/np.pi, orf_vals, alpha=1/255)
    
    xi_list = np.linspace(0, np.pi, 1000)
    HD = 3.0 / 2.0 * (1.0 / 3.0 + 0.5 * (1 - np.cos(xi_list)) * (np.log(0.5 * (1 - np.cos(xi_list))) - 1.0 / 6.0))
    
    # xi_sorted = sorted(xi_arr)
    # for i in range(len(xi_sorted)):
    #     xi_sorted[i] *= 180.0 / np.pi
    # with np.printoptions(threshold=np.inf):
    #     print(xi_sorted)
    # HD = (1.0/3.0 - 1.0/6.0 * (0.5 * (1 - np.cos(xi_list))) + (0.5 * (1 - np.cos(xi_list))) * np.log(0.5 * (1 - np.cos(xi_list)))) * 3/2
    plt.plot(xi_list * 180.0/np.pi, HD, color="black", linestyle="--", label="Hellings-Downs")
    plt.xlabel("Pulsar Pair Angular Separation (degrees)")
    plt.ylabel("ORF")
    plt.title("Overlap Reduction Function for Binaries")
    plt.legend()
    plt.grid()
    plt.savefig("figures/orf_plot.png")
    plt.show()

# def ind_pulsar_pair_SNR(binaries, freqs, pulsar1, pulsar2, T_obs, parsed_noise_params):
#     red_PSD_pulsar1 = pulsar_red_noise_psd(freqs, parsed_noise_params[pulsar1.name]['red_noise']['log10_A'])
#     red_PSD_pulsar2 = pulsar_red_noise_psd(freqs, parsed_noise_params[pulsar2.name]['red_noise']['log10_A'])
#     # White noise: one value per pulsar, depends only on TOA errors and cadence
#     white_noise_arr_p1 = pulsar_white_noise_psd(sigma_t=np.median(pulsar1.toaerrs), delta_t=1.0 / 20.0)
#     white_noise_arr_p2 = pulsar_white_noise_psd(sigma_t=np.median(pulsar2.toaerrs), delta_t=1.0 / 20.0)

# # ------------------------------------------------------------------ unpack binary arrays
#     bin_edges    = strain_data['bin_edges']
#     freqs        = np.array([b['f']           for b in binaries])   # (B,)
#     h_contribs   = np.array([b['h_c_contrib'] for b in binaries])   # (B,)
#     chirp_masses = np.array([b['Mc']          for b in binaries])   # (B,)
#     delta_fs     = np.array([
#         bin_edges[b['freq_bin'] + 1] - bin_edges[b['freq_bin']] for b in binaries
#     ])  # (B,)

#     B = len(binaries)

#     fyr = 1.0 / (365.25 * 86400)
#     for b in range(len(binaries)):
#         signal
   
#     return SNR_sq_binaries, chi_sq_pairs


# #Load pulsars (par/tim)
# psrs = [Pulsar(par, tim) for (par, tim) in par_tim_list]

# # Set up noise params
# params = {}
# for psr in psrs:
#     name = psr.name
#     params[f"{name}_red_noise_log10_A"] = noise_file[name]["log10_A_red"]
#     params[f"{name}_red_noise_gamma"]   = noise_file[name]["gamma_red"]

#     params[f"{name}_efac"]  = noise_file[name]["efac"]
#     params[f"{name}_equad"] = noise_file[name]["equad"]
#     params[f"{name}_ecorr"] = noise_file[name]["ecorr"] 


# #Build model
# efac  = white_signals.MeasurementNoise(efac=utils.get_parameter(params, "efac"))
# equad = white_signals.EquadNoise(log10_equad=utils.get_parameter(params, "equad"))
# ecorr = white_signals.EcorrKernelNoise(log10_ecorr=utils.get_parameter(params, "ecorr"))

# .....
# #Per-pulsar intrinsic red noise
# rn_pl = utils.powerlaw(log10_A=utils.get_parameter(params, "red_noise_log10_A"),
# gamma=utils.get_parameter(params, "red_noise_gamma"))
# red_noise = gp_signals.FourierBasisGP(spectrum=rn_pl, components=30)

# #combine signal model for each pulsar
# models = []
# for psr in psrs:
# s = (efac + equad + ecorr + red_noise)
# models.append(s(psr))

# pta = signal_base.PTA(models)

# # Build covariance
# lnL = pta.get_lnlikelihood(params)


from pta_builder import build_pta_and_params
import numpy as np
from enterprise.signals.gp_bases import createfourierdesignmatrix_red
from enterprise.signals.utils import create_quantization_matrix

def pulsar_PSD_using_enterprise(psrs, raw_noise_params, parsed_noise_params, Tspan, nmodes=30, debug_pulsar_idx=0):
    pta, model, params = build_pta_and_params(psrs=psrs, noise_params_15yr=raw_noise_params, Tspan=Tspan, include_GW=True, nmodes=nmodes)
    
    fyr = 1.0 / (365.25 * 86400)

    
    pulsar_PSD_total = np.zeros((len(psrs), nmodes))
    pulsar_PSD_red   = np.zeros((len(psrs), nmodes))
    pulsar_PSD_white = np.zeros((len(psrs), nmodes))
    freqs            = np.zeros((len(psrs), nmodes))

    for i, pulsar in enumerate(psrs):
        psr_tspan = pulsar.toas.max() - pulsar.toas.min()

        _, freq_full_list = createfourierdesignmatrix_red(pulsar.toas, nmodes=nmodes, Tspan=Tspan)
        freqs[i] = freq_full_list[:nmodes]


        # ---- find signal collection ----
        sc = None
        for _sc in pta._signalcollections:
            if _sc.psrname == pulsar.name:
                sc = _sc
                break
        if sc is None:
            raise RuntimeError(f"No signal collection found for {pulsar.name}")

        rn_signal = next(sig for sig in sc._signals if sig.name == f"{pulsar.name}_red_noise")
        kappa_full  = rn_signal.get_phi(params)   # shape (nmodes,) — RN only
        # print("sc._signals keys: ", sc._signals.keys())
        # ---- red noise via get_phi ----
        # kappa_full = sc.get_phi(params)

        red_PSD = kappa_full[:nmodes] * Tspan

        # ---- white noise ----
        Nvec_raw = sc.get_ndiag(params)
        Nvec_raw = Nvec_raw._nvec

        if np.ndim(Nvec_raw) == 0:
            Nvec = np.full(len(pulsar.toas), float(Nvec_raw))
        else:
            Nvec = np.array(Nvec_raw, dtype=float)

        if Nvec.shape != (len(pulsar.toas),):
            raise ValueError(
                f"{pulsar.name}: Nvec shape {Nvec.shape} != n_toa={len(pulsar.toas)}"
            )

        U, _ = create_quantization_matrix(pulsar.toas, nmin=1)
        n_epochs  = U.shape[1]
        cadence   = psr_tspan / n_epochs

        # EFAC + EQUAD contribution (diagonal Nvec)
        epoch_variance = np.zeros(n_epochs)
        for j in range(n_epochs):
            toa_mask = U[:, j].astype(bool)
            epoch_variance[j] = 1.0 / np.sum(1.0 / Nvec[toa_mask])

        # ECORR contribution — adds ecorr² to each epoch's variance per backend
        ecorr_variance = np.zeros(n_epochs)
        psr_wn = parsed_noise_params[pulsar.name]['white_noise']
        for backend, bp in psr_wn.items():
            ecorr = 10 ** bp['log10_ecorr']
            mask  = pulsar._flags['f'] == backend
            if not np.any(mask):
                continue
            U_be, _ = create_quantization_matrix(pulsar.toas[mask], nmin=2)
            n_epochs_be = U_be.shape[1]
            # Each epoch in this backend gets ecorr² added — pad/truncate to n_epochs
            ecorr_variance[:n_epochs_be] += ecorr**2

        sigma_epoch_sq = np.median(epoch_variance + ecorr_variance)
        white_PSD = np.full(nmodes, 2.0 * sigma_epoch_sq * cadence)

        pulsar_PSD_red[i]   = red_PSD
        pulsar_PSD_white[i] = white_PSD
        pulsar_PSD_total[i] = red_PSD + white_PSD

    return pulsar_PSD_total, pulsar_PSD_red, pulsar_PSD_white, freqs



def measured_strain_all_binaries_all_pulsars_old(
    bin_arrays:  dict,          # pre-extracted binary arrays, each shape (B,)
    pulsar_cache: dict,         # output of build_pulsar_cache_time_domain
    time_arr:    np.ndarray,    # (T,) common time grid [s]
) -> np.ndarray:
    """
    Compute GW strain time series for every binary × every pulsar simultaneously.

    Follows Eq. 40 of arXiv:2512.18822:
        h(t) = F+ * h+(t) + Fx * hx(t)
        h+(t) = h0 * (1 + cos²ι) * sin(2π f t + φ₀)
        hx(t) = h0 * (−2 cosι)   * cos(2π f t + φ₀)

    Parameters
    ----------
    bin_arrays : dict
        Pre-extracted binary arrays (each shape (B,)):
        'f', 'Mc', 'D_comov', 'z', 'ra', 'dec', 'psi', 'phi0', 'iota'
    pulsar_cache : dict
        Must contain 'raj_arr' (N,), 'decj_arr' (N,).
    time_arr : ndarray, shape (T,)
        Time samples starting at 0 [s].

    Returns
    -------
    h : ndarray, shape (B, N, T)
        Strain at every binary–pulsar–time combination.
    """
    B = bin_arrays['f'].size
    N = pulsar_cache['raj_arr'].size

    # Antenna patterns — (B, N) each
    Fp, Fx = antenna_response_vectorised(
        pulsar_cache['raj_arr'],
        pulsar_cache['decj_arr'],
        bin_arrays['ra'],
        bin_arrays['dec'],
        bin_arrays['psi'],
    )  # (B, N)

    # Strain amplitude — (B,)
    h0 = strain_amplitude(
        Mc     = bin_arrays['Mc'],
        fGW    = bin_arrays['f'],
        d_comov= bin_arrays['D_comov'],
        z      = bin_arrays['z'],
    )  # (B,)

    # Polarisation amplitudes — (B,)
    cos_iota = np.cos(bin_arrays['iota'])
    A_plus   = h0 * (1.0 + cos_iota**2)   # (B,)
    A_cross  = h0 * (-2.0 * cos_iota)     # (B,)

    # Phase — (B, T):  2π f_b t_k + φ₀_b
    # bin_arrays['f'][:, None] is (B, 1), time_arr[None, :] is (1, T)
    phase = 2.0 * np.pi * bin_arrays['f'][:, None] * time_arr[None, :] \
            + bin_arrays['phi0'][:, None]            # (B, T)

    # h+ and hx — (B, T)
    hp = A_plus[:, None]  * np.sin(phase)   # (B, T)
    hx = A_cross[:, None] * np.cos(phase)   # (B, T)

    # Fourier transform the time domain h+, hx into frequency domain
    hp_f = np.fft.rfft(hp, norm="forward") # norm = forward gets 1 / N normalisation
    hx_f = np.fft.rfft(hx, norm="forward")

    time_step = time_arr[1] - time_arr[0]
    hp_size = hp.size
    freq_p = np.fft.rfftfreq(hp_size, d = time_step)

    hx_size = hx.size
    freq_x = np.fft.rfftfreq(hx_size, d = time_step)

    # Where is the actual peak bin?
    peak_idx = np.argmax(np.abs(hp_f[0] + hx_f[0]))
    print(bin_arrays['h_c_contrib'], bin_arrays['f'])

    print(f"Peak at freq: {freq_p[peak_idx]:.3e}, amplitude: {2 * np.abs(hp_f[0, peak_idx] + hx_f[0, peak_idx]):.3e}")

    # And print a few bins around it
    print(freq_p[peak_idx-2:peak_idx+3])
    print(np.abs(hp_f[0, peak_idx-2:peak_idx+3]))
    # print(freq, freq.shape, time_arr.shape, hp_size, len(phase))
    
    plt.scatter(freq_p, np.abs(hp_f[0]))
    plt.scatter(freq_x, np.abs(hx_f[0]))
    plt.xscale("log")
    plt.show()


    # print(np.linalg.norm(np.sum(hp_f + hx_f))) # seems to be different to h_c by a factor of (2 * np.pi)**2
    # print(np.linalg.norm(np.sum(hp_f + hx_f))) # seems to be different to h_c by a factor of (2 * np.pi)**2
    
    # print(bin_arrays['h_c_contrib'], bin_arrays['f'])
    # #  make plots to check that the peak matches the binary frequency
    # # freq = np.array([1e-7])
    # # time_arr = np.linspace(0, 16*86400*365.25, 10000000)
    # h0 = 1e-12
    # A_plus   = h0 * np.ones(B)
    # f = 1e-8
    # bin_arrays['f'] = f * np.ones(B)

    # phase = 2.0 * np.pi * bin_arrays['f'][:, None] * time_arr[None, :] #\ 
    #   #  + bin_arrays['phi0'][:, None]            # (B, T)
    # hp = A_plus[:, None]  * np.sin(phase)   # (B, T)
    # hp_size = hp.size
    # hp_f = np.fft.rfft(hp, axis=-1, norm="forward")
    # freq = np.fft.rfftfreq(hp.shape[-1], d = time_step)

    # # Where is the actual peak bin?
    # peak_idx = np.argmax(np.abs(hp_f[0]))

    # print(len(time_arr))
    # print(f"Peak at freq: {freq[peak_idx]:.3e}, amplitude: {np.abs(hp_f[0, peak_idx]):.3e}")
    # print(np.abs(hp_f))

    # # And print a few bins around it
    # print(freq[peak_idx-2:peak_idx+3])
    # print(np.abs(hp_f[0, peak_idx-2:peak_idx+3]))
    # # print(freq, freq.shape, time_arr.shape, hp_size, len(phase))
    
    # plt.scatter(freq, np.abs(hp_f[0]))
    # plt.xscale("log")
    # plt.show()

    
    # Contract with antenna patterns:
    # Fp is (B, N), hp is (B, T) → need (B, N, T)
    # h[b, n, t] = Fp[b,n]*hp[b,t] + Fx[b,n]*hx[b,t]
    h_f = Fp[:, :, None] * hp_f[:, None, :] \
      + Fx[:, :, None] * hx_f[:, None, :]   # (B, N, T)


    return h_f

def measured_strain_all_binaries_all_pulsars(
    bin_arrays:  dict,          # pre-extracted binary arrays, each shape (B,)
    pulsar_cache: dict,         # output of build_pulsar_cache_time_domain
    time_arr:    np.ndarray,    # (T,) common time grid [s]
    test_case:  bool=False,
    plot_first_four: bool=True
) -> np.ndarray:
    """
    Compute GW strain time series for every binary × every pulsar simultaneously.

    Follows Eq. 40 of arXiv:2512.18822:
        h(t) = F+ * h+(t) + Fx * hx(t)
        h+(t) = h0 * (1 + cos²ι) * sin(2π f t + φ₀)
        hx(t) = h0 * (−2 cosι)   * cos(2π f t + φ₀)

    Parameters
    ----------
    bin_arrays : dict
        Pre-extracted binary arrays (each shape (B,)):
        'f', 'Mc', 'D_comov', 'z', 'ra', 'dec', 'psi', 'phi0', 'iota'
    pulsar_cache : dict
        Must contain 'raj_arr' (N,), 'decj_arr' (N,).
    time_arr : ndarray, shape (T,)
        Time samples starting at 0 [s].

    Returns
    -------
    h : ndarray, shape (B, N, T)
        Strain at every binary–pulsar–time combination.
    """
    B = bin_arrays['f'].size
    N = pulsar_cache['raj_arr'].size

    # Antenna patterns — (B, N) each
    Fp, Fx = antenna_response_vectorised(
        pulsar_cache['raj_arr'],
        pulsar_cache['decj_arr'],
        bin_arrays['ra'],
        bin_arrays['dec'],
        bin_arrays['psi'],
    )  # (B, N)

    # Strain amplitude — (B,)
    h0 = strain_amplitude(
        Mc     = bin_arrays['Mc'],
        fGW    = bin_arrays['f'],
        d_comov= bin_arrays['D_comov'],
        z      = bin_arrays['z'],
    )  # (B,)

    # Polarisation amplitudes — (B,)
    cos_iota = np.cos(bin_arrays['iota'])
    A_plus   = h0 * (1.0 + cos_iota**2)   # (B,)
    A_cross  = h0 * (-2.0 * cos_iota)     # (B,)

    # Phase — (B, T):  2π f_b t_k + φ₀_b
    # bin_arrays['f'][:, None] is (B, 1), time_arr[None, :] is (1, T)
    phase = 2.0 * np.pi * bin_arrays['f'][:, None] * time_arr[None, :] \
            + bin_arrays['phi0'][:, None]            # (B, T)

    # h+ and hx — (B, T)
    hp = A_plus[:, None]  * np.sin(phase)   # (B, T)
    hx = A_cross[:, None] * np.cos(phase)   # (B, T)

    # Fourier transform the time domain h+, hx into frequency domain
    hp_f = np.fft.rfft(hp, norm="forward") # norm = forward gets 1 / N normalisation
    hx_f = np.fft.rfft(hx, norm="forward")

    time_step = time_arr[1] - time_arr[0]
    hp_time = hp.shape[1]
    freq_p = np.fft.rfftfreq(hp_time, d = time_step)

    hx_time = hx.shape[1]
    freq_x = np.fft.rfftfreq(hx_time, d = time_step)



    # Find peak index per binary (along freq axis)
    peak_idx = np.argmax(np.abs(hp_f) + np.abs(hx_f), axis=1)  # (B,)

    peak_idx = peak_idx[:, None]  # (B,1)

    h_plus = np.take_along_axis(hp_f, peak_idx, axis=1).squeeze(axis=1)
    h_cross = np.take_along_axis(hx_f, peak_idx, axis=1).squeeze(axis=1)

    # Combine with antenna patterns
    # Fp, Fx are (B, N)
    # h_plus, h_cross are (B,)
    h_f = Fp * h_plus[:, None] + Fx * h_cross[:, None]  # (B, N)

    # h_det_f = Fp[..., None] * hp_f[:, None, :] + Fx[..., None] * hx_f[:, None, :]

    # Sanity checking code
    # print(bin_arrays['h_c_contrib'], bin_arrays['f'])

    if test_case:
        freqs = np.linspace(1e-9, 20e-9, 4)
        h0_fixed = 1e-12

        colors = ["#88CCEE", "#CC6677", "#DDCC77", "#117733"]

        fig, ax = plt.subplots(figsize=(8, 5))

        for i, f in enumerate(freqs):

            bin_arrays['f'] = f * np.ones(B)
            A_plus = h0_fixed * np.ones(B)
            A_cross = h0_fixed * np.ones(B)

            phase = 2.0 * np.pi * bin_arrays['f'][:, None] * time_arr[None, :] \
                    + bin_arrays['phi0'][:, None]

            hp = A_plus[:, None] * np.sin(phase)
            hx = A_cross[:, None] * np.cos(phase)

            hp_f = np.fft.rfft(hp, norm="forward")
            hx_f = np.fft.rfft(hx, norm="forward")

            time_step = time_arr[1] - time_arr[0]
            freq_axis = np.fft.rfftfreq(hp.shape[1], d=time_step)

            ax.scatter(freq_axis, np.abs(hp_f[0]),
                       color=colors[i], marker='o', s=18)

            ax.scatter(freq_axis, np.abs(hx_f[0]),
                       color=colors[i], marker='x', s=18)

        ax.set_xscale("log")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"$|\tilde{h}_A(f)|$")

        freq_handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=colors[i], markeredgecolor=colors[i],
                   markersize=7,
                   label=f"{freqs[i]*1e9:.0f} nHz")
            for i in range(len(freqs))
        ]

        pol_handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   color='black', markersize=7, label='+'),
            Line2D([0], [0], marker='x', linestyle='None',
                   color='black', markersize=7, label='×')
        ]

        leg1 = ax.legend(handles=freq_handles, title="Frequency",
                         loc="center right")
        ax.add_artist(leg1)
        ax.legend(handles=pol_handles, title="Polarisation",
                  loc="upper right")

        plt.tight_layout()
        plt.show()

    if plot_first_four:

        colors = ["#88CCEE", "#CC6677", "#DDCC77", "#117733"]

        fig, ax = plt.subplots(figsize=(8, 5))

        for b in range(min(4, B)):

            ax.scatter(freq_p, np.abs(hp_f[b]),
                       color=colors[b], marker='o', s=18)

            ax.scatter(freq_x, np.abs(hx_f[b]),
                       color=colors[b], marker='x', s=18)

        # ax.set_xscale("log")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"$|\tilde{h}_A(f)|$")

        bin_handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=colors[i], markeredgecolor=colors[i],
                   markersize=7,
                   label=f"Binary {i}")
            for i in range(min(4, B))
        ]

        pol_handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   color='black', markersize=7, label='+'),
            Line2D([0], [0], marker='x', linestyle='None',
                   color='black', markersize=7, label='×')
        ]

        leg1 = ax.legend(handles=bin_handles, title="Binary",
                         loc="center right")
        ax.add_artist(leg1)
        ax.legend(handles=pol_handles, title="Polarisation",
                  loc="upper right")

        plt.tight_layout()
        plt.show()

    return h_f

def build_pulsar_cache_time_domain(pulsars, parsed_noise_params, raw_noise_params):
    """
    Pulsar cache for the time-domain vectorised SNR² path.

    Identical to `build_pulsar_cache` but also stores raw position arrays
    for use by `antenna_response_vectorised`.
    """
    N = len(pulsars)

    white_noise_arr = np.array([
        analytic_white_noise_psd(p, raw_noise_params)
        for p in pulsars
    ])  # (N,)
    log10A_arr = np.array([parsed_noise_params[p.name]['red_noise']['log10_A'] for p in pulsars])
    gamma_arr  = np.array([parsed_noise_params[p.name]['red_noise']['gamma']   for p in pulsars])
    raj_arr    = np.array([p._raj  for p in pulsars])
    decj_arr   = np.array([p._decj for p in pulsars])
    i_idx, j_idx = np.triu_indices(N, k=1)

    return {
        'white_noise_arr': white_noise_arr,   # (N,)
        'log10A_arr':      log10A_arr,         # (N,)
        'gamma_arr':       gamma_arr,          # (N,)
        'raj_arr':         raj_arr,            # (N,)
        'decj_arr':        decj_arr,           # (N,)
        'i_idx':           i_idx,              # (P,)
        'j_idx':           j_idx,              # (P,)
        'pulsars':         pulsars,
    }


def find_N_needed(
    binaries:            list,
    pulsars:             list,
    parsed_noise_params: dict,
    raw_noise_params: dict,
    strain_data:         dict,
    Tspan:              float,
    target_SNR:          float,
    time_arr_npoints:    int   = 1_001,
    nmodes:              int = 30,
    chunk_size:          int   = None,
    target_memory_GB:    float = 1.0,
    inc_GW:              bool  = True,
    inc_red_noise:       bool  = False,
    inc_white_noise:     bool  = False,
) -> tuple[list, int, float, np.ndarray]:
    """
    Find the minimum number of binaries needed to reach a target cumulative SNR.

    Drop-in replacement for the original `find_N_needed` using the fully
    vectorised time-domain SNR² path.
    """
    snr_sq_arr = SNR_sq_all_pairs_all_binaries_vectorised(
        binaries            = binaries,
        pulsars             = pulsars,
        parsed_noise_params = parsed_noise_params,
        raw_noise_params    = raw_noise_params,
        strain_data         = strain_data,
        Tspan               = Tspan,
        time_arr_npoints    = time_arr_npoints,
        nmodes              = 30,
        chunk_size          = chunk_size,
        target_memory_GB    = target_memory_GB,
        inc_GW              = inc_GW,
        inc_red_noise       = inc_red_noise,
        inc_white_noise     = inc_white_noise
    )

    cum_snr_sq = np.cumsum(snr_sq_arr)
    cum_snr    = np.sqrt(np.abs(cum_snr_sq))

    neg_mask = snr_sq_arr < 0
    if neg_mask.any():
        print(f"Warning: negative SNR² at binary indices {np.where(neg_mask)[0].tolist()}. "
              "Check noise model for bugs.")

    crossing_idx = int(np.searchsorted(cum_snr, target_SNR))

    if crossing_idx < len(binaries):
        N_needed          = crossing_idx + 1
        SNR_current       = float(cum_snr[crossing_idx])
        selected_binaries = binaries[:N_needed]
        print(f"Target SNR {target_SNR:.3f} reached with {N_needed} binaries "
              f"(SNR = {SNR_current:.3f}, SNR² = {cum_snr_sq[crossing_idx]:.3e})")
    else:
        N_needed          = len(binaries)
        SNR_current       = float(cum_snr[-1])
        selected_binaries = binaries
        print(f"Target SNR {target_SNR:.3f} not reached. "
              f"Max SNR = {SNR_current:.3f} with all {N_needed} binaries.")


    # For one binary and one pulsar, both should agree
    Fp_vec, Fx_vec = antenna_response_vectorised(
        np.array([pulsars[0]._raj]), np.array([pulsars[0]._decj]),
        np.array([binaries[0]['ra']]), np.array([binaries[0]['dec']]), np.array([binaries[0]['psi']])
    )
    Fp_scalar, Fx_scalar = antenna_response(pulsars[0]._raj, pulsars[0]._decj, 
                                            binaries[0]['ra'], binaries[0]['dec'], binaries[0]['psi'])
    
    return selected_binaries, N_needed, SNR_current, snr_sq_arr


def SNR_sq_all_pairs_all_binaries_vectorised(
    binaries:            list,
    pulsars:             list,
    parsed_noise_params: dict,
    raw_noise_params:    dict,
    strain_data:         dict,
    Tspan:               float,
    time_arr_npoints:    int   = 301,
    nmodes:              int   = 301,
    chunk_size:          int   = None,
    target_memory_GB:    float = 1.0,
    inc_GW:              bool  = True,
    inc_red_noise:       bool  = True,
    inc_white_noise:     bool  = True,
    noise_method:        str   = 'enterprise',   # 'analytic' | 'enterprise'
) -> np.ndarray:
    """
    Compute ρ² for every binary summed over all unique pulsar pairs.

    noise_method : str
        'analytic'   — uses pulsar_red_noise_psd + analytic_white_noise_psd
                       (fast, no ENTERPRISE PTA build for noise)
        'enterprise' — uses pulsar_PSD_using_enterprise (full PTA build,
                       exact EFAC/EQUAD/epoch-averaged white noise)
    """
    from scipy.interpolate import interp1d

    assert noise_method in ('analytic', 'enterprise'), \
        f"noise_method must be 'analytic' or 'enterprise', got {noise_method!r}"

    # ----------------------------------------------------------------
    # Pulsar cache (positions, HD coefficients, pair indices)
    # ----------------------------------------------------------------
    cache = build_pulsar_cache_time_domain(pulsars, parsed_noise_params, raw_noise_params)
    i_idx = cache['i_idx']   # (P,)
    j_idx = cache['j_idx']   # (P,)
    N     = len(pulsars)

    # ----------------------------------------------------------------
    # Noise PSD — either enterprise or analytic
    # ----------------------------------------------------------------
    if noise_method == 'enterprise':
        print("Building ENTERPRISE noise PSDs...")
        psd_total, psd_red, psd_white, freqs_per_psr = \
            pulsar_PSD_using_enterprise(pulsars, raw_noise_params, parsed_noise_params, Tspan, nmodes=nmodes)

        # Select which components to include
        psd_to_interp = np.zeros_like(psd_total)
        if inc_red_noise:
            psd_to_interp += psd_red
        if inc_white_noise:
            psd_to_interp += psd_white

        psd_interpolators = []
        f_high = 1.0  # 1 Hz — safely above any PTA binary frequency
        for a in range(N):
            # Extend frequency grid to cover all possible binary frequencies
            # White noise is flat so we can safely extrapolate it as constant
            # by appending a point at high frequency with the white noise value
            f_grid   = np.append(freqs_per_psr[a], f_high)
            psd_grid = np.append(psd_to_interp[a], psd_white[a, -1] if inc_white_noise else psd_to_interp[a, -1])

            interp_fn = interp1d(
                np.log(f_grid),
                np.log(np.clip(psd_grid, 1e-300, None)),
                kind='linear',
                bounds_error=False,
                fill_value=(np.log(np.clip(psd_grid[0], 1e-300, None)),   # below range: use lowest
                            np.log(np.clip(psd_grid[-1], 1e-300, None))), # above range: use highest (white noise)
            )
            psd_interpolators.append(interp_fn)
        print("ENTERPRISE noise PSDs ready.")

    # ----------------------------------------------------------------
    # Binary arrays
    # ----------------------------------------------------------------
    B = len(binaries)
    bin_arrays = {
        'f':           np.array([b['f']                     for b in binaries]),
        'Mc':          np.array([b['Mc']                    for b in binaries]),
        'D_comov':     np.array([b['D_comov']               for b in binaries]),
        'z':           np.array([b['z']                     for b in binaries]),
        'ra':          np.array([b['ra']                    for b in binaries]),
        'dec':         np.array([b['dec']                   for b in binaries]),
        'psi':         np.array([b.get('psi',  0.0)         for b in binaries]),
        'phi0':        np.array([b.get('phi0', 0.0)         for b in binaries]),
        'iota':        np.array([b.get('iota', 0.0)         for b in binaries]),
        'h_c_contrib': np.array([b.get('h_c_contrib', 0.0)  for b in binaries]),
    }

    bin_edges = strain_data['bin_edges']
    delta_fs  = np.array([
        bin_edges[b['freq_bin'] + 1] - bin_edges[b['freq_bin']]
        for b in binaries
    ])

    time_arr_max = np.max(1.0 / bin_edges)
    time_arr_min = np.min(1.0 / bin_edges)
    time_arr     = np.linspace(time_arr_min, time_arr_max, time_arr_npoints)

    # ----------------------------------------------------------------
    # Per-pair Tspan
    # ----------------------------------------------------------------
    psr_tspans  = np.array([p.toas.max() - p.toas.min() for p in pulsars])
    tspan_pairs = np.minimum(psr_tspans[i_idx], psr_tspans[j_idx])
    unique_tspans, pair_group_ids = np.unique(tspan_pairs, return_inverse=True)

    # ----------------------------------------------------------------
    # Chunk sizing
    # ----------------------------------------------------------------
    T = time_arr_npoints
    if chunk_size is None:
        bytes_per_binary = N * T * 8 * 6
        chunk_size = max(1, int(target_memory_GB * 1024**3 / bytes_per_binary))
        print(f"Auto chunk size: {chunk_size} binaries (N={N}, T={T}, "
              f"target={target_memory_GB} GB)")

    snr_sq_arr = np.zeros(B)

    # ----------------------------------------------------------------
    # Main chunk loop
    # ----------------------------------------------------------------
    for start in range(0, B, chunk_size):
        end   = min(start + chunk_size, B)
        chunk = slice(start, end)
        Bc    = end - start
        print(f"  Chunk binaries {start}–{end-1}")

        ba     = {k: v[chunk] for k, v in bin_arrays.items()}
        f_bin  = ba['f']   # (Bc,)

        # ---- noise PSD matrix (Bc, N) --------------------------------
        if noise_method == 'enterprise':
            P_noise = np.zeros((Bc, N))
            for a in range(N):
                log_P         = psd_interpolators[a](np.log(f_bin))
                P_noise[:, a] = np.exp(log_P)

        else:  # analytic
            fyr   = 1.0 / (365.25 * 86400)
            P_noise = np.zeros((Bc, N))
            if inc_red_noise:
                A_red = 10.0 ** cache['log10A_arr']          # (N,)
                rn = (
                    A_red**2 / (12.0 * np.pi**2)
                    * (f_bin[:, None] / fyr) ** (-cache['gamma_arr'])
                    * fyr**-3.0
                )  # (Bc, N)
                P_noise += rn
            if inc_white_noise:
                P_noise += cache['white_noise_arr']           # (Bc, N) broadcast

        # ---- SNR² over Tspan groups ----------------------------------
        snr_sq_chunk = np.zeros(Bc)

        for g, Tspan_g in enumerate(unique_tspans):
            pair_mask = pair_group_ids == g
            gi        = i_idx[pair_mask]
            gj        = j_idx[pair_mask]
            Pg        = pair_mask.sum()
            if Pg == 0:
                continue

            h_f      = measured_strain_all_binaries_all_pulsars(ba, cache, time_arr)
            h_f_conj = h_f.conjugate()

            norm  = 1.0 / (12.0 * np.pi**2 * f_bin**3)   # (Bc,)

            Sh_ii = h_f[:, gi] * h_f_conj[:, gi] * norm[:, None]
            Sh_jj = h_f[:, gj] * h_f_conj[:, gj] * norm[:, None]
            Sh_ij = 0.5 * (
                h_f[:, gi] * h_f_conj[:, gj] +
                h_f_conj[:, gi] * h_f[:, gj]
            ) * norm[:, None]


            # # NEW METHOD: use Tspan_g normalisation instead of f_bin**-3
            # norm = 2 / Tspan_g  # (Bc,) — includes the 2 from the one-sided PSD
            # Sh_ii = h_f[:, gi] * h_f_conj[:, gi] * norm
            # Sh_jj = h_f[:, gj] * h_f_conj[:, gj] * norm
            # Sh_ij = h_f[:, gi] * h_f_conj[:, gj] * norm
            # Sh_ij = Sh_ij.real

            Ni = P_noise[:, gi] + Sh_ii
            Nj = P_noise[:, gj] + Sh_jj

            integrand = 2.0 * Tspan_g * Sh_ij**2 / (Ni * Nj)

            # Sanity check: integrand should be real
            real_imag_ratio = np.abs(np.real(integrand)) / (np.abs(np.imag(integrand)) + 1e-300)
            if np.any(real_imag_ratio < 1e10):
                print(f"WARNING: non-negligible imaginary component, "
                      f"min |Re/Im| = {real_imag_ratio.min():.3e}")

            snr_sq_chunk += delta_fs[chunk] * np.real(integrand).sum(axis=1)

        snr_sq_arr[chunk] = snr_sq_chunk

    return snr_sq_arr


def build_pulsar_cache_time_domain(pulsars, parsed_noise_params, raw_noise_params):
    """
    Pulsar cache for the time-domain vectorised SNR² path.

    Identical to `build_pulsar_cache` but also stores raw position arrays
    for use by `antenna_response_vectorised`.
    """
    N = len(pulsars)

    white_noise_arr = np.array([
        analytic_white_noise_psd(p, raw_noise_params)
        for p in pulsars
    ])  # (N,)
    log10A_arr = np.array([parsed_noise_params[p.name]['red_noise']['log10_A'] for p in pulsars])
    gamma_arr  = np.array([parsed_noise_params[p.name]['red_noise']['gamma']   for p in pulsars])
    raj_arr    = np.array([p._raj  for p in pulsars])
    decj_arr   = np.array([p._decj for p in pulsars])
    i_idx, j_idx = np.triu_indices(N, k=1)

    return {
        'white_noise_arr': white_noise_arr,   # (N,)
        'log10A_arr':      log10A_arr,         # (N,)
        'gamma_arr':       gamma_arr,          # (N,)
        'raj_arr':         raj_arr,            # (N,)
        'decj_arr':        decj_arr,           # (N,)
        'i_idx':           i_idx,              # (P,)
        'j_idx':           j_idx,              # (P,)
        'pulsars':         pulsars,
    }


def find_N_needed(
    binaries:            list,
    pulsars:             list,
    parsed_noise_params: dict,
    raw_noise_params:    dict, 
    strain_data:         dict,
    Tspan:               float,
    target_SNR:          float,
    time_arr_npoints:    int   = 301, # gets to 3e-7 Hz
    chunk_size:          int   = None,
    target_memory_GB:    float = 1.0,
    inc_GW:              bool  = True,
    inc_red_noise:       bool  = True,
    inc_white_noise:     bool  = True,
    noise_method:        str   = 'enterprise',   # 'analytic' | 'enterprise'
) -> tuple[list, int, float, np.ndarray]:
    """
    Find the minimum number of binaries needed to reach a target cumulative SNR.

    Drop-in replacement for the original `find_N_needed` using the fully
    vectorised time-domain SNR² path.
    """
    snr_sq_arr = SNR_sq_all_pairs_all_binaries_vectorised(
        binaries            = binaries,
        pulsars             = pulsars,
        parsed_noise_params = parsed_noise_params,
        raw_noise_params    = raw_noise_params,
        strain_data         = strain_data,
        Tspan               = Tspan,
        time_arr_npoints    = time_arr_npoints,
        chunk_size          = chunk_size,
        target_memory_GB    = target_memory_GB,
        inc_GW              = inc_GW,
        inc_red_noise       = inc_red_noise,
        inc_white_noise     = inc_white_noise,
        noise_method        = noise_method,
    )

    cum_snr_sq = np.cumsum(snr_sq_arr)
    cum_snr    = np.sqrt(np.abs(cum_snr_sq))

    neg_mask = snr_sq_arr < 0
    if neg_mask.any():
        print(f"Warning: negative SNR² at binary indices {np.where(neg_mask)[0].tolist()}. "
              "Check noise model for bugs.")

    crossing_idx = int(np.searchsorted(cum_snr, target_SNR))

    if crossing_idx < len(binaries):
        N_needed          = crossing_idx + 1
        SNR_current       = float(cum_snr[crossing_idx])
        selected_binaries = binaries[:N_needed]
        print(f"Target SNR {target_SNR:.3f} reached with {N_needed} binaries "
              f"(SNR = {SNR_current:.3f}, SNR² = {cum_snr_sq[crossing_idx]:.3e})")
    else:
        N_needed          = len(binaries)
        SNR_current       = float(cum_snr[-1])
        selected_binaries = binaries
        print(f"Target SNR {target_SNR:.3f} not reached. "
              f"Max SNR = {SNR_current:.3f} with all {N_needed} binaries.")


    # For one binary and one pulsar, both should agree
    Fp_vec, Fx_vec = antenna_response_vectorised(
        np.array([pulsars[0]._raj]), np.array([pulsars[0]._decj]),
        np.array([binaries[0]['ra']]), np.array([binaries[0]['dec']]), np.array([binaries[0]['psi']])
    )
    Fp_scalar, Fx_scalar = antenna_response(pulsars[0]._raj, pulsars[0]._decj, 
                                            binaries[0]['ra'], binaries[0]['dec'], binaries[0]['psi'])
    
    return selected_binaries, N_needed, SNR_current, snr_sq_arr


# comparison for pulsar PSDs
def compare_pulsar_psd_methods(
    psrs,
    raw_noise_params,
    parsed_noise_params,
    Tspan,
    nmodes=30,
    freq_array=None,
    rtol=0.5,
    verbose=True,
):
    """
    Compare PSD estimates from two methods:
      1. `pulsar_PSD_using_enterprise` — full ENTERPRISE-based calculation
         (per-pulsar Tspan, quantisation matrix, epoch-averaged white noise)
      2. `build_pulsar_cache_time_domain` + analytic PSD functions
         (median TOA error, fixed 20-obs/yr cadence, power-law red noise)

    Parameters
    ----------
    psrs : list
        Enterprise Pulsar objects.
    raw_noise_params : dict
        Noise parameter dict keyed by pulsar name.
    Tspan : float
        Common timespan [s] passed to the ENTERPRISE PTA builder.
    nmodes : int
        Number of Fourier modes to evaluate.
    freq_array : np.ndarray or None
        Frequencies [Hz] at which to evaluate the analytic PSDs.
        If None, the per-pulsar frequency grids from ENTERPRISE are used
        (one comparison per pulsar at its own grid).
    rtol : float
        Relative tolerance used to flag large discrepancies (default 50 %).
    verbose : bool
        Print a per-pulsar summary table.

    Returns
    -------
    results : dict
        Keys: pulsar name → dict with
          'freqs'              : (nmodes,) Hz
          'enterprise_red'     : (nmodes,) s³
          'enterprise_white'   : (nmodes,) s³  (scalar repeated)
          'enterprise_total'   : (nmodes,) s³
          'analytic_red'       : (nmodes,) s³
          'analytic_white'     : (nmodes,) s³  (scalar repeated)
          'analytic_total'     : (nmodes,) s³
          'red_reldiff'        : (nmodes,) fractional difference
          'white_reldiff'      : (nmodes,) fractional difference
          'total_reldiff'      : (nmodes,) fractional difference
          'max_red_reldiff'    : float
          'max_white_reldiff'  : float
          'max_total_reldiff'  : float
          'pass'               : bool  (all components within rtol)
    """
    # ------------------------------------------------------------------ #
    # 1.  ENTERPRISE-based PSDs                                           #
    # ------------------------------------------------------------------ #

    (
        ent_total,   # (N, nmodes)
        ent_red,     # (N, nmodes)
        ent_white,   # (N, nmodes)
        ent_freqs,   # (N, nmodes)
    ) = pulsar_PSD_using_enterprise(psrs, raw_noise_params, parsed_noise_params, Tspan, nmodes=nmodes)

    # ------------------------------------------------------------------ #
    # 2.  Analytic PSDs via the cache + helper functions                  #
    # ------------------------------------------------------------------ #
    cache = build_pulsar_cache_time_domain(psrs, parsed_noise_params, raw_noise_params)

    results = {}

    for i, psr in enumerate(psrs):
        name = psr.name

        # Frequency grid: prefer caller-supplied, fall back to ENTERPRISE grid
        freqs = freq_array if freq_array is not None else ent_freqs[i]  # (nmodes,)

        # --- red noise ---
        an_red = pulsar_red_noise_psd(
            freq=freqs,
            log10A_red=cache['log10A_arr'][i],
            gamma_red=cache['gamma_arr'][i],
        )  # (nmodes,)

        # --- white noise ---
        sigma_t  = np.median(psr.toaerrs)          # [s]
        cadence  = 1.0 / 20.0                       # [yr]  (matches cache builder)
        # an_white = pulsar_white_noise_psd(sigma_t=sigma_t, delta_t=cadence)
        an_white = pulsar_white_noise_psd(
            sigma_t=effective_toaerrs(psrs[i], parsed_noise_params),
            delta_t=actual_cadence_yr(psrs[i])
            )
        an_white = analytic_white_noise_psd(psr, raw_noise_params)
        an_white = np.full(len(freqs), an_white)    # broadcast to (nmodes,)

        an_total = an_red + an_white

        # --- ENTERPRISE values at same frequencies ---
        # ent_freqs[i] may differ from `freqs` if caller supplied freq_array;
        # in that case interpolate the ENTERPRISE values for a fair comparison.
        if freq_array is not None:
            e_red   = np.interp(freqs, ent_freqs[i], ent_red[i])
            e_white = np.interp(freqs, ent_freqs[i], ent_white[i])
            e_total = np.interp(freqs, ent_freqs[i], ent_total[i])
        else:
            e_red   = ent_red[i]
            e_white = ent_white[i]
            e_total = ent_total[i]

        # --- relative differences  |analytic - enterprise| / enterprise ---
        def _reldiff(a, b):
            denom = np.where(np.abs(b) > 0, np.abs(b), np.abs(a) + 1e-300)
            return np.abs(a - b) / denom

        rd_red   = _reldiff(an_red,   e_red)
        rd_white = _reldiff(an_white, e_white)
        rd_total = _reldiff(an_total, e_total)

        passed = (
            np.max(rd_red)   < rtol and
            np.max(rd_white) < rtol and
            np.max(rd_total) < rtol
        )

        results[name] = dict(
            freqs            = freqs,
            enterprise_red   = e_red,
            enterprise_white = e_white,
            enterprise_total = e_total,
            analytic_red     = an_red,
            analytic_white   = an_white,
            analytic_total   = an_total,
            red_reldiff      = rd_red,
            white_reldiff    = rd_white,
            total_reldiff    = rd_total,
            max_red_reldiff  = float(np.max(rd_red)),
            max_white_reldiff= float(np.max(rd_white)),
            max_total_reldiff= float(np.max(rd_total)),
            passed           = passed,
        )

    # ------------------------------------------------------------------ #
    # 3.  Optional console summary                                        #
    # ------------------------------------------------------------------ #
    if verbose:
        header = (
            f"{'Pulsar':<20} {'MaxΔred':>10} {'MaxΔwhite':>11} "
            f"{'MaxΔtotal':>11}  {'Pass?':>6}"
        )
        print(header)
        print("-" * len(header))
        for name, r in results.items():
            status = "✓" if r['passed'] else "✗"
            print(
                f"{name:<20} "
                f"{r['max_red_reldiff']:>10.3f} "
                f"{r['max_white_reldiff']:>11.3f} "
                f"{r['max_total_reldiff']:>11.3f}  "
                f"{status:>6}"
            )
        n_pass = sum(r['passed'] for r in results.values())
        print(f"\n{n_pass}/{len(results)} pulsars within rtol={rtol:.0%}")

    return results

def plot_psd_comparison(results, pulsar_name=None, figsize=(14, 10)):
    """
    Plot PSD comparison between ENTERPRISE and analytic methods for a single pulsar.

    Parameters
    ----------
    results : dict
        Output from `compare_pulsar_psd_methods`.
    pulsar_name : str or None
        Pulsar to plot. Defaults to the first pulsar in results.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if pulsar_name is None:
        pulsar_name = next(iter(results))
    r = results[pulsar_name]

    freqs = r['freqs']
    fyr   = 1.0 / (365.25 * 86400)
    f_norm = freqs / fyr  # in units of f/fyr for x-axis

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"PSD Comparison — {pulsar_name}", fontsize=14, fontweight='bold')

    # ── colour / style palette ──────────────────────────────────────────
    C = dict(
        ent_total   = '#1f77b4',
        ent_red     = '#d62728',
        ent_white   = '#ff7f0e',
        an_total    = '#1f77b4',
        an_red      = '#d62728',
        an_white    = '#ff7f0e',
    )

    # ================================================================== #
    # Panel 1 (top-left): absolute PSDs — ENTERPRISE                     #
    # ================================================================== #
    ax = axes[0, 0]
    ax.loglog(f_norm, r['enterprise_total'], color=C['ent_total'],
              lw=2,   label='Total')
    ax.loglog(f_norm, r['enterprise_red'],   color=C['ent_red'],
              lw=1.5, ls='--', label='Red noise')
    ax.loglog(f_norm, r['enterprise_white'], color=C['ent_white'],
              lw=1.5, ls=':',  label='White noise')
    ax.set_title('ENTERPRISE PSDs', fontsize=11)
    ax.set_xlabel(r'$f\,/\,f_\mathrm{yr}$')
    ax.set_ylabel(r'PSD  [s$^3$]')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # ================================================================== #
    # Panel 2 (top-right): absolute PSDs — Analytic                      #
    # ================================================================== #
    ax = axes[0, 1]
    ax.loglog(f_norm, r['analytic_total'], color=C['an_total'],
              lw=2,   label='Total')
    ax.loglog(f_norm, r['analytic_red'],   color=C['an_red'],
              lw=1.5, ls='--', label='Red noise')
    ax.loglog(f_norm, r['analytic_white'], color=C['an_white'],
              lw=1.5, ls=':',  label='White noise')
    ax.set_title('Analytic PSDs', fontsize=11)
    ax.set_xlabel(r'$f\,/\,f_\mathrm{yr}$')
    ax.set_ylabel(r'PSD  [s$^3$]')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # ================================================================== #
    # Panel 3 (bottom-left): relative contributions (red / total)        #
    # ================================================================== #
    ax = axes[1, 0]

    ent_red_frac   = r['enterprise_red']   / r['enterprise_total']
    ent_white_frac = r['enterprise_white'] / r['enterprise_total']
    an_red_frac    = r['analytic_red']     / r['analytic_total']
    an_white_frac  = r['analytic_white']   / r['analytic_total']

    ax.semilogx(f_norm, ent_red_frac,   color=C['ent_red'],
                lw=2,   label='ENT red / total')
    ax.semilogx(f_norm, ent_white_frac, color=C['ent_white'],
                lw=2,   label='ENT white / total')
    ax.semilogx(f_norm, an_red_frac,    color=C['an_red'],
                lw=1.5, ls='--', label='ANA red / total')
    ax.semilogx(f_norm, an_white_frac,  color=C['an_white'],
                lw=1.5, ls='--', label='ANA white / total')
    ax.axhline(0.5, color='grey', lw=0.8, ls=':')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Relative Contributions  (component / total)', fontsize=11)
    ax.set_xlabel(r'$f\,/\,f_\mathrm{yr}$')
    ax.set_ylabel('Fraction')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    # ================================================================== #
    # Panel 4 (bottom-right): fractional differences                     #
    # ================================================================== #
    ax = axes[1, 1]
    ax.semilogx(f_norm, r['red_reldiff']   * 100, color=C['ent_red'],
                lw=2,   label='Red noise')
    ax.semilogx(f_norm, r['white_reldiff'] * 100, color=C['ent_white'],
                lw=2,   label='White noise')
    ax.semilogx(f_norm, r['total_reldiff'] * 100, color=C['ent_total'],
                lw=2,   ls='--', label='Total')

    # tolerance band
    rtol_pct = 50.0
    ax.axhline(rtol_pct, color='grey', lw=1, ls=':', label=f'{rtol_pct:.0f}% tolerance')

    ax.set_title('Fractional Difference  |analytic − ENT| / ENT', fontsize=11)
    ax.set_xlabel(r'$f\,/\,f_\mathrm{yr}$')
    ax.set_ylabel('Relative difference [%]')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    return fig


def white_noise_residual_efac_only(pulsar, parsed_noise_params):
    """EFAC + EQUAD only — no ECORR — matches what ENTERPRISE get_ndiag returns."""
    wn_params   = parsed_noise_params[pulsar.name]['white_noise']
    sel         = selections.Selection(selections.by_backend)
    params_dict = {}
    for backend, bp in wn_params.items():
        params_dict[f'{pulsar.name}_{backend}_efac']          = bp['efac']
        params_dict[f'{pulsar.name}_{backend}_log10_t2equad'] = bp['log10_t2equad']

    efac_p    = parameter.Constant()
    equad_p   = parameter.Constant()
    mn        = white_signals.MeasurementNoise(efac=efac_p, log10_t2equad=equad_p, selection=sel)
    mn_signal = mn(pulsar)
    Nvec      = mn_signal.get_ndiag(params_dict)
    return np.sqrt(Nvec) * np.random.randn(len(pulsar.toas))


def test_psd_vs_residuals_consistency(
    psrs,
    parsed_noise_params,
    raw_noise_params,
    Tspan,
    nmodes           = 30,
    n_realisations   = 500,
    test_pulsar_idx  = 0,
    rtol_red         = 0.1,
    rtol_white       = 0.3,
    cond_threshold   = 1e10,
    plot             = True,
):
    import matplotlib.pyplot as plt

    psr = psrs[test_pulsar_idx]
    fyr = 1.0 / (365.25 * 86400)

    print(f"\n{'='*60}")
    print(f"PSD vs Residuals consistency test")
    print(f"Pulsar: {psr.name}  (idx {test_pulsar_idx})")
    print(f"n_realisations: {n_realisations},  nmodes: {nmodes}")
    print(f"NOTE: ENTERPRISE get_ndiag is missing ECORR (off-diagonal term)")
    print(f"      Fair comparison is empirical (EFAC only, epoch-averaged) vs ENTERPRISE.")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # 1. ENTERPRISE analytic PSDs
    # ----------------------------------------------------------------
    print("\nComputing ENTERPRISE analytic PSDs...")
    psd_total, psd_red, psd_white, freqs_per_psr = pulsar_PSD_using_enterprise(
        psrs, raw_noise_params, parsed_noise_params, Tspan, nmodes=nmodes
    )
    ent_red   = psd_red[test_pulsar_idx]
    ent_white = psd_white[test_pulsar_idx]
    freqs     = freqs_per_psr[test_pulsar_idx]

    print(f"  ENTERPRISE red   PSD range: [{ent_red.min():.3e}, {ent_red.max():.3e}] s³")
    print(f"  ENTERPRISE white PSD range: [{ent_white.min():.3e}, {ent_white.max():.3e}] s³")

    # ----------------------------------------------------------------
    # 2. Fourier basis + condition number
    # ----------------------------------------------------------------
    log10_A   = parsed_noise_params[psr.name]['red_noise']['log10_A']
    gamma     = parsed_noise_params[psr.name]['red_noise']['gamma']

    F, Ffreqs = createfourierdesignmatrix_red(psr.toas, nmodes=nmodes, Tspan=Tspan)
    FtF       = F.T @ F
    cond      = np.linalg.cond(FtF)
    white_use_time_domain = cond > cond_threshold

    print(f"\n  F^T F condition number: {cond:.3e}")
    if white_use_time_domain:
        print(f"  Poorly conditioned (> {cond_threshold:.0e}) "
              f"— using epoch-space time-domain variance for white noise")
    else:
        print(f"  Well conditioned — using Fourier projection for white noise")

    # ----------------------------------------------------------------
    # 3. Epoch structure + Nvec — computed once, reused throughout
    # ----------------------------------------------------------------
    psr_tspan  = psr.toas.max() - psr.toas.min()
    U, _       = create_quantization_matrix(psr.toas, nmin=1)
    n_epochs   = U.shape[1]
    cadence    = psr_tspan / n_epochs

    # Build Nvec from EFAC+EQUAD
    wn_params   = parsed_noise_params[psr.name]['white_noise']
    params_dict = {}
    for backend, bp in wn_params.items():
        params_dict[f'{psr.name}_{backend}_efac']          = bp['efac']
        params_dict[f'{psr.name}_{backend}_log10_t2equad'] = bp['log10_t2equad']
    sel       = selections.Selection(selections.by_backend)
    efac_p    = parameter.Constant()
    equad_p   = parameter.Constant()
    mn        = white_signals.MeasurementNoise(efac=efac_p, log10_t2equad=equad_p, selection=sel)
    mn_signal = mn(psr)
    Nvec_diag = mn_signal.get_ndiag(params_dict)

    # Epoch-averaged variance — exactly what ENTERPRISE computes
    epoch_variance_diag = np.zeros(n_epochs)
    for j in range(n_epochs):
        toa_mask = U[:, j].astype(bool)
        epoch_variance_diag[j] = 1.0 / np.sum(1.0 / Nvec_diag[toa_mask])
    sigma_epoch_sq = np.median(epoch_variance_diag)

    # Bandwidth in epoch space
    nyquist_bw_epoch = n_epochs / (2.0 * psr_tspan)
    fourier_bw       = nmodes / Tspan
    bw_ratio_epoch   = fourier_bw / nyquist_bw_epoch

    print(f"\n  --- WHITE NOISE DIAGNOSTICS ---")
    print(f"    n_toas                    = {len(psr.toas)}")
    print(f"    n_epochs                  = {n_epochs}")
    print(f"    avg TOAs/epoch            = {len(psr.toas)/n_epochs:.1f}")
    print(f"    cadence                   = {cadence:.3e} s")
    print(f"    nyquist_bw (epoch)        = {nyquist_bw_epoch:.3e} Hz")
    print(f"    fourier_bw                = {fourier_bw:.3e} Hz")
    print(f"    bw_ratio (epoch)          = {bw_ratio_epoch:.6f}")
    print(f"    median(Nvec)              = {np.median(Nvec_diag):.3e} s²")
    print(f"    sigma_epoch_sq            = {sigma_epoch_sq:.3e} s²  (epoch-averaged)")
    print(f"    median(toaerrs^2)         = {np.median(psr.toaerrs**2):.3e} s²")
    print(f"    ratio Nvec/toaerrs^2      = {np.median(Nvec_diag/psr.toaerrs**2):.3f}  (expect ~EFAC²~1)")
    print(f"    sqrt(Nvec) range          = [{np.sqrt(Nvec_diag.min())*1e9:.1f}, "
          f"{np.sqrt(Nvec_diag.max())*1e9:.1f}] ns")
    print(f"    toaerrs range             = [{psr.toaerrs.min()*1e9:.1f}, "
          f"{psr.toaerrs.max()*1e9:.1f}] ns")
    print(f"    implied PSD (per-TOA)     = {2 * np.median(Nvec_diag) * cadence:.3e} s³")
    print(f"    implied PSD (epoch-avg)   = {2 * sigma_epoch_sq * cadence:.3e} s³")
    print(f"    ent_white[0]              = {ent_white[0]:.3e} s³")
    print(f"    ratio per-TOA/ent         = {2 * np.median(Nvec_diag) * cadence / ent_white[0]:.3f}  (expect >> 1)")
    print(f"    ratio epoch-avg/ent       = {2 * sigma_epoch_sq * cadence / ent_white[0]:.3f}  (expect ~1)")

    # ----------------------------------------------------------------
    # 4. Kappa for red noise draws
    # ----------------------------------------------------------------
    kappa = (
        (10**log10_A)**2 / (12 * np.pi**2)
        * (Ffreqs / fyr)**(-gamma)
        * fyr**-3
        * (1.0 / Tspan)
    )

    # ----------------------------------------------------------------
    # 5. Draw realisations
    # ----------------------------------------------------------------
    red_coeffs_all             = np.zeros((n_realisations, 2 * nmodes))
    white_epoch_var_full_all   = np.zeros(n_realisations)
    white_epoch_var_efac_all   = np.zeros(n_realisations)
    white_coeffs_full_all      = np.zeros((n_realisations, 2 * nmodes))
    white_coeffs_efac_only_all = np.zeros((n_realisations, 2 * nmodes))

    print(f"\nDrawing {n_realisations} realisations...")

    for k in range(n_realisations):
        # Red noise
        a_red = np.sqrt(kappa) * np.random.randn(2 * nmodes)
        red_coeffs_all[k] = a_red

        # White noise — draw directly in epoch space from epoch_variance_diag
        # This exactly mirrors what ENTERPRISE does: one effective measurement
        # per epoch with variance = epoch_variance_diag[j]
        epoch_r_efac = np.sqrt(epoch_variance_diag) * np.random.randn(n_epochs)
        white_epoch_var_efac_all[k] = np.var(epoch_r_efac)

        # Full white noise — epoch variance including ECORR
        # Add ecorr² per epoch on top of EFAC+EQUAD epoch variance
        epoch_variance_full = epoch_variance_diag.copy()
        for backend, bp in wn_params.items():
            ecorr = 10 ** bp['log10_ecorr']
            mask  = psr._flags['f'] == backend
            if not np.any(mask):
                continue
            U_be, _ = create_quantization_matrix(psr.toas[mask], nmin=2)
            n_ep_be = U_be.shape[1]
            epoch_variance_full[:n_ep_be] += ecorr**2

        epoch_r_full = np.sqrt(epoch_variance_full) * np.random.randn(n_epochs)
        white_epoch_var_full_all[k] = np.var(epoch_r_full)

        if not white_use_time_domain:
            # For Fourier path, still need TOA-space residuals
            r_white_full = white_noise_residual(psr, parsed_noise_params)
            r_white_efac = white_noise_residual_efac_only(psr, parsed_noise_params)
            white_coeffs_full_all[k]      = np.linalg.solve(FtF, F.T @ r_white_full)
            white_coeffs_efac_only_all[k] = np.linalg.solve(FtF, F.T @ r_white_efac)

    # ----------------------------------------------------------------
    # 6. Empirical PSDs
    # ----------------------------------------------------------------

    # Red noise
    red_kappa_empirical = np.array([
        np.mean(0.5 * (red_coeffs_all[:, 2*m]**2 + red_coeffs_all[:, 2*m+1]**2))
        for m in range(nmodes)
    ])
    empirical_red_psd = red_kappa_empirical * Tspan
    rd_red = np.abs(empirical_red_psd - ent_red) / np.where(
        np.abs(ent_red) > 0, np.abs(ent_red), 1e-300
    )

    # White noise
    if white_use_time_domain:
        enterprise_white_var       = ent_white[0] * fourier_bw
        var_full_scaled            = np.median(white_epoch_var_full_all) * bw_ratio_epoch
        var_efac_only_scaled       = np.median(white_epoch_var_efac_all) * bw_ratio_epoch
        ecorr_fraction             = (
            np.median(white_epoch_var_full_all) - np.median(white_epoch_var_efac_all)
        ) / np.median(white_epoch_var_full_all)

        reldiff_full      = abs(var_full_scaled      - enterprise_white_var) / enterprise_white_var
        reldiff_efac_only = abs(var_efac_only_scaled - enterprise_white_var) / enterprise_white_var

        rd_white            = np.full(nmodes, reldiff_efac_only)
        empirical_white_psd = None

        print(f"\n  White noise (epoch-space, bandwidth-corrected):")
        print(f"    Raw epoch var (full)        = {np.median(white_epoch_var_full_all):.3e} s²")
        print(f"    Raw epoch var (EFAC only)   = {np.median(white_epoch_var_efac_all):.3e} s²")
        print(f"    Scaled var (full)           = {var_full_scaled:.3e} s²")
        print(f"    Scaled var (EFAC only)      = {var_efac_only_scaled:.3e} s²")
        print(f"    ENTERPRISE var              = {enterprise_white_var:.3e} s²")
        print(f"    ECORR fraction of var       = {ecorr_fraction*100:.1f}%")
        print(f"    Rel diff (full vs ENT)      = {reldiff_full*100:.2f}%")
        print(f"    Rel diff (EFAC only vs ENT) = {reldiff_efac_only*100:.2f}%  (fair comparison)")

    else:
        def fourier_psd(coeffs):
            return np.array([
                np.mean(0.5 * (coeffs[:, 2*m]**2 + coeffs[:, 2*m+1]**2))
                for m in range(nmodes)
            ]) * Tspan

        empirical_white_psd_full      = fourier_psd(white_coeffs_full_all)
        empirical_white_psd_efac_only = fourier_psd(white_coeffs_efac_only_all)
        empirical_white_psd           = empirical_white_psd_efac_only

        rd_white_full = np.abs(empirical_white_psd_full - ent_white) / np.where(
            np.abs(ent_white) > 0, np.abs(ent_white), 1e-300
        )
        rd_white = np.abs(empirical_white_psd_efac_only - ent_white) / np.where(
            np.abs(ent_white) > 0, np.abs(ent_white), 1e-300
        )

        print(f"\n  White noise (Fourier projection):")
        print(f"    Empirical PSD full      range: [{empirical_white_psd_full.min():.3e}, "
              f"{empirical_white_psd_full.max():.3e}] s³")
        print(f"    Empirical PSD EFAC only range: [{empirical_white_psd_efac_only.min():.3e}, "
              f"{empirical_white_psd_efac_only.max():.3e}] s³")
        print(f"    ENTERPRISE PSD          range: [{ent_white.min():.3e}, "
              f"{ent_white.max():.3e}] s³")
        print(f"    Max rel diff (full)           = {rd_white_full.max()*100:.2f}%")
        print(f"    Max rel diff (EFAC only)      = {rd_white.max()*100:.2f}%  (fair comparison)")

    # ----------------------------------------------------------------
    # 7. Pass / fail
    # ----------------------------------------------------------------
    passed = np.max(rd_red) < rtol_red and np.max(rd_white) < rtol_white

    print(f"\n{'─'*60}")
    print(f"{'Mode':>5}  {'f/fyr':>8}  {'ΔRed%':>8}  {'ΔWhite% (EFAC only)':>22}")
    print(f"{'─'*60}")
    for m in range(min(nmodes, 10)):
        w_str = f"{rd_white[0]*100:9.2f}*" if white_use_time_domain \
                else f"{rd_white[m]*100:9.2f}"
        print(f"  {m:3d}  {freqs[m]/fyr:8.4f}  {rd_red[m]*100:8.2f}  {w_str}")
    if nmodes > 10:
        print(f"  ... ({nmodes-10} more modes)")
    if white_use_time_domain:
        print(f"  (* single scalar epoch-space comparison)")

    print(f"\n  Max Δred               = {rd_red.max()*100:.2f}%  "
          f"(tol {rtol_red*100:.0f}%)  {'✓' if np.max(rd_red) < rtol_red else '✗'}")
    print(f"  Max Δwhite (EFAC only) = {rd_white.max()*100:.2f}%  "
          f"(tol {rtol_white*100:.0f}%)  {'✓' if np.max(rd_white) < rtol_white else '✗'}")
    print(f"\n  Overall: {'PASSED ✓' if passed else 'FAILED ✗'}")

    # ----------------------------------------------------------------
    # 8. Plot
    # ----------------------------------------------------------------
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"PSD vs Residuals — {psr.name}  ({n_realisations} realisations)"
            + ("  [white: epoch-space]" if white_use_time_domain else ""),
            fontsize=12, fontweight='bold'
        )
        fyr_arr = freqs / fyr

        # Panel 1: red noise
        ax = axes[0]
        ax.loglog(fyr_arr, ent_red,           'b-',  lw=2,   label='ENTERPRISE')
        ax.loglog(fyr_arr, empirical_red_psd,  'r--', lw=1.5, label='Empirical')
        ax.set_title('Red Noise PSD')
        ax.set_xlabel(r'$f / f_\mathrm{yr}$')
        ax.set_ylabel(r'PSD [s³]')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        # Panel 2: white noise
        ax = axes[1]
        ax.loglog(fyr_arr, ent_white, 'b-', lw=2, label='ENTERPRISE (EFAC+EQUAD)')
        if white_use_time_domain:
            bw = fourier_bw
            ax.axhline(var_full_scaled      / bw, color='r',  lw=1.5, ls='--',
                       label='Empirical full (EFAC+EQUAD+ECORR)')
            ax.axhline(var_efac_only_scaled / bw, color='g',  lw=1.5, ls='--',
                       label='Empirical EFAC only (fair)')
        else:
            ax.loglog(fyr_arr, empirical_white_psd_full,      'r--', lw=1.5,
                      label='Empirical full (EFAC+EQUAD+ECORR)')
            ax.loglog(fyr_arr, empirical_white_psd_efac_only, 'g--', lw=1.5,
                      label='Empirical EFAC only (fair)')
        ax.set_title('White Noise PSD')
        ax.set_xlabel(r'$f / f_\mathrm{yr}$')
        ax.set_ylabel(r'PSD [s³]')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', alpha=0.3)

        # Panel 3: fractional differences
        ax = axes[2]
        ax.semilogx(fyr_arr, rd_red * 100, 'r-', lw=2, label='Red noise')
        if white_use_time_domain:
            ax.axhline(reldiff_full      * 100, color='r', lw=1.5, ls='--',
                       label='White full (EFAC+EQUAD+ECORR)')
            ax.axhline(reldiff_efac_only * 100, color='g', lw=1.5, ls='--',
                       label='White EFAC only (fair)')
        else:
            ax.semilogx(fyr_arr, rd_white_full * 100, 'r--', lw=1.5,
                        label='White full (EFAC+EQUAD+ECORR)')
            ax.semilogx(fyr_arr, rd_white      * 100, 'g--', lw=1.5,
                        label='White EFAC only (fair)')
        ax.axhline(rtol_red   * 100, color='k',    lw=0.8, ls=':',
                   label=f'Red tol {rtol_red*100:.0f}%')
        ax.axhline(rtol_white * 100, color='grey', lw=0.8, ls=':',
                   label=f'White tol {rtol_white*100:.0f}%')
        ax.set_title('Fractional Difference |empirical − ENT| / ENT')
        ax.set_xlabel(r'$f / f_\mathrm{yr}$')
        ax.set_ylabel('Relative difference [%]')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
        ax.grid(True, which='both', alpha=0.3)

        fig.tight_layout()
        plt.show()

    return {
        'passed':                passed,
        'white_use_time_domain': white_use_time_domain,
        'cond_FtF':              cond,
        'rd_red':                rd_red,
        'rd_white':              rd_white,
        'empirical_red_psd':     empirical_red_psd,
        'empirical_white_psd':   empirical_white_psd,
        'enterprise_red':        ent_red,
        'enterprise_white':      ent_white,
        'freqs':                 freqs,
    }