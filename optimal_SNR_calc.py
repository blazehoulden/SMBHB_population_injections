import numpy as np
from SMBHB_pop_synth import H0_KMS_MPC, MEGAPARSEC_IN_METERS
import sys
from config import generate_population
from signal_injection import draw_red_noise_residuals, strain_amplitude, white_noise_residual
from pta_builder import build_pta_and_params
from enterprise.signals.gp_bases import createfourierdesignmatrix_red
from enterprise.signals.utils import create_quantization_matrix
from scipy.interpolate import interp1d
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


def estimate_chunk_size(N_pulsars, target_memory_GB=2.0, dtype=np.float64):
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

        rn_signal = next(sig for sig in sc._signals 
                 if sig.name == f"{pulsar.name}_red_noise")
        gw_signal = next((sig for sig in sc._signals 
                        if sig.name == f"{pulsar.name}_gw"), None)
        
        
        # only need every second bc values are identical for sine and cosine
        kappa_full = rn_signal.get_phi(params)
        red_PSD = kappa_full[::2][:nmodes] * Tspan

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
            mask = pulsar._flags['f'] == backend
            if not np.any(mask):
                continue
            
            # Find which global epochs this backend contributes to
            # An epoch j belongs to this backend if any TOA in that epoch has this flag
            for j in range(n_epochs):
                epoch_mask = U[:, j].astype(bool)
                if np.any(epoch_mask & mask):
                    ecorr_variance[j] += ecorr**2

        sigma_epoch_sq = np.median(epoch_variance + ecorr_variance)
        white_PSD = np.full(nmodes, 2.0 * sigma_epoch_sq * cadence)

        pulsar_PSD_red[i]   = red_PSD
        pulsar_PSD_white[i] = white_PSD
        pulsar_PSD_total[i] = red_PSD + white_PSD
    return pulsar_PSD_total, pulsar_PSD_red, pulsar_PSD_white, freqs

def measured_strain_all_binaries_all_pulsars(
    bin_arrays:   dict,
    pulsar_cache: dict,
    time_arr:     np.ndarray,
    n_neighbours: int  = 2,       # number of bins either side of peak to include
    full_spectrum: bool = True,
    test_case:    bool = False,
    plot_first_four: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    h_f      : (B, N, 2*n_neighbours+1)  strain at peak ± n_neighbours bins
    bin_freqs: (B, 2*n_neighbours+1)     corresponding frequencies per binary
    delta_f  : (B,)                      bin width (same for all bins, per binary)
    """
    B = bin_arrays['f'].size
    N = pulsar_cache['raj_arr'].size
    K = 2 * n_neighbours + 1          # total bins returned per binary

    # --- Antenna patterns, strain amplitude, phase (unchanged) ---
    Fp, Fx = antenna_response_vectorised(
        pulsar_cache['raj_arr'], pulsar_cache['decj_arr'],
        bin_arrays['ra'], bin_arrays['dec'], bin_arrays['psi'],
    )  # (B, N)


    h0       = strain_amplitude(
        Mc=bin_arrays['Mc'], fGW=bin_arrays['f'],
        d_comov=bin_arrays['D_comov'], z=bin_arrays['z'],
    )  # (B,)


    cos_iota = np.cos(bin_arrays['iota'])
    A_plus   = h0 * (1.0 + cos_iota**2)   # (B,)
    A_cross  = h0 * (-2.0 * cos_iota)     # (B,)

    phase = (2.0 * np.pi * bin_arrays['f'][:, None] * time_arr[None, :]
             + bin_arrays['phi0'][:, None])          # (B, T)

    hp = A_plus[:, None]  * np.sin(phase)            # (B, T)
    hx = A_cross[:, None] * np.cos(phase)            # (B, T)

    # by setting to "forward", we get the 1 / N_time_arr points normalisation factor
    hp_f = np.fft.rfft(hp, norm="forward")           # (B, F)
    hx_f = np.fft.rfft(hx, norm="forward")           # (B, F)

    time_step = time_arr[1] - time_arr[0]
    F_total   = hp_f.shape[1]
    freq_axis = np.fft.rfftfreq(hp.shape[1], d=time_step)  # (F,)
    delta_f   = freq_axis[1] - freq_axis[0]                # scalar, uniform spacing

    if full_spectrum:
        # Skip bin 0 (DC), return all remaining bins
        bin_idxs  = np.tile(np.arange(1, F_total), (B, 1))     # (B, F_total-1)
        hp_bins   = hp_f[:, 1:]                                  # (B, F_total-1)
        hx_bins   = hx_f[:, 1:]                                  # (B, F_total-1)
        bin_freqs = np.tile(freq_axis[1:], (B, 1))              # (B, F_total-1)
    else:
        # --- Peak index per binary, clamped so neighbours stay in bounds ---
        peak_idx_raw = np.argmax(np.abs(hp_f) + np.abs(hx_f), axis=1)  # (B,)
        peak_idx = np.clip(peak_idx_raw, n_neighbours, F_total - 1 - n_neighbours)  # (B,)

        # --- Gather K bins around each peak ---
        # offsets: (-n_neighbours, ..., 0, ..., +n_neighbours)
        offsets  = np.arange(-n_neighbours, n_neighbours + 1)       # (K,)
        bin_idxs = peak_idx[:, None] + offsets[None, :]             # (B, K)

        # Clamp neighbours to [1, F_total-1] — exclude bin 0 (f=0) but NOT the peak
        bin_idxs = np.clip(bin_idxs, 1, F_total - 1)               # (B, K)

        # Extract h+, hx at the K bins for all binaries
        hp_bins = np.take_along_axis(hp_f, bin_idxs, axis=1)     # (B, K)
        hx_bins = np.take_along_axis(hx_f, bin_idxs, axis=1)     # (B, K)

        # Corresponding frequencies
        bin_freqs = freq_axis[bin_idxs]                            # (B, K)

    # Combine with antenna patterns → (B, N, K)
    # Fp, Fx: (B, N) → (B, N, 1) for broadcasting over K
    h_f = (Fp[:, :, None] * hp_bins[:, None, :]
         + Fx[:, :, None] * hx_bins[:, None, :])              # (B, N, K)

    # print(Fp, Fx, abs(hp_f)/hp, abs(hx_f)/hx)

    # --- Plotting (updated to show neighbouring bins) ---
    if plot_first_four:
        _plot_neighbouring_bins(
            freq_axis, hp_f, hx_f, bin_idxs, B, n_neighbours
        )

    # print(np.median(Fp), np.mean(Fp), np.max(Fp), np.min(Fp), np.median(Fx),  np.mean(Fx), np.max(Fx), np.min(Fx),h0, np.median(abs(h_f) / abs(h0)))

    return h_f, bin_freqs, np.full(B, delta_f)


def _plot_neighbouring_bins(freq_axis, hp_f, hx_f, bin_idxs, B, n_neighbours):
    """Plot FFT magnitude for first 4 binaries, highlighting peak ± neighbours."""
    colors      = ["#88CCEE", "#CC6677", "#DDCC77", "#117733"]
    fig, axes   = plt.subplots(min(4, B), 1, figsize=(9, 3 * min(4, B)), sharex=True)
    if B == 1:
        axes = [axes]

    for b in range(min(4, B)):
        ax  = axes[b]
        mag = np.abs(hp_f[b]) + np.abs(hx_f[b])

        ax.plot(freq_axis, mag, color='grey', alpha=0.4, lw=1, label='full spectrum')

        # Highlight the selected K bins
        k_freqs = freq_axis[bin_idxs[b]]
        k_mags  = mag[bin_idxs[b]]
        ax.vlines(k_freqs, 0, k_mags, color=colors[b], lw=2)
        ax.scatter(k_freqs, k_mags, color=colors[b], zorder=5,
                   label=f'peak ± {n_neighbours} bins')

        ax.set_ylabel(r'$|\tilde{h}_+| + |\tilde{h}_\times|$')
        ax.set_title(f'Binary {b}')
        ax.legend(fontsize=8)

    axes[-1].set_xlabel('Frequency [Hz]')
    plt.tight_layout()
    plt.show()


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
    target_memory_GB:    float = 2.0,
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
    # Hellings-Downs coefficients for all pairs
    # ----------------------------------------------------------------
    chi_coeffs = chi_coeff_matrix(pulsars)  # (N, N)

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

        # ---- Strain over K bins: (B, N, K) + frequencies (B, K) -------
        h_f, bin_freqs, delta_f_arr = measured_strain_all_binaries_all_pulsars(
            ba, cache, time_arr, full_spectrum=True
        )
        # h_f:       (Bc, N, K)
        # bin_freqs: (Bc, K)   — frequencies at each of the K bins
        # delta_f_arr: (Bc,)   — bin width (same for all K bins)

        K = h_f.shape[2]
        snr_sq_chunk = np.zeros(Bc)


        # ---- Noise PSD at all K bin frequencies: (Bc, N, K) ------------
        if noise_method == 'enterprise':
            P_noise_K = np.zeros((Bc, N, K))
            for a in range(N):
                log_P             = psd_interpolators[a](np.log(bin_freqs))  # (Bc, K)
                P_noise_K[:, a, :] = np.exp(log_P)
        else:
            fyr = 1.0 / (365.25 * 86400)
            P_noise_K = np.zeros((Bc, N, K))
            if inc_red_noise:
                A_red = 10.0 ** cache['log10A_arr']   # (N,)
                # bin_freqs: (Bc, K), gamma: (N,) → need (Bc, N, K)
                rn = (
                    A_red[None, :, None]**2 / (12.0 * np.pi**2)
                    * (bin_freqs[:, None, :] / fyr) ** (-cache['gamma_arr'][None, :, None])
                    * fyr**-3.0
                )   # (Bc, N, K)
                P_noise_K += rn
            if inc_white_noise:
                P_noise_K += cache['white_noise_arr'][None, :, None]  # broadcast
        # ---- Integrand over K bins (for integration + plotting) ---------
        for g, Tspan_g in enumerate(unique_tspans):
            
            pair_mask = pair_group_ids == g
            gi        = i_idx[pair_mask]
            gj        = j_idx[pair_mask]
            if pair_mask.sum() == 0:
                continue

            # All shapes now have a K dimension at the end
            # h_f: (Bc, N, K) → index pulsars
            h_gi = h_f[:, gi, :]         # (Bc, Pg, K)
            h_gj = h_f[:, gj, :]         # (Bc, Pg, K)

            # we need to weight the pulsar pairs by considering how much they respond to the signal injected using Hellings-Down coefficientsß
            chi_coefficient = chi_coeffs[gi, gj]  # (Pg, Pg)



            # # norm over K bins: (Bc, K)
            # norm_K = 1.0 / (12.0 * np.pi**2 * bin_freqs**3)   # (Bc, K)

            # # T_obs  = time_arr[-1] - time_arr[0]    # [s]
            # # norm_K = T_obs / (4.0 * np.pi**2 * bin_freqs**2)   # (Bc, K)
            # Sh_ii = h_gi * h_gi.conj() * norm_K[:, None, :]    # (Bc, Pg, K)
            # Sh_jj = h_gj * h_gj.conj() * norm_K[:, None, :]
            # Sh_ij = 0.5 * (
            #     h_gi * h_gj.conj() + h_gi.conj() * h_gj
            # ) * norm_K[:, None, :]
            # Ni = P_noise_K[:, gi, :] + Sh_ii                   # (Bc, Pg, K)
            # Nj = P_noise_K[:, gj, :] + Sh_jj


            # # Integrand at each of the K bins: (Bc, Pg, K)
            # integrand_K = 2.0 * Tspan_g * chi_coefficient[None, :, None]**2 * Sh_ij**2 / (Ni * Nj)
            
            # Power spectral density: Sh_ij = abs(h_i * h_j_conj) / delta f - delta_f_arr is uniform
            Sh_ii = 2.0 * h_gi * h_gi.conj() / delta_f_arr[0] # [s]
            Sh_jj = 2.0 * h_gj * h_gj.conj() / delta_f_arr[0] # [s]
            Sh_ij = 2.0 * 0.5 * (h_gi * h_gj.conj() + h_gi.conj() * h_gj) / delta_f_arr[0] #[s]

            # We want the PSD of the timing residuals - S_tr(f) = Sh(f) / (12 * pi^2 * f^2)
            norm_K = 1.0 / (12.0 * np.pi**2 * bin_freqs**2)   # (Bc, K)
            Sr_ii = Sh_ii * norm_K[:, None, :]    # (Bc, Pg, K)
            Sr_jj = Sh_jj * norm_K[:, None, :]
            Sr_ij = Sh_ij * norm_K[:, None, :]


            Ni = P_noise_K[:, gi, :] + Sr_ii                   # (Bc, Pg, K)
            Nj = P_noise_K[:, gj, :] + Sr_jj

            integrand_K = 2.0 * Tspan_g * chi_coefficient[None, :, None]**2 * Sr_ij**2 / (Ni * Nj)

            # print(np.median(P_noise_K), np.median(np.abs(h_f)**2))
            # ---- Plot integrand vs frequency (first binary, first pair) --
            # Shows you how much it evolves across the K bins
            # _plot_integrand_vs_freq(bin_freqs, integrand_K, label=f'chunk {start}')

            # ---- Integrate over K bins using trapezoidal rule -------------------
            # integrand_K: (Bc, Pg, K) → sum over pairs first → (Bc, K), then trapz over K

            integrand_real = np.real(integrand_K).sum(axis=1)   # (Bc, K) — summed over pairs

            snr_sq_chunk += np.trapz(
                integrand_real,
                x=bin_freqs[0],    # (Bc, K) — non-uniform spacing handled automatically
                axis=-1         # integrate over K bins
            ) 

        snr_sq_arr[chunk] = snr_sq_chunk
    
    return snr_sq_arr

def _plot_integrand_vs_freq(bin_freqs, integrand_K, label=''):
    """
    Plot the real integrand vs frequency across the K neighbouring bins.
    bin_freqs:    (Bc, K)
    integrand_K:  (Bc, Pg, K)  — sum over pairs for display
    """
    integrand_summed = np.real(integrand_K[0]).sum(axis=0)   # (K,) first binary, all pairs
    freqs_plot       = bin_freqs[0]                           # (K,) first binary

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        freqs_plot,
        integrand_summed,
        width=np.diff(freqs_plot).mean() * 0.8,
        color='#88CCEE', edgecolor='k', alpha=0.8,
        label='integrand per bin'
    )
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$2 T_\mathrm{span}\, S_{ij}^2 / (N_i N_j)$')
    ax.set_title(f'Integrand across peak ± neighbours  [{label}]')
    ax.legend()
    plt.tight_layout()
    plt.show()


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
    target_memory_GB:    float = 2.0,
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


def compare_to_enterprise_os(
    os_obj,              # enterprise_extensions OptimalStatistic object
    raw_noise_params,    # your noise dict passed to build_pta_and_params
    pulsars,             # list of enterprise pulsar objects
    Tspan,               # float, seconds
    psd_interpolators,   # list of interp1d from your SNR function
    snr_sq_arr,          # (B,) your per-binary SNR^2 output
    binaries,            # list of binary dicts
    nmodes,              # int, same as used in pulsar_PSD_using_enterprise
):

    # ------------------------------------------------------------------
    # 1. Run enterprise OS at MAP noise params
    # ------------------------------------------------------------------
    pta_cmp, _, params_cmp = build_pta_and_params(
        psrs              = pulsars,
        noise_params_15yr = raw_noise_params,
        Tspan             = Tspan,
        include_GW        = True,
        nmodes            = nmodes,
    )
    os_obj_cmp = os_obj   # reuse passed-in object

    xi, rho, sig, OS, OS_sig = os_obj_cmp.compute_os(params=params_cmp)
    psrnames = [p.name for p in pulsars]
    pair_idx = [
        (i, j)
        for i in range(len(pulsars))
        for j in range(i+1, len(pulsars))
    ]

    print("\n" + "="*70)
    print("ENTERPRISE OS SUMMARY")
    print("="*70)
    print(f"  OS amplitude A^2        = {OS:.6e}")
    print(f"  OS sigma_A^2            = {OS_sig:.6e}")
    print(f"  OS SNR = A^2/sigma_A^2  = {OS / OS_sig:.6f}")
    print(f"  N pairs                 = {len(pair_idx)}")

    # ------------------------------------------------------------------
    # 2. Per-pair sigma_IJ from enterprise vs yours
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("PER-PAIR SIGMA_IJ COMPARISON (first 10 pairs)")
    print("="*70)
    print(f"  {'Pair':<40} {'ent sigma_IJ':>14}  {'note'}")
    print(f"  {'-'*40} {'-'*14}  {'-'*20}")
    for k, (i, j) in enumerate(pair_idx[:10]):
        print(f"  {psrnames[i][:18]}-{psrnames[j][:18]}  {sig[k]:>14.6e}")

    # ------------------------------------------------------------------
    # 3. Reconstruct YOUR sigma_IJ from the noise PSD interpolators
    #    using the same GWB template enterprise uses internally
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("YOUR SIGMA_IJ (from noise PSD interpolators, power-law template)")
    print("="*70)

    fyr    = 1.0 / (365.25 * 86400)
    gamma  = 13.0 / 3.0
    f_grid = np.array([k / Tspan for k in range(1, nmodes + 1)])   # Hz

    # GWB timing residual PSD template at unit amplitude (enterprise convention)
    # phi_gw(f) = A^2/(12pi^2) * f^{-gamma} * fyr^{gamma-3} / (2*df)
    # but as a continuous PSD: S_gw(f) = phi_gw * 2*df = A^2/(12pi^2) * f^{-gamma} * fyr^{gamma-3}
    # In timing residual units (already 1/(2pif)^2 folded in for GWB):
    # S_r_gw(f) = A^2/(12pi^2) * f^{-gamma} * fyr^{gamma-3} * 1/(2*df)  [enterprise phi units]
    # As continuous one-sided PSD:
    S_r_gw = (1.0 / (12.0 * np.pi**2)) * f_grid**(-gamma) * fyr**(gamma - 3)  # s^3, A=1

    your_sigma2_IJ = np.zeros(len(pair_idx))
    for k, (i, j) in enumerate(pair_idx):
        P_I = np.exp(psd_interpolators[i](np.log(f_grid)))   # s^3
        P_J = np.exp(psd_interpolators[j](np.log(f_grid)))   # s^3
        # sigma^2_IJ = integral S_gw^2 / (P_I * P_J) df
        integrand = S_r_gw**2 / (P_I * P_J)
        your_sigma2_IJ[k] = 2.0 * np.trapz(integrand, x=f_grid)

    your_sigma_IJ = np.sqrt(your_sigma2_IJ)

    print(f"  {'Pair':<40} {'ent sigma':>12}  {'yours':>12}  {'ratio y/e':>10}")
    print(f"  {'-'*40} {'-'*12}  {'-'*12}  {'-'*10}")
    for k, (i, j) in enumerate(pair_idx[:10]):
        ratio = your_sigma_IJ[k] / sig[k] if sig[k] > 0 else np.nan
        print(f"  {psrnames[i][:18]}-{psrnames[j][:18]}  "
              f"{sig[k]:>12.4e}  {your_sigma_IJ[k]:>12.4e}  {ratio:>10.4f}")

    # ------------------------------------------------------------------
    # 4. SNR comparison
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("SNR COMPARISON")
    print("="*70)

    # Enterprise OS SNR
    ent_snr = OS / OS_sig
    print(f"  Enterprise OS SNR                = {ent_snr:.6f}")

    # Your total SNR (sum in quadrature over all binaries)
    your_snr_total = np.sqrt(np.nansum(snr_sq_arr))
    print(f"  Your total SNR sqrt(sum rho^2)   = {your_snr_total:.6f}")
    print(f"  Ratio yours/enterprise           = {your_snr_total / ent_snr:.4f}")

    # Your sigma_A^2 from the noise-only sigma_IJ (comparable to enterprise)
    your_sigma_A2 = 1.0 / np.sqrt(np.sum(1.0 / your_sigma2_IJ))
    print(f"  Your sigma_A^2 (noise PSD only)  = {your_sigma_A2:.6e}")
    print(f"  Enterprise sigma_A^2             = {OS_sig:.6e}")
    print(f"  Ratio sigma_A^2 yours/enterprise = {your_sigma_A2 / OS_sig:.4f}")

    # ------------------------------------------------------------------
    # 5. Noise PSD spot check: yours vs enterprise at key frequencies
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("NOISE PSD SPOT CHECK — pulsar 0 at first 5 frequencies")
    print("="*70)

    sc_0 = next(
        sc for sc in pta_cmp._signalcollections
        if sc.psrname == pulsars[0].name
    )
    rn_0 = next(
        sig_obj for sig_obj in sc_0._signals
        if f'{pulsars[0].name}_red_noise' in sig_obj.name
    )
    phi_rn  = rn_0.get_phi(params_cmp)          # per-mode variance, pairs
    phi_rn_unique = phi_rn[::2][:nmodes]         # one per frequency

    # Enterprise noise PSD (timing residual): S_r(f) = phi_k * 2*df = phi_k * 2/Tspan
    # But in your interpolator units it is phi_k * Tspan (from pulsar_PSD_using_enterprise)
    # So the ratio should be Tspan^2/2 ... unless your white noise is in different units
    # We check both components separately

    Nvec_0     = np.array(sc_0.get_ndiag(params_cmp)._nvec, dtype=float)
    sigma2_toa = np.mean(Nvec_0)
    dt_toa     = Tspan / len(pulsars[0].toas)
    P_white_phys = sigma2_toa * dt_toa

    print(f"  {'f_k (Hz)':>14}  {'phi_rn (ent)':>14}  {'S_r_interp (yours)':>20}  "
          f"{'P_white_phys':>14}  {'ratio_RN':>10}")
    for k in range(5):
        f_k     = f_grid[k]
        P_yours = float(np.exp(psd_interpolators[0](np.log(f_k))))
        phi_k   = phi_rn_unique[k]
        # enterprise per-mode variance → continuous PSD
        S_r_ent = phi_k * Tspan   # your convention: phi * Tspan
        print(f"  {f_k:>14.4e}  {phi_k:>14.4e}  {P_yours:>20.4e}  "
              f"{P_white_phys:>14.4e}  {P_yours/S_r_ent:>10.4f}")

    # ------------------------------------------------------------------
    # 6. Per-frequency integrand comparison for pair (0,1)
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("PER-FREQUENCY INTEGRAND — pair (0,1), GWB template")
    print("comparing enterprise sigma^2_IJ integrand vs yours")
    print("="*70)

    i, j = 0, 1
    P_I_grid = np.exp(psd_interpolators[i](np.log(f_grid)))
    P_J_grid = np.exp(psd_interpolators[j](np.log(f_grid)))
    integrand_yours = S_r_gw**2 / (P_I_grid * P_J_grid)

    # Enterprise integrand: use phi_gw and per-pulsar phi from get_phi
    gw_sigs = [
        sig_obj for sc in pta_cmp._signalcollections
        for sig_obj in sc._signals
        if 'gw' in sig_obj.name.lower() and sc.psrname == pulsars[i].name
    ]
    if gw_sigs:
        phi_gw   = gw_sigs[0].get_phi(params_cmp)[::2][:nmodes]
        phi_tot_I = (rn_0.get_phi(params_cmp)[::2][:nmodes] + phi_gw)
        sc_j = next(sc for sc in pta_cmp._signalcollections
                    if sc.psrname == pulsars[j].name)
        rn_j = next(s for s in sc_j._signals
                    if f'{pulsars[j].name}_red_noise' in s.name)
        phi_tot_J = (rn_j.get_phi(params_cmp)[::2][:nmodes] + phi_gw)
        # Enterprise integrand in phi units: phi_gw^2 / (phi_I * phi_J)
        # Convert to PSD units (* Tspan for each phi): ratio is unchanged
        integrand_ent = phi_gw**2 / (phi_tot_I * phi_tot_J)
        integrand_ent_psd = S_r_gw**2 / (
            (phi_tot_I * Tspan) * (phi_tot_J * Tspan) / Tspan**2
        )

        print(f"  {'f_k':>12}  {'intgd yours':>14}  {'intgd ent':>14}  {'ratio':>10}")
        print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*10}")
        for k in range(10):
            ratio = integrand_yours[k] / integrand_ent[k] if integrand_ent[k] > 0 else np.nan
            print(f"  {f_grid[k]:>12.4e}  {integrand_yours[k]:>14.4e}  "
                  f"{integrand_ent[k]:>14.4e}  {ratio:>10.4f}")
    else:
        print("  (GW signal not found in PTA signals — run with include_GW=True)")


    # Enterprise total phi per mode (red + white)
    sc_0 = next(sc for sc in pta_cmp._signalcollections 
                if sc.psrname == pulsars[0].name)
    phi_total_ent = sc_0.get_phi(params_cmp)  # includes all noise sources
    phi_total_unique = phi_total_ent[::2][:nmodes]  # one per freq

    # Your interpolator
    P_yours = np.exp(psd_interpolators[0](np.log(f_grid)))

    print("f_k | phi_total_ent | P_yours/Tspan | ratio")
    for k in range(10):
        ent = phi_total_unique[k]
        yours = P_yours[k] / Tspan
        print(f"{f_grid[k]:.3e} | {ent:.4e} | {yours:.4e} | {yours/ent:.4f}")

    print("\n" + "="*70)
    print("SUMMARY OF LIKELY DISCREPANCY SOURCES")
    print("="*70)
    print(f"  sigma_IJ ratio (yours/enterprise): "
          f"median={np.median(your_sigma_IJ/sig):.4f}, "
          f"std={np.std(your_sigma_IJ/sig):.4f}")
    print(f"  If ratio ~1:    noise PSD is consistent, discrepancy is in signal")
    print(f"  If ratio >1:    your noise PSD is too large → SNR suppressed")
    print(f"  If ratio <1:    your noise PSD is too small → SNR inflated")
    print(f"  SNR ratio (yours/enterprise): {your_snr_total/ent_snr:.4f}")
    print("="*70)

    return {
        'xi':             xi,
        'rho':            rho,
        'sig_enterprise': sig,
        'sig_yours':      your_sigma_IJ,
        'OS':             OS,
        'OS_sig':         OS_sig,
        'snr_enterprise': ent_snr,
        'snr_yours':      your_snr_total,
        'snr_sq_arr':     snr_sq_arr,
    }


def sigma_ab(
    pulsar_a:             object,
    pulsar_b:             object,
    parsed_noise_params: dict,
    raw_noise_params:    dict,
    Tspan:               float,
    nmodes:              int   = 301,

) -> np.ndarray:
    """
    Compute sigma for pulsar pairs.
    This is the denominator of the optimal statistic SNR formula, and is given by:

        sigma_ab^2 = 1/(2 T) * [ integral (P_g^2 / (P_a P_b)) df ]^-1
    """

    
    f_min = 1.0 / Tspan
    f_max = 3e-7
    f_step = 1.0 / Tspan
    N_bin_f = int((f_max - f_min) / f_step) + 1

    bin_edges = np.linspace(f_min, f_min + N_bin_f * f_step, N_bin_f + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = bin_edges[1:] - bin_edges[:-1]


    time_arr_max = np.max(1.0 / bin_edges)
    time_arr_min = np.min(1.0 / bin_edges)

    # ----------------------------------------------------------------
    # Per-pair Tspan
    # ----------------------------------------------------------------
    psr_tspan_a  = pulsar_a.toas.max() - pulsar_a.toas.min() 
    psr_tspan_b  = pulsar_b.toas.max() - pulsar_b.toas.min() 
    tspan = np.minimum(psr_tspan_a, psr_tspan_b)

    # ----------------------------------------------------------------
    # Hellings-Downs coefficients for all pairs
    # ----------------------------------------------------------------
    pulsars = np.array([pulsar_a, pulsar_b])
    chi_coeffs = chi_coeff_matrix(pulsars)  # (N, N)
    # noise_arr_total, noise_red, noise_white, freqs = pulsar_PSD_using_enterprise(pulsars, raw_noise_params, parsed_noise_params, Tspan, nmodes=nmodes)

    freqs = np.array([k / Tspan for k in range(1, nmodes + 1)])   # (nmodes,)
    # freqs = freqs[0]
    A_gw = 1.
    alpha = -2. /3.
    fyr = 1 / (86400 * 365.25)
    P_g = A_gw**2 / (12 * np.pi**2) * (freqs/fyr)**(2 * alpha) * freqs**(-3)
    white_a = analytic_white_noise_psd(pulsar_a, parsed_noise_params)
    white_b = analytic_white_noise_psd(pulsar_b, parsed_noise_params)
    
    red_a = pulsar_red_noise_psd(freqs, parsed_noise_params[pulsar_a.name]['red_noise']['log10_A'], parsed_noise_params[pulsar_a.name]['red_noise']['gamma'], fyr)
    red_b = pulsar_red_noise_psd(freqs, parsed_noise_params[pulsar_b.name]['red_noise']['log10_A'], parsed_noise_params[pulsar_b.name]['red_noise']['gamma'], fyr)

    # print("red_noise:", noise_red[0]/red_a, "white_noise:", noise_white[0]/white_a)


    P_a = white_a + red_a
    P_b = white_b + red_b

    integrand = P_g**2 / (P_a * P_b)

    integrated_val = np.trapz(
            integrand,
            x=freqs,    # (Bc, K) — non-uniform spacing handled automatically
            axis=-1         # integrate over K bins
        ) 
    chi_coeff_ab = chi_coeffs[0, 1]
    sigma_ab = (2 * Tspan * chi_coeff_ab**2 * integrated_val)**(-0.5)
    return sigma_ab

def sigma_ab_all_pairs(
    psrs_clean:          list,
    parsed_noise_params: dict,
    raw_noise_params:    dict,
    Tspan:               float,
    nmodes:              int = 150,
    psd:                 list = None,  
) -> tuple[np.ndarray, list]:
    """
    Compute sigma_IJ for all unique pulsar pairs.
    Returns sigma_arr (N_pairs,) and pair_labels list.
    """
    fyr   = 1.0 / (86400.0 * 365.25)
    gamma = 13.0 / 3.0
    N     = len(psrs_clean)

    # ------------------------------------------------------------------
    # 1. Frequency grid — same for all pulsars
    # ------------------------------------------------------------------
    freqs = np.array([k / Tspan for k in range(1, nmodes + 1)])   # (nmodes,)

    # ------------------------------------------------------------------
    # 2. GWB template PSD — computed once
    # ------------------------------------------------------------------
    P_gw = (1.0 / (12.0 * np.pi**2)) * freqs**(-gamma) * fyr**(gamma - 3.0)  # (nmodes,) s^3

    # ------------------------------------------------------------------
    # 3. Per-pulsar noise PSD — computed once per pulsar, not per pair
    # ------------------------------------------------------------------
    if psd is None:
        P_noise = np.zeros((N, nmodes))   # (N, nmodes)
        for i, psr in enumerate(psrs_clean):
            white = analytic_white_noise_psd(psr, parsed_noise_params)
            red   = pulsar_red_noise_psd(
                freqs,
                parsed_noise_params[psr.name]['red_noise']['log10_A'],
                parsed_noise_params[psr.name]['red_noise']['gamma'],
                fyr,
            )
            P_noise[i] = white + red
    else:
        P_noise = np.array(psd)  # (N, nmodes)

    # ------------------------------------------------------------------
    # 4. HD coefficients — computed once for all pairs
    # ------------------------------------------------------------------
    chi_mat = chi_coeff_matrix(psrs_clean)   # (N, N)

    # ------------------------------------------------------------------
    # 5. Pair loop — now just arithmetic, no PTA builds
    # ------------------------------------------------------------------
    pair_labels = []
    sigma_arr   = []

     # Vectorised version of step 5 — replaces the pair loop entirely
    # Build index arrays for all unique pairs
    ii, jj     = np.tril_indices(N, k=-1)   # lower triangle indices
    # ii, jj give j < i; swap to get i < j convention
    i_idx, j_idx = jj, ii

    # Integrand for all pairs at once: (N_pairs, nmodes)
    integrand_all = P_gw[None, :]**2 / (P_noise[i_idx] * P_noise[j_idx])

    # Integrate each pair over frequency: (N_pairs,)
    integral_all  = np.trapz(integrand_all, x=freqs, axis=-1)

    # HD coefficients for all pairs: (N_pairs,)
    Gamma_all     = chi_mat[i_idx, j_idx]

    # sigma for all pairs: (N_pairs,)
    sigma_arr     = (2.0 * Tspan * Gamma_all**2 * integral_all) ** (-0.5)
    # sigma_arr     = (2.0 * Tspan  * integral_all) ** (-0.5)

    pair_labels   = [(psrs_clean[i].name, psrs_clean[j].name)
                     for i, j in zip(i_idx, j_idx)]

    return np.array(sigma_arr), pair_labels, P_noise

def get_enterprise_noise_per_mode(
    psrs_clean:       list,
    raw_noise_params: dict,
    Tspan:            float,
    nmodes:           int = 301,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract N_tilde_I diagonal for the Fourier modes only.
    Strips timing model columns before inverting.
    """
    N = len(psrs_clean)

    pta, _, params = build_pta_and_params(
        psrs              = psrs_clean,
        noise_params_15yr = raw_noise_params,
        Tspan             = Tspan,
        include_GW        = True,
        nmodes            = nmodes,
    )

    # Set GWB and BayesEphem params explicitly
    param_defaults = {
        'gw_log10_A':        np.log10(2.4e-15),
        'gw_gamma':          13.0 / 3.0,
        'd_jupiter_mass':     0.0,
        'd_saturn_mass':      0.0,
        'd_uranus_mass':      0.0,
        'd_neptune_mass':     0.0,
        'frame_drift_rate':   0.0,
        'jup_orb_elements_0': 0.0,
        'jup_orb_elements_1': 0.0,
        'jup_orb_elements_2': 0.0,
        'jup_orb_elements_3': 0.0,
        'jup_orb_elements_4': 0.0,
        'jup_orb_elements_5': 0.0,
    }
    for k, v in param_defaults.items():
        if k not in params:
            params[k] = v

    freqs   = np.array([k / Tspan for k in range(1, nmodes + 1)])
    N_tilde = np.zeros((N, nmodes))
    TNT_list = pta.get_TNT(params)

    for i, psr in enumerate(psrs_clean):
        sc       = next(sc for sc in pta._signalcollections
                        if sc.psrname == psr.name)
        phi_full = np.array(sc.get_phi(params))   # (n_tm + 2*nmodes,)
        TNT_I    = TNT_list[i]                     # (n_tm + 2*nmodes, n_tm + 2*nmodes)

        # Total size and Fourier block size
        n_total  = len(phi_full)
        n_fourier = 2 * nmodes                     # always last 2*nmodes entries

        # Verify the last 2*nmodes entries are the Fourier modes
        # (not the 1e40 timing model entries)
        phi_fourier = phi_full[-n_fourier:]
        phi_tm      = phi_full[:-n_fourier]

        print(f"  {psr.name}: phi_full size={n_total}, "
              f"n_tm={n_total - n_fourier}, "
              f"phi_tm max={phi_tm.max():.2e}, "
              f"phi_fourier max={phi_fourier.max():.2e}  "
              f"<-- phi_tm should be ~1e40, phi_fourier should be ~1e-12")

        # Extract Fourier-only block of TNT
        TNT_fourier = TNT_I[-n_fourier:, -n_fourier:]   # (2*nmodes, 2*nmodes)

        # Sigma = Phi_fourier^{-1} + TNT_fourier
        Sigma_fourier = np.diag(1.0 / phi_fourier) + TNT_fourier

        # N_tilde = Sigma^{-1}
        N_tilde_fourier = np.linalg.inv(Sigma_fourier)   # (2*nmodes, 2*nmodes)

        # One entry per frequency (sin and cos degenerate)
        N_tilde[i] = np.diag(N_tilde_fourier)[::2][:nmodes]

    return N_tilde, freqs


def sigma_ab_all_pairs_enterprise(
    psrs_clean:       list,
    raw_noise_params: dict,
    Tspan:            float,
    nmodes:           int = 301,
) -> tuple[np.ndarray, list]:
    """
    Compute sigma_IJ for all pairs using enterprise's exact N_tilde_I.

    The formula (Chamberlin+2015, Eq. A9) in per-mode form is:

        sigma_IJ^{-2} = sum_k  2 * Gamma_IJ^2 * phi_gw_k^2
                                / (N_tilde_I_k * N_tilde_J_k)

    where phi_gw_k is the unit-amplitude GWB per-mode variance.
    All quantities are in the same phi units (s^2 per mode) so no
    PSD conversion is needed.
    """
    fyr   = 1.0 / (86400.0 * 365.25)
    gamma = 13.0 / 3.0
    N     = len(psrs_clean)

    # ------------------------------------------------------------------
    # 1. Get N_tilde for all pulsars from enterprise
    # ------------------------------------------------------------------
    N_tilde, freqs = get_enterprise_noise_per_mode(
        psrs_clean       = psrs_clean,
        raw_noise_params = raw_noise_params,
        Tspan            = Tspan,
        nmodes           = nmodes,
    )
    # N_tilde: (N, nmodes), units s^2 (per-mode variance)

    # ------------------------------------------------------------------
    # 2. GWB template per-mode variance at unit amplitude
    #    phi_gw_k = 1/(12pi^2) * f^{-gamma} * fyr^{gamma-3} * Tspan/2
    #    This is what enterprise stores in phi for the GWB signal
    # ------------------------------------------------------------------
    phi_gw = (
        (1.0 / (12.0 * np.pi**2))
        * freqs**(-gamma)
        * fyr**(gamma - 3.0)
        * Tspan / 2.0
    )   # (nmodes,) s^2

    # ------------------------------------------------------------------
    # 3. HD coefficients for all pulsars
    # ------------------------------------------------------------------
    # Build PTA again to get chi — or reuse if you have it
    pta, _, params = build_pta_and_params(
        psrs              = psrs_clean,
        noise_params_15yr = raw_noise_params,
        Tspan             = Tspan,
        include_GW        = True,
        nmodes            = nmodes,
    )
    chi_mat = chi_coeff_matrix(psrs_clean)   # (N, N)

    # ------------------------------------------------------------------
    # 4. Vectorised pair computation
    # ------------------------------------------------------------------
    ii, jj  = np.tril_indices(N, k=-1)
    i_idx   = jj
    j_idx   = ii

    # Per-mode integrand for all pairs: (N_pairs, nmodes)
    # sigma^{-2} = sum_k  2 * Gamma^2 * phi_gw^2 / (N_tilde_I * N_tilde_J)
    integrand_all = (
        2.0 * phi_gw[None, :]**2
        / (N_tilde[i_idx] * N_tilde[j_idx])
    )   # (N_pairs, nmodes)

    # Sum over modes: (N_pairs,)
    sigma_inv2_all = np.sum(integrand_all, axis=-1)

    # HD factor
    Gamma_all      = chi_mat[i_idx, j_idx]   # (N_pairs,)
    sigma_inv2_all = sigma_inv2_all * Gamma_all**2

    sigma_arr   = 1.0 / np.sqrt(sigma_inv2_all)   # (N_pairs,)
    pair_labels = [
        (psrs_clean[i].name, psrs_clean[j].name)
        for i, j in zip(i_idx, j_idx)
    ]

    return sigma_arr, pair_labels

def get_pulsar_noise_psd(pta, params, pulsar_idx, T_span):
    '''
    Extract the Woodbury-marginalised total noise PSD for a single pulsar.
    This is the quantity that appears in the OS noise floor.

    Parameters
    ----------
    pta        : enterprise PTA object
    params     : dict of noise parameters (one posterior sample)
    pulsar_idx : int, index of the pulsar in pta.pulsars
    T_span     : float, total timing baseline in seconds

    Returns
    -------
    freqs   : array of shape (Nf,), frequencies in Hz
    psd     : array of shape (Nf,), one-sided noise PSD in s^3
    '''
    df = 1.0 / T_span


    # ── Step 1: Get the full Phi matrix (all pulsars) ──────────────────
    # Shape: (2*Npsr*Nf, 2*Npsr*Nf)
    # Diagonal block for pulsar a is at rows/cols:
    #   [ a*2*Nf : (a+1)*2*Nf, a*2*Nf : (a+1)*2*Nf ]
    phi_list = pta.get_phi(params)
    a = pulsar_idx
    phi_a_full = np.array(phi_list[a])   # shape (Ntm + 2*Nf,) — flat diagonal
    # Find where the 1e40 sentinels end — real entries are not 1e40
    real_mask = phi_a_full < 1e30
    phi_a_diag = phi_a_full[real_mask]   # shape (2*Nf,), actual red noise variances
    # Infer Nf from the data
    Nf = len(phi_a_diag) // 2
    freqs = np.arange(1, Nf + 1) * df

    # phi_a_diag[2k]   = variance of sin coefficient at freq bin k
    # phi_a_diag[2k+1] = variance of cos coefficient at freq bin k
    # Both equal P(fk) * df for an isotropic process

    # ── Step 2: Get TNT for this pulsar ────────────────────────────────
    # TNT is a list over pulsars, each shape (Ntm + 2Nf, Ntm + 2Nf)
    TNTs = pta.get_TNT(params)

    # ── Step 3: Slice out the Fourier-only block ───────────────────────
    # TNT_a has timing model columns first, then Fourier columns.
    # Ntm = number of timing model parameters for this pulsar.
    TNT_a = np.array(TNTs[a], dtype=float)  
    Ntm = TNT_a.shape[0] - 2 * Nf   # total columns minus the 2*Nf Fourier columns
    FtNF = TNT_a[Ntm:, Ntm:]

    # ── Step 4: Build the Fourier-basis total noise precision matrix ───
    # ΣF,a⁻¹ = Φred,a⁻¹ + FᵀN⁻¹F
    # For a diagonal Phi_a, the inverse is just 1/diag
    Phi_a_inv = np.diag(1.0 / phi_a_diag)
    Sigma_F_inv = Phi_a_inv + FtNF

    # ── Step 5: Invert to get the total noise covariance ───────────────
    # ΣF,a = (Φred,a⁻¹ + FᵀN⁻¹F)⁻¹
    Sigma_F = np.linalg.inv(Sigma_F_inv)

    # ── Step 6: Extract PSD from diagonal ─────────────────────────────
    # Average sin and cos components (equal by isotropy), divide by df
    psd = np.array([
        0.5 * (Sigma_F[2*k, 2*k] + Sigma_F[2*k+1, 2*k+1]) / df
        for k in range(Nf)
    ])

    return freqs, psd
