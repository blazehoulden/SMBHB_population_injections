import numpy as np
from SMBHB_pop_synth import H0_KMS_MPC, MEGAPARSEC_IN_METERS
from config import generate_population
from signal_injection import strain_amplitude

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

def plot_pulsar_psd(pulsar_noise_params, frequencies, sigma_ns = 100.0, delta_t_yr = 1.0/20.0):
    """
    Plot the PSD for a single pulsar's noise across frequencies.
    
    Parameters
    ----------
    pulsar_noise_params : dict
        Dictionary of noise parameters for a single pulsar
    frequencies : array_like
        Array of frequencies [Hz]
    """
    import matplotlib.pyplot as plt
    
    red_psd_values = pulsar_red_noise_psd(frequencies, pulsar_noise_params['red_noise']['log10_A'], pulsar_noise_params['red_noise']['gamma'])
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


def antenna_response(psr_ra, psr_dec, src_ra, src_dec, psi):
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

    denom = 1 + np.dot(omega_hat, p_hat)
    Fp = 0.5 * ((np.dot(p_hat, m_hat)**2 - np.dot(p_hat, n_hat)**2) / denom)
    Fx = (np.dot(p_hat, m_hat) * np.dot(p_hat, n_hat)) / denom

    return Fp, Fx

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


def build_pulsar_cache(pulsars, pulsar_noise_params):
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
    pulsar_noise_params : dict
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
        pulsar_noise_params[p.name]['red_noise']['log10_A'] for p in pulsars
    ])  # shape (N,)
    gamma_arr = np.array([
        pulsar_noise_params[p.name]['red_noise']['gamma'] for p in pulsars
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
    white_noise_arr = pulsar_cache['white_noise_arr']   # (N,)
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

    # --------------------------------------------------------------------- CHANGED: ORF branch
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
    # END CHANGED -----------------------------------------------------------

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
    # ------------------------------------------------------------------ unpack binary arrays
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

    # ------------------------------------------------------------------ diagnostics (kept from original)
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


def N_needed_for_population(binaries, pulsars, pulsar_noise_params, strain_data,
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
    pulsar_noise_params : dict
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
    pulsar_cache = build_pulsar_cache(pulsars, pulsar_noise_params)

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


def convergence_test(binaries, pulsars, pulsar_noise_params, strain_data, T_obs,
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
    pulsars, pulsar_noise_params, strain_data, T_obs : (see N_needed_for_population)
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

    pulsar_cache = build_pulsar_cache(pulsars, pulsar_noise_params)

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

def plot_overlap_reduction_function(pulsars, binaries, pulsar_noise_params):
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
    pulsar_cache = build_pulsar_cache(pulsars, pulsar_noise_params)  # build cache to get i_idx, j_idx, and pulsars
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

# def ind_pulsar_pair_SNR(binaries, freqs, pulsar1, pulsar2, T_obs, pulsar_noise_params):
#     red_PSD_pulsar1 = pulsar_red_noise_psd(freqs, pulsar_noise_params[pulsar1.name]['red_noise']['log10_A'])
#     red_PSD_pulsar2 = pulsar_red_noise_psd(freqs, pulsar_noise_params[pulsar2.name]['red_noise']['log10_A'])
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
def measured_strain_all_binaries_all_pulsars(
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

    # Contract with antenna patterns:
    # Fp is (B, N), hp is (B, T) → need (B, N, T)
    # h[b, n, t] = Fp[b,n]*hp[b,t] + Fx[b,n]*hx[b,t]
    h = Fp[:, :, None] * hp[:, None, :] \
      + Fx[:, :, None] * hx[:, None, :]   # (B, N, T)

    return h


def SNR_sq_all_pairs_all_binaries_vectorised(
    binaries:           list,
    pulsars:            list,
    pulsar_noise_params: dict,
    strain_data:        dict,
    time_arr_npoints:   int   = 10_001,
    chunk_size:         int   = None,
    target_memory_GB:   float = 1.0,
    inc_GW:              bool = True,
    inc_red_noise:       bool = True,
    inc_white_noise:     bool = True,
) -> np.ndarray:
    """
    Compute ρ² for every binary summed over all unique pulsar pairs — fully
    vectorised, no Python loops over binaries, pulsars, or time steps.

    The SNR² formula per binary, summed over pairs (i < j):

        ρ²_b = Σ_{i<j} Δf_b ∫ S_h^{ij}(t) / [N_i(t) N_j(t)] dt

    approximated as a midpoint Riemann sum over T time steps.

    Memory scales as  B × N × T × 8 bytes  per chunk.
    With N=50 pulsars, T=10001 time steps, float64:
        1 GB ≈ 25 binaries per chunk  →  auto chunk_size handles this.

    Parameters
    ----------
    binaries : list of dict
        Each must contain 'f', 'Mc', 'D_comov', 'z', 'ra', 'dec',
        and optionally 'psi', 'phi0', 'iota', 'freq_bin'.
    pulsars : list of Pulsar
    pulsar_noise_params : dict
    strain_data : dict
        Must contain 'bin_edges'.
    time_arr_npoints : int
        Number of time samples per pulsar span. Default 10 001.
    chunk_size : int or None
        Binaries per chunk. None = auto from target_memory_GB.
    target_memory_GB : float
        Memory budget for auto chunk sizing [GB].

    Returns
    -------
    snr_sq_arr : ndarray, shape (B,)
        ρ² for each binary, summed over all pulsar pairs.
    """
    cache     = build_pulsar_cache_time_domain(pulsars, pulsar_noise_params)
    i_idx     = cache['i_idx']   # (P,)
    j_idx     = cache['j_idx']   # (P,)

    # ------------------------------------------------------------------ binary arrays
    B = len(binaries)
    bin_arrays = {
        'f':       np.array([b['f']             for b in binaries]),
        'Mc':      np.array([b['Mc']            for b in binaries]),
        'D_comov': np.array([b['D_comov']       for b in binaries]),
        'z':       np.array([b['z']             for b in binaries]),
        'ra':      np.array([b['ra']            for b in binaries]),
        'dec':     np.array([b['dec']           for b in binaries]),
        'psi':     np.array([b.get('psi',  0.0) for b in binaries]),
        'phi0':    np.array([b.get('phi0', 0.0) for b in binaries]),
        'iota':    np.array([b.get('iota', 0.0) for b in binaries]),
    }
    bin_edges = strain_data['bin_edges']
    delta_fs  = np.array([
        bin_edges[b['freq_bin'] + 1] - bin_edges[b['freq_bin']] for b in binaries
    ])  # (B,)

    # ------------------------------------------------------------------ per-pair Tspan
    tspans      = np.array([p.toas.max() - p.toas.min() for p in pulsars])  # (N,)
    tspan_pairs = np.minimum(tspans[i_idx], tspans[j_idx])                  # (P,)

    # Group pairs by unique Tspan to avoid a per-pair Python loop
    unique_tspans, pair_group_ids = np.unique(tspan_pairs, return_inverse=True)
    # pair_group_ids[p] = index into unique_tspans for pair p

    # ------------------------------------------------------------------ chunk sizing
    N = len(pulsars)
    T = time_arr_npoints
    if chunk_size is None:
        bytes_per_binary = N * T * 8 * 6   # h, Sh_ii, Sh_jj, Sh_ij, Ni, Nj
        chunk_size = max(1, int(target_memory_GB * 1024**3 / bytes_per_binary))
        print(f"Auto chunk size: {chunk_size} binaries (N={N}, T={T}, "
              f"target={target_memory_GB} GB)")

    snr_sq_arr = np.zeros(B)

    for start in range(0, B, chunk_size):
        end   = min(start + chunk_size, B)
        chunk = slice(start, end)
        Bc    = end - start
        print(f"  Chunk binaries {start}–{end - 1}")

        ba = {k: v[chunk] for k, v in bin_arrays.items()}   # each (Bc,)

        # Noise base (Bc, N): red + white, no signal term yet
        fyr   = 1.0 / (365.25 * 86400)
        A_red = 10.0 ** cache['log10A_arr']
        rn    = (
            A_red**2 / (12.0 * np.pi**2)
            * (ba['f'][:, None] / fyr) ** (-cache['gamma_arr'])
            * fyr**-3.0
        )  # (Bc, N)

        N_base = np.zeros((Bc, N))
        if inc_red_noise:
            N_base += rn                              # (Bc, N)
        if inc_white_noise:
            N_base += cache['white_noise_arr']        # (Bc, N)

        # Accumulate SNR² over Tspan groups — one time grid per group
        snr_sq_chunk = np.zeros(Bc)

        for g, Tspan in enumerate(unique_tspans):
            # Which pairs belong to this Tspan group
            pair_mask = pair_group_ids == g          # (P,) boolean
            gi        = i_idx[pair_mask]             # pulsar i indices for this group
            gj        = j_idx[pair_mask]             # pulsar j indices for this group
            Pg        = pair_mask.sum()
            if Pg == 0:
                continue

            time_arr = np.linspace(0.0, Tspan, T)   # (T,)
            dt       = np.diff(time_arr)             # (T-1,)

            # Strain for all binaries × all pulsars on this time grid — (Bc, N, T)
            h = measured_strain_all_binaries_all_pulsars(ba, cache, time_arr)

            # PSD normalisation factor per binary — (Bc,)
            norm = 1.0 / (12.0 * np.pi**2 * ba['f']**3)

            # Auto-PSDs for pulsars in this group's pairs — (Bc, Pg, T)
            Sh_ii = h[:, gi, :]**2 * norm[:, None, None]   # (Bc, Pg, T)
            Sh_jj = h[:, gj, :]**2 * norm[:, None, None]   # (Bc, Pg, T)

            # Cross-PSD — (Bc, Pg, T)
            Sh_ij = h[:, gi, :] * h[:, gj, :] * norm[:, None, None]

            # Total noise: N_k(t) = N_base_k + S_h^{kk}(t)
            Ni = N_base[:, gi, None] + Sh_ii   # (Bc, Pg, T)
            Nj = N_base[:, gj, None] + Sh_jj   # (Bc, Pg, T)

            # Midpoint Riemann sum
            Sh_mid = 0.5 * (Sh_ij[:, :, :-1] + Sh_ij[:, :, 1:])   # (Bc, Pg, T-1)
            Ni_mid = 0.5 * (Ni[:, :, :-1]    + Ni[:, :, 1:])       # (Bc, Pg, T-1)
            Nj_mid = 0.5 * (Nj[:, :, :-1]    + Nj[:, :, 1:])       # (Bc, Pg, T-1)

            integrand = 2 * dt[None, None, :] * Sh_mid**2 / (Ni_mid * Nj_mid)  # (Bc, Pg, T-1)

            # Sum over pairs and time, weight by Δf
            snr_sq_chunk += delta_fs[chunk] * integrand.sum(axis=(1, 2))

        snr_sq_arr[chunk] = snr_sq_chunk

    return snr_sq_arr


def build_pulsar_cache_time_domain(pulsars, pulsar_noise_params):
    """
    Pulsar cache for the time-domain vectorised SNR² path.

    Identical to `build_pulsar_cache` but also stores raw position arrays
    for use by `antenna_response_vectorised`.
    """
    N = len(pulsars)

    white_noise_arr = np.array([
        pulsar_white_noise_psd(sigma_t=np.median(p.toaerrs), delta_t=1.0 / 20.0)
        for p in pulsars
    ])  # (N,)
    log10A_arr = np.array([pulsar_noise_params[p.name]['red_noise']['log10_A'] for p in pulsars])
    gamma_arr  = np.array([pulsar_noise_params[p.name]['red_noise']['gamma']   for p in pulsars])
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
    pulsar_noise_params: dict,
    strain_data:         dict,
    target_SNR:          float,
    time_arr_npoints:    int   = 1_001,
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
        pulsar_noise_params = pulsar_noise_params,
        strain_data         = strain_data,
        time_arr_npoints    = time_arr_npoints,
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