import numpy as np
from enterprise_extensions.frequentist.Fe_statistic import innerProduct_rr
from enterprise_extensions.deterministic import cw_delay
from optimal_SNR_calc import measured_strain_all_binaries_all_pulsars

def compute_cgw_signal_enterprise(psr, binary):
    """Compute CGW timing residual signal for a single pulsar using enterprise."""
    s_a = cw_delay(
            toas=psr.toas,
            pos=psr.pos,
            pdist=psr.pdist,
            cos_gwtheta=np.cos(np.pi / 2.0 - binary.dec),
            gwphi=binary.ra,
            cos_inc=np.cos(binary.iota),
            log10_mc=np.log10(binary.Mc),
            log10_fgw=np.log10(binary.f),
            # log10_dist=np.log10(lum_dist),
            log10_h=np.log10(binary.h0),
            phase0=binary.phi0,
            psi=binary.psi,
            psrTerm=False,
        )
    return s_a


def compute_cgw_optimal_snr(psrs, pta, noise_params, binary):
    """
    Compute matched-filter SNR for a single CGW source.

    Parameters
    ----------
    psrs         : list of enterprise Pulsar objects (e.g. psrs_clean),
                   ordered consistently with how the PTA was constructed
    pta          : enterprise PTA object
    noise_params : noise parameter dict (e.g. ML noise params from OS run)
    binary       : binary object with attributes (Mc, f, h0, ra, dec, iota,
                   psi, phi0, D_comov, z)

    Returns
    -------
    snr : float
    """
    phiinvs = pta.get_phiinv(noise_params, logdet=False)
    TNTs    = pta.get_TNT(noise_params)
    Ts      = pta.get_basis()
    Nvecs   = pta.get_ndiag(noise_params)

    # pta.pulsars is a list of strings (names); zip against the actual objects
    psr_map = {psr.name: psr for psr in psrs}

    rho_sq = 0.0
    for psr_name, Nvec, TNT, phiinv, T in zip(pta.pulsars, Nvecs, TNTs, phiinvs, Ts):
        psr = psr_map[psr_name]
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        s_a = compute_cgw_signal_enterprise(psr, binary)
        rho_sq += innerProduct_rr(s_a, s_a, Nvec, T, TNT, Sigma)

    return np.sqrt(rho_sq)

def compute_cgw_snr_optimal_population(
    psrs:          list,
    chol_factors:  dict,
    population:    list,
    verbose_top_n: int = 0,
) -> list[float]:
    """
    Compute CGW SNR for each binary in population.
    Cholesky factors must be precomputed once via
    get_per_pulsar_covariance_from_population.
    """
    return [
        compute_cgw_snr_single(
            psrs         = psrs,
            chol_factors = chol_factors,
            binary       = binary,
            verbose      = (i < verbose_top_n),
        )
        for i, binary in enumerate(population)
    ]



def compute_population_gwb_psd(
    binaries:      list,
    psrs:          list,
    pulsar_cache:  dict,
    time_arr:      np.ndarray,
    freq_axis:     np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the GWB timing-residual PSD from a discrete SMBHB population
    by incoherently summing |h_f|^2 over all binaries.

    The GWB characteristic strain spectrum is:
        h_c^2(f) = sum_b h_0,b^2 * f_b * T_obs   [dimensionless]

    The corresponding timing-residual one-sided PSD is:
        S_GWB(f) = h_c^2(f) / (12 pi^2 f^3)      [s^3]

    which is what enters the noise covariance C_a = N_a + S_RN,a + S_GWB,a.

    References
    ----------
    Phinney (2001) ApJ 153, L1  — incoherent sum over population
    Sesana, Vecchio & Colacino (2008) MNRAS 390, 192 — discrete population GWB
    Agazie et al. 2023 ApJ 951 L8 Eq. (7) — timing residual PSD convention

    Parameters
    ----------
    binaries      : list of binary objects (full population)
    psrs          : list of enterprise Pulsar objects
    pulsar_cache  : output of build_pulsar_cache_time_domain
    time_arr      : time array used in measured_strain_all_binaries_all_pulsars
    freq_axis     : optional — if provided, S_GWB is evaluated on this grid
                    (useful for matching to your noise PSD interpolators)

    Returns
    -------
    freqs   : (F,) frequency array [Hz]
    S_GWB   : (N, F) timing-residual PSD per pulsar [s^3]
              (isotropic background so identical for all pulsars,
               but returned per-pulsar for direct use in C_a)
    """
    B = len(binaries)
    N = len(psrs)

    bin_arrays = {
        'f':           np.array([b.f                     for b in binaries]),
        'Mc':          np.array([b.Mc                    for b in binaries]),
        'D_comov':     np.array([b.D_comov               for b in binaries]),
        'z':           np.array([b.z                     for b in binaries]),
        'ra':          np.array([b.ra                    for b in binaries]),
        'dec':         np.array([b.dec                   for b in binaries]),
        'psi':         np.array([b.psi                   for b in binaries]),
        'phi0':        np.array([b.phi0                  for b in binaries]),
        'iota':        np.array([b.iota                  for b in binaries]),
    }

    # --- Get full-spectrum strain from your existing function ---
    # h_f:       (B, N, F)  complex strain at each frequency bin
    # bin_freqs: (B, F)     same freq axis for all binaries (full_spectrum=True)
    # delta_f:   (B,)       uniform bin width
    h_f, bin_freqs, delta_f_arr = measured_strain_all_binaries_all_pulsars(
        bin_arrays  = bin_arrays,
        pulsar_cache= pulsar_cache,
        time_arr    = time_arr,
        full_spectrum = True,
    )
    # bin_freqs is (B, F) but identical rows when full_spectrum=True
    freqs   = bin_freqs[0]       # (F,)
    delta_f = delta_f_arr[0]     # scalar

    # --- Incoherent sum: GWB PSD = sum_b |h_f_b|^2 / delta_f ---
    # h_f: (B, N, F) — antenna-pattern-weighted strain per pulsar
    # |h_f|^2 / delta_f has units of [strain^2 / Hz] = [s^2 * Hz] ... 
    # but we want timing residual PSD S_r(f) [s^3].
    #
    # The one-sided GWB timing residual PSD per pulsar is:
    #   S_GWB_a(f) = (1/T_obs) * sum_b |h_f_{a,b}(f)|^2 / (4 pi^2 f^2)
    #                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                           converts strain -> residual, see below
    #
    # Your h_f is already the antenna-pattern-weighted strain Fourier amplitude
    # with norm="forward" (divided by N_time points), so:
    #   |h_f|^2 has units of [strain^2] (dimensionless^2)
    #   divide by delta_f to get one-sided PSD in [strain^2 / Hz]
    #   divide by (2 pi f)^2 to convert strain -> timing residual
    #   the factor of 2 gives one-sided PSD
    #
    # Residual PSD: S_r(f) = S_h(f) / (4 pi^2 f^2)
    # Eq. (3) of Hazboun et al. 2019 Phys. Rev. D 100, 104028

    # |h_f|^2 summed over all binaries, per pulsar: (N, F)
    h_sq_sum = np.sum(np.abs(h_f)**2, axis=0)        # (N, F)

    # One-sided strain PSD: S_h(f) = 2 * |h_f|^2 / delta_f  [Hz^{-1}]
    S_h = 2.0 * h_sq_sum / delta_f                    # (N, F)

    # Convert to timing residual PSD: S_r(f) = S_h(f) / (4 pi^2 f^2)
    S_GWB = S_h / (4.0 * np.pi**2 * freqs[None, :]**2)   # (N, F)  [s^3]

    # --- Optionally interpolate onto a provided frequency grid ---
    if freq_axis is not None:
        from scipy.interpolate import interp1d
        S_GWB_interp = np.zeros((N, len(freq_axis)))
        for a in range(N):
            fn = interp1d(
                np.log(freqs),
                np.log(np.clip(S_GWB[a], 1e-300, None)),
                kind='linear',
                bounds_error=False,
                fill_value=(
                    np.log(np.clip(S_GWB[a, 0],  1e-300, None)),
                    np.log(np.clip(S_GWB[a, -1], 1e-300, None)),
                ),
            )
            S_GWB_interp[a] = np.exp(fn(np.log(freq_axis)))
        return freq_axis, S_GWB_interp

    return freqs, S_GWB

def get_per_pulsar_covariance_from_population(
    psrs:         list,
    pta:          object,
    noise_params: dict,
    S_GWB:        np.ndarray,   # (N, F) [s^3]
    freqs_gwb:    np.ndarray,   # (F,)   [Hz]  — must be uniform
) -> tuple[dict, dict]:
    """
    Build C_a = N_a + S_{RN,a} + S_{GWB,aa} per pulsar and precompute
    Cholesky factorisations.

    The covariance matrix elements are given by the Wiener-Khinchin theorem:

        C_a(t_i, t_j) = integral_0^inf S_total,a(f) cos(2pi f dt_ij) df

    approximated as a discrete sum over the uniform FFT frequency grid:

        C_a[i,j] = sum_k S_total,a(f_k) * cos(2pi f_k * dt[i,j]) * df

    References: Lentati et al. 2013 PRD 87 104021, Eq. (A1);
                van Haasteren & Levin 2013 PRD 88 101501, Appendix A.

    Parameters
    ----------
    psrs         : list of enterprise Pulsar objects (same order as PTA)
    pta          : enterprise PTA object (for extracting N_a and S_RN,a)
    noise_params : ML noise parameter dict
    S_GWB        : (N, F) timing-residual PSD from compute_population_gwb_psd
    freqs_gwb    : (F,) Hz — must be uniform (asserted internally)

    Returns
    -------
    cov_matrices : dict  psr_name -> (n_toa, n_toa) ndarray [s^2]
    chol_factors : dict  psr_name -> cho_factor output
    """
    # Verify uniform spacing — required for the discrete Wiener-Khinchin sum
    df = freqs_gwb[1] - freqs_gwb[0]
    assert np.allclose(np.diff(freqs_gwb), df, rtol=1e-6), \
        "freqs_gwb must be uniformly spaced."

    Nvecs   = pta.get_ndiag(noise_params)
    Ts      = pta.get_basis()
    phiinvs = pta.get_phiinv(noise_params, logdet=False)

    psr_map = {psr.name: psr for psr in psrs}

    cov_matrices = {}
    chol_factors = {}

    for a, (psr_name, Nvec, T, phiinv) in enumerate(
        zip(pta.pulsars, Nvecs, Ts, phiinvs)
    ):
        psr   = psr_map[psr_name]
        n_toa = len(psr.toas)

        # --- White noise: N_a = diag(Nvec) ---
        # Nvec contains the diagonal of the white noise covariance in s^2.
        # enterprise computes this from EFAC/EQUAD/ECORR per the noise model.
        N_a = np.diag(Nvec)   # (n_toa, n_toa) [s^2]

        # --- Intrinsic red noise: S_RN,a = T phi T^T ---
        # phi is the per-pulsar red noise prior covariance in the Fourier basis.
        # phiinv is its inverse (1D diagonal or 2D matrix depending on PTA setup).
        # NOTE: if your PTA includes a common GWB process in phiinv, that
        # component is already in phiinv. We are adding S_GWB separately (from
        # the discrete population), so the common process should NOT be in your
        # PTA. If it is, you are double-counting. Build your PTA without a
        # common GWB signal for this use case.
        if phiinv.ndim == 1:
            # Diagonal case: safe inversion element-wise
            # Guard against zeros (can occur if a pulsar has no red noise)
            phi_diag = np.where(phiinv > 0, 1.0 / phiinv, 0.0)
            phi      = np.diag(phi_diag)
        else:
            phi = np.linalg.inv(phiinv)

        S_rn = T @ phi @ T.T   # (n_toa, n_toa) [s^2]

        # --- GWB auto-covariance via Wiener-Khinchin ---
        # S_GWB[a] is the per-pulsar timing-residual PSD (N, F); shape (F,)
        S_gwb_a = S_GWB[a]   # (F,) [s^3]

        # dt[i,j] = t_i - t_j in seconds: (n_toa, n_toa)
        dt = psr.toas[:, None] - psr.toas[None, :]

        # Vectorised Wiener-Khinchin sum:
        # C_GWB[i,j] = sum_k S_gwb_a[k] * cos(2pi f_k * dt[i,j]) * df
        # Shape: freqs (F,), dt (n_toa, n_toa) -> cos term (F, n_toa, n_toa)
        # einsum contracts over k (frequency axis).
        cos_term = np.cos(
            2.0 * np.pi * freqs_gwb[:, None, None] * dt[None, :, :]
        )   # (F, n_toa, n_toa)

        C_GWB = np.einsum('k,kij->ij', S_gwb_a * df, cos_term)   # (n_toa, n_toa) [s^2]

        # --- Full covariance ---
        C_a = N_a + S_rn + C_GWB   # (n_toa, n_toa) [s^2]

        # Symmetrise to guard against floating-point asymmetry
        C_a = 0.5 * (C_a + C_a.T)

        # Regularise: add small diagonal jitter proportional to trace.
        # This guards against near-singular matrices from near-zero GWB PSD
        # at high frequencies. Value 1e-10 is much smaller than any physical
        # noise level — it does not materially affect the SNR.
        jitter = 1e-10 * np.trace(C_a) / n_toa
        C_a   += np.eye(n_toa) * jitter

        cov_matrices[psr_name] = C_a
        try:
            chol_factors[psr_name] = cho_factor(C_a, lower=True)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Cholesky factorisation failed for pulsar {psr_name}. "
                f"C_a may not be positive definite. "
                f"Min eigenvalue: {np.linalg.eigvalsh(C_a).min():.3e}. "
                f"Original error: {e}"
            )

    return cov_matrices, chol_factors

def compute_population_gwb_psd_from_psrs(
    binaries:  list,
    psrs:      list,       # enterprise Pulsar objects directly
    time_arr:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as compute_population_gwb_psd but takes enterprise Pulsar objects
    directly instead of a pulsar_cache dict.

    The only thing pulsar_cache was providing to measured_strain_all_binaries_all_pulsars
    was raj_arr and decj_arr for antenna pattern computation — both are
    directly available on enterprise Pulsar objects as psr.raj and psr.decj.
    """
    bin_arrays = {
        'f':           np.array([b.f                     for b in binaries]),
        'Mc':          np.array([b.Mc                    for b in binaries]),
        'D_comov':     np.array([b.D_comov               for b in binaries]),
        'z':           np.array([b.z                     for b in binaries]),
        'ra':          np.array([b.ra                    for b in binaries]),
        'dec':         np.array([b.dec                   for b in binaries]),
        'psi':         np.array([b.psi                   for b in binaries]),
        'phi0':        np.array([b.phi0                  for b in binaries]),
        'iota':        np.array([b.iota                  for b in binaries]),
    }

    # Build the minimal cache-like namespace from enterprise pulsar objects
    # This is all that measured_strain_all_binaries_all_pulsars needs
    pulsar_cache_minimal = {
        'raj_arr':  np.array([psr._raj  for psr in psrs]),   # (N,) radians
        'decj_arr': np.array([psr._decj for psr in psrs]),   # (N,) radians
    }

    h_f, bin_freqs, delta_f_arr = measured_strain_all_binaries_all_pulsars(
        bin_arrays    = bin_arrays,
        pulsar_cache  = pulsar_cache_minimal,
        time_arr      = time_arr,
        full_spectrum = True,
    )

    freqs   = bin_freqs[0]
    delta_f = delta_f_arr[0]

    spacing = np.diff(freqs)
    assert np.allclose(spacing, spacing[0], rtol=1e-6), \
        "FFT frequency axis is not uniformly spaced."

    h_sq_sum = np.sum(np.abs(h_f)**2, axis=0)          # (N, F)
    S_h      = 2.0 * h_sq_sum / delta_f                 # (N, F)
    S_GWB    = S_h / (4.0 * np.pi**2 * freqs[None, :]**2)  # (N, F) [s^3]

    return freqs, S_GWB

def compute_cgw_snr_optimal_single(
    psrs:          list,
    chol_factors:  dict,
    binary,
    verbose:       bool = False,
) -> float:
    """
    Optimal matched-filter SNR for a single CGW source:

        rho^2 = sum_a  s_a^T C_a^{-1} s_a

    where C_a = N_a + S_{RN,a} + S_{GWB,aa} is precomputed and Cholesky-
    factorised. Cost is O(n_toa^2) per pulsar (triangular solve), not
    O(n_toa^3) (full inversion).

    References
    ----------
    Babak & Sesana (2012) PRD 85 044034, Eq. (13)
    Ellis, Siemens & Creighton (2012) ApJ 756 175, Eq. (14)
    Rosado, Sesana & Gair (2015) MNRAS 451 2417, Section 2.2
    """
    from enterprise_extensions.deterministic import cw_delay

    rho_sq = 0.0
    for psr in psrs:
        # CGW timing residual at this pulsar (Earth term only)
        s_a = cw_delay(
            toas        = psr.toas,
            pos         = psr.pos,
            pdist       = psr.pdist,
            cos_gwtheta = np.cos(np.pi / 2.0 - binary.dec),
            gwphi       = binary.ra,
            cos_inc     = np.cos(binary.iota),
            log10_mc    = np.log10(binary.Mc),
            log10_fgw   = np.log10(binary.f),
            log10_dist  = None,
            log10_h     = np.log10(binary.h0),
            phase0      = binary.phi0,
            psi         = binary.psi,
            psrTerm     = False,
        )

        # s_a^T C_a^{-1} s_a via Cholesky solve — numerically stable
        contrib = float(np.real(s_a @ cho_solve(chol_factors[psr.name], s_a)))

        if verbose:
            print(f"  [{psr.name}]  rho^2 contribution = {max(contrib, 0.0):.6f}")

        rho_sq += max(contrib, 0.0)

    snr = np.sqrt(rho_sq)
    if verbose:
        print(f"  Total optimal SNR = {snr:.4f}")
    return snr


def compute_cgw_snr_matched_filter(
    psrs:         list,
    chol_factors: dict,
    binary,
    snr_type:     str  = "matched",   # "optimal" | "matched" | "log_likelihood"
    verbose:      bool = False,
) -> float:
    """
    Compute CGW SNR for a single source, with choice of statistic:

        "optimal"       : S/N_s   = sqrt( (s|s) )            [Gardiner+2025 Eq. 9]
        "matched"       : S/N_rho = (d|s) / sqrt( (s|s) )    [Gardiner+2025 Eq. 13]
        "log_likelihood": S/N_Lam = sqrt( 2(d|s) - (s|s) )   [Gardiner+2025 Eq. 14]

    All use the same noise-weighted inner product (a|b) = a^T C^{-1} b
    with C_a = N_a + S_RN,a + S_GWB,aa precomputed and Cholesky-factorised.

    Parameters
    ----------
    psrs         : enterprise Pulsar objects — psr.residuals is d_a
    chol_factors : precomputed Cholesky factorisations from
                   get_per_pulsar_covariance_from_population
    binary       : single binary object with CGW parameters
    snr_type     : which SNR definition to use (see above)
    verbose      : print per-pulsar contributions

    References
    ----------
    Gardiner, Becsy, Kelley & Cornish (2025) arXiv:2502.16016, Eqs. 9, 13, 14
    Ellis, Siemens & Creighton (2012) ApJ 756 175, Eq. (14) for ln Lambda
    Di Matteo et al. (2019) for matched filter expectation value derivation
    """
    from enterprise_extensions.deterministic import cw_delay

    ss = 0.0   # (s|s) — accumulates over pulsars
    ds = 0.0   # (d|s) — accumulates over pulsars

    for psr in psrs:

        # --- Injected signal at this pulsar's TOAs ---
        s_a = cw_delay(
            toas        = psr.toas,
            pos         = psr.pos,
            pdist       = psr.pdist,
            cos_gwtheta = np.cos(np.pi / 2.0 - binary.dec),
            gwphi       = binary.ra,
            cos_inc     = np.cos(binary.iota),
            log10_mc    = np.log10(binary.Mc),
            log10_fgw   = np.log10(binary.f),
            log10_dist  = None,
            log10_h     = np.log10(binary.h0),
            phase0      = binary.phi0,
            psi         = binary.psi,
            psrTerm     = False,
        )

        # --- Data: timing residuals at this pulsar ---
        # psr.residuals is d_a = s_a + n_a after your injection pipeline
        # This is what Gardiner+2025 call d — residuals after timing model fit
        d_a = psr.residuals   # shape (n_toa,) in seconds

        # --- C_a^{-1} s_a via Cholesky solve (same for both inner products) ---
        Cinv_s_a = cho_solve(chol_factors[psr.name], s_a)   # C^{-1} s

        # --- Inner products ---
        ss_a = float(np.real(s_a @ Cinv_s_a))          # (s|s)_a = s^T C^{-1} s
        ds_a = float(np.real(d_a @ Cinv_s_a))          # (d|s)_a = d^T C^{-1} s

        if verbose:
            print(f"  [{psr.name}]  (s|s)={ss_a:.4f}  (d|s)={ds_a:.4f}")

        ss += max(ss_a, 0.0)
        ds += ds_a          # (d|s) can legitimately be negative for a bad noise draw

    # --- Form the requested SNR ---
    sqrt_ss = np.sqrt(max(ss, 0.0))

    if snr_type == "optimal":
        # S/N_s = sqrt((s|s))  [Eq. 9]
        snr = sqrt_ss

    elif snr_type == "matched":
        # S/N_rho = (d|s) / sqrt((s|s))  [Eq. 13]
        # Can be negative if the noise draw is particularly bad
        snr = ds / sqrt_ss if sqrt_ss > 0 else 0.0

    elif snr_type == "log_likelihood":
        # S/N_Lambda = sqrt(2(d|s) - (s|s))  [Eq. 14]
        # Can be imaginary (i.e. argument negative) for bad noise draws —
        # return signed sqrt to preserve the sign information, which is
        # meaningful: negative means noise-only hypothesis is preferred
        val = 2.0 * ds - ss
        snr = np.sign(val) * np.sqrt(abs(val))

    else:
        raise ValueError(f"snr_type must be 'optimal', 'matched', or 'log_likelihood', got '{snr_type}'")

    if verbose:
        print(f"  (s|s)={ss:.4f}  (d|s)={ds:.4f}  SNR_{snr_type}={snr:.4f}")

    return snr