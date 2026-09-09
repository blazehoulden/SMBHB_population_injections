import numpy as np
from enterprise_extensions.frequentist.Fe_statistic import innerProduct_rr
from enterprise_extensions.deterministic import cw_delay
from optimal_SNR_calc import measured_strain_all_binaries_all_pulsars
from signal_injection import population_residuals, get_base_name, population_residuals_eccentric
from scipy.linalg import cho_factor, cho_solve
import time
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


def compute_cgw_snr_optimal_population(psrs, pta, population, raw_noise_params, parsed_noise_params, Tspan, profile=False):
    # Compute these ONCE, reuse for every binary
    if profile:
        import time
        start_time = time.time()
    phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
    TNTs    = pta.get_TNT(raw_noise_params)
    Ts      = pta.get_basis()
    Nvecs   = pta.get_ndiag(raw_noise_params)
    psr_map = {psr.name: psr for psr in psrs}

    # Pre-build Sigma matrices once (also binary-independent)
    Sigmas = [
        TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        for TNT, phiinv in zip(TNTs, phiinvs)
    ]

    if profile:
        elapsed = time.time() - start_time
        print(f"Precomputation of phiinvs, TNTs, Ts, Nvecs, Sigmas took {elapsed:.2f} seconds.")

    results = []
    for binary in population:
        rho_sq = 0.0
        for psr_name, Nvec, TNT, Sigma, T in zip(pta.pulsars, Nvecs, TNTs, Sigmas, Ts):
            psr = psr_map[psr_name]
            psr_noise_params = parsed_noise_params[psr_name]
            s_a = population_residuals(psr.toas, psr, [binary], Tspan, psr_noise_params)
            # s_a = compute_cgw_signal_enterprise(psr, binary)
            rho_sq += innerProduct_rr(s_a, s_a, Nvec, T, TNT, Sigma)
        results.append(np.sqrt(rho_sq))

    if profile:
        elapsed = time.time() - start_time
        print(f"Total SNR computation for population took {elapsed:.2f} seconds.")
    return results


from scipy.linalg import cho_factor, cho_solve, LinAlgError
import numpy as np


def compute_cgw_snr_optimal_population_fast(
    psrs,
    pta,
    population,
    raw_noise_params,
    parsed_noise_params,
    Tspan,
    profile=False,
    return_breakdown=False,
    regularise_sigma=True,
    regularisation=1e-10,
    power_tol=1e-4,
    n_max_cap=100,
):

    import time
    t0 = time.time()

    phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
    TNTs    = pta.get_TNT(raw_noise_params)
    Ts      = pta.get_basis()
    Nvecs   = pta.get_ndiag(raw_noise_params)

    psr_map = {psr.name: psr for psr in psrs}

    precomputed = []

    for psr_name, Nvec, TNT, phiinv, T in zip(
        pta.pulsars, Nvecs, TNTs, phiinvs, Ts,
    ):
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

        if regularise_sigma:
            # Small diagonal regularisation, physically equivalent to a
            # tiny additional white-noise floor; negligible effect on SNR
            # for any reasonable signal amplitude, but improves robustness
            # against marginal ill-conditioning.
            Sigma += regularisation * np.eye(Sigma.shape[0])

        cf = None
        max_reg_attempts = 10
        for reg_attempt in range(max_reg_attempts):
            try:
                cf = cho_factor(Sigma)
                break
            except LinAlgError:
                reg = regularisation * (10 ** (reg_attempt + 1))
                if reg_attempt == 0 or reg_attempt == max_reg_attempts - 1:
                    print(f"  Warning: {psr_name} Sigma not PD via Cholesky, "
                          f"increasing regularisation to {reg:.2e} "
                          f"(attempt {reg_attempt + 1}/{max_reg_attempts})")
                Sigma += reg * np.eye(Sigma.shape[0])

        if cf is None:
            # Cholesky never succeeded even at max regularisation. Rather
            # than aborting the whole run, fall back to a direct solve
            # (matches enterprise_extensions' OptimalStatistic.compute_os
            # behaviour, which only requires Sigma to be non-singular, not
            # strictly positive-definite). We wrap this in a small wrapper
            # object exposing the same interface fast_inner_product_rr_exact
            # expects from cho_solve, so downstream code is unchanged.
            print(f"  Warning: {psr_name} Sigma still not PD after "
                  f"{max_reg_attempts} regularisation attempts; falling "
                  f"back to np.linalg.solve (non-PD-safe).")

            try:
                Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    f"Could not invert Sigma for {psr_name} even via "
                    f"pseudo-inverse. Check noise parameters for this "
                    f"pulsar (likely a degenerate/duplicated noise basis "
                    f"or pathological amplitude hierarchy). Original "
                    f"error: {e}"
                )

            class _SolveFallback:
                """Mimics scipy's cho_factor/cho_solve interface using a
                precomputed pseudo-inverse, for pulsars whose Sigma is not
                strictly positive-definite."""
                def __init__(self, Minv):
                    self.Minv = Minv

            cf = _SolveFallback(Sigma_inv)

        # Get TOAs safely — handle both property and method forms
        psr_obj = psr_map[psr_name]
        try:
            toas = psr_obj.toas
            if callable(toas):
                toas = toas()
            toas = np.asarray(toas)
        except Exception:
            # Last resort: get from the underlying enterprise object
            toas = np.asarray(psr_obj._psr.toas)

        precomputed.append((
            psr_name,
            psr_obj,
            toas,
            parsed_noise_params.get(psr_name,
                parsed_noise_params.get(get_base_name(psr_name), {})),
            Nvec,
            T,
            cf,
        ))

    if profile:
        print(f"Precompute time: {time.time()-t0:.2f} s")

    def _solve(cf, x):
        if isinstance(cf, tuple):
            return cho_solve(cf, x)
        else:
            return cf.Minv @ x

    def fast_inner_product_rr_exact(x, y, Nvec, Tmat, cf):
        TNy = Nvec.solve(y, left_array=Tmat)
        TNx = Nvec.solve(x, left_array=Tmat)
        xNy = Nvec.solve(y, left_array=x)
        SigmaTNy = _solve(cf, TNy)
        return xNy - TNx.T @ SigmaTNy

    results   = np.empty(len(population), dtype=np.float64)
    breakdowns = [] if return_breakdown else None

    N_binary = len(population)
    results    = np.empty(N_binary, dtype=np.float64)
    breakdowns = [] if return_breakdown else None

    for i in range(N_binary):
        binary = population[i]   # however your candidate list yields single binaries
                                  # (e.g. a _MiniPop-like scalar-attribute object,
                                  # NOT a PopulationArrays slice)
        rho_sq     = 0.0
        per_pulsar = {} if return_breakdown else None
        n_failed   = 0

        for (psr_name, psr_obj, toas, psr_noise_params,
             Nvec, T, cf) in precomputed:

            try:
                s_a = population_residuals_eccentric(
                    toas, psr_obj, [binary], Tspan,
                    pulsar_noise_params=psr_noise_params,
                    power_tol=power_tol, n_max_cap=n_max_cap,
                )
            except Exception as e:
                n_failed += 1
                print(f"  Warning: population_residuals failed for {psr_name}: {e}")
                continue

            contrib = float(np.real(
                fast_inner_product_rr_exact(s_a, s_a, Nvec, T, cf)
            ))
            rho_sq += contrib
            if return_breakdown:
                base_name = get_base_name(psr_name)
                per_pulsar[base_name] = per_pulsar.get(base_name, 0.0) + max(contrib, 0.0)

        if n_failed == len(precomputed):
            raise RuntimeError(
                f"population_residuals_eccentric failed for ALL {n_failed} "
                f"pulsars on binary {i} — likely structural, not per-pulsar. "
                f"See warnings above."
            )

        results[i] = np.sqrt(max(rho_sq, 0.0))
        if return_breakdown:
            breakdowns.append(per_pulsar)

    if profile:
        print(f"Total runtime: {time.time()-t0:.2f} s")

    return (results, breakdowns) if return_breakdown else results
 


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
        'h0':          np.array([b.h0               for b in binaries]),
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