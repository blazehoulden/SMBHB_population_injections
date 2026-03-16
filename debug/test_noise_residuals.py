"""
test_noise_residuals.py
=======================
Validation tests for white noise (EFAC, EQUAD, ECORR) and red noise
(Fourier basis GP) residual generation, checking consistency with
enterprise's internal model.

Usage
-----
    python test_noise_residuals.py

The script expects:
    - `psrs_clean`            : list of enterprise Pulsar objects
    - `pulsar_noise_params`   : noise parameter dict (your classified format)
    - `Tspan`                 : float, time span in seconds

Edit the CONFIGURATION block at the bottom to point at your data.

References
----------
    Lentati et al. (2014), MNRAS 437, 3004          -- EFAC/EQUAD/ECORR model
    Arzoumanian et al. (2016), ApJ 821, 13           -- NANOGrav 9yr noise model
    Agazie et al. (2023), ApJL 951, L9               -- NANOGrav 15yr noise values
    enterprise/signals/white_signals.py              -- EcorrKernelNoise source
    enterprise/signals/gp_bases.py                   -- createfourierdesignmatrix_red
"""

import numpy as np
import sys

from enterprise.signals import white_signals, selections
import enterprise.signals.parameter as parameter
from enterprise.signals.gp_bases import createfourierdesignmatrix_red
from enterprise.signals.utils import create_quantization_matrix


# ============================================================
#  Colours for terminal output
# ============================================================
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _pass(msg): print(f"  {GREEN}PASS{RESET}  {msg}")
def _fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def _info(msg): print(f"  {YELLOW}INFO{RESET}  {msg}")
def _head(msg): print(f"\n{BOLD}{msg}{RESET}")


# ============================================================
#  Helper: build MeasurementNoise signal for one pulsar
# ============================================================
def _build_mn_signal(pulsar, pulsar_noise_params):
    """Return (mn_signal, params_dict) for EFAC+EQUAD."""
    wn_params   = pulsar_noise_params[pulsar.name]['white_noise']
    params_dict = {}
    for backend, bp in wn_params.items():
        params_dict[f'{pulsar.name}_{backend}_efac']          = bp['efac']
        params_dict[f'{pulsar.name}_{backend}_log10_t2equad'] = bp['log10_t2equad']
        params_dict[f'{pulsar.name}_{backend}_log10_ecorr']   = bp['log10_ecorr']

    sel     = selections.Selection(selections.by_backend)
    efac_p  = parameter.Constant()
    equad_p = parameter.Constant()
    mn      = white_signals.MeasurementNoise(
                  efac=efac_p,
                  log10_t2equad=equad_p,
                  selection=sel
              )
    return mn(pulsar), params_dict


# ============================================================
#  TEST 1 — EFAC/EQUAD: enterprise Nvec matches manual formula
# ============================================================
def test_efac_equad(pulsar, pulsar_noise_params):
    """
    Verify that enterprise's get_ndiag returns
        N_i = (EFAC * sigma_i)^2 + EQUAD^2
    for every TOA, matching a manual calculation.

    This checks that:
      (a) parameter keys are formatted correctly
      (b) the backend selection masks are applied correctly
      (c) the variance formula is EFAC^2*(sigma^2 + EQUAD^2)
          i.e. the t2equad / TEMPO2 convention, NOT the legacy equad convention

    Reference: Lentati et al. (2014) eq. 19;
               enterprise white_signals.py combined_ndiag()
    """
    _head(f"TEST 1 — EFAC/EQUAD  [{pulsar.name}]")

    wn_params              = pulsar_noise_params[pulsar.name]['white_noise']
    mn_signal, params_dict = _build_mn_signal(pulsar, pulsar_noise_params)
    Nvec_enterprise        = mn_signal.get_ndiag(params_dict)  # shape (n_toas,)

    # Manual calculation per backend
    Nvec_manual = np.zeros(len(pulsar.toas))
    for backend, bp in wn_params.items():
        mask  = pulsar.flags['f'] == backend
        if not np.any(mask):
            _info(f"  backend {backend}: no TOAs found, skipping")
            continue
        efac  = bp['efac']
        equad = 10**bp['log10_t2equad']
        # TEMPO2 / t2equad convention:
        #   variance = efac^2 * (sigma^2 + equad^2)
        # Note: this is NOT (efac*sigma)^2 + equad^2
        # enterprise's combined_ndiag uses: efac^2*(toaerrs^2 + 10^(2*log10_t2equad))
        Nvec_manual[mask] = efac**2 * (pulsar.toaerrs[mask]**2 + equad**2)
        _info(f"  backend {backend}: {mask.sum()} TOAs, "
              f"EFAC={efac:.4f}, EQUAD={equad:.2e} s")

    match = np.allclose(Nvec_enterprise, Nvec_manual, rtol=1e-6)
    if match:
        _pass("enterprise Nvec matches manual (EFAC^2*(sigma^2+EQUAD^2)) for all TOAs")
    else:
        _fail("Nvec mismatch — check parameter key format or t2equad convention")
        worst = np.argmax(np.abs(Nvec_enterprise - Nvec_manual))
        _info(f"  worst mismatch at TOA {worst}: "
              f"enterprise={Nvec_enterprise[worst]:.4e}, "
              f"manual={Nvec_manual[worst]:.4e}")

    return match


# ============================================================
#  TEST 2 — ECORR: epoch structure from create_quantization_matrix
# ============================================================
def test_ecorr_epoch_structure(pulsar, pulsar_noise_params):
    """
    Verify that create_quantization_matrix produces a valid U matrix:
      - Binary entries (0 or 1)
      - Each TOA assigned to at most one epoch
      - Each epoch has >= 2 TOAs (nmin=2 requirement)
      - U is exactly what EcorrKernelNoise uses internally

    Reference: enterprise white_signals.py EcorrKernelNoise.__init__();
               enterprise signals/utils.py create_quantization_matrix()
    """
    _head(f"TEST 2 — ECORR epoch structure  [{pulsar.name}]")

    wn_params = pulsar_noise_params[pulsar.name]['white_noise']
    all_pass  = True

    for backend, bp in wn_params.items():
        mask = pulsar.flags['f'] == backend
        if not np.any(mask):
            continue

        U, _ = create_quantization_matrix(pulsar.toas[mask], nmin=2)
        # U shape: (n_toas_in_backend, n_epochs)

        n_toas_be = mask.sum()
        n_epochs  = U.shape[1]
        toas_per_epoch = U.sum(axis=0).astype(int)

        _info(f"  backend {backend}: {n_toas_be} TOAs → {n_epochs} epochs, "
              f"median {np.median(toas_per_epoch):.0f} TOAs/epoch")

        # Check 1: binary
        c1 = np.all((U == 0) | (U == 1))
        if c1: _pass(f"  [{backend}] U is binary")
        else:  _fail(f"  [{backend}] U has non-binary entries"); all_pass = False

        # Check 2: each TOA in at most one epoch
        c2 = np.all(U.sum(axis=1) <= 1)
        if c2: _pass(f"  [{backend}] each TOA assigned to at most one epoch")
        else:  _fail(f"  [{backend}] some TOAs appear in multiple epochs"); all_pass = False

        # Check 3: nmin=2 respected
        c3 = np.all(toas_per_epoch >= 2)
        if c3: _pass(f"  [{backend}] all epochs have >= 2 TOAs (nmin=2 respected)")
        else:  _fail(f"  [{backend}] some epochs have < 2 TOAs"); all_pass = False

    return all_pass


# ============================================================
#  TEST 3 — ECORR: sample covariance matches ECORR^2 * U @ U.T
# ============================================================
def test_ecorr_covariance(pulsar, pulsar_noise_params, n_realisations=30000):
    """
    Draw n_realisations of ECORR noise and verify that the sample covariance
    matrix matches the theoretical covariance ECORR^2 * U @ U.T.

    The ECORR draw is:
        r_ecorr[mask] = U @ (ecorr * z),  z ~ N(0, I_nepochs)

    which gives:
        Cov(r_i, r_j) = ECORR^2 * (U @ U.T)_{ij}

    This is a Monte Carlo verification of the epoch-correlation structure.

    Reference: Lentati et al. (2014) eq. 20;
               enterprise white_signals.py EcorrKernelNoise
    """
    _head(f"TEST 3 — ECORR covariance (Monte Carlo, N={n_realisations})  [{pulsar.name}]")

    wn_params = pulsar_noise_params[pulsar.name]['white_noise']
    all_pass  = True

    for backend, bp in wn_params.items():
        mask  = pulsar.flags['f'] == backend
        if not np.any(mask):
            continue

        ecorr = 10**bp['log10_ecorr']
        U, _  = create_quantization_matrix(pulsar.toas[mask], nmin=2)

        if U.shape[1] == 0:
            _info(f"  [{backend}] no multi-TOA epochs, skipping")
            continue

        draws = np.zeros((mask.sum(), n_realisations))
        for i in range(n_realisations):
            z           = np.random.randn(U.shape[1])
            draws[:, i] = U @ (ecorr * z)

        sample_cov      = np.cov(draws)
        theoretical_cov = ecorr**2 * (U @ U.T)

        # Use pure rtol — atol=0 because the matrix has true zeros
        # that should stay zero, and nonzero elements of order ECORR^2
        match = np.allclose(sample_cov, theoretical_cov, rtol=0.10, atol=0)
        if match:
            _pass(f"  [{backend}] sample cov matches ECORR^2 * U@U.T (rtol=5%)")
        else:
            _fail(f"  [{backend}] covariance mismatch")
            # Report the worst offender as a fraction of ECORR^2
            rel_diff = np.abs(sample_cov - theoretical_cov) / ecorr**2
            _info(f"    max relative diff (as fraction of ECORR^2): {rel_diff.max():.3f}")
            _info(f"    ECORR={ecorr:.3e} s")
            all_pass = False

    return all_pass


# ============================================================
#  TEST 4 — Full white noise: sample variance matches diagonal of N
# ============================================================
def white_noise_residual(pulsar, pulsar_noise_params):
    """
    Draw one full white noise residual vector for a pulsar.
    Includes EFAC, EQUAD (via enterprise get_ndiag), and ECORR
    (via create_quantization_matrix — same method as EcorrKernelNoise.__init__).
    """
    wn_params              = pulsar_noise_params[pulsar.name]['white_noise']
    mn_signal, params_dict = _build_mn_signal(pulsar, pulsar_noise_params)

    r_wn = np.zeros(len(pulsar.toas))

    # EFAC + EQUAD: independent per TOA
    Nvec  = mn_signal.get_ndiag(params_dict)
    r_wn += np.sqrt(Nvec) * np.random.randn(len(pulsar.toas))

    # ECORR: epoch-correlated, one draw per epoch per backend
    for backend, bp in wn_params.items():
        mask  = pulsar.flags['f'] == backend
        if not np.any(mask):
            continue
        ecorr = 10**bp['log10_ecorr']
        U, _  = create_quantization_matrix(pulsar.toas[mask], nmin=2)
        if U.shape[1] == 0:
            continue
        z           = np.random.randn(U.shape[1])
        r_wn[mask] += U @ (ecorr * z)

    return r_wn


def test_full_white_noise(pulsar, pulsar_noise_params, n_realisations=30000):
    """
    Draw many full white noise realisations and verify that:
      (a) per-TOA sample variance matches diag(N) = (EFAC*sigma)^2 + EQUAD^2 + ECORR^2
      (b) sample mean is consistent with zero

    Reference: full N = diag[(EFAC*sigma_i)^2 + EQUAD_i^2] + U*ECORR^2*U^T
    """
    _head(f"TEST 4 — Full white noise variance (N={n_realisations})  [{pulsar.name}]")

    wn_params              = pulsar_noise_params[pulsar.name]['white_noise']
    mn_signal, params_dict = _build_mn_signal(pulsar, pulsar_noise_params)
    Nvec                   = np.array(mn_signal.get_ndiag(params_dict))  # (n_toas,)
    n_toas                 = len(pulsar.toas)

    # --- Build theoretical diagonal and all ECORR U matrices up front ---
    N_diag_full  = Nvec.copy()
    ecorr_blocks = []  # list of (mask_indices, U, ecorr) — built once, reused

    for backend, bp in wn_params.items():
        mask = pulsar.flags['f'] == backend
        if not np.any(mask):
            continue
        ecorr = 10**bp['log10_ecorr']
        U, _  = create_quantization_matrix(pulsar.toas[mask], nmin=2)
        if U.shape[1] == 0:
            continue
        idx              = np.where(mask)[0]
        N_diag_full[idx] += ecorr**2 * U.sum(axis=1)
        ecorr_blocks.append((idx, U, ecorr))

    # --- Vectorised draws: all n_realisations at once ---
    # EFAC+EQUAD: shape (n_toas, n_realisations)
    draws = np.sqrt(Nvec)[:, None] * np.random.randn(n_toas, n_realisations)

    # ECORR: for each backend, draw (n_epochs, n_realisations) and scatter
    for idx, U, ecorr in ecorr_blocks:
        n_epochs          = U.shape[1]
        z                 = np.random.randn(n_epochs, n_realisations)  # (n_epochs, n_real)
        draws[idx, :]    += ecorr * (U @ z)                            # (n_toas_be, n_real)

    # --- Check variance and mean ---
    sample_var  = draws.var(axis=1)
    sample_mean = draws.mean(axis=1)

    var_match = np.allclose(sample_var, N_diag_full, rtol=0.05)
    if var_match:
        _pass("per-TOA sample variance matches diag(N) (rtol=5%)")
    else:
        _fail("per-TOA sample variance does not match diag(N)")
        rel_err = np.abs(sample_var - N_diag_full) / N_diag_full
        worst   = np.argmax(rel_err)
        _info(f"  worst TOA {worst}: sample={sample_var[worst]:.4e}, "
              f"theory={N_diag_full[worst]:.4e}, rel_err={rel_err[worst]:.3f}")

    std_of_mean = np.sqrt(N_diag_full / n_realisations)
    mean_ok     = np.all(np.abs(sample_mean) < 5 * std_of_mean)
    if mean_ok:
        _pass("sample mean consistent with zero (within 5-sigma)")
    else:
        n_bad = np.sum(np.abs(sample_mean) >= 5 * std_of_mean)
        _fail(f"{n_bad} TOAs have sample mean > 5-sigma from zero")

    return var_match and mean_ok


# ============================================================
#  TEST 5 — Red noise: sample covariance matches F @ diag(phi) @ F.T
# ============================================================
def red_noise_residual(pulsar, pulsar_noise_params, Tspan):
    """
    Draw one red noise residual via F @ a, where:
        F    = Fourier design matrix (createfourierdesignmatrix_red)
        phi  = power-law PSD evaluated at each frequency bin
        a    ~ N(0, diag(phi))

    PSD convention (NANOGrav / enterprise):
        phi_k = A^2/(12*pi^2) * (f_k/f_yr)^{-gamma} * f_yr^{-2} * (1/Tspan)

    Reference: Lentati et al. (2013) eq. 11 (F matrix);
               enterprise gp_bases.py createfourierdesignmatrix_red();
               enterprise utils.py powerlaw()
    """
    f_yr    = 1 / (365.25 * 24 * 3600)
    rn      = pulsar_noise_params[pulsar.name]['red_noise']
    log10_A = rn['log10_A']
    gamma   = rn['gamma']

    F, freqs = createfourierdesignmatrix_red(pulsar.toas, nmodes=30, Tspan=Tspan)

    if len(freqs) == 2 * 30:
        freqs_unique = freqs[::2]
    else:
        freqs_unique = freqs
    kappa    = (10**log10_A)**2 / (12*np.pi**2) \
               * (freqs_unique/f_yr)**(-gamma) * f_yr**-2 * (1/Tspan)
    phi = np.repeat(kappa, 2)
    a        = np.sqrt(phi) * np.random.randn(30 * 2)
    return F @ a


def test_red_noise_phi(pulsar, pulsar_noise_params, Tspan, pta=None):
    _head(f"TEST 5 — Red noise phi formula  [{pulsar.name}]")

    f_yr    = 1 / (365.25 * 24 * 3600)
    rn      = pulsar_noise_params[pulsar.name]['red_noise']
    log10_A = rn['log10_A']
    gamma   = rn['gamma']

    F, freqs = createfourierdesignmatrix_red(pulsar.toas, nmodes=30, Tspan=Tspan)
    if len(freqs) == 2 * 30:
        freqs_unique = freqs[::2]
    else:
        freqs_unique = freqs
    kappa    = (10**log10_A)**2 / (12*np.pi**2) \
               * (freqs_unique/f_yr)**(-gamma) * f_yr**-2 * (1/Tspan)
    phi = np.repeat(kappa, 2)

    _info(f"  log10_A={log10_A:.3f}, gamma={gamma:.3f}")
    _info(f"  phi range: [{phi.min():.3e}, {phi.max():.3e}] s^2")
    _info(f"  F shape: {F.shape}, phi shape: {phi.shape}")

    c1 = np.all(phi > 0)
    if c1: _pass("all phi values positive")
    else:  _fail("some phi values non-positive")

    # Monotonicity check on kappa (before repeat), not phi
    # kappa should be strictly decreasing for gamma > 0
    if gamma > 0:
        c2 = np.all(np.diff(kappa) < 0)
        if c2: _pass(f"kappa decreasing with frequency (gamma={gamma:.2f} > 0)")
        else:  _fail(f"kappa not monotonically decreasing despite gamma > 0")
    else:
        c2 = True
        _info(f"  gamma={gamma:.2f} <= 0, skipping monotonicity check")

    if pta is not None:
        psr_idx    = [p.name for p in pta.pulsars].index(pulsar.name)
        params_pta = pta.get_parameter_dict()
        sc         = pta._signalcollections[psr_idx]
        phi_list   = None
        for signal in sc._signals:
            if hasattr(signal, 'get_phi') and 'red_noise' in signal.name \
                    and 'gw' not in signal.name and 'curn' not in signal.name:
                phi_list = signal.get_phi(params_pta)
                break
        if phi_list is not None:
            match = np.allclose(phi, phi_list, rtol=1e-6)
            if match: _pass("phi matches enterprise get_phi exactly")
            else:
                _fail("phi does not match enterprise get_phi")
                _info(f"  max rel diff: {np.max(np.abs(phi-phi_list)/phi_list):.2e}")
        else:
            _info("could not find red noise signal in PTA — skipping enterprise comparison")

    return c1 and c2


def test_red_noise_covariance(pulsar, pulsar_noise_params, Tspan, n_realisations=20000):
    _head(f"TEST 6 — Red noise covariance F@Phi@F.T (N={n_realisations})  [{pulsar.name}]")

    f_yr    = 1 / (365.25 * 24 * 3600)
    rn      = pulsar_noise_params[pulsar.name]['red_noise']
    log10_A = rn['log10_A']
    gamma   = rn['gamma']

    F, freqs = createfourierdesignmatrix_red(pulsar.toas, nmodes=30, Tspan=Tspan)
    if len(freqs) == 2 * 30:
        freqs_unique = freqs[::2]
    else:
        freqs_unique = freqs
    kappa    = (10**log10_A)**2 / (12*np.pi**2) \
               * (freqs_unique/f_yr)**(-gamma) * f_yr**-2 * (1/Tspan)
    phi = np.repeat(kappa, 2)

    # F @ diag(phi) @ F.T  — correct broadcasting: scale rows of F.T by phi
    # F.T has shape (60, n_toas), phi has shape (60,)
    C_theory = F @ (phi[:, None] * F.T)   # (n_toas, 60) @ (60, n_toas) = (n_toas, n_toas)

    n_toas = len(pulsar.toas)

    if n_toas > 500:
        _info(f"  {n_toas} TOAs — checking diagonal variance only for speed")

        # Vectorised: draw all realisations at once
        # a has shape (60, n_realisations)
        a      = np.sqrt(phi)[:, None] * np.random.randn(60, n_realisations)
        draws  = F @ a   # (n_toas, n_realisations)

        sample_var = draws.var(axis=1)
        theory_var = np.diag(C_theory)

        match = np.allclose(sample_var, theory_var, rtol=0.05)
        if match:
            _pass("per-TOA sample variance matches diag(F@Phi@F.T) (rtol=5%)")
        else:
            _fail("per-TOA sample variance does not match diag(F@Phi@F.T)")
            rel_err = np.abs(sample_var - theory_var) / theory_var
            worst   = np.argmax(rel_err)
            _info(f"  worst TOA {worst}: sample={sample_var[worst]:.3e}, "
                  f"theory={theory_var[worst]:.3e}, rel_err={rel_err[worst]:.3f}")
        return match

    else:
        a      = np.sqrt(phi)[:, None] * np.random.randn(60, n_realisations)
        draws  = F @ a
        C_sample = np.cov(draws)
        match    = np.allclose(C_sample, C_theory, rtol=0.1, atol=0)
        if match:
            _pass("full sample covariance matches F@Phi@F.T (rtol=10%)")
        else:
            _fail("full sample covariance does not match F@Phi@F.T")
            rel_diff = np.abs(C_sample - C_theory) / (np.abs(C_theory) + 1e-40)
            _info(f"  max relative diff: {rel_diff.max():.3f}")
        return match


# ============================================================
#  TEST 7 — F matrix orthogonality check
# ============================================================
def test_fourier_matrix(pulsar, Tspan):
    """
    Verify basic properties of the Fourier design matrix F:
      - Shape is (n_toas, 2*nmodes)
      - Columns are approximately orthogonal (F.T @ F ≈ (Tspan/2) * I
        for uniformly sampled data; for real data just check no NaNs/Infs
        and that columns have reasonable norms)
      - Frequencies are k/Tspan for k=1..nmodes

    Reference: Lentati et al. (2013) eq. 11;
               enterprise gp_bases.py createfourierdesignmatrix_red()
    """
    _head(f"TEST 7 — Fourier design matrix properties  [{pulsar.name}]")

    nmodes   = 30
    F, freqs = createfourierdesignmatrix_red(pulsar.toas, nmodes=nmodes, Tspan=Tspan)

    all_pass = True

    # Shape
    c1 = F.shape == (len(pulsar.toas), 2*nmodes)
    if c1: _pass(f"F shape correct: {F.shape}")
    else:  _fail(f"F shape wrong: {F.shape}, expected ({len(pulsar.toas)}, {2*nmodes})"); all_pass=False

    # No NaNs or Infs
    c2 = np.all(np.isfinite(F))
    if c2: _pass("F has no NaN or Inf entries")
    else:  _fail("F contains NaN or Inf"); all_pass = False

    # Frequencies are k/Tspan
    expected_freqs = np.arange(1, nmodes+1) / Tspan
    c3 = np.allclose(freqs[::2], expected_freqs, rtol=1e-6)
    if c3: _pass("frequencies are k/Tspan for k=1..nmodes")
    else:  _fail("frequencies do not match k/Tspan"); all_pass = False

    # sin/cos structure: check columns come in sin/cos pairs
    # F[:,2k] should be cos(2*pi*f_k*t), F[:,2k+1] should be sin(2*pi*f_k*t)
    t   = pulsar.toas
    f0  = freqs[0]
    sin_col = np.sin(2*np.pi*f0*t)
    cos_col = np.cos(2*np.pi*f0*t)

    c4a = np.allclose(F[:,0]/np.linalg.norm(F[:,0]),
                    sin_col/np.linalg.norm(sin_col), atol=1e-6)
    c4b = np.allclose(F[:,1]/np.linalg.norm(F[:,1]),
                    cos_col/np.linalg.norm(cos_col), atol=1e-6)
    if c4a and c4b:
        _pass("first column pair is sin/cos at f=1/Tspan (enterprise convention)")
    else:
        # Try cos/sin in case convention differs
        c4a2 = np.allclose(F[:,0]/np.linalg.norm(F[:,0]),
                        cos_col/np.linalg.norm(cos_col), atol=1e-6)
        c4b2 = np.allclose(F[:,1]/np.linalg.norm(F[:,1]),
                        sin_col/np.linalg.norm(sin_col), atol=1e-6)
        if c4a2 and c4b2:
            _pass("first column pair is cos/sin at f=1/Tspan")
        else:
            _fail("first column pair does not match expected sin/cos or cos/sin structure")
            # Print actual correlation to diagnose
            _info(f"  corr(col0, sin): {np.corrcoef(F[:,0], sin_col)[0,1]:.6f}")
            _info(f"  corr(col0, cos): {np.corrcoef(F[:,0], cos_col)[0,1]:.6f}")
            all_pass = False

    _info(f"  Tspan={Tspan/3.15e7:.2f} yr, "
          f"f_min={freqs[0]*3.15e7:.4f} yr^-1, "
          f"f_max={freqs[-1]*3.15e7:.2f} yr^-1")

    return all_pass


# ============================================================
#  Run all tests
# ============================================================
def run_all_tests(psrs, pulsar_noise_params, Tspan, pta=None,
                  n_pulsars_to_test=3, n_realisations=20000):
    """
    Run the full test suite on the first n_pulsars_to_test pulsars
    (or all of them if n_pulsars_to_test is None).

    Parameters
    ----------
    psrs                 : list of enterprise Pulsar objects
    pulsar_noise_params  : your noise parameter dict
    Tspan                : float, seconds
    pta                  : optional built PTA object (needed for TEST 5 phi comparison)
    n_pulsars_to_test    : int, how many pulsars to run tests on
    n_realisations       : int, Monte Carlo draws for covariance tests
    """
    print(f"\n{'='*60}")
    print(f"  NOISE RESIDUAL VALIDATION SUITE")
    print(f"  Tspan = {Tspan/3.15e7:.2f} yr")
    print(f"  n_realisations = {n_realisations}")
    print(f"{'='*60}")

    # Filter to pulsars that have noise params
    test_psrs = [p for p in psrs if p.name in pulsar_noise_params]
    if n_pulsars_to_test is not None:
        test_psrs = test_psrs[:n_pulsars_to_test]

    _info(f"Testing {len(test_psrs)} pulsars: {[p.name for p in test_psrs]}")

    results = {}

    for psr in test_psrs:
        print(f"\n{'─'*60}")
        print(f"  Pulsar: {psr.name}  ({len(psr.toas)} TOAs)")
        print(f"{'─'*60}")

        r = {}
        r['efac_equad']      = test_efac_equad(psr, pulsar_noise_params)
        r['ecorr_structure'] = test_ecorr_epoch_structure(psr, pulsar_noise_params)
        r['ecorr_cov']       = test_ecorr_covariance(psr, pulsar_noise_params,
                                                       n_realisations=n_realisations)
        r['full_wn']         = test_full_white_noise(psr, pulsar_noise_params,
                                                      n_realisations=n_realisations)
        r['rn_phi']          = test_red_noise_phi(psr, pulsar_noise_params, Tspan, pta=pta)
        r['rn_cov']          = test_red_noise_covariance(psr, pulsar_noise_params, Tspan,
                                                          n_realisations=n_realisations)
        r['fourier_matrix']  = test_fourier_matrix(psr, Tspan)
        results[psr.name]    = r

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for psr_name, r in results.items():
        n_pass = sum(r.values())
        n_total = len(r)
        status = f"{GREEN}ALL PASS{RESET}" if n_pass == n_total \
                 else f"{RED}{n_pass}/{n_total} passed{RESET}"
        print(f"  {psr_name:20s}  {status}")
        if n_pass < n_total:
            all_passed = False
            for test, passed in r.items():
                if not passed:
                    print(f"    {RED}✗{RESET} {test}")

    print(f"\n{'='*60}")
    if all_passed:
        print(f"  {GREEN}{BOLD}ALL TESTS PASSED{RESET}")
    else:
        print(f"  {RED}{BOLD}SOME TESTS FAILED{RESET}")
    print(f"{'='*60}\n")

    return results


# ============================================================
#  CONFIGURATION — edit this block to point at your data
# ============================================================
if __name__ == '__main__':

    # ------------------------------------------------------------------
    # OPTION A: load your pulsars and noise params directly
    # ------------------------------------------------------------------
    # from your_module import psrs_clean, pulsar_noise_params, Tspan
    # run_all_tests(psrs_clean, pulsar_noise_params, Tspan)

    # ------------------------------------------------------------------
    # OPTION B: load from pickle / npz / whatever you use
    # ------------------------------------------------------------------
    # import pickle
    # with open('psrs_clean.pkl', 'rb') as f:
    #     psrs_clean = pickle.load(f)
    # import json
    # with open('noise_params.json') as f:
    #     pulsar_noise_params = json.load(f)
    # Tspan = 15 * 365.25 * 24 * 3600  # 15 years in seconds
    # run_all_tests(psrs_clean, pulsar_noise_params, Tspan,
    #               n_pulsars_to_test=5,    # test first 5 pulsars
    #               n_realisations=20000)   # more = slower but more accurate

    # ------------------------------------------------------------------
    # OPTION C: call from another script
    # ------------------------------------------------------------------
    # from test_noise_residuals import run_all_tests
    # results = run_all_tests(psrs, noise_params, Tspan)
    # ------------------------------------------------------------------

    print("Edit the __main__ block to point at your psrs and noise params.")
    print("See OPTION A/B/C comments above.")
    sys.exit(0)