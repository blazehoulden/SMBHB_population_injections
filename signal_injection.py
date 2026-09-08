"""
signal_injection.py
====================
SMBHB population injection into pulsar TOAs, with eccentricity handled
via a Peters-Mathews multi-harmonic decomposition.

Design principle
----------------
The eccentric harmonic formulation reduces EXACTLY to the circular
single-harmonic formula at e=0 (see _regression_check_circular_limit at
the bottom). There is therefore only ONE injection code path: every
binary is expanded into its harmonics (n_max=2 for a circular binary,
more for eccentric ones), and everything downstream — antenna pattern,
phi0 rotation, complex amplitude, NUFFT/direct summation — operates on
the flattened (binary, harmonic) representation uniformly. For an
all-circular population this costs exactly what a dedicated circular-only
implementation would cost (N_flat = N_binary); eccentricity is included
automatically wherever present, with no branching required from the user.

Main entry points (use these):
    change_in_TOAs_days_population(psrs, population, method='auto', ...)
    change_in_TOAs_days_population_direct(psrs, population, ...)
    change_in_TOAs_days_population_nufft(psrs, population, ...)

All return: list of [psr_name, time_change_days] pairs, time_change_days
shape (N_toa,) — apply to psr.stoas yourself.

IMPORTANT — upstream dependency
--------------------------------
h0 must be finite for every binary you want injected, INCLUDING eccentric
ones. If your population-synthesis code NaN-gates h0 based on a
circularity threshold (is_circular_enough), that gate must be removed —
it is superseded by the harmonic decomposition here, which handles
arbitrary e (up to e_max=0.999) correctly. filter_valid_population below
only drops non-finite h0/f or e >= e_max; it does NOT re-apply any
circularity gate.

Key assumptions (restated from prior discussion; see harmonic functions
for detail)
------------------------------------------------------------------------
  1. Earth-term only (no pulsar-term) — unchanged from the original
     circular code. Matters more for eccentric sources since the
     eccentric pulsar-term phase can differ substantially from the
     Earth-term phase; not addressed here.
  2. Stationary osculating elements: e and f_orb constant per binary over
     the dataset span (no periastron advance / GW-driven decay within
     Tspan).
  3. Leading-order eccentric waveform via the standard Peters-Mathews
     Bessel decomposition (source_harmonic_coeffs). Conventions vary
     across papers — validate numerically against a reference eccentric
     CW implementation before using for science; see
     _regression_check_circular_limit for the one guarantee made here
     (exact reduction to the circular formula at e=0).
  4. population.f is the pre-existing n=2 circular-equivalent GW
     frequency; f_orb = f / 2 is the fundamental used to build harmonics.

Dependencies: numpy, scipy.special (Bessel), finufft.
"""

from typing import Optional
import warnings

import numpy as np
from scipy.special import jv
import finufft

from SMBHB_pop_synth import PopulationArrays


# ──────────────────────────────────────────────────────────────────────────────
# PULSAR UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def get_base_name(psrname):
    """Strip telescope suffix (ao, gbt, vla, fast) from pulsar name."""
    for suffix in ['ao', 'gbt', 'vla', 'fast']:
        if psrname.endswith(suffix):
            return psrname[:-len(suffix)]
    return psrname


def _get_psr_radec(psr):
    """
    Extract (ra, dec) in radians from a libstempo or Enterprise pulsar
    object. Tries RAJ/DECJ, falls back to ELONG/ELAT (ecliptic), then
    falls back to _raj/_decj (Enterprise).
    """
    if hasattr(psr, 'pars'):
        try:
            pars = psr.pars()
            if 'RAJ' in pars and 'DECJ' in pars:
                return psr['RAJ'].val, psr['DECJ'].val
            elif 'ELONG' in pars and 'ELAT' in pars:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                coord = SkyCoord(lon=psr['ELONG'].val * u.rad,
                                  lat=psr['ELAT'].val * u.rad,
                                  frame='geocentricmeanecliptic')
                return coord.icrs.ra.rad, coord.icrs.dec.rad
        except Exception:
            pass
    if hasattr(psr, '_raj') and hasattr(psr, '_decj'):
        return psr._raj, psr._decj
    raise AttributeError(
        f"Cannot extract RA/Dec from pulsar object of type {type(psr)}. "
        f"Expected RAJ/DECJ or ELONG/ELAT params, or _raj/_decj attributes."
    )


def _antenna_response_vec(psr_ra, psr_dec, ra_arr, dec_arr, psi_arr):
    """
    Vectorised antenna response over N sources. Returns Fp, Fx each (N,).
    Geometric only — no frequency or harmonic dependence, hence computed
    once per (pulsar, binary) and reused across all harmonics.
    """
    cos_dec, sin_dec = np.cos(dec_arr), np.sin(dec_arr)
    cos_ra, sin_ra   = np.cos(ra_arr), np.sin(ra_arr)
    cos_psi, sin_psi = np.cos(psi_arr), np.sin(psi_arr)
    cos_pd, sin_pd   = np.cos(psr_dec), np.sin(psr_dec)
    cos_pr, sin_pr   = np.cos(psr_ra), np.sin(psr_ra)

    omega_hat = np.array([-cos_dec * cos_ra, -cos_dec * sin_ra, -sin_dec])
    p_hat     = np.array([cos_pd * cos_pr, cos_pd * sin_pr, sin_pd])
    m_hat     = np.array([sin_ra, -cos_ra, np.zeros(len(ra_arr))])
    n_hat     = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec])

    m_rot = cos_psi * m_hat + sin_psi * n_hat
    n_rot = -sin_psi * m_hat + cos_psi * n_hat

    denom = 1.0 + np.dot(p_hat, omega_hat)
    p_m, p_n = np.dot(p_hat, m_rot), np.dot(p_hat, n_rot)

    Fp = 0.5 * (p_m**2 - p_n**2) / denom
    Fx = (p_m * p_n) / denom
    return Fp, Fx


# ──────────────────────────────────────────────────────────────────────────────
# VALIDITY FILTER
# ──────────────────────────────────────────────────────────────────────────────

def filter_valid_population(population, e_max: float = 0.999, warn: bool = True):
    """
    Drop binaries that cannot be injected: non-finite h0/f, or e >= e_max
    (numerical stability of the Bessel-based harmonic truncation degrades
    as e -> 1; e_max=0.999 is a conservative numerical ceiling, not a
    physical one).

    Does NOT re-apply any upstream circularity gate — see module
    docstring. Handles populations without an `ecc` field (treated as
    all-circular, ecc=0) for backward compatibility.
    """
    ecc = np.asarray(getattr(population, 'ecc', np.zeros(len(population))), dtype=np.float64)

    finite_mask = np.isfinite(population.h0) & np.isfinite(population.f)
    ecc_mask    = ecc < e_max
    keep_mask   = finite_mask & ecc_mask

    n_total   = len(population)
    n_dropped = n_total - int(keep_mask.sum())

    if n_dropped > 0 and warn:
        warnings.warn(
            f"{n_dropped}/{n_total} binaries dropped: non-finite h0/f, or "
            f"eccentricity >= {e_max}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return population[keep_mask]


# ──────────────────────────────────────────────────────────────────────────────
# HARMONIC TRUNCATION  (Peters-Mathews power fraction, binned over e)
# ──────────────────────────────────────────────────────────────────────────────

def _g_power(n, e):
    """Relative GW power radiated into harmonic n at eccentricity e.
    Peters & Mathews (1963) Eq. 20. n: array, e: scalar."""
    n = np.asarray(n, dtype=np.float64)
    ne = n * e
    j_nm2, j_nm1, j_n = jv(n - 2, ne), jv(n - 1, ne), jv(n, ne)
    j_np1, j_np2      = jv(n + 1, ne), jv(n + 2, ne)

    term1 = j_nm2 - 2 * e * j_nm1 + (2.0 / n) * j_n + 2 * e * j_np1 - j_np2
    term2 = j_nm2 - 2 * j_n + j_np2
    return (n**4 / 32.0) * (term1**2 + (1 - e**2) * term2**2 + (4.0 / (3 * n**2)) * j_n**2)


def harmonic_truncation_binned(e_arr, power_tol=1e-4, n_max_cap=100, n_ebins=400):
    """
    Adaptive per-binary n_max, computed on a coarse e-grid (n_ebins points)
    and broadcast to every binary via np.digitize — turns an
    O(N_binary * n_max_cap) Bessel-evaluation cost into
    O(n_ebins * n_max_cap), independent of population size.

    Circular binaries (e < 1e-4) truncate to n_max=2 (quadrupole only) —
    this is what makes the flattened representation reduce to exactly the
    old circular-only cost for an all-circular population.

    NOTE on high e: as e -> 1, both n_max and the Bessel argument n*e
    grow, and jv()'s large-order/large-argument evaluation gets less
    numerically reliable. filter_valid_population's e_max default (0.999)
    is a conservative ceiling for this reason.
    """
    e_arr = np.clip(np.asarray(e_arr, dtype=np.float64), 0.0, 0.999)
    e_grid  = np.linspace(0.0, 0.999, n_ebins)
    n_range = np.arange(1, n_max_cap + 1)

    n_max_grid = np.full(n_ebins, 2, dtype=np.int32)
    for i, e in enumerate(e_grid):
        if e < 1e-4:
            n_max_grid[i] = 2
            continue
        gn = np.clip(_g_power(n_range, e), 0, None)
        total = gn.sum()
        if total <= 0:
            n_max_grid[i] = 2
            continue
        cumulative = np.cumsum(gn) / total
        idx = np.searchsorted(cumulative, 1 - power_tol)
        n_max_grid[i] = n_range[min(idx, len(n_range) - 1)]

    bin_idx = np.clip(np.digitize(e_arr, e_grid) - 1, 0, n_ebins - 1)
    return n_max_grid[bin_idx].astype(np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# FLATTEN (binary, harmonic) PAIRS
# ──────────────────────────────────────────────────────────────────────────────

def build_flattened_harmonics(population, power_tol=1e-4, n_max_cap=100):
    """
    CSR-style flattening: one row per (binary, harmonic) pair. For an
    all-circular population this yields exactly one row per binary
    (n_max=2), so downstream cost is identical to a dedicated circular-
    only implementation.

    Computed ONCE per population batch, reused across every pulsar —
    harmonic truncation and Bessel-based amplitude weights depend only
    on (e, iota, phi0, h0), not on the pulsar.
    """
    f    = np.asarray(population.f,    dtype=np.float64)
    ecc  = np.asarray(getattr(population, 'ecc', np.zeros(len(population))), dtype=np.float64)
    phi0 = np.asarray(population.phi0, dtype=np.float64)
    iota = np.asarray(population.iota, dtype=np.float64)
    h0   = np.asarray(population.h0,   dtype=np.float64)

    N = len(f)
    f_orb = f / 2.0

    n_max = harmonic_truncation_binned(ecc, power_tol=power_tol, n_max_cap=n_max_cap)
    offsets = np.concatenate([[0], np.cumsum(n_max)])
    N_flat = int(offsets[-1])

    binary_id     = np.repeat(np.arange(N), n_max)
    row_in_binary = np.arange(N_flat) - offsets[binary_id]
    harmonic_n    = (row_in_binary + 1).astype(np.int64)   # 1..n_max_i

    f_n    = harmonic_n * f_orb[binary_id]
    e_flat = ecc[binary_id]
    phi0_n = harmonic_n * phi0[binary_id]     # phase of harmonic n is n*phi0

    a_n, b_n = source_harmonic_coeffs(
        harmonic_n=harmonic_n, e_flat=e_flat,
        iota_flat=iota[binary_id], h0_flat=h0[binary_id], f_n=f_n,
    )

    return dict(
        binary_id=binary_id, harmonic_n=harmonic_n,
        f_n=f_n, phi0_n=phi0_n, a_n=a_n, b_n=b_n,
        N_binary=N, N_flat=N_flat, n_max=n_max,
    )


def source_harmonic_coeffs(harmonic_n, e_flat, iota_flat, h0_flat, f_n):
    """
    Per-harmonic source-frame amplitude coefficients (a_n, b_n), including
    the h0/(2*pi*f_n) normalisation, so that downstream:

        A_n = Fp(psr, binary) * a_n
        B_n = Fx(psr, binary) * b_n

    exactly mirrors the circular formula A=Fp*h0/(2pi f)(1+cos^2 i),
    B=Fx*h0/(2pi f)(-2 cos i).

    weight(n,e) is the Peters-Mathews term1 Bessel combination, which
    equals 1 identically at e=0, n=2 — this is what makes the e->0 limit
    reduce exactly to the circular formula (see
    _regression_check_circular_limit).

    ASSUMPTION FLAG: cross-validate against a reference eccentric CGW
    model before using for science (see module docstring).
    """
    n = harmonic_n.astype(np.float64)
    e = e_flat
    ne = n * e

    j_nm2, j_nm1, j_n = jv(n - 2, ne), jv(n - 1, ne), jv(n, ne)
    j_np1, j_np2      = jv(n + 1, ne), jv(n + 2, ne)
    weight = j_nm2 - 2 * e * j_nm1 + (2.0 / n) * j_n + 2 * e * j_np1 - j_np2

    norm_n = h0_flat / (2 * np.pi * f_n)
    ci = np.cos(iota_flat)

    a_n = norm_n * (1 + ci**2) * weight
    b_n = norm_n * (-2 * ci)   * weight
    return a_n, b_n


# ──────────────────────────────────────────────────────────────────────────────
# PER-PULSAR AMPLITUDE PRECOMPUTE
# ──────────────────────────────────────────────────────────────────────────────

def precompute_amplitudes(population, psr, flat):
    """
    Computes per-pulsar A_n/B_n on the flattened (binary, harmonic) rows
    and caches on population.amp_A[psr.name] / amp_B[psr.name] — reusing
    the existing PopulationArrays.amp_A/amp_B dict fields (no dataclass
    change needed; note their array length is now N_flat, not N_binary,
    when eccentric binaries are present).

    Cost: antenna pattern is O(N_binary) trig (same as the old circular
    code — NOT recomputed per harmonic), then broadcast onto N_flat rows
    via cheap fancy-indexing.
    """
    binary_id = flat['binary_id']
    Fp, Fx = _antenna_response_vec(
        *_get_psr_radec(psr),
        population.ra, population.dec, population.psi,
    )  # (N_binary,)

    population.amp_A[psr.name] = (Fp[binary_id] * flat['a_n']).astype(np.float64)
    population.amp_B[psr.name] = (Fx[binary_id] * flat['b_n']).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# DIRECT SUMMATION  (small populations, exact, no NUFFT)
# ──────────────────────────────────────────────────────────────────────────────

def change_in_TOAs_days_population_direct(
        psrs, population,
        verbose=False,
        chunk_size=50_000,
        power_tol=1e-4,
        n_max_cap=100,
        cache_precomputed_amplitudes=False,
        warn_on_filter=True,
        flat=None,
):
    """
    Direct-summation TOA-change calculation, chunked over the flattened
    (binary, harmonic) axis. Exact (no NUFFT tolerance error); cost scales
    as O(N_toa * N_flat) — use for N_flat below ~1e5, otherwise use the
    NUFFT variant.

    `flat` may be passed in precomputed (e.g. by the dispatcher, to avoid
    rebuilding it) — if None, it is built here.
    """
    population = filter_valid_population(population, warn=warn_on_filter)
    if flat is None:
        flat = build_flattened_harmonics(population, power_tol=power_tol, n_max_cap=n_max_cap)

    f_n, phi0_n = flat['f_n'], flat['phi0_n']
    cos_p, sin_p = np.cos(phi0_n), np.sin(phi0_n)
    N_flat = flat['N_flat']

    pulsar_time_changes_arr = []

    for psr in psrs:
        print(f"Processing {psr.name}...", flush=True)
        psr_name = psr.name
        computed_here = False
        if psr_name not in population.amp_A:
            precompute_amplitudes(population, psr, flat)
            computed_here = True

        A = population.amp_A[psr_name]
        B = population.amp_B[psr_name]

        S = A * cos_p - B * sin_p
        C = A * sin_p + B * cos_p

        t_sec = np.asarray(psr.stoas, dtype=np.float64) * 86400.0
        t_rel = t_sec - t_sec[0]

        time_change = np.zeros(len(t_rel), dtype=np.float64)
        for start in range(0, N_flat, chunk_size):
            end = min(start + chunk_size, N_flat)
            phase = (2 * np.pi * f_n[start:end][np.newaxis, :]
                      * t_rel[:, np.newaxis])
            time_change += (S[start:end] * np.sin(phase)
                             + C[start:end] * np.cos(phase)).sum(axis=1)

        if verbose:
            print(f"  {psr_name}: RMS = {time_change.std()*1e9:.3f} ns")

        pulsar_time_changes_arr.append([psr_name, time_change / 86400.0])

        if computed_here and not cache_precomputed_amplitudes:
            population.amp_A.pop(psr_name, None)
            population.amp_B.pop(psr_name, None)

    return pulsar_time_changes_arr


# ──────────────────────────────────────────────────────────────────────────────
# NUFFT  (large populations, O(N_flat log N_flat))
# ──────────────────────────────────────────────────────────────────────────────

def change_in_TOAs_days_population_nufft(
        psrs, population,
        verbose=False, eps=1e-6,
        power_tol=1e-4,
        n_max_cap=100,
        cache_precomputed_amplitudes=False,
        warn_on_filter=True,
        flat=None,
):
    """
    NUFFT type-3 TOA-change calculation.

    Uses finufft.nufft1d3: f(x_j) = sum_k c_k * exp(i * s_k * x_j), with
        x_j = t_j - t[0]                    (TOA times, seconds)
        s_k = 2*pi*f_n                      (harmonic frequencies, rad/s)
        c_k = (C_k - i*S_k) / 2             (complex amplitude)

    Derivation identical in form to the circular-only version — the only
    change is that "one row per binary" becomes "one row per (binary,
    harmonic)" via build_flattened_harmonics: s_k = 2*pi*f_n rather than
    2*pi*f, phase uses n*phi0 rather than phi0. One finufft call per
    pulsar covers all harmonics of all binaries simultaneously.

    `flat` may be passed in precomputed (see change_in_TOAs_days_population_direct).
    """
    population = filter_valid_population(population, warn=warn_on_filter)
    if flat is None:
        flat = build_flattened_harmonics(population, power_tol=power_tol, n_max_cap=n_max_cap)

    f_n, phi0_n = flat['f_n'], flat['phi0_n']
    s_arr = 2 * np.pi * f_n
    cos_p, sin_p = np.cos(phi0_n), np.sin(phi0_n)

    pulsar_time_changes_arr = []

    for psr in psrs:
        print(f"Processing {psr.name}...", flush=True)
        psr_name = psr.name
        computed_here = False
        if psr_name not in population.amp_A:
            precompute_amplitudes(population, psr, flat)
            computed_here = True

        A = population.amp_A[psr_name]
        B = population.amp_B[psr_name]

        S = A * cos_p - B * sin_p
        C = A * sin_p + B * cos_p
        c = (C - 1j * S) / 2

        t_sec = np.asarray(psr.stoas, dtype=np.float64) * 86400.0
        x = t_sec - t_sec[0]

        x       = np.ascontiguousarray(x,     dtype=np.float64)
        s_nufft = np.ascontiguousarray(s_arr, dtype=np.float64)
        c_nufft = np.ascontiguousarray(c,     dtype=np.complex128)

        f_out = finufft.nufft1d3(s_nufft, c_nufft, x, isign=+1, eps=eps)
        time_change = 2 * np.real(f_out)

        if verbose:
            print(f"  {psr_name}: RMS = {time_change.std()*1e9:.3f} ns")

        pulsar_time_changes_arr.append([psr_name, time_change / 86400.0])

        if computed_here and not cache_precomputed_amplitudes:
            population.amp_A.pop(psr_name, None)
            population.amp_B.pop(psr_name, None)

    return pulsar_time_changes_arr


# ──────────────────────────────────────────────────────────────────────────────
# DISPATCHER  (MAIN ENTRY POINT)
# ──────────────────────────────────────────────────────────────────────────────

def change_in_TOAs_days_population(
        psrs, population,
        method='auto',
        direct_threshold=100_000,     # compared against N_flat, not N_binary
        verbose=False,
        eps=1e-6,
        chunk_size=50_000,
        power_tol=1e-4,
        n_max_cap=100,
        cache_precomputed_amplitudes=False,
):
    """
    Main entry point. Filters once, builds the flattened harmonic
    representation once, then routes to direct or NUFFT — automatically
    handling eccentricity wherever present (no user branching required).

    method='auto':
        N_flat < direct_threshold  -> direct batched summation
        N_flat >= direct_threshold -> NUFFT

    Threshold is compared against N_flat (true injection cost), not
    N_binary, since eccentric binaries expand into multiple rows.
    """
    population = filter_valid_population(population, warn=True)
    flat = build_flattened_harmonics(population, power_tol=power_tol, n_max_cap=n_max_cap)
    N_flat, N_binary = flat['N_flat'], flat['N_binary']

    if method == 'auto':
        method = 'direct' if N_flat < direct_threshold else 'nufft'

    print(f"Computing TOA changes for N_binary={N_binary:,} "
          f"(N_flat={N_flat:,} binary-harmonic rows, mean n_max="
          f"{flat['n_max'].mean():.2f}) via method='{method}'")

    if method == 'direct':
        return change_in_TOAs_days_population_direct(
            psrs, population, verbose=verbose, chunk_size=chunk_size,
            power_tol=power_tol, n_max_cap=n_max_cap,
            cache_precomputed_amplitudes=cache_precomputed_amplitudes,
            warn_on_filter=False, flat=flat,
        )
    elif method == 'nufft':
        return change_in_TOAs_days_population_nufft(
            psrs, population, verbose=verbose, eps=eps,
            power_tol=power_tol, n_max_cap=n_max_cap,
            cache_precomputed_amplitudes=cache_precomputed_amplitudes,
            warn_on_filter=False, flat=flat,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: auto, direct, nufft")


# ──────────────────────────────────────────────────────────────────────────────
# LEGACY SUPPORT  (kept only for backward compatibility with the separate
# CGW_SNR script's `from signal_injection import population_residuals,
# get_base_name`). Circular-only — NOT eccentricity aware. If you need
# eccentric SNR, this needs rebuilding on build_flattened_harmonics the
# same way injection now is; flagging as a follow-up, not done here.
# ──────────────────────────────────────────────────────────────────────────────

def antenna_response(psr_ra, psr_dec, src_ra, src_dec, psi):
    """Scalar (single-source) antenna response. See Anholm et al. 2009."""
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
    m_hat = np.array([np.sin(src_azimuthal_angle), -np.cos(src_azimuthal_angle), 0.0])
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


def r_k(t, psr, binary):
    """
    Timing residual from a single CIRCULAR SMBHB (Earth term only).
    Legacy — used by population_residuals for the SNR pipeline. Does not
    use eccentricity even if binary.ecc is set.
    """
    f, ra, dec = binary.f, binary.ra, binary.dec
    psi, phi0, iota, h0 = binary.psi, binary.phi0, binary.iota, binary.h0

    psr_ra, psr_dec = psr._raj, psr._decj
    Fp, Fx = antenna_response(psr_ra, psr_dec, ra, dec, psi)

    t_rel = t - t[0]
    phase = 2 * np.pi * f * t_rel + phi0

    h_plus  = h0 * (1 + np.cos(iota)**2) * np.sin(phase)
    h_cross = h0 * (-2 * np.cos(iota))   * np.cos(phase)

    return (Fp * h_plus + Fx * h_cross) / (2 * np.pi * f)

def r_k_eccentric(t, psr, binary, power_tol=1e-4, n_max_cap=100):
    """
    Timing residual from a single SMBHB (Earth term only), including
    eccentricity via the Peters-Mathews harmonic decomposition — the
    matched-filter/SNR equivalent of what inject_population_* already does
    for real injection. Reduces exactly to r_k's circular formula at e=0
    (same harmonic machinery, same regression guarantee).

    Unlike r_k, this sums over ALL harmonics implied by binary.ecc, not
    just n=2. power_tol/n_max_cap should match whatever was used when the
    signal was actually injected, so the SNR template isn't mismatched
    against the true injected waveform (a template using fewer harmonics
    than were injected will underestimate SNR).
    """
    ecc = getattr(binary, 'ecc', 0.0)

    class _MiniPop:
        def __len__(self):
            return 1
    pop = _MiniPop()
    pop.f    = np.array([binary.f])
    pop.ecc  = np.array([ecc])
    pop.phi0 = np.array([binary.phi0])
    pop.iota = np.array([binary.iota])
    pop.h0   = np.array([binary.h0])

    flat = build_flattened_harmonics(pop, power_tol=power_tol, n_max_cap=n_max_cap)

    psr_ra, psr_dec = _get_psr_radec(psr)
    Fp, Fx = _antenna_response_vec(
        psr_ra, psr_dec,
        np.array([binary.ra]), np.array([binary.dec]), np.array([binary.psi]),
    )

    binary_id = flat['binary_id']          # all zeros — one binary here
    A = Fp[binary_id] * flat['a_n']         # (n_harmonics,)
    B = Fx[binary_id] * flat['b_n']

    f_n, phi0_n = flat['f_n'], flat['phi0_n']
    S = A * np.cos(phi0_n) - B * np.sin(phi0_n)
    C = A * np.sin(phi0_n) + B * np.cos(phi0_n)

    t_rel = t - t[0]
    phase = 2 * np.pi * f_n[None, :] * t_rel[:, None]     # (n_toa, n_harmonics)
    return (S[None, :] * np.sin(phase) + C[None, :] * np.cos(phase)).sum(axis=1)

def draw_red_noise_residuals(psr, log10_A, gamma, Tobs, nmodes=30):
    from enterprise_extensions.deterministic import createfourierdesignmatrix_red
    F, Ffreqs = createfourierdesignmatrix_red(psr.toas, nmodes=nmodes)
    f_yr = 1 / (365.25 * 24 * 3600)
    A = 10**log10_A
    kappa = (A**2 / (12 * np.pi**2)) * (Ffreqs / f_yr)**(-gamma) * f_yr**-3 * (1 / Tobs)
    a = np.sqrt(kappa) * np.random.randn(2 * nmodes)
    return F @ a


def white_noise_residual(pulsar, pulsar_noise_params):
    """
    NOTE: requires `enterprise.signals.{selections,parameter,white_signals}`
    and `enterprise.signals.utils.create_quantization_matrix` — these were
    used but never imported in the original file. Import them explicitly
    before use; verify against your installed enterprise version.
    """
    from enterprise.signals import selections, parameter, white_signals
    from enterprise.signals.utils import create_quantization_matrix

    wn_params = pulsar_noise_params[pulsar.name]['white_noise']
    r_wn = np.zeros(len(pulsar.toas))
    sel = selections.Selection(selections.by_backend)

    params_dict = {}
    for backend, bp in wn_params.items():
        params_dict[f'{pulsar.name}_{backend}_efac']          = bp['efac']
        params_dict[f'{pulsar.name}_{backend}_log10_t2equad'] = bp['log10_t2equad']
        params_dict[f'{pulsar.name}_{backend}_log10_ecorr']   = bp['log10_ecorr']

    mn = white_signals.MeasurementNoise(
        efac=parameter.Constant(), log10_t2equad=parameter.Constant(), selection=sel
    )
    Nvec = mn(pulsar).get_ndiag(params_dict)
    r_wn += np.sqrt(Nvec) * np.random.randn(len(pulsar.toas))

    for backend, bp in wn_params.items():
        mask = pulsar.flags['f'] == backend
        if not np.any(mask):
            continue
        ecorr = 10**bp['log10_ecorr']
        U, _ = create_quantization_matrix(pulsar.toas[mask], nmin=2)
        n_epochs = U.shape[1]
        z = np.random.randn(n_epochs)
        r_wn[mask] += U @ (ecorr * z)

    return r_wn


def population_residuals(t, psr, population, Tspan,
                         pulsar_noise_params=None,
                         include_GW=True,
                         include_RN=False,
                         include_WN=False,
                         power_tol=1e-4,
                         n_max_cap=100
                         ):
    """
    Scalar (loop-based) total timing residuals for one pulsar, for a small
    list of binary objects. Legacy — used by CGW_SNR script; circular only.
    """
    total_r = np.zeros_like(t, dtype=float)

    if include_GW:
        for binary in population:
            total_r += r_k_eccentric(t, psr, binary, power_tol=power_tol, n_max_cap=n_max_cap)

    if include_RN:
        if pulsar_noise_params is None:
            raise ValueError("pulsar_noise_params required when include_RN=True")
        rn = pulsar_noise_params[psr.name]['red_noise']
        total_r += draw_red_noise_residuals(psr, rn['log10_A'], rn['gamma'], Tspan)

    if include_WN:
        if pulsar_noise_params is None:
            raise ValueError("pulsar_noise_params required when include_WN=True")
        total_r += white_noise_residual(psr, pulsar_noise_params)

    return total_r


def population_residuals_eccentric(t, psr, binaries, Tspan,
                                    pulsar_noise_params=None,
                                    include_GW=True,
                                    power_tol=1e-4, n_max_cap=100):
    """
    Eccentric replacement for population_residuals(..., include_GW=True).

    `binaries` : a single binary-like object (scalar .f/.ecc/.../.h0),
    OR an iterable of such objects. Both are supported so this survives
    call sites written either way (population_residuals(t, psr, binary, ...)
    vs. population_residuals(t, psr, [binary], ...)) — the previous
    version only handled the list case and used len() to detect it,
    which raised on a bare _MiniPop (no __len__), got swallowed by the
    caller's except Exception, and silently zeroed every SNR.
    """
    total_r = np.zeros_like(t, dtype=np.float64)
    if not include_GW:
        return total_r

    # Detect "single binary" vs "iterable of binaries" WITHOUT relying on
    # len() or __iter__ existing — check for the presence of a scalar .f
    # attribute directly, since that's the one thing every binary-like
    # object here is guaranteed to have.
    if hasattr(binaries, 'f') and not hasattr(binaries, '__iter__'):
        binaries = [binaries]

    for binary in binaries:
        total_r += r_k_eccentric(
            t, psr, binary,
            power_tol=power_tol, n_max_cap=n_max_cap,
        )
    return total_r


# ──────────────────────────────────────────────────────────────────────────────
# REGRESSION CHECK: e -> 0 must reduce exactly to the circular formula
# ──────────────────────────────────────────────────────────────────────────────

def _regression_check_circular_limit():
    rng = np.random.default_rng(0)
    N = 500
    f    = rng.uniform(1e-8, 5e-8, N)
    ecc  = np.zeros(N)
    phi0 = rng.uniform(0, 2*np.pi, N)
    iota = np.arccos(rng.uniform(-1, 1, N))
    h0   = 10 ** rng.uniform(-16, -14, N)

    class _Pop:
        pass
    pop = _Pop()
    pop.f, pop.ecc, pop.phi0, pop.iota, pop.h0 = f, ecc, phi0, iota, h0

    flat = build_flattened_harmonics(pop, power_tol=1e-4, n_max_cap=50)
    assert np.all(flat['n_max'] == 2), "e=0 must truncate to n_max=2"
    assert np.all(flat['harmonic_n'] == 2), "only row per binary should be n=2"

    A_expected = h0 / (2*np.pi*f) * (1 + np.cos(iota)**2)
    B_expected = h0 / (2*np.pi*f) * (-2*np.cos(iota))
    assert np.allclose(flat['a_n'], A_expected, rtol=1e-10)
    assert np.allclose(flat['b_n'], B_expected, rtol=1e-10)
    print("✓ e=0 regression check passed: reduces exactly to circular formula.")


if __name__ == '__main__':
    _regression_check_circular_limit()

import math

day  = 86400.0
year = 3.15581498e7
NU_REF_MHZ = 1400.0  # MPTA reference frequency


def make_ideal_nofit(psr):
    """Zero out residuals by adjusting TOAs, without refitting."""
    res = psr.residuals(updatebats=True, formresiduals=True)
    psr.stoas[:] -= res / day


def _quantize_fast(times, flags=None, dt=1.0):
    """Bin TOAs into epochs (exact reimplementation of libstempo's quantize_fast)."""
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
    return (avetoas, aveflags, U) if flags is not None else (avetoas, U)


def _add_efac(psr, efac, flagid, flags, seed=None):
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
    _, aveflags, U = _quantize_fast(t, flags=f, dt=coarsegrain)
    epoch_mask = np.array([fv == flags for fv in aveflags])
    ecorrvec = np.where(epoch_mask, ecorr, 0.0)
    psr.stoas[:] += (1 / day) * np.dot(U * ecorrvec, np.random.randn(U.shape[1]))


def _add_red_noise_achromatic(psr, log10_A, gamma, components=10, tspan=None, seed=None, verbose=False):
    """Achromatic red noise: P(f) = (A^2/12pi^2) * (f/f_c)^(-gamma). No frequency dependence."""
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
    """DM noise: P(f) = (A_DM^2/12pi^2)*(f/f_c)^(-gamma_DM)*(nu/nu_ref)^(-4)."""
    if seed is not None:
        np.random.seed(seed)
    try:
        freqs_MHz = np.array(psr.freqs)
        if freqs_MHz is None or len(freqs_MHz) == 0:
            raise ValueError
    except Exception:
        if verbose:
            print(f"        WARNING: no frequency info, assuming {NU_REF_MHZ:.0f} MHz")
        freqs_MHz = np.full(psr.nobs, NU_REF_MHZ)

    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (day / year) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (day / year) * tspan

    A_DM = 10**log10_A_DM
    norm = A_DM**2 * year**2 / (12 * math.pi**2 * T)

    size = 2 * components
    F = np.zeros((psr.nobs, size), 'd')
    f = np.zeros(size, 'd')
    for i in range(components):
        F[:, 2*i]   = np.cos(2 * math.pi * (i+1) * x)
        F[:, 2*i+1] = np.sin(2 * math.pi * (i+1) * x)
        f[2*i] = f[2*i+1] = (i+1) / T

    prior = norm * f**(-gamma_DM)
    y = np.sqrt(prior) * np.random.randn(size)
    red_noise = np.dot(F, y)

    freq_scaling = (freqs_MHz / NU_REF_MHZ)**(-2.0)   # (nu/nu_ref)^-4 on power => ^-2 on amplitude
    psr.stoas[:] += (1.0 / day) * freq_scaling * red_noise


def _add_chromatic_noise(psr, log10_A_chrom, gamma_chrom, beta, components=10, tspan=None, seed=None, verbose=False):
    """Chromatic noise: P(f) = (A_ch^2/12pi^2)*(f/f_c)^(-gamma_ch)*(nu/nu_ref)^(-beta)."""
    if seed is not None:
        np.random.seed(seed)
    try:
        freqs_MHz = np.array(psr.freqs)
        if freqs_MHz is None or len(freqs_MHz) == 0:
            raise ValueError
    except Exception:
        if verbose:
            print(f"        WARNING: no frequency info, assuming {NU_REF_MHZ:.0f} MHz")
        freqs_MHz = np.full(psr.nobs, NU_REF_MHZ)

    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (day / year) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (day / year) * tspan

    A_chrom = 10**log10_A_chrom
    norm = A_chrom**2 * year**2 / (12 * math.pi**2 * T)

    size = 2 * components
    F = np.zeros((psr.nobs, size), 'd')
    f = np.zeros(size, 'd')
    for i in range(components):
        F[:, 2*i]   = np.cos(2 * math.pi * (i+1) * x)
        F[:, 2*i+1] = np.sin(2 * math.pi * (i+1) * x)
        f[2*i] = f[2*i+1] = (i+1) / T

    prior = norm * f**(-gamma_chrom)
    y = np.sqrt(prior) * np.random.randn(size)
    red_noise = np.dot(F, y)

    freq_scaling = (freqs_MHz / NU_REF_MHZ)**(-beta)
    psr.stoas[:] += (1.0 / day) * freq_scaling * red_noise


def _add_sw_noise(psr, log10_A_SW, gamma_SW, components=10, tspan=None, seed=None, verbose=False):
    """Solar wind noise, same (nu/nu_ref)^-4 scaling as DM noise."""
    if seed is not None:
        np.random.seed(seed)
    try:
        freqs_MHz = np.array(psr.freqs)
        if freqs_MHz is None or len(freqs_MHz) == 0:
            raise ValueError
    except Exception:
        freqs_MHz = np.full(psr.nobs, NU_REF_MHZ)

    t = psr.toas()
    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (day / year) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (day / year) * tspan

    A_SW = 10**log10_A_SW
    norm = A_SW**2 * year**2 / (12 * math.pi**2 * T)

    size = 2 * components
    F = np.zeros((psr.nobs, size), 'd')
    f = np.zeros(size, 'd')
    for i in range(components):
        F[:, 2*i]   = np.cos(2 * math.pi * (i+1) * x)
        F[:, 2*i+1] = np.sin(2 * math.pi * (i+1) * x)
        f[2*i] = f[2*i+1] = (i+1) / T

    prior = norm * f**(-gamma_SW)
    y = np.sqrt(prior) * np.random.randn(size)
    red_noise = np.dot(F, y)

    freq_scaling = (freqs_MHz / NU_REF_MHZ)**(-2.0)
    psr.stoas[:] += (1.0 / day) * freq_scaling * red_noise


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
    """Determine number of red-noise Fourier components from a .par file."""
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
                 seed=None, par_path=None, tspan_override=None, verbose=False):
    """
    Inject noise processes into a pulsar with proper frequency dependence:

        P_DM(f)    = (A_DM^2/12pi^2)   * (f/f_c)^(-gamma_DM)    * (nu/nu_ref)^(-4)
        P_chrom(f) = (A_chrom^2/12pi^2)* (f/f_c)^(-gamma_chrom) * (nu/nu_ref)^(-beta)

    noise_dict keys expected per system: {base}_red_log10_A/_red_gamma,
    {base}_dm_log10_A/_dm_gamma, {base}_chrom_log10_A/_chrom_gamma/_chrom_beta,
    {base}_sw_log10_A/_sw_gamma, {base}_{sys}_efac/_log10_equad/_log10_ecorr.
    """
    psrname  = psr.name
    basename = get_base_name(psrname)
    psr_keys = {k: v for k, v in noise_dict.items() if k.startswith(basename + '_')}

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
        print(f"  [{psrname}] zeroing residuals...", flush=True)
    make_ideal_nofit(psr)

    t = psr.toas()
    tspan_actual = t.max() - t.min()
    tspan = tspan_override if tspan_override is not None else tspan_actual

    if add_RN:
        k_A, k_g = f"{basename}_red_log10_A", f"{basename}_red_gamma"
        if k_A in psr_keys and k_g in psr_keys:
            rn_components = get_rn_components(par_path, default=120)
            _add_red_noise_achromatic(psr, psr_keys[k_A], psr_keys[k_g],
                                      components=rn_components, tspan=tspan,
                                      seed=rn_seed, verbose=verbose)

    if add_DM:
        k_A, k_g = f"{basename}_dm_log10_A", f"{basename}_dm_gamma"
        if k_A in psr_keys and k_g in psr_keys:
            dm_components = get_rn_components(par_path, default=120)
            _add_dm_noise(psr, psr_keys[k_A], psr_keys[k_g],
                          components=dm_components, tspan=tspan,
                          seed=dm_seed, verbose=verbose)

    if add_chrom:
        k_A, k_g, k_b = (f"{basename}_chrom_log10_A", f"{basename}_chrom_gamma",
                         f"{basename}_chrom_beta")
        if k_A in psr_keys and k_g in psr_keys and k_b in psr_keys:
            chrom_components = get_rn_components(par_path, default=120)
            _add_chromatic_noise(psr, psr_keys[k_A], psr_keys[k_g], psr_keys[k_b],
                                 components=chrom_components, tspan=tspan,
                                 seed=chrom_seed, verbose=verbose)

    if add_SW:
        k_A, k_g = f"{basename}_sw_log10_A", f"{basename}_sw_gamma"
        if k_A in psr_keys and k_g in psr_keys:
            sw_components = get_rn_components(par_path, default=120)
            _add_sw_noise(psr, psr_keys[k_A], psr_keys[k_g],
                         components=sw_components, tspan=tspan,
                         seed=sw_seed, verbose=verbose)

    if add_WN:
        try:
            flag_vals = np.array(psr.flagvals('f'))
        except Exception:
            flag_vals = np.array([''] * psr.nobs)

        systems = _get_system_list(basename, psr_keys,
                                   ['_efac', '_log10_equad', '_log10_ecorr'])
        for sys in systems:
            mask = np.array([sys in str(fv) for fv in flag_vals])
            if mask.sum() == 0:
                continue

            efac = psr_keys.get(f"{basename}_{sys}_efac", 1.0)
            log10_equad = psr_keys.get(f"{basename}_{sys}_log10_equad",
                          psr_keys.get(f"{basename}_{sys}_log10_t2equad", -100.0))
            equad = 10**log10_equad
            ecorr = 10**psr_keys.get(f"{basename}_{sys}_log10_ecorr", -100.0)

            if efac > 0.1:
                _add_efac(psr, efac, flagid='f', flags=sys, seed=wn_seed)
            if equad > 1e-8:
                _add_equad(psr, equad, flagid='f', flags=sys, seed=int(wn_seed+1) if wn_seed is not None else None)
            if ecorr > 1e-8:
                _add_ecorr(psr, ecorr, flagid='f', flags=sys, seed=int(wn_seed+2) if wn_seed is not None else None)

    if verbose:
        print(f"  [{psrname}] done", flush=True)
    return psr

def inject_population_direct(
        psrs, population,
        pure_signal=True,
        chunk_size=50_000,
        power_tol=1e-4,
        n_max_cap=100,
        cache_precomputed_amplitudes=False,
        warn_on_filter=True,
):
    """
    Direct-summation injection, mutating psr._residuals in place.

    pure_signal=True  -> replaces psr._residuals with the injected signal
    pure_signal=False -> adds the injected signal to existing psr.residuals

    Eccentricity handled automatically via the flattened (binary, harmonic)
    representation — see build_flattened_harmonics.
    """
    population = filter_valid_population(population, warn=warn_on_filter)
    flat = build_flattened_harmonics(population, power_tol=power_tol, n_max_cap=n_max_cap)

    f_n, phi0_n = flat['f_n'], flat['phi0_n']
    cos_p, sin_p = np.cos(phi0_n), np.sin(phi0_n)
    N_flat = flat['N_flat']

    for psr in psrs:
        psr_name = psr.name
        computed_here = False
        if psr_name not in population.amp_A:
            precompute_amplitudes(population, psr, flat)
            computed_here = True

        A = population.amp_A[psr_name]
        B = population.amp_B[psr_name]

        S = A * cos_p - B * sin_p
        C = A * sin_p + B * cos_p

        t_sec = np.asarray(psr.toas, dtype=np.float64)
        t_rel = t_sec - t_sec[0]

        r_new = np.zeros(len(t_rel), dtype=np.float64)
        for start in range(0, N_flat, chunk_size):
            end = min(start + chunk_size, N_flat)
            phase = (2 * np.pi * f_n[start:end][np.newaxis, :]
                      * t_rel[:, np.newaxis])
            r_new += (S[start:end] * np.sin(phase)
                       + C[start:end] * np.cos(phase)).sum(axis=1)

        psr._residuals = r_new if pure_signal else psr.residuals + r_new

        if computed_here and not cache_precomputed_amplitudes:
            population.amp_A.pop(psr_name, None)
            population.amp_B.pop(psr_name, None)

    return psrs


def inject_population_nufft(
        psrs, population,
        verbose=False, eps=1e-6,
        power_tol=1e-4,
        n_max_cap=100,
        cache_precomputed_amplitudes=True,
        warn_on_filter=True,
        track_contributors=False,
        top_k_global=50,
):
    """
    NUFFT type-3 injection, mutating psr.stoas in place directly.

    Same derivation as change_in_TOAs_days_population_nufft, but applies
    the time change to psr.stoas rather than returning it.

    If track_contributors=True, ranks the loudest binaries (by summed
    A^2+B^2 across their harmonics, aggregated over all pulsars) and
    attaches the result to population.contributor_summary.
    """
    population = filter_valid_population(population, warn=warn_on_filter)
    flat = build_flattened_harmonics(population, power_tol=power_tol, n_max_cap=n_max_cap)

    f_n, phi0_n = flat['f_n'], flat['phi0_n']
    binary_id   = flat['binary_id']
    s_arr = 2 * np.pi * f_n
    cos_p, sin_p = np.cos(phi0_n), np.sin(phi0_n)

    binary_score_total = np.zeros(flat['N_binary']) if track_contributors else None

    for psr in psrs:
        if verbose:
            print(f"Processing {psr.name}...", flush=True)
        psr_name = psr.name
        computed_here = False
        if psr_name not in population.amp_A:
            precompute_amplitudes(population, psr, flat)
            computed_here = True

        A = population.amp_A[psr_name]
        B = population.amp_B[psr_name]

        if track_contributors:
            # aggregate per-harmonic amplitude^2 back onto per-binary score
            np.add.at(binary_score_total, binary_id, A**2 + B**2)

        S = A * cos_p - B * sin_p
        C = A * sin_p + B * cos_p
        c = (C - 1j * S) / 2

        t_sec = np.asarray(psr.stoas, dtype=np.float64) * 86400.0
        x = t_sec - t_sec[0]

        x       = np.ascontiguousarray(x,     dtype=np.float64)
        s_nufft = np.ascontiguousarray(s_arr, dtype=np.float64)
        c_nufft = np.ascontiguousarray(c,     dtype=np.complex128)

        f_out = finufft.nufft1d3(s_nufft, c_nufft, x, isign=+1, eps=eps)
        time_change = 2 * np.real(f_out)

        if verbose:
            print(f"  {psr_name}: RMS = {time_change.std()*1e9:.3f} ns")

        psr.stoas[:] += time_change / 86400.0

        if computed_here and not cache_precomputed_amplitudes:
            population.amp_A.pop(psr_name, None)
            population.amp_B.pop(psr_name, None)

    if track_contributors:
        k = min(top_k_global, flat['N_binary'])
        top_idx = np.argpartition(binary_score_total, -k)[-k:]
        top_idx = top_idx[np.argsort(binary_score_total[top_idx])[::-1]]
        population.contributor_summary = {
            'score_definition': 'sum over harmonics of A^2+B^2, summed over pulsars',
            'indices': top_idx.tolist(),
            'score': binary_score_total[top_idx].tolist(),
        }

    return psrs