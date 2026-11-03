"""
CGW optimal S/N and population-GWB covariance utilities.

Run `python cgw_snr.py` to execute the mock-data self-test (no enterprise or
project modules required). On real data, call `cgw_live_check(...)` first.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import time
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve

LinAlgError = np.linalg.LinAlgError

# ---------------------------------------------------------------------------
# Optional project imports. Missing ones only raise when actually needed, so
# the module (and its self-test) imports cleanly without them.
# ---------------------------------------------------------------------------
try:
    from enterprise_extensions.frequentist.Fe_statistic import innerProduct_rr
except ImportError:
    innerProduct_rr = None
try:
    from enterprise_extensions.deterministic import cw_delay
except ImportError:
    cw_delay = None
try:
    from optimal_SNR_calc import measured_strain_all_binaries_all_pulsars
except ImportError:
    measured_strain_all_binaries_all_pulsars = None
try:
    from signal_injection import (
        antenna_response,
        get_base_name,
        population_residuals,
    )
except ImportError:
    antenna_response = get_base_name = population_residuals = None


def _resolve(override, default, name):
    fn = override if override is not None else default
    if fn is None:
        raise ImportError(
            f"{name} is unavailable: import it from your project or pass it explicitly."
        )
    return fn


def _supported_kwargs(fn, names):
    """Which of `names` can be passed to `fn` (decided by signature, not by catching TypeError)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return set(names)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return set(names)
    return {n for n in names if n in params}


def _psr_radec(psr):
    """(ra, dec) in radians from an enterprise Pulsar (or lookalike)."""
    try:
        return float(psr._raj), float(psr._decj)
    except AttributeError:
        return float(psr.phi), float(np.pi / 2.0 - psr.theta)


def _psr_toas(psr):
    toas = psr.toas
    if callable(toas):
        toas = toas()
    return np.asarray(toas)


# ---------------------------------------------------------------------------
# Debug helpers (printing can never crash a run)
# ---------------------------------------------------------------------------
_SOURCE_FIELDS = ("f", "Mc", "Mtot", "D_comov", "z", "h0", "ecc",
                  "ra", "dec", "psi", "iota", "phi0")


def _fmt(value):
    try:
        return f"{float(value):.17g}"
    except Exception:
        return "<unprintable>"


def _debug_source_header(source_index, binary, reason):
    print("\n" + "=" * 86)
    print(f"CGW debug source {source_index}")
    print(f"Reason: {reason}")
    print("=" * 86)
    for field in _SOURCE_FIELDS:
        if hasattr(binary, field):
            print(f"  source.{field:<8} = {_fmt(getattr(binary, field))}")


def _debug_pulsar_summary(source_index, pulsar_name, psr_obj, toas, noise, residuals,
                          contribution, cumulative_rho_sq, denominator=None,
                          separation=None, fp=None, fx=None):
    print(f"\n[CGW debug] source={source_index} pulsar={pulsar_name}")
    print(f"  type              = {type(psr_obj).__name__}")
    print(f"  n_toa             = {len(toas)}")
    try:
        print(f"  TOA range         = {np.min(toas):.17g} .. {np.max(toas):.17g} s")
    except Exception:
        print("  TOA range         = <unavailable>")
    try:
        te = np.asarray(psr_obj.toaerrs, dtype=float)
        print(f"  TOA errors        = min {te.min():.6e}, "
              f"median {np.median(te):.6e}, max {te.max():.6e} s")
    except Exception:
        print("  TOA errors        = <unavailable>")
    if isinstance(noise, dict):
        print(f"  noise param keys  = {sorted(noise)}")
    else:
        print(f"  noise parameters  = <{type(noise).__name__}>")
    if denominator is not None:
        print(f"  antenna denom     = {denominator:.16e}")
    if separation is not None:
        print(f"  source separation = {np.degrees(separation):.12f} deg")
    if fp is not None and fx is not None:
        print(f"  F_plus / F_cross  = {float(fp):.16e}, {float(fx):.16e}")
    try:
        r = np.asarray(residuals, dtype=float)
        print(f"  residuals         = min {r.min():.6e}, "
              f"rms {np.sqrt(np.mean(r ** 2)):.6e}, max {r.max():.6e} s")
    except Exception:
        print("  residuals         = <unavailable>")
    print(f"  rho^2 contribution= {contribution:.16e}")
    print(f"  cumulative rho^2  = {cumulative_rho_sq:.16e}")


def _antenna_diagnostics(psr_ra, psr_dec, binary, ant_fn):
    """Return (F+, Fx, 1 + p.Omega, angular separation) or Nones on failure."""
    try:
        src_ra, src_dec, psi = float(binary.ra), float(binary.dec), float(binary.psi)
        fp, fx = ant_fn(psr_ra, psr_dec, src_ra, src_dec, psi)
        src_dir = -np.array([np.cos(src_dec) * np.cos(src_ra),
                             np.cos(src_dec) * np.sin(src_ra),
                             np.sin(src_dec)])
        psr_dir = np.array([np.cos(psr_dec) * np.cos(psr_ra),
                            np.cos(psr_dec) * np.sin(psr_ra),
                            np.sin(psr_dec)])
        den = float(1.0 + psr_dir @ src_dir)
        sep = float(np.arccos(np.clip(
            np.sin(psr_dec) * np.sin(src_dec)
            + np.cos(psr_dec) * np.cos(src_dec) * np.cos(psr_ra - src_ra), -1.0, 1.0)))
        return fp, fx, den, sep
    except Exception:
        return None, None, None, None


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------
class _SigmaSolver:
    """Solve Sigma x = b via a Cholesky factor, or a pseudo-inverse fallback."""

    def __init__(self, cf=None, inv=None):
        self.cf, self.inv = cf, inv

    def solve(self, x):
        return cho_solve(self.cf, x) if self.cf is not None else self.inv @ x


def _factor_sigma(Sigma, name, regularise=True, regularisation=1e-10, max_attempts=10):
    Sigma = np.array(Sigma, dtype=float)
    if not np.all(np.isfinite(Sigma)):
        raise RuntimeError(f"Sigma for {name} contains non-finite values.")
    eye = np.eye(Sigma.shape[0])
    if regularise:
        Sigma = Sigma + regularisation * eye

    for attempt in range(max_attempts + 1):
        try:
            return _SigmaSolver(cf=cho_factor(Sigma))
        except LinAlgError:
            if attempt == max_attempts:
                break
            reg = regularisation * 10 ** (attempt + 1)
            if attempt in (0, max_attempts - 1):
                print(f"  Warning: {name} Sigma not PD via Cholesky, increasing "
                      f"regularisation to {reg:.2e} (attempt {attempt + 1}/{max_attempts})")
            Sigma = Sigma + reg * eye

    print(f"  Warning: {name} Sigma still not PD after {max_attempts} "
          f"regularisation attempts; pinv fallback.")
    try:
        return _SigmaSolver(inv=np.linalg.pinv(Sigma, hermitian=True))
    except LinAlgError as exc:
        raise RuntimeError(f"Could not invert Sigma for {name} even via pinv: {exc}")


def fast_inner_product_rr_exact(x, y, Nvec, T, solver):
    """(x|y) = x^T N^-1 y - (T^T N^-1 x)^T Sigma^-1 (T^T N^-1 y),  Sigma = T^T N^-1 T + phi^-1."""
    TNy = Nvec.solve(y, left_array=T)
    TNx = TNy if x is y else Nvec.solve(x, left_array=T)
    xNy = Nvec.solve(y, left_array=x)
    return float(np.real(xNy - np.dot(TNx, solver.solve(TNy))))


# ---------------------------------------------------------------------------
# Simple (reference) S/N and single-pulsar signal
# ---------------------------------------------------------------------------
def compute_cgw_signal_enterprise(psr, binary):
    """CGW timing residuals for one pulsar via enterprise's cw_delay (earth term only)."""
    fn = _resolve(None, cw_delay, "cw_delay")
    return fn(
        toas=psr.toas, pos=psr.pos, pdist=psr.pdist,
        cos_gwtheta=np.cos(np.pi / 2.0 - binary.dec),
        gwphi=binary.ra,
        cos_inc=np.cos(binary.iota),
        log10_mc=np.log10(binary.Mc),
        log10_fgw=np.log10(binary.f),
        log10_h=np.log10(binary.h0),
        phase0=binary.phi0,
        psi=binary.psi,
        psrTerm=False,
    )


def compute_cgw_snr_optimal_population(psrs, pta, population, raw_noise_params,
                                       parsed_noise_params, Tspan, profile=False):
    """Straightforward S/N using enterprise's innerProduct_rr (slower reference path)."""
    inner = _resolve(None, innerProduct_rr, "innerProduct_rr")
    pop_res = _resolve(None, population_residuals, "population_residuals")
    t0 = time.time()

    phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
    TNTs = pta.get_TNT(raw_noise_params)
    Ts = pta.get_basis()
    Nvecs = pta.get_ndiag(raw_noise_params)
    psr_map = {psr.name: psr for psr in psrs}
    Sigmas = [TNT + (np.diag(p) if p.ndim == 1 else p) for TNT, p in zip(TNTs, phiinvs)]

    if profile:
        print(f"Precomputation took {time.time() - t0:.2f} s")

    results = []
    for binary in population:
        rho_sq = 0.0
        for psr_name, Nvec, TNT, Sigma, T in zip(pta.pulsars, Nvecs, TNTs, Sigmas, Ts):
            psr = psr_map[psr_name]
            s_a = pop_res(psr.toas, psr, [binary], Tspan, parsed_noise_params[psr_name])
            rho_sq += inner(s_a, s_a, Nvec, T, TNT, Sigma)
        results.append(np.sqrt(rho_sq))

    if profile:
        print(f"Total S/N computation took {time.time() - t0:.2f} s")
    return results


# ---------------------------------------------------------------------------
# Fast S/N with diagnostics
# ---------------------------------------------------------------------------
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
    strict=False,
    debug=False,
    debug_top_k=10,
    debug_dominance_fraction=0.95,
    debug_denominator_threshold=1e-3,
    debug_rho_sq_threshold=1.0,
    population_residuals_fn=None,
    antenna_response_fn=None,
    get_base_name_fn=None,
):
    """
    Optimal CW S/N (rho) for every binary in `population`.

    strict=True      raise on any per-pulsar failure instead of skipping it
                     (skipping silently biases rho low; a warning is printed).
    debug=True       after the main pass, re-evaluate and print diagnostics for
                     the top-k sources, any source where one pulsar supplies
                     >= `debug_dominance_fraction` of rho^2, and non-finite results.
    return_breakdown also return per-source {pulsar_base_name: rho^2 contribution}.
    """
    t0 = time.time()
    pop_res = _resolve(population_residuals_fn, population_residuals, "population_residuals")
    ant_fn = _resolve(antenna_response_fn, antenna_response, "antenna_response")
    base_fn = _resolve(get_base_name_fn, get_base_name, "get_base_name")
    pop_kw = _supported_kwargs(pop_res, ("pulsar_noise_params", "power_tol", "n_max_cap"))
    extra_kw = {"power_tol": power_tol, "n_max_cap": n_max_cap}

    phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
    TNTs = pta.get_TNT(raw_noise_params)
    Ts = pta.get_basis()
    Nvecs = pta.get_ndiag(raw_noise_params)
    psr_map = {psr.name: psr for psr in psrs}

    # ---- binary-independent precomputation -------------------------------
    precomputed = []
    for psr_name, Nvec, TNT, phiinv, T in zip(pta.pulsars, Nvecs, TNTs, phiinvs, Ts):
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        solver = _factor_sigma(Sigma, psr_name, regularise_sigma, regularisation)
        psr_obj = psr_map[psr_name]
        ra, dec = _psr_radec(psr_obj)
        noise = (parsed_noise_params.get(psr_name)
                 or parsed_noise_params.get(base_fn(psr_name)) or {})
        precomputed.append((psr_name, psr_obj, _psr_toas(psr_obj), noise, Nvec, T, solver, ra, dec))

    if profile:
        print(f"Precompute time: {time.time() - t0:.2f} s")

    def _evaluate_source(idx, emit_debug=False):
        binary = population[idx]
        rho_sq, n_failed, per_pulsar = 0.0, 0, {}

        for psr_name, psr_obj, toas, noise, Nvec, T, solver, ra, dec in precomputed:
            kwargs = {k: v for k, v in extra_kw.items() if k in pop_kw}
            if "pulsar_noise_params" in pop_kw:
                kwargs["pulsar_noise_params"] = noise
            try:
                s_a = np.asarray(pop_res(toas, psr_obj, [binary], Tspan, **kwargs), dtype=float)
                contrib = fast_inner_product_rr_exact(s_a, s_a, Nvec, T, solver)
            except Exception as exc:
                if strict:
                    raise
                n_failed += 1
                print(f"Warning: source {idx}, pulsar {psr_name} failed: {exc}")
                continue

            if np.isfinite(contrib):
                contrib = max(contrib, 0.0)   # tiny negatives are numerical noise
            rho_sq += contrib
            base = base_fn(psr_name)
            per_pulsar[base] = per_pulsar.get(base, 0.0) + contrib

            if emit_debug:
                fp, fx, den, sep = _antenna_diagnostics(ra, dec, binary, ant_fn)
                if den is None or abs(den) <= debug_denominator_threshold \
                        or contrib >= debug_rho_sq_threshold:
                    _debug_pulsar_summary(idx, psr_name, psr_obj, toas, noise, s_a,
                                          contrib, rho_sq, den, sep, fp, fx)

        if n_failed == len(precomputed):
            raise RuntimeError(f"population_residuals failed for all {n_failed} pulsars "
                               f"on source {idx}.")
        return float(np.sqrt(rho_sq)), per_pulsar, n_failed

    # ---- pass 1: all sources, no debug output ----------------------------
    N_binary = len(population)
    results = np.empty(N_binary, dtype=np.float64)
    all_breakdowns, total_failed = [], 0
    for i in range(N_binary):
        results[i], bd, nf = _evaluate_source(i)
        all_breakdowns.append(bd)
        total_failed += nf

    if total_failed:
        print(f"WARNING: {total_failed} source/pulsar evaluations failed and were skipped; "
              f"affected S/N values are biased low. Use strict=True to raise instead.")

    # ---- pass 2: diagnostics for selected sources ------------------------
    if debug:
        finite_idx = np.flatnonzero(np.isfinite(results))
        n_top = min(int(debug_top_k), len(finite_idx))
        top = set(finite_idx[np.argsort(results[finite_idx])[-n_top:]].tolist()) if n_top else set()
        nonfinite = set(np.flatnonzero(~np.isfinite(results)).tolist())

        dominant = {}
        for i, bd in enumerate(all_breakdowns):
            pos = {k: (v if v > 0 else 0.0) for k, v in bd.items()}   # NaN -> 0
            total = sum(pos.values())
            if total <= 0.0:
                continue
            name, largest = max(pos.items(), key=lambda kv: kv[1])
            if largest / total >= float(debug_dominance_fraction):
                dominant[i] = (name, largest / total)

        selected = sorted(top | set(dominant) | nonfinite)
        print(f"\nCGW debug selection: {len(selected)} source(s): {len(top)} top-{n_top}, "
              f"{len(dominant)} single-pulsar dominated, {len(nonfinite)} non-finite.")

        for i in selected:
            reasons = []
            if i in top:
                rank = int(1 + np.sum(results[finite_idx] > results[i]))
                reasons.append(f"top-{debug_top_k} by CW S/N (rank {rank}, rho={results[i]:.6e})")
            if i in dominant:
                name, frac = dominant[i]
                reasons.append(f"{name} contributes {100.0 * frac:.3f}% of rho^2")
            if i in nonfinite:
                reasons.append("non-finite S/N")
            _debug_source_header(i, population[i], "; ".join(reasons))
            _evaluate_source(i, emit_debug=True)
            print(f"[CGW debug] source={i} final rho^2={results[i] ** 2:.16e}, "
                  f"rho={results[i]:.16e}")

    if profile:
        print(f"Total runtime: {time.time() - t0:.2f} s")
    return (results, all_breakdowns) if return_breakdown else results


# ---------------------------------------------------------------------------
# Population GWB PSD
# ---------------------------------------------------------------------------
_BIN_FIELDS = ("f", "Mc", "D_comov", "h0", "z", "ra", "dec", "psi", "phi0", "iota")


def _gwb_psd_core(binaries, pulsar_cache, time_arr, strain_fn=None):
    """
    Incoherent sum of per-binary strain spectra -> timing-residual PSD per pulsar.

        S_h(f)  = 2 * sum_b |h_f,b(f)|^2 / delta_f        one-sided strain PSD   [1/Hz]
        S_r(f)  = S_h(f) / (4 pi^2 f^2)                   timing-residual PSD    [s^3]

    Assumes `measured_strain_all_binaries_all_pulsars(full_spectrum=True)` returns
    h_f of shape (B, N, F) normalised so |h_f|^2/delta_f is a PSD (check this on
    your side). Bins with f <= 0 are set to zero.
    """
    if len(binaries) == 0:
        raise ValueError("binaries is empty.")
    fn = _resolve(strain_fn, measured_strain_all_binaries_all_pulsars,
                  "measured_strain_all_binaries_all_pulsars")
    bin_arrays = {k: np.array([getattr(b, k) for b in binaries]) for k in _BIN_FIELDS}

    h_f, bin_freqs, delta_f_arr = fn(bin_arrays=bin_arrays, pulsar_cache=pulsar_cache,
                                     time_arr=time_arr, full_spectrum=True)
    freqs = np.asarray(bin_freqs[0], dtype=float)
    delta_f = float(np.asarray(delta_f_arr).ravel()[0])
    if delta_f <= 0:
        raise ValueError("delta_f must be positive.")
    spacing = np.diff(freqs)
    if not np.allclose(spacing, spacing[0], rtol=1e-6):
        raise ValueError("FFT frequency axis is not uniformly spaced.")

    S_h = 2.0 * np.sum(np.abs(h_f) ** 2, axis=0) / delta_f             # (N, F)
    denom = 4.0 * np.pi ** 2 * freqs ** 2
    S_GWB = np.zeros_like(S_h, dtype=float)
    np.divide(S_h, denom[None, :], out=S_GWB, where=(freqs > 0)[None, :])
    return freqs, S_GWB


def compute_population_gwb_psd(binaries, psrs, pulsar_cache, time_arr, freq_axis=None,
                               strain_fn=None):
    """
    GWB timing-residual PSD from a discrete SMBHB population (incoherent sum).
    `pulsar_cache` must provide 'raj_arr' and 'decj_arr'.

    Returns (freqs, S_GWB) with S_GWB of shape (N_psr, F) [s^3]; if `freq_axis`
    is given, S_GWB is log-interpolated onto it.
    """
    freqs, S_GWB = _gwb_psd_core(binaries, pulsar_cache, time_arr, strain_fn)
    if freq_axis is None:
        return freqs, S_GWB

    freq_axis = np.asarray(freq_axis, dtype=float)
    out = np.zeros((S_GWB.shape[0], len(freq_axis)))
    pos = freqs > 0
    for a in range(S_GWB.shape[0]):
        logS = np.log(np.clip(S_GWB[a, pos], 1e-300, None))
        fn = interp1d(np.log(freqs[pos]), logS, kind="linear", bounds_error=False,
                      fill_value=(logS[0], logS[-1]))
        out[a] = np.exp(fn(np.log(np.clip(freq_axis, 1e-300, None))))
    return freq_axis, out


def compute_population_gwb_psd_from_psrs(binaries, psrs, time_arr, strain_fn=None):
    """Same as compute_population_gwb_psd, building the minimal cache from Pulsar objects."""
    radec = np.array([_psr_radec(p) for p in psrs])
    cache = {"raj_arr": radec[:, 0], "decj_arr": radec[:, 1]}
    return _gwb_psd_core(binaries, cache, time_arr, strain_fn)


# ---------------------------------------------------------------------------
# Per-pulsar covariance C_a = N_a + S_RN,a + C_GWB,a
# ---------------------------------------------------------------------------
def _dense_white_noise(Nvec, n_toa):
    """Dense N from an ndarray diagonal, or from an enterprise Nvec object (incl. ECORR)."""
    if isinstance(Nvec, np.ndarray):
        return np.diag(Nvec) if Nvec.ndim == 1 else np.asarray(Nvec)
    try:
        return np.linalg.inv(np.asarray(Nvec.solve(np.eye(n_toa))))
    except Exception:
        for attr in ("_nvec", "diag", "_diag"):
            v = getattr(Nvec, attr, None)
            if v is not None:
                print(f"  Warning: dense N built from `{attr}` only (diagonal approximation).")
                return np.diag(np.asarray(v, dtype=float))
        raise


def get_per_pulsar_covariance_from_population(
    psrs, pta, noise_params, S_GWB, freqs_gwb,
    max_prior_variance=1e20, return_improper_basis=False,
):
    """
    Build C_a = N_a + T_p phi_p T_p^T + C_GWB,a and its Cholesky factor per pulsar.

    C_GWB[i,j] = sum_k S_GWB[a,k] cos(2 pi f_k (t_i - t_j)) df   (Wiener-Khinchin),
    computed via cos(a-b) = cos a cos b + sin a sin b, so memory is O(n_toa * F).

    Basis columns whose prior variance exceeds `max_prior_variance` (timing-model
    columns, phiinv ~ 0 or ~1e-40) are NOT put into C. They are improper priors and
    must be marginalised analytically; pass `return_improper_basis=True` to get them
    and use `cov_inner_product(..., M=basis)`.

    The PTA must NOT contain a common GWB process (it would be double counted).

    Returns (cov_matrices, chol_factors) or (cov_matrices, chol_factors, improper_bases).
    """
    freqs_gwb = np.asarray(freqs_gwb, dtype=float)
    df = freqs_gwb[1] - freqs_gwb[0]
    if not np.allclose(np.diff(freqs_gwb), df, rtol=1e-6):
        raise ValueError("freqs_gwb must be uniformly spaced.")
    if S_GWB.shape != (len(pta.pulsars), len(freqs_gwb)):
        raise ValueError(f"S_GWB shape {S_GWB.shape} != "
                         f"({len(pta.pulsars)}, {len(freqs_gwb)}).")

    Nvecs = pta.get_ndiag(noise_params)
    Ts = pta.get_basis()
    phiinvs = pta.get_phiinv(noise_params, logdet=False)
    psr_map = {psr.name: psr for psr in psrs}
    phiinv_cut = 1.0 / max_prior_variance

    cov_matrices, chol_factors, improper = {}, {}, {}
    for a, (name, Nvec, T, phiinv) in enumerate(zip(pta.pulsars, Nvecs, Ts, phiinvs)):
        toas = _psr_toas(psr_map[name])
        n_toa = len(toas)
        T = np.asarray(T, dtype=float)

        # --- white noise ---
        N_a = _dense_white_noise(Nvec, n_toa)

        # --- red noise on proper columns only ---
        diag = phiinv if phiinv.ndim == 1 else np.diag(phiinv)
        proper = diag > phiinv_cut
        improper[name] = T[:, ~proper]
        if np.any(proper):
            Tp = T[:, proper]
            if phiinv.ndim == 1:
                S_rn = (Tp / phiinv[proper]) @ Tp.T
            else:
                S_rn = Tp @ np.linalg.inv(phiinv[np.ix_(proper, proper)]) @ Tp.T
        else:
            S_rn = 0.0

        # --- GWB auto-covariance ---
        phase = 2.0 * np.pi * toas[:, None] * freqs_gwb[None, :]     # (n_toa, F)
        cosp, sinp = np.cos(phase), np.sin(phase)
        w = S_GWB[a] * df
        C_GWB = (cosp * w) @ cosp.T + (sinp * w) @ sinp.T

        C_a = N_a + S_rn + C_GWB
        C_a = 0.5 * (C_a + C_a.T)
        C_a = C_a + np.eye(n_toa) * (1e-10 * np.trace(C_a) / n_toa)

        cov_matrices[name] = C_a
        try:
            chol_factors[name] = cho_factor(C_a, lower=True)
        except LinAlgError as exc:
            raise RuntimeError(
                f"Cholesky failed for {name}; min eigenvalue "
                f"{np.linalg.eigvalsh(C_a).min():.3e}. Original error: {exc}")

    if return_improper_basis:
        return cov_matrices, chol_factors, improper
    return cov_matrices, chol_factors


def cov_inner_product(x, y, chol, M=None):
    """
    x^T C^-1 y, optionally marginalising improper columns M (e.g. timing model):
        x^T C^-1 y - (M^T C^-1 x)^T (M^T C^-1 M)^-1 (M^T C^-1 y)
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    Cy = cho_solve(chol, y)
    val = float(x @ Cy)
    if M is not None and M.shape[1] > 0:
        CM = cho_solve(chol, M)
        MCx, MCy = CM.T @ x, M.T @ Cy
        val -= float(MCx @ np.linalg.lstsq(M.T @ CM, MCy, rcond=None)[0])
    return val


# ---------------------------------------------------------------------------
# Mock world for the self-test
# ---------------------------------------------------------------------------
class _MockNvec:
    def __init__(self, diag):
        self.diag = np.asarray(diag, float)

    def solve(self, other, left_array=None, logdet=False):
        o = np.asarray(other, float)
        Ninv = o / (self.diag if o.ndim == 1 else self.diag[:, None])
        return Ninv if left_array is None else np.asarray(left_array).T @ Ninv


class _MockPTA:
    def __init__(self, psrs, T, phi, nvec):
        self.pulsars = [p.name for p in psrs]
        self._T, self._phi, self._N = T, phi, nvec

    def get_phiinv(self, params, logdet=False):
        return [1.0 / p for p in self._phi]

    def get_TNT(self, params):
        return [T.T @ (T / n[:, None]) for T, n in zip(self._T, self._N)]

    def get_basis(self, params=None):
        return self._T

    def get_ndiag(self, params=None):
        return [_MockNvec(n) for n in self._N]


def _mock_antenna(pra, pdec, sra, sdec, psi):
    sd = np.array([-np.cos(sdec) * np.cos(sra), -np.cos(sdec) * np.sin(sra), -np.sin(sdec)])
    pd = np.array([np.cos(pdec) * np.cos(pra), np.cos(pdec) * np.sin(pra), np.sin(pdec)])
    m = np.array([np.sin(sra), -np.cos(sra), 0.0])
    n = np.array([-np.sin(sdec) * np.cos(sra), -np.sin(sdec) * np.sin(sra), np.cos(sdec)])
    mr, nr = np.cos(psi) * m + np.sin(psi) * n, -np.sin(psi) * m + np.cos(psi) * n
    pm, pn, den = pd @ mr, pd @ nr, 1.0 + pd @ sd
    return 0.5 * (pm ** 2 - pn ** 2) / den, pm * pn / den


def _mock_pop_res(toas, psr, pop, Tspan, pulsar_noise_params=None):
    out = np.zeros(len(toas))
    for b in pop:
        fp, fx = _mock_antenna(float(psr._raj), float(psr._decj), b.ra, b.dec, b.psi)
        amp, c = b.h0 / (2 * np.pi * b.f), np.cos(b.iota)
        ph = 2 * np.pi * b.f * toas + b.phi0
        out += amp * (fp * (1 + c ** 2) * np.sin(ph) - fx * 2 * c * np.cos(ph))
    return out


def _mock_world(n_psr=4, n_toa=200, n_basis=6, seed=0):
    rng = np.random.default_rng(seed)
    Tspan = 10 * 3.15576e7
    psrs, Ts, phis, Ns = [], [], [], []
    for k in range(n_psr):
        toas = np.sort(rng.uniform(0, Tspan, n_toa))
        psrs.append(SimpleNamespace(
            name=f"J{k:04d}+0000_NANOGrav" if k % 2 else f"J{k:04d}+0000",
            toas=toas, toaerrs=np.full(n_toa, 1e-7),
            _raj=rng.uniform(0, 2 * np.pi), _decj=rng.uniform(-1, 1)))
        Ts.append(rng.normal(size=(n_toa, n_basis)))
        phis.append(np.full(n_basis, 1e-12))
        Ns.append((1e-7 * rng.uniform(1, 2, n_toa)) ** 2)
    return psrs, _MockPTA(psrs, Ts, phis, Ns), Tspan


def _mock_binary(ra=1.0, dec=0.3, h0=1e-13, f=3e-8):
    return SimpleNamespace(ra=ra, dec=dec, psi=0.4, iota=0.7, phi0=0.2, h0=h0, f=f,
                           Mc=1e9, Mtot=3e9, D_comov=100.0, z=0.05, ecc=0.0)


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


_MOCKS = dict(population_residuals_fn=_mock_pop_res, antenna_response_fn=_mock_antenna,
              get_base_name_fn=lambda n: n.split("_")[0])


# ---------------------------------------------------------------------------
# Self-test (mock data; needs no enterprise / project modules)
# ---------------------------------------------------------------------------
def cgw_selftest():
    F = compute_cgw_snr_optimal_population_fast
    tests = {}

    def test(fn):
        tests[fn.__name__] = fn
        return fn

    @test
    def dense_reference():
        psrs, pta, Ts = _mock_world()
        b = _mock_binary()
        r, _ = _quiet(F, psrs, pta, [b], None, {}, Ts, regularise_sigma=False, **_MOCKS)
        tot = 0.0
        for k, p in enumerate(psrs):
            s = _mock_pop_res(p.toas, p, [b], Ts)
            C = np.diag(pta._N[k]) + (pta._T[k] * pta._phi[k]) @ pta._T[k].T
            tot += s @ np.linalg.solve(C, s)
        rel = abs(r[0] ** 2 - tot) / tot
        assert rel < 1e-6, f"rel diff {rel:.2e}"
        return f"matches dense N+T*phi*T' reference (rel diff {rel:.1e})"

    @test
    def debug_does_not_change_result():
        psrs, pta, Ts = _mock_world()
        pop = [_mock_binary(1.0 + .1 * k) for k in range(3)]
        a, _ = _quiet(F, psrs, pta, pop, None, {}, Ts, debug=False, **_MOCKS)
        b, txt = _quiet(F, psrs, pta, pop, None, {}, Ts, debug=True,
                        debug_rho_sq_threshold=0.0, **_MOCKS)
        assert np.array_equal(a, b)
        assert "CGW debug source 0" in txt and "[CGW debug] source=2" in txt
        return "debug on/off identical; debug sections printed"

    @test
    def breakdown_sums_to_rho_sq():
        psrs, pta, Ts = _mock_world()
        (r, bd), _ = _quiet(F, psrs, pta, [_mock_binary()], None, {}, Ts,
                            return_breakdown=True, **_MOCKS)
        assert np.isclose(sum(bd[0].values()), r[0] ** 2, rtol=1e-10)
        return "breakdown sums to rho^2"

    @test
    def non_pd_sigma_falls_back_to_pinv():
        psrs, pta, Ts = _mock_world()
        pta._phi[2][0] = -1e-17
        r, txt = _quiet(F, psrs, pta, [_mock_binary()], None, {}, Ts, **_MOCKS)
        assert np.isfinite(r[0]) and "pinv fallback" in txt
        return "non-PD Sigma -> pinv fallback, finite result"

    @test
    def per_pulsar_failure_handling():
        psrs, pta, Ts = _mock_world()

        def flaky(toas, psr, pop, T, pulsar_noise_params=None):
            if psr is psrs[0]:
                raise ValueError("boom")
            return _mock_pop_res(toas, psr, pop, T)

        kw = dict(_MOCKS, population_residuals_fn=flaky)
        r, txt = _quiet(F, psrs, pta, [_mock_binary()], None, {}, Ts, **kw)
        assert np.isfinite(r[0]) and "failed" in txt and "biased low" in txt
        try:
            _quiet(F, psrs, pta, [_mock_binary()], None, {}, Ts, strict=True, **kw)
            raise AssertionError("strict=True should raise")
        except ValueError:
            pass

        def dead(*a, **k):
            raise ValueError("all")
        try:
            _quiet(F, psrs, pta, [_mock_binary()], None, {}, Ts,
                   **dict(_MOCKS, population_residuals_fn=dead))
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass
        return "skip+warn, strict raises, all-fail -> RuntimeError"

    @test
    def hostile_debug_inputs():
        psrs, pta, Ts = _mock_world()

        class Bad:
            def __repr__(self): raise RuntimeError("no repr")
            def __float__(self): raise RuntimeError("no float")

        b = _mock_binary()
        b.z, b.Mc = Bad(), np.array([1e9, 2e9])
        for p in psrs:
            del p.toaerrs
        r, _ = _quiet(F, psrs, pta, [b], None, {psrs[0].name: Bad()}, Ts,
                      debug=True, debug_rho_sq_threshold=0.0, **_MOCKS)
        assert np.isfinite(r[0])
        return "unprintable fields / missing toaerrs do not crash"

    @test
    def denominator_triggers_debug():
        psrs, pta, Ts = _mock_world()
        p = psrs[0]
        b = _mock_binary(ra=p._raj, dec=p._decj)      # source on pulsar -> denom ~ 0
        _, txt = _quiet(F, psrs, pta, [b], None, {}, Ts, debug=True,
                        debug_rho_sq_threshold=1e99, **_MOCKS)
        assert "antenna denom" in txt
        return "denominator threshold triggers debug block"

    @test
    def gwb_psd_scaling():
        B, N, Fn, df = 3, 2, 8, 1e-9
        freqs = np.arange(Fn) * df                    # includes f = 0
        h = np.full((B, N, Fn), 1e-10 + 0j)

        def fake(bin_arrays, pulsar_cache, time_arr, full_spectrum):
            assert "h0" in bin_arrays
            return h, np.tile(freqs, (B, 1)), np.full(B, df)

        bins = [_mock_binary() for _ in range(B)]
        fr, S = compute_population_gwb_psd(bins, None, {}, None, strain_fn=fake)
        expect = 2 * B * 1e-20 / df / (4 * np.pi ** 2 * freqs[1:] ** 2)
        assert np.allclose(S[:, 1:], expect) and np.all(S[:, 0] == 0)
        fr2, S2 = compute_population_gwb_psd(bins, None, {}, None,
                                             freq_axis=freqs[1:] * 1.5, strain_fn=fake)
        assert S2.shape == (N, Fn - 1) and np.all(np.isfinite(S2))
        return "S = 2 B |h|^2 / (df 4 pi^2 f^2); f=0 bin zeroed; interpolation ok"

    @test
    def covariance_matches_direct_sum():
        psrs, pta, Ts = _mock_world(n_toa=60)
        freqs = np.arange(1, 21) * 1e-9
        S = np.abs(np.random.default_rng(1).normal(size=(4, 20))) * 1e3
        cov, chol = get_per_pulsar_covariance_from_population(psrs, pta, {}, S, freqs)
        for a, p in enumerate(psrs):
            dt = p.toas[:, None] - p.toas[None, :]
            C = np.diag(pta._N[a]) + (pta._T[a] * pta._phi[a]) @ pta._T[a].T
            C += sum(S[a, k] * np.cos(2 * np.pi * freqs[k] * dt) * 1e-9 for k in range(20))
            err = np.abs(cov[p.name] - C).max() / np.abs(C).max()
            assert err < 1e-6, f"{p.name}: {err:.2e}"
        return "fast cos/sin Wiener-Khinchin matches direct sum"

    @test
    def improper_columns_marginalised():
        psrs, pta, Ts = _mock_world(n_toa=80)
        pta._phi[0][0] = 1e30                          # improper (timing-model-like)
        freqs = np.arange(1, 6) * 1e-9
        S = np.full((4, 5), 1e2)
        cov, chol, M = get_per_pulsar_covariance_from_population(
            psrs, pta, {}, S, freqs, return_improper_basis=True)
        name = pta.pulsars[0]
        assert M[name].shape[1] == 1 and np.all(np.isfinite(cov[name]))
        assert np.linalg.eigvalsh(cov[name]).min() > 0
        s = _mock_pop_res(psrs[0].toas, psrs[0], [_mock_binary()], Ts)
        Cinv_s = np.linalg.solve(cov[name], s)
        Mm = M[name]
        ref = s @ Cinv_s - (Mm.T @ Cinv_s) @ np.linalg.solve(
            Mm.T @ np.linalg.solve(cov[name], Mm), Mm.T @ Cinv_s)
        got = cov_inner_product(s, s, chol[name], M=Mm)
        assert np.isclose(got, ref, rtol=1e-6), (got, ref)
        return "improper columns excluded from C and marginalised in inner product"

    failed = 0
    for name, fn in tests.items():
        try:
            print(f"PASS  {name}: {fn()}")
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed == 0


# ---------------------------------------------------------------------------
# Live sanity check on YOUR real objects
# ---------------------------------------------------------------------------
def cgw_live_check(psrs, pta, population, raw_noise_params, parsed_noise_params,
                   Tspan, n_sources=3):
    F = compute_cgw_snr_optimal_population_fast
    pop = [population[i] for i in range(min(n_sources, len(population)))]
    args = (psrs, pta, pop, raw_noise_params, parsed_noise_params, Tspan)

    off, _ = _quiet(F, *args, debug=False)
    on, _ = _quiet(F, *args, debug=True, debug_rho_sq_threshold=0.0)
    print("debug on/off identical :", np.array_equal(off, on))

    (r, bd), _ = _quiet(F, *args, debug=False, return_breakdown=True)
    print("breakdown sum == rho^2 :", np.allclose([sum(d.values()) for d in bd], r ** 2, rtol=1e-8))

    # independent check for source 0: plain np.linalg.solve on unregularised Sigma
    phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
    TNTs, Ts = pta.get_TNT(raw_noise_params), pta.get_basis()
    Nvecs = pta.get_ndiag(raw_noise_params)
    psr_map = {p.name: p for p in psrs}
    ref = 0.0
    for name, N, TNT, pi, T in zip(pta.pulsars, Nvecs, TNTs, phiinvs, Ts):
        p = psr_map[name]
        noise = parsed_noise_params.get(name, parsed_noise_params.get(get_base_name(name), {})) or {}
        s = population_residuals(_psr_toas(p), p, [pop[0]], Tspan, pulsar_noise_params=noise)
        Sig = TNT + (np.diag(pi) if pi.ndim == 1 else pi)
        TNs = N.solve(s, left_array=T)
        ref += float(N.solve(s, left_array=s) - TNs @ np.linalg.solve(Sig, TNs))
    print(f"independent rho^2 check : fast={r[0] ** 2:.10e} ref={ref:.10e} "
          f"rel diff={abs(r[0] ** 2 - ref) / max(abs(ref), 1e-300):.2e}")
    print("rho per source          :", r)
    return r


if __name__ == "__main__":
    raise SystemExit(0 if cgw_selftest() else 1)