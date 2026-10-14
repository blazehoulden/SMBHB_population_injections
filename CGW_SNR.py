import numpy as np
from enterprise_extensions.frequentist.Fe_statistic import innerProduct_rr
from enterprise_extensions.deterministic import cw_delay
from optimal_SNR_calc import measured_strain_all_binaries_all_pulsars
from signal_injection import (
    population_residuals,
    get_base_name,
    population_residuals_eccentric,
    antenna_response,
)
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


# def compute_cgw_snr_optimal_population_fast(
#     psrs,
#     pta,
#     population,
#     raw_noise_params,
#     parsed_noise_params,
#     Tspan,
#     profile=False,
#     return_breakdown=False,
#     regularise_sigma=True,
#     regularisation=1e-10,
#     power_tol=1e-4,
#     n_max_cap=100,
#     debug=True,
#     debug_denominator_threshold=1e-3,
#     debug_rho_sq_threshold=1.0,
# ):

#     import time
#     t0 = time.time()

#     phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
#     TNTs    = pta.get_TNT(raw_noise_params)
#     Ts      = pta.get_basis()
#     Nvecs   = pta.get_ndiag(raw_noise_params)

#     psr_map = {psr.name: psr for psr in psrs}

#     precomputed = []

#     for psr_name, Nvec, TNT, phiinv, T in zip(
#         pta.pulsars, Nvecs, TNTs, phiinvs, Ts,
#     ):
#         Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

#         if regularise_sigma:
#             # Small diagonal regularisation, physically equivalent to a
#             # tiny additional white-noise floor; negligible effect on SNR
#             # for any reasonable signal amplitude, but improves robustness
#             # against marginal ill-conditioning.
#             Sigma += regularisation * np.eye(Sigma.shape[0])

#         cf = None
#         max_reg_attempts = 10
#         for reg_attempt in range(max_reg_attempts):
#             try:
#                 cf = cho_factor(Sigma)
#                 break
#             except LinAlgError:
#                 reg = regularisation * (10 ** (reg_attempt + 1))
#                 if reg_attempt == 0 or reg_attempt == max_reg_attempts - 1:
#                     print(f"  Warning: {psr_name} Sigma not PD via Cholesky, "
#                           f"increasing regularisation to {reg:.2e} "
#                           f"(attempt {reg_attempt + 1}/{max_reg_attempts})")
#                 Sigma += reg * np.eye(Sigma.shape[0])

#         if cf is None:
#             # Cholesky never succeeded even at max regularisation. Rather
#             # than aborting the whole run, fall back to a direct solve
#             # (matches enterprise_extensions' OptimalStatistic.compute_os
#             # behaviour, which only requires Sigma to be non-singular, not
#             # strictly positive-definite). We wrap this in a small wrapper
#             # object exposing the same interface fast_inner_product_rr_exact
#             # expects from cho_solve, so downstream code is unchanged.
#             print(f"  Warning: {psr_name} Sigma still not PD after "
#                   f"{max_reg_attempts} regularisation attempts; falling "
#                   f"back to np.linalg.solve (non-PD-safe).")

#             try:
#                 Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)
#             except np.linalg.LinAlgError as e:
#                 raise RuntimeError(
#                     f"Could not invert Sigma for {psr_name} even via "
#                     f"pseudo-inverse. Check noise parameters for this "
#                     f"pulsar (likely a degenerate/duplicated noise basis "
#                     f"or pathological amplitude hierarchy). Original "
#                     f"error: {e}"
#                 )

#             class _SolveFallback:
#                 """Mimics scipy's cho_factor/cho_solve interface using a
#                 precomputed pseudo-inverse, for pulsars whose Sigma is not
#                 strictly positive-definite."""
#                 def __init__(self, Minv):
#                     self.Minv = Minv

#             cf = _SolveFallback(Sigma_inv)

#         # Get TOAs safely — handle both property and method forms
#         psr_obj = psr_map[psr_name]
#         try:
#             toas = psr_obj.toas
#             if callable(toas):
#                 toas = toas()
#             toas = np.asarray(toas)
#         except Exception:
#             # Last resort: get from the underlying enterprise object
#             toas = np.asarray(psr_obj._psr.toas)

#         precomputed.append((
#             psr_name,
#             psr_obj,
#             toas,
#             parsed_noise_params.get(psr_name,
#                 parsed_noise_params.get(get_base_name(psr_name), {})),
#             Nvec,
#             T,
#             cf,
#         ))

#     if profile:
#         print(f"Precompute time: {time.time()-t0:.2f} s")

#     def _solve(cf, x):
#         if isinstance(cf, tuple):
#             return cho_solve(cf, x)
#         else:
#             return cf.Minv @ x

#     def fast_inner_product_rr_exact(x, y, Nvec, Tmat, cf):
#         TNy = Nvec.solve(y, left_array=Tmat)
#         TNx = Nvec.solve(x, left_array=Tmat)
#         xNy = Nvec.solve(y, left_array=x)
#         SigmaTNy = _solve(cf, TNy)
#         return xNy - TNx.T @ SigmaTNy

#     results   = np.empty(len(population), dtype=np.float64)
#     breakdowns = [] if return_breakdown else None

#     N_binary = len(population)
#     results    = np.empty(N_binary, dtype=np.float64)
#     breakdowns = [] if return_breakdown else None

#     for i in range(N_binary):
#         binary = population[i]   # however your candidate list yields single binaries
#                                   # (e.g. a _MiniPop-like scalar-attribute object,
#                                   # NOT a PopulationArrays slice)
#         rho_sq     = 0.0
#         per_pulsar = {} if return_breakdown else None
#         n_failed   = 0
#         source_debugged = False

#         for (psr_name, psr_obj, toas, psr_noise_params,
#              Nvec, T, cf) in precomputed:

#             try: # In PTA searches, they do not scan over the eccentricity parameter space
#                 # s_a = population_residuals_eccentric(
#                 s_a = population_residuals(
#                     toas, psr_obj, [binary], Tspan,
#                     pulsar_noise_params=psr_noise_params,
#                     # power_tol=power_tol, n_max_cap=n_max_cap,
#                 )
#             except Exception as e:
#                 n_failed += 1
#                 print(f"  Warning: population_residuals failed for {psr_name}: {e}")
#                 continue

#             psr_ra = float(psr_obj._raj)
#             psr_dec = float(psr_obj._decj)
#             src_ra = float(binary.ra)
#             src_dec = float(binary.dec)
#             psi = float(binary.psi)
#             fp, fx = antenna_response(psr_ra, psr_dec, src_ra, src_dec, psi)
#             source_direction = np.array([
#                 -np.cos(src_dec) * np.cos(src_ra),
#                 -np.cos(src_dec) * np.sin(src_ra),
#                 -np.sin(src_dec),
#             ])
#             pulsar_direction = np.array([
#                 np.cos(psr_dec) * np.cos(psr_ra),
#                 np.cos(psr_dec) * np.sin(psr_ra),
#                 np.sin(psr_dec),
#             ])
#             m_hat = np.array([np.sin(src_ra), -np.cos(src_ra), 0.0])
#             n_hat = np.array([
#                 -np.sin(src_dec) * np.cos(src_ra),
#                 -np.sin(src_dec) * np.sin(src_ra),
#                 np.cos(src_dec),
#             ])
#             m_rot = np.cos(psi) * m_hat + np.sin(psi) * n_hat
#             n_rot = -np.sin(psi) * m_hat + np.cos(psi) * n_hat
#             p_m = float(np.dot(pulsar_direction, m_rot))
#             p_n = float(np.dot(pulsar_direction, n_rot))
#             denominator = float(1.0 + np.dot(pulsar_direction, source_direction))
#             separation = float(np.arccos(np.clip(
#                 np.sin(psr_dec) * np.sin(src_dec)
#                 + np.cos(psr_dec) * np.cos(src_dec)
#                 * np.cos(psr_ra - src_ra),
#                 -1.0,
#                 1.0,
#             )))
#             contrib = float(np.real(
#                 fast_inner_product_rr_exact(s_a, s_a, Nvec, T, cf)
#             ))
#             rho_sq += contrib

#             if debug and (
#                 abs(denominator) <= debug_denominator_threshold
#                 or contrib >= debug_rho_sq_threshold
#             ):
#                 if not source_debugged:
#                     print("\n" + "=" * 78)
#                     print(f"CGW debug source {i}")
#                     for field in ("f", "Mc", "Mtot", "D_comov", "z", "h0",
#                                   "ra", "dec", "psi", "iota", "phi0", "ecc"):
#                         if hasattr(binary, field):
#                             print(f"  source.{field:<8} = "
#                                   f"{getattr(binary, field)!r} "
#                                   f"(float64={float(getattr(binary, field)):.17g})")
#                     print("=" * 78)
#                     source_debugged = True
#                 print(f"\n[CGW debug] source={i} pulsar={psr_name}")
#                 print(f"  pulsar type       = {type(psr_obj).__name__}")
#                 print(f"  pulsar RA/Dec     = {psr_ra:.17g}, {psr_dec:.17g} rad")
#                 print(f"  pulsar RA/Dec     = {np.degrees(psr_ra):.12f}, "
#                       f"{np.degrees(psr_dec):.12f} deg")
#                 print(f"  pulsar n_toa      = {len(toas)}")
#                 print(f"  pulsar TOA range  = {np.min(toas):.17g} .. "
#                       f"{np.max(toas):.17g} s")
#                 if hasattr(psr_obj, "toaerrs"):
#                     toaerrs = np.asarray(psr_obj.toaerrs, dtype=float)
#                     print(f"  TOA errors        = min {toaerrs.min():.6e}, "
#                           f"median {np.median(toaerrs):.6e}, "
#                           f"max {toaerrs.max():.6e} s")
#                 print(f"  noise parameters  = {psr_noise_params}")
#                 print(f"  Nvec              = min {np.min(Nvec):.6e}, "
#                       f"median {np.median(Nvec):.6e}, "
#                       f"max {np.max(Nvec):.6e}")
#                 print(f"  T shape           = {np.shape(T)}")
#                 print(f"  source separation = {np.degrees(separation):.12f} deg")
#                 print(f"  antenna denom     = {denominator:.16e}")
#                 print(f"  p·m_rot, p·n_rot = {p_m:.16e}, {p_n:.16e}")
#                 print(f"  tensor numerators = F+ {0.5 * (p_m**2 - p_n**2):.16e}, "
#                       f"Fx {p_m * p_n:.16e}")
#                 print(f"  F_plus/F_cross    = {float(fp):.16e}, {float(fx):.16e}")
#                 residual_norm = float(binary.h0) / (2.0 * np.pi * float(binary.f))
#                 print(f"  h0/(2πf)         = {residual_norm:.16e} s")
#                 print(f"  h+ coefficient    = "
#                       f"{residual_norm * (1.0 + np.cos(float(binary.iota))**2):.16e} s")
#                 print(f"  hx coefficient    = "
#                       f"{residual_norm * (-2.0 * np.cos(float(binary.iota))):.16e} s")
#                 print(f"  residual stats    = min {np.min(s_a):.6e}, "
#                       f"rms {np.sqrt(np.mean(s_a**2)):.6e}, "
#                       f"max {np.max(s_a):.6e} s")
#                 print(f"  inner product     = {contrib:.16e}")
#                 print(f"  cumulative rho^2  = {rho_sq:.16e}")

#             if return_breakdown:
#                 base_name = get_base_name(psr_name)
#                 per_pulsar[base_name] = per_pulsar.get(base_name, 0.0) + max(contrib, 0.0)

#         if n_failed == len(precomputed):
#             raise RuntimeError(
#                 f"population_residuals failed for ALL {n_failed} "
#                 f"pulsars on binary {i} — likely structural, not per-pulsar. "
#                 f"See warnings above."
#             )

#         results[i] = np.sqrt(max(rho_sq, 0.0))
#         if debug and source_debugged:
#             print(f"[CGW debug] source={i} final rho^2={rho_sq:.16e}, "
#                   f"rho={results[i]:.16e}")
#         if return_breakdown:
#             breakdowns.append(per_pulsar)

#     if profile:
#         print(f"Total runtime: {time.time()-t0:.2f} s")

#     return (results, breakdowns) if return_breakdown else results
 
# =====================================================================
# cgw_snr_paste_block.py  --  paste this whole chunk into a cell/file.
# Needs in your namespace (only for REAL runs): population_residuals,
# antenna_response, get_base_name.  The self-test needs nothing extra.
#
#   cgw_selftest()                                   # mock data, no inputs
#   cgw_live_check(psrs, pta, population, raw_noise_params,
#                  parsed_noise_params, Tspan)       # your real objects
# =====================================================================
import io, time, contextlib
from types import SimpleNamespace
import numpy as np
from scipy.linalg import cho_factor, cho_solve

LinAlgError = np.linalg.LinAlgError


# ---------------------------------------------------------------------
# helpers (debug printing can never crash a run)
# ---------------------------------------------------------------------
def _cgw_nvec_diag(Nvec):
    for attr in ("_nvec", "diag", "_diag"):
        v = getattr(Nvec, attr, None)
        if v is not None:
            try:
                return np.asarray(v, dtype=float)
            except Exception:
                pass
    return np.asarray(Nvec, dtype=float)


def _cgw_f64(x):
    try:
        return f"{float(x):.17g}"
    except Exception:
        return "n/a"


def _cgw_print_debug(i, psr_name, psr_obj, toas, noise, Nvec, T, binary, s_a,
                     psr_ra, psr_dec, fp, fx, denominator, p_m, p_n,
                     separation, contrib, rho_sq, header_done):
    try:
        if not header_done:
            print("\n" + "=" * 78)
            print(f"CGW debug source {i}")
            for field in ("f", "Mc", "Mtot", "D_comov", "z", "h0",
                          "ra", "dec", "psi", "iota", "phi0", "ecc"):
                if hasattr(binary, field):
                    v = getattr(binary, field)
                    try:
                        r = repr(v)
                    except Exception:
                        r = "<unprintable>"
                    print(f"  source.{field:<8} = {r} (float64={_cgw_f64(v)})")
            print("=" * 78)
        print(f"\n[CGW debug] source={i} pulsar={psr_name}")
        print(f"  pulsar type       = {type(psr_obj).__name__}")
        print(f"  pulsar RA/Dec     = {psr_ra:.17g}, {psr_dec:.17g} rad")
        print(f"  pulsar n_toa      = {len(toas)}")
        print(f"  pulsar TOA range  = {np.min(toas):.17g} .. {np.max(toas):.17g} s")
        if hasattr(psr_obj, "toaerrs"):
            te = np.asarray(psr_obj.toaerrs, dtype=float)
            print(f"  TOA errors        = min {te.min():.6e}, "
                  f"median {np.median(te):.6e}, max {te.max():.6e} s")
        else:
            print("  TOA errors        = unavailable")
        try:
            print(f"  noise parameters  = {noise}")
        except Exception:
            print("  noise parameters  = <unprintable>")
        try:
            d = _cgw_nvec_diag(Nvec)
            print(f"  Nvec              = min {d.min():.6e}, "
                  f"median {np.median(d):.6e}, max {d.max():.6e}")
        except Exception:
            print("  Nvec              = unavailable")
        print(f"  T shape           = {np.shape(T)}")
        print(f"  source separation = {np.degrees(separation):.12f} deg")
        print(f"  antenna denom     = {denominator:.16e}")
        print(f"  p.m_rot, p.n_rot  = {p_m:.16e}, {p_n:.16e}")
        print(f"  F_plus/F_cross    = {float(fp):.16e}, {float(fx):.16e}")
        print(f"  residual stats    = min {np.min(s_a):.6e}, "
              f"rms {np.sqrt(np.mean(s_a**2)):.6e}, max {np.max(s_a):.6e} s")
        print(f"  inner product     = {contrib:.16e}")
        print(f"  cumulative rho^2  = {rho_sq:.16e}")
    except Exception as e:
        print(f"  [CGW debug] output failed ({type(e).__name__}: {e})")


# ---------------------------------------------------------------------
# the function under test (your commented-out block, uncommented)
# last three kwargs let the self-test inject mocks; default = your globals
# ---------------------------------------------------------------------
def compute_cgw_snr_optimal_population_fast(
    psrs, pta, population, raw_noise_params, parsed_noise_params, Tspan,
    profile=False, return_breakdown=False, regularise_sigma=True,
    regularisation=1e-10, power_tol=1e-4, n_max_cap=100, debug=True,
    debug_denominator_threshold=1e-3, debug_rho_sq_threshold=1.0,
    population_residuals_fn=None, antenna_response_fn=None, get_base_name_fn=None,
):
    t0 = time.time()
    _pop_res = population_residuals_fn or population_residuals
    _ant = antenna_response_fn or antenna_response
    _base = get_base_name_fn or get_base_name

    phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
    TNTs = pta.get_TNT(raw_noise_params)
    Ts = pta.get_basis()
    Nvecs = pta.get_ndiag(raw_noise_params)
    psr_map = {psr.name: psr for psr in psrs}
    precomputed = []

    for psr_name, Nvec, TNT, phiinv, T in zip(pta.pulsars, Nvecs, TNTs, phiinvs, Ts):
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        if regularise_sigma:
            Sigma = Sigma + regularisation * np.eye(Sigma.shape[0])

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
                Sigma = Sigma + reg * np.eye(Sigma.shape[0])

        if cf is None:
            print(f"  Warning: {psr_name} Sigma still not PD after "
                  f"{max_reg_attempts} regularisation attempts; pinv fallback.")
            try:
                Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"Could not invert Sigma for {psr_name} "
                                   f"even via pseudo-inverse: {e}")

            class _SolveFallback:
                def __init__(self, Minv):
                    self.Minv = Minv
            cf = _SolveFallback(Sigma_inv)

        psr_obj = psr_map[psr_name]
        try:
            toas = psr_obj.toas
            if callable(toas):
                toas = toas()
            toas = np.asarray(toas)
        except Exception:
            toas = np.asarray(psr_obj._psr.toas)

        precomputed.append((
            psr_name, psr_obj, toas,
            parsed_noise_params.get(psr_name,
                parsed_noise_params.get(_base(psr_name), {})) or {},
            Nvec, T, cf,
        ))

    if profile:
        print(f"Precompute time: {time.time()-t0:.2f} s")

    def _solve(cf, x):
        return cho_solve(cf, x) if isinstance(cf, tuple) else cf.Minv @ x

    def inner(x, y, Nvec, Tmat, cf):
        TNy = Nvec.solve(y, left_array=Tmat)
        TNx = Nvec.solve(x, left_array=Tmat)
        xNy = Nvec.solve(y, left_array=x)
        return xNy - TNx.T @ _solve(cf, TNy)

    def radec(p):
        try:
            return float(p._raj), float(p._decj)
        except AttributeError:
            return tuple(float(v) for v in globals()["_get_psr_radec"](p))

    N_binary = len(population)
    results = np.empty(N_binary, dtype=np.float64)
    breakdowns = [] if return_breakdown else None

    for i in range(N_binary):
        binary = population[i]
        rho_sq = 0.0
        per_pulsar = {} if return_breakdown else None
        n_failed = 0
        source_debugged = False

        for (psr_name, psr_obj, toas, noise, Nvec, T, cf) in precomputed:
            try:
                s_a = _pop_res(toas, psr_obj, [binary], Tspan,
                               pulsar_noise_params=noise)
            except Exception as e:
                n_failed += 1
                print(f"  Warning: population_residuals failed for {psr_name}: {e}")
                continue

            psr_ra, psr_dec = radec(psr_obj)
            src_ra, src_dec, psi = float(binary.ra), float(binary.dec), float(binary.psi)
            fp, fx = _ant(psr_ra, psr_dec, src_ra, src_dec, psi)
            contrib = float(np.real(inner(s_a, s_a, Nvec, T, cf)))
            rho_sq += contrib

            if debug:
                src_dir = np.array([-np.cos(src_dec) * np.cos(src_ra),
                                    -np.cos(src_dec) * np.sin(src_ra),
                                    -np.sin(src_dec)])
                p_dir = np.array([np.cos(psr_dec) * np.cos(psr_ra),
                                  np.cos(psr_dec) * np.sin(psr_ra),
                                  np.sin(psr_dec)])
                denominator = float(1.0 + p_dir @ src_dir)
                if (abs(denominator) <= debug_denominator_threshold
                        or contrib >= debug_rho_sq_threshold):
                    m_hat = np.array([np.sin(src_ra), -np.cos(src_ra), 0.0])
                    n_hat = np.array([-np.sin(src_dec) * np.cos(src_ra),
                                      -np.sin(src_dec) * np.sin(src_ra),
                                      np.cos(src_dec)])
                    m_rot = np.cos(psi) * m_hat + np.sin(psi) * n_hat
                    n_rot = -np.sin(psi) * m_hat + np.cos(psi) * n_hat
                    sep = float(np.arccos(np.clip(
                        np.sin(psr_dec) * np.sin(src_dec)
                        + np.cos(psr_dec) * np.cos(src_dec) * np.cos(psr_ra - src_ra),
                        -1.0, 1.0)))
                    _cgw_print_debug(i, psr_name, psr_obj, toas, noise, Nvec, T,
                                     binary, s_a, psr_ra, psr_dec, fp, fx,
                                     denominator, float(p_dir @ m_rot),
                                     float(p_dir @ n_rot), sep, contrib, rho_sq,
                                     source_debugged)
                    source_debugged = True

            if return_breakdown:
                b = _base(psr_name)
                per_pulsar[b] = per_pulsar.get(b, 0.0) + max(contrib, 0.0)

        if n_failed == len(precomputed):
            raise RuntimeError(f"population_residuals failed for ALL {n_failed} "
                               f"pulsars on binary {i}. See warnings above.")

        results[i] = np.sqrt(max(rho_sq, 0.0))
        if debug and source_debugged:
            print(f"[CGW debug] source={i} final rho^2={rho_sq:.16e}, rho={results[i]:.16e}")
        if return_breakdown:
            breakdowns.append(per_pulsar)

    if profile:
        print(f"Total runtime: {time.time()-t0:.2f} s")
    return (results, breakdowns) if return_breakdown else results


# ---------------------------------------------------------------------
# mock world for the self-test
# ---------------------------------------------------------------------
class _MockNvec:
    def __init__(self, diag): self.diag = np.asarray(diag, float)
    def solve(self, other, left_array=None, logdet=False):
        o = np.asarray(other, float)
        Ninv = o / (self.diag if o.ndim == 1 else self.diag[:, None])
        return Ninv if left_array is None else np.asarray(left_array).T @ Ninv


class _MockPTA:
    def __init__(self, psrs, T, phi, nvec):
        self.pulsars = [p.name for p in psrs]
        self._T, self._phi, self._N = T, phi, nvec
    def get_phiinv(self, params, logdet=False): return [1.0 / p for p in self._phi]
    def get_TNT(self, params): return [T.T @ (T / n[:, None]) for T, n in zip(self._T, self._N)]
    def get_basis(self, params=None): return self._T
    def get_ndiag(self, params=None): return [_MockNvec(n) for n in self._N]


def _mock_antenna(pra, pdec, sra, sdec, psi):
    sd = np.array([-np.cos(sdec) * np.cos(sra), -np.cos(sdec) * np.sin(sra), -np.sin(sdec)])
    pd = np.array([np.cos(pdec) * np.cos(pra), np.cos(pdec) * np.sin(pra), np.sin(pdec)])
    m = np.array([np.sin(sra), -np.cos(sra), 0.0])
    n = np.array([-np.sin(sdec) * np.cos(sra), -np.sin(sdec) * np.sin(sra), np.cos(sdec)])
    mr, nr = np.cos(psi) * m + np.sin(psi) * n, -np.sin(psi) * m + np.cos(psi) * n
    pm, pn, den = pd @ mr, pd @ nr, 1.0 + pd @ sd
    return 0.5 * (pm**2 - pn**2) / den, pm * pn / den


def _mock_pop_res(toas, psr, pop, Tspan, pulsar_noise_params=None):
    out = np.zeros(len(toas))
    for b in pop:
        fp, fx = _mock_antenna(float(psr._raj), float(psr._decj), b.ra, b.dec, b.psi)
        amp, c = b.h0 / (2 * np.pi * b.f), np.cos(b.iota)
        ph = 2 * np.pi * b.f * toas + b.phi0
        out += amp * (fp * (1 + c**2) * np.sin(ph) - fx * 2 * c * np.cos(ph))
    return out


def _mock_world(n_psr=4, n_toa=200, n_basis=6, seed=0):
    rng = np.random.default_rng(seed)
    Tspan = 10 * 3.15576e7
    psrs, Ts, phis, Ns = [], [], [], []
    for k in range(n_psr):
        toas = np.sort(rng.uniform(0, Tspan, n_toa))
        psrs.append(SimpleNamespace(name=f"J{k:04d}+0000_NANOGrav" if k % 2 else f"J{k:04d}+0000",
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


# ---------------------------------------------------------------------
# self-test (mock data)
# ---------------------------------------------------------------------
def cgw_selftest():
    F = compute_cgw_snr_optimal_population_fast
    tests = {}

    def test(fn): tests[fn.__name__] = fn; return fn

    @test
    def dense_reference():
        psrs, pta, Ts = _mock_world(); b = _mock_binary()
        r, _ = _quiet(F, psrs, pta, [b], None, {}, Ts, regularise_sigma=False,
                      debug=False, **_MOCKS)
        tot = 0.0
        for k, p in enumerate(psrs):
            s = _mock_pop_res(p.toas, p, [b], Ts)
            C = np.diag(pta._N[k]) + (pta._T[k] * pta._phi[k]) @ pta._T[k].T
            tot += s @ np.linalg.solve(C, s)
        rel = abs(r[0]**2 - tot) / tot
        assert rel < 1e-6, f"rel diff {rel:.2e}"
        return f"matches dense N+T*phi*T' reference (rel diff {rel:.1e})"

    @test
    def debug_does_not_change_result():
        psrs, pta, Ts = _mock_world(); pop = [_mock_binary(1.0 + .1 * k) for k in range(3)]
        a, _ = _quiet(F, psrs, pta, pop, None, {}, Ts, debug=False, **_MOCKS)
        b, txt = _quiet(F, psrs, pta, pop, None, {}, Ts, debug=True,
                        debug_rho_sq_threshold=0.0, **_MOCKS)
        assert np.array_equal(a, b)
        assert "CGW debug source 0" in txt and "[CGW debug] source=2" in txt
        return "debug on/off identical; debug sections printed"

    @test
    def breakdown_sums_to_rho_sq():
        psrs, pta, Ts = _mock_world(); b = _mock_binary()
        (r, bd), _ = _quiet(F, psrs, pta, [b], None, {}, Ts, return_breakdown=True,
                            debug=False, **_MOCKS)
        assert np.isclose(sum(bd[0].values()), r[0]**2, rtol=1e-10)
        return "breakdown sums to rho^2"

    @test
    def non_pd_sigma_falls_back_to_pinv():
        psrs, pta, Ts = _mock_world(); pta._phi[2][0] = -1e-17   # phiinv very negative
        r, txt = _quiet(F, psrs, pta, [_mock_binary()], None, {}, Ts, debug=False, **_MOCKS)
        assert np.isfinite(r[0]) and "pinv fallback" in txt
        return "non-PD Sigma -> pinv fallback, finite result"

    @test
    def per_pulsar_failure_contained():
        psrs, pta, Ts = _mock_world()
        def flaky(toas, psr, pop, T, pulsar_noise_params=None):
            if psr is psrs[0]: raise ValueError("boom")
            return _mock_pop_res(toas, psr, pop, T)
        kw = dict(_MOCKS, population_residuals_fn=flaky)
        r, txt = _quiet(F, psrs, pta, [_mock_binary()], None, {}, Ts, debug=False, **kw)
        assert np.isfinite(r[0]) and "failed for" in txt
        def dead(*a, **k): raise ValueError("all")
        try:
            _quiet(F, psrs, pta, [_mock_binary()], None, {}, Ts, debug=False,
                   **dict(_MOCKS, population_residuals_fn=dead))
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass
        return "one failing pulsar skipped; all failing -> RuntimeError"

    @test
    def hostile_debug_inputs():
        psrs, pta, Ts = _mock_world()
        class Bad:
            def __repr__(self): raise RuntimeError("no repr")
            def __float__(self): raise RuntimeError("no float")
        b = _mock_binary(); b.z = Bad(); b.Mc = np.array([1e9, 2e9])
        for p in psrs: del p.toaerrs
        r, txt = _quiet(F, psrs, pta, [b], None, {psrs[0].name: Bad()}, Ts,
                        debug=True, debug_rho_sq_threshold=0.0, **_MOCKS)
        assert np.isfinite(r[0])
        return "unprintable fields / missing toaerrs do not crash"

    @test
    def far_source_triggers_denominator_debug():
        psrs, pta, Ts = _mock_world()
        p = psrs[0]
        b = _mock_binary(ra=p._raj, dec=p._decj)                 # source on pulsar -> denom ~ 0
        _, txt = _quiet(F, psrs, pta, [b], None, {}, Ts, debug=True,
                        debug_rho_sq_threshold=1e99, **_MOCKS)
        assert "antenna denom" in txt
        return "denominator threshold triggers debug block"

    failed = 0
    for name, fn in tests.items():
        try:
            print(f"PASS  {name}: {fn()}")
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed == 0


# ---------------------------------------------------------------------
# live sanity check on YOUR real objects (uses your real functions)
# ---------------------------------------------------------------------
def cgw_live_check(psrs, pta, population, raw_noise_params, parsed_noise_params,
                   Tspan, n_sources=3):
    F = compute_cgw_snr_optimal_population_fast
    pop = [population[i] for i in range(min(n_sources, len(population)))]
    args = (psrs, pta, pop, raw_noise_params, parsed_noise_params, Tspan)

    off, _ = _quiet(F, *args, debug=False)
    on, txt = _quiet(F, *args, debug=True, debug_rho_sq_threshold=0.0)
    print("debug on/off identical :", np.array_equal(off, on))

    (r, bd), _ = _quiet(F, *args, debug=False, return_breakdown=True)
    print("breakdown sum == rho^2 :", np.allclose([sum(d.values()) for d in bd], r**2, rtol=1e-8))

    # independent check: np.linalg.solve on unregularised Sigma, no Cholesky
    phiinvs = pta.get_phiinv(raw_noise_params, logdet=False)
    TNTs, Ts, Nvecs = pta.get_TNT(raw_noise_params), pta.get_basis(), pta.get_ndiag(raw_noise_params)
    psr_map = {p.name: p for p in psrs}
    ref = 0.0
    for name, N, TNT, pi, T in zip(pta.pulsars, Nvecs, TNTs, phiinvs, Ts):
        p = psr_map[name]
        noise = parsed_noise_params.get(name, parsed_noise_params.get(get_base_name(name), {})) or {}
        s = population_residuals(np.asarray(p.toas), p, [pop[0]], Tspan, pulsar_noise_params=noise)
        Sig = TNT + (np.diag(pi) if pi.ndim == 1 else pi)
        TNs = N.solve(s, left_array=T)
        ref += float(N.solve(s, left_array=s) - TNs @ np.linalg.solve(Sig, TNs))
    print(f"independent rho^2 check : fast={r[0]**2:.10e} ref={ref:.10e} "
          f"rel diff={abs(r[0]**2 - ref) / max(abs(ref), 1e-300):.2e}")
    print("rho per source          :", r)
    return r


# cgw_selftest()      # <- uncomment to run immediately

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