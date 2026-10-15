from enterprise.signals.selections import Selection
from enterprise_extensions.blocks import red_noise_block, common_red_noise_block
from collections import defaultdict
import numpy as np

from enterprise.signals import (
    gp_signals,
    parameter,
    selections,
    signal_base,
    utils,
    white_signals,
)
 
 
def _has(noise_params, *keys):
    return all(k in noise_params for k in keys)
 
 
def _detect_backend_systems(noise_params, pname, suffix):
    """
    Find every backend/system tag `sys` such that
    f"{pname}_{sys}_{suffix}" exists in noise_params.
 
    e.g. suffix="efac" -> finds every backend with an EFAC defined.
    """
    prefix = f"{pname}_"
    found = []
    for key in noise_params:
        if key.startswith(prefix) and key.endswith(f"_{suffix}"):
            sys = key[len(prefix):-len(f"_{suffix}")]
            if sys:
                found.append(sys)
    return sorted(set(found))
 
 
from enterprise.signals import (
    gp_signals,
    parameter,
    selections,
    signal_base,
    utils,
    white_signals,
)
 
 
def _has(noise_params, *keys):
    return all(k in noise_params for k in keys)
 
 
def _detect_backend_systems(noise_params, pname, suffix):
    """
    Find every backend/system tag `sys` such that
    f"{pname}_{sys}_{suffix}" exists in noise_params.
 
    e.g. suffix="efac" -> finds every backend with an EFAC defined.
    """
    prefix = f"{pname}_"
    found = []
    for key in noise_params:
        if key.startswith(prefix) and key.endswith(f"_{suffix}"):
            sys = key[len(prefix):-len(f"_{suffix}")]
            if sys:
                found.append(sys)
    return sorted(set(found))
 
 
def build_pta_and_params(
    psrs,
    noise_params,
    Tspan,
    crn_name="gw",
    gw_log10_A=np.log10(4.8e-15), # MPTA value at 1/yr
    gw_gamma=13.0 / 3.0,
    include_GW=True,
    include_RN=True,
    include_WN=True,
    include_SW=False,
    nmodes=120,
    curn_components=None,
    rn_components=None,
    verbose=True,
):
    selection = selections.Selection(selections.by_backend)
    ecorr_selection = selections.Selection(selections.by_backend)
 
    model_list = []
    params = dict(noise_params)  # will be filled with any missing defaults
 
    for psr in psrs:
        pname = psr.name
        model = gp_signals.TimingModel(use_svd=True)
        ncomp = rn_components or nmodes
 
        # ---------------- Achromatic red noise ----------------
        if include_RN and _has(noise_params, f"{pname}_red_log10_A", f"{pname}_red_gamma"):
            pl = utils.powerlaw(
                log10_A=parameter.Constant(noise_params[f"{pname}_red_log10_A"]),
                gamma=parameter.Constant(noise_params[f"{pname}_red_gamma"]),
            )
            model += gp_signals.FourierBasisGP(
                spectrum=pl, components=ncomp, Tspan=Tspan, name="red_noise"
            )
 
        # ---------------- DM noise ----------------
        if include_RN and _has(noise_params, f"{pname}_dm_log10_A", f"{pname}_dm_gamma"):
            # IMPORTANT: call with kwargs only, NO positional toas/freqs and
            # NO functools.partial. createfourierdesignmatrix_dm is already
            # decorated with @function; calling it this way (missing the
            # required positional args) makes the decorator return a Function
            # *class* - which is exactly what BasisGP expects as basisFunction.
            # functools.partial bypasses this branching and breaks BasisGP.
            dm_basis = utils.createfourierdesignmatrix_dm(
                nmodes=ncomp,
                Tspan=Tspan,
            )
            dm_pl = utils.powerlaw(
                log10_A=parameter.Constant(noise_params[f"{pname}_dm_log10_A"]),
                gamma=parameter.Constant(noise_params[f"{pname}_dm_gamma"]),
            )
            model += gp_signals.BasisGP(dm_pl, dm_basis, name="dm_gp")
 
        # ---------------- Chromatic noise ----------------
        if include_RN and _has(
            noise_params,
            f"{pname}_chrom_log10_A",
            f"{pname}_chrom_gamma",
            f"{pname}_chrom_beta",
        ):
            beta = noise_params[f"{pname}_chrom_beta"]
            # GUARD: createfourierdesignmatrix_chromatic(idx=2) is
            # mathematically IDENTICAL to createfourierdesignmatrix_dm
            # (both reduce to (1400/freq)^2 * F_red on the same toas/freqs).
            # If this pulsar also has a dm_gp signal and chrom_beta happens
            # to equal 2, the combined design matrix becomes ill-conditioned/
            # near-singular in exactly the same way the old DM+SW collision
            # was - Sigma = TNT + diag(phiinv) can fail Cholesky factorisation
            # downstream (e.g. in CGW SNR code) even though it remains
            # non-singular enough for np.linalg.solve to silently succeed
            # elsewhere. Guard against it explicitly rather than relying on
            # beta never landing on exactly 2.0 in practice.
            has_dm_here = _has(noise_params, f"{pname}_dm_log10_A", f"{pname}_dm_gamma")
            if has_dm_here and float(beta) == 2.0:
                raise ValueError(
                    f"{pname}: chrom_beta == 2.0 makes the chromatic basis "
                    f"identical to the DM basis (both pulsars use "
                    f"createfourierdesignmatrix_*(idx=2) on the same toas/"
                    f"freqs). Building both as separate GPs would create a "
                    f"near-singular design matrix and silently corrupt "
                    f"downstream Cholesky-based SNR calculations. Either "
                    f"exclude this pulsar's chromatic noise, or merge it "
                    f"with the DM term upstream before calling this function."
                )
            chrom_basis = utils.createfourierdesignmatrix_chromatic(
                nmodes=ncomp,
                Tspan=Tspan,
                idx=beta,
            )
            chrom_pl = utils.powerlaw(
                log10_A=parameter.Constant(noise_params[f"{pname}_chrom_log10_A"]),
                gamma=parameter.Constant(noise_params[f"{pname}_chrom_gamma"]),
            )
            model += gp_signals.BasisGP(chrom_pl, chrom_basis, name="chrom_gp")
 
        # ---------------- Solar-wind-like noise (DM-style, idx fixed at 2 via dm basis) ----------------
        if include_SW and _has(noise_params, f"{pname}_sw_log10_A", f"{pname}_sw_gamma"):
            # GUARD: this SW model uses createfourierdesignmatrix_dm, the
            # exact same basis as DM noise (both idx=2 on the same toas/
            # freqs). If this pulsar also has dm_gp, the two design matrices
            # are byte-identical, making the combined basis ill-conditioned/
            # near-singular - this was the original cause of "N-th leading
            # minor is not positive definite" Cholesky failures in CGW SNR
            # code downstream, even though it stayed non-singular enough for
            # np.linalg.solve to silently succeed elsewhere (e.g. the OS
            # step). include_SW defaults to False specifically to avoid
            # this; if you re-enable it, this guard stops the collision
            # from silently reappearing rather than failing far downstream.
            has_dm_here = _has(noise_params, f"{pname}_dm_log10_A", f"{pname}_dm_gamma")
            if has_dm_here:
                raise ValueError(
                    f"{pname}: has both DM and SW noise parameters, and this "
                    f"SW model uses the same basis as DM noise (idx=2, same "
                    f"toas/freqs) - building both as separate GPs creates a "
                    f"near-singular design matrix that breaks downstream "
                    f"Cholesky-based SNR calculations. Either drop SW noise "
                    f"for this pulsar, drop DM noise, or merge them into one "
                    f"combined-PSD GP before calling this function."
                )
            sw_basis = utils.createfourierdesignmatrix_dm(
                nmodes=ncomp,
                Tspan=Tspan,
            )
            sw_pl = utils.powerlaw(
                log10_A=parameter.Constant(noise_params[f"{pname}_sw_log10_A"]),
                gamma=parameter.Constant(noise_params[f"{pname}_sw_gamma"]),
            )
            model += gp_signals.BasisGP(sw_pl, sw_basis, name="sw_gp")
 
        # ---------------- White noise ----------------
        if include_WN:
            efac_systems = _detect_backend_systems(noise_params, pname, "efac")
            t2equad_systems = _detect_backend_systems(noise_params, pname, "log10_t2equad")
            equad_systems = _detect_backend_systems(noise_params, pname, "log10_equad")
            ecorr_systems = _detect_backend_systems(noise_params, pname, "log10_ecorr")
 
            has_t2equad = len(t2equad_systems) > 0
            has_equad = len(equad_systems) > 0
 
            # One MeasurementNoise signal covers ALL backends at once via
            # selection=by_backend; per-backend values come from
            # pta.set_default_params(noise_params), not from baked Constants.
            if has_t2equad:
                # tempo/tempo2/pint convention: variance = efac^2*(toaerr^2 + t2equad^2)
                model += white_signals.MeasurementNoise(
                    efac=parameter.Constant(),
                    log10_t2equad=parameter.Constant(),
                    selection=selection,
                )
                for sys in t2equad_systems:
                    params.setdefault(f"{pname}_{sys}_log10_t2equad", -10.0)
                for sys in efac_systems:
                    params.setdefault(f"{pname}_{sys}_efac", 1.0)
            elif has_equad:
                # legacy TNEQUAD convention: separate EFAC and TNEquad signals,
                # NOT multiplied together (use TNEquadNoise, not MeasurementNoise).
                # NOTE: TNEquadNoise's parameter is named log10_tnequad internally
                # (not log10_equad) - the dict key must match that exactly.
                model += white_signals.MeasurementNoise(
                    efac=parameter.Constant(),
                    selection=selection,
                )
                model += white_signals.TNEquadNoise(
                    log10_tnequad=parameter.Constant(),
                    selection=selection,
                )
                for sys in equad_systems:
                    # noise_params uses "_log10_equad"; map it to the
                    # "_log10_tnequad" key that enterprise actually expects.
                    src_key = f"{pname}_{sys}_log10_equad"
                    dst_key = f"{pname}_{sys}_log10_tnequad"
                    if src_key in noise_params:
                        params[dst_key] = noise_params[src_key]
                    params.setdefault(dst_key, -10.0)
                for sys in efac_systems:
                    params.setdefault(f"{pname}_{sys}_efac", 1.0)
            else:
                model += white_signals.MeasurementNoise(
                    efac=parameter.Constant(),
                    selection=selection,
                )
                for sys in efac_systems:
                    params.setdefault(f"{pname}_{sys}_efac", 1.0)
 
            if not efac_systems:
                # ensure at least one efac default exists so set_default_params
                # has something to fill if enterprise expects one
                params.setdefault(f"{pname}_KAT_MKBF_efac", 1.0)
 
            if ecorr_systems:
                model += white_signals.EcorrKernelNoise(
                    log10_ecorr=parameter.Constant(),
                    selection=ecorr_selection,
                )
                for sys in ecorr_systems:
                    params.setdefault(f"{pname}_{sys}_log10_ecorr", -10.0)
        else:
            # include_WN=False: still add a negligible-amplitude white noise
            # floor (EFAC=1, tiny EQUAD) so every pulsar's SignalCollection
            # always has at least one non-GP, non-empty signal. Mirrors the
            # original 15yr builder's "still need a tiny value" behavior, and
            # avoids enterprise choking on a pulsar model with zero signals
            # when include_GW/RN/WN are all False.
            model += white_signals.MeasurementNoise(
                efac=parameter.Constant(val=1.0),
                log10_t2equad=parameter.Constant(val=-12.0),
                selection=selection,
            )
 
 
        # ---------------- Common red noise / GWB ----------------
        if include_GW:
            gw_pl = utils.powerlaw(
                log10_A=parameter.Constant(gw_log10_A),
                gamma=parameter.Constant(gw_gamma),
            )
            model += gp_signals.FourierBasisGP(
                spectrum=gw_pl,
                components=curn_components or nmodes,
                Tspan=Tspan,
                name=crn_name,
            )
        else:
            # Mirrors the original builder: even when GW is "off", still add
            # a negligible-amplitude term under the same crn_name so that
            # any downstream code expecting a `{crn_name}_log10_A` /
            # `{crn_name}_gamma` parameter (e.g. for consistent SNR
            # comparisons across configs) doesn't break.
            negligible_pl = utils.powerlaw(
                log10_A=parameter.Constant(-20.0),
                gamma=parameter.Constant(gw_gamma),
            )
            model += gp_signals.FourierBasisGP(
                spectrum=negligible_pl,
                components=curn_components or nmodes,
                Tspan=Tspan,
                name=crn_name,
            )
 
        model_list.append(model(psr))
 
    pta = signal_base.PTA(model_list)
 
    # Fill in any remaining expected parameters with sane defaults, then push
    # everything (including the literal Constant params built above) through
    # set_default_params so that per-backend Constant() factories actually
    # pick up per-backend values.
    expected = {p.name for p in pta.params}
    for pname_ in expected:
        if pname_ not in params:
            if pname_.endswith("_efac"):
                params[pname_] = 1.0
            elif pname_.endswith("_log10_ecorr"):
                params[pname_] = -10.0
            elif pname_.endswith("_log10_t2equad") or pname_.endswith("_log10_equad"):
                params[pname_] = -10.0
            elif pname_.endswith("_red_noise_gamma") or pname_.endswith("_gamma"):
                params[pname_] = 13.0 / 3.0
            elif pname_.endswith("_red_noise_log10_A") or pname_.endswith("_log10_A"):
                params[pname_] = -14.0
            else:
                raise KeyError(f"Unhandled PTA parameter with no default rule: {pname_}")
 
    missing = expected - params.keys()
    if missing:
        raise RuntimeError(f"Still missing params after defaults: {missing}")
 
    pta.set_default_params(params)
 
    # if verbose:
    #     print(f"\nBuilt PTA with {len(psrs)} pulsars")
    #     for sc in pta._signalcollections:
    #         sigs = [s.signal_name for s in sc._signals]
    #         print(f"  {sc.psrname}: {sigs}")
 
    return pta, model_list, params
