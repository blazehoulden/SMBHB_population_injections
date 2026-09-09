from enterprise.signals import signal_base, gp_signals, white_signals, selections, parameter, utils
from enterprise.signals.selections import Selection
from enterprise_extensions.blocks import red_noise_block, common_red_noise_block
from collections import defaultdict
import numpy as np

# def build_pta_and_params(psrs, noise_params, Tspan, crn_name="gw",
#                          gw_log10_A=np.log10(2.4e-15), gw_gamma=13.0/3.0, 
#                          include_GW=True, include_RN=True, include_WN=True,
#                          nmodes=150, curn_components=None, rn_components=None):
#     """
#     Build PTA model and ensure all required parameters exist.
    
#     Parameters
#     ----------
#     psrs : list
#         List of pulsar objects with .name attribute
#     noise_params : dict
#         Full noise parameter dictionary keyed as {pulsar}_{receiver}_{backend}_{param}
#     Tspan : float
#         Time baseline for common red noise / GWB block
#     crn_name : str
#         Name for the common red noise / GWB signal
#     gw_log10_A : float
#         Fixed GWB log10 amplitude
#     gw_gamma : float
#         Fixed GWB spectral index
#     """

#     # ---- Selection and white noise parameters ----
#     selection = selections.Selection(selections.by_backend)

#     efac    = parameter.Constant()
#     t2equad = parameter.Constant()
#     ecorr   = parameter.Constant()

#     # ---- Red noise parameters ----
#     log10_A = parameter.Constant()
#     gamma   = parameter.Constant()

#     # ---- Build signals once ----
#     tm   = gp_signals.TimingModel(use_svd=True)
#     mn   = white_signals.MeasurementNoise(efac=efac, log10_t2equad=t2equad, selection=selection)
#     ec   = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
#     if include_WN == False: # need to still include a tiny value for this to work, but it should be negligible enough to not affect results
#         mn = white_signals.MeasurementNoise(efac=parameter.Constant(val=1e-8), log10_t2equad=parameter.Constant(val=-12), selection=selection)

#     pl   = utils.powerlaw(log10_A=log10_A, gamma=gamma)
#     if rn_components is not None:
#         rn = gp_signals.FourierBasisGP(spectrum=pl, components=rn_components, Tspan=Tspan)
#     else:
#         rn = gp_signals.FourierBasisGP(spectrum=pl, components=nmodes, Tspan=Tspan)
#     cpl  = utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
#     if curn_components is not None:
#         curn = gp_signals.FourierBasisGP(spectrum=cpl, components=curn_components, Tspan=Tspan, name=crn_name)
#     else:
#         curn = gp_signals.FourierBasisGP(spectrum=cpl, components=nmodes, Tspan=Tspan, name=crn_name)

#     model = tm
#     if include_GW:
#         model += curn
#     if include_RN:
#         model += rn
#     if include_WN:
#         model += ec + mn
#     if include_WN == False:
#         model += mn
#     if include_GW == False:
#         gw_log10_A=-20.0 # should be negligible enough - similar to WN above
#         cpl  = utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
#         if curn_components is not None:
#             curn = gp_signals.FourierBasisGP(spectrum=cpl, components=curn_components, Tspan=Tspan, name=crn_name)
#         else:
#             curn = gp_signals.FourierBasisGP(spectrum=cpl, components=nmodes, Tspan=Tspan, name=crn_name)
#         model += curn

#     # ---- Instantiate per pulsar ----
#     pta = signal_base.PTA([model(psr) for psr in psrs])
#     pta.set_default_params(noise_params)

#     # ---- Fill any remaining expected params with defaults ----
#     params = dict(noise_params)
#     expected = {p.name for p in pta.params}

#     for pname in expected:
#         if pname not in params:
#             if "_efac" in pname:
#                 params[pname] = 1.0
#             elif "_log10_ecorr" in pname:
#                 params[pname] = -7.0
#             elif "_log10_t2equad" in pname:
#                 params[pname] = -7.0
#             elif "red_noise_gamma" in pname:
#                 params[pname] = 4.33
#             elif "red_noise_log10_A" in pname:
#                 params[pname] = -14.0
#             elif pname == f"{crn_name}_log10_A":
#                 params[pname] = gw_log10_A
#             elif pname == f"{crn_name}_gamma":
#                 params[pname] = gw_gamma
#             else:
#                 raise KeyError(f"Unhandled PTA parameter: {pname}")

#     missing = expected - params.keys()
#     if missing:
#         raise RuntimeError(f"Still missing params after defaults: {missing}")

#     return pta, model, params



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
    gw_log10_A=-14.0,
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


# import numpy as np
# from typing import Dict, List, Set, Tuple, Optional
# from enterprise.pulsar import Pulsar
# from enterprise.signals import parameter, signal_base
# from enterprise.signals import gp_signals, white_signals, utils
# from enterprise.signals import selections
 
 
# def build_pta_and_params(
#     psrs,
#     noise_params,
#     Tspan,
#     crn_name="gw",
#     gw_log10_A=-14.0,
#     gw_gamma=13.0/3.0,
#     include_GW=True,
#     include_RN=True,
#     include_WN=True,
#     nmodes=120,
#     curn_components=None,
#     rn_components=None,
#     verbose=True,
#     eqad_all=False,
#     ecorr_all=False,
#     debug=True,  # NEW: Enable debug mode
# ):
#     """
#     Build PTA with extensive debugging output.
    
#     Set debug=True to see detailed information at each step.
#     """
 
#     if verbose:
#         print("=" * 75)
#         print("Building PTA (per-pulsar noise model)")
#         print("=" * 75)
 
#     # =========================================================================
#     # PHASE 0: INPUT VALIDATION
#     # =========================================================================
#     if debug:
#         print("\n[DEBUG] PHASE 0: Input Validation")
#         print(f"  psrs type: {type(psrs)}, length: {len(psrs)}")
#         print(f"  noise_params type: {type(noise_params)}, length: {len(noise_params)}")
#         print(f"  Tspan: {Tspan}")
#         print(f"  nmodes: {nmodes}, rn_components: {rn_components}, curn_components: {curn_components}")
#         print(f"  include_GW: {include_GW}, include_RN: {include_RN}, include_WN: {include_WN}")
        
#         # Sample parameters
#         print(f"\n[DEBUG] Sample parameter keys (first 10):")
#         for i, key in enumerate(list(noise_params.keys())[:10]):
#             print(f"    {i}: {key} = {noise_params[key]}")
        
#         # Check for NaN/Inf
#         nan_count = sum(1 for v in noise_params.values() if isinstance(v, float) and np.isnan(v))
#         inf_count = sum(1 for v in noise_params.values() if isinstance(v, float) and np.isinf(v))
#         if nan_count > 0 or inf_count > 0:
#             print(f"\n[WARNING] Found {nan_count} NaN and {inf_count} Inf values!")
 
#     # =========================================================================
#     # PHASE 1: SELECTION AND MODEL LIST INIT
#     # =========================================================================
#     if debug:
#         print("\n[DEBUG] PHASE 1: Selection Setup")
    
#     try:
#         selection = selections.Selection(selections.by_backend)
#         if debug:
#             print(f"  ✓ Selection created: {type(selection)}")
#     except Exception as e:
#         print(f"  ✗ FAILED to create selection: {e}")
#         raise
 
#     model_list = []
    
#     # =========================================================================
#     # PHASE 2: PER-PULSAR MODEL BUILDING
#     # =========================================================================
#     if debug:
#         print(f"\n[DEBUG] PHASE 2: Building models for {len(psrs)} pulsars")
    
#     for psr_idx, psr in enumerate(psrs):
#         pname = psr.name
        
#         if debug:
#             print(f"\n[DEBUG] Pulsar {psr_idx}: {pname}")
#             print(f"  nobs: {psr.nobs}, type: {type(psr)}")
        
#         # Check if this pulsar has parameters
#         psr_param_keys = [k for k in noise_params.keys() if k.startswith(pname + '_')]
#         if debug:
#             print(f"  Parameters for this pulsar: {len(psr_param_keys)}")
#             if len(psr_param_keys) == 0:
#                 print(f"    ⚠️  WARNING: No parameters found for {pname}!")
#             elif len(psr_param_keys) < 5:
#                 for k in psr_param_keys:
#                     print(f"    - {k}: {noise_params[k]}")
        
#         # Initialize model
#         try:
#             model = gp_signals.TimingModel(use_svd=True)
#             if debug:
#                 print(f"  ✓ TimingModel created")
#         except Exception as e:
#             print(f"  ✗ FAILED to create TimingModel: {e}")
#             raise
 
#         # =====================================================================
#         # RED NOISE
#         # =====================================================================
#         red_added = False
#         if include_RN:
#             red_A_key = f"{pname}_red_noise_log10_A"
#             red_g_key = f"{pname}_red_noise_gamma"
            
#             if debug:
#                 print(f"  Checking RED_NOISE: {red_A_key} in params? {red_A_key in noise_params}")
#                 print(f"                      {red_g_key} in params? {red_g_key in noise_params}")
            
#             if red_A_key in noise_params and red_g_key in noise_params:
#                 try:
#                     log10_A = noise_params[red_A_key]
#                     gamma = noise_params[red_g_key]
                    
#                     if debug:
#                         print(f"    Values: log10_A={log10_A}, gamma={gamma}")
                    
#                     pl = utils.powerlaw(
#                         log10_A=parameter.Constant(),
#                         gamma=parameter.Constant(),
#                     )
                    
#                     if debug:
#                         print(f"    ✓ Powerlaw spectrum created")
                    
#                     model += gp_signals.FourierBasisGP(
#                         spectrum=pl,
#                         components=rn_components or nmodes,
#                         Tspan=Tspan,
#                         name="red_noise",
#                     )
                    
#                     red_added = True
#                     if debug:
#                         print(f"    ✓ RED_NOISE added (components={rn_components or nmodes})")
                
#                 except Exception as e:
#                     print(f"    ✗ FAILED to add RED_NOISE: {e}")
#                     raise
#             elif debug:
#                 print(f"    → RED_NOISE skipped (missing parameters)")
 
#         # =====================================================================
#         # DM NOISE
#         # =====================================================================
#         dm_added = False
#         if include_RN:
#             dm_A_key = f"{pname}_dm_noise_log10_A"
#             dm_g_key = f"{pname}_dm_noise_gamma"
            
#             if debug:
#                 print(f"  Checking DM_NOISE: {dm_A_key} in params? {dm_A_key in noise_params}")
#                 print(f"                     {dm_g_key} in params? {dm_g_key in noise_params}")
            
#             if dm_A_key in noise_params and dm_g_key in noise_params:
#                 try:
#                     log10_A = noise_params[dm_A_key]
#                     gamma = noise_params[dm_g_key]
                    
#                     if debug:
#                         print(f"    Values: log10_A={log10_A}, gamma={gamma}")
                    
#                     pl = utils.powerlaw(
#                         log10_A=parameter.Constant(),
#                         gamma=parameter.Constant(),
#                     )
                    
#                     model += gp_signals.FourierBasisGP(
#                         spectrum=pl,
#                         components=rn_components or nmodes,
#                         Tspan=Tspan,
#                         name="dm_noise",
#                     )
                    
#                     dm_added = True
#                     if debug:
#                         print(f"    ✓ DM_NOISE added (components={rn_components or nmodes})")
                
#                 except Exception as e:
#                     print(f"    ✗ FAILED to add DM_NOISE: {e}")
#                     raise
#             elif debug:
#                 print(f"    → DM_NOISE skipped (missing parameters)")
 
#         # =====================================================================
#         # CHROMATIC NOISE
#         # =====================================================================
#         chrom_added = False
#         if include_RN:
#             chrom_A_key = f"{pname}_chrom_noise_log10_A"
#             chrom_g_key = f"{pname}_chrom_noise_gamma"
            
#             if debug:
#                 print(f"  Checking CHROM_NOISE: {chrom_A_key} in params? {chrom_A_key in noise_params}")
#                 print(f"                        {chrom_g_key} in params? {chrom_g_key in noise_params}")
            
#             if chrom_A_key in noise_params and chrom_g_key in noise_params:
#                 try:
#                     log10_A = noise_params[chrom_A_key]
#                     gamma = noise_params[chrom_g_key]
                    
#                     if debug:
#                         print(f"    Values: log10_A={log10_A}, gamma={gamma}")
                    
#                     pl = utils.powerlaw(
#                         log10_A=parameter.Constant(),
#                         gamma=parameter.Constant(),
#                     )
                    
#                     model += gp_signals.FourierBasisGP(
#                         spectrum=pl,
#                         components=rn_components or nmodes,
#                         Tspan=Tspan,
#                         name="chrom_noise",
#                     )
                    
#                     chrom_added = True
#                     if debug:
#                         print(f"    ✓ CHROM_NOISE added (components={rn_components or nmodes})")
                
#                 except Exception as e:
#                     print(f"    ✗ FAILED to add CHROM_NOISE: {e}")
#                     raise
#             elif debug:
#                 print(f"    → CHROM_NOISE skipped (missing parameters)")
 
#         # =====================================================================
#         # SCATTERING NOISE
#         # =====================================================================
#         scatter_added = False
#         if include_RN:
#             scatter_A_key = f"{pname}_scattering_noise_log10_A"
#             scatter_g_key = f"{pname}_scattering_noise_gamma"
            
#             if debug:
#                 print(f"  Checking SCATTERING_NOISE: {scatter_A_key} in params? {scatter_A_key in noise_params}")
#                 print(f"                             {scatter_g_key} in params? {scatter_g_key in noise_params}")
            
#             if scatter_A_key in noise_params and scatter_g_key in noise_params:
#                 try:
#                     log10_A = noise_params[scatter_A_key]
#                     gamma = noise_params[scatter_g_key]
                    
#                     if debug:
#                         print(f"    Values: log10_A={log10_A}, gamma={gamma}")
                    
#                     pl = utils.powerlaw(
#                         log10_A=parameter.Constant(),
#                         gamma=parameter.Constant(),
#                     )
                    
#                     model += gp_signals.FourierBasisGP(
#                         spectrum=pl,
#                         components=rn_components or nmodes,
#                         Tspan=Tspan,
#                         name="scattering_noise",
#                     )
                    
#                     scatter_added = True
#                     if debug:
#                         print(f"    ✓ SCATTERING_NOISE added (components={rn_components or nmodes})")
                
#                 except Exception as e:
#                     print(f"    ✗ FAILED to add SCATTERING_NOISE: {e}")
#                     raise
#             elif debug:
#                 print(f"    → SCATTERING_NOISE skipped (missing parameters)")
 
#         # =====================================================================
#         # BAND NOISE
#         # =====================================================================
#         band_added = False
#         if include_RN:
#             band_A_key = f"{pname}_band_noise_log10_A"
#             band_g_key = f"{pname}_band_noise_gamma"
            
#             if debug:
#                 print(f"  Checking BAND_NOISE: {band_A_key} in params? {band_A_key in noise_params}")
#                 print(f"                       {band_g_key} in params? {band_g_key in noise_params}")
            
#             if band_A_key in noise_params and band_g_key in noise_params:
#                 try:
#                     log10_A = noise_params[band_A_key]
#                     gamma = noise_params[band_g_key]
                    
#                     if debug:
#                         print(f"    Values: log10_A={log10_A}, gamma={gamma}")
                    
#                     pl = utils.powerlaw(
#                         log10_A=parameter.Constant(),
#                         gamma=parameter.Constant(),
#                     )
                    
#                     model += gp_signals.FourierBasisGP(
#                         spectrum=pl,
#                         components=rn_components or nmodes,
#                         Tspan=Tspan,
#                         name="band_noise",
#                     )
                    
#                     band_added = True
#                     if debug:
#                         print(f"    ✓ BAND_NOISE added (components={rn_components or nmodes})")
                
#                 except Exception as e:
#                     print(f"    ✗ FAILED to add BAND_NOISE: {e}")
#                     raise
#             elif debug:
#                 print(f"    → BAND_NOISE skipped (missing parameters)")
 
#         # =====================================================================
#         # WHITE NOISE
#         # =====================================================================
#         wn_added = False
#         if include_WN:
#             if debug:
#                 print(f"  Checking WHITE NOISE:")
            
#             t2equad_key = f"{pname}_KAT_MKBF_log10_t2equad"
#             ecorr_key = f"{pname}_KAT_MKBF_log10_ecorr"
            
#             has_t2equad = t2equad_key in noise_params or eqad_all
#             has_ecorr = ecorr_key in noise_params or ecorr_all
            
#             if debug:
#                 print(f"    Has t2equad: {has_t2equad} ({t2equad_key} in params? {t2equad_key in noise_params}, eqad_all? {eqad_all})")
#                 print(f"    Has ecorr: {has_ecorr} ({ecorr_key} in params? {ecorr_key in noise_params}, ecorr_all? {ecorr_all})")
            
#             try:
#                 if has_t2equad:
#                     if debug:
#                         print(f"    → Adding MeasurementNoise with t2equad")
#                     model += white_signals.MeasurementNoise(
#                         efac=parameter.Constant(),
#                         log10_t2equad=parameter.Constant(),
#                         selection=selection,
#                     )
#                 else:
#                     if debug:
#                         print(f"    → Adding MeasurementNoise without t2equad")
#                     model += white_signals.MeasurementNoise(
#                         efac=parameter.Constant(),
#                         selection=selection,
#                     )
                
#                 wn_added = True
#                 if debug:
#                     print(f"    ✓ MeasurementNoise added")
                
#                 if has_ecorr:
#                     if debug:
#                         print(f"    → Adding EcorrKernelNoise")
#                     model += white_signals.EcorrKernelNoise(
#                         log10_ecorr=parameter.Constant(),
#                         selection=selection
#                     )
#                     if debug:
#                         print(f"    ✓ EcorrKernelNoise added")
            
#             except Exception as e:
#                 print(f"    ✗ FAILED to add white noise: {e}")
#                 raise
 
#         # =====================================================================
#         # COMMON RED NOISE / GW
#         # =====================================================================
#         gw_added = False
#         if include_GW:
#             if debug:
#                 print(f"  Checking GW/CRN:")
            
#             try:
#                 gw_pl = utils.powerlaw(
#                     log10_A=parameter.Constant(val=gw_log10_A),
#                     gamma=parameter.Constant(val=gw_gamma),
#                 )
                
#                 if debug:
#                     print(f"    Powerlaw values: log10_A={gw_log10_A}, gamma={gw_gamma}")
                
#                 model += gp_signals.FourierBasisGP(
#                     spectrum=gw_pl,
#                     components=curn_components or nmodes,
#                     Tspan=Tspan,
#                     name=crn_name,
#                 )
                
#                 gw_added = True
#                 if debug:
#                     print(f"    ✓ GW/CRN added ({crn_name}, components={curn_components or nmodes})")
            
#             except Exception as e:
#                 print(f"    ✗ FAILED to add GW: {e}")
#                 raise
 
#         # =====================================================================
#         # SUMMARY FOR THIS PULSAR
#         # =====================================================================
#         if debug:
#             signals_added = sum([red_added, dm_added, chrom_added, scatter_added, band_added, wn_added, gw_added])
#             print(f"  → Summary: {signals_added} signals added")
#             print(f"    - TimingModel: ✓")
#             print(f"    - red_noise: {'✓' if red_added else '✗'}")
#             print(f"    - dm_noise: {'✓' if dm_added else '✗'}")
#             print(f"    - chrom_noise: {'✓' if chrom_added else '✗'}")
#             print(f"    - scattering_noise: {'✓' if scatter_added else '✗'}")
#             print(f"    - band_noise: {'✓' if band_added else '✗'}")
#             print(f"    - white_noise: {'✓' if wn_added else '✗'}")
#             print(f"    - gw: {'✓' if gw_added else '✗'}")
 
#         # Add to model list
#         try:
#             model_psr = model(psr)
#             model_list.append(model_psr)
#             if debug:
#                 print(f"  ✓ Model applied to pulsar")
#         except Exception as e:
#             print(f"  ✗ FAILED to apply model to pulsar: {e}")
#             raise
 
#     # =========================================================================
#     # PHASE 3: PTA CONSTRUCTION
#     # =========================================================================
#     if debug:
#         print(f"\n[DEBUG] PHASE 3: PTA Construction")
#         print(f"  Creating PTA with {len(model_list)} models...")
    
#     try:
#         pta = signal_base.PTA(model_list)
#         if debug:
#             print(f"  ✓ PTA created successfully")
#             print(f"  Number of signal collections: {len(pta._signalcollections)}")
#     except Exception as e:
#         print(f"  ✗ FAILED to create PTA: {e}")
#         import traceback
#         traceback.print_exc()
#         raise
 
#     # =========================================================================
#     # PHASE 4: PARAMETER SETTING
#     # =========================================================================
#     if debug:
#         print(f"\n[DEBUG] PHASE 4: Setting Default Parameters")
#         print(f"  Parameters to set: {len(noise_params)}")
    
#     try:
#         pta.set_default_params(noise_params)
#         if debug:
#             print(f"  ✓ Default parameters set successfully")
#     except Exception as e:
#         print(f"  ✗ FAILED to set default parameters: {e}")
#         import traceback
#         traceback.print_exc()
#         raise
 
#     # =========================================================================
#     # PHASE 5: BUILD PARAM DICT
#     # =========================================================================
#     if debug:
#         print(f"\n[DEBUG] PHASE 5: Building Parameter Dictionary")
    
#     params = dict(noise_params)
    
#     if debug:
#         print(f"  ✓ Parameter dictionary created ({len(params)} entries)")
 
#     # =========================================================================
#     # PHASE 6: VERBOSE OUTPUT
#     # =========================================================================
#     if verbose:
#         print(f"\n[VERBOSE] Built PTA with {len(psrs)} pulsars")
        
#         print(f"\n[VERBOSE] Signal Summary:")
#         for sc in pta._signalcollections:
#             psrname = sc.psrname
#             sigs = [sig.signal_name for sig in sc._signals]
#             print(f"  {psrname}: {sigs}")
 
#     # =========================================================================
#     # PHASE 7: DETAILED DEBUG OUTPUT
#     # =========================================================================
#     if debug:
#         print(f"\n[DEBUG] PHASE 7: Detailed Signal Inspection")
#         for sc_idx, sc in enumerate(pta._signalcollections):
#             psrname = sc.psrname
#             print(f"\n  Signal Collection {sc_idx}: {psrname}")
#             print(f"    Number of signals: {len(sc._signals)}")
            
#             for sig_idx, sig in enumerate(sc._signals):
#                 print(f"    Signal {sig_idx}: {sig.signal_name}")
                
#                 if hasattr(sig, "_params"):
#                     param_keys = list(sig._params.keys())
#                     print(f"      Parameters ({len(param_keys)}): {param_keys[:5]}...")
                    
#                     # Check if these params are in noise_params
#                     for pk in param_keys[:3]:
#                         if pk in noise_params:
#                             print(f"        {pk}: ✓ in noise_params = {noise_params[pk]}")
#                         else:
#                             print(f"        {pk}: ✗ NOT in noise_params")
#                 else:
#                     print(f"      No _params attribute")
 
#     return pta, model_list, params