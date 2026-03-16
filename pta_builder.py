from enterprise.signals import signal_base, gp_signals, white_signals, selections, parameter, utils
from enterprise.signals.selections import Selection
from enterprise_extensions.blocks import red_noise_block, common_red_noise_block
from collections import defaultdict
import numpy as np

def build_pta_and_params(psrs, noise_params_15yr, Tspan, crn_name="gw",
                         gw_log10_A=np.log10(2.4e-15), gw_gamma=13.0/3.0, 
                         include_GW=True, include_RN=True, include_WN=True,
                         nmodes=30):
    """
    Build PTA model and ensure all required parameters exist.
    
    Parameters
    ----------
    psrs : list
        List of pulsar objects with .name attribute
    noise_params_15yr : dict
        Full noise parameter dictionary keyed as {pulsar}_{receiver}_{backend}_{param}
    Tspan : float
        Time baseline for common red noise / GWB block
    crn_name : str
        Name for the common red noise / GWB signal
    gw_log10_A : float
        Fixed GWB log10 amplitude
    gw_gamma : float
        Fixed GWB spectral index
    """

    # ---- Selection and white noise parameters ----
    selection = selections.Selection(selections.by_backend)

    efac    = parameter.Constant()
    t2equad = parameter.Constant()
    ecorr   = parameter.Constant()

    # ---- Red noise parameters ----
    log10_A = parameter.Constant()
    gamma   = parameter.Constant()

    # ---- Build signals once ----
    tm   = gp_signals.TimingModel(use_svd=True)
    mn   = white_signals.MeasurementNoise(efac=efac, log10_t2equad=t2equad, selection=selection)
    ec   = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
    if include_WN == False:
        mn = white_signals.MeasurementNoise(efac=parameter.Constant(val=1e-8), log10_t2equad=parameter.Constant(val=-12), selection=selection)

    pl   = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn   = gp_signals.FourierBasisGP(spectrum=pl, components=nmodes, Tspan=Tspan)
    cpl  = utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
    curn = gp_signals.FourierBasisGP(spectrum=cpl, components=nmodes, Tspan=Tspan, name=crn_name)

    model = tm
    if include_GW:
        model += curn
    if include_RN:
        model += rn
    if include_WN:
        model += ec + mn
    if include_WN == False:
        model += mn
    if include_GW == False:
        gw_log10_A=-18.0 # should be negligible enough
        cpl  = utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
        curn = gp_signals.FourierBasisGP(spectrum=cpl, components=nmodes, Tspan=Tspan, name=crn_name)
        model += curn

    # ---- Instantiate per pulsar ----
    pta = signal_base.PTA([model(psr) for psr in psrs])
    pta.set_default_params(noise_params_15yr)

    # ---- Fill any remaining expected params with defaults ----
    params = dict(noise_params_15yr)
    expected = {p.name for p in pta.params}

    for pname in expected:
        if pname not in params:
            if "_efac" in pname:
                params[pname] = 1.0
            elif "_log10_ecorr" in pname:
                params[pname] = -7.0
            elif "_log10_t2equad" in pname:
                params[pname] = -7.0
            elif "red_noise_gamma" in pname:
                params[pname] = 4.33
            elif "red_noise_log10_A" in pname:
                params[pname] = -14.0
            elif pname == f"{crn_name}_log10_A":
                params[pname] = gw_log10_A
            elif pname == f"{crn_name}_gamma":
                params[pname] = gw_gamma
            else:
                raise KeyError(f"Unhandled PTA parameter: {pname}")

    missing = expected - params.keys()
    if missing:
        raise RuntimeError(f"Still missing params after defaults: {missing}")

    return pta, model, params
