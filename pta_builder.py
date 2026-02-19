from enterprise.signals import signal_base, gp_signals, white_signals, selections, parameter
from enterprise.signals.selections import Selection
from enterprise_extensions.blocks import red_noise_block, common_red_noise_block


def build_pta_and_params(psrs, noise_params_15yr, Tspan, use_efac_only=True, crn_name="gw"):
    """Build PTA model and ensure all required parameters exist."""
    
    # Timing model
    tm = gp_signals.TimingModel(use_svd=True)

    # White noise
    if use_efac_only:
        efac = parameter.Constant(val=0)
        selection = Selection(selections.by_backend)
        wn = white_signals.MeasurementNoise(efac=efac, selection=selection, name=None)
    else:
        raise NotImplementedError("Only EFAC-only supported currently")

    # Red noise per pulsar
    rn = red_noise_block(
        prior="log-uniform",
        psd="powerlaw",
        components=30,
        gamma_val=None,
        coefficients=False
    )

    # Common red noise (GWB)
    crn = common_red_noise_block(
        psd="powerlaw",
        prior="log-uniform",
        Tspan=Tspan,
        components=5,
        gamma_val=13/3,
        name=crn_name,
        coefficients=False
    )

    model = tm + wn + rn + crn
    # model = tm + rn + crn
    pta = signal_base.PTA([model(psr) for psr in psrs])
    pta.set_default_params(noise_params_15yr) # check if this is necessary - saw it https://colab.research.google.com/drive/1VNLbutN7cKJM2jl6LId0IgkGJDszDloC#scrollTo=XlmoCSjvQhnI

    # Parameter completion
    params = dict(noise_params_15yr)
    expected = {p.name for p in pta.params}

    for pname in expected:
        if pname not in params:
            if "_efac" in pname:
                params[pname] = 1.0
            elif "red_noise_gamma" in pname:
                params[pname] = 4.33
            elif "red_noise_log10_A" in pname:
                params[pname] = -14.0
            elif pname == f"{crn_name}_log10_A":
                params[pname] = -14.5
            elif pname == f"{crn_name}_gamma":
                params[pname] = 13 / 3
            else:
                raise KeyError(f"Unhandled PTA parameter: {pname}")

    missing = expected - params.keys()
    if missing:
        raise RuntimeError(f"Still missing params: {missing}")

    return pta, model, params