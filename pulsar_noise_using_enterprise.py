from pta_builder import build_pta_and_params
import numpy as np

def get_noise_matrix(psrs, noise_params, Tspan):
    pta, model, params = build_pta_and_params(psrs=psrs, noise_params_15yr=noise_params, Tspan=Tspan, include_GW=False, include_RN=True)
    ln_likelihood = pta.get_lnlikelihood(params)
    print(ln_likelihood)
    # the noise covariance matrix doesn't change when the red noise is included, this is solely composed of the white noise it seems, I thought it would
    return