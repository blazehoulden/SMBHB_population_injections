from pta_builder import build_pta_and_params
import numpy as np
from enterprise.signals.gp_bases import createfourierdesignmatrix_red
from enterprise.signals.utils import create_quantization_matrix

def pulsar_PSD_using_enterprise(psrs, noise_params, Tspan, nmodes=30):
    pta, model, params = build_pta_and_params(psrs=psrs, noise_params_15yr=noise_params, Tspan=Tspan, include_GW=True)
    pulsar_PSD_total = np.zeros((len(psrs), nmodes))
    pulsar_PSD_red = np.zeros((len(psrs), nmodes))
    pulsar_PSD_white = np.zeros((len(psrs), nmodes))
    for i, pulsar in enumerate(psrs):
        psr_tspan = pulsar.toas.max() - pulsar.toas.min()
        _, freq_full_list = createfourierdesignmatrix_red(pulsar.toas, nmodes=nmodes, Tspan=psr_tspan)
        freqs = freq_full_list[:nmodes]
        sc = None
        for _sc in pta._signalcollections:
            if _sc._pulsar.name == pulsar.name:
                sc = _sc
                break
        kappa_full = sc.get_phi(params)
        red_PSD = kappa_full[:nmodes] * psr_tspan

        Nvec = np.array(sc.get_ndiag(params))
        U, _ = create_quantization_matrix(pulsar.toas, nmin=1)
        n_epochs = U.shape[1]
        cadence = psr_tspan / n_epochs

        epoch_variance = np.zeros(n_epochs)
        for j in range(n_epochs):
            toa_mask = U[:, j].astype(bool)
            epoch_variance[j] = 1.0 / np.sum(1.0 / Nvec[toa_mask])

        sigma_epoch_sq = np.median(epoch_variance)
        white_PSD = np.full(nmodes, 2.0 * sigma_epoch_sq * cadence)
        
        pulsar_PSD_red[i] = red_PSD
        pulsar_PSD_white[i] = white_PSD
        pulsar_PSD_total[i] = red_PSD + white_PSD

    return pulsar_PSD_total, pulsar_PSD_red, pulsar_PSD_white