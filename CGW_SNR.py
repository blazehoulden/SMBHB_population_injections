import numpy as np
from enterprise_extensions.frequentist.Fe_statistic import innerProduct_rr
from enterprise_extensions.deterministic import cw_delay


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


def compute_cgw_snr(psrs, pta, noise_params, binary):
    """
    Compute matched-filter SNR for a single CGW source.

    Parameters
    ----------
    psrs         : list of enterprise Pulsar objects (e.g. psrs_clean),
                   ordered consistently with how the PTA was constructed
    pta          : enterprise PTA object
    noise_params : noise parameter dict (e.g. ML noise params from OS run)
    binary       : binary object with attributes (Mc, f, h0, ra, dec, iota,
                   psi, phi0, D_comov, z)

    Returns
    -------
    snr : float
    """
    phiinvs = pta.get_phiinv(noise_params, logdet=False)
    TNTs    = pta.get_TNT(noise_params)
    Ts      = pta.get_basis()
    Nvecs   = pta.get_ndiag(noise_params)

    # pta.pulsars is a list of strings (names); zip against the actual objects
    psr_map = {psr.name: psr for psr in psrs}

    rho_sq = 0.0
    for psr_name, Nvec, TNT, phiinv, T in zip(pta.pulsars, Nvecs, TNTs, phiinvs, Ts):
        psr = psr_map[psr_name]
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        s_a = compute_cgw_signal_enterprise(psr, binary)
        rho_sq += innerProduct_rr(s_a, s_a, Nvec, T, TNT, Sigma)

    return np.sqrt(rho_sq)


def compute_cgw_snr_population(psrs, pta, noise_params, population):
    """
    Compute CGW SNR for each binary in a population.

    Parameters
    ----------
    psrs         : list of enterprise Pulsar objects
    pta          : enterprise PTA object
    noise_params : noise parameter dict
    population   : iterable of binary objects

    Returns
    -------
    snrs : list of float, same length and order as population
    """
    return [compute_cgw_snr(psrs, pta, noise_params, binary) for binary in population]