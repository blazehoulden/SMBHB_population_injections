import numpy as np
from enterprise_extensions.frequentist.Fe_statistic import innerProduct_rr
from enterprise_extensions.deterministic import cw_delay

def compute_cgw_signal_enterprise(psr, binary):
    lum_dist = binary.D_comov * (1 + binary.z)  # convert comoving to luminosity distance
    
    s_a = cw_delay(
        toas    = psr.toas,
        pos     = psr.pos,          # pulsar sky position unit vector
        pdist   = psr.pdist,        # pulsar distance (for pulsar term)
        cos_gwtheta = np.cos(np.pi / 2.0 - binary.dec),  # GW source sky position (cos(theta))
        gwphi   = binary.ra,       # GW source sky position (radians)
        cos_inc = np.cos(binary.iota),     # cos(inclination) of the binary's orbital plane
        log10_mc = np.log10(binary.Mc),    # chirp mass in solar masses
        log10_fgw = np.log10(binary.f),  # GW frequency in Hz
        log10_dist = np.log10(lum_dist),# luminosity distance in Mpc
        log10_h = np.log10(binary.h0),             # if None, computed from mc/dist/fgw
        phase0  = binary.phi0,
        psi     = binary.psi,
        psrTerm = False             # set True to include pulsar term
    )

def compute_CGW_snr(pta, params, signal_fn, binary):
    """
    Compute matched-filter SNR for a CGW with known parameters.
    
    pta          : enterprise PTA object (same one used for OS)
    params       : noise dict from your OS run (ML noise params)
    signal_fn    : function(toas, psr_pos, binary_params) -> residuals s_a(t)
    binary: dict of binary parameters (chirp mass, dist, freq, sky pos, etc.)
    """
    
    # These are the same calls the Fe-statistic makes internally
    phiinvs = pta.get_phiinv(params, logdet=False)   # red noise prior inverse
    TNTs    = pta.get_TNT(params)                     # T^T N^{-1} T per pulsar
    Ts      = pta.get_basis()                         # combined basis T per pulsar
    Nvecs   = pta.get_ndiag(params)                   # white noise N per pulsar

    rho_sq = 0.0

    for psr, Nvec, TNT, phiinv, T in zip(pta.pulsars, Nvecs, TNTs, phiinvs, Ts):

        # Build Sigma = phi^{-1} + T^T N^{-1} T  (same as Fe_statistic.py line-for-line)
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

        # Compute the injected signal at this pulsar's TOAs
        # Use enterprise_extensions.deterministic.cw_delay for this
        s_a = signal_fn(psr.toas, psr.pos, binary)  # shape (n_toas,)

        # Inner product (s_a | s_a) using the exact same innerProduct_rr
        # that the Fe-statistic uses — this is (s|s) with full noise model
        rho_sq_a = innerProduct_rr(s_a, s_a, Nvec, T, TNT, Sigma)

        rho_sq += rho_sq_a

    return np.sqrt(rho_sq)

def compute_CGW_snr_binary_population(pta, params, signal_fn, population):
    """
    Compute CGW SNR for each binary in the population.
    
    Returns list of SNRs corresponding to the input population list.
    """
    snrs = []
    for binary in population:
        snr = compute_CGW_snr(pta, params, signal_fn, binary)
        snrs.append(snr)
    return snrs