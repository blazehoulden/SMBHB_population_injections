from enterprise_extensions.frequentist import F_statistic

def CGW_SNR(
    population,
    psrs_clean,
    raw_noise_params,
    pta,
):
    """
    Compute the SNR of a CGW signal for a given population and set of pulsars.

    Parameters
    ----------
    population : list of dict
        A list of dictionaries, each containing the parameters of a SMBHB in the population.
    psrs_clean : list of str
        A list of pulsar names to be used in the analysis.
    raw_noise_params : dict
        A dictionary containing the noise parameters for each pulsar.
    Tspan : float
        The total observation time span in years.

    Returns
    -------
    snr_results : dict
        A dictionary containing the SNR results for each SMBHB in the population.
    """
    Fpstat = F_statistic.Fpstat(
        psrs=psrs_clean, 
        noise_dict=raw_noise_params, 
        pta=pta, 
        psrTerm=True, 
        bayesephem=True, 
        tnequad=False)
    
    F_stat = Fpstat.compute_Fp(fgw=frequency)