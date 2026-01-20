import time
import numpy as np
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from signal_injection import inject_population_into_psrs
from pta_builder import build_pta_and_params


def compute_os_for_N_binaries_incremental(N_total, N_prev, population, psrs_prev,
                                          psrs_clean, params, Tspan):
    """Compute OS by incrementally injecting binaries."""
    start_time = time.time()
    
    try:
        binaries_to_inject = population[N_prev:N_total]
        ΔN = len(binaries_to_inject)
        
        if N_prev == 0:
            psrs_new = inject_population_into_psrs(
                psrs_clean, binaries_to_inject, pure_signal=True, add=False, verbose=False
            )
        else:
            psrs_new = inject_population_into_psrs(
                psrs_prev, binaries_to_inject, pure_signal=True, add=True, verbose=False
            )
        
        pta_temp, _, params_out = build_pta_and_params(
            psrs=psrs_new, noise_params_15yr=params, Tspan=Tspan, use_efac_only=True
        )
        
        ostat = opt_stat.OptimalStatistic(psrs_new, pta=pta_temp, orf='hd')
        xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_out)
        
        return {
            'N': N_total,
            'ΔN': ΔN,
            'OS': OS,
            'OS_sig': OS_sig,
            'SNR': OS / OS_sig,
            'mean_rho': np.mean(rho),
            'std_rho': np.std(rho),
            'time': time.time() - start_time,
            'success': True,
            'psrs_new': psrs_new
        }
        
    except Exception as e:
        print(f"✗ Failed for N={N_total}: {e}")
        return {'N': N_total, 'success': False, 'error': str(e)}