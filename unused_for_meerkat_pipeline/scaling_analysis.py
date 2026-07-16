import numpy as np
from scipy.interpolate import interp1d
from optimal_statistic import compute_os_for_N_binaries_incremental


def find_N_for_target_SNR(results, target_SNR):
    """Find number of binaries needed for target SNR via interpolation."""
    N_vals = np.array(results['N_binaries'])
    SNR_vals = np.array(results['SNR'])
    
    if target_SNR < SNR_vals.min():
        return None, "Target SNR below minimum achieved"
    if target_SNR > SNR_vals.max():
        return None, "Target SNR above maximum achieved"
    
    interp = interp1d(SNR_vals, N_vals, kind='cubic', fill_value='extrapolate')
    N_needed = int(interp(target_SNR))
    
    return N_needed, "Interpolated"


def run_scaling_analysis(population, psrs_clean, params, Tspan, target_SNR=4.0, n_test_points=5):
    """Run SNR scaling analysis with incremental injection."""
    N_max = len(population)
    N_values = np.unique(np.logspace(np.log10(N_max/2), np.log10(N_max), n_test_points, dtype=int))
    N_values = sorted([n for n in N_values if n <= N_max])
    
    results = {
        'N_binaries': [],
        'OS': [],
        'OS_sig': [],
        'SNR': [],
        'mean_rho': [],
        'std_rho': [],
        'time': []
    }
    
    psrs_prev = None
    N_prev = 0
    
    for N_current in N_values:
        result = compute_os_for_N_binaries_incremental(
            N_total=N_current,
            N_prev=N_prev,
            population=population,
            psrs_prev=psrs_prev,
            psrs_clean=psrs_clean,
            params=params,
            Tspan=Tspan
        )
        
        if result['success']:
            results['N_binaries'].append(result['N'])
            results['OS'].append(result['OS'])
            results['OS_sig'].append(result['OS_sig'])
            results['SNR'].append(result['SNR'])
            results['mean_rho'].append(result['mean_rho'])
            results['std_rho'].append(result['std_rho'])
            results['time'].append(result['time'])
            
            psrs_prev = result['psrs_new']
            N_prev = N_current
    
    # Find N needed
    N_needed = None
    if len(results['N_binaries']) > 2:
        N_needed, status = find_N_for_target_SNR(results, target_SNR)
    
    return results, N_needed