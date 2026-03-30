import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator
from signal_injection import inject_population_nufft
from pta_builder import build_pta_and_params
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from config import Msun


def compute_single_binary_os(binary, psrs_clean, noise_params_15yr, Tspan):
    """Compute OS for single binary with detailed diagnostics."""
    try:
        psrs_single = inject_population_nufft(psrs_clean, [binary], pure_signal=True, verbose=False)
        
        pta_single, _, params = build_pta_and_params(
            psrs=psrs_single, noise_params_15yr=noise_params_15yr, 
            Tspan=Tspan, crn_name="gw"
        )
        
        ostat = opt_stat.OptimalStatistic(psrs_single, pta=pta_single, orf='hd')
        xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params)
        
        snr = OS / OS_sig
        h_0 = binary.h0
        r_amp = h_0 / (2 * np.pi * binary.f)
        
        # Sky location metrics relative to pulsars
        psr_separations = []
        for psr in psrs_clean:
            psr_ra = psr._raj
            psr_dec = psr._decj
            # Angular separation (small angle approximation)
            delta_ra = binary.ra - psr_ra
            delta_dec = binary.dec - psr_dec
            sep = np.sqrt(delta_ra**2 + delta_dec**2)
            psr_separations.append(np.degrees(sep))
        
        return {
            'binary': binary,
            'OS': OS,
            'OS_sig': OS_sig,
            'SNR': snr,
            'abs_SNR': abs(snr),
            'h_0': h_0,
            'residual_amplitude_us': r_amp * 1e6,
            'frequency_nHz': binary.f * 1e9,
            'chirp_mass_Msun': binary.Mc / Msun,
            'comoving_distance_Mpc': binary.D_comov,
            'ra_deg': np.degrees(binary.ra),
            'dec_deg': np.degrees(binary.dec),
            'mean_rho': np.mean(rho),
            'std_rho': np.std(rho),
            'pos_corr': np.sum(rho > 0),
            'neg_corr': np.sum(rho < 0),
            'min_psr_separation_deg': np.min(psr_separations),
            'mean_psr_separation_deg': np.mean(psr_separations),
            'success': True
        }
    except Exception as e:
        return {'binary': binary, 'success': False, 'error': str(e)}


def analyze_individual_binaries(population, psrs_clean, noise_params_15yr, Tspan,
                                max_binaries=50, sort_by='SNR'):
    """Analyze individual binaries to find loudest sources."""
    N_analyze = min(max_binaries, len(population)) if max_binaries else len(population)
    
    print(f"Analyzing top {N_analyze} binaries by strain amplitude...")
    
    # Pre-screen by strain
    lum_dist = np.array([b.D_comov for b in population])
    lum_dist *= (1 + np.array([b.z for b in population]))
    h0_values = np.array([b.h0 for b in population])
    top_indices = np.argsort(h0_values)[::-1][:N_analyze]
    
    results = []
    for i, idx in enumerate(top_indices):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{N_analyze}")
        
        result = compute_single_binary_os(population[idx], psrs_clean, noise_params_15yr, Tspan)

        if result['success']:
            result['index'] = idx
            result['pre_rank'] = i + 1
            results.append(result)
    if len(results) == 0:
        return None
    
    df = pd.DataFrame(results)
    df = df.sort_values(by=sort_by, ascending=False, key=abs if 'abs' in sort_by else lambda x: x)
    df['final_rank'] = range(1, len(df) + 1)
    
    print(f"\n✓ Successfully analyzed {len(df)} binaries")
    
    return df


# Example usage
"""
# Run the analysis
df = analyze_individual_binaries(
    population, psrs_clean, noise_params_15yr, Tspan,
    max_binaries=50, sort_by='abs_SNR'
)

# Print detailed statistics
print_binary_statistics(df, top_n=10)

# Create comprehensive plots
plot_binary_analysis(df, save_prefix='top_binaries')

# Access top binary
top_binary = df.iloc[0]
print(f"\nTop binary has SNR = {top_binary['SNR']:.2f}")
"""