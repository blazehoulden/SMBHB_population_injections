"""
ULTRA-DETAILED SNR DEBUGGING
Shows complete breakdown of every calculation for every binary
"""

import numpy as np
from SMBHB_pop_synth import H0, h_circ
import json


def analyze_snr_calculation_complete(population, strain_data, pulsars, pulsar_noise_params, 
                                     T_obs, output_file='complete_snr_breakdown.json'):
    """
    Complete breakdown of SNR calculation showing EVERYTHING.
    
    This will show for EACH binary:
    - All input values
    - h_c and h_square
    - omega_GW calculation
    - PSD for each pulsar
    - Numerator and denominator for each pulsar pair
    - Integrand for each pulsar pair
    - Final SNR contribution
    
    Parameters
    ----------
    population : list
        Binary population
    strain_data : dict
        Strain data
    pulsars : list
        Pulsar objects
    pulsar_noise_params : dict
        Noise parameters
    T_obs : float
        Observation time [s]
    output_file : str
        JSON file to save complete breakdown
    """
    
    print("\n" + "="*80)
    print("COMPLETE SNR BREAKDOWN FOR ALL BINARIES")
    print("="*80)
    
    # Constants
    fyr = 1.0 / (365.25 * 24 * 3600)
    
    # Precompute pulsar pairs
    N_pulsars = len(pulsars)
    N_pairs = N_pulsars * (N_pulsars - 1) // 2
    
    pairs = []
    gamma_vals = []
    
    idx = 0
    for i in range(N_pulsars):
        for j in range(i + 1, N_pulsars):
            cos_xi = np.dot(pulsars[i].pos, pulsars[j].pos)
            gamma = 3 * (1.0/3.0 + (1.0 - cos_xi) / 2.0 * (np.log((1.0 - cos_xi) / 2.0) - 1.0/6.0))
            pairs.append([i, j])
            gamma_vals.append(gamma)
            idx += 1
    
    pairs = np.array(pairs)
    gamma_vals = np.array(gamma_vals)
    
    print(f"\nPulsar Array Setup:")
    print(f"  N_pulsars: {N_pulsars}")
    print(f"  N_pairs: {N_pairs}")
    print(f"  Gamma range: [{np.min(gamma_vals):.4f}, {np.max(gamma_vals):.4f}]")
    print(f"  Gamma mean: {np.mean(gamma_vals):.4f}")
    
    print(f"\nConstants:")
    print(f"  H0 = {H0:.6e}")
    print(f"  fyr = {fyr:.6e} Hz")
    print(f"  T_obs = {T_obs:.6e} s ({T_obs/(365.25*86400):.4f} yr)")
    
    # Prefactor for SNR
    prefactor = (H0**2 / (4 * np.pi**2))**2 * T_obs
    print(f"  Prefactor = (H0^2/(4π^2))^2 * T_obs = {prefactor:.6e}")
    
    # Get bin edges if available
    bin_edges = strain_data.get('bin_edges', None)
    
    # Storage for all results
    all_results = {
        'metadata': {
            'N_pulsars': N_pulsars,
            'N_pairs': N_pairs,
            'N_binaries': len(population),
            'T_obs_seconds': T_obs,
            'T_obs_years': T_obs/(365.25*86400),
            'H0': H0,
            'prefactor': prefactor
        },
        'pulsar_info': [],
        'binaries': []
    }
    
    # Save pulsar info
    for i, pulsar in enumerate(pulsars):
        pulsar_info = {
            'index': i,
            'name': pulsar.name,
            'has_red_noise': pulsar.name in pulsar_noise_params and bool(pulsar_noise_params[pulsar.name]['red_noise'])
        }
        if pulsar_info['has_red_noise']:
            params = pulsar_noise_params[pulsar.name]['red_noise']
            pulsar_info['log10_A'] = params['log10_A']
            pulsar_info['gamma'] = params['gamma']
        all_results['pulsar_info'].append(pulsar_info)
    
    # Process each binary
    SNR_squared_cumulative = 0.0
    
    print("\n" + "="*80)
    print("PROCESSING BINARIES")
    print("="*80)
    
    for bin_idx, binary in enumerate(population):
        print(f"\n{'='*80}")
        print(f"BINARY {bin_idx}")
        print(f"{'='*80}")
        
        # Extract binary parameters
        freq = binary['f']
        Mc = binary['Mc']
        D_comov = binary['D_comov']
        h_square = binary.get('h_square', 0)
        h_c_contrib = binary.get('h_c_contrib', strain_data.get('h_c_individual', [0])[bin_idx])
        freq_bin_str = binary.get('freq_bin', str(bin_idx))
        
        print(f"\nBinary Parameters:")
        print(f"  Index: {bin_idx}")
        print(f"  Chirp Mass (Mc): {Mc:.6e} solar masses")
        print(f"  Comoving Distance (D): {D:.6e} Mpc")
        print(f"  Frequency (f): {freq:.6e} Hz = {freq*1e9:.6f} nHz")
        print(f"  Frequency bin: {freq_bin_str}")
        
        print(f"\nStrain Values:")
        print(f"  h_square: {h_square:.6e}")
        print(f"  h_c_contrib: {h_c_contrib:.6e}")
        
        # Calculate omega_GW
        omega_GW = 2 * np.pi**2 / (3 * H0**2) * freq**3 * h_c_contrib**2
        
        print(f"\nOmega_GW Calculation:")
        print(f"  Formula: Ω_GW = (2π²)/(3H₀²) * f³ * h_c²")
        print(f"  2π²/(3H₀²) = {2 * np.pi**2 / (3 * H0**2):.6e}")
        print(f"  f³ = {freq**3:.6e}")
        print(f"  h_c² = {h_c_contrib**2:.6e}")
        print(f"  Ω_GW = {omega_GW:.6e}")
        
        # Get frequency bin width
        if bin_edges is not None and int(freq_bin_str) < len(bin_edges) - 1:
            bin_idx_int = int(freq_bin_str)
            delta_f = bin_edges[bin_idx_int + 1] - bin_edges[bin_idx_int]
        else:
            delta_f = freq * 0.1
        
        print(f"\nFrequency Bin:")
        print(f"  Delta_f: {delta_f:.6e} Hz")
        print(f"  Delta_f/f: {delta_f/freq:.6f} (fractional bandwidth)")
        
        # Calculate PSD for each pulsar at this frequency
        print(f"\nPulsar PSDs at f={freq:.6e} Hz:")
        psd_values = np.zeros(N_pulsars)
        
        # for i, pulsar in enumerate(pulsars):
        #     if pulsar.name in pulsar_noise_params and pulsar_noise_params[pulsar.name]['red_noise']:
        #         params = pulsar_noise_params[pulsar.name]['red_noise']
        #         log10_A = params['log10_A']
        #         gamma = params['gamma']
                
        #         psd = 10**(2 * log10_A) * (freq / fyr)**(-gamma)
        #         psd_values[i] = psd
                
        #         print(f"  Pulsar {i} ({pulsar.name}):")
        #         print(f"    log10_A = {log10_A:.4f}, gamma = {gamma:.4f}")
        #         print(f"    10^(2*log10_A) = {10**(2*log10_A):.6e}")
        #         print(f"    (f/fyr)^(-gamma) = {(freq/fyr)**(-gamma):.6e}")
        #         print(f"    PSD = {psd:.6e}")
        #     else:
        #         print(f"  Pulsar {i} ({pulsar.name}): NO RED NOISE")
        
        # Calculate contribution from each pulsar pair
        print(f"\nPulsar Pair Contributions (showing all {N_pairs} pairs):")
        print(f"  Formula: integrand = (Ω²γ²)/(f⁶ * PSD_i * PSD_j) * Δf")
        
        pair_integrands = []
        
        for pair_idx in range(N_pairs):
            i, j = pairs[pair_idx]
            gamma = gamma_vals[pair_idx]
            psd_i = psd_values[i]
            psd_j = psd_values[j]
            
            # Calculate integrand
            numerator = omega_GW**2 * gamma**2
            denominator = freq**6 * psd_i * psd_j
            
            if denominator > 0:
                integrand = numerator / denominator * delta_f
            else:
                integrand = 0.0
            
            pair_integrands.append(integrand)
            
            # # Print detailed info
            # print(f"\n  Pair {pair_idx}: Pulsars ({i},{j}) - {pulsars[i].name} & {pulsars[j].name}")
            # print(f"    Gamma (HD coeff): {gamma:.6f}")
            # print(f"    PSD_i: {psd_i:.6e}")
            # print(f"    PSD_j: {psd_j:.6e}")
            # print(f"    Numerator (Ω²γ²): {numerator:.6e}")
            # print(f"      Ω² = {omega_GW**2:.6e}")
            # print(f"      γ² = {gamma**2:.6e}")
            # print(f"    Denominator (f⁶ * PSD_i * PSD_j): {denominator:.6e}")
            # print(f"      f⁶ = {freq**6:.6e}")
            # print(f"      PSD_i * PSD_j = {psd_i * psd_j:.6e}")
            # print(f"    Integrand (before Δf): {numerator/denominator if denominator > 0 else 0:.6e}")
            # print(f"    Integrand (with Δf): {integrand:.6e}")
        
        pair_integrands = np.array(pair_integrands)
        
        # Sum over all pairs
        sum_integrand = np.sum(pair_integrands)
        
        print(f"\n  Sum of integrands over all pairs: {sum_integrand:.6e}")
        print(f"  Top 5 pair contributions:")
        top_5_idx = np.argsort(pair_integrands)[-5:][::-1]
        for rank, idx in enumerate(top_5_idx, 1):
            i, j = pairs[idx]
            print(f"    {rank}. Pair ({i},{j}): {pair_integrands[idx]:.6e}")
        
        # Calculate SNR² for this binary
        SNR_sq_binary = sum_integrand * prefactor
        SNR_binary = np.sqrt(SNR_sq_binary)
        
        SNR_squared_cumulative += SNR_sq_binary
        SNR_cumulative = np.sqrt(SNR_squared_cumulative)
        
        print(f"\nSNR Calculation for this Binary:")
        print(f"  Sum(integrand) = {sum_integrand:.6e}")
        print(f"  Prefactor = {prefactor:.6e}")
        print(f"  SNR² = Sum(integrand) × Prefactor = {SNR_sq_binary:.6e}")
        print(f"  SNR = {SNR_binary:.6e}")
        
        print(f"\nCumulative SNR:")
        print(f"  Cumulative SNR² = {SNR_squared_cumulative:.6e}")
        print(f"  Cumulative SNR = {SNR_cumulative:.6e}")
        
        # Save to results
        binary_result = {
            'index': bin_idx,
            'frequency_Hz': freq,
            'frequency_nHz': freq * 1e9,
            'Mc': Mc,
            'D_comov': D_comov,
            'h_square': h_square,
            'h_c_contrib': h_c_contrib,
            'omega_GW': omega_GW,
            'delta_f': delta_f,
            'delta_f_over_f': delta_f / freq,
            'psd_values': psd_values.tolist(),
            'pair_integrands': pair_integrands.tolist(),
            'sum_integrand': sum_integrand,
            'SNR_squared_this_binary': SNR_sq_binary,
            'SNR_this_binary': SNR_binary,
            'SNR_squared_cumulative': SNR_squared_cumulative,
            'SNR_cumulative': SNR_cumulative
        }
        all_results['binaries'].append(binary_result)
        
        # CRITICAL DIAGNOSTIC
        if SNR_binary > 1.0:
            print(f"\n  ⚠️⚠️⚠️  WARNING: SNR > 1 from single binary! ⚠️⚠️⚠️")
        if SNR_binary > 100:
            print(f"\n  🚨🚨🚨  CRITICAL: SNR > 100 from single binary! 🚨🚨🚨")
            print(f"  🚨  This indicates a SERIOUS ERROR in the calculation!")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total binaries: {len(population)}")
    print(f"Final SNR² = {SNR_squared_cumulative:.6e}")
    print(f"Final SNR = {SNR_cumulative:.6e}")
    
    # Per-binary statistics
    snr_per_binary = [b['SNR_this_binary'] for b in all_results['binaries']]
    print(f"\nPer-binary SNR statistics:")
    print(f"  Min: {np.min(snr_per_binary):.6e}")
    print(f"  Max: {np.max(snr_per_binary):.6e}")
    print(f"  Mean: {np.mean(snr_per_binary):.6e}")
    print(f"  Median: {np.median(snr_per_binary):.6e}")
    
    # Expected vs actual
    expected_per_binary = 4.0 / np.sqrt(4000)
    print(f"\nExpected SNR per binary (if SNR=4 for N=4000): {expected_per_binary:.6e}")
    print(f"Actual mean SNR per binary: {np.mean(snr_per_binary):.6e}")
    print(f"Ratio (actual/expected): {np.mean(snr_per_binary)/expected_per_binary:.2f}x")
    
    # Save to JSON
    print(f"\nSaving complete breakdown to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved!")
    
    return all_results


def diagnose_high_snr(results_file='complete_snr_breakdown.json'):
    """
    Analyze the complete breakdown to diagnose why SNR is too high.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("DIAGNOSIS: WHY IS SNR SO HIGH?")
    print("="*80)
    
    binaries = results['binaries']
    
    # Check h_c values
    h_c_values = [b['h_c_contrib'] for b in binaries]
    print(f"\n1. CHARACTERISTIC STRAIN (h_c):")
    print(f"   Range: [{np.min(h_c_values):.3e}, {np.max(h_c_values):.3e}]")
    print(f"   Mean: {np.mean(h_c_values):.3e}")
    print(f"   Expected: ~1e-16 to 1e-13")
    if np.mean(h_c_values) > 1e-13:
        print(f"   ❌ TOO HIGH by factor of {np.mean(h_c_values)/1e-15:.1f}x")
    elif np.mean(h_c_values) < 1e-17:
        print(f"   ❌ TOO LOW by factor of {1e-15/np.mean(h_c_values):.1f}x")
    else:
        print(f"   ✓ Reasonable")
    
    # Check omega_GW
    omega_values = [b['omega_GW'] for b in binaries]
    print(f"\n2. OMEGA_GW:")
    print(f"   Range: [{np.min(omega_values):.3e}, {np.max(omega_values):.3e}]")
    print(f"   Mean: {np.mean(omega_values):.3e}")
    print(f"   Expected: ~1e-12 to 1e-8")
    if np.mean(omega_values) > 1e-8:
        print(f"   ❌ TOO HIGH by factor of {np.mean(omega_values)/1e-10:.1f}x")
    
    # Check PSD values
    all_psds = []
    for b in binaries:
        all_psds.extend([p for p in b['psd_values'] if p > 0])
    print(f"\n3. PULSAR PSDs:")
    print(f"   Range: [{np.min(all_psds):.3e}, {np.max(all_psds):.3e}]")
    print(f"   Mean: {np.mean(all_psds):.3e}")
    print(f"   Expected at ~10 nHz: ~1e-26 to 1e-24")
    if np.mean(all_psds) < 1e-27:
        print(f"   ❌ TOO LOW by factor of {1e-25/np.mean(all_psds):.1f}x")
    
    # Check delta_f
    delta_f_values = [b['delta_f'] for b in binaries]
    delta_f_frac = [b['delta_f_over_f'] for b in binaries]
    print(f"\n4. FREQUENCY BINNING (Δf):")
    print(f"   Δf range: [{np.min(delta_f_values):.3e}, {np.max(delta_f_values):.3e}] Hz")
    print(f"   Δf/f range: [{np.min(delta_f_frac):.3f}, {np.max(delta_f_frac):.3f}]")
    print(f"   Mean Δf/f: {np.mean(delta_f_frac):.3f}")
    print(f"   Expected Δf/f: ~0.01 to 0.2")
    if np.mean(delta_f_frac) > 0.5:
        print(f"   ❌ TOO LARGE - bins are too wide!")
    
    # Check integrands
    all_integrands = []
    for b in binaries:
        all_integrands.extend(b['pair_integrands'])
    print(f"\n5. PAIR INTEGRANDS:")
    print(f"   Range: [{np.min(all_integrands):.3e}, {np.max(all_integrands):.3e}]")
    print(f"   Mean: {np.mean(all_integrands):.3e}")
    
    # Identify the problem
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS:")
    print(f"{'='*80}")
    
    problems = []
    if np.mean(h_c_values) > 1e-13:
        problems.append("h_c values are TOO HIGH")
    if np.mean(omega_values) > 1e-8:
        problems.append("omega_GW is TOO HIGH (likely due to high h_c)")
    if np.mean(all_psds) < 1e-27:
        problems.append("PSD values are TOO LOW")
    if np.mean(delta_f_frac) > 0.5:
        problems.append("Frequency bins are TOO WIDE")
    
    if problems:
        print("Identified problems:")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
    else:
        print("No obvious problems found in individual components.")
        print("Issue may be in the formula or prefactor.")
    
    return results


if __name__ == "__main__":
    print(__doc__)