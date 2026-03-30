#!/usr/bin/env python3
"""
Wrapper to cross-validate your generate_smbhb_population against GWB_SMBHB.py
Uses your actual population generation code.
"""

from SMBHB_pop_synth import generate_smbhb_population
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Import your population generator
# Adjust this import based on your file structure:
# from your_module import generate_smbhb_population

# Import the GWB calculation from the validation script
from gwb_strain_cross_validation import (
    compute_gwb_strain_method1,
    comoving_distance,
    cosmo_par
)

def prepare_population_for_gwb_code(population, strain_data=None):
    """
    Ensure population has all fields needed for GWB_SMBHB.py method.
    
    Parameters
    ----------
    population : list of dict
        From generate_smbhb_population
    strain_data : dict, optional
        If provided, adds h_c_contrib to each binary
        
    Returns
    -------
    list of dict with added 'D_comov' field if missing
    """
    
    for i, binary in enumerate(population):
        # Ensure comoving distance is computed
        if 'D_comov' not in binary:
            z = binary.z
            binary['D_comov'] = comoving_distance(z, cosmo_par)
        
        # Ensure eccentricity (default to nearly circular)
        if 'e' not in binary:
            binary['e'] = 1e-6
        
        # # Add strain contribution if available
        # if strain_data is not None and 'h_c_individual' in strain_data:
        #     binary['h_c_contrib'] = strain_data['h_c_individual'][i]
    
    return population


def cross_validate_with_your_population(
    n_binaries=10000,
    T_obs=16.0,
    n_freq_bins=50,
    plot=True,
    **population_kwargs
):
    """
    Generate population with your code and cross-validate strain calculation.
    
    Parameters
    ----------
    n_binaries : int
        Number of binaries to generate
    T_obs : float
        Observation time [years]
    n_freq_bins : int
        Number of frequency bins
    plot : bool
        Create comparison plots
    **population_kwargs
        Additional arguments passed to generate_smbhb_population
        
    Returns
    -------
    dict with comparison results
    """
    
    print("="*70)
    print("CROSS-VALIDATION WITH YOUR POPULATION CODE")
    print("="*70)
    print(f"Generating {n_binaries} binaries...")
    print()
    
    # ========================================================================
    # STEP 1: Generate population using YOUR code
    # ========================================================================
    
    # Uncomment and adjust based on your actual function:
    population, strain_data = generate_smbhb_population(
        n_binaries=n_binaries,
        compute_strain=True,
        n_freq_bins=n_freq_bins,
        T_obs=T_obs,
        **population_kwargs
    )
    
    # For now, use test population
    from gwb_strain_cross_validation import create_test_population
    # population = create_test_population(n_binaries, z_max=2.0)
    # strain_data = None  # Your code would return this
    
    print(f"✓ Generated {len(population)} binaries")
    print()
    
    # Prepare population
    population = prepare_population_for_gwb_code(population, strain_data)
    
    # ========================================================================
    # STEP 2: Compute strain with YOUR method (if available)
    # ========================================================================
    
    if strain_data is not None:
        print("Your method results:")
        print(f"  Frequency bins: {len(strain_data['bin_centres'])}")
        print(f"  Peak h_c: {np.max(strain_data['h_c_total']):.3e}")
        print()
    
    # ========================================================================
    # STEP 3: Compute strain with GWB_SMBHB.py method
    # ========================================================================
    
    print("Computing strain using GWB_SMBHB.py method...")
    gwb_result = compute_gwb_strain_method1(population, T_obs, n_freq_bins)
    print(f"  Peak h_c: {np.max(gwb_result['h_c']):.3e}")
    print()
    
    # ========================================================================
    # STEP 4: Compare if both methods available
    # ========================================================================
    
    if strain_data is not None and 'h_c_total' in strain_data:
        print("COMPARISON:")
        print("-" * 70)
        
        # Interpolate to common grid
        f_your = strain_data['bin_centres']
        h_c_your = strain_data['h_c_total']
        
        f_gwb = gwb_result['frequencies']
        h_c_gwb = gwb_result['h_c']
        
        # Common frequency range
        f_min = max(f_your.min(), f_gwb.min())
        f_max = min(f_your.max(), f_gwb.max())
        f_common = np.logspace(np.log10(f_min), np.log10(f_max), n_freq_bins)
        
        # Interpolate
        mask_your = h_c_your > 0
        mask_gwb = h_c_gwb > 0
        
        if np.sum(mask_your) > 1 and np.sum(mask_gwb) > 1:
            interp_your = interp1d(f_your[mask_your], h_c_your[mask_your],
                                  kind='linear', bounds_error=False, fill_value=0.0)
            interp_gwb = interp1d(f_gwb[mask_gwb], h_c_gwb[mask_gwb],
                                 kind='linear', bounds_error=False, fill_value=0.0)
            
            h_c_your_interp = interp_your(f_common)
            h_c_gwb_interp = interp_gwb(f_common)
            
            mask_both = (h_c_your_interp > 0) & (h_c_gwb_interp > 0)
            
            if np.sum(mask_both) > 0:
                frac_diff = np.abs(h_c_your_interp[mask_both] - h_c_gwb_interp[mask_both]) / \
                           (0.5 * (h_c_your_interp[mask_both] + h_c_gwb_interp[mask_both]))
                
                print(f"  Mean fractional difference: {np.mean(frac_diff):.3e}")
                print(f"  Median fractional difference: {np.median(frac_diff):.3e}")
                print(f"  Max fractional difference: {np.max(frac_diff):.3e}")
                
                if np.mean(frac_diff) < 0.1:
                    print("\n✓ Methods agree within 10% - EXCELLENT!")
                elif np.mean(frac_diff) < 0.5:
                    print("\n⚠ Methods differ by 10-50% - investigate")
                else:
                    print("\n❌ Methods differ by >50% - likely bug!")
        
        # Plot comparison
        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # Top: strain comparison
            ax = axes[0]
            ax.loglog(f_your, h_c_your, 'b-', label='Your method', linewidth=2)
            ax.loglog(f_gwb, h_c_gwb, 'r--', label='GWB_SMBHB.py', linewidth=2, alpha=0.7)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Characteristic Strain $h_c$')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Strain Comparison (N={n_binaries}, T={T_obs} yr)')
            
            # Bottom: difference
            ax = axes[1]
            if np.sum(mask_both) > 0:
                ax.semilogx(f_common[mask_both], frac_diff * 100, 'k-', linewidth=2)
                ax.axhline(10, color='orange', linestyle='--', alpha=0.5)
                ax.axhline(50, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('Frequency [Hz]')
                ax.set_xlim(1e-9, 1e-7)
                ax.set_ylabel('Fractional Difference [%]')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('your_population_validation.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print("\nPlot saved to: your_population_validation.png")
    
    else:
        print("⚠ Your method didn't return strain_data - showing GWB_SMBHB.py result only")
        
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.loglog(gwb_result['frequencies'], gwb_result['h_c'], 'b-', linewidth=2)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Characteristic Strain $h_c$')
            ax.set_xlim(1e-9, 1e-7)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'GWB Strain (GWB_SMBHB.py method, N={n_binaries}, T={T_obs} yr)')
            plt.tight_layout()
            plt.savefig('gwb_strain_only.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    print("="*70)
    
    return {
        'population': population,
        'strain_data_yours': strain_data,
        'gwb_result': gwb_result
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Quick test with default parameters
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick validation test")
    print("="*70 + "\n")
    
    results = cross_validate_with_your_population(
        n_binaries=5000,
        T_obs=15.0,
        n_freq_bins=50,
        plot=True
    )
    
    # Example 2: With custom population parameters
    # Uncomment when you have generate_smbhb_population imported:
    # 
    # print("\n" + "="*70)
    # print("EXAMPLE 2: Custom population parameters")
    # print("="*70 + "\n")
    # 
    # results = cross_validate_with_your_population(
    #     n_binaries=10000,
    #     T_obs=30.0,
    #     n_freq_bins=100,
    #     z_max=2.0,
    #     mass_distribution='exponential_damping',
    #     alpha_0=1.21,
    #     mass_min=1e7,
    #     mass_max=1e10,
    #     plot=True
    # )