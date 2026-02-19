#!/usr/bin/env python3
"""
DEEP DEBUGGING VERSION of GWB strain cross-validation.
Tracks every intermediate calculation step to diagnose discrepancies.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import jn as bessel_jn

# ============================================================================
# CONSTANTS (matching GWB_SMBHB.py)
# ============================================================================

G_SI = 6.67384e-11          # m^3/(kg s^2)
c_SI = 299792458.           # m/s
parsec = 3.08567758e16      # m
year = 31557600.            # s
solar_mass = 1.98855e30     # kg
Mpc = parsec * 1e6          # m

H0 = 67.3 * 1000 / Mpc      # s^-1
G = G_SI / (Mpc**3) * solar_mass   # Mpc^3/(s^2 Msun)
c = c_SI / Mpc              # Mpc/s

G_5_3 = G ** (5./3.)
c_8 = c ** 8
c_4 = c ** 4
pi = np.pi

cosmo_par = [H0, 0., 0.315, 0., 0.683]

# ============================================================================
# COSMOLOGY
# ============================================================================

def E_function(z, cosmo_par):
    H0, Omega_r, Omega_M, Omega_k, Omega_L = cosmo_par
    return 1.0 / np.sqrt(
        Omega_r * (1+z)**4 + 
        Omega_M * (1+z)**3 + 
        Omega_k * (1+z)**2 + 
        Omega_L
    )

def comoving_distance(z, cosmo_par):
    NN = 500
    D_M = 0.0
    for l in range(NN):
        z_l = z / NN * l
        z_u = z / NN * (l + 1)
        D_M += (E_function(z_l, cosmo_par) + E_function(z_u, cosmo_par)) * (z_u - z_l) / 2.0
    return c / H0 * D_M

def g_func(n, e):
    n_e = n * e
    jn_vals = np.array([
        bessel_jn(n - 2, n_e),
        bessel_jn(n - 1, n_e),
        bessel_jn(n, n_e),
        bessel_jn(n + 1, n_e),
        bessel_jn(n + 2, n_e)
    ])
    
    term1 = jn_vals[0] - 2.0*e*jn_vals[1] + 2.0/n*jn_vals[2] + 2.0*e*jn_vals[3] - jn_vals[4]
    term2 = jn_vals[0] - 2.0*jn_vals[2] + jn_vals[4]
    term3 = 4.0 / (3.0 * n**2) * jn_vals[2]**2
    
    return n**4 / 32.0 * (term1**2 + (1.0 - e**2) * term2**2 + term3)

def n_peak(e):
    c1, c2, c3, c4 = -1.01678, 5.57372, -4.9271, 1.68506
    return 2.0 * (1.0 + c1*e + c2*e**2 + c3*e**3 + c4*e**4) * (1.0 - e**2)**(-1.5)

# ============================================================================
# METHOD 1: GWB_SMBHB.py with DETAILED LOGGING
# ============================================================================

def compute_gwb_strain_method1_debug(population, T_obs=30.0, n_freq_bins=100, debug_binary_idx=0):
    """
    Method 1 with step-by-step logging for debugging.
    """
    
    T_obs_s = T_obs * year
    
    # Setup frequency grid
    min_f = 1.0 / T_obs_s
    max_f = 1e-7
    step_fobs = 1.0 / T_obs_s
    N_bin_f = int((max_f - min_f) / step_fobs) + 1
    
    F_obs = np.linspace(min_f, min_f + N_bin_f * step_fobs, N_bin_f + 1)
    hc_gwb_2_vec = np.zeros(N_bin_f + 1)
    
    print("\n" + "="*70)
    print("METHOD 1: GWB_SMBHB.py (DETAILED DEBUG)")
    print("="*70)
    print(f"Frequency grid: {min_f:.3e} to {max_f:.3e} Hz")
    print(f"Frequency step: {step_fobs:.3e} Hz")
    print(f"Number of bins: {N_bin_f}")
    print(f"T_obs: {T_obs} years = {T_obs_s:.3e} seconds")
    print(f"Constants: G={G:.6e}, c={c:.6e}")
    print(f"           G^(5/3)={G_5_3:.6e}, c^4={c_4:.6e}, c^8={c_8:.6e}")
    
    # Pre-compute comoving distance lookup
    N_zD = 10000
    min_zD = 0.001
    max_zD = 2.5
    delta_zD = (max_zD - min_zD) / N_zD
    
    zD_array = np.linspace(min_zD, max_zD, N_zD + 1)
    comoving_distance_array = np.array([comoving_distance(z, cosmo_par) for z in zD_array])
    
    # Debug single binary in detail
    binary = population[debug_binary_idx]
    
    print(f"\n{'='*70}")
    print(f"DEBUGGING BINARY {debug_binary_idx} (DETAILED STEP-BY-STEP)")
    print(f"{'='*70}")
    
    Mc = binary['Mc']
    z_loc = binary['z']
    f = binary['f']
    i_loc = binary.get('iota', 0.0)
    e_loc = binary.get('e', 1e-6)
    
    print(f"\nInput parameters:")
    print(f"  Chirp mass Mc = {Mc:.6e} M_sun")
    print(f"  Redshift z = {z_loc:.6f}")
    print(f"  GW frequency f = {f:.6e} Hz")
    print(f"  Inclination i = {i_loc:.6f} rad = {np.degrees(i_loc):.2f} deg")
    print(f"  Eccentricity e = {e_loc:.6e}")
    
    # Step 1: Compute Mc^(5/3)
    Mc_5_3 = Mc ** (5./3.)
    print(f"\nStep 1: Mc^(5/3) = {Mc_5_3:.6e}")
    
    # Step 2: Orbital frequencies
    f_orb = f / 2.0
    f_r = f_orb * (1.0 + z_loc)
    print(f"\nStep 2: Frequencies")
    print(f"  GW frequency (input): {f:.6e} Hz")
    print(f"  Orbital frequency f_orb = f/2 = {f_orb:.6e} Hz")
    print(f"  Rest-frame orbital freq f_r = f_orb*(1+z) = {f_r:.6e} Hz")
    
    # Step 3: Comoving distance
    id_z = int((z_loc - min_zD) / delta_zD)
    if id_z >= N_zD:
        id_z = N_zD - 1
    xd = (z_loc - zD_array[id_z]) / (zD_array[id_z + 1] - zD_array[id_z])
    D_M_distance = comoving_distance_array[id_z] * (1 - xd) + \
                   comoving_distance_array[id_z + 1] * xd
    
    print(f"\nStep 3: Comoving distance")
    print(f"  z = {z_loc:.6f}")
    print(f"  D_c = {D_M_distance:.6e} Mpc")
    print(f"  D_c^2 = {D_M_distance**2:.6e} Mpc^2")
    
    # Step 4: Polarization contribution
    a = 1.0 + np.cos(i_loc)**2
    b = -2.0 * np.cos(i_loc)
    MeanAng = np.sqrt(0.5 * (a**2 + b**2))
    
    print(f"\nStep 4: Polarization factor")
    print(f"  cos(i) = {np.cos(i_loc):.6f}")
    print(f"  a = 1 + cos²(i) = {a:.6f}")
    print(f"  b = -2*cos(i) = {b:.6f}")
    print(f"  MeanAng = sqrt(0.5*(a² + b²)) = {MeanAng:.6f}")
    print(f"  MeanAng² = {MeanAng**2:.6f}")
    
    # Step 5: Harmonic decomposition
    n_max_val = n_peak(e_loc)
    n_max = int(np.ceil(4.0 * n_max_val))
    
    print(f"\nStep 5: Harmonic decomposition")
    print(f"  n_peak(e={e_loc:.3e}) = {n_max_val:.3f}")
    print(f"  n_max = ceil(4 * n_peak) = {n_max}")
    
    # Process harmonics
    total_h2_contribution = 0.0
    
    for n in range(1, min(n_max + 1, 5)):  # Show first few harmonics
        g_n_e = g_func(n, e_loc)
        
        if g_n_e == 0:
            continue
        
        f_n = n * f_r
        index = (f_n / (1.0 + z_loc) - min_f) / step_fobs
        
        print(f"\n  Harmonic n={n}:")
        print(f"    g(n, e) = {g_n_e:.6e}")
        print(f"    f_n = n * f_r = {f_n:.6e} Hz")
        print(f"    f_n / (1+z) = {f_n / (1.0 + z_loc):.6e} Hz (observed)")
        print(f"    Bin index = {index:.2f}")
        
        if index < 0 or index > N_bin_f:
            print(f"    → Outside frequency range, skipped")
            continue
        
        # Amaro-Seoane formula broken down
        term1 = 2.0 * 2.0 * MeanAng**2
        term2 = 4.0 * G_5_3**2 * Mc_5_3**2
        term3 = (2.0 * pi * f_r)**(4./3.)
        term4 = 1.0 / (c_8 * D_M_distance**2)
        term5 = 1.0 / (n**2)
        term6 = f_n / (1.0 + z_loc) / step_fobs
        
        h2_n_square = term1 * term2 * term3 * term4 * term5 * g_n_e * term6
        
        print(f"    h² breakdown:")
        print(f"      4 * MeanAng² = {term1:.6e}")
        print(f"      4 * G^(10/3) * Mc^(10/3) = {term2:.6e}")
        print(f"      (2π f_r)^(4/3) = {term3:.6e}")
        print(f"      1/(c^8 * D_c²) = {term4:.6e}")
        print(f"      1/n² = {term5:.6e}")
        print(f"      g(n,e) = {g_n_e:.6e}")
        print(f"      f_n/(1+z)/Δf = {term6:.6e}")
        print(f"    → h²_n = {h2_n_square:.6e}")
        
        total_h2_contribution += h2_n_square
    
    if n_max > 4:
        print(f"\n  ... (showing first 4 harmonics, total n_max={n_max})")
    
    print(f"\n  TOTAL h² contribution from this binary = {total_h2_contribution:.6e}")
    print(f"  TOTAL h_c contribution = {np.sqrt(total_h2_contribution):.6e}")
    
    # Now compute for all binaries
    print(f"\n{'='*70}")
    print("Computing full population...")
    
    for n_s, binary in enumerate(population):
        Mc = binary['Mc']
        z_loc = binary['z']
        f = binary['f']
        i_loc = binary.get('iota', 0.0)
        e_loc = binary.get('e', 1e-6)
        
        if e_loc == 0:
            e_loc = 1e-6
        
        Mc_5_3 = Mc ** (5./3.)
        f_orb = f / 2.0
        f_r = f_orb * (1.0 + z_loc)
        
        id_z = int((z_loc - min_zD) / delta_zD)
        if id_z >= N_zD:
            id_z = N_zD - 1
        xd = (z_loc - zD_array[id_z]) / (zD_array[id_z + 1] - zD_array[id_z])
        D_M_distance = comoving_distance_array[id_z] * (1 - xd) + \
                       comoving_distance_array[id_z + 1] * xd
        
        a = 1.0 + np.cos(i_loc)**2
        b = -2.0 * np.cos(i_loc)
        MeanAng = np.sqrt(0.5 * (a**2 + b**2))
        
        n_max = int(np.ceil(4.0 * n_peak(e_loc)))
        
        for n in range(1, n_max + 1):
            g_n_e = g_func(n, e_loc)
            if g_n_e == 0:
                continue
            
            f_n = n * f_r
            index = (f_n / (1.0 + z_loc) - min_f) / step_fobs
            
            if index > N_bin_f or index < 0:
                continue
            
            h2_n_square = (2.0 * 2.0 * MeanAng**2 * 4.0 * G_5_3**2 * Mc_5_3**2 / 
                          (c_8 * D_M_distance**2) * (2.0 * pi * f_r)**(4./3.) / 
                          (n**2) * g_n_e * f_n / (1.0 + z_loc) / step_fobs)
            
            hc_gwb_2_vec[int(index)] += h2_n_square
    
    frequencies = 0.5 * (F_obs[:-1] + F_obs[1:])
    h_c_squared = hc_gwb_2_vec[:-1]
    h_c = np.sqrt(h_c_squared)
    
    print(f"Peak h_c: {np.max(h_c):.3e}")
    
    return {
        'frequencies': frequencies,
        'h_c_squared': h_c_squared,
        'h_c': h_c,
        'method': 'GWB_SMBHB'
    }

# ============================================================================
# METHOD 2: Circular orbit formula with DETAILED LOGGING
# ============================================================================

def compute_gwb_strain_method2_debug(population, T_obs=30.0, n_freq_bins=100, debug_binary_idx=0):
    """
    Method 2 with step-by-step logging for debugging.
    """
    
    YEAR_IN_SECONDS = 365.25 * 24 * 3600
    
    # Constants in SI
    G_SI_val = 6.67430e-11
    c_SI_val = 299792458.0
    MEGAPARSEC_IN_METERS = 3.085677581491367e22
    SOLAR_MASS_IN_KG = 1.98847e30
    
    print("\n" + "="*70)
    print("METHOD 2: Circular orbit formula (DETAILED DEBUG)")
    print("="*70)
    print(f"Constants (SI units):")
    print(f"  G = {G_SI_val:.6e} m³/(kg·s²)")
    print(f"  c = {c_SI_val:.6e} m/s")
    print(f"  1 Mpc = {MEGAPARSEC_IN_METERS:.6e} m")
    print(f"  1 M_sun = {SOLAR_MASS_IN_KG:.6e} kg")
    
    # Debug single binary
    binary = population[debug_binary_idx]
    
    print(f"\n{'='*70}")
    print(f"DEBUGGING BINARY {debug_binary_idx} (DETAILED STEP-BY-STEP)")
    print(f"{'='*70}")
    
    Mc = binary['Mc']
    z = binary['z']
    f_gw = binary['f']
    D_c = binary['D_comov']
    
    print(f"\nInput parameters:")
    print(f"  Chirp mass Mc = {Mc:.6e} M_sun")
    print(f"  Redshift z = {z:.6f}")
    print(f"  GW frequency f = {f_gw:.6e} Hz")
    print(f"  Comoving distance D_c = {D_c:.6e} Mpc")
    
    # Step 1: Convert to SI
    Mc_SI = Mc * SOLAR_MASS_IN_KG
    D_SI = D_c * MEGAPARSEC_IN_METERS
    f_obs = f_gw / (1.0 + z)
    
    print(f"\nStep 1: Convert to SI units")
    print(f"  Mc (SI) = {Mc_SI:.6e} kg")
    print(f"  D_c (SI) = {D_SI:.6e} m")
    print(f"  f_obs = f/(1+z) = {f_obs:.6e} Hz")
    
    # Step 2: Compute prefactor
    prefactor = 4.0 * (G_SI_val / c_SI_val**2)**(5./3.)
    
    print(f"\nStep 2: Compute prefactor")
    print(f"  G/c² = {G_SI_val / c_SI_val**2:.6e}")
    print(f"  (G/c²)^(5/3) = {(G_SI_val / c_SI_val**2)**(5./3.):.6e}")
    print(f"  4 * (G/c²)^(5/3) = {prefactor:.6e}")
    
    # Step 3: Compute h²
    Mc_term = Mc_SI**(5./3.)
    freq_term = (np.pi * f_obs)**(2./3.)
    denom = D_SI**2 * f_obs
    
    h_squared = prefactor * Mc_term * freq_term / denom
    
    print(f"\nStep 3: Compute h²")
    print(f"  Mc^(5/3) = {Mc_term:.6e}")
    print(f"  (π f_obs)^(2/3) = {freq_term:.6e}")
    print(f"  D_c² = {D_SI**2:.6e}")
    print(f"  D_c² * f_obs = {denom:.6e}")
    print(f"  h² = {h_squared:.6e}")
    print(f"  h_c = sqrt(h²) = {np.sqrt(h_squared):.6e}")
    
    # Step 4: Check against formula
    print(f"\nStep 4: Verification")
    print(f"  Formula: h² = 4 * (G Mc / c²)^(5/3) * (π f_obs)^(2/3) / (D_c² f_obs)")
    
    # Now compute for all binaries
    print(f"\n{'='*70}")
    print("Computing full population...")
    
    n_binaries = len(population)
    gw_frequencies = np.array([b['f'] for b in population])
    chirp_masses = np.array([b['Mc'] for b in population])
    comoving_dist = np.array([b['D_comov'] for b in population])
    redshift = np.array([b['z'] for b in population])
    
    # Convert to SI
    Mc_SI = chirp_masses * SOLAR_MASS_IN_KG
    D_SI = comoving_dist * MEGAPARSEC_IN_METERS
    f_obs = gw_frequencies / (1.0 + redshift)
    
    # Compute h²
    h_squared = prefactor * Mc_SI**(5./3.) * (np.pi * f_obs)**(2./3.) / (D_SI**2 * f_obs)
    
    # Bin into frequency bins
    f_min = np.min(gw_frequencies)
    f_max = np.max(gw_frequencies)
    
    output_bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_freq_bins + 1)
    h_c_squared_binned = np.zeros(n_freq_bins)
    
    for i in range(n_freq_bins):
        mask = (gw_frequencies >= output_bin_edges[i]) & (gw_frequencies < output_bin_edges[i+1])
        if np.any(mask):
            delta_f = output_bin_edges[i+1] - output_bin_edges[i]
            h_c_squared_binned[i] = np.sum(h_squared[mask]) / delta_f
    
    frequencies = 0.5 * (output_bin_edges[:-1] + output_bin_edges[1:])
    h_c = np.sqrt(h_c_squared_binned)
    
    print(f"Peak h_c: {np.max(h_c):.3e}")
    
    return {
        'frequencies': frequencies,
        'h_c_squared': h_c_squared_binned,
        'h_c': h_c,
        'method': 'Circular'
    }

# ============================================================================
# COMPARISON WITH DEBUGGING
# ============================================================================

def debug_strain_comparison(population, T_obs=30.0, n_freq_bins=50, 
                           debug_binary_idx=0, plot=True):
    """
    Compare both methods with detailed debugging output.
    """
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  DEEP DEBUG: GWB STRAIN CALCULATION COMPARISON".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print(f"\nPopulation size: {len(population)}")
    print(f"Debug binary index: {debug_binary_idx}")
    print(f"Observation time: {T_obs} years")
    
    # Method 1
    result1 = compute_gwb_strain_method1_debug(population, T_obs, n_freq_bins, debug_binary_idx)
    
    # Method 2  
    result2 = compute_gwb_strain_method2_debug(population, T_obs, n_freq_bins, debug_binary_idx)
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\nMethod 1 (GWB_SMBHB):")
    print(f"  Peak h_c: {np.max(result1['h_c']):.6e}")
    print(f"  Frequency range: {result1['frequencies'].min():.3e} - {result1['frequencies'].max():.3e} Hz")
    
    print(f"\nMethod 2 (Circular):")
    print(f"  Peak h_c: {np.max(result2['h_c']):.6e}")
    print(f"  Frequency range: {result2['frequencies'].min():.3e} - {result2['frequencies'].max():.3e} Hz")
    
    # Interpolate and compare
    f_min = max(result1['frequencies'].min(), result2['frequencies'].min())
    f_max = min(result1['frequencies'].max(), result2['frequencies'].max())
    f_common = np.logspace(np.log10(f_min), np.log10(f_max), n_freq_bins)
    
    mask1 = result1['h_c'] > 0
    mask2 = result2['h_c'] > 0
    
    if np.sum(mask1) > 1 and np.sum(mask2) > 1:
        interp1 = interp1d(result1['frequencies'][mask1], result1['h_c'][mask1],
                          kind='linear', bounds_error=False, fill_value=0.0)
        interp2 = interp1d(result2['frequencies'][mask2], result2['h_c'][mask2],
                          kind='linear', bounds_error=False, fill_value=0.0)
        
        h_c_1 = interp1(f_common)
        h_c_2 = interp2(f_common)
        
        mask_both = (h_c_1 > 0) & (h_c_2 > 0)
        
        if np.sum(mask_both) > 0:
            frac_diff = np.abs(h_c_1[mask_both] - h_c_2[mask_both]) / \
                       (0.5 * (h_c_1[mask_both] + h_c_2[mask_both]))
            
            print(f"\nFractional difference statistics:")
            print(f"  Mean: {np.mean(frac_diff)*100:.2f}%")
            print(f"  Median: {np.median(frac_diff)*100:.2f}%")
            print(f"  Max: {np.max(frac_diff)*100:.2f}%")
            print(f"  Min: {np.min(frac_diff)*100:.2f}%")
            
            # Identify where differences are largest
            max_diff_idx = np.argmax(frac_diff)
            print(f"\nLargest difference at:")
            print(f"  Frequency: {f_common[mask_both][max_diff_idx]:.3e} Hz")
            print(f"  Method 1: {h_c_1[mask_both][max_diff_idx]:.3e}")
            print(f"  Method 2: {h_c_2[mask_both][max_diff_idx]:.3e}")
            print(f"  Difference: {frac_diff[max_diff_idx]*100:.2f}%")
    
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        ax = axes[0]
        ax.loglog(result1['frequencies'], result1['h_c'], 'b-', 
                 label='Method 1 (GWB_SMBHB)', linewidth=2)
        ax.loglog(result2['frequencies'], result2['h_c'], 'r--',
                 label='Method 2 (Circular)', linewidth=2, alpha=0.7)
        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel('Characteristic Strain $h_c$', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Strain Comparison (N={len(population)}, T={T_obs} yr)', fontsize=14)
        
        ax = axes[1]
        if np.sum(mask_both) > 0:
            ax.semilogx(f_common[mask_both], frac_diff * 100, 'k-', linewidth=2)
            ax.axhline(10, color='orange', linestyle='--', label='10%', alpha=0.5)
            ax.axhline(50, color='red', linestyle='--', label='50%', alpha=0.5)
            ax.set_xlabel('Frequency [Hz]', fontsize=12)
            ax.set_ylabel('Fractional Difference [%]', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('debug_strain_comparison.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved to: debug_strain_comparison.png")
        plt.show()
    
    return {
        'result1': result1,
        'result2': result2
    }

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    from gwb_strain_cross_validation import create_test_population
    
    print("\nCreating test population...")
    pop = create_test_population(n_binaries=1000, z_max=2.0, random_seed=42)
    
    print("\nRunning deep debug comparison...")
    results = debug_strain_comparison(
        pop, 
        T_obs=15.0,
        n_freq_bins=50,
        debug_binary_idx=0,  # Change this to debug different binaries
        plot=True
    )