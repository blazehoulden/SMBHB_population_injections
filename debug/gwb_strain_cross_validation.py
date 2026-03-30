#!/usr/bin/env python3
"""
Cross-validation script corrected to match YOUR strain calculation formula.
Uses the exact formula from compute_characteristic_strain_squared_circular.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import jn as bessel_jn

# ============================================================================
# CONSTANTS (matching your code)
# ============================================================================

# Physical constants (SI units)
GRAVITATIONAL_CONSTANT = 6.67430e-11        # m^3 kg^-1 s^-2
SPEED_OF_LIGHT_MS = 299792458.0             # m/s
SOLAR_MASS_KG = 1.98847e30                  # kg
MEGAPARSEC_IN_METERS = 3.085677581491367e22 # m
YEAR_IN_SECONDS = 365.25 * 24 * 3600        # s

# For GWB_SMBHB.py compatibility
parsec = 3.08567758e16
Mpc = parsec * 1e6
year = 31557600.0
solar_mass = 1.98855e30

# Cosmological parameters
H0 = 67.3 * 1000 / Mpc  # s^-1
cosmo_par = [H0, 0., 0.315, 0., 0.683]

pi = np.pi

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
    """Compute comoving distance in Mpc"""
    c_SI = SPEED_OF_LIGHT_MS
    c_Mpc = c_SI / MEGAPARSEC_IN_METERS
    
    NN = 500
    D_M = 0.0
    for l in range(NN):
        z_l = z / NN * l
        z_u = z / NN * (l + 1)
        D_M += (E_function(z_l, cosmo_par) + E_function(z_u, cosmo_par)) * (z_u - z_l) / 2.0
    
    return c_Mpc / H0 * D_M

def g_func(n, e):
    """Bessel function combination for harmonic n and eccentricity e"""
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
    """Harmonic at which emission is brightest"""
    c1, c2, c3, c4 = -1.01678, 5.57372, -4.9271, 1.68506
    return 2.0 * (1.0 + c1*e + c2*e**2 + c3*e**3 + c4*e**4) * (1.0 - e**2)**(-1.5)

# ============================================================================
# METHOD 1: GWB_SMBHB.py (Eccentric, harmonics)
# ============================================================================

def compute_gwb_strain_method1(population, T_obs=30.0, n_freq_bins=100):
    """
    Compute GWB strain using GWB_SMBHB.py method.
    Handles eccentric binaries with harmonic decomposition.
    """
    
    # GWB code uses its own unit system
    G_SI = 6.67384e-11
    c_SI = 299792458.0
    G = G_SI / (Mpc**3) * solar_mass
    c = c_SI / Mpc
    G_5_3 = G ** (5./3.)
    c_8 = c ** 8
    
    T_obs_s = T_obs * year
    
    # Setup frequency grid
    min_f = 1.0 / T_obs_s
    max_f = 1e-7
    step_fobs = 1.0 / T_obs_s
    N_bin_f = int((max_f - min_f) / step_fobs) + 1
    
    F_obs = np.linspace(min_f, min_f + N_bin_f * step_fobs, N_bin_f + 1)
    hc_gwb_2_vec = np.zeros(N_bin_f + 1)
    
    # Pre-compute comoving distance lookup
    N_zD = 10000
    min_zD = 0.001
    max_zD = 2.5
    delta_zD = (max_zD - min_zD) / N_zD
    
    zD_array = np.linspace(min_zD, max_zD, N_zD + 1)
    comoving_distance_array = np.array([comoving_distance(z, cosmo_par) for z in zD_array])
    
    # Process each source
    for n_s, binary in enumerate(population):
        Mc = binary.Mc
        z_loc = binary.z
        f = binary.f  # GW frequency
        i_loc = binary.get('iota', 0.0)
        e_loc = binary.get('e', 1e-6)
        
        if e_loc == 0:
            e_loc = 1e-6
        
        Mc_5_3 = Mc ** (5./3.)
        
        # Orbital frequencies
        f_orb = f / 2.0
        f_r = f_orb * (1.0 + z_loc)
        
        # Get comoving distance
        id_z = int((z_loc - min_zD) / delta_zD)
        if id_z >= N_zD:
            id_z = N_zD - 1
        
        xd = (z_loc - zD_array[id_z]) / (zD_array[id_z + 1] - zD_array[id_z])
        D_M_distance = comoving_distance_array[id_z] * (1 - xd) + \
                       comoving_distance_array[id_z + 1] * xd
        
        # Polarization contribution
        a = 1.0 + np.cos(i_loc)**2
        b = -2.0 * np.cos(i_loc)
        MeanAng = np.sqrt(0.5 * (a**2 + b**2))
        
        # Harmonic decomposition
        n_max = int(np.ceil(4.0 * n_peak(e_loc)))
        
        for n in range(1, n_max + 1):
            g_n_e = g_func(n, e_loc)
            if g_n_e == 0:
                continue
            
            f_n = n * f_r
            index = (f_n / (1.0 + z_loc) - min_f) / step_fobs
            
            if index > N_bin_f or index < 0:
                continue
            
            # Amaro-Seoane formula
            h2_n_square = (2.0 * 2.0 * MeanAng**2 * 4.0 * G_5_3**2 * Mc_5_3**2 / 
                          (c_8 * D_M_distance**2) * (2.0 * pi * f_r)**(4./3.) / 
                          (n**2) * g_n_e * f_n / (1.0 + z_loc) / step_fobs)
            
            hc_gwb_2_vec[int(index)] += h2_n_square
    
    frequencies = 0.5 * (F_obs[:-1] + F_obs[1:])
    h_c_squared = hc_gwb_2_vec[:-1]
    h_c = np.sqrt(h_c_squared)
    
    return {
        'frequencies': frequencies,
        'h_c_squared': h_c_squared,
        'h_c': h_c,
        'method': 'GWB_SMBHB (eccentric)'
    }

# ============================================================================
# METHOD 2: YOUR CODE (Circular, orientation-averaged)
# ============================================================================

def compute_gwb_strain_method2(population, T_obs=30.0, n_freq_bins=100, 
                               use_inclination=False):
    """
    Compute GWB strain using YOUR exact formula from compute_characteristic_strain_squared_circular.
    
    Formula: h² = (32/5) * (G*Mc)^(10/3) / (c^8 * D_c²) * (2π * f_rest)^(4/3)
    where f_rest = 0.5 * (1+z) * f_GW is the rest-frame orbital frequency.
    
    Then: h_c = sqrt(h² * f / Δf)
    
    Parameters
    ----------
    population : list of dict
        Each binary must have: 'Mc', 'z', 'f', 'D_comov'
    T_obs : float
        Observation time [years]
    n_freq_bins : int
        Number of frequency bins
    use_inclination : bool
        If True and 'iota' in population, use polarization factor
        
    Returns
    -------
    dict with 'frequencies', 'h_c_squared', 'h_c'
    """
    
    # Extract arrays
    n_binaries = len(population)
    gw_frequencies = np.array([b.f for b in population])
    chirp_masses = np.array([b.Mc for b in population])
    comoving_dist = np.array([b.D_comov for b in population])
    redshift = np.array([b.z for b in population])
    
    # Check for inclination
    if use_inclination and 'iota' in population[0]:
        inclinations = np.array([b['iota'] for b in population])
        has_inclination = True
    else:
        inclinations = None
        has_inclination = False
    
    # ========================================================================
    # STEP 1: Compute h² for each binary (YOUR EXACT FORMULA)
    # ========================================================================
    
    h_squared = np.zeros(n_binaries)
    
    # Constant factor
    if has_inclination:
        const = 2.0 * 2.0 / (SPEED_OF_LIGHT_MS**8)  # With inclination
    else:
        const = 32.0 / (5.0 * SPEED_OF_LIGHT_MS**8)  # Orientation-averaged
    
    for i in range(n_binaries):
        # Rest-frame orbital frequency: f_orb = f_GW/2 in source frame
        f_rest_orbital = 0.5 * (1.0 + redshift[i]) * gw_frequencies[i]
        
        # Convert to SI units
        Mc_SI = chirp_masses[i] * SOLAR_MASS_KG
        D_comov_SI = comoving_dist[i] * MEGAPARSEC_IN_METERS
        
        # h² formula for circular orbits
        if has_inclination:
            # Polarization contribution (matching GWB code)
            i_loc = inclinations[i]
            a = 1.0 + np.cos(i_loc)**2
            b = -2.0 * np.cos(i_loc)
            MeanAng = np.sqrt(0.5 * (a**2 + b**2))
            
            h_squared[i] = const * MeanAng**2 * \
                          (GRAVITATIONAL_CONSTANT * Mc_SI)**(10.0/3.0) / D_comov_SI**2 * \
                          (2.0 * np.pi * f_rest_orbital)**(4.0/3.0)
        else:
            # Orientation-averaged strain
            h_squared[i] = const * \
                          (GRAVITATIONAL_CONSTANT * Mc_SI)**(10.0/3.0) / D_comov_SI**2 * \
                          (2.0 * np.pi * f_rest_orbital)**(4.0/3.0)
    
    # ========================================================================
    # STEP 2: Bin into frequency bins (YOUR EXACT BINNING)
    # ========================================================================
    
    # Use YOUR binning strategy
    f_min = 1.0 / (T_obs * YEAR_IN_SECONDS)
    f_max = 3e-7
    f_step = 1.0 / (T_obs * YEAR_IN_SECONDS)
    N_bin_f = int((f_max - f_min) / f_step) + 1
    
    bin_edges = np.linspace(f_min, f_min + N_bin_f * f_step, N_bin_f + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Assign each binary to a bin
    bin_indices = np.zeros(n_binaries, dtype=int)
    
    for i in range(n_binaries):
        f = gw_frequencies[i]
        
        # Skip if outside frequency range
        if f < bin_edges[0] or f >= bin_edges[-1]:
            bin_indices[i] = -1  # Mark as invalid
            continue
        
        # Find which bin
        for b in range(N_bin_f):
            if bin_edges[b] <= f < bin_edges[b+1]:
                bin_indices[i] = b
                break
    
    # Sum h² contributions in each bin
    h_squared_sum_per_bin = np.zeros(N_bin_f)
    
    for i in range(n_binaries):
        bin_idx = bin_indices[i]
        if bin_idx >= 0:  # Only sum valid bins
            h_squared_sum_per_bin[bin_idx] += h_squared[i]
    
    # ========================================================================
    # STEP 3: Convert to characteristic strain: h_c = sqrt(h² * f / Δf)
    # ========================================================================
    
    h_c_total = np.sqrt(h_squared_sum_per_bin * bin_centres / bin_widths)
    
    # Individual contributions (only for valid bins)
    h_c_individual = np.zeros(n_binaries)
    valid_mask = bin_indices >= 0
    h_c_individual[valid_mask] = np.sqrt(
        h_squared[valid_mask] * bin_centres[bin_indices[valid_mask]] / bin_widths[bin_indices[valid_mask]]
    )
    
    print("maximum bin index:", np.max(bin_indices[valid_mask]) if np.any(valid_mask) else -1, 
          "N_bin_f:", N_bin_f, "max frequency:", np.max(gw_frequencies))

    
    return {
        'frequencies': bin_centres,
        'h_c_squared': h_c_total**2,
        'h_c': h_c_total,
        'h_squared_individual': h_squared,
        'h_c_individual': h_c_individual,
        'bin_indices': bin_indices,
        'method': 'Your code (circular, orientation-averaged)'
    }

# ============================================================================
# COMPARISON
# ============================================================================

def compare_strain_calculations(population, T_obs=30.0, n_freq_bins=100, 
                                plot=True, savefig=None, use_inclination=False):
    """
    Compare two methods of computing GWB strain.
    
    Parameters
    ----------
    population : list of dict
        Must have fields for both methods
    T_obs : float
        Observation time [years]
    n_freq_bins : int
        Number of frequency bins for output
    plot : bool
        Create comparison plot
    savefig : str, optional
        Save figure to this path
    use_inclination : bool
        Whether to use inclination angles in Method 2
        
    Returns
    -------
    dict with comparison results
    """
    
    print("="*70)
    print("GWB STRAIN CROSS-VALIDATION (CORRECTED)")
    print("="*70)
    print(f"Population size: {len(population)}")
    print(f"Observation time: {T_obs} years")
    print(f"Using inclination: {use_inclination}")
    print()
    
    # Method 1: GWB_SMBHB.py
    print("Computing strain using Method 1 (GWB_SMBHB.py)...")
    result1 = compute_gwb_strain_method1(population, T_obs, n_freq_bins)
    print(f"  Peak h_c: {np.max(result1['h_c']):.3e}")
    print(f"  Frequency range: {result1['frequencies'].min():.3e} - {result1['frequencies'].max():.3e} Hz")
    print()
    
    # Method 2: Your code
    print("Computing strain using Method 2 (Your code)...")
    result2 = compute_gwb_strain_method2(population, T_obs, n_freq_bins, use_inclination)
    print(f"  Peak h_c: {np.max(result2['h_c']):.3e}")
    print(f"  Frequency range: {result2['frequencies'].min():.3e} - {result2['frequencies'].max():.3e} Hz")
    print()
    
    # Compare
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
        
        h_c_1_interp = interp1(f_common)
        h_c_2_interp = interp2(f_common)
        
        mask_both = (h_c_1_interp > 0) & (h_c_2_interp > 0)
        
        if np.sum(mask_both) > 0:
            frac_diff = np.abs(h_c_1_interp[mask_both] - h_c_2_interp[mask_both]) / \
                       (0.5 * (h_c_1_interp[mask_both] + h_c_2_interp[mask_both]))
            
            print("COMPARISON STATISTICS:")
            print(f"  Mean fractional difference: {np.mean(frac_diff)*100:.2f}%")
            print(f"  Median fractional difference: {np.median(frac_diff)*100:.2f}%")
            print(f"  Max fractional difference: {np.max(frac_diff)*100:.2f}%")
            print()
            
            if np.mean(frac_diff) < 0.1:
                print("✓ Methods agree within 10% (EXCELLENT)")
            elif np.mean(frac_diff) < 0.3:
                print("⚠ Methods differ by 10-30% (ACCEPTABLE - different assumptions)")
            elif np.mean(frac_diff) < 0.5:
                print("⚠ Methods differ by 30-50% (CHECK - may be orientation vs eccentric)")
            else:
                print("❌ Methods differ by >50% (PROBLEM)")
        else:
            print("⚠ No overlapping non-zero strain values")
            frac_diff = None
    else:
        print("⚠ Insufficient data for comparison")
        frac_diff = None
    
    print("="*70)
    
    # Plotting
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        ax = axes[0]
        ax.loglog(result1['frequencies'], result1['h_c'], 'b-', 
                 label=result1['method'], linewidth=2)
        ax.loglog(result2['frequencies'], result2['h_c'], 'r--', 
                 label=result2['method'], linewidth=2)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Characteristic Strain $h_c$')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'GWB Strain Comparison (N={len(population)}, T={T_obs} yr)')
        
        ax = axes[1]
        if frac_diff is not None:
            ax.semilogx(f_common[mask_both], frac_diff * 100, 'k-', linewidth=2)
            ax.axhline(10, color='orange', linestyle='--', label='10%', alpha=0.5)
            ax.axhline(30, color='red', linestyle='--', label='30%', alpha=0.5)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Fractional Difference [%]')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Fractional Difference: |Method1 - Method2| / Average')
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to: {savefig}")
        
        plt.show()
    
    return {
        'method1': result1,
        'method2': result2,
        'f_common': f_common,
        'h_c_1_interp': h_c_1_interp if frac_diff is not None else None,
        'h_c_2_interp': h_c_2_interp if frac_diff is not None else None,
        'fractional_difference': frac_diff,
        'frequencies_overlap': f_common[mask_both] if frac_diff is not None else None
    }

# ============================================================================
# TEST POPULATION
# ============================================================================

def create_test_population(n_binaries=1000, z_max=2.0, random_seed=42):
    """Create test population compatible with both methods."""
    
    np.random.seed(random_seed)
    
    population = []
    
    # Sample frequencies
    f_min = 1e-9
    f_max = 1e-7
    u = np.random.uniform(0, 1, n_binaries)
    frequencies = (f_min**(-8/3) + u * (f_max**(-8/3) - f_min**(-8/3)))**(-3/8)
    
    # Sample redshifts
    z_samples = np.random.uniform(0.01, z_max, n_binaries)
    
    # Sample masses
    alpha = 1.21
    m_min = 1e7
    m_max = 1e10
    u_m = np.random.uniform(0, 1, n_binaries)
    masses = (m_min**(1-alpha) + u_m * (m_max**(1-alpha) - m_min**(1-alpha)))**(1/(1-alpha))
    
    # Chirp masses (q=0.25)
    q = 0.25
    chirp_masses = masses * (q / (1+q)**2)**(3/5)
    
    # Sky positions
    inclinations = np.random.uniform(0, np.pi, n_binaries)
    
    for i in range(n_binaries):
        z = z_samples[i]
        D_comov = comoving_distance(z, cosmo_par)
        
        binary = {
            'Mc': chirp_masses[i],
            'Mtot': masses[i] * (1 + q),
            'f': frequencies[i],
            'z': z,
            'D_comov': D_comov,
            'iota': inclinations[i],
            'e': 1e-6,
        }
        
        population.append(binary)
    
    return population

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Creating test population...")
    test_pop = create_test_population(n_binaries=5000, z_max=2.0)
    
    print("\nRunning cross-validation (orientation-averaged)...\n")
    results = compare_strain_calculations(
        test_pop, 
        T_obs=15.0,
        n_freq_bins=50,
        plot=True,
        savefig='gwb_strain_comparison_corrected.png',
        use_inclination=False
    )
    
    print("\n" + "="*70)
    print("Testing with inclination angles...")
    print("="*70 + "\n")
    
    results_incl = compare_strain_calculations(
        test_pop,
        T_obs=15.0,
        n_freq_bins=50,
        plot=False,
        use_inclination=True
    )