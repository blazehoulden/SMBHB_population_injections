"""
Supermassive Black Hole Binary (SMBHB) Population Synthesis

This module generates synthetic populations of SMBHBs for gravitational wave
background studies. It samples binary properties (masses, frequencies, distances)
from astrophysically-motivated distributions and computes their characteristic
gravitational wave strain.

Author: Blaze Houlden
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numba as nb
from numba import njit, prange

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Cosmology (ΛCDM)
HUBBLE_CONSTANT_H = 0.67                   # Dimensionless Hubble parameter
OMEGA_MATTER = 0.3                         # Matter density parameter
OMEGA_LAMBDA = 0.7                         # Dark energy density parameter
H0_KMS_MPC = 100 * HUBBLE_CONSTANT_H       # Hubble constant [km/s/Mpc]

# Physical constants
SPEED_OF_LIGHT_KMS = 2.9979e5              # Speed of light [km/s]
SPEED_OF_LIGHT_MS = SPEED_OF_LIGHT_KMS * 1e3  # Speed of light [m/s]
GRAVITATIONAL_CONSTANT = 6.67e-11          # Newton's constant [m^3 kg^-1 s^-2]

# Length scales
PARSEC_IN_METERS = 3.086e16                # 1 parsec [m]
MEGAPARSEC_IN_METERS = 1e6 * PARSEC_IN_METERS  # 1 Megaparsec [m]

# Mass and time scales
SOLAR_MASS_KG = 1.989e30                   # Solar mass [kg]
YEAR_IN_SECONDS = 86400 * 365.25           # 1 year [s]
FREQUENCY_PER_YEAR = 1.0 / YEAR_IN_SECONDS # 1/year [Hz]

# ============================================================================
# COSMOLOGICAL FUNCTIONS
# ============================================================================

def hubble_parameter(redshift):
    """
    Hubble parameter H(z) for flat ΛCDM cosmology.
    
    Parameters
    ----------
    redshift : float or array
        Cosmological redshift
        
    Returns
    -------
    H_z : float or array
        Hubble parameter at redshift z [km/s/Mpc]
        
    Notes
    -----
    For flat ΛCDM: H(z) = H0 * sqrt(Ωm(1+z)^3 + ΩΛ)
    """
    return H0_KMS_MPC * np.sqrt(
        OMEGA_MATTER * (1 + redshift)**3 + OMEGA_LAMBDA
    )


def build_comoving_distance_interpolator(z_max=20.0, n_points=2000):
    """
    Build interpolator for comoving distance as a function of redshift.
    
    Parameters
    ----------
    z_max : float, optional
        Maximum redshift for interpolation grid (default: 20.0)
    n_points : int, optional
        Number of grid points for interpolation (default: 2000)
        
    Returns
    -------
    comoving_distance_fn : callable
        Function that takes redshift(s) and returns comoving distance(s) [Mpc]
        
    Notes
    -----
    Comoving distance: χ(z) = c/H0 * ∫[0 to z] dz'/E(z')
    where E(z) = H(z)/H0
    """
    z_grid = np.linspace(0, z_max, n_points)
    
    # Compute comoving distance at each redshift via integration
    chi_grid = np.array([
        quad(lambda zp: SPEED_OF_LIGHT_KMS / hubble_parameter(zp), 0, zi)[0] 
        for zi in z_grid
    ])
    
    # Create interpolator
    interpolator = interp1d(
        z_grid, chi_grid, 
        kind="cubic", 
        fill_value="extrapolate"
    )
    
    def comoving_distance(z):
        """Comoving distance [Mpc] as function of redshift."""
        z = np.atleast_1d(z)
        return interpolator(z)
    
    return comoving_distance


def build_inverse_comoving_to_redshift(comoving_dist_fn, z_max=5.0, n_points=2000):
    """
    Build interpolator for the inverse: comoving distance → redshift.
    
    This is used for efficient sampling of distances in the population synthesis.
    
    Parameters
    ----------
    comoving_dist_fn : callable
        Function that computes comoving distance from redshift
    z_max : float, optional
        Maximum redshift (default: 5.0)
    n_points : int, optional
        Number of grid points (default: 2000)
        
    Returns
    -------
    inverse_interpolator : callable
        Function: comoving distance [Mpc] → redshift
    """
    # Build forward lookup table
    z_grid = np.linspace(0, z_max, n_points)
    chi_grid = comoving_dist_fn(z_grid)
    
    # Ensure monotonicity (required for inversion)
    assert np.all(np.diff(chi_grid) > 0), "Comoving distance must be monotonic"
    
    # Create inverse interpolator
    inverse_interp = interp1d(
        chi_grid, z_grid,
        kind='linear',
        fill_value=(0, z_max),
        bounds_error=False
    )
    
    return inverse_interp


# Initialize global comoving distance functions
COMOVING_DISTANCE_FN = build_comoving_distance_interpolator(z_max=5.0)
INVERSE_COMOVING_TO_Z = build_inverse_comoving_to_redshift(
    COMOVING_DISTANCE_FN, 
    z_max=2.0
)

# Build Numba-compatible lookup tables for fast inverse transform
Z_MAX_NUMBA = 5.0
N_POINTS_NUMBA = 2000
Z_GRID_NUMBA = np.linspace(0, Z_MAX_NUMBA, N_POINTS_NUMBA)
CHI_GRID_NUMBA = COMOVING_DISTANCE_FN(Z_GRID_NUMBA)


@njit
def inverse_comoving_to_redshift_numba(comoving_distances):
    """
    Numba-accelerated inverse transform: comoving distance → redshift.
    
    Uses binary search on precomputed lookup table for speed.
    
    Parameters
    ----------
    comoving_distances : ndarray
        Array of comoving distances [Mpc]
        
    Returns
    -------
    redshifts : ndarray
        Corresponding redshifts
    """
    n = len(comoving_distances)
    redshifts = np.empty(n)
    
    for i in range(n):
        chi = comoving_distances[i]
        
        # Binary search in chi_grid
        left = 0
        right = len(CHI_GRID_NUMBA) - 1
        
        while right - left > 1:
            mid = (left + right) // 2
            if CHI_GRID_NUMBA[mid] <= chi:
                left = mid
            else:
                right = mid
        
        # Linear interpolation between grid points
        slope = (Z_GRID_NUMBA[right] - Z_GRID_NUMBA[left]) / \
                (CHI_GRID_NUMBA[right] - CHI_GRID_NUMBA[left])
        redshifts[i] = Z_GRID_NUMBA[left] + slope * (chi - CHI_GRID_NUMBA[left])
    
    return redshifts


# ============================================================================
# POPULATION SAMPLING FUNCTIONS
# ============================================================================

@nb.njit(parallel=True, fastmath=True)
def sample_gw_frequencies(n_binaries, random_seeds, 
                         t_obs_max=30*YEAR_IN_SECONDS, 
                         t_obs_min=YEAR_IN_SECONDS/12):
    """
    Sample gravitational wave frequencies from merger-time-weighted distribution.
    
    p(f) ∝ f^(-11/3), then through inverse transform sampling we get for a uniform distribution ∝ f^(-8/3).
    
    We sample uniformly in x = f^(-8/3) space for efficiency.
    
    Parameters
    ----------
    n_binaries : int
        Number of binaries to sample
    random_seeds : ndarray
        Random seeds for parallel threads
    t_obs_max : float, optional
        Maximum observation time [s] (default: 30 years)
    t_obs_min : float, optional
        Minimum observation time [s] (default: 1 month)
        
    Returns
    -------
    frequencies : ndarray
        GW frequencies [Hz], sampled from p(f) ∝ f^(-11/3)
        
    Notes
    -----
    - Lower frequencies are more probable (more time in band)
    - This assumes the mission observes the full inspiral
    - For mission duration T: f_min = 1/T, f_max = 1/t_min
    """
    f_min = 1.0 / t_obs_max
    f_max = 1.0 / t_obs_min
    
    # Transform to uniform variable: u ~ Uniform[x_min, x_max]
    # where x = f^(-8/3)
    x_min = f_min**(-8.0/3.0)
    x_max = f_max**(-8.0/3.0)
    x_range = x_max - x_min
    
    frequencies = np.empty(n_binaries)
    n_threads = len(random_seeds)
    chunk_size = n_binaries // n_threads
    
    # Parallel sampling across threads
    for thread_id in nb.prange(n_threads):
        np.random.seed(random_seeds[thread_id])
        
        start_idx = thread_id * chunk_size
        end_idx = n_binaries if thread_id == n_threads - 1 else (thread_id + 1) * chunk_size
        
        for i in range(start_idx, end_idx):
            u = np.random.rand()
            x = x_min + x_range * u
            frequencies[i] = x**(-3.0/8.0)  # Inverse transform: f = x^(-3/8)
    
    return frequencies


@njit
def sample_comoving_distances(n_binaries, distance_max, distance_min=1.0):
    """
    Sample comoving distances uniformly in volume.
    
    Also computes redshift and luminosity distance for each binary.
    
    Parameters
    ----------
    n_binaries : int
        Number of binaries to sample
    distance_min : float, optional
        Minimum comoving distance [Mpc] (default: 1.0)
    distance_max : float
        Maximum comoving distance [Mpc]
        
    Returns
    -------
    comoving_dist : ndarray
        Comoving distances [Mpc]
    luminosity_dist : ndarray
        Luminosity distances [Mpc]
    redshift : ndarray
        Cosmological redshifts
        
    Notes
    -----
    Number density n(D) ∝ D² (uniform in comoving volume)
    We sample uniformly in D³ space, then take cube root
    
    Luminosity distance: D_L = (1+z) * D_comoving
    """
    # Sample uniformly in volume: V ∝ D³
    vol_min = distance_min**3
    vol_max = distance_max**3
    
    u = np.random.rand(n_binaries)
    volumes = vol_min + (vol_max - vol_min) * u
    comoving_dist = volumes**(1.0/3.0)
    
    # Convert comoving distance to redshift via lookup table
    redshift = inverse_comoving_to_redshift_numba(comoving_dist)
    
    # Luminosity distance includes redshift factor
    luminosity_dist = (1.0 + redshift) * comoving_dist
    
    return comoving_dist, luminosity_dist, redshift


# ============================================================================
# MASS SAMPLING FUNCTIONS
# ============================================================================

def sample_masses_power_law(n_binaries, redshift_array,
                            alpha_0=1.21, alpha_z=0.0,
                            mass_min=1e7, mass_max=1e11):
    """
    Sample black hole masses from a power law distribution.
    
    Parameters
    ----------
    n_binaries : int
        Number of masses to sample
    redshift_array : ndarray
        Redshift for each binary (allows redshift evolution)
    alpha_0 : float, optional
        Power law index at z=0 (default: 1.21)
    alpha_z : float, optional
        Redshift evolution of index: α(z) = α₀ + α_z * z (default: 0.0)
    mass_min : float, optional
        Minimum mass [M_sun] (default: 10^7)
    mass_max : float, optional
        Maximum mass [M_sun] (default: 10^11)
        
    Returns
    -------
    masses : ndarray
        Black hole masses [M_sun], sampled from p(M) ∝ M^(-α)
        
    Notes
    -----
    Uses inverse transform sampling:
    If p(M) ∝ M^(-α), then CDF ∝ M^(1-α)
    Sample u ~ Uniform[0,1], invert CDF to get M(u)
    """
    # Redshift-dependent power law index
    alpha = alpha_0 + alpha_z * redshift_array
    
    # CDF limits: F(M) ∝ M^(1-α)
    cdf_max = mass_max**(1.0 - alpha)
    cdf_min = mass_min**(1.0 - alpha)
    cdf_range = cdf_max - cdf_min
    
    # Inverse transform sampling
    u = np.random.rand(n_binaries)
    cdf_values = cdf_min + cdf_range * u
    masses = cdf_values**(1.0 / (1.0 - alpha))
    
    return masses


# --- Helper functions for exponentially-damped mass function ---

@nb.njit
def exponentially_damped_pdf(mass, alpha, mass_cutoff):
    """
    Exponentially damped power law: p(M) ∝ M^(-α) * exp(-M/M_c)
    
    This suppresses very massive black holes.
    """
    return mass**(-alpha) * np.exp(-mass / mass_cutoff)


@nb.njit
def power_law_pdf(mass, alpha):
    """Simple power law: p(M) ∝ M^(-α)"""
    return mass**(-alpha)


def build_cdf_exponential_damping(mass_min, mass_max, alpha, mass_cutoff, 
                                  n_grid=20000):
    """
    Build CDF for exponentially damped mass function via numerical integration.
    
    Parameters
    ----------
    mass_min, mass_max : float
        Mass range [M_sun]
    alpha : float
        Power law index
    mass_cutoff : float
        Exponential cutoff mass [M_sun]
    n_grid : int, optional
        Number of grid points for integration (default: 20000)
        
    Returns
    -------
    mass_grid : ndarray
        Mass values
    cdf_grid : ndarray
        Cumulative distribution function (normalized to 1)
    """
    mass_grid = np.linspace(mass_min, mass_max, n_grid)
    pdf_grid = exponentially_damped_pdf(mass_grid, alpha, mass_cutoff)
    
    # Trapezoidal integration
    cdf_grid = np.empty(n_grid)
    cdf_grid[0] = 0.0
    
    for i in range(1, n_grid):
        delta_m = mass_grid[i] - mass_grid[i-1]
        avg_pdf = 0.5 * (pdf_grid[i] + pdf_grid[i-1])
        cdf_grid[i] = cdf_grid[i-1] + avg_pdf * delta_m
    
    # Normalize
    cdf_grid /= cdf_grid[-1]
    
    return mass_grid, cdf_grid


def build_cdf_power_law(mass_min, mass_max, alpha, n_grid=20000):
    """Build CDF for pure power law (for comparison/testing)."""
    mass_grid = np.linspace(mass_min, mass_max, n_grid)
    pdf_grid = power_law_pdf(mass_grid, alpha)
    
    cdf_grid = np.empty(n_grid)
    cdf_grid[0] = 0.0
    
    for i in range(1, n_grid):
        delta_m = mass_grid[i] - mass_grid[i-1]
        avg_pdf = 0.5 * (pdf_grid[i] + pdf_grid[i-1])
        cdf_grid[i] = cdf_grid[i-1] + avg_pdf * delta_m
    
    cdf_grid /= cdf_grid[-1]
    
    return mass_grid, cdf_grid


@nb.njit(parallel=True)
def sample_from_precomputed_cdf(n_samples, mass_grid, cdf_grid):
    """
    Sample from arbitrary distribution using precomputed CDF.
    
    Uses inverse transform sampling with binary search for efficiency.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to draw
    mass_grid : ndarray
        Mass values corresponding to CDF
    cdf_grid : ndarray
        Cumulative distribution values (must be monotonic)
        
    Returns
    -------
    samples : ndarray
        Sampled mass values
    """
    samples = np.empty(n_samples)
    
    for i in nb.prange(n_samples):
        u = np.random.random()
        
        # Binary search in CDF to find u
        low = 0
        high = len(cdf_grid) - 1
        
        while high - low > 1:
            mid = (low + high) // 2
            if cdf_grid[mid] < u:
                low = mid
            else:
                high = mid
        
        # Linear interpolation between grid points
        cdf_low, cdf_high = cdf_grid[low], cdf_grid[high]
        mass_low, mass_high = mass_grid[low], mass_grid[high]
        
        if cdf_high - cdf_low > 0:
            weight = (u - cdf_low) / (cdf_high - cdf_low)
            samples[i] = mass_low + weight * (mass_high - mass_low)
        else:
            samples[i] = mass_low
    
    return samples


def sample_masses_exponential_damping(n_binaries, redshift_array,
                                     mass_cutoff_0=1e9, mass_cutoff_z=0.0,
                                     alpha_0=1.21, alpha_z=0.0,
                                     mass_min=1e7, mass_max=1e11,
                                     use_pure_power_law=False):
    """
    Sample masses from exponentially damped power law.
    
    Distribution: p(M) ∝ M^(-α) * exp(-M/M_c)
    
    Parameters
    ----------
    n_binaries : int
        Number of masses to sample
    redshift_array : ndarray
        Redshift for each binary
    mass_cutoff_0 : float, optional
        Exponential cutoff mass at z=0 [M_sun] (default: 10^9)
    mass_cutoff_z : float, optional
        Redshift evolution: M_c(z) = M_c,0 + M_c,z * z (default: 0.0)
    alpha_0 : float, optional
        Power law index at z=0 (default: 1.21)
    alpha_z : float, optional
        Redshift evolution of α (default: 0.0)
    mass_min, mass_max : float, optional
        Mass range [M_sun]
    use_pure_power_law : bool, optional
        If True, ignore exponential damping (default: False)
        
    Returns
    -------
    masses : ndarray
        Sampled black hole masses [M_sun]
    """
    # For simplicity, use z=0 parameters to build CDF
    # (Could extend to redshift-dependent sampling if needed)
    alpha = alpha_0 + alpha_z * 0
    mass_cutoff = mass_cutoff_0 + mass_cutoff_z * 0
    
    if use_pure_power_law:
        mass_grid, cdf_grid = build_cdf_power_law(mass_min, mass_max, alpha)
    else:
        mass_grid, cdf_grid = build_cdf_exponential_damping(
            mass_min, mass_max, alpha, mass_cutoff
        )
    
    # Sample all binaries from this CDF
    masses = sample_from_precomputed_cdf(n_binaries, mass_grid, cdf_grid)
    
    return masses


# ============================================================================
# MASS RATIO AND CHIRP MASS CALCULATION
# ============================================================================

@nb.njit(parallel=True)
def sample_mass_ratios_and_compute_chirp_mass(n_binaries, primary_masses, 
                                              use_equal_mass=False):
    """
    Sample mass ratios and compute chirp masses.
    
    Parameters
    ----------
    n_binaries : int
        Number of binaries
    primary_masses : ndarray
        Primary (more massive) black hole masses [M_sun]
    use_equal_mass : bool, optional
        If True, set q=1 for all binaries (default: False)
        
    Returns
    -------
    total_masses : ndarray
        Total masses M = M1 + M2 [M_sun]
    chirp_masses : ndarray
        Chirp masses Mc = (M1*M2)^(3/5) / (M1+M2)^(1/5) [M_sun]
        
    Notes
    -----
    Mass ratio q = M2/M1 where M1 ≥ M2
    
    If use_equal_mass=False:
        Sample q uniformly from [0, 1]
        
    Chirp mass is the combination that enters GW frequency evolution:
        Mc = (M1 * M2)^(3/5) / (M1 + M2)^(1/5)
    """
    total_masses = np.empty(n_binaries)
    chirp_masses = np.empty(n_binaries)
    
    for i in nb.prange(n_binaries):
        if use_equal_mass:
            q = 1.0
        else:
            q = np.random.rand()  # Uniform in [0, 1]
        
        M1 = primary_masses[i]
        M2 = q * M1
        
        total_masses[i] = M1 + M2
        
        # Chirp mass: Mc = (M1*M2)^(3/5) / (M1+M2)^(1/5)
        chirp_masses[i] = (M1 * M2)**(3.0/5.0) / (M1 + M2)**(1.0/5.0)
    
    return total_masses, chirp_masses


# ============================================================================
# CHARACTERISTIC STRAIN CALCULATION
# ============================================================================

@nb.njit(parallel=True)
def compute_characteristic_strain_squared_circular(
    gw_frequencies, chirp_masses, comoving_distances, redshifts, inclination_angle=None
):
    """
    Compute h² for circular binaries (RMS strain, orientation-averaged).
    
    For a circular binary at frequency f and chirp mass Mc:
    
        h² = (32/5) * (G*Mc)^(10/3) / (c^8 * D²) * (2π * f_rest)^(4/3)
    
    where f_rest = (1+z) * f_obs / 2 is the rest-frame orbital frequency.
    
    Parameters
    ----------
    gw_frequencies : ndarray
        Observed GW frequencies [Hz]
    chirp_masses : ndarray
        Chirp masses [M_sun]
    comoving_distances : ndarray
        Comoving distances [Mpc]
    redshifts : ndarray
        Cosmological redshifts
    inclination_angle : ndarray or None, optional
        
    Returns
    -------
    h_squared : ndarray
        Squared characteristic strain h² [dimensionless]
        
    Notes
    -----
    This is the RMS strain averaged over all orientations:
        <F²> = 4/5  (orientation average)
    
    The factor 32/5 includes this averaging and other numerical constants
    from Peters & Mathews (1963).
    """
    n = gw_frequencies.size
    h_squared = np.empty(n, dtype=np.float64)
    
    # Constant factor: 32/(5*c^8)
    if inclination_angle is None:
        const = 32.0 / (5.0 * SPEED_OF_LIGHT_MS**8)
    else:
        const = 2.0 * 2.0 / (SPEED_OF_LIGHT_MS**8)
    
    for i in nb.prange(n):
        # Rest-frame orbital frequency: f_orb = f_GW/2 in source frame
        f_rest_orbital = 0.5 * (1.0 + redshifts[i]) * gw_frequencies[i]
        
        # Convert to SI units
        Mc_SI = chirp_masses[i] * SOLAR_MASS_KG
        D_SI = comoving_distances[i] * MEGAPARSEC_IN_METERS
        
        # h² formula for circular orbits
        if inclination_angle is None:
            # Orientation-averaged strain
            h_squared[i] = const * \
                        (GRAVITATIONAL_CONSTANT * Mc_SI)**(10.0/3.0) / D_SI**2 * \
                        (2.0 * np.pi * f_rest_orbital)**(4.0/3.0)
        else: 
            # Polarization contribution
            i_loc = inclination_angle[i]
            a = 1.0 + np.cos(i_loc)**2
            b = -2.0 * np.cos(i_loc)
            MeanAng = np.sqrt(0.5 * (a**2 + b**2))
            h_squared[i] = const * MeanAng**2 \
                        (GRAVITATIONAL_CONSTANT * Mc_SI)**(10.0/3.0) / D_SI**2 * \
                        (2.0 * np.pi * f_rest_orbital)**(4.0/3.0) 
    
    return h_squared


@njit
def bin_characteristic_strain(gw_frequencies, h_squared, n_freq_bins, T_obs=15):
    """
    Bin individual strain contributions to compute population spectrum.
    
    The characteristic strain spectrum is defined as:
        h_c(f) = sqrt(h² * f / Δf)
    
    For a population, we sum h² values in each frequency bin, then convert.
    
    Parameters
    ----------
    gw_frequencies : ndarray
        GW frequencies [Hz] for each binary
    h_squared : ndarray
        Squared strain h² for each binary
    n_freq_bins : int
        Number of logarithmically-spaced frequency bins
    T_obs : float, optional
        Observation time [years] (default: 15)
        
    Returns
    -------
    bin_centres : ndarray
        Center frequency of each bin [Hz]
    h_c_total : ndarray
        Total characteristic strain in each bin
    h_c_individual : ndarray
        Individual contribution of each binary to h_c
        
    Notes
    -----
    We use logarithmic binning since GW frequencies span many decades.
    
    The total h_c in a bin is:
        h_c,total = sqrt( Σ h² * f / Δf )
    
    where the sum is over all binaries in that bin.
    """
    f_min = np.min(gw_frequencies)
    f_max = np.max(gw_frequencies)

    f_step = 1.0 / (T_obs * YEAR_IN_SECONDS)
    N_bin_f = int((f_max - f_min) / f_step) + 1
    
    bin_edges = np.linspace(f_min, f_min + N_bin_f * f_step, N_bin_f + 1)
    
    # # Logarithmically spaced bin edges
    # bin_edges = np.logspace(
    #     np.log10(f_min), 
    #     np.log10(f_max), 
    #     n_freq_bins + 1
    # )
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Assign each binary to a bin
    bin_indices = np.empty(gw_frequencies.size, dtype=np.int64)
    
    for i in range(gw_frequencies.size):
        f = gw_frequencies[i]
        
        # Find which bin this frequency belongs to
        for b in range(n_freq_bins):
            if bin_edges[b] <= f < bin_edges[b+1]:
                bin_indices[i] = b
                break
        else:
            # If frequency is at upper edge, assign to last bin
            bin_indices[i] = n_freq_bins - 1
    
    # Sum h² contributions in each bin
    h_squared_sum_per_bin = np.zeros(n_freq_bins, dtype=np.float64)
    
    for i in range(gw_frequencies.size):
        bin_idx = bin_indices[i]
        h_squared_sum_per_bin[bin_idx] += h_squared[i]
    
    # Convert to characteristic strain: h_c = sqrt(h² * f / Δf)
    h_c_total = np.sqrt(
        h_squared_sum_per_bin * bin_centres / bin_widths
    )
    
    # Individual contributions (for diagnostics)
    h_c_individual = np.sqrt(
        h_squared * bin_centres[bin_indices] / bin_widths[bin_indices]
    )
    
    return bin_centres, h_c_total, h_c_individual


# ============================================================================
# MAIN POPULATION GENERATION FUNCTION
# ============================================================================

def generate_smbhb_population(
    n_binaries,
    z_max=2.0,
    mass_distribution='power_law',
    alpha_0=1.21,
    alpha_z=0.0,
    mass_min=1e7,
    mass_max=1e11,
    mass_cutoff_0=1e9,
    mass_cutoff_z=0.0,
    compute_strain=False,
    n_freq_bins=50,
    T_obs=15,
    random_seed=None
):
    """
    Generate a synthetic population of supermassive black hole binaries.
    
    This function samples binary properties from astrophysically-motivated
    distributions and optionally computes their gravitational wave strain.
    
    Parameters
    ----------
    n_binaries : int
        Number of SMBHBs to generate
    z_max : float, optional
        Maximum redshift for population (default: 2.0)
    mass_distribution : str, optional
        Mass distribution model:
        - 'power_law': Simple power law M^(-α)
        - 'exponential_damping': M^(-α) * exp(-M/M_c)
        (default: 'power_law')
    alpha_0 : float, optional
        Power law index at z=0 (default: 1.21)
    alpha_z : float, optional
        Redshift evolution of power law (default: 0.0)
    mass_min : float, optional
        Minimum black hole mass [M_sun] (default: 10^7)
    mass_max : float, optional
        Maximum black hole mass [M_sun] (default: 10^11)
    mass_cutoff_0 : float, optional
        Exponential cutoff mass at z=0 [M_sun] (default: 10^9)
        Only used if mass_distribution='exponential_damping'
    mass_cutoff_z : float, optional
        Redshift evolution of cutoff mass (default: 0.0)
    compute_strain : bool, optional
        If True, compute characteristic strain spectrum (default: False)
    n_freq_bins : int, optional
        Number of frequency bins for strain calculation (default: 50)
    random_seed : int, optional
        Random seed for reproducibility (default: None)
        
    Returns
    -------
    population : list of dict
        List of binary parameter dictionaries, each containing:
        - 'Mc': chirp mass [M_sun]
        - 'Mtot': total mass [M_sun]
        - 'f': GW frequency [Hz]
        - 'D_comov': comoving distance [Mpc]
        - 'z': redshift
        - 'ra': right ascension [radians]
        - 'dec': declination [radians]
        - 'psi': polarization angle [radians]
        - 'iota': inclination angle [radians]
        - 'phi0': initial GW phase [radians]
        
        If compute_strain=True, also includes:
        - 'h_square': squared strain h²
        - 'h_c_contrib': individual contribution to h_c
        - 'freq_bin': frequency bin assignment
        
    strain_data : dict (only if compute_strain=True)
        Contains:
        - 'bin_centres': frequency bin centers [Hz]
        - 'h_c_total': total characteristic strain per bin
        - 'h_square_individual': h² for each binary
        - 'bin_assignment': frequency bin for each binary
        - 'h_c_individual': h_c contribution for each binary
        - 'bin_edges': frequency bin edges [Hz]
        
    Examples
    --------
    Generate a simple population:
    
    >>> pop = generate_smbhb_population(
    ...     n_binaries=10000,
    ...     z_max=1.0,
    ...     mass_distribution='power_law'
    ... )
    
    Generate population with strain calculation:
    
    >>> pop, strain = generate_smbhb_population(
    ...     n_binaries=10000,
    ...     compute_strain=True,
    ...     n_freq_bins=100
    ... )
    >>> print(f"Peak h_c: {strain['h_c_total'].max():.2e}")
    
    Notes
    -----
    The population synthesis includes:
    1. Frequency sampling weighted by time-in-band (f^(-11/3))
    2. Distance sampling uniform in comoving volume
    3. Mass sampling from specified distribution
    4. Random sky positions and orientations
    5. Optional strain calculation for circular orbits
    
    For references on the astrophysical models, see:
    - Sesana et al. (2008) for mass functions
    - Ravi et al. (2014) for GW background modeling
    """
    
    # Initialize random number generator
    if random_seed is not None:
        np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)
    
    # ========================================================================
    # STEP 1: Sample frequencies
    # ========================================================================
    
    # Generate random seeds for parallel frequency sampling
    n_threads = nb.get_num_threads()
    thread_seeds = rng.integers(0, 2**32 - 1, size=n_threads)
    
    gw_frequencies = sample_gw_frequencies(n_binaries, thread_seeds)
    
    # ========================================================================
    # STEP 2: Sample distances and redshifts
    # ========================================================================
    
    distance_max = COMOVING_DISTANCE_FN(z_max)
    
    comoving_dist, luminosity_dist, redshift = sample_comoving_distances(
        n_binaries, 
        distance_max
    )
    
    # ========================================================================
    # STEP 3: Sample masses
    # ========================================================================
    
    if mass_distribution == 'exponential_damping':
        primary_masses = sample_masses_exponential_damping(
            n_binaries, redshift,
            mass_cutoff_0=mass_cutoff_0,
            mass_cutoff_z=mass_cutoff_z,
            alpha_0=alpha_0,
            alpha_z=alpha_z,
            mass_min=mass_min,
            mass_max=mass_max,
            use_pure_power_law=False
        )
    else:  # 'power_law'
        primary_masses = sample_masses_power_law(
            n_binaries, redshift,
            alpha_0=alpha_0,
            alpha_z=alpha_z,
            mass_min=mass_min,
            mass_max=mass_max
        )
    
    # ========================================================================
    # STEP 4: Sample mass ratios and compute chirp masses
    # ========================================================================
    
    total_masses, chirp_masses = sample_mass_ratios_and_compute_chirp_mass(
        n_binaries, 
        primary_masses
    )
    
    # ========================================================================
    # STEP 5: Sample sky positions and orientations
    # ========================================================================
    
    # Right ascension: uniform on [0, 2π]
    right_ascension = rng.uniform(0, 2*np.pi, size=n_binaries)
    
    # Declination: uniform on sphere → sample sin(dec) uniformly
    declination = np.arcsin(rng.uniform(-1, 1, size=n_binaries))
    
    # Polarization angle: uniform on [0, π]
    polarization = rng.uniform(0, np.pi, size=n_binaries)
    
    # Inclination angle: uniform on [0, π]
    inclination = rng.uniform(0, np.pi, size=n_binaries)
    
    # Initial GW phase: uniform on [0, 2π]
    initial_phase = rng.uniform(0, 2*np.pi, size=n_binaries)
    
    # ========================================================================
    # STEP 6: Compute strain (optional)
    # ========================================================================
    
    strain_data = None
    
    if compute_strain:
        # Compute h² for each binary (circular orbits)
        h_squared = compute_characteristic_strain_squared_circular(
            gw_frequencies, 
            chirp_masses, 
            comoving_dist, 
            redshift
        )
        
        # Bin into frequency bins and compute h_c
        bin_centres, h_c_total, h_c_individual = bin_characteristic_strain(
            gw_frequencies, 
            h_squared, 
            n_freq_bins,
            T_obs=T_obs
        )
        
        # Find which bin each binary belongs to
        f_min = np.min(gw_frequencies)
        f_max = np.max(gw_frequencies)
        bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_freq_bins+1)
        bin_assignment = np.digitize(gw_frequencies, bin_edges) - 1
        bin_assignment = np.clip(bin_assignment, 0, n_freq_bins-1)
        
        strain_data = {
            'bin_centres': bin_centres,
            'h_c_total': h_c_total,
            'h_square_individual': h_squared,
            'bin_assignment': bin_assignment,
            'h_c_individual': h_c_individual,
            'bin_edges': bin_edges
        }
    
    # ========================================================================
    # STEP 7: Assemble population catalog
    # ========================================================================
    
    population = []
    
    for i in range(n_binaries):
        binary_params = {
            'Mc': chirp_masses[i],
            'Mtot': total_masses[i],
            'f': gw_frequencies[i],
            'D_comov': comoving_dist[i],
            'z': redshift[i],
            'ra': right_ascension[i],
            'dec': declination[i],
            'psi': polarization[i],
            'iota': inclination[i],
            'phi0': initial_phase[i]
        }
        
        # Add strain information if computed
        if compute_strain:
            binary_params['h_square'] = h_squared[i]
            binary_params['h_c_contrib'] = h_c_individual[i]
            binary_params['freq_bin'] = bin_assignment[i]
        
        population.append(binary_params)
    
    # ========================================================================
    # Return results
    # ========================================================================
    
    if compute_strain:
        return population, strain_data
    else:
        return population


# ============================================================================
# UTILITY FUNCTIONS (Plotting, etc.)
# ============================================================================

def plot_population_histogram(
    populations_dict,
    parameter_key,
    xlabel="",
    logx=False,
    logy=False,
    n_bins=30,
    normalize=True,
    figsize=(6.5, 3.25),
    save_path=None
):
    """
    Plot histogram comparing multiple populations.
    
    Parameters
    ----------
    populations_dict : dict
        Dictionary of {label: population_list} where each population_list
        is the output from generate_smbhb_population()
    parameter_key : str
        Which parameter to plot (e.g., 'Mc', 'f', 'z', 'D_comov')
    xlabel : str, optional
        X-axis label
    logx, logy : bool, optional
        Use log scale for x/y axis (default: False)
    n_bins : int, optional
        Number of histogram bins (default: 30)
    normalize : bool, optional
        Normalize histogram to PDF (default: True)
    figsize : tuple, optional
        Figure size (default: (6.5, 3.25))
    save_path : str, optional
        Path to save figure (default: None, don't save)
        
    Examples
    --------
    >>> pop1 = generate_smbhb_population(10000, mass_distribution='power_law')
    >>> pop2 = generate_smbhb_population(10000, mass_distribution='exponential_damping')
    >>> 
    >>> plot_population_histogram(
    ...     {'Power Law': pop1, 'Exp. Damping': pop2},
    ...     parameter_key='Mc',
    ...     xlabel='Chirp Mass [M$_\\odot$]',
    ...     logx=True
    ... )
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color scheme
    colors = {'Power Law': '#1f77b4', 'Exponential Damping': '#2ca02c'}
    
    for label, population in populations_dict.items():
        # Extract parameter values
        values = np.array([binary[parameter_key] for binary in population])
        
        # Plot histogram
        ax.hist(
            values,
            bins=n_bins if not logx else np.logspace(np.log10(values.min()), 
                                                     np.log10(values.max()), 
                                                     n_bins),
            alpha=0.6,
            label=label,
            color=colors.get(label, None),
            density=normalize,
            histtype='stepfilled'
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density" if normalize else "Count")
    ax.legend()
    
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    """
    Example usage and basic tests.
    """
    print("=" * 70)
    print("SMBHB Population Synthesis - Example")
    print("=" * 70)
    
    # Generate a small test population
    print("\nGenerating population of 1000 SMBHBs...")
    
    population, strain_data = generate_smbhb_population(
        n_binaries=1000,
        z_max=2.0,
        mass_distribution='power_law',
        compute_strain=True,
        n_freq_bins=50,
        random_seed=42
    )
    
    print(f"✓ Generated {len(population)} binaries")
    
    # Print summary statistics
    chirp_masses = np.array([b['Mc'] for b in population])
    frequencies = np.array([b['f'] for b in population])
    redshifts = np.array([b['z'] for b in population])
    
    print("\nPopulation Statistics:")
    print(f"  Chirp mass: {chirp_masses.min()/1e9:.2f} - {chirp_masses.max()/1e9:.2f} × 10⁹ M☉")
    print(f"  Frequency:  {frequencies.min()*1e9:.2f} - {frequencies.max()*1e9:.2f} nHz")
    print(f"  Redshift:   {redshifts.min():.3f} - {redshifts.max():.3f}")
    
    print("\nStrain Spectrum:")
    peak_idx = np.argmax(strain_data['h_c_total'])
    print(f"  Peak h_c:    {strain_data['h_c_total'][peak_idx]:.2e}")
    print(f"  Peak freq:   {strain_data['bin_centres'][peak_idx]*1e9:.2f} nHz")
    
    print("\n" + "=" * 70)
