"""
Supermassive Black Hole Binary (SMBHB) Population Synthesis

This module generates synthetic populations of SMBHBs for gravitational wave
background studies. It samples binary properties (masses, frequencies, distances)
from astrophysically-motivated distributions and computes their characteristic
gravitational wave strain.

Author: Blaze Houlden
Date: February 2026
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np
import numba as nb
from numba import njit, prange
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
import astropy.units as u
 
# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
 
HUBBLE_CONSTANT_H   = 0.67
OMEGA_MATTER        = 0.3
OMEGA_LAMBDA        = 0.7
H0_KMS_MPC          = 100 * HUBBLE_CONSTANT_H          # km/s/Mpc
SPEED_OF_LIGHT_KMS  = 2.9979e5                          # km/s
SPEED_OF_LIGHT_MS   = SPEED_OF_LIGHT_KMS * 1e3          # m/s
GRAVITATIONAL_CONST = 6.67e-11                          # m³ kg⁻¹ s⁻²
PARSEC_M            = 3.086e16                          # m
MEGAPARSEC_M        = 1e6 * PARSEC_M                    # m
SOLAR_MASS_KG       = 1.989e30                          # kg
YEAR_S              = 86400 * 365.25                    # s
FREQ_PER_YEAR       = 1.0 / YEAR_S                      # Hz
 
# ============================================================================
# POPULATION STORAGE
# ============================================================================
 
@dataclass
class PopulationArrays:
    """
    Compact storage for a SMBHB population as contiguous numpy arrays.
 
    All angular quantities in radians.
    Mc is in solar masses (consistent with the rest of your pipeline —
    convert to SI at the point of use with Mc * SOLAR_MASS_KG).
    D_comov is in Mpc.
    h0 is the dimensionless peak strain amplitude.
 
    Per-pulsar injection amplitudes (A, B) are populated later by
    precompute_amplitudes() from population_array.py and stored in
    amp_A / amp_B keyed by pulsar name.
    """
    f       : np.ndarray    # GW frequency [Hz]              shape (N,)
    Mc      : np.ndarray    # chirp mass [M_sun]              shape (N,)
    Mtot    : np.ndarray    # total mass [M_sun]              shape (N,)
    D_comov : np.ndarray    # comoving distance [Mpc]         shape (N,)
    z       : np.ndarray    # redshift                        shape (N,)
    h0      : np.ndarray    # strain amplitude                shape (N,)
    ra      : np.ndarray    # right ascension [rad]           shape (N,)
    dec     : np.ndarray    # declination [rad]               shape (N,)
    psi     : np.ndarray    # polarisation angle [rad]        shape (N,)
    iota    : np.ndarray    # inclination angle [rad]         shape (N,)
    phi0    : np.ndarray    # initial GW phase [rad]          shape (N,)
    cgw_snr : np.ndarray    # optimal CGW SNR                 shape (N,)
 
    amp_A   : Dict[str, np.ndarray] = field(default_factory=dict)
    amp_B   : Dict[str, np.ndarray] = field(default_factory=dict)
 
    def __len__(self) -> int:
        return len(self.f)
 
    def __getitem__(self, idx):
        """Slice the population (e.g. pop[mask], pop[:1000])."""
        new = PopulationArrays(
            f       = self.f[idx],
            Mc      = self.Mc[idx],
            Mtot    = self.Mtot[idx],
            D_comov = self.D_comov[idx],
            z       = self.z[idx],
            h0      = self.h0[idx],
            ra      = self.ra[idx],
            dec     = self.dec[idx],
            psi     = self.psi[idx],
            iota    = self.iota[idx],
            phi0    = self.phi0[idx],
            cgw_snr = self.cgw_snr[idx]
        )
        # carry over amplitude arrays if they exist, sliced to match
        for psr_name, A in self.amp_A.items():
            new.amp_A[psr_name] = A[idx]
        for psr_name, B in self.amp_B.items():
            new.amp_B[psr_name] = B[idx]
        return new
 
    def memory_gb(self) -> float:
        """Approximate RAM for source arrays (excludes amp_A/B)."""
        return len(self) * 12 * 8 / 1024**3
 
    def drop_amplitudes(self, psr_name: Optional[str] = None) -> None:
        """Free per-pulsar amplitude arrays after injection."""
        if psr_name is None:
            self.amp_A.clear()
            self.amp_B.clear()
        else:
            self.amp_A.pop(psr_name, None)
            self.amp_B.pop(psr_name, None)
 
    def to_dict_list(self):
        """
        Convert back to list-of-dicts format for compatibility with
        code that still expects the old format.  Avoid for large N.
        """
        keys = ['f','Mc','Mtot','D_comov','z','h0','ra','dec','psi','iota','phi0','cgw_snr']
        arrays = [self.f, self.Mc, self.Mtot, self.D_comov, self.z, self.h0,
                  self.ra, self.dec, self.psi, self.iota, self.phi0, self.cgw_snr]
        return [dict(zip(keys, vals)) for vals in zip(*arrays)]
 
 
# ============================================================================
# COSMOLOGICAL FUNCTIONS
# ============================================================================
 
def hubble_parameter(z: np.ndarray) -> np.ndarray:
    """H(z) for flat ΛCDM [km/s/Mpc]."""
    return H0_KMS_MPC * np.sqrt(OMEGA_MATTER * (1 + z)**3 + OMEGA_LAMBDA)
 
 
def build_comoving_distance_interpolator(z_max: float = 20.0,
                                         n_points: int = 2000):
    """Cubic interpolator: z → comoving distance [Mpc]."""
    z_grid   = np.linspace(0, z_max, n_points)
    chi_grid = np.array([
        quad(lambda zp: SPEED_OF_LIGHT_KMS / hubble_parameter(zp), 0, zi)[0]
        for zi in z_grid
    ])
    interp = interp1d(z_grid, chi_grid, kind='cubic', fill_value='extrapolate')
    return lambda z: interp(np.atleast_1d(z)).squeeze()[()]
 
 
# Build module-level interpolation grids once
_CHI_FN   = build_comoving_distance_interpolator(z_max=20.0)
 
# Fine grid for fast vectorised inversion (chi → z via np.interp)
_Z_GRID   = np.linspace(0, 20.0, 500_000)
_CHI_GRID = _CHI_FN(_Z_GRID)          # monotone increasing, shape (500000,)
 
# Expose for backward compatibility
COMOVING_DISTANCE_FN         = _CHI_FN
Z_GRID_NUMBA                 = _Z_GRID
CHI_GRID_NUMBA               = _CHI_GRID
 
 
def comoving_to_redshift(chi: np.ndarray) -> np.ndarray:
    """
    Vectorised chi [Mpc] → z via np.interp on the precomputed grid.
 
    Replaces the Numba binary-search loop.  np.interp is a single C call
    and is faster for any N > ~100.
    """
    return np.interp(chi, _CHI_GRID, _Z_GRID)
 
 
# Keep the Numba version for any downstream code that still calls it
@njit
def inverse_comoving_to_redshift_numba(comoving_distances):
    n         = len(comoving_distances)
    redshifts = np.empty(n)
    for i in range(n):
        chi   = comoving_distances[i]
        left  = 0
        right = len(CHI_GRID_NUMBA) - 1
        while right - left > 1:
            mid = (left + right) // 2
            if CHI_GRID_NUMBA[mid] <= chi:
                left = mid
            else:
                right = mid
        slope       = ((Z_GRID_NUMBA[right] - Z_GRID_NUMBA[left]) /
                       (CHI_GRID_NUMBA[right] - CHI_GRID_NUMBA[left]))
        redshifts[i] = Z_GRID_NUMBA[left] + slope * (chi - CHI_GRID_NUMBA[left])
    return redshifts
 
 
# ============================================================================
# SAMPLING — FREQUENCIES
# ============================================================================
 
@nb.njit(parallel=True, fastmath=True)
def sample_gw_frequencies(n_binaries, random_seeds,
                          t_obs_max=30 * YEAR_S,
                          t_obs_min=YEAR_S / 12):
    """
    Sample GW frequencies from p(f) ∝ f^(-11/3) via inverse-CDF.
 
    Sampling uniformly in x = f^(-8/3), then f = x^(-3/8).
    Parallelised across Numba threads (unchanged from original).
    """
    f_min   = 1.0 / t_obs_max
    f_max   = 1.0 / t_obs_min
    x_min   = f_min**(-8.0 / 3.0)
    x_max   = f_max**(-8.0 / 3.0)
    x_range = x_max - x_min
 
    frequencies = np.empty(n_binaries)
    n_threads   = len(random_seeds)
    chunk_size  = n_binaries // n_threads
 
    for tid in nb.prange(n_threads):
        np.random.seed(random_seeds[tid])
        start = tid * chunk_size
        end   = n_binaries if tid == n_threads - 1 else (tid + 1) * chunk_size
        for i in range(start, end):
            u              = np.random.rand()
            frequencies[i] = (x_min + x_range * u)**(-3.0 / 8.0)
 
    return frequencies
 
 
# ============================================================================
# SAMPLING — DISTANCES  (vectorised, no Numba needed)
# ============================================================================
 
def sample_comoving_distances(n_binaries: int,
                              distance_max: float,
                              distance_min: float = 1.0,
                              rng: Optional[np.random.Generator] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample comoving distances uniformly in volume, return (D_comov, D_lum, z).
 
    Replaces the Numba version — np.interp for the inversion is a single
    vectorised C call and avoids Numba's per-element binary search overhead.
    """
    if rng is None:
        rng = np.random.default_rng()
 
    vol_min     = distance_min**3
    vol_max     = distance_max**3
    u           = rng.random(n_binaries)
    D_comov     = (vol_min + (vol_max - vol_min) * u) ** (1.0 / 3.0)  # Mpc
    z           = comoving_to_redshift(D_comov)                         # vectorised
    D_lum       = (1.0 + z) * D_comov
    return D_comov, D_lum, z
 
 
# ============================================================================
# SAMPLING — MASSES
# ============================================================================
 
def sample_masses_power_law(n_binaries: int,
                            redshift_array: np.ndarray,
                            alpha_0: float = 1.21,
                            alpha_z: float = 0.0,
                            mass_min: float = 10**(7.5),
                            mass_max: float = 10**(12.5)) -> np.ndarray:
    """
    Sample primary masses from p(M) ∝ M^(-α) via vectorised inverse-CDF.
 
    Pure numpy — one call, no loop.  For redshift-independent α this is
    exact; redshift-dependent α is handled per-binary using broadcasting.
    """
    alpha    = alpha_0 + alpha_z * redshift_array          # (N,)
    exp      = 1.0 - alpha                                  # (N,)
    cdf_min  = mass_min ** exp
    cdf_max  = mass_max ** exp
    u        = np.random.rand(n_binaries)
    return (cdf_min + (cdf_max - cdf_min) * u) ** (1.0 / exp)
 
 
# Numba CDF builders — unchanged, fast for the exponential-damping case
@nb.njit
def _exponentially_damped_pdf(mass, alpha, mass_cutoff):
    return mass**(-alpha) * np.exp(-mass / mass_cutoff)
 
 
def build_cdf_exponential_damping(mass_min, mass_max, alpha, mass_cutoff,
                                   n_grid=20_000):
    mass_grid = np.linspace(mass_min, mass_max, n_grid)
    pdf_grid  = _exponentially_damped_pdf(mass_grid, alpha, mass_cutoff)
    cdf_grid  = np.empty(n_grid)
    cdf_grid[0] = 0.0
    for i in range(1, n_grid):
        cdf_grid[i] = cdf_grid[i-1] + 0.5*(pdf_grid[i]+pdf_grid[i-1])*(mass_grid[i]-mass_grid[i-1])
    cdf_grid /= cdf_grid[-1]
    return mass_grid, cdf_grid
 
 
@nb.njit(parallel=True)
def sample_from_precomputed_cdf(n_samples, mass_grid, cdf_grid):
    """Parallel inverse-CDF sampling via binary search on precomputed grid."""
    samples = np.empty(n_samples)
    for i in nb.prange(n_samples):
        u    = np.random.random()
        low  = 0
        high = len(cdf_grid) - 1
        while high - low > 1:
            mid = (low + high) // 2
            if cdf_grid[mid] < u:
                low = mid
            else:
                high = mid
        df = cdf_grid[high] - cdf_grid[low]
        if df > 0:
            w         = (u - cdf_grid[low]) / df
            samples[i] = mass_grid[low] + w * (mass_grid[high] - mass_grid[low])
        else:
            samples[i] = mass_grid[low]
    return samples
 
 
def sample_masses_exponential_damping(n_binaries, redshift_array,
                                      mass_cutoff_0=1e9, mass_cutoff_z=0.0,
                                      alpha_0=1.21, alpha_z=0.0,
                                      mass_min=10**(7.5), mass_max=10**(12.5),
                                      use_pure_power_law=False):
    alpha      = alpha_0
    mass_cutoff = mass_cutoff_0
    if use_pure_power_law:
        mass_grid, cdf_grid = build_cdf_exponential_damping.__wrapped__\
            if hasattr(build_cdf_exponential_damping, '__wrapped__') else \
            _build_cdf_power_law_numpy(mass_min, mass_max, alpha)
    else:
        mass_grid, cdf_grid = build_cdf_exponential_damping(
            mass_min, mass_max, alpha, mass_cutoff)
    return sample_from_precomputed_cdf(n_binaries, mass_grid, cdf_grid)
 
 
def _build_cdf_power_law_numpy(mass_min, mass_max, alpha, n_grid=20_000):
    mass_grid = np.linspace(mass_min, mass_max, n_grid)
    pdf_grid  = mass_grid**(-alpha)
    cdf_grid  = np.zeros(n_grid)
    dm        = mass_grid[1] - mass_grid[0]
    cdf_grid[1:] = np.cumsum(0.5*(pdf_grid[:-1] + pdf_grid[1:]) * dm)
    cdf_grid /= cdf_grid[-1]
    return mass_grid, cdf_grid
 
 
# ============================================================================
# MASS RATIOS AND CHIRP MASSES  (vectorised)
# ============================================================================
 
def compute_chirp_mass(primary_masses: np.ndarray,
                       use_equal_mass: bool = False,
                       rng: Optional[np.random.Generator] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised chirp mass computation.  No Python loop, no Numba overhead.
 
    Returns (total_masses, chirp_masses) both in the same units as primary_masses.
    """
    if rng is None:
        rng = np.random.default_rng()
 
    N  = len(primary_masses)
    q  = np.ones(N) if use_equal_mass else rng.random(N)   # [0, 1]
    M2 = q * primary_masses
    Mt = primary_masses + M2
    Mc = (primary_masses * M2)**(3.0/5.0) / Mt**(1.0/5.0)
    return Mt, Mc
 
 
# Keep Numba version for any code that still calls it directly
@nb.njit(parallel=True)
def sample_mass_ratios_and_compute_chirp_mass(n_binaries, primary_masses,
                                               use_equal_mass=False):
    total_masses = np.empty(n_binaries)
    chirp_masses = np.empty(n_binaries)
    for i in nb.prange(n_binaries):
        q               = 1.0 if use_equal_mass else np.random.rand()
        M2              = q * primary_masses[i]
        total_masses[i] = primary_masses[i] + M2
        chirp_masses[i] = (primary_masses[i] * M2)**(3.0/5.0) / total_masses[i]**(1.0/5.0)
    return total_masses, chirp_masses
 
 
# ============================================================================
# STRAIN AMPLITUDE  (vectorised, replaces Numba prange version)
# ============================================================================
 
def compute_strain_amplitude(gw_frequencies: np.ndarray,
                             chirp_masses_msun: np.ndarray,
                             D_comov_mpc: np.ndarray,
                             redshifts: np.ndarray,
                             inclinations: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised characteristic strain.  Pure numpy — no Numba overhead.
 
    Returns (h_squared, h0_amplitude) both shape (N,).
 
    h0  = 2 (G Mc)^(5/3) (2π f_rest)^(2/3) / (c^4 D_comov)
    h²  = (32/5) * (h0/2)²    [orientation-averaged]
       or the inclination-weighted version if inclinations is supplied.
    """
    f_rest  = 0.5 * (1.0 + redshifts) * gw_frequencies         # rest-frame orbital f
    Mc_SI   = chirp_masses_msun * SOLAR_MASS_KG                 # kg
    D_SI    = D_comov_mpc * MEGAPARSEC_M                        # m
 
    h0 = (2.0 * (GRAVITATIONAL_CONST * Mc_SI)**(5.0/3.0)
              * (2.0 * np.pi * f_rest)**(2.0/3.0)
              / (SPEED_OF_LIGHT_MS**4 * D_SI))                  # (N,)
 
    if inclinations is None:
        const   = 32.0 / (5.0 * SPEED_OF_LIGHT_MS**8)
        h_sq    = const * (0.5 * h0)**2
    else:
        a       = 1.0 + np.cos(inclinations)**2
        b       = -2.0 * np.cos(inclinations)
        MeanAng = np.sqrt(2.0 * (a**2 + b**2))
        const   = 1.0 / SPEED_OF_LIGHT_MS**8
        h_sq    = const * (0.5 * h0)**2 * MeanAng**2
 
    return h_sq, h0
 
 
# ============================================================================
# STRAIN BINNING  (vectorised, replaces Numba O(N*N_bins) loop)
# ============================================================================
 
def bin_characteristic_strain(gw_frequencies: np.ndarray,
                               h_squared: np.ndarray,
                               T_obs_seconds: float = 16.03 * YEAR_S
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin h² contributions onto a uniform frequency grid and compute h_c.
 
    Replaces the Numba loop version.  Key speedup: np.searchsorted for
    bin assignment (O(N log N_bins)) and np.bincount for accumulation (O(N)).
    For N=10^6 this is ~100× faster than the original inner loop.
 
    Returns
    -------
    bin_edges    : (N_bins+1,)  frequency bin edges [Hz]
    bin_centres  : (N_bins,)    bin centre frequencies [Hz]
    h_c_total    : (N_bins,)    sqrt(sum h² * f / Δf) per bin
    bin_idx      : (N,)         which bin each binary falls in (-1 = out of range)
    """
    f_min   = 1.0 / T_obs_seconds
    f_max   = 3e-7
    f_step  = f_min                                   # 1/T resolution
    N_bins  = int(round((f_max - f_min) / f_step)) + 1
 
    bin_edges   = np.linspace(f_min, f_min + N_bins * f_step, N_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths  = bin_edges[1:] - bin_edges[:-1]      # uniform, but keep explicit
 
    # Assign each binary to a bin — O(N log N_bins)
    raw_idx = np.searchsorted(bin_edges, gw_frequencies, side='right') - 1
 
    # Mask out-of-range sources
    in_range = (raw_idx >= 0) & (raw_idx < N_bins)
    bin_idx  = np.where(in_range, raw_idx, -1).astype(np.int64)
 
    # Accumulate h² per bin — O(N), fully vectorised
    h_sq_per_bin = np.bincount(bin_idx[in_range],
                               weights=h_squared[in_range],
                               minlength=N_bins)        # (N_bins,)
 
    # h_c = sqrt(h² * f / Δf)
    h_c_total = np.sqrt(h_sq_per_bin * bin_centres / bin_widths)
 
    return bin_edges, bin_centres, h_c_total, bin_idx
 
 
# ============================================================================
# MAIN POPULATION GENERATOR
# ============================================================================
 
def generate_smbhb_population(
        n_binaries: int,
        z_max: float = 2.0,
        mass_distribution: str = 'power_law',
        alpha_0: float = 1.21,
        alpha_z: float = 0.0,
        mass_min: float = 10**(7.5),
        mass_max: float = 10**(12.5),
        mass_cutoff_0: float = 10**(9),
        mass_cutoff_z: float = 0.0,
        compute_strain: bool = True,
        T_obs_seconds: float = 16.03 * YEAR_S,
        random_seed: Optional[int] = None,
) -> PopulationArrays | Tuple[PopulationArrays, dict]:
    """
    Generate a synthetic SMBHB population.
 
    Returns a PopulationArrays instead of a list of dicts.
    Set compute_strain=True to also get the strain spectrum dict.
 
    All sampling and computation is vectorised — no Python loops over binaries.
    """
    rng = np.random.default_rng(random_seed)
    if random_seed is not None:
        np.random.seed(random_seed)       # also seed the global state for Numba
 
    # ── frequencies ─────────────────────────────────────────────────────────
    n_threads    = nb.get_num_threads()
    thread_seeds = rng.integers(0, 2**32 - 1, size=n_threads)
    f = sample_gw_frequencies(n_binaries, thread_seeds, t_obs_max=T_obs_seconds)
 
    # ── distances & redshifts ────────────────────────────────────────────────
    D_max               = float(_CHI_FN(z_max))
    D_comov, D_lum, z   = sample_comoving_distances(n_binaries, D_max, rng=rng)
 
    # ── masses ───────────────────────────────────────────────────────────────
    if mass_distribution == 'exponential_damping':
        M1 = sample_masses_exponential_damping(
            n_binaries, z,
            mass_cutoff_0=mass_cutoff_0, mass_cutoff_z=mass_cutoff_z,
            alpha_0=alpha_0, alpha_z=alpha_z,
            mass_min=mass_min, mass_max=mass_max,
        )
    else:
        M1 = sample_masses_power_law(
            n_binaries, z,
            alpha_0=alpha_0, alpha_z=alpha_z,
            mass_min=mass_min, mass_max=mass_max,
        )
 
    Mtot, Mc = compute_chirp_mass(M1, rng=rng)     # solar masses
 
    # ── sky positions & orientations ─────────────────────────────────────────
    ra   = rng.uniform(0,       2*np.pi, n_binaries)
    dec  = np.arcsin(rng.uniform(-1, 1,  n_binaries))
    psi  = rng.uniform(0,       np.pi,   n_binaries)
    iota = rng.uniform(0,       np.pi,   n_binaries)
    phi0 = rng.uniform(0,       2*np.pi, n_binaries)
 
    # ── strain amplitude ─────────────────────────────────────────────────────
    h_sq, h0 = compute_strain_amplitude(f, Mc, D_comov, z, iota)
 
    # ── assemble PopulationArrays (no Python loop) ───────────────────────────
    pop = PopulationArrays(
        f=f, Mc=Mc, Mtot=Mtot, D_comov=D_comov, z=z, h0=h0,
        ra=ra, dec=dec, psi=psi, iota=iota, phi0=phi0, cgw_snr=np.zeros(n_binaries, dtype=np.float16)
    )
 
    if not compute_strain:
        return pop
 
    # ── strain binning ───────────────────────────────────────────────────────
    bin_edges, bin_centres, h_c_total, bin_idx = bin_characteristic_strain(
        f, h_sq, T_obs_seconds=T_obs_seconds
    )
 
    strain_data = {
        'bin_edges'          : bin_edges,
        'bin_centres'        : bin_centres,
        'h_c_total'          : h_c_total,
        'h_square_individual': h_sq,
        'bin_assignment'     : bin_idx,
        'h_c_individual'     : np.sqrt(h_sq),   # per-binary contribution
    }
 
    return pop, strain_data
 
 
# ============================================================================
# FIXED-PROPERTY POPULATION
# ============================================================================
 
def chosen_population(
        n_binaries: int,
        chirp_mass_msun: float = 1e10,
        mass_ratio: float = 0.5,
        gw_frequency: float = 1e-8,
        redshift: float = 0.5,
        polarization: float = 0.0,
        inclination: float = 0.0,
        initial_phase: float = 0.0,
        right_ascension: float = 0.0,
        declination: float = 0.0,
        compute_strain: bool = True,
        T_obs_seconds: float = 16.03 * YEAR_S,
) -> PopulationArrays | Tuple[PopulationArrays, dict]:
    """
    Population of identical SMBHBs with user-specified properties.
 
    Returns PopulationArrays (+ strain_data if compute_strain=True).
    """
    Mc_arr  = np.full(n_binaries, chirp_mass_msun)
    Mtot    = Mc_arr * (1 + mass_ratio)**(1/5) / mass_ratio**(3/5)
    f_arr   = np.full(n_binaries, gw_frequency)
    z_arr   = np.full(n_binaries, redshift)
    D_arr   = np.full(n_binaries, float(_CHI_FN(redshift)))
    iota_arr= np.full(n_binaries, inclination)
 
    h_sq, h0 = compute_strain_amplitude(f_arr, Mc_arr, D_arr, z_arr, iota_arr)
 
    pop = PopulationArrays(
        f       = f_arr,
        Mc      = Mc_arr,
        Mtot    = Mtot,
        D_comov = D_arr,
        z       = z_arr,
        h0      = h0,
        ra      = np.full(n_binaries, right_ascension),
        dec     = np.full(n_binaries, declination),
        psi     = np.full(n_binaries, polarization),
        iota    = iota_arr,
        phi0    = np.full(n_binaries, initial_phase),
        cgw_snr = np.zeros(n_binaries, dtype=np.float16)
    )
 
    if not compute_strain:
        return pop
 
    bin_edges, bin_centres, h_c_total, bin_idx = bin_characteristic_strain(
        f_arr, h_sq, T_obs_seconds=T_obs_seconds
    )
 
    strain_data = {
        'bin_edges'          : bin_edges,
        'bin_centres'        : bin_centres,
        'h_c_total'          : h_c_total,
        'h_square_individual': h_sq,
        'bin_assignment'     : bin_idx,
        'h_c_individual'     : np.sqrt(h_sq),
    }
 
    return pop, strain_data
 
 
# ============================================================================
# PLOTTING UTILITY
# ============================================================================
 
def plot_population_histogram(
        populations_dict: dict,
        parameter_key: str,
        xlabel: str = '',
        logx: bool = False,
        logy: bool = False,
        n_bins: int = 30,
        normalize: bool = True,
        figsize=(6.5, 3.25),
        save_path: Optional[str] = None,
):
    """
    Compare parameter distributions across multiple PopulationArrays.
 
    populations_dict: {label: PopulationArrays}
    parameter_key:    field name, e.g. 'Mc', 'f', 'z'
    """
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=figsize)
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
 
    for idx, (label, pop) in enumerate(populations_dict.items()):
        vals = getattr(pop, parameter_key)               # direct array access
 
        if logx:
            bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), n_bins)
        else:
            bins = n_bins
 
        ax.hist(vals, bins=bins, alpha=0.6, label=label,
                color=colors[idx % len(colors)],
                density=normalize, histtype='stepfilled')
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density' if normalize else 'Count')
    ax.legend()
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def precompute_amplitudes(pop: PopulationArrays, psr, chunk_size=10_000_000):
    """
    Compute and store A[j], B[j] for pulsar psr into pop.amp_A/B[psr.name].

    A[j] = (Fp * h0 * (1 + cos²ι)) / (2π f)   — raw sin coefficient
    B[j] = (Fx * h0 * (-2 cosι))   / (2π f)   — raw cos coefficient

    These are the amplitude prefactors BEFORE phase rotation.
    phi0 is applied at injection time, matching _gw_residuals_vec exactly:
        phase = 2π f (t - t[0]) + phi0
        r(t)  = A sin(phase) + B cos(phase)

    No t_ref dependence here — phi0 is handled in inject_population_nufft.
    """
    N        = len(pop)
    psr_name = psr.name

    pars = psr.pars()

    if 'RAJ' not in pars or 'DECJ' not in pars:
        elong = psr['ELONG'].val  # radians
        elat  = psr['ELAT'].val   # radians

        coord = SkyCoord(lon=elong*u.rad, lat=elat*u.rad, frame='geocentricmeanecliptic')
        equatorial = coord.icrs

        psr_ra  = equatorial.ra.rad
        psr_dec = equatorial.dec.rad

    else:
        psr_ra   = psr._raj
        psr_dec  = psr._decj

    A_full = np.empty(N, dtype=np.float64)
    B_full = np.empty(N, dtype=np.float64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sl  = slice(start, end)

        f    = pop.f[sl]
        h0   = pop.h0[sl]
        ra   = pop.ra[sl]
        dec  = pop.dec[sl]
        psi  = pop.psi[sl]
        iota = pop.iota[sl]

        Fp, Fx = _antenna_response_vec(psr_ra, psr_dec, ra, dec, psi)

        # Raw amplitude prefactors — no phi0, no t_ref
        # Matches _gw_residuals_vec:
        #   weighted = Fp*h0*(1+cos²ι)*sin(phase) + Fx*h0*(-2cosι)*cos(phase)
        #   r = weighted / (2π f)
        A_full[sl] = (Fp * h0 * (1 + np.cos(iota)**2)) / (2 * np.pi * f)
        B_full[sl] = (Fx * h0 * (-2 * np.cos(iota)))   / (2 * np.pi * f)

    pop.amp_A[psr_name] = A_full
    pop.amp_B[psr_name] = B_full
 
 
def _antenna_response_vec(psr_ra, psr_dec, ra, dec, psi):
    """Vectorised antenna response. Returns Fp, Fx each shape (N,)."""
    src_polar = np.pi/2 - dec
    psr_polar = np.pi/2 - psr_dec
 
    sin_sp = np.sin(src_polar); cos_sp = np.cos(src_polar)
    sin_sa = np.sin(ra);        cos_sa = np.cos(ra)
 
    omega = np.stack([-sin_sp*cos_sa, -sin_sp*sin_sa, -cos_sp])  # (3,N)
    p     = np.array([
        np.sin(psr_polar)*np.cos(psr_ra),
        np.sin(psr_polar)*np.sin(psr_ra),
        np.cos(psr_polar),
    ])  # (3,)
 
    m = np.stack([sin_sa, -cos_sa, np.zeros_like(ra)])             # (3,N)
    n = np.stack([-cos_sp*cos_sa, -cos_sp*sin_sa, sin_sp])        # (3,N)
 
    cos_psi = np.cos(psi); sin_psi = np.sin(psi)
    m_rot   =  cos_psi*m + sin_psi*n
    n_rot   = -sin_psi*m + cos_psi*n
 
    denom = 1 + p @ omega    # (N,)
    p_m   = p @ m_rot        # (N,)
    p_n   = p @ n_rot        # (N,)
 
    Fp = 0.5*(p_m**2 - p_n**2) / denom
    Fx = (p_m * p_n) / denom
    return Fp, Fx