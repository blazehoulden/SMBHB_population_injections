import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.special import gammaincc, gammainccinv  # gammaincc = Q(s,x)
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import json
import numba as nb
import tqdm
import pathos


# Constants
h = 0.67
omega_M = 0.3
omega_L = 0.7
H0 = 100 * h # km/s/Mpc
c_kms = 2.9979e5      # km/s
c_ms = c_kms * 1e3    # m/s
G = 6.67e-11 # kg^-1 m^3 s^-2
pc = 3.086e16 # m
Mpc = 1e6 * pc # m
Msun = 1.989e30 # kg
yr = 86400 * 365.25 # s
inv_yr = 1 / yr # s^-1
inv_10yr = 1 / (10 *yr) # s^-1

# Cosmological functions
def H(z):
    return H0 * np.sqrt(omega_M * (1 + z)**3 + omega_L) # km/s/Mpc

def build_comov_dist(z_max=20.0):
    """Return interpolator for comoving distance (Mpc)."""
    z_grid = np.linspace(0, z_max, 2000)
    chi_grid = np.array([quad(lambda zp: c_kms/H(zp), 0, zi)[0] for zi in z_grid])
    interp = interp1d(z_grid, chi_grid, kind="cubic", fill_value="extrapolate")
    def comov_dist(z):
        z = np.atleast_1d(z)
        return interp(z)
    return comov_dist # Mpc

comov_dist_fn = build_comov_dist(z_max=5) # Mpc


def solve_for_z_from_comov_dist(comov_dist, z_max = 20):
    func  = lambda z: comov_dist_fn(z) - comov_dist
    return brentq(func, 0, z_max) # unitless


def build_inverse_comov_to_z(comov_dist_fn, z_max=5.0, n_points=2000):
    # Build the forward table
    z_grid = np.linspace(0, z_max, n_points)
    chi_grid = comov_dist_fn(z_grid)
    
    # Ensure monotonicity (always increasing)
    # Then invert: chi → z
    inv_interp = interp1d(
        chi_grid, z_grid, kind='linear',
        fill_value=(0, z_max), bounds_error=False
    )
    return inv_interp

inv_comov_to_z = build_inverse_comov_to_z(comov_dist_fn, z_max=2.0)



# Forward comoving distance
z_max = 5.0
n_points = 2000
z_grid = np.linspace(0, z_max, n_points)
chi_grid = comov_dist_fn(z_grid)  # Mpc

# Ensure monotonicity
assert np.all(np.diff(chi_grid) > 0)

from numba import njit

@njit
def inv_comov_to_z_numba(chi):
    """
    chi must be a 1D array
    """
    n = len(chi)
    z_out = np.empty(n)
    for i in range(n):
        x = chi[i]
        left = 0
        right = len(chi_grid) - 1
        while right - left > 1:
            mid = (left + right) // 2
            if chi_grid[mid] <= x:
                left = mid
            else:
                right = mid
        slope = (z_grid[right] - z_grid[left]) / (chi_grid[right] - chi_grid[left])
        z_out[i] = z_grid[left] + slope * (x - chi_grid[left])
    return z_out


# SAMPLING FUNCTIONS

# Frequency Sampling
@nb.njit(parallel=True, fastmath=True)
def f_sampler(N_binaries, seeds, tmax = 30 * yr, tmin = yr / 12):
    fmin = 1.0 / tmax
    fmax = 1.0 / tmin
    xmin = fmin ** (-8.0/3.0)
    xmax = fmax ** (-8.0/3.0)
    xdiff = xmax - xmin
    out = np.empty(N_binaries)
    n_threads = len(seeds)
    chunk = N_binaries // n_threads

    for t in nb.prange(n_threads):
        np.random.seed(seeds[t])
        start = t * chunk
        end = N_binaries if t == n_threads - 1 else (t + 1) * chunk
        for i in range(start, end):
            u = np.random.rand()
            x = xmin + xdiff * u
            out[i] = x ** (-3.0/8.0)
    return out

# Distance Sampling
@njit
def dist_sampler(N_binaries,  dmax=None, dmin=1.0,inv_comov_to_z_numba=inv_comov_to_z_numba):
    """
    Sample comoving distances and compute redshift and luminosity distance,
    Numba-accelerated.
    
    Parameters
    ----------
    N_binaries : int
        Number of binaries to sample
    dmin : float
        Minimum comoving distance
    dmax : float
        Maximum comoving distance
    inv_comov_to_z_numba : function
        Numba-jitted function for comoving distance → z
    """

    # Sample distances ∝ volume
    vol_min = dmin ** 3
    vol_max = dmax ** 3
    x = np.random.rand(N_binaries)
    x *= vol_max - vol_min
    x += vol_min
    dist = x ** (1/3)

    # Compute redshift
    z = inv_comov_to_z_numba(dist)

    # Luminosity distance
    lum_dist = (1.0 + z) * dist

    return dist, lum_dist, z


# Mass Sampling

# Simple power-law
def mass_sampler(N_binaries, z, alpha_con = 1.21, alpha_z = 0.0, m_min = 1e7, m_max = 1e11): # take out redshift dependence for now
    alpha = alpha_con + alpha_z * z
    mass_dist_max = m_max **(( -alpha + 1))
    mass_dist_min = m_min **(( -alpha + 1))

    mass_dist_diff = mass_dist_max - mass_dist_min

    mass_dist = mass_dist_diff * np.random.rand(N_binaries) + mass_dist_min
    mass = mass_dist ** (1/( -alpha + 1))
    return mass

# ------------------------------
# PDF and CDF helpers
# ------------------------------

@nb.njit
def exp_damp_pdf(m, alpha, m_c):
    return m**(-alpha) * np.exp(-m / m_c)

@nb.njit
def power_law_pdf(m, alpha):
    return m**(-alpha)

def build_cdf_exp_damp(m_min, m_max, alpha, m_c, n_grid=20000):
    m_grid = np.linspace(m_min, m_max, n_grid)
    pdf_grid = exp_damp_pdf(m_grid, alpha, m_c)
    cdf_grid = np.empty(n_grid)
    cdf_grid[0] = 0.0
    for i in range(1, n_grid):
        cdf_grid[i] = cdf_grid[i-1] + 0.5 * (pdf_grid[i] + pdf_grid[i-1]) * (m_grid[i] - m_grid[i-1])
    cdf_grid /= cdf_grid[-1]  # normalize
    return m_grid, cdf_grid

def build_cdf_power_law(m_min, m_max, alpha, n_grid=20000):
    m_grid = np.linspace(m_min, m_max, n_grid)
    pdf_grid = power_law_pdf(m_grid, alpha)
    cdf_grid = np.empty(n_grid)
    cdf_grid[0] = 0.0
    for i in range(1, n_grid):
        cdf_grid[i] = cdf_grid[i-1] + 0.5 * (pdf_grid[i] + pdf_grid[i-1]) * (m_grid[i] - m_grid[i-1])
    cdf_grid /= cdf_grid[-1]
    return m_grid, cdf_grid

# ------------------------------
# Numba sampling from precomputed CDF
# ------------------------------
@nb.njit(parallel=True)
def sample_from_precomputed_cdf(N, m_grid, cdf_grid):
    samples = np.empty(N)
    for i in nb.prange(N):
        u = np.random.random()
        # binary search in CDF
        low, high = 0, len(cdf_grid)-1
        while high - low > 1:
            mid = (low + high) // 2
            if cdf_grid[mid] < u:
                low = mid
            else:
                high = mid
        # linear interpolation
        c1, c2 = cdf_grid[low], cdf_grid[high]
        m1, m2 = m_grid[low], m_grid[high]
        if c2 - c1 > 0:
            samples[i] = m1 + (u - c1)/(c2 - c1) * (m2 - m1)
        else:
            samples[i] = m1
    return samples

# ------------------------------
# Main sampler
# ------------------------------
def mass_sampler_exp_damp(N_binaries, z_array,
                          m_c_con=1e9, m_c_z=0, #0.11e9, # take out redshift dependence for now
                          alpha_con=1.21, alpha_z=0.0, # 0.03, # take out redshift dependence for now
                          m_min=1e7, m_max=1e11,
                          power_law=False):
    masses = np.empty(N_binaries)
    
    # If alpha and m_c are constant, precompute CDF once for speed
    alpha0 = alpha_con + alpha_z * 0
    m_c0 = m_c_con + m_c_z * 0
    if power_law:
        m_grid, cdf_grid = build_cdf_power_law(m_min, m_max, alpha0)
    else:
        m_grid, cdf_grid = build_cdf_exp_damp(m_min, m_max, alpha0, m_c0)

    # Sample all binaries
    masses = sample_from_precomputed_cdf(N_binaries, m_grid, cdf_grid)
    return masses

# Mass ratio sampler
@nb.njit(parallel=True)
def q_sampler(N_binaries, tot_mass_array, simple=False):
    chirp_mass = np.empty(N_binaries)
    q_array = np.empty(N_binaries)

    if simple:
        for i in nb.prange(N_binaries):
            q_array[i] = 1.0
            chirp_mass[i] = (1 / (1 + 1)**2)**(3/5) * tot_mass_array[i] * Msun
        return tot_mass_array, chirp_mass

    # uniform mass ratio distribution
    q_min = 0.05
    q_max = 1.0
    q_diff = q_max - q_min

    for i in nb.prange(N_binaries):
        q = q_min + q_diff * np.random.rand()
        q_array[i] = q
        chirp_mass[i] = (q / (1 + q)**2)**(3/5) * tot_mass_array[i] * Msun

    return tot_mass_array, chirp_mass

# TOTAL CHARACTERISTIC STRAIN
@nb.njit(parallel=True)
def compute_h_square_circ_nb(fGW, chirp_mass, dist, z):
    # calculate the RMS strain averaged over orbital orientations
    N = fGW.size
    h2 = np.empty(N, dtype=np.float64)
    const = 32.0 / (5.0 * c_ms**8)
    for i in nb.prange(N):
        f_rest_orb = 0.5 * (1.0 + z[i]) * fGW[i]
        h2[i] = const * (G * chirp_mass[i])**(10/3) / (dist[i] * Mpc)**2 * (2*np.pi*f_rest_orb)**(4/3)
    return h2

# --- Helper: bin h^2 and compute h_c_total and per-binary contribution ---
@nb.njit
def bin_h_c_nb(fGW, h_square_circ, n_freq_bins):
    # calculate the characteristic strain spectrum of population
    f_min = np.min(fGW)
    f_max = np.max(fGW)
    bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_freq_bins+1)
    bin_centres = 0.5*(bin_edges[:-1] + bin_edges[1:])
    delta_f_bin = bin_edges[1:] - bin_edges[:-1]

    # Digitize
    bin_idx = np.empty(fGW.size, dtype=np.int64)
    n_bins = n_freq_bins
    for i in range(fGW.size):
        x = fGW[i]
        for b in range(n_bins):
            if bin_edges[b] <= x < bin_edges[b+1]:
                bin_idx[i] = b
                break
        else:
            bin_idx[i] = n_bins-1

    # Compute per-bin sum
    h_square_sum = np.zeros(n_bins, dtype=np.float64)
    for i in range(fGW.size):
        h_square_sum[bin_idx[i]] += h_square_circ[i]

    h_c_total = np.sqrt(h_square_sum * (bin_centres / delta_f_bin))
    h_c_contrib = np.sqrt(h_square_circ * (bin_centres[bin_idx] / delta_f_bin[bin_idx]))

    return bin_centres, h_c_total, h_c_contrib

def h_circ(N_binaries, diagnostics=False, n_freq_bins=50,
           alpha_con=1.21, alpha_z=0.0,
           m_min=1e7, m_max=1e11,
           reduce_mass=False, divide_spec=False,
           dmax=comov_dist_fn(z=1),
           mass_exp_damp_flag=False, m_c_con=1e9, m_c_z=0.0, power_law=False,
           inv_comov_to_z_numba=inv_comov_to_z_numba, rng=None):

    # --- Sample binaries ---
    if rng is None:
        rng = np.random.default_rng()  # create if not provided
    n_threads = nb.get_num_threads()
    seeds = rng.integers(0, 2**32 - 1, size=n_threads)
    fGW = f_sampler(N_binaries, seeds)
    dist, lum_dist, z = dist_sampler(N_binaries, dmax=dmax, inv_comov_to_z_numba=inv_comov_to_z_numba)

    if mass_exp_damp_flag:
        mass = mass_sampler_exp_damp(N_binaries=N_binaries, z_array=z, alpha_con=alpha_con, alpha_z=alpha_z,
                                     m_min=m_min, m_max=m_max,
                                     power_law=power_law,
                                     m_c_con=m_c_con, m_c_z=m_c_z)
    else:
        mass = mass_sampler(N_binaries, z, alpha_con, alpha_z, m_min, m_max)

    tot_mass, chirp_mass = q_sampler(N_binaries, mass)  # your Numba function

    if reduce_mass:
        chirp_mass *= 0.1

    # --- h^2 computation - RMS strain ---
    h_square_circ = compute_h_square_circ_nb(fGW, chirp_mass, dist, z)

    # --- Bin h^2 ---
    bin_centres, h_c_total, h_c_contrib = bin_h_c_nb(fGW, h_square_circ, n_freq_bins)

    # --- Reference power law ---
    alpha = 2/3
    h_ref = h_c_total[0]
    h_c_powerlaw = h_ref * (bin_centres / bin_centres[0])**(-alpha)

    # --- NANOGrav reference ---
    A_NG = 2.4e-15
    h_c_NG = A_NG * (bin_centres / inv_yr)**(-alpha)

    if divide_spec:
        h_c_total /= 100
        h_c_contrib /= 100
        h_c_powerlaw /= 100

    if diagnostics:
        return bin_centres, h_c_total, h_c_contrib, h_c_powerlaw, h_c_NG, fGW, dist, z, tot_mass, chirp_mass
    else:
        return bin_centres, h_c_total, h_c_contrib, h_c_powerlaw, h_c_NG


# Diagnostic plotting functions

# From sampling
def diagnostics(f, tot_mass, z, h_c_cont = None, h_c_flag = False):
    # make diagnostic plots
    # frequency diagnostics
    fmaxplot = 30e-9
    fmin = np.min(f)
    fi = np.arange(fmin, fmaxplot, fmin)
    mask = f < fmaxplot
    nf, _ = np.histogram(f[mask], bins=fi)

    plt.figure()
    plt.plot(fi[:-1], np.log10(nf), 'b')
    plt.xlabel('f (Hz)')
    plt.xscale('log')
    plt.ylabel('log10(N)')
    plt.title('Frequency distribution of SMBHB population')
    plt.tight_layout()
    plt.savefig('freq.png', dpi=150)
    plt.show()

    # mass diagnostics
    m_maxplot = 1e11
    m_min = np.min(tot_mass)
    mi = np.arange(2 * m_min, m_maxplot, 2 * m_min)
    mask = tot_mass < m_maxplot
    nm, _ = np.histogram(tot_mass[mask], bins=mi)

    plt.figure()
    plt.plot(mi[:-1], np.log10(nm), 'b')
    plt.xlabel(r'Mass (M$_{\odot}$)')
    plt.xscale('log')
    plt.ylabel('log10(N)')
    plt.title('Mass distribution of SMBHB population')
    plt.tight_layout()
    plt.savefig('mass.png', dpi=150)
    plt.show()

    dmin = 1 # Mpc
    dmax = comov_dist_fn(z = np.max(z))
    # comoving volume distribution
    v_min = 4 * np.pi / 3 * dmin**3 
    v_maxplot = 4 * np.pi / 3 * dmax[0]**3
    vi = np.linspace((v_min), (v_maxplot), 100)

    vol = 4 * np.pi / 3 * comov_dist_fn(z)**3
    mask = vol < v_maxplot
    nv, _ = np.histogram(vol[mask], bins=vi)

    plt.figure()
    plt.plot(vi[:-1], np.log10(nv), 'b')
    plt.xlabel(r'Volume (cMpc$^{-3}$)')
    plt.ylabel('log10(N)')
    plt.title('Volume distribution of SMBHB population')
    plt.tight_layout()
    plt.savefig('vol.png', dpi=150)
    plt.show()

    if h_c_flag == True:
        h_min = np.min(h_c_cont)
        h_max = np.max(h_c_cont)
        # hi = np.arange((h_min), (h_max), h_min)
        # hi = np.logspace(np.log10(h_min), np.log10(h_max), 100)
        hi = np.linspace((h_min), (h_max), 1000000)

        # mask = vol < v_maxplot
        nh, _ = np.histogram(h_c_cont, bins=hi)

        plt.figure()
        plt.plot(hi[:-1], np.log10(nh), 'b')
        plt.xlabel(r'$h_{\mathrm{circ}}$')
        plt.ylabel('log10(N)')
        plt.xscale('log')
        plt.title('Individual strain distribution of SMBHB population')
        plt.tight_layout()
        plt.savefig('hc_circ.png', dpi=150)
        plt.show()
    return

# Characteristic strain plotting
def plot_char_str(bin_centres, h_c, h_c_powerlaw, h_c_NG, f_max_residual=None, savefile=None):
    """
    Plot characteristic strain h_c(f) and residual relative to NANOGrav constraints.
    
    Parameters
    ----------
    bin_centres : array
        Frequencies of bins.
    h_c : array
        Calculated characteristic strain.
    h_c_powerlaw : array
        Reference power-law strain.
    h_c_NG : array
        NANOGrav reference strain.
    f_max_residual : float, optional
        Maximum frequency (Hz) to plot residuals; residuals above this are ignored.
    savefile : str, optional
        If provided, save figure to this PDF filename.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Mask for residuals if f_max_residual is set
    if f_max_residual is not None:
        mask = bin_centres <= f_max_residual
    else:
        mask = np.ones_like(bin_centres, dtype=bool)
    
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={'height_ratios':[2,1]})

    # --- Top panel: characteristic strain ---
    ax[0].loglog(bin_centres, h_c, label='$h_c(f)$', lw=2)
    ax[0].loglog(bin_centres, h_c_powerlaw, '--', lw=1.8, label=r'$\propto f^{-2/3}$')
    ax[0].loglog(bin_centres, h_c_NG, ':', lw=1.8, label='NG')
    
    # Reference lines
    ax[0].vlines(inv_10yr, ymin=min(h_c), ymax=max(h_c), colors='red', linestyles='-.', label='1/10yr')
    ax[0].vlines(inv_yr, ymin=min(h_c), ymax=max(h_c), colors='cyan', linestyles='-.', label='1/yr')
    
    ax[0].set_ylabel(r'$h_c$')
    ax[0].legend(fontsize=9)
    ax[0].grid(True, which='both', ls=':', lw=0.5)

    # --- Bottom panel: residual ---
    residual = (h_c - h_c_NG)/h_c_NG
    ax[1].plot(bin_centres[mask], residual[mask], lw=1.8, color='black')
    ax[1].axhline(0, color='gray', ls='--', lw=1)
    ax[1].set_xlabel(r'$f$ [Hz]')
    ax[1].set_yscale('symlog', linthresh=0.01)
    ax[1].set_ylabel(r'Residual $(h_c - h_{\rm NG})/h_{\rm NG}$')
    ax[1].grid(True, ls=':', lw=0.5)
    
    plt.tight_layout()
    
    if savefile is not None:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure as {savefile}")
    else:
        plt.show()
    return


# Comparison plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set(style="whitegrid", context="talk")

# Colors for each subset
subset_colors = {
    "Pessimistic": "#f72d2d",  # red
    "Realistic": "#1f77b4",    # blue
    "Optimistic": "#2ca02c"    # green
}

def plot_overlay(results, key_loudest, key_nearest=None, xlabel="", 
                 logx=False, logy=False, bins=30, save_name=None, 
                 div_Msun=False, vol_conv=False, log_bins=False, alpha=0.5):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "ytick.right": False,
        "ytick.left": True,
        "xtick.top": False,
        "xtick.bottom": True,

    })

    width_in = 3.25
    fig, ax = plt.subplots(figsize=(2 * width_in, width_in))

    for subset in results:
        data = np.array(results[subset][key_loudest], dtype=float)
        if div_Msun:
            data = data / Msun
        if vol_conv:
            data = 4/3 * np.pi * data**3

        if log_bins:
            data = data[data > 0]
            bins_edges = np.logspace(np.log10(data.min()), np.log10(data.max()), bins + 1)
        else:
            bins_edges = bins

        ax.hist(
            data,
            bins=bins_edges,
            alpha=alpha,
            label=subset,
            color=subset_colors.get(subset, "gray"),
            density=True
        )

        if key_nearest is not None:
            data_near = np.array(results[subset][key_nearest], dtype=float)
            if div_Msun:
                data_near = data_near / Msun
            if vol_conv:
                data_near = 4/3 * np.pi * data_near**3
            if log_bins:
                data_near = data_near[data_near > 0]
            ax.hist(
                data_near,
                bins=bins_edges,
                alpha=0.3,
                color=subset_colors.get(subset, "gray"),
                density=False,
                linestyle='dashed'
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDF")
    ax.legend(fontsize=12)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    # Set black outline for all four sides
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    # Automatic ticks
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, dpi=150)
    plt.show()
    return


def generate_SMBHB_population(
        N_binaries,
        mass_exp_damp_flag=False,
        alpha_con=1.21,
        alpha_z=0,# 0.03, # take out redshift dependence for now
        m_min=1e7,
        m_max=1e11,
        power_law=True,
        m_c_con=1e9,
        m_c_z=0.0, #0.11e9, # take out redshift dependence for now
        z_max=2.0,
        rng=None,
        compute_strain=False,
        n_freq_bins=50
    ):
    """
    Generate a full SMBHB population using popsyn samplers and return a list
    of dictionaries, one per binary, in the form:

        {
            'Mc':  chirp mass
            'f':   GW frequency
            'D_comov':   comoving distance (Mpc)
            'ra':  right ascension (radians)
            'dec': declination (radians)
            'psi': polarization angle
            'iota': inclination
            'phi0': initial GW phase
        }

    All popsyn mass/frequency/distance sampling is included inside this wrapper.

    Parameters
    ----------
    N_binaries : int
        Number of SMBHBs to sample.
    popsyn : module
        Your population synthesis module (with f_sampler, dist_sampler, etc.).
    nb : module
        Your numba-threading helper for seeds.
    mass_exp_damp_flag : bool
        Whether to use the exponential-damped mass sampler.
    compute_strain : bool
        If True, also compute strain and return binned h_c values with individual contributions.
    n_freq_bins : int
        Number of frequency bins for strain calculation (if compute_strain=True).
    The rest are passed to your popsyn mass samplers.

    Returns
    -------
    population : list of dict
        Each entry is an SMBHB parameter dictionary.
    strain_data : dict (only if compute_strain=True)
        Dictionary containing:
            'bin_centres': frequency bin centers
            'h_c_total': total characteristic strain per bin
            'h_square_individual': h^2 for each binary
            'bin_assignment': which bin each binary belongs to
            'h_c_individual': individual h_c contribution for each binary
    """

    # RNG ------------------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    # comoving cutoff
    dmax = comov_dist_fn(z=z_max)

    # threads + seeds for your f_sampler
    n_threads = nb.get_num_threads()
    seeds = rng.integers(0, 2**32 - 1, size=n_threads)

    # ------------------------------------------------------------
    # SAMPLE POPULATION FROM POPSYN
    # ------------------------------------------------------------

    # frequencies
    fGW = f_sampler(N_binaries=N_binaries, seeds=seeds)

    # (comoving D, luminosity distance, redshift)
    dist, lum_dist, z = dist_sampler(N_binaries, dmax=dmax)

    # masses
    if mass_exp_damp_flag:
        mass = mass_sampler_exp_damp(
            N_binaries=N_binaries, z_array=z,
            alpha_con=alpha_con, alpha_z=alpha_z,
            m_min=m_min, m_max=m_max,
            power_law=power_law,
            m_c_con=m_c_con, m_c_z=m_c_z
        )
    else:
        mass = mass_sampler(
            N_binaries, z,
            alpha_con, alpha_z,
            m_min, m_max
        )

    # convert mass → (total mass, chirp mass)
    tot_mass, chirp_mass = q_sampler(N_binaries, mass)

    # ------------------------------------------------------------
    # RANDOM ORIENTATION / ANGLES (vectorized)
    # ------------------------------------------------------------

    gw_ra  = rng.uniform(0, 2*np.pi, size=N_binaries)
    gw_dec = np.arcsin(rng.uniform(-1, 1, size=N_binaries))
    psi    = rng.uniform(0, np.pi, size=N_binaries)
    iota   = rng.uniform(0, np.pi, size=N_binaries)
    phi0   = rng.uniform(0, 2*np.pi, size=N_binaries)

    # ------------------------------------------------------------
    # STRAIN CALCULATION (optional)
    # ------------------------------------------------------------
    strain_data = None
    if compute_strain:
        # Compute h^2 for each binary
        h_square_circ = compute_h_square_circ_nb(fGW, chirp_mass, dist, z)
        
        # Bin the strain contributions
        bin_centres, h_c_total, h_c_contrib = bin_h_c_nb(fGW, h_square_circ, n_freq_bins)
        
        # Find which bin each binary belongs to
        f_min = np.min(fGW)
        f_max = np.max(fGW)
        bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_freq_bins+1)
        bin_assignment = np.digitize(fGW, bin_edges) - 1
        bin_assignment = np.clip(bin_assignment, 0, n_freq_bins-1)
        
        strain_data = {
            'bin_centres': bin_centres,
            'h_c_total': h_c_total,
            'h_square_individual': h_square_circ,
            'bin_assignment': bin_assignment,
            'h_c_individual': h_c_contrib,
            'bin_edges': bin_edges
        }

    # ------------------------------------------------------------
    # ASSEMBLE LIST OF DICTIONARIES
    # ------------------------------------------------------------
    population = []

    for i in range(N_binaries):
        pop_dict = {
            'Mc':   chirp_mass[i],
            'f':    fGW[i],
            'D_comov':    dist[i],   # comoving distance
            'z':    z[i],
            'ra':   gw_ra[i],
            'dec':  gw_dec[i],
            'psi':  psi[i],
            'iota': iota[i],
            'phi0': phi0[i],
            'Mtot': tot_mass[i]
        }
        
        # Add strain info if computed
        if compute_strain:
            pop_dict['h_square'] = h_square_circ[i]
            pop_dict['h_c_contrib'] = h_c_contrib[i]
            pop_dict['freq_bin'] = bin_assignment[i]
        
        population.append(pop_dict)

    if compute_strain:
        return population, strain_data
    else:
        return population