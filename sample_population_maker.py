import numpy as np
from SMBHB_pop_synth import (
    GRAVITATIONAL_CONSTANT, SOLAR_MASS_KG, COMOVING_DISTANCE_FN,
    compute_characteristic_strain_squared_circular,
    bin_characteristic_strain,
    YEAR_IN_SECONDS
)

def chosen_population(
    n_binaries,
    chirp_mass_msun=1e10,
    mass_ratio=0.5,
    gw_frequency=1e-8,
    redshift=0.5,
    polarization=0.0,
    inclination=0.0,
    initial_phase=0.0,
    right_ascension=0.0,
    declination=0.0,
    compute_strain=True,
    n_freq_bins=50,
    T_obs=15,
):
    """
    Generate a population of SMBHBs with fixed (user-specified) properties.

    Parameters
    ----------
    n_binaries : int
        Number of binaries to generate.
    chirp_mass_msun : float
        Chirp mass in solar masses (default: 1e10).
    mass_ratio : float
        Mass ratio q = m2/m1, must be in (0, 1] (default: 0.5).
    gw_frequency : float
        GW frequency in Hz (default: 1e-8).
    redshift : float
        Redshift of all binaries (default: 0.5).
    polarization : float
        Polarization angle in radians (default: 0.0).
    inclination : float
        Inclination angle in radians (default: 0.0).
    initial_phase : float
        Initial GW phase in radians (default: 0.0).
    right_ascension : float
        Right ascension in radians (default: 0.0).
    declination : float
        Declination in radians (default: 0.0).
    compute_strain : bool
        If True, compute characteristic strain (default: True).
    n_freq_bins : int
        Number of frequency bins (default: 50).
    T_obs : float
        Observation time in years (default: 15).

    Returns
    -------
    population : list of dict
        Same format as generate_smbhb_population.
    strain_data : dict (only if compute_strain=True)
        Same format as generate_smbhb_population.
    """

    # --- Masses ---
    chirp_masses = (chirp_mass_msun * SOLAR_MASS_KG) * np.ones(n_binaries)  # kg
    # Mtot from Mc and q: Mc = Mtot * q^(3/5) / (1+q)^(1/5)
    # => Mtot = Mc * (1+q)^(1/5) / q^(3/5)
    total_masses = chirp_masses * (1 + mass_ratio) ** (1/5) / mass_ratio ** (3/5)  # kg

    # --- Orbital parameters ---
    gw_frequencies  = gw_frequency  * np.ones(n_binaries)  # Hz
    redshifts       = redshift       * np.ones(n_binaries)
    comoving_dist   = COMOVING_DISTANCE_FN(redshift) * np.ones(n_binaries)  # Mpc (or metres, match your convention)
    polarizations   = polarization   * np.ones(n_binaries)  # rad
    inclinations    = inclination    * np.ones(n_binaries)  # rad
    initial_phases  = initial_phase  * np.ones(n_binaries)  # rad
    right_ascensions = right_ascension * np.ones(n_binaries)  # rad
    declinations    = declination    * np.ones(n_binaries)  # rad

    # --- Optional strain ---
    strain_data = None

    if compute_strain:
        h_squared = compute_characteristic_strain_squared_circular(
            gw_frequencies,
            chirp_masses,
            comoving_dist,
            redshifts,
            inclinations,
        )

        bin_centres, h_c_total, h_c_individual = bin_characteristic_strain(
            gw_frequencies,
            h_squared,
            n_freq_bins,
            T_obs=T_obs,
        )

        f_min  = 1.0 / (T_obs * YEAR_IN_SECONDS)
        f_max  = 3e-7
        f_step = 1.0 / (T_obs * YEAR_IN_SECONDS)
        N_bin_f = int((f_max - f_min) / f_step) + 1
        bin_edges = np.linspace(f_min, f_min + N_bin_f * f_step, N_bin_f + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        bin_assignment = np.digitize(gw_frequencies, bin_edges) - 1
        bin_assignment = np.clip(bin_assignment, 0, n_freq_bins - 1)

        strain_data = {
            'bin_centres':          bin_centres,
            'h_c_total':            h_c_total,
            'h_square_individual':  h_squared,
            'bin_assignment':       bin_assignment,
            'h_c_individual':       h_c_individual,
            'bin_edges':            bin_edges,
        }

    # --- Assemble catalog ---
    population = []
    for i in range(n_binaries):
        binary_params = {
            'Mc':      chirp_masses[i],
            'Mtot':    total_masses[i],
            'f':       gw_frequencies[i],
            'D_comov': comoving_dist[i],
            'z':       redshifts[i],
            'ra':      right_ascensions[i],
            'dec':     declinations[i],
            'psi':     polarizations[i],
            'iota':    inclinations[i],
            'phi0':    initial_phases[i],
        }
        if compute_strain:
            binary_params['h_square']    = h_squared[i]
            binary_params['h_c_contrib'] = h_c_individual[i]
            binary_params['freq_bin']    = bin_assignment[i]

        population.append(binary_params)

    return (population, strain_data) if compute_strain else population