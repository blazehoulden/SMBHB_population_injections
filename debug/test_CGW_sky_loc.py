from types import SimpleNamespace

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from SMBHB_pop_synth import PopulationArrays, chosen_population
from plot_cgw_snr import _aitoff_xy, _snr_colormap, _star_marker


def _concat_population_arrays(populations):
    """Concatenate a list of PopulationArrays into a single PopulationArrays."""
    if not populations:
        raise ValueError("No populations provided for concatenation.")

    fields = ("f", "Mc", "Mtot", "D_comov", "z", "h0", "ra", "dec", "psi", "iota", "phi0")
    merged = {
        field: np.concatenate([getattr(pop, field) for pop in populations])
        for field in fields
    }
    return PopulationArrays(**merged)


def _population_arrays_to_binary_rows(population):
    """Convert PopulationArrays to a list of per-binary attribute objects."""
    return [
        SimpleNamespace(
            f=float(population.f[i]),
            Mc=float(population.Mc[i]),
            Mtot=float(population.Mtot[i]),
            D_comov=float(population.D_comov[i]),
            z=float(population.z[i]),
            h0=float(population.h0[i]),
            ra=float(population.ra[i]),
            dec=float(population.dec[i]),
            psi=float(population.psi[i]),
            iota=float(population.iota[i]),
            phi0=float(population.phi0[i]),
        )
        for i in range(len(population))
    ]


# USED TO INFORM THE FILTERING OF THE SKY LOCATION DEPENDENCE FOR CGW ANALYSIS:
import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata

# ---------------------------------------------------------------------------
# Sky-sensitivity weight map (built once from your SNR survey, 400 points)
# Same h0/Mc/distance for all entries, so SNR variation is purely sky position.
# We normalise so the median weight = 1.0, keeping the proxy scale meaningful.
# ---------------------------------------------------------------------------
_SKY_SURVEY = [
    (5.03, -0.08, 20.88), (5.03, -0.25, 20.84), (5.03,  0.08, 20.81),
    (4.71, -0.08, 20.72), (4.71,  0.08, 20.59), (4.71, -0.25, 20.59),
    (4.71, -0.41, 20.49), (5.03,  0.25, 20.44), (5.03, -0.41, 20.42),
    (4.71,  0.25, 20.35), (5.03,  0.41, 20.02), (4.71, -0.58, 19.97),
    (4.40,  0.08, 19.93), (5.34,  0.08, 19.92), (5.03, -0.58, 19.92),
    (5.34, -0.08, 19.89), (5.34, -0.25, 19.86), (4.40, -0.08, 19.82),
    (4.40, -0.25, 19.75), (4.71,  0.41, 19.69), (5.34,  0.25, 19.62),
    (5.34, -0.41, 19.60), (4.40,  0.25, 19.49), (4.40, -0.41, 19.46),
    (4.40,  0.41, 19.32), (5.03,  0.58, 19.30), (4.71, -0.74, 19.26),
    (5.03, -0.74, 19.17), (5.34,  0.41, 19.14), (5.34, -0.58, 19.13),
    (4.40, -0.58, 19.01), (4.71,  0.58, 18.97), (4.40,  0.58, 18.65),
    (4.08, -0.08, 18.64), (4.08,  0.08, 18.56), (4.08, -0.25, 18.53),
    (5.34, -0.74, 18.44), (5.65,  0.08, 18.44), (5.34,  0.58, 18.39),
    (5.65, -0.08, 18.37), (4.40, -0.74, 18.36), (4.71, -0.91, 18.32),
    (5.65, -0.25, 18.31), (5.03,  0.74, 18.31), (5.65,  0.25, 18.25),
    (5.03, -0.91, 18.24), (4.08, -0.41, 18.22), (5.65, -0.41, 18.18),
    (4.71,  0.74, 18.18), (4.08,  0.25, 18.11), (5.65, -0.58, 17.84),
    (4.08, -0.58, 17.82), (5.65,  0.41, 17.81), (4.40,  0.74, 17.77),
    (5.34,  0.74, 17.65), (5.34, -0.91, 17.62), (4.40, -0.91, 17.61),
    (4.08,  0.41, 17.59), (5.65,  0.58, 17.38), (5.65, -0.74, 17.33),
    (4.08, -0.74, 17.26), (4.08,  0.58, 17.24), (4.71, -1.07, 17.20),
    (5.03,  0.91, 17.20), (4.71,  0.91, 17.17), (5.03, -1.07, 17.16),
    (4.40,  0.91, 16.79), (5.97, -0.08, 16.77), (5.34,  0.91, 16.76),
    (5.65,  0.74, 16.75), (5.97,  0.08, 16.71), (5.65, -0.91, 16.69),
    (4.40, -1.07, 16.69), (5.97, -0.25, 16.68), (5.34, -1.07, 16.68),
    (4.08,  0.74, 16.64), (3.77, -0.25, 16.63), (4.08, -0.91, 16.62),
    (3.77, -0.08, 16.60), (3.77, -0.41, 16.54), (5.97, -0.41, 16.51),
    (3.77,  0.08, 16.42), (5.97,  0.25, 16.37), (3.77, -0.58, 16.33),
    (5.97, -0.58, 16.27), (3.77,  0.25, 16.09), (5.03,  1.07, 16.03),
    (5.65,  0.91, 16.02), (4.71,  1.07, 16.02), (5.97,  0.41, 16.00),
    (4.71, -1.24, 15.98), (5.03, -1.24, 15.97), (5.65, -1.07, 15.95),
    (5.97, -0.74, 15.95), (3.77, -0.74, 15.94), (4.08, -1.07, 15.89),
    (4.08,  0.91, 15.88), (5.34,  1.07, 15.76), (4.40,  1.07, 15.73),
    (3.77,  0.41, 15.69), (5.34, -1.24, 15.64), (4.40, -1.24, 15.63),
    (5.97,  0.58, 15.53), (5.97, -0.91, 15.53), (3.77, -0.91, 15.50),
    (3.77,  0.58, 15.33), (5.65,  1.07, 15.22), (5.97,  0.74, 15.22),
    (5.65, -1.24, 15.12), (4.08, -1.24, 15.07), (4.08,  1.07, 15.06),
    (5.97, -1.07, 15.03), (3.77, -1.07, 15.01), (3.77,  0.74, 14.96),
    (5.03,  1.24, 14.81), (4.71,  1.24, 14.80), (5.97,  0.91, 14.78),
    (5.03, -1.41, 14.70), (5.34,  1.24, 14.69), (4.71, -1.41, 14.69),
    (4.40,  1.24, 14.63), (3.77,  0.91, 14.52), (5.34, -1.41, 14.52),
    (3.77, -1.24, 14.48), (4.40, -1.41, 14.47), (5.97, -1.24, 14.47),
    (0.00, -0.08, 14.44), (0.00, -0.25, 14.43), (3.46, -0.58, 14.40),
    (0.00, -0.41, 14.40), (3.46, -0.41, 14.38), (0.00,  0.08, 14.37),
    (5.65,  1.24, 14.35), (0.00, -0.58, 14.33), (3.46, -0.74, 14.33),
    (3.46, -0.25, 14.29), (5.97,  1.07, 14.24), (0.00, -0.74, 14.23),
    (5.65, -1.41, 14.22), (0.00,  0.25, 14.21), (4.08,  1.24, 14.20),
    (3.46, -0.91, 14.19), (4.08, -1.41, 14.18), (0.00,  0.41, 14.17),
    (3.46, -0.08, 14.16), (0.00, -0.91, 14.07), (3.46, -1.07, 14.03),
    (3.77,  1.07, 14.02), (3.46,  0.08, 13.99), (3.77, -1.41, 13.92),
    (0.00,  0.58, 13.88), (0.00, -1.07, 13.87), (5.97, -1.41, 13.87),
    (3.46, -1.24, 13.84), (3.46,  0.25, 13.78), (0.00, -1.24, 13.67),
    (5.97,  1.24, 13.65), (3.46, -1.41, 13.64), (5.34,  1.41, 13.59),
    (5.03,  1.41, 13.58), (4.71,  1.41, 13.56), (3.46,  0.41, 13.55),
    (0.00,  0.74, 13.53), (4.40,  1.41, 13.52), (0.00, -1.41, 13.50),
    (3.77,  1.24, 13.48), (5.65,  1.41, 13.43), (0.31, -1.57, 13.41),
    (3.46, -1.57, 13.41), (1.88, -1.57, 13.41), (5.03, -1.57, 13.41),
    (4.71, -1.57, 13.36), (3.14, -1.57, 13.36), (1.57, -1.57, 13.36),
    (0.00, -1.57, 13.36), (2.20, -1.57, 13.34), (0.63, -1.57, 13.34),
    (3.77, -1.57, 13.34), (5.34, -1.57, 13.34), (4.08,  1.41, 13.34),
    (3.46,  0.58, 13.33), (1.26, -1.57, 13.27), (5.97, -1.57, 13.27),
    (2.83, -1.57, 13.27), (4.40, -1.57, 13.27), (0.00,  0.91, 13.26),
    (2.51, -1.57, 13.26), (0.94, -1.57, 13.26), (4.08, -1.57, 13.26),
    (5.65, -1.57, 13.26), (3.14, -1.41, 13.22), (0.31, -1.41, 13.15),
    (3.46,  0.74, 13.14), (3.14, -1.24, 13.06), (5.97,  1.41, 13.03),
    (0.00,  1.07, 13.01), (3.46,  0.91, 12.97), (3.77,  1.41, 12.95),
    (3.14, -1.07, 12.88), (0.31, -1.24, 12.86), (3.46,  1.07, 12.80),
    (0.00,  1.24, 12.78), (0.63, -1.41, 12.77), (2.83, -1.41, 12.70),
    (3.14, -0.91, 12.69), (3.46,  1.24, 12.63), (0.31, -1.07, 12.61),
    (0.00,  1.41, 12.56), (5.65,  1.57, 12.49), (2.51,  1.57, 12.49),
    (4.08,  1.57, 12.49), (0.94,  1.57, 12.49), (3.14, -0.74, 12.49),
    (0.31, -0.91, 12.47), (3.46,  1.41, 12.47), (2.20,  1.57, 12.47),
    (3.77,  1.57, 12.47), (5.34,  1.57, 12.47), (0.63,  1.57, 12.47),
    (1.26,  1.57, 12.42), (5.97,  1.57, 12.42), (2.83,  1.57, 12.42),
    (4.40,  1.57, 12.42), (0.31, -0.74, 12.40), (0.31,  1.57, 12.37),
    (1.88,  1.57, 12.37), (3.46,  1.57, 12.37), (5.03,  1.57, 12.37),
    (0.94, -1.41, 12.34), (4.71,  1.57, 12.34), (0.00,  1.57, 12.34),
    (1.57,  1.57, 12.34), (3.14,  1.57, 12.34), (0.31, -0.58, 12.32),
    (3.14, -0.58, 12.29), (2.51, -1.41, 12.28), (0.31,  1.41, 12.27),
    (0.31, -0.41, 12.24), (0.31, -0.25, 12.18), (0.63, -1.24, 12.18),
    (2.83, -1.24, 12.16), (0.31, -0.08, 12.14), (0.31,  1.24, 12.14),
    (0.31,  0.58, 12.14), (0.31,  0.41, 12.13), (2.20, -1.41, 12.13),
    (1.88, -1.41, 12.12), (0.31,  0.74, 12.10), (0.31,  0.25, 12.10),
    (0.63,  1.41, 12.08), (3.14,  1.41, 12.08), (3.14, -0.41, 12.07),
    (1.26, -1.41, 12.07), (1.57, -1.41, 12.06), (0.31,  1.07, 12.06),
    (0.31,  0.91, 12.05), (0.31,  0.08, 12.03), (3.14, -0.25, 11.85),
    (3.14,  1.24, 11.84), (2.83,  1.41, 11.83), (0.94,  1.41, 11.75),
    (0.63,  1.24, 11.69), (3.14,  1.07, 11.68), (3.14, -0.08, 11.65),
    (2.83, -1.07, 11.64), (2.51,  1.41, 11.60), (0.63, -1.07, 11.54),
    (3.14,  0.91, 11.53), (3.14,  0.08, 11.52), (0.94, -1.24, 11.50),
    (3.14,  0.25, 11.44), (3.14,  0.74, 11.42), (3.14,  0.41, 11.38),
    (3.14,  0.58, 11.37), (2.51, -1.24, 11.36), (2.20,  1.41, 11.36),
    (1.26,  1.41, 11.33), (0.63,  1.07, 11.29), (1.88,  1.41, 11.24),
    (2.83,  1.24, 11.21), (1.57,  1.41, 11.17), (0.94,  1.24, 11.15),
    (2.83, -0.91, 11.14), (0.63, -0.91, 10.98), (0.63,  0.91, 10.95),
    (1.26, -1.24, 10.91), (2.20, -1.24, 10.91), (1.57, -1.24, 10.85),
    (1.88, -1.24, 10.84), (0.94, -1.07, 10.75), (2.83,  1.07, 10.74),
    (0.63, -0.74, 10.72), (2.51,  1.24, 10.72), (0.63,  0.74, 10.69),
    (2.83, -0.74, 10.65), (2.51, -1.07, 10.54), (0.63, -0.58, 10.51),
    (0.94,  1.07, 10.50), (0.63,  0.58, 10.47), (1.26,  1.24, 10.45),
    (2.20,  1.24, 10.36), (2.83,  0.91, 10.31), (0.63, -0.41, 10.28),
    (0.63,  0.41, 10.24), (2.83, -0.58, 10.21), (1.88,  1.24, 10.21),
    (0.63, -0.25, 10.09), (0.63,  0.25, 10.06), (1.57,  1.24,  9.99),
    (0.63, -0.08,  9.98), (2.51,  1.07,  9.96), (0.63,  0.08,  9.96),
    (2.83,  0.74,  9.91), (0.94, -0.91,  9.87), (2.83, -0.41,  9.85),
    (1.26, -1.07,  9.85), (0.94,  0.91,  9.84), (2.51, -0.91,  9.81),
    (2.20, -1.07,  9.78), (1.26,  1.07,  9.78), (1.57, -1.07,  9.74),
    (2.83, -0.25,  9.61), (2.83,  0.58,  9.58), (1.88, -1.07,  9.53),
    (2.20,  1.07,  9.40), (0.94, -0.74,  9.40), (2.51,  0.91,  9.38),
    (2.83, -0.08,  9.37), (2.83,  0.41,  9.36), (0.94,  0.74,  9.33),
    (2.83,  0.08,  9.28), (2.83,  0.25,  9.25), (2.51, -0.74,  9.17),
    (1.88,  1.07,  9.15), (1.57,  1.07,  9.14), (1.26, -0.91,  9.02),
    (0.94,  0.58,  8.96), (0.94, -0.58,  8.95), (1.26,  0.91,  8.93),
    (2.20, -0.91,  8.86), (2.51,  0.74,  8.75), (2.20,  0.91,  8.72),
    (0.94,  0.41,  8.65), (2.51, -0.58,  8.62), (0.94, -0.41,  8.52),
    (1.57, -0.91,  8.49), (0.94,  0.25,  8.42), (1.57,  0.91,  8.39),
    (1.88, -0.91,  8.38), (1.26, -0.74,  8.33), (0.94, -0.25,  8.28),
    (1.26,  0.74,  8.26), (0.94,  0.08,  8.26), (1.88,  0.91,  8.24),
    (2.51,  0.58,  8.21), (0.94, -0.08,  8.20), (2.51, -0.41,  8.19),
    (2.20, -0.74,  8.14), (2.20,  0.74,  8.00), (1.26,  0.58,  7.85),
    (2.51,  0.41,  7.82), (2.51, -0.25,  7.79), (1.57, -0.74,  7.73),
    (1.57,  0.74,  7.72), (1.88, -0.74,  7.63), (1.26, -0.58,  7.63),
    (2.51,  0.25,  7.60), (2.51, -0.08,  7.57), (2.51,  0.08,  7.56),
    (2.20, -0.58,  7.56), (1.26,  0.41,  7.55), (1.88,  0.74,  7.51),
    (2.20,  0.58,  7.37), (1.26,  0.25,  7.28), (1.57, -0.58,  7.27),
    (1.57,  0.58,  7.26), (1.26, -0.41,  7.24), (1.88, -0.58,  7.09),
    (1.26,  0.08,  7.08), (2.20, -0.41,  7.07), (1.26, -0.25,  7.03),
    (1.88,  0.58,  7.00), (1.26, -0.08,  6.97), (2.20,  0.41,  6.95),
    (1.57,  0.41,  6.89), (1.57, -0.41,  6.79), (2.20, -0.25,  6.68),
    (2.20,  0.25,  6.68), (1.88, -0.41,  6.66), (1.88,  0.41,  6.65),
    (1.57,  0.25,  6.62), (2.20,  0.08,  6.53), (2.20, -0.08,  6.51),
    (1.57, -0.25,  6.51), (1.57,  0.08,  6.46), (1.57, -0.08,  6.42),
    (1.88,  0.25,  6.40), (1.88, -0.25,  6.37), (1.88,  0.08,  6.23),
    (1.88, -0.08,  6.21),
]

def _build_sky_weight_interpolator():
    ras   = np.array([r for r, d, _ in _SKY_SURVEY])
    decs  = np.array([d for _, d, _ in _SKY_SURVEY])
    snrs  = np.array([s for _, _, s in _SKY_SURVEY])

    # Tile in RA to handle wrap-around at 0/2π
    ras_tiled  = np.concatenate([ras - 2*np.pi, ras, ras + 2*np.pi])
    decs_tiled = np.concatenate([decs, decs, decs])
    snrs_tiled = np.concatenate([snrs, snrs, snrs])

    weights = snrs_tiled / np.median(snrs)  # median-normalised

    points = np.column_stack([ras_tiled, decs_tiled])
    interp = LinearNDInterpolator(points, weights, fill_value=1.0)
    return interp

_SKY_WEIGHT_INTERP = _build_sky_weight_interpolator()


def sky_sensitivity_weight(ra_arr: np.ndarray, dec_arr: np.ndarray) -> np.ndarray:
    """
    Interpolated sky sensitivity weight at (ra, dec).
    Weight > 1  →  hotspot; Weight < 1  →  coldspot.
    Median sky position gives weight ≈ 1.0.
    """
    points = np.column_stack([ra_arr, dec_arr])
    return _SKY_WEIGHT_INTERP(points).astype(np.float32)




# NICER VERSION OF THE PLOTTING AND TESTING WITH TILING
def test_sky_CGW_SNR_location(psrs_clean, raw_noise_params, parsed_noise_params, Tspan):
    """
    Build a dense sky-grid test population as PopulationArrays, inject it,
    and compute per-source CGW optimal SNRs for tiled skymap rendering.
    """
    import os
    os.makedirs("figures", exist_ok=True)
 
    # Use a 72×36 grid (lon × lat) for smooth tiling coverage
    n_ra  = 72
    n_dec = 36
 
    dec_list = np.linspace(-np.pi / 2, np.pi / 2, n_dec)
    ra_list  = np.linspace(0, 2 * np.pi, n_ra, endpoint=False)
    sky_locations = [(ra, dec) for ra in ra_list for dec in dec_list]
 
    sub_populations = []
    for ra, dec in sky_locations:
        pop = chosen_population(
            n_binaries=1,
            chirp_mass_msun=10**9,
            right_ascension=float(ra),
            declination=float(dec),
            compute_strain=False,
            T_obs_seconds=Tspan,
        )
        sub_populations.append(pop)
    population = _concat_population_arrays(sub_populations)
 
    from consistent_pop_synth import compute_population_snr
 
    original_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}
    _, pta, enterprise_psrs = compute_population_snr(
        population=population,
        psrs_clean=psrs_clean,
        current_stoas=original_stoas,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan,
        return_psrs_pta=True,
    )
 
    from CGW_SNR import compute_cgw_snr_optimal_population
 
    binary_rows = _population_arrays_to_binary_rows(population)
    cgw_snrs_optimal = compute_cgw_snr_optimal_population(
        psrs=enterprise_psrs,
        pta=pta,
        population=binary_rows,
        Tspan=Tspan,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
    )
 
    plot_cgw_analysis(
        binaries=binary_rows,
        snrs=cgw_snrs_optimal,
        psrs=enterprise_psrs,
        n_ra=n_ra,
        n_dec=n_dec,
        save_path="figures/cgw_snr_sky_map_comp.pdf",
    )
 
    print("\nRanked CGW candidates by optimal SNR:")
    for i, (binary, snr) in enumerate(
        sorted(zip(binary_rows, cgw_snrs_optimal), key=lambda x: x[1], reverse=True)
    ):
        print(f"{i+1:2d}. SNR={snr:.2f}, RA={binary.ra:.2f} rad, Dec={binary.dec:.2f} rad")
 
    return population, cgw_snrs_optimal
 
 
# ---------------------------------------------------------------------------
# Skymap panel
# ---------------------------------------------------------------------------
 
def plot_skymap(ax, binaries, snrs, psrs, cmap, norm, n_ra=72, n_dec=36):
    """
    Aitoff-projection tiled SNR heatmap.
 
    The SNR values at the sampled grid points are interpolated onto a fine
    regular (RA, Dec) mesh and rendered as a continuous heatmap via
    ``pcolormesh``.  Pulsars are overlaid as white stars with black outlines.
    """
    snra     = np.array(snrs, dtype=float)
    star_path = _star_marker()
 
    # ------------------------------------------------------------------
    # 1. Build the fine evaluation grid in Aitoff-projected coordinates
    # ------------------------------------------------------------------
    # Native (RA, Dec) grid — same resolution as the sampled population
    ra_vals  = np.array([b.ra  for b in binaries])
    dec_vals = np.array([b.dec for b in binaries])
 
    # Dense grid for smooth rendering
    ra_fine  = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    dec_fine = np.linspace(-np.pi / 2, np.pi / 2, 180)
    RA_fine, DEC_fine = np.meshgrid(ra_fine, dec_fine)
 
    # Interpolate SNR from the sampled points onto the fine grid
    snr_grid = griddata(
        points=np.column_stack([ra_vals, dec_vals]),
        values=snra,
        xi=np.column_stack([RA_fine.ravel(), DEC_fine.ravel()]),
        method="linear",
    ).reshape(RA_fine.shape)
 
    # Fill any NaN edges (convex-hull boundary artefacts) with nearest value
    snr_grid_nn = griddata(
        points=np.column_stack([ra_vals, dec_vals]),
        values=snra,
        xi=np.column_stack([RA_fine.ravel(), DEC_fine.ravel()]),
        method="nearest",
    ).reshape(RA_fine.shape)
    snr_grid = np.where(np.isnan(snr_grid), snr_grid_nn, snr_grid)
 
    # ------------------------------------------------------------------
    # 2. Project onto Aitoff coordinates and render with pcolormesh
    # ------------------------------------------------------------------
    # pcolormesh needs corner coordinates → use cell-edge grid
    ra_edges  = np.linspace(0, 2 * np.pi, 361, endpoint=True)
    dec_edges = np.linspace(-np.pi / 2, np.pi / 2, 181)
    RA_e, DEC_e = np.meshgrid(ra_edges, dec_edges)
    X_e, Y_e = _aitoff_xy(RA_e, DEC_e)
    X_e = -X_e                          # flip to match NANOGrav convention

 
    ax.pcolormesh(
        X_e, Y_e, snr_grid,
        cmap=cmap, norm=norm,
        shading="auto", zorder=1, rasterized=True,
    )
 
    # ------------------------------------------------------------------
    # 3. Grid lines
    # ------------------------------------------------------------------
    ra_grid  = np.linspace(0, 2 * np.pi, 500)
    dec_grid = np.linspace(-np.pi / 2, np.pi / 2, 500)
    grid_color = "0.55"
 
    for dec_val in np.radians([-60, -30, 0, 30, 60]):
        xs, ys = _aitoff_xy(ra_grid, np.full_like(ra_grid, dec_val))
        xs = -xs  # flip to match NANOGrav convention
        ax.plot(xs, ys, color=grid_color, lw=0.4, zorder=2)
 
    for ra_val in np.linspace(0, 2 * np.pi, 13)[:-1]:
        xs, ys = _aitoff_xy(np.full_like(dec_grid, ra_val), dec_grid)
        xs = -xs  # flip to match NANOGrav convention
        ax.plot(xs, ys, color=grid_color, lw=0.4, zorder=2)
 
    # Equator
    xs, ys = _aitoff_xy(ra_grid, np.zeros_like(ra_grid))
    xs = -xs  # flip to match NANOGrav convention
    ax.plot(xs, ys, color="0.35", lw=0.8, zorder=2)
 
    # Projection boundary
    for ra_bnd in [0, 2 * np.pi]:
        xs, ys = _aitoff_xy(np.full_like(dec_grid, ra_bnd), dec_grid)
        xs = -xs  # flip to match NANOGrav convention
        ax.plot(xs, ys, color="0.35", lw=1.0, zorder=3)


    # ------------------------------------------------------------------
    # 3b. Galactic plane
    # ------------------------------------------------------------------
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    l_vals = np.linspace(0, 360, 1000)
    gal = SkyCoord(l=l_vals * u.deg, b=np.zeros(1000) * u.deg, frame="galactic")
    eq  = gal.icrs

    ra_gal  = np.deg2rad(eq.ra.deg)
    dec_gal = np.deg2rad(eq.dec.deg)

    # Sort by RA so the line doesn't jump across the projection boundary
    order   = np.argsort(ra_gal)
    ra_gal  = ra_gal[order]
    dec_gal = dec_gal[order]

    # Project and flip (matching the horizontal flip applied everywhere else)
    xs_gal, ys_gal = _aitoff_xy(ra_gal, dec_gal)
    xs_gal = -xs_gal

    # Split at discontinuities caused by the RA=0/2π wrap-around
    gaps = np.where(np.abs(np.diff(xs_gal)) > 0.5)[0] + 1
    segs = np.split(np.column_stack([xs_gal, ys_gal]), gaps)

    for seg in segs:
        ax.plot(seg[:, 0], seg[:, 1],
                color="0.45", lw=2.0, zorder=4, solid_capstyle="round")
 
    # ------------------------------------------------------------------
    # 4. Pulsars — white fill, black outline
    # ------------------------------------------------------------------
    for psr in psrs:
        ra_psr  = float(psr.phi)
        dec_psr = float(np.pi / 2.0 - psr.theta)
        px, py  = _aitoff_xy(ra_psr, dec_psr)
        px = -px  # flip to match NANOGrav convention
        ax.scatter(
            px, py,
            marker=star_path, s=100,
            facecolors="white", edgecolors="black", linewidths=0.8,
            zorder=6,
        )
 
    # ------------------------------------------------------------------
    # 5. Legend
    # ------------------------------------------------------------------
    from matplotlib.lines import Line2D
    handles = [
        Line2D(
            [0], [0], marker=star_path, color="w",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=0.8, markersize=9, label="Pulsar",
        ),
        Line2D(
            [0], [0], color="0.45", lw=2.0, solid_capstyle="round",
            label="Galactic plane",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.7, edgecolor="0.4")
 
    ax.set_axis_off()
 
 
# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------
 
def plot_cgw_analysis(
    binaries,
    snrs,
    psrs,
    n_ra=72,
    n_dec=36,
    save_path=None,
    figsize=(7, 4),
    annotate_top=0,
):
    """
    Generate a paper-ready single-panel Aitoff SNR skymap.
 
    Parameters
    ----------
    binaries : list
        Binary objects (must have .ra, .dec).
    snrs : list or array-like
        Optimal SNR values matching binaries order.
    psrs : list
        enterprise Pulsar objects (must have .phi, .theta).
    n_ra, n_dec : int
        Grid dimensions used to sample the population (passed for reference).
    save_path : str or None
        If given, saves to this path.  If None, calls plt.show().
    figsize : tuple
    annotate_top : int
        Number of top-SNR sources to label (0 = none).
    """
    snra = np.array(snrs, dtype=float)
    cmap = _snr_colormap()
    norm = Normalize(vmin=snra.min() * 0.9, vmax=snra.max() * 1.05)
 
    # White background, paper-ready style
    with plt.style.context("default"):
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "font.family":      "serif",
            "font.size":        10,
        })
 
        fig = plt.figure(figsize=figsize, constrained_layout=False)
 
        gs = gridspec.GridSpec(
            1, 1, figure=fig,
            left=0.03, right=0.91, top=0.97, bottom=0.03,
        )
        ax_sky = fig.add_subplot(gs[0, 0])
        ax_sky.set_facecolor("white")
 
        plot_skymap(ax_sky, binaries, snra, psrs, cmap, norm, n_ra=n_ra, n_dec=n_dec)
 
        # Colourbar on the right
        cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Optimal SNR", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
 
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            print(f"Saved to {save_path}")
        else:
            plt.show()
 
    return fig
 