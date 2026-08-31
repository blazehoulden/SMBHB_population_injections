from types import SimpleNamespace

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from SMBHB_pop_synth import PopulationArrays, chosen_population
from plot_cgw_snr import _aitoff_xy, _snr_colormap, _star_marker
import os 

# top of test_CGW_sky_loc.py, next to the sys.path fix
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def _concat_population_arrays(populations):
    """Concatenate a list of PopulationArrays into a single PopulationArrays."""
    if not populations:
        raise ValueError("No populations provided for concatenation.")

    fields = (
        "f", "Mc", "Mtot", "D_comov", "z", "h0",
        "ra", "dec", "psi", "iota", "phi0", "cgw_snr",   # cgw_snr added
    )
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
            cgw_snr=float(population.cgw_snr[i]),  # cgw_snr added
            
        )
        for i in range(len(population))
    ]

# ---------------------------------------------------------------------------
# Sky-sensitivity weight map — now loaded from disk instead of hardcoded.
#
# The 400-point _SKY_SURVEY table used to be a frozen, hand-pasted snapshot
# from an old noise model / pulsar timing array. That's fragile: every time
# the noise model, pulsar set, or Tspan changes, the table silently goes
# stale with no way to tell. Instead we now load the same kind of
# (ra, dec, snr) triples from disk, saved via curve_io.save_sky_snr_data(),
# and regenerate that file with regenerate_sky_survey.py whenever the
# pipeline changes.
#
# load_sky_snr_data() (curve_io.py) returns a plain dict of numpy arrays
# keyed by ra, dec, f, Mc, Mtot, D_comov, z, h0, psi, iota, phi0,
# cgw_snr_input, cgw_snrs_optimal, psr_phi, psr_theta -- confirmed directly
# against curve_io.py, no more guessing involved.
# ---------------------------------------------------------------------------
 
import numpy as np
from scipy.interpolate import LinearNDInterpolator
 
from curve_io import load_sky_snr_data
 
# Default location of the saved sky-SNR survey. Override by calling
# _build_sky_weight_interpolator(path=...) directly, or by setting
# SKY_SURVEY_PATH before this module's interpolator is first built.
SKY_SURVEY_PATH = "data/sky_snr/sky_survey"
 
 
def _load_sky_survey_arrays(path):
    """
    Load (ra, dec, snr) triples for the sky-sensitivity survey from disk.
 
    Pulls ra/dec/cgw_snrs_optimal out of the dict returned by
    load_sky_snr_data(). Note this is the *optimal* per-binary SNR computed
    post-injection (what test_sky_CGW_SNR_location actually ranks
    candidates by) -- not cgw_snr_input, which is just the value stamped
    onto the PopulationArrays before injection and isn't what we want the
    sky-sensitivity weighting to reflect.
    """
    loaded = load_sky_snr_data(path)
 
    ra = np.asarray(loaded["ra"], dtype=float)
    dec = np.asarray(loaded["dec"], dtype=float)
    snr = np.asarray(loaded["cgw_snrs_optimal"], dtype=float)
 
    if ra.shape != dec.shape or ra.shape != snr.shape:
        raise ValueError(
            f"Loaded sky survey arrays have mismatched shapes: "
            f"ra={ra.shape}, dec={dec.shape}, snr={snr.shape}."
        )
 
    return ra, dec, snr
 
 
def get_sky_survey_points(path=None):
    """
    Public accessor for the raw (ra, dec, snr) survey points, as a list of
    (ra, dec, snr) tuples -- i.e. a drop-in replacement for the old
    _SKY_SURVEY list for any code (like _best_sky_location()) that needs
    the raw points rather than the interpolated weight map.
    """
    ras, decs, snrs = _load_sky_survey_arrays(path or SKY_SURVEY_PATH)
    return list(zip(ras.tolist(), decs.tolist(), snrs.tolist()))
 
 
def _build_sky_weight_interpolator(path=None):
    """Build the RA/Dec -> weight interpolator from a saved sky survey."""
    ras, decs, snrs = _load_sky_survey_arrays(path or SKY_SURVEY_PATH)
 
    # Tile in RA to handle wrap-around at 0/2pi (unchanged from before).
    ras_tiled = np.concatenate([ras - 2 * np.pi, ras, ras + 2 * np.pi])
    decs_tiled = np.concatenate([decs, decs, decs])
    snrs_tiled = np.concatenate([snrs, snrs, snrs])
 
    weights = snrs_tiled / np.median(snrs)  # median-normalised, as before
 
    points = np.column_stack([ras_tiled, decs_tiled])
    interp = LinearNDInterpolator(points, weights, fill_value=1.0)
    return interp
 
 
# Interpolator is now built lazily on first use rather than at import time,
# so importing this module doesn't require the survey file to already
# exist on disk (e.g. right after a fresh checkout, before anyone has run
# regenerate_sky_survey.py yet).
_SKY_WEIGHT_INTERP = None
 
 
def sky_sensitivity_weight(ra_arr: np.ndarray, dec_arr: np.ndarray) -> np.ndarray:
    """
    Interpolated sky sensitivity weight at (ra, dec).
    Weight > 1  ->  hotspot; Weight < 1  ->  coldspot.
    Median sky position gives weight ~= 1.0.
    """
    global _SKY_WEIGHT_INTERP
    if _SKY_WEIGHT_INTERP is None:
        _SKY_WEIGHT_INTERP = _build_sky_weight_interpolator()
 
    points = np.column_stack([ra_arr, dec_arr])
    return _SKY_WEIGHT_INTERP(points).astype(np.float32)


# ---------------------------------------------------------------------------
# Save computed sky-SNR data -- lets the (slow) population-synth + CGW-SNR
# computation run once here, with plotting/iteration happening later from
# disk (e.g. in population_analysis.ipynb) without re-running
# test_sky_CGW_SNR_location.
#
# The actual save/load logic lives in curve_io.py (numpy only, no
# enterprise/hasasia/SMBHB_pop_synth dependency) so population_analysis.ipynb
# can `from curve_io import load_sky_snr_data` without dragging in
# everything this module needs to *compute* the data in the first place.
# ---------------------------------------------------------------------------

from curve_io import save_sky_snr_data


# NICER VERSION OF THE PLOTTING AND TESTING WITH TILING
def test_sky_CGW_SNR_location(psrs_clean, raw_noise_params, parsed_noise_params, Tspan,
                               save_data_path=None, make_plot=True):
    """
    Build a dense sky-grid test population as PopulationArrays, inject it,
    and compute per-source CGW optimal SNRs for tiled skymap rendering.

    save_data_path : if given, save the computed population/SNRs/pulsar
        positions to this path via save_sky_snr_data() -- e.g.
        'data/sky_snr/run01' -- so plotting/iteration can happen later
        from disk (population_analysis.ipynb) without re-running this
        (slow) function.
    make_plot : set False to skip plot_cgw_analysis entirely -- useful
        together with save_data_path when you only want to compute and
        save, not render a figure here.
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
 
    if save_data_path is not None:
        save_sky_snr_data(population, cgw_snrs_optimal, enterprise_psrs, save_data_path)

    if make_plot:
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
# Plotting: moved to sky_plot.py
#
# _wrap_ra, _STROKE, _style_skyax, plot_skymap, and plot_cgw_analysis used
# to be defined here, but none of them actually depend on SMBHB_pop_synth,
# CGW_SNR, enterprise, or hasasia -- only plain floats/arrays. They're now
# in sky_plot.py, which has no heavy compute-package imports, so
# population_analysis.ipynb (or anything else that just wants to re-plot
# already-computed sky-SNR data) can import plotting from there directly
# without triggering this module's SMBHB_pop_synth -> numba import chain.
# ---------------------------------------------------------------------------

from sky_plot import plot_cgw_analysis, plot_skymap, _style_skyax, _wrap_ra