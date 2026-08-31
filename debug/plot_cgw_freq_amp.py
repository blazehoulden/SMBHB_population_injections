"""
plot_cgw_freq_amp_continuous.py

Same as plot_cgw_freq_amp.py (GW frequency f vs strain amplitude h0
detectability, fixed sky position / chirp mass CW source), but with a
single continuous S/N colour bar instead of discrete threshold bins.

Everything upstream of plotting (h0(z) inversion, grid building, S/N
computation) is byte-for-byte identical to plot_cgw_freq_amp.py -- only
_discrete_snr_cmap / plot_freq_amp_analysis differ:
  - _discrete_snr_cmap + BoundaryNorm/ListedColormap  ->  plain cmap + LogNorm
  - no threshold list / tick-labels-with-"+"; colour bar just reads off
    the continuous S/N value at any point on the grid.

LogNorm is used (not Normalize) because S/N here typically spans several
orders of magnitude across the grid, and a linear colour scale would
crush all the low-S/N structure into a single dark colour at the low end.
If your S/N range is actually narrow, switch norm=LogNorm(...) to
norm=Normalize(...) below.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import brentq

from apj_style import apply_apj_style, APJ_COL_WIDTH
from SMBHB_pop_synth import PopulationArrays, chosen_population, compute_strain_amplitude, _Z_GRID, _CHI_GRID
from plot_cgw_snr import _snr_colormap

# Reuse the helpers already defined for the sky-map script rather than
# duplicating them -- adjust this import path/module name to wherever
# _concat_population_arrays / _population_arrays_to_binary_rows /
# get_sky_survey_points actually live in your repo.
from debug.test_CGW_sky_loc import (
    _concat_population_arrays,
    _population_arrays_to_binary_rows,
)
from debug.test_CGW_sky_loc import get_sky_survey_points  # or wherever you put the patch

def _best_sky_location():
    """
    Most sensitive (ra, dec) from the sky-sensitivity survey used to build
    the interpolated weight map in the sky-map script. Because that map is
    built with *linear* interpolation, its maximum over the whole sky is
    necessarily attained at one of the sampled vertices -- so the argmax
    over the raw survey points, not the interpolator, is the true best
    sky location (no optimisation loop needed).
    """
    survey = get_sky_survey_points()
    ra_best, dec_best, snr_best = max(survey, key=lambda entry: entry[2])
    return ra_best, dec_best


# ---------------------------------------------------------------------------
# h0(z) inversion via chosen_population's own strain calculation
# ---------------------------------------------------------------------------

def _h0_at_redshift(f_hz, z, chirp_mass_msun, iota):
    """h0 for a single (f, z) point, using the pipeline's own strain calc."""
    D_comov = np.interp(z, _Z_GRID, _CHI_GRID)
    _, h0 = compute_strain_amplitude(
        np.array([f_hz]), np.array([chirp_mass_msun]),
        np.array([D_comov]), np.array([z]), np.array([iota]),
    )
    return float(h0[0])


# The cosmology lookup used inside chosen_population only covers
# z in [_Z_GRID.min(), _Z_GRID.max()] -- np.interp does NOT extrapolate
# beyond that, it just returns the edge value flat, so h0 stops changing
# past those bounds. Root-finding must stay within this range.
_Z_MIN_SAFE = max(float(_Z_GRID.min()), 1e-6)  # avoid z=0 exactly (D_comov=0)
_Z_MAX_SAFE = float(_Z_GRID.max())


def _reachable_h0_range(f_hz, chirp_mass_msun, iota):
    """
    (h0_max, h0_min) reachable at this (f, Mc, iota) across the full
    z range the cosmology lookup covers -- h0_max at the closest usable
    redshift, h0_min at the farthest.
    """
    h0_max = _h0_at_redshift(f_hz, _Z_MIN_SAFE, chirp_mass_msun, iota)
    h0_min = _h0_at_redshift(f_hz, _Z_MAX_SAFE, chirp_mass_msun, iota)
    return h0_max, h0_min


def _redshift_for_target_h0(f_hz, target_h0, chirp_mass_msun, iota):
    """
    Root-find the redshift z such that h0(f, z) == target_h0, at fixed
    (Mc, iota), within z in [_Z_MIN_SAFE, _Z_MAX_SAFE] -- the range the
    cosmology lookup actually covers. h0 decreases monotonically with z
    (larger D_comov), so the reachable range at fixed (f, Mc, iota) is
    bounded above by z -> _Z_MIN_SAFE and below by z -> _Z_MAX_SAFE.
    Raises a clear ValueError (with the actual reachable range) if
    target_h0 falls outside that, rather than looping or silently
    extrapolating.
    """
    def resid(z):
        return _h0_at_redshift(f_hz, z, chirp_mass_msun, iota) - target_h0

    r_lo, r_hi = resid(_Z_MIN_SAFE), resid(_Z_MAX_SAFE)

    if r_lo < 0:
        h0_max_reachable = r_lo + target_h0
        raise ValueError(
            f"target h0={target_h0:.3e} at f={f_hz:.3e} Hz exceeds the max "
            f"reachable h0={h0_max_reachable:.3e} (at z={_Z_MIN_SAFE:.2e}, "
            "the closest usable redshift) for this chirp mass/inclination -- "
            "lower h0_max or increase chirp_mass_msun."
        )
    if r_hi > 0:
        h0_min_reachable = r_hi + target_h0
        raise ValueError(
            f"target h0={target_h0:.3e} at f={f_hz:.3e} Hz is below the min "
            f"reachable h0={h0_min_reachable:.3e} (at z={_Z_MAX_SAFE:.2e}, "
            "the farthest usable redshift) for this chirp mass/inclination -- "
            "raise h0_min or increase chirp_mass_msun."
        )

    return brentq(resid, _Z_MIN_SAFE, _Z_MAX_SAFE)


def suggest_h0_bounds(freqs_hz, chirp_mass_msun, iota=0.0):
    """
    For every frequency in freqs_hz, compute the h0 range reachable
    across the cosmology lookup's full redshift span (see
    _reachable_h0_range), and return the intersection across all of them:
    an (h0_min, h0_max) pair guaranteed reachable at *every* frequency in
    the grid. Call this first to pick sensible h0_min/h0_max for
    compute_freq_h0_snr_grid / test_freq_amp_CGW_SNR before running the
    (slow) full S/N grid, e.g.:

        freqs_hz = np.geomspace(1e-9, 1e-7, 40)
        h0_min, h0_max = suggest_h0_bounds(freqs_hz, chirp_mass_msun=1e9)
    """
    h0_maxes, h0_mins = [], []
    for f_val in freqs_hz:
        h0_max, h0_min = _reachable_h0_range(f_val, chirp_mass_msun, iota)
        h0_maxes.append(h0_max)
        h0_mins.append(h0_min)
    return max(h0_mins), min(h0_maxes)


# ---------------------------------------------------------------------------
# Grid population builder
# ---------------------------------------------------------------------------

def _build_freq_h0_grid_population(
    chirp_mass_msun,
    freqs_hz,
    h0_targets,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
):
    """
    Build a PopulationArrays with one binary per (f, h0) grid point, all
    sharing the same sky position, chirp mass, mass ratio, and
    inclination. For each grid point, the redshift is root-found so the
    resulting h0 matches the target (see _redshift_for_target_h0).

    Returns (population, grid_shape, ra, dec, h0_actual) where h0_actual
    is the (n_f, n_h0) array of h0 values actually realised (read back
    from the population, for exact self-consistency with the pipeline).
    """
    if ra is None or dec is None:
        best_ra, best_dec = _best_sky_location()
        ra = best_ra if ra is None else ra
        dec = best_dec if dec is None else dec

    n_f, n_h0 = len(freqs_hz), len(h0_targets)
    sub_populations = []
    h0_actual = np.empty((n_f, n_h0), dtype=float)

    for i, f_val in enumerate(freqs_hz):
        for j, h0_target in enumerate(h0_targets):
            z_val = _redshift_for_target_h0(f_val, h0_target, chirp_mass_msun, iota)
            pop = chosen_population(
                n_binaries=1,
                T_obs_seconds=None,          # unused when compute_strain=False
                chirp_mass_msun=chirp_mass_msun,
                mass_ratio=mass_ratio,
                gw_frequency=float(f_val),
                redshift=float(z_val),
                polarization=0.0,
                inclination=float(iota),
                initial_phase=0.0,
                right_ascension=float(ra),
                declination=float(dec),
                compute_strain=False,
            )
            sub_populations.append(pop)
            h0_actual[i, j] = float(pop.h0[0])

    population = _concat_population_arrays(sub_populations)
    return population, (n_f, n_h0), ra, dec, h0_actual


# ---------------------------------------------------------------------------
# SNR grid computation
# ---------------------------------------------------------------------------

def compute_freq_h0_snr_grid(
    psrs_clean,
    raw_noise_params,
    parsed_noise_params,
    Tspan,
    chirp_mass_msun=2e9,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
    f_min_hz=1e-9,
    f_max_hz=3e-7,
    n_f=50,
    h0_min=1e-17,
    h0_max=1e-12,
    n_h0=40,
):
    """
    Build the (f, h0) grid, inject it, and compute per-point optimal CGW
    S/N, returning (freqs_hz, h0_actual, snr_grid, (ra, dec)) with
    snr_grid.shape == h0_actual.shape == (n_f, n_h0). h0_actual is read
    back from the pipeline itself, so it may differ very slightly from
    the geomspace targets used to seed the root-finder.
    """
    freqs_hz = np.geomspace(f_min_hz, f_max_hz, n_f)
    h0_targets = np.geomspace(h0_min, h0_max, n_h0)

    population, grid_shape, ra_used, dec_used, h0_actual = _build_freq_h0_grid_population(
        chirp_mass_msun=chirp_mass_msun,
        freqs_hz=freqs_hz,
        h0_targets=h0_targets,
        mass_ratio=mass_ratio,
        ra=ra,
        dec=dec,
        iota=iota,
    )

    from consistent_pop_synth import compute_population_snr
    from CGW_SNR import compute_cgw_snr_optimal_population

    original_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}
    _, pta, enterprise_psrs = compute_population_snr(
        population=population,
        psrs_clean=psrs_clean,
        current_stoas=original_stoas,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan,
        return_psrs_pta=True,
    )

    binary_rows = _population_arrays_to_binary_rows(population)
    cgw_snrs = compute_cgw_snr_optimal_population(
        psrs=enterprise_psrs,
        pta=pta,
        population=binary_rows,
        Tspan=Tspan,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
    )

    snr_grid = np.array(cgw_snrs, dtype=float).reshape(grid_shape)
    return freqs_hz, h0_actual, snr_grid, (ra_used, dec_used)


# ---------------------------------------------------------------------------
# Plot -- continuous colour bar (no discrete S/N binning)
# ---------------------------------------------------------------------------

def plot_freq_amp_analysis(
    freqs_hz,
    h0_grid,
    snr_grid,
    save_path=None,
    figsize=None,
    sky_location=None,
    iota=None,
    snr_floor=None,
):
    """
    Publication-style f vs h0 plot with a single continuous S/N colour
    bar (LogNorm), styled consistently with the rest of the paper via
    apj_style. No threshold binning -- colour reads off the actual S/N
    value continuously.

    h0_grid is the full (n_f, n_h0) array of *actual* h0 values (not a
    1D axis) since h0 was root-found per frequency and may vary slightly
    across the frequency axis for the same target index -- pcolormesh
    is given the true (F, h0_grid) coordinates so the plot is exact even
    if that grid is very slightly non-rectangular.

    snr_floor: clip S/N values below this before taking the log colour
    scale (LogNorm chokes on exactly-zero or negative values). Defaults
    to a small fraction of the smallest positive finite S/N in the grid.

    Pass sky_location=(ra, dec) and/or iota to annotate the fixed
    parameters used to generate the grid (e.g. "most sensitive sky
    location, face-on").
    """
    if figsize is None:
        figsize = (APJ_COL_WIDTH, APJ_COL_WIDTH * 0.8)

    finite_snr = snr_grid[np.isfinite(snr_grid) & (snr_grid > 0)]
    if snr_floor is None:
        snr_floor = finite_snr.min() * 0.5 if finite_snr.size else 1e-3
    snr_plot = np.clip(snr_grid, snr_floor, None)

    cmap = _snr_colormap()
    norm = LogNorm(vmin=snr_plot.min(), vmax=snr_plot.max())

    with plt.style.context("default"):
        apply_apj_style()

        fig, ax = plt.subplots(figsize=figsize)

        F = np.repeat(freqs_hz[:, None], h0_grid.shape[1], axis=1)
        mesh = ax.pcolormesh(
            F, h0_grid, snr_plot,
            cmap=cmap, norm=norm,
            shading="auto", rasterized=True,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$f_{\mathrm{GW}}$ [Hz]")
        ax.set_ylabel(r"$h_0$")
        ax.set_xlim(freqs_hz.min(), freqs_hz.max())
        ax.set_ylim(h0_grid.min(), h0_grid.max())

        if sky_location is not None or iota is not None:
            parts = []
            if sky_location is not None:
                ra_deg = np.degrees(sky_location[0])
                dec_deg = np.degrees(sky_location[1])
                parts.append(rf"RA={ra_deg:.1f}$^\circ$, Dec={dec_deg:.1f}$^\circ$")
            if iota is not None:
                parts.append(r"face-on" if iota == 0.0 else rf"$\iota$={iota:.2f}")
            ax.set_title(", ".join(parts), fontsize=7)

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(r"(S/N)$_{\mathrm{CW}}$", labelpad=1)
        cbar.ax.tick_params(colors="black")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
            png_path = Path(save_path).with_suffix(".png")
            fig.savefig(png_path)
            print(f"Saved to {save_path} (and {png_path})")
        else:
            plt.show()

    return fig


# ---------------------------------------------------------------------------
# Master driver (mirrors test_sky_CGW_SNR_location)
# ---------------------------------------------------------------------------

def test_freq_amp_CGW_SNR(
    psrs_clean,
    raw_noise_params,
    parsed_noise_params,
    Tspan,
    chirp_mass_msun=2e9,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
    f_min_hz=1e-9,
    f_max_hz=3e-7,
    n_f=50,
    h0_min=1e-17,
    h0_max=1e-10,
    n_h0=40,
):
    """
    Build the (f, h0) grid, inject and compute S/N, plot, and print a
    ranked summary -- analogous to test_sky_CGW_SNR_location but scanning
    frequency/amplitude at fixed sky position/inclination instead of sky
    position at fixed frequency/amplitude.

    By default this uses the most sensitive sky location (peak of the
    sky-sensitivity survey used in the sky-map script) and a face-on
    binary (iota=0), i.e. the S/N-maximising choice of the parameters
    that aren't being scanned. Pass ra/dec/iota explicitly to override.
    """
    import os
    os.makedirs("figures", exist_ok=True)

    freqs_hz = np.geomspace(f_min_hz, f_max_hz, n_f)
    suggested_min, suggested_max = suggest_h0_bounds(freqs_hz, chirp_mass_msun, iota)
    print(
        f"Reachable h0 range across f=[{f_min_hz:.2e}, {f_max_hz:.2e}] Hz "
        f"for chirp_mass_msun={chirp_mass_msun:.2e}: "
        f"[{suggested_min:.3e}, {suggested_max:.3e}]"
    )
    if not (suggested_min <= h0_min and h0_max <= suggested_max):
        print(
            f"  Requested h0 range [{h0_min:.3e}, {h0_max:.3e}] falls outside "
            f"that -- clamping to the reachable range. Pass h0_min/h0_max "
            "explicitly within the printed bounds to avoid this."
        )
        h0_min = max(h0_min, suggested_min)
        h0_max = min(h0_max, suggested_max)

    freqs_hz, h0_grid, snr_grid, (ra_used, dec_used) = compute_freq_h0_snr_grid(
        psrs_clean=psrs_clean,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan=Tspan,
        chirp_mass_msun=chirp_mass_msun,
        mass_ratio=mass_ratio,
        ra=ra,
        dec=dec,
        iota=iota,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        n_f=n_f,
        h0_min=h0_min,
        h0_max=h0_max,
        n_h0=n_h0,
    )

    plot_freq_amp_analysis(
        freqs_hz, h0_grid, snr_grid,
        save_path="figures/cgw_snr_freq_amp_continuous.pdf",
        sky_location=(ra_used, dec_used),
        iota=iota,
    )

    print("\nS/N grid summary:")
    print(f"  Sky location: RA={ra_used:.3f} rad, Dec={dec_used:.3f} rad (most sensitive)")
    print(f"  Inclination:  iota={iota:.3f} rad")
    print(f"  f range:  {freqs_hz.min():.2e} - {freqs_hz.max():.2e} Hz")
    print(f"  h0 range: {h0_grid.min():.2e} - {h0_grid.max():.2e}")
    print(f"  S/N range: {np.nanmin(snr_grid):.2f} - {np.nanmax(snr_grid):.2f}")

    return freqs_hz, h0_grid, snr_grid


"""
plot_cgw_freq_redshift.py

Redshift (z) vs GW frequency (f) detectability plot for a fixed
chirp-mass, fixed sky-position/inclination CW source, colour-binned by
S/N thresholds -- reproduces the style of Rosado et al. (2015) Fig. 4,
in the same apj_style / _snr_colormap convention as the other CGW
figures in this repo.

WHAT'S FIXED vs SCANNED
------------------------
Fixed: chirp mass, mass ratio, sky position (ra, dec), inclination
(iota) -- by default the most sensitive sky location (see
_best_sky_location) and face-on (iota=0), same convention as
plot_cgw_freq_amp.py. Pass ra/dec/iota explicitly to override.
Scanned: GW frequency f and redshift z, on a 2D grid.

WHY THIS IS SIMPLER THAN THE FREQ-VS-H0 SCRIPT
-------------------------------------------------
In plot_cgw_freq_amp.py, h0 is not a direct input to chosen_population
(only redshift is, via D_comov), so hitting a target h0 requires
root-finding the redshift that produces it. Here z *is* the axis we
want, so there's no inversion needed at all: we scan (f, z) directly,
call chosen_population once per grid point, and read h0 back purely
for annotation/sanity-checking -- it plays no role in placing points on
the grid.

CAVEAT (see accompanying discussion): chosen_population passes
D_comov -- not D_L = (1+z) * D_comov -- into compute_strain_amplitude.
If compute_strain_amplitude does not itself apply the (1+z) factor,
every h0/SNR here is biased by that missing factor, growing with z.
Worth confirming against compute_strain_amplitude's source before
trusting absolute SNR levels (relative structure of the contours is
less affected since the same convention is used everywhere in the
repo).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from apj_style import apply_apj_style, APJ_COL_WIDTH
from SMBHB_pop_synth import PopulationArrays, chosen_population, _Z_GRID
from plot_cgw_snr import _snr_colormap

def _best_sky_location():
    """
    Most sensitive (ra, dec) from the sky-sensitivity survey used to build
    the interpolated weight map in the sky-map script. Because that map is
    built with *linear* interpolation, its maximum over the whole sky is
    necessarily attained at one of the sampled vertices -- so the argmax
    over the raw survey points, not the interpolator, is the true best
    sky location (no optimisation loop needed).
    """
    survey = get_sky_survey_points()
    ra_best, dec_best, snr_best = max(survey, key=lambda entry: entry[2])
    return ra_best, dec_best


# The cosmology lookup used inside chosen_population only covers
# z in [_Z_GRID.min(), _Z_GRID.max()] -- np.interp does NOT extrapolate
# beyond that, it just flat-lines D_comov (and hence h0) past the edges.
# Default z bounds below are clamped into this range so every point on
# the grid is using a real (not flat-extrapolated) D_comov.
_Z_MIN_SAFE = max(float(_Z_GRID.min()), 1e-3)  # avoid z=0 exactly (D_comov=0)
_Z_MAX_SAFE = float(_Z_GRID.max())


def _clamp_redshift_bounds(z_min, z_max):
    if z_min < _Z_MIN_SAFE or z_max > _Z_MAX_SAFE:
        print(
            f"  Requested z range [{z_min:.3e}, {z_max:.3e}] falls outside "
            f"the cosmology lookup's usable range "
            f"[{_Z_MIN_SAFE:.3e}, {_Z_MAX_SAFE:.3e}] -- clamping. Points "
            "outside this range would silently use a flat-extrapolated "
            "D_comov rather than a real one."
        )
    return max(z_min, _Z_MIN_SAFE), min(z_max, _Z_MAX_SAFE)


# ---------------------------------------------------------------------------
# Grid population builder
# ---------------------------------------------------------------------------

def _build_freq_z_grid_population(
    chirp_mass_msun,
    freqs_hz,
    redshifts,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
):
    """
    Build a PopulationArrays with one binary per (f, z) grid point, all
    sharing the same sky position, chirp mass, mass ratio, and
    inclination. Unlike the freq-vs-h0 script, no root-finding is
    needed here -- z is a direct input to chosen_population.

    Returns (population, grid_shape, ra, dec, h0_grid) where h0_grid is
    the (n_f, n_z) array of h0 values realised at each grid point,
    kept purely for annotation/sanity-checking.
    """
    if ra is None or dec is None:
        best_ra, best_dec = _best_sky_location()
        ra = best_ra if ra is None else ra
        dec = best_dec if dec is None else dec

    n_f, n_z = len(freqs_hz), len(redshifts)
    sub_populations = []
    h0_grid = np.empty((n_f, n_z), dtype=float)

    for i, f_val in enumerate(freqs_hz):
        for j, z_val in enumerate(redshifts):
            pop = chosen_population(
                n_binaries=1,
                T_obs_seconds=None,          # unused when compute_strain=False
                chirp_mass_msun=chirp_mass_msun,
                mass_ratio=mass_ratio,
                gw_frequency=float(f_val),
                redshift=float(z_val),
                polarization=0.0,
                inclination=float(iota),
                initial_phase=0.0,
                right_ascension=float(ra),
                declination=float(dec),
                compute_strain=False,
            )
            sub_populations.append(pop)
            h0_grid[i, j] = float(pop.h0[0])

    population = _concat_population_arrays(sub_populations)
    return population, (n_f, n_z), ra, dec, h0_grid


# ---------------------------------------------------------------------------
# SNR grid computation
# ---------------------------------------------------------------------------

def compute_freq_z_snr_grid(
    psrs_clean,
    raw_noise_params,
    parsed_noise_params,
    Tspan,
    chirp_mass_msun=1e9,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
    f_min_hz=1e-9,
    f_max_hz=1e-7,
    n_f=40,
    z_min=0.01,
    z_max=5.0,
    n_z=40,
    log_z=True,
):
    """
    Build the (f, z) grid, inject it, and compute per-point optimal CGW
    S/N, returning (freqs_hz, redshifts, snr_grid, h0_grid, (ra, dec))
    with snr_grid.shape == h0_grid.shape == (n_f, n_z).
    """
    z_min, z_max = _clamp_redshift_bounds(z_min, z_max)

    freqs_hz = np.geomspace(f_min_hz, f_max_hz, n_f)
    redshifts = (
        np.geomspace(z_min, z_max, n_z) if log_z
        else np.linspace(z_min, z_max, n_z)
    )

    population, grid_shape, ra_used, dec_used, h0_grid = _build_freq_z_grid_population(
        chirp_mass_msun=chirp_mass_msun,
        freqs_hz=freqs_hz,
        redshifts=redshifts,
        mass_ratio=mass_ratio,
        ra=ra,
        dec=dec,
        iota=iota,
    )

    from consistent_pop_synth import compute_population_snr
    from CGW_SNR import compute_cgw_snr_optimal_population

    original_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}
    _, pta, enterprise_psrs = compute_population_snr(
        population=population,
        psrs_clean=psrs_clean,
        current_stoas=original_stoas,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan,
        return_psrs_pta=True,
    )

    binary_rows = _population_arrays_to_binary_rows(population)
    cgw_snrs = compute_cgw_snr_optimal_population(
        psrs=enterprise_psrs,
        pta=pta,
        population=binary_rows,
        Tspan=Tspan,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
    )

    snr_grid = np.array(cgw_snrs, dtype=float).reshape(grid_shape)
    return freqs_hz, redshifts, snr_grid, h0_grid, (ra_used, dec_used)


# ---------------------------------------------------------------------------
# Plot -- continuous colour bar (no discrete S/N binning)
# ---------------------------------------------------------------------------

def plot_freq_redshift_analysis(
    freqs_hz,
    redshifts,
    snr_grid,
    snr_thresholds=(1, 2, 4, 8),
    save_path=None,
    figsize=None,
    sky_location=None,
    iota=None,
    chirp_mass_msun=None,
    contours=True,
    snr_floor=None,
):
    """
    Rosado et al. (2015) Fig. 4-style redshift vs frequency plot, with a
    single continuous S/N colour bar (LogNorm) rather than discrete
    threshold bins, styled via apj_style.

    contours=True still overlays solid S/N contour lines at
    snr_thresholds on top of the continuous colour field -- these are
    kept because they're the actual detectability boundaries Rosado's
    Fig. 4 is about, even though the fill underneath is now continuous
    rather than binned. Set False to drop them and show only the
    continuous colour field.

    snr_floor: clip S/N values below this before taking the log colour
    scale (LogNorm chokes on exactly-zero or negative values). Defaults
    to half the smallest positive finite S/N in the grid.
    """
    if figsize is None:
        figsize = (APJ_COL_WIDTH, APJ_COL_WIDTH * 0.8)

    finite_snr = snr_grid[np.isfinite(snr_grid) & (snr_grid > 0)]
    if snr_floor is None:
        snr_floor = finite_snr.min() * 0.5 if finite_snr.size else 1e-3
    snr_plot = np.clip(snr_grid, snr_floor, None)

    cmap = _snr_colormap()
    norm = LogNorm(vmin=snr_plot.min(), vmax=snr_plot.max())

    with plt.style.context("default"):
        apply_apj_style()

        fig, ax = plt.subplots(figsize=figsize)

        F, Z = np.meshgrid(freqs_hz, redshifts, indexing="ij")
        mesh = ax.pcolormesh(
            F, Z, snr_plot,
            cmap=cmap, norm=norm,
            shading="auto", rasterized=True,
        )

        if contours:
            cs = ax.contour(
                F, Z, snr_grid,
                levels=snr_thresholds,
                colors="black", linewidths=0.6,
            )
            ax.clabel(cs, inline=True, fontsize=6, fmt=lambda v: f"{v:g}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$f_{\mathrm{GW}}$ [Hz]")
        ax.set_ylabel(r"$z$")
        ax.set_xlim(freqs_hz.min(), freqs_hz.max())
        ax.set_ylim(redshifts.min(), redshifts.max())

        title_parts = []
        if chirp_mass_msun is not None:
            title_parts.append(rf"$\mathcal{{M}}_c={chirp_mass_msun:.1e}\,M_\odot$")
        if sky_location is not None:
            ra_deg = np.degrees(sky_location[0])
            dec_deg = np.degrees(sky_location[1])
            title_parts.append(rf"RA={ra_deg:.1f}$^\circ$, Dec={dec_deg:.1f}$^\circ$")
        if iota is not None:
            title_parts.append(r"face-on" if iota == 0.0 else rf"$\iota$={iota:.2f}")
        if title_parts:
            ax.set_title(", ".join(title_parts), fontsize=7)

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(r"(S/N)$_{\mathrm{CW}}$", labelpad=1)
        cbar.ax.tick_params(colors="black")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
            png_path = Path(save_path).with_suffix(".png")
            fig.savefig(png_path)
            print(f"Saved to {save_path} (and {png_path})")
        else:
            plt.show()

    return fig


# ---------------------------------------------------------------------------
# Master driver (mirrors test_freq_amp_CGW_SNR)
# ---------------------------------------------------------------------------

def test_freq_redshift_CGW_SNR(
    psrs_clean,
    raw_noise_params,
    parsed_noise_params,
    Tspan,
    chirp_mass_msun=1e9,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
    f_min_hz=1e-9,
    f_max_hz=1e-7,
    n_f=40,
    z_min=0.01,
    z_max=5.0,
    n_z=40,
    log_z=True,
):
    """
    Build the (f, z) grid, inject and compute S/N, plot, and print a
    summary -- Rosado et al. (2015) Fig. 4 analogue: redshift vs
    frequency at fixed chirp mass, colour/contour-binned by S/N,
    using the most sensitive sky location and face-on inclination by
    default (pass ra/dec/iota to override).
    """
    import os
    os.makedirs("figures", exist_ok=True)

    freqs_hz, redshifts, snr_grid, h0_grid, (ra_used, dec_used) = compute_freq_z_snr_grid(
        psrs_clean=psrs_clean,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan=Tspan,
        chirp_mass_msun=chirp_mass_msun,
        mass_ratio=mass_ratio,
        ra=ra,
        dec=dec,
        iota=iota,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        n_f=n_f,
        z_min=z_min,
        z_max=z_max,
        n_z=n_z,
        log_z=log_z,
    )

    plot_freq_redshift_analysis(
        freqs_hz, redshifts, snr_grid,
        save_path="figures/cgw_snr_freq_redshift_continuous.pdf",
        sky_location=(ra_used, dec_used),
        iota=iota,
        chirp_mass_msun=chirp_mass_msun,
    )

    print("\nS/N grid summary:")
    print(f"  Chirp mass:   {chirp_mass_msun:.2e} Msun")
    print(f"  Sky location: RA={ra_used:.3f} rad, Dec={dec_used:.3f} rad (most sensitive)")
    print(f"  Inclination:  iota={iota:.3f} rad")
    print(f"  f range:  {freqs_hz.min():.2e} - {freqs_hz.max():.2e} Hz")
    print(f"  z range:  {redshifts.min():.3f} - {redshifts.max():.3f}")
    print(f"  h0 range: {h0_grid.min():.2e} - {h0_grid.max():.2e}")
    print(f"  S/N range: {np.nanmin(snr_grid):.2f} - {np.nanmax(snr_grid):.2f}")

    return freqs_hz, redshifts, snr_grid, h0_grid