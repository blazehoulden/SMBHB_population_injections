"""
plot_cgw_mass_amp.py

Chirp mass (Mc) vs strain amplitude (h0) detectability plot for a fixed
GW frequency, fixed sky-position/inclination CW source, continuous S/N
colour bar (LogNorm) with optional overlaid threshold contours -- same
family as plot_cgw_freq_amp_continuous.py, but scanning Mc instead of f
(f is fixed here instead).

WHAT'S FIXED vs SCANNED
------------------------
Fixed: GW frequency f, mass ratio, sky position (ra, dec), inclination
(iota) -- by default the most sensitive sky location and face-on.
Pass ra/dec/iota explicitly to override.
Scanned: chirp mass Mc and strain amplitude h0, on a 2D grid.

HOW h0 IS CONTROLLED
---------------------
Same situation as plot_cgw_freq_amp.py: chosen_population() does not
take h0 directly, only redshift (via D_comov). So for each (Mc, h0)
grid point we root-find the redshift z such that
compute_strain_amplitude(f, Mc, D_comov(z), z, iota) equals the target
h0, using chosen_population's own compute_strain_amplitude function
directly. h0 is monotonically decreasing in z at fixed (f, Mc, iota)
(D_comov increases monotonically with z), so the root is unique and
well-bracketed -- same guarantee as the freq-amp script, just with Mc
now also varying across the grid rather than fixed.

The h0 values actually plotted are read back from the resulting
PopulationArrays.h0, not just the root-finder's target, so the plot
stays self-consistent with the pipeline even if the solver lands a
hair off target.

CAVEAT: same D_comov-vs-D_L=(1+z)*D_comov caveat as the other scripts
applies here too.
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
    """h0 for a single (f, Mc, z) point, using the pipeline's own strain calc."""
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
    z range the cosmology lookup covers.
    """
    h0_max = _h0_at_redshift(f_hz, _Z_MIN_SAFE, chirp_mass_msun, iota)
    h0_min = _h0_at_redshift(f_hz, _Z_MAX_SAFE, chirp_mass_msun, iota)
    return h0_max, h0_min


def _redshift_for_target_h0(f_hz, target_h0, chirp_mass_msun, iota):
    """
    Root-find the redshift z such that h0(f, Mc, z) == target_h0, at
    fixed (f, Mc, iota), within z in [_Z_MIN_SAFE, _Z_MAX_SAFE].

    Returns None (rather than raising) if target_h0 is outside the
    reachable range at this chirp mass. This is expected/normal for a
    wide Mc scan: since h0 ~ Mc^(5/3) at fixed (f, z), the reachable h0
    range shifts by orders of magnitude across a several-decade Mc
    range, so a single (h0_min, h0_max) box essentially never covers
    every Mc row simultaneously. The caller marks these grid points as
    NaN rather than the whole grid computation failing.
    """
    def resid(z):
        return _h0_at_redshift(f_hz, z, chirp_mass_msun, iota) - target_h0

    r_lo, r_hi = resid(_Z_MIN_SAFE), resid(_Z_MAX_SAFE)

    if r_lo < 0 or r_hi > 0:
        return None

    return brentq(resid, _Z_MIN_SAFE, _Z_MAX_SAFE)


def suggest_h0_bounds(f_hz, chirp_masses_msun, iota=0.0):
    """
    For every chirp mass in chirp_masses_msun, compute the h0 range
    reachable across the cosmology lookup's full redshift span, and
    return the intersection across all of them: an (h0_min, h0_max)
    pair guaranteed reachable at *every* chirp mass in the grid. Call
    this first to pick sensible h0_min/h0_max before running the (slow)
    full S/N grid, e.g.:

        chirp_masses_msun = np.geomspace(1e7, 1e11, 50)
        h0_min, h0_max = suggest_h0_bounds(1e-8, chirp_masses_msun)
    """
    h0_maxes, h0_mins = [], []
    for mc_val in chirp_masses_msun:
        h0_max, h0_min = _reachable_h0_range(f_hz, mc_val, iota)
        h0_maxes.append(h0_max)
        h0_mins.append(h0_min)
    return max(h0_mins), min(h0_maxes)


# ---------------------------------------------------------------------------
# Grid population builder
# ---------------------------------------------------------------------------

def _build_mass_h0_grid_population(
    gw_frequency_hz,
    chirp_masses_msun,
    h0_targets,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
):
    """
    Build a PopulationArrays with one binary per *reachable* (Mc, h0)
    grid point, all sharing the same GW frequency, sky position, mass
    ratio, and inclination. Grid points where no redshift in
    [_Z_MIN_SAFE, _Z_MAX_SAFE] reaches the target h0 (common across a
    wide Mc range, since h0 ~ Mc^(5/3)) are skipped rather than
    included -- they're reported back via `valid_mask` so the caller
    can leave them as NaN in the final grid instead of crashing or
    silently mis-plotting them.

    Returns (population, grid_shape, ra, dec, h0_actual, valid_mask):
      - h0_actual: (n_mc, n_h0) array, NaN at unreachable points
      - valid_mask: (n_mc, n_h0) bool array, True where a binary was
        actually built (in the same flattened order as `population`)
    """
    if ra is None or dec is None:
        best_ra, best_dec = _best_sky_location()
        ra = best_ra if ra is None else ra
        dec = best_dec if dec is None else dec

    n_mc, n_h0 = len(chirp_masses_msun), len(h0_targets)
    sub_populations = []
    h0_actual = np.full((n_mc, n_h0), np.nan, dtype=float)
    valid_mask = np.zeros((n_mc, n_h0), dtype=bool)

    for i, mc_val in enumerate(chirp_masses_msun):
        for j, h0_target in enumerate(h0_targets):
            z_val = _redshift_for_target_h0(gw_frequency_hz, h0_target, mc_val, iota)
            if z_val is None:
                continue
            pop = chosen_population(
                n_binaries=1,
                T_obs_seconds=None,          # unused when compute_strain=False
                chirp_mass_msun=float(mc_val),
                mass_ratio=mass_ratio,
                gw_frequency=float(gw_frequency_hz),
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
            valid_mask[i, j] = True

    n_skipped = (~valid_mask).sum()
    if n_skipped:
        print(
            f"  {n_skipped}/{n_mc * n_h0} (Mc, h0) grid points are outside "
            "the reachable redshift range at their chirp mass and will "
            "show as blank/NaN -- this is expected for a wide Mc range "
            "since h0 ~ Mc^(5/3) shifts the reachable h0 band a lot "
            "across decades of chirp mass."
        )

    population = _concat_population_arrays(sub_populations)
    return population, (n_mc, n_h0), ra, dec, h0_actual, valid_mask


# ---------------------------------------------------------------------------
# SNR grid computation
# ---------------------------------------------------------------------------

def compute_mass_h0_snr_grid(
    psrs_clean,
    raw_noise_params,
    parsed_noise_params,
    Tspan,
    gw_frequency_hz=1e-8,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
    mc_min=1e7,
    mc_max=1e11,
    n_mc=50,
    h0_min=1e-17,
    h0_max=1e-12,
    n_h0=40,
):
    """
    Build the (Mc, h0) grid, inject it, and compute per-point optimal
    CGW S/N, returning (chirp_masses_msun, h0_actual, snr_grid,
    (ra, dec)) with snr_grid.shape == h0_actual.shape == (n_mc, n_h0).
    """
    chirp_masses_msun = np.geomspace(mc_min, mc_max, n_mc)
    h0_targets = np.geomspace(h0_min, h0_max, n_h0)

    population, grid_shape, ra_used, dec_used, h0_actual, valid_mask = _build_mass_h0_grid_population(
        gw_frequency_hz=gw_frequency_hz,
        chirp_masses_msun=chirp_masses_msun,
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

    # cgw_snrs is in the same flattened order as the valid (Mc, h0)
    # points were appended in _build_mass_h0_grid_population (row-major
    # over valid_mask); scatter back into the full grid, NaN elsewhere.
    snr_grid = np.full(grid_shape, np.nan, dtype=float)
    snr_grid[valid_mask] = np.array(cgw_snrs, dtype=float)
    return chirp_masses_msun, h0_actual, snr_grid, (ra_used, dec_used)


# ---------------------------------------------------------------------------
# Plot -- continuous colour bar
# ---------------------------------------------------------------------------

def plot_mass_amp_analysis(
    chirp_masses_msun,
    h0_grid,
    snr_grid,
    snr_thresholds=(1, 2, 4, 8),
    save_path=None,
    figsize=None,
    sky_location=None,
    iota=None,
    gw_frequency_hz=None,
    contours=True,
    snr_floor=None,
):
    """
    Chirp mass vs h0 plot with a single continuous S/N colour bar
    (LogNorm), styled via apj_style. contours=True overlays solid S/N
    threshold contours on top of the continuous colour field.

    h0_grid is the full (n_mc, n_h0) array of *actual* h0 values (not a
    1D axis) since h0 was root-found per chirp mass and may vary
    slightly across the Mc axis for the same target index --
    pcolormesh is given the true (Mc, h0_grid) coordinates so the plot
    is exact even if that grid is very slightly non-rectangular.
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

        MC = np.repeat(chirp_masses_msun[:, None], h0_grid.shape[1], axis=1)
        mesh = ax.pcolormesh(
            MC, h0_grid, snr_plot,
            cmap=cmap, norm=norm,
            shading="auto", rasterized=True,
        )

        if contours:
            cs = ax.contour(
                MC, h0_grid, snr_grid,
                levels=snr_thresholds,
                colors="black", linewidths=0.6,
            )
            ax.clabel(cs, inline=True, fontsize=6, fmt=lambda v: f"{v:g}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\mathcal{M}_c$ [$M_\odot$]")
        ax.set_ylabel(r"$h_0$")
        ax.set_xlim(chirp_masses_msun.min(), chirp_masses_msun.max())
        ax.set_ylim(h0_grid.min(), h0_grid.max())

        title_parts = []
        if gw_frequency_hz is not None:
            title_parts.append(rf"$f_{{\mathrm{{GW}}}}={gw_frequency_hz:.2e}$ Hz")
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
# Master driver
# ---------------------------------------------------------------------------

def test_mass_amp_CGW_SNR(
    psrs_clean,
    raw_noise_params,
    parsed_noise_params,
    Tspan,
    gw_frequency_hz=1e-8,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
    mc_min=1e7,
    mc_max=1e11,
    n_mc=50,
    h0_min=1e-17,
    h0_max=1e-12,
    n_h0=40,
):
    """
    Build the (Mc, h0) grid, inject and compute S/N, plot, and print a
    summary -- chirp mass vs strain amplitude analogue of
    test_freq_amp_CGW_SNR, at fixed GW frequency, using the most
    sensitive sky location and face-on inclination by default (pass
    ra/dec/iota to override).
    """
    import os
    os.makedirs("figures", exist_ok=True)

    chirp_masses_msun = np.geomspace(mc_min, mc_max, n_mc)
    suggested_min, suggested_max = suggest_h0_bounds(gw_frequency_hz, chirp_masses_msun, iota)
    print(
        f"Reachable h0 range across Mc=[{mc_min:.2e}, {mc_max:.2e}] Msun "
        f"at f={gw_frequency_hz:.2e} Hz: "
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

    chirp_masses_msun, h0_grid, snr_grid, (ra_used, dec_used) = compute_mass_h0_snr_grid(
        psrs_clean=psrs_clean,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan=Tspan,
        gw_frequency_hz=gw_frequency_hz,
        mass_ratio=mass_ratio,
        ra=ra,
        dec=dec,
        iota=iota,
        mc_min=mc_min,
        mc_max=mc_max,
        n_mc=n_mc,
        h0_min=h0_min,
        h0_max=h0_max,
        n_h0=n_h0,
    )

    plot_mass_amp_analysis(
        chirp_masses_msun, h0_grid, snr_grid,
        save_path="figures/cgw_snr_mass_amp.pdf",
        sky_location=(ra_used, dec_used),
        iota=iota,
        gw_frequency_hz=gw_frequency_hz,
    )

    print("\nS/N grid summary:")
    print(f"  GW frequency: {gw_frequency_hz:.2e} Hz")
    print(f"  Sky location: RA={ra_used:.3f} rad, Dec={dec_used:.3f} rad (most sensitive)")
    print(f"  Inclination:  iota={iota:.3f} rad")
    print(f"  Mc range: {chirp_masses_msun.min():.2e} - {chirp_masses_msun.max():.2e} Msun")
    print(f"  h0 range: {h0_grid.min():.2e} - {h0_grid.max():.2e}")
    print(f"  S/N range: {np.nanmin(snr_grid):.2f} - {np.nanmax(snr_grid):.2f}")

    return chirp_masses_msun, h0_grid, snr_grid