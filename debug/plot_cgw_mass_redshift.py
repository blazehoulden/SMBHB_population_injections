"""
plot_cgw_mass_redshift.py

Chirp mass (Mc) vs redshift (z) detectability plot for a fixed GW
frequency, fixed sky-position/inclination CW source, continuous S/N
colour bar (LogNorm) with optional overlaid threshold contours -- same
family as plot_cgw_freq_redshift.py and plot_cgw_freq_amp_continuous.py,
just with Mc and z as the scanned pair instead of f and z or f and h0.

WHAT'S FIXED vs SCANNED
------------------------
Fixed: GW frequency f, mass ratio, sky position (ra, dec), inclination
(iota) -- by default the most sensitive sky location and face-on,
same convention as the other CGW scripts in this repo. Pass ra/dec/iota
explicitly to override.
Scanned: chirp mass Mc and redshift z, on a 2D grid.

WHY THIS IS A DIRECT BUILD (no root-finding)
----------------------------------------------
Both chirp_mass_msun and redshift are direct inputs to
chosen_population(), so -- exactly like plot_cgw_freq_redshift.py --
there's nothing to invert here. We scan (Mc, z) directly and read h0
back purely for annotation/cross-checking against the Mc-h0 script.

CAVEAT: same D_comov-vs-D_L=(1+z)*D_comov caveat as the other scripts
applies here too -- if compute_strain_amplitude doesn't itself apply
the (1+z) factor, h0/SNR here is biased by a factor growing with z.
Worth confirming against compute_strain_amplitude's source.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from apj_style import apply_apj_style, APJ_COL_WIDTH
from SMBHB_pop_synth import PopulationArrays, chosen_population, _Z_GRID
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


# The cosmology lookup used inside chosen_population only covers
# z in [_Z_GRID.min(), _Z_GRID.max()] -- np.interp does NOT extrapolate
# beyond that, it just flat-lines D_comov (and hence h0) past the edges.
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

def _build_mass_z_grid_population(
    gw_frequency_hz,
    chirp_masses_msun,
    redshifts,
    mass_ratio=0.5,
    ra=None,
    dec=None,
    iota=0.0,
):
    """
    Build a PopulationArrays with one binary per (Mc, z) grid point, all
    sharing the same GW frequency, sky position, mass ratio, and
    inclination. No root-finding needed -- both Mc and z are direct
    inputs to chosen_population.

    Returns (population, grid_shape, ra, dec, h0_grid) where h0_grid is
    the (n_Mc, n_z) array of h0 values realised at each grid point,
    kept for annotation/cross-checking.
    """
    if ra is None or dec is None:
        best_ra, best_dec = _best_sky_location()
        ra = best_ra if ra is None else ra
        dec = best_dec if dec is None else dec

    n_mc, n_z = len(chirp_masses_msun), len(redshifts)
    sub_populations = []
    h0_grid = np.empty((n_mc, n_z), dtype=float)

    for i, mc_val in enumerate(chirp_masses_msun):
        for j, z_val in enumerate(redshifts):
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
            h0_grid[i, j] = float(pop.h0[0])

    population = _concat_population_arrays(sub_populations)
    return population, (n_mc, n_z), ra, dec, h0_grid


# ---------------------------------------------------------------------------
# SNR grid computation
# ---------------------------------------------------------------------------

def compute_mass_z_snr_grid(
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
    z_min=0.01,
    z_max=20.0,
    n_z=80,
    log_z=True,
):
    """
    Build the (Mc, z) grid, inject it, and compute per-point optimal
    CGW S/N, returning (chirp_masses_msun, redshifts, snr_grid, h0_grid,
    (ra, dec)) with snr_grid.shape == h0_grid.shape == (n_mc, n_z).
    """
    z_min, z_max = _clamp_redshift_bounds(z_min, z_max)

    chirp_masses_msun = np.geomspace(mc_min, mc_max, n_mc)
    redshifts = (
        np.geomspace(z_min, z_max, n_z) if log_z
        else np.linspace(z_min, z_max, n_z)
    )

    population, grid_shape, ra_used, dec_used, h0_grid = _build_mass_z_grid_population(
        gw_frequency_hz=gw_frequency_hz,
        chirp_masses_msun=chirp_masses_msun,
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
    return chirp_masses_msun, redshifts, snr_grid, h0_grid, (ra_used, dec_used)


# ---------------------------------------------------------------------------
# Plot -- continuous colour bar
# ---------------------------------------------------------------------------

def plot_mass_redshift_analysis(
    chirp_masses_msun,
    redshifts,
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
    Chirp mass vs redshift plot with a single continuous S/N colour bar
    (LogNorm), styled via apj_style. contours=True overlays solid S/N
    threshold contours on top of the continuous colour field.

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

        MC, Z = np.meshgrid(chirp_masses_msun, redshifts, indexing="ij")
        mesh = ax.pcolormesh(
            MC, Z, snr_plot,
            cmap=cmap, norm=norm,
            shading="auto", rasterized=True,
        )

        if contours:
            cs = ax.contour(
                MC, Z, snr_grid,
                levels=snr_thresholds,
                colors="black", linewidths=0.6,
            )
            ax.clabel(cs, inline=True, fontsize=6, fmt=lambda v: f"{v:g}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\mathcal{M}_c$ [$M_\odot$]")
        ax.set_ylabel(r"$z$")
        ax.set_xlim(chirp_masses_msun.min(), chirp_masses_msun.max())
        ax.set_ylim(redshifts.min(), redshifts.max())

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

def test_mass_redshift_CGW_SNR(
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
    z_min=0.01,
    z_max=20.0,
    n_z=80,
    log_z=True,
):
    """
    Build the (Mc, z) grid, inject and compute S/N, plot, and print a
    summary -- chirp mass vs redshift analogue of test_freq_redshift_CGW_SNR,
    at fixed GW frequency, using the most sensitive sky location and
    face-on inclination by default (pass ra/dec/iota to override).
    """
    import os
    os.makedirs("figures", exist_ok=True)

    chirp_masses_msun, redshifts, snr_grid, h0_grid, (ra_used, dec_used) = compute_mass_z_snr_grid(
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
        z_min=z_min,
        z_max=z_max,
        n_z=n_z,
        log_z=log_z,
    )

    plot_mass_redshift_analysis(
        chirp_masses_msun, redshifts, snr_grid,
        save_path="figures/cgw_snr_mass_redshift.pdf",
        sky_location=(ra_used, dec_used),
        iota=iota,
        gw_frequency_hz=gw_frequency_hz,
    )

    print("\nS/N grid summary:")
    print(f"  GW frequency: {gw_frequency_hz:.2e} Hz")
    print(f"  Sky location: RA={ra_used:.3f} rad, Dec={dec_used:.3f} rad (most sensitive)")
    print(f"  Inclination:  iota={iota:.3f} rad")
    print(f"  Mc range: {chirp_masses_msun.min():.2e} - {chirp_masses_msun.max():.2e} Msun")
    print(f"  z range:  {redshifts.min():.3f} - {redshifts.max():.3f}")
    print(f"  h0 range: {h0_grid.min():.2e} - {h0_grid.max():.2e}")
    print(f"  S/N range: {np.nanmin(snr_grid):.2f} - {np.nanmax(snr_grid):.2f}")

    return chirp_masses_msun, redshifts, snr_grid, h0_grid