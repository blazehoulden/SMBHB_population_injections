"""
plot_cgw_freq_amp.py

GW frequency (f) vs strain amplitude (h0) detectability plot for a fixed
sky position / fixed chirp mass continuous-wave (CW) source, colour-binned
by S/N thresholds (e.g. 1, 2, 4, 8), in the same style as the CGW sky-map
figures in this repo (apj_style, _snr_colormap, compute_population_snr /
compute_cgw_snr_optimal_population).

WHAT'S FIXED vs SCANNED
------------------------
Fixed: chirp mass, sky position (ra, dec), inclination (iota), redshift,
and everything else that goes into chosen_population(). By default sky
position and inclination are set to whatever maximises S/N: the most
sensitive point in the sky-sensitivity survey used by the sky-map script
(see _best_sky_location) and face-on (iota=0), so the resulting f-h0
detectability plot is the best-case boundary for this chirp mass/redshift
rather than an arbitrary sky slice. Pass ra/dec/iota explicitly to
override.
Scanned: GW frequency f and strain amplitude h0, on a 2D grid.

WHY DISTANCE IS BACKED OUT
---------------------------
h0 is not a direct input to chosen_population() in this codebase -- it's
derived from Mc, D_comov, z and f. So for every (f, h0) grid point we
invert the standard circular, quadrupole CW strain formula

    h0 = 4 * G^(5/3) / c^4 * Mc^(5/3) * (pi f)^(2/3) / D_L

for the luminosity distance D_L that would produce that h0 at that Mc
and f, then pass THAT distance into chosen_population(). Everything
downstream (the actual SNR calculation) is untouched -- it still uses
the pipeline's own strain/response machinery, this just chooses the
distance so the injected h0 lands where you asked for it.

>>> IMPORTANT: chosen_population()'s exact kwarg names for setting GW
>>> frequency and distance directly aren't visible from the sky-map code
>>> alone. Two placeholders below (`gw_freq_hz=`, `luminosity_distance_mpc=`)
>>> need to be checked against your actual chosen_population() signature
>>> and renamed if they differ (e.g. it might be `f_gw=`, `D_L_mpc=`,
>>> `distance_mpc=`, etc). If chosen_population only accepts D_comov
>>> directly (no separate D_L/z handling), just pass the distance as
>>> D_comov and drop the (1+z) correction below.

Also double check the strain-formula convention (factor of 4 vs 2,
sky/inclination-averaging, etc.) against whatever chosen_population uses
internally when compute_strain=True -- if your pipeline uses a different
convention the (f, h0) grid points won't correspond to exactly the h0
values in the axis labels, just adjust `_h0_to_distance_m` to match.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

from apj_style import apply_apj_style, APJ_COL_WIDTH
from SMBHB_pop_synth import PopulationArrays, chosen_population
from plot_cgw_snr import _snr_colormap

# Reuse the helpers already defined for the sky-map script rather than
# duplicating them -- adjust this import path/module name to wherever
# _concat_population_arrays / _population_arrays_to_binary_rows /
# _SKY_SURVEY actually live in your repo.
from plot_cgw_full_sky_analysis import (
    _concat_population_arrays,
    _population_arrays_to_binary_rows,
    _SKY_SURVEY,
)


def _best_sky_location():
    """
    Most sensitive (ra, dec) from the 400-point sky-sensitivity survey
    used to build the interpolated weight map in the sky-map script.
    Because that map is built with *linear* interpolation, its maximum
    over the whole sky is necessarily attained at one of the sampled
    vertices -- so the argmax over the raw survey points, not the
    interpolator, is the true best sky location (no optimisation loop
    needed).
    """
    ra_best, dec_best, snr_best = max(_SKY_SURVEY, key=lambda entry: entry[2])
    return ra_best, dec_best

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
G = 6.67430e-11              # m^3 kg^-1 s^-2
C = 2.99792458e8              # m/s
MSUN = 1.98892e30             # kg
MPC = 3.0856775814913673e22   # m


def _h0_to_distance_m(chirp_mass_msun, f_gw_hz, h0):
    """
    Invert h0 = 4 G^(5/3)/c^4 * Mc^(5/3) * (pi f)^(2/3) / D_L for D_L
    (metres), given Mc (Msun) and f (Hz). Vectorised over f_gw_hz / h0.
    """
    Mc_kg = np.asarray(chirp_mass_msun) * MSUN
    numerator = 4.0 * G ** (5.0 / 3.0) / C ** 4
    numerator *= Mc_kg ** (5.0 / 3.0) * (np.pi * np.asarray(f_gw_hz)) ** (2.0 / 3.0)
    return numerator / np.asarray(h0)


# ---------------------------------------------------------------------------
# Grid population builder
# ---------------------------------------------------------------------------

def _build_freq_h0_grid_population(
    chirp_mass_msun,
    freqs_hz,
    h0_vals,
    ra=None,
    dec=None,
    iota=0.0,
    z=0.02,
    T_obs_seconds=None,
):
    """
    Build a PopulationArrays with one binary per (f, h0) grid point, all
    sharing the same sky position, chirp mass, inclination, and redshift.
    Returns the population plus the (n_f, n_h0) grid shape for reshaping
    SNRs later.

    ra/dec default to the most sensitive sky location (see
    _best_sky_location); iota defaults to 0.0 (face-on), which maximises
    h0 in the standard convention since both h_plus ~ (1+cos^2 iota)/2
    and h_cross ~ cos(iota) peak at iota=0.
    """
    if ra is None or dec is None:
        best_ra, best_dec = _best_sky_location()
        ra = best_ra if ra is None else ra
        dec = best_dec if dec is None else dec

    F, H0 = np.meshgrid(freqs_hz, h0_vals, indexing="ij")
    f_flat = F.ravel()
    h0_flat = H0.ravel()

    D_L_m = _h0_to_distance_m(chirp_mass_msun, f_flat, h0_flat)
    # small-z comoving-distance correction; drop this if chosen_population
    # wants D_L directly rather than D_comov
    D_comov_mpc = D_L_m / (1.0 + z) / MPC

    sub_populations = []
    for f_val, d_val in zip(f_flat, D_comov_mpc):
        pop = chosen_population(
            n_binaries=1,
            chirp_mass_msun=chirp_mass_msun,
            right_ascension=float(ra),
            declination=float(dec),
            compute_strain=False,
            T_obs_seconds=T_obs_seconds,
            gw_freq_hz=float(f_val),                 # <-- verify/rename against your signature
            luminosity_distance_mpc=float(d_val),     # <-- verify/rename against your signature
            inclination=float(iota),                  # <-- verify/rename against your signature (iota)
            redshift=z,
        )
        sub_populations.append(pop)

    population = _concat_population_arrays(sub_populations)
    return population, F.shape, ra, dec


# ---------------------------------------------------------------------------
# SNR grid computation
# ---------------------------------------------------------------------------

def compute_freq_h0_snr_grid(
    psrs_clean,
    raw_noise_params,
    parsed_noise_params,
    Tspan,
    chirp_mass_msun=1e9,
    ra=None,
    dec=None,
    iota=0.0,
    z=0.02,
    f_min_hz=1e-9,
    f_max_hz=1e-7,
    n_f=40,
    h0_min=1e-16,
    h0_max=1e-13,
    n_h0=40,
):
    """
    Build the (f, h0) grid, inject it, and compute per-point optimal CGW
    S/N, returning (freqs_hz, h0_vals, snr_grid, (ra, dec)) with
    snr_grid.shape == (n_f, n_h0). ra/dec default to the most sensitive
    sky location and iota defaults to face-on (0.0) -- see
    _build_freq_h0_grid_population.
    """
    freqs_hz = np.geomspace(f_min_hz, f_max_hz, n_f)
    h0_vals = np.geomspace(h0_min, h0_max, n_h0)

    population, grid_shape, ra_used, dec_used = _build_freq_h0_grid_population(
        chirp_mass_msun=chirp_mass_msun,
        freqs_hz=freqs_hz,
        h0_vals=h0_vals,
        ra=ra,
        dec=dec,
        iota=iota,
        z=z,
        T_obs_seconds=Tspan,
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
    return freqs_hz, h0_vals, snr_grid, (ra_used, dec_used)


# ---------------------------------------------------------------------------
# Discrete S/N colour binning
# ---------------------------------------------------------------------------

def _discrete_snr_cmap(boundaries):
    """
    Sample the same continuous colormap used elsewhere in the paper
    (_snr_colormap) at evenly spaced points and turn it into a discrete,
    solid-colour ListedColormap + BoundaryNorm -- dark for low S/N bins,
    light for high S/N bins, consistent tone with the sky-map figures.
    """
    base = _snr_colormap()
    n_bins = len(boundaries) - 1
    colors = [base(x) for x in np.linspace(0.15, 0.95, n_bins)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_freq_amp_analysis(
    freqs_hz,
    h0_vals,
    snr_grid,
    snr_thresholds=(1, 2, 4, 8),
    save_path=None,
    figsize=None,
    sky_location=None,
    iota=None,
):
    """
    Publication-style f vs h0 plot, colour-binned by S/N thresholds
    (default 1/2/4/8), styled consistently with the rest of the paper
    via apj_style. Pass sky_location=(ra, dec) and/or iota to annotate
    the fixed parameters used to generate the grid (e.g. "most sensitive
    sky location, face-on").
    """
    if figsize is None:
        figsize = (APJ_COL_WIDTH, APJ_COL_WIDTH * 0.8)

    finite_max = snr_grid[np.isfinite(snr_grid)].max()
    top_edge = max(finite_max * 1.05, snr_thresholds[-1] * 1.5)
    boundaries = [0.0, *snr_thresholds, top_edge]

    cmap, norm = _discrete_snr_cmap(boundaries)

    with plt.style.context("default"):
        apply_apj_style()

        fig, ax = plt.subplots(figsize=figsize)

        F, H0 = np.meshgrid(freqs_hz, h0_vals, indexing="ij")
        ax.pcolormesh(
            F, H0, snr_grid,
            cmap=cmap, norm=norm,
            shading="auto", rasterized=True,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$f_{\mathrm{GW}}$ [Hz]")
        ax.set_ylabel(r"$h_0$")
        ax.set_xlim(freqs_hz.min(), freqs_hz.max())
        ax.set_ylim(h0_vals.min(), h0_vals.max())

        if sky_location is not None or iota is not None:
            parts = []
            if sky_location is not None:
                ra_deg = np.degrees(sky_location[0])
                dec_deg = np.degrees(sky_location[1])
                parts.append(rf"RA={ra_deg:.1f}$^\circ$, Dec={dec_deg:.1f}$^\circ$")
            if iota is not None:
                parts.append(r"face-on" if iota == 0.0 else rf"$\iota$={iota:.2f}")
            ax.set_title(", ".join(parts), fontsize=7)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, ticks=boundaries[:-1])
        cbar.set_label(r"(S/N)$_{\mathrm{CW}}$", labelpad=1)
        tick_labels = [f"{v:g}" for v in boundaries[:-2]] + [f"{snr_thresholds[-1]:g}+"]
        cbar.ax.set_yticklabels(tick_labels)
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
    chirp_mass_msun=1e9,
    ra=None,
    dec=None,
    iota=0.0,
    z=0.02,
    f_min_hz=1e-9,
    f_max_hz=1e-7,
    n_f=40,
    h0_min=1e-16,
    h0_max=1e-13,
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

    freqs_hz, h0_vals, snr_grid, (ra_used, dec_used) = compute_freq_h0_snr_grid(
        psrs_clean=psrs_clean,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan=Tspan,
        chirp_mass_msun=chirp_mass_msun,
        ra=ra,
        dec=dec,
        iota=iota,
        z=z,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        n_f=n_f,
        h0_min=h0_min,
        h0_max=h0_max,
        n_h0=n_h0,
    )

    plot_freq_amp_analysis(
        freqs_hz, h0_vals, snr_grid,
        save_path="figures/cgw_snr_freq_amp.pdf",
        sky_location=(ra_used, dec_used),
        iota=iota,
    )

    print("\nS/N grid summary:")
    print(f"  Sky location: RA={ra_used:.3f} rad, Dec={dec_used:.3f} rad (most sensitive)")
    print(f"  Inclination:  iota={iota:.3f} rad")
    print(f"  f range:  {freqs_hz.min():.2e} - {freqs_hz.max():.2e} Hz")
    print(f"  h0 range: {h0_vals.min():.2e} - {h0_vals.max():.2e}")
    print(f"  S/N range: {np.nanmin(snr_grid):.2f} - {np.nanmax(snr_grid):.2f}")

    return freqs_hz, h0_vals, snr_grid