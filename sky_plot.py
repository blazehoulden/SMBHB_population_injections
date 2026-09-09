"""
sky_plot.py

Plotting-only code for the CGW sky-location SNR map, extracted from
test_CGW_sky_loc.py.

WHY THIS MODULE EXISTS: test_CGW_sky_loc.py's top-level imports include
`from SMBHB_pop_synth import PopulationArrays, chosen_population`, and
SMBHB_pop_synth imports numba. In some environments (numpy 2.x + certain
numba versions) numba's import-time typing registration
(`typeof_impl.register(np.polynomial.polynomial.Polynomial)`) recurses
infinitely against numpy's `numpy.core` -> `numpy._core` compatibility
shim, raising RecursionError before you even get to use anything -- purely
from importing the module, with no way to opt out short of fixing the
numba/numpy version pin.

None of that machinery is actually needed to *plot* an already-computed
sky map: plot_skymap/plot_cgw_analysis/_style_skyax/_wrap_ra only ever
touch plain floats/arrays (ra, dec, snr, psr phi/theta) -- they don't
import SMBHB_pop_synth, CGW_SNR, enterprise, or hasasia. So they're moved
here, where the only outside dependency is plot_cgw_snr (for
_snr_colormap/_star_marker) plus matplotlib/numpy/scipy/apj_style.

test_CGW_sky_loc.py should import these back from here (rather than
redefining them) so there's exactly one copy. population_analysis.ipynb
can `from sky_plot import plot_cgw_analysis` directly and never import
test_CGW_sky_loc.py (and therefore never import SMBHB_pop_synth/numba) at
all.

If plot_cgw_snr.py itself turns out to import anything numba-heavy,
_snr_colormap/_star_marker should get pulled out the same way -- check
that file's own import list if this module still triggers the
RecursionError.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.patheffects import withStroke

from apj_style import apply_apj_style, APJ_COL_WIDTH
from plot_cgw_snr import _snr_colormap, _star_marker


def _wrap_ra(ra_rad):
    ra_shifted = (ra_rad - np.pi) % (2 * np.pi)
    ra_shifted = np.where(ra_shifted > np.pi, ra_shifted - 2 * np.pi, ra_shifted)
    return -ra_shifted


# white stroke used on all axis labels so they read over any background
_STROKE = [withStroke(linewidth=2.5, foreground='white')]


def _style_skyax(ax, fig, ra_label_size=6, dec_label_size=6, dec_nudge=0.028):
    """
    RA-hour / Dec-rim labels + light dashed grid for a projection='aitoff'
    axes. See plot_cgw_skymaps.py for the original version this was
    factored out of.
    """
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    ra_hours = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    for h in ra_hours:
        rad = _wrap_ra(np.deg2rad(h * 15))
        ax.annotate(
            f'{h}h',
            xy=(rad, 0),
            xycoords='data',
            ha='center', va='center',
            fontsize=ra_label_size, fontweight='bold', color='dimgrey',
            path_effects=_STROKE,
        )

    fig.canvas.draw()
    data_to_axes = ax.transData + ax.transAxes.inverted()

    for deg in range(-75, 76, 15):
        dec_rad = np.deg2rad(deg)
        ra_rim  = _wrap_ra(np.deg2rad(359.9))
        x_ax, y_ax = data_to_axes.transform((ra_rim, dec_rad))

        ax.annotate(
            f'{deg}°',
            xy=(x_ax + dec_nudge, y_ax),
            xycoords='axes fraction',
            ha='left', va='center',
            fontsize=dec_label_size, fontweight='bold', color='dimgrey',
            path_effects=_STROKE,
            annotation_clip=False,
        )


# ---------------------------------------------------------------------------
# Skymap panel
# ---------------------------------------------------------------------------

def plot_skymap(ax, binaries, snrs, psrs, cmap, norm, n_ra=72, n_dec=36):
    """
    Aitoff-projection tiled SNR heatmap. `ax` must be created with
    projection='aitoff' — grid, boundary, and RA/Dec rim labels are
    handled by _style_skyax (same helper used by the other skymap
    figures), not drawn manually here.

    `binaries` needs .ra/.dec attributes; `psrs` needs .phi/.theta
    (colatitude) -- plain SimpleNamespace objects work fine, the real
    PopulationArrays/enterprise.Pulsar objects are not required.
    """
    from scipy.interpolate import griddata

    snra      = np.array(snrs, dtype=float)
    star_path = _star_marker()

    ra_vals  = np.array([b.ra  for b in binaries])
    dec_vals = np.array([b.dec for b in binaries])
    lon_vals = _wrap_ra(ra_vals)

    lon_fine = np.linspace(-np.pi, np.pi, 360)
    lat_fine = np.linspace(-np.pi / 2, np.pi / 2, 180)
    LON_fine, LAT_fine = np.meshgrid(lon_fine, lat_fine)

    snr_grid = griddata(
        points=np.column_stack([lon_vals, dec_vals]),
        values=snra,
        xi=np.column_stack([LON_fine.ravel(), LAT_fine.ravel()]),
        method="linear",
    ).reshape(LON_fine.shape)

    snr_grid_nn = griddata(
        points=np.column_stack([lon_vals, dec_vals]),
        values=snra,
        xi=np.column_stack([LON_fine.ravel(), LAT_fine.ravel()]),
        method="nearest",
    ).reshape(LON_fine.shape)
    snr_grid = np.where(np.isnan(snr_grid), snr_grid_nn, snr_grid)

    lon_edges = np.linspace(-np.pi, np.pi, 361)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, 181)
    LON_e, LAT_e = np.meshgrid(lon_edges, lat_edges)

    ax.pcolormesh(
        LON_e, LAT_e, snr_grid,
        cmap=cmap, norm=norm,
        shading="auto", zorder=1, rasterized=True,
    )

    # ---- pulsars: white fill, black outline, same size (s=20) used
    # everywhere else. NOTE: this was previously facecolors="black",
    # edgecolors="white" -- backwards relative to the comment/intent --
    # swapped so pulsars actually render white-centred. ----
    if psrs:
        psr_ra  = np.array([float(psr.phi) for psr in psrs])
        psr_dec = np.array([float(np.pi / 2.0 - psr.theta) for psr in psrs])
        psr_lon = _wrap_ra(psr_ra)
        ax.scatter(
            psr_lon, psr_dec,
            marker=star_path, s=30,
            facecolors="white", edgecolors="black", linewidths=0.4,
            zorder=4,
        )

    # ---- legend: same styling as the other skymap scripts (frameless,
    # bottom-anchored) rather than this file's own framealpha=0.7 box ----
    from matplotlib.lines import Line2D
    handles = [
        Line2D(
            [0], [0], marker=star_path, color="w",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=0.6, markersize=7, label="Pulsars",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.3,
    )

    _style_skyax(ax, ax.figure)


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
    figsize=None,
    annotate_top=0,
):
    """
    Publication-quality single-column ApJ sky-map, styled consistently
    with every other skymap in the paper via apj_style / _style_skyax.
    """
    snra = np.array(snrs, dtype=float)
    cmap = _snr_colormap()
    norm = Normalize(vmin=snra.min() * 0.9, vmax=snra.max() * 1.05)

    if figsize is None:
        figsize = (APJ_COL_WIDTH, APJ_COL_WIDTH * 0.8)

    apply_apj_style()

    fig = plt.figure(figsize=figsize, constrained_layout=False)

    ax_sky = fig.add_axes(
        [0.06, 0.28, 0.88, 0.63],
        projection="aitoff",
    )

    plot_skymap(ax_sky, binaries, snra, psrs, cmap, norm, n_ra=n_ra, n_dec=n_dec)

    # colorbar dropped from y=0.16 to y=0.08 (and shortened slightly)
    # so the pulsar legend -- anchored just below the sky axes, whose
    # bottom edge is now at 0.28 -- has room to sit above it instead
    # of overlapping/clipping the colorbar's own label
    cax = fig.add_axes([0.16, 0.08, 0.68, 0.03])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.locator = MaxNLocator(nbins=6)   # was showing only ~2 ticks
    cbar.update_ticks()
    cbar.set_label(r"(S/N)$_{\mathrm{CW}}$", labelpad=1)   # matches the other panels'
                                                    # label format, not
                                                    # $(\mathrm{S/N})_{\mathrm{CW}}$
    cbar.ax.tick_params(colors="black", labelsize=7)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        from pathlib import Path as FilePath
        pdf_path = FilePath(save_path).with_suffix(".pdf")
        fig.savefig(pdf_path)
        print(f"Saved to {save_path} (and {pdf_path})")
    else:
        plt.show()

    return fig