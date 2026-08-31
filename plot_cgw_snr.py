"""
CGW SNR Analysis Plots
======================
Plotting utilities for visualising the properties of continuous gravitational
wave (CGW) binary candidates ranked by their optimal matched-filter SNR.

Usage
-----
    from plot_cgw_snr import plot_cgw_analysis

    plot_cgw_analysis(
        top_binaries = top_binaries,   # list of binary objects
        top_snrs     = top_snrs,       # list/array of SNR values (same order)
        psrs         = psrs,           # list of enterprise Pulsar objects
        save_path    = "cgw_plots.pdf" # optional; omit to show interactively
    )

Binary objects must expose at least:
    b.h0   – strain amplitude
    b.Mc   – chirp mass [M_sun]
    b.f    – GW frequency [Hz]
    b.ra   – right ascension [radians]
    b.dec  – declination [radians]

Optional attributes used when present:
    b.D_comov  – comoving distance [metres]  (fallback: estimated from h0/Mc/f)
    b.z        – redshift

Pulsar objects must expose:
    psr.name   – string
    psr.phi    – azimuthal sky angle = RA [radians]
    psr.theta  – polar sky angle, where theta = pi/2 - dec [radians]
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path as FilePath

from apj_style import apply_apj_style, APJ_COL_WIDTH
from signal_injection import _get_psr_radec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 6.674e-11     # m^3 kg^-1 s^-2
c = 3.0e8         # m s^-1
M_sun = 1.989e30  # kg
pc_to_m = 3.086e16
Mpc_to_m = 3.086e22

_STROKE = [withStroke(linewidth=2.5, foreground='white')]

# Tweak these two to change the marker sizes independently:
PULSAR_STAR_SIZE = 30     # pulsar star marker size
BINARY_MARKER_SIZE = 30   # fixed binary circle size — colour alone encodes SNR


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _snr_colormap():
    """Blue → purple → coral ramp matching SNR intensity."""
    return plt.cm.get_cmap("plasma")


def _set_apj_style():
    """Kept for backward compatibility with existing call sites."""
    apply_apj_style()


def _aitoff_xy(ra, dec):
    """
    Convert (ra, dec) in radians to Aitoff projected (x, y).
    ra  in [0, 2pi] — shifted so that ra=pi maps to centre (x=0).

    NOTE: plot_skymap below now uses matplotlib's native 'aitoff' projection
    axes directly (with _wrap_ra), so this manual projection is no longer
    used there. Kept here only for backward compatibility in case other
    code in your pipeline still calls it directly.
    """
    lon = ra - np.pi
    lat = dec
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.arccos(np.cos(lat) * np.cos(lon / 2.0))
        sinc_a = np.where(np.abs(alpha) < 1e-10, 1.0, np.sin(alpha) / alpha)
        x = 2.0 * np.cos(lat) * np.sin(lon / 2.0) / sinc_a
        y = np.sin(lat) / sinc_a
    return x, y


def _star_marker(n=5, inner=0.45):
    """
    Return a matplotlib Path for an n-pointed star.

    NOTE: plot_skymap below now just uses marker='*' to match snippet 1
    exactly, so this custom path is no longer used there. Kept here only
    for backward compatibility.
    """
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 2 * n + 1)
    radii = np.tile([1.0, inner], n + 1)[:2 * n + 1]
    verts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    codes = [Path.MOVETO] + [Path.LINETO] * (2 * n - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _wrap_ra(ra_rad):
    """Wrap RA (radians) into [-pi, pi] and flip sign, matching the sky-map
    convention used throughout this module (RA increasing right-to-left)."""
    ra_shifted = (ra_rad - np.pi) % (2 * np.pi)
    ra_shifted = np.where(ra_shifted > np.pi, ra_shifted - 2 * np.pi, ra_shifted)
    return -ra_shifted


def _style_skyax(ax, fig, ra_label_size=6, dec_label_size=6, dec_nudge=0.028):
    """Grid + RA/Dec label styling for the aitoff sky map: dashed grid, hour
    labels through the equator, degree labels hugging the right rim."""
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
        ra_rim = _wrap_ra(np.deg2rad(359.9))
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


def _thin_overlapping_ticklabels(ax, axis='x'):
    """
    Hide tick labels (major + minor) that would visually overlap their
    neighbour, instead of showing every one and letting them collide.
    Keeps the first label in each overlapping cluster (in position order),
    drops the rest. Must be called after the figure has a renderer, so it
    triggers its own fig.canvas.draw() internally.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    get_labels = ax.get_xticklabels if axis == 'x' else ax.get_yticklabels
    labels = list(get_labels(minor=False)) + list(get_labels(minor=True))

    entries = []
    for label in labels:
        if not label.get_text():
            continue
        bbox = label.get_window_extent(renderer=renderer)
        pos = bbox.x0 if axis == 'x' else bbox.y0
        entries.append((pos, bbox, label))

    entries.sort(key=lambda e: e[0])

    last_bbox = None
    for _, bbox, label in entries:
        if last_bbox is not None and bbox.overlaps(last_bbox):
            label.set_visible(False)
        else:
            last_bbox = bbox


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_mc_vs_distance(ax, binaries, snrs, cmap, norm, annotate_top=5):
    """Chirp mass vs luminosity distance, with SNR as colour and size."""
    mcs = np.array([b.Mc for b in binaries])
    dists = np.array([b.D_comov for b in binaries])
    zs = np.array([b.z for b in binaries])
    lum_dists = dists * (1 + zs)  # D_L = D_comov * (1 + z)
    snra = np.array(snrs)

    sizes = 30 + snra * 40
    valid = np.isfinite(mcs) & np.isfinite(lum_dists)

    sc = ax.scatter(
        mcs[valid], lum_dists[valid],
        s=sizes[valid],
        c=snra[valid],
        cmap=cmap, norm=norm,
        edgecolors="black", linewidths=0.35, zorder=3
    )

    if annotate_top > 0:
        top_idx = np.where(valid)[0][np.argsort(snra[valid])[-annotate_top:]]
        for i in top_idx:
            ax.annotate(
                f"#{i+1}",
                (mcs[i], lum_dists[i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=plt.rcParams['legend.fontsize'] - 2,
                color="0.15"
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathcal{M}$ [M$_\odot$]")
    ax.set_ylabel(r"$D_{\rm{L}}$ [Mpc]")
    ax.tick_params(which="both")
    ax.grid(True, which="both", ls="--", lw=0.35, alpha=0.28, color="0.75")

    # Cap how many major ticks the log locator proposes, then drop any
    # minor/major labels that still end up overlapping once rendered.
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    _thin_overlapping_ticklabels(ax, axis='x')

    return sc


def plot_h0_vs_frequency(ax, binaries, snrs, cmap, norm, annotate_top=0):
    """Strain amplitude vs frequency, with SNR as colour and size."""
    h0s = np.array([b.h0 for b in binaries])
    freqs = np.array([b.f for b in binaries])
    snra = np.array(snrs)

    sizes = 30 + snra * 40
    valid = np.isfinite(h0s) & np.isfinite(freqs)

    sc = ax.scatter(
        h0s[valid], freqs[valid],
        s=sizes[valid],
        c=snra[valid],
        cmap=cmap, norm=norm,
        edgecolors="black", linewidths=0.35, zorder=3
    )

    if annotate_top > 0:
        top_idx = np.where(valid)[0][np.argsort(snra[valid])[-annotate_top:]]
        for i in top_idx:
            ax.annotate(
                f"#{i+1}",
                (h0s[i], freqs[i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=plt.rcParams['legend.fontsize'] - 2,
                color="0.15"
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$h_0$")
    ax.set_ylabel(r"$f$ [Hz]")
    ax.tick_params(which="both")
    ax.grid(True, which="both", ls="--", lw=0.35, alpha=0.28, color="0.75")

    # Same overlap fix as plot_mc_vs_distance, applied here too.
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    _thin_overlapping_ticklabels(ax, axis='x')

    return sc


def plot_skymap(ax, binaries, snrs, psrs, cmap, norm, fig=None):
    """
    Aitoff projection skymap:
      - matplotlib native 'aitoff' projection (ax must be created with
        projection='aitoff')
      - RA wrapped/negated via _wrap_ra so orientation matches the other
        sky-map figures in this project
      - draw order: grid/RA-Dec labels -> pulsar stars -> binary circles
        (each with an explicit zorder too, so the stacking is guaranteed
        regardless of call order)
      - binaries: circles, ALL the same fixed size (BINARY_MARKER_SIZE),
        colour ∝ SNR (plasma)
      - pulsars: white 5-pointed stars, black edge, size = PULSAR_STAR_SIZE
    """
    if fig is None:
        fig = ax.figure

    snra = np.array(snrs, dtype=float)

    # ---- 1) grid + RA/Dec labels first (background layer) ----
    _style_skyax(ax, fig)

    # ---- 2) pulsars ----
    if len(psrs) > 0:
        psr_ra, psr_dec = [], []
        for psr in psrs:
            ra_psr, dec_psr = _get_psr_radec(psr)
            psr_ra.append(ra_psr)
            psr_dec.append(dec_psr)
        psr_ra = _wrap_ra(np.asarray(psr_ra, dtype=float))
        psr_dec = np.asarray(psr_dec, dtype=float)

        ax.scatter(
            psr_ra, psr_dec,
            s=PULSAR_STAR_SIZE, marker='*',
            color='black',
            edgecolors='black',
            linewidths=0.4,
            alpha=0.95,
            label='Pulsars',
            zorder=4,
        )

    # ---- 3) binaries — all one size now, colour still encodes SNR ----
    bin_ra = _wrap_ra(np.array([float(b.ra) for b in binaries]))
    bin_dec = np.array([float(b.dec) for b in binaries])

    ax.scatter(
        bin_ra, bin_dec,
        s=BINARY_MARKER_SIZE,
        c=snra,
        cmap=cmap, norm=norm,
        edgecolors='black', linewidths=0.3,
        zorder=5,
        label='SMBHB',
    )

    # ---- dummy legend handles (fixed-size proxy, matches actual markers) ----
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(norm(np.median(snra))),
               markeredgecolor="black", markersize=9, label="SMBHB"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFFFFF",
               markeredgecolor="#000000", markersize=9, label="Pulsars"),
    ]
    ax._legend_handles_labels_override = (handles, [h.get_label() for h in handles])


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def plot_cgw_analysis(
    top_binaries,
    top_snrs,
    psrs,
    save_path=None,
    figsize=None,
    style="default",
    annotate_top=0,
):
    snra = np.array(top_snrs, dtype=float)
    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=snra.min() * 0.9, vmax=snra.max() * 1.05)

    if figsize is None:
        figsize = (APJ_COL_WIDTH, APJ_COL_WIDTH * 1.9)

    with plt.style.context(style):
        apply_apj_style()
        fig = plt.figure(figsize=figsize, constrained_layout=False)

        panel_left, panel_right = 0.22, 0.95

        # Sky map takes up most of the figure width (wider than the h0/mc
        # panels above it, which are narrower because they carry y-axis
        # tick labels). Centered on the full figure.
        sky_width = 0.92
        sky_left = (1.0 - sky_width) / 2.0

        # Legend and colorbar are centered on the sky map itself (same
        # left/width as ax_sky) so they visually align with the map above.
        legend_width = sky_width
        legend_left = sky_left

        cbar_width = sky_width * 0.95
        cbar_left = (1.0 - cbar_width) / 2.0

        h0_bottom, h0_height = 0.74, 0.22
        gap_h0_mc = 0.075
        mc_height = 0.22
        mc_bottom = h0_bottom - gap_h0_mc - mc_height

        gap_mc_sky = 0.065
        sky_height = 0.27
        sky_bottom = mc_bottom - gap_mc_sky - sky_height

        # Tightened: sky map sits almost flush against its legend.
        gap_sky_legend = 0.0
        legend_height = 0.045
        legend_bottom = sky_bottom - gap_sky_legend - legend_height

        gap_legend_cbar = 0.0
        cbar_height = 0.025
        cbar_bottom = legend_bottom - gap_legend_cbar - cbar_height

        ax_h0_freq = fig.add_axes([panel_left, h0_bottom, panel_right - panel_left, h0_height])
        ax_mc_dist = fig.add_axes([panel_left, mc_bottom, panel_right - panel_left, mc_height])
        ax_sky = fig.add_axes([sky_left, sky_bottom, sky_width, sky_height], projection='aitoff')
        ax_legend = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

        plot_h0_vs_frequency(ax_h0_freq, top_binaries, snra, cmap, norm, annotate_top)
        plot_mc_vs_distance(ax_mc_dist, top_binaries, snra, cmap, norm, annotate_top)
        plot_skymap(ax_sky, top_binaries, snra, psrs, cmap, norm, fig=fig)

        # aitoff axes don't support get_legend_handles_labels() the same way
        # after set_axis_off(), so plot_skymap stashes its own handles/labels
        handles, labels = getattr(
            ax_sky, "_legend_handles_labels_override", ([], [])
        )
        ax_legend.axis("off")
        if handles:
            ax_legend.legend(
                handles, labels, loc="center", ncol=2,
                frameon=False, handletextpad=0.4, columnspacing=1.2,
            )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(r"(S/N)$_{\rm{CW}}$")
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            save_stem = FilePath(save_path)
            png_path = save_stem.with_suffix(".png")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

    return fig


# ---------------------------------------------------------------------------
# Convenience: per-population wrapper matching the loop in your pipeline
# ---------------------------------------------------------------------------

def plot_population_results(
    consistent_results,
    noise_params,
    save_prefix="cgw_pop",
    **plot_kwargs,
):
    """
    Iterate over consistent_results["populations"] and call plot_cgw_analysis
    for each one. Expects each entry to have keys:
        "population", "pta", "psrs", "top_binaries", "top_snrs"

    If top_binaries / top_snrs are not yet stored on the dict, pass them in
    before calling this, or compute them inline.

    Parameters
    ----------
    consistent_results : dict
        Output dict from your pipeline.
    noise_params : dict
        Noise parameter dict passed through (used only if you want to recompute
        SNRs here; otherwise assumed pre-computed).
    save_prefix : str
        Files will be saved as {save_prefix}_pop1.pdf, _pop2.pdf, etc.
    **plot_kwargs
        Forwarded to plot_cgw_analysis (e.g. style=, figsize=, annotate_top=).
    """
    for pop_idx, result in enumerate(consistent_results["populations"], start=1):
        top_binaries = result.get("top_binaries")
        top_snrs = result.get("top_snrs")

        if top_binaries is None or top_snrs is None:
            print(f"Population {pop_idx}: top_binaries/top_snrs not found — skipping.")
            continue

        psrs = result["psrs"]
        path = f"{save_prefix}_pop{pop_idx}.pdf"
        print(f"Plotting population {pop_idx} → {path}")
        plot_cgw_analysis(
            top_binaries=top_binaries,
            top_snrs=top_snrs,
            psrs=psrs,
            save_path=path,
            **plot_kwargs,
        )