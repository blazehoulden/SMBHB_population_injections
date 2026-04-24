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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G   = 6.674e-11   # m^3 kg^-1 s^-2
c   = 3.0e8       # m s^-1
M_sun = 1.989e30  # kg
pc_to_m = 3.086e16
Mpc_to_m = 3.086e22


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snr_colormap():
    """Blue → purple → coral ramp matching SNR intensity."""
    return plt.cm.get_cmap("plasma")


def _aitoff_xy(ra, dec):
    """
    Convert (ra, dec) in radians to Aitoff projected (x, y).
    ra  in [0, 2pi] — shifted so that ra=pi maps to centre (x=0).
    """
    lon = ra - np.pi
    lat = dec
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha  = np.arccos(np.cos(lat) * np.cos(lon / 2.0))
        sinc_a = np.where(np.abs(alpha) < 1e-10, 1.0, np.sin(alpha) / alpha)
        x = 2.0 * np.cos(lat) * np.sin(lon / 2.0) / sinc_a
        y = np.sin(lat) / sinc_a
    return x, y


def _star_marker(n=5, inner=0.45):
    """Return a matplotlib Path for an n-pointed star."""
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 2 * n + 1)
    radii  = np.tile([1.0, inner], n + 1)[:2 * n + 1]
    verts  = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    codes  = [Path.MOVETO] + [Path.LINETO] * (2 * n - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------
def plot_mc_vs_distance(ax, binaries, snrs, cmap, norm, annotate_top=5):
    """Chirp mass vs comoving distance, with SNR as color and size."""
    mcs   = np.array([b.Mc for b in binaries])
    dists = np.array([b.D_comov for b in binaries])
    snra  = np.array(snrs)

    sizes = 30 + snra * 40
    valid = np.isfinite(mcs) & np.isfinite(dists)

    sc = ax.scatter(
        mcs[valid], dists[valid],
        s=sizes[valid],
        c=snra[valid],
        cmap=cmap, norm=norm,
        edgecolors="white", linewidths=0.5, zorder=3
    )

    top_idx = np.where(valid)[0][np.argsort(snra[valid])[-annotate_top:]]
    for i in top_idx:
        ax.annotate(
            f"#{i+1}",
            (mcs[i], dists[i]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="0.3"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathcal{M}_c$ [M$_\odot$]")
    ax.set_ylabel(r"$D_{\rm comov}$ [Mpc]")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)

    return sc


def plot_h0_vs_frequency(ax, binaries, snrs, cmap, norm, annotate_top=5):
    """Strain amplitude vs frequency, with SNR as color and size."""
    h0s  = np.array([b.h0 for b in binaries])
    freqs = np.array([b.f for b in binaries])   # observed GW frequency
    snra = np.array(snrs)

    sizes = 30 + snra * 40
    valid = np.isfinite(h0s) & np.isfinite(freqs)

    sc = ax.scatter(
        h0s[valid], freqs[valid],
        s=sizes[valid],
        c=snra[valid],
        cmap=cmap, norm=norm,
        edgecolors="white", linewidths=0.5, zorder=3
    )

    top_idx = np.where(valid)[0][np.argsort(snra[valid])[-annotate_top:]]
    for i in top_idx:
        ax.annotate(
            f"#{i+1}",
            (h0s[i], freqs[i]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="0.3"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$h_0 = 2\frac{(G\mathcal{M})^{5/3}}{c^4D_{\rm{comov}}}(\pi f_{\rm GW}(1 + z))^{2/3}$")
    ax.set_ylabel(r"$f_{\rm GW}$ [Hz]")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)

    return sc

def plot_skymap(ax, binaries, snrs, psrs, cmap, norm):
    """
    Aitoff projection skymap.
    Binaries: circles, size ∝ SNR, colour ∝ SNR.
    Pulsars:  red 5-pointed stars.
    """
    snra      = np.array(snrs)
    max_snr   = snra.max()
    star_path = _star_marker()

    # ---- draw grid lines in RA/Dec ----
    ra_grid  = np.linspace(0, 2 * np.pi, 500)
    dec_grid = np.linspace(-np.pi / 2, np.pi / 2, 500)

    for dec_val in np.radians([-60, -30, 0, 30, 60]):
        xs, ys = _aitoff_xy(ra_grid, np.full_like(ra_grid, dec_val))
        ax.plot(xs, ys, color="0.7", lw=0.4, zorder=0)

    for ra_val in np.linspace(0, 2 * np.pi, 13)[:-1]:
        xs, ys = _aitoff_xy(np.full_like(dec_grid, ra_val), dec_grid)
        ax.plot(xs, ys, color="0.7", lw=0.4, zorder=0)

    # equator
    xs, ys = _aitoff_xy(ra_grid, np.zeros_like(ra_grid))
    ax.plot(xs, ys, color="0.5", lw=0.8, zorder=1)

    # ---- draw Aitoff projection boundary ----
    # Right edge (RA = 0)
    xs, ys = _aitoff_xy(np.zeros_like(dec_grid), dec_grid)
    ax.plot(xs, ys, color="0.5", lw=1.0, zorder=2)
    
    # Left edge (RA = π, the seam)
    xs, ys = _aitoff_xy(np.full_like(dec_grid, np.pi), dec_grid)
    ax.plot(xs, ys, color="0.5", lw=1.0, zorder=2)

    # ---- pulsars ----
    for psr in psrs:
        ra_psr  = float(psr.phi)
        dec_psr = float(np.pi / 2.0 - psr.theta)
        px, py  = _aitoff_xy(ra_psr, dec_psr)
        ax.scatter(
            px, py,
            marker=star_path, s=80,
            c="#E24B4A", edgecolors="#8B1A1A", linewidths=0.5,
            zorder=5, label="_pulsar",
        )

    # ---- binaries ----
    for i, (b, snr) in enumerate(zip(binaries, snra)):
        bx, by = _aitoff_xy(float(b.ra), float(b.dec))
        r = 20 + (snr / max_snr) * 200
        ax.scatter(
            bx, by, s=r,
            c=[[cmap(norm(snr))]],
            edgecolors="white", linewidths=0.5,
            zorder=4,
        )
        if snr >= sorted(snra)[-5]:
            ax.annotate(
                f"#{i+1}",
                (bx, by), xytext=(4, 4),
                textcoords="offset points",
                fontsize=7, color="0.2", zorder=6,
            )

    # ---- dummy legend handles ----
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o",  color="w", markerfacecolor=cmap(0.8),
               markersize=9, label="CGW binary (size ∝ SNR)"),
        Line2D([0], [0], marker=star_path, color="w", markerfacecolor="#E24B4A",
               markersize=9, label="Pulsar"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.6)

    ax.set_axis_off()
    ax.set_title("Sky positions", fontsize=11)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def plot_cgw_analysis(
    top_binaries,
    top_snrs,
    psrs,
    save_path=None,
    figsize=(14, 11),
    style="dark_background",
    annotate_top=5,
):
    """
    Generate a 2×2 panel figure:
        top-left  : h0 vs SNR
        top-right : Mc vs SNR
        bottom-left : D_L vs SNR
        bottom-right: Skymap (Aitoff)

    Parameters
    ----------
    top_binaries : list
        Binary objects (must have .h0, .Mc, .f, .ra, .dec).
    top_snrs : list or array-like
        Optimal SNR values matching top_binaries order.
    psrs : list
        enterprise Pulsar objects (must have .phi, .theta, .name).
    save_path : str or None
        If given, saves to this path (PDF/PNG/SVG inferred from extension).
        If None, calls plt.show().
    figsize : tuple
    style : str
        Any valid matplotlib style, e.g. "dark_background", "seaborn-v0_8-darkgrid".
    annotate_top : int
        How many top-SNR sources to label.
    """
    snra = np.array(top_snrs, dtype=float)
    cmap = _snr_colormap()
    norm = Normalize(vmin=snra.min() * 0.9, vmax=snra.max() * 1.05)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        fig.suptitle("CGW candidate properties — optimal SNR", fontsize=13, y=0.98)

        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.35, wspace=0.32,
                               left=0.08, right=0.93, top=0.93, bottom=0.07)

        ax_h0_freq   = fig.add_subplot(gs[0, 0])
        ax_mc_dist   = fig.add_subplot(gs[1, 0])
        ax_sky  = fig.add_subplot(gs[2, 0])

        plot_h0_vs_frequency(ax_h0_freq,   top_binaries, snra, cmap, norm, annotate_top)
        plot_mc_vs_distance(ax_mc_dist,   top_binaries, snra, cmap, norm, annotate_top)
        plot_skymap(ax_sky, top_binaries, snra, psrs, cmap, norm)

        # shared colourbar on right edge
        cbar_ax = fig.add_axes([0.95, 0.1, 0.015, 0.8])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Optimal SNR", fontsize=10)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    for each one.  Expects each entry to have keys:
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
        top_snrs     = result.get("top_snrs")

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

