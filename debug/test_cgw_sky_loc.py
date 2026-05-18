from SMBHB_pop_synth import chosen_population
from plot_cgw_snr import _snr_colormap

def test_sky_CGW_SNR_location(psrs_clean, raw_noise_params, parsed_noise_params, Tspan):
    
    # generate uniformly distributed sky locations
    num_bhs = 36
    dec_list = np.linspace(-np.pi / 2, np.pi / 2, int(np.sqrt(num_bhs)))
    ra_list = np.linspace(0, 2 * np.pi, int(np.sqrt(num_bhs)))
    sky_locations = [(ra, dec) for ra in ra_list for dec in dec_list]
    # create a population with these sky locations

    population = chosen_population(right_ascension=[loc[0] for loc in sky_locations], declination=[loc[1] for loc in sky_locations])
    from consistent_pop_synth import compute_population_snr
    original_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}
    SGWB_snrs, pta, enterprise_psrs = compute_population_snr(population, psrs_clean=psrs_clean, current_stoas=original_stoas, return_psrs_pta=True)
    from CGW_SNR import compute_cgw_snr_optimal_population
    cgw_snrs_optimal = compute_cgw_snr_optimal_population(psrs=enterprise_psrs, pta=pta, population=population, Tspan=Tspan, raw_noise_params=raw_noise_params, parsed_noise_params=parsed_noise_params)

    # plot the sky map of CGW SNRs
    plot_cgw_analysis(binaries=population, snrs=cgw_snrs_optimal, psrs=enterprise_psrs, save_path="cgw_snr_sky_map_comp.pdf", annotate_top=0)
    return
    




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
    # Left edge (RA = 0)
    xs, ys = _aitoff_xy(np.zeros_like(dec_grid), dec_grid)
    ax.plot(xs, ys, color="0.5", lw=1.0, zorder=2)
    
    # Right edge (RA = 2π)
    xs, ys = _aitoff_xy(np.full_like(dec_grid, 2 * np.pi), dec_grid)
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
def plot_cgw_analysis(
    binaries,
    snrs,
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
    binaries : list
        Binary objects (must have .h0, .Mc, .f, .ra, .dec).
    snrs : list or array-like
        Optimal SNR values matching binaries order.
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
    snra = np.array(snrs, dtype=float)
    cmap = _snr_colormap()
    norm = Normalize(vmin=snra.min() * 0.9, vmax=snra.max() * 1.05)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        fig.suptitle("CGW candidate properties — optimal SNR", fontsize=13, y=0.98)

        gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.35, wspace=0.32,
                               left=0.08, right=0.93, top=0.93, bottom=0.07)

        ax_sky  = fig.add_subplot(gs[0, 0])

        plot_skymap(ax_sky, binaries, snra, psrs, cmap, norm)

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