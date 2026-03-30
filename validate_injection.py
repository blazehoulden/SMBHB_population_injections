"""
validate_injection.py  —  Compare inject_population_nufft vs old vectorised method
====================================================================================
Run this with a small population (N~10-50) where both methods should agree exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from signal_injection import inject_population_direct, inject_population_nufft, inject_population_nufft, population_residuals_vectorised
from SMBHB_pop_synth import precompute_amplitudes
from consistent_pop_synth import compute_population_snr


def compare_injection_methods(
    psrs,
    population,          # PopulationArrays
    Tspan,
    verbose=True,
    plot=True,
    rtol=1e-3,           # relative tolerance for agreement check
):
    """
    Inject the same population with both methods and compare residuals.

    Parameters
    ----------
    psrs       : list of enterprise Pulsar objects
    population : PopulationArrays (small N recommended, ~10-50)
    Tspan      : observation span [s]
    rtol       : relative tolerance for pass/fail (default 1e-3 = 0.1%)

    Returns
    -------
    results : dict with per-pulsar residuals and agreement metrics
    """

    # Convert PopulationArrays to list-of-dicts for the old method
    # (old vectorised code expects dicts with 'f', 'Mc', 'D_comov', etc.)
    pop_dict = population.to_dict_list()
    for i in range(len(pop_dict)):
        pop_dict[i]['Mc'] *= 1.98847e30  # convert from Msun to kg if needed

    # ── precompute amplitudes for NUFFT method ───────────────────────────────
    for psr in psrs:
        precompute_amplitudes(population, psr)

    results = {}

    for psr in psrs:
        psr_name = psr.name

        # ── Method 1: NUFFT ──────────────────────────────────────────────────
        psr._residuals = np.zeros(len(psr.toas))
        inject_population_nufft(
            [psr], population,
            N_freq      = None,
            pure_signal = True,
            verbose     = True,
        )
        r_nufft = psr._residuals.copy()

        # ── Method 2: old vectorised ─────────────────────────────────────────
        # population_residuals_vectorised expects list-of-dicts and returns
        # the residual array directly (doesn't set psr._residuals)
        t_sec = np.asarray(psr.toas, dtype=np.float64)
        r_old = population_residuals_vectorised(
            t_sec, psr, pop_dict, Tspan=Tspan,
            include_GW=True, include_RN=False, include_WN=False,
        )
        # print(r_old)

        # ── metrics ──────────────────────────────────────────────────────────
        diff        = r_nufft - r_old
        rms_nufft   = r_nufft.std()
        rms_old     = r_old.std()
        rms_diff    = diff.std()
        rel_diff    = rms_diff / rms_old if rms_old > 0 else np.nan
        max_abs_err = np.abs(diff).max()
        corr        = np.corrcoef(r_nufft, r_old)[0, 1]
        passed      = rel_diff < rtol

        results[psr_name] = {
            'r_nufft'    : r_nufft,
            'r_old'      : r_old,
            'diff'       : diff,
            'rms_nufft'  : rms_nufft,
            'rms_old'    : rms_old,
            'rms_diff'   : rms_diff,
            'rel_diff'   : rel_diff,
            'max_abs_err': max_abs_err,
            'corr'       : corr,
            'passed'     : passed,
        }

        if verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}  {psr_name}")
            print(f"       RMS (NUFFT):  {rms_nufft*1e9:.4f} ns")
            print(f"       RMS (old):    {rms_old*1e9:.4f} ns")
            print(f"       RMS (diff):   {rms_diff*1e9:.4f} ns")
            print(f"       Rel diff:     {rel_diff*100:.4f} %")
            print(f"       Max |err|:    {max_abs_err*1e9:.4f} ns")
            print(f"       Correlation:  {corr:.8f}")

    # ── summary ──────────────────────────────────────────────────────────────
    n_pass = sum(r['passed'] for r in results.values())
    n_tot  = len(results)

    if verbose:
        print(f"\n{'='*50}")
        print(f"SUMMARY: {n_pass}/{n_tot} pulsars within rtol={rtol*100:.1f}%")
        worst = max(results.items(), key=lambda x: x[1]['rel_diff'])
        print(f"Worst:   {worst[0]}  rel_diff={worst[1]['rel_diff']*100:.4f}%")
        print(f"{'='*50}")

    # ── plot ─────────────────────────────────────────────────────────────────
    if plot:
        n_psrs  = min(4, len(psrs))   # plot first 4 pulsars
        fig, axes = plt.subplots(n_psrs, 2, figsize=(12, 3*n_psrs))
        if n_psrs == 1:
            axes = axes[np.newaxis, :]

        for i, psr in enumerate(psrs[:n_psrs]):
            r     = results[psr.name]
            t_yrs = (psr.toas - psr.toas.min()) / (365.25 * 86400)

            # Left: overlay residuals
            ax = axes[i, 0]
            ax.plot(t_yrs, r['r_old'  ]*1e9, 'k-',  lw=1.5, label='Old vectorised', alpha=0.8)
            ax.plot(t_yrs, r['r_nufft']*1e9, 'r--', lw=1.0, label='NUFFT',          alpha=0.8)
            ax.set_ylabel('Residual [ns]')
            ax.set_title(f"{psr.name}  (corr={r['corr']:.6f})")
            ax.legend(fontsize=8)
            if i == n_psrs - 1:
                ax.set_xlabel('Time [yr]')

            # Right: difference
            ax2 = axes[i, 1]
            ax2.plot(t_yrs, r['diff']*1e9, 'b-', lw=0.8)
            ax2.axhline(0, color='k', lw=0.5, ls='--')
            ax2.set_ylabel('NUFFT − Old [ns]')
            ax2.set_title(f"rel diff = {r['rel_diff']*100:.3f}%")
            if i == n_psrs - 1:
                ax2.set_xlabel('Time [yr]')

        plt.suptitle(f'Injection comparison  (N={len(population)} binaries)',
                     fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig('injection_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: injection_comparison.png")

    return results


def compare_os_snr(psrs, population, Tspan, detailed_noise_params,
                   verbose=True):
    """
    Run the full OS SNR comparison between both injection methods.

    Calls compare_injection_methods first to check residuals agree,
    then computes OS SNR with each set of residuals.
    """
    from copy import deepcopy

    # Need independent copies of psrs so the two methods don't interfere
    pop_dict = population.to_dict_list()
    # ── inject old ───────────────────────────────────────────────────────────
    for psr in psrs:
        t_sec = np.asarray(psr.toas, dtype=np.float64)
        psr._residuals = population_residuals_vectorised(
            t_sec, psr, pop_dict, Tspan=Tspan,
            include_GW=True, include_RN=False, include_WN=False,
        )

    snr_old = compute_population_snr(population, psrs, raw_noise_params=detailed_noise_params, Tspan=Tspan)
    
    from consistent_pop_synth import _restore_zero_residuals
    _restore_zero_residuals(psrs)
        # ── precompute for NUFFT ─────────────────────────────────────────────────
    for psr in psrs:
        precompute_amplitudes(population, psr)

    # ── inject NUFFT ─────────────────────────────────────────────────────────
    inject_population_nufft(
        psrs, population,
        N_freq=None, pure_signal=True, verbose=False,
    )

    snr_nufft = compute_population_snr(population, psrs, raw_noise_params=detailed_noise_params, Tspan=Tspan)

    if verbose:
        print(f"\n{'='*50}")
        print(f"OS SNR COMPARISON")
        print(f"{'='*50}")
        print(f"  NUFFT method:  SNR = {snr_nufft:.6f}")
        print(f"  Old method:    SNR = {snr_old:.6f}")
        print(f"  Relative diff: {abs(snr_nufft - snr_old)/abs(snr_old)*100:.4f}%")
        print(f"{'='*50}")

    return {
        'snr_nufft': snr_nufft,
        'snr_old'  : snr_old,
        'rel_diff' : abs(snr_nufft - snr_old) / abs(snr_old),
    }

