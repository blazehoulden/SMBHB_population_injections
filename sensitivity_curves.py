from data_loader import BEST_PSRS, SCENARIOS
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import glob, pickle, json
import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky
from enterprise.pulsar import Pulsar as ePulsar


def _set_apj_style():
    """Apply a clean, publication-friendly ApJ-style appearance."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.edgecolor": "0.2",
        "legend.facecolor": "white",
        "legend.frameon": True,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_corr(psr, raw_noise_params, thin=1):
    toas    = psr.toas[::thin]
    toaerrs = psr.toaerrs[::thin]
    flags   = psr.flags['f'][::thin]

    N = toaerrs.size
    _, _, fl, _, bi = hsen.quantize_fast(toas, toaerrs, flags=flags, dt=1)
    backends  = np.unique(flags)
    sigma_sqr = np.zeros(N)
    ecorrs    = np.zeros_like(fl, dtype=float)

    for be in backends:
        mask   = np.where(flags == be)
        key_ef = '{0}_{1}_efac'.format(psr.name, be)
        key_eq = '{0}_{1}_log10_t2equad'.format(psr.name, be)
        key_ec = '{0}_{1}_log10_ecorr'.format(psr.name, be)

        if not all(k in raw_noise_params for k in [key_ef, key_eq, key_ec]):
            missing = [k for k in [key_ef, key_eq, key_ec] if k not in raw_noise_params]
            raise KeyError(f"Missing noise params for backend '{be}': {missing}")

        sigma_sqr[mask] = (raw_noise_params[key_ef]**2 * toaerrs[mask]**2
                           + (10**raw_noise_params[key_eq])**2)
        mask_ec           = np.where(fl == be)
        ecorrs[mask_ec]   = 10**raw_noise_params[key_ec]

    j = [ecorrs[ii]**2 * np.ones((len(bucket), len(bucket)))
         for ii, bucket in enumerate(bi)]
    J = sl.block_diag(*j)
    return np.diag(sigma_sqr) + J


# ---------------------------------------------------------------------------
# Real curves
# ---------------------------------------------------------------------------

def build_real_curves(ePsrs, parsed_noise_params, raw_noise_params, freqs, thin=10):
    """Build real sensitivity curves from ePulsar objects."""
    print('--- Building real sensitivity curves ---')
    real_psrs = []
    real_specs_per_psr = {}

    for ePsr in ePsrs:
        corr = make_corr(ePsr, raw_noise_params, thin=thin)
        plaw = hsen.red_noise_powerlaw(A=9e-16, gamma=13/3., freqs=freqs)

        if (ePsr.name in parsed_noise_params and
                'red_noise' in parsed_noise_params[ePsr.name]):
            rn = parsed_noise_params[ePsr.name]['red_noise']
            plaw += hsen.red_noise_powerlaw(
                A=10**rn['log10_A'], gamma=rn['gamma'], freqs=freqs)

        corr += hsen.corr_from_psd(freqs=freqs, psd=plaw, toas=ePsr.toas[::thin])
        psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                          toaerrs=ePsr.toaerrs[::thin],
                          phi=ePsr.phi, theta=ePsr.theta,
                          N=corr, designmatrix=ePsr.Mmat[::thin, :])
        psr.name = ePsr.name
        real_psrs.append(psr)
        print(f'\r  real PSR {ePsr.name} complete', end='', flush=True)
    print()

    real_specs = []
    for p in real_psrs:
        sp = hsen.Spectrum(p, freqs=freqs)
        _ = sp.NcalInv
        real_specs_per_psr[p.name] = sp
        real_specs.append(sp)

    real_sc  = hsen.GWBSensitivityCurve(real_specs)
    real_dsc = hsen.DeterSensitivityCurve(real_specs)
    return real_sc, real_dsc, real_specs_per_psr


# ---------------------------------------------------------------------------
# Synthetic curve builder — realistic modifications of the real data
# ---------------------------------------------------------------------------

def _interleave_toas(toas, toaerrs, Mmat, flags, factor=2):
    """
    Increase cadence by inserting `factor-1` new TOAs between every consecutive
    real pair.  New TOAs inherit the noise properties of their nearest real
    neighbour and the design-matrix rows are linearly interpolated.

    Parameters
    ----------
    factor : int
        Cadence multiplier.  factor=2 → twice as many observations.

    Returns
    -------
    new_toas, new_toaerrs, new_Mmat, new_flags  (sorted by time)
    """
    n = len(toas)
    extra_toas    = []
    extra_errs    = []
    extra_Mmat    = []
    extra_flags   = []

    for i in range(n - 1):
        dt = toas[i+1] - toas[i]
        for k in range(1, factor):
            frac   = k / factor
            t_new  = toas[i] + frac * dt
            # inherit noise from nearest neighbour
            err_new = toaerrs[i] if frac < 0.5 else toaerrs[i+1]
            flag_new = flags[i] if frac < 0.5 else flags[i+1]
            # linearly interpolate design matrix row
            row_new  = (1 - frac) * Mmat[i] + frac * Mmat[i+1]
            extra_toas.append(t_new)
            extra_errs.append(err_new)
            extra_flags.append(flag_new)
            extra_Mmat.append(row_new)

    if not extra_toas:
        return toas, toaerrs, Mmat, flags

    all_toas  = np.concatenate([toas,    extra_toas])
    all_errs  = np.concatenate([toaerrs, extra_errs])
    all_flags = np.concatenate([flags,   extra_flags])
    all_Mmat  = np.vstack([Mmat, extra_Mmat])

    idx        = np.argsort(all_toas)
    return all_toas[idx], all_errs[idx], all_Mmat[idx], all_flags[idx]
def make_pta_sensitivity(
    lPsrs,
    parsed_noise_params,
    raw_noise_params,
    Tspan_seconds,
    thin              = 10,
    scenarios         = None,        # NEW: pass your SCENARIOS dict directly
    best_psrs         = None,        # NEW: pass your BEST_PSRS tuple directly
    synthetic_configs = None,        # kept for manual override as before
    outdir            = 'figures',
):
    import os; os.makedirs(outdir, exist_ok=True)
    _set_apj_style()

    _scenarios  = scenarios  if scenarios  is not None else SCENARIOS
    _best_psrs  = best_psrs  if best_psrs  is not None else BEST_PSRS

    # ------------------------------------------------------------------
    # Auto-derive synthetic_configs from SCENARIOS if not supplied manually
    # ------------------------------------------------------------------
    SCENARIO_COLORS = [
        'black', 'magenta', 'lime', 'navy',
        'purple', 'gold', 'crimson', 'teal',
    ]

    if synthetic_configs is None:
        synthetic_configs = []
        color_iter = iter(SCENARIO_COLORS)
        for scen_name, cfg in _scenarios.items():
            if scen_name == 'baseline':
                continue   # baseline is the "real" curve

            color = next(color_iter, 'grey')

            # Scenario-level defaults
            cf         = cfg.get('cadence_factor', 1)
            ef         = cfg.get('toaerr_factor',  1.0)
            best_only  = cfg.get('best_only',       True)
            scen_bpsrs = cfg.get('best_psrs',       _best_psrs)
            per_pulsar = cfg.get('per_pulsar',       None)

            # Choose mode
            if per_pulsar:
                mode = 'per_pulsar'
            elif best_only:
                if cf != 1 and ef != 1.0:
                    mode = 'best_both'
                elif cf != 1:
                    mode = 'best_cadence'
                else:
                    mode = 'best_precision'
            else:
                if cf != 1 and ef != 1.0:
                    mode = 'all_both'
                elif cf != 1:
                    mode = 'all_cadence'
                else:
                    mode = 'all_precision'

            synthetic_configs.append(dict(
                label          = scen_name,
                color          = color,
                mode           = mode,
                best_psrs      = scen_bpsrs,
                toaerr_factor  = ef,
                cadence_factor = cf,
                per_pulsar     = per_pulsar,   # None for non-per-pulsar scenarios
            ))

    ePsrs = [ePulsar(psr, ephem='DE440', backend='tempo2') for psr in lPsrs]
    freqs = np.logspace(np.log10(1 / (5 * Tspan_seconds)), np.log10(3e-7), 900)

    # --- real curves ---
    real_sc, real_dsc, _ = build_real_curves(
        ePsrs, parsed_noise_params, raw_noise_params, freqs, thin=thin)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.loglog(real_dsc.freqs, real_dsc.h_c, color='black', lw=2.2, label='NG15 deterministic')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'Characteristic strain $h_c$')
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
    ax.grid(which='both', ls='--', lw=0.45, alpha=0.35)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f'{outdir}/sensitivity_curves_real.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- synthetic curves ---
    synthetic_curves = []
    for cfg in synthetic_configs:
        try:
            sc, dsc = build_realistic_synthetic_curves(
                ePsrs               = ePsrs,
                parsed_noise_params = parsed_noise_params,
                raw_noise_params    = raw_noise_params,
                freqs               = freqs,
                thin                = thin,
                label               = cfg['label'],
                best_psrs           = cfg.get('best_psrs',       _best_psrs),
                mode                = cfg.get('mode',            'best_cadence'),
                toaerr_factor       = cfg.get('toaerr_factor',   1.0),
                cadence_factor      = cfg.get('cadence_factor',  1),
                per_pulsar          = cfg.get('per_pulsar',      None),
            )
            synthetic_curves.append((cfg['label'], cfg['color'], sc, dsc))
            print(f'  OK: "{cfg["label"]}"')
        except Exception as e:
            print(f'  ERROR for "{cfg["label"]}": {e}')
            import traceback; traceback.print_exc()

    if not synthetic_curves:
        print('WARNING: no synthetic curves built, skipping comparison plot')
        return real_sc, real_dsc, []

    plot_sensitivity_comparison(real_sc, real_dsc, synthetic_curves, outdir=outdir)
    return real_sc, real_dsc, synthetic_curves


def build_realistic_synthetic_curves(
    ePsrs,
    parsed_noise_params,
    raw_noise_params,
    freqs,
    thin                = 10,
    label               = 'Synthetic PTA',
    best_psrs           = BEST_PSRS,
    mode                = 'best_cadence',
    toaerr_factor       = 1.0,
    cadence_factor      = 1,
    per_pulsar          = None,   # NEW: dict of {psr_name: {cadence_factor, toaerr_factor}}
):
    """
    Build synthetic sensitivity curves identical to the real NG15 curves
    except for controlled modifications.

    Modes
    -----
    best_cadence    cadence_factor applied to best_psrs only
    best_precision  toaerr_factor  applied to best_psrs only
    best_both       both applied to best_psrs only
    all_cadence     cadence_factor applied to all pulsars
    all_precision   toaerr_factor  applied to all pulsars
    all_both        both applied to all pulsars
    per_pulsar      per_pulsar dict controls each pulsar individually;
                    pulsars absent from the dict and not in best_psrs
                    are left unchanged; best_psrs absent from the dict
                    fall back to the scenario-level cadence/toaerr factors.
    """
    print(f'--- Building realistic synthetic curves: [{label}] mode={mode} ---')

    psrs = []
    for ePsr in ePsrs:
        name    = ePsr.name
        is_best = name in best_psrs

        if mode == 'per_pulsar':
            if per_pulsar and name in per_pulsar:
                # Explicit per-pulsar override
                overrides = per_pulsar[name]
                cf = overrides.get('cadence_factor', cadence_factor)
                ef = overrides.get('toaerr_factor',  toaerr_factor)
            elif is_best:
                # best_psrs not individually listed fall back to scenario defaults
                cf = cadence_factor
                ef = toaerr_factor
            else:
                # Non-best pulsars are untouched
                cf = 1
                ef = 1.0

        elif mode == 'best_cadence':
            cf = cadence_factor if is_best else 1
            ef = 1.0

        elif mode == 'best_precision':
            cf = 1
            ef = toaerr_factor if is_best else 1.0

        elif mode == 'best_both':
            cf = cadence_factor if is_best else 1
            ef = toaerr_factor  if is_best else 1.0

        elif mode == 'all_cadence':
            cf = cadence_factor
            ef = 1.0

        elif mode == 'all_precision':
            cf = 1
            ef = toaerr_factor

        elif mode == 'all_both':
            cf = cadence_factor
            ef = toaerr_factor

        else:
            raise ValueError(f'Unknown mode: {mode!r}')

        try:
            psr = _build_one_psr_realistic(
                ePsr, parsed_noise_params, raw_noise_params, freqs,
                thin=thin, toaerr_factor=ef, cadence_factor=cf,
            )
            tag = ''
            if cf > 1: tag += f' cadence×{cf}'
            if ef < 1: tag += f' precision×{1/ef:.1f}'
            print(f'  [{label}] {name:20s}  n_obs={len(psr.toas):4d}{tag}')
            psrs.append(psr)
        except Exception as e:
            print(f'  WARNING: {name} skipped — {e}')

    if not psrs:
        raise RuntimeError(f'No pulsars built for "{label}"')

    specs = []
    for p in psrs:
        try:
            sp      = hsen.Spectrum(p, freqs=freqs)
            NcalInv = sp.NcalInv
            n_bad   = np.sum(~np.isfinite(NcalInv)) + np.sum(NcalInv <= 0)
            if n_bad > 0:
                print(f'  WARNING: {p.name} has {n_bad} bad NcalInv values, skipping')
                continue
            specs.append(sp)
        except Exception as e:
            print(f'  WARNING: Spectrum failed for {p.name}: {e}')

    if not specs:
        raise RuntimeError(f'No valid spectra for "{label}"')

    print(f'  {len(specs)} valid spectra for "{label}"')
    sc  = hsen.GWBSensitivityCurve(specs)
    dsc = hsen.DeterSensitivityCurve(specs)
    print(f'  h_c (GWB)   [{np.nanmin(sc.h_c):.2e}, {np.nanmax(sc.h_c):.2e}]')
    print(f'  h_c (Deter) [{np.nanmin(dsc.h_c):.2e}, {np.nanmax(dsc.h_c):.2e}]')
    return sc, dsc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sensitivity_comparison(real_sc, real_dsc, synthetic_curves, outdir='figures'):
    import os; os.makedirs(outdir, exist_ok=True)

    _set_apj_style()

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    ax.loglog(real_dsc.freqs, real_dsc.h_c, color='black', lw=2.2, label='NG15')

    for label, color, syn_sc, syn_dsc in synthetic_curves:
        print(f'  plotting {label}: h_c range [{syn_dsc.h_c.min():.2e}, {syn_dsc.h_c.max():.2e}]')
        ax.loglog(syn_dsc.freqs, syn_dsc.h_c, color=color, lw=1.8, label=label)

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'Characteristic strain $h_c$')
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
    ax.grid(which='both', ls='--', lw=0.45, alpha=0.35)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(f'{outdir}/sensitivity_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
