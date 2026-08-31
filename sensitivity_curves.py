"""
Sensitivity curve pipeline.

MERGE NOTES (why things are organized this way):

- The CANONICAL forecasting path is `make_pta_sensitivity` (== the old
  `make_pta_sensitivity_from_tims`, now just one function). It builds every
  scenario's pulsars via `data_loader.load_pulsars(scenario=...)`, which
  reads REAL (or properly-simulated) .tim files already extended to the
  forecast baseline on disk. This is the only path that should be used for
  actual forecasting comparisons.

- `_interleave_toas` / `_build_one_psr_realistic` / `build_realistic_synthetic_curves`
  are kept as a LEGACY in-memory fallback for cadence/precision changes
  WITHOUT extending the time baseline (e.g. "what if I just observed my
  existing 4.5yr span more densely/precisely"). They deliberately do NOT
  support extending Tspan anymore -- an earlier version added
  `_extend_toas_to_tspan`, which forecast a longer baseline by copying the
  LAST real epoch's design-matrix row onto every fabricated future epoch.
  That's physically wrong for a forecast: design-matrix rows are partial
  derivatives of timing residuals w.r.t. fit parameters (spindown, proper
  motion, parallax, binary params, ...) and those derivatives grow with
  time -- freezing them at the last real epoch misrepresents exactly the
  thing a longer baseline is supposed to improve, and it silently biases
  NcalInv (and therefore h_c) for every forecast scenario. It also always
  inherited the LAST backend epoch's radio frequency for every fabricated
  epoch, which can bias chromatic/DM corrections under multi-band
  scheduling. That function has been removed entirely -- if you need Tspan
  extension, do it via real .tim files (data_loader/load_pulsars), not in
  memory here.

- Single source of truth for SCENARIOS: imported ONCE from data_loader.
  (An earlier draft defined a second, DIFFERENT module-level `SCENARIOS`
  dict here, and the two orchestration functions defaulted to different
  dicts -- editing one wouldn't update the other. Don't reintroduce that.)

- Visual style: matplotlib rcParams for ApJ single-column figures
  (3.5 x 2.8 in), legend placed below the axes via bbox_to_anchor +
  subplots_adjust (so it's visible both inline via plt.show() and in the
  saved file, not just after bbox_inches='tight' rescues it at save time),
  frameon=False, output as .pdf. `apj_style.py` is a HARD dependency --
  there is deliberately no try/except fallback here. An earlier version
  fell back to a different, smaller-font inline rcParams dict whenever the
  import failed, which silently produced plots that didn't match the rest
  of the paper (including population_analysis.ipynb) with no error or
  warning. If `apj_style` isn't importable, this module should fail loudly
  at import time rather than degrade quietly.
"""

from data_loader import BEST_PSRS, SCENARIOS, load_pulsars
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import json
import re
from collections import defaultdict
import hasasia.sensitivity as hsen
from enterprise.pulsar import Pulsar as ePulsar

from apj_style import apply_apj_style, APJ_FIGSIZE

apply_apj_style()


# ---------------------------------------------------------------------------
# White-noise covariance
# ---------------------------------------------------------------------------

def make_corr(psr, raw_noise_params, thin=1):
    """
    Build the white-noise (+ecorr) covariance matrix for a pulsar.

    efac is the only required white-noise term per backend; log10_t2equad
    and log10_ecorr are optional (many MPTA/MeerKAT backends only fit a
    subset), contributing zero if absent rather than raising.
    """
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

        if key_ef not in raw_noise_params:
            raise KeyError(f"Missing required efac for backend '{be}': {key_ef}")

        efac = raw_noise_params[key_ef]
        equad_sqr = (10**raw_noise_params[key_eq])**2 if key_eq in raw_noise_params else 0.0

        sigma_sqr[mask] = efac**2 * toaerrs[mask]**2 + equad_sqr

        if key_ec in raw_noise_params:
            mask_ec = np.where(fl == be)
            ecorrs[mask_ec] = 10**raw_noise_params[key_ec]

    j = [ecorrs[ii]**2 * np.ones((len(bucket), len(bucket)))
         for ii, bucket in enumerate(bi)]
    J = sl.block_diag(*j)
    return np.diag(sigma_sqr) + J


# ---------------------------------------------------------------------------
# Noise-parameter parsing (red / DM / chromatic), straight from raw file
# ---------------------------------------------------------------------------

_NOISE_SUFFIX_PATTERNS = [
    (re.compile(r'^(?P<name>.+)_red_log10_A$'),   'red_noise',   'log10_A'),
    (re.compile(r'^(?P<name>.+)_red_gamma$'),     'red_noise',   'gamma'),
    (re.compile(r'^(?P<name>.+)_dm_log10_A$'),    'dm_noise',    'log10_A'),
    (re.compile(r'^(?P<name>.+)_dm_gamma$'),      'dm_noise',    'gamma'),
    (re.compile(r'^(?P<name>.+)_chrom_log10_A$'), 'chrom_noise', 'log10_A'),
    (re.compile(r'^(?P<name>.+)_chrom_gamma$'),   'chrom_noise', 'gamma'),
    (re.compile(r'^(?P<name>.+)_chrom_beta$'),    'chrom_noise', 'beta'),
]


def parse_pulsar_parameters(json_file_path):
    """Parse red/DM/chromatic noise parameters from a raw noise JSON file
    (path version -- see parse_pulsar_parameters_from_dict for details)."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return parse_pulsar_parameters_from_dict(data)


def parse_pulsar_parameters_from_dict(raw_noise_params):
    """
    Extract red_noise/dm_noise/chrom_noise sub-dicts directly from
    raw_noise_params by matching known key suffixes (anchored, so pulsar
    names containing '+'/'-' don't break the match). White-noise
    (efac/ecorr/t2equad) is intentionally NOT parsed here -- make_corr()
    already reads those directly from raw_noise_params by name.
    """
    out = defaultdict(lambda: {'red_noise': {}, 'dm_noise': {}, 'chrom_noise': {}})
    for key, value in raw_noise_params.items():
        for pattern, group, field in _NOISE_SUFFIX_PATTERNS:
            m = pattern.match(key)
            if m:
                out[m.group('name')][group][field] = value
                break
    return {k: dict(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Optional: DM / chromatic red-noise contributions
# ---------------------------------------------------------------------------
#
# hasasia's `corr_from_psd` builds an ACHROMATIC correlation matrix. DM and
# generic chromatic noise are NOT achromatic -- their effect on a TOA scales
# with that TOA's radio frequency, as (ref_freq/freq)**2 for DM and
# (ref_freq/freq)**chrom_beta for generic chromatic noise. This is an
# approximate treatment (see caveats): normalization convention vs.
# enterprise's actual basis construction is unvalidated, ref_freq_mhz=1400
# is a placeholder pending confirmation, and thin>1 subsamples uniformly in
# time (not frequency), which can bias chromatic scaling for pulsars with
# uneven multi-band coverage.

def _powerlaw_psd(log10_A, gamma, freqs):
    return hsen.red_noise_powerlaw(A=10**log10_A, gamma=gamma, freqs=freqs)


def add_chromatic_corr(corr, freqs, toas, radio_freqs_mhz, log10_A, gamma,
                        chrom_idx=2.0, ref_freq_mhz=1400.0):
    """chrom_idx=2.0 -> standard DM noise; chrom_idx=chrom_beta -> generic
    chromatic noise. Treat as a first-pass approximation (see caveats)."""
    psd = _powerlaw_psd(log10_A, gamma, freqs)
    base_corr = hsen.corr_from_psd(freqs=freqs, psd=psd, toas=toas)
    scale = (ref_freq_mhz / radio_freqs_mhz) ** chrom_idx
    return corr + base_corr * np.outer(scale, scale)


# ---------------------------------------------------------------------------
# Real curves -- used for EVERY scenario (baseline and forecast alike),
# since scenario extension now lives entirely in the .tim files themselves
# ---------------------------------------------------------------------------

def build_real_curves(ePsrs, parsed_noise_params, raw_noise_params, freqs,
                       thin=10, include_dm=False, include_chrom=False,
                       ref_freq_mhz=1400.0):
    """
    Build sensitivity curves from ePulsar objects, whatever TOAs they
    contain (real 4.5yr baseline OR a scenario's forecast-extended .tim --
    this function doesn't care which, it just fits the noise model to
    whatever epochs are present).
    """
    print('--- Building sensitivity curves ---')
    real_psrs = []
    real_specs_per_psr = {}

    for ePsr in ePsrs:
        corr = make_corr(ePsr, raw_noise_params, thin=thin)
        plaw = hsen.red_noise_powerlaw(A=9e-16, gamma=13/3., freqs=freqs)

        psr_params = parsed_noise_params.get(ePsr.name, {})

        if 'red_noise' in psr_params:
            rn = psr_params['red_noise']
            if 'log10_A' in rn and 'gamma' in rn:
                plaw += hsen.red_noise_powerlaw(
                    A=10**rn['log10_A'], gamma=rn['gamma'], freqs=freqs)
            elif rn:
                print(f"  WARNING: incomplete red_noise entry for {ePsr.name}, "
                      f"skipping (found keys: {list(rn.keys())})")

        corr += hsen.corr_from_psd(freqs=freqs, psd=plaw, toas=ePsr.toas[::thin])

        radio_freqs = None
        if include_dm or include_chrom:
            radio_freqs = ePsr.freqs[::thin]

        if include_dm and 'dm_noise' in psr_params and psr_params['dm_noise']:
            dm = psr_params['dm_noise']
            corr = add_chromatic_corr(
                corr, freqs, ePsr.toas[::thin], radio_freqs,
                log10_A=dm['log10_A'], gamma=dm['gamma'],
                chrom_idx=2.0, ref_freq_mhz=ref_freq_mhz)

        if include_chrom and 'chrom_noise' in psr_params and psr_params['chrom_noise']:
            ch = psr_params['chrom_noise']
            corr = add_chromatic_corr(
                corr, freqs, ePsr.toas[::thin], radio_freqs,
                log10_A=ch['log10_A'], gamma=ch['gamma'],
                chrom_idx=ch['beta'], ref_freq_mhz=ref_freq_mhz)

        psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                          toaerrs=ePsr.toaerrs[::thin],
                          phi=ePsr.phi, theta=ePsr.theta,
                          N=corr, designmatrix=ePsr.Mmat[::thin, :])
        psr.name = ePsr.name
        real_psrs.append(psr)
        print(f'\r  PSR {ePsr.name} complete', end='', flush=True)
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
# LEGACY in-memory synthetic path -- cadence/precision changes ONLY, no
# time-baseline extension. Kept for quick "what if we just observed the
# EXISTING 4.5yr span more densely/precisely" checks that don't need a real
# forecast .tim. Do NOT use this for genuine forecasting; use
# make_pta_sensitivity (real .tim files) for that.
# ---------------------------------------------------------------------------

def _interleave_toas(toas, toaerrs, Mmat, flags, radio_freqs=None, factor=2):
    """Increase cadence by inserting `factor-1` new TOAs between every
    consecutive real pair, inheriting nearest-neighbour noise properties
    and linearly interpolated design-matrix rows. Does not extend Tspan."""
    n = len(toas)
    extra_toas, extra_errs, extra_Mmat, extra_flags = [], [], [], []
    extra_freqs = [] if radio_freqs is not None else None

    for i in range(n - 1):
        dt = toas[i+1] - toas[i]
        for k in range(1, factor):
            frac  = k / factor
            t_new = toas[i] + frac * dt
            near_i = i if frac < 0.5 else i + 1
            extra_toas.append(t_new)
            extra_errs.append(toaerrs[near_i])
            extra_flags.append(flags[near_i])
            extra_Mmat.append((1 - frac) * Mmat[i] + frac * Mmat[i+1])
            if radio_freqs is not None:
                extra_freqs.append(radio_freqs[near_i])

    if not extra_toas:
        return toas, toaerrs, Mmat, flags, radio_freqs

    all_toas  = np.concatenate([toas, extra_toas])
    all_errs  = np.concatenate([toaerrs, extra_errs])
    all_flags = np.concatenate([flags, extra_flags])
    all_Mmat  = np.vstack([Mmat, extra_Mmat])
    idx = np.argsort(all_toas)

    if radio_freqs is not None:
        all_freqs = np.concatenate([radio_freqs, extra_freqs])
        return all_toas[idx], all_errs[idx], all_Mmat[idx], all_flags[idx], all_freqs[idx]
    return all_toas[idx], all_errs[idx], all_Mmat[idx], all_flags[idx], None


def _build_one_psr_realistic(
    ePsr, parsed_noise_params, raw_noise_params, freqs,
    thin=10, toaerr_factor=1.0, cadence_factor=1,
    include_dm=False, include_chrom=False, ref_freq_mhz=1400.0,
):
    """Legacy: build a modified hsen.Pulsar from the EXISTING TOA span only
    (no Tspan extension -- use real .tim files for that)."""
    name    = ePsr.name
    toas    = ePsr.toas[::thin].copy()
    toaerrs = ePsr.toaerrs[::thin].copy()
    Mmat    = ePsr.Mmat[::thin, :].copy()
    flags   = ePsr.flags['f'][::thin].copy()
    radio_freqs = ePsr.freqs[::thin].copy() if (include_dm or include_chrom) else None

    if cadence_factor > 1:
        toas, toaerrs, Mmat, flags, radio_freqs = _interleave_toas(
            toas, toaerrs, Mmat, flags, radio_freqs=radio_freqs, factor=cadence_factor)

    toaerrs = toaerrs * toaerr_factor

    N_obs = len(toas)
    _, _, fl, _, bi = hsen.quantize_fast(toas, toaerrs, flags=flags, dt=1)
    backends  = np.unique(flags)
    sigma_sqr = np.zeros(N_obs)
    ecorrs    = np.zeros_like(fl, dtype=float)

    for be in backends:
        mask   = np.where(flags == be)
        key_ef = f'{name}_{be}_efac'
        key_eq = f'{name}_{be}_log10_t2equad'
        key_ec = f'{name}_{be}_log10_ecorr'

        if key_ef not in raw_noise_params:
            raise KeyError(f"[{name}] Missing required efac for backend '{be}': {key_ef}")

        efac = raw_noise_params[key_ef]
        equad_sqr = (10**raw_noise_params[key_eq])**2 if key_eq in raw_noise_params else 0.0
        sigma_sqr[mask] = efac**2 * toaerrs[mask]**2 + equad_sqr

        if key_ec in raw_noise_params:
            mask_ec = np.where(fl == be)
            ecorrs[mask_ec] = 10**raw_noise_params[key_ec]

    j    = [ecorrs[ii]**2 * np.ones((len(bucket), len(bucket)))
            for ii, bucket in enumerate(bi)]
    J    = sl.block_diag(*j)
    corr = np.diag(sigma_sqr) + J

    plaw = hsen.red_noise_powerlaw(A=9e-16, gamma=13/3., freqs=freqs)
    psr_params = parsed_noise_params.get(name, {})
    if 'red_noise' in psr_params:
        rn = psr_params['red_noise']
        if 'log10_A' in rn and 'gamma' in rn:
            plaw += hsen.red_noise_powerlaw(A=10**rn['log10_A'], gamma=rn['gamma'], freqs=freqs)
        elif rn:
            print(f"  WARNING: incomplete red_noise entry for {name}, skipping")

    corr += hsen.corr_from_psd(freqs=freqs, psd=plaw, toas=toas)

    if include_dm and 'dm_noise' in psr_params and psr_params['dm_noise']:
        dm = psr_params['dm_noise']
        corr = add_chromatic_corr(corr, freqs, toas, radio_freqs,
                                   log10_A=dm['log10_A'], gamma=dm['gamma'],
                                   chrom_idx=2.0, ref_freq_mhz=ref_freq_mhz)
    if include_chrom and 'chrom_noise' in psr_params and psr_params['chrom_noise']:
        ch = psr_params['chrom_noise']
        corr = add_chromatic_corr(corr, freqs, toas, radio_freqs,
                                   log10_A=ch['log10_A'], gamma=ch['gamma'],
                                   chrom_idx=ch['beta'], ref_freq_mhz=ref_freq_mhz)

    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        raise ValueError(f'[{name}] Noise matrix not positive definite after modification')

    psr = hsen.Pulsar(toas=toas, toaerrs=toaerrs, phi=ePsr.phi, theta=ePsr.theta,
                       N=corr, designmatrix=Mmat)
    psr.name = name
    return psr


def build_realistic_synthetic_curves(
    ePsrs, parsed_noise_params, raw_noise_params, freqs,
    thin=10, label='Synthetic PTA', best_psrs=BEST_PSRS, mode='best_cadence',
    toaerr_factor=1.0, cadence_factor=1, per_pulsar=None,
    include_dm=False, include_chrom=False, ref_freq_mhz=1400.0,
):
    """Legacy in-memory synthetic-curve builder (no Tspan extension). See
    module docstring for why real .tim files are preferred for forecasting."""
    print(f'--- [LEGACY in-memory] Building synthetic curves: [{label}] mode={mode} ---')
    psrs = []
    for ePsr in ePsrs:
        name, is_best = ePsr.name, ePsr.name in best_psrs

        if mode == 'per_pulsar':
            if per_pulsar and name in per_pulsar:
                ov = per_pulsar[name]
                cf, ef = ov.get('cadence_factor', cadence_factor), ov.get('toaerr_factor', toaerr_factor)
            elif is_best:
                cf, ef = cadence_factor, toaerr_factor
            else:
                cf, ef = 1, 1.0
        elif mode == 'best_cadence':
            cf, ef = (cadence_factor if is_best else 1), 1.0
        elif mode == 'best_precision':
            cf, ef = 1, (toaerr_factor if is_best else 1.0)
        elif mode == 'best_both':
            cf = cadence_factor if is_best else 1
            ef = toaerr_factor if is_best else 1.0
        elif mode == 'all_cadence':
            cf, ef = cadence_factor, 1.0
        elif mode == 'all_precision':
            cf, ef = 1, toaerr_factor
        elif mode == 'all_both':
            cf, ef = cadence_factor, toaerr_factor
        else:
            raise ValueError(f'Unknown mode: {mode!r}')

        try:
            psr = _build_one_psr_realistic(
                ePsr, parsed_noise_params, raw_noise_params, freqs,
                thin=thin, toaerr_factor=ef, cadence_factor=cf,
                include_dm=include_dm, include_chrom=include_chrom, ref_freq_mhz=ref_freq_mhz)
            tag = (f' cadence×{cf}' if cf > 1 else '') + (f' precision×{1/ef:.1f}' if ef < 1 else '')
            print(f'  [{label}] {name:20s}  n_obs={len(psr.toas):4d}{tag}')
            psrs.append(psr)
        except Exception as e:
            print(f'  WARNING: {name} skipped — {e}')

    if not psrs:
        raise RuntimeError(f'No pulsars built for "{label}"')

    specs = []
    for p in psrs:
        try:
            sp = hsen.Spectrum(p, freqs=freqs)
            NcalInv = sp.NcalInv
            n_bad = np.sum(~np.isfinite(NcalInv)) + np.sum(NcalInv <= 0)
            if n_bad > 0:
                print(f'  WARNING: {p.name} has {n_bad} bad NcalInv values, skipping')
                continue
            specs.append(sp)
        except Exception as e:
            print(f'  WARNING: Spectrum failed for {p.name}: {e}')

    if not specs:
        raise RuntimeError(f'No valid spectra for "{label}"')

    sc, dsc = hsen.GWBSensitivityCurve(specs), hsen.DeterSensitivityCurve(specs)
    print(f'  h_c (GWB)   [{np.nanmin(sc.h_c):.2e}, {np.nanmax(sc.h_c):.2e}]')
    print(f'  h_c (Deter) [{np.nanmin(dsc.h_c):.2e}, {np.nanmax(dsc.h_c):.2e}]')
    return sc, dsc


# ---------------------------------------------------------------------------
# Scenario -> color / label helpers (structural, not name-lookup, so they
# work regardless of which scenario dict/labels get passed in)
# ---------------------------------------------------------------------------

def _scenario_color(cadence_factor, toaerr_factor, is_forecast_only=False):
    """Assign color by what the scenario actually modifies."""
    changes_cadence   = cadence_factor != 1
    changes_precision = toaerr_factor  != 1.0
    if changes_cadence and changes_precision:
        return 'navy'
    if changes_cadence:
        return 'lime'
    if changes_precision:
        return 'magenta'
    return '0.5' if is_forecast_only else 'black'


# Default display labels for data_loader's internal scenario keys. Line
# breaks are placed EXACTLY where wanted (not computed by a width-based
# wrapper) since there are only a handful of known scenarios and manual
# breaks are more reliable than a generic wrap algorithm guessing at them.
DEFAULT_SCENARIO_LABELS = {
    'baseline':          'Fiducial\n(4.5 yr)',
    'baseline_forecast': 'Fiducial\n(9.0 yr)',
    '4x_cadence':        r'$4 \times$ Cadence',
    '2x_precision':      r'$2 \times$ Precision',
    '4x_cad_2x_prec':    r'$4 \times$ Cadence,' + '\n' + r'$2 \times$ Precision',
}


# ---------------------------------------------------------------------------
# CANONICAL orchestration: real .tim-based forecasting
# ---------------------------------------------------------------------------

def make_pta_sensitivity(
    scenario_names,
    raw_noise_params,
    thin              = 10,
    scenarios         = None,
    outdir            = 'figures',
    load_kwargs       = None,
    labels            = None,
    colors            = None,
    include_dm        = False,
    include_chrom     = False,
    ref_freq_mhz      = 1400.0,
    figsize           = None,
    baseline_name     = None,
    legend_ncol       = 3,
    save_data_path    = None,
    make_plot         = True,
):
    """
    Build sensitivity curves directly from real/forecast TOAs already
    written to disk via load_pulsars' scenario-tim mechanism. This is the
    canonical path -- no in-memory Tspan extension happens here; each
    scenario's cadence/precision/baseline-length is exactly what's in its
    .tim files on disk.

    Parameters
    ----------
    scenario_names : list of scenario keys to build & compare (must match
        keys in `scenarios`, e.g. SCENARIOS from data_loader)
    raw_noise_params : dict loaded from your noise JSON
    baseline_name : which scenario_names entry to plot as the solid
        reference curve; defaults to the first name containing "baseline"
        or "unchanged" (case-insensitive), else scenario_names[0]
    save_data_path : if given, save the computed curves (freqs/h_c per
        scenario) to this path via save_curves() -- e.g.
        'data/sensitivity_curves/run01' -- so plotting/iteration can
        happen later from disk (population_analysis.ipynb) without
        re-running this (slow) function.
    make_plot : set False to skip plot_sensitivity_comparison entirely --
        useful together with save_data_path when you only want to compute
        and save, not render a figure here.

    Returns
    -------
    dict: {scenario_name: (sc, dsc)}
    """
    parsed_noise_params = parse_pulsar_parameters_from_dict(raw_noise_params)

    _scenarios  = scenarios if scenarios is not None else SCENARIOS
    load_kwargs = load_kwargs or {}
    labels      = labels or {name: DEFAULT_SCENARIO_LABELS.get(name, name) for name in scenario_names}

    loaded, max_tspan = {}, 0.0
    for name in scenario_names:
        print(f'--- Loading pulsars for scenario "{name}" ---')
        lPsrs = load_pulsars(scenario=name, scenarios=_scenarios, **load_kwargs)
        if not lPsrs:
            print(f'  WARNING: no pulsars loaded for "{name}", skipping')
            continue
        tspan = max(p.toas().max() - p.toas().min() for p in lPsrs) * 86400.0
        max_tspan = max(max_tspan, tspan)
        loaded[name] = lPsrs
        print(f'  "{name}": {len(lPsrs)} pulsars, Tspan={tspan / (365.25*86400):.2f} yr')

    if not loaded:
        raise RuntimeError('No scenarios produced any pulsars')

    freqs = np.logspace(np.log10(1 / (5 * max_tspan)), np.log10(3e-7), 900)

    curves = {}
    for name, lPsrs in loaded.items():
        ePsrs = [ePulsar(psr, ephem='DE440', backend='tempo2') for psr in lPsrs]
        sc, dsc, _ = build_real_curves(
            ePsrs, parsed_noise_params, raw_noise_params, freqs, thin=thin,
            include_dm=include_dm, include_chrom=include_chrom, ref_freq_mhz=ref_freq_mhz)
        curves[name] = (sc, dsc)
        print(f'  "{name}" h_c(Deter) range [{np.nanmin(dsc.h_c):.2e}, {np.nanmax(dsc.h_c):.2e}]')

    if baseline_name is None:
        baseline_name = next(
            (n for n in loaded if 'baseline' in n.lower() or 'fiducial' in n.lower()),
            scenario_names[0])
    if baseline_name not in curves:
        baseline_name = list(curves.keys())[0]

    base_sc, base_dsc = curves[baseline_name]

    default_colors = ['magenta', 'lime', 'navy', 'cyan', 'crimson', 'teal', 'purple']
    color_iter = iter(default_colors)
    overlay = []
    resolved_colors = {}
    for name, (sc, dsc) in curves.items():
        if name == baseline_name:
            continue
        color = (colors or {}).get(name, next(color_iter, 'grey'))
        resolved_colors[name] = color
        overlay.append((labels.get(name, name), color, sc, dsc))

    if save_data_path is not None:
        save_curves(curves, save_data_path, labels=labels,
                    baseline_name=baseline_name, colors=resolved_colors)

    if make_plot:
        plot_sensitivity_comparison(
            base_sc, base_dsc, overlay, outdir=outdir, figsize=figsize or APJ_FIGSIZE,
            baseline_label=labels.get(baseline_name, DEFAULT_SCENARIO_LABELS.get(baseline_name, baseline_name)),
            legend_ncol=legend_ncol)
    return curves


# ---------------------------------------------------------------------------
# Save / load computed curves -- lets the (slow) sensitivity computation
# run once here, with plotting/iteration happening later from disk (e.g.
# in population_analysis.ipynb) without re-running make_pta_sensitivity.
#
# The actual save/load logic lives in curve_io.py (numpy/json only, no
# hasasia/enterprise/data_loader dependency) so population_analysis.ipynb
# can `from curve_io import load_curves` without dragging in everything
# this module needs to *compute* the curves in the first place.
# ---------------------------------------------------------------------------

from curve_io import save_curves as _save_curves


def save_curves(curves, path, labels=None, baseline_name=None, colors=None):
    return _save_curves(curves, path, labels=labels, baseline_name=baseline_name,
                         colors=colors, default_scenario_labels=DEFAULT_SCENARIO_LABELS)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sensitivity_comparison(
    real_sc, real_dsc, synthetic_curves,
    outdir          = 'figures',
    figsize         = None,
    legend_ncol     = 3,
    filename        = 'sensitivity_curves_comparison.pdf',
    baseline_label  = 'Unchanged\n(4.5 yr)',
    legend_fontsize = None,
    ylabel          = 'Characteristic\nstrain $h_c$',
):
    """
    ApJ single-column figure, held EXACTLY at APJ_FIGSIZE.

    IMPORTANT: apj_style.py sets rcParams['savefig.bbox'] = 'tight'
    globally. Passing bbox_inches=None to savefig() does NOT disable that
    -- matplotlib's savefig() treats None as "use rcParams['savefig.bbox']",
    so None was silently equivalent to 'tight' all along, which is why the
    canvas kept growing to fit the legend regardless of font size. Fixed
    below by passing bbox_inches=fig.bbox_inches (a concrete Bbox equal to
    `figsize`), which bypasses the rcParam fallback entirely and holds the
    saved file at exactly `figsize`, no exceptions.

    Legend/ylabel line breaks are NOT computed by a width-based wrapper --
    they're literal '\\n' characters in the label text (see
    DEFAULT_SCENARIO_LABELS and the ylabel default), so breaks land
    exactly where wanted rather than wherever a character-count heuristic
    guesses. legend_fontsize defaults to None, which means "use
    apj_style's rcParams legend.fontsize (10pt) as-is" -- only override it
    if the legend still doesn't fit at that size.

    apply_apj_style() is re-applied here (in addition to the module-level
    call on import) as cheap insurance: it's idempotent, and it means this
    function doesn't silently inherit rcParams drift from whatever else
    ran between import time and this call.
    """
    import os; os.makedirs(outdir, exist_ok=True)
    apply_apj_style()

    figsize = figsize or APJ_FIGSIZE
    fig, ax = plt.subplots(figsize=figsize)

    ax.loglog(real_dsc.freqs, real_dsc.h_c, color='black', lw=1.3, label=baseline_label)
    for label, color, syn_sc, syn_dsc in synthetic_curves:
        print(f'  plotting {label}: h_c range [{syn_dsc.h_c.min():.2e}, {syn_dsc.h_c.max():.2e}]')
        ax.loglog(syn_dsc.freqs, syn_dsc.h_c, color=color, lw=1.0, label=label)

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(ylabel, labelpad=6 if '\n' in ylabel else 3)
    ax.set_xlim(1e-9, 3e-7)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.grid(which='both', ls='--', lw=0.3, alpha=0.3)

    n_entries = len(synthetic_curves) + 1
    ncol = legend_ncol
    n_rows = -(-n_entries // ncol)  # ceil division

    # Assume up to 2 lines per entry when reserving vertical space, since
    # a couple of these labels wrap onto 2 lines while others stay on 1.
    row_height = 0.13
    # Extra fixed margin here accounts for tick labels + the x-axis label
    # itself, since the legend is now anchored below the x-label (not
    # just below the panel) -- without this, the top legend row can
    # crowd into the x-label at larger row counts.
    bottom_margin = 0.28 + row_height * n_rows
    fig.subplots_adjust(left=0.24, right=0.96, top=0.95,
                         bottom=min(bottom_margin, 0.62))

    # Center the legend on the FULL FIGURE, not the panel. The y-axis
    # label sits to the left of the panel, shifting the panel itself
    # right of figure-center -- if the legend is centered on the panel
    # instead, a wide legend (e.g. 3 columns) gets pushed past the
    # figure's actual right edge and clips. Centering on the full canvas
    # keeps the whole legend block inside the page regardless of how far
    # right the panel sits.
    panel_pos = ax.get_position()
    figure_center_x = 0.5

    # Anchor the legend's TOP edge just below the x-AXIS LABEL's actual
    # rendered bottom edge -- not a fixed guessed margin below the panel.
    # panel_pos.y0 (the axes bounding box bottom) is well ABOVE the tick
    # labels and the x-axis label text, both of which live further down;
    # a small fixed gap below panel_pos.y0 would overlap them. Instead,
    # force a draw so the renderer has real text metrics, then read the
    # x-axis label's own bounding box (in display/pixel space) and
    # convert its bottom edge to figure-fraction coordinates. This stays
    # correct regardless of font size, label wrapping, or figure size --
    # no hand-tuned margin number to re-guess every time something changes.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
    xlabel_bottom_fig_frac = fig.transFigure.inverted().transform(
        (0, xlabel_bbox.y0))[1]

    legend_gap = 0.03
    legend_top_y = xlabel_bottom_fig_frac - legend_gap

    legend_kwargs = dict(
        loc='upper center', bbox_to_anchor=(figure_center_x, legend_top_y),
        bbox_transform=fig.transFigure,
        ncol=ncol, frameon=False, handlelength=1.2,
        columnspacing=0.7, handletextpad=0.35, borderaxespad=0.2,
        labelspacing=0.6,
    )
    if legend_fontsize is not None:
        legend_kwargs['fontsize'] = legend_fontsize

    leg = ax.legend(**legend_kwargs)

    # CRITICAL: passing bbox_inches=None here does NOT disable tight
    # cropping -- matplotlib's savefig() falls back to
    # rcParams['savefig.bbox'] whenever bbox_inches is None, and
    # apj_style.py sets that rcParam to 'tight' globally. So None was
    # silently equivalent to 'tight' the whole time, which is why the
    # canvas kept growing regardless. The only way to actually bypass the
    # rcParam is to pass a concrete Bbox instead of None/'tight' --
    # fig.bbox_inches IS exactly that: a Bbox equal to `figsize` in
    # inches. Passing it explicitly skips both the None-fallback and the
    # tight-bbox-computation code paths, so the saved file is guaranteed
    # to be exactly `figsize`, full stop.
    fig.savefig(f'{outdir}/{filename}', dpi=300, bbox_inches=fig.bbox_inches)
    plt.show()