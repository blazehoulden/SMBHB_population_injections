import os
import pickle
import json
import numpy as np
import gc
from copy import deepcopy
from enterprise.pulsar import Pulsar
import sys
from collections import defaultdict
from config import NANOGRAV_PULSARS, PAR_DIR, TIM_DIR, USE_PULSAR_CACHE, PULSAR_CACHE, NOISEFILE
import libstempo as T
from signal_injection import get_base_name

def tim_has_toas(tim_path):
    """Quick check if .tim file contains valid TOA data."""
    try:
        with open(tim_path, 'r') as fh:
            lines = [ln.strip() for ln in fh 
                    if ln.strip() and not ln.strip().startswith(('*', 'FORMAT', 'C'))]
        for ln in lines:
            parts = ln.split()
            for p in parts:
                try:
                    float(p)
                    return True
                except Exception:
                    continue
        return False
    except Exception:
        return False
    
SKIP_PULSARS = {}
MAX_SYNTH_TOAS = None

def filter_pulsars_15yr(psrs, min_baseline_years=0.0, verbose=True):
    """Filter to 15yr pulsars with sufficient baseline."""
    with open(NOISEFILE, 'r') as f:
        params = json.load(f)
    
    pulsars_in_15yr = list(set([k.split('_')[0] for k in params.keys() if '_' in k]))
    
    psrs_after_15yr = [psr for psr in psrs if psr.name in pulsars_in_15yr]
    
    psrs_filtered = []
    total_tmin = None
    total_tmax = None
    for psr in psrs_after_15yr:
        # fix for libstempo pulsars
        tmin = min(psr.toas())
        tmax = max(psr.toas())
        # baseline_years = (psr.toas.max() - psr.toas.min()) / (365.25 * 86400)
        baseline_years = (tmax - tmin) / (365.25)

        if baseline_years >= min_baseline_years:
            psrs_filtered.append(psr)
        if total_tmin is None or tmin < total_tmin:
            total_tmin = tmin
        if total_tmax is None or tmax > total_tmax:
            total_tmax = tmax
        
    if total_tmin is None or total_tmax is None:
        if verbose:
            print(f"\nFiltered: {len(psrs)} → 0 pulsars")
        return [], params, 0.0
    
    Tspan = float(total_tmax - total_tmin) # days
    Tspan_seconds = Tspan * 86400 # seconds
    if verbose:
        print(f"\nFiltered: {len(psrs)} → {len(psrs_filtered)} pulsars")
    
    return psrs_filtered, params, Tspan_seconds



def get_clean_pulsars_and_tspan(psrs_filtered):
    """
    Get pulsars and calculate Tspan.
    
    Note: Returns original pulsars (not copies) to save memory.
    Original residuals are saved for restoration between injections.
    """
    # Calculate Tspan
    tmin = min(min(p.toas()) for p in psrs_filtered)
    tmax = max(max(p.toas()) for p in psrs_filtered)
    Tspan = tmax - tmin
    
    # Save original residuals ONCE for each pulsar
    for psr in psrs_filtered:
        if not hasattr(psr, '_original_residuals'):
            psr._original_residuals = np.copy(psr.residuals)
    
    return psrs_filtered, Tspan


def restore_original_residuals(psrs):
    """Restore pulsars to original state before next injection."""
    for psr in psrs:
        if hasattr(psr, '_original_residuals'):
            psr._residuals = np.copy(psr._original_residuals)
    gc.collect()


def parse_pulsar_parameters(json_file_path):
    """
    Parse pulsar parameters from JSON file.
    
    Parameters
    ----------
    json_file_path : str
        Path to the JSON file containing pulsar parameters
        
    Returns
    -------
    dict
        Organized pulsar parameters with structure:
        {
            'pulsar_name': {
                'red_noise': {'gamma': float, 'log10_A': float},
                'white_noise': {
                    'backend_name': {
                        'efac': float, 
                        'log10_ecorr': float, 
                        'log10_t2equad': float
                    }
                }
            }
        }
    
    Examples
    --------
    >>> noise_params = parse_pulsar_parameters('noise_params.json')
    >>> # Access red noise for a specific pulsar
    >>> log10_A = noise_params['B1855+09']['red_noise']['log10_A']
    >>> gamma = noise_params['B1855+09']['red_noise']['gamma']
    >>> # Access white noise for a specific backend
    >>> efac = noise_params['B1855+09']['white_noise']['430_ASP']['efac']
    """
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store organized parameters
    pulsar_params = defaultdict(lambda: {'red_noise': {}, 'white_noise': {}})
    
    # Process each parameter
    for key, value in data.items():
        # Check if this is a red noise parameter
        if 'red_noise' in key:
            # Extract pulsar name (everything before '_red_noise')
            pulsar_name = key.split('_red_noise')[0]
            
            # Extract parameter type (gamma or log10_A)
            if 'gamma' in key:
                pulsar_params[pulsar_name]['red_noise']['gamma'] = value
            elif 'log10_A' in key:
                pulsar_params[pulsar_name]['red_noise']['log10_A'] = value
        
        # Otherwise, it's a white noise parameter
        elif 'efac' in key or 'ecorr' in key or 't2equad' in key:
            # Identify the parameter type and split point
            if '_efac' in key:
                split_idx = key.rfind('_efac')
                param_type = 'efac'
            elif '_log10_ecorr' in key:
                split_idx = key.rfind('_log10_ecorr')
                param_type = 'log10_ecorr'
            elif '_log10_t2equad' in key:
                split_idx = key.rfind('_log10_t2equad')
                param_type = 'log10_t2equad'
            else:
                continue
            
            # Everything before the parameter type is pulsar_backend
            base_key = key[:split_idx]
            
            # Now find where pulsar name ends
            # Pulsar names contain + or - (but not L-wide)
            parts = base_key.split('_')
            
            pulsar_name = None
            for i, part in enumerate(parts):
                if '+' in part:
                    # Found pulsar name ending with +
                    pulsar_name = '_'.join(parts[:i+1])
                    backend_name = '_'.join(parts[i+1:])
                    break
                elif '-' in part:
                    # Check if this is L-wide or part of pulsar name
                    if i > 0 and parts[i-1] == 'L' and part == 'wide':
                        # This is L-wide, not pulsar name
                        continue
                    else:
                        # This is part of pulsar name (e.g., J0437-4715)
                        pulsar_name = '_'.join(parts[:i+1])
                        backend_name = '_'.join(parts[i+1:])
                        break
            
            if pulsar_name is None:
                # Fallback: first part is pulsar name
                pulsar_name = parts[0]
                backend_name = '_'.join(parts[1:])
            
            # Initialize backend entry if it doesn't exist
            if backend_name not in pulsar_params[pulsar_name]['white_noise']:
                pulsar_params[pulsar_name]['white_noise'][backend_name] = {}
            
            # Store the parameter value
            pulsar_params[pulsar_name]['white_noise'][backend_name][param_type] = value
    
    # Convert defaultdict to regular dict for cleaner output
    return {k: dict(v) for k, v in pulsar_params.items()}



#### Adapting to simulate pulsars with different cadences and errors

# BEST_PSRS = (
#     'J0437-4715', 'J1909-3744', 'J1713+0747',
#     'J0030+0451', 'J1744-1134',
# )

BEST_PSRS_NANOGrav = (
    'J1713+0747', 'J1909-3744', 'J2043+1711',
    'J1741+1351', 'J1918-0642'
)
BEST_PSRS_MEERKAT_SENS = (
    'J2241-5236', 'J1744-1134', 'J0437-4715', 
    'J2010-1323', 'J2124-3358', 'J1918-0642',
    'J1732-5049', 'J1603-7202', 'J2129-5721',
    'J0711-6830', 'J2145-0750', 'J1022+1001',
    'J0030+0451', 'J1547-5709', 'J1435-6100',
    'J1446-4701', 'J1455-3330', 'J2150-0326',
    'J2322-2650', 'J1946-5403'
)

# Only want top 10 to conserve telescope time
BEST_PSRS_MEERKAT = (
    'J2241-5236', 'J1909-3744', 'J0711-6830',
    'J1744-1134', 'J1629-6902', 'J2129-5721',
    'J1946-5403', 'J1125-6014', 'J0437-4715',
    'J0125-2327', #'J2010-1323', 'J1446-4701',
    # 'J2039-3616', 'J1216-6410', 'J1545-4550',
    # 'J1732-5049', 'J0613-0200', 'J1918-0642',
    # 'J1811-2405', 'J2124-3358',
)

BEST_PSRS = BEST_PSRS_NANOGrav if NANOGRAV_PULSARS else BEST_PSRS_MEERKAT
top20 = list(BEST_PSRS_MEERKAT_SENS)        # all 20, ranked
top10 = top20[:10]
top5  = top20[:5]

# Each scenario entry:
#   cadence_factor  int    — TOA grid multiplier   (1 = unchanged)
#   toaerr_factor   float  — TOA error multiplier  (1.0 = unchanged)
#   best_only       bool   — apply only to BEST_PSRS if True
#   best_psrs       tuple  — override BEST_PSRS for this scenario (optional)
SCENARIOS = {
    'baseline': dict(
        cadence_factor = 1,
        toaerr_factor  = 1.0,
        best_only      = True,
        # no extension_years -> stays real-only 4.5yr, used for the
        # SGWB consistency check + CW candidate baseline
    ),

    'baseline_forecast': dict(
        cadence_factor  = 1,
        toaerr_factor   = 1.0,
        best_only       = True,
        extension_years = 4.46,
    ),

    '4x_cadence': dict(
        cadence_factor  = 4,
        toaerr_factor   = 1.0,
        best_only       = True,
        extension_years = 4.46,
    ),

    '2x_precision': dict(
        cadence_factor  = 1,
        toaerr_factor   = 0.5,
        best_only       = True,
        extension_years = 4.46,
    ),

    '4x_cad_2x_prec': dict(
        cadence_factor  = 4,
        toaerr_factor   = 0.5,
        best_only       = True,
        extension_years = 4.46,
    ),
    '4x_cadence_conserved': dict(
        cadence_factor          = 4,
        toaerr_factor           = 1.0,
        best_only               = True,
        extension_years         = 4.46,
        conserve_telescope_time = True,  
    ),
    '2x_precision_conserved': dict(
        cadence_factor          = 2,
        toaerr_factor           = 1.0,
        best_only               = True,
        extension_years         = 4.46,
        conserve_telescope_time = True,  
    ),
    '4x_cad_2x_prec_conserved': dict(
        cadence_factor  = 4,
        toaerr_factor   = 0.5,
        best_only       = True,
        extension_years = 4.46,
        conserve_telescope_time = True,  
    ),
}




def _interleave_toas(toas, errs, flags, freqs, factor):
    """Insert (factor-1) linearly-spaced TOAs between every consecutive pair."""
    extra_t, extra_e, extra_fl, extra_fr = [], [], [], []
    for i in range(len(toas) - 1):
        dt = toas[i + 1] - toas[i]
        for k in range(1, factor):
            frac = k / factor
            extra_t.append(toas[i] + frac * dt)
            extra_e.append(errs[i]  if frac < 0.5 else errs[i + 1])
            extra_fl.append(flags[i] if frac < 0.5 else flags[i + 1])
            extra_fr.append(freqs[i] if frac < 0.5 else freqs[i + 1])
 
    if not extra_t:
        return toas, errs, flags, freqs
 
    all_t  = np.concatenate([toas,  extra_t])
    all_e  = np.concatenate([errs,  extra_e])
    all_fl = np.concatenate([flags, extra_fl])
    all_fr = np.concatenate([freqs, extra_fr])
    idx    = np.argsort(all_t)
    return all_t[idx], all_e[idx], all_fl[idx], all_fr[idx]
 
def _find_par_files(par_dir: str, psr_name: str) -> dict:
    """
    Return a dict mapping variant suffix -> par_path for a given base pulsar name.
    e.g. {'': '/path/J1713+0747.par', 'ao': '/path/J1713+0747ao.par', 'gbt': ...}
    """
    _suffixes = ('ao', 'gbt', 'vla', 'fast')
    result = {}
    for fname in sorted(os.listdir(par_dir)):
        if not fname.endswith('.par'):
            continue
        stem = fname.replace('.par', '').split('_')[0]
        base = stem
        variant = ''
        for sfx in _suffixes:
            if base.endswith(sfx):
                base = base[:-len(sfx)]
                variant = sfx
                break
        if base == psr_name:
            result[variant] = os.path.join(par_dir, fname)
    return result

def _find_tim_files(tim_dir: str, psr_name: str) -> list:
    """
    Return all .tim files in tim_dir whose basename starts with psr_name.
    Handles variants like J1713+0747.tim, J1713+0747ao.tim, J1713+0747gbt.tim.
    """
    matches = []
    for fname in os.listdir(tim_dir):
        if not fname.endswith('.tim'):
            continue
        # strip everything after the first underscore or dot to get the stem pulsar name
        stem = fname.replace('.tim', '').split('_')[0]
        if stem == psr_name or get_base_name(stem) == psr_name:
            matches.append(os.path.join(tim_dir, fname))
    return sorted(matches)

def _parse_tim_lines(tim_path):
    """Parse .tim into (header_lines, toa_records)."""
    header_lines, toa_records = [], []

    with open(tim_path) as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            stripped = line.strip()

            if (not stripped or stripped.startswith('FORMAT')
                    or stripped.startswith('C ') or stripped.startswith('C\t')
                    or stripped == 'C'):
                header_lines.append(line)
                continue

            parts = stripped.split()

            if len(parts) < 4:
                header_lines.append(line)
                continue

            try:
                toa = float(parts[2])
                err = float(parts[3])

                toa_decimals = (
                    len(parts[2].split('.')[1])
                    if '.' in parts[2] else 0
                )

                err_decimals = (
                    len(parts[3].split('.')[1])
                    if '.' in parts[3] else 0
                )

            except ValueError:
                header_lines.append(line)
                continue

            toa_records.append({
                'line': line,
                'toa': toa,
                'err': err,
                'toa_col': 2,
                'err_col': 3,
                'toa_decimals': toa_decimals,
                'err_decimals': err_decimals,
            })

    return header_lines, toa_records

def _rebuild_line(record, new_toa=None, new_err=None):
    """Swap only TOA/error columns; everything else verbatim."""

    parts = record['line'].split()

    if new_toa is not None:
        nd = record.get('toa_decimals', 16)
        parts[record['toa_col']] = f'{new_toa:.{nd}f}'

    if new_err is not None:
        nd = record.get('err_decimals', 4)
        parts[record['err_col']] = f'{new_err:.{nd}f}'

    return ' '.join(parts)
 
def _write_scenario_tim(orig_tim_path,
                        psr_name,
                        outpath,
                        cadence_factor=1,
                        toaerr_factor=1.0):
    """
    Write synthetic .tim while preserving every metadata field exactly.
    Only TOA and uncertainty columns are modified.
    """

    header_lines, toa_records = _parse_tim_lines(orig_tim_path)

    if not toa_records:
        raise ValueError(f'No TOA records found in {orig_tim_path}')

    orig_toas = np.array([r['toa'] for r in toa_records])

    n_orig = len(orig_toas)

    entries = [
        (orig_toas[i], i, False)
        for i in range(n_orig)
    ]

    if cadence_factor > 1:

        for i in range(n_orig - 1):

            dt = orig_toas[i + 1] - orig_toas[i]

            for k in range(1, cadence_factor):

                frac = k / cadence_factor

                entries.append(
                    (
                        orig_toas[i] + frac * dt,
                        i if frac < 0.5 else i + 1,
                        True,
                    )
                )

        entries.sort(key=lambda x: x[0])

    if MAX_SYNTH_TOAS is not None and len(entries) > MAX_SYNTH_TOAS:

        print(
            f'  ⚠ [{psr_name}] capping '
            f'{len(entries)} → {MAX_SYNTH_TOAS} TOAs'
        )

        idx = np.round(
            np.linspace(
                0,
                len(entries) - 1,
                MAX_SYNTH_TOAS
            )
        ).astype(int)

        entries = [entries[i] for i in idx]

    with open(outpath, 'w') as fout:

        for line in header_lines:
            fout.write(line + '\n')

        for toa_mjd, rec_idx, is_synth in entries:

            rec = toa_records[rec_idx]

            new_err = rec['err'] * toaerr_factor

            if is_synth:

                fout.write(
                    _rebuild_line(
                        rec,
                        new_toa=toa_mjd,
                        new_err=new_err,
                    ) + '\n'
                )

            elif toaerr_factor != 1.0:

                fout.write(
                    _rebuild_line(
                        rec,
                        new_err=new_err,
                    ) + '\n'
                )

            else:

                fout.write(rec['line'] + '\n')

    # sanity check
    written = _tim_nobs(outpath)

    expected = len(entries)

    if written != expected:
        raise RuntimeError(
            f'{psr_name}: wrote {written} TOAs '
            f'but expected {expected}'
        )

def _load_tim_template(tim_path):
    """
    Parse a real NANOGrav .tim file into:
    - header lines
    - structured TOA rows (kept as token lists)
    """

    header = []
    rows = []

    with open(tim_path, "r") as f:
        for line in f:
            s = line.rstrip("\n")

            if not s or s.startswith(("C", "FORMAT")):
                header.append(s)
                continue

            parts = s.split()
            if len(parts) < 4:
                continue

            rows.append(parts)

    return header, rows

 
def _tim_nobs(tim_path):
    """
    Count real TOA lines in a tempo2 .tim file — delegates to
    _parse_tim_lines so there is exactly one definition of "is this a TOA
    line" used everywhere (previously _tim_nobs had its own, looser rule
    that miscounted non-FORMAT/C/* header lines like "MODE 1" as TOAs).
    """
    _, toa_records = _parse_tim_lines(tim_path)
    return len(toa_records)

import hashlib
import json
import numpy as np
 
 
# ──────────────────────────────────────────────────────────────────────────
# NEW: helper functions — add these anywhere above _write_scenario_tim
# ──────────────────────────────────────────────────────────────────────────
 
def _deterministic_seed(*parts: str) -> int:
    """
    Stable seed derived from string parts. Do NOT use python's hash() here —
    it's randomized per-process for str objects (PYTHONHASHSEED), so cache
    validity would never reproduce the same forecast draw between runs.
    """
    h = hashlib.sha256('::'.join(parts).encode()).hexdigest()
    return int(h[:8], 16)
 
 
def _mean_cadence_days(toas: np.ndarray) -> float:
    """
    Mean spacing between sorted real TOAs, in days. Same-epoch multi-frequency
    TOAs (dt ~ 0) are excluded so they don't drag the mean cadence toward
    zero — we want "how often does a new observing epoch happen", not
    "how often does a new TOA line appear".
    """
    toas_sorted = np.sort(toas)
    dts = np.diff(toas_sorted)
    dts = dts[dts > 1e-6]
    if len(dts) == 0:
        return 30.0  # degenerate fallback, shouldn't happen for real PTA data
    return float(np.mean(dts))
 
 
def _future_toa_times(orig_toas: np.ndarray, cadence_factor: float,
                       extension_days: float) -> np.ndarray:
    """
    MJDs for the forecast segment: starts one future-cadence step past the
    last real TOA, spaced at (mean real cadence / cadence_factor), out to
    last_real_toa + extension_days.
 
    cadence_factor > 1 -> denser future observing than today's average.
    cadence_factor < 1 -> sparser future observing than today's average.
    """
    if cadence_factor <= 0:
        raise ValueError('cadence_factor must be > 0')
    if extension_days <= 0:
        return np.array([])
 
    mean_dt = _mean_cadence_days(orig_toas)
    dt_future = mean_dt / cadence_factor
 
    last_toa = float(np.max(orig_toas))
    n_future = int(np.floor(extension_days / dt_future))
    if n_future <= 0:
        return np.array([])
 
    offsets = np.arange(1, n_future + 1) * dt_future
    return last_toa + offsets
 
 
def _sample_future_errs(orig_errs: np.ndarray, n_future: int,
                         toaerr_factor: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Bootstrap-resample (with replacement) future TOA uncertainties from the
    empirical distribution of this pulsar's real uncertainties, then scale
    by toaerr_factor. Bootstrapping preserves the real spread/skew/receiver
    mix without assuming a parametric error distribution.
    """
    if len(orig_errs) == 0:
        raise ValueError('no real TOA errors available to sample from')
    draws = rng.choice(orig_errs, size=n_future, replace=True)
    return draws * toaerr_factor
_TOBS_CACHE: dict = {}
 
 
def _parse_toa_flags(tim_path):
    """
    Yield (toa_mjd, flags_dict) for every real TOA line in a tempo2 .tim
    file. Standard format: FILE FREQ TOA TOAERR SITE [-flag value]...
    """
    with open(tim_path, 'r', errors='ignore') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(('C', 'FORMAT', '*')):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                toa = float(parts[2])
            except ValueError:
                continue
 
            flags = {}
            rest = parts[5:]
            i = 0
            while i < len(rest) - 1:
                key, val = rest[i], rest[i + 1]
                if key.startswith('-'):
                    flags[key[1:].lower()] = val
                    i += 2
                else:
                    i += 1
            yield toa, flags
 
 
def _pulsar_mean_tobs(psr_name: str, tim_dir: str,
                       flag_key: str = 'tobs',
                       fallback: float = 1.0) -> float:
    """
    Mean real per-TOA integration time (seconds, per the -tobs flag) for
    this pulsar, averaged across ALL its real .tim variants (ao/gbt/vla/
    fast/etc). Falls back to `fallback` (a neutral weight) if no such flag
    is found for this pulsar, so pulsars/datasets without dwell-time info
    degrade to the uniform-weight behavior instead of erroring. Cached —
    this only depends on real data, never on scenario config, so it's
    computed once per pulsar per run regardless of how many scenarios use
    conserve_telescope_time.
    """
    key = (psr_name, tim_dir, flag_key)
    if key in _TOBS_CACHE:
        return _TOBS_CACHE[key]
 
    tim_paths = [t for t in _find_tim_files(tim_dir, psr_name) if tim_has_toas(t)]
    vals = []
    for tp in tim_paths:
        for _, flags in _parse_toa_flags(tp):
            if flag_key in flags:
                try:
                    vals.append(float(flags[flag_key]))
                except ValueError:
                    pass
 
    weight = float(np.mean(vals)) if vals else fallback
    _TOBS_CACHE[key] = weight
    return weight
 
 
_TELESCOPE_BUDGET_CACHE: dict = {}
_TOBS_CACHE: dict = {}
 
 
def _list_base_pulsar_names(par_dir: str) -> list:
    """Unique base pulsar names present in par_dir — mirrors the scan
    load_pulsars() already does to build `base_names`."""
    _suffixes = ('ao', 'gbt', 'vla', 'fast')
    parfiles = sorted([f for f in os.listdir(par_dir) if f.endswith('.par')])
    seen, base_names = set(), []
    for par in parfiles:
        stem = par.replace('.par', '').split('_')[0]
        base = stem
        for sfx in _suffixes:
            if base.endswith(sfx):
                base = base[:-len(sfx)]
                break
        if base not in seen:
            seen.add(base)
            base_names.append(base)
    return base_names
 
 
def _pulsar_time_cost(cadence_factor: float, toaerr_factor: float) -> float:
    """
    Relative telescope-time cost multiplier: cadence_factor scales
    linearly (more epochs = more time); toaerr_factor scales as
    1/toaerr_factor^2 (radiometer equation — halving the error costs 4x
    the per-epoch integration time).
    """
    return cadence_factor * (1.0 / toaerr_factor ** 2)
 
 
def _parse_toa_flags(tim_path):
    """
    Yield (toa_mjd, flags_dict) for every real TOA line in a tempo2 .tim
    file. Standard format: FILE FREQ TOA TOAERR SITE [-flag value]...
    """
    with open(tim_path, 'r', errors='ignore') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(('C', 'FORMAT', '*')):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                toa = float(parts[2])
            except ValueError:
                continue
 
            flags = {}
            rest = parts[5:]
            i = 0
            while i < len(rest) - 1:
                key, val = rest[i], rest[i + 1]
                if key.startswith('-'):
                    flags[key[1:].lower()] = val
                    i += 2
                else:
                    i += 1
            yield toa, flags
 
 
def _pulsar_mean_tobs(psr_name: str, tim_dir: str,
                       flag_key: str = 'tobs',
                       fallback: float = 1.0) -> float:
    """
    Mean real per-TOA integration time (seconds, per the -tobs flag) for
    this pulsar, averaged across all its real .tim variants. Falls back to
    `fallback` if no such flag is found, so pulsars/datasets without
    dwell-time info degrade to a neutral weight instead of erroring.
    Cached — only depends on real data, computed once per pulsar per run.
    """
    key = (psr_name, tim_dir, flag_key)
    if key in _TOBS_CACHE:
        return _TOBS_CACHE[key]
 
    tim_paths = [t for t in _find_tim_files(tim_dir, psr_name) if tim_has_toas(t)]
    vals = []
    for tp in tim_paths:
        for _, flags in _parse_toa_flags(tp):
            if flag_key in flags:
                try:
                    vals.append(float(flags[flag_key]))
                except ValueError:
                    pass
 
    weight = float(np.mean(vals)) if vals else fallback
    _TOBS_CACHE[key] = weight
    return weight
 
 
def _compute_telescope_time_budget(cfg: dict, scenario_label: str,
                                    best_only: bool, best_psrs,
                                    par_dir: str, tim_dir: str,
                                    dwell_flag_key: str = 'tobs') -> dict:
    """
    Compute the uniform forecast cadence_factor for every non-upgraded
    ("non-best") pulsar so total array-wide telescope time — in real
    seconds, weighted by each pulsar's own mean -tobs — matches
    business-as-usual (every pulsar at cadence_factor=1, toaerr_factor=1).
    """
    all_names = _list_base_pulsar_names(par_dir)
    weights = {name: _pulsar_mean_tobs(name, tim_dir, flag_key=dwell_flag_key)
               for name in all_names}
    budget_total = sum(weights.values())
 
    per_pulsar_cfg = cfg.get('per_pulsar', {})
    top_cadence    = cfg.get('cadence_factor', 1)
    top_toaerr     = cfg.get('toaerr_factor', 1.0)
 
    upgraded_names = set(per_pulsar_cfg.keys())
    if best_only:
        upgraded_names |= set(best_psrs)
    upgraded_names &= set(all_names)
 
    cost_upgraded = 0.0
    for name in upgraded_names:
        override = per_pulsar_cfg.get(name)
        if override:
            c = override.get('cadence_factor', top_cadence)
            e = override.get('toaerr_factor',  top_toaerr)
        else:
            c, e = top_cadence, top_toaerr
        cost_upgraded += weights[name] * _pulsar_time_cost(c, e)
 
    remaining_names = set(all_names) - upgraded_names
    sum_w_remaining = sum(weights[n] for n in remaining_names)
    n_upgraded  = len(upgraded_names)
    n_remaining = len(remaining_names)
    leftover    = budget_total - cost_upgraded
 
    MIN_CADENCE_FACTOR = 0.02   # near-zero but never exactly 0 — keeps
                                 # _future_toa_times' cadence_factor > 0
                                 # requirement satisfied instead of raising
 
    if n_remaining == 0 or sum_w_remaining <= 0:
        cadence_nonbest = 1.0
        if leftover < -1e-6:
            print(f'  ⚠ conserve_telescope_time [{scenario_label}]: every '
                  f'pulsar is upgraded (or none left with nonzero weight) — '
                  f'over budget by {-leftover:.0f}s')
    else:
        cadence_nonbest = leftover / sum_w_remaining
        if cadence_nonbest < MIN_CADENCE_FACTOR:
            overshoot_pct = 100.0 * (cost_upgraded - budget_total) / budget_total
            print(f'  🛑 conserve_telescope_time [{scenario_label}]: upgrade '
                  f'cost ({cost_upgraded:.0f}s) exceeds the ENTIRE array '
                  f'budget ({budget_total:.0f}s) by {overshoot_pct:.0f}% even '
                  f'at near-zero cadence on the remaining {n_remaining} '
                  f'pulsars. This scenario config is asking for more '
                  f'telescope time than physically exists — consider '
                  f'reducing cadence_factor/toaerr_factor or the number of '
                  f'upgraded pulsars for this scenario. Clamping to '
                  f'{MIN_CADENCE_FACTOR} rather than 0 so the run doesnt '
                  f'crash, but this result should not be trusted physically.')
            cadence_nonbest = MIN_CADENCE_FACTOR
 
    print(f'  ⏱ conserve_telescope_time [{scenario_label}]: {len(all_names)} '
          f'pulsars, weighted by real mean tobs (sum={budget_total:.0f}s); '
          f'{n_upgraded} upgraded (cost={cost_upgraded:.0f}s), '
          f'leftover={leftover:.0f}s over {n_remaining} remaining pulsars '
          f'(sum_w={sum_w_remaining:.0f}s) → cadence_factor={cadence_nonbest:.3f}')
 
    return {
        'n_total': len(all_names), 'n_upgraded': n_upgraded,
        'n_remaining': n_remaining, 'weights': weights,
        'budget_total': budget_total, 'cost_upgraded': cost_upgraded,
        'leftover': leftover, 'sum_w_remaining': sum_w_remaining,
        'cadence_nonbest': cadence_nonbest,
    }
 
 
def _get_telescope_time_budget(cfg: dict, scenario_label: str,
                                best_only: bool, best_psrs,
                                par_dir: str, tim_dir: str,
                                dwell_flag_key: str = 'tobs') -> dict:
    """Cached wrapper — computed once per scenario, not once per pulsar."""
    key = (id(cfg), tim_dir, dwell_flag_key)
    if key not in _TELESCOPE_BUDGET_CACHE:
        _TELESCOPE_BUDGET_CACHE[key] = _compute_telescope_time_budget(
            cfg, scenario_label, best_only, best_psrs, par_dir, tim_dir,
            dwell_flag_key=dwell_flag_key)
    return _TELESCOPE_BUDGET_CACHE[key]
 
 
# ──────────────────────────────────────────────────────────────────────────
# REPLACES your existing _resolve_pulsar_scenario_params
# ──────────────────────────────────────────────────────────────────────────
 
def _resolve_pulsar_scenario_params(cfg: dict, psr_name: str,
                                     best_only: bool, best_psrs,
                                     par_dir: str = None,
                                     tim_dir: str = None,
                                     scenario_label: str = 'scenario') -> tuple:
    """
    Resolve (cadence_factor, toaerr_factor) for a given base pulsar name.
 
      - Explicit per_pulsar override -> used verbatim, bypasses best_only.
      - best_only=True and this pulsar not in best_psrs:
          * conserve_telescope_time not set / False (default) -> (1, 1.0),
            identical to current behavior.
          * conserve_telescope_time=True -> (cadence_nonbest, 1.0), a
            uniform cadence reduction computed once per scenario, weighted
            by each pulsar's real mean -tobs dwell time, so total
            array-wide telescope time matches business-as-usual.
      - Otherwise -> the scenario's top-level cadence_factor/toaerr_factor.
    """
    override = cfg.get('per_pulsar', {}).get(psr_name)
    if override:
        return (override.get('cadence_factor', cfg.get('cadence_factor', 1)),
                override.get('toaerr_factor',  cfg.get('toaerr_factor', 1.0)))
 
    if best_only and psr_name not in best_psrs:
        if cfg.get('conserve_telescope_time', False):
            budget = _get_telescope_time_budget(
                cfg, scenario_label, best_only, best_psrs,
                par_dir or PAR_DIR, tim_dir or TIM_DIR,
                dwell_flag_key=cfg.get('dwell_flag_key', 'tobs'))
            return budget['cadence_nonbest'], 1.0
        return 1, 1.0
 
    return cfg.get('cadence_factor', 1), cfg.get('toaerr_factor', 1.0)

def _scenario_params_cache_key(cadence_factor, toaerr_factor,
                                extension_years, extension_fraction) -> str:
    """
    Cache key based on the pulsar's actually-resolved observation strategy,
    not the scenario name. Two scenarios (or two pulsars within different
    scenarios, e.g. via conserve_telescope_time) that resolve to the same
    parameters share one physical tim file.
 
    extension_years takes precedence if set (absolute forecast length);
    otherwise extension_fraction (relative to that pulsar's own real
    span) is used — either way, two scenarios with the same value here
    will always produce the same actual forecast length for a given
    pulsar, since that pulsar's real span is itself deterministic.
    """
    ext_key = (f'{extension_years:.4f}y' if extension_years is not None
               else f'{extension_fraction:.4f}xspan')
    return f'cad{cadence_factor:.6f}_err{toaerr_factor:.6f}_{ext_key}'


def _pulsar_needs_scenario_tim(cfg: dict, cadence_factor, toaerr_factor) -> bool:
    """
    True if this pulsar needs a scenario .tim written — either because its
    resolved cadence/precision differ from (1, 1.0), OR because the
    scenario has a forecast extension configured (in which case EVERY
    pulsar in the scenario gets a forecast segment appended, even ones
    continuing at today's cadence/precision).
    """
    modified_rate = not (cadence_factor == 1 and toaerr_factor == 1.0)
    has_extension = (
        cfg.get('extension_years') is not None
        or bool(cfg.get('extension_fraction', 0.0))
    )
    return modified_rate or has_extension

import tempfile   # add near your other top-of-file imports if not already present
 
 
def _group_into_epochs(toa_records):
    """
    Group real TOA records into observing epochs by source file path
    (the first whitespace-separated field of each line). Every channel/
    sub-band line from the same observation session shares an identical
    file path — differing only in frequency/TOA/error/channel — so this
    is a completely reliable epoch key, no time-clustering heuristics
    needed. Returns a list of epochs (each a list of records, in original
    order); a dict is used internally to also tolerate non-contiguous
    lines sharing a filename, though in practice tim files are contiguous
    per epoch.
    """
    groups = {}
    order = []
    for rec in toa_records:
        key = rec['line'].split()[0]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(rec)
    return [groups[k] for k in order]
 
 
def _write_scenario_tim(orig_tim_path,
                         psr_name,
                         outpath,
                         cadence_factor=1,
                         toaerr_factor=1.0,
                         extension_days=None,
                         extension_fraction=1.0,
                         scenario_label='scenario'):
    """
    Write a synthetic .tim consisting of:
      1. Every real TOA line, reproduced exactly — cadence/error changes
         are never applied to recorded data.
      2. A forecast segment of synthetic FULL EPOCHS strictly after the
         last real epoch: spaced at (mean real epoch cadence /
         cadence_factor). Each future epoch replicates every channel/
         sub-band line of a resampled real epoch — same flags, frequency,
         receiver/backend mix — shifted onto the new epoch's TOA, with
         each line's error scaled by toaerr_factor. This keeps a
         synthetic future epoch structurally like a real multi-channel
         observation instead of a single point.
 
    extension_days, if given, overrides extension_fraction. Otherwise the
    forecast spans `extension_fraction * real_span` days (default 1.0 ->
    forecast length matches the real dataset span, i.e. total baseline
    doubles).
 
    Writes a sidecar `<outpath>.meta.json` used by _scenario_tim_is_valid
    to check cache freshness without re-deriving expected counts.
 
    Both the .tim and .meta.json are written atomically (temp file +
    os.replace) so concurrent processes sharing the same scenario_tim_dir
    cache can never interleave writes into the same output file.
    """
    header_lines, toa_records = _parse_tim_lines(orig_tim_path)
    if not toa_records:
        raise ValueError(f'No TOA records found in {orig_tim_path}')
 
    epochs = _group_into_epochs(toa_records)
    epoch_toas = np.array([
        float(np.mean([r['toa'] for r in ep])) for ep in epochs
    ])
    real_span = float(np.max(epoch_toas) - np.min(epoch_toas))
 
    if extension_days is None:
        extension_days = extension_fraction * real_span
 
    future_epoch_toas = _future_toa_times(epoch_toas, cadence_factor, extension_days)
 
    seed = _deterministic_seed(psr_name, scenario_label, orig_tim_path)
    rng = np.random.default_rng(seed)
 
    if len(future_epoch_toas) > 0:
        # Each future epoch borrows the FULL structure (every channel line)
        # of a resampled real epoch, rather than a single borrowed line.
        template_epoch_idx = rng.integers(0, len(epochs), size=len(future_epoch_toas))
    else:
        template_epoch_idx = np.array([], dtype=int)
 
    if len(template_epoch_idx) > 0 and MAX_SYNTH_TOAS is not None:
        n_future_toas_precap = sum(len(epochs[int(i)]) for i in template_epoch_idx)
        avg_epoch_size = max(1.0, n_future_toas_precap / len(template_epoch_idx))
        max_future_epochs = max(1, int(MAX_SYNTH_TOAS / avg_epoch_size))
        if len(template_epoch_idx) > max_future_epochs:
            print(f'  ⚠ [{psr_name}] capping forecast epochs '
                  f'{len(template_epoch_idx)} → {max_future_epochs} '
                  f'(~{avg_epoch_size:.1f} lines/epoch, limit {MAX_SYNTH_TOAS} lines)')
            keep = np.round(
                np.linspace(0, len(template_epoch_idx) - 1, max_future_epochs)
            ).astype(int)
            future_epoch_toas  = future_epoch_toas[keep]
            template_epoch_idx = template_epoch_idx[keep]
 
    # ── atomic .tim write: temp file in the same dir, sanity-check it,
    #    then os.replace() into place ────────────────────────────────────
    out_dir = os.path.dirname(outpath) or '.'
    os.makedirs(out_dir, exist_ok=True)
 
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=out_dir, prefix=f'.{os.path.basename(outpath)}.', suffix='.tmp')
    n_future_written = 0
    try:
        with os.fdopen(tmp_fd, 'w') as fout:
            for line in header_lines:
                fout.write(line + '\n')
 
            # real data, byte-for-byte — no rescaling, ever
            for rec in toa_records:
                fout.write(rec['line'] + '\n')
 
            # forecast segment — full synthetic epochs, one real epoch's
            # worth of channel lines per future epoch
            for epoch_toa, t_idx in zip(future_epoch_toas, template_epoch_idx):
                template_epoch = epochs[int(t_idx)]
                for rec in template_epoch:
                    new_err = rec['err'] * toaerr_factor
                    fout.write(
                        _rebuild_line(rec, new_toa=epoch_toa, new_err=new_err) + '\n'
                    )
                    n_future_written += 1
 
        written = _tim_nobs(tmp_path)
        expected = len(toa_records) + n_future_written
        if written != expected:
            os.unlink(tmp_path)
            raise RuntimeError(
                f'{psr_name}: wrote {written} TOAs but expected {expected}'
            )
 
        os.replace(tmp_path, outpath)   # atomic — no reader ever sees a partial file
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
 
    meta = {
        'n_real':          len(toa_records),
        'n_future':         n_future_written,
        'n_future_epochs':  len(future_epoch_toas),
        'cadence_factor':   cadence_factor,
        'toaerr_factor':    toaerr_factor,
        'extension_days':   extension_days,
        'seed':             seed,
    }
 
    # ── atomic .meta.json write, same pattern ─────────────────────────────
    meta_path = outpath + '.meta.json'
    meta_tmp_fd, meta_tmp_path = tempfile.mkstemp(
        dir=out_dir, prefix=f'.{os.path.basename(meta_path)}.', suffix='.tmp')
    try:
        with os.fdopen(meta_tmp_fd, 'w') as fh:
            json.dump(meta, fh)
        os.replace(meta_tmp_path, meta_path)   # atomic
    except Exception:
        if os.path.exists(meta_tmp_path):
            os.unlink(meta_tmp_path)
        raise
 
    return meta
 
def _scenario_tim_is_valid(scen_tim, orig_tim, cadence_factor, toaerr_factor,
                            extension_days=None, extension_fraction=1.0):
    """
    A cached scenario .tim is valid iff it exists, is non-empty, and its
    sidecar metadata matches the forecast parameters currently requested.
    Comparing recorded params (rather than recomputing an expected count
    from scratch) keeps this in lockstep with whatever _write_scenario_tim
    actually did, even if its internals change later.
    """
    if not os.path.exists(scen_tim):
        return False
 
    meta_path = scen_tim + '.meta.json'
    if not os.path.exists(meta_path):
        return False
 
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except Exception:
        return False
 
    if extension_days is None:
        _, toa_records = _parse_tim_lines(orig_tim)
        if not toa_records:
            return False
        orig_toas = np.array([r['toa'] for r in toa_records])
        real_span = float(np.max(orig_toas) - np.min(orig_toas))
        extension_days = extension_fraction * real_span
 
    n_scen = _tim_nobs(scen_tim)
    if n_scen == 0:
        return False
 
    ok = (
        abs(meta.get('cadence_factor', -1) - cadence_factor) < 1e-9
        and abs(meta.get('toaerr_factor', -1) - toaerr_factor) < 1e-9
        and abs(meta.get('extension_days', -1) - extension_days) < 1.0
        and n_scen == meta.get('n_real', -1) + meta.get('n_future', -1)
    )
    return ok

def load_single_pulsar(
    par: str,
    verbose: bool = False,
    scenario: str = 'baseline',
    scenarios = None,
    scenario_tim_dir: str = 'scenario_tims',
    par_dir = None,
    tim_dir = None,
    skip_pulsars = None,
):
    """
    Load libstempo pulsars for a given .par file.
 
    If the par file is a base par (e.g. J1713+0747.par), loads ALL tim
    variants (J1713+0747, J1713+0747ao, J1713+0747gbt).
    If the par file is a variant par (e.g. J1713+0747ao.par), loads ONLY
    the matching tim variant.
 
    Returns a list of tempopulsar objects.
    """
    _par_dir   = par_dir       or PAR_DIR
    _tim_dir   = tim_dir       or TIM_DIR
    _skip      = skip_pulsars  if skip_pulsars is not None else SKIP_PULSARS
    _scenarios = scenarios     if scenarios    is not None else SCENARIOS
 
    if scenario not in _scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {list(_scenarios.keys())}"
        )
 
    cfg                = _scenarios[scenario]
    best_only          = cfg.get('best_only', True)
    best_psrs          = cfg.get('best_psrs', BEST_PSRS)
    per_pulsar_cfg     = cfg.get('per_pulsar', {})
    extension_years    = cfg.get('extension_years', None)
    extension_fraction = cfg.get('extension_fraction', 0.0)
    extension_days     = (extension_years * 365.25
                           if extension_years is not None else None)
 
    # Derive base pulsar name AND the variant this par file represents
    _suffixes = ('ao', 'gbt', 'vla', 'fast')
    par_stem     = par.replace('.par', '').split('_')[0]
    psr_name     = par_stem
    par_variant  = ''
    for sfx in _suffixes:
        if psr_name.endswith(sfx):
            psr_name    = psr_name[:-len(sfx)]
            par_variant = sfx
            break
 
    if psr_name in _skip:
        return []
  
    cadence_factor, toaerr_factor = _resolve_pulsar_scenario_params(
        cfg, psr_name, best_only, best_psrs,
        par_dir=_par_dir, tim_dir=_tim_dir, scenario_label=scenario)
    modified = _pulsar_needs_scenario_tim(cfg, cadence_factor, toaerr_factor)
 
    BASE_MAXOBS = 60000
    maxobs = int(BASE_MAXOBS * max(cadence_factor, 1.0) * 1.2)
 
    par_map   = _find_par_files(_par_dir, psr_name)
    all_tims  = [t for t in _find_tim_files(_tim_dir, psr_name) if tim_has_toas(t)]
 
    if not par_map or not all_tims:
        return []
 
    if par_variant:
        tim_paths = [
            t for t in all_tims
            if os.path.basename(t).replace('.tim', '').split('_')[0]
               == f'{psr_name}{par_variant}'
        ]
    else:
        tim_paths = [
            t for t in all_tims
            if os.path.basename(t).replace('.tim', '').split('_')[0][len(psr_name):]
               not in par_map or
               os.path.basename(t).replace('.tim', '').split('_')[0][len(psr_name):]
               == ''
        ]
 
    if not tim_paths:
        return []
 
    loaded = []
    for tim_path in tim_paths:
        tim_stem  = os.path.basename(tim_path).replace('.tim', '').split('_')[0]
        variant   = tim_stem[len(psr_name):]
        load_name = f'{psr_name}{variant}'
 
        par_path = par_map.get(variant) or par_map.get('')
        if par_path is None:
            if verbose:
                print(f'⚠ No par for variant {load_name}, skipping')
            continue
 
        effective_tim = tim_path
 
        if modified:
            cache_key = _scenario_params_cache_key(
                cadence_factor, toaerr_factor, extension_years, extension_fraction)
            scen_dir = os.path.join(scenario_tim_dir, '_by_params', cache_key)
            os.makedirs(scen_dir, exist_ok=True)
            scen_tim = os.path.join(scen_dir, f'{load_name}.tim')
 
            if (not os.path.exists(scen_tim)
                    or not _scenario_tim_is_valid(
                        scen_tim, tim_path, cadence_factor, toaerr_factor,
                        extension_days=extension_days,
                        extension_fraction=extension_fraction)):
                if verbose:
                    print(f'  ✎ Writing scenario tim for {load_name}...',
                          end=' ', flush=True)
                try:
                    psr_tmp = T.tempopulsar(
                        parfile=par_path, timfile=tim_path,
                        maxobs=maxobs, dofit=False)
                    meta = _write_scenario_tim(
                        tim_path, load_name, scen_tim,
                        cadence_factor=cadence_factor,
                        toaerr_factor=toaerr_factor,
                        extension_days=extension_days,
                        extension_fraction=extension_fraction,
                        scenario_label=cache_key)
                    if verbose:
                        print(f'✓ ({meta["n_real"]} real + {meta["n_future"]} '
                              f'forecast = {meta["n_real"] + meta["n_future"]} TOAs)')
                    del psr_tmp
                    gc.collect()
                except Exception as e:
                    if verbose:
                        print(f'✗ failed: {e}')
                    continue
            else:
                if verbose:
                    print(f'  ✓ {load_name:20s} scenario tim exists, reusing')
 
            effective_tim = scen_tim
 
        elif verbose and (cfg.get('cadence_factor', 1) != 1
                       or cfg.get('toaerr_factor', 1.0) != 1.0
                       or per_pulsar_cfg):
            print(f'  ✓ {load_name:20s} not upgraded (best_only) — original tim')
 
        try:
            psr = T.tempopulsar(
                parfile=par_path, timfile=effective_tim,
                maxobs=maxobs, dofit=False)
            if psr.nobs > 0:
                loaded.append(psr)
                if verbose:
                    print(f'✓ Loaded {psr.name} ({psr.nobs} TOAs) [tim={tim_stem}]')
        except Exception as e:
            if verbose:
                print(f'✗ Failed {load_name}: {e}')
 
    return loaded


def load_pulsars(
    verbose          = True,
    scenario         = 'baseline',
    scenarios        = None,
    scenario_tim_dir = 'scenario_tims',
    par_dir          = None,
    tim_dir          = None,
    skip_pulsars     = None,
    use_cache        = None,
    cache_path       = None,
):
    """
    Load NANOGrav pulsars as libstempo tempopulsar objects.
 
    For 'baseline' (no per_pulsar block, cadence_factor=toaerr_factor=1)
    behaviour is identical to before, pickle cache included. For other
    scenarios, cadence_factor/toaerr_factor are resolved per pulsar (via
    cfg['per_pulsar'] if present, else the scenario's top-level defaults),
    and modified pulsars get a real-data-preserved forecast tim written
    via _write_scenario_tim.
 
    Scenario tim caching is keyed by the RESOLVED PARAMETERS for each
    pulsar (cadence_factor, toaerr_factor, extension), not by scenario
    name — so any two scenarios (or two pulsars within a
    conserve_telescope_time scenario) landing on identical parameters
    share one physical tim file instead of regenerating separately.
 
    Returns
    -------
    list of libstempo.tempopulsar
    """
    _par_dir    = par_dir      or PAR_DIR
    _tim_dir    = tim_dir      or TIM_DIR
    _skip       = skip_pulsars if skip_pulsars is not None else SKIP_PULSARS
    _use_cache  = use_cache    if use_cache    is not None else USE_PULSAR_CACHE
    _cache_path = cache_path   or PULSAR_CACHE
    _scenarios  = scenarios    if scenarios    is not None else SCENARIOS
 
    if scenario not in _scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {list(_scenarios.keys())}"
        )
 
    cfg                 = _scenarios[scenario]
    default_cadence     = cfg.get('cadence_factor', 1)
    default_toaerr      = cfg.get('toaerr_factor',  1.0)
    best_only           = cfg.get('best_only',       True)
    best_psrs           = cfg.get('best_psrs',       BEST_PSRS)
    per_pulsar_cfg      = cfg.get('per_pulsar', {})
    extension_years     = cfg.get('extension_years', None)
    extension_fraction  = cfg.get('extension_fraction', 0.0)
    extension_days      = (extension_years * 365.25
                            if extension_years is not None else None)
 
    # A scenario is a true no-op baseline only if the top-level defaults
    # are unmodified AND there's no per-pulsar override at all — this is
    # what gates the pickle cache below.
    is_scenario_baseline = (
        default_cadence == 1 and default_toaerr == 1.0
        and not per_pulsar_cfg
    )
 
    if verbose:
        print('=' * 70)
        print(f'LOADING NANOGRAV PULSARS (libstempo)  —  scenario: {scenario}')
        if not is_scenario_baseline:
            print(f'  default: cadence×{default_cadence}  err×{default_toaerr}'
                  f'  best_only={best_only}')
            if per_pulsar_cfg:
                print(f'  per-pulsar overrides: {list(per_pulsar_cfg.keys())}')
        print('=' * 70)
 
    # ------------------------------------------------------------------
    # Baseline path: identical to original, cache included
    # ------------------------------------------------------------------
    if is_scenario_baseline:
        if _use_cache and os.path.exists(_cache_path):
            if verbose:
                print(f'\n📦 Loading from cache: {_cache_path}')
            try:
                with open(_cache_path, 'rb') as f:
                    psrs = pickle.load(f)
                if verbose:
                    print(f'✓ Loaded {len(psrs)} pulsars from cache\n')
                return psrs
            except Exception as e:
                if verbose:
                    print(f'⚠ Cache load failed: {e}')
 
    # ------------------------------------------------------------------
    # Build sorted list of unique base pulsar names from par directory
    # ------------------------------------------------------------------
    _suffixes = ('ao', 'gbt', 'vla', 'fast')
    parfiles   = sorted([f for f in os.listdir(_par_dir) if f.endswith('.par')])
 
    seen_set   = set()
    base_names = []
    for par in parfiles:
        stem = par.replace('.par', '').split('_')[0]
        base = stem
        for sfx in _suffixes:
            if base.endswith(sfx):
                base = base[:-len(sfx)]
                break
        if base not in seen_set:
            seen_set.add(base)
            base_names.append(base)
 
    if verbose:
        print(f'Found {len(parfiles)} .par files → {len(base_names)} unique pulsars')
 
    # NOTE: scen_dir is no longer computed here — it depends on each
    # pulsar's own RESOLVED (cadence_factor, toaerr_factor), which isn't
    # known until inside the per-pulsar loop below. Computing it here
    # (keyed only on `scenario`) was the bug — every pulsar in a scenario
    # would share one directory even though conserve_telescope_time (or
    # per_pulsar) can give different pulsars different resolved params
    # within the SAME scenario call.
    psrs           = []
    failed_pulsars = []
 
    for psr_name in base_names:
        if psr_name in _skip:
            if verbose:
                print(f'⚠ Skipping {psr_name} (in SKIP_PULSARS)')
            continue
 
        par_map   = _find_par_files(_par_dir, psr_name)
        tim_paths = [t for t in _find_tim_files(_tim_dir, psr_name) if tim_has_toas(t)]
 
        if not par_map:
            if verbose:
                print(f'⚠ No par file found for {psr_name}')
            failed_pulsars.append(psr_name)
            continue
        if not tim_paths:
            if verbose:
                print(f'⚠ No valid tim for {psr_name}')
            failed_pulsars.append(psr_name)
            continue
 
        cadence_factor, toaerr_factor = _resolve_pulsar_scenario_params(
            cfg, psr_name, best_only, best_psrs,
            par_dir=_par_dir, tim_dir=_tim_dir, scenario_label=scenario)
        modified = _pulsar_needs_scenario_tim(cfg, cadence_factor, toaerr_factor)
 
        # Params-keyed cache: computed HERE, per pulsar, now that this
        # pulsar's own resolved cadence_factor/toaerr_factor are known.
        # Any other pulsar (this scenario or another) landing on the same
        # (cadence_factor, toaerr_factor, extension) shares this directory.
        cache_key = _scenario_params_cache_key(
            cadence_factor, toaerr_factor, extension_years, extension_fraction)
        scen_dir  = os.path.join(scenario_tim_dir, '_by_params', cache_key)
 
        BASE_MAXOBS = 60000
        maxobs = int(BASE_MAXOBS * max(cadence_factor, 1.0) * 1.2)
 
        for tim_path in tim_paths:
            tim_stem  = os.path.basename(tim_path).replace('.tim', '').split('_')[0]
            variant   = tim_stem[len(psr_name):]
            load_name = f'{psr_name}{variant}'
 
            par_path = par_map.get(variant) or par_map.get('')
            if par_path is None:
                if verbose:
                    print(f'⚠ No par for variant {load_name}, skipping')
                failed_pulsars.append(load_name)
                continue
 
            effective_tim = tim_path
 
            if modified:
                os.makedirs(scen_dir, exist_ok=True)
                scen_tim = os.path.join(scen_dir, f'{load_name}.tim')
 
                if (not os.path.exists(scen_tim)
                        or not _scenario_tim_is_valid(
                            scen_tim, tim_path, cadence_factor, toaerr_factor,
                            extension_days=extension_days,
                            extension_fraction=extension_fraction)):
                    if verbose:
                        print(f'  ✎ Writing scenario tim for {load_name} '
                              f'[{cache_key}]...', end=' ', flush=True)
                    try:
                        psr_tmp = T.tempopulsar(
                            parfile=par_path, timfile=tim_path,
                            maxobs=maxobs, dofit=False)
                        meta = _write_scenario_tim(
                            tim_path, load_name, scen_tim,
                            cadence_factor=cadence_factor,
                            toaerr_factor=toaerr_factor,
                            extension_days=extension_days,
                            extension_fraction=extension_fraction,
                            scenario_label=cache_key)
                        if verbose:
                            print(f'✓ ({meta["n_real"]} real + {meta["n_future"]} '
                                  f'forecast = {meta["n_real"] + meta["n_future"]} TOAs)')
                        del psr_tmp
                    except Exception as e:
                        if verbose:
                            print(f'✗ failed: {e}')
                        failed_pulsars.append(load_name)
                        continue
                else:
                    if verbose:
                        print(f'  ✓ {load_name:20s} [{cache_key}] '
                              f'scenario tim exists, reusing')
 
                effective_tim = scen_tim
 
            elif verbose and (cfg.get('cadence_factor', 1) != 1
                       or cfg.get('toaerr_factor', 1.0) != 1.0
                       or per_pulsar_cfg):
                print(f'  ✓ {load_name:20s} not upgraded (best_only) — original tim')
 
            try:
                raw_n  = _tim_nobs(effective_tim)
                psr = T.tempopulsar(
                    parfile=par_path,
                    timfile=effective_tim,
                    maxobs=maxobs,
                    dofit=False,
                )
                if psr.nobs == 0:
                    if verbose:
                        print(f'⚠ Zero TOAs for {load_name}')
                    failed_pulsars.append(load_name)
                    continue
 
                psrs.append(psr)
                if verbose:
                    print(f'✓ Loaded {psr.name} ({psr.nobs} TOAs) [tim={tim_stem}]')
 
            except Exception as e:
                if verbose:
                    print(f'✗ Failed {load_name}: {e}')
                failed_pulsars.append(load_name)
 
    # ------------------------------------------------------------------
    # Cache true baseline only
    # ------------------------------------------------------------------
    if is_scenario_baseline and _use_cache and psrs:
        try:
            with open(_cache_path, 'wb') as f:
                pickle.dump(psrs, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f'\n💾 Saved cache: {_cache_path}')
        except Exception as e:
            if verbose:
                print(f'⚠ Could not save cache: {e}')
 
    if verbose:
        print(f'\n✓ Loaded {len(psrs)} pulsars [scenario={scenario}]')
        print(f'✗ Failed: {len(failed_pulsars)} — {failed_pulsars}')
 
    return psrs