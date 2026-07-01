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
BEST_PSRS_MEERKAT = (
    'J1327-0755', 'J1545-4550', 'J0613-0200', 
    'J2322-2650', 'J1918-0642', 'J1446-4701', 
    'J2039-3616', 'J1744-1134', 'J1125-6014', 
    'J2124-3358', 'J1732-5049', 'J1946-5403', 
    'J0711-6830', 'J1629-6902', 'J2010-1323', 
    'J2129-5721', 'J1909-3744', 'J0125-2327', 
    'J0437-4715', 'J2241-5236'
)

BEST_PSRS = BEST_PSRS_NANOGrav if NANOGRAV_PULSARS else BEST_PSRS_MEERKAT
 
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
    ),
    '4x_cadence': dict(
        cadence_factor = 4,
        toaerr_factor  = 1.0,
        best_only      = True,
    ),
    #     # New: per-pulsar control
    # 'hi_prec': dict(
    #     cadence_factor = 1,      # default for any best pulsar not listed below
    #     toaerr_factor  = 1.0,    # default for any best pulsar not listed below
    #     best_only      = True,
    #     per_pulsar     = {
    #         'J1713+0747':  dict(cadence_factor=1, toaerr_factor=0.5),
    #         'J1909-3744':  dict(cadence_factor=1,  toaerr_factor=0.5),
    #         'J2043+1711':  dict(cadence_factor=1,  toaerr_factor=0.25),
    #         'J1741+1351':  dict(cadence_factor=1,  toaerr_factor=0.25),
    #         'J1918-0642':  dict(cadence_factor=1,  toaerr_factor=0.25),
    #     },
    # ),
    '2x_precision': dict(
        cadence_factor = 1,
        toaerr_factor  = 0.5,
        best_only      = True,
    ),
    '2x_cad_2x_prec': dict(
        cadence_factor = 2,
        toaerr_factor  = 0.5,
        best_only      = True,
    ),
    '4x_cad_2x_prec': dict(
        cadence_factor = 4,
        toaerr_factor  = 0.5,
        best_only      = True,
    ),
    # '5x_cad_4x_prec': dict(
    #     cadence_factor = 5,
    #     toaerr_factor  = 0.25,
    #     best_only      = True,
    # ),
    # # New: per-pulsar control
    # 'hi_prec_hi_cad': dict(
    #     cadence_factor = 1,      # default for any best pulsar not listed below
    #     toaerr_factor  = 1.0,    # default for any best pulsar not listed below
    #     best_only      = True,
    #     per_pulsar     = {
    #         'J1713+0747':  dict(cadence_factor=5, toaerr_factor=0.5),
    #         'J1909-3744':  dict(cadence_factor=5,  toaerr_factor=0.5),
    #         'J2043+1711':  dict(cadence_factor=5,  toaerr_factor=0.25),
    #         'J1741+1351':  dict(cadence_factor=5,  toaerr_factor=0.25),
    #         'J1918-0642':  dict(cadence_factor=5,  toaerr_factor=0.25),
    #     },
    # ),
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
 
def _augmented_nobs(nobs, cadence_factor):
    """Return the exact TOA count after interleaving cadence_factor-1 TOAs per gap."""
    if cadence_factor <= 1 or nobs <= 0:
        return int(nobs)
    return int(nobs + (nobs - 1) * (cadence_factor - 1))
 
def _tim_nobs(tim_path):
    """Count non-comment TOA lines in a tempo2 .tim file."""
    count = 0
    with open(tim_path, 'r') as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith(('FORMAT', 'C', '*')):
                continue
            count += 1
    return count


def _scenario_tim_is_valid(
    scen_tim,
    orig_tim,
    cadence_factor,
    toaerr_factor,
):
    if not os.path.exists(scen_tim):
        return False

    n_scen = _tim_nobs(scen_tim)

    if n_scen == 0:
        return False

    n_orig = _tim_nobs(orig_tim)

    expected = (
        n_orig +
        (n_orig - 1) * (cadence_factor - 1)
    )

    if MAX_SYNTH_TOAS is not None:
        expected = min(expected, MAX_SYNTH_TOAS)

    if abs(n_scen - expected) > max(5, int(0.05 * expected)):
        return False

    return True
 
def _safe_maxobs(tim_path):

    n = _tim_nobs(tim_path)

    return int(n + max(500, 0.05 * n))

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
    the matching tim variant (J1713+0747ao) to avoid duplicates when the
    caller iterates over all par files.

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

    cfg            = _scenarios[scenario]
    cadence_factor = cfg.get('cadence_factor', 1)
    toaerr_factor  = cfg.get('toaerr_factor', 1.0)
    best_only      = cfg.get('best_only', True)
    best_psrs      = cfg.get('best_psrs', BEST_PSRS)
    is_baseline    = (cadence_factor == 1 and toaerr_factor == 1.0)

    BASE_MAXOBS = 60000
    maxobs = int(BASE_MAXOBS * cadence_factor * 1.2)
    # Derive base pulsar name AND the variant this par file represents
    _suffixes = ('ao', 'gbt', 'vla', 'fast')
    par_stem     = par.replace('.par', '').split('_')[0]
    psr_name     = par_stem
    par_variant  = ''                      # variant encoded in the par filename
    for sfx in _suffixes:
        if psr_name.endswith(sfx):
            psr_name    = psr_name[:-len(sfx)]
            par_variant = sfx
            break

    if psr_name in _skip:
        return []

    par_map   = _find_par_files(_par_dir, psr_name)
    all_tims  = [t for t in _find_tim_files(_tim_dir, psr_name) if tim_has_toas(t)]

    if not par_map or not all_tims:
        return []

    # If called with a variant par (e.g. J1713+0747ao.par), only load the
    # matching tim variant so the caller doesn't get duplicates when it
    # iterates over every par file.
    # If called with the base par (e.g. J1713+0747.par), load ALL variants
    # whose tim has no dedicated par file of its own (they rely on the base par).
    if par_variant:
        # variant par — only the one matching tim
        tim_paths = [
            t for t in all_tims
            if os.path.basename(t).replace('.tim', '').split('_')[0]
               == f'{psr_name}{par_variant}'
        ]
    else:
        # base par — only tims whose variant has no dedicated par file
        tim_paths = [
            t for t in all_tims
            if os.path.basename(t).replace('.tim', '').split('_')[0][len(psr_name):]
               not in par_map or
               os.path.basename(t).replace('.tim', '').split('_')[0][len(psr_name):]
               == ''
        ]

    if not tim_paths:
        return []

    is_best  = psr_name in best_psrs
    modified = not is_baseline and ((not best_only) or is_best)

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
            scen_dir = os.path.join(scenario_tim_dir, scenario)
            os.makedirs(scen_dir, exist_ok=True)
            scen_tim = os.path.join(scen_dir, f'{load_name}.tim')

            if (not os.path.exists(scen_tim)
                    or not _scenario_tim_is_valid(
                        scen_tim, tim_path, cadence_factor, toaerr_factor)):
                if verbose:
                    print(f'  ✎ Writing scenario tim for {load_name}...',
                          end=' ', flush=True)
                try:
                    psr_tmp = T.tempopulsar(
                        parfile=par_path, timfile=tim_path,
                        maxobs=maxobs, dofit=False)
                    _write_scenario_tim(
                        tim_path, load_name, scen_tim,
                        cadence_factor=cadence_factor,
                        toaerr_factor=toaerr_factor)
                    if verbose:
                        n_new = _augmented_nobs(psr_tmp.nobs, cadence_factor)
                        print(f'✓ ({psr_tmp.nobs} → ~{n_new} TOAs)')
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

        elif not is_baseline and verbose:
            print(f'  ✓ {load_name:20s} not in best_psrs — original tim')

        try:
            raw_n  = _tim_nobs(effective_tim)
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

    Drop-in replacement for the original load_pulsars(). When scenario is
    'baseline' (default) behaviour is identical including the pickle cache.
    For other scenarios the function transparently writes / reuses augmented
    .tim files and returns the modified pulsars, skipping the cache.

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

    cfg            = _scenarios[scenario]
    cadence_factor = cfg.get('cadence_factor', 1)
    toaerr_factor  = cfg.get('toaerr_factor',  1.0)
    best_only      = cfg.get('best_only',       True)
    best_psrs      = cfg.get('best_psrs',       BEST_PSRS)
    is_baseline    = (cadence_factor == 1 and toaerr_factor == 1.0)

    BASE_MAXOBS = 60000
    maxobs = int(BASE_MAXOBS * cadence_factor * 1.2)

    if verbose:
        print('=' * 70)
        print(f'LOADING NANOGRAV PULSARS (libstempo)  —  scenario: {scenario}')
        if not is_baseline:
            print(f'  cadence×{cadence_factor}  err×{toaerr_factor}'
                  f'  best_only={best_only}')
        print('=' * 70)

    # ------------------------------------------------------------------
    # Baseline path: identical to original, cache included
    # ------------------------------------------------------------------
    if is_baseline:
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

    scen_dir       = os.path.join(scenario_tim_dir, scenario)
    psrs           = []
    failed_pulsars = []

    # load_pulsars iterates by unique base name, so no deduplication
    # issue here — _find_tim_files returns all variants and we load them all
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

        is_best  = psr_name in best_psrs
        modified = not is_baseline and ((not best_only) or is_best)

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
                            scen_tim, tim_path, cadence_factor, toaerr_factor)):
                    if verbose:
                        print(f'  ✎ Writing scenario tim for {load_name}...',
                              end=' ', flush=True)
                    try:
                        raw_n  = _tim_nobs(tim_path)
                        psr_tmp = T.tempopulsar(
                            parfile=par_path, timfile=tim_path,
                            maxobs=maxobs, dofit=False)
                        _write_scenario_tim(
                            tim_path, load_name, scen_tim,
                            cadence_factor=cadence_factor,
                            toaerr_factor=toaerr_factor)
                        n_new = _augmented_nobs(psr_tmp.nobs, cadence_factor)
                        if verbose:
                            print(f'✓ ({psr_tmp.nobs} → ~{n_new} TOAs)')
                        del psr_tmp
                    except Exception as e:
                        if verbose:
                            print(f'✗ failed: {e}')
                        failed_pulsars.append(load_name)
                        continue
                else:
                    if verbose:
                        print(f'  ✓ {load_name:20s} scenario tim exists, reusing')

                effective_tim = scen_tim

            elif not is_baseline and verbose:
                print(f'  ✓ {load_name:20s} not in best_psrs — original tim')

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
    # Cache baseline only
    # ------------------------------------------------------------------
    if is_baseline and _use_cache and psrs:
        try:
            with open(_cache_path, 'wb') as f:
                pickle.dump(psrs, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f'\n💾 Saved cache: {_cache_path}')
        except Exception as e:
            if verbose:
                print(f'⚠ Could not save cache: {e}')

    if verbose:
        print(f'\n✓ Loaded {len(psrs)} pulsars '
              f'[scenario={scenario}  cadence×{cadence_factor}  err×{toaerr_factor}]')
        print(f'✗ Failed: {len(failed_pulsars)} — {failed_pulsars}')

    return psrs