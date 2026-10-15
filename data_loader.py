import os
import pickle
import json
import numpy as np
import gc
import hashlib
import tempfile
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
MAX_SYNTH_TOAS = 120_000


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
        tmin = min(psr.toas())
        tmax = max(psr.toas())
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

    Tspan = float(total_tmax - total_tmin)  # days
    Tspan_seconds = Tspan * 86400  # seconds
    if verbose:
        print(f"\nFiltered: {len(psrs)} → {len(psrs_filtered)} pulsars")

    return psrs_filtered, params, Tspan_seconds


_ORIGINAL_RESIDUALS = {}  # module-level cache: psr.name -> np.ndarray


def get_clean_pulsars_and_tspan(psrs_filtered):
    """
    Get pulsars and calculate Tspan.

    Note: Returns original pulsars (not copies) to save memory.
    Original residuals are cached externally (keyed by pulsar name) for
    restoration between injections, since libstempo's tempopulsar is a
    Cython extension type and doesn't support arbitrary attribute assignment.
    """
    tmin = min(min(p.toas()) for p in psrs_filtered)
    tmax = max(max(p.toas()) for p in psrs_filtered)
    Tspan = tmax - tmin

    for psr in psrs_filtered:
        if psr.name not in _ORIGINAL_RESIDUALS:
            _ORIGINAL_RESIDUALS[psr.name] = np.copy(psr.residuals)

    return psrs_filtered, Tspan

def restore_original_residuals(psrs):
    """Restore pulsars to original state before next injection."""
    for psr in psrs:
        if psr.name in _ORIGINAL_RESIDUALS:
            psr.residuals[:] = _ORIGINAL_RESIDUALS[psr.name]
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
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    pulsar_params = defaultdict(lambda: {'red_noise': {}, 'white_noise': {}})

    for key, value in data.items():
        if 'red_noise' in key:
            pulsar_name = key.split('_red_noise')[0]

            if 'gamma' in key:
                pulsar_params[pulsar_name]['red_noise']['gamma'] = value
            elif 'log10_A' in key:
                pulsar_params[pulsar_name]['red_noise']['log10_A'] = value

        elif 'efac' in key or 'ecorr' in key or 't2equad' in key:
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

            base_key = key[:split_idx]
            parts = base_key.split('_')

            pulsar_name = None
            for i, part in enumerate(parts):
                if '+' in part:
                    pulsar_name = '_'.join(parts[:i + 1])
                    backend_name = '_'.join(parts[i + 1:])
                    break
                elif '-' in part:
                    if i > 0 and parts[i - 1] == 'L' and part == 'wide':
                        continue
                    else:
                        pulsar_name = '_'.join(parts[:i + 1])
                        backend_name = '_'.join(parts[i + 1:])
                        break

            if pulsar_name is None:
                pulsar_name = parts[0]
                backend_name = '_'.join(parts[1:])

            if backend_name not in pulsar_params[pulsar_name]['white_noise']:
                pulsar_params[pulsar_name]['white_noise'][backend_name] = {}

            pulsar_params[pulsar_name]['white_noise'][backend_name][param_type] = value

    return {k: dict(v) for k, v in pulsar_params.items()}


#### Adapting to simulate pulsars with different cadences and errors

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

# Only want top 10 to conserve telescope time -- used for 6-7/07/26 sims
BEST_PSRS_MEERKAT = (
    'J2241-5236', 'J1909-3744', 'J0711-6830',
    'J1744-1134', 'J1629-6902', 'J2129-5721',
    'J1946-5403', 'J1125-6014', 'J0437-4715',
    'J0125-2327',
)

# Best psrs from 6-7/07/26 runs
BEST_PSRS_COMBINED = (
    'J2241-5236', 'J1909-3744', 'J1946-5403',
    'J2129-5721', 'J0437-4715', 'J0711-6830',
    'J1744-1134', 'J2010-1323', 'J1629-6902',
    'J1125-6014', 'J1600-3053', 'J1446-4701',
    'J2039-3616', 'J2124-3358', 'J1545-4550',
    'J1732-5049', 'J0125-2327', 'J1811-2405',
    'J1918-0642', 'J1216-6410', 'J1933-6211',
    'J1036-8317', 'J1614-2230', 'J1903-7051',
    'J1017-7156', 'J1658-5324', 'J1843-1113',
    'J1543-5149', 'J2322-2650', 'J1757-5322',
    'J1902-5105', 'J0613-0200', 'J1730-2304',
    'J1455-3330', 'J1832-0836', 'J2150-0326',
    'J1327-0755', 'J2145-0750', 'J1024-0719',
    'J1603-7202', 'J0614-3329', 'J0101-6422',
)

BEST_PSRS = BEST_PSRS_NANOGrav if NANOGRAV_PULSARS else BEST_PSRS_MEERKAT

top42 = list(BEST_PSRS_COMBINED)        # all 42, ranked
top40 = top42[:40]
top30 = top42[:30]
top25 = top42[:25]
top20 = top42[:20]
top15 = top42[:15]
top10 = top42[:10]
top5  = top42[:5]

# ~3 months -- the default floor used by the max-cadence scenarios below.
# Override per-scenario via _make_max_cadence_scenario(..., min_cadence_days=...)
MIN_CADENCE_DAYS_DEFAULT = 90.0

# Each scenario entry:
#   cadence_factor  int    — TOA grid multiplier   (1 = unchanged)
#   toaerr_factor   float  — TOA error multiplier  (1.0 = unchanged)
#   best_only       bool   — apply only to BEST_PSRS if True
#   best_psrs       tuple  — override BEST_PSRS for this scenario (optional)
SCENARIOS = {
    "baseline": dict(
        cadence_factor=1,
        toaerr_factor=1.0,
        best_only=True,
        # No forecast extension. Uses the real observing data.
    ),

    "baseline_forecast": dict(
        cadence_factor=1,
        toaerr_factor=1.0,
        best_only=True,
        # Add one shared extension equal to the longest real pulsar span.
        extension_to_longest_span=True,
    ),

    "4x_cadence": dict(
        cadence_factor=4,
        toaerr_factor=1.0,
        best_only=True,
        extension_to_longest_span=True,
    ),

    "2x_precision": dict(
        cadence_factor=1,
        toaerr_factor=0.5,
        best_only=True,
        extension_to_longest_span=True,
    ),

    "4x_cad_2x_prec": dict(
        cadence_factor=4,
        toaerr_factor=0.5,
        best_only=True,
        extension_to_longest_span=True,
    ),

    "4x_cadence_conserved": dict(
        cadence_factor=4,
        toaerr_factor=1.0,
        best_only=True,
        extension_to_longest_span=True,
        conserve_telescope_time=True,
    ),

    "2x_precision_conserved": dict(
        cadence_factor=2,
        toaerr_factor=1.0,
        best_only=True,
        extension_to_longest_span=True,
        conserve_telescope_time=True,
    ),

    "4x_cad_2x_prec_conserved": dict(
        cadence_factor=4,
        toaerr_factor=0.5,
        best_only=True,
        extension_to_longest_span=True,
        conserve_telescope_time=True,
    ),
    "cgw_weighted_cadence": dict(
        cgw_weighted_cadence=True,
        min_cadence_days=90.0,
        toaerr_factor=1.0,
        best_only=False,
        conserve_telescope_time=True,
        extension_to_longest_span=True,
        cgw_weight_dir=(
            PROJECT_ROOT / "config" / "cgw_cadence_weights"
        ),
    ),
}


def _bounded_maxobs(tim_path: str) -> int:
    """
    maxobs sized to the actual possible maximum TOA count for a scenario
    tim built from this real tim_path — n_real (from the real file) +
    MAX_SYNTH_TOAS (the fixed forecast-segment cap) — NOT scaled by
    cadence_factor, which has no real bound. A generous 1.5x safety
    margin on top covers any minor cap-estimation slop in
    _write_scenario_tim's epoch-based subsampling.
    """
    n_real = _tim_nobs(tim_path)
    synth_cap = MAX_SYNTH_TOAS if MAX_SYNTH_TOAS is not None else n_real * 20
    return int((n_real + synth_cap) * 1.5)

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
    line" used everywhere.
    """
    _, toa_records = _parse_tim_lines(tim_path)
    return len(toa_records)


# ──────────────────────────────────────────────────────────────────────────
# Forecast helpers
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
    by toaerr_factor.
    """
    if len(orig_errs) == 0:
        raise ValueError('no real TOA errors available to sample from')
    draws = rng.choice(orig_errs, size=n_future, replace=True)
    return draws * toaerr_factor


# ──────────────────────────────────────────────────────────────────────────
# Telescope-time weighting (mean real -tobs per pulsar) — single set of
# caches / definitions, used by both conserve_telescope_time and the new
# maximize_best_cadence budget solver.
# ──────────────────────────────────────────────────────────────────────────

_TOBS_CACHE: dict = {}
_TELESCOPE_BUDGET_CACHE: dict = {}
_MAX_CADENCE_BUDGET_CACHE: dict = {}
_MIN_CADENCE_FACTOR_CACHE: dict = {}


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
    is found. Cached — only depends on real data, computed once per pulsar
    per run regardless of how many scenarios use it.
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


def _pulsar_min_cadence_factor(psr_name, tim_dir, min_cadence_days):
    """
    Cadence-factor floor for a "held back" pulsar: the smallest
    cadence_factor allowed for this pulsar such that its forecast spacing
    (mean_real_cadence_days / cadence_factor) never exceeds min_cadence_days.

    floor = min(1.0, mean_real_cadence_days / min_cadence_days)

    Capped at 1.0 on purpose: if a pulsar's own real cadence is ALREADY
    sparser than min_cadence_days, we don't force it denser (cadence_factor
    > 1) just to hit the floor — this mode only spends saved budget on the
    upgraded top-N, never on the held-back set. Such a pulsar simply stays
    at its natural (cadence_factor=1) rate, which is the best available
    without diverting budget from the upgraded pulsars.
    """
    key = (psr_name, tim_dir, min_cadence_days)
    if key in _MIN_CADENCE_FACTOR_CACHE:
        return _MIN_CADENCE_FACTOR_CACHE[key]

    tim_paths = [t for t in _find_tim_files(tim_dir, psr_name) if tim_has_toas(t)]
    all_toas = []
    for tp in tim_paths:
        _, recs = _parse_tim_lines(tp)
        all_toas.extend(r['toa'] for r in recs)

    if len(all_toas) < 2 or min_cadence_days <= 0:
        val = 1.0
    else:
        mean_dt = _mean_cadence_days(np.array(all_toas))
        val = float(min(1.0, mean_dt / min_cadence_days))

    _MIN_CADENCE_FACTOR_CACHE[key] = val
    return val


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


def _compute_max_cadence_budget(cfg: dict, scenario_label: str, best_psrs,
                                 par_dir: str, tim_dir: str,
                                 min_cadence_days: float,
                                 dwell_flag_key: str = 'tobs') -> dict:
    """
    Solve for the single uniform cadence_factor applied to `best_psrs` that
    exactly spends the array-wide telescope-time budget left over once every
    OTHER pulsar is held at its own min_cadence_days floor (see
    _pulsar_min_cadence_factor). toaerr_factor is fixed at 1.0 on both sides
    — this mode maxes out observing FREQUENCY for the upgraded set, not
    measurement precision.
    """
    all_names = _list_base_pulsar_names(par_dir)
    weights = {name: _pulsar_mean_tobs(name, tim_dir, flag_key=dwell_flag_key)
               for name in all_names}
    budget_total = sum(weights.values())

    upgraded_names = set(best_psrs) & set(all_names)
    remaining_names = set(all_names) - upgraded_names

    floor_factor = {
        name: _pulsar_min_cadence_factor(name, tim_dir, min_cadence_days)
        for name in remaining_names
    }
    cost_remaining = sum(
        weights[name] * _pulsar_time_cost(floor_factor[name], 1.0)
        for name in remaining_names
    )

    sum_w_upgraded = sum(weights[n] for n in upgraded_names)
    leftover = budget_total - cost_remaining

    MIN_CADENCE_FACTOR = 0.02  # matches conserve_telescope_time's floor
    max_cap = cfg.get('max_cadence_cap', None)  # optional safety cap

    if sum_w_upgraded <= 0:
        cadence_best = 1.0
        print(f'  🛑 max_cadence [{scenario_label}]: no upgraded pulsars with '
              f'nonzero weight among best_psrs={sorted(upgraded_names)}')
    else:
        cadence_best = leftover / sum_w_upgraded
        if cadence_best < MIN_CADENCE_FACTOR:
            print(f'  🛑 max_cadence [{scenario_label}]: holding the other '
                  f'{len(remaining_names)} pulsars at their '
                  f'{min_cadence_days:.0f}-day floor already costs more than '
                  f'the whole array budget — clamping upgraded cadence_factor '
                  f'to {MIN_CADENCE_FACTOR} instead of {cadence_best:.4f}. '
                  f'This result should not be trusted physically.')
            cadence_best = MIN_CADENCE_FACTOR
        elif max_cap is not None and cadence_best > max_cap:
            cadence_best = max_cap

    print(f'  ⏱ max_cadence [{scenario_label}]: {len(all_names)} pulsars '
          f'total, {len(upgraded_names)} upgraded (sum_w={sum_w_upgraded:.0f}s); '
          f'{len(remaining_names)} held at ≥{min_cadence_days:.0f}d floor '
          f'(cost={cost_remaining:.0f}s of {budget_total:.0f}s) → '
          f'leftover={leftover:.0f}s → upgraded cadence_factor={cadence_best:.3f}')

    return {
        'cadence_best': cadence_best, 'floor_factor': floor_factor,
        'budget_total': budget_total, 'cost_remaining': cost_remaining,
        'leftover': leftover, 'sum_w_upgraded': sum_w_upgraded,
    }


def _get_max_cadence_budget(cfg: dict, scenario_label: str, best_psrs,
                             par_dir: str, tim_dir: str,
                             min_cadence_days: float,
                             dwell_flag_key: str = 'tobs') -> dict:
    """Cached wrapper — computed once per scenario, not once per pulsar."""
    key = (id(cfg), tim_dir, min_cadence_days, dwell_flag_key)
    if key not in _MAX_CADENCE_BUDGET_CACHE:
        _MAX_CADENCE_BUDGET_CACHE[key] = _compute_max_cadence_budget(
            cfg, scenario_label, best_psrs, par_dir, tim_dir,
            min_cadence_days, dwell_flag_key=dwell_flag_key)
    return _MAX_CADENCE_BUDGET_CACHE[key]



def _load_cgw_cadence_weights(
    configuration=None,
    weight_dir=None,
):
    """
    Load and validate the fixed median CGW contribution weights for the
    active population configuration.
    """
    resolved_configuration = _resolve_population_configuration(
        configuration
    )

    directory = Path(weight_dir) if weight_dir else CGW_CADENCE_WEIGHT_DIR
    path = directory / f"{resolved_configuration}.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"No CGW cadence-weight file exists for population "
            f"configuration {resolved_configuration!r}: {path}. "
            "Run the population-analysis calibration cell first."
        )

    cache_key = (
        str(path.resolve()),
        path.stat().st_mtime_ns,
    )

    if cache_key in _CGW_CADENCE_WEIGHT_CACHE:
        return (
            resolved_configuration,
            _CGW_CADENCE_WEIGHT_CACHE[cache_key].copy(),
        )

    with path.open() as fh:
        payload = json.load(fh)

    file_configuration = payload.get("configuration")
    if file_configuration is not None:
        file_configuration = _resolve_population_configuration(
            file_configuration
        )

        if file_configuration != resolved_configuration:
            raise ValueError(
                f"Weight file {path} is labelled "
                f"{file_configuration!r}, but the active population "
                f"configuration is {resolved_configuration!r}."
            )

    raw_weights = payload.get("weights")

    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError(
            f"Weight file {path} does not contain a non-empty "
            "'weights' mapping."
        )

    weights = {}

    for pulsar, value in raw_weights.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid weight for pulsar {pulsar!r} in {path}: "
                f"{value!r}"
            ) from exc

        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Weight for pulsar {pulsar!r} in {path} is invalid: "
                f"{value!r}"
            )

        weights[str(pulsar)] = value

    total = float(sum(weights.values()))

    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"CGW cadence weights in {path} have invalid total {total!r}."
        )

    weights = {
        pulsar: value / total
        for pulsar, value in weights.items()
    }

    _CGW_CADENCE_WEIGHT_CACHE.clear()
    _CGW_CADENCE_WEIGHT_CACHE[cache_key] = weights.copy()

    return resolved_configuration, weights

def _compute_cgw_weighted_cadence_budget(
    cfg,
    scenario_label,
    par_dir,
    tim_dir,
    population_configuration=None,
):
    """
    Allocate the baseline telescope-time budget according to fixed,
    configuration-level median CGW rho^2 contribution weights.

    Every pulsar receives its minimum cadence floor first. The remaining
    telescope time is allocated proportionally to the calibrated CGW
    weights.

    Pulsars absent from the weight file receive zero weight and therefore
    receive only their minimum cadence.
    """
    all_names = _list_base_pulsar_names(par_dir)

    if not all_names:
        raise ValueError(
            f"No pulsars found in par directory {par_dir}"
        )

    dwell_flag_key = cfg.get("dwell_flag_key", "tobs")

    baseline_cost = {
        name: float(_pulsar_mean_tobs(
            name,
            tim_dir,
            flag_key=dwell_flag_key,
        ))
        for name in all_names
    }

    if any(value <= 0.0 for value in baseline_cost.values()):
        bad = [
            name for name, value in baseline_cost.items()
            if value <= 0.0
        ]
        raise ValueError(
            f"Non-positive baseline telescope-time cost for pulsars: "
            f"{bad}"
        )

    total_budget = float(sum(baseline_cost.values()))

    min_cadence_days = float(cfg.get(
        "min_cadence_days",
        MIN_CADENCE_DAYS_DEFAULT,
    ))

    floor_factor = {
        name: float(_pulsar_min_cadence_factor(
            name,
            tim_dir,
            min_cadence_days,
        ))
        for name in all_names
    }

    floor_cost = {
        name: baseline_cost[name] * floor_factor[name]
        for name in all_names
    }

    minimum_cost = float(sum(floor_cost.values()))

    if minimum_cost > total_budget + 1e-8:
        raise ValueError(
            f"Minimum cadence floor for {scenario_label!r} costs "
            f"{minimum_cost:.6g}, exceeding the baseline telescope-time "
            f"budget {total_budget:.6g}."
        )

    configuration, calibrated_weights = _load_cgw_cadence_weights(
        configuration=population_configuration,
        weight_dir=cfg.get(
            "cgw_weight_dir",
            CGW_CADENCE_WEIGHT_DIR,
        ),
    )

    # Zero weight is intentional for pulsars absent from the calibration
    # file: they receive the minimum cadence only.
    weights = {
        name: max(
            float(calibrated_weights.get(name, 0.0)),
            0.0,
        )
        for name in all_names
    }

    weight_sum = float(sum(weights.values()))

    if weight_sum <= 0.0:
        raise ValueError(
            f"None of the pulsars in {par_dir} appear in the "
            f"{configuration!r} CGW weight file."
        )

    weights = {
        name: value / weight_sum
        for name, value in weights.items()
    }

    remaining_budget = total_budget - minimum_cost

    cadence_factor = {}

    for name in all_names:
        extra_cost = remaining_budget * weights[name]
        cadence_factor[name] = (
            floor_factor[name]
            + extra_cost / baseline_cost[name]
        )

    achieved_budget = float(sum(
        baseline_cost[name] * cadence_factor[name]
        for name in all_names
    ))

    relative_error = (
        achieved_budget - total_budget
    ) / max(total_budget, 1e-30)

    if abs(relative_error) > 1e-8:
        raise RuntimeError(
            f"CGW weighted cadence failed telescope-time conservation: "
            f"relative error={relative_error:.3e}"
        )

    if cfg.get("max_cadence_cap") is not None:
        cap = float(cfg["max_cadence_cap"])

        if cap <= 0.0:
            raise ValueError(
                f"max_cadence_cap must be positive, got {cap}"
            )

        over_cap = [
            name for name in all_names
            if cadence_factor[name] > cap
        ]

        if over_cap:
            raise ValueError(
                f"CGW weighted cadence exceeds max_cadence_cap={cap} "
                f"for pulsars {over_cap}. A cap requires a redistribution "
                "solver; refusing to silently violate telescope-time "
                "conservation."
            )

    print(
        f"  weighted_cadence [{scenario_label}]: "
        f"configuration={configuration}, "
        f"{len(all_names)} pulsars, "
        f"{len(calibrated_weights)} calibrated weights, "
        f"minimum cost={minimum_cost:.0f}s, "
        f"budget={total_budget:.0f}s, "
        f"relative error={relative_error:.3e}"
    )

    return {
        "configuration": configuration,
        "weights": weights,
        "floor_factor": floor_factor,
        "cadence_factor": cadence_factor,
        "baseline_cost": baseline_cost,
        "minimum_cost": minimum_cost,
        "budget_total": total_budget,
        "achieved_budget": achieved_budget,
        "relative_error": relative_error,
    }


CGW_CADENCE_WEIGHT_DIR = (
    PROJECT_ROOT / "config" / "cgw_cadence_weights"
)

_CGW_CADENCE_WEIGHT_CACHE = {}
_CGW_WEIGHTED_CADENCE_BUDGET_CACHE = {}

def _get_cgw_weighted_cadence_budget(
    cfg,
    scenario_label,
    par_dir,
    tim_dir,
    population_configuration=None,
):
    configuration = _resolve_population_configuration(
        population_configuration
        or cfg.get("population_configuration")
    )

    weight_dir = Path(cfg.get(
        "cgw_weight_dir",
        CGW_CADENCE_WEIGHT_DIR,
    ))

    weight_path = weight_dir / f"{configuration}.json"

    key = (
        id(cfg),
        str(Path(par_dir).resolve()),
        str(Path(tim_dir).resolve()),
        configuration,
        str(weight_path.resolve()),
        weight_path.stat().st_mtime_ns
        if weight_path.is_file()
        else None,
        float(cfg.get(
            "min_cadence_days",
            MIN_CADENCE_DAYS_DEFAULT,
        )),
        cfg.get("dwell_flag_key", "tobs"),
    )

    if key not in _CGW_WEIGHTED_CADENCE_BUDGET_CACHE:
        _CGW_WEIGHTED_CADENCE_BUDGET_CACHE[key] = (
            _compute_cgw_weighted_cadence_budget(
                cfg=cfg,
                scenario_label=scenario_label,
                par_dir=par_dir,
                tim_dir=tim_dir,
                population_configuration=configuration,
            )
        )

    return _CGW_WEIGHTED_CADENCE_BUDGET_CACHE[key]

def _resolve_pulsar_scenario_params(cfg: dict, psr_name: str,
                                     best_only: bool, best_psrs,
                                     par_dir: str = None,
                                     tim_dir: str = None,
                                     scenario_label: str = 'scenario',
                                     population_configuration=None) -> tuple:
    """
    Resolve (cadence_factor, toaerr_factor) for a given base pulsar name.

      - Explicit per_pulsar override -> used verbatim, bypasses everything else.
      - cfg['maximize_best_cadence']=True:
          * pulsar in best_psrs -> (cadence_best, 1.0), the single uniform
            factor that spends the whole leftover budget (see
            _compute_max_cadence_budget).
          * pulsar not in best_psrs -> (floor_factor, 1.0), the sparsest
            allowed cadence that still respects cfg['min_cadence_days'].
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
    
    if cfg.get("cgw_weighted_cadence", False):
        budget = _get_cgw_weighted_cadence_budget(
            cfg=cfg,
            scenario_label=scenario_label,
            par_dir=par_dir or PAR_DIR,
            tim_dir=tim_dir or TIM_DIR,
            population_configuration=population_configuration or cfg.get(
                "population_configuration"
            ),
        )

        return (
            budget["cadence_factor"].get(psr_name, 1.0),
            cfg.get("toaerr_factor", 1.0),
        )
    
    if cfg.get('maximize_best_cadence', False):
        min_cadence_days = cfg.get('min_cadence_days', MIN_CADENCE_DAYS_DEFAULT)
        budget = _get_max_cadence_budget(
            cfg, scenario_label, best_psrs,
            par_dir or PAR_DIR, tim_dir or TIM_DIR, min_cadence_days,
            dwell_flag_key=cfg.get('dwell_flag_key', 'tobs'))
        if psr_name in best_psrs:
            return budget['cadence_best'], 1.0
        return budget['floor_factor'].get(psr_name, 1.0), 1.0

    if best_only and psr_name not in best_psrs:
        if cfg.get('conserve_telescope_time', False):
            budget = _get_telescope_time_budget(
                cfg, scenario_label, best_only, best_psrs,
                par_dir or PAR_DIR, tim_dir or TIM_DIR,
                dwell_flag_key=cfg.get('dwell_flag_key', 'tobs'))
            return budget['cadence_nonbest'], 1.0
        return 1, 1.0

    return cfg.get('cadence_factor', 1), cfg.get('toaerr_factor', 1.0)


def _resolve_population_configuration(configuration=None):
    """
    Resolve the active population configuration.

    The preferred route is an explicit value passed from Stage 1/Stage 2.
    SMBHB_CONFIG is supported for shell/HPC workflows.

    Do not infer this from the pulsar scenario name: 'baseline',
    'baseline_forecast', and 'cgw_weighted_cadence' are observing scenarios,
    not population configurations.
    """
    value = (
        configuration
        or os.environ.get("SMBHB_CONFIG")
    )

    if value is None:
        raise ValueError(
            "The population configuration is unknown. Pass "
            "population_configuration='optimistic'/'realistic'/"
            "'pessimistic' to load_pulsars(), or set SMBHB_CONFIG."
        )

    value = str(value).strip().lower()

    aliases = {
        "intermediate": "realistic",
        "realistic": "realistic",
        "optimistic": "optimistic",
        "pessimistic": "pessimistic",
    }

    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"Unknown population configuration {value!r}. "
            f"Expected one of {sorted(aliases)}."
        ) from exc

def _scenario_params_cache_key(
    cadence_factor,
    toaerr_factor,
    extension_years,
    extension_fraction,
    extension_to_longest_span=False,
    global_extension_days=None,
):
    """
    Build a cache key for a scenario timing transformation.

    The actual global extension is included in the key so that timing files
    generated from a different data release or pulsar baseline are not reused.
    """
    if extension_to_longest_span:
        if global_extension_days is None:
            raise ValueError(
                "global_extension_days is required when "
                "extension_to_longest_span=True"
            )
        ext_key = f"longest-span-{float(global_extension_days):.4f}d"
    elif extension_years is not None:
        ext_key = f"{float(extension_years):.4f}y"
    else:
        ext_key = f"{float(extension_fraction):.4f}xspan"

    return (
        f"cad{float(cadence_factor):.2f}_"
        f"err{float(toaerr_factor):.2f}_"
        f"{ext_key}"
    )


def _longest_real_pulsar_span_days(
    tim_dir,
    par_dir,
    selected_names=None,
    skip_pulsars=None,
):
    """
    Return the longest real observing span among eligible pulsars.

    The span is measured in days from the actual real TOAs:

        max(real TOA) - min(real TOA)

    The returned value is a single global duration. It is not calculated
    separately for each pulsar.
    """
    selected_names = (
        set(selected_names)
        if selected_names is not None
        else None
    )
    skip_pulsars = set(skip_pulsars or ())

    suffixes = ("ao", "gbt", "vla", "fast")
    parfiles = sorted(
        filename
        for filename in os.listdir(par_dir)
        if filename.endswith(".par")
    )

    seen_names = set()
    spans_days = []

    for par_filename in parfiles:
        stem = par_filename.replace(".par", "").split("_")[0]
        base_name = stem

        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        if base_name in seen_names:
            continue
        seen_names.add(base_name)

        if base_name in skip_pulsars:
            continue

        if (
            selected_names is not None
            and base_name not in selected_names
        ):
            continue

        tim_paths = [
            tim_path
            for tim_path in _find_tim_files(tim_dir, base_name)
            if tim_has_toas(tim_path)
        ]

        for tim_path in tim_paths:
            try:
                _, records = _parse_tim_lines(tim_path)
            except (OSError, ValueError):
                continue

            if len(records) < 2:
                continue

            toas = np.asarray(
                [record["toa"] for record in records],
                dtype=float,
            )

            if not np.all(np.isfinite(toas)):
                continue

            span_days = float(np.max(toas) - np.min(toas))

            if span_days > 0.0:
                spans_days.append(span_days)

    if not spans_days:
        raise RuntimeError(
            "Could not determine a valid real observing span from the "
            "eligible pulsar timing files."
        )

    return max(spans_days)

def _resolve_global_extension_days(
    cfg,
    tim_dir,
    par_dir,
    selected_names=None,
    skip_pulsars=None,
):
    """
    Resolve the single extension duration used by every pulsar in a scenario.

    Precedence:
      1. extension_to_longest_span=True
      2. extension_years
      3. extension_fraction, which remains per-pulsar behavior
    """
    extension_to_longest_span = bool(
        cfg.get("extension_to_longest_span", False)
    )

    extension_years = cfg.get("extension_years", None)
    extension_fraction = float(
        cfg.get("extension_fraction", 0.0)
    )

    if extension_to_longest_span:
        return _longest_real_pulsar_span_days(
            tim_dir=tim_dir,
            par_dir=par_dir,
            selected_names=selected_names,
            skip_pulsars=skip_pulsars,
        )

    if extension_years is not None:
        extension_years = float(extension_years)

        if extension_years < 0.0:
            raise ValueError(
                f"extension_years must be non-negative, got {extension_years}"
            )

        return extension_years * 365.25

    # None means _write_scenario_tim() may use extension_fraction and
    # calculate a separate extension from each pulsar's own span.
    if extension_fraction < 0.0:
        raise ValueError(
            "extension_fraction must be non-negative"
        )

    return None


def _pulsar_needs_scenario_tim(
    cfg,
    cadence_factor,
    toaerr_factor,
):
    """
    Return True if this scenario requires a generated timing file.

    A generated file is needed when cadence, TOA precision, or forecast
    duration changes. A longest-span extension is global, but it still
    applies to every pulsar and therefore must force scenario generation.
    """
    modified_rate = not (
        cadence_factor == 1
        and toaerr_factor == 1.0
    )

    has_extension = (
        cfg.get("extension_years") is not None
        or bool(cfg.get("extension_fraction", 0.0))
        or bool(cfg.get("extension_to_longest_span", False))
    )

    return modified_rate or has_extension


def _group_into_epochs(toa_records):
    """
    Group real TOA records into observing epochs by source file path
    (the first whitespace-separated field of each line).
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
      1. Every real TOA line, reproduced exactly.
      2. A forecast segment of synthetic FULL EPOCHS strictly after the
         last real epoch: spaced at (mean real epoch cadence /
         cadence_factor). Each future epoch replicates every channel/
         sub-band line of a resampled real epoch, shifted onto the new
         epoch's TOA, with each line's error scaled by toaerr_factor.

    Both the .tim and .meta.json are written atomically (temp file +
    os.replace).
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

    out_dir = os.path.dirname(outpath) or '.'
    os.makedirs(out_dir, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=out_dir, prefix=f'.{os.path.basename(outpath)}.', suffix='.tmp')
    n_future_written = 0
    try:
        with os.fdopen(tmp_fd, 'w') as fout:
            for line in header_lines:
                fout.write(line + '\n')

            for rec in toa_records:
                fout.write(rec['line'] + '\n')

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

        os.replace(tmp_path, outpath)
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

    meta_path = outpath + '.meta.json'
    meta_tmp_fd, meta_tmp_path = tempfile.mkstemp(
        dir=out_dir, prefix=f'.{os.path.basename(meta_path)}.', suffix='.tmp')
    try:
        with os.fdopen(meta_tmp_fd, 'w') as fh:
            json.dump(meta, fh)
        os.replace(meta_tmp_path, meta_path)
    except Exception:
        if os.path.exists(meta_tmp_path):
            os.unlink(meta_tmp_path)
        raise

    return meta


def _scenario_tim_is_valid(
    scen_tim,
    orig_tim,
    cadence_factor,
    toaerr_factor,
    extension_days=None,
    extension_fraction=1.0,
):
    """
    Check whether a cached scenario timing file matches the requested
    transformation.

    If extension_days is supplied, it is treated as a global duration and
    compared directly. If it is None, extension_fraction is evaluated from
    the individual original timing file.
    """
    if not os.path.exists(scen_tim):
        return False

    meta_path = scen_tim + ".meta.json"
    if not os.path.exists(meta_path):
        return False

    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (OSError, ValueError, TypeError):
        return False

    if extension_days is None:
        try:
            _, toa_records = _parse_tim_lines(orig_tim)
        except (OSError, ValueError):
            return False

        if not toa_records:
            return False

        orig_toas = np.asarray(
            [record["toa"] for record in toa_records],
            dtype=float,
        )

        if not np.all(np.isfinite(orig_toas)):
            return False

        real_span = float(
            np.max(orig_toas) - np.min(orig_toas)
        )
        extension_days = float(extension_fraction) * real_span

    n_scen = _tim_nobs(scen_tim)

    if n_scen <= 0:
        return False

    return (
        abs(
            float(meta.get("cadence_factor", -1))
            - float(cadence_factor)
        ) < 1e-9
        and abs(
            float(meta.get("toaerr_factor", -1))
            - float(toaerr_factor)
        ) < 1e-9
        and abs(
            float(meta.get("extension_days", -1))
            - float(extension_days)
        ) < 1.0
        and n_scen
        == int(meta.get("n_real", -1))
        + int(meta.get("n_future", -1))
    )


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

    cfg = _scenarios[scenario]
    best_only = cfg.get("best_only", True)
    best_psrs = cfg.get("best_psrs", BEST_PSRS)
    per_pulsar_cfg = cfg.get("per_pulsar", {})

    extension_years = cfg.get("extension_years", None)
    extension_fraction = float(
        cfg.get("extension_fraction", 0.0)
    )
    extension_to_longest_span = bool(
        cfg.get("extension_to_longest_span", False)
    )

    # load_single_pulsar() is normally used for one pulsar at a time, so resolve
    # the shared extension from the complete data set rather than from this one
    # pulsar's own TOAs.
    global_extension_days = _resolve_global_extension_days(
        cfg=cfg,
        tim_dir=_tim_dir,
        par_dir=_par_dir,
        selected_names=None,
        skip_pulsars=_skip,
    )

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
        maxobs = _bounded_maxobs(tim_path)

        if modified:
            cache_key = _scenario_params_cache_key(
                cadence_factor=cadence_factor,
                toaerr_factor=toaerr_factor,
                extension_years=extension_years,
                extension_fraction=extension_fraction,
                extension_to_longest_span=extension_to_longest_span,
                global_extension_days=global_extension_days,
            )
            scen_dir = os.path.join(scenario_tim_dir, '_by_params', cache_key)
            os.makedirs(scen_dir, exist_ok=True)
            scen_tim = os.path.join(scen_dir, f'{load_name}.tim')

            if (
                not os.path.exists(scen_tim)
                or not _scenario_tim_is_valid(
                    scen_tim,
                    tim_path,
                    cadence_factor,
                    toaerr_factor,
                    extension_days=global_extension_days,
                    extension_fraction=(
                        0.0
                        if global_extension_days is not None
                        else extension_fraction
                    ),
                )
            ):
                if verbose:
                    print(f'  ✎ Writing scenario tim for {load_name}...',
                          end=' ', flush=True)
                try:
                    psr_tmp = T.tempopulsar(
                        parfile=par_path, timfile=tim_path,
                        maxobs=maxobs, dofit=False)
                    meta = _write_scenario_tim(
                        tim_path,
                        load_name,
                        scen_tim,
                        cadence_factor=cadence_factor,
                        toaerr_factor=toaerr_factor,
                        extension_days=global_extension_days,
                        extension_fraction=(
                            0.0
                            if global_extension_days is not None
                            else extension_fraction
                        ),
                        scenario_label=cache_key,
                    )
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


def _resolve_global_extension_days(
    cfg,
    tim_dir,
    par_dir,
    selected_names=None,
    skip_pulsars=None,
):
    """
    Resolve the forecast extension duration.

    Returns:
      - one global duration in days when extension_to_longest_span=True;
      - one fixed duration in days when extension_years is used;
      - None when extension_fraction should remain per-pulsar.
    """
    extension_to_longest_span = bool(
        cfg.get("extension_to_longest_span", False)
    )
    extension_years = cfg.get("extension_years", None)
    extension_fraction = float(
        cfg.get("extension_fraction", 0.0)
    )

    if extension_to_longest_span:
        return _longest_real_pulsar_span_days(
            tim_dir=tim_dir,
            par_dir=par_dir,
            selected_names=selected_names,
            skip_pulsars=skip_pulsars,
        )

    if extension_years is not None:
        extension_years = float(extension_years)

        if extension_years < 0.0:
            raise ValueError(
                "extension_years must be non-negative, got "
                f"{extension_years}"
            )

        return extension_years * 365.25

    if extension_fraction < 0.0:
        raise ValueError(
            "extension_fraction must be non-negative, got "
            f"{extension_fraction}"
        )

    # None means that _write_scenario_tim() will resolve the extension
    # separately from each pulsar's real observing span.
    return None


def load_pulsars(
    verbose=True,
    scenario="baseline",
    scenarios=None,
    scenario_tim_dir="scenario_tims",
    par_dir=None,
    tim_dir=None,
    skip_pulsars=None,
    use_cache=None,
    cache_path=None,
    population_configuration=None,
):
    """
    Load NANOGrav pulsars as libstempo tempopulsar objects.

    For scenarios using extension_to_longest_span=True, one shared forecast
    duration is calculated from the longest eligible real pulsar span. That
    same duration is appended to every pulsar.

    Therefore, if the longest real pulsar has span T_max, its final span is:

        T_max + T_max = 2 * T_max

    Shorter pulsars receive the same number of future days, rather than an
    extension calculated from each pulsar's own baseline.

    For baseline scenarios, the existing pickle-cache behavior is preserved.
    """
    _par_dir = par_dir or PAR_DIR
    _tim_dir = tim_dir or TIM_DIR
    _skip = (
        skip_pulsars
        if skip_pulsars is not None
        else SKIP_PULSARS
    )
    _use_cache = (
        use_cache
        if use_cache is not None
        else USE_PULSAR_CACHE
    )
    _cache_path = cache_path or PULSAR_CACHE
    _scenarios = scenarios or SCENARIOS
    
    resolved_population_configuration = (
        _resolve_population_configuration(
            population_configuration
        )
        if cfg.get("cgw_weighted_cadence", False)
        else None
    )
    if scenario not in _scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {list(_scenarios.keys())}"
        )

    cfg = _scenarios[scenario]

    default_cadence = cfg.get("cadence_factor", 1)
    default_toaerr = cfg.get("toaerr_factor", 1.0)
    best_only = cfg.get("best_only", True)
    best_psrs = cfg.get("best_psrs", BEST_PSRS)
    per_pulsar_cfg = cfg.get("per_pulsar", {})

    extension_years = cfg.get("extension_years", None)
    extension_fraction = float(
        cfg.get("extension_fraction", 0.0)
    )
    extension_to_longest_span = bool(
        cfg.get("extension_to_longest_span", False)
    )

    # A scenario is a true no-op baseline only when it does not modify
    # cadence, precision, per-pulsar settings, or observing duration.
    is_scenario_baseline = (
        default_cadence == 1
        and default_toaerr == 1.0
        and not per_pulsar_cfg
        and not cfg.get("maximize_best_cadence", False)
        and extension_years is None
        and extension_fraction == 0.0
        and not extension_to_longest_span
    )

    if verbose:
        print("=" * 70)
        print(
            "LOADING NANOGRAV PULSARS "
            f"(libstempo)  —  scenario: {scenario}"
        )

        if not is_scenario_baseline:
            print(
                f"  default: cadence×{default_cadence} "
                f"err×{default_toaerr} "
                f"best_only={best_only}"
            )

            if per_pulsar_cfg:
                print(
                    "  per-pulsar overrides: "
                    f"{list(per_pulsar_cfg.keys())}"
                )

            if cfg.get("maximize_best_cadence", False):
                print(
                    "  maximize_best_cadence=True "
                    f"min_cadence_days={cfg.get(
                        'min_cadence_days',
                        MIN_CADENCE_DAYS_DEFAULT,
                    )} "
                    f"best_psrs={list(best_psrs)}"
                )

            if extension_to_longest_span:
                print(
                    "  forecast extension: one shared duration equal "
                    "to the longest eligible real pulsar span"
                )
            elif extension_years is not None:
                print(
                    f"  forecast extension: {float(extension_years):.4f} years"
                )
            elif extension_fraction != 0.0:
                print(
                    "  forecast extension: "
                    f"{extension_fraction:.4f} × each pulsar span"
                )

        print("=" * 70)

    # ------------------------------------------------------------------
    # Baseline path: preserve the original pickle cache behavior.
    # ------------------------------------------------------------------
    if is_scenario_baseline and _use_cache:
        if _cache_path is None:
            raise ValueError(
                "use_cache=True but no pulsar cache path was provided."
            )

        if os.path.exists(_cache_path):
            if verbose:
                print(f"\n📦 Loading from cache: {_cache_path}")

            try:
                with open(_cache_path, "rb") as fh:
                    psrs = pickle.load(fh)

                if verbose:
                    print(f"✓ Loaded {len(psrs)} pulsars from cache\n")

                return psrs

            except (OSError, EOFError, pickle.PickleError) as exc:
                if verbose:
                    print(f"⚠ Cache load failed: {exc}")

    # ------------------------------------------------------------------
    # Build sorted unique base-pulsar names.
    # ------------------------------------------------------------------
    suffixes = ("ao", "gbt", "vla", "fast")

    parfiles = sorted(
        filename
        for filename in os.listdir(_par_dir)
        if filename.endswith(".par")
    )

    seen_names = set()
    base_names = []

    for par_filename in parfiles:
        stem = par_filename.replace(".par", "").split("_")[0]
        base_name = stem

        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        if base_name not in seen_names:
            seen_names.add(base_name)
            base_names.append(base_name)

    if verbose:
        print(
            f"Found {len(parfiles)} .par files "
            f"→ {len(base_names)} unique pulsars"
        )

    # ------------------------------------------------------------------
    # Resolve the one shared extension after the available pulsars are known.
    # This must not be done at module import time.
    # ------------------------------------------------------------------
    global_extension_days = _resolve_global_extension_days(
        cfg=cfg,
        tim_dir=_tim_dir,
        par_dir=_par_dir,
        selected_names=set(base_names),
        skip_pulsars=_skip,
    )

    if global_extension_days is not None and verbose:
        print(
            "  Shared forecast extension: "
            f"{global_extension_days / 365.25:.4f} years "
            "(based on the longest eligible real pulsar span)"
        )

    psrs = []
    failed_pulsars = []

    for psr_name in base_names:
        if psr_name in _skip:
            if verbose:
                print(
                    f"⚠ Skipping {psr_name} "
                    "(in SKIP_PULSARS)"
                )
            continue

        par_map = _find_par_files(_par_dir, psr_name)
        tim_paths = [
            tim_path
            for tim_path in _find_tim_files(_tim_dir, psr_name)
            if tim_has_toas(tim_path)
        ]

        if not par_map:
            if verbose:
                print(f"⚠ No par file found for {psr_name}")
            failed_pulsars.append(psr_name)
            continue

        if not tim_paths:
            if verbose:
                print(f"⚠ No valid tim for {psr_name}")
            failed_pulsars.append(psr_name)
            continue

        cadence_factor, toaerr_factor = _resolve_pulsar_scenario_params(
            cfg,
            psr_name,
            best_only,
            best_psrs,
            par_dir=_par_dir,
            tim_dir=_tim_dir,
            scenario_label=scenario,
            population_configuration=(
                resolved_population_configuration
            ),
        )

        modified = _pulsar_needs_scenario_tim(
            cfg,
            cadence_factor,
            toaerr_factor,
        )

        cache_key = _scenario_params_cache_key(
            cadence_factor=cadence_factor,
            toaerr_factor=toaerr_factor,
            extension_years=extension_years,
            extension_fraction=extension_fraction,
            extension_to_longest_span=extension_to_longest_span,
            global_extension_days=global_extension_days,
        )

        scen_dir = os.path.join(
            scenario_tim_dir,
            "_by_params",
            cache_key,
        )

        for tim_path in tim_paths:
            tim_stem = (
                os.path.basename(tim_path)
                .replace(".tim", "")
                .split("_")[0]
            )
            variant = tim_stem[len(psr_name):]
            load_name = f"{psr_name}{variant}"

            par_path = (
                par_map.get(variant)
                or par_map.get("")
            )

            if par_path is None:
                if verbose:
                    print(
                        f"⚠ No par for variant {load_name}, skipping"
                    )
                failed_pulsars.append(load_name)
                continue

            effective_tim = tim_path
            maxobs = _bounded_maxobs(tim_path)

            if modified:
                os.makedirs(scen_dir, exist_ok=True)
                scen_tim = os.path.join(
                    scen_dir,
                    f"{load_name}.tim",
                )

                validation_extension_fraction = (
                    0.0
                    if global_extension_days is not None
                    else extension_fraction
                )

                needs_rebuild = (
                    not os.path.exists(scen_tim)
                    or not _scenario_tim_is_valid(
                        scen_tim,
                        tim_path,
                        cadence_factor,
                        toaerr_factor,
                        extension_days=global_extension_days,
                        extension_fraction=(
                            validation_extension_fraction
                        ),
                    )
                )

                if needs_rebuild:
                    if verbose:
                        print(
                            f"  ✎ Writing scenario tim for {load_name} "
                            f"[{cache_key}]...",
                            end=" ",
                            flush=True,
                        )

                    try:
                        # This construction checks that the original timing
                        # file is usable before generating the scenario file.
                        psr_tmp = T.tempopulsar(
                            parfile=par_path,
                            timfile=tim_path,
                            maxobs=maxobs,
                            dofit=False,
                        )

                        meta = _write_scenario_tim(
                            tim_path,
                            load_name,
                            scen_tim,
                            cadence_factor=cadence_factor,
                            toaerr_factor=toaerr_factor,
                            extension_days=global_extension_days,
                            extension_fraction=(
                                validation_extension_fraction
                            ),
                            scenario_label=cache_key,
                        )

                        if verbose:
                            print(
                                f"✓ ({meta['n_real']} real + "
                                f"{meta['n_future']} forecast = "
                                f"{meta['n_real'] + meta['n_future']} TOAs)"
                            )

                        del psr_tmp

                    except (OSError, ValueError, RuntimeError) as exc:
                        if verbose:
                            print(f"✗ failed: {exc}")
                        failed_pulsars.append(load_name)
                        continue

                elif verbose:
                    print(
                        f"  ✓ {load_name:20s} "
                        f"[{cache_key}] scenario tim exists, reusing"
                    )

                effective_tim = scen_tim

            elif verbose and (
                cfg.get("cadence_factor", 1) != 1
                or cfg.get("toaerr_factor", 1.0) != 1.0
                or per_pulsar_cfg
            ):
                print(
                    f"  ✓ {load_name:20s} not upgraded "
                    "(best_only) — original tim"
                )

            try:
                psr = T.tempopulsar(
                    parfile=par_path,
                    timfile=effective_tim,
                    maxobs=maxobs,
                    dofit=False,
                )

                if psr.nobs == 0:
                    if verbose:
                        print(f"⚠ Zero TOAs for {load_name}")
                    failed_pulsars.append(load_name)
                    continue

                psrs.append(psr)

                if verbose:
                    print(
                        f"✓ Loaded {psr.name} "
                        f"({psr.nobs} TOAs) "
                        f"[tim={tim_stem}]"
                    )

            except (OSError, ValueError, RuntimeError) as exc:
                if verbose:
                    print(f"✗ Failed {load_name}: {exc}")
                failed_pulsars.append(load_name)

    # ------------------------------------------------------------------
    # Cache true baseline only.
    # ------------------------------------------------------------------
    if is_scenario_baseline and _use_cache and psrs:
        try:
            with open(_cache_path, "wb") as fh:
                pickle.dump(
                    psrs,
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            if verbose:
                print(f"\n💾 Saved cache: {_cache_path}")

        except (OSError, pickle.PickleError) as exc:
            if verbose:
                print(f"⚠ Could not save cache: {exc}")

    if verbose:
        print(
            f"\n✓ Loaded {len(psrs)} pulsars "
            f"[scenario={scenario}]"
        )
        print(
            f"✗ Failed: {len(failed_pulsars)} "
            f"— {failed_pulsars}"
        )

    return psrs


# ──────────────────────────────────────────────────────────────────────────
# New: max-cadence scenarios for top5 / top10 / top15 / top20 / top40
# ──────────────────────────────────────────────────────────────────────────

def _make_max_cadence_scenario(best_psrs, min_cadence_days=MIN_CADENCE_DAYS_DEFAULT,
                                extension_years=4.46):
    """
    Scenario factory: max out cadence on `best_psrs`, subject to holding
    every other pulsar's forecast spacing at, or just above,
    min_cadence_days. Uses the array-wide real telescope-time budget
    (weighted by each pulsar's own mean -tobs) as the conserved resource
    — see _compute_max_cadence_budget for the solve.
    """
    return dict(
        maximize_best_cadence = True,
        min_cadence_days      = min_cadence_days,
        best_only             = True,
        best_psrs              = tuple(best_psrs),
        toaerr_factor          = 1.0,
        extension_years       = extension_years,
    )


SCENARIOS.update({
    'max_cadence_top10': _make_max_cadence_scenario(top10),
    'max_cadence_top20': _make_max_cadence_scenario(top20),
    'max_cadence_top30': _make_max_cadence_scenario(top30),
    'max_cadence_top40': _make_max_cadence_scenario(top40),
})