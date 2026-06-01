import os
import pickle
import json
import numpy as np
import gc
from copy import deepcopy
from enterprise.pulsar import Pulsar
import sys
from collections import defaultdict
from config import PAR_DIR, TIM_DIR, USE_PULSAR_CACHE, NANOGRAV_PULSAR_CACHE, NOISEFILE
import libstempo as T

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
# def load_pulsars(verbose=True):
#     """Load NANOGrav pulsars as libstempo tempopulsar objects."""
#     if verbose:
#         print("="*70)
#         print("LOADING NANOGRAV PULSARS (libstempo)")
#         print("="*70)

#     # Try cache
#     if USE_PULSAR_CACHE and os.path.exists(NANOGRAV_PULSAR_CACHE):
#         if verbose:
#             print(f"\n📦 Loading from cache: {NANOGRAV_PULSAR_CACHE}")
#         try:
#             with open(NANOGRAV_PULSAR_CACHE, 'rb') as f:
#                 psrs = pickle.load(f)
#             if verbose:
#                 print(f"✓ Loaded {len(psrs)} pulsars from cache\n")
#             return psrs
#         except Exception as e:
#             if verbose:
#                 print(f"⚠ Cache load failed: {e}")

#     parfiles = sorted([f for f in os.listdir(PAR_DIR) if f.endswith(".par")])

#     if verbose:
#         print(f"[DEBUG] Found {len(parfiles)} .par files")

#     psrs = []
#     failed_pulsars = []

#     for par in parfiles:
#         # Skip known problematic pulsars
#         psr_name = par.split('_')[0]
#         if psr_name in SKIP_PULSARS:
#             if verbose:
#                 print(f"⚠ Skipping {par} (known hang)")
#             continue

#         tim = par.replace(".par", ".tim")
#         par_path = os.path.join(PAR_DIR, par)
#         tim_path = os.path.join(TIM_DIR, tim)

#         if not os.path.exists(tim_path) or not tim_has_toas(tim_path):
#             if verbose:
#                 print(f"⚠ No valid tim for {par}")
#             failed_pulsars.append(par)
#             continue

#         try:
#             psr = T.tempopulsar(
#                 parfile=par_path,
#                 timfile=tim_path,
#                 maxobs=60000,
#                 dofit=False,   # skip internal fit to avoid hangs
#             )
#             if psr.nobs == 0:
#                 if verbose:
#                     print(f"⚠ Zero TOAs for {par}")
#                 failed_pulsars.append(par)
#                 continue

#             psrs.append(psr)
#             if verbose:
#                 print(f"✓ Loaded {psr.name} ({psr.nobs} TOAs)")

#         except Exception as e:
#             if verbose:
#                 print(f"✗ Failed {par}: {e}")
#             failed_pulsars.append(par)
#             continue

#     # Save cache
#     if USE_PULSAR_CACHE and len(psrs) > 0:
#         try:
#             with open(NANOGRAV_PULSAR_CACHE, 'wb') as f:
#                 pickle.dump(psrs, f, protocol=pickle.HIGHEST_PROTOCOL)
#             if verbose:
#                 print(f"\n💾 Saved cache: {NANOGRAV_PULSAR_CACHE}")
#         except Exception as e:
#             if verbose:
#                 print(f"⚠ Could not save cache: {e}")

#     if verbose:
#         print(f"\n✓ Loaded {len(psrs)} pulsars")
#         print(f"✗ Failed: {len(failed_pulsars)} — {failed_pulsars}")

#     return psrs



def filter_pulsars_15yr(psrs, min_baseline_years=3.0, verbose=True):
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


BEST_PSRS = (
    'J0437-4715', 'J1909-3744', 'J1713+0747',
    'J0030+0451', 'J1744-1134',
)
 
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
    '5x_cadence': dict(
        cadence_factor = 5,
        toaerr_factor  = 1.0,
        best_only      = True,
    ),
    '4x_precision': dict(
        cadence_factor = 1,
        toaerr_factor  = 0.25,
        best_only      = True,
    ),
    '5x_cad_4x_prec': dict(
        cadence_factor = 5,
        toaerr_factor  = 0.25,
        best_only      = True,
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
 
 
def _write_scenario_tim(psr, outpath, cadence_factor=1, toaerr_factor=1.0):
    """
    Write a tempo2 FORMAT 1 .tim file with interleaved TOAs and/or scaled
    errors.  Observing frequency and backend flag are preserved per-TOA.
    """
    toas  = psr.stoas.copy()      # MJD barycentric
    errs  = psr.toaerrs.copy()   # microseconds
    flags = psr.flagvals('f').copy()
    try:
        freqs = psr.ssbfreqs().copy()
    except Exception:
        freqs = np.full(len(toas), 1400.0)
 
    if cadence_factor > 1:
        toas, errs, flags, freqs = _interleave_toas(
            toas, errs, flags, freqs, cadence_factor)
 
    errs = errs * toaerr_factor
 
    with open(outpath, 'w') as f:
        f.write('FORMAT 1\n')
        for t, e, fl, fr in zip(toas, errs, flags, freqs):
            f.write(f'{psr.name}  {fr:.4f}  {t:.15f}  {e:.4f}  @  -f {fl}\n')
 
 
# ---------------------------------------------------------------------------
# Extended loader
# ---------------------------------------------------------------------------
 
def load_pulsars(
    verbose          = True,
    scenario         = 'baseline',
    scenarios        = None,            # pass None to use module-level SCENARIOS
    scenario_tim_dir = 'scenario_tims',
    par_dir          = None,            # falls back to module-level PAR_DIR
    tim_dir          = None,            # falls back to module-level TIM_DIR
    skip_pulsars     = None,            # falls back to module-level SKIP_PULSARS
    use_cache        = None,            # falls back to module-level USE_PULSAR_CACHE
    cache_path       = None,            # falls back to module-level NANOGRAV_PULSAR_CACHE
):
    """
    Load NANOGrav pulsars as libstempo tempopulsar objects.
 
    Drop-in replacement for the original load_pulsars().  When scenario is
    'baseline' (default) behaviour is identical including the pickle cache.
    For other scenarios the function transparently writes / reuses augmented
    .tim files and returns the modified pulsars, skipping the cache.
 
    Parameters
    ----------
    scenario : str
        Key into `scenarios`.  Default 'baseline'.
    scenarios : dict or None
        Scenario definitions (see module-level SCENARIOS for format).
        If None, the module-level SCENARIOS dict is used.
    scenario_tim_dir : str
        Root directory for augmented .tim files; one sub-dir per scenario.
    par_dir, tim_dir : str or None
        Override module-level PAR_DIR / TIM_DIR.
    skip_pulsars : set or None
        Override module-level SKIP_PULSARS.
    use_cache : bool or None
        Override module-level USE_PULSAR_CACHE.
    cache_path : str or None
        Override module-level NANOGRAV_PULSAR_CACHE.
 
    Returns
    -------
    list of libstempo.tempopulsar
    """
    # resolve module-level globals as fallbacks
    _par_dir      = par_dir      or PAR_DIR
    _tim_dir      = tim_dir      or TIM_DIR
    _skip         = skip_pulsars if skip_pulsars is not None else SKIP_PULSARS
    _use_cache    = use_cache    if use_cache    is not None else USE_PULSAR_CACHE
    _cache_path   = cache_path   or NANOGRAV_PULSAR_CACHE
    _scenarios    = scenarios    if scenarios    is not None else SCENARIOS
 
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
    # Non-baseline: write / reuse scenario .tim files, then load
    # ------------------------------------------------------------------
    scen_dir  = os.path.join(scenario_tim_dir, scenario)
    parfiles  = sorted([f for f in os.listdir(_par_dir) if f.endswith('.par')])
 
    if verbose:
        print(f'Found {len(parfiles)} .par files in {_par_dir}')
 
    psrs           = []
    failed_pulsars = []
 
    for par in parfiles:
        psr_name = par.split('_')[0]
 
        if psr_name in _skip:
            if verbose:
                print(f'⚠ Skipping {par} (in SKIP_PULSARS)')
            continue
 
        tim      = par.replace('.par', '.tim')
        par_path = os.path.join(_par_dir, par)
        tim_path = os.path.join(_tim_dir, tim)
 
        if not os.path.exists(tim_path) or not tim_has_toas(tim_path):
            if verbose:
                print(f'⚠ No valid tim for {par}')
            failed_pulsars.append(par)
            continue
 
        # ---- determine which .tim to load --------------------------------
        if not is_baseline:
            is_best  = any(b in psr_name for b in best_psrs)
            modified = (not best_only) or is_best
 
            if modified:
                os.makedirs(scen_dir, exist_ok=True)
                scen_tim = os.path.join(scen_dir, f'{psr_name}.tim')
 
                if not os.path.exists(scen_tim):
                    if verbose:
                        print(f'  ✎ Writing scenario tim for {psr_name}...',
                              end=' ', flush=True)
                    try:
                        psr_tmp = T.tempopulsar(
                            parfile=par_path, timfile=tim_path,
                            maxobs=60000, dofit=False)
                        _write_scenario_tim(
                            psr_tmp, scen_tim,
                            cadence_factor=cadence_factor,
                            toaerr_factor=toaerr_factor)
                        n_new = len(psr_tmp.stoas) * cadence_factor
                        if verbose:
                            print(f'✓ ({psr_tmp.nobs} → ~{n_new} TOAs)')
                        del psr_tmp
                    except Exception as e:
                        if verbose:
                            print(f'✗ failed: {e}')
                        failed_pulsars.append(par)
                        continue
                else:
                    if verbose:
                        print(f'  ✓ {psr_name:20s} scenario tim exists, reusing')
 
                tim_path = scen_tim
            else:
                if verbose:
                    print(f'  ✓ {psr_name:20s} not in best_psrs — original tim')
 
        # ---- load --------------------------------------------------------
        try:
            psr = T.tempopulsar(
                parfile=par_path,
                timfile=tim_path,
                maxobs=60000,
                dofit=False,
            )
            if psr.nobs == 0:
                if verbose:
                    print(f'⚠ Zero TOAs for {par}')
                failed_pulsars.append(par)
                continue
 
            psrs.append(psr)
            if verbose:
                print(f'✓ Loaded {psr.name} ({psr.nobs} TOAs)')
 
        except Exception as e:
            if verbose:
                print(f'✗ Failed {par}: {e}')
            failed_pulsars.append(par)
 
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