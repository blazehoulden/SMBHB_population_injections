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
def load_pulsars(verbose=True):
    """Load NANOGrav pulsars as libstempo tempopulsar objects."""
    if verbose:
        print("="*70)
        print("LOADING NANOGRAV PULSARS (libstempo)")
        print("="*70)

    # Try cache
    if USE_PULSAR_CACHE and os.path.exists(NANOGRAV_PULSAR_CACHE):
        if verbose:
            print(f"\n📦 Loading from cache: {NANOGRAV_PULSAR_CACHE}")
        try:
            with open(NANOGRAV_PULSAR_CACHE, 'rb') as f:
                psrs = pickle.load(f)
            if verbose:
                print(f"✓ Loaded {len(psrs)} pulsars from cache\n")
            return psrs
        except Exception as e:
            if verbose:
                print(f"⚠ Cache load failed: {e}")

    parfiles = sorted([f for f in os.listdir(PAR_DIR) if f.endswith(".par")])

    if verbose:
        print(f"[DEBUG] Found {len(parfiles)} .par files")

    psrs = []
    failed_pulsars = []

    for par in parfiles:
        # Skip known problematic pulsars
        psr_name = par.split('_')[0]
        if psr_name in SKIP_PULSARS:
            if verbose:
                print(f"⚠ Skipping {par} (known hang)")
            continue

        tim = par.replace(".par", ".tim")
        par_path = os.path.join(PAR_DIR, par)
        tim_path = os.path.join(TIM_DIR, tim)

        if not os.path.exists(tim_path) or not tim_has_toas(tim_path):
            if verbose:
                print(f"⚠ No valid tim for {par}")
            failed_pulsars.append(par)
            continue

        try:
            psr = T.tempopulsar(
                parfile=par_path,
                timfile=tim_path,
                maxobs=60000,
                dofit=False,   # skip internal fit to avoid hangs
            )
            if psr.nobs == 0:
                if verbose:
                    print(f"⚠ Zero TOAs for {par}")
                failed_pulsars.append(par)
                continue

            psrs.append(psr)
            if verbose:
                print(f"✓ Loaded {psr.name} ({psr.nobs} TOAs)")

        except Exception as e:
            if verbose:
                print(f"✗ Failed {par}: {e}")
            failed_pulsars.append(par)
            continue

    # Save cache
    if USE_PULSAR_CACHE and len(psrs) > 0:
        try:
            with open(NANOGRAV_PULSAR_CACHE, 'wb') as f:
                pickle.dump(psrs, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"\n💾 Saved cache: {NANOGRAV_PULSAR_CACHE}")
        except Exception as e:
            if verbose:
                print(f"⚠ Could not save cache: {e}")

    if verbose:
        print(f"\n✓ Loaded {len(psrs)} pulsars")
        print(f"✗ Failed: {len(failed_pulsars)} — {failed_pulsars}")

    return psrs



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
    Tspan = total_tmax - total_tmin # days
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