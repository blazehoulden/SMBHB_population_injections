import os
import pickle
import json
import numpy as np
from copy import deepcopy
from enterprise.pulsar import Pulsar
from config import PAR_DIR, TIM_DIR, USE_PULSAR_CACHE, NANOGRAV_PULSAR_CACHE, NOISEFILE


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


def load_pulsars(verbose=True):
    """Load NANOGrav pulsars with caching."""
    if verbose:
        print("="*70)
        print("LOADING NANOGRAV PULSARS")
        print("="*70)
    psrs = None
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
            psrs = None

    # Load from files
    if verbose:
        print(f"Loading pulsars from {PAR_DIR}...")
    # 🔍 DEBUG: check contents of PAR_DIR
    if verbose:
        print(f"[DEBUG] PAR_DIR exists: {os.path.exists(PAR_DIR)}")
        if os.path.exists(PAR_DIR):
            print(f"[DEBUG] PAR_DIR contains: {os.listdir(PAR_DIR)}")

    parfiles = sorted([f for f in os.listdir(PAR_DIR) if f.endswith(".par")])

    # 🔍 DEBUG: show parfiles found
    if verbose:
        print(f"[DEBUG] Found {len(parfiles)} .par files: {parfiles}")

    psrs = []
    failed_pulsars = []

    for par in parfiles:
        tim = par.replace(".par", ".tim")
        par_path = os.path.join(PAR_DIR, par)
        tim_path = os.path.join(TIM_DIR, tim)

        # 🔍 DEBUG: pairing check
        if verbose:
            print(f"[DEBUG] pairing: {par} → {tim}")
            print(f"        par_exists={os.path.exists(par_path)}, tim_exists={os.path.exists(tim_path)}")

        # 🔍 DEBUG: display tim header if exists
        if verbose and os.path.exists(tim_path):
            print(f"[DEBUG] Checking {tim_path}")
            print(f"        size={os.path.getsize(tim_path)} bytes")
            with open(tim_path, 'r') as fh:
                for i, ln in enumerate(fh):
                    if i > 3:
                        break
                    print("        > ", ln.strip())

        # TIM validation
        if not os.path.exists(tim_path) or not tim_has_toas(tim_path):
            if verbose:
                print(f"[DEBUG] tim_has_toas() returned False for {tim_path}")
            failed_pulsars.append(par)
            continue

        try:
            psr = Pulsar(par_path, tim_path)
        except Exception as e:
            if verbose:
                print(f"[DEBUG] enterprise failed on {par}: {e}")
            try:
                psr = Pulsar(par_path, tim_path, use_pint=True, ephem="DE440",
                            clk_corr=False, maxobs=None)
            except Exception as e2:
                if verbose:
                    print(f"[DEBUG] fallback also failed on {par}: {e2}")
                failed_pulsars.append(par)
                continue

        if len(np.asarray(psr.toas, dtype=float)) == 0:
            if verbose:
                print(f"[DEBUG] Pulsar {par} has zero TOAs after load")
            failed_pulsars.append(par)
            continue

        psrs.append(psr)
        if verbose:
            print(f"✓ Loaded {par}")

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
        print(f"❌ Failed on {len(failed_pulsars)} pars: {failed_pulsars}")

    return psrs



def filter_pulsars_15yr(psrs, min_baseline_years=3.0, verbose=True):
    """Filter to 15yr pulsars with sufficient baseline."""
    with open(NOISEFILE, 'r') as f:
        params = json.load(f)
    
    pulsars_in_15yr = list(set([k.split('_')[0] for k in params.keys() if '_' in k]))
    
    psrs_after_15yr = [psr for psr in psrs if psr.name in pulsars_in_15yr]
    
    psrs_filtered = []
    for psr in psrs_after_15yr:
        baseline_years = (psr.toas.max() - psr.toas.min()) / (365.25 * 86400)
        if baseline_years >= min_baseline_years:
            psrs_filtered.append(psr)
    
    if verbose:
        print(f"\nFiltered: {len(psrs)} → {len(psrs_filtered)} pulsars")
    
    return psrs_filtered, params


def get_clean_pulsars_and_tspan(psrs_filtered):
    """
    Get pulsars and calculate Tspan.
    
    Note: Returns original pulsars (not copies) to save memory.
    Original residuals are saved for restoration between injections.
    """
    # Calculate Tspan
    tmin = min(p.toas.min() for p in psrs_filtered)
    tmax = max(p.toas.max() for p in psrs_filtered)
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