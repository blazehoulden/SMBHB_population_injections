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

    parfiles = sorted([f for f in os.listdir(PAR_DIR) if f.endswith(".par")])

    if verbose:
        print(f"Found {len(parfiles)} .par files")

    psrs = []
    failed_pulsars = []

    for par in parfiles:
        # CHANGED: More flexible tim file matching for alternate directory
        # Extract pulsar name by removing tempo2 date suffix
        psr_name = par.split("_tempo2_")[0] if "_tempo2_" in par else par.replace(".par", "")
        par_path = os.path.join(PAR_DIR, par)
        
        # Look for matching .tim file (handles different naming conventions)
        tim_candidates = [
            f for f in os.listdir(TIM_DIR) 
            if f.startswith(psr_name) and f.endswith(".tim")
        ]
        
        if not tim_candidates:
            if verbose:
                print(f"  ⚠ No .tim file found for {par}")
            failed_pulsars.append(par)
            continue
        
        # Use first matching tim file
        tim_path = os.path.join(TIM_DIR, tim_candidates[0])

        # TIM validation
        if not os.path.exists(tim_path) or not tim_has_toas(tim_path):
            failed_pulsars.append(par)
            continue

        try:
            psr = Pulsar(
                par_path, 
                tim_path, 
                timing_package='tempo2',
                drop_t2pulsar=False
            )
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed {par}: {str(e)[:80]}")
            failed_pulsars.append(par)
            continue

        if len(np.asarray(psr.toas, dtype=float)) == 0:
            failed_pulsars.append(par)
            continue

        psrs.append(psr)
        if verbose and len(psrs) % 10 == 0:
            print(f"  Loaded {len(psrs)} pulsars...")

    # Save cache (disabled for tempo2 - objects can't be pickled)
    # if USE_PULSAR_CACHE and len(psrs) > 0:
    #     try:
    #         with open(NANOGRAV_PULSAR_CACHE, 'wb') as f:
    #             pickle.dump(psrs, f, protocol=pickle.HIGHEST_PROTOCOL)
    #         if verbose:
    #             print(f"\n💾 Saved cache: {NANOGRAV_PULSAR_CACHE}")
    #     except Exception as e:
    #         if verbose:
    #             print(f"⚠ Could not save cache: {e}")

    if verbose:
        print(f"\n✓ Loaded {len(psrs)} pulsars")
        if failed_pulsars:
            print(f"❌ Failed on {len(failed_pulsars)} pulsars: {failed_pulsars[:5]}")

    return psrs


def filter_pulsars_15yr(psrs, min_baseline_years=3.0, verbose=True):
    """Filter to 15yr pulsars with sufficient baseline."""
    with open(NOISEFILE, 'r') as f:
        params = json.load(f)
    
    pulsars_in_15yr = list(set([k.split('_')[0] for k in params.keys() if '_' in k]))
    
    if verbose:
        print(f"\nFiltering pulsars...")
        print(f"Pulsars in noise file: {len(pulsars_in_15yr)}")
    
    psrs_after_15yr = [psr for psr in psrs if psr.name in pulsars_in_15yr]
    
    if verbose:
        missing_from_noise = [psr.name for psr in psrs if psr.name not in pulsars_in_15yr]
        if missing_from_noise:
            print(f"⚠ {len(missing_from_noise)} pulsars not in noise file:")
            for name in missing_from_noise[:5]:
                print(f"  - {name}")
            if len(missing_from_noise) > 5:
                print(f"  ... and {len(missing_from_noise) - 5} more")
    
    psrs_filtered = []
    short_baseline = []
    for psr in psrs_after_15yr:
        baseline_years = (psr.toas.max() - psr.toas.min()) / (365.25 * 86400)
        if baseline_years >= min_baseline_years:
            psrs_filtered.append(psr)
        else:
            short_baseline.append((psr.name, baseline_years))
    
    if verbose:
        if short_baseline:
            print(f"⚠ {len(short_baseline)} pulsars with baseline < {min_baseline_years} years:")
            for name, baseline in short_baseline[:5]:
                print(f"  - {name}: {baseline:.2f} years")
        print(f"\nFiltered: {len(psrs)} → {len(psrs_filtered)} pulsars")
    
    return psrs_filtered, params


def get_clean_pulsars_and_tspan(psrs_filtered):
    """Get clean copies and calculate Tspan.
    
    Note: With tempo2, pulsar objects can't be deep copied due to 
    underlying C objects. We return the original list and save original
    residuals for restoration between runs.
    """
    # Calculate Tspan
    tmin = min(p.toas.min() for p in psrs_filtered)
    tmax = max(p.toas.max() for p in psrs_filtered)
    Tspan = tmax - tmin
    
    # Save original residuals for each pulsar (for restoration between injections)
    for psr in psrs_filtered:
        if not hasattr(psr, '_original_residuals'):
            psr._original_residuals = np.copy(psr.residuals)
    
    # Return original pulsars
    # (tempo2 pulsar objects can't be deepcopied)
    return psrs_filtered, Tspan


def restore_original_residuals(psrs):
    """Restore original residuals before next injection."""
    for psr in psrs:
        if hasattr(psr, '_original_residuals'):
            psr._residuals = np.copy(psr._original_residuals)