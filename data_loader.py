import os
import json
import numpy as np
from enterprise.pulsar import Pulsar
from config import PAR_DIR, TIM_DIR, USE_PULSAR_CACHE, NOISEFILE


# Use NPZ format instead of pickle - portable and works across machines
PULSAR_DATA_CACHE = "./cache/pulsar_data.npz"


def extract_pulsar_data(psr):
    """Extract essential data from pulsar object into portable dict."""
    return {
        'name': psr.name,
        'toas': np.asarray(psr.toas, dtype=float),
        'toaerrs': np.asarray(psr.toaerrs, dtype=float),
        'residuals': np.asarray(psr.residuals, dtype=float),
        'freqs': np.asarray(psr.freqs, dtype=float),
        'raj': float(psr._raj),
        'decj': float(psr._decj),
        # Add any other arrays you need
    }


def save_pulsar_cache(psrs, cache_file=PULSAR_DATA_CACHE, verbose=True):
    """Save pulsar data to portable NPZ format."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    # Extract data from all pulsars
    cache_data = {}
    for i, psr in enumerate(psrs):
        data = extract_pulsar_data(psr)
        # Store each field with pulsar index
        for key, value in data.items():
            cache_key = f"psr{i}_{key}"
            cache_data[cache_key] = value
    
    # Save metadata
    cache_data['n_pulsars'] = len(psrs)
    cache_data['pulsar_names'] = [psr.name for psr in psrs]
    
    np.savez_compressed(cache_file, **cache_data)
    
    if verbose:
        size_mb = os.path.getsize(cache_file) / (1024**2)
        print(f"💾 Saved cache: {cache_file} ({size_mb:.1f} MB)")


def load_pulsar_cache(cache_file=PULSAR_DATA_CACHE, verbose=True):
    """Load pulsar data from NPZ cache."""
    if not os.path.exists(cache_file):
        return None
    
    try:
        data = np.load(cache_file, allow_pickle=True)
        n_pulsars = int(data['n_pulsars'])
        
        # Reconstruct pulsar-like objects
        class CachedPulsar:
            """Lightweight pulsar object from cached data."""
            def __init__(self, data_dict):
                for key, value in data_dict.items():
                    setattr(self, key if not key.startswith('_') else key, value)
                # Make sure residuals are accessible as both attribute and _residuals
                if hasattr(self, 'residuals'):
                    self._residuals = self.residuals
                    self._original_residuals = np.copy(self.residuals)
        
        psrs = []
        for i in range(n_pulsars):
            psr_data = {}
            for key in ['name', 'toas', 'toaerrs', 'residuals', 'freqs', 'raj', 'decj']:
                cache_key = f"psr{i}_{key}"
                if cache_key in data:
                    value = data[cache_key]
                    # Convert numpy strings to Python strings
                    if isinstance(value, np.ndarray) and value.dtype.kind in ['U', 'S', 'O']:
                        value = str(value)
                    psr_data[key if key != 'raj' else '_raj'] = value
                    psr_data[key if key != 'decj' else '_decj'] = value
            
            psrs.append(CachedPulsar(psr_data))
        
        if verbose:
            print(f"✓ Loaded {len(psrs)} pulsars from cache")
        
        return psrs
        
    except Exception as e:
        if verbose:
            print(f"⚠ Cache load failed: {e}")
        return None


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
    """Load NANOGrav pulsars with efficient caching."""
    if verbose:
        print("="*70)
        print("LOADING NANOGRAV PULSARS")
        print("="*70)
    
    # Try cache first
    if USE_PULSAR_CACHE:
        psrs = load_pulsar_cache(verbose=verbose)
        if psrs is not None:
            return psrs
    
    # Load from files
    if verbose:
        print(f"\nLoading pulsars from {PAR_DIR}...")

    parfiles = sorted([f for f in os.listdir(PAR_DIR) if f.endswith(".par")])

    if verbose:
        print(f"Found {len(parfiles)} .par files")

    psrs = []
    failed_pulsars = []

    for par in parfiles:
        # Extract pulsar name by removing tempo2 date suffix
        psr_name = par.split("_tempo2_")[0] if "_tempo2_" in par else par.replace(".par", "")
        par_path = os.path.join(PAR_DIR, par)
        
        # Look for matching .tim file
        tim_candidates = [
            f for f in os.listdir(TIM_DIR) 
            if f.startswith(psr_name) and f.endswith(".tim")
        ]
        
        if not tim_candidates:
            if verbose:
                print(f"  ⚠ No .tim file found for {par}")
            failed_pulsars.append(par)
            continue
        
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

    # Save cache
    if USE_PULSAR_CACHE and len(psrs) > 0:
        save_pulsar_cache(psrs, verbose=verbose)

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
    """Get clean pulsars and calculate Tspan."""
    # Calculate Tspan
    tmin = min(p.toas.min() for p in psrs_filtered)
    tmax = max(p.toas.max() for p in psrs_filtered)
    Tspan = tmax - tmin
    
    # Save original residuals for restoration between injections
    for psr in psrs_filtered:
        if not hasattr(psr, '_original_residuals'):
            psr._original_residuals = np.copy(psr.residuals)
    
    return psrs_filtered, Tspan


def restore_original_residuals(psrs):
    """Restore original residuals before next injection."""
    for psr in psrs:
        if hasattr(psr, '_original_residuals'):
            psr._residuals = np.copy(psr._original_residuals)