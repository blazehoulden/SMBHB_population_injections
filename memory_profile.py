#!/usr/bin/env python3
"""
Memory profiling script - run this locally to estimate HPC memory needs
"""

import psutil
import os
from pympler import asizeof
import sys

import psutil
import os

def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory(label=""):
    """Print current memory usage with optional label."""
    mem_mb = get_memory_mb()
    print(f"[MEM] {label}: {mem_mb:.1f} MB")
    return mem_mb

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def monitor_script():
    """Run your main script with memory monitoring"""
    import time
    
    print("="*70)
    print("MEMORY PROFILING")
    print("="*70)
    
    initial_mem = get_memory_usage()
    max_mem = initial_mem
    
    print(f"Initial memory: {initial_mem:.1f} MB\n")
    
    # Import heavy modules
    print("Importing modules...")
    import config
    from enterprise_extensions.frequentist import optimal_statistic as opt_stat
    from data_loader import load_pulsars, filter_pulsars_15yr, get_clean_pulsars_and_tspan
    from signal_injection import inject_population_into_psrs
    from pta_builder import build_pta_and_params
    
    mem_after_imports = get_memory_usage()
    max_mem = max(max_mem, mem_after_imports)
    print(f"After imports: {mem_after_imports:.1f} MB (Δ {mem_after_imports - initial_mem:.1f} MB)\n")
    
    # Load SMBHB module
    print("Loading SMBHB module...")
    smbhb_module = config.load_smbhb_module()
    
    mem_after_smbhb = get_memory_usage()
    max_mem = max(max_mem, mem_after_smbhb)
    print(f"After SMBHB: {mem_after_smbhb:.1f} MB (Δ {mem_after_smbhb - mem_after_imports:.1f} MB)\n")
    
    # Generate population
    print("Generating population...")
    selected_config = config.POPULATION_CONFIGS['optimistic']
    population = config.generate_population(selected_config, smbhb_module)
    
    mem_after_pop = get_memory_usage()
    max_mem = max(max_mem, mem_after_pop)
    print(f"After population: {mem_after_pop:.1f} MB (Δ {mem_after_pop - mem_after_smbhb:.1f} MB)\n")
    
    # Load pulsars
    print("Loading pulsars...")
    psrs = load_pulsars(verbose=True)
    
    mem_after_psrs = get_memory_usage()
    max_mem = max(max_mem, mem_after_psrs)
    print(f"After loading pulsars: {mem_after_psrs:.1f} MB (Δ {mem_after_psrs - mem_after_pop:.1f} MB)\n")
    
    # Filter pulsars
    print("Filtering pulsars...")
    psrs_filtered, noise_params = filter_pulsars_15yr(psrs, verbose=True)
    psrs_clean, Tspan = get_clean_pulsars_and_tspan(psrs_filtered)
    
    mem_after_filter = get_memory_usage()
    max_mem = max(max_mem, mem_after_filter)
    print(f"After filtering: {mem_after_filter:.1f} MB (Δ {mem_after_filter - mem_after_psrs:.1f} MB)\n")
    
    # Inject population (most memory intensive)
    print("Injecting population (memory intensive)...")
    psrs_injected = inject_population_into_psrs(
        psrs_filtered, population, pure_signal=True, verbose=False, pulsar_noise_params=noise_params
    )
    
    mem_after_injection = get_memory_usage()
    max_mem = max(max_mem, mem_after_injection)
    print(f"After injection: {mem_after_injection:.1f} MB (Δ {mem_after_injection - mem_after_filter:.1f} MB)\n")
    
    # Build PTA (also memory intensive)
    print("Building PTA...")
    pta, model, params_complete = build_pta_and_params(
        psrs=psrs_injected, noise_params_15yr=noise_params, Tspan=Tspan
    )
    
    mem_after_pta = get_memory_usage()
    max_mem = max(max_mem, mem_after_pta)
    print(f"After PTA build: {mem_after_pta:.1f} MB (Δ {mem_after_pta - mem_after_injection:.1f} MB)\n")
    
    # Summary
    print("="*70)
    print("MEMORY USAGE SUMMARY")
    print("="*70)
    print(f"Peak memory usage: {max_mem:.1f} MB")
    print(f"Total increase: {max_mem - initial_mem:.1f} MB")
    print(f"\nRecommended HPC settings:")
    
    # Add safety margin (2x for safety)
    recommended_mb = max_mem * 2
    recommended_gb = recommended_mb / 1024
    
    print(f"  --mem={int(recommended_gb) + 1}GB  (with 2x safety margin)")
    print(f"  Minimum: --mem={int(max_mem/1024) + 1}GB")
    
    return max_mem

if __name__ == "__main__":
    try:
        max_mem = monitor_script()
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)