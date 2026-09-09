#!/usr/bin/env python3
"""
Merge batch results and validate COMBINED population SNR.

This script:
1. Loads all batch_results_*.json files
2. Combines all populations into one
3. Computes SNR of the combined injection
4. Validates/scales if needed
5. Writes individual zarr files (with scaling applied)
6. Saves consistency summary
"""
import gc
import os
import sys
import json
import numpy as np
from pathlib import Path

import config
from consistent_pop_synth import merge_batch_results, validate_and_scale_combined_populations
from data_loader import filter_pulsars_15yr, get_clean_pulsars_and_tspan, load_pulsars, parse_pulsar_parameters
from io_backends import population_to_zarr
from utils import compact_consistent_results_for_storage, save_results, save_results_dual

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # Parse arguments
    config_name = sys.argv[1]
    target_snr = float(sys.argv[2])
    snr_low = float(sys.argv[3])
    snr_high = float(sys.argv[4])
    n_sims = int(sys.argv[5])
    batch_output_dir = sys.argv[6]
    output_root = sys.argv[7]
    n_chunks = int(sys.argv[8])
    minimal_pop_storage = sys.argv[9].lower() == "true"
    
    print("="*70)
    print("MERGE AND VALIDATE COMBINED POPULATIONS")
    print("="*70)
    print(f"Config: {config_name}")
    print(f"Total sims: {n_sims}")
    print(f"Target SNR: {target_snr}")
    print(f"SNR range: [{snr_low}, {snr_high}]")
    print(f"Batch output dir: {batch_output_dir}")
    print(f"="*70 + "\n")
    
    # Step 1: Merge batch results
    print("Step 1: Merging batch results...")
    merged_results = merge_batch_results(
        batch_output_dir=batch_output_dir,
        N_sims=n_sims,
        n_batches=n_sims,  # Each batch is one array task
        verbose=True,
    )
    
    populations_for_validation = merged_results.get("populations", [])
    if not populations_for_validation:
        raise ValueError("No populations found in merged results!")
    
    print(f"\n✓ Merged {len(populations_for_validation)} populations\n")
    
    # Step 2: Load pulsar data
    print("Step 2: Loading pulsar data for combined SNR validation...")
    psrs_unfiltered = load_pulsars(verbose=False)
    psrs_filtered = filter_pulsars_15yr(psrs_unfiltered)
    psrs_clean, Tspan_seconds = get_clean_pulsars_and_tspan(psrs_filtered)
    raw_noise_params = parse_pulsar_parameters(config.NOISEFILE)
    original_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}
    print(f"✓ Loaded {len(psrs_clean)} pulsars\n")
    
    # Step 3: Validate combined population SNR
    print("Step 3: Validating COMBINED population SNR...")
    validation_result = validate_and_scale_combined_populations(
        populations=populations_for_validation,
        psrs_clean=psrs_clean,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan_seconds,
        target_SNR=target_snr,
        SNR_range=(snr_low, snr_high),
        original_stoas=original_stoas,
        verbose=True,
        timer=True,
        inject_eps=1e-6,
        precompute_parallel=True,
        max_retries=25,
    )
    
    if validation_result is None:
        raise RuntimeError(
            f"SGWB consistency check FAILED for combined population. "
            f"Could not achieve target SNR {target_snr} in range [{snr_low}, {snr_high}]"
        )
    
    print(f"\n✓ COMBINED population validated successfully!")
    print(f"  Combined SNR: {validation_result['SNR_final']:.4f}")
    print(f"  Total binaries: {validation_result['n_binaries_total']}")
    print(f"  Distance scaling factor: {validation_result['scaling_factor']:.6f}\n")
    
    # Step 4: Write zarr files using scaled combined population
    print("Step 4: Writing zarr files with scaling applied...")
    
    chunks_base = os.path.join(output_root, "chunks")
    os.makedirs(chunks_base, exist_ok=True)
    
    combined_pop = validation_result['combined_population']
    scaling_factor = validation_result['scaling_factor']
    
    if minimal_pop_storage:
        field_dtypes = {
            "f": np.float32,
            "h0": np.float32,
            "ra": np.float16,
            "dec": np.float16,
            "psi": np.float16,
            "iota": np.float16,
            "phi0": np.float16,
        }
    else:
        field_dtypes = None
    
    # Track which binaries belong to which population
    binary_start_idx = 0
    pop_boundary_indices = []
    for pop in populations_for_validation:
        if hasattr(pop, 'binaries'):
            n_binaries = len(pop.binaries)
        elif isinstance(pop, dict) and 'binaries' in pop:
            n_binaries = len(pop['binaries'])
        else:
            n_binaries = 0
        pop_boundary_indices.append((binary_start_idx, binary_start_idx + n_binaries))
        binary_start_idx += n_binaries
    
    # Write zarr for each population using scaled combined binaries
    for pop_idx, (start_idx, end_idx) in enumerate(pop_boundary_indices):
        combined_binaries = combined_pop.binaries
        pop_binaries = combined_binaries[start_idx:end_idx]
        
        if not pop_binaries:
            print(f"  Skipping population {pop_idx} (no binaries)")
            continue
        
        class _SimplePop:
            pass
        
        sp = _SimplePop()
        sp.f = combined_pop.f[start_idx:end_idx]
        sp.Mc = combined_pop.Mc[start_idx:end_idx]
        sp.D_comov = combined_pop.D_comov[start_idx:end_idx]
        sp.z = combined_pop.z[start_idx:end_idx]
        sp.h0 = combined_pop.h0[start_idx:end_idx]
        sp.ra = combined_pop.ra[start_idx:end_idx]
        sp.dec = combined_pop.dec[start_idx:end_idx]
        sp.psi = combined_pop.psi[start_idx:end_idx]
        sp.iota = combined_pop.iota[start_idx:end_idx]
        sp.phi0 = combined_pop.phi0[start_idx:end_idx]
        
        zarr_path = os.path.join(chunks_base, f"population_pop{pop_idx}.zarr")
        print(f"  Writing population {pop_idx}: {len(pop_binaries)} scaled binaries -> {zarr_path}")
        population_to_zarr(
            zarr_path,
            sp,
            dtype=np.float32,
            chunk_size=1_000_000,
            field_dtypes=field_dtypes,
        )
        del sp
        gc.collect()
    
    # Step 5: Save consistency summary
    print("\nStep 5: Saving consistency summary...")
    consistency_summary_path = os.path.join(chunks_base, "sgwb_consistency_summary.json")
    consistency_summary = {
        "validation_mode": "combined",
        "description": "All populations combined, single SNR calculated and validated",
        "target_snr": float(target_snr),
        "snr_range": [float(snr_low), float(snr_high)],
        "combined_snr_final": float(validation_result['SNR_final']),
        "scaling_factor": float(validation_result['scaling_factor']),
        "n_populations": int(len(populations_for_validation)),
        "n_binaries_total": int(validation_result['n_binaries_total']),
        "n_binaries_per_population": [
            int(len(pop.get('binaries', [])) if isinstance(pop, dict) else (len(pop.binaries) if hasattr(pop, 'binaries') else 0))
            for pop in populations_for_validation
        ],
    }
    
    with open(consistency_summary_path, 'w') as f:
        json.dump(consistency_summary, f, indent=2)
    print(f"✓ Consistency summary saved: {consistency_summary_path}")
    
    # Step 6: Save merged compact results
    print("\nStep 6: Saving merged population results...")
    save_path = os.path.join(output_root, f"consistent_population_{config_name}_targetSNR{snr_high}_merged.json")
    compact_results = compact_consistent_results_for_storage(
        merged_results,
        max_mb_per_sim=1.0,
        n_nearest=100,
        n_loudest=10000,
    )
    for pop in compact_results.get("populations", []):
        pop.pop("pta", None)
        pop.pop("psrs", None)
    
    save_results_dual(compact_results, save_path, save_compact_npz=False)
    print(f"✓ Merged population results saved: {save_path}")
    
    print("\n" + "="*70)
    print("MERGE AND VALIDATION COMPLETE")
    print("="*70)
    print(f"Output zarr files: {chunks_base}/population_pop*.zarr")
    print(f"Consistency summary: {consistency_summary_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
