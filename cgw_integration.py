#!/usr/bin/env python3
"""
CGW Analysis Integration Module

Provides functions for post-processing CGW analysis results and aggregating
loudest binary candidates across multiple populations for notebook analysis.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_loudest_cgw_metadata(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load 'loudest CGW candidate' metadata files from all populations.
    
    Args:
        input_dir: Directory containing loudest_cgw_pop*.json files
        
    Returns:
        List of metadata dicts, one per population
    """
    loudest_list = []
    input_path = Path(input_dir)
    
    # Find all loudest_cgw_pop*.json files
    loudest_files = sorted(input_path.glob("loudest_cgw_pop*.json"))
    
    for loudest_file in loudest_files:
        with open(loudest_file, 'r') as f:
            data = json.load(f)
            loudest_list.append(data)
    
    return loudest_list


def aggregate_cgw_results(loudest_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics from loudest CGW candidates.
    
    Args:
        loudest_list: List of loudest CGW metadata dicts
        
    Returns:
        Dict with statistics and distributions
    """
    if not loudest_list:
        return {}
    
    # Extract CGW SNRs
    cgw_snrs = np.array([item["loudest_cgw"]["cgw_snr"] for item in loudest_list])
    
    # Extract binary properties
    frequencies = np.array([item["loudest_cgw"]["f"] for item in loudest_list])
    chirp_masses = np.array([item["loudest_cgw"]["Mc"] for item in loudest_list])
    h0s = np.array([item["loudest_cgw"]["h0"] for item in loudest_list])
    
    aggregated = {
        "n_populations": len(loudest_list),
        "cgw_snr": {
            "mean": float(np.mean(cgw_snrs)),
            "median": float(np.median(cgw_snrs)),
            "std": float(np.std(cgw_snrs)),
            "min": float(np.min(cgw_snrs)),
            "max": float(np.max(cgw_snrs)),
            "all_values": cgw_snrs.tolist(),
        },
        "frequency": {
            "mean": float(np.mean(frequencies)),
            "median": float(np.median(frequencies)),
            "std": float(np.std(frequencies)),
            "min": float(np.min(frequencies)),
            "max": float(np.max(frequencies)),
        },
        "chirp_mass": {
            "mean": float(np.mean(chirp_masses)),
            "median": float(np.median(chirp_masses)),
            "std": float(np.std(chirp_masses)),
            "min": float(np.min(chirp_masses)),
            "max": float(np.max(chirp_masses)),
        },
        "h0": {
            "mean": float(np.mean(h0s)),
            "median": float(np.median(h0s)),
            "std": float(np.std(h0s)),
            "min": float(np.min(h0s)),
            "max": float(np.max(h0s)),
        },
    }
    
    return aggregated


def create_notebook_results(
    loudest_list: List[Dict[str, Any]],
    config_name: str,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create unified results file suitable for notebook analysis.
    
    Args:
        loudest_list: List of loudest CGW metadata dicts
        config_name: Configuration name (e.g., 'pessimistic')
        output_file: Optional path to save JSON results
        
    Returns:
        Unified results dict
    """
    aggregated = aggregate_cgw_results(loudest_list)
    
    results = {
        "metadata": {
            "config": config_name,
            "n_populations": len(loudest_list),
            "created_at": str(Path.cwd()),
        },
        "summary_statistics": aggregated,
        "loudest_cgw_candidates": loudest_list,
    }
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved notebook results to: {output_file}")
    
    return results


def print_cgw_summary(aggregated: Dict[str, Any]) -> None:
    """Print human-readable CGW statistics."""
    print("\n" + "="*70)
    print("CGW ANALYSIS SUMMARY ACROSS ALL POPULATIONS")
    print("="*70)
    print(f"\nPopulations analyzed: {aggregated.get('n_populations', 'N/A')}")
    
    if "cgw_snr" in aggregated:
        snr_stats = aggregated["cgw_snr"]
        print(f"\nLoudest CGW SNR per population:")
        print(f"  Mean:   {snr_stats.get('mean', 'N/A'):.4f}")
        print(f"  Median: {snr_stats.get('median', 'N/A'):.4f}")
        print(f"  Std:    {snr_stats.get('std', 'N/A'):.4f}")
        print(f"  Range:  [{snr_stats.get('min', 'N/A'):.4f}, {snr_stats.get('max', 'N/A'):.4f}]")
    
    if "frequency" in aggregated:
        freq_stats = aggregated["frequency"]
        print(f"\nFrequency of loudest binaries:")
        print(f"  Mean:   {freq_stats.get('mean', 'N/A'):.2e} Hz")
        print(f"  Median: {freq_stats.get('median', 'N/A'):.2e} Hz")
        print(f"  Range:  [{freq_stats.get('min', 'N/A'):.2e}, {freq_stats.get('max', 'N/A'):.2e}] Hz")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cgw_integration.py <input_dir> [config_name]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    config_name = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    
    print(f"Loading CGW results from: {input_dir}")
    loudest_list = load_loudest_cgw_metadata(input_dir)
    
    if loudest_list:
        aggregated = aggregate_cgw_results(loudest_list)
        print_cgw_summary(aggregated)
        
        results = create_notebook_results(
            loudest_list,
            config_name,
            output_file=str(Path(input_dir) / "final_cgw_results.json")
        )
    else:
        print(f"No loudest_cgw_pop*.json files found in {input_dir}")
