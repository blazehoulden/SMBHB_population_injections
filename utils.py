import numpy as np
import json


def save_results(results, filename):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Saved: {filename}")


def print_population_diagnostics(population):
    """Print population diagnostics."""
    from config import Msun
    from signal_injection import strain_amplitude
    
    print("\n" + "="*70)
    print("POPULATION DIAGNOSTICS")
    print("="*70)
    
    freqs_Hz = [b['f'] for b in population]
    masses = [b['Mc'] for b in population]
    distances = [b['D'] for b in population]
    
    print(f"\nSize: {len(population)} binaries")
    print(f"Frequency range: {min(freqs_Hz)*1e9:.2f} - {max(freqs_Hz)*1e9:.2f} nHz")
    print(f"Mass range: {min(masses)/Msun:.2e} - {max(masses)/Msun:.2e} M☉")
    print(f"Distance range: {min(distances):.1f} - {max(distances):.1f} Mpc")
    
    # Check detectability
    detectable = sum(1 for b in population if 1e-9 < b['f'] < 1e-7)
    print(f"In detectable range (1-100 nHz): {detectable}/{len(population)}")


def print_scaling_summary(results, N_needed, target_SNR):
    """Print scaling analysis summary."""
    print("\n" + "="*70)
    print("SCALING ANALYSIS SUMMARY")
    print("="*70)
    print(f"Target SNR: {target_SNR}σ")
    print(f"SNR Range: {min(results['SNR']):.2f} - {max(results['SNR']):.2f}")
    if N_needed:
        print(f"N required: {N_needed} binaries")
    print("="*70)