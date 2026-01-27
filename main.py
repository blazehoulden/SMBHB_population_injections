#!/usr/bin/env python3
"""
Main execution script for SMBHB population analysis.
Run with: python main.py
"""
import argparse
import os
from datetime import datetime
import config
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from data_loader import load_pulsars, filter_pulsars_15yr, get_clean_pulsars_and_tspan
from signal_injection import inject_population_into_psrs
from pta_builder import build_pta_and_params
from scaling_analysis import run_scaling_analysis
from individual_binary import analyze_individual_binaries
from memory_profile import log_memory
from ensemble_analysis import find_N_ensemble, find_N_binaries_for_target_snr
from visualisation import plot_binaries_vs_frequency_mc, plot_scaling_results, plot_individual_binaries, plot_ensemble_results, plot_initial_injection_analysis, print_binary_statistics, plot_binaries_vs_frequency
from utils import save_results, print_population_diagnostics, print_scaling_summary
import gc

def parse_args():
    parser = argparse.ArgumentParser(
        description="SMBHB population analysis pipeline"
    )

    parser.add_argument(
        "--config", "-c",
        default="optimistic",
        choices=list(config.POPULATION_CONFIGS.keys()),
        help="Which population configuration to use"
    )

    parser.add_argument(
        "--target-snr", type=float, default=3.75,
        help="Target SNR for ensemble/scaling analyses"
    )

    parser.add_argument(
        "--snr-range", nargs=2, type=float, default=[3.5, 4.0],
        help="SNR range for ensemble search (low high)"
    )

    parser.add_argument(
        "--realisations", "-n", type=int, default=10,
        help="Number of ensemble realisations"
    )

    parser.add_argument(
        "--initial-guess", default="auto",
        help="Initial guess for N_binaries (number or 'auto')"
    )

    parser.add_argument(
        "--simulations", "-s", type=int, default=10,
        help="Number of simulations for consistent population synthesis"
    )

    parser.add_argument(
        "--save-name", type=str, default=None,
        help="Optional custom save name (e.g., 'run_001' or job array index)"
    )

    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Optional custom save directory (default: data/YYYY-MM-DD/)"
    )

    return parser.parse_args()


def setup_save_directory(args):
    """
    Setup organized save directory structure.
    
    Creates: data/{date}/{config}_{savename}/
    
    Returns:
        save_dir: full path to save directory
        run_name: name for this run (config_savename)
    """
    # Determine run name
    if args.save_name:
        run_name = f"{args.config}_{args.save_name}"
    else:
        run_name = args.config
    
    # Determine save directory
    if args.save_dir:
        # User-specified directory
        save_dir = args.save_dir
    else:
        # Default: data/{date}/{run_name}/
        date_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join("data", date_str, run_name)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📁 Save directory: {save_dir}")
    print(f"📝 Run name: {run_name}\n")
    
    return save_dir, run_name


def main():
    """Main analysis workflow."""
    args = parse_args()
    
    # Setup save directory
    save_dir, run_name = setup_save_directory(args)

    toggle_memory_profiling = config.MEMORY_PROFILE_ENABLED
    if toggle_memory_profiling:
        log_memory("Start")
    
    # ========== SETUP ==========
    print("\n" + "="*70)
    print("SMBHB POPULATION ANALYSIS")
    print("="*70)
    
    # Load SMBHB module
    smbhb_module = config.load_smbhb_module()
    
    # Select configuration
    CONFIG_NAME = args.config
    selected_config = config.POPULATION_CONFIGS[CONFIG_NAME]
    
    print(f"\nConfiguration: {CONFIG_NAME}")
    print(f"  {selected_config['description']}")
    print(f"  N_binaries: {selected_config['N_binaries']}")
    
    # Generate population
    print("\n📊 Generating sample SMBHB population...")
    population = config.generate_population(selected_config, smbhb_module)
    print_population_diagnostics(population)
    
    # ========== LOAD PULSARS ==========
    print("\n📡 Loading pulsars...")
    psrs = load_pulsars(verbose=True)
    if toggle_memory_profiling:
        log_memory("After loading pulsars")
    
    print("\n🔍 Filtering pulsars...")
    psrs_filtered, noise_params = filter_pulsars_15yr(psrs, verbose=True)
    if toggle_memory_profiling:
        log_memory("After filtering pulsars")
    
    psrs_clean, Tspan = get_clean_pulsars_and_tspan(psrs_filtered)
    print(f"\n✓ Ready: {len(psrs_clean)} pulsars, Tspan = {Tspan/(365.25*86400):.1f} years")
    if toggle_memory_profiling:
        log_memory("After getting clean pulsars and Tspan")

    # Force garbage collection
    gc.collect()
    if toggle_memory_profiling:
        log_memory("After garbage collection")
    
    # ========== INITIAL INJECTION (OPTIONAL) ==========
    if config.RUN_INITIAL_INJECTION_ANALYSIS:
        print("\n" + "="*70)
        print("INITIAL INJECTION ANALYSIS")
        print("="*70)
        
        psrs_injected = inject_population_into_psrs(
            psrs_filtered, population, pure_signal=True, verbose=True
        )
        
        pta, model, params_complete = build_pta_and_params(
            psrs=psrs_injected, noise_params_15yr=noise_params, Tspan=Tspan
        )
        
        print(f"✓ PTA built with {len(pta.params)} parameters")
        ostat = opt_stat.OptimalStatistic(psrs_injected, pta=pta, orf='hd')
        xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_complete)

        snr = OS / OS_sig
        print(f"\n✓ Initial Injection SNR: {snr:.3f}")
        
        # Save with organized naming
        plot_initial_injection_analysis(
            psrs_injected, population, snr, xi, rho,
            save_dir=save_dir, run_name=run_name
        )
    
    # ========== SCALING ANALYSIS ==========
    if config.RUN_SCALING_ANALYSIS:
        print("\n" + "="*70)
        print("SCALING ANALYSIS")
        print("="*70)
        
        results, N_needed = run_scaling_analysis(
            population, psrs_clean, noise_params, Tspan, 
            target_SNR=4.0, n_test_points=5
        )
        
        print_scaling_summary(results, N_needed, target_SNR=4.0)
        
        # Save results
        save_path = os.path.join(save_dir, f'scaling_results.json')
        save_results({'scaling': results, 'N_needed': N_needed}, save_path)
        
        # Save plots
        plot_scaling_results(
            results, N_needed, 4.0, 
            save_dir=save_dir, run_name=run_name
        )
    
    # ========== INDIVIDUAL BINARY ANALYSIS ==========
    if config.RUN_INDIVIDUAL_BINARY_ANALYSIS:
        print("\n" + "="*70)
        print("INDIVIDUAL BINARY ANALYSIS")
        print("="*70)
        
        df = analyze_individual_binaries(
            population, psrs_clean, noise_params, Tspan, max_binaries=50
        )
        print(df.head())
        
        if df is not None:
            print_binary_statistics(df, top_n=10)
            print(f"\n✓ Analyzed {len(df)} binaries")
            print(f"  Loudest SNR: {df.iloc[0]['SNR']:+.3f}")
            
            # Save results
            csv_path = os.path.join(save_dir, 'individual_binary_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"💾 Saved: {csv_path}")
            
            # Save plots
            plot_individual_binaries(
                df, psrs_injected=psrs_clean, top_N=20,
                save_dir=save_dir, run_name=run_name
            )
    
    # ========== ENSEMBLE ANALYSIS ==========
    if config.RUN_ENSEMBLE_ANALYSIS:
        print("\n" + "="*70)
        print("ENSEMBLE ANALYSIS")
        print("="*70)
        
        # auto guess: default = 0.5 * N_binaries
        if args.initial_guess == "auto":
            N_initial_guess = int(0.5 * selected_config['N_binaries'])
        else:
            N_initial_guess = int(args.initial_guess)

        SNR_low, SNR_high = args.snr_range

        ensemble_results = find_N_ensemble(
            selected_config, smbhb_module, psrs_clean, noise_params, Tspan,
            target_SNR=args.target_snr,
            SNR_range=(SNR_low, SNR_high),
            n_realisations=args.realisations,
            N_initial_guess=N_initial_guess,
            N_max_initial=selected_config['N_binaries'] * 3
        )
        
        if 'statistics' in ensemble_results:
            stats = ensemble_results['statistics']
            print(f"\nN_binaries statistics:")
            print(f"  Mean: {stats['mean']:.0f}")
            print(f"  Median: {stats['median']:.0f}")
            print(f"  Std: {stats['std']:.0f}")

        # Save results
        save_path = os.path.join(save_dir, 'ensemble_results.json')
        save_results(ensemble_results, save_path)
        
    # ========== CONSISTENT POPULATION SYNTHESIS ==========
    if config.RUN_CONSISTENT_POP_SYNTH:
        print("\n" + "="*70)
        print("CONSISTENT POPULATION SYNTHESIS")
        print("="*70)
        
        from consistent_pop_synth import generate_snr_consistent_populations
        
        # auto guess: default = full N_binaries for consistent pop
        if args.initial_guess == "auto":
            N_initial_guess = int(selected_config['N_binaries'])
        else:
            N_initial_guess = int(args.initial_guess)
        
        SNR_low, SNR_high = args.snr_range
        
        consistent_results = generate_snr_consistent_populations(
            selected_config, smbhb_module, psrs_clean, noise_params, Tspan,
            SNR_range=(SNR_low, SNR_high),
            N_sims=args.simulations,
            N_initial_guess=N_initial_guess,
            N_max_initial=selected_config['N_binaries'] * 3,
            verbose=True,
            profile=False,
            use_cache=False,
            cache_threshold=0
        )
        
        # Save results
        save_path = os.path.join(save_dir, 'consistent_pop_synth.json')
        save_results(consistent_results, save_path)

    # ========== NG R&G COMPARISON ==========
    if config.RUN_NG_RG_COMPARISON:
        print("\n" + "="*70)
        print("NANOGrav Rohan & Gondor COMPARISON")
        print("="*70)
        
        plot_binaries_vs_frequency(
            population, subset_name=run_name,
            candidate_frequencies=[14e-9, 21e-9],
            candidate_labels=['Gondor 14 nHz', 'Rohan 21 nHz'],
            candidate_masses=[9.75, 10.05],
            save_dir=save_dir
        )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()