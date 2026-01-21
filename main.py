#!/usr/bin/env python3
"""
Main execution script for SMBHB population analysis.
Run with: python main.py
"""
import argparse
# from memory_profiler import profile
import config
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from data_loader import load_pulsars, filter_pulsars_15yr, get_clean_pulsars_and_tspan
from signal_injection import inject_population_into_psrs
from pta_builder import build_pta_and_params
from scaling_analysis import run_scaling_analysis
from individual_binary import analyze_individual_binaries
from ensemble_analysis import find_N_ensemble, find_N_binaries_for_target_snr
from visualisation import plot_binaries_vs_frequency_mc, plot_scaling_results, plot_individual_binaries, plot_ensemble_results, plot_initial_injection_analysis, print_binary_statistics, plot_binaries_vs_frequency
from utils import save_results, print_population_diagnostics, print_scaling_summary

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

    return parser.parse_args()

# @profile
def main():
    """Main analysis workflow."""
    args = parse_args()
    
    # ========== SETUP ==========
    print("\n" + "="*70)
    print("SMBHB POPULATION ANALYSIS")
    print("="*70)
    
    # Load SMBHB module
    smbhb_module = config.load_smbhb_module()
    
    # Select configuration
    CONFIG_NAME = args.config  # Change as needed
    # CONFIG_NAME = 'realistic'  # Change as needed
    selected_config = config.POPULATION_CONFIGS[CONFIG_NAME]
    
    print(f"\nConfiguration: {CONFIG_NAME}")
    print(f"  {selected_config['description']}")
    print(f"  N_binaries: {selected_config['N_binaries']}")
    
    # Generate population
    print("\n📊 Generating SMBHB population...")
    population = config.generate_population(selected_config, smbhb_module)
    print_population_diagnostics(population)
    
    # ========== LOAD PULSARS ==========
    print("\n📡 Loading pulsars...")
    psrs = load_pulsars(verbose=True)
    
    print("\n🔍 Filtering pulsars...")
    psrs_filtered, noise_params = filter_pulsars_15yr(psrs, verbose=True)
    
    psrs_clean, Tspan = get_clean_pulsars_and_tspan(psrs_filtered)
    print(f"\n✓ Ready: {len(psrs_clean)} pulsars, Tspan = {Tspan/(365.25*86400):.1f} years")
    
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
        # Compute SNR and other diagnostics
        # Initialize Optimal Statistic with Hellings-Downs correlation
        ostat = opt_stat.OptimalStatistic(psrs_injected, pta=pta, orf='hd')

        # Compute OS
        # Returns: xi (ORF), rho (correlations), sig (uncertainties), OS, OS_sig
        xi, rho, sig, OS, OS_sig = ostat.compute_os(params=params_complete)

        snr = OS / OS_sig
        print(f"\n✓ Initial Injection SNR: {snr:.3f}")
        plot_initial_injection_analysis(psrs_injected, population, snr, xi, rho)
    
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
        plot_scaling_results(results, N_needed, 4.0, CONFIG_NAME)
        save_results({'scaling': results, 'N_needed': N_needed}, 
                    f'data/scaling_results_{CONFIG_NAME}.json')
    
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
            plot_individual_binaries(df, psrs_injected=psrs_clean, top_N=20)
            df.to_csv('data/individual_binary_results.csv', index=False)
            print("💾 Saved: data/individual_binary_results.csv")
    
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


        save_results(ensemble_results, f'data/ensemble_results_{CONFIG_NAME}.json')
        
    if config.RUN_CONSISTENT_POP_SYNTH:
        print("\n" + "="*70)
        print("CONSISTENT POPULATION SYNTHESIS")
        print("="*70)
        
        from consistent_pop_synth import generate_snr_consistent_populations
        # auto guess: default = 0.5 * N_binaries
        if args.initial_guess == "auto":
            N_initial_guess = int(0.5 * selected_config['N_binaries'])
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
            profile=False
        )
        
        save_results(consistent_results, f'data/consistent_pop_synth_{CONFIG_NAME}.json')

    if config.RUN_NG_RG_COMPARISON:
        print("\n" + "="*70)
        print("NANOGrav Rohan & Gondor COMPARISON")
        print("="*70)
        plot_binaries_vs_frequency(population, subset_name=CONFIG_NAME, candidate_frequencies=[14e-9, 21e-9],  # optional
            candidate_labels=['Gondor 14 nHz', 'Rohan 21 nHz'], candidate_masses=[9.75, 10.05])  # NEW

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()