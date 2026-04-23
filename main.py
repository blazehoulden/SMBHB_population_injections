#!/usr/bin/env python3
"""
Main execution script for SMBHB population analysis.
Run with: python main.py
"""
import os
import sys

from consistent_pop_synth import generate_snr_consistent_populations_distance_scaling, suppress_enterprise_warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import config
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from SMBHB_pop_synth import chosen_population
from data_loader import load_pulsars, filter_pulsars_15yr, get_clean_pulsars_and_tspan, parse_pulsar_parameters
from signal_injection import inject_population_nufft
from pta_builder import build_pta_and_params
from scaling_analysis import run_scaling_analysis
from individual_binary import analyze_individual_binaries
from memory_profile import log_memory
from consistent_pop_synth import compute_population_snr
# from ensemble_analysis import find_N_ensemble, find_N_binaries_for_target_snr
from optimal_SNR_calc import N_needed_for_population, SNR_sq_all_pairs_all_binaries_vectorised, convergence_test, plot_overlap_reduction_function, plot_overlap_reduction_function, find_N_needed, compare_pulsar_psd_methods, plot_psd_comparison, sigma_ab, test_psd_vs_residuals_consistency
from CGW_SNR import compute_cgw_snr_optimal_population, compute_population_gwb_psd_from_psrs, get_per_pulsar_covariance_from_population
from visualisation import plot_binaries_vs_frequency_mc, plot_scaling_results, plot_individual_binaries, plot_ensemble_results, plot_initial_injection_analysis, plot_snr_population, print_binary_statistics, plot_binaries_vs_frequency
from utils import save_results, save_results_dual, print_population_diagnostics, print_scaling_summary, compact_consistent_results_for_storage
# from pulsar_noise_using_enterprise import get_noise_matrix
from enterprise.signals.gp_bases import createfourierdesignmatrix_red

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
        "--target-snr", type=float, default=4.0,
        help="Target SNR for ensemble/scaling analyses"
    )

    parser.add_argument(
        "--snr-range", nargs=2, type=float, default=[3.5, 4.25],
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

    parser.add_argument(
        "--max-save-mb-per-sim", type=float, default=5.0,
        help="Target max saved size per simulation for consistent-pop outputs"
    )

    parser.add_argument(
        "--save-nearest", type=int, default=100,
        help="Number of nearest binaries (smallest D_comov) always kept per simulation"
    )

    parser.add_argument(
        "--save-loudest", type=int, default=10000,
        help="Number of loudest binaries (highest h0) always kept per simulation"
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

import io

class TeeLogger:
    """Writes to both stdout and a log file simultaneously."""
    def __init__(self, filepath, mode='w'):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, buffering=1)  # line-buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()
        sys.stdout = self.terminal

logger = TeeLogger("run_log.txt")
sys.stdout = logger

def main():
    """Main analysis workflow."""
    args = parse_args()
    
    # Setup save directory
    save_dir, run_name = setup_save_directory(args)

    toggle_memory_profiling = config.MEMORY_PROFILE_ENABLED
    if toggle_memory_profiling:
        log_memory("Start")
    
    # add time toggle for loading pulsars
    # ========== LOAD PULSARS ========== Takes ~ 1 second per pulsar
    print("\n📡 Loading pulsars...") 
    psrs_unfiltered = load_pulsars(verbose=True)
    if toggle_memory_profiling:
        log_memory("After loading pulsars")
    
    print("\n🔍 Filtering pulsars...")
    with suppress_enterprise_warnings():

        psrs_clean, raw_noise_params, Tspan_seconds = filter_pulsars_15yr(psrs_unfiltered, verbose=True)
        if toggle_memory_profiling:
            log_memory("After filtering pulsars")
    
    # psrs_clean, Tspan_seconds = get_clean_pulsars_and_tspan(psrs_filtered)
    # print(f"\n✓ Ready: {len(psrs_clean)} pulsars, Tspan = {Tspan_seconds/(365.25*86400):.1f} years")
    # if toggle_memory_profiling:
    #     log_memory("After getting clean pulsars and Tspan")
    parsed_noise_params = parse_pulsar_parameters(config.NOISEFILE)

    # testing code 
    # from debug.test_noise_residuals import run_all_tests
    # results = run_all_tests(psrs_clean, parsed_noise_params, Tspan, n_pulsars_to_test=5)

    # Force garbage collection
    gc.collect()
    if toggle_memory_profiling:
        log_memory("After garbage collection")

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
    print(f"  N_binaries: {selected_config['n_binaries']}")
    
    # Generate population
    if config.GEN_POP:
        print("\n📊 Generating sample SMBHB population...")
        population = config.generate_population(selected_config, smbhb_module, T_obs_seconds=Tspan_seconds)
    # print_population_diagnostics(population)
    
    # ========== INITIAL INJECTION (OPTIONAL) ==========
    if config.RUN_INITIAL_INJECTION_ANALYSIS:
        print("\n" + "="*70)
        print("INITIAL INJECTION ANALYSIS")
        print("="*70)
        
        psrs_injected = inject_population_nufft(
            psrs_clean, population, pure_signal=True, verbose=True
        )
        
        pta, model, params_complete = build_pta_and_params(
            psrs=psrs_injected, noise_params_15yr=raw_noise_params, Tspan=Tspan_seconds
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
            population, psrs_clean, raw_noise_params, Tspan_seconds, 
            target_SNR=4.0, n_test_points=5
        )
        
        print_scaling_summary(results, N_needed, target_SNR=4.0)
        
        # Save results
        save_path = os.path.join(save_dir, f'scaling_results.json')
        save_results_dual({'scaling': results, 'N_needed': N_needed}, save_path)
        
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
            population, psrs_clean, raw_noise_params, Tspan_seconds, max_binaries=50
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
    # if config.RUN_ENSEMBLE_ANALYSIS:
        # print("\n" + "="*70)
        # print("ENSEMBLE ANALYSIS")
        # print("="*70)
        
        # # auto guess: default = 0.5 * n_binaries
        # if args.initial_guess == "auto":
        #     N_initial_guess = int(0.5 * selected_config['n_binaries'])
        # else:
        #     N_initial_guess = int(args.initial_guess)

        # SNR_low, SNR_high = args.snr_range

        # ensemble_results = find_N_ensemble(
        #     selected_config, smbhb_module, psrs_clean, raw_noise_params, Tspan_seconds,
        #     target_SNR=args.target_snr,
        #     SNR_range=(SNR_low, SNR_high),
        #     n_realisations=args.realisations,
        #     N_initial_guess=N_initial_guess,
        #     N_max_initial=selected_config['n_binaries'] * 3
        # )
        
        # if 'statistics' in ensemble_results:
        #     stats = ensemble_results['statistics']
        #     print(f"\nn_binaries statistics:")
        #     print(f"  Mean: {stats['mean']:.0f}")
        #     print(f"  Median: {stats['median']:.0f}")
        #     print(f"  Std: {stats['std']:.0f}")

        # # Save results
        # save_path = os.path.join(save_dir, 'ensemble_results.json')
        # save_results(ensemble_results, save_path)
        
    # ========== CONSISTENT POPULATION SYNTHESIS ==========
    if config.RUN_CONSISTENT_POP_SYNTH:
        print("\n" + "="*70)
        print("CONSISTENT POPULATION SYNTHESIS")
        print("="*70)

        N_initial_guess = (
            int(selected_config['n_binaries'])
            if args.initial_guess == "auto"
            else int(args.initial_guess)
        )

        SNR_low, SNR_high = args.snr_range

        original_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}

        consistent_results = generate_snr_consistent_populations_distance_scaling(
            config_template=selected_config,
            smbhb_module=smbhb_module,
            psrs_clean=psrs_clean,
            raw_noise_params=raw_noise_params,
            Tspan=Tspan_seconds,
            original_stoas=original_stoas,
            target_SNR=args.target_snr,
            SNR_range=args.snr_range,
            N_sims=args.simulations,
            verbose=True,
            save_populations=True,
            profile=True,
            n_iterations=4,
            toggle_memory_profiling=False,
            keep_amplitudes_in_result=False,
            precompute_parallel=True,
            inject_eps=1e-6,
        )

        file_name = f'consistent_population_{CONFIG_NAME}_targetSNR{SNR_high}_sims{args.simulations}.json'
        save_path = os.path.join(save_dir, file_name)
        compact_results = compact_consistent_results_for_storage(
            consistent_results,
            max_mb_per_sim=args.max_save_mb_per_sim,
            n_nearest=args.save_nearest,
            n_loudest=args.save_loudest,
        )
        # Strip unpicklable pta objects before saving
        for pop in compact_results.get("populations", []):
            pop.pop("pta", None)
            pop.pop("psrs", None)

        save_results_dual(compact_results, save_path, save_compact_npz=False)


    if config.CGW_SNR_ANALYSIS:
        print("\n" + "="*70)
        print("CONTINUOUS WAVE SNR ANALYSIS")
        print("="*70)

        if not config.RUN_CONSISTENT_POP_SYNTH or consistent_results is None:
            raise RuntimeError(
                "CGW_SNR_ANALYSIS requires RUN_CONSISTENT_POP_SYNTH to have run first."
            )

        N_PRE_FILTER  = 1000  # candidates pre-screened by characteristic strain proxy
        N_TOP_SOURCES = 25   # loudest sources to report per population

        all_population_cgw_snrs = []
        T_obs    = 15.0 * 365.25 * 24 * 3600   # 15 years in seconds
        # Cadence: match your PTA cadence (~2 weeks for NANOGrav)
        # but coarser is fine for GWB PSD — you just need f_max >> f_GW of your highest binary
        cadence  = 14 * 24 * 3600              # 2 weeks in seconds

        time_arr = np.arange(0, Tspan_seconds, cadence)  # uniform grid, seconds ***** this needs to be fixed up I believe *****
        for pop_idx, result in enumerate(
            consistent_results["populations"]
        ):
            print(f"\n--- Population {pop_idx + 1} ---")
            population = result["population"]
            pta = result["pta"]
            psrs = result["psrs"]

            # Pre-filter by characteristic strain proxy h0 / (2 pi f)
            pre_filtered = sorted(
                population,
                key=lambda b: b.h0 / (2.0 * np.pi * b.f),
                reverse=True,
            )[:N_PRE_FILTER]

            
            pre_filter_snrs = compute_cgw_snr_optimal_population(
                psrs          = psrs,
                pta           = pta,
                population    = population,
                noise_params  = raw_noise_params,
                profile       = True,
            )

            top_sources = sorted(
                zip(pre_filtered, pre_filter_snrs),
                key=lambda x: x[1],
                reverse=True,
            )[:N_TOP_SOURCES]

            top_binaries, top_snrs = zip(*top_sources) if top_sources else ([], [])

            print(f"Top {N_TOP_SOURCES} CGW candidates:")
            for rank, (b, snr) in enumerate(zip(top_binaries, top_snrs), start=1):
                print(
                    f"  {rank:2d}. f={b.f:.2e} Hz  "
                    f"Mc={b.Mc:.2e} Msun  "
                    f"h0={b.h0:.2e}  "
                    f"SNR={snr:.3f}"
                )

            all_population_cgw_snrs.append(list(top_snrs))
        from plot_cgw_snr import plot_cgw_analysis
        plot_cgw_analysis(
            top_binaries=top_binaries,
            top_snrs=top_snrs,
            psrs=psrs,
            save_path=f"figures/cgw_analysis_{CONFIG_NAME}.pdf",
            style="dark_background",
            annotate_top=5,
        )

    # ========== NG R&G COMPARISON ==========
    if config.RUN_NG_RG_COMPARISON:
        print("\n" + "="*70)
        print("NANOGrav Rohan & Gondor COMPARISON")
        print("="*70)
        population = consistent_results["populations"]["population"]
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

    if config.OPTIMAL_SNR_POPULATION:
        population, strain_data = config.generate_population(selected_config, smbhb_module, compute_strain=True, T_obs_seconds=Tspan_seconds)

            # print("\nFinding optimal SNR population using enterprise noise model...")
            # selected_population, N_needed, final_SNR, SNR_sq_binaries = find_N_needed(
            #         population, psrs_clean, parsed_noise_params, raw_noise_params, strain_data, Tspan_seconds,
            #         target_SNR=args.target_snr, noise_method='enterprise')
            
            # # print("\nFinding optimal SNR population using analytic noise model...")
            # # selected_population, N_needed, final_SNR, SNR_sq_binaries = find_N_needed(
            # #         population, psrs_clean, parsed_noise_params, raw_noise_params, strain_data, Tspan_seconds,
            # #         target_SNR=args.target_snr, noise_method='analytic')
            # results = {
            #     'population': (selected_population[:N_needed], N_needed, final_SNR, SNR_sq_binaries),
            #         }
            # # Save results
            # save_path = os.path.join(save_dir, 'optimal_SNR_population.json')
            # save_results(results, save_path)

        
        # plot_snr_population(
        #     binaries=selected_population,
        #     SNR_sq_binaries=SNR_sq_binaries[:N_needed],
        #     psrs=psrs_clean,
        #     top_N=50,
        #     selected_binaries=selected_population,   # marks N_needed on the cumulative plot
        #     savepath='figures/snr_population_analysis.png'
        # )
        # convergence_results = convergence_test(
        #     binaries=population, pulsars=psrs_clean, pulsar_noise_params=pulsar_noise_params, strain_data=strain_data, T_obs=Tspan_seconds
        # )
    
        # plot_overlap_reduction_function(
        #     pulsars=psrs_clean, binaries=population, pulsar_noise_params=pulsar_noise_params
        # )
        # pta, model, params_complete = build_pta_and_params(
        #     psrs=psrs_clean, noise_params_15yr=noise_params, Tspan=Tspan_seconds
        # )
        # get_noise_matrix(psrs=psrs_clean, noise_params=noise_params, Tspan=Tspan_seconds)
        # save_results({
        #     'N_needed': N_needed,
        #     'final_SNR': final_SNR,
        #     'selected_population': selected_population
        # }, os.path.join(save_dir, 'optimal_snr_population.json'))

        # from debug_snr import analyze_snr_calculation_complete, diagnose_high_snr
        # results = analyze_snr_calculation_complete(
        #     population, 
        #     strain_data, 
        #     psrs_clean, 
        #     pulsar_noise_params, 
        #     Tspan_seconds,
        #     output_file='complete_breakdown.json'
        # )
        
        # Then diagnose:
        # diagnose_high_snr('complete_breakdown.json')
    if config.SNR_COMPARISON_CHOSEN_POP:

        sample_pop, strain_data = chosen_population(
            n_binaries = 1,
            chirp_mass_msun=1e10,
            mass_ratio=0.5,
            gw_frequency=5e-9,
            redshift=0.5,
            polarization=0.0,
            inclination=0.0,
            initial_phase=0.0,
            right_ascension=np.pi/4,
            declination=np.pi/4,
            compute_strain=True,
            T_obs_seconds=Tspan_seconds,
            )
        print(sample_pop)
        snr_sq, psd_interpolators = SNR_sq_all_pairs_all_binaries_vectorised(
            binaries=sample_pop,
            pulsars=psrs_clean,
            parsed_noise_params=parsed_noise_params,
            raw_noise_params=raw_noise_params,
            strain_data=strain_data,
            Tspan=Tspan_seconds
            )
        snr, ostat = compute_population_snr(
            population=sample_pop, 
            psrs_clean=psrs_clean, 
            raw_noise_params=raw_noise_params, 
            parsed_noise_params=parsed_noise_params,
            Tspan=Tspan_seconds
            )
        snr_optimal = np.sqrt(np.sum(snr_sq))
        print(f"Chosen population SNR, from enterprise: {snr:.8f}")
        print(f"Chosen population SNR, from optimal: {snr_optimal:.8f}")
        # from optimal_SNR_calc import compare_to_enterprise_os

        # results = compare_to_enterprise_os(
        #         os_obj            = ostat,
        #         raw_noise_params  = raw_noise_params,
        #         pulsars           = psrs_clean,
        #         Tspan             = Tspan_seconds,
        #         psd_interpolators = psd_interpolators,   # from inside your SNR function — return these
        #         snr_sq_arr        = snr_sq,
        #         binaries          = sample_pop,
        #         nmodes            = 301,
        #     )
    if config.PSD_COMPARISON:
        population, strain_data = config.generate_population(selected_config, smbhb_module, compute_strain=True, T_obs_seconds=Tspan_seconds)


        from optimal_SNR_calc import get_pulsar_noise_psd
        pta, model, params_complete = build_pta_and_params(
            psrs=psrs_clean, noise_params_15yr=raw_noise_params, Tspan=Tspan_seconds
        )
        psd_list = []
        for i in range(len(psrs_clean)):
            psr_name = psrs_clean[i].name
            freqs, psd = get_pulsar_noise_psd(pta, params=params_complete, pulsar_idx=i, T_span=Tspan_seconds)
            psd_list.append(psd)
            # # plt.loglog(freqs, psd)
            # # plt.show()
        # print(freqs)

        from optimal_SNR_calc import sigma_ab, sigma_ab_all_pairs
        pulsar_a = psrs_clean[10]
        pulsar_b = psrs_clean[31]
        psrs_test = np.array([pulsar_a, pulsar_b])
        # test all the pulsar pairs

        # sigma_ab_arr = []
        # for i in range(len(psrs_clean)):
        #     pulsar_a = psrs_clean[i]
        #     for j in range(i+1, len(psrs_clean)):
        #         pulsar_b = psrs_clean[j]
        #         sigma_psrs_ab = sigma_ab(pulsar_a, pulsar_b, parsed_noise_params, raw_noise_params, Tspan_seconds, nmodes=301)
        #         sigma_ab_arr.append(sigma_psrs_ab)

        sigma_ab_arr, _, noise = sigma_ab_all_pairs(psrs_clean, parsed_noise_params, raw_noise_params, Tspan_seconds, nmodes=150, psd=psd_list)
        # sigma_psrs_ab = sigma_ab(pulsar_a, pulsar_b, parsed_noise_params, raw_noise_params, Tspan_seconds, nmodes=301)
        from consistent_pop_synth import compute_population_snr
        snr, sig = compute_population_snr(population, psrs_clean, raw_noise_params, parsed_noise_params, Tspan_seconds)
        print(noise)
        
        print(sigma_ab_arr.shape, sigma_ab_arr)

        print(sig.shape, sig)

        print("sigma ratio mine/enterprise:", sigma_ab_arr/sig)
        print("median ratio:", np.median(sigma_ab_arr/sig), "mean ratio:", np.mean(sigma_ab_arr/sig))
        #     # selected_population, N_needed, final_SNR, SNR_sq_binaries = N_needed_for_population(
        #     #         population, psrs_clean, pulsar_noise_params, strain_data,
        #     #         target_SNR=args.target_snr, T_obs=Tspan_seconds )

        #     # results = compare_pulsar_psd_methods(psrs_clean, raw_noise_params, parsed_noise_params, Tspan_seconds)
        #     # for psr in psrs_clean:
        #     #     fig = plot_psd_comparison(results, pulsar_name=psr.name)                 
        #     #     plt.show()

        #     # fig = plot_psd_comparison(results, pulsar_name="B1937+21")                        # first pulsar
        #     # plt.show()
        #     # fig = plot_psd_comparison(results, pulsar_name="B1953+29")                        # first pulsar
        #     # plt.show()
        #     # fig = plot_psd_comparison(results)                        # first pulsar
        #     # plt.show()
        #     # results = test_psd_vs_residuals_consistency(
        #     #     psrs                = psrs_clean,
        #     #     parsed_noise_params = parsed_noise_params,
        #     #     raw_noise_params    = raw_noise_params,
        #     #     Tspan               = Tspan_seconds,
        #     #     nmodes              = 30,
        #     #     n_realisations      = 500,
        #     #     test_pulsar_idx     = 57,
        #     # )
        #     # Your SNR: signal / sigma
        #     # signal for pair (i,j) = Gamma_ij * sum_k P_gw(fk)  (at A=1)
        #     # then combine pairs optimally



        from enterprise_extensions.frequentist.optimal_statistic import OptimalStatistic
        pta_clean, _, params_clean = build_pta_and_params(psrs_clean, raw_noise_params, Tspan_seconds, gw_log10_A=0.0)
        ostat_clean = opt_stat.OptimalStatistic(psrs_clean, pta=pta_clean, orf='hd')
        xi_c, rho_c, sig_c, OS_c, OS_sig_c = ostat_clean.compute_os(params=params_clean)

        def hd_from_xi(xi):
            x = 0.5 * (1 - np.cos(xi))
            return 1.5 * x * np.log(x) - 0.25 * x + 0.5

        Gamma_enterprise = hd_from_xi(xi_c)

        # Recompute your Gamma for the same pairs using pulsar positions
        N = len(psrs_clean)
        ii, jj = np.tril_indices(N, k=-1)
        i_idx, j_idx = jj, ii

        Gamma_mine = np.array([
            hd_from_xi(np.arccos(np.clip(np.dot(psrs_clean[i].pos, psrs_clean[j].pos), -1, 1)))
            for i, j in zip(i_idx, j_idx)
        ])

        print("Gamma enterprise[:5]: ", Gamma_enterprise[:5])
        print("Gamma mine[:5]:       ", Gamma_mine[:5])
        print("ratio[:5]:            ", Gamma_enterprise[:5] / Gamma_mine[:5])

        # Check if the ratio sig/sigma is cleaner when you remove Gamma dependence
        # i.e. compute sigma without Gamma to isolate the integral part
        df = 1.0 / Tspan_seconds
        freqs = np.arange(1, 151) * df
        fyr = 1.0 / (86400.0 * 365.25)
        gamma = 13.0/3.0
        P_gw = (1.0/(12*np.pi**2)) * freqs**(-gamma) * fyr**(gamma-3)
        P_noise = np.array(psd_list)

        integral_all = np.sum(P_gw[None,:]**2 / (P_noise[i_idx] * P_noise[j_idx]), axis=-1) * df
        sigma_no_gamma = (2.0 * Tspan_seconds * integral_all) ** (-0.5)

        print("P_gw[:3]:         ", P_gw[:3])
        print("P_noise[0,:3]:    ", P_noise[0,:3])
        print("ratio P_gw/P_n:   ", (P_gw/P_noise[0])[:3])
        print("integrand[:3]:    ", (P_gw**2/P_noise[0]**2)[:3])


        # Get the GWB signal matrix that enterprise actually uses
        # This is S_ab(f) = Gamma_ab * P_gw(f) used inside compute_os
        phi_full = pta_clean.get_phi(params_clean)

        # The GWB contribution to phi for pulsar 0 is the diagonal
        # phi_a includes both red noise AND GWB — red noise alone is from the per-pulsar model
        # Difference = GWB contribution
        phi_0 = np.array(phi_full[0], dtype=float)
        real_mask = phi_0 < 1e30
        phi_0_real = phi_0[real_mask]

        # For Gamma_aa = 0.5 (self-overlap), GWB adds 0.5*P_gwb*df to each diagonal
        # So P_gwb = phi_gwb_contribution / (0.5 * df)
        # But we need to separate GWB from red noise — build a noise-only PTA to get phi_rn
        print("phi_0_real[:6]:  ", phi_0_real[:6])
        print("P_gw[:3]:        ", P_gw[:3])
        print("P_gw*df[:3]:     ", (P_gw*df)[:3])
        print("phi_0_real[::2] / (P_gw*df)[:3] =", phi_0_real[::2][:3] / (P_gw*df)[:3])
logger.close()



                

if __name__ == "__main__":
    main()