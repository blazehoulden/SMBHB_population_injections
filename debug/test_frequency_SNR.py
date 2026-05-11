# copy & paste this into main
# test binaries with same properties but different frequencies
        freqs = np.linspace(1e-9, 3e-7, 3000)
        sub_populations = []
        for freq in freqs:
            pop = chosen_population(
                n_binaries=1,
                gw_frequency=freq,
                chirp_mass_msun=10**9,
                compute_strain=False,
                T_obs_seconds=Tspan_seconds,
            )
            sub_populations.append(pop)

        from debug.test_CGW_sky_loc import _concat_population_arrays, _population_arrays_to_binary_rows
        population = _concat_population_arrays(sub_populations)
    
        from consistent_pop_synth import compute_population_snr
    
        original_stoas = {psr.name: np.copy(psr.stoas[:]) for psr in psrs_clean}
        _, pta, enterprise_psrs = compute_population_snr(
            population=population,
            psrs_clean=psrs_clean,
            current_stoas=original_stoas,
            raw_noise_params=raw_noise_params,
            Tspan=Tspan_seconds,
            return_psrs_pta=True,
        )
    
        from CGW_SNR import compute_cgw_snr_optimal_population

        binary_rows = _population_arrays_to_binary_rows(population)
        cgw_snrs_optimal = compute_cgw_snr_optimal_population(
            psrs=enterprise_psrs,
            pta=pta,
            population=binary_rows,
            Tspan=Tspan_seconds,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
        )
    
        print("\nRanked CGW candidates by optimal SNR:")
        for i, (binary, snr) in enumerate(
            sorted(zip(binary_rows, cgw_snrs_optimal), key=lambda x: x[1], reverse=True)
        ):
            print(f"{i+1:2d}. SNR={snr:.2f}, freq={binary.f * 10**9:.5f} nHz")
        plt.figure(figsize=(6, 4))  # width, height in inches
        plt.plot(freqs, cgw_snrs_optimal, color='red', lw=2.0, label='Binary CGW SNR')
        plt.xlabel(r'$f$ [Hz]')
        plt.ylabel(r'SNR')          # also fixed: you had xlabel twice
        plt.xlim(1e-9, 3e-7)
        plt.xscale('log')
        yr = 365.25*86400
        plt.axvline(1/yr, color='black', linestyle='dashed', label=r'$1/\rm{yr}$')
        plt.legend(loc='best')
        plt.savefig("figures/frequency_snr_comp.png", dpi=300)
        plt.savefig("figures/frequency_snr_comp.pdf", dpi=300)