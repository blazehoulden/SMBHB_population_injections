pop, strain_data = chosen_population(
            n_binaries = 5,
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
snr_tot = 0
for i in range(5):
    from consistent_pop_synth import compute_population_snr
    snr = compute_population_snr(pop[:i + 1], psrs_clean, raw_noise_params, parsed_noise_params, Tspan_seconds)
    print(f"Population SNR for first {i + 1} binaries: {snr}")
    snr_i = compute_population_snr(pop[[i]], psrs_clean, raw_noise_params, parsed_noise_params, Tspan_seconds)
    print(f"SNR for binary {i}: {snr_i}")
    snr_tot += np.sqrt(snr_i**2)
    print(f"Quadrature sum of individual SNRs for first {i + 1} binaries: {snr_tot}")