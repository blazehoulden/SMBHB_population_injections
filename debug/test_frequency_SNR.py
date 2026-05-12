"""
Lightweight smoke test for frequency-varying CGW SNR.

This test is skippable when project dependencies are not available
(e.g., running outside the intended conda env). It verifies the
population-building path for several frequencies at a small scale.
"""
import pytest
import numpy as np

try:
    from SMBHB_pop_synth import chosen_population
    from debug.test_CGW_sky_loc import _concat_population_arrays
    HAS_DEPS = True
except Exception:
    HAS_DEPS = False


@pytest.mark.skipif(not HAS_DEPS, reason="requires SMBHB_pop_synth and debug helpers")
def test_frequency_snr_smoke():
    freqs = np.linspace(1e-9, 3e-7, 10)
    sub_populations = []
    for freq in freqs:
        pop = chosen_population(
            n_binaries=1,
            gw_frequency=freq,
            chirp_mass_msun=1e9,
            compute_strain=False,
            T_obs_seconds=1.0,
        )
        sub_populations.append(pop)

    population = _concat_population_arrays(sub_populations)

    # Basic sanity: number of binaries equals number of frequencies
    assert len(population) == len(freqs)