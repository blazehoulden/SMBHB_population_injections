import numpy as np
from pathlib import Path
import importlib.util
import sys

# Physical constants
c = 2.99792458e8      # Speed of light [m/s]
G = 6.67430e-11       # Gravitational constant [m^3 kg^-1 s^-2]
Msun = 1.98847e30     # Solar mass [kg]
pc = 3.085677581e16   # Parsec [m]

# Population configuration presets
POPULATION_CONFIGS = {
    'test': {
        'n_binaries': 20_000,
        'mass_distribution': 'exponential_damping',
        'mass_cutoff_0': 10**(12.0),
        'z_max': 0.5,
        'description': 'Tiny population for local testing'
    },
    'pessimistic': {
        'n_binaries': 200_000_000,
        'mass_distribution': 'exponential_damping',
        'mass_cutoff_0': 10**(8.7),
        'z_max': 2.0,
        'description': 'Lower mass, higher population size'
    },
    'realistic': {
        'n_binaries': 20_000_000,
        'mass_distribution': 'exponential_damping',
        'mass_cutoff_0': 10**(9.0),
        'z_max': 2.0,
        'description': 'Medium mass, medium population size'
    },
    'optimistic': {
        'n_binaries': 2_250_000,
        'mass_distribution': 'exponential_damping',
        'mass_cutoff_0': 10**(9.3),
        'z_max': 2.0,
        'description': 'Higher mass, lower population size'
    }
}

# Data directories
NANOGRAV_PULSARS = False
MEERKAT_PULSARS = True

if NANOGRAV_PULSARS:
    # PAR_DIR = "./psars_narrowband/alternate/tempo2"
    # TIM_DIR = "./psars_narrowband/alternate/tim/initial"
    PAR_DIR = "./psars_narrowband/par/"
    TIM_DIR = "./psars_narrowband/tim/"
    USE_PULSAR_CACHE = True
    PULSAR_CACHE = "nanograv_pulsars_cache.pkl"
    NOISEFILE = '15yr_noise.json'

elif MEERKAT_PULSARS:
    PAR_DIR = "meerkat_partim/"
    TIM_DIR = "meerkat_partim/"
    USE_PULSAR_CACHE = False
    PULSAR_CACHE = None
    NOISEFILE = 'meerkat_45yr_noise.json'
else:
    PAR_DIR = "pulsars/"
    TIM_DIR = "pulsars/"
    USE_PULSAR_CACHE = False
    NOISEFILE = '15yr_noise.json'

# Analysis flags
# RUN_INITIAL_INJECTION_ANALYSIS = False
# RUN_SCALING_ANALYSIS = False
# RUN_INDIVIDUAL_BINARY_ANALYSIS = False
# RUN_ENSEMBLE_ANALYSIS = False
RUN_NG_RG_COMPARISON = False
RUN_CONSISTENT_POP_SYNTH = True
# OPTIMAL_SNR_POPULATION = False
# SNR_COMPARISON_CHOSEN_POP = False
# PSD_COMPARISON = False
GEN_POP = False
CGW_SNR_ANALYSIS = True
MAKE_SENSITIVITY_CURVES = False

# Memory profiling
MEMORY_PROFILE_ENABLED = True

def load_smbhb_module(module_path="SMBHB_pop_synth.py"):
    """Load the SMBHB population synthesis module."""
    file_path = Path(module_path)
    module_name = "SMBHB_pop_synth"
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def generate_population(config, smbhb_module, T_obs_seconds, n_binaries = None, compute_strain=False, seed=None, f_obs_min=2e-9):
    """
    Generate SMBHB population with given configuration. 
    Set minimum frequency of population to f_obs_min -- we leave it as 2 nHz for simplicity.
    NANOGrav's lowest frequency is 1/T_obs ~ 2 nHz, while MPTA's is ~7 nHz for 4.5yrs, and ~3.5 nHz for 6yrs.
    2 nHz allows for a slight buffer at the lowest frequency, and avoids requiring large numbers of binaries.
    """
    if compute_strain:

        population, strain_data = smbhb_module.generate_smbhb_population(
            n_binaries=n_binaries if n_binaries is not None else config['n_binaries'],
            z_max=config['z_max'],
            mass_distribution=config['mass_distribution'],
            alpha_0=1.21,
            alpha_z=0.03,
            mass_min=10**(7.5),
            mass_max=10**(12.5),
            mass_cutoff_0=config['mass_cutoff_0'],
            mass_cutoff_z=0.0,
            compute_strain=compute_strain,
            # n_freq_bins=50,
            random_seed=seed,
            T_obs_seconds=T_obs_seconds,
            f_obs_min=f_obs_min
        )
        # Convert masses if needed
        if max([b.Mc for b in population]) < 1e20:
            for binary in population:
                binary.Mc = binary.Mc * Msun
        return population, strain_data
    elif not compute_strain:
        population = smbhb_module.generate_smbhb_population(
            n_binaries=n_binaries if n_binaries is not None else config['n_binaries'],
            mass_distribution=config['mass_distribution'],
            z_max=config['z_max'],
            alpha_0=1.21,
            alpha_z=0.03,
            mass_min=10**(7.5),
            mass_max=10**(12.5),
            mass_cutoff_0=config['mass_cutoff_0'],
            mass_cutoff_z=0.0,
            compute_strain=compute_strain,
            # n_freq_bins=50,
            random_seed=seed,
            T_obs_seconds=T_obs_seconds,
            f_obs_min=f_obs_min
        )
        strain_data = None
        # Convert masses if needed - this is needed throughout the rest of the directory, should come back and change this, but works for now
        if max([b.Mc for b in population]) < 1e20:
            for binary in population:
                binary.Mc = binary.Mc * Msun
        
        return population