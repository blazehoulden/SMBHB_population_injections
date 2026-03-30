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
    'optimistic': {
        'n_binaries': 1_000,
        'mass_distribution': 'power_law',
        'mass_cutoff_0': 1e9,
        'z_max': 2.0,
        'description': 'Larger mass, higher spread in distance, small population'
    },
    'realistic': {
        'n_binaries': 20_000,
        'mass_distribution': 'exponential_damping',
        'mass_cutoff_0': 1e10,
        'z_max': 1.2,
        'description': 'Medium mass, medium spread in distance, medium population'
    },
    'pessimistic': {
        'n_binaries': 10_000_000,
        'mass_distribution': 'exponential_damping',
        'mass_cutoff_0': 10**(9),
        'z_max': 1.0,
        'description': 'Lower mass, lower spread in distance, large population'
    }
}

# Data directories
NANOGRAV_PULSARS = True

if NANOGRAV_PULSARS:
    # PAR_DIR = "./psars_narrowband/alternate/tempo2"
    # TIM_DIR = "./psars_narrowband/alternate/tim/initial"
    PAR_DIR = "./psars_narrowband/par/"
    TIM_DIR = "./psars_narrowband/tim/"
    USE_PULSAR_CACHE = True
    NANOGRAV_PULSAR_CACHE = "nanograv_pulsars_cache.pkl"
else:
    PAR_DIR = "pulsars/"
    TIM_DIR = "pulsars/"
    USE_PULSAR_CACHE = False

# Noise file
NOISEFILE = '15yr_noise.json'

# Analysis flags
RUN_INITIAL_INJECTION_ANALYSIS = False
RUN_SCALING_ANALYSIS = False
RUN_INDIVIDUAL_BINARY_ANALYSIS = False
RUN_ENSEMBLE_ANALYSIS = False
RUN_NG_RG_COMPARISON = False
RUN_CONSISTENT_POP_SYNTH = True
OPTIMAL_SNR_POPULATION = False
SNR_COMPARISON_CHOSEN_POP = False
PSD_COMPARISON = False

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


def generate_population(config, smbhb_module, compute_strain=False, T_obs_seconds=16.03 * 365.25 * 86400):
    """Generate SMBHB population with given configuration."""
    if compute_strain:

        population, strain_data = smbhb_module.generate_smbhb_population(
            n_binaries=config['n_binaries'],
            z_max=config['z_max'],
            mass_distribution=config['mass_distribution'],
            alpha_0=1.21,
            alpha_z=0.0,
            mass_min=1e7,
            mass_max=1e11,
            mass_cutoff_0=config['mass_cutoff_0'],
            mass_cutoff_z=0.0,
            compute_strain=compute_strain,
            # n_freq_bins=50,
            random_seed=None,
            T_obs_seconds=T_obs_seconds
        )
        # Convert masses if needed
        if max([b.Mc for b in population]) < 1e20:
            for binary in population:
                binary.Mc = binary.Mc * Msun
        return population, strain_data
    elif not compute_strain:
        population = smbhb_module.generate_smbhb_population(
            n_binaries=config['n_binaries'],
            mass_distribution=config['mass_distribution'],
            alpha_0=1.21,
            alpha_z=0.0,
            mass_min=1e7,
            mass_max=1e11,
            mass_cutoff_0=config['mass_cutoff_0'],
            mass_cutoff_z=0.0,
            compute_strain=compute_strain,
            # n_freq_bins=50,
            random_seed=None,
            T_obs_seconds=T_obs_seconds
        )
        strain_data = None
        # Convert masses if needed - this is needed throughout the rest of the directory, should come back and change this, but works for now
        if max([b.Mc for b in population]) < 1e20:
            for binary in population:
                binary.Mc = binary.Mc * Msun
        
        return population