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
        'N_binaries': 2_000,
        'mass_exp_damp_flag': False,
        'power_law': True,
        'm_c_con': 1e9,
        'z_max': 2.0,
        'description': 'Larger mass, higher spread in distance, small population'
    },
    'realistic': {
        'N_binaries': 120_000,
        'mass_exp_damp_flag': True,
        'power_law': False,
        'm_c_con': 1e10,
        'z_max': 1.2,
        'description': 'Medium mass, medium spread in distance, medium population'
    },
    'pessimistic': {
        'N_binaries': 2_000_000,
        'mass_exp_damp_flag': True,
        'power_law': False,
        'm_c_con': 1e9,
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

# Memory profiling
MEMORY_PROFILE_ENABLED = False

def load_smbhb_module(module_path="SMBHB_pop_synth.py"):
    """Load the SMBHB population synthesis module."""
    file_path = Path(module_path)
    module_name = "SMBHB_pop_synth"
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def generate_population(config, smbhb_module):
    """Generate SMBHB population with given configuration."""
    population = smbhb_module.generate_SMBHB_population(
        N_binaries=config['N_binaries'],
        mass_exp_damp_flag=config['mass_exp_damp_flag'],
        alpha_con=1.21,
        alpha_z=0.03,
        m_min=1e7,
        m_max=1e11,
        power_law=config['power_law'],
        m_c_con=config['m_c_con'],
        m_c_z=0.11e9,
        z_max=config['z_max'],
        rng=None
    )
    
    # Convert masses if needed
    if max([b['Mc'] for b in population]) < 1e20:
        for binary in population:
            binary['Mc'] = binary['Mc'] * Msun
    
    return population