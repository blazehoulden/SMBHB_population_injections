import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_loader import load_single_pulsar
from signal_injection import simulate_psr
from config import PAR_DIR, NOISEFILE

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run_one_psr.py PARFILE [SCENARIO]')
        sys.exit(2)
    parfile = sys.argv[1]
    scenario = sys.argv[2] if len(sys.argv) > 2 else '5x_cadence'

    with open(NOISEFILE, 'r') as f:
        noise = json.load(f)

    psr = load_single_pulsar(parfile, scenario=scenario, verbose=True)
    if psr is None:
        print('Failed to load', parfile)
        sys.exit(3)

    try:
        simulate_psr(psr, noise_dict=noise, add_WN=True, add_RN=True)
        print('OK')
        sys.exit(0)
    except Exception as e:
        print('EXC', e)
        raise
