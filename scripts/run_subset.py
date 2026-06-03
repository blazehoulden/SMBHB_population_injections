import sys
import os
import json
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_loader import load_pulsars
from signal_injection import simulate_psr
from config import NOISEFILE

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python run_subset.py START END [SCENARIO]')
        sys.exit(2)
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    scenario = sys.argv[3] if len(sys.argv) > 3 else '5x_cadence'

    with open(NOISEFILE, 'r') as f:
        noise = json.load(f)

    psrs = load_pulsars(verbose=False, scenario=scenario)
    total = len(psrs)
    start = max(0, start)
    end = min(end, total)
    print(f'Running subset [{start}:{end}] of {total} pulsars for scenario {scenario}')

    try:
        for i in range(start, end):
            psr = psrs[i]
            name = getattr(psr, 'name', f'psr_{i}')
            print(f'  [{i+1}/{end}] {name}', flush=True)
            simulate_psr(psr, noise_dict=noise, add_WN=True, add_RN=True)
        print('COMPLETED')
        sys.exit(0)
    except Exception as e:
        print('EXCEPTION', e)
        traceback.print_exc()
        sys.exit(3)
