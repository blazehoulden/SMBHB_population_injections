"""
Iterate over pulsars in a scenario and run simulate_psr(psr, noise_dict, add_WN=True, add_RN=True)
Logs progress to `logs/find_crash_psr.log`. Run under your `smbhb312` env.
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_loader import load_pulsars
from signal_injection import simulate_psr
from config import NOISEFILE

scenario = '5x_cadence'
log_path = os.path.join('logs', 'find_crash_psr.log')
os.makedirs('logs', exist_ok=True)

with open(NOISEFILE, 'r') as f:
    noise_dict = json.load(f)

print('Loading pulsars for scenario:', scenario)
psrs = load_pulsars(verbose=True, scenario=scenario)
print('Loaded', len(psrs), 'pulsars')

with open(log_path, 'w') as log:
    for i, psr in enumerate(psrs):
        name = getattr(psr, 'name', f'psr_{i}')
        print(f'[{i+1}/{len(psrs)}] Running simulate_psr for', name)
        log.write(f'[{time.ctime()}] START {name}\n')
        log.flush()
        try:
            simulate_psr(psr, noise_dict=noise_dict, add_WN=True, add_RN=True)
            print('  OK')
            log.write(f'[{time.ctime()}] OK {name}\n')
            log.flush()
        except Exception as e:
            print('  FAILED with exception:', e)
            log.write(f'[{time.ctime()}] EXC {name} {e}\n')
            log.flush()
        # small pause to ensure logs flush before potential crash
        time.sleep(0.1)

print('Done. See', log_path)
