import os
import time
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_loader import load_single_pulsar
from config import PAR_DIR
from signal_injection import simulate_psr

psr_name = 'J0030+0451'
scenario = '5x_cadence'

# Check scenario tim file
tim_path = os.path.join('scenario_tims', scenario, f'{psr_name}.tim')
print('tim_path=', tim_path)
if os.path.exists(tim_path):
    mtime = time.ctime(os.path.getmtime(tim_path))
    print('tim mtime:', mtime)
    with open(tim_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 10:
                print('TIM>', line.rstrip())
            else:
                break
else:
    print('tim file not found')

print('\nLoading pulsar via data_loader.load_single_pulsar...')
# find par file for this pulsar
par_candidates = [p for p in os.listdir(PAR_DIR) if p.startswith(psr_name)]
if not par_candidates:
    print('No .par file found for', psr_name, 'in', PAR_DIR)
    sys.exit(1)
parfile = par_candidates[0]
print('Using parfile:', parfile)
psr = load_single_pulsar(parfile, scenario=scenario, verbose=True)
if psr is None:
    print('load_single_pulsar returned None — failed to load', parfile)
    sys.exit(1)
print('Loaded psr:', psr.name, 'nobs=', getattr(psr,'nobs',None))

print('\nRunning simulate_psr with no noise (dry run of make_ideal_nofit)')
try:
    simulate_psr(psr, noise_dict={}, add_WN=False, add_RN=False)
    print('simulate_psr(dry) completed OK')
except Exception as e:
    print('simulate_psr(dry) raised exception:', e)

print('\nRunning simulate_psr with RN enabled (may reproduce crash)')
try:
    simulate_psr(psr, noise_dict={}, add_WN=False, add_RN=True)
    print('simulate_psr(RN) completed OK')
except Exception as e:
    print('simulate_psr(RN) raised exception:', e)
