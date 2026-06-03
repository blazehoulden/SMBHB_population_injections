import os
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import PAR_DIR

parfiles = sorted([f for f in os.listdir(PAR_DIR) if f.endswith('.par')])

for i, par in enumerate(parfiles, 1):
    print(f'[{i}/{len(parfiles)}] Running {par} ...', flush=True)
    res = subprocess.run(['python', 'scripts/run_one_psr.py', par, '5x_cadence'], capture_output=True, text=True)
    print('  exit', res.returncode)
    print('  stdout:\n', res.stdout)
    print('  stderr:\n', res.stderr)
    if res.returncode != 0:
        print('>>> Crash or failure for', par)
        break
