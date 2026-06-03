"""
Find minimal prefix length of pulsar list that reproduces the crash when processed in a single process.
Uses `scripts/run_subset.py START END` in subprocesses to test ranges.

Usage:
    python find_crash_by_bsearch.py [scenario]

Output:
    Prints the minimal index where running pulsars[0:index] crashes.
"""
import os
import subprocess
import sys
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import PAR_DIR

scenario = sys.argv[1] if len(sys.argv) > 1 else '5x_cadence'
# get total pulsars by listing PAR_DIR
parfiles = sorted([f for f in os.listdir(PAR_DIR) if f.endswith('.par')])
total = len(parfiles)
print('Total parfiles:', total)

# helper to run prefix 0..k (exclusive)
def test_prefix(k):
    if k <= 0:
        return False, 'k<=0'
    print(f'Testing prefix length {k}...')
    res = subprocess.run(['python', 'scripts/run_subset.py', '0', str(k), scenario], capture_output=True, text=True)
    print('  exit', res.returncode)
    print('  stdout last lines:\n', '\n'.join(res.stdout.splitlines()[-10:]))
    print('  stderr last lines:\n', '\n'.join(res.stderr.splitlines()[-10:]))
    failed = res.returncode != 0
    return failed, res

# doubling search to find an upper bound where failure occurs
low = 0
high = 1
while high <= total:
    failed, _ = test_prefix(high)
    if failed:
        break
    low = high
    high = min(total, high * 2)
    if high == total:
        # test full
        failed, _ = test_prefix(high)
        break

if high > total:
    print('No failure found up to full list')
    sys.exit(0)

# binary search between low and high (low passes, high fails)
while high - low > 1:
    mid = (low + high) // 2
    failed, _ = test_prefix(mid)
    if failed:
        high = mid
    else:
        low = mid

print(f'Minimal failing prefix length: {high} (1-based count)')
print('Offending parfile:', parfiles[high-1])
