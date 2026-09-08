#!/usr/bin/env python3
"""
check_toa_integration_times.py

Diagnostic: does your real PTA dataset carry per-TOA integration/dwell time
info (e.g. a "-tobs" flag), and does it vary meaningfully across pulsars?
This determines whether the telescope-time budget in PATCH 7 needs to be
weighted by real per-pulsar dwell time, or whether "1 unit = 1 visit" is a
fine approximation.

Run against your real data:

    python check_toa_integration_times.py

Edit PAR_DIR / TIM_DIR below if they're not importable from your config.py.
"""

import os
import statistics as stats
from collections import defaultdict, Counter

try:
    from config import TIM_DIR
except ImportError:
    TIM_DIR = '/path/to/tim'   # <-- edit if config.py isn't importable here

_SUFFIXES = ('ao', 'gbt', 'vla', 'fast')

# Candidate flag names that typically encode per-TOA dwell/integration time.
# 'tobs' is the standard NANOGrav convention (seconds); the rest are here in
# case your files use a different naming scheme.
TIME_LIKE_KEYS = ('tobs', 'length', 'integ', 'inttime', 'exposure',
                   'exp', 'dur', 'duration', 'obslen')


def base_pulsar_name(stem: str) -> str:
    base = stem
    for sfx in _SUFFIXES:
        if base.endswith(sfx):
            return base[:-len(sfx)]
    return base


def parse_toa_flags(tim_path):
    """
    Yield (toa_mjd, flags_dict) for every TOA line in a tempo2 .tim file.
    Standard format: FILE FREQ TOA TOAERR SITE [-flag value]...
    """
    with open(tim_path, 'r', errors='ignore') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(('C', 'FORMAT', '*')):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                toa = float(parts[2])
            except ValueError:
                continue

            flags = {}
            rest = parts[5:]
            i = 0
            while i < len(rest) - 1:
                key, val = rest[i], rest[i + 1]
                if key.startswith('-'):
                    flags[key[1:].lower()] = val
                    i += 2
                else:
                    i += 1
            yield toa, flags


def main():
    if not os.path.isdir(TIM_DIR):
        print(f'ERROR: TIM_DIR not found or not a directory: {TIM_DIR}')
        print('Edit TIM_DIR at the top of this script to point at your real .tim files.')
        return

    tim_files = sorted(f for f in os.listdir(TIM_DIR) if f.endswith('.tim'))
    print(f'Scanning {len(tim_files)} .tim files in {TIM_DIR}...\n')

    all_flag_keys   = Counter()
    example_values  = {}
    per_pulsar_vals = defaultdict(list)                    # base -> [float,...]
    per_pulsar_backend_vals = defaultdict(lambda: defaultdict(list))  # base -> backend -> [float,...]
    n_toas_total          = 0
    n_toas_with_time_flag = 0

    for fname in tim_files:
        stem = fname.replace('.tim', '').split('_')[0]
        base = base_pulsar_name(stem)
        fpath = os.path.join(TIM_DIR, fname)

        for toa, flags in parse_toa_flags(fpath):
            n_toas_total += 1
            for k, v in flags.items():
                all_flag_keys[k] += 1
                example_values.setdefault(k, v)

            found_key = next((k for k in TIME_LIKE_KEYS if k in flags), None)
            if found_key is not None:
                try:
                    val = float(flags[found_key])
                except ValueError:
                    continue
                per_pulsar_vals[base].append(val)
                backend = flags.get('be') or flags.get('backend') or flags.get('f') or 'unknown'
                per_pulsar_backend_vals[base][backend].append(val)
                n_toas_with_time_flag += 1

    print('=' * 72)
    print('ALL FLAG KEYS SEEN (name: count, example value)')
    print('=' * 72)
    for k, count in all_flag_keys.most_common():
        print(f'  -{k:<15s} count={count:<8d} example="{example_values[k]}"')

    print()
    print('=' * 72)
    print(f'TIME-LIKE FLAGS: {n_toas_with_time_flag}/{n_toas_total} TOAs carry '
          f'one of {TIME_LIKE_KEYS}')
    print('=' * 72)

    if not per_pulsar_vals:
        print('No time-like flag found in this dataset — the .tim files do not '
              'record per-TOA integration/dwell time under any of the names '
              'this script checks. If your files use a different flag name, '
              'add it to TIME_LIKE_KEYS above and rerun. Otherwise, the "1 '
              'unit = 1 visit" budget assumption in PATCH 7 is the best '
              'available without an external source for real dwell times '
              '(e.g. an observing log).')
        return

    print(f'\n{"pulsar":<16s} {"n_toas":>8s} {"mean":>10s} {"median":>10s} '
          f'{"std":>10s} {"min":>8s} {"max":>8s}')
    overall_means = []
    for base in sorted(per_pulsar_vals):
        vals = per_pulsar_vals[base]
        mean = stats.mean(vals)
        overall_means.append(mean)
        print(f'{base:<16s} {len(vals):>8d} {mean:>10.1f} '
              f'{stats.median(vals):>10.1f} '
              f'{(stats.stdev(vals) if len(vals) > 1 else 0.0):>10.1f} '
              f'{min(vals):>8.1f} {max(vals):>8.1f}')

    print()
    spread = (max(overall_means) / min(overall_means)) if min(overall_means) > 0 else float('inf')
    print(f'Per-pulsar mean dwell time ranges {min(overall_means):.1f} to '
          f'{max(overall_means):.1f} (units as given by the flag — likely '
          f'seconds) — a {spread:.1f}x spread across pulsars.')
    if spread > 1.5:
        print('⚠ Meaningful spread — the telescope-time budget in PATCH 7 '
              'should be weighted by each pulsar\'s real mean dwell time, '
              'not treated as 1 uniform unit per pulsar/visit. See PATCH 8 '
              'below once you confirm this.')
    else:
        print('✓ Dwell times are fairly uniform across pulsars — the "1 unit '
              '= 1 visit" simplification in PATCH 7 is a reasonable '
              'approximation as-is.')

    print()
    print('=' * 72)
    print('WITHIN-PULSAR BACKEND VARIATION (top 10 pulsars by n_toas)')
    print('=' * 72)
    top_pulsars = sorted(
        per_pulsar_backend_vals,
        key=lambda b: sum(len(v) for v in per_pulsar_backend_vals[b].values()),
        reverse=True,
    )[:10]
    for base in top_pulsars:
        print(f'\n  {base}:')
        for backend, vals in sorted(per_pulsar_backend_vals[base].items()):
            print(f'    {backend:<20s} n={len(vals):<6d} mean={stats.mean(vals):.1f}')


if __name__ == '__main__':
    main()