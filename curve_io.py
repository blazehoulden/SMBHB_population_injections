"""
curve_io.py

Save/load helpers shared by sensitivity_curves.py and test_CGW_sky_loc.py.

DELIBERATELY has no dependency on hasasia, enterprise, data_loader, or
SMBHB_pop_synth -- only numpy, json, os. Those heavy packages are what make
the actual analysis slow to import/run; this module exists specifically so
population_analysis.ipynb (or anything else that just wants to re-plot
already-computed results) can load the saved arrays without dragging any of
that in.

sensitivity_curves.py and test_CGW_sky_loc.py both import their save_*
functions from here rather than defining them locally, so there's exactly
one on-disk format for each, used by both the writer and the reader.
"""

import os
import json
import numpy as np


# ---------------------------------------------------------------------------
# Sensitivity curves (sensitivity_curves.py <-> population_analysis.ipynb)
# ---------------------------------------------------------------------------

def save_curves(curves, path, labels=None, baseline_name=None, colors=None,
                 default_scenario_labels=None):
    """
    Save a {scenario_name: (sc, dsc)} dict (as returned by
    make_pta_sensitivity) to disk as a single .npz + a metadata .json
    sidecar with the same stem.

    Only `freqs` and `h_c` are pulled off `dsc` (DeterSensitivityCurve) and
    `sc` (GWBSensitivityCurve) -- those are the only two per-frequency
    products the plotting code actually uses. The `sc`/`dsc` objects
    themselves aren't saved -- just the arrays.
    """
    path = str(path)
    if not path.endswith('.npz'):
        path += '.npz'
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    default_scenario_labels = default_scenario_labels or {}

    arrays = {}
    for name, (sc, dsc) in curves.items():
        arrays[f'{name}__freqs']  = np.asarray(dsc.freqs)
        arrays[f'{name}__hc_det'] = np.asarray(dsc.h_c)
        arrays[f'{name}__hc_gwb'] = np.asarray(sc.h_c)
    np.savez_compressed(path, **arrays)

    meta = {
        'scenario_names': list(curves.keys()),
        'labels':         labels or {name: default_scenario_labels.get(name, name) for name in curves},
        'baseline_name':  baseline_name,
        'colors':         colors or {},
    }
    meta_path = path[:-4] + '.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Saved {len(curves)} scenario curves to {path}')
    print(f'Saved metadata to {meta_path}')
    return path, meta_path


def load_curves(path):
    """
    Inverse of save_curves(). Returns (arrays, meta):
      arrays: {scenario_name: {'freqs': ..., 'hc_det': ..., 'hc_gwb': ...}}
      meta:   {'scenario_names', 'labels', 'baseline_name', 'colors'}
    """
    path = str(path)
    if not path.endswith('.npz'):
        path += '.npz'
    meta_path = path[:-4] + '.json'

    npz = np.load(path)
    with open(meta_path) as f:
        meta = json.load(f)

    arrays = {}
    for name in meta['scenario_names']:
        arrays[name] = {
            'freqs':  npz[f'{name}__freqs'],
            'hc_det': npz[f'{name}__hc_det'],
            'hc_gwb': npz[f'{name}__hc_gwb'],
        }
    return arrays, meta


# ---------------------------------------------------------------------------
# Sky-location CGW SNR (test_CGW_sky_loc.py <-> population_analysis.ipynb)
# ---------------------------------------------------------------------------

def save_sky_snr_data(population, cgw_snrs_optimal, enterprise_psrs, path):
    """
    Save everything the skymap plot needs to a single .npz:
      - binary ra/dec (+ the other PopulationArrays fields, for reference)
      - cgw_snrs_optimal
      - pulsar sky positions (phi, theta) used for the star markers
    """
    path = str(path)
    if not path.endswith('.npz'):
        path += '.npz'
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    psr_phi   = np.array([float(psr.phi) for psr in enterprise_psrs])
    psr_theta = np.array([float(psr.theta) for psr in enterprise_psrs])

    np.savez_compressed(
        path,
        ra=population.ra, dec=population.dec, f=population.f,
        Mc=population.Mc, Mtot=population.Mtot, D_comov=population.D_comov,
        z=population.z, h0=population.h0, psi=population.psi,
        iota=population.iota, phi0=population.phi0,
        cgw_snr_input=population.cgw_snr,
        cgw_snrs_optimal=np.asarray(cgw_snrs_optimal, dtype=float),
        psr_phi=psr_phi, psr_theta=psr_theta,
    )
    print(f'Saved sky-SNR data ({len(population.ra)} binaries, '
          f'{len(psr_phi)} pulsars) to {path}')
    return path


def load_sky_snr_data(path):
    """
    Inverse of save_sky_snr_data(). Returns a dict of plain numpy arrays:
    ra, dec, f, Mc, Mtot, D_comov, z, h0, psi, iota, phi0,
    cgw_snrs_optimal, psr_phi, psr_theta.
    """
    path = str(path)
    if not path.endswith('.npz'):
        path += '.npz'
    npz = np.load(path)
    return {key: npz[key] for key in npz.files}