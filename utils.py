import numpy as np
import json
import pickle
import gzip
from pathlib import Path


_POP_FIELDS = ('f', 'Mc', 'Mtot', 'D_comov', 'z', 'h0', 'ra', 'dec', 'psi', 'iota', 'phi0')


def _k_smallest_indices(arr, k):
    arr = np.asarray(arr)
    n = arr.size
    if n == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    idx = np.argpartition(arr, k - 1)[:k]
    return idx[np.argsort(arr[idx])]


def _k_largest_indices(arr, k):
    arr = np.asarray(arr)
    n = arr.size
    if n == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    idx = np.argpartition(arr, n - k)[-k:]
    return idx[np.argsort(arr[idx])[::-1]]


def _estimate_max_binaries_per_sim(
    max_mb_per_sim,
    n_fields,
    float_dtype,
    json_compatible=True,
):
    budget_bytes = max(float(max_mb_per_sim), 0.1) * 1024**2
    if json_compatible:
        # Conservative estimate for JSON numbers serialized from Python lists.
        approx_bytes_per_value = 28
        per_binary_bytes = n_fields * approx_bytes_per_value
    else:
        per_binary_bytes = n_fields * np.dtype(float_dtype).itemsize
    return max(int(budget_bytes // max(per_binary_bytes, 1)), 1)


def _select_representative_indices(
    arrays,
    max_binaries,
    n_nearest=100,
    n_loudest=100,
    extreme_fields=('Mc', 'Mtot', 'D_comov', 'z', 'f', 'h0'),
):
    D = arrays.get('D_comov')
    h0 = arrays.get('h0')
    f = arrays.get('f')
    if D is None or h0 is None:
        n = len(next(iter(arrays.values()))) if arrays else 0
        return np.arange(min(n, max_binaries), dtype=np.int64), {'fallback': min(n, max_binaries)}

    n_total = D.size
    max_binaries = min(max_binaries, n_total)

    selected = []
    selected.extend(_k_smallest_indices(D, min(n_nearest, n_total)).tolist())
    selected.extend(_k_largest_indices(h0/(2 * np.pi * f), min(n_loudest, n_total)).tolist())

    for field in extreme_fields:
        arr = arrays.get(field)
        if arr is None or arr.size == 0:
            continue
        selected.append(int(np.argmin(arr)))
        selected.append(int(np.argmax(arr)))

    # Keep insertion order and uniqueness.
    selected_unique = list(dict.fromkeys(selected))

    if len(selected_unique) < max_binaries:
        need = max_binaries - len(selected_unique)
        extra = _k_largest_indices(h0/(2 * np.pi * f), min(n_total, len(selected_unique) + need)).tolist()
        for idx in extra:
            if idx not in selected_unique:
                selected_unique.append(idx)
            if len(selected_unique) >= max_binaries:
                break

    if len(selected_unique) > max_binaries:
        selected_unique = selected_unique[:max_binaries]

    stats = {
        'selected': len(selected_unique),
        'requested_nearest': int(n_nearest),
        'requested_loudest': int(n_loudest),
        'extreme_fields': list(extreme_fields),
    }
    return np.asarray(selected_unique, dtype=np.int64), stats


def compact_consistent_results_for_storage(
    results,
    max_mb_per_sim=5.0,
    n_nearest=100,
    n_loudest=10000,
    float_dtype=np.float32,
):
    """
    Build a compact, analysis-focused copy of consistent-pop results.

    Per simulation, keeps representative binaries only:
    1) nearest by D_comov
    2) loudest proxy by h0
    3) min/max extremes for key parameters
    4) fills remaining budget with additional highest-h0 binaries
    """
    if not isinstance(results, dict):
        return results

    pops = results.get('populations', [])
    if not isinstance(pops, list):
        return results

    max_bins = _estimate_max_binaries_per_sim(
        max_mb_per_sim=max_mb_per_sim,
        n_fields=len(_POP_FIELDS),
        float_dtype=float_dtype,
        json_compatible=True,
    )

    compact_pops = []
    for entry in pops:
        if not isinstance(entry, dict):
            compact_pops.append(entry)
            continue

        pop = entry.get('population')
        arrays = _population_to_array_dict(pop)
        if arrays is None or 'f' not in arrays:
            compact_pops.append(entry)
            continue

        n_total = len(arrays['f'])
        sel_idx, sel_stats = _select_representative_indices(
            arrays=arrays,
            max_binaries=max_bins,
            n_nearest=n_nearest,
            n_loudest=n_loudest,
        )

        compact_population = {
            field: np.asarray(arrays[field])[sel_idx].astype(float_dtype, copy=False).tolist()
            for field in _POP_FIELDS if field in arrays
        }

        entry_compact = {k: v for k, v in entry.items() if k != 'population'}
        entry_compact['population'] = compact_population
        entry_compact['population_storage'] = {
            'mode': 'representative_subset',
            'n_total': int(n_total),
            'n_saved': int(len(sel_idx)),
            'selected_global_indices': sel_idx.tolist(),
            'selection': sel_stats,
            'target_max_mb_per_sim': float(max_mb_per_sim),
            'float_dtype': str(np.dtype(float_dtype)),
        }
        compact_pops.append(entry_compact)

    compact = {
        **results,
        'populations': compact_pops,
    }
    compact['storage_profile'] = {
        'population_mode': 'representative_subset',
        'target_max_mb_per_sim': float(max_mb_per_sim),
        'max_binaries_per_sim_estimate': int(max_bins),
        'n_nearest': int(n_nearest),
        'n_loudest': int(n_loudest),
        'float_dtype': str(np.dtype(float_dtype)),
    }
    return compact


def save_results(results, filename):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Saved: {filename}")


def save_results_pickle(results, filename):
    """Save full Python object graph (including PopulationArrays) via pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"💾 Saved pickle: {filename}")


def save_results_pickle_gz(results, filename, compresslevel=5):
    """Save full Python objects to compressed pickle (.pkl.gz)."""
    with gzip.open(filename, 'wb', compresslevel=compresslevel) as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"💾 Saved compressed pickle: {filename}")


def save_results_dual(
    results,
    json_filename,
    pickle_filename=None,
    save_compact_npz=False,
    npz_filename=None,
    npz_float_dtype=np.float32,
):
    """
    Save JSON (human-readable) and companion compressed pickle (full-fidelity objects).

    JSON is convenient for inspection, while compressed pickle preserves full
    PopulationArrays objects for exact reloading later.

    Optionally also save compact NPZ for plotting-focused workflows.
    """
    save_results(results, json_filename)

    if pickle_filename is None:
        p = Path(json_filename)
        pickle_filename = str(p.with_suffix('.pkl.gz'))

    if str(pickle_filename).endswith('.gz'):
        save_results_pickle_gz(results, pickle_filename)
    else:
        save_results_pickle(results, pickle_filename)

    npz_out = None
    if save_compact_npz:
        if npz_filename is None:
            p = Path(json_filename)
            npz_filename = str(p.with_suffix('.npz'))
        save_results_compact_npz(results, npz_filename, float_dtype=npz_float_dtype)
        npz_out = npz_filename

    return {
        'json': json_filename,
        'pickle': pickle_filename,
        'npz': npz_out,
    }


def load_results_pickle(filename):
    """Load a full-fidelity results object previously saved with pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_results_pickle_gz(filename):
    """Load compressed pickle (.pkl.gz)."""
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def _population_to_array_dict(population):
    """
    Convert population representation to plain numpy arrays.

    Supports:
    - PopulationArrays-like objects with attributes
    - list of dict rows
    """
    keys = ('f', 'Mc', 'Mtot', 'D_comov', 'z', 'h0', 'ra', 'dec', 'psi', 'iota', 'phi0')

    # PopulationArrays-like object
    if hasattr(population, 'f') and hasattr(population, 'Mc'):
        out = {}
        for k in keys:
            if hasattr(population, k):
                arr = getattr(population, k)
                if arr is not None:
                    out[k] = np.asarray(arr)
        return out if out else None

    # list[dict] fallback
    if isinstance(population, list) and population and isinstance(population[0], dict):
        out = {}
        for k in keys:
            vals = [row.get(k, np.nan) for row in population]
            out[k] = np.asarray(vals)
        return out

    return None


def save_results_compact_npz(results, filename, float_dtype=np.float32):
    """
    Save a compact plotting-focused representation to compressed NPZ.

    This is typically much smaller than pickle and faster to load for plotting,
    but it does not preserve full Python objects/methods.

    Stored keys per simulation:
    - sim{idx}_f, sim{idx}_Mc, sim{idx}_D_comov, sim{idx}_h0, ...
    - sim{idx}_SNR_final (scalar), sim{idx}_n_binaries (scalar)
    - meta_json: JSON string with summary/config/metadata
    """
    pops = results.get('populations', []) if isinstance(results, dict) else []
    arrays_out = {}

    for i, entry in enumerate(pops):
        if not isinstance(entry, dict):
            continue

        pop = entry.get('population')
        arrs = _population_to_array_dict(pop)
        if arrs is None:
            continue

        for k, arr in arrs.items():
            arr = np.asarray(arr)
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(float_dtype, copy=False)
            arrays_out[f'sim{i}_{k}'] = arr

        if 'SNR_final' in entry:
            arrays_out[f'sim{i}_SNR_final'] = np.asarray([entry['SNR_final']], dtype=float_dtype)
        if 'SNR_achieved' in entry:
            arrays_out[f'sim{i}_SNR_achieved'] = np.asarray([entry['SNR_achieved']], dtype=float_dtype)
        if 'n_bininaries' in entry:
            arrays_out[f'sim{i}_n_binaries'] = np.asarray([entry['n_bininaries']], dtype=np.int32)
        if 'sim_index' in entry:
            arrays_out[f'sim{i}_sim_index'] = np.asarray([entry['sim_index']], dtype=np.int32)

    meta = {
        'summary_statistics': results.get('summary_statistics') if isinstance(results, dict) else None,
        'config': results.get('config') if isinstance(results, dict) else None,
        'metadata': results.get('metadata') if isinstance(results, dict) else None,
        'n_sims_serialized': len([k for k in arrays_out if k.endswith('_sim_index')]),
        'float_dtype': str(np.dtype(float_dtype)),
    }
    arrays_out['meta_json'] = np.asarray(json.dumps(meta), dtype=object)

    np.savez_compressed(filename, **arrays_out)
    print(f"💾 Saved compact NPZ: {filename}")


def load_results_compact_npz(filename):
    """
    Load compact NPZ produced by save_results_compact_npz.

    Returns numpy NpzFile; access arrays by keys like 'sim0_f'.
    """
    return np.load(filename, allow_pickle=True)


def print_population_diagnostics(population):
    """Print population diagnostics."""
    from config import Msun
    from signal_injection import strain_amplitude
    
    print("\n" + "="*70)
    print("POPULATION DIAGNOSTICS")
    print("="*70)
    
    freqs_Hz = [b.f for b in population]
    masses = [b.Mc for b in population]
    distances = [b.D_comov for b in population]
    
    print(f"\nSize: {len(population)} binaries")
    print(f"Frequency range: {min(freqs_Hz)*1e9:.2f} - {max(freqs_Hz)*1e9:.2f} nHz")
    print(f"Mass range: {min(masses)/Msun:.2e} - {max(masses)/Msun:.2e} M☉")
    print(f"Comoving distance range: {min(distances):.1f} - {max(distances):.1f} Mpc")
    
    # Check detectability
    detectable = sum(1 for b in population if 1e-9 < b.f < 1e-7)
    print(f"In detectable range (1-100 nHz): {detectable}/{len(population)}")


def print_scaling_summary(results, N_needed, target_SNR):
    """Print scaling analysis summary."""
    print("\n" + "="*70)
    print("SCALING ANALYSIS SUMMARY")
    print("="*70)
    print(f"Target SNR: {target_SNR}σ")
    print(f"SNR Range: {min(results['SNR']):.2f} - {max(results['SNR']):.2f}")
    if N_needed:
        print(f"N required: {N_needed} binaries")
    print("="*70)