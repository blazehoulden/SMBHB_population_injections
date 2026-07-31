#!/usr/bin/env python3
"""
Stage 1 — Population synthesis + TOA delta computation.

For each simulation chunk (Slurm array task) the work is divided into
N_SUB_CHUNKS sub-chunks.  Pulsars are loaded **once per scenario per task**
and reused across every sub-chunk — this is the dominant speedup relative
to the per-chunk-per-scenario original design.

Memory strategy (this revision)
────────────────────────────────
The previous sub-chunk revision loaded *all* synthetic-scenario pulsar
sets up front and held them in memory simultaneously (`syn_psrs`), plus
the baseline set, for the entire sub-chunk loop. For configs with several
synthetic scenarios (especially cadence-boosted ones, which carry several×
more TOAs per pulsar) this multiplies peak RSS by roughly
`1 + n_synthetic_scenarios`.

This revision instead processes **one PTA's worth of pulsars at a time**:

    1. Generate ALL sub-chunk populations up front (cheap — these are
       small `PopulationArrays`, tens of MB total even for chunk_size
       ~10^6, with no per-pulsar amplitude arrays yet).
    2. Load baseline pulsars ONCE. Run baseline NUFFT for every sub-chunk
       (baseline pulsars are kept in memory because they're needed again
       for the population filter step at the end).
    3. For each synthetic scenario, ONE AT A TIME: load its pulsars, run
       NUFFT for every sub-chunk, then delete that scenario's pulsars
       before loading the next.
    4. Filter + write shards for every sub-chunk using the (still-loaded)
       baseline pulsars, then free them.

Total pulsar loads and NUFFT calls are IDENTICAL to the previous revision
(one load per scenario, one NUFFT per (scenario, sub-chunk) pair) — only
the loop order changes, so wall time is essentially unchanged. Peak RSS
drops from "baseline + all synthetic scenarios" to "baseline + at most one
synthetic scenario".

Sub-chunk file naming
─────────────────────
  populations/   subpop_{chunk_id:03d}_{sub_id:03d}.pkl.gz
  stoas/         chunk_{chunk_id:04d}_{sub_id:04d}[_<scenario>].npz

Stage 2 discovers shards via ShardedPickleStore.available(), which returns
(chunk_id, sub_id) tuples so it can glob the new pattern. The TOA delta
filenames follow the same two-index scheme so stage 2 can reconstruct them
from the shard index.

Arguments added (carried over from the previous revision)
───────────────────────────────────────────────────────────
  --n-sub-chunks   number of sub-chunks per Slurm array task  [1]
                   chunk_size is divided equally across sub-chunks.
                   Must divide chunk_size evenly.

  --sub-chunk-id   process a single sub-chunk only (0-based).
                   When omitted all sub-chunks are run in sequence.
                   Useful for manual reruns or finer Slurm arrays.

Optional: frequency/mass/redshift histogram (--freq-mass-hist)
─────────────────────────────────────────────────────────────
Stage 1's population filter step keeps only proxy-ranked CGW candidates +
parameter extremes — correct for the CW/SGWB analysis, but it means the
TRUE frequency/mass/redshift distribution of "all binaries generated"
doesn't survive anywhere on disk once filtering runs. Passing
--freq-mass-hist writes a compact 3D histogram (f x log10(Mtot) x z) of
each sub-chunk's FULL population batch — computed BEFORE that sub-chunk
gets filtered away — as its OWN file:

    metadata/freq_mass_z_hist/{chunk_id:03d}_{sub_id:03d}.npz

One file per sub-chunk (not merged) so stage 2 can later delete exactly
the sub-chunks it decides are "inactive" (via phase_handoff.json's
active_shard_ids — see _prune_inactive_freq_mass_hist in stage2_inject.py),
the same way it already prunes unused population shards. This is for
reproducing population-distribution plots (e.g. Fig 9 of Agarwal et al.
2026) from the real, unbiased population — NOT from the filtered shards.
Off by default: zero cost, zero behavior change, unless explicitly
requested.
"""

import argparse
import gc
import gzip
import json
import os
import pickle
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from consistent_pop_synth import suppress_enterprise_warnings
from data_loader import (
    load_pulsars,
    load_single_pulsar,
    filter_pulsars_15yr,
    SCENARIOS as DEFAULT_SCENARIOS,
)
from signal_injection import change_in_TOAs_days_population_nufft, _antenna_response_vec, _get_psr_radec
from SMBHB_pop_synth import PopulationArrays
from debug.test_CGW_sky_loc import sky_sensitivity_weight


# =============================================================================
# Memory logging helper
# =============================================================================

def _log_mem(label: str) -> None:
    """Best-effort RSS logging — silently does nothing if psutil is absent."""
    try:
        import psutil
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        print(f'  [mem] {label}: {rss_mb:,.0f} MB RSS')
    except Exception:
        pass


# =============================================================================
# ShardedPickleStore  (two-index: chunk × sub-chunk)
# =============================================================================

FIELD_DTYPES: Dict[str, type] = {
    "f":        np.float32,
    "Mc":       np.float32,
    "Mtot":     np.float32,
    "D_comov":  np.float32,
    "z":        np.float32,
    "h0":       np.float32,
    "ra":       np.float16,
    "dec":      np.float16,
    "psi":      np.float16,
    "iota":     np.float16,
    "phi0":     np.float16,
    "cgw_snr":  np.float16,
}
SCALAR_FIELDS = list(FIELD_DTYPES.keys())
N_PRE_FILTER_PER_CHUNK = 12_500


class ShardedPickleStore:
    """
    One pkl.gz per (chunk_id, sub_id) pair stored in a single directory.

    File pattern:  subpop_{chunk_id:03d}_{sub_id:03d}.pkl.gz

    available() returns a sorted list of (chunk_id, sub_id) int tuples.
    """

    def __init__(self, directory: str, compress_level: int = 6):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.compress_level = compress_level

    # ── Public API ────────────────────────────────────────────────────────────

    def write(self, chunk_id: int, sub_id: int, pop: PopulationArrays) -> None:
        compact = self._downcast(pop)
        self._dump(self._path(chunk_id, sub_id), compact)

    def read(self, chunk_id: int, sub_id: int) -> PopulationArrays:
        with gzip.open(self._path(chunk_id, sub_id), "rb") as f:
            return pickle.load(f)

    def update(
        self,
        chunk_id: int,
        sub_id: int,
        h0:        Optional[np.ndarray] = None,
        D_comov:   Optional[np.ndarray] = None,
        z:         Optional[np.ndarray] = None,
        cgw_snr:   Optional[np.ndarray] = None,
        cgw_proxy: Optional[np.ndarray] = None,
        amp_A:     Optional[Dict[str, np.ndarray]] = None,
        amp_B:     Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        pop = self.read(chunk_id, sub_id)
        if h0        is not None: pop.h0        = h0.astype(np.float32)
        if D_comov   is not None: pop.D_comov   = D_comov.astype(np.float32)
        if z         is not None: pop.z         = z.astype(np.float32)
        if cgw_snr   is not None: pop.cgw_snr   = cgw_snr.astype(np.float32)
        if cgw_proxy is not None: pop.cgw_proxy = cgw_proxy.astype(np.float32)
        if amp_A is not None:
            for psr, arr in amp_A.items():
                pop.amp_A[psr] = arr.astype(np.float32)
        if amp_B is not None:
            for psr, arr in amp_B.items():
                pop.amp_B[psr] = arr.astype(np.float32)
        self._dump(self._path(chunk_id, sub_id), pop)

    def available(self):
        results = []
        pattern = re.compile(r"subpop_(\d{3})_(\d{3})\.pkl\.gz$")

        for p in self.dir.glob("subpop_*.pkl.gz"):
            m = pattern.match(p.name)
            if m:
                results.append((int(m.group(1)), int(m.group(2))))

        return sorted(results)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _path(self, chunk_id: int, sub_id: int) -> Path:
        return self.dir / f"subpop_{chunk_id:03d}_{sub_id:03d}.pkl.gz"

    def _dump(self, path: Path, obj) -> None:
        with gzip.open(path, "wb", compresslevel=self.compress_level) as f:
            pickle.dump(obj, f, protocol=5)

    @staticmethod
    def _downcast(pop: PopulationArrays) -> PopulationArrays:
        for name, dtype in FIELD_DTYPES.items():
            if hasattr(pop, name):
                setattr(pop, name, getattr(pop, name).astype(dtype))
        for psr in list(pop.amp_A):
            pop.amp_A[psr] = pop.amp_A[psr].astype(np.float32)
        for psr in list(pop.amp_B):
            pop.amp_B[psr] = pop.amp_B[psr].astype(np.float32)
        return pop


# =============================================================================
# Pulsar geometry helpers  (unchanged)
# =============================================================================

def _get_psr_radec(psr):
    if hasattr(psr, 'pars'):
        try:
            pars = psr.pars()
            if 'RAJ' in pars and 'DECJ' in pars:
                return psr['RAJ'].val, psr['DECJ'].val
            elif 'ELONG' in pars and 'ELAT' in pars:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                coord = SkyCoord(
                    lon=psr['ELONG'].val * u.rad,
                    lat=psr['ELAT'].val  * u.rad,
                    frame='geocentricmeanecliptic'
                )
                return coord.icrs.ra.rad, coord.icrs.dec.rad
        except Exception:
            pass
    if hasattr(psr, '_raj') and hasattr(psr, '_decj'):
        return psr._raj, psr._decj
    raise AttributeError(
        f"Cannot extract RA/Dec from pulsar object of type {type(psr)}. "
        f"Expected RAJ/DECJ or ELONG/ELAT params, or _raj/_decj attributes."
    )


def _antenna_response_vec(psr_ra, psr_dec, ra_arr, dec_arr, psi_arr):
    cos_dec = np.cos(dec_arr)
    sin_dec = np.sin(dec_arr)
    cos_ra  = np.cos(ra_arr)
    sin_ra  = np.sin(ra_arr)
    cos_psi = np.cos(psi_arr)
    sin_psi = np.sin(psi_arr)

    cos_psr_dec = np.cos(psr_dec)
    sin_psr_dec = np.sin(psr_dec)
    cos_psr_ra  = np.cos(psr_ra)
    sin_psr_ra  = np.sin(psr_ra)

    omega_hat = np.array([
        -cos_dec * cos_ra,
        -cos_dec * sin_ra,
        -sin_dec,
    ])
    p_hat = np.array([
        cos_psr_dec * cos_psr_ra,
        cos_psr_dec * sin_psr_ra,
        sin_psr_dec,
    ])
    m_hat = np.array([
        sin_ra,
        -cos_ra,
        np.zeros(len(ra_arr)),
    ])
    n_hat = np.array([
        -sin_dec * cos_ra,
        -sin_dec * sin_ra,
         cos_dec,
    ])

    m_rot = cos_psi * m_hat + sin_psi * n_hat
    n_rot = -sin_psi * m_hat + cos_psi * n_hat

    denom = 1.0 + np.dot(p_hat, omega_hat)
    p_m   = np.dot(p_hat, m_rot)
    p_n   = np.dot(p_hat, n_rot)

    Fp = 0.5 * (p_m**2 - p_n**2) / denom
    Fx = (p_m * p_n) / denom

    return Fp, Fx


# =============================================================================
# Analytic proxy SNR  (unchanged)
# =============================================================================

def _compute_analytic_proxy(pop, psrs, Nvecs=None, Tspan_seconds=None):
    if Nvecs is not None:
        inv_Nvec_sums = np.array([np.sum(1.0 / Nvec) for Nvec in Nvecs])
    else:
        inv_Nvec_sums = []
        for psr in psrs:
            toaerrs = psr.toaerrs
            if np.median(toaerrs) > 1e-3:
                toaerrs = toaerrs * 1e-6
            inv_Nvec_sums.append(np.sum(1.0 / toaerrs**2))
        inv_Nvec_sums = np.array(inv_Nvec_sums)

    psr_coords = [_get_psr_radec(psr) for psr in psrs]

    if hasattr(pop, 'f'):
        f   = np.asarray(pop.f,   dtype=np.float64)
        h0  = np.asarray(pop.h0,  dtype=np.float64)
        ci  = np.cos(np.asarray(pop.iota, dtype=np.float64))
        ra  = np.asarray(pop.ra,  dtype=np.float64)
        dec = np.asarray(pop.dec, dtype=np.float64)
        psi = np.asarray(pop.psi, dtype=np.float64)
    else:
        f   = np.array([b.f    for b in pop])
        h0  = np.array([b.h0   for b in pop])
        ci  = np.cos(np.array([b.iota for b in pop]))
        ra  = np.array([b.ra   for b in pop])
        dec = np.array([b.dec  for b in pop])
        psi = np.array([b.psi  for b in pop])

    norm = h0 / (2.0 * np.pi * f)
    Aamp = norm * (1.0 + ci**2)
    Bamp = norm * (-2.0 * ci)

    if Tspan_seconds is not None:
        f_T   = f * Tspan_seconds
        phase = 2.0 * np.pi * f_T
        sinc_term = np.where(
            phase > 1e-6,
            np.sin(phase) / phase,
            1.0 - phase**2 / 6.0
        )
        time_avg = np.clip(0.5 * (1.0 - sinc_term), 0.0, 0.5)
    else:
        time_avg = 0.5

    antenna_sum = np.zeros(len(f))
    for (psr_ra, psr_dec), inv_Nvec_sum in zip(psr_coords, inv_Nvec_sums):
        Fp, Fx = _antenna_response_vec(psr_ra, psr_dec, ra, dec, psi)
        A = Fp * Aamp
        B = Fx * Bamp
        antenna_sum += (A**2 + B**2) * inv_Nvec_sum

    rho_sq      = time_avg * antenna_sum
    sky_weights = sky_sensitivity_weight(ra, dec).astype(np.float64)
    return np.sqrt(rho_sq) * sky_weights


# =============================================================================
# Population filter  (unchanged)
# =============================================================================

def _filter_population_extremes(
    pop: PopulationArrays,
    psrs,
    n_keep: int = 100,
    n_total: int = None,
    n_sub_chunks: int = 1,
    Tspan_seconds: Optional[float] = None,
    extra_tspans_seconds: Optional[List[float]] = None,
) -> tuple:
    """
    Filter down to the top-N proxy-ranked + parameter/regime-extreme
    binaries, evaluated at `Tspan_seconds` (the real baseline — this is
    also what gets STORED as pop.cgw_proxy, since stage 2's baseline CW
    candidate ranking expects proxies computed at the real Tspan).

    If `extra_tspans_seconds` is given, the SAME top-N proxy pass and
    low/mid-f rescue are ALSO run at each of those Tspans, and the
    resulting indices are UNIONED into the keep-set — so a binary that's
    only loud/resolvable at a longer (forecast) Tspan isn't dropped just
    because it didn't rank at the real baseline. Their proxy VALUES are
    not stored; only their INDICES widen what gets kept.
    """
    n = len(pop)

    if n_total is None:
        n_total = N_PRE_FILTER_PER_CHUNK // n_sub_chunks
    n_total = min(n_total, n)
    n_keep  = min(n_keep, n // 2)

    indices = set()

    def _proxy_topn_and_rescue(tspan):
        """Run the top-N proxy pass + low/mid-f rescue at one Tspan.
        Returns the proxy array (only meaningful/used for the primary,
        real-baseline call) and updates `indices` in place."""
        proxies = _compute_analytic_proxy(pop, psrs, Nvecs=None,
                                          Tspan_seconds=tspan)
        n_cgw = min(n_total, n)
        if n_cgw == n:
            top = np.argsort(proxies)[::-1]
        else:
            top = np.argpartition(proxies, -n_cgw)[-n_cgw:]
            top = top[np.argsort(proxies[top])[::-1]]
        indices.update(top.tolist())

        if tspan is not None:
            f_T = np.asarray(pop.f, dtype=np.float64) * tspan
            for lo, hi in ((0.0, 10.0), (10.0, 50.0)):
                mask = (f_T >= lo) & (f_T < hi)
                if mask.sum() == 0:
                    continue
                regime_idx = np.where(mask)[0]
                top_h0 = regime_idx[np.argsort(pop.h0[regime_idx])[-n_keep:]]
                indices.update(top_h0.tolist())

        return proxies, n_cgw

    # ── primary pass: real baseline Tspan — THIS is what gets stored ────────
    proxies, n_cgw = _proxy_topn_and_rescue(Tspan_seconds)

    # ── parameter extremes — Tspan-independent, do once ─────────────────────
    for arr in (pop.f, pop.D_comov, pop.h0, pop.Mc, pop.Mtot):
        order = np.argsort(arr)
        indices.update(order[:n_keep].tolist())
        indices.update(order[-n_keep:].tolist())

    # ── extra pass(es): widen the keep-set for longer (forecast) Tspans ────
    n_before_extra = len(indices)
    if extra_tspans_seconds:
        for extra_tspan in extra_tspans_seconds:
            _proxy_topn_and_rescue(extra_tspan)
    n_rescued_extra = len(indices) - n_before_extra

    idx = np.array(sorted(indices))
    extra_note = (f' + {n_rescued_extra:,} rescued for forecast Tspan'
                  if extra_tspans_seconds else '')
    print(
        f"  Population filtered: {n:,} → {len(idx):,} kept "
        f"(top-{n_cgw} proxy @ real Tspan + "
        f"{n_before_extra - n_cgw:,} parameter/regime extremes"
        f"{extra_note})"
    )
    return pop[idx], proxies[idx]


# =============================================================================
# Frequency / mass / redshift histogram  (optional, --freq-mass-hist)
# =============================================================================
# Preserves the TRUE (unfiltered) population distribution — everything
# above discards ~99% of the generated binaries down to CGW-candidate-
# biased extremes, which would badly distort a population-distribution
# plot (e.g. reproducing Fig 9 of Agarwal et al. 2026). Each histogram is
# tiny (~140K cells) regardless of population size, since only bin counts
# are stored, never per-binary data.
#
# ONE FILE PER SUB-CHUNK (not merged into a running sum): this lets stage 2
# delete exactly the sub-chunks it decides are "inactive" — via
# phase_handoff.json's active_shard_ids, the same authoritative source
# already used to prune unused population shards — without needing to
# parse log output or re-derive anything after the fact. As a bonus, since
# no two sub-chunks ever share a file, there's no read-merge-write race to
# guard against either.
#
# Fixed bin edges across every sim/run so histograms combine cleanly:
#   f:    1nHz - 1000nHz, 60 log bins
#   Mtot: log10(Mtot/Msun) 7.5-10.6, 0.1 dex bins (fine — coarser bins
#         requested at plot time are built by summing these)
#   z:    0-3, 60 bins (covers every POPULATION_CONFIGS z_max; a config
#         with smaller z_max just has zero counts above its own range)

_HIST_F_BINS    = np.logspace(-9, -6, 61)
_HIST_MTOT_BINS = np.arange(7.5, 10.61, 0.1)
_HIST_Z_BINS    = np.linspace(0.0, 3.0, 61)

def _accumulate_and_save_freq_mass_z_hist(
    population_batch, sim_out_dir: str, chunk_id: int, sub_id: int
) -> None:
    """
    Histogram this sub-chunk's FULL (unfiltered) population batch over
    (f, log10(Mtot), z) and write it to its own file:
    metadata/freq_mass_z_hist/{chunk_id:03d}_{sub_id:03d}.npz

    Mtot is stored in solar masses directly in PopulationArrays (confirmed
    via its docstring — no unit conversion needed here).

    Written atomically (temp file + os.replace) and verified by reading the
    file back and checking counts.sum() matches len(population_batch) before
    the temp file is trusted — so a partial/failed write can never silently
    end up looking like a valid shard on disk.
    """
    f        = np.asarray(population_batch.f,    dtype=np.float64)
    mtot_log = np.log10(np.asarray(population_batch.Mtot, dtype=np.float64))
    z        = np.asarray(population_batch.z,     dtype=np.float64)
    n_binaries = len(f)

    if not (len(mtot_log) == n_binaries and len(z) == n_binaries):
        raise ValueError(
            f'[{chunk_id:03d}_{sub_id:03d}] Mismatched array lengths: '
            f'f={n_binaries}, Mtot={len(mtot_log)}, z={len(z)}'
        )

    counts, _ = np.histogramdd(
        np.column_stack([f, mtot_log, z]),
        bins=[_HIST_F_BINS, _HIST_MTOT_BINS, _HIST_Z_BINS],
    )

    hist_dir = os.path.join(sim_out_dir, 'metadata', 'freq_mass_z_hist')
    os.makedirs(hist_dir, exist_ok=True)
    out_path = os.path.join(hist_dir, f'{chunk_id:03d}_{sub_id:03d}.npz')

    # Each (chunk_id, sub_id) writes its own file — never shared with any
    # other sub-chunk — but still write atomically (temp + replace) for
    # general safety against e.g. a killed job leaving a partial file.
    #
    # IMPORTANT: tmp_path must itself end in '.npz'. np.savez() silently
    # APPENDS '.npz' to whatever path you give it if the path doesn't
    # already end that way — so a suffix like '.npz.tmp' causes savez to
    # write to a completely different stray path ('*.npz.tmp.npz') while
    # leaving the intended tmp_path an empty, untouched 0-byte file. That
    # empty file then gets os.replace()'d into out_path, producing a
    # silently-corrupt (0-byte) final shard. Ending the suffix in '.npz'
    # guarantees np.savez writes to exactly the path we pass it.
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=hist_dir, prefix=f'.{chunk_id:03d}_{sub_id:03d}.tmp.', suffix='.npz')
    os.close(tmp_fd)

    try:
        np.savez(tmp_path, counts=counts,
                 f_bins=_HIST_F_BINS, mtot_log_bins=_HIST_MTOT_BINS,
                 z_bins=_HIST_Z_BINS)

        # Verify before trusting the write: re-open the temp file and
        # confirm it's a real npz with the expected keys and a count total
        # matching the population size. Catches truncated writes, silent
        # np.savez path-mangling (like the bug above), disk-full errors
        # that don't raise, etc. — anything that would otherwise produce a
        # shard that "exists" but is wrong or unreadable.
        with np.load(tmp_path) as check:
            required_keys = {'counts', 'f_bins', 'mtot_log_bins', 'z_bins'}
            missing = required_keys - set(check.files)
            if missing:
                raise IOError(
                    f'[{chunk_id:03d}_{sub_id:03d}] Written file is missing '
                    f'keys after save: {missing}'
                )
            written_total = check['counts'].sum()
            if written_total != n_binaries:
                raise IOError(
                    f'[{chunk_id:03d}_{sub_id:03d}] Round-trip count mismatch: '
                    f'wrote {n_binaries} binaries but read back '
                    f'{written_total:.0f} from {tmp_path}'
                )

        os.replace(tmp_path, out_path)

    except Exception:
        # Never leave a bad temp file lying around, and never let a bad
        # write masquerade as a successful shard at out_path.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f'  ✓ Wrote + verified freq/mass/z hist: {out_path} '
          f'({n_binaries:,} binaries)')


# =============================================================================
# TOA delta I/O  (two-index naming, unchanged)
# =============================================================================

def _delta_filename(chunk_id: int, sub_id: int, scenario: Optional[str] = None) -> str:
    """
    chunk_{chunk_id:04d}_{sub_id:04d}[_<scenario>].npz
    e.g.  chunk_0002_0005.npz  or  chunk_0002_0005_5x_cadence.npz
    """
    suffix = f'_{scenario}' if scenario else ''
    return f'chunk_{chunk_id:04d}_{sub_id:04d}{suffix}.npz'


def _save_toa_deltas(
    delta_stoas,
    out_dir: str,
    chunk_id: int,
    sub_id: int,
    scenario: Optional[str] = None,
) -> None:
    stoa_dir = os.path.join(out_dir, 'stoas')
    os.makedirs(stoa_dir, exist_ok=True)
    outpath  = os.path.join(stoa_dir, _delta_filename(chunk_id, sub_id, scenario))
    save_dict = {
        psr_name: delta.astype(np.float64)
        for psr_name, delta in delta_stoas
    }
    np.savez(outpath, **save_dict)
    print(f'  Saved Δstoas ({len(save_dict)} pulsars) → {outpath}')


def _max_scenario_extension_days(scenarios_dict: dict, real_span_days: float) -> float:
    """
    Longest forecast extension (in days) across every scenario in
    scenarios_dict, so the population can be generated with a frequency
    floor low enough for whichever scenario needs the longest Tspan.

    Mirrors the same extension_years-takes-precedence-else-extension_fraction
    logic used everywhere else (_pulsar_needs_scenario_tim,
    _scenario_params_cache_key, etc.) so this stays consistent with how
    each scenario's actual forecast length gets resolved elsewhere.
    """
    max_ext_days = 0.0
    for label, cfg in scenarios_dict.items():
        ext_years = cfg.get('extension_years', None)
        ext_frac  = cfg.get('extension_fraction', 0.0)
        ext_days  = (ext_years * 365.25 if ext_years is not None
                     else ext_frac * real_span_days)
        max_ext_days = max(max_ext_days, ext_days)
    return max_ext_days

# =============================================================================
# Seeding helper  (unchanged)
# =============================================================================

def _make_rng_for_sub(
    noise_seed_base: int,
    n_sims: int,
    sim_id: int,
    n_chunks: int,
    chunk_id: int,
    n_sub_chunks: int,
    sub_id: int,
) -> Tuple[np.random.Generator, int]:
    """
    Deterministic, independent seed for each (sim, chunk, sub_chunk) triple.
    Returns (rng, seed_int).
    """
    root_sq  = np.random.SeedSequence(noise_seed_base)
    sim_seq  = root_sq.spawn(max(n_sims, sim_id + 1))[sim_id]
    chunk_seq = sim_seq.spawn(n_chunks)[chunk_id]
    sub_seq  = chunk_seq.spawn(n_sub_chunks)[sub_id]
    rng      = np.random.default_rng(sub_seq)
    seed_int = rng.integers(2**31).item()
    return rng, seed_int


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Stage 1: population synthesis (memory-efficient sub-chunk edition)')
    p.add_argument('--config', '-c', default='optimistic',
                   choices=list(config.POPULATION_CONFIGS.keys()))
    p.add_argument('--target-snr',     type=float, default=4.0)
    p.add_argument('--snr-range',      nargs=2, type=float, default=[3.5, 4.0])
    p.add_argument('--output-dir',     type=str, required=True)
    p.add_argument('--chunk-size',     type=int, default=1_000_000,
                   help='Total binaries per Slurm task. '
                        'Divided equally across --n-sub-chunks.')
    p.add_argument('--n-chunks',       type=int, default=10,
                   help='Number of Slurm array tasks per sim (used to decode task_id).')
    p.add_argument('--n-sub-chunks',   type=int, default=1,
                   help='Number of sub-chunks per Slurm array task. '
                        'chunk_size must be divisible by this value. '
                        'Pulsars are loaded once (per scenario) and reused '
                        'across sub-chunks.')
    p.add_argument('--sub-chunk-id',   type=int, default=None,
                   help='Process only this sub-chunk (0-based). '
                        'When omitted, all sub-chunks are run in sequence.')
    p.add_argument('--task-id',        type=int, default=None,
                   help='Flat array task ID (overrides $SLURM_ARRAY_TASK_ID).')
    p.add_argument('--sim-id',         type=int, required=True)
    p.add_argument('--noise-seed-base', type=int, default=None)
    p.add_argument('--n-sims',         type=int, default=1,
                   help='Total number of simulations (for seeding purposes).')
    p.add_argument(
        '--synthetic-pta-config',
        type=str,
        default=None,
        help='JSON dict of synthetic PTA scenario definitions. '
             'If omitted, the three default scenarios are used when '
             '--synthetic-ptas is set.',
    )
    p.add_argument(
        '--synthetic-ptas',
        action='store_true',
        default=False,
        help='Enable synthetic PTA scenarios alongside baseline.',
    )
    p.add_argument(
        '--n-keep-extremes',
        type=int,
        default=20,
        help='Per-parameter (and per-frequency-regime) top/bottom count kept '
             'in addition to the proxy-ranked candidates. Each of '
             '{f, D_comov, h0, Mc, Mtot} contributes up to 2x this many '
             '(top + bottom), plus up to 1x this many from each of the '
             'low-f and mid-f rescue regimes — so up to ~12x this value '
             'before deduplication/overlap with the proxy set. '
             'Lower this to shrink the "parameter/regime extremes" count. '
             '[20]',
    )
    p.add_argument(
        '--freq-mass-hist',
        action='store_true',
        default=True,
        help='Write a per-sub-chunk (frequency x log10(Mtot) x redshift) '
             'histogram of the FULL, unfiltered population to '
             'metadata/freq_mass_z_hist/{chunk:03d}_{sub:03d}.npz — the '
             'only surviving record of the true population distribution '
             'once filtering runs. Off by default: zero cost, zero '
             'behavior change, unless explicitly requested. Stage 2 '
             'automatically prunes the files for any sub-chunk it decides '
             'is inactive. Used to reproduce population-distribution plots '
             '(e.g. Fig 9 of Agarwal et al. 2026) from the unbiased '
             'population rather than the CGW-candidate-biased filtered '
             'shards.',
    )
    return p.parse_args()


# =============================================================================
# Step 1 — generate all sub-chunk populations up front
# =============================================================================

def _generate_population_batches(
    *,
    Tspan_seconds:    float,
    sub_ids:         List[int],
    chunk_id:        int,
    sim_id:          int,
    args,
    sub_chunk_size:  int,
    selected_config,
    smbhb_module,
) -> List[PopulationArrays]:
    """
    Generate one PopulationArrays per sub-chunk.

    These are small (no per-pulsar amplitude arrays at this stage — just
    the 12 scalar fields × sub_chunk_size), so holding all of them in
    memory simultaneously for the rest of this task is cheap (tens of MB
    even at chunk_size ~10^6) and lets every scenario's NUFFT reuse the
    exact same binaries without regenerating or re-seeding.
    """
    batches = []
    for sub_id in sub_ids:
        _, seed_int = _make_rng_for_sub(
            noise_seed_base=args.noise_seed_base,
            n_sims=args.n_sims,
            sim_id=sim_id,
            n_chunks=args.n_chunks,
            chunk_id=chunk_id,
            n_sub_chunks=args.n_sub_chunks,
            sub_id=sub_id,
        )
        pop_batch = config.generate_population(
            config=selected_config,
            smbhb_module=smbhb_module,
            T_obs_seconds=Tspan_seconds,
            n_binaries=sub_chunk_size,
            seed=seed_int,
        )
        print(f'  ✓ sub-chunk {sub_id:03d}: generated {len(pop_batch):,} binaries '
              f'(seed={seed_int})')
        batches.append(pop_batch)
    return batches


# =============================================================================
# Step 2/3 — run NUFFT for one scenario's pulsars across all sub-chunks
# =============================================================================

def _run_nufft_for_scenario(
    *,
    scenario_label: str,
    is_baseline:    bool,
    psrs,
    population_batches: List[PopulationArrays],
    sub_ids:         List[int],
    chunk_id:        int,
    sim_out_dir:     str,
    t_nufft_ref:     List[float],   # mutable accumulator [total_nufft_time]
) -> None:
    """
    Compute and save Δstoas for ONE scenario's pulsar set, for every
    sub-chunk's population batch.

    `psrs` is whatever pulsar set is currently loaded (baseline or one
    synthetic scenario) — this function does not load or free pulsars,
    it just consumes whatever the caller has in memory right now.
    """
    for sub_id, pop_batch in zip(sub_ids, population_batches):
        t_n0 = time.time()
        delta_stoas = change_in_TOAs_days_population_nufft(
            psrs, pop_batch, verbose=False)
        t_nufft_ref[0] += time.time() - t_n0

        _save_toa_deltas(
            delta_stoas, sim_out_dir, chunk_id, sub_id,
            scenario=None if is_baseline else scenario_label,
        )
        del delta_stoas
        gc.collect()


# =============================================================================
# Step 4 — filter + write shard for one sub-chunk
# =============================================================================

def _filter_and_write_shard(
    *,
    sub_id:               int,
    chunk_id:              int,
    population_batch:      PopulationArrays,
    psrs_baseline,
    Tspan_seconds:         float,
    pop_dir:               str,
    args,
    extra_tspans_seconds:  Optional[List[float]] = None,
) -> None:
    population_batch, proxy_scores = _filter_population_extremes(
        population_batch,
        n_keep=args.n_keep_extremes,
        n_sub_chunks=args.n_sub_chunks,
        psrs=psrs_baseline,
        Tspan_seconds=Tspan_seconds,
        extra_tspans_seconds=extra_tspans_seconds,
    )

    store = ShardedPickleStore(pop_dir)
    store.write(chunk_id, sub_id, population_batch)
    store.update(chunk_id, sub_id, cgw_proxy=proxy_scores)
    print(f'  ✓ Shard subpop_{chunk_id:03d}_{sub_id:03d}.pkl.gz written')


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t0   = time.time()

    # ── Validate sub-chunk args ───────────────────────────────────────────────
    if args.chunk_size % args.n_sub_chunks != 0:
        sys.exit(
            f'ERROR: --chunk-size ({args.chunk_size}) must be divisible '
            f'by --n-sub-chunks ({args.n_sub_chunks}).'
        )
    sub_chunk_size = args.chunk_size // args.n_sub_chunks

    # ── Decode task_id → chunk_id ─────────────────────────────────────────────
    task_id = args.task_id
    if task_id is None:
        env_val = os.environ.get('SLURM_ARRAY_TASK_ID')
        if env_val is None:
            sys.exit('ERROR: --task-id not set and $SLURM_ARRAY_TASK_ID undefined.')
        task_id = int(env_val)

    sim_id   = args.sim_id
    chunk_id = task_id   # task_id == chunk_id (0..N_CHUNKS-1)

    # Sub-chunk range for this task
    if args.sub_chunk_id is not None:
        if not (0 <= args.sub_chunk_id < args.n_sub_chunks):
            sys.exit(
                f'ERROR: --sub-chunk-id {args.sub_chunk_id} out of range '
                f'[0, {args.n_sub_chunks - 1}].'
            )
        sub_ids = [args.sub_chunk_id]
    else:
        sub_ids = list(range(args.n_sub_chunks))

    # Per-simulation output directory
    sim_out_dir = os.path.join(args.output_dir, f'sim{sim_id:03d}')
    pop_dir     = os.path.join(sim_out_dir, 'populations')
    meta_dir    = os.path.join(sim_out_dir, 'metadata')
    os.makedirs(pop_dir,  exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    print(f'\n{"="*62}')
    print(f'Stage 1 — task_id={task_id}  sim_id={sim_id}  chunk_id={chunk_id}')
    print(f'  config={args.config}  chunk_size={args.chunk_size:,}')
    print(f'  n_sub_chunks={args.n_sub_chunks}  sub_chunk_size={sub_chunk_size:,}')
    print(f'  sub_ids to process: {sub_ids}')
    print(f'  freq_mass_hist={args.freq_mass_hist}')
    print(f'  output → {sim_out_dir}')
    print(f'{"="*62}')

    # ── Resolve scenario configs ──────────────────────────────────────────────
    if args.synthetic_ptas:
        if args.synthetic_pta_config:
            syn_scenarios = json.loads(args.synthetic_pta_config)
        else:
            syn_scenarios = {
                k: v for k, v in DEFAULT_SCENARIOS.items()
                if k != 'baseline'
            }
        print(f'  Synthetic scenarios: {list(syn_scenarios.keys())}')
    else:
        syn_scenarios = {}

    parfiles = sorted([f for f in os.listdir(config.PAR_DIR) if f.endswith('.par')])

    combined_scenarios = dict(DEFAULT_SCENARIOS)
    combined_scenarios.update(syn_scenarios)

    t_load_total  = 0.0
    t_nufft_total = [0.0]   # mutable accumulator

    selected_config = config.POPULATION_CONFIGS[args.config]
    smbhb_module    = config.load_smbhb_module()


    # =========================================================================
    # STEP 1 — baseline pulsars: loaded once, kept until the filter step
    # =========================================================================
    t_l0 = time.time()
    print(f'\n📡 Loading baseline pulsars...')
    psrs_unfiltered = load_pulsars(
        verbose=True,
        scenario='baseline',
        scenarios=combined_scenarios,
    )
    with suppress_enterprise_warnings():
        psrs_baseline, _, Tspan_seconds = filter_pulsars_15yr(
            psrs_unfiltered, verbose=True)
    del psrs_unfiltered
    gc.collect()

    t_load_total += time.time() - t_l0
    print(f'✓ {len(psrs_baseline)} baseline pulsars, '
          f'Tspan={Tspan_seconds / (365.25 * 86400):.1f} yr')
    _log_mem('after loading baseline pulsars')

    # ── Frequency floor must cover the LONGEST scenario Tspan, not just
    #    the real baseline — otherwise low-frequency binaries that only
    #    become resolvable/loud with the extra forecast years are never
    #    sampled in the first place, regardless of how the PTA's own
    #    sensitivity improves with more observing time.
    real_span_days          = Tspan_seconds / 86400.0
    max_extension_days      = _max_scenario_extension_days(
        combined_scenarios, real_span_days)
    population_gen_tspan_seconds = Tspan_seconds + max_extension_days * 86400.0
    if max_extension_days > 0:
        print(f'  ℹ Longest scenario extension: {max_extension_days/365.25:.2f} yr '
              f'→ population generated with T_obs={population_gen_tspan_seconds/(365.25*86400):.2f} yr '
              f'(f_obs_min={1.0/population_gen_tspan_seconds*1e9:.3f} nHz), '
              f'not the real {Tspan_seconds/(365.25*86400):.2f} yr baseline')


    # =========================================================================
    # STEP 2 — generate every sub-chunk's population batch up front
    # =========================================================================
    # Cheap and small (no amplitude arrays yet) — generating once here lets
    # every scenario below reuse the *same* binaries without re-seeding.
    print(f'\n🌌 Generating {len(sub_ids)} sub-chunk population batch(es)...')
    population_batches = _generate_population_batches(
        Tspan_seconds=population_gen_tspan_seconds,
        sub_ids=sub_ids,
        chunk_id=chunk_id,
        sim_id=sim_id,
        args=args,
        sub_chunk_size=sub_chunk_size,
        selected_config=selected_config,
        smbhb_module=smbhb_module,
    )
    _log_mem('after generating population batches')


    # =========================================================================
    # STEP 3 — baseline NUFFT for every sub-chunk
    # =========================================================================
    print(f'\n⚡ Baseline NUFFT ({len(sub_ids)} sub-chunk(s))...')
    _run_nufft_for_scenario(
        scenario_label='baseline',
        is_baseline=True,
        psrs=psrs_baseline,
        population_batches=population_batches,
        sub_ids=sub_ids,
        chunk_id=chunk_id,
        sim_out_dir=sim_out_dir,
        t_nufft_ref=t_nufft_total,
    )

    # =========================================================================
    # STEP 4 — synthetic scenarios, ONE PTA's pulsars in memory at a time
    # =========================================================================
    for scenario_label in syn_scenarios:
        t_l0 = time.time()
        print(f'\n📡 Loading pulsars for scenario "{scenario_label}"...')
        kept_psrs = []
        for par in parfiles:
            psr_variants = load_single_pulsar(
                par,
                verbose=False,
                scenario=scenario_label,
                scenarios=combined_scenarios,
            )
            for psr in psr_variants:
                with suppress_enterprise_warnings():
                    kept, _, _ = filter_pulsars_15yr([psr], verbose=False)
                if kept:
                    kept_psrs.extend(kept)
                del psr
                gc.collect()
        t_load_total += time.time() - t_l0
        print(f'✓ {len(kept_psrs)} pulsars loaded for "{scenario_label}"')
        _log_mem(f'after loading "{scenario_label}" pulsars')

        print(f'\n⚡ NUFFT → scenario={scenario_label} '
              f'({len(sub_ids)} sub-chunk(s))...')
        _run_nufft_for_scenario(
            scenario_label=scenario_label,
            is_baseline=False,
            psrs=kept_psrs,
            population_batches=population_batches,
            sub_ids=sub_ids,
            chunk_id=chunk_id,
            sim_out_dir=sim_out_dir,
            t_nufft_ref=t_nufft_total,
        )

        # Free this scenario's pulsars before the next one is loaded —
        # this is the key memory saving vs. the previous revision.
        del kept_psrs
        gc.collect()
        _log_mem(f'after freeing "{scenario_label}" pulsars')

    # =========================================================================
    # STEP 5 — histogram (optional) + filter + write shards
    # =========================================================================
    print(f'\n💾 Filtering and writing {len(sub_ids)} shard(s)...')
    extra_tspans = ([population_gen_tspan_seconds]
                     if max_extension_days > 0 else None)
    for i, sub_id in enumerate(sub_ids):
        if args.freq_mass_hist:
            # MUST run before _filter_and_write_shard — that call discards
            # ~99% of population_batches[i] down to CGW-candidate-biased
            # extremes. This is the only chance to see the full population.
            _accumulate_and_save_freq_mass_z_hist(
                population_batches[i], sim_out_dir, chunk_id, sub_id)

        _filter_and_write_shard(
            sub_id=sub_id,
            chunk_id=chunk_id,
            population_batch=population_batches[i],
            psrs_baseline=psrs_baseline,
            Tspan_seconds=Tspan_seconds,
            pop_dir=pop_dir,
            args=args,
            extra_tspans_seconds=extra_tspans,
        )
        # Free each sub-chunk's population batch as soon as it's written.
        population_batches[i] = None
        gc.collect()

    # Baseline pulsars no longer needed.
    del psrs_baseline, population_batches
    gc.collect()
    _log_mem('after freeing baseline pulsars')

    # ── Write metadata (chunk 0, sub-chunk 0 wins; content is idempotent) ────
    config_path = os.path.join(meta_dir, 'config.json')
    if (chunk_id == 0 and 0 in sub_ids) or not os.path.exists(config_path):
        config_json = {
            'config':               args.config,
            'target_snr':           args.target_snr,
            'snr_range':            args.snr_range,
            'n_chunks':             args.n_chunks,
            'n_sub_chunks':         args.n_sub_chunks,
            'chunk_size':           args.chunk_size,
            'sub_chunk_size':       sub_chunk_size,
            'Tspan_seconds':        Tspan_seconds,
            'synthetic_scenarios':  list(syn_scenarios.keys()),
            'freq_mass_hist':       args.freq_mass_hist,
        }
        with open(config_path, 'w') as fh:
            json.dump(config_json, fh, indent=2)
        print('\n✓ Wrote metadata/config.json')

    elapsed = time.time() - t0
    print(f'\n  Pulsar loading total:  {t_load_total/60:.1f} min')
    print(f'  NUFFT injection total: {t_nufft_total[0]/60:.1f} min')
    n_done = len(sub_ids)
    print(f'\n✅ Stage 1 chunk={chunk_id} (sim={sim_id})  '
          f'{n_done} sub-chunk(s) complete in {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()