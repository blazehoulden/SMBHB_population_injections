#!/usr/bin/env python3
"""
Stage 1 — Population synthesis + TOA delta computation.

For each simulation chunk:
  1. Load pulsars for each enabled scenario (one at a time, freed after use).
  2. Generate one chunk of binaries.
  3. Compute per-pulsar TOA deltas via NUFFT for every scenario.
  4. Save deltas:  <output_dir>/sim{sim_id}/stoas/chunk_{chunk_id:04d}[_<scenario>].npz
  5. Filter and save population shard.
  6. Write metadata.

Synthetic PTA scenarios are driven by --synthetic-pta-config, which accepts
a JSON string of scenario definitions (same format as data_loader.SCENARIOS).
Pass an empty dict '{}' or omit the flag to run baseline only.

Memory strategy
───────────────
Pulsars for each scenario are loaded, used to compute TOA deltas, and
immediately deleted before the next scenario is processed. Only the baseline
pulsars are retained (briefly) for the population pre-filter. The large
population_batch is also deleted after the filter step.
"""

import argparse
import gc
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from consistent_pop_synth import suppress_enterprise_warnings
from data_loader import load_pulsars, load_single_pulsar, filter_pulsars_15yr, SCENARIOS as DEFAULT_SCENARIOS
from signal_injection import change_in_TOAs_days_population_nufft, _antenna_response_vec, _get_psr_radec
from SMBHB_pop_synth import PopulationArrays
from debug.test_CGW_sky_loc import sky_sensitivity_weight


# =============================================================================
# ShardedPickleStore
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
    """One pkl.gz per chunk stored in a single directory."""

    def __init__(self, directory: str, compress_level: int = 6):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.compress_level = compress_level

    def write(self, idx: int, pop: PopulationArrays) -> None:
        compact = self._downcast(pop)
        self._dump(self._path(idx), compact)

    def read(self, idx: int) -> PopulationArrays:
        with gzip.open(self._path(idx), "rb") as f:
            return pickle.load(f)

    def update(
        self,
        idx: int,
        h0:        Optional[np.ndarray] = None,
        D_comov:   Optional[np.ndarray] = None,
        z:         Optional[np.ndarray] = None,
        cgw_snr:   Optional[np.ndarray] = None,
        cgw_proxy: Optional[np.ndarray] = None,
        amp_A:     Optional[Dict[str, np.ndarray]] = None,
        amp_B:     Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        pop = self.read(idx)
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
        self._dump(self._path(idx), pop)

    def available(self) -> list:
        return sorted(
            int(p.name.split("_")[1].split(".")[0])
            for p in self.dir.glob("subpop_*.pkl.gz")
        )

    def _path(self, idx: int) -> Path:
        return self.dir / f"subpop_{idx:03d}.pkl.gz"

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
# Pulsar geometry helpers
# =============================================================================

def _get_psr_radec(psr):
    """
    Extract (ra, dec) in radians from either a libstempo or Enterprise pulsar object.
    Tries RAJ/DECJ first, falls back to ELONG/ELAT ecliptic coords,
    then falls back to _raj/_decj (Enterprise).
    """
    # --- libstempo path ---
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
            pass  # fall through to _raj/_decj

    # --- Enterprise path (or libstempo fallback) ---
    if hasattr(psr, '_raj') and hasattr(psr, '_decj'):
        return psr._raj, psr._decj

    raise AttributeError(
        f"Cannot extract RA/Dec from pulsar object of type {type(psr)}. "
        f"Expected RAJ/DECJ or ELONG/ELAT params, or _raj/_decj attributes."
    )


def _antenna_response_vec(psr_ra, psr_dec, ra_arr, dec_arr, psi_arr):
    """
    Vectorised antenna response over N binaries.
    Returns Fp_arr, Fx_arr each of shape (N,).
    Avoids redundant trig via sin(pi/2 - x) = cos(x) identities.
    """
    cos_dec = np.cos(dec_arr)        # (N,) = sin(src_polar)
    sin_dec = np.sin(dec_arr)        # (N,) = cos(src_polar)
    cos_ra  = np.cos(ra_arr)         # (N,)
    sin_ra  = np.sin(ra_arr)         # (N,)
    cos_psi = np.cos(psi_arr)        # (N,)
    sin_psi = np.sin(psi_arr)        # (N,)

    cos_psr_dec = np.cos(psr_dec)    # scalar
    sin_psr_dec = np.sin(psr_dec)    # scalar
    cos_psr_ra  = np.cos(psr_ra)     # scalar
    sin_psr_ra  = np.sin(psr_ra)     # scalar

    omega_hat = np.array([           # (3, N)
        -cos_dec * cos_ra,
        -cos_dec * sin_ra,
        -sin_dec,
    ])
    p_hat = np.array([               # (3,)
        cos_psr_dec * cos_psr_ra,
        cos_psr_dec * sin_psr_ra,
        sin_psr_dec,
    ])
    m_hat = np.array([               # (3, N)
        sin_ra,
        -cos_ra,
        np.zeros(len(ra_arr)),
    ])
    n_hat = np.array([               # (3, N)
        -sin_dec * cos_ra,
        -sin_dec * sin_ra,
         cos_dec,
    ])

    m_rot = cos_psi * m_hat + sin_psi * n_hat     # (3, N)
    n_rot = -sin_psi * m_hat + cos_psi * n_hat    # (3, N)

    denom = 1.0 + np.dot(p_hat, omega_hat)        # (N,)
    p_m   = np.dot(p_hat, m_rot)                  # (N,)
    p_n   = np.dot(p_hat, n_rot)                  # (N,)

    Fp = 0.5 * (p_m**2 - p_n**2) / denom
    Fx = (p_m * p_n) / denom

    return Fp, Fx


# =============================================================================
# Analytic proxy SNR
# =============================================================================

def _compute_analytic_proxy(pop, psrs, Nvecs=None, Tspan_seconds=None):
    """
    Analytic approximation to sqrt(s^T N^{-1} s), vectorized over all binaries.

    rho^2 ≈ sum_psr  <sin^2> * (A^2 + B^2) * sum(1/Nvec)

    where A = Fp * h0 * (1 + cos^2(iota)) / (2*pi*f)
          B = Fx * h0 * (-2*cos(iota))    / (2*pi*f)

    and <sin^2> = 0.5 * (1 - sin(2*pi*f*T) / (2*pi*f*T))
        is the time-averaged sin^2 correction (exact for a stationary
        sinusoid over baseline T). For f*T >> 1 this -> 0.5 (standard
        result); for f*T ~ 1 (low-frequency regime) it suppresses rho
        relative to the naive 0.5 assumption.

    An additional cycle_penalty drives sub-cycle sources (f*T < 10) to
    near-zero proxy score, preventing the timing model absorption regime
    from being overestimated.

    Nvecs: list of per-pulsar Nvec arrays (Stage 2, from PTA object).
           If None, falls back to toaerrs^2 (Stage 1, libstempo objects,
           toaerrs in microseconds -> converted to seconds here).

    pop: either a PopulationArrays object (Stage 1) with array attributes,
         or a list of binary objects (Stage 2) with scalar attributes.

    Tspan_seconds: observation baseline in seconds. Required when the
                   low-frequency correction is desired; if None the
                   correction is skipped and <sin^2> = 0.5 is used.
    """
    # ── Per-pulsar noise weights — computed once, binary-independent ──────────
    if Nvecs is not None:
        inv_Nvec_sums = np.array([np.sum(1.0 / Nvec) for Nvec in Nvecs])
    else:
        # libstempo toaerrs in microseconds → convert to seconds
        # Detect libstempo (microseconds) vs Enterprise (seconds) by magnitude
        inv_Nvec_sums = []
        for psr in psrs:
            toaerrs = psr.toaerrs
            if np.median(toaerrs) > 1e-3:
                toaerrs = toaerrs * 1e-6   # microseconds → seconds
            inv_Nvec_sums.append(np.sum(1.0 / toaerrs**2))
        inv_Nvec_sums = np.array(inv_Nvec_sums)

    psr_coords = [_get_psr_radec(psr) for psr in psrs]

    # ── Extract binary params as arrays ──────────────────────────────────────
    if hasattr(pop, 'f'):
        # PopulationArrays — attributes are already arrays
        f   = np.asarray(pop.f,   dtype=np.float64)
        h0  = np.asarray(pop.h0,  dtype=np.float64)
        ci  = np.cos(np.asarray(pop.iota, dtype=np.float64))
        ra  = np.asarray(pop.ra,  dtype=np.float64)
        dec = np.asarray(pop.dec, dtype=np.float64)
        psi = np.asarray(pop.psi, dtype=np.float64)
    else:
        # List of binary objects — extract scalar attributes into arrays
        f   = np.array([b.f    for b in pop])
        h0  = np.array([b.h0   for b in pop])
        ci  = np.cos(np.array([b.iota for b in pop]))
        ra  = np.array([b.ra   for b in pop])
        dec = np.array([b.dec  for b in pop])
        psi = np.array([b.psi  for b in pop])

    # ── Amplitude factors — shape (N_binaries,) ───────────────────────────────
    norm = h0 / (2.0 * np.pi * f)
    Aamp = norm * (1.0 + ci**2)    # + polarization amplitude
    Bamp = norm * (-2.0 * ci)      # x polarization amplitude

    # ── Time-averaged sin^2 correction — shape (N_binaries,) ─────────────────
    # <sin^2(2*pi*f*t)> over [0, T] = 0.5 * (1 - sin(2*pi*f*T) / (2*pi*f*T))
    # This is the exact mean for a single stationary sinusoid; reduces to 0.5
    # when f*T >> 1.
    if Tspan_seconds is not None:
        f_T   = f * Tspan_seconds                             # number of cycles
        phase = 2.0 * np.pi * f_T                            # 2*pi*f*T

        # Exact time-average of sin^2(2*pi*f*t) over [0, T]
        sinc_term = np.where(
            phase > 1e-6,
            np.sin(phase) / phase,
            1.0 - phase**2 / 6.0                             # Taylor near 0
        )
        time_avg = 0.5 * (1.0 - sinc_term)                   # exact <sin^2>

        # Cycle penalty: sources with f*T < 10 have signal partially/fully
        # absorbed by the timing model and red noise basis (Sigma correction).
        # This is NOT captured by time_avg above — it requires the full
        # innerProduct_rr to evaluate. We apply a smooth penalty:
        #   f*T <  1 → penalty = 0   (sub-cycle, fully absorbed)
        #   f*T =  5 → penalty ~ 0.4
        #   f*T = 10 → penalty = 1   (no penalty above 10 cycles)
        cycle_penalty = np.clip((f_T - 1.0) / 9.0, 0.0, 1.0)

        time_avg = time_avg * cycle_penalty

        # Hard cap at 0.5 (theoretical maximum); no floor so low-f → 0
        time_avg = np.clip(time_avg, 0.0, 0.5)
    else:
        time_avg = 0.5                                        # high-f scalar limit

    # ── Sum antenna response over pulsars ─────────────────────────────────────
    # rho^2 = time_avg * sum_psr [ (Fp*Aamp)^2 + (Fx*Bamp)^2 ] * inv_Nvec_sum
    antenna_sum = np.zeros(len(f))
    for (psr_ra, psr_dec), inv_Nvec_sum in zip(psr_coords, inv_Nvec_sums):
        Fp, Fx = _antenna_response_vec(psr_ra, psr_dec, ra, dec, psi)
        A = Fp * Aamp
        B = Fx * Bamp
        antenna_sum += (A**2 + B**2) * inv_Nvec_sum

    rho_sq = time_avg * antenna_sum

    # ── Array-level sky sensitivity weight ────────────────────────────────────
    # sky_sensitivity_weight is already vectorized (accepts arrays)
    sky_weights = sky_sensitivity_weight(ra, dec).astype(np.float64)

    return np.sqrt(rho_sq) * sky_weights


# =============================================================================
# Population filter
# =============================================================================

def _filter_population_extremes(
    pop: PopulationArrays,
    psrs,
    n_keep: int = 100,
    n_total: int = None,
    Tspan_seconds: Optional[float] = None,
) -> tuple[PopulationArrays, np.ndarray]:
    """
    Keep extreme binaries across all parameters PLUS the top binaries by
    analytic CGW proxy SNR, with explicit rescue of proxy failure regimes.

    Returns (filtered_pop, proxy_scores) where proxy_scores align with
    filtered_pop so they can be saved directly to the shard.

    psrs: libstempo pulsar objects — used to compute per-pulsar noise weights.
    Tspan_seconds: observation baseline — used for low-frequency correction.
    """
    n = len(pop)

    if n_total is None:
        n_total = N_PRE_FILTER_PER_CHUNK
    n_total = min(n_total, n)
    n_keep  = min(n_keep, n // 2)

    indices = set()

    # ── 1. Analytic proxy SNR (primary filter) ────────────────────────────────
    proxies = _compute_analytic_proxy(pop, psrs, Nvecs=None,
                                      Tspan_seconds=Tspan_seconds)

    n_cgw = min(n_total, n)
    if n_cgw == n:
        cgw_top = np.argsort(proxies)[::-1]
    else:
        cgw_top = np.argpartition(proxies, -n_cgw)[-n_cgw:]
        cgw_top = cgw_top[np.argsort(proxies[cgw_top])[::-1]]
    indices.update(cgw_top.tolist())

    # ── 2. Parameter extremes (union'd in) ───────────────────────────────────
    for arr in (pop.f, pop.D_comov, pop.h0, pop.Mc, pop.Mtot):
        order = np.argsort(arr)
        indices.update(order[:n_keep].tolist())
        indices.update(order[-n_keep:].tolist())
    h0_order = np.argsort(pop.h0)
    indices.update(h0_order[-n_keep:].tolist())

    # ── 3. Explicit rescue of proxy failure regimes ───────────────────────────
    # Force top-h0 sources from frequency regimes where the proxy is known
    # to fail into the candidate list unconditionally.
    if Tspan_seconds is not None:
        f_T = np.asarray(pop.f, dtype=np.float64) * Tspan_seconds

        # Regime 1: very low f (f*T < 10 cycles)
        # cycle_penalty drives these to zero proxy score, but some may be
        # genuinely loud — keep top-h0 from this regime as insurance
        low_f_mask = f_T < 10.0
        if low_f_mask.sum() > 0:
            low_f_idx    = np.where(low_f_mask)[0]
            top_h0_low_f = low_f_idx[np.argsort(pop.h0[low_f_idx])[-n_keep:]]
            indices.update(top_h0_low_f.tolist())

        # Regime 2: moderate f (10 < f*T < 50)
        # Red noise Sigma correction is large here — proxy underestimates
        # because it only sees white noise diagonal. Keep top-h0 as rescue.
        mid_f_mask = (f_T >= 10.0) & (f_T < 50.0)
        if mid_f_mask.sum() > 0:
            mid_f_idx    = np.where(mid_f_mask)[0]
            top_h0_mid_f = mid_f_idx[np.argsort(pop.h0[mid_f_idx])[-n_keep:]]
            indices.update(top_h0_mid_f.tolist())

    idx = np.array(sorted(indices))
    print(
        f"  Population filtered: {n:,} → {len(idx):,} kept "
        f"(top-{n_cgw} proxy + {len(idx) - n_cgw} parameter/regime extremes)"
    )
    return pop[idx], proxies[idx]


# =============================================================================
# TOA delta I/O
# =============================================================================

def _delta_filename(chunk_id: int, scenario: Optional[str] = None) -> str:
    suffix = f'_{scenario}' if scenario else ''
    return f'chunk_{chunk_id:04d}{suffix}.npz'


def _save_toa_deltas(
    delta_stoas,          # list of (psr_name, delta_days) pairs
    out_dir: str,
    chunk_id: int,
    scenario: Optional[str] = None,
) -> None:
    stoa_dir = os.path.join(out_dir, 'stoas')
    os.makedirs(stoa_dir, exist_ok=True)
    outpath  = os.path.join(stoa_dir, _delta_filename(chunk_id, scenario))
    save_dict = {
        psr_name: delta.astype(np.float64)
        for psr_name, delta in delta_stoas
    }
    np.savez(outpath, **save_dict)
    print(f'  Saved Δstoas ({len(save_dict)} pulsars) → {outpath}')


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Stage 1: population synthesis')
    p.add_argument('--config', '-c', default='optimistic',
                   choices=list(config.POPULATION_CONFIGS.keys()))
    p.add_argument('--target-snr',  type=float, default=4.0)
    p.add_argument('--snr-range',   nargs=2, type=float, default=[3.5, 4.0])
    p.add_argument('--output-dir',  type=str, required=True)
    p.add_argument('--chunk-size',  type=int, default=1_000_000)
    p.add_argument('--n-chunks',    type=int, default=10,
                   help='Number of chunks per simulation (used to decode task_id)')
    p.add_argument('--task-id',     type=int, default=None,
                   help='Flat array task ID (overrides $SLURM_ARRAY_TASK_ID)')
    p.add_argument('--sim-id',      type=int, required=True)
    p.add_argument('--noise-seed-base',           type=int, default=None)
    # JSON string defining synthetic PTA scenarios.
    # Keys are scenario labels; values match data_loader.SCENARIOS format.
    # Example: '{"5x_cadence": {"cadence_factor": 5, "toaerr_factor": 1.0, "best_only": true}}'
    # Pass '{}' or omit to run baseline only.
    p.add_argument('--n-sims', type=int, default=1, help='Total number of simulations (for seeding purposes)')
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
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t0   = time.time()

    # ── Decode task_id → chunk_id ─────────────────────────────────────────────
    task_id = args.task_id
    if task_id is None:
        env_val = os.environ.get('SLURM_ARRAY_TASK_ID')
        if env_val is None:
            sys.exit('ERROR: --task-id not set and $SLURM_ARRAY_TASK_ID undefined.')
        task_id = int(env_val)

    sim_id   = args.sim_id
    chunk_id = task_id   # task_id is now directly the chunk_id (0..N_CHUNKS-1)

    root_sq = np.random.SeedSequence(args.noise_seed_base)
    sim_seq = root_sq.spawn(max(args.n_sims, args.sim_id + 1))[args.sim_id]
    chunk_seq = sim_seq.spawn(args.n_chunks)[args.task_id]

    rng = np.random.default_rng(chunk_seq)
    seed_int = rng.integers(2**31).item()

    # Per-simulation output directory
    sim_out_dir = os.path.join(args.output_dir, f'sim{sim_id:03d}')
    pop_dir     = os.path.join(sim_out_dir, 'populations')
    meta_dir    = os.path.join(sim_out_dir, 'metadata')
    os.makedirs(pop_dir,  exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Stage 1 — task_id={task_id}  sim_id={sim_id}  chunk_id={chunk_id}')
    print(f'  config={args.config}  chunk_size={args.chunk_size:,}')
    print(f'  output → {sim_out_dir}')
    print(f'{"="*60}')

    # ── Resolve scenario configs ──────────────────────────────────────────────
    # Build ordered list: baseline first, then any synthetic scenarios.
    if args.synthetic_ptas:
        if args.synthetic_pta_config:
            syn_scenarios = json.loads(args.synthetic_pta_config)
        else:
            # default: all non-baseline scenarios from data_loader
            syn_scenarios = {
                k: v for k, v in DEFAULT_SCENARIOS.items()
                if k != 'baseline'
            }
        print(f'  Synthetic scenarios: {list(syn_scenarios.keys())}')
    else:
        syn_scenarios = {}

    all_scenario_labels: List[str] = ['baseline'] + list(syn_scenarios.keys())

    parfiles = sorted([f for f in os.listdir(config.PAR_DIR) if f.endswith('.par')])

    # Combined scenarios dict for load_pulsars resolution
    combined_scenarios = dict(DEFAULT_SCENARIOS)
    combined_scenarios.update(syn_scenarios)

    # ── 1. Generate population chunk (before pulsar loops to keep it in scope)
    print(f'\n🌌 Generating {args.chunk_size:,} binaries...')
    selected_config  = config.POPULATION_CONFIGS[args.config]
    smbhb_module     = config.load_smbhb_module()
    population_batch = config.generate_population(
        config=selected_config,
        smbhb_module=smbhb_module,
        n_binaries=args.chunk_size,
        seed=seed_int,
    )
    print(f'✓ Generated {len(population_batch):,} binaries')

    # ── 2. Process scenarios one at a time ────────────────────────────────────
    # Memory strategy: load a scenario's pulsars, compute + save TOA deltas,
    # then delete the pulsar objects before loading the next scenario.
    # Only the baseline pulsars and Tspan are kept past their own iteration
    # (held briefly for the population filter step after the loop).

    print('\n📡 Processing scenarios (one at a time for memory efficiency)...')
    t_load_total  = 0.0
    t_nufft_total = 0.0

    psrs_baseline  = None
    Tspan_seconds  = None   # set from baseline; used for population filter

    for scenario_label in all_scenario_labels:
        t_scen = time.time()
        is_baseline = (scenario_label == 'baseline')
        print(f'\n  ── Scenario: {scenario_label} {"(baseline)" if is_baseline else ""} ──')

        if is_baseline:
            # Baseline can stay list-based because it is the reference PTA.
            t_l0 = time.time()
            print(f'  📡 Loading pulsars...')
            psrs_unfiltered = load_pulsars(
                verbose=True,
                scenario=scenario_label,
                scenarios=combined_scenarios,
            )
            with suppress_enterprise_warnings():
                psrs_clean, _, T_span = filter_pulsars_15yr(psrs_unfiltered, verbose=True)
            del psrs_unfiltered
            gc.collect()
            t_load_total += time.time() - t_l0
            print(f'  ✓ {len(psrs_clean)} pulsars, Tspan={T_span / (365.25*86400):.1f} yr')

            psrs_baseline = psrs_clean
            Tspan_seconds = T_span

            t_n0 = time.time()
            print(f'  ⚡ Computing TOA Δs via NUFFT...')
            delta_stoas = change_in_TOAs_days_population_nufft(
                psrs_clean, population_batch, verbose=True)
            t_nufft_total += time.time() - t_n0
            print(f'  ✓ Computed Δstoas for {len(delta_stoas)} pulsars')

            _save_toa_deltas(
                delta_stoas, sim_out_dir, chunk_id,
                scenario=None if is_baseline else scenario_label,
            )

            del delta_stoas, psrs_clean
            gc.collect()
        else:
            # Stream synthetic scenarios one pulsar at a time to avoid keeping
            # the augmented TOA sets for all pulsars in memory simultaneously.
            t_l0 = time.time()
            print(f'  📡 Loading pulsars (streaming)...')
            delta_stoas = []
            n_kept = 0
            for par in parfiles:
                psr_variants = load_single_pulsar(
                    par,
                    verbose=True,
                    scenario=scenario_label,
                    scenarios=combined_scenarios,
                )
                for psr in psr_variants:
                    with suppress_enterprise_warnings():
                        kept, _, _ = filter_pulsars_15yr([psr], verbose=False)
                    if not kept:
                        del psr
                        gc.collect()
                        continue

                    t_n0 = time.time()
                    delta_one = change_in_TOAs_days_population_nufft(
                        [psr], population_batch, verbose=True)
                    t_nufft_total += time.time() - t_n0
                    delta_stoas.extend(delta_one)
                    n_kept += 1
                    del psr, delta_one
                    gc.collect()

            t_load_total += time.time() - t_l0
            print(f'  ✓ Computed Δstoas for {n_kept} pulsars')

            _save_toa_deltas(
                delta_stoas, sim_out_dir, chunk_id,
                scenario=scenario_label,
            )

            del delta_stoas
            gc.collect()

        elapsed_scen = time.time() - t_scen
        print(f'  ✓ Scenario {scenario_label} done in {elapsed_scen/60:.1f} min')

    # ── 3. Filter and save population shard ──────────────────────────────────
    print(f'\n💾 Filtering to extreme binaries '
          f'(using baseline pulsars, Tspan={Tspan_seconds/(365.25*86400):.1f} yr)...')
    population_batch, proxy_scores = _filter_population_extremes(
        population_batch,
        n_keep=100,
        psrs=psrs_baseline,
        Tspan_seconds=Tspan_seconds,
    )

    # Baseline pulsars no longer needed
    del psrs_baseline
    gc.collect()

    print(f'\n💾 Saving shard subpop_{chunk_id:03d}.pkl.gz ...')
    store = ShardedPickleStore(pop_dir)
    store.write(chunk_id, population_batch)
    store.update(chunk_id, cgw_proxy=proxy_scores)
    print('✓ Shard written')

    del population_batch, proxy_scores
    gc.collect()

    # ── 4. Write metadata (chunk 0 wins; content is idempotent) ──────────────
    config_path = os.path.join(meta_dir, 'config.json')
    if chunk_id == 0 or not os.path.exists(config_path):
        config_json = {
            'config':               args.config,
            'target_snr':           args.target_snr,
            'snr_range':            args.snr_range,
            'n_chunks':             args.n_chunks,
            'chunk_size':           args.chunk_size,
            'Tspan_seconds':        Tspan_seconds,
            'synthetic_scenarios':  list(syn_scenarios.keys()),
        }
        with open(config_path, 'w') as fh:
            json.dump(config_json, fh, indent=2)
        print('✓ Wrote metadata/config.json')

    elapsed = time.time() - t0
    print(f'\n  Pulsar loading total:  {t_load_total/60:.1f} min')
    print(f'  NUFFT injection total: {t_nufft_total/60:.1f} min')
    print(f'\n✅ Stage 1 chunk={chunk_id} (sim={sim_id}) '
          f'complete in {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()