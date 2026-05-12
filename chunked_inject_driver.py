"""
Prototype chunked injection driver.

Usage (example):
    python chunked_inject_driver.py \
        --population-zarr data/population.zarr \
        --chunk-index 0 --n-chunks 10 --top-k 500 \
        --output-dir data/chunks

This prototype currently computes a simple pre-filter proxy for each binary
(proxy = h0/(2π f) * sky_weight) and writes the top-K global indices and
scores for the assigned chunk. It demonstrates the chunking I/O and a
memory-safe compute pattern.
"""

import argparse
import os
import numpy as np
from io_backends import get_population_length, population_slice

def antenna_response_vec(psr_ra, psr_dec, ra_arr, dec_arr, psi_arr):
    """Local copy of vectorised antenna response. Returns Fp, Fx each (N,)."""
    N             = len(ra_arr)
    src_polar     = np.pi / 2 - dec_arr
    psr_polar     = np.pi / 2 - psr_dec

    omega_hat = np.array([
        -np.sin(src_polar) * np.cos(ra_arr),
        -np.sin(src_polar) * np.sin(ra_arr),
        -np.cos(src_polar),
    ])  # (3, N)

    p_hat = np.array([
        np.sin(psr_polar) * np.cos(psr_ra),
        np.sin(psr_polar) * np.sin(psr_ra),
        np.cos(psr_polar),
    ])  # (3,)

    m_hat = np.array([np.sin(ra_arr), -np.cos(ra_arr), np.zeros(N)])       # (3, N)
    n_hat = np.array([
        -np.cos(src_polar) * np.cos(ra_arr),
        -np.cos(src_polar) * np.sin(ra_arr),
         np.sin(src_polar),
    ])  # (3, N)

    cos_psi = np.cos(psi_arr); sin_psi = np.sin(psi_arr)
    m_rot   =  cos_psi * m_hat + sin_psi * n_hat
    n_rot   = -sin_psi * m_hat + cos_psi * n_hat

    denom = 1 + p_hat @ omega_hat      # (N,)
    p_m   = p_hat @ m_rot              # (N,)
    p_n   = p_hat @ n_rot              # (N,)

    Fp = 0.5 * (p_m**2 - p_n**2) / denom
    Fx =       (p_m   * p_n)     / denom
    return Fp, Fx

try:
    import finufft
    _HAS_FINUFFT = True
except Exception:
    finufft = None
    _HAS_FINUFFT = False

try:
    from debug.test_CGW_sky_loc import sky_sensitivity_weight
except Exception:
    def sky_sensitivity_weight(ra, dec):
        # Fallback: uniform weight
        return np.ones_like(ra)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--population-zarr', required=True)
    p.add_argument('--chunk-index', type=int, required=True)
    p.add_argument('--n-chunks', type=int, required=True)
    p.add_argument('--top-k', type=int, default=1000)
    p.add_argument('--output-dir', default='data/chunks')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--nufft', action='store_true', help='Compute NUFFT residuals for a test pulsar')
    p.add_argument('--psr-ra', type=float, default=None, help='Pulsar RA in radians (required with --nufft)')
    p.add_argument('--psr-dec', type=float, default=None, help='Pulsar DEC in radians (required with --nufft)')
    p.add_argument('--T-obs-years', type=float, default=15.0, help='Observation span in years for synthetic TOAs')
    p.add_argument('--cadence-days', type=float, default=14.0, help='Cadence in days for synthetic TOAs')
    p.add_argument('--eps', type=float, default=1e-6, help='NUFFT tolerance')
    p.add_argument('--accumulate', action='store_true', help='Accumulate complex amplitudes onto a uniform frequency grid for later IFFT')
    p.add_argument('--accum-grid-size', type=int, default=2**14, help='Number of frequency bins for accumulation')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    N = get_population_length(args.population_zarr)
    # Determine slice for this chunk
    per_chunk = int(np.ceil(N / args.n_chunks))
    start = args.chunk_index * per_chunk
    end = min(N, start + per_chunk)

    if start >= N:
        raise ValueError(f"chunk-index {args.chunk_index} out of range for N={N} (start={start})")

    if args.verbose:
        print(f"Processing chunk {args.chunk_index}/{args.n_chunks}: indices [{start}:{end}) of {N}")

    pop = population_slice(args.population_zarr, start, end)

    # Compute proxy: h0 / (2π f) * sky_weight(ra,dec)
    f = pop['f']
    h0 = pop['h0']
    ra = pop['ra']
    dec = pop['dec']

    # sky_sensitivity_weight expects scalars; vectorise safely here
    try:
        sky_w = sky_sensitivity_weight(ra, dec)
    except Exception:
        sky_w = np.array([sky_sensitivity_weight(r, d) for r, d in zip(ra, dec)])
    proxy = (h0 / (2.0 * np.pi * f)) * sky_w

    # Pick top-K entries in this chunk
    K = min(args.top_k, len(proxy))
    if K <= 0:
        top_local_idx = np.array([], dtype=int)
        top_scores = np.array([], dtype=float)
    else:
        # partial selection
        if K < len(proxy):
            part = np.argpartition(proxy, -K)[-K:]
            top_local_idx = part[np.argsort(proxy[part])[::-1]]
        else:
            top_local_idx = np.argsort(proxy)[::-1]
        top_scores = proxy[top_local_idx]

    top_global_idx = start + top_local_idx

    out_path = os.path.join(args.output_dir, f"chunk_{args.chunk_index:04d}.npz")
    # If accumulation or NUFFT requested, compute amplitudes for this pulsar
    if args.accumulate or args.nufft:
        if args.psr_ra is None or args.psr_dec is None:
            raise RuntimeError('psr-ra and psr-dec must be provided for --accumulate or --nufft')

        # Build synthetic TOA grid (for NUFFT time-series only)
        T_obs_seconds = args.T_obs_years * 365.25 * 24 * 3600
        cadence_seconds = args.cadence_days * 24 * 3600
        time_arr = np.arange(0, T_obs_seconds, cadence_seconds)
        x = time_arr - time_arr[0]

        # Compute antenna responses and A,B amplitudes for this pulsar
        Fp, Fx = antenna_response_vec(args.psr_ra, args.psr_dec, ra, dec, pop.get('psi', np.zeros_like(ra)))
        iota = pop.get('iota', np.zeros_like(ra))
        phi0 = pop.get('phi0', np.zeros_like(ra))

        A = (Fp * h0 * (1 + np.cos(iota)**2)) / (2 * np.pi * f)
        B = (Fx * h0 * (-2 * np.cos(iota)))   / (2 * np.pi * f)

        # Convert to complex coefficients for NUFFT/accumulation
        S = A * np.cos(phi0) - B * np.sin(phi0)
        C = A * np.sin(phi0) + B * np.cos(phi0)
        c_k = (C - 1j * S) / 2.0

        s_k = f  # use Hz for accumulation grid (not rad/s)

        if args.accumulate:
            # Build frequency grid covering this chunk (small margin)
            fmin = np.maximum(0.0, np.min(s_k) * 0.9)
            fmax = np.max(s_k) * 1.1
            if fmax <= fmin:
                fmax = fmin + 1.0 / (args.accum_grid_size)
            freqs = np.linspace(fmin, fmax, args.accum_grid_size)
            df = freqs[1] - freqs[0]
            # Bin complex amplitudes onto nearest bin (simple, memory efficient)
            idx = np.round((s_k - fmin) / df).astype(int)
            valid = (idx >= 0) & (idx < args.accum_grid_size)
            accum = np.zeros(args.accum_grid_size, dtype=np.complex64)
            # accumulate
            for ii, vv in zip(idx[valid], c_k[valid]):
                accum[ii] += vv

            # Optionally also compute NUFFT time-series if requested
            if args.nufft:
                if not _HAS_FINUFFT:
                    raise RuntimeError('finufft not available in this environment')
                if args.psr_ra is None or args.psr_dec is None:
                    raise RuntimeError('psr-ra and psr-dec must be provided for --nufft')
                f_out = finufft.nufft1d3(2.0 * np.pi * s_k, c_k, x, isign=+1, eps=args.eps)
                r = 2.0 * np.real(f_out)
                np.savez_compressed(out_path,
                                    chunk_index=args.chunk_index,
                                    start=start,
                                    end=end,
                                    top_indices=top_global_idx.astype(np.int64),
                                    top_scores=top_scores.astype(np.float64),
                                    freq_grid=freqs.astype(np.float32),
                                    freq_accum=accum.astype(np.complex64),
                                    toa_times=time_arr.astype(np.float32),
                                    partial_residual=r.astype(np.float32))
            else:
                np.savez_compressed(out_path,
                                    chunk_index=args.chunk_index,
                                    start=start,
                                    end=end,
                                    top_indices=top_global_idx.astype(np.int64),
                                    top_scores=top_scores.astype(np.float64),
                                    freq_grid=freqs.astype(np.float32),
                                    freq_accum=accum.astype(np.complex64))
    else:
        np.savez_compressed(out_path,
                            chunk_index=args.chunk_index,
                            start=start,
                            end=end,
                            top_indices=top_global_idx.astype(np.int64),
                            top_scores=top_scores.astype(np.float64))

    if args.verbose:
        print(f"Wrote: {out_path}  (found {len(top_global_idx)} top candidates)")


if __name__ == '__main__':
    main()
