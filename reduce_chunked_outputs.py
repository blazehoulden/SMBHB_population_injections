"""
Reducer for chunked outputs. Scans a directory for chunk_*.npz files,
merges the per-chunk top lists and writes a global top-N results file.

Usage:
    python reduce_chunked_outputs.py --chunks-dir data/chunks --top-k 500 --out-file data/global_topk.npz
"""

import argparse
import glob
import numpy as np
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--chunks-dir', required=True)
    p.add_argument('--top-k', type=int, default=500)
    p.add_argument('--out-file', default='data/global_topk.npz')
    p.add_argument('--merge-accum', action='store_true', help='Merge per-chunk frequency accumulators and optionally IFFT to time domain')
    p.add_argument('--ifft-out', default=None, help='If set, write IFFT time-series to this file (npz) after merging accumulators')
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.chunks_dir, 'chunk_*.npz')))
    if len(files) == 0:
        raise RuntimeError(f'No chunk files found in {args.chunks_dir}')

    all_idx = []
    all_scores = []
    for fpath in files:
        data = np.load(fpath)
        idx = data['top_indices']
        scores = data['top_scores']
        all_idx.append(idx)
        all_scores.append(scores)

    if len(all_idx) == 0:
        raise RuntimeError('No indices collected from chunks')

    all_idx = np.concatenate(all_idx)
    all_scores = np.concatenate(all_scores)

    # Keep best score per unique index
    order = np.argsort(all_scores)[::-1]
    all_idx = all_idx[order]
    all_scores = all_scores[order]

    uniq_idx, uniq_pos = np.unique(all_idx, return_index=True)
    uniq_scores = all_scores[uniq_pos]

    # select top-k
    k = min(args.top_k, len(uniq_idx))
    top_idx = uniq_idx[:k]
    top_scores = uniq_scores[:k]

    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)
    np.savez_compressed(args.out_file, top_indices=top_idx.astype(np.int64), top_scores=top_scores.astype(np.float64))
    print(f'Wrote global top-{k}: {args.out_file}')
    # Optionally merge frequency accumulators
    if args.merge_accum:
        freq_files = sorted(glob.glob(os.path.join(args.chunks_dir, 'chunk_*.npz')))
        freq_grid = None
        accum_total = None
        for fpath in freq_files:
            data = np.load(fpath)
            if 'freq_accum' in data:
                g = data['freq_grid']
                a = data['freq_accum']
                if freq_grid is None:
                    freq_grid = g
                    accum_total = np.array(a, dtype=np.complex128)
                else:
                    # require same grid
                    if not np.allclose(freq_grid, g):
                        raise RuntimeError('Mismatched freq grids across chunks')
                    accum_total += a

        if accum_total is None:
            print('No frequency accumulators found in chunk files.')
        else:
            print('Merged frequency accumulators from chunks.')
            if args.ifft_out:
                # perform inverse FFT (complex -> real time-series)
                # Use ifft of the complex array; assume uniform freq grid
                time_series = np.fft.ifft(accum_total)
                np.savez_compressed(args.ifft_out, freq_grid=freq_grid, accum=accum_total, time_series=time_series.astype(np.complex128))
                print(f'Wrote IFFT result to {args.ifft_out}')


if __name__ == '__main__':
    main()
