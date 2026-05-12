Chunked injection prototype

Files added:
- `io_backends.py` : zarr/h5py helpers to write/read population slices
- `chunked_inject_driver.py` : per-chunk driver that selects top-K proxies
- `reduce_chunked_outputs.py` : aggregator to merge chunk outputs
- `submit_slurm_array.sh` : example SBATCH wrapper for array jobs

Quick local test (small population saved as zarr):

1. Create a tiny PopulationArrays-like object and write it with `io_backends.population_to_zarr`.

2. Run a single chunk locally:

```bash
python chunked_inject_driver.py --population-zarr data/population.zarr --chunk-index 0 --n-chunks 1 --top-k 20 --output-dir data/chunks --verbose
```

3. Reduce the chunk outputs:

```bash
python reduce_chunked_outputs.py --chunks-dir data/chunks --top-k 20 --out-file data/global_topk.npz
```

Submit to Slurm as an array (example):

```bash
sbatch --array=0-99 submit_slurm_array.sh --population-zarr data/population.zarr --n-chunks 100 --top-k 500 --output-dir data/chunks
```

Notes:
- This is a prototype focusing on I/O and chunking. Replace the proxy computation
  with your preferred injection method (NUFFT/FFT) inside `chunked_inject_driver.py`.
- `io_backends` uses `zarr` if available, otherwise falls back to `h5py`.
- Use local scratch (e.g., `$SLURM_TMPDIR`) inside the array job for heavy temporary writes.
