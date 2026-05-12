import os
import sys
import shutil
import subprocess
import numpy as np
# ensure repo root on path for test imports
sys.path.insert(0, os.getcwd())
from io_backends import population_to_zarr, population_slice, get_population_length


def make_test_population(path):
    class B:
        def __init__(self, f, Mc, D, h0, ra, dec, psi, iota, phi0):
            self.f = f
            self.Mc = Mc
            self.Mtot = Mc*1.5
            self.D_comov = D
            self.z = 0.1
            self.h0 = h0
            self.ra = ra
            self.dec = dec
            self.psi = psi
            self.iota = iota
            self.phi0 = phi0

    N = 200
    rng = np.random.RandomState(0)
    freqs = rng.uniform(1e-9, 100e-9, size=N)
    h0 = rng.lognormal(mean=-50, sigma=2.0, size=N)
    pop = [B(f=freqs[i], Mc=1e9, D=100.0, h0=h0[i], ra=rng.uniform(0,2*np.pi), dec=rng.uniform(-np.pi/2, np.pi/2), psi=0.0, iota=0.0, phi0=0.0) for i in range(N)]

    class SP:
        def __len__(self):
            return len(self.f)
    sp = SP()
    fields = ['f','Mc','Mtot','D_comov','z','h0','ra','dec','psi','iota','phi0']
    for fld in fields:
        setattr(sp, fld, np.array([getattr(b, fld) for b in pop], dtype=np.float64))

    os.makedirs(path, exist_ok=True)
    zpath = os.path.join(path, 'population_test.zarr')
    population_to_zarr(zpath, sp, dtype=np.float32, chunk_size=100)
    return zpath


def test_chunked_end_to_end(tmp_path):
    work = str(tmp_path)
    zpath = make_test_population(work)

    outdir = os.path.join(work, 'chunks')
    os.makedirs(outdir, exist_ok=True)

    # run two chunks
    cmd0 = ["python", "chunked_inject_driver.py", "--population-zarr", zpath, "--chunk-index", "0", "--n-chunks", "2", "--output-dir", outdir, "--accumulate", "--accum-grid-size", "1024", "--psr-ra", "1.0", "--psr-dec", "0.1"]
    cmd1 = ["python", "chunked_inject_driver.py", "--population-zarr", zpath, "--chunk-index", "1", "--n-chunks", "2", "--output-dir", outdir, "--accumulate", "--accum-grid-size", "1024", "--psr-ra", "1.0", "--psr-dec", "0.1"]

    subprocess.check_call(cmd0)
    subprocess.check_call(cmd1)

    # reduce accumulators
    reduce_cmd = ["python", "reduce_chunked_outputs.py", "--chunks-dir", outdir, "--merge-accum", "--ifft-out", os.path.join(work, 'ifft_out.npz')]
    subprocess.check_call(reduce_cmd)

    # read global topk from reducer default file
    global_top = os.path.join(os.getcwd(), 'data', 'global_topk.npz')
    # if global_top doesn't exist (since reducer writes to args.out-file default), check chunk files
    found = any([name.startswith('chunk_') for name in os.listdir(outdir)])
    assert found

    # basic check: IFFT output exists
    assert os.path.exists(os.path.join(work, 'ifft_out.npz'))

