"""
Simple population I/O helpers: prefer zarr, fall back to h5py.
Provides functions to write PopulationArrays-like objects to disk
and to read slices without loading the entire population into memory.
"""

from typing import Tuple
import numpy as np

try:
    import zarr
    from numcodecs import Blosc
    # Quick runtime check: ensure zarr Group supports creating arrays/datasets
    import tempfile, shutil
    _HAS_ZARR = True
    try:
        tmp = tempfile.mkdtemp(prefix='zarr_test_')
        g = zarr.open_group(tmp, mode='w')
        # try to create a small dataset; if unsupported, mark zarr unusable
        try:
            if hasattr(g, 'create_dataset'):
                g.create_dataset('test', data=[1,2,3])
            elif hasattr(zarr, 'array'):
                zarr.array([1,2,3], store=tmp)
            else:
                raise RuntimeError('zarr API missing create methods')
        except Exception:
            _HAS_ZARR = False
        finally:
            try:
                shutil.rmtree(tmp)
            except Exception:
                pass
    except Exception:
        _HAS_ZARR = False
except Exception:
    _HAS_ZARR = False
    zarr = None

# zarr 3 moved DirectoryStore under zarr.storage; support both locations
if _HAS_ZARR:
    _DirectoryStore = getattr(zarr, 'DirectoryStore', None)
    if _DirectoryStore is None:
        _DirectoryStore = getattr(getattr(zarr, 'storage', None), 'DirectoryStore', None)
    DirectoryStore = _DirectoryStore

try:
    import h5py
    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False
    h5py = None


FIELDS = [
    "f",
    "Mc",
    "Mtot",
    "D_comov",
    "z",
    "h0",
    "ra",
    "dec",
    "psi",
    "iota",
    "phi0",
]


def population_to_zarr(
    path: str,
    pop,
    dtype=np.float32,
    chunk_size: int = 1_000_000,
    field_dtypes: dict | None = None,
):
    """Write a PopulationArrays-like object to disk.

    If zarr is available, create a zarr group with each field as a 1D array.
    Otherwise fall back to HDF5 using h5py (gzip compression).

    Args:
        path: destination path (for zarr this is a directory).
        pop: object with attributes named in FIELDS that are NumPy arrays.
        dtype: storage dtype (default float32).
        chunk_size: chunk length for 1D arrays.
    """
    n = len(pop)

    if _HAS_ZARR:
        # Prefer creating a zarr group directly from a path when supported
        try:
            root = zarr.open_group(path, mode='w')
        except Exception:
            if DirectoryStore is None:
                raise RuntimeError('zarr found but cannot create DirectoryStore or open_group for this zarr version')
            store = DirectoryStore(path)
            root = zarr.group(store=store, overwrite=True)
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
        fields_to_write = list(field_dtypes.keys()) if field_dtypes else FIELDS
        for field in fields_to_write:
            arr = getattr(pop, field)
            target_dtype = field_dtypes.get(field, dtype) if field_dtypes else dtype
            root.create_dataset(
                field,
                data=arr.astype(target_dtype),
                chunks=(min(chunk_size, n),),
                compressor=compressor,
            )
        root.attrs['n'] = n
        root.attrs['fields'] = fields_to_write
        return True

    if _HAS_H5PY:
        with h5py.File(path, 'w') as f:
            f.attrs['n'] = n
            fields_to_write = list(field_dtypes.keys()) if field_dtypes else FIELDS
            for field in fields_to_write:
                arr = getattr(pop, field)
                target_dtype = field_dtypes.get(field, dtype) if field_dtypes else dtype
                f.create_dataset(
                    field,
                    data=arr.astype(target_dtype),
                    compression='gzip',
                    chunks=(min(chunk_size, n),),
                )
            f.attrs['fields'] = fields_to_write
        return True

    raise RuntimeError("Neither zarr nor h5py available — please install one to use io_backends.")


def get_population_length(path: str) -> int:
    if _HAS_ZARR and zarr.util.path_exists(path):
        root = zarr.open_group(path, mode='r')
        return int(root.attrs.get('n', root['f'].shape[0]))
    if _HAS_H5PY:
        with h5py.File(path, 'r') as f:
            return int(f.attrs.get('n', f['f'].shape[0]))
    raise RuntimeError("Cannot determine population length: no supported backend available.")


def population_slice(path: str, start: int, end: int, fields: list[str] | None = None) -> dict:
    """Read a slice [start:end) of the stored population and return a dict of NumPy arrays.

    The returned dict contains keys matching FIELDS.
    """
    if _HAS_ZARR and zarr.util.path_exists(path):
        root = zarr.open_group(path, mode='r')
        available = list(root.attrs.get('fields', [])) or [f for f in FIELDS if f in root]
        selected = fields if fields is not None else available
        out = {field: np.array(root[field][start:end]) for field in selected if field in root}
        return out
    if _HAS_H5PY:
        with h5py.File(path, 'r') as f:
            available = list(f.attrs.get('fields', [])) or [fld for fld in FIELDS if fld in f]
            selected = fields if fields is not None else available
            out = {field: np.array(f[field][start:end]) for field in selected if field in f}
        return out
    raise RuntimeError("No supported backend available to read population slice.")
