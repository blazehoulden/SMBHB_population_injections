"""
build_mpta_pulsar_skymap.py
============================
Build a pulsar sky-location .npz for the MeerKAT Pulsar Timing Array (MPTA),
in the same format used by your CGW sky-map plotting code
(``data/pulsar_sky_locations.npz`` with ``ras`` and ``decs`` arrays, in
radians, ICRS RA/Dec) -- so it's a drop-in replacement/addition next to your
existing NANOGrav npz.

Two ways to get positions -- use whichever matches what you have on hand:

  (A) From a directory of .par files (recommended). These are the exact
      positions used in your timing solutions, and this needs no network
      access at all.

  (B) From a plain-text list of pulsar J-names, querying the ATNF pulsar
      catalogue via `psrqpy` (needs an internet connection and
      `pip install psrqpy`).

Usage
-----
    # (A) from a directory of MPTA par files
    python build_mpta_pulsar_skymap.py \
        --par-dir /path/to/mpta_pars \
        --out data/mpta_pulsar_sky_locations.npz

    # (B) from a list of pulsar names, one J-name per line
    python build_mpta_pulsar_skymap.py \
        --names-file mpta_names.txt \
        --out data/mpta_pulsar_sky_locations.npz

Either way, load it downstream exactly like the NANOGrav one:

    data       = np.load("data/mpta_pulsar_sky_locations.npz")
    pulsar_ra  = _wrap_ra(data["ras"])
    pulsar_dec = data["decs"]
"""

import argparse
import glob
import os

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def _parse_par_file(path):
    """
    Extract a sky position from a tempo/tempo2-style .par file.

    Looks for RAJ/DECJ (equatorial, the common case) and falls back to
    ELONG/ELAT (ecliptic, used by some pipelines) if those aren't present.
    Returns an astropy SkyCoord, or None if no position field was found.
    """
    raj = decj = elong = elat = None

    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            key = parts[0].upper()
            if key == "RAJ" and len(parts) > 1:
                raj = parts[1]
            elif key == "DECJ" and len(parts) > 1:
                decj = parts[1]
            elif key == "ELONG" and len(parts) > 1:
                elong = float(parts[1])
            elif key == "ELAT" and len(parts) > 1:
                elat = float(parts[1])

    if raj is not None and decj is not None:
        # RAJ is sexagesimal hours (hh:mm:ss.sss), DECJ sexagesimal degrees
        return SkyCoord(ra=raj, dec=decj, unit=(u.hourangle, u.deg), frame="icrs")

    if elong is not None and elat is not None:
        return SkyCoord(
            lon=elong * u.deg, lat=elat * u.deg, frame="barycentrictrueecliptic"
        ).icrs

    return None


def positions_from_par_dir(par_dir):
    """Scan a directory of .par files and extract (name, ra_rad, dec_rad) for each."""
    names, ras, decs = [], [], []
    par_paths = sorted(glob.glob(os.path.join(par_dir, "*.par")))
    if not par_paths:
        raise FileNotFoundError(f"No .par files found in {par_dir}")

    for path in par_paths:
        coord = _parse_par_file(path)
        if coord is None:
            print(f"  [skip] no RAJ/DECJ or ELONG/ELAT found in {os.path.basename(path)}")
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        names.append(name)
        ras.append(coord.ra.rad)
        decs.append(coord.dec.rad)
        print(f"  {name:>14s}  RA={coord.ra.deg:8.4f} deg  Dec={coord.dec.deg:8.4f} deg")

    return names, np.array(ras), np.array(decs)


def positions_from_names(names_file):
    """Look up (name, ra_rad, dec_rad) for a list of pulsar names via the ATNF catalogue."""
    try:
        import psrqpy
    except ImportError as exc:
        raise ImportError(
            "psrqpy is required for name-based lookup: pip install psrqpy"
        ) from exc

    with open(names_file) as f:
        names = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Querying ATNF pulsar catalogue for {len(names)} pulsars...")
    query = psrqpy.QueryATNF(psrs=names, params=["RAJ", "DECJ"])
    table = query.table

    ras, decs, found_names = [], [], []
    for row in table:
        coord = SkyCoord(ra=row["RAJ"], dec=row["DECJ"], unit=(u.hourangle, u.deg), frame="icrs")
        found_names.append(row["PSRJ"])
        ras.append(coord.ra.rad)
        decs.append(coord.dec.rad)
        print(f"  {row['PSRJ']:>14s}  RA={coord.ra.deg:8.4f} deg  Dec={coord.dec.deg:8.4f} deg")

    missing = sorted(set(names) - set(found_names))
    if missing:
        print(f"  [warning] not found in ATNF catalogue: {missing}")

    return found_names, np.array(ras), np.array(decs)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--par-dir", help="Directory of MPTA .par files")
    parser.add_argument("--names-file", help="Text file of MPTA pulsar J-names, one per line")
    parser.add_argument("--out", required=True, help="Output .npz path")
    args = parser.parse_args()

    if args.par_dir:
        names, ras, decs = positions_from_par_dir(args.par_dir)
    elif args.names_file:
        names, ras, decs = positions_from_names(args.names_file)
    else:
        raise SystemExit("Provide either --par-dir or --names-file")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 'ras'/'decs' match the existing pulsar_sky_locations.npz format exactly
    # (radians, ICRS). 'names' is extra metadata -- harmless for the plotting
    # code above since it only ever reads data['ras'] and data['decs'].
    np.savez(args.out, ras=ras, decs=decs, names=np.array(names))
    print(f"\nSaved {len(ras)} pulsar positions to {args.out}")


if __name__ == "__main__":
    main()