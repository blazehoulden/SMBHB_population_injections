from pathlib import Path
import argparse

import libstempo as lt
import numpy as np


def _check_pulsar(par_path: Path, tim_path: Path, do_fit: bool = True) -> dict:
    """Load one pulsar and return residual diagnostics."""
    psr = lt.tempopulsar(parfile=str(par_path), timfile=str(tim_path), maxobs=60000)

    if do_fit:
        psr.fit()

    resid = psr.residuals()
    toas = psr.toas()
    toaerrs = psr.toaerrs

    n_toas = len(resid)
    has_nan_resid = bool(np.any(np.isnan(resid)))
    has_inf_resid = bool(np.any(np.isinf(resid)))
    rms = float(np.std(resid)) if n_toas > 0 else float("nan")

    return {
        "name": par_path.name,
        "n_toas": n_toas,
        "nan_resid": has_nan_resid,
        "inf_resid": has_inf_resid,
        "rms": rms,
        "nan_toas": bool(np.any(np.isnan(toas))),
        "nan_toaerrs": bool(np.any(np.isnan(toaerrs))),
        "toa_min": float(np.min(toas)) if len(toas) > 0 else float("nan"),
        "toa_max": float(np.max(toas)) if len(toas) > 0 else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test residual quality for all pulsars.")
    parser.add_argument("--par-dir", default="psars_narrowband/par", help="Directory with .par files")
    parser.add_argument("--tim-dir", default="psars_narrowband/tim", help="Directory with .tim files")
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Skip psr.fit() and inspect pre-fit residuals only",
    )
    args = parser.parse_args()

    par_dir = Path(args.par_dir)
    tim_dir = Path(args.tim_dir)

    if not par_dir.exists() or not tim_dir.exists():
        print(f"ERROR: par/tim directory not found: par={par_dir}, tim={tim_dir}")
        return 2

    par_files = sorted(par_dir.glob("*.par"))
    if not par_files:
        print(f"ERROR: no .par files found in {par_dir}")
        return 2

    n_ok = 0
    n_fail = 0

    print(f"Found {len(par_files)} pulsars to test")
    print("=" * 80)

    for par_path in par_files:
        tim_path = tim_dir / f"{par_path.stem}.tim"
        print(f"Testing {par_path.name}...")

        if not tim_path.exists():
            print(f"  FAIL: missing TIM file: {tim_path.name}")
            n_fail += 1
            continue

        try:
            result = _check_pulsar(par_path, tim_path, do_fit=not args.no_fit)
            print(f"  N TOAs: {result['n_toas']}")
            print(f"  Residual RMS: {result['rms']:.3e} s")
            print(f"  NaN residuals: {result['nan_resid']} | Inf residuals: {result['inf_resid']}")
            print(f"  NaN TOAs: {result['nan_toas']} | NaN toaerrs: {result['nan_toaerrs']}")
            print(f"  TOA range: {result['toa_min']:.3f} to {result['toa_max']:.3f} MJD")

            bad = (
                result["n_toas"] == 0
                or result["nan_resid"]
                or result["inf_resid"]
                or result["nan_toas"]
                or result["nan_toaerrs"]
            )
            if bad:
                print("  RESULT: FAIL (invalid residual/toa diagnostics)")
                n_fail += 1
            else:
                print("  RESULT: OK")
                n_ok += 1
        except Exception as exc:
            print(f"  FAIL: exception: {exc}")
            n_fail += 1

        print("-" * 80)

    print("Summary")
    print("=" * 80)
    print(f"Passed: {n_ok}")
    print(f"Failed: {n_fail}")
    print(f"Total:  {len(par_files)}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())