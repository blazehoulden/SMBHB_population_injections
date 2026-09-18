#!/usr/bin/env python3
"""Measure how changing one saved CW source changes its optimal S/N.

The script never writes inside a simulation directory.  It accepts either the
normal ``residuals/combined`` directory or a ``combined.tar.gz`` archive and
uses the saved summary to find the loudest source.

Example (on a machine with the project environment installed)::

    python analyze_cgw_source_perturbations.py \
        --runs-root runs --parameter ra --values 0.0,1.0,2.0

The values use the native units in the summary: radians for angles and Hz for
frequency. ``--parameter sky`` is the exception: instead of absolute
coordinates, it takes *offsets in degrees* from the loudest source's own
saved (ra, dec) -- i.e. its "standard" initial sky location. Offsets are
converted to radians and added to that initial position when the source is
perturbed. Each value must be a ``dra_deg,ddec_deg`` pair, and multiple
offsets are separated by semicolons.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np


SOURCE_FIELDS = ("f", "ecc", "phi0", "iota", "h0", "ra", "dec", "psi")
PERTURBABLE_FIELDS = ("f", "ecc", "phi0", "iota", "h0", "ra", "dec", "psi")
SOURCE_DEFAULTS = {"ecc": 0.0}


def load_summary(path: Path) -> dict:
    with gzip.open(path, "rb") as stream:
        summary = pickle.load(stream)
    if not isinstance(summary, dict) or "arrays" not in summary:
        raise ValueError(f"Unsupported summary format: {path}")
    return summary


def find_simulations(runs_root: Path) -> list[Path]:
    return sorted(p.parent for p in runs_root.rglob("summary.pkl.gz"))


def select_loudest(summary: dict) -> tuple[int, str, float]:
    arrays = summary["arrays"]
    candidates = [
        name for name in summary.get("meta", {}).get("snr_fields", [])
        if name in arrays and name.startswith("cgw_snr_baseline_forecast")
    ]
    if "cgw_snr_baseline_forecast" in arrays:
        field = "cgw_snr_baseline_forecast"
    elif candidates:
        field = candidates[0]
    else:
        raise KeyError("summary contains no cgw_snr_baseline_forecast array")
    values = np.asarray(arrays[field], dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        raise ValueError(f"SNR array {field!r} is empty or non-finite")
    index = int(np.nanargmax(values))
    return index, field, float(values[index])


def select_loudest_sim(simulations: Iterable[Path]) -> Path:
    ranked = []
    for sim_dir in simulations:
        index, field, snr = select_loudest(load_summary(sim_dir / "summary.pkl.gz"))
        ranked.append((snr, sim_dir, index, field))
    return max(ranked, key=lambda item: item[0])[1]


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in archive {archive}: {member.name}")
        tar.extractall(destination)


def load_combined_residuals(
    sim_dir: Path,
    names: Iterable[str],
    scenario: str,
) -> tuple[dict[str, np.ndarray], tempfile.TemporaryDirectory | None]:
    residual_root = sim_dir / "residuals"
    if scenario != "baseline":
        residual_root = sim_dir / f"residuals_{scenario}"
    residual_dir = residual_root / "combined"
    temporary = None
    if not residual_dir.is_dir():
        archive = next(iter(sorted(residual_root.glob("combined*.tar.gz"))), None)
        if archive is None:
            raise FileNotFoundError(
                f"No combined residual directory/archive in {residual_root}"
            )
        temporary = tempfile.TemporaryDirectory(prefix="cgw-residuals-")
        extracted = Path(temporary.name)
        _safe_extract(archive, extracted)
        matches = list(extracted.rglob("*.npy"))
        if not matches:
            raise FileNotFoundError(f"Archive contains no .npy residuals: {archive}")
        residual_dir = matches[0].parent

    result = {}
    for name in names:
        path = residual_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing combined residual: {path}")
        result[name] = np.load(path).astype(np.float64, copy=False)
    return result, temporary


def scenario_from_snr_field(snr_field: str) -> str:
    prefix = "cgw_snr_"
    if snr_field == "cgw_snr":
        return "baseline"
    if not snr_field.startswith(prefix):
        raise ValueError(f"Cannot infer scenario from S/N field {snr_field!r}")
    return snr_field[len(prefix):]


def make_source(summary: dict, index: int) -> SimpleNamespace:
    arrays = summary["arrays"]
    missing = [
        field for field in SOURCE_FIELDS
        if field not in arrays and field not in SOURCE_DEFAULTS
    ]
    if missing:
        raise KeyError(f"summary is missing source fields: {', '.join(missing)}")
    source = {}
    for field in SOURCE_FIELDS:
        if field in arrays:
            source[field] = float(np.asarray(arrays[field])[index])
        else:
            source[field] = SOURCE_DEFAULTS[field]
    return SimpleNamespace(**source)


def parse_values(parameter: str, text: str) -> list[dict[str, float]]:
    """Parse --values into a list of "change" dicts.

    For ``parameter == "sky"`` each change dict holds *degree offsets*
    (``ra_offset_deg`` / ``dec_offset_deg``) rather than absolute
    coordinates. The offsets are only resolved into absolute radian
    coordinates later, in ``run_analysis``, once the source's own initial
    (ra, dec) is known.
    """
    if parameter == "sky":
        values = []
        for item in text.split(";"):
            pair = [float(value.strip()) for value in item.split(",")]
            if len(pair) != 2:
                raise ValueError(
                    "sky values must be dra_deg,ddec_deg offset pairs separated by ';'"
                )
            values.append({"ra_offset_deg": pair[0], "dec_offset_deg": pair[1]})
        return values
    if parameter not in PERTURBABLE_FIELDS:
        raise ValueError(f"parameter must be one of {PERTURBABLE_FIELDS} or 'sky'")
    return [{parameter: float(item.strip())} for item in text.split(",") if item.strip()]


def _apply_change(source: SimpleNamespace, change: dict[str, float]) -> tuple[SimpleNamespace, dict]:
    """Return a perturbed copy of ``source`` plus a change record for output.

    Handles the special sky-offset keys (degrees, relative to the source's
    own saved ra/dec) as well as ordinary absolute-field overrides.
    """
    perturbed = SimpleNamespace(**vars(source))
    if "ra_offset_deg" in change or "dec_offset_deg" in change:
        ra_offset_deg = change.get("ra_offset_deg", 0.0)
        dec_offset_deg = change.get("dec_offset_deg", 0.0)
        perturbed.ra = source.ra + np.deg2rad(ra_offset_deg)
        perturbed.dec = source.dec + np.deg2rad(dec_offset_deg)
        record = {
            "ra_offset_deg": ra_offset_deg,
            "dec_offset_deg": dec_offset_deg,
            "ra": perturbed.ra,
            "dec": perturbed.dec,
        }
        return perturbed, record

    for field, value in change.items():
        setattr(perturbed, field, value)
    return perturbed, dict(change)


def run_analysis(sim_dir: Path, parameter: str, values: list[dict[str, float]]) -> dict:
    # Scientific imports are delayed so --help and summary discovery remain usable
    # without the HPC-only enterprise/libstempo stack installed.
    import config
    from CGW_SNR import compute_cgw_snr_optimal_population_fast
    from consistent_pop_synth import compute_population_snr
    from data_loader import (
        SCENARIOS,
        filter_pulsars_15yr,
        load_pulsars,
        parse_pulsar_parameters,
    )
    from signal_injection import population_residuals

    summary = load_summary(sim_dir / "summary.pkl.gz")
    index, snr_field, saved_snr = select_loudest(summary)
    scenario = scenario_from_snr_field(snr_field)
    if scenario not in SCENARIOS:
        raise KeyError(
            f"Summary uses scenario {scenario!r}, which is not configured in "
            f"data_loader.SCENARIOS"
        )
    source = make_source(summary, index)
    psrs = load_pulsars(verbose=True, scenario=scenario, scenarios=SCENARIOS)
    psrs_clean, raw_noise, tspan = filter_pulsars_15yr(psrs, verbose=True)
    parsed_noise = parse_pulsar_parameters(config.NOISEFILE)
    names = [psr.name for psr in psrs_clean]
    combined, temporary = load_combined_residuals(sim_dir, names, scenario)
    try:
        toas = {psr.name: np.array(psr.stoas, copy=True) for psr in psrs_clean}
        for psr in psrs_clean:
            if len(combined[psr.name]) != len(psr.stoas):
                raise ValueError(f"Residual/TOA length mismatch for {psr.name}")

        original_signal = {
            psr.name: population_residuals(
                toas[psr.name], psr, [source], tspan,
            )
            for psr in psrs_clean
        }
        baseline_pta = _build_pta(compute_population_snr, psrs_clean, raw_noise,
                                  combined, tspan)
        baseline = float(compute_cgw_snr_optimal_population_fast(
            baseline_pta[1], baseline_pta[0], [source], raw_noise, parsed_noise,
            tspan)[0])
        records = []
        for change in values:
            perturbed, record_change = _apply_change(source, change)
            modified = {
                name: combined[name] - original_signal[name] + population_residuals(
                    toas[name], psr, [perturbed], tspan,
                )
                for name, psr in ((p.name, p) for p in psrs_clean)
            }
            pta, enterprise_psrs = _build_pta(
                compute_population_snr, psrs_clean, raw_noise, modified,
                tspan,
            )[0:2]
            snr = float(compute_cgw_snr_optimal_population_fast(
                enterprise_psrs, pta, [perturbed], raw_noise, parsed_noise,
                tspan)[0])
            records.append({"change": record_change, "cgw_snr_baseline_forecast": snr, "delta": snr - baseline})
        return {
            "simulation": str(sim_dir), "source_index": index,
            "snr_field": snr_field, "scenario": scenario, "saved_snr": saved_snr,
            "tspan_seconds": float(tspan),
            "baseline_recomputed_snr": baseline, "source": vars(source),
            "results": records,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


def _tspan(sim_dir: Path) -> float:
    with (sim_dir / "metadata" / "config.json").open() as stream:
        return float(json.load(stream)["Tspan_seconds"])


def _build_pta(compute_population_snr, psrs, noise, residuals, tspan):
    _, pta, enterprise_psrs = compute_population_snr(
        population=None, psrs_clean=psrs, raw_noise_params=noise,
        Tspan=tspan, current_stoas=residuals, return_psrs_pta=True,
    )
    return pta, enterprise_psrs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs-root", type=Path)
    group.add_argument("--sim-dir", type=Path)
    parser.add_argument("--parameter", required=True, help="Source field or 'sky'")
    parser.add_argument(
        "--values", required=True,
        help=(
            "Comma-separated values; for 'sky' these are semicolon-separated "
            "dra_deg,ddec_deg offsets from the source's saved position, e.g. "
            "'0.5,-0.5;1.0,0.0'"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("cgw_source_perturbations.json"))
    args = parser.parse_args()
    simulations = [args.sim_dir] if args.sim_dir else find_simulations(args.runs_root)
    if not simulations:
        raise FileNotFoundError("No simulation directories containing summary.pkl.gz were found")
    selected = simulations[0] if args.sim_dir else select_loudest_sim(simulations)
    result = run_analysis(selected, args.parameter, parse_values(args.parameter, args.values))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()


"""
Sample Use:

python analyze_cgw_source_perturbations.py \
  --sim-dir runs/2026-07-17_pessimistic/sim342 \
  --parameter sky \
  --values '1.0,0.0;0.0,1.0;1.0,1.0'

Each value is a "dra_deg,ddec_deg" offset (in degrees) applied to the
loudest source's own saved (ra, dec) -- its standard initial sky location,
read straight from the simulation's summary.pkl.gz. Offsets are converted
to radians internally before being added, so you no longer need to know or
pass the absolute coordinates by hand.
"""