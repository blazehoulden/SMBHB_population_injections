#!/usr/bin/env python3
"""Measure how changing one saved CW source changes its optimal S/N.

The script never writes inside a simulation directory.  It accepts either the
normal ``residuals/combined`` directory or a ``combined.tar.gz`` archive and
uses the saved summary to find the loudest source.

Example (on a machine with the project environment installed)::

    python analyze_cgw_source_perturbations.py \
        --runs-root runs --parameter ra --values 0.0,1.0,2.0

The values use the native units in the summary: radians for angles and Hz for
frequency.  ``--parameter sky`` changes both right ascension and declination;
its values must be ``ra,dec`` pairs separated by semicolons.
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
        if name in arrays and name.startswith("cgw_snr")
    ]
    if "cgw_snr" in arrays:
        field = "cgw_snr"
    elif candidates:
        field = candidates[0]
    else:
        raise KeyError("summary contains no cgw_snr array")
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


def load_combined_residuals(sim_dir: Path, names: Iterable[str]) -> tuple[dict[str, np.ndarray], tempfile.TemporaryDirectory | None]:
    residual_dir = sim_dir / "residuals" / "combined"
    temporary = None
    if not residual_dir.is_dir():
        archive = next(iter(sorted((sim_dir / "residuals").glob("combined*.tar.gz"))), None)
        if archive is None:
            raise FileNotFoundError(f"No combined residual directory/archive in {sim_dir / 'residuals'}")
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
    if parameter == "sky":
        values = []
        for item in text.split(";"):
            pair = [float(value.strip()) for value in item.split(",")]
            if len(pair) != 2:
                raise ValueError("sky values must be ra,dec pairs separated by ';'")
            values.append({"ra": pair[0], "dec": pair[1]})
        return values
    if parameter not in PERTURBABLE_FIELDS:
        raise ValueError(f"parameter must be one of {PERTURBABLE_FIELDS} or 'sky'")
    return [{parameter: float(item.strip())} for item in text.split(",") if item.strip()]


def run_analysis(sim_dir: Path, parameter: str, values: list[dict[str, float]]) -> dict:
    # Scientific imports are delayed so --help and summary discovery remain usable
    # without the HPC-only enterprise/libstempo stack installed.
    import config
    from CGW_SNR import compute_cgw_snr_optimal_population_fast
    from consistent_pop_synth import compute_population_snr
    from data_loader import filter_pulsars_15yr, load_pulsars, parse_pulsar_parameters
    from signal_injection import population_residuals_eccentric

    summary = load_summary(sim_dir / "summary.pkl.gz")
    index, snr_field, saved_snr = select_loudest(summary)
    source = make_source(summary, index)
    psrs = load_pulsars(verbose=True)
    psrs_clean, raw_noise, _ = filter_pulsars_15yr(psrs, verbose=True)
    parsed_noise = parse_pulsar_parameters(config.NOISEFILE)
    names = [psr.name for psr in psrs_clean]
    combined, temporary = load_combined_residuals(sim_dir, names)
    try:
        toas = {psr.name: np.array(psr.stoas, copy=True) for psr in psrs_clean}
        for psr in psrs_clean:
            if len(combined[psr.name]) != len(psr.stoas):
                raise ValueError(f"Residual/TOA length mismatch for {psr.name}")

        original_signal = {
            psr.name: population_residuals_eccentric(
                toas[psr.name], psr, [source], float(_tspan(sim_dir)),
            )
            for psr in psrs_clean
        }
        baseline_pta = _build_pta(compute_population_snr, psrs_clean, raw_noise,
                                  combined, float(_tspan(sim_dir)))
        baseline = float(compute_cgw_snr_optimal_population_fast(
            baseline_pta[1], baseline_pta[0], [source], raw_noise, parsed_noise,
            float(_tspan(sim_dir)))[0])
        records = []
        for change in values:
            perturbed = SimpleNamespace(**vars(source))
            for field, value in change.items():
                setattr(perturbed, field, value)
            modified = {
                name: combined[name] - original_signal[name] + population_residuals_eccentric(
                    toas[name], psr, [perturbed], float(_tspan(sim_dir)),
                )
                for name, psr in ((p.name, p) for p in psrs_clean)
            }
            pta, enterprise_psrs = _build_pta(
                compute_population_snr, psrs_clean, raw_noise, modified,
                float(_tspan(sim_dir)),
            )[0:2]
            snr = float(compute_cgw_snr_optimal_population_fast(
                enterprise_psrs, pta, [perturbed], raw_noise, parsed_noise,
                float(_tspan(sim_dir)))[0])
            records.append({"change": change, "cgw_snr": snr, "delta": snr - baseline})
        return {
            "simulation": str(sim_dir), "source_index": index,
            "snr_field": snr_field, "saved_snr": saved_snr,
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
        curn_components=14, rn_components=30,
    )
    return pta, enterprise_psrs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs-root", type=Path)
    group.add_argument("--sim-dir", type=Path)
    parser.add_argument("--parameter", required=True, help="Source field or 'sky'")
    parser.add_argument("--values", required=True, help="Comma-separated values; sky uses ra,dec;ra,dec")
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
