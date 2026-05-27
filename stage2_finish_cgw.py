#!/usr/bin/env python3
"""
Stage 2 resume path for CGW-only completion.

This script is meant for timeout jobs that already finished the heavy stage-2
work, saved the residual snapshots, and only need the final CGW candidate pass.
It reloads the saved combined residuals, reconstructs the PTA from those
residuals, computes CGW SNRs for the global candidate set, and writes the same
per-sim summary artifacts as the full stage-2 pipeline.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def _load_combined_residuals(residual_dir: Path, psr_names: list[str]) -> dict[str, np.ndarray]:
    combined: dict[str, np.ndarray] = {}
    missing: list[str] = []

    for name in psr_names:
        fpath = residual_dir / f"{name}.npy"
        if not fpath.is_file():
            missing.append(str(fpath))
            continue
        combined[name] = np.load(fpath).astype(np.float64, copy=False)

    if missing:
        raise FileNotFoundError(
            "Missing combined residual files:\n  " + "\n  ".join(missing)
        )

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume stage 2 from saved residuals and finish CGW analysis",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sim-id", type=int, required=True)
    parser.add_argument(
        "--config",
        default="optimistic",
        choices=list(config.POPULATION_CONFIGS.keys()),
    )
    parser.add_argument(
        "--validate-proxy",
        action="store_true",
        help="Validate the CGW proxy before the full SNR pass",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1_000,
        help="Number of binaries to sample when validating the proxy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after checking the residual inputs and chunk store",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    sim_out_dir = Path(args.output_dir) / f"sim{args.sim_id:03d}"
    config_path = sim_out_dir / "metadata" / "config.json"
    residual_dir = sim_out_dir / "residuals" / "combined"
    pop_dir = sim_out_dir / "populations"

    print(f"\n{'='*60}")
    print(f"Stage 2 CGW resume — sim_id={args.sim_id}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Config     : {args.config}")
    print(f"  Residuals  : {residual_dir}")
    print(f"{'='*60}\n")

    if not config_path.is_file():
        print(f"ERROR: config.json not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    if not residual_dir.is_dir():
        print(f"ERROR: combined residuals directory not found at {residual_dir}", file=sys.stderr)
        sys.exit(1)

    with config_path.open() as fh:
        run_config = json.load(fh)
    Tspan_seconds = run_config["Tspan_seconds"]

    if args.dry_run:
        manifest_path = sim_out_dir / "residuals" / "manifest.json"
        print("\nDry run complete.")
        print(f"  Residual manifest: {'present' if manifest_path.is_file() else 'missing'}")
        print(f"  Combined residual files: {len(list(residual_dir.glob('*.npy')))}")
        return

    from stage1_setup import ShardedPickleStore
    from data_loader import filter_pulsars_15yr, load_pulsars, parse_pulsar_parameters
    from consistent_pop_synth import compute_population_snr
    from stage2_inject import _build_summary_object, _compute_cgw_snrs

    print("📡 Loading pulsars...")
    psrs_unfiltered = load_pulsars(verbose=True)
    psrs_clean, raw_noise_params, _ = filter_pulsars_15yr(psrs_unfiltered, verbose=True)
    print(f"✓ {len(psrs_clean)} pulsars loaded\n")

    parsed_noise_params = parse_pulsar_parameters(config.NOISEFILE)
    psr_names = [psr.name for psr in psrs_clean]
    combined_stoas = _load_combined_residuals(residual_dir, psr_names)

    for psr in psrs_clean:
        residuals = combined_stoas[psr.name]
        if len(residuals) != len(psr.stoas):
            raise ValueError(
                f"Residual length mismatch for {psr.name}: "
                f"{len(residuals)} saved vs {len(psr.stoas)} pulsar stoas"
            )

    store = ShardedPickleStore(pop_dir)
    chunk_ids = store.available()
    if not chunk_ids:
        print(f"ERROR: no population shards found in {pop_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"📦 Found {len(chunk_ids)} chunks: {chunk_ids}")

    print("\n📂 Rebuilding PTA from combined residuals...")
    snr, pta, enterprise_psrs = compute_population_snr(
        population=None,
        psrs_clean=psrs_clean,
        raw_noise_params=raw_noise_params,
        Tspan=Tspan_seconds,
        current_stoas=combined_stoas,
        return_psrs_pta=True,
        curn_components=14,
        rn_components=30,
    )
    print(f"✓ PTA rebuilt; OS SNR = {snr:.4f}")

    if args.validate_proxy:
        from debug.test_cgw_proxy import validate_cgw_proxy

        validate_cgw_proxy(
            store=store,
            chunk_ids=chunk_ids,
            pta=pta,
            enterprise_psrs=enterprise_psrs,
            raw_noise_params=raw_noise_params,
            parsed_noise_params=parsed_noise_params,
            Tspan_seconds=Tspan_seconds,
            n_test=args.n_test,
        )

    print("\n🔭 Computing CGW SNRs from saved shards...")
    _compute_cgw_snrs(
        store=store,
        chunk_ids=chunk_ids,
        pta=pta,
        enterprise_psrs=enterprise_psrs,
        raw_noise_params=raw_noise_params,
        parsed_noise_params=parsed_noise_params,
        Tspan_seconds=Tspan_seconds,
    )
    print("✓ CGW complete")

    print("\n🧾 Building summary object...")
    _build_summary_object(args.output_dir, sim_id=args.sim_id, n_keep_per_category=200)

    sentinel = sim_out_dir / "metadata" / "stage2_complete.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    with sentinel.open("w") as fh:
        json.dump(
            {
                "sim_id": args.sim_id,
                "resumed_from_residuals": True,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            fh,
            indent=2,
        )

    elapsed = time.time() - t0
    print(f"\n✅ Stage 2 CGW resume complete — sim_id={args.sim_id} in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()