#!/usr/bin/env python3
"""
Scan SMBHB run directories for stage2 Slurm logs, resubmit timeout jobs with
longer wallclock, and tally non-timeout failures by configuration.

Typical use:

  python retry_stage2_timeouts.py --submit --s2-time 04:00:00

By default the script scans yesterday's run directories under ./runs and only
prints a report. Use --submit to actually queue reruns.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


JOB_REPORT_RE = re.compile(r"Job Report:\s+(\d+)\s+\(([^)]+)\)")
STAGE2_LOG_RE = re.compile(r"^stage2_sim(?P<sim>\d{3})_attempt(?P<attempt>\d+?)_(?P<jobid>\d+)\.out$")


@dataclasses.dataclass
class Stage2Record:
    run_dir: Path
    config: str
    sim_id: int
    attempt: int
    job_id: str
    status: str
    log_path: Path
    has_complete_sentinel: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resubmit timed-out stage2 jobs and tally other failures by config."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Directory containing run folders (default: runs).",
    )
    parser.add_argument(
        "--date",
        action="append",
        default=None,
        help="Run date to scan in YYYY-MM-DD format. Can be provided multiple times.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=None,
        help="Explicit run directory to scan. Can be provided multiple times.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit sbatch reruns for timeout jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force report-only mode even if --submit is given.",
    )
    parser.add_argument(
        "--s2-time",
        default="04:00:00",
        help="Wallclock to use for resubmitted stage2 jobs.",
    )
    parser.add_argument(
        "--s2-mem",
        default="26G",
        help="Memory to use for resubmitted stage2 jobs.",
    )
    parser.add_argument(
        "--s2-cpus",
        type=int,
        default=1,
        help="CPUs per resubmitted stage2 job.",
    )
    parser.add_argument(
        "--target-snr",
        type=float,
        default=None,
        help="Override target SNR instead of reading metadata/config.json.",
    )
    parser.add_argument(
        "--snr-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Override stage2 SNR range instead of reading metadata/config.json.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Override stage2 --n-test.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root containing stage2_inject.py.",
    )
    parser.add_argument(
        "--env-setup",
        default="module purge; unset PYTHONPATH; module load mamba; mamba activate smbhb312;",
        help="Shell snippet used to activate the job environment.",
    )
    parser.add_argument(
        "--job-name-prefix",
        default="s2_retry",
        help="Prefix used for resubmitted job names.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write the scan report as JSON.",
    )
    return parser.parse_args()


def _default_yesterday() -> str:
    return (dt.date.today() - dt.timedelta(days=1)).isoformat()


def _discover_run_dirs(output_root: Path, date_filters: list[str] | None, explicit_dirs: list[Path] | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_dirs:
        candidates.extend(explicit_dirs)

    if date_filters is None:
        date_filters = [] if explicit_dirs else [_default_yesterday()]

    for date_str in date_filters:
        candidates.extend(sorted(output_root.glob(f"{date_str}_*")))

    seen: set[Path] = set()
    run_dirs: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        if candidate.is_dir():
            seen.add(candidate)
            run_dirs.append(candidate)
    return run_dirs


def _read_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _read_tail_status(log_path: Path) -> tuple[str, str | None]:
    """Return (status, job_id). Status defaults to INCOMPLETE if no footer is found."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return "MISSING_LOG", None

    for line in reversed(lines):
        match = JOB_REPORT_RE.search(line)
        if match:
            job_id, status = match.group(1), match.group(2).strip().upper()
            return status, job_id
    return "INCOMPLETE", None


def _load_run_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / "sim000" / "metadata" / "config.json"
    if meta_path.is_file():
        try:
            return _read_json(meta_path)
        except Exception:
            pass
    return {}


def _resolve_run_config(run_dir: Path, override_args: argparse.Namespace) -> dict:
    config_data = _load_run_metadata(run_dir)
    resolved = {
        "config": run_dir.name.split("_", 1)[1] if "_" in run_dir.name else run_dir.name,
        "target_snr": config_data.get("target_snr"),
        "snr_range": config_data.get("snr_range"),
        "n_chunks": config_data.get("n_chunks"),
        "chunk_size": config_data.get("chunk_size"),
        "proxy_only": False,
        "cgw": True,
        "n_test": None,
    }

    if override_args.target_snr is not None:
        resolved["target_snr"] = override_args.target_snr
    if override_args.snr_range is not None:
        resolved["snr_range"] = [float(override_args.snr_range[0]), float(override_args.snr_range[1])]
    if override_args.n_test is not None:
        resolved["n_test"] = override_args.n_test

    if resolved["target_snr"] is None:
        resolved["target_snr"] = 3.75
    if resolved["snr_range"] is None:
        resolved["snr_range"] = [3.5, 4.0]
    if resolved["n_chunks"] is None:
        raise RuntimeError(f"Missing n_chunks metadata for {run_dir}")
    if resolved["chunk_size"] is None:
        raise RuntimeError(f"Missing chunk_size metadata for {run_dir}")
    if resolved["n_test"] is None:
        resolved["n_test"] = 1000

    return resolved


def _iter_stage2_logs(run_dir: Path) -> Iterable[Path]:
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return []
    return sorted(logs_dir.glob("stage2_sim*_attempt*_*.out"))


def _iter_sim_dirs(run_dir: Path) -> Iterable[Path]:
    return sorted(p for p in run_dir.glob("sim[0-9][0-9][0-9]") if p.is_dir())


def _scan_run_dir(run_dir: Path, override_args: argparse.Namespace) -> tuple[list[Stage2Record], dict]:
    run_config = _resolve_run_config(run_dir, override_args)
    records: list[Stage2Record] = []
    seen_sims: set[int] = set()

    for log_path in _iter_stage2_logs(run_dir):
        match = STAGE2_LOG_RE.match(log_path.name)
        if not match:
            continue

        sim_id = int(match.group("sim"))
        attempt = int(match.group("attempt"))
        job_id = match.group("jobid")
        status, footer_job_id = _read_tail_status(log_path)

        if footer_job_id is not None:
            job_id = footer_job_id

        sentinel = run_dir / f"sim{sim_id:03d}" / "metadata" / "stage2_complete.json"
        has_complete = sentinel.is_file()

        records.append(
            Stage2Record(
                run_dir=run_dir,
                config=run_config["config"],
                sim_id=sim_id,
                attempt=attempt,
                job_id=job_id,
                status=status,
                log_path=log_path,
                has_complete_sentinel=has_complete,
            )
        )
        seen_sims.add(sim_id)

    for sim_dir in _iter_sim_dirs(run_dir):
        sim_id = int(sim_dir.name[3:])
        if sim_id in seen_sims:
            continue

        sentinel = sim_dir / "metadata" / "stage2_complete.json"
        has_complete = sentinel.is_file()
        status = "COMPLETED" if has_complete else "MISSING_LOG"

        records.append(
            Stage2Record(
                run_dir=run_dir,
                config=run_config["config"],
                sim_id=sim_id,
                attempt=0,
                job_id="0",
                status=status,
                log_path=run_dir / "logs" / f"stage2_sim{sim_id:03d}_attempt0_missing.out",
                has_complete_sentinel=has_complete,
            )
        )

    return records, run_config


def _choose_retry_targets(records: list[Stage2Record]) -> tuple[list[Stage2Record], Counter]:
    latest_by_sim: dict[tuple[Path, int], Stage2Record] = {}
    for record in records:
        key = (record.run_dir, record.sim_id)
        current = latest_by_sim.get(key)
        record_key = (record.attempt, int(record.job_id) if record.job_id.isdigit() else -1)
        current_key = (current.attempt, int(current.job_id) if current and current.job_id.isdigit() else -1) if current else None
        if current is None or record_key > current_key:
            latest_by_sim[key] = record

    retry_targets: list[Stage2Record] = []
    tally = Counter()

    for record in latest_by_sim.values():
        if record.has_complete_sentinel or record.status == "COMPLETED":
            tally[(record.config, "COMPLETED")] += 1
            continue

        if record.status == "TIMEOUT":
            retry_targets.append(record)
            tally[(record.config, record.status)] += 1
        elif record.status == "MISSING_LOG":
            tally[(record.config, record.status)] += 1
        elif record.status in {"FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}:
            tally[(record.config, record.status)] += 1
        elif record.status == "INCOMPLETE":
            tally[(record.config, "INCOMPLETE")] += 1
        else:
            tally[(record.config, f"OTHER:{record.status}")] += 1

    return retry_targets, tally


def _sbatch_command(record: Stage2Record, run_config: dict, args: argparse.Namespace) -> list[str]:
    out_dir = record.run_dir
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cgw_flag = "--cgw" if run_config.get("cgw", True) else ""
    proxy_only_flag = "--proxy-only" if run_config.get("proxy_only", False) else ""

    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={args.job_name_prefix}_{record.config}_sim{record.sim_id:03d}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={args.s2_cpus}",
        f"--mem={args.s2_mem}",
        f"--time={args.s2_time}",
        f"--output={log_dir / f'stage2_retry_sim{record.sim_id:03d}_%j.out'}",
        f"--error={log_dir / f'stage2_retry_sim{record.sim_id:03d}_%j.err'}",
        "--wrap",
        (
            f"{args.env_setup} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            f"python -u {args.repo_dir / 'stage2_inject.py'} "
            f"--output-dir {out_dir} "
            f"--config {run_config['config']} "
            f"--target-snr {run_config['target_snr']} "
            f"--snr-range {run_config['snr_range'][0]} {run_config['snr_range'][1]} "
            f"--sim-id {record.sim_id} "
            f"--n-chunks {run_config['n_chunks']} "
            f"--n-test {run_config['n_test']} "
            f"{cgw_flag} {proxy_only_flag}"
        ).strip(),
    ]
    return cmd


def _submit_retry(record: Stage2Record, run_config: dict, args: argparse.Namespace) -> str:
    cmd = _sbatch_command(record, run_config, args)
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _format_tally(tally: Counter) -> list[str]:
    lines: list[str] = []
    grouped = defaultdict(Counter)
    for (config, status), count in tally.items():
        grouped[config][status] = count

    for config in sorted(grouped):
        lines.append(f"{config}:")
        for status, count in grouped[config].most_common():
            lines.append(f"  {status}: {count}")
    return lines


def main() -> int:
    args = _parse_args()
    do_submit = args.submit and not args.dry_run

    run_dirs = _discover_run_dirs(args.output_root, args.date, args.run_dir)
    if not run_dirs:
        print("No matching run directories found.", file=sys.stderr)
        return 1

    all_records: list[Stage2Record] = []
    run_configs: dict[Path, dict] = {}
    for run_dir in run_dirs:
        try:
            records, run_config = _scan_run_dir(run_dir, args)
        except Exception as exc:
            print(f"{run_dir}: {exc}", file=sys.stderr)
            continue
        all_records.extend(records)
        run_configs[run_dir] = run_config

    retry_targets, tally = _choose_retry_targets(all_records)

    print(f"Scanned {len(run_dirs)} run directory(ies)")
    print(f"Found {len(all_records)} stage2 record(s)")
    print(f"Retry jobs eligible for resubmission: {len(retry_targets)}")
    print()

    if retry_targets:
        print("Retry submissions:")
        for record in retry_targets:
            print(f"  {record.run_dir.name} sim{record.sim_id:03d} attempt{record.attempt} job {record.job_id} -> {record.status}")
        print()

    print("Failure tally:")
    if tally:
        for line in _format_tally(tally):
            print(line)
    else:
        print("  no incomplete or failed stage2 jobs found")

    report = {
        "run_dirs": [str(p) for p in run_dirs],
        "total_logs": len(all_records),
        "timeout_targets": [
            {
                "run_dir": str(record.run_dir),
                "config": record.config,
                "sim_id": record.sim_id,
                "attempt": record.attempt,
                "job_id": record.job_id,
                "log_path": str(record.log_path),
            }
            for record in retry_targets
        ],
        "tally": {
            f"{config}:{status}": count
            for (config, status), count in tally.items()
        },
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote report JSON: {args.report_json}")

    if not do_submit:
        if args.submit and args.dry_run:
            print("\nSubmission suppressed because --dry-run was set.")
        return 0

    if not retry_targets:
        return 0

    print("\nSubmitting retry jobs:")
    for record in retry_targets:
        run_config = run_configs[record.run_dir]
        submitted_job_id = _submit_retry(record, run_config, args)
        print(
            f"  {record.run_dir.name} sim{record.sim_id:03d} "
            f"attempt{record.attempt + 1} -> {submitted_job_id}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())