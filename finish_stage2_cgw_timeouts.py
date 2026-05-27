#!/usr/bin/env python3
"""
Scan SMBHB run directories for stage-2 timeout jobs that can be finished from
saved residuals, tally other failures, and optionally submit CGW-only resume
jobs.

Typical use:

  python finish_stage2_cgw_timeouts.py --run-dir runs/2026-05-23_pessimistic
  python finish_stage2_cgw_timeouts.py --run-dir runs/2026-05-23_pessimistic --submit
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
class FinishRecord:
    run_dir: Path
    config: str
    sim_id: int
    attempt: int
    job_id: str
    status: str
    log_path: Path
    has_complete_sentinel: bool
    complete_sentinel_mtime: float | None
    residuals_present: bool
    log_mtime: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan stage2 timeout jobs that can be finished from saved residuals.",
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
        help="Actually submit sbatch finish jobs for eligible timeouts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force report-only mode even if --submit is given.",
    )
    parser.add_argument(
        "--wrapper-script",
        type=Path,
        default=Path(__file__).resolve().parent / "submit_stage2_finish_cgw.sh",
        help="Slurm wrapper used to submit a single finish job.",
    )
    parser.add_argument(
        "--validate-proxy",
        action="store_true",
        help="Forward --validate-proxy to the finish job wrapper.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help="Number of binaries to sample when proxy validation is enabled.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root used in the sbatch wrapper command.",
    )
    parser.add_argument(
        "--env-setup",
        default="module purge; unset PYTHONPATH; module load mamba; mamba activate smbhb312;",
        help="Shell snippet used to activate the job environment.",
    )
    parser.add_argument(
        "--job-name-prefix",
        default="s2_finish",
        help="Prefix used for submitted job names.",
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
        "cgw": True,
        "proxy_only": False,
        "n_test": None,
    }

    if override_args.n_test is not None:
        resolved["n_test"] = override_args.n_test
    if resolved["n_test"] is None:
        resolved["n_test"] = config_data.get("n_test", 1000)

    return resolved


def _iter_stage2_logs(run_dir: Path) -> Iterable[Path]:
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return []
    return sorted(logs_dir.glob("stage2_sim*_attempt*_*.out"))


def _iter_sim_dirs(run_dir: Path) -> Iterable[Path]:
    return sorted(p for p in run_dir.glob("sim[0-9][0-9][0-9]") if p.is_dir())


def _scan_run_dir(run_dir: Path, override_args: argparse.Namespace) -> tuple[list[FinishRecord], dict]:
    run_config = _resolve_run_config(run_dir, override_args)
    records: list[FinishRecord] = []
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

        sim_dir = run_dir / f"sim{sim_id:03d}"
        sentinel = sim_dir / "metadata" / "stage2_complete.json"
        residuals_present = (sim_dir / "residuals" / "combined").is_dir()
        sentinel_mtime = sentinel.stat().st_mtime if sentinel.is_file() else None
        log_mtime = log_path.stat().st_mtime

        records.append(
            FinishRecord(
                run_dir=run_dir,
                config=run_config["config"],
                sim_id=sim_id,
                attempt=attempt,
                job_id=job_id,
                status=status,
                log_path=log_path,
                has_complete_sentinel=sentinel.is_file(),
                complete_sentinel_mtime=sentinel_mtime,
                residuals_present=residuals_present,
                log_mtime=log_mtime,
            )
        )
        seen_sims.add(sim_id)

    for sim_dir in _iter_sim_dirs(run_dir):
        sim_id = int(sim_dir.name[3:])
        if sim_id in seen_sims:
            continue

        sentinel = sim_dir / "metadata" / "stage2_complete.json"
        residuals_present = (sim_dir / "residuals" / "combined").is_dir()
        sentinel_mtime = sentinel.stat().st_mtime if sentinel.is_file() else None
        status = "COMPLETED" if sentinel.is_file() else "MISSING_LOG"

        records.append(
            FinishRecord(
                run_dir=run_dir,
                config=run_config["config"],
                sim_id=sim_id,
                attempt=0,
                job_id="0",
                status=status,
                log_path=run_dir / "logs" / f"stage2_sim{sim_id:03d}_attempt0_missing.out",
                has_complete_sentinel=sentinel.is_file(),
                complete_sentinel_mtime=sentinel_mtime,
                residuals_present=residuals_present,
                log_mtime=None,
            )
        )

    return records, run_config


def _choose_finish_targets(records: list[FinishRecord]) -> tuple[list[FinishRecord], Counter]:
    latest_by_sim: dict[tuple[Path, int], FinishRecord] = {}
    for record in records:
        key = (record.run_dir, record.sim_id)
        current = latest_by_sim.get(key)
        record_key = (record.attempt, int(record.job_id) if record.job_id.isdigit() else -1)
        current_key = (current.attempt, int(current.job_id) if current and current.job_id.isdigit() else -1) if current else None
        if current is None or record_key > current_key:
            latest_by_sim[key] = record

    finish_targets: list[FinishRecord] = []
    tally = Counter()

    for record in latest_by_sim.values():
        sentinel_is_newer = (
            record.has_complete_sentinel
            and record.complete_sentinel_mtime is not None
            and record.log_mtime is not None
            and record.complete_sentinel_mtime >= record.log_mtime
        )

        if record.status == "COMPLETED" or sentinel_is_newer:
            tally[(record.config, "COMPLETED")] += 1
            continue

        if record.status == "TIMEOUT" and record.residuals_present:
            finish_targets.append(record)
            tally[(record.config, "TIMEOUT")] += 1
        elif record.status == "TIMEOUT" and not record.residuals_present:
            tally[(record.config, "TIMEOUT_NO_RESIDUALS")] += 1
        elif record.status == "MISSING_LOG":
            tally[(record.config, record.status)] += 1
        elif record.status in {"FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}:
            tally[(record.config, record.status)] += 1
        elif record.status == "INCOMPLETE":
            tally[(record.config, "INCOMPLETE")] += 1
        else:
            tally[(record.config, f"OTHER:{record.status}")] += 1

    return finish_targets, tally


def _sbatch_command(record: FinishRecord, run_config: dict, args: argparse.Namespace) -> list[str]:
    log_dir = record.run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={args.job_name_prefix}_{record.config}_sim{record.sim_id:03d}",
        "--nodes=1",
        "--ntasks=1",
        f"--output={log_dir / f'stage2_finish_sim{record.sim_id:03d}_%j.out'}",
        f"--error={log_dir / f'stage2_finish_sim{record.sim_id:03d}_%j.err'}",
        str(args.wrapper_script),
        "--output-dir",
        str(record.run_dir),
        "--sim-id",
        str(record.sim_id),
        "--config",
        run_config["config"],
    ]
    if args.validate_proxy:
        cmd.append("--validate-proxy")
        cmd.extend(["--n-test", str(run_config["n_test"])])
    return cmd


def _submit_finish(record: FinishRecord, run_config: dict, args: argparse.Namespace) -> str:
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

    all_records: list[FinishRecord] = []
    run_configs: dict[Path, dict] = {}
    for run_dir in run_dirs:
        try:
            records, run_config = _scan_run_dir(run_dir, args)
        except Exception as exc:
            print(f"{run_dir}: {exc}", file=sys.stderr)
            continue
        all_records.extend(records)
        run_configs[run_dir] = run_config

    finish_targets, tally = _choose_finish_targets(all_records)

    print(f"Scanned {len(run_dirs)} run directory(ies)")
    print(f"Found {len(all_records)} stage2 record(s)")
    print(f"Finish jobs eligible for submission: {len(finish_targets)}")
    print()

    if finish_targets:
        print("Finish submissions:")
        for record in finish_targets:
            residual_flag = "residuals" if record.residuals_present else "no-residuals"
            print(
                f"  {record.run_dir.name} sim{record.sim_id:03d} "
                f"attempt{record.attempt} job {record.job_id} -> {record.status} ({residual_flag})"
            )
        print()

    print("Failure tally:")
    if tally:
        for line in _format_tally(tally):
            print(line)
    else:
        print("  no incomplete or failed stage2 jobs found")

    report = {
        "run_dirs": [str(p) for p in run_dirs],
        "total_records": len(all_records),
        "finish_targets": [
            {
                "run_dir": str(record.run_dir),
                "config": record.config,
                "sim_id": record.sim_id,
                "attempt": record.attempt,
                "job_id": record.job_id,
                "status": record.status,
                "log_path": str(record.log_path),
                "residuals_present": record.residuals_present,
            }
            for record in finish_targets
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

    if not finish_targets:
        return 0

    print("\nSubmitting finish jobs:")
    for record in finish_targets:
        run_config = run_configs[record.run_dir]
        submitted_job_id = _submit_finish(record, run_config, args)
        print(f"  {record.run_dir.name} sim{record.sim_id:03d} attempt{record.attempt + 1} -> {submitted_job_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())