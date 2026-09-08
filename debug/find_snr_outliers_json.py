#!/usr/bin/env python3
"""
Scan a structured data directory for simulation runs and report any where
the mean SNR (from SNR_final in the .json file) falls outside [SNR_MIN, SNR_MAX].

Directory structure expected:
    data/
      2026-04-30/
        optimistic_run_1/
          *.json
          *.pkl.gz
        pessimistic_run_40/
          ...

Usage:
    python find_snr_outliers_json.py /path/to/data
    python find_snr_outliers_json.py /path/to/data --range 3.5 4.25
    python find_snr_outliers_json.py /path/to/data --outdir ~/results
    python find_snr_outliers_json.py /path/to/data --delete   # DELETE outlier dirs!
"""

import json
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime


SCENARIOS = ("optimistic", "pessimistic", "realistic")


def find_json_file(sim_dir):
    """Return the first .json file found in a simulation directory, or None."""
    candidates = list(sim_dir.glob("*.json"))
    return candidates[0] if candidates else None


def extract_mean_snr(json_path):
    """
    Parse the .json file and return the mean SNR_final across all populations,
    or None if missing. Returns (mean_snr, error_message).
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        populations = data.get("populations", [])
        if not populations:
            return None, "populations list is empty"
        
        snr_values = []
        for pop in populations:
            snr = pop.get("SNR_final")
            if snr is not None:
                snr_values.append(float(snr))
        
        if not snr_values:
            return None, "SNR_final not found in any population entry"
        
        return sum(snr_values) / len(snr_values), None  # mean across all entries

    except json.JSONDecodeError as e:
        return None, "JSON parse error: {}".format(e)
    except Exception as e:
        return None, "Error reading JSON: {}".format(e)


def parse_scenario(sim_name):
    """Extract scenario type from directory name, e.g. 'optimistic_run_1' -> 'optimistic'."""
    for s in SCENARIOS:
        if sim_name.startswith(s):
            return s
    return "unknown"


def scan_data(data_dir, snr_min, snr_max, outdir, do_delete, dry_run):
    """
    Walk data_dir -> date dirs -> sim dirs, check SNR, report outliers.
    """

    # Collect all simulation directories (any depth-2 subdir matching the pattern)
    # Auto-detect whether the user passed the root data dir or a specific date dir.
    # A "sim dir" is identified by containing a .json file or matching the naming pattern.
    # If the immediate children look like sim dirs (e.g. optimistic_run_1), treat
    # data_dir itself as the single date dir. Otherwise treat children as date dirs.

    def looks_like_sim_dir(d):
        return any(d.glob("*.json")) or any(
            d.name.startswith(s) for s in SCENARIOS
        )

    immediate_children = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not immediate_children:
        print("No subdirectories found in {}".format(data_dir))
        sys.exit(1)

    # If children look like sim dirs, wrap data_dir as a single date dir
    if any(looks_like_sim_dir(c) for c in immediate_children):
        date_dirs = [data_dir]   # data_dir IS the date dir
    else:
        date_dirs = immediate_children  # data_dir contains date dirs

    skipped = []
    results = []
    for date_dir in date_dirs:
        sim_dirs = sorted([s for s in date_dir.iterdir() if s.is_dir()])
        for sim_dir in sim_dirs:
            json_path = find_json_file(sim_dir)
            if json_path is None:
                skipped.append("{}/{}".format(date_dir.name, sim_dir.name))
                continue

            mean_snr, err = extract_mean_snr(json_path)
            scenario = parse_scenario(sim_dir.name)

            results.append({
                "date":     date_dir.name,
                "sim":      sim_dir.name,
                "scenario": scenario,
                "path":     sim_dir,
                "snr":      mean_snr,
                "error":    err,
            })

    # Categorise
    outliers   = [r for r in results if r["snr"] is not None and
                  (r["snr"] < snr_min or r["snr"] > snr_max)]
    in_range   = [r for r in results if r["snr"] is not None and
                  snr_min <= r["snr"] <= snr_max]
    parse_errs = [r for r in results if r["error"] is not None]

    # Scenario breakdown for outliers
    scenario_counts = {}
    for r in outliers:
        scenario_counts[r["scenario"]] = scenario_counts.get(r["scenario"], 0) + 1

    # ── Build report ─────────────────────────────────────────────────────────
    now = datetime.now()
    timestamp_display = now.strftime("%A %d %B %Y  %H:%M:%S")
    timestamp_file    = now.strftime("%Y%m%d_%H%M%S")

    sep  = "=" * 70
    sep2 = "-" * 70
    lines = []

    lines.append(sep)
    lines.append("  SNR Outlier Report (JSON-based)")
    lines.append("  Generated : {}".format(timestamp_display))
    lines.append("  Data dir  : {}".format(data_dir))
    lines.append("  SNR range : [{}, {}]".format(snr_min, snr_max))
    if do_delete:
        mode = "DRY RUN - no files deleted" if dry_run else "DELETE MODE ACTIVE"
        lines.append("  Mode      : *** {} ***".format(mode))
    lines.append(sep)

    # Outliers table
    if outliers:
        lines.append("")
        lines.append("  OUTSIDE RANGE  ({} simulation(s)):".format(len(outliers)))
        lines.append("")
        lines.append("  {:<12} {:<35} {:<14} {:>10}  {}".format(
            "Date", "Simulation", "Scenario", "Mean SNR", "Flag"))
        lines.append("  " + sep2)
        for r in sorted(outliers, key=lambda x: x["snr"]):
            flag = "LOW" if r["snr"] < snr_min else "HIGH"
            lines.append("  {:<12} {:<35} {:<14} {:>10.4f}  [{}]".format(
                r["date"], r["sim"], r["scenario"], r["snr"], flag))
    else:
        lines.append("")
        lines.append("  All simulations are within the SNR range. No outliers found.")

    # Scenario breakdown
    if scenario_counts:
        lines.append("")
        lines.append("  OUTLIER BREAKDOWN BY SCENARIO:")
        for scenario in SCENARIOS:
            count = scenario_counts.get(scenario, 0)
            if count:
                lines.append("    {:>12} : {} outlier(s)".format(scenario, count))

    # Parse errors
    if parse_errs:
        lines.append("")
        lines.append("  JSON ERRORS  ({} file(s)):".format(len(parse_errs)))
        for r in parse_errs:
            lines.append("    {}/{} — {}".format(r["date"], r["sim"], r["error"]))

    # Incomplete runs (no json)
    if skipped:
        lines.append("")
        lines.append("  INCOMPLETE RUNS (no .json found)  ({} dir(s)):".format(len(skipped)))
        for s in skipped:
            lines.append("    {}".format(s))

    # Summary
    total = len(results) + len(skipped)
    lines.append("")
    lines.append("  " + sep2)
    lines.append("  Summary:")
    lines.append("    Total sim dirs found  : {}".format(total))
    lines.append("    Complete (has .json)  : {}".format(len(results)))
    lines.append("    In range              : {}".format(len(in_range)))
    lines.append("    Outside range         : {}".format(len(outliers)))
    lines.append("    JSON errors           : {}".format(len(parse_errs)))
    lines.append("    Incomplete (no .json) : {}".format(len(skipped)))
    lines.append("  " + sep2)

    # Deletion section
    deleted, delete_failed = [], []
    if do_delete and outliers:
        lines.append("")
        if dry_run:
            lines.append("  DRY RUN — directories that WOULD be deleted:")
        else:
            lines.append("  DELETED DIRECTORIES:")
        lines.append("")
        for r in outliers:
            if dry_run:
                lines.append("    [DRY RUN] {}".format(r["path"]))
            else:
                try:
                    shutil.rmtree(r["path"])
                    deleted.append(str(r["path"]))
                    lines.append("    [DELETED] {}".format(r["path"]))
                except Exception as e:
                    delete_failed.append((str(r["path"]), str(e)))
                    lines.append("    [FAILED]  {} — {}".format(r["path"], e))
        if delete_failed:
            lines.append("")
            lines.append("  DELETION FAILURES: {}".format(len(delete_failed)))

    lines.append(sep)
    lines.append("")

    report = "\n".join(lines)

    # Print to console
    print("\n" + report)

    # Save report to file
    outdir = Path(outdir).expanduser().resolve() if outdir else data_dir
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = "snr_outliers_dryrun_" if (do_delete and dry_run) else "snr_outliers_"
    out_path = outdir / "{}{}.txt".format(prefix, timestamp_file)
    out_path.write_text(report, encoding="utf-8")
    print("  Report saved to: {}\n".format(out_path))


def main():
    ap = argparse.ArgumentParser(
        description="Find simulation runs with mean SNR outside a target range."
    )
    ap.add_argument("data_dir",
                    help="Root data directory (contains date subdirs)")
    ap.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"),
                    default=[3.5, 4.25],
                    help="Acceptable SNR range (default: 3.5 4.25)")
    ap.add_argument("--outdir", default=None,
                    help="Where to save the report (default: data_dir)")
    ap.add_argument("--delete", action="store_true",
                    help="Delete outlier simulation directories (use with --no-dry-run to confirm)")
    ap.add_argument("--no-dry-run", action="store_true",
                    help="Actually perform deletions (only used if --delete is set)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print("Error: '{}' is not a directory.".format(data_dir))
        sys.exit(1)

    # Delete is only real when BOTH --delete AND --no-dry-run are passed
    do_delete = args.delete
    dry_run   = not args.no_dry_run   # dry_run=True unless --no-dry-run is explicitly given

    if args.no_dry_run and not args.delete:
        print("Warning: --no-dry-run has no effect without --delete.")

    scan_data(data_dir, args.range[0], args.range[1], args.outdir, do_delete, dry_run)


if __name__ == "__main__":
    main()
