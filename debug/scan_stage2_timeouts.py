#!/usr/bin/env python3
"""
Stage 2 Timeout Scanner and Recovery Classifier

Reads all stage2 .out files, classifies each simulation's completion status,
and identifies which ones are resumable (got past SNR shard selection).

Usage:
  python3 scan_stage2_timeouts.py --logs-dir /path/to/logs --output results.csv
  
Output CSV:
  sim_id,status,phase_reached,resumable,reason,log_file
"""

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path


def classify_sim(content, sim_id):
    """
    Classify a simulation's completion status.
    
    Returns dict with:
      sim_id: int
      status: "incomplete_early" | "incomplete_late" | "complete" | "error" | "unknown"
      phase_reached: str (descriptive)
      resumable: bool (can retry and finish)
      reason: str (explanation)
    """
    
    has_timeout = "TIMEOUT" in content
    has_error = bool(re.search(r"\nERROR:|Traceback|RuntimeError|ValueError|sys\.exit", content))
    
    # Check for successful SNR shard selection (critical gate for resumability)
    has_snr_convergence = bool(re.search(
        r"✓ Converged.*?k=\d+.*?sub-chunks.*?SNR=|"
        r"✓ Chunk-addition found:|"
        r"Using closest: k=",
        content, re.DOTALL | re.IGNORECASE
    ))
    
    # Check baseline completion markers
    has_handoff = "✓ [baseline] Wrote phase_handoff.json" in content
    has_cgw_complete = "✓ Baseline CGW complete" in content

    has_stage2_complete = bool(
        re.search(
            rf"Stage 2 .*complete.*sim_id={sim_id}|"
            rf"Stage 2 complete.*sim_id={sim_id}",
            content,
            re.IGNORECASE
        )
    )

    if has_stage2_complete:
        return {
            "sim_id": sim_id,
            "status": "complete",
            "phase_reached": "all phases",
            "resumable": False,
            "reason": "Explicit Stage 2 completion marker found"
        }
    
    # Count scenario progress
    scenario_starts = len(re.findall(r"▶ Subprocess: sim\d+/(\S+)", content))
    scenario_completes = len(re.findall(r"✓ \[(\S+)\] Phase complete", content))
    
    # Decision logic
    if has_error and not has_snr_convergence:
        return {
            "sim_id": sim_id,
            "status": "error",
            "phase_reached": "early (before SNR selection)",
            "resumable": False,
            "reason": "Explicit error before SNR convergence; do not retry"
        }
    
    if not has_snr_convergence and has_timeout:
        return {
            "sim_id": sim_id,
            "status": "incomplete_early",
            "phase_reached": "SNR shard selection (failed to converge)",
            "resumable": False,
            "reason": "Timed out before finding population SNR in target band; no resumable state"
        }
    
    if not has_snr_convergence and not has_timeout:
        return {
            "sim_id": sim_id,
            "status": "incomplete_early",
            "phase_reached": "SNR shard selection (no convergence, no timeout)",
            "resumable": False,
            "reason": "Job stopped without finding population SNR; unclear why"
        }
    
    # SNR convergence achieved
    if not has_timeout:
        if scenario_completes == scenario_starts and has_cgw_complete and has_handoff:
            return {
                "sim_id": sim_id,
                "status": "complete",
                "phase_reached": "all phases" if scenario_starts > 0 else "baseline",
                "resumable": False,
                "reason": "Completed successfully; no retry needed"
            }
        else:
            # Job still running or incomplete but no timeout signal
            return {
                "sim_id": sim_id,
                "status": "incomplete_late",
                "phase_reached": f"baseline+scenarios ({scenario_completes}/{scenario_starts} done)",
                "resumable": True,
                "reason": "SNR converged but all scenarios not done; can resume"
            }
    
    # SNR convergence achieved AND timeout
    if has_snr_convergence and has_timeout:
        if has_cgw_complete and has_handoff:
            if scenario_starts > 0:
                return {
                    "sim_id": sim_id,
                    "status": "incomplete_late",
                    "phase_reached": f"scenario {scenario_completes+1}/{scenario_starts}",
                    "resumable": True,
                    "reason": f"Baseline+CGW done; timed out in scenario phases ({scenario_completes}/{scenario_starts} completed)"
                }
            else:
                return {
                    "sim_id": sim_id,
                    "status": "incomplete_late",
                    "phase_reached": "baseline completion",
                    "resumable": True,
                    "reason": "Baseline phase complete; timed out before scenario phases started"
                }
        else:
            # SNR converged but CGW or handoff incomplete
            return {
                "sim_id": sim_id,
                "status": "incomplete_late",
                "phase_reached": "baseline (CGW/handoff incomplete)",
                "resumable": True,
                "reason": "SNR shard selection complete; timed out in baseline CGW computation or handoff write"
            }
    
    return {
        "sim_id": sim_id,
        "status": "unknown",
        "phase_reached": "unknown",
        "resumable": False,
        "reason": "Could not classify"
    }


def scan_logs(logs_dir, output_csv):
    """
    Scan all stage2_*.out files in logs_dir and classify them.
    Write results to output_csv.
    """
    logs_path = Path(logs_dir)
    if not logs_path.is_dir():
        print(f"ERROR: {logs_dir} not found", file=sys.stderr)
        sys.exit(1)
    
    # Find all stage2 .out files
    stage2_pattern = str(logs_path / "stage2_sim*.out")
    out_files = sorted(glob.glob(stage2_pattern))
    
    if not out_files:
        print(f"WARNING: No stage2_sim*.out files found in {logs_dir}", file=sys.stderr)
        return
    
    print(f"Found {len(out_files)} stage2_sim*.out files")
    
    results = []
    for out_file in out_files:
        # Extract sim_id from filename: stage2_sim{id:03d}_try{attempt}_{jobid}.out
        match = re.search(r"stage2_sim(\d+)_", os.path.basename(out_file))
        if not match:
            print(f"  Skipping {os.path.basename(out_file)} (no sim_id)", file=sys.stderr)
            continue
        
        sim_id = int(match.group(1))
        
        # Read file
        try:
            with open(out_file, 'r', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"  ERROR reading {out_file}: {e}", file=sys.stderr)
            results.append({
                "sim_id": sim_id,
                "status": "error",
                "phase_reached": "unknown (file read error)",
                "resumable": False,
                "reason": f"Could not read log file: {e}",
                "log_file": out_file
            })
            continue
        
        # Classify
        result = classify_sim(content, sim_id)
        result["log_file"] = out_file
        results.append(result)
    
    # Write CSV
    if not results:
        print("No results to write", file=sys.stderr)
        return
    
    fieldnames = ["sim_id", "status", "phase_reached", "resumable", "reason", "log_file"]
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    print(f"\n✓ Wrote {len(results)} results to {output_csv}")
    
    # Summary
    resumable = [r for r in results if r["resumable"]]
    early_fail = [r for r in results if r["status"] == "incomplete_early"]
    complete = [r for r in results if r["status"] == "complete"]
    errors = [r for r in results if r["status"] == "error"]
    
    print(f"\nSummary:")
    print(f"  Complete:           {len(complete):4d}")
    print(f"  Incomplete (early):  {len(early_fail):4d}  (do not retry)")
    print(f"  Incomplete (late):   {len(resumable):4d}  ← RESUMABLE")
    print(f"  Errors:             {len(errors):4d}")
    
    if resumable:
        print(f"\nResumable sim_ids: {', '.join(str(r['sim_id']) for r in resumable)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify stage2 timeout logs and identify resumable simulations"
    )
    parser.add_argument("--logs-dir", required=True, 
                       help="Directory containing stage2_sim*.out files")
    parser.add_argument("--output", default="stage2_classification.csv",
                       help="Output CSV filename (default: stage2_classification.csv)")
    
    args = parser.parse_args()
    scan_logs(args.logs_dir, args.output)