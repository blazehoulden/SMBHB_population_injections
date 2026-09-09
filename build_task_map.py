#!/usr/bin/env python3
"""
Build the stage-2 task map after stage 1 completes.

Writes  <output_dir>/metadata/task_map.json:
  { "0": [pop_idx, chunk_idx], "1": [...], ... }

Also prints the --array argument for sbatch, e.g. --array=0-2399

Usage
─────
  python build_task_map.py --output-dir /scratch/runs/my_run
"""

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Build stage-2 task map")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = args.output_dir
    meta_dir = os.path.join(out_dir, "metadata")

    config_path = os.path.join(meta_dir, "config.json")
    if not os.path.exists(config_path):
        sys.exit(f"ERROR: config.json not found at {config_path}  "
                 "(did stage 1 finish?)")

    with open(config_path) as fh:
        run_config = json.load(fh)

    task_map = {}   # task_id (str) -> [pop_idx, chunk_idx]
    task_id  = 0

    for pc in sorted(run_config["populations"], key=lambda x: x["pop_idx"]):
        pop_idx  = pc["pop_idx"]
        n_chunks = pc["n_chunks"]
        for chunk_idx in range(n_chunks):
            task_map[str(task_id)] = [pop_idx, chunk_idx]
            task_id += 1

    n_tasks    = task_id
    array_arg  = f"0-{n_tasks - 1}"

    task_map_path = os.path.join(meta_dir, "task_map.json")
    with open(task_map_path, "w") as fh:
        json.dump(task_map, fh)

    print(f"Task map written: {task_map_path}")
    print(f"Total tasks      : {n_tasks}")
    print(f"sbatch --array   : {array_arg}")

    # Summary per population
    print(f"\n{'pop_idx':>8}  {'n_binaries':>12}  {'n_chunks':>10}  {'task_ids':>20}")
    print("-" * 60)
    offset = 0
    for pc in sorted(run_config["populations"], key=lambda x: x["pop_idx"]):
        n_c = pc["n_chunks"]
        print(f"{pc['pop_idx']:>8}  {pc['n_binaries']:>12,}  {n_c:>10}  "
              f"{offset}-{offset+n_c-1}")
        offset += n_c


if __name__ == "__main__":
    main()
