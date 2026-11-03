#!/usr/bin/env bash
#
# Summarise how many generated binaries each completed simulation needed to
# reach its SGWB S/N target.  Reports are written below runs/logs/.
#
# Examples:
#   bash debug/report_sgwb_population_requirements.sh runs
#   bash debug/report_sgwb_population_requirements.sh /path/to/runs
#
# The script does not rerun simulations. It reads:
#   simXXX/metadata/config.json
#   simXXX/metadata/phase_handoff.json
#
# "Active binaries" is calculated from the active sub-chunks selected by
# Stage 2. With sub-chunks enabled, one sub-chunk contains
# chunk_size / n_sub_chunks binaries.

set -euo pipefail

RUNS_ROOT="${1:-runs}"

if [[ ! -d "$RUNS_ROOT" ]]; then
    echo "ERROR: runs directory does not exist: $RUNS_ROOT" >&2
    exit 1
fi

LOG_DIR="$RUNS_ROOT/logs"
mkdir -p "$LOG_DIR"

python3 - "$RUNS_ROOT" "$LOG_DIR" <<'PY'
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

runs_root = Path(sys.argv[1]).expanduser().resolve()
log_dir = Path(sys.argv[2]).expanduser().resolve()
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def iter_config_paths(root):
    """Find metadata files without recursively stat-ing large data trees."""
    candidates = []

    def add_sim_dir(sim_dir):
        metadata = sim_dir / "metadata"
        config = metadata / "config.json"
        if config.is_file():
            candidates.append(config)

    # Standard layout: runs/<population-config>/simXXX/metadata/config.json
    try:
        first_level = list(root.iterdir())
    except OSError as exc:
        raise SystemExit(f"Could not read runs directory {root}: {exc}")

    for entry in first_level:
        if not entry.is_dir() or entry.name == "logs":
            continue
        if entry.name.startswith("sim"):
            add_sim_dir(entry)
            continue
        try:
            for sim_entry in entry.iterdir():
                if sim_entry.is_dir() and sim_entry.name.startswith("sim"):
                    add_sim_dir(sim_entry)
        except OSError as exc:
            print(f"WARNING: cannot scan {entry}: {exc}", file=sys.stderr)

    return sorted(candidates)


records = []
for config_path in iter_config_paths(runs_root):
    sim_dir = config_path.parent.parent
    handoff_path = sim_dir / "metadata" / "phase_handoff.json"
    if not handoff_path.is_file():
        continue

    try:
        config = json.loads(config_path.read_text())
        handoff = json.loads(handoff_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: skipping {sim_dir}: {exc}", file=sys.stderr)
        continue

    try:
        n_chunks = int(config["n_chunks"])
        n_sub_chunks = int(config.get("n_sub_chunks", 1))
        chunk_size = int(config["chunk_size"])
        active_sub_chunks = int(handoff["n_active_sub_chunks"])
        total_sub_chunks = int(
            handoff.get("n_total_sub_chunks", n_chunks * n_sub_chunks)
        )
        if n_chunks <= 0 or n_sub_chunks <= 0 or chunk_size <= 0:
            raise ValueError("n_chunks, n_sub_chunks, and chunk_size must be positive")
        if chunk_size % n_sub_chunks:
            raise ValueError("chunk_size is not divisible by n_sub_chunks")
    except (KeyError, TypeError, ValueError) as exc:
        print(f"WARNING: invalid metadata in {sim_dir}: {exc}", file=sys.stderr)
        continue

    active_shard_ids = handoff.get("active_shard_ids", [])
    if active_shard_ids and len(active_shard_ids) != active_sub_chunks:
        print(
            f"WARNING: {sim_dir}: active_shard_ids has "
            f"{len(active_shard_ids)} entries, metadata says "
            f"{active_sub_chunks}; using n_active_sub_chunks.",
            file=sys.stderr,
        )

    sub_chunk_size = int(config.get("sub_chunk_size", chunk_size // n_sub_chunks))
    requested_binaries = n_chunks * chunk_size
    active_binaries = active_sub_chunks * sub_chunk_size
    active_fraction = active_binaries / requested_binaries
    snr = handoff.get("snr_achieved")
    sim_name = sim_dir.name
    population_config = str(config.get("config", sim_dir.parent.name))

    records.append(
        {
            "config": population_config,
            "sim": sim_name,
            "snr": float(snr) if snr is not None else math.nan,
            "active_sub_chunks": active_sub_chunks,
            "total_sub_chunks": total_sub_chunks,
            "active_binaries": active_binaries,
            "requested_binaries": requested_binaries,
            "active_fraction": active_fraction,
            "chunk_size": chunk_size,
            "n_chunks": n_chunks,
            "n_sub_chunks": n_sub_chunks,
        }
    )

if not records:
    raise SystemExit(
        f"No completed simulations with phase_handoff.json found below {runs_root}"
    )

records.sort(key=lambda row: (row["config"], row["sim"]))
groups = defaultdict(list)
for row in records:
    groups[row["config"]].append(row)

combined_lines = [
    f"SGWB population requirement report",
    f"Generated: {datetime.now().isoformat(timespec='seconds')}",
    f"Runs root: {runs_root}",
    "",
    "Active binaries are the Stage 2-selected sub-chunks; requested binaries "
    "are n_chunks * chunk_size.",
    "",
]

for config, rows in groups.items():
    config_lines = [
        f"Configuration: {config}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runs root: {runs_root}",
        "",
        "sim  S/N     active_subchunks/total  active_binaries  "
        "requested_binaries  fraction",
        "---  ------  ----------------------  ---------------  "
        "------------------  --------",
    ]

    for row in rows:
        snr_text = f"{row['snr']:.4f}" if math.isfinite(row["snr"]) else "n/a"
        config_lines.append(
            f"{row['sim']:<4} {snr_text:>6}  "
            f"{row['active_sub_chunks']:>7}/{row['total_sub_chunks']:<12}  "
            f"{row['active_binaries']:>15,}  "
            f"{row['requested_binaries']:>18,}  "
            f"{row['active_fraction']:>7.3f}"
        )

    active_values = [row["active_binaries"] for row in rows]
    snrs = [row["snr"] for row in rows if math.isfinite(row["snr"])]
    mean_active = sum(active_values) / len(active_values)
    median_active = sorted(active_values)[len(active_values) // 2]
    mean_snr = sum(snrs) / len(snrs) if snrs else math.nan
    config_lines += [
        "",
        f"Simulations counted: {len(rows)}",
        f"Minimum active binaries: {min(active_values):,}",
        f"Median active binaries: {median_active:,}",
        f"Mean active binaries: {mean_active:,.1f}",
        f"Maximum active binaries: {max(active_values):,}",
        (
            f"Mean achieved S/N: {mean_snr:.4f}"
            if math.isfinite(mean_snr)
            else "Mean achieved S/N: n/a"
        ),
        "",
    ]

    config_log = log_dir / f"sgwb_population_requirements_{config}_{stamp}.log"
    config_log.write_text("\n".join(config_lines))
    combined_lines.extend(config_lines)

combined_log = log_dir / f"sgwb_population_requirements_all_{stamp}.log"
combined_log.write_text("\n".join(combined_lines))

print(f"Wrote {combined_log}")
for config in groups:
    print(f"Wrote {log_dir / f'sgwb_population_requirements_{config}_{stamp}.log'}")
PY
