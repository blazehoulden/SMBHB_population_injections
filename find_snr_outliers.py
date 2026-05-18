#!/usr/bin/env python3
"""
Scan a directory of simulation log files and report any logs where the
mean SNR falls outside [SNR_MIN, SNR_MAX]. Results are printed to the
console and saved to a timestamped .txt file.

Usage:
    python find_snr_outliers.py /path/to/logs
    python find_snr_outliers.py /path/to/logs --ext .out --range 3.5 4.25
    python find_snr_outliers.py /path/to/logs --outdir /path/to/save/results
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

SNR_MEAN_PATTERN = re.compile(
    r"SNR achieved:\s*\n  Mean .{1,5} std\s*:\s*([\d.]+)",
    re.MULTILINE
)
SNR_MEAN_FALLBACK = re.compile(
    r"SNR achieved:.*?Mean\s*.{1,5}\s*std\s*:\s*([\d.]+)",
    re.DOTALL
)


def extract_mean_snr(text):
    m = SNR_MEAN_PATTERN.search(text) or SNR_MEAN_FALLBACK.search(text)
    return float(m.group(1)) if m else None


def scan_logs(log_dir, ext, snr_min, snr_max, outdir):
    log_files = sorted(log_dir.glob("*" + ext))
    if not log_files:
        print("No '*{}' files found in {}".format(ext, log_dir))
        sys.exit(1)

    outliers, no_snr, in_range = [], [], 0

    for lp in log_files:
        try:
            text = lp.read_text(errors="replace")
        except OSError as e:
            print("[WARN] Cannot read {}: {}".format(lp.name, e))
            continue
        snr = extract_mean_snr(text)
        if snr is None:
            no_snr.append(lp.name)
        elif snr < snr_min or snr > snr_max:
            outliers.append((lp.name, snr))
        else:
            in_range += 1

    # Build report lines
    now = datetime.now()
    timestamp_display = now.strftime("%A %d %B %Y  %H:%M:%S")
    timestamp_file    = now.strftime("%Y%m%d_%H%M%S")

    total = len(log_files)
    sep = "=" * 62
    lines = []

    lines.append(sep)
    lines.append("  SNR Outlier Report")
    lines.append("  Generated  : {}".format(timestamp_display))
    lines.append("  Log dir    : {}".format(log_dir))
    lines.append("  Scanned    : {}   Range: [{}, {}]".format(total, snr_min, snr_max))
    lines.append(sep)

    if outliers:
        lines.append("")
        lines.append("  OUTSIDE RANGE  ({} file(s)):".format(len(outliers)))
        lines.append("")
        lines.append("  {:<48} {:>9}  {}".format("Filename", "Mean SNR", "Flag"))
        lines.append("  " + "-" * 65)
        for fname, snr in outliers:
            flag = "LOW" if snr < snr_min else "HIGH"
            lines.append("  {:<48} {:>9.4f}  [{}]".format(fname, snr, flag))
    else:
        lines.append("")
        lines.append("  All files are within the SNR range. No outliers found.")

    if no_snr:
        lines.append("")
        lines.append("  SNR BLOCK NOT FOUND in {} file(s):".format(len(no_snr)))
        for f in no_snr:
            lines.append("    {}".format(f))

    lines.append("")
    lines.append("  Summary: {} in-range | {} outlier(s) | {} no-SNR | {} total".format(
        in_range, len(outliers), len(no_snr), total))
    lines.append(sep)
    lines.append("")

    report = "\n".join(lines)

    # Print to console
    print("\n" + report)

    # Save to file
    outdir = outdir or log_dir
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    out_filename = "snr_outliers_{}.txt".format(timestamp_file)
    out_path = outdir / out_filename

    out_path.write_text(report, encoding="utf-8")
    print("  Report saved to: {}\n".format(out_path))


def main():
    ap = argparse.ArgumentParser(description="Find logs with mean SNR outside a given range.")
    ap.add_argument("log_dir",          help="Directory containing log files")
    ap.add_argument("--ext",   default=".log",
                    help="File extension to scan (default: .log)")
    ap.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"),
                    default=[3.5, 4.25], help="Acceptable SNR range (default: 3.5 4.25)")
    ap.add_argument("--outdir", default=None,
                    help="Directory to save the report (default: same as log_dir)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.is_dir():
        print("Error: '{}' is not a directory.".format(log_dir))
        sys.exit(1)

    scan_logs(log_dir, args.ext, args.range[0], args.range[1], args.outdir)


if __name__ == "__main__":
    main()
