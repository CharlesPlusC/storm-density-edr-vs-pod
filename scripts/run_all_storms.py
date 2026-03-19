"""
Run density inversion (POD + EDR) on all storms listed in a CSV storm file.

Reads the storm list, finds matching ephemeris and TU Delft files, and runs the
full pipeline for each storm. Saves per-storm outputs and a summary metrics CSV.

Usage:
    python scripts/run_all_storms.py [--storms-file misc/selected_storms.txt] [--output-dir output]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

from src.pipeline import analyze_storm_from_files
from src.data_loaders import (find_ephemeris_file, find_tudelft_files,
                              get_ephemeris_date_range, find_accel_density_file)


def load_storm_list(path):
    """
    Read storm CSV file.

    Returns list of (satellite, storm_date, kp_category) where storm_date is a date object.
    """
    storms = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sat = row['satellite'].strip()
            storm_date = date.fromisoformat(row['storm_date'].strip())
            kp = row['kp_category'].strip()
            storms.append((sat, storm_date, kp))
    return storms


def main():
    parser = argparse.ArgumentParser(description='Run all storms (POD + EDR)')
    parser.add_argument('--storms-file', default='misc/selected_storms.txt',
                        help='CSV storm list (satellite, storm_date, kp_category)')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--diagnostics', action='store_true', help='Save diagnostic plots')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip storms that already have output')
    parser.add_argument('--ephem-dir', default='external/ephems',
                        help='Ephemeris base directory')
    args = parser.parse_args()

    storms = load_storm_list(args.storms_file)
    print(f"Loaded {len(storms)} storms from {args.storms_file}\n")

    # Check data availability for each storm
    ready = []
    missing = []
    for sat, storm_date, kp in storms:
        label = f"{sat}_{storm_date}"
        try:
            ephem_path = find_ephemeris_file(sat, storm_date, args.ephem_dir)
        except FileNotFoundError as e:
            missing.append((label, f"ephemeris: {e}"))
            continue

        ep_start, ep_end = get_ephemeris_date_range(ephem_path)

        # Try unified finder first (handles Swarm + TU Delft)
        accel_path, accel_fmt = find_accel_density_file(sat, storm_date)
        tudelft_paths = []
        if accel_fmt == 'tudelft':
            # For TU Delft, also find all monthly files covering the full ephemeris range
            try:
                tudelft_paths = find_tudelft_files(sat, ep_start, ep_end)
            except (FileNotFoundError, ValueError):
                tudelft_paths = []

        ready.append((sat, storm_date, kp, ephem_path, tudelft_paths,
                       ep_start, ep_end, accel_path, accel_fmt))

    print(f"Ready: {len(ready)} storms")
    print(f"Missing data: {len(missing)} storms")
    if missing:
        print("\nMissing storms:")
        for label, reason in missing:
            print(f"  {label}: {reason}")
    if not ready:
        print("\nNo storms ready to process. Exiting.")
        return

    print(f"\nStorms to process:")
    for sat, storm_date, kp, ep, _, s, e, ap, af in ready:
        truth_str = f" truth: {af}" if ap else ""
        print(f"  {sat:15s}  {storm_date}  {kp:3s}  ephem: {s} → {e}{truth_str}")

    all_metrics = []
    failed = []

    for i, (sat, storm_date, kp, ephem_path, tudelft_paths,
            ep_start, ep_end, accel_path, accel_fmt) in enumerate(ready):
        storm_label = f"{sat}_{storm_date}"
        storm_dir = os.path.join(args.output_dir, storm_label)

        if args.skip_existing and os.path.isfile(os.path.join(storm_dir, 'edr_effective.csv')):
            print(f"\n[{i+1}/{len(ready)}] {storm_label} — skipping (already complete)")
            continue

        os.makedirs(storm_dir, exist_ok=True)
        print(f"\n{'='*72}")
        print(f"[{i+1}/{len(ready)}] {sat}  {storm_date}  ({kp})")
        print(f"{'='*72}")

        # Use full ephemeris range (files are pre-trimmed to correct windows)
        storm_start = None
        storm_end = None

        try:
            results = analyze_storm_from_files(
                ephem_path=ephem_path,
                tudelft_path=tudelft_paths if tudelft_paths else None,
                sat_name=sat,
                storm_start=storm_start,
                storm_end=storm_end,
                diagnostics=args.diagnostics,
                output_dir=storm_dir,
                accel_path=accel_path,
                accel_fmt=accel_fmt,
            )

            if results.get('metrics') is not None:
                met = results['metrics'].copy()
                met['satellite'] = sat
                met['storm_date'] = str(storm_date)
                met['kp_category'] = kp
                met['method'] = met.index
                all_metrics.append(met)
                print(f"\n  Metrics:\n{results['metrics'].to_string()}")
            else:
                print("  WARNING: No metrics computed (missing truth data?)")
                failed.append((storm_label, "no metrics"))

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append((storm_label, str(e)))

    # ---- Summary ----
    print(f"\n\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"Completed: {len(all_metrics)} storms")
    print(f"Failed/skipped: {len(failed)}")

    if failed:
        print("\nFailed storms:")
        for label, reason in failed:
            print(f"  {label}: {reason}")

    if all_metrics:
        summary = pd.concat(all_metrics, ignore_index=True)
        summary_path = os.path.join(args.output_dir, 'all_storms_metrics.csv')
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")

        for method in ['POD', 'EDR']:
            subset = summary[summary['method'] == method]
            if len(subset) > 0:
                print(f"\n--- {method} ---")
                print(f"  Mean r:   {subset['r'].mean():.4f}")
                print(f"  Mean SD%: {subset['SD_pct'].mean():.2f}")
                print(f"  N storms: {len(subset)}")


if __name__ == '__main__':
    main()
