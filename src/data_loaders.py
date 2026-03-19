"""
Data loaders for raw ephemeris and TU Delft density files.

Ephemeris files: alternating data/sigma line pairs, positions in km, velocities in km/s.
TU Delft files: comment lines starting with #, whitespace-separated columns.
"""

import os
import numpy as np
from datetime import datetime, date, timedelta

from .config import SATELLITES


# ====================================================================
# Ephemeris loader
# ====================================================================
def load_ephemeris(filepath, gap_threshold=60.0):
    """
    Load an ephemeris file (alternating data + sigma lines).

    If there are time gaps > gap_threshold seconds, the data is split into
    contiguous segments and only the longest segment is returned.

    Returns
    -------
    pos_vel_eci : (N, 6) ndarray — ECI [m, m/s]
    times       : list of datetime (length N)
    """
    times = []
    states = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        dt = datetime.strptime(parts[0] + ' ' + parts[1], '%Y-%m-%d %H:%M:%S.%f')
        state = [float(p) for p in parts[2:8]]
        times.append(dt)
        states.append(state)

    pos_vel_eci = np.array(states) * 1000.0  # km, km/s → m, m/s

    # Detect and handle time gaps
    dt_sec = np.array([(times[i+1] - times[i]).total_seconds()
                        for i in range(len(times) - 1)])
    gap_idx = np.where(dt_sec > gap_threshold)[0]

    if len(gap_idx) > 0:
        boundaries = [0] + list(gap_idx + 1) + [len(times)]
        segments = [(boundaries[i], boundaries[i+1])
                    for i in range(len(boundaries) - 1)]
        best = max(segments, key=lambda s: s[1] - s[0])
        gap_info = ", ".join([f"{dt_sec[g]/3600:.1f}h at {times[g].strftime('%Y-%m-%d %H:%M')}"
                              for g in gap_idx])
        print(f"  WARNING: {len(gap_idx)} gap(s) in ephemeris: {gap_info}")
        print(f"  Using longest contiguous segment: idx {best[0]}-{best[1]} "
              f"({best[1]-best[0]} of {len(times)} points)")
        s, e = best
        pos_vel_eci = pos_vel_eci[s:e]
        times = times[s:e]

    return pos_vel_eci, times


# ====================================================================
# TU Delft density loader
# ====================================================================
def load_tudelft_density(filepath, start_date=None, end_date=None):
    """
    Load TU Delft accelerometer density file(s).

    Parameters
    ----------
    filepath   : str or list of str — one or more monthly files
    start_date : datetime or None — trim to this start
    end_date   : datetime or None — trim to this end

    Returns
    -------
    times   : list of datetime
    density : (M,) ndarray — dens_x [kg/m³], flag==0 only
    """
    if isinstance(filepath, str):
        filepath = [filepath]

    times = []
    densities = []

    for fp in filepath:
        with open(fp, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 12:
                    continue

                dt = datetime.strptime(parts[0] + ' ' + parts[1], '%Y-%m-%d %H:%M:%S.%f')

                # Column 11 (0-indexed 10) is flag1 — filter out flag=1
                flag = float(parts[10])
                if flag != 0.0:
                    continue

                if start_date and dt < start_date:
                    continue
                if end_date and dt > end_date:
                    continue

                times.append(dt)
                densities.append(float(parts[8]))  # dens_x

    # Sort by time when loading from multiple files
    if len(filepath) > 1 and len(times) > 1:
        order = sorted(range(len(times)), key=lambda i: times[i])
        times = [times[i] for i in order]
        densities = [densities[i] for i in order]

    return times, np.array(densities)


def get_ephemeris_date_range(filepath):
    """Read first and last data line to get the actual date range of an ephemeris file."""
    first_dt = last_dt = None
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i % 2 == 1:
                continue
            parts = line.split()
            dt = datetime.strptime(parts[0] + ' ' + parts[1], '%Y-%m-%d %H:%M:%S.%f')
            if first_dt is None:
                first_dt = dt
            last_dt = dt
    return first_dt.date(), last_dt.date()


# ====================================================================
# File finders
# ====================================================================
def find_ephemeris_file(sat_name, storm_date, base_dir='external/ephems'):
    """
    Find the ephemeris file whose date range contains storm_date.

    Parameters
    ----------
    sat_name   : str — e.g. 'GRACE-FO-A', 'CHAMP'
    storm_date : date or datetime
    base_dir   : str — path to ephems directory

    Returns
    -------
    str — full path to the ephemeris file
    """
    if isinstance(storm_date, datetime):
        storm_date = storm_date.date()
    if isinstance(storm_date, str):
        storm_date = date.fromisoformat(storm_date)

    sat_info = SATELLITES.get(sat_name, {})
    ephem_subdir = sat_info.get('ephem_dir', sat_name)
    sat_dir = os.path.join(base_dir, ephem_subdir)
    if not os.path.isdir(sat_dir):
        raise FileNotFoundError(f"No ephemeris directory: {sat_dir}")

    norad_id = sat_info.get('norad')
    best_file = None
    best_span = None

    for fname in os.listdir(sat_dir):
        if not fname.endswith('.txt'):
            continue
        stem = fname.replace('.txt', '')
        parts = stem.split('-')
        try:
            if len(parts) >= 7:
                # Old format: NORAD{ID}-{YYYY}-{MM}-{DD}-{YYYY}-{MM}-{DD}.txt
                start = date(int(parts[1]), int(parts[2]), int(parts[3]))
                end = date(int(parts[4]), int(parts[5]), int(parts[6]))
            elif '_' in stem:
                # New format: {SAT}_NORAD{ID}_{YYYY}-{MM}-{DD}.txt
                # The date range is determined by scanning file content
                date_part = stem.rsplit('_', 1)[-1]
                dp = date_part.split('-')
                storm_ref = date(int(dp[0]), int(dp[1]), int(dp[2]))
                # Accept if storm_date is within ±5 days of the reference date
                if abs((storm_date - storm_ref).days) > 5:
                    continue
                start = storm_ref - timedelta(days=5)
                end = storm_ref + timedelta(days=5)
            else:
                continue
        except (IndexError, ValueError):
            continue

        if start <= storm_date <= end:
            span = (end - start).days
            if best_span is None or span > best_span:
                best_file = os.path.join(sat_dir, fname)
                best_span = span

    if best_file is None:
        raise FileNotFoundError(
            f"No ephemeris file containing {storm_date} in {sat_dir}")
    return best_file


def find_tudelft_files(sat_name, start_date, end_date,
                        base_dir='external/accelerometer_densities'):
    """
    Find all TU Delft monthly files covering [start_date, end_date].

    Returns list of file paths (one per month spanned).
    """
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    if 'GRACE' in sat_name.upper():
        subdir = 'version_02_GRACE-FO_data'
        pattern = 'GC_DNS_ACC_{year:04d}_{month:02d}_v02c.txt'
    elif 'CHAMP' in sat_name.upper():
        subdir = 'version_02_CHAMP_data'
        pattern = 'CH_DNS_ACC_{year:04d}-{month:02d}_v02.txt'
    else:
        return []

    paths = []
    d = start_date
    seen = set()
    while d <= end_date:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            fname = pattern.format(year=d.year, month=d.month)
            path = os.path.join(base_dir, subdir, fname)
            if os.path.isfile(path):
                paths.append(path)
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)

    if not paths:
        raise FileNotFoundError(
            f"No TU Delft files for {sat_name} {start_date}→{end_date}")
    return paths


def find_tudelft_file(sat_name, storm_date,
                       base_dir='external/accelerometer_densities'):
    """
    Find the TU Delft density file for the month containing storm_date.

    Parameters
    ----------
    sat_name   : str — 'GRACE-FO-A', 'GRACE-FO', or 'CHAMP'
    storm_date : date or datetime
    base_dir   : str

    Returns
    -------
    str — full path to the density file
    """
    if isinstance(storm_date, datetime):
        storm_date = storm_date.date()
    if isinstance(storm_date, str):
        storm_date = date.fromisoformat(storm_date)

    year = storm_date.year
    month = storm_date.month

    if 'GRACE' in sat_name.upper():
        # GRACE-FO: GC_DNS_ACC_YYYY_MM_v02c.txt (underscore-separated)
        subdir = 'version_02_GRACE-FO_data'
        fname = f'GC_DNS_ACC_{year:04d}_{month:02d}_v02c.txt'
    elif 'CHAMP' in sat_name.upper():
        # CHAMP: CH_DNS_ACC_YYYY-MM_v02.txt (hyphen-separated)
        subdir = 'version_02_CHAMP_data'
        fname = f'CH_DNS_ACC_{year:04d}-{month:02d}_v02.txt'
    else:
        return None

    path = os.path.join(base_dir, subdir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TU Delft file not found: {path}")
    return path


# ====================================================================
# Swarm density loader
# ====================================================================
def load_swarm_density(filepath, start_date=None, end_date=None):
    """
    Load a Swarm POD-derived or accelerometer density CSV.

    Supports two formats:
    - POD: columns include validity_flag (filter flag != 0), 30s cadence
    - ACC: no validity_flag column, 10s cadence

    Parameters
    ----------
    filepath   : str — path to CSV file
    start_date : datetime or None
    end_date   : datetime or None

    Returns
    -------
    times   : list of datetime
    density : (M,) ndarray — [kg/m³]
    """
    import csv

    times = []
    densities = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        has_flag = 'validity_flag' in reader.fieldnames

        for row in reader:
            if has_flag:
                flag = float(row['validity_flag'])
                if flag != 0.0:
                    continue

            density_str = row['density'].strip()
            if not density_str:
                continue

            dt = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')

            if start_date and dt < start_date:
                continue
            if end_date and dt > end_date:
                continue

            times.append(dt)
            densities.append(float(density_str))

    return times, np.array(densities)


# ====================================================================
# Unified accelerometer density finder
# ====================================================================
def find_accel_density_file(sat_name, storm_date,
                             base_dir='external/accelerometer_densities'):
    """
    Find accelerometer/truth density file for any supported satellite.

    For GRACE-FO/CHAMP: delegates to find_tudelft_file().
    For Swarm: looks for {base_dir}/{sat_name}/{storm_date}/{sat_name}_{storm_date}_density_*.csv

    Parameters
    ----------
    sat_name   : str — e.g. 'GRACE-FO-A', 'CHAMP', 'Swarm-B'
    storm_date : date, datetime, or str
    base_dir   : str — base directory for all accelerometer density data

    Returns
    -------
    (filepath, fmt) : tuple — filepath and format string ('tudelft' or 'swarm')
    Returns (None, None) if no file found.
    """
    if isinstance(storm_date, datetime):
        storm_date_d = storm_date.date()
    elif isinstance(storm_date, str):
        storm_date_d = date.fromisoformat(storm_date)
    else:
        storm_date_d = storm_date

    # GRACE-FO / CHAMP → TU Delft format
    if 'GRACE' in sat_name.upper() or 'CHAMP' in sat_name.upper():
        try:
            path = find_tudelft_file(sat_name, storm_date_d, base_dir)
            return path, 'tudelft'
        except FileNotFoundError:
            return None, None

    # Swarm → CSV in {base_dir}/{sat_name}/{storm_date}/
    storm_str = storm_date_d.isoformat()
    sat_dir = os.path.join(base_dir, sat_name, storm_str)
    if not os.path.isdir(sat_dir):
        return None, None

    # Try POD first, then ACC
    for suffix in ['density_POD.csv', 'density_ACC.csv']:
        fname = f'{sat_name}_{storm_str}_{suffix}'
        path = os.path.join(sat_dir, fname)
        if os.path.isfile(path):
            return path, 'swarm'

    return None, None
