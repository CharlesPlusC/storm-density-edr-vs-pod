#!/usr/bin/env python3
"""
Compute all 5 configurations for Table 3 and the scatter heatmap.

For each storm, loads ephemeris + TU Delft truth, runs POD + EDR force models,
computes genuine multi-orbit densities (not step-function approximation),
runs the sub-orbital sweep, and pools all density pairs.

Saves per-storm NPZ files and generates the full 5-row scatter heatmap + Table 3.

Usage:
    cd source/arc_length_analysis
    python scripts/compute_full_table3.py [--parallel N]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import traceback
from datetime import datetime, timedelta

from src.data_loaders import (
    load_ephemeris, load_tudelft_density, find_tudelft_files,
    find_ephemeris_file, get_ephemeris_date_range,
)
from src.edr import find_perigees
from src.edr_hires import precompute_forces, suborbit_edr_density, multiorbit_edr_density
from src.pod_accelerometry import pod_density_from_positions
from src.pod_accelerometry_hires import (
    interpolate_tudelft_to_grid, debias_density, pod_effective_density,
    multiorbit_pod_density,
)
from src.effective_density import compute_effective_density
from src.config import SATELLITES

# ── Config ──
V2_DIR = 'output/orbit-effective'
NPZ_DIR = os.path.join(V2_DIR, 'npz')
EPHEM_BASE = 'external/ephems'

# EDR and POD candidate arc lengths for sub-orbital sweep (same as paper)
CANDIDATE_ARCS = [10, 15, 20, 30, 45, 60, 75, 90]


def compute_r(rho, truth):
    valid = np.isfinite(rho) & np.isfinite(truth) & (rho > 0) & (truth > 0)
    if valid.sum() < 20:
        return np.nan
    return np.corrcoef(rho[valid], truth[valid])[0, 1]


def process_storm(sat, storm_date_str):
    """Process a single storm: all 5 configurations."""
    storm_date = datetime.strptime(storm_date_str, '%Y-%m-%d').date()

    # ── Load ephemeris ──
    ephem_path = find_ephemeris_file(sat, storm_date, EPHEM_BASE)
    pos_vel_eci, times_sp3 = load_ephemeris(ephem_path)
    N_sp3 = len(times_sp3)
    print(f"  SP3: {N_sp3} points, {times_sp3[0]} → {times_sp3[-1]}")

    # ── Load TU Delft ──
    ep_start, ep_end = get_ephemeris_date_range(ephem_path)
    td_files = find_tudelft_files(sat, ep_start, ep_end)
    td_times, td_density = load_tudelft_density(
        td_files,
        start_date=times_sp3[0] - timedelta(hours=1),
        end_date=times_sp3[-1] + timedelta(hours=1),
    )
    td_on_sp3 = interpolate_tudelft_to_grid(td_times, td_density, times_sp3)
    n_td = np.isfinite(td_on_sp3).sum()
    print(f"  TU Delft: {n_td}/{N_sp3} matched on SP3 grid")
    if n_td < 50:
        raise ValueError("Too few TU Delft matches")

    # ── Perigees on SP3 grid ──
    perigees_sp3 = find_perigees(pos_vel_eci, times_sp3)
    n_orb_sp3 = len(perigees_sp3) - 1
    print(f"  Perigees: {n_orb_sp3} orbits (SP3)")

    # ── EDR: precompute forces + multi-orbit ──
    print("  EDR: pre-computing forces...")
    forces = precompute_forces(pos_vel_eci, times_sp3, sat)

    # Multi-orbit EDR (genuine Jacobi energy balance)
    edr_multiorbit = {}
    for N in [1, 2, 3]:
        edr_N = multiorbit_edr_density(forces, sat, perigees_sp3, n_orbits=N)
        truth_N, _ = compute_effective_density(td_on_sp3, pos_vel_eci,
                                                perigees_sp3[::N])
        n_common = min(len(edr_N), len(truth_N))
        edr_multiorbit[N] = {
            'model': np.array(edr_N[:n_common]),
            'truth': np.array(truth_N[:n_common]),
        }
        print(f"  EDR {N}-orbit: {n_common} arcs")

    # Sub-orbital EDR sweep
    print(f"  EDR: sweeping {len(CANDIDATE_ARCS)} arc lengths...")
    edr_best_r = -1
    edr_best_rho = None
    edr_best_arc = None
    for am in CANDIDATE_ARCS:
        rho_k, _ = suborbit_edr_density(pos_vel_eci, times_sp3, forces, am, sat)
        rho_k, _ = debias_density(rho_k, td_on_sp3)
        r_k = compute_r(rho_k, td_on_sp3)
        if np.isfinite(r_k) and r_k > edr_best_r:
            edr_best_r = r_k
            edr_best_rho = rho_k.copy()
            edr_best_arc = am
    print(f"  EDR best sub-orbital: {edr_best_arc} min, r={edr_best_r:.4f}")

    # ── POD: run density inversion ──
    print("  POD: running density inversion...")
    pod_result = pod_density_from_positions(
        pos_vel_eci, times_sp3, sat_name=sat,
        output_cadence=15, density_smooth_min=0,  # no smoothing — raw for sweep
    )
    pod_times = pod_result['times_out']
    pod_pv = pod_result['pos_vel_out']
    _, _, l_drag = pod_result['drag_hcl']
    N_pod = len(pod_times)

    # Perigees on POD 15s grid
    perigees_pod = find_perigees(pod_pv, pod_times)
    n_orb_pod = len(perigees_pod) - 1
    print(f"  POD perigees: {n_orb_pod} orbits (15s grid)")

    # TU Delft on POD grid
    td_on_pod = interpolate_tudelft_to_grid(td_times, td_density, pod_times)

    # Multi-orbit POD (genuine Picone drag-power integral)
    pod_multiorbit = {}
    for N in [1, 2, 3]:
        pod_N = multiorbit_pod_density(l_drag, pod_pv, pod_times, sat,
                                        perigees_pod, n_orbits=N)
        truth_N_pod, _ = compute_effective_density(td_on_pod, pod_pv,
                                                    perigees_pod[::N])
        n_common = min(len(pod_N), len(truth_N_pod))
        pod_multiorbit[N] = {
            'model': np.array(pod_N[:n_common]),
            'truth': np.array(truth_N_pod[:n_common]),
        }
        print(f"  POD {N}-orbit: {n_common} arcs")

    # Sub-orbital POD sweep (Picone-weighted sliding window)
    print(f"  POD: sweeping {len(CANDIDATE_ARCS)} windows...")
    pod_best_r = -1
    pod_best_rho = None
    pod_best_win = None
    for wm in CANDIDATE_ARCS:
        rho_k = pod_effective_density(l_drag, pod_pv, pod_times, sat, wm,
                                       output_cadence=15)
        rho_k, _ = debias_density(rho_k, td_on_pod)
        # Downsample to 30s for fair comparison
        rho_30s = rho_k[::2][:N_sp3]
        r_k = compute_r(rho_30s, td_on_sp3[:len(rho_30s)])
        if np.isfinite(r_k) and r_k > pod_best_r:
            pod_best_r = r_k
            # Store the density on the 30s SP3 grid for the scatter plot
            pod_best_rho = np.full(N_sp3, np.nan)
            n = min(len(rho_30s), N_sp3)
            pod_best_rho[:n] = rho_30s[:n]
            pod_best_win = wm
    print(f"  POD best sub-orbital: {pod_best_win} min, r={pod_best_r:.4f}")

    # ── 1-orbit native: expand multi-orbit values to step function ──
    # De-bias 1-orbit values against per-orbit truth first
    edr_1orb_model = edr_multiorbit[1]['model']
    edr_1orb_truth = edr_multiorbit[1]['truth']
    edr_1orb_db, _ = debias_density(edr_1orb_model, edr_1orb_truth)

    pod_1orb_model = pod_multiorbit[1]['model']
    pod_1orb_truth = pod_multiorbit[1]['truth']
    pod_1orb_db, _ = debias_density(pod_1orb_model, pod_1orb_truth)

    # Expand to step function on SP3 grid (EDR uses SP3 perigees)
    edr_step = np.full(N_sp3, np.nan)
    for k in range(min(len(edr_1orb_db), len(perigees_sp3) - 1)):
        edr_step[perigees_sp3[k]:perigees_sp3[k + 1]] = edr_1orb_db[k]

    # Expand to step function on POD grid, then downsample to SP3
    pod_step_15s = np.full(N_pod, np.nan)
    for k in range(min(len(pod_1orb_db), len(perigees_pod) - 1)):
        pod_step_15s[perigees_pod[k]:perigees_pod[k + 1]] = pod_1orb_db[k]
    pod_step = pod_step_15s[::2][:N_sp3]
    pod_step_full = np.full(N_sp3, np.nan)
    n = min(len(pod_step), N_sp3)
    pod_step_full[:n] = pod_step[:n]

    # ── Save NPZ ──
    os.makedirs(NPZ_DIR, exist_ok=True)
    npz_path = os.path.join(NPZ_DIR, f'{sat}_{storm_date_str}.npz')
    save_data = {
        'td_on_sp3': td_on_sp3,
        'perigees_sp3': perigees_sp3,
        'perigees_pod': perigees_pod,
        # 1-orbit step functions for native comparison
        'edr_step_sp3': edr_step,
        'pod_step_sp3': pod_step_full,
        # Sub-orbital best densities (on SP3 30s grid)
        'edr_suborb': edr_best_rho if edr_best_rho is not None else np.array([]),
        'pod_suborb': pod_best_rho if pod_best_rho is not None else np.array([]),
        'edr_best_arc': edr_best_arc or 0,
        'pod_best_win': pod_best_win or 0,
    }
    # Multi-orbit data
    for N in [1, 2, 3]:
        save_data[f'edr_{N}orb'] = edr_multiorbit[N]['model']
        save_data[f'edr_truth_{N}orb'] = edr_multiorbit[N]['truth']
        save_data[f'pod_{N}orb'] = pod_multiorbit[N]['model']
        save_data[f'pod_truth_{N}orb'] = pod_multiorbit[N]['truth']

    np.savez_compressed(npz_path, **save_data)
    print(f"  Saved: {npz_path}")

    return {
        'sat': sat, 'storm': storm_date_str,
        'n_orb_sp3': n_orb_sp3, 'n_orb_pod': n_orb_pod,
        'edr_best_arc': edr_best_arc, 'edr_best_r': edr_best_r,
        'pod_best_win': pod_best_win, 'pod_best_r': pod_best_r,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    print(f"Working directory: {os.getcwd()}")

    # Load storm list
    storms = []
    with open('misc/selected_storms.txt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('satellite'):
                continue
            parts = line.split(',')
            sat, date_str = parts[0], parts[1]
            if sat in ['CHAMP', 'GRACE-FO-A']:
                storms.append((sat, date_str))

    print(f"\nProcessing {len(storms)} storms\n")

    results = []
    failed = []
    for i, (sat, date_str) in enumerate(storms):
        label = f"{sat}_{date_str}"
        npz_path = os.path.join(NPZ_DIR, f'{label}.npz')

        # Skip if already computed
        if os.path.isfile(npz_path):
            print(f"[{i+1}/{len(storms)}] {label} — SKIP (exists)")
            results.append({'sat': sat, 'storm': date_str})
            continue

        print(f"\n[{i+1}/{len(storms)}] {label}")
        try:
            r = process_storm(sat, date_str)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append((sat, date_str, str(e)))

    print(f"\n{'='*60}")
    print(f"Completed: {len(results)}, Failed: {len(failed)}")
    if failed:
        print("Failed storms:")
        for s, d, e in failed:
            print(f"  {s} {d}: {e}")


if __name__ == '__main__':
    main()
