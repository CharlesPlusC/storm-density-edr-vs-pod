#!/usr/bin/env python3
"""Reproduce Figure 3 (rms_vs_drag) and Figure 4 (per-orbit-hist) from orbit-effective CSVs."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OE_DIR = os.path.join(REPO, 'data', 'orbit_effective')
FIG_DIR = os.path.join(REPO, 'figures')

SAT_PARAMS = {
    'CHAMP':      {'mass': 522.0, 'area': 1.0,  'Cd': 2.2, 'v': 7700.0},
    'GRACE-FO-A': {'mass': 600.2, 'area': 1.04, 'Cd': 3.2, 'v': 7500.0},
}

# Ray et al. (2024) Figure 13, digitised
RAY_POD = np.array([
    [5e-9, 30], [1e-8, 25], [5e-8, 14], [1e-7, 13],
    [5e-7,  9], [1e-6,  8], [2e-6,  5],
])
RAY_EDR = np.array([
    [5e-9, 55], [1e-8, 48], [5e-8, 50], [2e-7, 20],
    [5e-7, 15], [1e-6, 14],
])

NUM_BINS = 9


def drag_acc(rho, sat_name):
    """a_drag = 0.5 * rho * v^2 * Cd * A / m"""
    p = SAT_PARAMS[sat_name]
    return 0.5 * rho * p['v']**2 * p['Cd'] * p['area'] / p['mass']


def parse_sat_name(dirname):
    if dirname.startswith('GRACE-FO-A_'):
        return 'GRACE-FO-A'
    return dirname.split('_', 1)[0]


def load_all_storms():
    result = {m: {'drag': [], 'pct_err': []} for m in ['POD', 'EDR']}
    hist_data = {'CHAMP': [], 'GRACE-FO-A': []}

    for dirname in sorted(os.listdir(OE_DIR)):
        pod_path = os.path.join(OE_DIR, dirname, 'pod_acc_effective.csv')
        edr_path = os.path.join(OE_DIR, dirname, 'edr_effective.csv')
        if not os.path.isfile(edr_path) or not os.path.isfile(pod_path):
            continue

        sat = parse_sat_name(dirname)
        if sat not in SAT_PARAMS:
            continue

        pod = pd.read_csv(pod_path)
        edr = pd.read_csv(edr_path)

        acc = pod['acc_effective'].values
        pod_db = pod['pod_debiased'].values
        edr_db = edr['edr_debiased'].values

        n = min(len(acc), len(pod_db), len(edr_db))
        v = (np.isfinite(acc[:n]) & (acc[:n] > 0)
             & np.isfinite(pod_db[:n]) & (pod_db[:n] > 0)
             & np.isfinite(edr_db[:n]) & (edr_db[:n] > 0))

        drags = np.array([drag_acc(a, sat) for a in acc[:n][v]])
        hist_data[sat].extend(drags.tolist())

        pct_pod = (pod_db[:n][v] - acc[:n][v]) / acc[:n][v] * 100.0
        result['POD']['drag'].extend(drags)
        result['POD']['pct_err'].extend(pct_pod)

        pct_edr = (edr_db[:n][v] - acc[:n][v]) / acc[:n][v] * 100.0
        result['EDR']['drag'].extend(drags)
        result['EDR']['pct_err'].extend(pct_edr)

    for m in result:
        result[m]['drag'] = np.array(result[m]['drag'])
        result[m]['pct_err'] = np.array(result[m]['pct_err'])
        print(f"  {m}: {len(result[m]['drag'])} orbits")

    return result, hist_data


def logbin_rms(drag, pct_err, num_bins=NUM_BINS):
    """Bin percentage errors into log-spaced drag bins, return (centers, rms, std, counts)."""
    mask = (drag > 0) & np.isfinite(pct_err)
    x, y = drag[mask], pct_err[mask]
    if x.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), num_bins + 1)
    inds = np.clip(np.digitize(x, edges) - 1, 0, num_bins - 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    rms_vals = np.full(num_bins, np.nan)
    std_vals = np.full(num_bins, np.nan)
    counts = np.zeros(num_bins, dtype=int)

    for i in range(num_bins):
        in_bin = y[inds == i]
        counts[i] = len(in_bin)
        if len(in_bin) >= 3:
            rms_vals[i] = np.sqrt(np.mean(in_bin**2))
            std_vals[i] = np.std(np.abs(in_bin))

    return centers, rms_vals, std_vals, counts


def plot_rms_vs_drag(data):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_facecolor('#e8e0d4')

    styles = {
        'POD': {'color': 'magenta', 'label': 'POD-A (This study)'},
        'EDR': {'color': 'cyan',    'label': 'EDR (This study)'},
    }

    for method, sty in styles.items():
        drag = data[method]['drag']
        pct_err = data[method]['pct_err']
        clip = np.abs(pct_err) < 200
        centers, rms, std, counts = logbin_rms(drag[clip], pct_err[clip])
        valid = np.isfinite(rms)
        ax.errorbar(centers[valid], rms[valid], yerr=std[valid],
                    color=sty['color'], marker='o', ms=4, lw=1.8,
                    capsize=3, capthick=1.0, label=sty['label'], zorder=3)

    ax.plot(RAY_POD[:, 0], RAY_POD[:, 1], '--', color='magenta', lw=1.5,
            label='POD-A (Ray et al.)', zorder=2)
    ax.plot(RAY_EDR[:, 0], RAY_EDR[:, 1], '--', color='cyan', lw=1.5,
            label='EDR (Ray et al.)', zorder=2)

    ax.set_xscale('log')
    ax.set_xlabel(r'Per-Orbit Mean Drag Acceleration [m/s$^2$]', fontsize=11)
    ax.set_ylabel(r'Method $\tilde{\rho}_{\mathrm{RMS}}$ [%]', fontsize=11)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=9, loc='upper right',
              fancybox=True, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()

    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'rms_vs_drag_orbit_effective.{ext}'),
                    bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print("Saved: figures/rms_vs_drag_orbit_effective.png + .svg")


def plot_per_orbit_hist(hist_data):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_facecolor('#e8e0d4')

    n_gfo = len(hist_data['GRACE-FO-A'])
    n_champ = len(hist_data['CHAMP'])

    all_drags = hist_data['GRACE-FO-A'] + hist_data['CHAMP']
    bins = np.logspace(np.log10(min(all_drags)), np.log10(max(all_drags)), 30)

    ax.hist([hist_data['GRACE-FO-A'], hist_data['CHAMP']], bins=bins,
            stacked=True, color=['#2980b9', '#c0392b'],
            label=[f'GRACE-FO-A  (n={n_gfo})', f'CHAMP  (n={n_champ})'],
            zorder=2)

    ax.set_xscale('log')
    ax.set_xlabel(r'Per-Orbit Mean Drag Acceleration [m/s$^2$]', fontsize=11)
    ax.set_ylabel('Number of Orbits', fontsize=11)
    ax.legend(fontsize=11, loc='upper left',
              fancybox=True, framealpha=0.9, edgecolor='black')
    fig.tight_layout()

    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'per-orbit-hist.{ext}'),
                    bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print("Saved: figures/per-orbit-hist.png + .svg")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading orbit-effective data...")
    data, hist_data = load_all_storms()

    print("\n=== Figure 3: RMS% vs Drag ===")
    plot_rms_vs_drag(data)

    print("\n=== Figure 4: Per-Orbit Histogram ===")
    plot_per_orbit_hist(hist_data)


if __name__ == '__main__':
    main()
